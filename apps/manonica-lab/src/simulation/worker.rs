//! Fixed-step simulation thread. The window never waits for a physics step.
//! A bounded, lossy snapshot queue keeps memory constant if rendering stalls.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use manonica_physics::FreeManaField;

use super::{BackendPreference, Simulation};
use crate::experiment::Experiment;

mod runtime;
use runtime::{RuntimeChannels, run_simulation};

pub enum ConfigurationResult {
    Applied {
        revision: u64,
        snapshot: Box<SimulationSnapshot>,
    },
    Rejected {
        revision: u64,
        message: String,
    },
}
fn worker_stopped() -> manonica_error::BaseError {
    manonica_error::BaseError::new(
        manonica_error::ErrorCode::InvalidState,
        "simulation_worker_stopped",
    )
}

const SNAPSHOT_PERIOD: Duration = Duration::from_millis(50);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeBackend {
    Cpu,
    Gpu,
}
impl ComputeBackend {
    pub fn label(self) -> &'static str {
        match self {
            Self::Cpu => "CPU",
            Self::Gpu => "GPU",
        }
    }
}
#[derive(Debug, Clone, Copy)]
pub struct ComputeTelemetry {
    pub backend: ComputeBackend,
    /// True if auto selection tried GPU and switched to CPU.
    pub cpu_fallback: bool,
    /// End-to-end GPU operator wall time, including upload and readback.
    pub batch_wall_ms: f64,
    pub batch_steps: u32,
}
impl Default for ComputeTelemetry {
    fn default() -> Self {
        Self {
            backend: ComputeBackend::Cpu,
            cpu_fallback: false,
            batch_wall_ms: 0.0,
            batch_steps: 0,
        }
    }
}

#[derive(Clone)]
pub struct SimulationSnapshot {
    pub revision: u64,
    pub mechanics: Option<super::MechanicsObservation>,
    pub field: Arc<FreeManaField>,
    pub elapsed: f64,
    pub steps: u64,
    pub total: f64,
    pub bound: f64,
    pub formula_bound: Vec<f64>,
    pub material_energy: f64,
    pub potential_peak: f64,
    pub field_energy_joules: f64,
    pub field_heat_joules: f64,
    pub field_work_joules: f64,
    pub potential_transport: f64,
    pub trap_temperature_kelvin: f64,
    pub exported_heat_joules: f64,
    pub field_shutdown: bool,
    pub equilibrium: super::EquilibriumState,
    pub sim_cpu_ms: f64,
    pub compute: ComputeTelemetry,
    pub paused: bool,
    pub error: Option<String>,
    pub captured_at: Instant,
}

impl SimulationSnapshot {
    fn capture(
        sim: &Simulation,
        cpu_ms: f64,
        paused: bool,
        error: Option<String>,
        compute: ComputeTelemetry,
    ) -> Self {
        Self {
            revision: sim.revision,
            mechanics: sim.mechanics_observation(),
            field: Arc::new(sim.field().clone()),
            elapsed: sim.elapsed(),
            steps: sim.steps(),
            total: sim.total().map_or(f64::NAN, |amount| amount.value()),
            bound: sim
                .material_balance()
                .map_or(f64::NAN, |balance| balance.bound),
            formula_bound: sim.formula_bound().unwrap_or_default(),
            material_energy: sim
                .material_balance()
                .map_or(f64::NAN, |balance| balance.energy()),
            potential_peak: sim.field_ledger().map_or(0.0, |p| p.peak_potential),
            field_energy_joules: sim.field_ledger().map_or(0.0, |p| p.field_energy_joules),
            field_heat_joules: sim.field_ledger().map_or(0.0, |p| p.dissipated_heat_joules),
            field_work_joules: sim.field_ledger().map_or(0.0, |p| p.operating_work_joules),
            potential_transport: sim.field_ledger().map_or(0.0, |p| p.last_transport_manons),
            trap_temperature_kelvin: sim.peak_trap_temperature_kelvin().unwrap_or(0.0),
            exported_heat_joules: sim.field_ledger().map_or(0.0, |p| p.exported_heat_joules),
            field_shutdown: sim.shutdown_active(),
            equilibrium: sim.equilibrium_state(),
            sim_cpu_ms: cpu_ms,
            compute,
            paused,
            error,
            captured_at: Instant::now(),
        }
    }
}

#[derive(Debug)]
enum Command {
    Pause(bool),
    Reset,
    Replace {
        revision: u64,
        experiment: Box<Experiment>,
    },
}

/// Owned by the app; only the worker owns `Simulation`.
/// Control messages are asynchronous; `poll_latest` never blocks.
pub struct SimulationWorker {
    commands: Option<Sender<Command>>,
    snapshots: Receiver<SimulationSnapshot>,
    thread: Option<JoinHandle<()>>,
    stop: Arc<AtomicBool>,
    next_revision: AtomicU64,
    configuration_results: Receiver<ConfigurationResult>,
}

impl SimulationWorker {
    pub fn spawn(simulation: Simulation) -> (Self, SimulationSnapshot) {
        let backend =
            BackendPreference::resolve(None, std::env::var("MANONICA_SIM_BACKEND").ok().as_deref())
                .expect("Invalid MANONICA_SIM_BACKEND");
        Self::spawn_with_backend(simulation, backend)
    }

    pub fn spawn_with_backend(
        simulation: Simulation,
        backend: BackendPreference,
    ) -> (Self, SimulationSnapshot) {
        let initial =
            SimulationSnapshot::capture(&simulation, 0.0, false, None, ComputeTelemetry::default());
        let (commands_tx, commands_rx) = mpsc::channel();
        let (snapshots_tx, snapshots_rx) = mpsc::sync_channel(1);
        let (result_tx, result_rx) = mpsc::channel();
        let stop = Arc::new(AtomicBool::new(false));
        let worker_stop = Arc::clone(&stop);
        let handle = thread::Builder::new()
            .name("manonica-physics".into())
            .spawn(move || {
                run_simulation(
                    simulation,
                    backend,
                    RuntimeChannels {
                        commands: commands_rx,
                        updates: snapshots_tx,
                        results: result_tx,
                        stop: worker_stop,
                    },
                )
            })
            .expect("failed to start physics worker");
        (
            Self {
                commands: Some(commands_tx),
                snapshots: snapshots_rx,
                thread: Some(handle),
                stop,
                next_revision: AtomicU64::new(1),
                configuration_results: result_rx,
            },
            initial,
        )
    }

    pub fn pause(&self, pause: bool) {
        if let Some(tx) = &self.commands {
            let _ = tx.send(Command::Pause(pause));
        }
    }

    pub fn reset(&self) {
        if let Some(tx) = &self.commands {
            let _ = tx.send(Command::Reset);
        }
    }

    pub fn replace_experiment(
        &self,
        experiment: Experiment,
    ) -> Result<u64, manonica_error::BaseError> {
        let revision = self.next_revision.fetch_add(1, Ordering::Relaxed);
        self.commands
            .as_ref()
            .ok_or_else(worker_stopped)?
            .send(Command::Replace {
                revision,
                experiment: Box::new(experiment),
            })
            .map_err(|_| worker_stopped())?;
        Ok(revision)
    }
    pub fn poll_configuration(&self) -> Option<ConfigurationResult> {
        self.configuration_results.try_recv().ok()
    }
    pub fn request_shutdown(&mut self) {
        self.stop.store(true, Ordering::Release);
        self.commands.take();
    }
    pub fn is_finished(&self) -> bool {
        self.thread.as_ref().is_none_or(JoinHandle::is_finished)
    }

    pub fn poll_latest(&self) -> Option<SimulationSnapshot> {
        let mut newest = None;
        while let Ok(snapshot) = self.snapshots.try_recv() {
            newest = Some(snapshot);
        }
        newest
    }
}

impl Drop for SimulationWorker {
    fn drop(&mut self) {
        // Closing the control channel wakes recv_timeout/recv, and the join
        // ensures the physics thread cannot outlive its owning application.
        self.request_shutdown();
        if let Some(handle) = self.thread.take() {
            let _ = handle.join();
        }
    }
}

#[cfg(test)]
mod tests;
