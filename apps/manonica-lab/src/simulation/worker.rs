//! Fixed-step simulation thread. The window never waits for a physics step.
//! A bounded, lossy snapshot queue keeps memory constant if rendering stalls.

use std::sync::Arc;
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender, SyncSender};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use manonica_physics::FreeManaField;

use super::Simulation;
use crate::experiment::Experiment;

const SNAPSHOT_PERIOD: Duration = Duration::from_millis(50);

#[derive(Clone)]
pub struct SimulationSnapshot {
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
    pub sim_cpu_ms: f64,
    pub paused: bool,
    pub error: Option<String>,
    pub captured_at: Instant,
}

impl SimulationSnapshot {
    fn capture(sim: &Simulation, cpu_ms: f64, paused: bool, error: Option<String>) -> Self {
        Self {
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
            sim_cpu_ms: cpu_ms,
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
    Replace(Experiment),
}

/// Owned by the app; only the worker owns `Simulation`.
/// Control messages are asynchronous; `poll_latest` never blocks.
pub struct SimulationWorker {
    commands: Option<Sender<Command>>,
    snapshots: Receiver<SimulationSnapshot>,
    thread: Option<JoinHandle<()>>,
}

impl SimulationWorker {
    pub fn spawn(simulation: Simulation) -> (Self, SimulationSnapshot) {
        let initial = SimulationSnapshot::capture(&simulation, 0.0, false, None);
        let (commands_tx, commands_rx) = mpsc::channel();
        let (snapshots_tx, snapshots_rx) = mpsc::sync_channel(1);
        let handle = thread::Builder::new()
            .name("manonica-physics".into())
            .spawn(move || run_simulation(simulation, commands_rx, snapshots_tx))
            .expect("failed to start physics worker");
        (
            Self {
                commands: Some(commands_tx),
                snapshots: snapshots_rx,
                thread: Some(handle),
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

    pub fn replace_experiment(&self, experiment: Experiment) {
        if let Some(tx) = &self.commands {
            let _ = tx.send(Command::Replace(experiment));
        }
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
        self.commands.take();
        if let Some(handle) = self.thread.take() {
            let _ = handle.join();
        }
    }
}

fn publish(
    tx: &SyncSender<SimulationSnapshot>,
    sim: &Simulation,
    cpu_ms: f64,
    paused: bool,
    error: &Option<String>,
) {
    // A delayed renderer may keep the previous snapshot. Never block physics
    // on the graphics thread or queue unlimited full-grid copies.
    let snapshot = SimulationSnapshot::capture(sim, cpu_ms, paused, error.clone());
    let _ = tx.try_send(snapshot);
}

fn run_simulation(
    mut sim: Simulation,
    commands: Receiver<Command>,
    updates: SyncSender<SimulationSnapshot>,
) {
    let mut period = Duration::from_secs_f64(sim.experiment.physics.dt);
    let mut max_catch_up = sim.experiment.physics.steps_per_frame;
    let mut next_tick = Instant::now() + period;
    let mut next_snapshot = Instant::now() + SNAPSHOT_PERIOD;
    let mut paused = false;
    let mut error: Option<String> = None;
    let mut cpu_ms = 0.0;

    loop {
        let now = Instant::now();
        let deadline = if paused {
            next_snapshot
        } else {
            next_tick.min(next_snapshot)
        };
        let wait = deadline.saturating_duration_since(now);
        match commands.recv_timeout(wait) {
            Ok(Command::Pause(next)) => {
                paused = next;
                next_tick = Instant::now() + period;
                publish(&updates, &sim, cpu_ms, paused, &error);
            }
            Ok(Command::Reset) => {
                match sim.reset() {
                    Ok(()) => {
                        error = None;
                        cpu_ms = 0.0;
                        next_tick = Instant::now() + period;
                    }
                    Err(err) => {
                        error = Some(err.to_string());
                        paused = true;
                    }
                }
                publish(&updates, &sim, cpu_ms, paused, &error);
            }
            Ok(Command::Replace(experiment)) => match Simulation::new(experiment) {
                Ok(next) => {
                    sim = next;
                    period = Duration::from_secs_f64(sim.experiment.physics.dt);
                    max_catch_up = sim.experiment.physics.steps_per_frame;
                    error = None;
                    paused = false;
                    cpu_ms = 0.0;
                    next_tick = Instant::now() + period;
                    next_snapshot = Instant::now() + SNAPSHOT_PERIOD;
                    publish(&updates, &sim, cpu_ms, paused, &error);
                }
                Err(err) => {
                    error = Some(err.to_string());
                    paused = true;
                    publish(&updates, &sim, cpu_ms, paused, &error);
                }
            },
            Err(RecvTimeoutError::Disconnected) => break,
            Err(RecvTimeoutError::Timeout) => {}
        }

        let now = Instant::now();
        if !paused {
            let mut ran = 0;
            while now >= next_tick && ran < max_catch_up {
                let started = Instant::now();
                if let Err(err) = sim.step() {
                    error = Some(err.to_string());
                    paused = true;
                    break;
                }
                cpu_ms = started.elapsed().as_secs_f64() * 1000.0;
                ran += 1;
                next_tick += period;
            }
            if now >= next_tick {
                // Drop accumulated debt rather than turning every slow
                // simulation frame into an unbounded burst of solver work.
                next_tick = Instant::now() + period;
            }
        }
        if now >= next_snapshot || paused {
            publish(&updates, &sim, cpu_ms, paused, &error);
            next_snapshot = Instant::now() + SNAPSHOT_PERIOD;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experiment::ExperimentBuilder;

    #[test]
    fn physics_advances_independently_of_renderer_then_pauses() {
        let exp = ExperimentBuilder::new("worker")
            .grid([8, 8, 8], 0.5)
            .diffusion(0.15, 0.01, 2)
            .gaussian([0.0; 3], 200.0, 0.6)
            .build()
            .unwrap();
        let (worker, initial) = SimulationWorker::spawn(Simulation::new(exp).unwrap());
        assert_eq!(initial.steps, 0);
        thread::sleep(Duration::from_millis(110));
        let snapshot = worker.poll_latest().expect("worker publishes snapshots");
        assert!(snapshot.steps > 0);
        assert!((snapshot.total - initial.total).abs() < 1e-6);
        worker.pause(true);
        thread::sleep(Duration::from_millis(80));
        let stopped = worker.poll_latest().unwrap();
        assert!(stopped.paused);
        let step = stopped.steps;
        thread::sleep(Duration::from_millis(90));
        assert_eq!(worker.poll_latest().unwrap().steps, step);
        worker.reset();
        thread::sleep(Duration::from_millis(70));
        assert_eq!(worker.poll_latest().unwrap().steps, 0);
    }

    #[test]
    fn dropping_worker_terminates_without_waiting_for_render() {
        let exp = ExperimentBuilder::new("drop")
            .grid([5, 5, 5], 1.0)
            .build()
            .unwrap();
        let (worker, _) = SimulationWorker::spawn(Simulation::new(exp).unwrap());
        drop(worker);
    }
}
