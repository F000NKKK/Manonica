//! Fixed-step coordination, transactional configuration and graceful stop.
use super::{
    BackendPreference, Command, ComputeBackend, ComputeTelemetry, ConfigurationResult,
    SNAPSHOT_PERIOD, Simulation, SimulationSnapshot,
};
use manonica_compute_gpu::GpuCoupled;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
    mpsc::{Receiver, RecvTimeoutError, Sender, SyncSender},
};
use std::time::{Duration, Instant};

fn publish(
    tx: &SyncSender<SimulationSnapshot>,
    sim: &Simulation,
    cpu_ms: f64,
    paused: bool,
    error: &Option<String>,
    compute: ComputeTelemetry,
) {
    // A delayed renderer may keep the previous snapshot. Never block physics
    // on the graphics thread or queue unlimited full-grid copies.
    let snapshot = SimulationSnapshot::capture(sim, cpu_ms, paused, error.clone(), compute);
    let _ = tx.try_send(snapshot);
}

pub(super) struct RuntimeChannels {
    pub(super) commands: Receiver<Command>,
    pub(super) updates: SyncSender<SimulationSnapshot>,
    pub(super) results: Sender<ConfigurationResult>,
    pub(super) stop: Arc<AtomicBool>,
}

pub(super) fn run_simulation(
    mut sim: Simulation,
    backend: BackendPreference,
    channels: RuntimeChannels,
) {
    let RuntimeChannels {
        commands,
        updates,
        results,
        stop,
    } = channels;
    let mut period = Duration::from_secs_f64(sim.experiment.physics.dt);
    let mut max_catch_up = sim.experiment.physics.steps_per_frame;
    let mut paused = false;
    let mut error: Option<String> = None;
    let mut cpu_ms = 0.0;
    let create_gpu = |sim: &Simulation| -> Result<Option<GpuCoupled>, String> {
        if sim.has_mechanics() {
            if backend == BackendPreference::Gpu {
                return Err("dynamic mechanics requires --backend cpu or auto".into());
            }
            return Ok(None);
        }
        if !backend.attempts_gpu() {
            return Ok(None);
        }
        match GpuCoupled::new(sim.coupled_state(), &sim.step_parameters()) {
            Ok(device) => {
                eprintln!(
                    "Manonica: resident GPU physics selected: {}",
                    device.adapter_name()
                );
                Ok(Some(device))
            }
            Err(error) if backend.allows_cpu_fallback() => {
                eprintln!(
                    "Manonica: GPU initialization unavailable ({error}); falling back to CPU"
                );
                Ok(None)
            }
            Err(error) => Err(error),
        }
    };
    if stop.load(Ordering::Acquire) {
        return;
    }
    let mut gpu = create_gpu(&sim).unwrap_or_else(|error| {
        eprintln!("Manonica: --backend gpu requires a working GPU: {error}");
        std::process::exit(2)
    });
    let mut next_tick = Instant::now() + period;
    let mut next_snapshot = Instant::now() + SNAPSHOT_PERIOD;
    let mut compute = ComputeTelemetry {
        backend: if gpu.is_some() {
            ComputeBackend::Gpu
        } else {
            ComputeBackend::Cpu
        },
        cpu_fallback: gpu.is_none() && backend.attempts_gpu() && !sim.has_mechanics(),
        batch_wall_ms: 0.0,
        batch_steps: 0,
    };

    while !stop.load(Ordering::Acquire) {
        let now = Instant::now();
        let deadline = if paused || gpu.is_some() {
            next_snapshot
        } else {
            next_tick.min(next_snapshot)
        };
        let wait = deadline.saturating_duration_since(now);
        let command = commands.recv_timeout(wait);
        if stop.load(Ordering::Acquire) {
            break;
        }
        match command {
            Ok(Command::Pause(next)) => {
                paused = next;
                next_tick = Instant::now() + period;
                publish(&updates, &sim, cpu_ms, paused, &error, compute);
            }
            Ok(Command::Reset) => {
                match Simulation::new_cancellable(sim.experiment.clone(), &|| {
                    stop.load(Ordering::Acquire)
                }) {
                    Ok(mut next) => {
                        if stop.load(Ordering::Acquire) {
                            break;
                        }
                        next.revision = sim.revision;
                        match create_gpu(&next) {
                            Ok(device) => {
                                sim = next;
                                gpu = device;
                                error = None;
                                cpu_ms = 0.0;
                                compute.backend = if gpu.is_some() {
                                    ComputeBackend::Gpu
                                } else {
                                    ComputeBackend::Cpu
                                };
                                compute.cpu_fallback =
                                    gpu.is_none() && backend.attempts_gpu() && !sim.has_mechanics();
                                compute.batch_wall_ms = 0.0;
                                compute.batch_steps = 0;
                                next_tick = Instant::now() + period;
                            }
                            Err(err) => {
                                error = Some(err);
                                paused = true;
                            }
                        }
                    }
                    Err(err) => {
                        error = Some(err.to_string());
                        paused = true;
                    }
                }
                publish(&updates, &sim, cpu_ms, paused, &error, compute);
            }
            Ok(Command::Replace {
                revision,
                experiment,
            }) => {
                let candidate =
                    Simulation::new_cancellable(*experiment, &|| stop.load(Ordering::Acquire))
                        .map_err(|err| err.to_string())
                        .and_then(|mut next| {
                            if stop.load(Ordering::Acquire) {
                                return Err("configuration cancelled".into());
                            }
                            next.revision = revision;
                            create_gpu(&next).map(|device| (next, device))
                        });
                if stop.load(Ordering::Acquire) {
                    break;
                }
                match candidate {
                    Ok((next, device)) => {
                        sim = next;
                        gpu = device;
                        compute.backend = if gpu.is_some() {
                            ComputeBackend::Gpu
                        } else {
                            ComputeBackend::Cpu
                        };
                        compute.cpu_fallback =
                            gpu.is_none() && backend.attempts_gpu() && !sim.has_mechanics();
                        compute.batch_wall_ms = 0.0;
                        compute.batch_steps = 0;
                        period = Duration::from_secs_f64(sim.experiment.physics.dt);
                        max_catch_up = sim.experiment.physics.steps_per_frame;
                        error = None;
                        paused = false;
                        cpu_ms = 0.0;
                        next_tick = Instant::now() + period;
                        next_snapshot = Instant::now() + SNAPSHOT_PERIOD;
                        let snapshot = SimulationSnapshot::capture(&sim, 0.0, false, None, compute);
                        let _ = results.send(ConfigurationResult::Applied {
                            revision,
                            snapshot: Box::new(snapshot),
                        });
                        publish(&updates, &sim, cpu_ms, paused, &error, compute);
                    }
                    Err(message) => {
                        let _ = results.send(ConfigurationResult::Rejected { revision, message });
                    }
                }
            }
            Err(RecvTimeoutError::Disconnected) => break,
            Err(RecvTimeoutError::Timeout) => {}
        }

        let now = Instant::now();
        if !paused {
            let mut ran = 0;
            // Bound submission length so overdue physics cannot monopolize
            // the rendering device with an ever-growing catch-up batch.
            let catch_up_limit = if gpu.is_some() {
                ((SNAPSHOT_PERIOD.as_secs_f64() / period.as_secs_f64()).ceil() as usize)
                    .max(max_catch_up)
                    .clamp(1, 64)
            } else {
                max_catch_up
            };
            while now >= next_tick && ran < catch_up_limit && !stop.load(Ordering::Acquire) {
                let started = Instant::now();
                // GPU runs the complete coupled interval, with one readback
                // at the observation cadence rather than each physics step.
                let overdue = (now.duration_since(next_tick).as_secs_f64() / period.as_secs_f64())
                    .floor() as usize
                    + 1;
                let steps = if gpu.is_some() {
                    overdue.min(catch_up_limit - ran)
                } else {
                    1
                };
                let result = if let Some(device) = gpu.as_mut() {
                    sim.step_coupled_gpu(device, steps as u32)
                } else {
                    sim.step()
                };
                if let Some(device) = gpu.as_ref() {
                    compute.batch_wall_ms = device.last_wall_ms();
                    compute.batch_steps = steps as u32;
                }
                if let Err(err) = result {
                    if gpu.is_some() && backend.allows_cpu_fallback() {
                        eprintln!("Manonica: GPU step failed ({err}); retrying batch on CPU");
                        gpu = None;
                        compute.backend = ComputeBackend::Cpu;
                        compute.cpu_fallback = true;
                        compute.batch_wall_ms = 0.0;
                        compute.batch_steps = 0;
                        // GPU candidate is transactional. On fallback replay
                        // *every* step, not only the first.
                        for _ in 0..steps {
                            if let Err(cpu_error) = sim.step() {
                                error = Some(cpu_error.to_string());
                                paused = true;
                                break;
                            }
                        }
                        if paused {
                            break;
                        }
                    } else if gpu.is_some() && backend == BackendPreference::Gpu {
                        eprintln!("Manonica: forced GPU backend failed: {err}");
                        std::process::exit(3);
                    } else {
                        error = Some(err.to_string());
                        paused = true;
                        break;
                    }
                }
                // Per-simulated-step wall time, including GPU synchronization.
                cpu_ms = started.elapsed().as_secs_f64() * 1000.0 / steps as f64;
                ran += steps;
                next_tick += period * steps as u32;
            }
            if now >= next_tick {
                // Drop accumulated debt rather than turning every slow
                // simulation frame into an unbounded burst of solver work.
                next_tick = Instant::now() + period;
            }
        }
        if now >= next_snapshot || paused {
            publish(&updates, &sim, cpu_ms, paused, &error, compute);
            next_snapshot = Instant::now() + SNAPSHOT_PERIOD;
        }
    }
}
