//! Hardware integration contracts, run explicitly on a Vulkan adapter.
use super::{BackendPreference, ComputeBackend, Simulation, SimulationSnapshot, SimulationWorker};
use crate::experiment::{Experiment, ExperimentBuilder};
use manonica_compute_gpu::GpuCoupled;
use std::time::{Duration, Instant};

#[test]
#[ignore = "requires a Vulkan compute adapter"]
fn coupled_formula_geometry_matches_cpu_reference() {
    let experiment = Experiment::parse(include_str!(
        "../../experiments/local_boundary_conductor.toml"
    ))
    .unwrap();
    let mut cpu = Simulation::new(experiment.clone()).unwrap();
    let mut actual = Simulation::new(experiment).unwrap();
    let mut gpu = GpuCoupled::new(actual.coupled_state(), &actual.step_parameters()).unwrap();
    // Exclude first-use driver compilation from the steady-state sample.
    cpu.step().unwrap();
    actual.step_coupled_gpu(&mut gpu, 1).unwrap();
    for _ in 0..3 {
        cpu.step().unwrap();
    }
    let gpu_start = Instant::now();
    actual.step_coupled_gpu(&mut gpu, 3).unwrap();
    println!(
        "Lab conductor: {} cells, 3 steps, {:.3} ms total, {:.3} ms/step on {}",
        actual.field().shape().len(),
        gpu_start.elapsed().as_secs_f64() * 1000.0,
        gpu.last_wall_ms() / 3.0,
        gpu.adapter_name()
    );
    assert_eq!(actual.steps(), 4);
    assert!((actual.elapsed() - cpu.elapsed()).abs() < 1e-12);
    for (a, b) in cpu
        .field()
        .densities()
        .iter()
        .zip(actual.field().densities())
    {
        assert!((a.value() - b.value()).abs() < a.value().abs().max(1.0) * 2e-4);
    }
    for (a, b) in cpu
        .formula_bound()
        .unwrap()
        .iter()
        .zip(actual.formula_bound().unwrap())
    {
        assert!((a - b).abs() < a.abs().max(1.0) * 2e-4);
    }
    assert!(
        (actual.total().unwrap().value() - cpu.total().unwrap().value()).abs()
            < cpu.total().unwrap().value() * 5e-4
    );
    assert!(
        (actual.tracked_energy().unwrap() - cpu.tracked_energy().unwrap()).abs()
            < cpu.tracked_energy().unwrap().abs().max(1.0) * 2e-3
    );
}

fn wait_for(
    worker: &SimulationWorker,
    predicate: impl Fn(&SimulationSnapshot) -> bool,
) -> SimulationSnapshot {
    let deadline = Instant::now() + Duration::from_secs(30);
    while Instant::now() < deadline {
        if let Some(snapshot) = worker.poll_latest() {
            assert!(
                snapshot.error.is_none(),
                "worker failed: {:?}",
                snapshot.error
            );
            if predicate(&snapshot) {
                return snapshot;
            }
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    panic!("worker did not produce expected state before timeout");
}
#[test]
#[ignore = "requires a Vulkan compute adapter"]
fn resident_worker_resets_device_state_and_resumes() {
    let experiment = ExperimentBuilder::new("resident worker reset")
        .grid([8, 6, 4], 0.5)
        .diffusion(0.12, 0.01, 4)
        .gaussian([0.0; 3], 200.0, 0.6)
        .build()
        .unwrap();
    let (worker, _) = SimulationWorker::spawn_with_backend(
        Simulation::new(experiment).unwrap(),
        BackendPreference::Auto,
    );
    let running = wait_for(&worker, |s| s.steps >= 5);
    assert_eq!(
        running.compute.backend,
        ComputeBackend::Gpu,
        "test must exercise resident GPU path"
    );
    worker.pause(true);
    wait_for(&worker, |s| s.paused);
    worker.reset();
    let reset = wait_for(&worker, |s| s.paused && s.steps == 0);
    assert!((reset.total - 200.0).abs() < 1e-6);
    worker.pause(false);
    let resumed = wait_for(&worker, |s| !s.paused && s.steps > 0);
    assert_eq!(resumed.compute.backend, ComputeBackend::Gpu);
    assert!((resumed.total - 200.0).abs() < 0.1);
}
