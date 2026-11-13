use super::*;
use crate::experiment::ExperimentBuilder;

#[test]
fn backend_labels_are_explicit_and_unambiguous() {
    assert_eq!(ComputeBackend::Cpu.label(), "CPU");
    assert_eq!(ComputeBackend::Gpu.label(), "GPU");
    assert!(!ComputeTelemetry::default().cpu_fallback);
    assert_eq!(ComputeTelemetry::default().batch_wall_ms, 0.0);
}

#[test]
fn physics_advances_independently_of_renderer_then_pauses() {
    let exp = ExperimentBuilder::new("worker")
        .grid([8, 8, 8], 0.5)
        .diffusion(0.15, 0.01, 2)
        .gaussian([0.0; 3], 200.0, 0.6)
        .build()
        .unwrap();
    let (worker, initial) =
        SimulationWorker::spawn_with_backend(Simulation::new(exp).unwrap(), BackendPreference::Cpu);
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
    let (worker, _) =
        SimulationWorker::spawn_with_backend(Simulation::new(exp).unwrap(), BackendPreference::Cpu);
    drop(worker);
}

fn wait_configuration(worker: &SimulationWorker) -> ConfigurationResult {
    let deadline = Instant::now() + Duration::from_secs(2);
    while Instant::now() < deadline {
        if let Some(result) = worker.poll_configuration() {
            return result;
        }
        thread::sleep(Duration::from_millis(5));
    }
    panic!("configuration acknowledgement was lost");
}
fn small_experiment(total: f64) -> Experiment {
    ExperimentBuilder::new("configuration transaction")
        .grid([8, 6, 4], 0.5)
        .diffusion(0.1, 0.01, 4)
        .gaussian([0.0; 3], total, 0.6)
        .build()
        .unwrap()
}
#[test]
fn replacement_acknowledgement_survives_a_full_snapshot_queue() {
    let (worker, _) = SimulationWorker::spawn_with_backend(
        Simulation::new(small_experiment(100.0)).unwrap(),
        BackendPreference::Cpu,
    );
    thread::sleep(Duration::from_millis(70));
    let revision = worker.replace_experiment(small_experiment(250.0)).unwrap();
    match wait_configuration(&worker) {
        ConfigurationResult::Applied {
            revision: accepted,
            snapshot,
        } => {
            assert_eq!(accepted, revision);
            assert_eq!(snapshot.revision, revision);
            assert!((snapshot.total - 250.0).abs() < 1e-8);
            assert_eq!(snapshot.steps, 0);
        }
        ConfigurationResult::Rejected { message, .. } => {
            panic!("valid replacement rejected: {message}")
        }
    }
}
#[test]
fn invalid_replacement_does_not_pause_or_replace_the_live_simulation() {
    let (worker, _) = SimulationWorker::spawn_with_backend(
        Simulation::new(small_experiment(100.0)).unwrap(),
        BackendPreference::Cpu,
    );
    let mut invalid = small_experiment(250.0);
    invalid.physics.dt = -0.1;
    let revision = worker.replace_experiment(invalid).unwrap();
    assert!(
        matches!(wait_configuration(&worker),ConfigurationResult::Rejected {revision: rejected,..} if rejected==revision)
    );
    thread::sleep(Duration::from_millis(80));
    let snapshot = worker.poll_latest().unwrap();
    assert_eq!(snapshot.revision, 0);
    assert!(!snapshot.paused);
    assert!(snapshot.error.is_none());
    assert!((snapshot.total - 100.0).abs() < 1e-8);
}
#[test]
fn explicit_shutdown_wakes_a_paused_worker_without_blocking_the_caller() {
    let (mut worker, _) = SimulationWorker::spawn_with_backend(
        Simulation::new(small_experiment(100.0)).unwrap(),
        BackendPreference::Cpu,
    );
    worker.pause(true);
    thread::sleep(Duration::from_millis(30));
    let started = Instant::now();
    worker.request_shutdown();
    assert!(started.elapsed() < Duration::from_millis(100));
    let deadline = Instant::now() + Duration::from_secs(1);
    while !worker.is_finished() && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(5));
    }
    assert!(worker.is_finished());
    assert!(worker.replace_experiment(small_experiment(250.0)).is_err());
}

#[test]
fn mechanical_worker_pause_reset_replace_and_shutdown_preserve_lifecycle() {
    let experiment =
        Experiment::parse(include_str!("../../../experiments/dynamic_mechanics.toml")).unwrap();
    let (mut worker, initial) = SimulationWorker::spawn_with_backend(
        Simulation::new(experiment.clone()).unwrap(),
        BackendPreference::Cpu,
    );
    worker.pause(true);
    let wait_snapshot = |predicate: &dyn Fn(&SimulationSnapshot) -> bool| {
        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            if let Some(snapshot) = worker.poll_latest()
                && predicate(&snapshot)
            {
                break snapshot;
            }
            assert!(Instant::now() < deadline, "mechanics snapshot timeout");
            thread::sleep(Duration::from_millis(5));
        }
    };
    let paused = wait_snapshot(&|s| s.paused);
    thread::sleep(Duration::from_millis(60));
    let stable = wait_snapshot(&|s| s.paused);
    assert_eq!(stable.steps, paused.steps);
    worker.reset();
    let reset = wait_snapshot(&|s| s.paused && s.steps == 0);
    assert_eq!(
        reset.mechanics.unwrap().positions,
        initial.mechanics.unwrap().positions
    );
    let mut replacement = experiment;
    replacement.mechanics.as_mut().unwrap().bodies[0].position[0] = -1.2;
    let revision = worker.replace_experiment(replacement).unwrap();
    match wait_configuration(&worker) {
        ConfigurationResult::Applied {
            revision: accepted,
            snapshot,
        } => {
            assert_eq!(accepted, revision);
            assert_eq!(snapshot.mechanics.as_ref().unwrap().positions[0][0], -1.2);
            assert_eq!(snapshot.steps, 0);
        }
        ConfigurationResult::Rejected { message, .. } => {
            panic!("mechanics replacement rejected: {message}")
        }
    }
    worker.request_shutdown();
    let deadline = Instant::now() + Duration::from_secs(2);
    while !worker.is_finished() && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(5));
    }
    assert!(worker.is_finished());
}

#[test]
#[ignore = "requires a Vulkan compute adapter"]
fn mechanical_gpu_worker_reset_and_replace_recreate_the_device_state() {
    let experiment =
        Experiment::parse(include_str!("../../../experiments/dynamic_mechanics.toml")).unwrap();
    let (mut worker, initial) = SimulationWorker::spawn_with_backend(
        Simulation::new(experiment.clone()).unwrap(),
        BackendPreference::Gpu,
    );
    worker.pause(true);
    let wait_snapshot = |predicate: &dyn Fn(&SimulationSnapshot) -> bool| {
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            if let Some(snapshot) = worker.poll_latest()
                && predicate(&snapshot)
            {
                break snapshot;
            }
            assert!(Instant::now() < deadline, "GPU mechanics snapshot timeout");
            thread::sleep(Duration::from_millis(5));
        }
    };
    let paused = wait_snapshot(&|s| s.paused && s.compute.backend == ComputeBackend::Gpu);
    assert!(!paused.compute.cpu_fallback);
    assert!(paused.error.is_none());
    worker.reset();
    let reset = wait_snapshot(&|s| s.paused && s.steps == 0);
    assert_eq!(reset.compute.backend, ComputeBackend::Gpu);
    assert_eq!(
        reset.mechanics.unwrap().positions,
        initial.mechanics.unwrap().positions
    );
    let mut next = experiment;
    next.mechanics.as_mut().unwrap().bodies[0].position[0] = -1.2;
    let revision = worker.replace_experiment(next).unwrap();
    match wait_configuration(&worker) {
        ConfigurationResult::Applied {
            revision: accepted,
            snapshot,
        } => {
            assert_eq!(accepted, revision);
            assert_eq!(snapshot.compute.backend, ComputeBackend::Gpu);
            assert_eq!(snapshot.mechanics.as_ref().unwrap().positions[0][0], -1.2);
            assert_eq!(snapshot.steps, 0);
        }
        ConfigurationResult::Rejected { message, .. } => {
            panic!("GPU mechanics replacement rejected: {message}")
        }
    }
    worker.request_shutdown();
    let deadline = Instant::now() + Duration::from_secs(5);
    while !worker.is_finished() && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(5));
    }
    assert!(worker.is_finished());
}
