use super::*;
use manonica_core::{Evolution, TimeStep, units::Diffusivity};
use manonica_physics::{FreeDiffusion, GridShape};

/// Stage-GPU-001: compare a complete 3D GPU diffusion step against
/// independent f64 CPU finite-volume reference; no display required.
#[test]
fn gpu_diffusion_matches_cpu_when_an_adapter_exists() {
    // GitHub-hosted CI has no guaranteed physical compute device.
    // Device parity is required on the target hardware before release.
    if std::env::var_os("CI").is_some() && std::env::var("MANONICA_GPU_TEST").as_deref() != Ok("1")
    {
        eprintln!("Skipping hardware GPU parity in CI; set MANONICA_GPU_TEST=1 on a GPU runner");
        return;
    }
    let shape = GridShape::new(9, 7, 5).unwrap();
    let mut cpu = FreeManaField::uniform(
        shape,
        0.5,
        Quantity::<ManonDensity>::nonnegative(0.0).unwrap(),
    )
    .unwrap();
    cpu.set(3, 3, 2, Quantity::nonnegative(100.0).unwrap())
        .unwrap();
    cpu.set(6, 4, 3, Quantity::nonnegative(45.0).unwrap())
        .unwrap();
    let mut accelerated = cpu.clone();
    let mut gpu = match GpuDiffusion::new(&accelerated) {
        Ok(device) => device,
        Err(error) if std::env::var("MANONICA_GPU_TEST").as_deref() == Ok("1") => {
            panic!("GPU parity explicitly required, but no usable compute device: {error}");
        }
        Err(error) => {
            eprintln!("Skipping optional hardware parity: {error}");
            return;
        }
    };
    let before = accelerated.total_manons().unwrap().value();
    let diffusivity = 0.25;
    let dt = 0.02;
    FreeDiffusion::new(Quantity::<Diffusivity>::nonnegative(diffusivity).unwrap())
        .unwrap()
        .evolve(&mut cpu, TimeStep::from_seconds(dt).unwrap())
        .unwrap();
    gpu.evolve(&mut accelerated, diffusivity, dt).unwrap();
    for (i, (expected, actual)) in cpu
        .densities()
        .iter()
        .zip(accelerated.densities())
        .enumerate()
    {
        assert!(
            (expected.value() - actual.value()).abs() <= 4e-5,
            "GPU/CPU stencil mismatch at cell {i}: {} vs {}",
            expected.value(),
            actual.value(),
        );
    }
    assert!(
        (accelerated.total_manons().unwrap().value() - before).abs() < 1e-7,
        "GPU solver must conserve physical manon count",
    );
}
/// Stage-GPU-002: many device-resident GPU diffusion steps must match
/// a CPU numerical simulation and preserve the total number of manons.
#[test]
fn resident_diffusion_batch_matches_cpu_reference() {
    if std::env::var_os("CI").is_some() && std::env::var("MANONICA_GPU_TEST").as_deref() != Ok("1")
    {
        return;
    }
    let shape = GridShape::new(10, 6, 4).unwrap();
    let mut reference = FreeManaField::uniform(
        shape,
        0.5,
        Quantity::<ManonDensity>::nonnegative(1.0).unwrap(),
    )
    .unwrap();
    reference
        .set(3, 2, 1, Quantity::nonnegative(90.0).unwrap())
        .unwrap();
    let mut actual = reference.clone();
    let mut device = match GpuDiffusion::new(&actual) {
        Ok(device) => device,
        Err(error) if std::env::var("MANONICA_GPU_TEST").as_deref() == Ok("1") => {
            panic!("hardware GPU parity required: {error}")
        }
        Err(_) => return,
    };
    let initial = actual.total_manons().unwrap().value();
    let diffusivity = 0.12;
    let dt = 0.02;
    let solver =
        FreeDiffusion::new(Quantity::<Diffusivity>::nonnegative(diffusivity).unwrap()).unwrap();
    for _ in 0..25 {
        solver
            .evolve(&mut reference, TimeStep::from_seconds(dt).unwrap())
            .unwrap();
    }
    device
        .evolve_resident_batch(&mut actual, diffusivity, dt, 25)
        .unwrap();
    let max_error = reference
        .densities()
        .iter()
        .zip(actual.densities())
        .map(|(a, b)| (a.value() - b.value()).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_error < 1e-4,
        "25-step GPU resident batch mismatch: {max_error}"
    );
    assert!(
        (actual.total_manons().unwrap().value() - initial).abs() < 1e-7,
        "resident batch must preserve total manons"
    );
}

/// LAW-GPU-ADV-001: exact upwind transfer, no exterior leakage.
/// On three unit cells, v=+1, dt=0.25: [100,0,0] -> [75,25,0].
#[test]
fn gpu_uniform_advection_matches_exact_face_flux() {
    if std::env::var_os("CI").is_some() && std::env::var("MANONICA_GPU_TEST").as_deref() != Ok("1")
    {
        return;
    }
    let shape = GridShape::new(3, 1, 1).unwrap();
    let mut field = FreeManaField::uniform(
        shape,
        1.0,
        Quantity::<ManonDensity>::nonnegative(0.0).unwrap(),
    )
    .unwrap();
    field
        .set(0, 0, 0, Quantity::nonnegative(100.0).unwrap())
        .unwrap();
    let mut gpu = GpuDiffusion::new(&field).expect("GPU compute required");
    let before = field.total_manons().unwrap().value();
    gpu.evolve_uniform_flow(&mut field, 0.0, 0.25, [1.0, 0.0, 0.0])
        .unwrap();
    for (x, expected) in [75.0, 25.0, 0.0].into_iter().enumerate() {
        assert!(
            (field.at(x, 0, 0).unwrap().value() - expected).abs() <= 1e-5,
            "upwind conservation failure at x={x}"
        );
    }
    assert!((field.total_manons().unwrap().value() - before).abs() < 1e-6);
}

/// Manual performance comparison; skipped in CI without an adapter.
/// Measures real end-to-end wall time, including every transfer.
#[test]
#[ignore = "run explicitly on an available compute adapter"]
fn benchmark_gpu_vs_cpu_including_transfer_overhead() {
    let shape = GridShape::new(32, 32, 32).unwrap();
    let mut cpu = FreeManaField::uniform(
        shape,
        0.5,
        Quantity::<ManonDensity>::nonnegative(1.0).unwrap(),
    )
    .unwrap();
    for z in 14..18 {
        for y in 14..18 {
            for x in 14..18 {
                cpu.set(x, y, z, Quantity::nonnegative(100.0).unwrap())
                    .unwrap();
            }
        }
    }
    let mut accelerated = cpu.clone();
    let Ok(mut gpu) = GpuDiffusion::new(&accelerated) else {
        eprintln!("GPU benchmark unavailable: no suitable compute adapter");
        return;
    };
    const ITERATIONS: usize = 100;
    let dt = 0.01;
    let d = 0.08;
    let solver = FreeDiffusion::new(Quantity::<Diffusivity>::nonnegative(d).unwrap()).unwrap();
    let start_cpu = Instant::now();
    for _ in 0..ITERATIONS {
        solver
            .evolve(&mut cpu, TimeStep::from_seconds(dt).unwrap())
            .unwrap();
    }
    let cpu_wall = start_cpu.elapsed().as_secs_f64() * 1000.0;
    let start_gpu = Instant::now();
    for _ in 0..ITERATIONS {
        gpu.evolve(&mut accelerated, d, dt).unwrap();
    }
    let gpu_wall = start_gpu.elapsed().as_secs_f64() * 1000.0;
    let maximum_error = cpu
        .densities()
        .iter()
        .zip(accelerated.densities())
        .map(|(a, b)| (a.value() - b.value()).abs())
        .fold(0.0_f64, f64::max);
    let total_error =
        (cpu.total_manons().unwrap().value() - accelerated.total_manons().unwrap().value()).abs();
    println!(
        "{} cells x {} steps: CPU {:.2} ms, GPU {:.2} ms, ratio CPU/GPU {:.3}, max cell error {:.3e}",
        shape.len(),
        ITERATIONS,
        cpu_wall,
        gpu_wall,
        cpu_wall / gpu_wall,
        maximum_error
    );
    assert!(
        maximum_error < 0.001,
        "GPU must remain close to CPU reference"
    );
    assert!(total_error < 1e-6, "GPU must conserve total manons");
}

/// Compare every split step with independent f64 laws, including negative
/// velocities, closed boundaries and both even and odd batch lengths.
#[test]
fn resident_uniform_flow_matches_cpu_operator_split() {
    if std::env::var_os("CI").is_some() && std::env::var("MANONICA_GPU_TEST").as_deref() != Ok("1")
    {
        return;
    }
    use manonica_core::{
        Vec3,
        units::{Metres, MetresPerSecond},
    };
    use manonica_physics::{FreeAdvection, PhysicsError, VelocityField};
    struct Uniform([f64; 3]);
    impl VelocityField for Uniform {
        fn at(&self, _: Vec3<Metres>) -> Result<Vec3<MetresPerSecond>, PhysicsError> {
            Ok(Vec3::try_values(self.0[0], self.0[1], self.0[2])?)
        }
    }
    let shape = GridShape::new(7, 5, 3).unwrap();
    let mut initial = FreeManaField::uniform(
        shape,
        0.5,
        Quantity::<ManonDensity>::nonnegative(1.0).unwrap(),
    )
    .unwrap();
    initial
        .set(0, 4, 0, Quantity::nonnegative(90.0).unwrap())
        .unwrap();
    initial
        .set(5, 2, 1, Quantity::nonnegative(40.0).unwrap())
        .unwrap();
    let mut gpu = GpuDiffusion::new(&initial).expect("GPU compute required");
    let diffusion =
        FreeDiffusion::new(Quantity::<Diffusivity>::nonnegative(0.12).unwrap()).unwrap();
    for steps in [1, 2, 25, 26] {
        for velocity in [[0.3, -0.2, 0.1], [-0.3, 0.2, -0.1]] {
            let mut expected = initial.clone();
            let mut actual = initial.clone();
            let flow = FreeAdvection::new(Uniform(velocity));
            for _ in 0..steps {
                diffusion
                    .evolve(&mut expected, TimeStep::from_seconds(0.02).unwrap())
                    .unwrap();
                flow.evolve(&mut expected, TimeStep::from_seconds(0.02).unwrap())
                    .unwrap();
            }
            gpu.evolve_uniform_flow_batch(&mut actual, 0.12, 0.02, velocity, steps)
                .unwrap();
            for (a, b) in actual.densities().iter().zip(expected.densities()) {
                assert!((a.value() - b.value()).abs() < 2e-4);
            }
            assert!(
                (actual.total_manons().unwrap().value() - initial.total_manons().unwrap().value())
                    .abs()
                    < 1e-7
            );
        }
    }
    for (d, dt, steps) in [
        (0.12, 0.02, 0),
        (-0.12, -0.02, 1),
        (0.12, 0.0, 1),
        (0.12, f64::NAN, 1),
    ] {
        let mut actual = initial.clone();
        assert!(
            gpu.evolve_uniform_flow_batch(&mut actual, d, dt, [0.3, 0.0, 0.0], steps)
                .is_err()
        );
        assert_eq!(actual.densities(), initial.densities());
    }
}
