//! Headless numerical validation of resident transport; no window or Lab.
use manonica_compute_gpu::GpuDiffusion;
use manonica_core::{
    Evolution, Quantity, TimeStep, Vec3,
    units::{Diffusivity, ManonDensity, Metres, MetresPerSecond},
};
use manonica_physics::{
    FreeAdvection, FreeDiffusion, FreeManaField, GridShape, PhysicsError, VelocityField,
};
use std::time::Instant;

struct Uniform([f64; 3]);
impl VelocityField for Uniform {
    fn at(&self, _: Vec3<Metres>) -> Result<Vec3<MetresPerSecond>, PhysicsError> {
        Ok(Vec3::try_values(self.0[0], self.0[1], self.0[2])?)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut reference = FreeManaField::uniform(
        GridShape::new(16, 12, 8)?,
        0.5,
        Quantity::<ManonDensity>::nonnegative(1.0)?,
    )?;
    reference.set(3, 4, 2, Quantity::nonnegative(100.0)?)?;
    let mut accelerated = reference.clone();
    let initial = reference.total_manons()?.value();
    let velocity = [0.3, -0.2, 0.1];
    let diffusion = FreeDiffusion::new(Quantity::<Diffusivity>::nonnegative(0.12)?)?;
    let flow = FreeAdvection::new(Uniform(velocity));
    let mut gpu = GpuDiffusion::new(&accelerated).map_err(std::io::Error::other)?;
    let started = Instant::now();
    for _ in 0..100 {
        diffusion.evolve(&mut reference, TimeStep::from_seconds(0.02)?)?;
        flow.evolve(&mut reference, TimeStep::from_seconds(0.02)?)?;
    }
    let cpu_ms = started.elapsed().as_secs_f64() * 1000.0;
    gpu.evolve_uniform_flow_batch(&mut accelerated, 0.12, 0.02, velocity, 100)?;
    let max_error = reference
        .densities()
        .iter()
        .zip(accelerated.densities())
        .map(|(a, b)| (a.value() - b.value()).abs())
        .fold(0.0_f64, f64::max);
    let mass_error = (initial - accelerated.total_manons()?.value()).abs();
    println!(
        "1536 cells, 100 steps: CPU {cpu_ms:.3} ms, GPU with transfers {:.3} ms, max cell error {max_error:.3e}, mass error {mass_error:.3e}",
        gpu.last_wall_ms()
    );
    if max_error > 5e-4 || mass_error > 1e-7 {
        return Err(std::io::Error::other("GPU transport differs from CPU reference").into());
    }
    Ok(())
}
