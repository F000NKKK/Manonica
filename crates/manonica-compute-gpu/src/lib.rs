//! Optional GPU-backed, finite-volume free-manon diffusion.
//!
//! This crate is independent of the Manonica Lab application and renderer.
//!
//! This is deliberately a *hybrid* backend: free-field diffusion is
//! evaluated on a headless Meridian/wgpu compute device; bindings,
//! temperature, energy audits and field potential still execute on CPU.
//! GPU f32 results are mass-renormalized to the previous f64 inventory
//! after readback. CPU remains the reference implementation.
use manonica_core::{Quantity, units::ManonDensity};
use manonica_error::{BaseError, ErrorCode};
use manonica_physics::FreeManaField;
use meridian_gpu_wgpu::{BindGroup, Buffer, BufferUsage, ComputePipeline};
use meridian_sdk::GpuComputeDevice;
use std::time::Instant;

mod potential;
pub use potential::GpuPotential;

const SHADER: &str = include_str!("gpu_diffusion.wgsl");

fn invalid(name: &'static str) -> BaseError {
    BaseError::new(ErrorCode::InvalidRange, name)
}

pub struct GpuDiffusion {
    runtime: tokio::runtime::Runtime,
    device: GpuComputeDevice,
    pipeline: ComputePipeline,
    input: Buffer,
    output: Buffer,
    parameters: Buffer,
    bind: BindGroup,
    reverse_bind: BindGroup,
    advection_pipeline: ComputePipeline,
    advection_parameters: Buffer,
    advection_bind: BindGroup,
    dimensions: [usize; 3],
    last_wall_ms: f64,
}

impl GpuDiffusion {
    pub fn new(field: &FreeManaField) -> Result<Self, String> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .map_err(|error| error.to_string())?;
        let device = runtime
            .block_on(GpuComputeDevice::new())
            .map_err(|error| error.to_string())?;
        let raw = device.gpu_driver_device();
        let shader = raw.create_shader("manonica-3d-diffusion", SHADER);
        let pipeline = raw.create_compute_pipeline(&shader, "diffusion");
        let dimensions = [field.shape().nx(), field.shape().ny(), field.shape().nz()];
        let len = field.shape().len();
        let input = raw.create_buffer(len * 4, BufferUsage::Storage);
        let output = raw.create_buffer(len * 4, BufferUsage::Storage);
        let parameters = raw.create_buffer(16, BufferUsage::Storage);
        let layout = pipeline.wgpu_pipeline().get_bind_group_layout(0);
        let bind = raw.create_bind_group(&layout, &[&input, &output, &parameters]);
        let reverse_bind = raw.create_bind_group(&layout, &[&output, &input, &parameters]);
        let advection_shader = raw.create_shader(
            "manonica-uniform-advection",
            include_str!("gpu_advection.wgsl"),
        );
        let advection_pipeline =
            raw.create_compute_pipeline(&advection_shader, "advect_uniform");
        let advection_parameters = raw.create_buffer(24, BufferUsage::Storage);
        let advection_layout = advection_pipeline.wgpu_pipeline().get_bind_group_layout(0);
        // The advection step reads the diffusion output and writes the input
        // buffer; both physical operations happen in one GPU submission.
        let advection_bind = raw.create_bind_group(
            &advection_layout,
            &[&output, &input, &advection_parameters],
        );
        Ok(Self {
            runtime,
            device,
            pipeline,
            input,
            output,
            parameters,
            bind,
            reverse_bind,
            advection_pipeline,
            advection_parameters,
            advection_bind,
            dimensions,
            last_wall_ms: 0.0,
        })
    }

    /// Execute several *diffusion-only* steps on persistent GPU buffers,
    /// without CPU↔GPU round trips between steps. No other physical operators
    /// may be skipped: callers must use this only for a diffusion-only scenario.
    /// A full multi-physics resident pipeline is a separate milestone.
    pub fn evolve_resident_batch(
        &mut self,
        field: &mut FreeManaField,
        diffusivity: f64,
        dt: f64,
        steps: u32,
    ) -> Result<(), BaseError> {
        let started = Instant::now();
        let shape = field.shape();
        if [shape.nx(), shape.ny(), shape.nz()] != self.dimensions || steps == 0 {
            return Err(invalid("gpu_resident_batch_parameters"));
        }
        let dx = field.cell_width_metres();
        let alpha = diffusivity * dt / dx.powi(2);
        if !alpha.is_finite() || !(0.0..=1.0 / 6.0).contains(&alpha) || alpha > f32::MAX as f64 {
            return Err(invalid("gpu_diffusion_cfl"));
        }
        let source: Vec<f32> = field.densities().iter().map(|n| n.value() as f32).collect();
        if source.iter().any(|v| !v.is_finite() || *v < 0.0) {
            return Err(invalid("gpu_diffusion_input"));
        }
        let before = field.total_manons()?.value();
        let raw = self.device.gpu_driver_device();
        let mut bytes = Vec::with_capacity(source.len() * 4);
        for n in source {
            bytes.extend_from_slice(&n.to_le_bytes());
        }
        raw.write_buffer(&self.input, &bytes);
        let mut params = Vec::with_capacity(16);
        for n in self.dimensions {
            params.extend_from_slice(&(n as u32).to_le_bytes());
        }
        params.extend_from_slice(&(alpha as f32).to_bits().to_le_bytes());
        raw.write_buffer(&self.parameters, &params);
        let workgroups = (shape.len() as u32).div_ceil(64);
        let mut commands = raw.create_command_buffer();
        for step in 0..steps {
            commands.dispatch_compute_with_bind_group(
                &self.pipeline,
                if step % 2 == 0 {
                    &self.bind
                } else {
                    &self.reverse_bind
                },
                workgroups,
            );
        }
        commands.submit();
        let last = if steps % 2 == 0 {
            &self.input
        } else {
            &self.output
        };
        let result = self.runtime.block_on(self.device.read_buffer(last));
        if result.len() != shape.len() * 4 {
            return Err(invalid("gpu_diffusion_readback"));
        }
        let densities: Vec<f64> = result
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64)
            .collect();
        if densities.iter().any(|v| !v.is_finite() || *v < -1e-5) {
            return Err(invalid("gpu_diffusion_result"));
        }
        let total = densities.iter().sum::<f64>() * dx.powi(3);
        if !total.is_finite()
            || (before > 0.0 && total <= 0.0)
            || (before > 0.0 && ((before - total) / before).abs() > 5e-4)
        {
            return Err(invalid("gpu_diffusion_mass"));
        }
        let factor = if total > 0.0 { before / total } else { 1.0 };
        let updated = densities
            .into_iter()
            .map(|n| Quantity::<ManonDensity>::nonnegative(n.max(0.0) * factor))
            .collect::<Result<Vec<_>, _>>()?;
        field.try_replace_densities(updated)?;
        self.last_wall_ms = started.elapsed().as_secs_f64() * 1000.0;
        Ok(())
    }

    /// Fuses finite-volume diffusion with conservative upwind uniform
    /// advection. No intermediate CPU readback, preserving operator order.
    pub fn evolve_uniform_flow(
        &mut self,
        field: &mut FreeManaField,
        diffusivity: f64,
        dt: f64,
        velocity: [f64; 3],
    ) -> Result<(), BaseError> {
        let started = Instant::now();
        let shape = field.shape();
        if [shape.nx(), shape.ny(), shape.nz()] != self.dimensions {
            return Err(invalid("gpu_flow_shape"));
        }
        let dx = field.cell_width_metres();
        let alpha = diffusivity * dt / dx.powi(2);
        let fractions = velocity.map(|v| v * dt / dx);
        if !alpha.is_finite()
            || !(0.0..=1.0 / 6.0).contains(&alpha)
            || fractions.iter().any(|v| !v.is_finite())
            || fractions.iter().map(|v| v.abs()).sum::<f64>() > 1.0
        {
            return Err(invalid("gpu_uniform_flow_cfl"));
        }
        let source: Vec<f32> = field.densities().iter().map(|n| n.value() as f32).collect();
        if source.iter().any(|n| !n.is_finite() || *n < 0.0) {
            return Err(invalid("gpu_uniform_flow_input"));
        }
        let before = field.total_manons()?.value();
        let raw = self.device.gpu_driver_device();
        let mut bytes = Vec::with_capacity(source.len() * 4);
        for number in source { bytes.extend_from_slice(&number.to_le_bytes()); }
        raw.write_buffer(&self.input, &bytes);
        let mut diffusion_params = Vec::with_capacity(16);
        for n in self.dimensions {
            diffusion_params.extend_from_slice(&(n as u32).to_le_bytes());
        }
        diffusion_params.extend_from_slice(&(alpha as f32).to_bits().to_le_bytes());
        raw.write_buffer(&self.parameters, &diffusion_params);
        let mut advection_params = Vec::with_capacity(24);
        for n in self.dimensions {
            advection_params.extend_from_slice(&(n as u32).to_le_bytes());
        }
        for fraction in fractions {
            if !fraction.is_finite() || fraction.abs() > f32::MAX as f64 {
                return Err(invalid("gpu_uniform_flow_precision"));
            }
            advection_params.extend_from_slice(&(fraction as f32).to_bits().to_le_bytes());
        }
        raw.write_buffer(&self.advection_parameters, &advection_params);
        let mut commands = raw.create_command_buffer();
        let workgroups = (shape.len() as u32).div_ceil(64);
        commands.dispatch_compute_with_bind_group(&self.pipeline, &self.bind, workgroups);
        commands.dispatch_compute_with_bind_group(
            &self.advection_pipeline,
            &self.advection_bind,
            workgroups,
        );
        commands.submit();
        let result = self.runtime.block_on(self.device.read_buffer(&self.input));
        if result.len() != shape.len() * 4 {
            return Err(invalid("gpu_uniform_flow_readback"));
        }
        let densities: Vec<f64> = result.chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64)
            .collect();
        if densities.iter().any(|n| !n.is_finite() || *n < -1e-5) {
            return Err(invalid("gpu_uniform_flow_nonfinite"));
        }
        let after = densities.iter().sum::<f64>() * dx.powi(3);
        if !after.is_finite()
            || (before > 0.0 && after <= 0.0)
            || (after - before).abs() > before.max(1.0) * 5e-4
        {
            return Err(invalid("gpu_uniform_flow_conservation"));
        }
        let scale = if after > 0.0 { before / after } else { 1.0 };
        let updated = densities.into_iter()
            .map(|n| Quantity::<ManonDensity>::nonnegative(n.max(0.0) * scale))
            .collect::<Result<Vec<_>, _>>()?;
        field.try_replace_densities(updated)?;
        self.last_wall_ms = started.elapsed().as_secs_f64() * 1000.0;
        Ok(())
    }

    /// Wall-clock upload + compute + readback + validation (not GPU kernel timer).
    pub fn last_wall_ms(&self) -> f64 {
        self.last_wall_ms
    }

    /// Advances a single explicit diffusion step: J=-D grad(n).
    /// The input field is mutated only after the GPU output has been validated.
    pub fn evolve(
        &mut self,
        field: &mut FreeManaField,
        diffusivity: f64,
        dt: f64,
    ) -> Result<(), BaseError> {
        let started = Instant::now();
        let shape = field.shape();
        if [shape.nx(), shape.ny(), shape.nz()] != self.dimensions {
            return Err(invalid("gpu_grid_changed"));
        }
        let dx = field.cell_width_metres();
        let alpha = diffusivity * dt / dx.powi(2);
        if !alpha.is_finite() || !(0.0..=1.0 / 6.0).contains(&alpha) {
            return Err(invalid("gpu_diffusion_cfl"));
        }
        if !alpha.is_finite() || alpha > f32::MAX as f64 {
            return Err(invalid("gpu_diffusion_range"));
        }
        let source: Vec<f32> = field.densities().iter().map(|n| n.value() as f32).collect();
        if source.iter().any(|n| !n.is_finite() || *n < 0.0) {
            return Err(invalid("gpu_diffusion_input"));
        }
        let before = field.total_manons()?.value();
        let raw = self.device.gpu_driver_device();
        let mut bytes = Vec::with_capacity(source.len() * 4);
        for number in &source {
            bytes.extend_from_slice(&number.to_le_bytes());
        }
        raw.write_buffer(&self.input, &bytes);
        let mut settings = Vec::with_capacity(16);
        for n in self.dimensions {
            settings.extend_from_slice(&(n as u32).to_le_bytes());
        }
        settings.extend_from_slice(&(alpha as f32).to_bits().to_le_bytes());
        raw.write_buffer(&self.parameters, &settings);
        let mut commands = raw.create_command_buffer();
        commands.dispatch_compute_with_bind_group(
            &self.pipeline,
            &self.bind,
            (shape.len() as u32).div_ceil(64),
        );
        commands.submit();
        let result = self.runtime.block_on(self.device.read_buffer(&self.output));
        if result.len() != shape.len() * 4 {
            return Err(invalid("gpu_diffusion_readback"));
        }
        let densities: Vec<f64> = result
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64)
            .collect();
        if densities.iter().any(|v| !v.is_finite() || *v < -1e-5) {
            return Err(invalid("gpu_diffusion_result"));
        }
        let total = densities.iter().sum::<f64>() * dx.powi(3);
        if !total.is_finite() || (before > 0.0 && total <= 0.0) {
            return Err(invalid("gpu_diffusion_mass"));
        }
        if before > 0.0 && ((before - total) / before).abs() > 5e-4 {
            return Err(invalid("gpu_diffusion_accuracy"));
        }
        let mass_scale = if total > 0.0 { before / total } else { 1.0 };
        let updated = densities
            .into_iter()
            .map(|n| Quantity::<ManonDensity>::nonnegative(n.max(0.0) * mass_scale))
            .collect::<Result<Vec<_>, _>>()?;
        field.try_replace_densities(updated)?;
        self.last_wall_ms = started.elapsed().as_secs_f64() * 1000.0;
        Ok(())
    }
}

impl manonica_physics::FreeManaDiffusionBackend for GpuDiffusion {
    fn evolve_free_diffusion(
        &mut self,
        field: &mut FreeManaField,
        diffusivity: f64,
        dt_seconds: f64,
    ) -> Result<(), BaseError> {
        self.evolve(field, diffusivity, dt_seconds)
    }

    fn evolve_diffusion_and_uniform_flow(
        &mut self,
        field: &mut FreeManaField,
        diffusivity: f64,
        dt_seconds: f64,
        velocity: [f64; 3],
    ) -> Result<bool, BaseError> {
        self.evolve_uniform_flow(field, diffusivity, dt_seconds, velocity)?;
        Ok(true)
    }

    fn evolve_diffusion_batch(
        &mut self,
        field: &mut FreeManaField,
        diffusivity: f64,
        dt_seconds: f64,
        steps: u32,
    ) -> Result<(), BaseError> {
        self.evolve_resident_batch(field, diffusivity, dt_seconds, steps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use manonica_core::{Evolution, TimeStep, units::Diffusivity};
    use manonica_physics::{FreeDiffusion, GridShape};

    /// Stage-GPU-001: compare a complete 3D GPU diffusion step against
    /// independent f64 CPU finite-volume reference; no display required.
    #[test]
    fn gpu_diffusion_matches_cpu_when_an_adapter_exists() {
        // GitHub-hosted CI has no guaranteed physical compute device.
        // Device parity is required on the target hardware before release.
        if std::env::var_os("CI").is_some()
            && std::env::var("MANONICA_GPU_TEST").as_deref() != Ok("1")
        {
            eprintln!(
                "Skipping hardware GPU parity in CI; set MANONICA_GPU_TEST=1 on a GPU runner"
            );
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
        if std::env::var_os("CI").is_some()
            && std::env::var("MANONICA_GPU_TEST").as_deref() != Ok("1")
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
        if std::env::var_os("CI").is_some()
            && std::env::var("MANONICA_GPU_TEST").as_deref() != Ok("1")
        {
            return;
        }
        let shape = GridShape::new(3, 1, 1).unwrap();
        let mut field = FreeManaField::uniform(
            shape, 1.0, Quantity::<ManonDensity>::nonnegative(0.0).unwrap()
        ).unwrap();
        field.set(0, 0, 0, Quantity::nonnegative(100.0).unwrap()).unwrap();
        let mut gpu = GpuDiffusion::new(&field).expect("GPU compute required");
        let before = field.total_manons().unwrap().value();
        gpu.evolve_uniform_flow(&mut field, 0.0, 0.25, [1.0, 0.0, 0.0]).unwrap();
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
        let total_error = (cpu.total_manons().unwrap().value()
            - accelerated.total_manons().unwrap().value())
        .abs();
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
}
