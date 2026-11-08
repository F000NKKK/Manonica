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
use meridian_gpu_wgpu::{Buffer, BufferUsage, ComputePipeline};
use meridian_sdk::GpuComputeDevice;
use std::time::Instant;

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
        let dimensions = [
            field.shape().nx(),
            field.shape().ny(),
            field.shape().nz(),
        ];
        let len = field.shape().len();
        let input = raw.create_buffer(len * 4, BufferUsage::Storage);
        let output = raw.create_buffer(len * 4, BufferUsage::Storage);
        let parameters = raw.create_buffer(16, BufferUsage::Storage);
        Ok(Self {
            runtime, device, pipeline, input, output, parameters, dimensions,
            last_wall_ms: 0.0,
        })
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
        let layout = self.pipeline.wgpu_pipeline().get_bind_group_layout(0);
        let bind = raw.create_bind_group(
            &layout, &[&self.input, &self.output, &self.parameters],
        );
        let mut commands = raw.create_command_buffer();
        commands.dispatch_compute_with_bind_group(
            &self.pipeline, &bind, (shape.len() as u32).div_ceil(64),
        );
        commands.submit();
        let result = self.runtime.block_on(self.device.read_buffer(&self.output));
        if result.len() != shape.len() * 4 {
            return Err(invalid("gpu_diffusion_readback"));
        }
        let densities: Vec<f64> = result.chunks_exact(4)
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
        let updated = densities.into_iter()
            .map(|n| Quantity::<ManonDensity>::nonnegative(n.max(0.0) * mass_scale))
            .collect::<Result<Vec<_>, _>>()?;
        field.try_replace_densities(updated)?;
        self.last_wall_ms = started.elapsed().as_secs_f64() * 1000.0;
        Ok(())
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
        let shape = GridShape::new(9, 7, 5).unwrap();
        let mut cpu = FreeManaField::uniform(
            shape, 0.5, Quantity::<ManonDensity>::nonnegative(0.0).unwrap(),
        ).unwrap();
        cpu.set(3, 3, 2, Quantity::nonnegative(100.0).unwrap()).unwrap();
        cpu.set(6, 4, 3, Quantity::nonnegative(45.0).unwrap()).unwrap();
        let mut accelerated = cpu.clone();
        let Some(mut gpu) = GpuDiffusion::new(&accelerated).ok() else {
            eprintln!("Skipping hardware parity: no GPU compute adapter in CI");
            return;
        };
        let before = accelerated.total_manons().unwrap().value();
        let diffusivity = 0.25;
        let dt = 0.02;
        FreeDiffusion::new(Quantity::<Diffusivity>::nonnegative(diffusivity).unwrap())
            .unwrap()
            .evolve(&mut cpu, TimeStep::from_seconds(dt).unwrap())
            .unwrap();
        gpu.evolve(&mut accelerated, diffusivity, dt).unwrap();
        for (i, (expected, actual)) in cpu.densities().iter()
            .zip(accelerated.densities()).enumerate()
        {
            assert!(
                (expected.value() - actual.value()).abs() <= 4e-5,
                "GPU/CPU stencil mismatch at cell {i}: {} vs {}",
                expected.value(), actual.value(),
            );
        }
        assert!(
            (accelerated.total_manons().unwrap().value() - before).abs() < 1e-7,
            "GPU solver must conserve physical manon count",
        );
    }
}
