//! Uniform upwind transport and its matching WGSL pipeline.
use crate::{backend::GpuDiffusion, diffusion::DiffusionKernel, error::invalid};
use manonica_core::{Quantity, units::ManonDensity};
use manonica_error::BaseError;
use manonica_physics::FreeManaField;
use meridian_gpu_wgpu::{BindGroup, Buffer, BufferUsage, ComputePipeline, Device};
use std::time::Instant;

pub(super) struct AdvectionKernel {
    pub(super) pipeline: ComputePipeline,
    pub(super) parameters: Buffer,
    pub(super) bind: BindGroup,
}
impl AdvectionKernel {
    pub(super) fn new(raw: &Device, diffusion: &DiffusionKernel) -> Self {
        let shader =
            raw.create_shader("manonica-uniform-advection", include_str!("advection.wgsl"));
        let pipeline = raw.create_compute_pipeline(&shader, "advect_uniform");
        let parameters = raw.create_buffer(24, BufferUsage::Storage);
        let layout = pipeline.wgpu_pipeline().get_bind_group_layout(0);
        let bind =
            raw.create_bind_group(&layout, &[&diffusion.output, &diffusion.input, &parameters]);
        Self {
            pipeline,
            parameters,
            bind,
        }
    }
}

impl GpuDiffusion {
    /// Fuses finite-volume diffusion with conservative upwind uniform
    /// advection. No intermediate CPU readback, preserving operator order.
    pub fn evolve_uniform_flow(
        &mut self,
        field: &mut FreeManaField,
        diffusivity: f64,
        dt: f64,
        velocity: [f64; 3],
    ) -> Result<(), BaseError> {
        self.evolve_uniform_flow_batch(field, diffusivity, dt, velocity, 1)
    }

    /// Advance diffusion followed by uniform advection for every step, keeping
    /// intermediate fields on the device. Only the final validated state is
    /// committed; material, potential and thermal operators are not included.
    pub fn evolve_uniform_flow_batch(
        &mut self,
        field: &mut FreeManaField,
        diffusivity: f64,
        dt: f64,
        velocity: [f64; 3],
        steps: u32,
    ) -> Result<(), BaseError> {
        let started = Instant::now();
        if steps == 0
            || !diffusivity.is_finite()
            || diffusivity < 0.0
            || !dt.is_finite()
            || dt <= 0.0
        {
            return Err(invalid("gpu_uniform_flow_parameters"));
        }
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
        for number in source {
            bytes.extend_from_slice(&number.to_le_bytes());
        }
        raw.write_buffer(&self.diffusion.input, &bytes);
        let mut diffusion_params = Vec::with_capacity(16);
        for n in self.dimensions {
            diffusion_params.extend_from_slice(&(n as u32).to_le_bytes());
        }
        diffusion_params.extend_from_slice(&(alpha as f32).to_bits().to_le_bytes());
        raw.write_buffer(&self.diffusion.parameters, &diffusion_params);
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
        raw.write_buffer(&self.advection.parameters, &advection_params);
        let mut commands = raw.create_command_buffer();
        let workgroups = (shape.len() as u32).div_ceil(64);
        for _ in 0..steps {
            commands.dispatch_compute_with_bind_group(
                &self.diffusion.pipeline,
                &self.diffusion.forward,
                workgroups,
            );
            commands.dispatch_compute_with_bind_group(
                &self.advection.pipeline,
                &self.advection.bind,
                workgroups,
            );
        }
        commands.submit();
        let result = self
            .runtime
            .block_on(self.device.read_buffer(&self.diffusion.input));
        if result.len() != shape.len() * 4 {
            return Err(invalid("gpu_uniform_flow_readback"));
        }
        let densities: Vec<f64> = result
            .as_chunks::<4>()
            .0
            .iter()
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
        let updated = densities
            .into_iter()
            .map(|n| Quantity::<ManonDensity>::nonnegative(n.max(0.0) * scale))
            .collect::<Result<Vec<_>, _>>()?;
        field.try_replace_densities(updated)?;
        self.last_wall_ms = started.elapsed().as_secs_f64() * 1000.0;
        Ok(())
    }
}
