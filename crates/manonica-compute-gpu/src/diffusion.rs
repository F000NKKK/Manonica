//! Conservative finite-volume diffusion and its matching WGSL pipeline.
use crate::{backend::GpuDiffusion, error::invalid};
use manonica_core::{Quantity, units::ManonDensity};
use manonica_error::BaseError;
use manonica_physics::FreeManaField;
use meridian_gpu_wgpu::{BindGroup, Buffer, BufferUsage, ComputePipeline, Device};
use std::time::Instant;

pub(super) struct DiffusionKernel {
    pub(super) pipeline: ComputePipeline,
    pub(super) input: Buffer,
    pub(super) output: Buffer,
    pub(super) parameters: Buffer,
    pub(super) forward: BindGroup,
    pub(super) reverse: BindGroup,
}
impl DiffusionKernel {
    pub(super) fn new(raw: &Device, len: usize) -> Self {
        let shader = raw.create_shader("manonica-3d-diffusion", include_str!("diffusion.wgsl"));
        let pipeline = raw.create_compute_pipeline(&shader, "diffusion");
        let input = raw.create_buffer(len * 4, BufferUsage::Storage);
        let output = raw.create_buffer(len * 4, BufferUsage::Storage);
        let parameters = raw.create_buffer(16, BufferUsage::Storage);
        let layout = pipeline.wgpu_pipeline().get_bind_group_layout(0);
        let forward = raw.create_bind_group(&layout, &[&input, &output, &parameters]);
        let reverse = raw.create_bind_group(&layout, &[&output, &input, &parameters]);
        Self {
            pipeline,
            input,
            output,
            parameters,
            forward,
            reverse,
        }
    }
}
impl GpuDiffusion {
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
        raw.write_buffer(&self.diffusion.input, &bytes);
        let mut params = Vec::with_capacity(16);
        for n in self.dimensions {
            params.extend_from_slice(&(n as u32).to_le_bytes());
        }
        params.extend_from_slice(&(alpha as f32).to_bits().to_le_bytes());
        raw.write_buffer(&self.diffusion.parameters, &params);
        let workgroups = (shape.len() as u32).div_ceil(64);
        let mut commands = raw.create_command_buffer();
        for step in 0..steps {
            commands.dispatch_compute_with_bind_group(
                &self.diffusion.pipeline,
                if step % 2 == 0 {
                    &self.diffusion.forward
                } else {
                    &self.diffusion.reverse
                },
                workgroups,
            );
        }
        commands.submit();
        let last = if steps.is_multiple_of(2) {
            &self.diffusion.input
        } else {
            &self.diffusion.output
        };
        let result = self.runtime.block_on(self.device.read_buffer(last));
        if result.len() != shape.len() * 4 {
            return Err(invalid("gpu_diffusion_readback"));
        }
        let densities: Vec<f64> = result
            .as_chunks::<4>()
            .0
            .iter()
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
        raw.write_buffer(&self.diffusion.input, &bytes);
        let mut settings = Vec::with_capacity(16);
        for n in self.dimensions {
            settings.extend_from_slice(&(n as u32).to_le_bytes());
        }
        settings.extend_from_slice(&(alpha as f32).to_bits().to_le_bytes());
        raw.write_buffer(&self.diffusion.parameters, &settings);
        let mut commands = raw.create_command_buffer();
        commands.dispatch_compute_with_bind_group(
            &self.diffusion.pipeline,
            &self.diffusion.forward,
            (shape.len() as u32).div_ceil(64),
        );
        commands.submit();
        let result = self
            .runtime
            .block_on(self.device.read_buffer(&self.diffusion.output));
        if result.len() != shape.len() * 4 {
            return Err(invalid("gpu_diffusion_readback"));
        }
        let densities: Vec<f64> = result
            .as_chunks::<4>()
            .0
            .iter()
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
