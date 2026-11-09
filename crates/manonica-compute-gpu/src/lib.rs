//! Optional GPU finite-volume diffusion, uniform transport and potential solve.
//!
//! This crate is independent of the Manonica Lab application and renderer.
//!
//! Lab remains a hybrid solver: diffusion and uniform transport run on a
//! headless Meridian/wgpu compute device; binding, temperature, energy audits
//! and the Lab potential cycle still execute on CPU. `GpuPotential` exposes
//! the independent GPU potential solver for headless validation.
//! GPU f32 results are mass-renormalized to the previous f64 inventory
//! after readback. CPU remains the reference implementation.
use manonica_core::{Quantity, units::ManonDensity};
use manonica_error::{BaseError, ErrorCode};
use manonica_physics::FreeManaField;
use meridian_gpu_wgpu::{BindGroup, Buffer, BufferUsage, ComputePipeline};
use meridian_sdk::GpuComputeDevice;
use std::time::Instant;

mod potential;
mod transport;
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
        let advection_pipeline = raw.create_compute_pipeline(&advection_shader, "advect_uniform");
        let advection_parameters = raw.create_buffer(24, BufferUsage::Storage);
        let advection_layout = advection_pipeline.wgpu_pipeline().get_bind_group_layout(0);
        // The advection step reads the diffusion output and writes the input
        // buffer; both physical operations happen in one GPU submission.
        let advection_bind =
            raw.create_bind_group(&advection_layout, &[&output, &input, &advection_parameters]);
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
        let last = if steps.is_multiple_of(2) {
            &self.input
        } else {
            &self.output
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

    fn evolve_uniform_flow_batch(
        &mut self,
        field: &mut FreeManaField,
        diffusivity: f64,
        dt_seconds: f64,
        velocity: [f64; 3],
        steps: u32,
    ) -> Result<(), BaseError> {
        GpuDiffusion::evolve_uniform_flow_batch(
            self,
            field,
            diffusivity,
            dt_seconds,
            velocity,
            steps,
        )
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
mod tests;
