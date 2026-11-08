//! Optional GPU-backed, finite-volume free-manon diffusion.
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
        })
    }

    /// Advances a single explicit diffusion step: J=-D grad(n).
    /// The input field is mutated only after the GPU output has been validated.
    pub fn evolve(
        &mut self,
        field: &mut FreeManaField,
        diffusivity: f64,
        dt: f64,
    ) -> Result<(), BaseError> {
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
        field.replace(updated);
        Ok(())
    }
}
