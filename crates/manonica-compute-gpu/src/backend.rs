//! Headless device ownership and the physics backend adapter.
use crate::{advection::AdvectionKernel, diffusion::DiffusionKernel};
use manonica_error::BaseError;
use manonica_physics::FreeManaField;
use meridian_sdk::GpuComputeDevice;

pub struct GpuDiffusion {
    pub(super) runtime: tokio::runtime::Runtime,
    pub(super) device: GpuComputeDevice,
    pub(super) diffusion: DiffusionKernel,
    pub(super) advection: AdvectionKernel,
    pub(super) dimensions: [usize; 3],
    pub(super) last_wall_ms: f64,
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
        let diffusion = DiffusionKernel::new(raw, field.shape().len());
        let advection = AdvectionKernel::new(raw, &diffusion);
        Ok(Self {
            runtime,
            device,
            diffusion,
            advection,
            dimensions: [field.shape().nx(), field.shape().ny(), field.shape().nz()],
            last_wall_ms: 0.0,
        })
    }
    /// Wall-clock upload + compute + readback + validation, not GPU kernel time.
    pub fn last_wall_ms(&self) -> f64 {
        self.last_wall_ms
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
