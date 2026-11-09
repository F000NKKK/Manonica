//! Stable acceleration boundary shared by the headless physics solver,
//! Manonica Lab, and future Minecraft integration.
//!
//! No WGPU, window, Minecraft, or shader types appear in the physical core.
use manonica_error::BaseError;

use super::FreeManaField;

/// Accelerate conservative diffusion and optionally uniform transport.
/// Implementations must atomically reject invalid inputs and maintain the
/// inventory invariant. Material, field, and thermal steps remain owned by
/// the deterministic physics simulation.
pub trait FreeManaDiffusionBackend {
    fn evolve_free_diffusion(
        &mut self,
        field: &mut FreeManaField,
        diffusivity: f64,
        dt_seconds: f64,
    ) -> Result<(), BaseError>;

    /// Optional fused, conservative operator split: diffuse, then advect by
    /// a uniform surrounding medium without an intermediate CPU readback.
    /// Returns false if the backend does not implement this operation;
    /// the caller must then run the two reference laws separately.
    fn evolve_diffusion_and_uniform_flow(
        &mut self,
        _field: &mut FreeManaField,
        _diffusivity: f64,
        _dt_seconds: f64,
        _velocity: [f64; 3],
    ) -> Result<bool, BaseError> {
        Ok(false)
    }

    /// Executes diffusion then uniform advection at each step. The default
    /// commits only after every step succeeds. A backend without fused
    /// transport rejects this operation without modifying the field.
    fn evolve_uniform_flow_batch(
        &mut self,
        field: &mut FreeManaField,
        diffusivity: f64,
        dt_seconds: f64,
        velocity: [f64; 3],
        steps: u32,
    ) -> Result<(), BaseError> {
        if steps == 0 {
            return Err(BaseError::new(
                manonica_error::ErrorCode::InvalidRange,
                "flow_batch_steps",
            ));
        }
        let mut candidate = field.clone();
        for _ in 0..steps {
            if !self.evolve_diffusion_and_uniform_flow(
                &mut candidate,
                diffusivity,
                dt_seconds,
                velocity,
            )? {
                return Err(BaseError::new(
                    manonica_error::ErrorCode::InvalidRange,
                    "flow_batch_unsupported",
                ));
            }
        }
        *field = candidate;
        Ok(())
    }

    /// Executes consecutive diffusion steps without exposing intermediate
    /// states. The default implementation is reference-equivalent; GPU
    /// implementations can keep intermediate buffers device-resident.
    fn evolve_diffusion_batch(
        &mut self,
        field: &mut FreeManaField,
        diffusivity: f64,
        dt_seconds: f64,
        steps: u32,
    ) -> Result<(), BaseError> {
        if steps == 0 {
            return Err(BaseError::new(
                manonica_error::ErrorCode::InvalidRange,
                "diffusion_batch_steps",
            ));
        }
        let mut candidate = field.clone();
        for _ in 0..steps {
            self.evolve_free_diffusion(&mut candidate, diffusivity, dt_seconds)?;
        }
        *field = candidate;
        Ok(())
    }
}
