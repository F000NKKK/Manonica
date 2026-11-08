//! Stable acceleration boundary shared by the headless physics solver,
//! Manonica Lab, and future Minecraft integration.
//!
//! No WGPU, window, Minecraft, or shader types appear in the physical core.
use manonica_error::BaseError;

use super::FreeManaField;

/// Replace only the conservative free-mana diffusion operator.
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
}
