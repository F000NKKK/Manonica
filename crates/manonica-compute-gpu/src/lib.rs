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

mod advection;
mod backend;
mod channels;
mod coupled;
mod coupled_transport;
mod diffusion;
mod energy;
mod error;
mod kernel;
mod layout;
mod material_exchange;
mod potential;
mod potential_drift;
mod thermal;

pub use backend::GpuDiffusion;
pub use coupled::GpuCoupled;
pub use potential::GpuPotential;

#[cfg(test)]
mod tests;
