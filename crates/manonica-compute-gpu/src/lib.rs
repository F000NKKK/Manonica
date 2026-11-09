//! Optional GPU finite-volume diffusion, uniform transport and potential solve.
//!
//! This crate is independent of the Manonica Lab application and renderer.
//!
//! `GpuCoupled` keeps the complete coupled solver on the device: transport,
//! potential drift, binding, channels, heat and energy. Cell operators and
//! hierarchical reductions are parallel; dependent edge exchanges preserve
//! the reference order. Readback is validated before committing a snapshot.
//! `GpuDiffusion` also exposes the independent diffusion/flow adapter.
//! CPU f64 laws remain the numerical reference for the f32 GPU backend.

mod advection;
mod backend;
mod channels;
mod cooling;
mod coupled;
mod coupled_transport;
mod diffusion;
mod drift_plan;
mod drift_prepare;
mod energy;
mod energy_control;
mod error;
mod kernel;
mod layout;
mod material_exchange;
mod potential;
mod potential_drift;
mod potential_scale;
mod reduction;
mod schedule;
mod thermal;

pub use backend::GpuDiffusion;
pub use coupled::GpuCoupled;
pub use potential::GpuPotential;

#[cfg(test)]
mod tests;
