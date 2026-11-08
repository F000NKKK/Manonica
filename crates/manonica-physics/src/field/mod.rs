mod advection;
mod diffusion;
mod grid;

pub use advection::{FreeAdvection, VelocityField};
pub use diffusion::{ConservationReport, FreeDiffusion};
pub use grid::{FreeManaField, GridShape};
