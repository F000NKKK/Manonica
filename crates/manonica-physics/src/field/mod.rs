mod advection;
mod backend;
mod diffusion;
mod grid;
mod potential;

pub use advection::{FreeAdvection, VelocityField};
pub use backend::FreeManaDiffusionBackend;
pub use diffusion::{ConservationReport, FreeDiffusion};
pub use grid::{FreeManaField, GridShape};
pub use potential::{
    LocalBoundarySite, LocalPotentialBoundary, PotentialBoundary, PotentialField, PotentialStep,
};
