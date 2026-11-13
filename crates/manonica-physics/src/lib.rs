#![forbid(unsafe_code)]

//! Фундаментальные маноны, движение и численные поля.
//! Нет кристаллов, руд, биологии, игровых объектов или готовых заклинаний.

pub mod dynamics;
pub mod error;
pub mod field;
pub mod manon;
pub mod mechanics;
pub mod system;

pub use dynamics::{FreeMotion, ParticleLaw};
pub use error::PhysicsError;
pub use field::{
    ConservationReport, FreeAdvection, FreeDiffusion, FreeManaDiffusionBackend, FreeManaField,
    GridShape, LocalBoundarySite, LocalPotentialBoundary, PotentialBoundary, PotentialField,
    PotentialStep, VelocityField,
};
pub use manon::{Manon, ManonId, ManonState};
pub use system::ManonSystem;

pub use mechanics::{
    ExternalLoadLedger, GaussianBody, MechanicalBody, PeriodicSpectralDomain, SpectralMechanics,
    SpectralMode,
};
