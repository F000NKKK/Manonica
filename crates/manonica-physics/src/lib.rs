#![forbid(unsafe_code)]

//! Фундаментальные маноны, движение и численные поля.
//! Нет кристаллов, руд, биологии, игровых объектов или готовых заклинаний.

pub mod binding;
pub mod dynamics;
pub mod error;
pub mod field;
pub mod manon;
pub mod system;

pub use binding::{Balance, DischargeReport, ManaChannel, MaterialLattice, TrapCell, TrapMaterial};
pub use dynamics::{FreeMotion, ParticleLaw};
pub use error::PhysicsError;
pub use field::{
    ConservationReport, FreeAdvection, FreeDiffusion, FreeManaField, GridShape, PotentialBoundary,
    PotentialField, PotentialStep, VelocityField,
};
pub use manon::{Manon, ManonId, ManonState};
pub use system::ManonSystem;
