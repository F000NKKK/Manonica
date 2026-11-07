#![forbid(unsafe_code)]

//! Физика манонов и полей. Без знаний о кристаллах, биологических органах,
//! Minecraft-блоках, рунах и других высокоуровневых сущностях.

pub mod dynamics;
pub mod error;
pub mod field;
pub mod manon;
pub mod system;

pub use dynamics::FreeMotion;
pub use error::PhysicsError;
pub use field::{ConservationReport, FreeDiffusion, FreeManaField, GridShape};
pub use manon::{Manon, ManonId, ManonState};
pub use system::ManonSystem;
