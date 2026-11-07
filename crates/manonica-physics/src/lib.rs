#![forbid(unsafe_code)]

//! Базовая физика свободных манонов.
//! Без кристаллов, материалов и преждевременно зафиксированных законов связи.

pub mod dynamics;
pub mod error;
pub mod manon;
pub mod system;

pub use dynamics::FreeMotion;
pub use error::PhysicsError;
pub use manon::{Manon, ManonId, ManonState};
pub use system::ManonSystem;
