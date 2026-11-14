#![forbid(unsafe_code)]
//! Demo game runtime: immutable content, chunked state and conservative local laws.
pub mod chunk;
mod error;
pub mod inventory;
pub mod metadata;
pub mod registry;
pub mod runtime;
pub use error::{WorldError, WorldResult};

pub mod gameplay;
pub mod magic;

pub mod matter;
pub mod thermal;
