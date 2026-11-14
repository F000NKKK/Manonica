#![forbid(unsafe_code)]
//! Material responses of ordinary matter and crystalline mana reservoirs.
//!
//! The fundamental field and differential operators live in
//! `manonica-physics`. This crate owns material-specific binding,
//! conduction, heat, discharge and tensor constitutive laws. No dependency
//! on Minecraft, graphics APIs, or experimental TOML.

mod binding;
mod conductivity;

pub use binding::{Balance, DischargeReport, ManaChannel, MaterialLattice, TrapCell, TrapMaterial};
pub use conductivity::DirectionalConductivity;
mod thermal_contact;
pub use thermal_contact::{ThermalReservoir, exchange_heat_exact, radiate_to_bath};
