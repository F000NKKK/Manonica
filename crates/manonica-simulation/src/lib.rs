//! Reusable coupled physical state and CPU reference, independent of Lab and Minecraft.
#![forbid(unsafe_code)]
mod cpu;
mod flow;
mod parameters;
mod state;
pub use flow::MediumFlow;
pub use parameters::{FieldControl, StepParameters};
pub use state::{CoupledState, FieldLedger};
