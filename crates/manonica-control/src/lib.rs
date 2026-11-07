#![forbid(unsafe_code)]

//! Платформенно независимая библиотека систем автоматического управления.
//! Регуляторы и сигналы не знают о манонах или Minecraft.

pub mod analysis;
pub mod blocks;
pub mod error;
pub mod signals;
pub mod systems;

pub use analysis::{AnalysisReport, controllability, observability};
pub use blocks::{
    ControllerSnapshot, ControllerTerm, Derivative, DerivativeMode, Gain,
    Integral, IntegralMode, IntegralTerm, Limits, Pid, Proportional, ProportionalMode,
};
pub use error::ControlError;
pub use signals::{
    Constant, Delay, Exponential, Pulse, Ramp, Scale, Sine, Square, Step, Sum,
};
pub use systems::{DynamicSystem, LinearSystem};
