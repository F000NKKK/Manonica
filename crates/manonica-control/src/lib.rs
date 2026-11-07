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
    ControllerSnapshot, ControllerTerm, Derivative, DerivativeMode, Gain, Integral, IntegralMode,
    IntegralTerm, Limits, Pid, Proportional, ProportionalMode,
};
pub use error::ControlError;
pub use signals::{
    Chirp, Constant, DampedSine, Delay, Exponential, Gaussian, Pulse, Ramp, SampledNoise, Sawtooth,
    Scale, Sine, Square, Step, Sum, Triangle,
};
pub use systems::{DynamicSystem, LinearSystem};
