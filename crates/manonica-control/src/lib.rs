#![forbid(unsafe_code)]

//! Универсальная библиотека ТАУ для моделей Manonica и других систем.

pub mod analysis;
pub mod blocks;
pub mod error;
pub mod signals;
pub mod systems;

pub use analysis::{controllability, observability};
pub use blocks::{ControllerTerm, Derivative, Gain, Integral, Pid, Proportional};
pub use error::ControlError;
pub use signals::{Constant, Delay, Exponential, Pulse, Ramp, Scale, Sine, Square, Step, Sum};
pub use systems::LinearSystem;
