#![forbid(unsafe_code)]

//! Платформенно независимые величины и интерфейсы Manonica.

pub mod math;
pub mod quantity;
pub mod signal;
pub mod time;
pub mod units;

pub use math::{Matrix, Vec3};
pub use quantity::Quantity;
pub use signal::{Instrumented, Signal};
pub use time::{Time, TimeStep};
pub use units::*;
