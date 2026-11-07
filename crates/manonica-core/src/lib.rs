#![forbid(unsafe_code)]

//! Общая математическая основа Manonica: величины, размерности и интерфейсы.
//! Не зависит от предметной физики, материалов, Minecraft и загрузчиков.

pub mod evolution;
pub mod math;
pub mod quantity;
pub mod signal;
pub mod time;
pub mod units;

pub use evolution::Evolution;
pub use math::{GainMatrix, Matrix, Vec3};
pub use quantity::Quantity;
pub use signal::{Instrumented, Signal};
pub use time::{Time, TimeStep};
pub use units::*;
