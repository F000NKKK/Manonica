#![forbid(unsafe_code)]

//! Базовые типы ошибок, общие для всех подсистем.

mod base;
mod code;
mod traits;

pub use base::BaseError;
pub use code::ErrorCode;
pub use traits::ClassifiedError;
