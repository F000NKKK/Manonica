use crate::ErrorCode;

/// Общая ошибка с машинно читаемым кодом и указанием параметра.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BaseError {
    pub code: ErrorCode,
    pub parameter: &'static str,
}

impl BaseError {
    pub const fn new(code: ErrorCode, parameter: &'static str) -> Self {
        Self { code, parameter }
    }
}

impl std::fmt::Display for BaseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}: {}", self.code, self.parameter)
    }
}

impl std::error::Error for BaseError {}
