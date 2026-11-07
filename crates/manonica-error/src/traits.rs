use crate::{BaseError, ErrorCode};

/// Общий контракт типизированных ошибок доменных подсистем.
pub trait ClassifiedError: std::error::Error {
    fn code(&self) -> ErrorCode;
}

impl ClassifiedError for BaseError {
    fn code(&self) -> ErrorCode {
        self.code
    }
}
