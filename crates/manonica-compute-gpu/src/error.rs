//! Shared validation errors at the GPU boundary.
use manonica_error::{BaseError, ErrorCode};
pub(super) fn invalid(name: &'static str) -> BaseError {
    BaseError::new(ErrorCode::InvalidRange, name)
}
