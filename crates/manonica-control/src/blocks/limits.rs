use manonica_core::{Quantity, units::Unit};
use manonica_error::{BaseError, ErrorCode};

#[derive(Debug, Clone, Copy)]
pub struct Limits<U: Unit> {
    min: Quantity<U>,
    max: Quantity<U>,
}

impl<U: Unit> Limits<U> {
    pub fn new(min: Quantity<U>, max: Quantity<U>) -> Result<Self, BaseError> {
        if min.value() > max.value() {
            return Err(BaseError::new(ErrorCode::InvalidRange, "limits"));
        }
        Ok(Self { min, max })
    }

    pub fn apply(self, value: Quantity<U>) -> Result<(Quantity<U>, bool), BaseError> {
        let clamped = value.value().clamp(self.min.value(), self.max.value());
        Ok((Quantity::new(clamped)?, clamped != value.value()))
    }

    pub fn min(self) -> Quantity<U> {
        self.min
    }

    pub fn max(self) -> Quantity<U> {
        self.max
    }
}
