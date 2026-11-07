use std::marker::PhantomData;

use manonica_error::{BaseError, ErrorCode};

use crate::units::Unit;

/// Значение, размерность которого проверяется на уровне типов.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quantity<U: Unit> {
    value: f64,
    unit: PhantomData<U>,
}

impl<U: Unit> Quantity<U> {
    pub fn new(value: f64) -> Result<Self, BaseError> {
        if !value.is_finite() {
            return Err(BaseError::new(ErrorCode::NonFinite, "quantity"));
        }
        Ok(Self {
            value,
            unit: PhantomData,
        })
    }

    pub fn nonnegative(value: f64) -> Result<Self, BaseError> {
        let result = Self::new(value)?;
        if value < 0.0 {
            return Err(BaseError::new(ErrorCode::InvalidRange, "quantity"));
        }
        Ok(result)
    }

    pub fn value(self) -> f64 {
        self.value
    }
}

#[cfg(test)]
mod tests {
    use crate::{Quantity, units::{Joules, Metres}};

    #[test]
    fn quantities_reject_non_finite_values() {
        assert!(Quantity::<Joules>::new(f64::NAN).is_err());
        let metres = Quantity::<Metres>::new(2.0).unwrap();
        assert_eq!(metres.value(), 2.0);
    }
}
