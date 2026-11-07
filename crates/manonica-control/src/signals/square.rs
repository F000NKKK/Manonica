use manonica_core::{Quantity, Signal, Time, units::{Hertz, Unit}};
use manonica_error::{BaseError, ErrorCode};

/// Прямоугольный периодический сигнал с коэффициентом заполнения.
pub struct Square<U: Unit> {
    low: Quantity<U>,
    high: Quantity<U>,
    frequency: Quantity<Hertz>,
    duty: f64,
}

impl<U: Unit> Square<U> {
    pub fn new(
        low: Quantity<U>,
        high: Quantity<U>,
        frequency: Quantity<Hertz>,
        duty: f64,
    ) -> Result<Self, BaseError> {
        if frequency.value() <= 0.0 || !duty.is_finite() || !(0.0..=1.0).contains(&duty) {
            return Err(BaseError::new(ErrorCode::InvalidRange, "square"));
        }
        Ok(Self { low, high, frequency, duty })
    }
}

impl<U: Unit> Signal<U> for Square<U> {
    fn sample(&self, time: Time) -> Result<Quantity<U>, BaseError> {
        let cycle = (time.seconds() * self.frequency.value()).rem_euclid(1.0);
        Ok(if cycle < self.duty { self.high } else { self.low })
    }
}
