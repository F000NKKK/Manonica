use manonica_core::{Quantity, Signal, Time, units::Unit};
use manonica_error::BaseError;

/// Единичный переход произвольной амплитуды и начального момента.
pub struct Step<U: Unit> {
    at: Time,
    before: Quantity<U>,
    after: Quantity<U>,
}

impl<U: Unit> Step<U> {
    pub fn new(at: Time, before: Quantity<U>, after: Quantity<U>) -> Self {
        Self { at, before, after }
    }
}

impl<U: Unit> Signal<U> for Step<U> {
    fn sample(&self, time: Time) -> Result<Quantity<U>, BaseError> {
        Ok(if time.seconds() < self.at.seconds() { self.before } else { self.after })
    }
}
