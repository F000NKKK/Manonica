use manonica_core::{Quantity, Signal, Time, units::Unit};
use manonica_error::BaseError;

pub struct Constant<U: Unit> {
    value: Quantity<U>,
}

impl<U: Unit> Constant<U> {
    pub fn new(value: Quantity<U>) -> Self { Self { value } }
}

impl<U: Unit> Signal<U> for Constant<U> {
    fn sample(&self, _: Time) -> Result<Quantity<U>, BaseError> { Ok(self.value) }
}
