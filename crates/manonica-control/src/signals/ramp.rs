use manonica_core::{Quantity, Signal, Time, units::{PerSecond, Unit}};
use manonica_error::BaseError;

/// Линейный сигнал y(t) = initial + slope * (t - start).
pub struct Ramp<U: Unit> {
    initial: Quantity<U>,
    slope: Quantity<PerSecond<U>>,
    start: Time,
}

impl<U: Unit> Ramp<U> {
    pub fn new(initial: Quantity<U>, slope: Quantity<PerSecond<U>>, start: Time) -> Self {
        Self { initial, slope, start }
    }
}

impl<U: Unit> Signal<U> for Ramp<U> {
    fn sample(&self, time: Time) -> Result<Quantity<U>, BaseError> {
        Quantity::new(
            self.initial.value() + self.slope.value() * (time.seconds() - self.start.seconds()),
        )
    }
}
