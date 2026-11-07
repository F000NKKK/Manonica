use manonica_core::{Quantity, Signal, Time, TimeStep, units::Unit};
use manonica_error::BaseError;

/// Одиночный прямоугольный импульс, не дельта-функция Дирака.
pub struct Pulse<U: Unit> {
    start: Time,
    width: TimeStep,
    baseline: Quantity<U>,
    peak: Quantity<U>,
}

impl<U: Unit> Pulse<U> {
    pub fn new(start: Time, width: TimeStep, baseline: Quantity<U>, peak: Quantity<U>) -> Self {
        Self { start, width, baseline, peak }
    }
}

impl<U: Unit> Signal<U> for Pulse<U> {
    fn sample(&self, time: Time) -> Result<Quantity<U>, BaseError> {
        let elapsed = time.seconds() - self.start.seconds();
        Ok(if elapsed >= 0.0 && elapsed < self.width.seconds() {
            self.peak
        } else {
            self.baseline
        })
    }
}
