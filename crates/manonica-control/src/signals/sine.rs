use manonica_core::{Quantity, Signal, Time, units::{Hertz, Radians, Unit}};
use manonica_error::{BaseError, ErrorCode};

/// y(t) = offset + amplitude * sin(2π f t + phase).
pub struct Sine<U: Unit> {
    amplitude: Quantity<U>,
    offset: Quantity<U>,
    frequency: Quantity<Hertz>,
    phase: Quantity<Radians>,
}

impl<U: Unit> Sine<U> {
    pub fn new(
        amplitude: Quantity<U>,
        offset: Quantity<U>,
        frequency: Quantity<Hertz>,
        phase: Quantity<Radians>,
    ) -> Result<Self, BaseError> {
        if frequency.value() < 0.0 {
            return Err(BaseError::new(ErrorCode::InvalidRange, "frequency"));
        }
        Ok(Self { amplitude, offset, frequency, phase })
    }
}

impl<U: Unit> Signal<U> for Sine<U> {
    fn sample(&self, time: Time) -> Result<Quantity<U>, BaseError> {
        let angle = std::f64::consts::TAU * self.frequency.value() * time.seconds()
            + self.phase.value();
        Quantity::new(self.offset.value() + self.amplitude.value() * angle.sin())
    }
}
