use manonica_core::{
    Quantity, Signal, Time,
    units::{Dimensionless, Hertz, PerSecond, Radians, Unit},
};
use manonica_error::{BaseError, ErrorCode};

/// Затухающие гармонические колебания после момента запуска.
pub struct DampedSine<U: Unit> {
    amplitude: Quantity<U>,
    offset: Quantity<U>,
    start: Time,
    frequency: Quantity<Hertz>,
    decay: Quantity<PerSecond<Dimensionless>>,
    phase: Quantity<Radians>,
}

impl<U: Unit> DampedSine<U> {
    pub fn new(
        amplitude: Quantity<U>,
        offset: Quantity<U>,
        start: Time,
        frequency: Quantity<Hertz>,
        decay: Quantity<PerSecond<Dimensionless>>,
        phase: Quantity<Radians>,
    ) -> Result<Self, BaseError> {
        if frequency.value() < 0.0 || decay.value() < 0.0 {
            return Err(BaseError::new(ErrorCode::InvalidRange, "damped_sine"));
        }
        Ok(Self { amplitude, offset, start, frequency, decay, phase })
    }
}

impl<U: Unit> Signal<U> for DampedSine<U> {
    fn sample(&self, time: Time) -> Result<Quantity<U>, BaseError> {
        let elapsed = time.seconds() - self.start.seconds();
        if elapsed < 0.0 { return Ok(self.offset); }
        let envelope = (-self.decay.value() * elapsed).exp();
        let phase = std::f64::consts::TAU * self.frequency.value() * elapsed + self.phase.value();
        Quantity::new(self.offset.value() + self.amplitude.value() * envelope * phase.sin())
    }
}
