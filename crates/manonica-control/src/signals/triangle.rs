use manonica_core::{
    Quantity, Signal, Time,
    units::{Hertz, Radians, Unit},
};
use manonica_error::{BaseError, ErrorCode};

/// Треугольный сигнал: за период плавно изменяется между offset±amplitude.
pub struct Triangle<U: Unit> {
    amplitude: Quantity<U>,
    offset: Quantity<U>,
    frequency: Quantity<Hertz>,
    phase: Quantity<Radians>,
}

impl<U: Unit> Triangle<U> {
    pub fn new(
        amplitude: Quantity<U>,
        offset: Quantity<U>,
        frequency: Quantity<Hertz>,
        phase: Quantity<Radians>,
    ) -> Result<Self, BaseError> {
        if frequency.value() <= 0.0 {
            return Err(BaseError::new(ErrorCode::InvalidRange, "triangle_frequency"));
        }
        Ok(Self { amplitude, offset, frequency, phase })
    }
}

impl<U: Unit> Signal<U> for Triangle<U> {
    fn sample(&self, time: Time) -> Result<Quantity<U>, BaseError> {
        let cycles = time.seconds() * self.frequency.value()
            + self.phase.value() / std::f64::consts::TAU;
        let phase = cycles.rem_euclid(1.0);
        let wave = 1.0 - 4.0 * (phase - 0.5).abs();
        Quantity::new(self.offset.value() + self.amplitude.value() * wave)
    }
}
