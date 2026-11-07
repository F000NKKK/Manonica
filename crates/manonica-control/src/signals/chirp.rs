use manonica_core::{
    Quantity, Signal, Time, TimeStep,
    units::{Hertz, Radians, Unit},
};
use manonica_error::{BaseError, ErrorCode};

/// Линейное изменение мгновенной частоты от start_frequency до end_frequency.
/// После конца интервала сохраняется конечная частота.
pub struct Chirp<U: Unit> {
    amplitude: Quantity<U>,
    offset: Quantity<U>,
    start: Time,
    duration: TimeStep,
    start_frequency: Quantity<Hertz>,
    end_frequency: Quantity<Hertz>,
    phase: Quantity<Radians>,
}

impl<U: Unit> Chirp<U> {
    pub fn new(
        amplitude: Quantity<U>,
        offset: Quantity<U>,
        start: Time,
        duration: TimeStep,
        start_frequency: Quantity<Hertz>,
        end_frequency: Quantity<Hertz>,
        phase: Quantity<Radians>,
    ) -> Result<Self, BaseError> {
        if start_frequency.value() < 0.0 || end_frequency.value() < 0.0 {
            return Err(BaseError::new(ErrorCode::InvalidRange, "chirp_frequency"));
        }
        Ok(Self {
            amplitude,
            offset,
            start,
            duration,
            start_frequency,
            end_frequency,
            phase,
        })
    }
}

impl<U: Unit> Signal<U> for Chirp<U> {
    fn sample(&self, time: Time) -> Result<Quantity<U>, BaseError> {
        let elapsed = time.seconds() - self.start.seconds();
        let duration = self.duration.seconds();
        let f0 = self.start_frequency.value();
        let f1 = self.end_frequency.value();
        let cycles = if elapsed < 0.0 {
            f0 * elapsed
        } else if elapsed <= duration {
            f0 * elapsed + 0.5 * (f1 - f0) * elapsed * elapsed / duration
        } else {
            0.5 * (f0 + f1) * duration + f1 * (elapsed - duration)
        };
        let angle = std::f64::consts::TAU * cycles + self.phase.value();
        Quantity::new(self.offset.value() + self.amplitude.value() * angle.sin())
    }
}
