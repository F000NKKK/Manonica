use manonica_core::{
    Quantity, Signal, Time,
    units::{Hertz, Radians, Unit},
};
use manonica_error::{BaseError, ErrorCode};

/// Пилообразный сигнал с разрывом в конце каждого периода.
pub struct Sawtooth<U: Unit> {
    amplitude: Quantity<U>,
    offset: Quantity<U>,
    frequency: Quantity<Hertz>,
    phase: Quantity<Radians>,
}

impl<U: Unit> Sawtooth<U> {
    pub fn new(
        amplitude: Quantity<U>,
        offset: Quantity<U>,
        frequency: Quantity<Hertz>,
        phase: Quantity<Radians>,
    ) -> Result<Self, BaseError> {
        if frequency.value() <= 0.0 {
            return Err(BaseError::new(
                ErrorCode::InvalidRange,
                "sawtooth_frequency",
            ));
        }
        Ok(Self {
            amplitude,
            offset,
            frequency,
            phase,
        })
    }
}

impl<U: Unit> Signal<U> for Sawtooth<U> {
    fn sample(&self, time: Time) -> Result<Quantity<U>, BaseError> {
        let cycles =
            time.seconds() * self.frequency.value() + self.phase.value() / std::f64::consts::TAU;
        let wave = 2.0 * cycles.rem_euclid(1.0) - 1.0;
        Quantity::new(self.offset.value() + self.amplitude.value() * wave)
    }
}
