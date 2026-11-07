use manonica_core::{
    Quantity, Signal, Time,
    units::{Dimensionless, PerSecond, Unit},
};
use manonica_error::{BaseError, ErrorCode};

/// y(t) = offset + amplitude * exp(-decay * (t-start)), t >= start.
/// До начала возвращает offset + amplitude.
pub struct Exponential<U: Unit> {
    start: Time,
    amplitude: Quantity<U>,
    offset: Quantity<U>,
    decay: Quantity<PerSecond<Dimensionless>>,
}

impl<U: Unit> Exponential<U> {
    pub fn new(
        start: Time,
        amplitude: Quantity<U>,
        offset: Quantity<U>,
        decay: Quantity<PerSecond<Dimensionless>>,
    ) -> Result<Self, BaseError> {
        if decay.value() < 0.0 {
            return Err(BaseError::new(ErrorCode::InvalidRange, "decay"));
        }
        Ok(Self {
            start,
            amplitude,
            offset,
            decay,
        })
    }
}

impl<U: Unit> Signal<U> for Exponential<U> {
    fn sample(&self, time: Time) -> Result<Quantity<U>, BaseError> {
        let t = (time.seconds() - self.start.seconds()).max(0.0);
        Quantity::new(
            self.offset.value() + self.amplitude.value() * (-self.decay.value() * t).exp(),
        )
    }
}
