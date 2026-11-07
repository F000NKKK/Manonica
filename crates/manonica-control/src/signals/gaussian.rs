use manonica_core::{Quantity, Signal, Time, TimeStep, units::Unit};
use manonica_error::BaseError;

/// Гауссов импульс с максимумом в центре и шириной sigma.
pub struct Gaussian<U: Unit> {
    amplitude: Quantity<U>,
    offset: Quantity<U>,
    center: Time,
    sigma: TimeStep,
}

impl<U: Unit> Gaussian<U> {
    pub fn new(amplitude: Quantity<U>, offset: Quantity<U>, center: Time, sigma: TimeStep) -> Self {
        Self { amplitude, offset, center, sigma }
    }
}

impl<U: Unit> Signal<U> for Gaussian<U> {
    fn sample(&self, time: Time) -> Result<Quantity<U>, BaseError> {
        let normal = (time.seconds() - self.center.seconds()) / self.sigma.seconds();
        Quantity::new(self.offset.value() + self.amplitude.value() * (-0.5 * normal * normal).exp())
    }
}
