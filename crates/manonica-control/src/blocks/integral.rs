use manonica_core::{Quantity, TimeStep, units::Unit};
use manonica_error::{BaseError, ErrorCode};

use crate::{ControlError, blocks::{ControllerTerm, Gain, IntegralMode, IntegralTerm}};

/// Дискретный интегратор по правилу трапеций.
/// Интегральное состояние хранится отдельно от коэффициента Ki.
#[derive(Debug, Clone, Copy)]
pub struct Integral<In: Unit, Out: Unit> {
    gain: Gain<In, Out, IntegralMode>,
    integral: f64,
    previous_error: Option<f64>,
}

impl<In: Unit, Out: Unit> Integral<In, Out> {
    pub fn new(gain: Gain<In, Out, IntegralMode>) -> Self {
        Self {
            gain,
            integral: 0.0,
            previous_error: None,
        }
    }
}

impl<In: Unit, Out: Unit> ControllerTerm<In, Out> for Integral<In, Out> {
    fn update(
        &mut self,
        error: Quantity<In>,
        dt: TimeStep,
    ) -> Result<Quantity<Out>, ControlError> {
        let area = self.previous_error.map_or(
            error.value() * dt.seconds(),
            |previous| 0.5 * (previous + error.value()) * dt.seconds(),
        );
        let next = self.integral + area;
        let output = self.gain.coefficient() * next;
        if !next.is_finite() || !output.is_finite() {
            return Err(BaseError::new(ErrorCode::NonFinite, "integral").into());
        }
        self.integral = next;
        self.previous_error = Some(error.value());
        Ok(Quantity::new(output)?)
    }

    fn reset(&mut self) {
        self.integral = 0.0;
        self.previous_error = None;
    }
}

impl<In: Unit, Out: Unit> IntegralTerm<In, Out> for Integral<In, Out> {
    fn current_output(&self) -> Result<Quantity<Out>, ControlError> {
        Ok(Quantity::new(self.gain.coefficient() * self.integral)?)
    }
}
