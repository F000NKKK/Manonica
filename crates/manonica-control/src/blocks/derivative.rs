use manonica_core::{Quantity, TimeStep, units::Unit};
use manonica_error::{BaseError, ErrorCode};

use crate::{
    ControlError,
    blocks::{ControllerTerm, DerivativeMode, Gain},
};

/// Дифференциатор ошибки, опционально фильтруемый апериодическим звеном.
/// Первый шаг возвращает 0, не вызывая искусственного выброса.
#[derive(Debug, Clone, Copy)]
pub struct Derivative<In: Unit, Out: Unit> {
    gain: Gain<In, Out, DerivativeMode>,
    filter_time: Option<TimeStep>,
    previous_error: Option<f64>,
    filtered_slope: f64,
}

impl<In: Unit, Out: Unit> Derivative<In, Out> {
    pub fn new(gain: Gain<In, Out, DerivativeMode>, filter_time: Option<TimeStep>) -> Self {
        Self {
            gain,
            filter_time,
            previous_error: None,
            filtered_slope: 0.0,
        }
    }
}

impl<In: Unit, Out: Unit> ControllerTerm<In, Out> for Derivative<In, Out> {
    fn update(&mut self, error: Quantity<In>, dt: TimeStep) -> Result<Quantity<Out>, ControlError> {
        let raw_slope = self
            .previous_error
            .map_or(0.0, |previous| (error.value() - previous) / dt.seconds());
        let slope = self.filter_time.map_or(raw_slope, |tau| {
            let alpha = tau.seconds() / (tau.seconds() + dt.seconds());
            alpha * self.filtered_slope + (1.0 - alpha) * raw_slope
        });
        let output = self.gain.coefficient() * slope;
        if !slope.is_finite() || !output.is_finite() {
            return Err(BaseError::new(ErrorCode::NonFinite, "derivative").into());
        }
        self.previous_error = Some(error.value());
        self.filtered_slope = slope;
        Ok(Quantity::new(output)?)
    }

    fn reset(&mut self) {
        self.previous_error = None;
        self.filtered_slope = 0.0;
    }
}
