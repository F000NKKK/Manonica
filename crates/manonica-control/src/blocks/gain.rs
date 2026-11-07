use std::marker::PhantomData;

use manonica_core::units::Unit;
use manonica_error::{BaseError, ErrorCode};

pub trait GainMode: Copy + Send + Sync + 'static {}

#[derive(Debug, Clone, Copy)]
pub struct ProportionalMode;
#[derive(Debug, Clone, Copy)]
pub struct IntegralMode;
#[derive(Debug, Clone, Copy)]
pub struct DerivativeMode;

impl GainMode for ProportionalMode {}
impl GainMode for IntegralMode {}
impl GainMode for DerivativeMode {}

/// Kp: Out/In; Ki: Out/(In·s); Kd: Out·s/In.
/// Маркер режима предотвращает ошибочную подстановку Kp вместо Ki.
#[derive(Debug, Clone, Copy)]
pub struct Gain<In: Unit, Out: Unit, Mode: GainMode> {
    coefficient: f64,
    marker: PhantomData<(In, Out, Mode)>,
}

impl<In: Unit, Out: Unit, Mode: GainMode> Gain<In, Out, Mode> {
    pub fn new(coefficient: f64) -> Result<Self, BaseError> {
        if !coefficient.is_finite() {
            return Err(BaseError::new(ErrorCode::NonFinite, "gain"));
        }
        Ok(Self {
            coefficient,
            marker: PhantomData,
        })
    }

    pub fn coefficient(self) -> f64 {
        self.coefficient
    }
}
