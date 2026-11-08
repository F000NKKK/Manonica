use manonica_core::{Quantity, TimeStep, units::Unit};

use crate::ControlError;

/// Универсальный блок регулятора со строго заданными размерностями входа и выхода.
pub trait ControllerTerm<In: Unit, Out: Unit>: Send {
    fn update(&mut self, error: Quantity<In>, dt: TimeStep) -> Result<Quantity<Out>, ControlError>;

    fn reset(&mut self);
}

/// Интегрирующий блок раскрывает текущее выходное значение для антивиндапа.
pub trait IntegralTerm<In: Unit, Out: Unit>: ControllerTerm<In, Out> {
    fn current_output(&self) -> Result<Quantity<Out>, ControlError>;
}

/// Отчёт о вычислении регулятора — не математическая наблюдаемость системы.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ControllerSnapshot {
    pub error: Option<f64>,
    pub proportional: Option<f64>,
    pub integral: Option<f64>,
    pub derivative: Option<f64>,
    pub output: Option<f64>,
    pub saturated: bool,
}
