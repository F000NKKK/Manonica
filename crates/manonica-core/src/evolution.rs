use manonica_error::ClassifiedError;

use crate::TimeStep;

/// Общий контракт численного шага любой динамической модели.
/// Физические законы и решатели реализуют собственные состояния и отчёты.
pub trait Evolution<State> {
    type Error: ClassifiedError + Send + Sync + 'static;
    type Report;

    /// При ошибке состояние должно остаться без изменений.
    fn evolve(&self, state: &mut State, dt: TimeStep) -> Result<Self::Report, Self::Error>;
}
