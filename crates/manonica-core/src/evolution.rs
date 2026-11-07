use crate::TimeStep;

/// Общий контракт численного шага любой динамической модели.
/// Физика, ТАУ и симуляторы подставляют собственные типы состояния и ошибок.
pub trait Evolution<State> {
    type Error: std::error::Error + Send + Sync + 'static;
    type Report;

    /// При ошибке состояние должно оставаться неизменным (транзакционная семантика).
    fn evolve(&self, state: &mut State, dt: TimeStep) -> Result<Self::Report, Self::Error>;
}
