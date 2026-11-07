use manonica_error::BaseError;

use crate::{Quantity, Time, units::Unit};

/// Произвольный сигнал с типизированной единицей измерения.
/// Генераторы и композиции реализуют один и тот же контракт.
pub trait Signal<U: Unit>: Send + Sync {
    fn sample(&self, time: Time) -> Result<Quantity<U>, BaseError>;
}

/// Диагностика состояния компонента в ходе расчёта.
/// Не равнозначна математической наблюдаемости системы.
pub trait Instrumented {
    type Snapshot: Clone;
    fn snapshot(&self) -> Self::Snapshot;
}
