/// Стабильная классификация ошибки для обработки и диагностики.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ErrorCode {
    NonFinite,
    InvalidRange,
    InvalidTimeStep,
    InvalidShape,
    InvalidState,
    DuplicateId,
    NumericalInstability,
    NotObservable,
    NotControllable,
}
