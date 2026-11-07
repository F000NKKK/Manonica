use manonica_error::{BaseError, ErrorCode};

/// Момент времени; допускает отрицательные значения для анализа сигналов.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Time(f64);

impl Time {
    pub fn from_seconds(value: f64) -> Result<Self, BaseError> {
        if !value.is_finite() {
            return Err(BaseError::new(ErrorCode::NonFinite, "time"));
        }
        Ok(Self(value))
    }

    pub fn seconds(self) -> f64 {
        self.0
    }
}

/// Положительный шаг интегрирования. Не смешивается с моментом времени.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct TimeStep(f64);

impl TimeStep {
    pub fn from_seconds(value: f64) -> Result<Self, BaseError> {
        if !value.is_finite() {
            return Err(BaseError::new(ErrorCode::NonFinite, "time_step"));
        }
        if value <= 0.0 {
            return Err(BaseError::new(ErrorCode::InvalidTimeStep, "time_step"));
        }
        Ok(Self(value))
    }

    pub fn seconds(self) -> f64 {
        self.0
    }
}
