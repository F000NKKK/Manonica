use manonica_error::{BaseError, ClassifiedError, ErrorCode};

/// Ошибка регулятора или анализа системы управления.
#[derive(Debug)]
pub enum ControlError {
    Base(BaseError),
    NumericalFailure(&'static str),
}

impl From<BaseError> for ControlError {
    fn from(value: BaseError) -> Self {
        Self::Base(value)
    }
}

impl ClassifiedError for ControlError {
    fn code(&self) -> ErrorCode {
        match self {
            Self::Base(error) => error.code,
            Self::NumericalFailure(_) => ErrorCode::NumericalInstability,
        }
    }
}

impl std::fmt::Display for ControlError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Base(error) => write!(f, "{error}"),
            Self::NumericalFailure(component) => write!(f, "numerical failure in {component}"),
        }
    }
}

impl std::error::Error for ControlError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Base(error) => Some(error),
            Self::NumericalFailure(_) => None,
        }
    }
}
