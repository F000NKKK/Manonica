use manonica_error::{BaseError, ClassifiedError, ErrorCode};

#[derive(Debug)]
pub enum PhysicsError {
    Base(BaseError),
    DuplicateManonId(u64),
}

impl From<BaseError> for PhysicsError {
    fn from(value: BaseError) -> Self {
        Self::Base(value)
    }
}

impl ClassifiedError for PhysicsError {
    fn code(&self) -> ErrorCode {
        match self {
            Self::Base(base) => base.code,
            Self::DuplicateManonId(_) => ErrorCode::DuplicateId,
        }
    }
}

impl std::fmt::Display for PhysicsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Base(error) => write!(f, "{error}"),
            Self::DuplicateManonId(id) => write!(f, "duplicate manon id: {id}"),
        }
    }
}

impl std::error::Error for PhysicsError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Base(error) => Some(error),
            Self::DuplicateManonId(_) => None,
        }
    }
}
