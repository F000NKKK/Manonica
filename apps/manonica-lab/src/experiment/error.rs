use manonica_error::{BaseError, ClassifiedError, ErrorCode};

#[derive(Debug)]
pub enum ExperimentError {
    Base(BaseError),
    Parse(toml::de::Error),
    Io(std::io::Error),
    Physics(manonica_physics::PhysicsError),
}

impl From<BaseError> for ExperimentError {
    fn from(error: BaseError) -> Self {
        Self::Base(error)
    }
}
impl From<toml::de::Error> for ExperimentError {
    fn from(error: toml::de::Error) -> Self {
        Self::Parse(error)
    }
}
impl From<std::io::Error> for ExperimentError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}
impl From<manonica_physics::PhysicsError> for ExperimentError {
    fn from(error: manonica_physics::PhysicsError) -> Self {
        Self::Physics(error)
    }
}

impl ClassifiedError for ExperimentError {
    fn code(&self) -> ErrorCode {
        match self {
            Self::Base(error) => error.code,
            Self::Parse(_) | Self::Io(_) => ErrorCode::InvalidState,
            Self::Physics(error) => error.code(),
        }
    }
}

impl std::fmt::Display for ExperimentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Base(error) => write!(f, "{error}"),
            Self::Parse(error) => write!(f, "invalid experiment TOML: {error}"),
            Self::Io(error) => write!(f, "experiment I/O: {error}"),
            Self::Physics(error) => write!(f, "simulation: {error}"),
        }
    }
}

impl std::error::Error for ExperimentError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Base(error) => Some(error),
            Self::Parse(error) => Some(error),
            Self::Io(error) => Some(error),
            Self::Physics(error) => Some(error),
        }
    }
}
