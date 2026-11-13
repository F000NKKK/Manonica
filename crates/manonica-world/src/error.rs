use manonica_error::{BaseError, ClassifiedError, ErrorCode};
#[derive(Debug)]
pub enum WorldError {
    Base(BaseError),
    Content {
        source: &'static str,
        line: usize,
        detail: String,
    },
    Codec(String),
    Io(std::io::Error),
    Invalid(&'static str),
}
pub type WorldResult<T> = Result<T, WorldError>;
impl From<BaseError> for WorldError {
    fn from(e: BaseError) -> Self {
        Self::Base(e)
    }
}
impl From<std::io::Error> for WorldError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}
impl ClassifiedError for WorldError {
    fn code(&self) -> ErrorCode {
        match self {
            Self::Base(e) => e.code,
            _ => ErrorCode::InvalidState,
        }
    }
}
impl std::fmt::Display for WorldError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Base(e) => write!(f, "{e}"),
            Self::Content {
                source,
                line,
                detail,
            } => write!(f, "{source}:{line}: {detail}"),
            Self::Codec(e) => write!(f, "world codec: {e}"),
            Self::Io(e) => write!(f, "world I/O: {e}"),
            Self::Invalid(p) => write!(f, "invalid world state: {p}"),
        }
    }
}
impl std::error::Error for WorldError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Base(e) => Some(e),
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}
