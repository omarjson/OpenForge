//! Error types for the materials subsystem.

/// Errors produced by material/texture operations.
#[derive(Debug, thiserror::Error)]
pub enum MaterialError {
    #[error("failed to read image file `{path}`: {message}")]
    Io { path: String, message: String },
    #[error("failed to decode image `{path}`: {message}")]
    Decode { path: String, message: String },
}

impl MaterialError {
    /// Build an [`MaterialError::Io`] from a path and a displayable cause.
    pub fn io(path: &std::path::Path, cause: impl std::fmt::Display) -> Self {
        MaterialError::Io {
            path: path.display().to_string(),
            message: cause.to_string(),
        }
    }

    /// Build a [`MaterialError::Decode`] from a source label and cause.
    pub fn decode(source: impl Into<String>, cause: impl std::fmt::Display) -> Self {
        MaterialError::Decode {
            path: source.into(),
            message: cause.to_string(),
        }
    }
}
