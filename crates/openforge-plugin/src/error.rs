//! Errors raised while registering or using plugins.

use thiserror::Error;

/// Everything that can go wrong when a plugin is registered, or when the host
/// is asked to do something no plugin provides.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum PluginError {
    /// Two plugins tried to own the same name in the same registry.
    #[error("{kind} `{name}` is already registered")]
    Duplicate { kind: &'static str, name: String },
    /// A plugin declared an empty name, which could never be looked up.
    #[error("a {kind} was registered with an empty name")]
    EmptyName { kind: &'static str },
    /// Nothing registered handles this request.
    #[error("no plugin provides {kind} `{name}`")]
    NotFound { kind: &'static str, name: String },
}

impl PluginError {
    pub(crate) fn duplicate(kind: &'static str, name: impl Into<String>) -> Self {
        PluginError::Duplicate { kind, name: name.into() }
    }

    pub(crate) fn not_found(kind: &'static str, name: impl Into<String>) -> Self {
        PluginError::NotFound { kind, name: name.into() }
    }
}
