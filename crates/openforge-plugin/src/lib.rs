//! The OpenForge plugin host.
//!
//! OpenForge's reference architecture is a plugin host: a fixed set of
//! *services* that plugins both consume and extend, pluggable *tools*, and
//! pluggable *format filters*. This crate is that seam. The engine's own tools
//! and formats are ordinary plugins here ([`builtin_host`]), so a third-party
//! filter or tool is a registration rather than a change to the app.
//!
//! ```
//! use openforge_plugin::builtin_host;
//!
//! let host = builtin_host();
//! assert!(host.has_tool("Move"));
//! assert_eq!(host.filters.for_extension("glb").unwrap().id(), "glb");
//! ```
//!
//! Three rules keep the host honest:
//!
//! - **Names are unique.** Registering a second service, tool, filter id or
//!   extension is an error naming the clash, never a silent replacement.
//! - **Registration is atomic.** A plugin rejected part way through leaves the
//!   host exactly as it was ([`PluginHost::register`]).
//! - **Tools are factories.** A tool carries the state of an in-progress drag,
//!   so the host hands out a fresh one per request instead of sharing.

pub mod error;
pub mod filter;
pub mod host;
pub mod service;

pub use error::PluginError;
pub use filter::{engine_filters, FilterRegistry, FormatFilter};
pub use host::{
    builtin_host, builtin_plugins, CoreToolsPlugin, InterchangePlugin, LoadedPlugin, Plugin,
    PluginHost, ToolFactory,
};
pub use service::{Service, ServiceRegistry, SimpleService};

/// Returns the crate name, used by a smoke test to confirm linkage.
pub fn name() -> &'static str {
    "openforge-plugin"
}

#[cfg(test)]
mod tests {
    #[test]
    fn links() {
        assert_eq!(super::name(), "openforge-plugin");
    }
}
