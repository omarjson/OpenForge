//! The plugin host: the registries a plugin registers into, and the loaded
//! plugin list the UI shows.

use crate::error::PluginError;
use crate::filter::{engine_filters, FilterRegistry};
use crate::service::ServiceRegistry;
use openforge_core::tool::{SelectTool, Tool, TransformMode, TransformTool};
use openforge_io::{LoadedScene, MeshData};
use std::collections::BTreeMap;
use std::path::Path;

/// A plugin: a named bundle that registers services, tools and filters.
///
/// Registration is fallible and takes `&self`, because a host may run the same
/// plugin twice (in tests, or under a second host) and the plugin should not
/// carry state between them. A plugin that is rejected half way through leaves
/// the host as it was — see [`PluginHost::register`].
pub trait Plugin {
    /// Unique plugin name, e.g. `"openforge.interchange"`.
    fn name(&self) -> &str;
    fn version(&self) -> &str {
        "0.1.0"
    }
    /// A one-line description for the plugins panel.
    fn description(&self) -> &str {
        ""
    }
    /// Publish this plugin's services, tools and filters.
    fn register(&self, host: &mut PluginHost) -> Result<(), PluginError>;
}

/// One loaded plugin, as the UI reports it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoadedPlugin {
    pub name: String,
    pub version: String,
    pub description: String,
}

/// A factory for one tool.
///
/// Tools are *constructed* per use rather than shared: a `Tool` carries the
/// state of an in-progress drag, so handing the same instance to two viewports
/// (or to the next drag) would leak that state.
pub type ToolFactory = fn() -> Box<dyn Tool>;

/// What a plugin registers into, and what the app asks for a tool by name.
#[derive(Default)]
pub struct PluginHost {
    pub services: ServiceRegistry,
    pub filters: FilterRegistry,
    tools: BTreeMap<String, (String, ToolFactory)>,
    plugins: Vec<LoadedPlugin>,
}

impl PluginHost {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a plugin.
    ///
    /// The plugin fills a *staged* host of its own, and only if that succeeds
    /// is anything merged in — so a plugin that clashes on its second
    /// registration cannot leave its first one behind, and a plugin that fails
    /// halfway through its own `register` leaves this host untouched.
    pub fn register(&mut self, plugin: &dyn Plugin) -> Result<(), PluginError> {
        let mut staged = PluginHost::new();
        plugin.register(&mut staged)?;
        if let Err(err) = self.merge(staged) {
            log::warn!("plugin `{}` rejected: {err}", plugin.name());
            return Err(err);
        }
        self.plugins.push(LoadedPlugin {
            name: plugin.name().to_string(),
            version: plugin.version().to_string(),
            description: plugin.description().to_string(),
        });
        log::info!("registered plugin `{}` {}", plugin.name(), plugin.version());
        Ok(())
    }

    /// Merge another host's registrations into this one, all or nothing.
    ///
    /// Every service, tool, filter id and extension is checked for a clash
    /// *before* the first entry is moved across, so the merge either applies
    /// completely or not at all.
    fn merge(&mut self, mut other: PluginHost) -> Result<(), PluginError> {
        for name in other.services.names() {
            if self.services.contains(name) {
                return Err(PluginError::duplicate("service", name));
            }
        }
        for name in other.tools.keys() {
            if self.tools.contains_key(name) {
                return Err(PluginError::duplicate("tool", name.clone()));
            }
        }
        for filter in other.filters.filters() {
            if self.filters.get(filter.id()).is_some() {
                return Err(PluginError::duplicate("filter", filter.id()));
            }
            for extension in filter.extensions() {
                if self.filters.for_extension(extension).is_some() {
                    return Err(PluginError::duplicate("extension", *extension));
                }
            }
        }

        for service in other.services.drain() {
            self.services.insert(service)?;
        }
        for filter in other.filters.drain() {
            self.filters.insert(filter)?;
        }
        for (name, tool) in std::mem::take(&mut other.tools) {
            self.tools.insert(name, tool);
        }
        Ok(())
    }

    /// Register a tool factory. The name must be unique.
    pub fn add_tool(
        &mut self,
        name: impl Into<String>,
        description: impl Into<String>,
        factory: ToolFactory,
    ) -> Result<(), PluginError> {
        let name = name.into();
        if name.is_empty() {
            return Err(PluginError::EmptyName { kind: "tool" });
        }
        if self.tools.contains_key(&name) {
            return Err(PluginError::duplicate("tool", name));
        }
        self.tools.insert(name, (description.into(), factory));
        Ok(())
    }

    /// Build the tool registered under `name` (or `None` when there is none).
    pub fn make_tool(&self, name: &str) -> Option<Box<dyn Tool>> {
        self.tools.get(name).map(|(_, factory)| factory())
    }

    pub fn has_tool(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    /// Tool names, alphabetically.
    pub fn tool_names(&self) -> impl Iterator<Item = &str> {
        self.tools.keys().map(|k| k.as_str())
    }

    /// `(name, description)` for the UI, alphabetically.
    pub fn tool_entries(&self) -> impl Iterator<Item = (&str, &str)> {
        self.tools.iter().map(|(k, (d, _))| (k.as_str(), d.as_str()))
    }

    pub fn tool_count(&self) -> usize {
        self.tools.len()
    }

    /// The plugins that loaded, in registration order.
    pub fn plugins(&self) -> &[LoadedPlugin] {
        &self.plugins
    }

    /// Read a file through the registered filters.
    pub fn import(&self, path: &Path) -> Result<LoadedScene, PluginError> {
        self.filters.import(path)
    }

    /// Write a file through the registered filters.
    pub fn export(&self, path: &Path, data: &MeshData) -> Result<(), PluginError> {
        self.filters.export(path, data)
    }
}

// --- The plugins OpenForge ships with -----------------------------------------

/// The editing tools that live in `openforge-core`.
pub struct CoreToolsPlugin;

impl Plugin for CoreToolsPlugin {
    fn name(&self) -> &str {
        "openforge.core-tools"
    }

    fn description(&self) -> &str {
        "Selection and the transform tools"
    }

    fn register(&self, host: &mut PluginHost) -> Result<(), PluginError> {
        host.add_tool("Select", "Click to select objects", || Box::new(SelectTool::default()))?;
        host.add_tool("Move", "Translate the selection", || {
            Box::new(TransformTool::new(TransformMode::Move))
        })?;
        host.add_tool("Scale", "Scale the selection", || {
            Box::new(TransformTool::new(TransformMode::Scale))
        })?;
        host.add_tool("Rotate", "Rotate the selection", || {
            Box::new(TransformTool::new(TransformMode::Rotate))
        })?;
        host.services.insert(Box::new(crate::service::SimpleService::new(
            "selection",
            "Which nodes and sub-elements are selected",
        )))?;
        host.services.insert(Box::new(crate::service::SimpleService::new(
            "undo",
            "Snapshot-based undo/redo of the scene",
        )))?;
        Ok(())
    }
}

/// The interchange formats in `openforge-io`.
pub struct InterchangePlugin;

impl Plugin for InterchangePlugin {
    fn name(&self) -> &str {
        "openforge.interchange"
    }

    fn description(&self) -> &str {
        "OBJ, glTF/GLB, PLY and STL import/export"
    }

    fn register(&self, host: &mut PluginHost) -> Result<(), PluginError> {
        for filter in engine_filters() {
            host.filters.insert(filter)?;
        }
        host.services.insert(Box::new(crate::service::SimpleService::new(
            "io",
            "Reading and writing mesh files",
        )))?;
        Ok(())
    }
}

/// The plugins a stock OpenForge starts with.
pub fn builtin_plugins() -> Vec<Box<dyn Plugin>> {
    vec![Box::new(CoreToolsPlugin), Box::new(InterchangePlugin)]
}

/// A host with every built-in plugin registered.
///
/// The built-ins are compiled in and cannot fail to load; if one ever did, that
/// is a programming error rather than a user-facing condition.
pub fn builtin_host() -> PluginHost {
    let mut host = PluginHost::new();
    for plugin in builtin_plugins() {
        if let Err(err) = host.register(plugin.as_ref()) {
            log::error!("built-in plugin `{}` failed to register: {err}", plugin.name());
        }
    }
    host
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter::FormatFilter;
    use crate::service::SimpleService;

    /// A plugin that registers one service and one filter.
    struct Demo;

    struct DemoFilter;
    impl FormatFilter for DemoFilter {
        fn id(&self) -> &str {
            "demo"
        }
        fn label(&self) -> &str {
            "Demo format"
        }
        fn extensions(&self) -> &[&str] {
            &["demo"]
        }
        fn import(&self, _path: &Path) -> openforge_io::IoResult<LoadedScene> {
            Ok(LoadedScene { meshes: Vec::new(), materials: Vec::new() })
        }
        fn export(&self, _path: &Path, _data: &MeshData) -> openforge_io::IoResult<()> {
            Ok(())
        }
    }

    impl Plugin for Demo {
        fn name(&self) -> &str {
            "demo"
        }
        fn description(&self) -> &str {
            "a demonstration plugin"
        }
        fn register(&self, host: &mut PluginHost) -> Result<(), PluginError> {
            host.services.insert(Box::new(SimpleService::new("demo-service", "does demo things")))?;
            host.filters.insert(Box::new(DemoFilter))?;
            host.add_tool("Demo Tool", "does nothing", || Box::new(SelectTool::default()))?;
            Ok(())
        }
    }

    /// A plugin that clashes on its *second* registration, to test rollback.
    struct Clashing;
    impl Plugin for Clashing {
        fn name(&self) -> &str {
            "clashing"
        }
        fn register(&self, host: &mut PluginHost) -> Result<(), PluginError> {
            host.services.insert(Box::new(SimpleService::new("clashing-service", "")))?;
            host.add_tool("Select", "a clash with the built-ins", || {
                Box::new(SelectTool::default())
            })?;
            Ok(())
        }
    }

    #[test]
    fn the_builtin_host_has_the_tools_and_formats_the_app_needs() {
        let host = builtin_host();
        assert_eq!(
            host.tool_names().collect::<Vec<_>>(),
            ["Move", "Rotate", "Scale", "Select"]
        );
        assert!(host.has_tool("Move") && host.has_tool("Select"));
        assert_eq!(host.filters.ids().collect::<Vec<_>>(), ["glb", "obj", "ply", "stl"]);
        assert!(host.services.contains("selection") && host.services.contains("io"));
        assert_eq!(host.plugins().len(), 2);
        assert_eq!(host.plugins()[0].name, "openforge.core-tools");
        assert_eq!(host.plugins()[1].version, "0.1.0");
        assert_eq!(host.plugins()[1].description, "OBJ, glTF/GLB, PLY and STL import/export");
    }

    #[test]
    fn a_tool_factory_hands_back_a_fresh_tool() {
        let host = builtin_host();
        let tool = host.make_tool("Scale").expect("registered");
        assert_eq!(tool.name(), "Scale");
        assert!(host.make_tool("Nope").is_none());
        // Two lookups are two tool instances, not one shared drag state.
        assert!(host.make_tool("Move").is_some());
        assert!(host.make_tool("Move").is_some());
    }

    #[test]
    fn a_plugin_can_extend_the_host() {
        let mut host = builtin_host();
        host.register(&Demo).unwrap();
        assert!(host.services.contains("demo-service"));
        assert_eq!(host.filters.for_extension("demo").unwrap().id(), "demo");
        assert_eq!(host.tool_count(), 5);
        assert_eq!(host.plugins().len(), 3);
        assert_eq!(host.plugins()[2].description, "a demonstration plugin");
    }

    #[test]
    fn a_rejected_plugin_leaves_the_host_untouched() {
        let mut host = builtin_host();
        let tools = host.tool_count();
        let services = host.services.len();

        let err = host.register(&Clashing).unwrap_err();
        assert_eq!(err, PluginError::Duplicate { kind: "tool", name: "Select".into() });
        assert_eq!(host.tool_count(), tools, "the tool it did not clash on is gone");
        assert_eq!(host.services.len(), services, "…and so is its service");
        assert!(!host.services.contains("clashing-service"));
        assert_eq!(host.plugins().len(), 2, "a rejected plugin is not reported as loaded");
    }

    #[test]
    fn a_plugin_looking_for_an_unknown_tool_gets_nothing() {
        let host = builtin_host();
        assert!(host.make_tool("Bevel").is_none());
        assert_eq!(host.tool_entries().count(), 4);
        assert!(host.tool_entries().all(|(name, desc)| !name.is_empty() && !desc.is_empty()));
    }

    #[test]
    fn an_empty_host_is_usable() {
        let host = PluginHost::new();
        assert_eq!(host.tool_count(), 0);
        assert!(host.plugins().is_empty());
        assert!(host.services.is_empty());
        assert!(host.filters.is_empty());
        assert!(host.import(Path::new("nothing.obj")).is_err());
    }
}
