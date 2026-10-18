//! Format filters: importers and exporters behind the plugin host.
//!
//! The engine's seven formats are ordinary plugins here, so a third-party
//! filter for another extension is a registration rather than a change to the
//! app: [`FilterRegistry::import`] dispatches on a file's extension, and the
//! app never names a format again.

use crate::error::PluginError;
use openforge_io::{dae, fbx, gltf, obj, off, ply, stl, Format, IoResult, LoadedScene, MeshData};
use std::collections::BTreeMap;
use std::path::Path;

/// One file format, as a plugin sees it.
pub trait FormatFilter: Send + Sync {
    /// Stable id, e.g. `"obj"`. Unique within the registry.
    fn id(&self) -> &str;
    /// Display name, e.g. `"Wavefront OBJ"`.
    fn label(&self) -> &str;
    /// Extensions handled, lowercase and without the dot.
    fn extensions(&self) -> &[&str];
    /// Whether this filter can read. Defaults to `true`.
    fn can_import(&self) -> bool {
        true
    }
    /// Whether this filter can write. Defaults to `true`.
    fn can_export(&self) -> bool {
        true
    }
    /// Read `path` into scene-ready meshes.
    fn import(&self, path: &Path) -> IoResult<LoadedScene>;
    /// Write `data` to `path`.
    fn export(&self, path: &Path, data: &MeshData) -> IoResult<()>;
}

/// Build one of the engine's filters for a [`Format`].
fn engine_filter(format: Format) -> Box<dyn FormatFilter> {
    match format {
        Format::Obj => Box::new(EngineFilter::new(Format::Obj)),
        Format::Gltf => Box::new(EngineFilter::new(Format::Gltf)),
        Format::Ply => Box::new(EngineFilter::new(Format::Ply)),
        Format::Stl => Box::new(EngineFilter::new(Format::Stl)),
        Format::Collada => Box::new(EngineFilter::new(Format::Collada)),
        Format::Off => Box::new(EngineFilter::new(Format::Off)),
        Format::Fbx => Box::new(EngineFilter::new(Format::Fbx)),
    }
}

/// A thin adapter over `openforge_io`'s per-format modules.
struct EngineFilter {
    format: Format,
    extensions: Vec<&'static str>,
}

impl EngineFilter {
    fn new(format: Format) -> Self {
        // glTF covers two extensions; the rest map one to one.
        let extensions = match format {
            Format::Gltf => vec!["gltf", "glb"],
            other => vec![other.extension()],
        };
        Self { format, extensions }
    }
}

impl FormatFilter for EngineFilter {
    fn id(&self) -> &str {
        self.format.extension()
    }

    fn label(&self) -> &str {
        self.format.filter_label()
    }

    fn extensions(&self) -> &[&str] {
        &self.extensions
    }

    fn import(&self, path: &Path) -> IoResult<LoadedScene> {
        match self.format {
            Format::Obj => obj::import(path),
            Format::Gltf => gltf::import(path),
            Format::Ply => ply::import(path),
            Format::Stl => stl::import(path),
            Format::Collada => dae::import(path),
            Format::Off => off::import(path),
            Format::Fbx => fbx::import(path),
        }
    }

    fn export(&self, path: &Path, data: &MeshData) -> IoResult<()> {
        match self.format {
            Format::Obj => obj::export(path, data),
            Format::Gltf => gltf::export(path, data),
            Format::Ply => ply::export(path, data),
            Format::Stl => stl::export(path, data),
            Format::Collada => dae::export(path, data),
            Format::Off => off::export(path, data),
            Format::Fbx => fbx::export(path, data),
        }
    }
}

/// Filters keyed by id, plus extension dispatch.
#[derive(Default)]
pub struct FilterRegistry {
    filters: BTreeMap<String, Box<dyn FormatFilter>>,
    /// Extension → filter id.
    by_extension: BTreeMap<String, String>,
}

impl FilterRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a filter. Duplicate ids *and* duplicate extensions are both
    /// rejected: dispatch by extension has to be unambiguous.
    pub fn insert(&mut self, filter: Box<dyn FormatFilter>) -> Result<(), PluginError> {
        let id = filter.id().to_string();
        if id.is_empty() {
            return Err(PluginError::EmptyName { kind: "filter" });
        }
        if self.filters.contains_key(&id) {
            return Err(PluginError::duplicate("filter", id));
        }
        for extension in filter.extensions() {
            let ext = extension.to_ascii_lowercase();
            if ext.is_empty() {
                return Err(PluginError::EmptyName { kind: "extension" });
            }
            if let Some(owner) = self.by_extension.get(&ext) {
                return Err(PluginError::duplicate(
                    "extension",
                    format!("{ext} (also handled by `{owner}`)"),
                ));
            }
        }
        for extension in filter.extensions() {
            self.by_extension.insert(extension.to_ascii_lowercase(), id.clone());
        }
        self.filters.insert(id, filter);
        Ok(())
    }

    pub fn get(&self, id: &str) -> Option<&dyn FormatFilter> {
        self.filters.get(id).map(|f| f.as_ref())
    }

    /// The filter that handles `extension` (case-insensitive, no dot).
    pub fn for_extension(&self, extension: &str) -> Option<&dyn FormatFilter> {
        let ext = extension.trim_start_matches('.').to_ascii_lowercase();
        self.by_extension.get(&ext).and_then(|id| self.get(id))
    }

    /// The filter that handles `path`'s extension.
    pub fn for_path(&self, path: &Path) -> Option<&dyn FormatFilter> {
        path.extension().and_then(|e| e.to_str()).and_then(|e| self.for_extension(e))
    }

    /// Read a file, dispatching on its extension.
    pub fn import(&self, path: &Path) -> Result<LoadedScene, PluginError> {
        match self.for_path(path) {
            Some(filter) if filter.can_import() => filter.import(path).map_err(|e| {
                log::error!("plugin filter `{}` failed to import: {e}", filter.id());
                PluginError::not_found("import error", format!("{}: {e}", filter.id()))
            }),
            Some(filter) => Err(PluginError::not_found("importer", filter.id())),
            None => Err(PluginError::not_found("filter for extension", extension_of(path))),
        }
    }

    /// Write a file, dispatching on its extension.
    pub fn export(&self, path: &Path, data: &MeshData) -> Result<(), PluginError> {
        match self.for_path(path) {
            Some(filter) if filter.can_export() => filter.export(path, data).map_err(|e| {
                log::error!("plugin filter `{}` failed to export: {e}", filter.id());
                PluginError::not_found("export error", format!("{}: {e}", filter.id()))
            }),
            Some(filter) => Err(PluginError::not_found("exporter", filter.id())),
            None => Err(PluginError::not_found("filter for extension", extension_of(path))),
        }
    }

    /// Filter ids, alphabetically.
    pub fn ids(&self) -> impl Iterator<Item = &str> {
        self.filters.keys().map(|k| k.as_str())
    }

    /// `(id, label, extensions)` for the UI, alphabetically by id.
    pub fn entries(&self) -> impl Iterator<Item = (&str, &str, &[&str])> {
        self.filters.iter().map(|(k, f)| (k.as_str(), f.label(), f.extensions()))
    }

    /// Every extension served, alphabetically — what a file dialog should offer.
    pub fn extensions(&self) -> impl Iterator<Item = &str> {
        self.by_extension.keys().map(|k| k.as_str())
    }

    /// Every filter, unsorted, for merging one registry into another.
    pub fn filters(&self) -> impl Iterator<Item = &dyn FormatFilter> {
        self.filters.values().map(|f| f.as_ref())
    }

    /// Remove and hand back every filter (used when merging registries).
    pub fn drain(&mut self) -> Vec<Box<dyn FormatFilter>> {
        self.by_extension.clear();
        std::mem::take(&mut self.filters).into_values().collect()
    }

    pub fn len(&self) -> usize {
        self.filters.len()
    }

    pub fn is_empty(&self) -> bool {
        self.filters.is_empty()
    }
}

fn extension_of(path: &Path) -> String {
    path.extension().and_then(|e| e.to_str()).unwrap_or("<none>").to_string()
}

/// The engine's own filters, as a plugin would register them.
pub fn engine_filters() -> Vec<Box<dyn FormatFilter>> {
    [Format::Obj, Format::Gltf, Format::Ply, Format::Stl, Format::Collada, Format::Off, Format::Fbx]
        .into_iter()
        .map(engine_filter)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use openforge_core::mesh::Mesh;
    use openforge_core::Vec3;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static COUNTER: AtomicUsize = AtomicUsize::new(0);

    /// A unique temp path, so parallel tests do not collide.
    fn temp_path(extension: &str) -> std::path::PathBuf {
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("openforge-plugin-{}-{n}.{extension}", std::process::id()))
    }

    fn triangle(name: &str) -> MeshData {
        let mut mesh = Mesh::new();
        mesh.add_vertex(Vec3::ZERO);
        mesh.add_vertex(Vec3::X);
        mesh.add_vertex(Vec3::Y);
        mesh.add_face(&[0, 1, 2]);
        mesh.compute_flat_normals();
        MeshData { name: name.into(), mesh, material: None }
    }

    fn registry_with_engine_filters() -> FilterRegistry {
        let mut registry = FilterRegistry::new();
        for filter in engine_filters() {
            registry.insert(filter).unwrap();
        }
        registry
    }

    #[test]
    fn engine_filters_register_by_id_and_extension() {
        let registry = registry_with_engine_filters();
        assert_eq!(registry.len(), 7);
        assert_eq!(
            registry.ids().collect::<Vec<_>>(),
            ["dae", "fbx", "glb", "obj", "off", "ply", "stl"]
        );
        // glTF covers both extensions.
        assert_eq!(registry.for_extension("glb").unwrap().id(), "glb");
        assert_eq!(registry.for_extension("gltf").unwrap().id(), "glb");
        assert_eq!(registry.for_extension("OBJ").unwrap().id(), "obj");
        assert_eq!(registry.for_extension(".StL").unwrap().id(), "stl");
        assert_eq!(registry.for_extension("dae").unwrap().label(), "COLLADA");
        assert_eq!(registry.for_extension("off").unwrap().label(), "OFF");
        assert_eq!(registry.for_extension("fbx").unwrap().label(), "FBX (ASCII)");
    }

    #[test]
    fn a_duplicate_id_or_extension_is_rejected() {
        let mut registry = registry_with_engine_filters();
        // Same id, same extensions.
        let err = registry.insert(Box::new(EngineFilter::new(Format::Obj))).unwrap_err();
        assert_eq!(err, PluginError::Duplicate { kind: "filter", name: "obj".into() });

        // A different id claiming an extension someone already serves.
        struct Claimer;
        impl FormatFilter for Claimer {
            fn id(&self) -> &str {
                "obj2"
            }
            fn label(&self) -> &str {
                "Also OBJ"
            }
            fn extensions(&self) -> &[&str] {
                &["obj"]
            }
            fn import(&self, _path: &Path) -> IoResult<LoadedScene> {
                Err(openforge_io::IoError::Unsupported("no"))
            }
            fn export(&self, _path: &Path, _data: &MeshData) -> IoResult<()> {
                Err(openforge_io::IoError::Unsupported("no"))
            }
        }
        let err = registry.insert(Box::new(Claimer)).unwrap_err();
        assert!(matches!(err, PluginError::Duplicate { kind: "extension", .. }), "{err:?}");
        assert_eq!(registry.len(), 7, "the failed filter left nothing behind");
    }

    #[test]
    fn import_and_export_dispatch_on_the_extension() {
        let registry = registry_with_engine_filters();
        let path = temp_path("obj");
        registry.export(&path, &triangle("tri")).unwrap();
        assert!(path.exists());

        let loaded = registry.import(&path).unwrap();
        assert_eq!(loaded.mesh_count(), 1);
        assert_eq!(loaded.meshes[0].mesh.vertex_count(), 3);
        assert_eq!(loaded.meshes[0].mesh.face_count(), 1);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn an_unknown_extension_is_reported_not_guessed() {
        let registry = registry_with_engine_filters();
        // `LoadedScene` is not `Debug`, so `err()` rather than `unwrap_err()`.
        let err = registry.import(&temp_path("usdz")).err().expect("an error");
        match err {
            PluginError::NotFound { kind, name } => {
                assert_eq!(kind, "filter for extension");
                assert_eq!(name, "usdz");
            }
            other => panic!("expected NotFound, got {other:?}"),
        }
    }

    #[test]
    fn a_filter_can_be_import_only() {
        struct Writer;
        impl FormatFilter for Writer {
            fn id(&self) -> &str {
                "writeonly"
            }
            fn label(&self) -> &str {
                "Write only"
            }
            fn extensions(&self) -> &[&str] {
                &["wo"]
            }
            fn can_import(&self) -> bool {
                false
            }
            fn import(&self, _path: &Path) -> IoResult<LoadedScene> {
                panic!("must not be called")
            }
            fn export(&self, _path: &Path, _data: &MeshData) -> IoResult<()> {
                Ok(())
            }
        }
        let mut registry = FilterRegistry::new();
        registry.insert(Box::new(Writer)).unwrap();
        assert!(!registry.for_extension("wo").unwrap().can_import());
        let err = registry.import(&temp_path("wo")).err().expect("an error");
        assert_eq!(err, PluginError::NotFound { kind: "importer", name: "writeonly".into() });
        registry.export(&temp_path("wo"), &triangle("x")).unwrap();
    }

    #[test]
    fn extensions_lists_what_a_file_dialog_should_offer() {
        let registry = registry_with_engine_filters();
        assert_eq!(
            registry.extensions().collect::<Vec<_>>(),
            ["dae", "fbx", "glb", "gltf", "obj", "off", "ply", "stl"]
        );
    }
}
