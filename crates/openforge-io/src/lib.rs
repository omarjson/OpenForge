//! Import/export of mesh data across common interchange formats.
//!
//! Every loader returns [`LoadedScene`], a lightweight description of imported
//! geometry that the app can turn into core scene nodes. Every writer takes
//! [`MeshData`].

use openforge_core::mesh::Mesh;
use openforge_core::Vec3;
use std::path::Path;
use thiserror::Error;

pub mod gltf;
pub mod obj;
pub mod ply;
pub mod stl;

/// Errors from loading/saving.
#[derive(Debug, Error)]
pub enum IoError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("parse error in {format}: {message}")]
    Parse { format: &'static str, message: String },
    #[error("unsupported: {0}")]
    Unsupported(&'static str),
    #[error("missing data: {0}")]
    Missing(&'static str),
}

pub type IoResult<T> = Result<T, IoError>;

/// A named mesh with an optional material name, ready for scene insertion.
pub struct MeshData {
    pub name: String,
    pub mesh: Mesh,
    pub material: Option<String>,
}

/// The result of importing a file.
pub struct LoadedScene {
    pub meshes: Vec<MeshData>,
    /// Names of materials referenced by the meshes, if the format carried them.
    pub materials: Vec<String>,
}

impl LoadedScene {
    pub fn mesh_count(&self) -> usize {
        self.meshes.len()
    }
}

/// Detect a format from a file extension (lowercase, no dot).
pub fn format_from_extension(ext: &str) -> Option<Format> {
    match ext {
        "obj" => Some(Format::Obj),
        "gltf" | "glb" => Some(Format::Gltf),
        "ply" => Some(Format::Ply),
        "stl" => Some(Format::Stl),
        _ => None,
    }
}

/// Supported formats.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Format {
    Obj,
    Gltf,
    Ply,
    Stl,
}

impl Format {
    pub fn extension(self) -> &'static str {
        match self {
            Format::Obj => "obj",
            Format::Gltf => "glb",
            Format::Ply => "ply",
            Format::Stl => "stl",
        }
    }

    pub fn filter_label(self) -> &'static str {
        match self {
            Format::Obj => "Wavefront OBJ",
            Format::Gltf => "glTF 2.0",
            Format::Ply => "Stanford PLY",
            Format::Stl => "STL",
        }
    }
}

/// Import a file, dispatching on its extension.
pub fn import_path(path: &Path) -> IoResult<LoadedScene> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    match format_from_extension(&ext) {
        Some(Format::Obj) => obj::import(path),
        Some(Format::Gltf) => gltf::import(path),
        Some(Format::Ply) => ply::import(path),
        Some(Format::Stl) => stl::import(path),
        None => Err(IoError::Unsupported("unrecognized file extension")),
    }
}

/// Export a single mesh to a file, dispatching on its extension.
pub fn export_path(path: &Path, data: &MeshData) -> IoResult<()> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    match format_from_extension(&ext) {
        Some(Format::Obj) => obj::export(path, data),
        Some(Format::Gltf) => gltf::export(path, data),
        Some(Format::Ply) => ply::export(path, data),
        Some(Format::Stl) => stl::export(path, data),
        None => Err(IoError::Unsupported("unrecognized file extension")),
    }
}

/// Convenience used by loaders: normalize a mesh so normals exist.
pub(crate) fn finalize_mesh(mut mesh: Mesh) -> Mesh {
    if mesh.normals.is_empty() || mesh.normals.len() == 1 && mesh.normals[0] == Vec3::Y {
        mesh.compute_smooth_normals();
    }
    mesh
}
