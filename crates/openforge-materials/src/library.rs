//! The material library: the app-facing collection of materials and textures.

use std::path::Path;
use std::sync::OnceLock;

use openforge_core::{CoreError, CoreResult};

use crate::material::Material;
use crate::texture::Texture;
use crate::error::MaterialError;

/// Owns the project's materials and textures.
///
/// Mesh faces reference materials by slot (a `u32`), and materials reference
/// textures by index into this library's texture list. The library is created
/// with a default grey material at index 0 so that every slot resolves to
/// *something* renderable.
#[derive(Clone, PartialEq, Debug)]
pub struct MaterialLibrary {
    /// Materials; slot `n` in a mesh resolves to `materials[n]`.
    pub materials: Vec<Material>,
    /// Textures referenced by `Material::{diffuse,normal,specular}_map`.
    pub textures: Vec<Texture>,
}

/// The canonical grey fallback, built once and shared.
fn fallback_material() -> &'static Material {
    static FALLBACK: OnceLock<Material> = OnceLock::new();
    FALLBACK.get_or_init(Material::default)
}

impl Default for MaterialLibrary {
    fn default() -> Self {
        MaterialLibrary::new()
    }
}

impl MaterialLibrary {
    /// An empty library that already contains the default grey material at
    /// index 0 (see [`MaterialLibrary::default_material`]).
    pub fn new() -> Self {
        MaterialLibrary {
            materials: vec![MaterialLibrary::default_material()],
            textures: Vec::new(),
        }
    }

    /// The canonical grey fallback material also added at index 0 by
    /// [`MaterialLibrary::new`].
    pub fn default_material() -> Material {
        Material::default()
    }

    /// Append a material and return its slot index.
    pub fn add_material(&mut self, material: Material) -> usize {
        self.materials.push(material);
        self.materials.len() - 1
    }

    /// Append a texture and return its index.
    pub fn add_texture(&mut self, texture: Texture) -> usize {
        self.textures.push(texture);
        self.textures.len() - 1
    }

    /// Number of materials in the library.
    pub fn material_count(&self) -> usize {
        self.materials.len()
    }

    /// Number of textures in the library.
    pub fn texture_count(&self) -> usize {
        self.textures.len()
    }

    /// The material for a mesh's slot, falling back to the default material
    /// when the slot is out of range. Never fails.
    pub fn resolve(&self, slot: u32) -> &Material {
        let idx = slot as usize;
        if let Some(m) = self.materials.get(idx) {
            return m;
        }
        log::debug!("material slot {slot} out of range; using default material");
        if let Some(m) = self.materials.first() {
            return m;
        }
        fallback_material()
    }

    /// Look up a texture by optional index (`None` or an out-of-range index
    /// yields `None`).
    pub fn resolve_texture(&self, index: Option<usize>) -> Option<&Texture> {
        index.and_then(|i| self.textures.get(i))
    }

    /// Decode a PNG or JPEG file into a texture and append it to the library,
    /// returning its index.
    ///
    /// I/O and decode failures are logged with details and reported as
    /// [`CoreError::Invalid`]; use [`MaterialLibrary::try_load_texture_from_file`]
    /// when the caller needs the full [`MaterialError`].
    pub fn load_texture_from_file(&mut self, path: &Path) -> CoreResult<usize> {
        self.try_load_texture_from_file(path).map_err(|e| {
            log::error!("failed to load texture `{}`: {e}", path.display());
            match &e {
                MaterialError::Io { .. } => CoreError::Invalid("failed to read texture file"),
                MaterialError::Decode { .. } => CoreError::Invalid("failed to decode texture file"),
            }
        })
    }

    /// Like [`MaterialLibrary::load_texture_from_file`] but surfaces the full
    /// [`MaterialError`] (I/O vs. decode, with details).
    pub fn try_load_texture_from_file(&mut self, path: &Path) -> Result<usize, MaterialError> {
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.display().to_string());
        let bytes = std::fs::read(path).map_err(|e| MaterialError::io(path, e))?;
        let texture = Texture::from_encoded(name, &bytes)?;
        Ok(self.add_texture(texture))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_library_starts_with_default_material() {
        let lib = MaterialLibrary::new();
        assert_eq!(lib.materials.len(), 1);
        assert!(lib.textures.is_empty());
        assert_eq!(lib.resolve(0), &MaterialLibrary::default_material());
    }

    #[test]
    fn add_material_returns_slot_and_resolves() {
        let mut lib = MaterialLibrary::new();
        let slot = lib.add_material(Material::new("Rusty Iron"));
        assert_eq!(slot, 1);
        assert_eq!(lib.materials.len(), 2);
        assert_eq!(lib.resolve(1).name, "Rusty Iron");
        assert_eq!(lib.resolve(0).name, "Default");
    }

    #[test]
    fn resolve_falls_back_to_default_for_bad_slots() {
        let mut lib = MaterialLibrary::new();
        lib.add_material(Material::new("Custom"));
        // Out-of-range and u32::MAX slots resolve to the grey default.
        assert_eq!(lib.resolve(7).name, "Default");
        assert_eq!(lib.resolve(u32::MAX).diffuse, Material::default().diffuse);
        // Even a manually emptied library resolves without panicking.
        lib.materials.clear();
        assert_eq!(lib.resolve(0).name, "Default");
    }

    #[test]
    fn add_texture_returns_indices_and_resolves() {
        let mut lib = MaterialLibrary::new();
        assert_eq!(lib.add_texture(Texture::solid("a", 1, 1, [1, 2, 3, 4])), 0);
        assert_eq!(lib.add_texture(Texture::solid("b", 1, 1, [5, 6, 7, 8])), 1);
        assert_eq!(lib.texture_count(), 2);
        assert_eq!(lib.resolve_texture(Some(0)).unwrap().name, "a");
        assert_eq!(lib.resolve_texture(Some(1)).unwrap().name, "b");
        assert_eq!(lib.resolve_texture(Some(9)).map(|_| ()), None);
        assert_eq!(lib.resolve_texture(None).map(|_| ()), None);
    }

    #[test]
    fn material_maps_index_into_library_textures() {
        let mut lib = MaterialLibrary::new();
        let tex = lib.add_texture(Texture::checkerboard("chk", 8, 2, [0; 4], [255; 4]));
        let mut m = Material::new("Checked");
        m.diffuse_map = Some(tex);
        lib.add_material(m);
        let resolved = lib.resolve(1);
        let map = lib.resolve_texture(resolved.diffuse_map).expect("diffuse map exists");
        assert_eq!(map.sample_nearest(0.1, 0.1), [0, 0, 0, 0]);
        // x = 7, y = 0 falls in an odd cell: the second checker color.
        assert_eq!(map.sample_nearest(0.9, 0.1), [255, 255, 255, 255]);
    }

    #[test]
    fn load_texture_from_file_missing_path_errors() {
        let mut lib = MaterialLibrary::new();
        let err = lib
            .load_texture_from_file(Path::new("/nonexistent/openforge/no_such_texture.png"))
            .unwrap_err();
        assert!(matches!(err, CoreError::Invalid(_)));
        // The library is unchanged by the failed load.
        assert_eq!(lib.texture_count(), 0);

        // The detailed error distinguishes I/O from decode failures.
        let detailed = lib
            .try_load_texture_from_file(Path::new("/nonexistent/openforge/no_such_texture.png"))
            .unwrap_err();
        assert!(matches!(detailed, MaterialError::Io { .. }));
    }

    #[test]
    fn load_texture_from_file_decodes_png() {
        // Build a tiny PNG in memory, write it to a temp file, load it back.
        let source = Texture::solid("src", 2, 2, [10, 20, 30, 40]);
        let mut png: Vec<u8> = Vec::new();
        image::DynamicImage::ImageRgba8(
            image::RgbaImage::from_raw(2, 2, source.pixels.clone()).expect("raw image"),
        )
        .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
        .expect("png encode");

        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "openforge-materials-test-{}-{}.png",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        std::fs::write(&path, &png).expect("write temp png");

        let mut lib = MaterialLibrary::new();
        let idx = lib.load_texture_from_file(&path).expect("decode png");
        let tex = &lib.textures[idx];
        assert_eq!((tex.width, tex.height), (2, 2));
        assert_eq!(tex.sample_nearest(0.25, 0.25), [10, 20, 30, 40]);

        let _ = std::fs::remove_file(&path);
    }
}
