//! Surface material description.

use glam::{Vec3, Vec4};

/// A surface material: colors, optional texture maps and rendering flags.
///
/// Texture maps are indices into a [`crate::MaterialLibrary`]'s texture list;
/// `None` means the map is absent. `Face::material` values in
/// `openforge-core` meshes index the library's material list, and
/// [`crate::MaterialLibrary::resolve`] maps such a slot to a `Material`.
#[derive(Clone, PartialEq, Debug)]
pub struct Material {
    /// Human-readable name (shown in material pickers, preserved by importers).
    pub name: String,
    /// Base color, RGBA with components in `0.0..=1.0`.
    pub diffuse: Vec4,
    /// Specular (highlight) color, RGB in `0.0..=1.0`.
    pub specular: Vec3,
    /// Self-illumination color, RGB in `0.0..=1.0`.
    pub emissive: Vec3,
    /// Specular exponent; higher values give tighter highlights.
    pub shininess: f32,
    /// Overall opacity in `0.0..=1.0` (multiplied with `diffuse.a` when shading).
    pub opacity: f32,
    /// Index into the library's texture list for the base color map.
    pub diffuse_map: Option<usize>,
    /// Index into the library's texture list for the normal map.
    pub normal_map: Option<usize>,
    /// Index into the library's texture list for the specular intensity map.
    pub specular_map: Option<usize>,
    /// When `true`, back faces are rendered instead of being culled.
    pub two_sided: bool,
}

impl Default for Material {
    /// A neutral grey material: mid-grey diffuse, faint specular, fully opaque.
    fn default() -> Self {
        Material {
            name: "Default".to_string(),
            diffuse: Vec4::new(0.5, 0.5, 0.5, 1.0),
            specular: Vec3::new(0.1, 0.1, 0.1),
            emissive: Vec3::ZERO,
            shininess: 32.0,
            opacity: 1.0,
            diffuse_map: None,
            normal_map: None,
            specular_map: None,
            two_sided: false,
        }
    }
}

impl Material {
    /// A default grey material carrying the given name.
    pub fn new(name: impl Into<String>) -> Self {
        Material { name: name.into(), ..Material::default() }
    }

    /// The effective base color: `diffuse` with `opacity` folded into alpha.
    pub fn effective_color(&self) -> Vec4 {
        Vec4::new(self.diffuse.x, self.diffuse.y, self.diffuse.z, self.diffuse.w * self.opacity)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_material_is_opaque_grey() {
        let m = Material::default();
        assert_eq!(m.name, "Default");
        assert_eq!(m.diffuse, Vec4::new(0.5, 0.5, 0.5, 1.0));
        assert_eq!(m.specular, Vec3::new(0.1, 0.1, 0.1));
        assert_eq!(m.emissive, Vec3::ZERO);
        assert_eq!(m.shininess, 32.0);
        assert_eq!(m.opacity, 1.0);
        assert_eq!(m.diffuse_map, None);
        assert_eq!(m.normal_map, None);
        assert_eq!(m.specular_map, None);
        assert!(!m.two_sided);
    }

    #[test]
    fn new_material_keeps_defaults_and_sets_name() {
        let m = Material::new("Painted Metal");
        assert_eq!(m.name, "Painted Metal");
        assert_eq!(m.diffuse, Material::default().diffuse);
        assert_eq!(m.opacity, 1.0);
    }

    #[test]
    fn effective_color_folds_opacity_into_alpha() {
        let mut m = Material::default();
        m.diffuse = Vec4::new(1.0, 0.0, 0.0, 0.8);
        m.opacity = 0.5;
        let c = m.effective_color();
        assert!((c.w - 0.4).abs() < 1e-6);
        assert_eq!((c.x, c.y, c.z), (1.0, 0.0, 0.0));
    }
}
