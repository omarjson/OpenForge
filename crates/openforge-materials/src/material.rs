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
    /// Index into the library's texture list for the self-illumination map.
    pub emissive_map: Option<usize>,
    /// When `true`, back faces are rendered instead of being culled.
    pub two_sided: bool,
}

/// The four texture maps a material can carry.
///
/// The fields stay separate on [`Material`] (the project file, the importers
/// and the GPU vertex all name them one by one), but the *code* — the material
/// editor, the sample helpers — loops over them through this enum, so a fifth
/// map would be one arm here rather than a fourth near-identical branch in
/// every caller.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum MapKind {
    /// Base colour map, multiplied into the surface colour.
    Diffuse,
    /// Tangent-space normal map.
    Normal,
    /// Specular intensity map, scaling the material's highlight.
    Specular,
    /// Self-illumination map, added to the lit colour.
    Emissive,
}

impl MapKind {
    /// Every kind, in the order the material editor lists them.
    pub const ALL: [MapKind; 4] = [
        MapKind::Diffuse,
        MapKind::Normal,
        MapKind::Specular,
        MapKind::Emissive,
    ];

    /// The kind's name, as the material editor labels it.
    pub fn label(self) -> &'static str {
        match self {
            MapKind::Diffuse => "Diffuse",
            MapKind::Normal => "Normal",
            MapKind::Specular => "Specular",
            MapKind::Emissive => "Emissive",
        }
    }
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
            emissive_map: None,
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

    /// The texture index of one of the maps, `None` when that map is absent.
    pub fn map(&self, kind: MapKind) -> Option<usize> {
        match kind {
            MapKind::Diffuse => self.diffuse_map,
            MapKind::Normal => self.normal_map,
            MapKind::Specular => self.specular_map,
            MapKind::Emissive => self.emissive_map,
        }
    }

    /// Set (or clear, with `None`) one of the maps.
    pub fn set_map(&mut self, kind: MapKind, map: Option<usize>) {
        match kind {
            MapKind::Diffuse => self.diffuse_map = map,
            MapKind::Normal => self.normal_map = map,
            MapKind::Specular => self.specular_map = map,
            MapKind::Emissive => self.emissive_map = map,
        }
    }

    /// The maps this material carries, as `(kind, texture index)` pairs in
    /// [`MapKind::ALL`] order — what the material editor's rows are built from.
    pub fn maps(&self) -> Vec<(MapKind, usize)> {
        MapKind::ALL
            .into_iter()
            .filter_map(|kind| self.map(kind).map(|index| (kind, index)))
            .collect()
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
        assert_eq!(m.emissive_map, None);
        assert!(!m.two_sided);
    }

    #[test]
    fn maps_are_read_and_written_by_kind() {
        let mut m = Material::default();
        assert!(m.maps().is_empty(), "no maps to begin with");
        for (n, kind) in MapKind::ALL.into_iter().enumerate() {
            assert_eq!(m.map(kind), None, "{kind:?} starts absent");
            m.set_map(kind, Some(n));
            assert_eq!(m.map(kind), Some(n), "{kind:?} reads back");
            // Each kind has its own field, so the list grows by exactly one.
            assert_eq!(m.maps().len(), n + 1, "{kind:?} joined the list once");
        }
        assert_eq!(
            m.maps(),
            vec![
                (MapKind::Diffuse, 0),
                (MapKind::Normal, 1),
                (MapKind::Specular, 2),
                (MapKind::Emissive, 3),
            ],
            "the list is in MapKind::ALL order"
        );

        // Clearing one map leaves the rest.
        m.set_map(MapKind::Normal, None);
        assert_eq!(m.map(MapKind::Normal), None);
        assert_eq!(m.map(MapKind::Diffuse), Some(0));
        assert_eq!(m.map(MapKind::Emissive), Some(3));

        for kind in MapKind::ALL {
            m.set_map(kind, None);
        }
        assert!(m.maps().is_empty(), "and every map can be cleared");
    }

    #[test]
    fn every_map_kind_has_its_own_label() {
        let labels: Vec<&str> = MapKind::ALL.iter().map(|kind| kind.label()).collect();
        assert_eq!(labels, vec!["Diffuse", "Normal", "Specular", "Emissive"]);
        let unique: std::collections::HashSet<&str> = labels.iter().copied().collect();
        assert_eq!(unique.len(), MapKind::ALL.len(), "no two kinds share a label");
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
