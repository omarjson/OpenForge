//! The material maps, sampled: what a diffuse, normal, specular or emissive
//! map contributes at one surface sample.
//!
//! Both shading paths take their "no map" values from [`MapTexels::IDENTITY`],
//! so an absent map is a no-op in the same way everywhere — the GPU shader
//! samples the reserved white layer, this sampler returns the identity
//! without a texture at all — and the two cannot drift apart on what a map
//! does when it *is* there.
//!
//! The GPU path samples per fragment. The CPU preview shades one triangle at a
//! time — its one per-pixel feature is the interpolated diffuse texel the
//! painter draws through an egui mesh — so it samples the other three maps
//! once at the triangle's middle: resolution, not behaviour.

use crate::{MapKind, Material, MaterialLibrary};
use glam::{Vec2, Vec3};

/// What a material's maps contribute at one sample point.
///
/// The identity values are the ones a *missing* map contributes, and they are
/// what the WGSL fragment stage gets from the reserved white layer, so an
/// untextured face and a white-textured one shade the same.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct MapTexels {
    /// Base-colour factor; `1, 1, 1, 1` without a diffuse map.
    pub diffuse: [f32; 4],
    /// Tangent-space normal; `+Z` (straight out of the surface) without a
    /// normal map.
    pub normal: Vec3,
    /// Highlight intensity; `1.0` without a specular map.
    pub specular: f32,
    /// Self-illumination added to the lit colour; black without an emissive
    /// map.
    pub emissive: Vec3,
}

impl MapTexels {
    /// The identity: every map absent, nothing changed.
    pub const IDENTITY: MapTexels = MapTexels {
        diffuse: [1.0, 1.0, 1.0, 1.0],
        normal: Vec3::Z,
        specular: 1.0,
        emissive: Vec3::ZERO,
    };

    /// Sample every map `material` carries at `uv`.
    ///
    /// A map whose index the library does not hold, or whose texture is empty,
    /// contributes the identity — the same rule the GPU path gets by sampling
    /// the reserved white layer — so a dangling index can never sample past
    /// the end of the texture list.
    ///
    /// Texels are read as bytes and scaled to `0.0..=1.0`; a normal map's
    /// three channels are a tangent-space direction, so `0.5` (byte 128) means
    /// "straight out"; a specular map's intensity is its luminance, which is
    /// the whole value for the usual greyscale map; an emissive map is a
    /// colour.
    pub fn sample(library: &MaterialLibrary, material: &Material, uv: Vec2) -> MapTexels {
        let mut out = MapTexels::IDENTITY;
        for kind in MapKind::ALL {
            let Some(texture) = material
                .map(kind)
                .and_then(|index| library.textures.get(index))
            else {
                continue;
            };
            if texture.is_empty() {
                continue;
            }
            let [r, g, b, a] = texture.sample_nearest(uv.x, uv.y);
            let rgb = [r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0];
            match kind {
                MapKind::Diffuse => {
                    out.diffuse = [rgb[0], rgb[1], rgb[2], a as f32 / 255.0];
                }
                MapKind::Normal => {
                    let decoded = Vec3::new(
                        rgb[0] * 2.0 - 1.0,
                        rgb[1] * 2.0 - 1.0,
                        rgb[2] * 2.0 - 1.0,
                    );
                    out.normal = if decoded.length_squared() > 1e-12 {
                        decoded.normalize()
                    } else {
                        Vec3::Z
                    };
                }
                MapKind::Specular => out.specular = (rgb[0] + rgb[1] + rgb[2]) / 3.0,
                MapKind::Emissive => out.emissive = Vec3::from_array(rgb),
            }
        }
        out
    }
}

/// The surface's tangent frame at one triangle — `(tangent, bitangent)` — from
/// its world positions and UV coordinates.
///
/// `None` for a triangle with no area in UV space (all three UVs equal, a
/// collinear set) or no area at all: a frame is not defined there, and a
/// normal map has nothing to be relative to.
pub fn tangent_frame(a: (Vec3, Vec2), b: (Vec3, Vec2), c: (Vec3, Vec2)) -> Option<(Vec3, Vec3)> {
    let edge1 = b.0 - a.0;
    let edge2 = c.0 - a.0;
    let duv1 = b.1 - a.1;
    let duv2 = c.1 - a.1;
    let det = duv1.x * duv2.y - duv2.x * duv1.y;
    if !det.is_finite() || det.abs() < 1e-12 {
        return None;
    }
    let r = 1.0 / det;
    let tangent = ((edge1 * duv2.y - edge2 * duv1.y) * r).normalize_or_zero();
    let bitangent = ((edge2 * duv1.x - edge1 * duv2.x) * r).normalize_or_zero();
    if tangent == Vec3::ZERO || bitangent == Vec3::ZERO {
        return None;
    }
    Some((tangent, bitangent))
}

/// Perturb `normal` by a tangent-space map normal.
///
/// The map's axes are put into the surface's frame: the tangent is
/// re-orthogonalized against the normal and the bitangent is rebuilt from the
/// cross product, so a slightly skew frame still yields an orthonormal one.
/// A `None` frame, a zero map normal and a normal that is not a direction all
/// leave `normal` as it was rather than producing a NaN.
pub fn apply_normal_map(normal: Vec3, frame: Option<(Vec3, Vec3)>, map_normal: Vec3) -> Vec3 {
    let n = normal.normalize_or_zero();
    if n == Vec3::ZERO {
        return normal;
    }
    let Some((tangent, _bitangent)) = frame else {
        return n;
    };
    let tangent = (tangent - n * n.dot(tangent)).normalize_or_zero();
    if tangent == Vec3::ZERO {
        return n;
    }
    let bitangent = n.cross(tangent).normalize_or_zero();
    let perturbed = tangent * map_normal.x + bitangent * map_normal.y + n * map_normal.z;
    if perturbed.length_squared() > 1e-12 {
        perturbed.normalize()
    } else {
        n
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Material, MaterialLibrary, Texture};

    /// A 2x2 texture with one distinct texel per quadrant.
    fn quadrant_texture() -> Texture {
        let mut pixels = Vec::new();
        for texel in [
            [255, 0, 0, 255],
            [0, 255, 0, 255],
            [0, 0, 255, 255],
            [128, 128, 128, 64],
        ] {
            pixels.extend_from_slice(&texel);
        }
        Texture {
            name: "quadrants".into(),
            width: 2,
            height: 2,
            pixels,
        }
    }

    #[test]
    fn the_identity_changes_nothing() {
        let id = MapTexels::IDENTITY;
        assert_eq!(id.diffuse, [1.0, 1.0, 1.0, 1.0]);
        assert_eq!(id.normal, Vec3::Z);
        assert_eq!(id.specular, 1.0);
        assert_eq!(id.emissive, Vec3::ZERO);
    }

    #[test]
    fn a_material_without_maps_samples_the_identity() {
        let mut library = MaterialLibrary::new();
        library.add_texture(quadrant_texture());
        let material = Material::new("Plain");
        assert_eq!(
            MapTexels::sample(&library, &material, Vec2::new(0.25, 0.25)),
            MapTexels::IDENTITY
        );
    }

    #[test]
    fn every_map_is_sampled_at_the_uv() {
        let mut library = MaterialLibrary::new();
        let index = library.add_texture(quadrant_texture());
        // One material wearing the same texture as all four maps: each kind
        // reads its own channel convention from the same texel.
        let mut material = Material::new("Mapped");
        for kind in MapKind::ALL {
            material.set_map(kind, Some(index));
        }
        // The first quadrant: red, byte alpha 255.
        let maps = MapTexels::sample(&library, &material, Vec2::new(0.25, 0.25));
        assert_eq!(maps.diffuse, [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(
            maps.normal,
            Vec3::new(1.0, -1.0, -1.0).normalize(),
            "0.5 is straight out, so red 255 and the zeros tilt"
        );
        assert!((maps.specular - 1.0 / 3.0).abs() < 1e-6, "luminance of 255,0,0");
        assert_eq!(maps.emissive, Vec3::new(1.0, 0.0, 0.0));

        // The fourth quadrant: mid grey with byte alpha 64.
        let maps = MapTexels::sample(&library, &material, Vec2::new(0.75, 0.75));
        assert!((maps.diffuse[0] - 128.0 / 255.0).abs() < 1e-6);
        assert!((maps.diffuse[3] - 64.0 / 255.0).abs() < 1e-6);
        assert_eq!(
            maps.normal,
            Vec3::splat(128.0 / 255.0 * 2.0 - 1.0).normalize(),
            "a mid-grey texel decodes to a diagonal, as the bytes say"
        );
        assert!((maps.specular - 128.0 / 255.0).abs() < 1e-6);
    }

    #[test]
    fn a_flat_normal_map_texel_is_straight_out() {
        let mut library = MaterialLibrary::new();
        // A texture that is exactly 128,128,255 everywhere: blue is +Z.
        let mut pixels = Vec::new();
        for _ in 0..4 {
            pixels.extend_from_slice(&[128, 128, 255, 255]);
        }
        let index = library.add_texture(Texture {
            name: "flat".into(),
            width: 2,
            height: 2,
            pixels,
        });
        let mut material = Material::new("Flat");
        material.normal_map = Some(index);
        let maps = MapTexels::sample(&library, &material, Vec2::new(0.1, 0.9));
        // Byte 128 is half a step above the midpoint of 0..255, so the decoded
        // normal leans by about 0.004 — a real texel would too.
        assert!((maps.normal - Vec3::Z).length() < 1e-2, "{:?}", maps.normal);
    }

    #[test]
    fn a_dangling_or_empty_map_is_the_identity() {
        let mut library = MaterialLibrary::new();
        library.add_texture(Texture {
            name: "empty".into(),
            width: 0,
            height: 0,
            pixels: Vec::new(),
        });
        let mut material = Material::new("Broken");
        material.diffuse_map = Some(9);
        material.normal_map = Some(0);
        material.specular_map = Some(9);
        material.emissive_map = Some(0);
        assert_eq!(
            MapTexels::sample(&library, &material, Vec2::new(0.5, 0.5)),
            MapTexels::IDENTITY
        );
    }

    #[test]
    fn the_tangent_frame_follows_the_uvs() {
        // A triangle in the XY plane whose UVs are its positions: the tangent
        // runs along +X and the bitangent along +Y.
        let frame = tangent_frame(
            (Vec3::ZERO, Vec2::ZERO),
            (Vec3::X, Vec2::X),
            (Vec3::Y, Vec2::Y),
        )
        .expect("a frame is defined");
        assert!((frame.0 - Vec3::X).length() < 1e-6, "{:?}", frame.0);
        assert!((frame.1 - Vec3::Y).length() < 1e-6, "{:?}", frame.1);

        // The same triangle with the UV axes swapped flips both.
        let frame = tangent_frame(
            (Vec3::ZERO, Vec2::ZERO),
            (Vec3::X, Vec2::Y),
            (Vec3::Y, Vec2::X),
        )
        .expect("a frame is defined");
        assert!(frame.0.dot(Vec3::X).abs() < 1e-6, "{:?}", frame.0);
        assert!(frame.1.dot(Vec3::Y).abs() < 1e-6, "{:?}", frame.1);
    }

    #[test]
    fn a_degenerate_triangle_or_uv_set_has_no_frame() {
        assert_eq!(
            tangent_frame(
                (Vec3::ZERO, Vec2::ZERO),
                (Vec3::X, Vec2::ZERO),
                (Vec3::Y, Vec2::ZERO)
            ),
            None,
            "every corner at the same UV"
        );
        assert_eq!(
            tangent_frame(
                (Vec3::ZERO, Vec2::ZERO),
                (Vec3::ZERO, Vec2::X),
                (Vec3::ZERO, Vec2::Y)
            ),
            None,
            "a triangle with no area"
        );
        assert_eq!(
            tangent_frame(
                (Vec3::ZERO, Vec2::ZERO),
                (Vec3::X, Vec2::new(0.5, 0.5)),
                (Vec3::Y, Vec2::new(1.0, 1.0))
            ),
            None,
            "collinear UVs leave no area to read a frame from"
        );
    }

    #[test]
    fn a_normal_map_tilts_the_normal_in_its_frame() {
        let frame = Some((Vec3::X, Vec3::Y));
        // A 45 degree tilt towards +X.
        let tilted = apply_normal_map(Vec3::Z, frame, Vec3::new(0.7071, 0.0, 0.7071));
        assert!((tilted - Vec3::new(0.7071, 0.0, 0.7071)).length() < 1e-3, "{tilted:?}");
        assert!((tilted.length() - 1.0).abs() < 1e-5, "still a direction");

        // Straight out leaves the normal alone; so does no frame at all.
        assert!((apply_normal_map(Vec3::Z, frame, Vec3::Z) - Vec3::Z).length() < 1e-6);
        assert!((apply_normal_map(Vec3::Z, None, Vec3::new(1.0, 0.0, 0.0)) - Vec3::Z).length() < 1e-6);

        // A zero map normal and a zero surface normal are both no-ops.
        assert!((apply_normal_map(Vec3::Z, frame, Vec3::ZERO) - Vec3::Z).length() < 1e-6);
        assert_eq!(apply_normal_map(Vec3::ZERO, frame, Vec3::X), Vec3::ZERO);
    }

    #[test]
    fn a_skew_tangent_is_re_orthogonalized() {
        // A tangent leaning into the normal: the result is still a unit
        // direction and still tilts towards the tangent's own side, and a
        // frame parallel to the normal falls back to the normal itself.
        let skew = Some((Vec3::new(1.0, 0.0, 1.0).normalize(), Vec3::Y));
        let tilted = apply_normal_map(Vec3::Z, skew, Vec3::new(1.0, 0.0, 0.0));
        assert!((tilted.length() - 1.0).abs() < 1e-5);
        assert!(tilted.x > 0.0 && tilted.z.abs() < 1e-5, "{tilted:?}");

        let parallel = Some((Vec3::Z, Vec3::Y));
        assert!((apply_normal_map(Vec3::Z, parallel, Vec3::X) - Vec3::Z).length() < 1e-6);
    }
}
