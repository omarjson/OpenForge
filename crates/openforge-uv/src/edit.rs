//! Affine UV edits, bounds queries, and per-face UV area measurement.
//!
//! The [`UvOps`](crate::UvOps) methods that call into this module live in the
//! crate root's single impl block.

use glam::Vec2;
use openforge_core::mesh::Mesh;

use crate::{normalize_in_place, sanitize_uvs, uv_bounds_of, UvBounds};

/// Rescale all UVs so their bounding box maps into `0..1` per axis.
pub(crate) fn uv_normalize(mesh: &mut Mesh) {
    normalize_in_place(&mut mesh.uvs);
}

/// Flip the V coordinate of every UV (`v = 1 - v`).
pub(crate) fn uv_flip_v(mesh: &mut Mesh) {
    sanitize_uvs(&mut mesh.uvs);
    for uv in mesh.uvs.iter_mut() {
        uv.y = 1.0 - uv.y;
    }
}

/// Apply the affine transform `uv' = uv * scale + offset` to every UV.
pub(crate) fn uv_scale_translate(mesh: &mut Mesh, scale: Vec2, offset: Vec2) {
    sanitize_uvs(&mut mesh.uvs);
    for uv in mesh.uvs.iter_mut() {
        *uv = *uv * scale + offset;
    }
}

/// 2D area of a face in UV space (fan triangulation, absolute values).
pub(crate) fn uv_face_area_uv(mesh: &Mesh, face: usize) -> f32 {
    let Some(f) = mesh.faces.get(face) else {
        return 0.0;
    };
    let n = f.corners.len();
    if n < 3 {
        return 0.0;
    }
    let corner_uv = |i: usize| -> Option<Vec2> {
        mesh.uvs
            .get(f.corners[i].uv as usize)
            .copied()
            .filter(|v| v.is_finite())
    };
    // Fan triangulation around corner 0; sum absolute triangle areas so
    // reversed windings still report a positive area.
    let mut area = 0.0f32;
    for i in 1..n - 1 {
        let (Some(a), Some(b), Some(c)) = (corner_uv(0), corner_uv(i), corner_uv(i + 1))
        else {
            continue;
        };
        // 2D cross product (glam's Vec2 has no `cross`); its magnitude is
        // twice the triangle area.
        let ab = b - a;
        let ac = c - a;
        area += 0.5 * (ab.x * ac.y - ab.y * ac.x).abs();
    }
    if area.is_finite() {
        area
    } else {
        0.0
    }
}

/// Bounding box of all UV entries, or `None` when there are no finite ones.
pub(crate) fn uv_bounds(mesh: &Mesh) -> Option<UvBounds> {
    uv_bounds_of(&mesh.uvs)
}

/// `(uv index, value)` for every corner of `face`, in corner order.
///
/// Empty for an out-of-range face. Values are read defensively (an
/// out-of-range or non-finite entry reads as `Vec2::ZERO`) so the result is
/// always safe to display.
pub(crate) fn uv_face_uvs(mesh: &Mesh, face: usize) -> Vec<(u32, Vec2)> {
    let Some(f) = mesh.faces.get(face) else {
        return Vec::new();
    };
    f.corners
        .iter()
        .map(|c| (c.uv, crate::uv_at(&mesh.uvs, c.uv)))
        .collect()
}

/// Bounding box over a subset of UV entries (by index).
///
/// Out-of-range and non-finite entries are skipped; `None` when nothing usable
/// remains.
pub(crate) fn uv_bounds_selected(mesh: &Mesh, indices: &[u32]) -> Option<UvBounds> {
    let mut min = Vec2::ZERO;
    let mut max = Vec2::ZERO;
    let mut first = true;
    for &i in indices {
        let Some(v) = mesh.uvs.get(i as usize) else {
            continue;
        };
        if !v.is_finite() {
            continue;
        }
        if first {
            min = *v;
            max = *v;
            first = false;
        } else {
            min = min.min(*v);
            max = max.max(*v);
        }
    }
    if first {
        None
    } else {
        Some(UvBounds { min, max })
    }
}

/// Apply `uv' = R(angle) * ((uv - pivot) * scale) + pivot + translate` to a
/// subset of UV entries, leaving every other entry untouched.
///
/// Scale is applied before the rotation, both about `pivot`, then the
/// translation is added. Out-of-range indices are skipped and non-finite input
/// (or a non-finite angle) makes the whole call a no-op; non-finite *entries*
/// in the mesh are reset to `Vec2::ZERO` first, so the result is always finite
/// and every corner `uv` index stays valid.
pub(crate) fn uv_transform_selected(
    mesh: &mut Mesh,
    indices: &[u32],
    translate: Vec2,
    scale: Vec2,
    rotate: f32,
    pivot: Vec2,
) {
    if indices.is_empty()
        || !translate.is_finite()
        || !scale.is_finite()
        || !pivot.is_finite()
        || !rotate.is_finite()
    {
        return;
    }
    let (sin, cos) = rotate.sin_cos();
    for &i in indices {
        let Some(v) = mesh.uvs.get_mut(i as usize) else {
            continue;
        };
        if !v.is_finite() {
            *v = Vec2::ZERO;
        }
        let p = (*v - pivot) * scale;
        *v = Vec2::new(p.x * cos - p.y * sin, p.x * sin + p.y * cos) + pivot + translate;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::{assert_uv_indices_valid, unit_quad};
    use crate::UvOps;
    use glam::Vec3;
    use openforge_core::mesh::Face;

    #[test]
    fn bounds_none_on_empty_mesh() {
        let m = Mesh::new();
        assert_eq!(m.uv_bounds(), None);
    }

    #[test]
    fn normalize_maps_min_max_to_unit_range() {
        let mut m = Mesh::new();
        m.uvs = vec![Vec2::new(2.0, 4.0), Vec2::new(6.0, 8.0)];
        m.uv_normalize();
        assert_eq!(m.uvs[0], Vec2::ZERO);
        assert_eq!(m.uvs[1], Vec2::ONE);
        let b = m.uv_bounds().unwrap();
        assert!(b.min.abs().max_element() < 1e-6);
        assert!((b.max - Vec2::ONE).abs().max_element() < 1e-6);
    }

    #[test]
    fn normalize_maps_full_box_to_01() {
        // Values below and above the final range must be stretched into it.
        let mut m = Mesh::new();
        m.uvs = vec![Vec2::new(-3.0, 0.5), Vec2::new(1.0, 0.5), Vec2::new(0.0, 9.0)];
        m.uv_normalize();
        assert!(m
            .uvs
            .iter()
            .all(|v| (0.0..=1.0).contains(&v.x) && (0.0..=1.0).contains(&v.y)));
        assert!(m.uvs.iter().any(|v| v.x == 0.0));
        assert!(m.uvs.iter().any(|v| v.x == 1.0));
        assert!(m.uvs.iter().any(|v| v.y == 1.0));
    }

    #[test]
    fn normalize_handles_degenerate_and_empty() {
        let mut m = Mesh::new();
        m.uv_normalize(); // empty: no-op
        assert_eq!(m.uvs.len(), 0);
        // A flat (zero-extent) axis maps to 0.5.
        m.uvs = vec![Vec2::new(4.0, 7.0), Vec2::new(4.0, 3.0)];
        m.uv_normalize();
        for v in &m.uvs {
            assert_eq!(v.x, 0.5);
            assert!((0.0..=1.0).contains(&v.y));
        }
    }

    #[test]
    fn flip_v_inverts_the_v_axis() {
        let mut m = Mesh::new();
        m.uvs = vec![Vec2::new(0.25, 0.25), Vec2::new(0.75, 0.75)];
        m.uv_flip_v();
        assert_eq!(m.uvs[0], Vec2::new(0.25, 0.75));
        assert_eq!(m.uvs[1], Vec2::new(0.75, 0.25));
        // Flipping twice restores the original values.
        m.uv_flip_v();
        assert_eq!(m.uvs[0], Vec2::new(0.25, 0.25));
        // Empty mesh is a no-op.
        let mut e = Mesh::new();
        e.uv_flip_v();
        assert!(e.uvs.is_empty());
    }

    #[test]
    fn scale_translate_applies_affine_transform() {
        let mut m = Mesh::new();
        m.uvs = vec![Vec2::new(0.5, 0.5), Vec2::new(0.0, 1.0)];
        m.uv_scale_translate(Vec2::new(2.0, 4.0), Vec2::new(1.0, -1.0));
        assert_eq!(m.uvs[0], Vec2::new(2.0, 1.0));
        assert_eq!(m.uvs[1], Vec2::new(1.0, 3.0));
        // Identity transform changes nothing.
        m.uv_scale_translate(Vec2::ONE, Vec2::ZERO);
        assert_eq!(m.uvs[0], Vec2::new(2.0, 1.0));
    }

    #[test]
    fn face_area_of_unit_square_uv_quad_is_one() {
        let mut m = unit_quad();
        m.uvs = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(0.0, 1.0),
        ];
        for (i, c) in m.faces[0].corners.iter_mut().enumerate() {
            c.uv = i as u32;
        }
        let area = m.uv_face_area_uv(0);
        assert!((area - 1.0).abs() < 1e-6, "area was {area}");
        assert_uv_indices_valid(&m);
    }

    #[test]
    fn face_area_handles_out_of_range_and_invalid_uvs() {
        let mut m = unit_quad();
        m.uvs = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(0.0, 1.0),
        ];
        for (i, c) in m.faces[0].corners.iter_mut().enumerate() {
            c.uv = i as u32;
        }
        // Out-of-range face index and faces with too few corners are guarded.
        assert_eq!(m.uv_face_area_uv(1), 0.0);
        assert_eq!(m.uv_face_area_uv(usize::MAX), 0.0);
        let mut no_corners = Mesh::new();
        no_corners.faces.push(Face {
            corners: Vec::new(),
            material: 0,
        });
        assert_eq!(no_corners.uv_face_area_uv(0), 0.0);
        // An unresolvable uv index contributes no area instead of panicking.
        m.faces[0].corners[2].uv = 999;
        assert_eq!(m.uv_face_area_uv(0), 0.0);
    }

    #[test]
    fn transform_selected_only_touches_the_given_indices() {
        let mut m = Mesh::new();
        m.uvs = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(0.5, 0.5),
            Vec2::new(1.0, 1.0),
        ];
        m.uv_transform_selected(&[1], Vec2::new(0.25, -0.25), Vec2::ONE, 0.0, Vec2::ZERO);
        assert_eq!(m.uvs[0], Vec2::new(0.0, 0.0));
        assert_eq!(m.uvs[1], Vec2::new(0.75, 0.25));
        assert_eq!(m.uvs[2], Vec2::new(1.0, 1.0));
    }

    #[test]
    fn transform_selected_scales_about_the_pivot() {
        let mut m = Mesh::new();
        // Pivot at (0.5, 0.5): scaling by 2 pushes the corners out to +-0.5.
        m.uvs = vec![
            Vec2::new(0.25, 0.5),
            Vec2::new(0.5, 0.5),
            Vec2::new(0.75, 0.5),
        ];
        m.uv_transform_selected(&[0, 1, 2], Vec2::ZERO, Vec2::splat(2.0), 0.0, Vec2::splat(0.5));
        assert!((m.uvs[0] - Vec2::new(0.0, 0.5)).length() < 1e-6);
        assert!((m.uvs[1] - Vec2::new(0.5, 0.5)).length() < 1e-6);
        assert!((m.uvs[2] - Vec2::new(1.0, 0.5)).length() < 1e-6);
    }

    #[test]
    fn transform_selected_rotates_about_the_pivot() {
        let mut m = Mesh::new();
        m.uvs = vec![Vec2::new(1.0, 0.0), Vec2::new(0.0, 1.0)];
        // Quarter turn about the origin: (1,0) -> (0,1) and (0,1) -> (-1,0).
        let quarter = std::f32::consts::FRAC_PI_2;
        m.uv_transform_selected(&[0, 1], Vec2::ZERO, Vec2::ONE, quarter, Vec2::ZERO);
        assert!((m.uvs[0] - Vec2::new(0.0, 1.0)).length() < 1e-6, "{:?}", m.uvs[0]);
        assert!((m.uvs[1] - Vec2::new(-1.0, 0.0)).length() < 1e-6, "{:?}", m.uvs[1]);
        // A full turn is the identity.
        m.uv_transform_selected(&[0, 1], Vec2::ZERO, Vec2::ONE, std::f32::consts::TAU, Vec2::ZERO);
        assert!((m.uvs[0] - Vec2::new(0.0, 1.0)).length() < 1e-5);
        assert!((m.uvs[1] - Vec2::new(-1.0, 0.0)).length() < 1e-5);
    }

    #[test]
    fn transform_selected_is_defensive() {
        let mut m = Mesh::new();
        m.uvs = vec![Vec2::new(0.25, 0.25), Vec2::new(f32::NAN, 1.0)];
        // Empty index list: no-op (and no panic).
        m.uv_transform_selected(&[], Vec2::splat(1.0), Vec2::ONE, 0.0, Vec2::ZERO);
        assert_eq!(m.uvs[0], Vec2::new(0.25, 0.25));
        // Out-of-range indices are skipped.
        m.uv_transform_selected(&[7, 1], Vec2::splat(0.5), Vec2::ONE, 0.0, Vec2::ZERO);
        assert_eq!(m.uvs[0], Vec2::new(0.25, 0.25));
        // A non-finite entry is reset instead of poisoning the result.
        assert!(m.uvs[1].is_finite());
        // Non-finite parameters make the whole call a no-op.
        let before = m.uvs.clone();
        m.uv_transform_selected(&[0], Vec2::new(f32::NAN, 0.0), Vec2::ONE, 0.0, Vec2::ZERO);
        m.uv_transform_selected(&[0], Vec2::ZERO, Vec2::ONE, f32::INFINITY, Vec2::ZERO);
        m.uv_transform_selected(&[0], Vec2::ZERO, Vec2::ONE, 0.0, Vec2::splat(f32::NAN));
        assert_eq!(m.uvs, before);
    }

    #[test]
    fn face_uvs_lists_indices_and_values_in_corner_order() {
        let mut m = unit_quad();
        m.uvs = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(0.0, 1.0),
        ];
        for (i, c) in m.faces[0].corners.iter_mut().enumerate() {
            c.uv = i as u32;
        }
        let f = m.uv_face_uvs(0);
        assert_eq!(f.len(), 4);
        assert_eq!(f[0], (0, Vec2::new(0.0, 0.0)));
        assert_eq!(f[3], (3, Vec2::new(0.0, 1.0)));
        // Out-of-range face and a dangling uv index are handled.
        assert!(m.uv_face_uvs(9).is_empty());
        m.faces[0].corners[1].uv = 42;
        assert_eq!(m.uv_face_uvs(0)[1], (42, Vec2::ZERO));
    }

    #[test]
    fn bounds_selected_covers_only_the_subset() {
        let mut m = Mesh::new();
        m.uvs = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(0.5, 0.25),
            Vec2::new(4.0, 4.0),
            Vec2::new(f32::NAN, 1.0),
        ];
        assert_eq!(m.uv_bounds_selected(&[]), None);
        let b = m.uv_bounds_selected(&[0, 1]).expect("two finite entries");
        assert_eq!(b.min, Vec2::new(0.0, 0.0));
        assert_eq!(b.max, Vec2::new(0.5, 0.25));
        // The far corner is only included when it is part of the subset.
        assert_eq!(m.uv_bounds_selected(&[2]).unwrap().max, Vec2::new(4.0, 4.0));
        // Non-finite and out-of-range indices are skipped, never panic.
        assert_eq!(m.uv_bounds_selected(&[3, 99]), None);
        assert!(m.uv_bounds_selected(&[3, 99, 0]).is_some());
    }

    #[test]
    fn face_area_of_collinear_triangle_is_zero() {
        let mut m = Mesh::new();
        m.uvs = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(2.0, 0.0),
        ];
        m.add_vertex(Vec3::new(0.0, 0.0, 0.0));
        m.add_vertex(Vec3::new(1.0, 1.0, 0.0));
        m.add_vertex(Vec3::new(2.0, 2.0, 0.0));
        m.add_face(&[0, 1, 2]);
        m.faces[0].corners[0].uv = 0;
        m.faces[0].corners[1].uv = 1;
        m.faces[0].corners[2].uv = 2;
        assert_eq!(m.uv_face_area_uv(0), 0.0);
    }
}
