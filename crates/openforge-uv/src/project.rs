//! Projection of vertex positions into UV space.
//!
//! Each projection computes a raw UV per vertex (or per face corner for the
//! box projection) and then installs it through the shared commit helpers in
//! the crate root, which guarantee that every corner `uv` index stays valid.
//! The [`UvOps`](crate::UvOps) methods that call into this module live in the
//! crate root's single impl block.

use glam::{Vec2, Vec3};
use openforge_core::mesh::Mesh;

use crate::{
    commit_corner_uvs, commit_vertex_uvs, face_normal_opt, normalize_in_place,
    projection_basis,
};

/// TAU = 2π, used by the angular projections.
pub(crate) const TAU: f32 = std::f32::consts::TAU;

/// Pick the dominant (largest absolute) component of `n`
/// as `0 = X`, `1 = Y`, `2 = Z`. Ties prefer X.
pub(crate) fn dominant_axis(n: Vec3) -> usize {
    let (x, y, z) = (n.x.abs(), n.y.abs(), n.z.abs());
    if x >= y && x >= z {
        0
    } else if y >= z {
        1
    } else {
        2
    }
}

/// Project `p` by dropping the dominant axis coordinate:
/// drop X → `(y, z)`, drop Y → `(x, z)`, drop Z → `(x, y)`.
pub(crate) fn drop_axis(p: Vec3, axis: usize) -> Vec2 {
    match axis {
        0 => Vec2::new(p.y, p.z),
        1 => Vec2::new(p.x, p.z),
        _ => Vec2::new(p.x, p.y),
    }
}

/// Center of the mesh bounds (`Vec3::ZERO` when there are no finite
/// positions). The angular projections spin around an axis through here.
fn mesh_center(mesh: &Mesh) -> Vec3 {
    let mut min = Vec3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY);
    let mut max = Vec3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
    let mut any = false;
    for p in &mesh.positions {
        if p.is_finite() {
            min = min.min(*p);
            max = max.max(*p);
            any = true;
        }
    }
    if any {
        (min + max) * 0.5
    } else {
        Vec3::ZERO
    }
}

/// Project every vertex onto the plane perpendicular to `axis` (one UV per
/// vertex), then stretch the range into `0..1` per axis.
pub(crate) fn uv_project_planar(mesh: &mut Mesh, axis: Vec3) {
    let (_, u, v) = projection_basis(axis);
    let raw: Vec<Vec2> = mesh
        .positions
        .iter()
        .map(|p| Vec2::new(p.dot(u), p.dot(v)))
        .collect();
    commit_vertex_uvs(mesh, raw);
    normalize_in_place(&mut mesh.uvs);
}

/// Wrap the mesh around `axis`: `U = angle / 2π` shifted into `0..1`,
/// `V` = height along the axis normalized into `0..1`.
pub(crate) fn uv_project_cylindrical(mesh: &mut Mesh, axis: Vec3) {
    let (n, u, v) = projection_basis(axis);
    let center = mesh_center(mesh);

    // Raw pass: U from the in-plane angle (already 0..1), V as raw height.
    let mut raw: Vec<Vec2> = Vec::with_capacity(mesh.positions.len());
    for p in &mesh.positions {
        let local = *p - center;
        let angle = local.dot(v).atan2(local.dot(u)); // -π..π
        let uu = angle / TAU + 0.5; // 0..1
        raw.push(Vec2::new(uu, local.dot(n)));
    }

    // Normalize the height column into 0..1 (a degenerate span maps to 0.5).
    let (mut hmin, mut hmax, mut any_h) = (f32::INFINITY, f32::NEG_INFINITY, false);
    for uv in &raw {
        if uv.y.is_finite() {
            hmin = hmin.min(uv.y);
            hmax = hmax.max(uv.y);
            any_h = true;
        }
    }
    let span = hmax - hmin;
    if any_h {
        for uv in raw.iter_mut() {
            uv.y = if uv.y.is_finite() && span > 0.0 && span.is_finite() {
                (uv.y - hmin) / span
            } else {
                0.5
            };
        }
    }

    commit_vertex_uvs(mesh, raw);
}

/// Map the mesh onto a sphere around the axis through the bounds center:
/// `U` = longitude, `V` = latitude, both inherently in `0..1`.
pub(crate) fn uv_project_spherical(mesh: &mut Mesh, axis: Vec3) {
    let (n, u, v) = projection_basis(axis);
    let center = mesh_center(mesh);
    let raw: Vec<Vec2> = mesh
        .positions
        .iter()
        .map(|p| {
            let local = *p - center;
            let r = local.length();
            let angle = local.dot(v).atan2(local.dot(u)); // -π..π
            let uu = angle / TAU + 0.5; // 0..1
            // Latitude from the axis: -π/2 (south pole) .. +π/2 (north pole).
            let lat = if r > 0.0 && r.is_finite() {
                (local.dot(n) / r).clamp(-1.0, 1.0).asin()
            } else {
                0.0
            };
            let vv = lat / std::f32::consts::PI + 0.5; // 0..1
            Vec2::new(uu, vv)
        })
        .collect();
    commit_vertex_uvs(mesh, raw);
}

/// Box (triplanar) projection: every face picks its normal's dominant axis
/// and drops that coordinate, so each face gets its own UVs. The X-, Y-, and
/// Z-facing groups are each normalized into `0..1` together, keeping texel
/// density consistent across the two faces of an axis pair.
pub(crate) fn uv_project_box(mesh: &mut Mesh) {
    // Pass 1: dominant-axis group per face (degenerate faces fall back to X).
    let mut group_of_face: Vec<usize> = Vec::with_capacity(mesh.faces.len());
    for fi in 0..mesh.faces.len() {
        let axis = face_normal_opt(mesh, fi).map(dominant_axis).unwrap_or(0);
        group_of_face.push(axis);
    }

    // Pass 2: one UV per corner, dropping the group's dominant axis.
    let mut raw: Vec<Vec2> = Vec::with_capacity(
        mesh.faces.iter().map(|f| f.corners.len()).sum(),
    );
    for (fi, face) in mesh.faces.iter().enumerate() {
        let g = group_of_face[fi];
        for c in &face.corners {
            let pos = mesh
                .positions
                .get(c.position as usize)
                .copied()
                .unwrap_or(Vec3::ZERO);
            raw.push(drop_axis(pos, g));
        }
    }
    commit_corner_uvs(mesh, raw);

    // Pass 3: normalize each dominant-axis group into 0..1 independently.
    // `commit_corner_uvs` numbers corners in face order, so uv index == the
    // corner's global offset.
    let mut group_uv: [Vec<usize>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    let mut offset = 0usize;
    for (fi, face) in mesh.faces.iter().enumerate() {
        let g = group_of_face[fi];
        for _ in 0..face.corners.len() {
            group_uv[g].push(offset);
            offset += 1;
        }
    }
    for idxs in group_uv {
        if idxs.is_empty() {
            continue;
        }
        let mut gmin = Vec2::new(f32::INFINITY, f32::INFINITY);
        let mut gmax = Vec2::new(f32::NEG_INFINITY, f32::NEG_INFINITY);
        for &i in &idxs {
            let uv = mesh.uvs[i];
            gmin = gmin.min(uv);
            gmax = gmax.max(uv);
        }
        let span = gmax - gmin;
        for &i in &idxs {
            let uv = mesh.uvs[i];
            mesh.uvs[i].x = if span.x > 0.0 && span.x.is_finite() && uv.x.is_finite() {
                (uv.x - gmin.x) / span.x
            } else {
                0.5
            };
            mesh.uvs[i].y = if span.y > 0.0 && span.y.is_finite() && uv.y.is_finite() {
                (uv.y - gmin.y) / span.y
            } else {
                0.5
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::{assert_uv_indices_valid, cube, open_cylinder, unit_quad};
    use crate::UvOps;

    /// Assert every finite uv entry lies within the unit square.
    fn assert_all_in_unit(m: &Mesh) {
        for uv in &m.uvs {
            assert!(
                (0.0..=1.0).contains(&uv.x) && (0.0..=1.0).contains(&uv.y),
                "uv out of range: {uv:?}"
            );
        }
    }

    #[test]
    fn planar_on_unit_quad_spans_the_unit_square() {
        let mut m = unit_quad();
        m.uv_project_planar(Vec3::Z);
        assert_uv_indices_valid(&m);
        assert_eq!(m.uvs.len(), 4); // one UV per vertex
        let b = m.uv_bounds().unwrap();
        assert_eq!(b.min, Vec2::ZERO);
        assert_eq!(b.max, Vec2::ONE);
        // Each face corner resolves to its vertex's UV: area of a unit square.
        assert!((m.uv_face_area_uv(0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn planar_zero_axis_falls_back_without_panicking() {
        let mut m = unit_quad();
        m.uv_project_planar(Vec3::ZERO);
        assert_uv_indices_valid(&m);
        assert_all_in_unit(&m);
    }

    #[test]
    fn cylindrical_on_open_cylinder_rims_get_extreme_v() {
        let mut m = open_cylinder(8);
        m.uv_project_cylindrical(Vec3::Y);
        assert_uv_indices_valid(&m);
        assert_eq!(m.uvs.len(), m.vertex_count()); // one UV per vertex
        assert_all_in_unit(&m);
        // open_cylinder interleaves bottom (y=0) and top (y=1) vertices.
        for (i, uv) in m.uvs.iter().enumerate() {
            let expected = if i % 2 == 0 { 0.0 } else { 1.0 };
            assert!((uv.y - expected).abs() < 1e-6, "v={}", uv.y);
        }
        // The wrap covers distinct angles around the axis.
        let umin = m.uvs.iter().map(|u| u.x).fold(f32::INFINITY, f32::min);
        let umax = m.uvs.iter().map(|u| u.x).fold(f32::NEG_INFINITY, f32::max);
        assert!(umax - umin > 0.5, "u should span most of the wrap");
    }

    #[test]
    fn spherical_poles_land_on_the_v_extremes() {
        let mut m = Mesh::new();
        m.add_vertex(Vec3::new(0.0, 0.0, 0.0));
        m.add_vertex(Vec3::new(0.0, 2.0, 0.0));
        m.uv_project_spherical(Vec3::Y);
        assert_uv_indices_valid(&m);
        assert_all_in_unit(&m);
        // Bounds center is (0,1,0): the second vertex is the north pole.
        assert!((m.uvs[1].y - 1.0).abs() < 1e-6, "v={}", m.uvs[1].y);
        assert!((m.uvs[0].y - 0.0).abs() < 1e-6, "v={}", m.uvs[0].y);
    }

    #[test]
    fn spherical_on_cube_stays_in_range() {
        let mut m = cube();
        m.uv_project_spherical(Vec3::Y);
        assert_uv_indices_valid(&m);
        assert_all_in_unit(&m);
    }

    #[test]
    fn box_on_cube_gives_every_face_unit_area() {
        let mut m = cube();
        m.uv_project_box();
        assert_uv_indices_valid(&m);
        assert_eq!(m.uvs.len(), 24); // 6 faces x 4 corners, per-corner UVs
        for fi in 0..m.face_count() {
            let area = m.uv_face_area_uv(fi);
            assert!((area - 1.0).abs() < 1e-6, "face {fi} area was {area}");
        }
    }

    #[test]
    fn box_opposite_faces_of_a_pair_overlap() {
        let mut m = cube();
        m.uv_project_box();
        // -Z face (0) and +Z face (1) drop the same coordinate and share a
        // normalization group, so after projection they coincide.
        let corners_of = |face: usize| -> Vec<Vec2> {
            let mut v: Vec<Vec2> = m.faces[face]
                .corners
                .iter()
                .map(|c| m.uvs[c.uv as usize])
                .collect();
            // Two-key sort: sorting by x alone leaves equal-x ties unordered.
            v.sort_by(|a, b| {
                a.x.partial_cmp(&b.x)
                    .unwrap()
                    .then(a.y.partial_cmp(&b.y).unwrap())
            });
            v
        };
        let (f0, f1) = (corners_of(0), corners_of(1));
        for (a, b) in f0.iter().zip(&f1) {
            assert!((*a - *b).length() < 1e-6, "opposite faces must overlap");
        }
    }

    #[test]
    fn box_covers_side_faces_that_planar_projection_collapses() {
        // The point of triplanar: planar-Z flattens the ±X/±Y faces onto
        // lines (zero UV area), while box projection gives every face a full
        // unit square.
        let mut planar = cube();
        planar.uv_project_planar(Vec3::Z);
        assert_eq!(planar.uv_face_area_uv(4), 0.0); // -X face degenerate
        let mut boxed = cube();
        boxed.uv_project_box();
        assert!((boxed.uv_face_area_uv(4) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn box_handles_degenerate_and_empty_meshes() {
        // Empty mesh: no-op, no panic.
        let mut empty = Mesh::new();
        empty.uv_project_box();
        assert!(empty.uvs.is_empty());
        // A zero-area (collinear) face has no normal: falls back to a group
        // and still produces valid corner indices without panicking.
        let mut degenerate = Mesh::new();
        degenerate.add_vertex(Vec3::new(0.0, 0.0, 0.0));
        degenerate.add_vertex(Vec3::new(1.0, 0.0, 0.0));
        degenerate.add_vertex(Vec3::new(2.0, 0.0, 0.0));
        degenerate.add_face(&[0, 1, 2]);
        degenerate.uv_project_box();
        assert_uv_indices_valid(&degenerate);
    }
}
