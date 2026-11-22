//! Mesh boolean (CSG) operations for OpenForge: union, difference, and
//! intersection of two triangle meshes, implemented with binary space
//! partitioning (BSP).
//!
//! The engine lives in [`bsp`]; this module exposes the public, mesh-level API
//! plus two useful mesh predicates — [`mesh_volume`] and [`is_watertight`].
//!
//! BSP CSG is exact for convex and many non-convex solids but can produce
//! sliver faces where input faces are exactly coplanar; inputs are
//! fan-triangulated first so every splitting polygon is a triangle.
//!
//! The work is done in **normalized `f64` space** (see [`bsp::Fit`]): both
//! solids are centred and scaled to about a unit across before anything is
//! split, and the result is mapped back afterwards. Every tolerance in the
//! engine is therefore relative to the model, so the same scene at millimetre
//! and at kilometre scale takes the same path with the same accuracy, and the
//! only rounding to single precision happens once, on the way out.

mod bsp;

use bsp::{mesh_to_polygons, polygons_to_mesh, Fit};
use glam::Vec3;
use openforge_core::mesh::Mesh;
use thiserror::Error;

/// Errors produced by a boolean operation.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum BooleanError {
    /// One of the inputs had no faces to operate on.
    #[error("mesh has no faces")]
    EmptyInput,
    /// The requested splitting plane had no usable direction.
    #[error("plane has a zero direction")]
    DegeneratePlane,
    /// The face index named by a shave was not a face of the mesh.
    #[error("face {0} does not exist")]
    InvalidFace(usize),
    /// A shave needs a positive depth.
    #[error("shave depth must be positive")]
    InvalidDepth,
}

/// Union of two meshes: everything inside `a` or `b` (or both).
pub fn union(a: &Mesh, b: &Mesh) -> Result<Mesh, BooleanError> {
    boolean(a, b, bsp::union)
}

/// Difference `a - b`: everything inside `a` that is not inside `b`.
pub fn subtract(a: &Mesh, b: &Mesh) -> Result<Mesh, BooleanError> {
    boolean(a, b, bsp::subtract)
}

/// Intersection of two meshes: everything inside both `a` and `b`.
pub fn intersect(a: &Mesh, b: &Mesh) -> Result<Mesh, BooleanError> {
    boolean(a, b, bsp::intersect)
}

/// Split a solid by a plane, returning `(front, back)` — the piece on the
/// side the plane's normal points to first.
///
/// Both halves are closed solids: the cut face comes from the CSG, so there is
/// no open boundary at the plane. The cutter is an oriented box whose near
/// face is the plane and whose other five faces sit well outside the mesh, so
/// only the requested plane can cut it.
pub fn split_by_plane(
    mesh: &Mesh,
    origin: Vec3,
    normal: Vec3,
) -> Result<(Mesh, Mesh), BooleanError> {
    if mesh.faces.is_empty() {
        return Err(BooleanError::EmptyInput);
    }
    let n = normal.normalize_or_zero();
    if !n.is_finite() || n.length_squared() < 0.5 {
        return Err(BooleanError::DegeneratePlane);
    }
    let reach = mesh_extent(mesh) * 4.0;
    let cutter = half_space_prism(origin, n, reach, reach);
    let front = intersect(mesh, &cutter)?;
    let back = subtract(mesh, &cutter)?;
    Ok((front, back))
}

/// Shave a face inward by `depth`, returning the smaller solid.
///
/// The face's winding gives the outward direction. A half-space box starts
/// `depth` behind that face and covers everything outside it, so subtracting
/// it removes a uniform slab from that side and caps the new face.
pub fn shave_face(mesh: &Mesh, face: usize, depth: f32) -> Result<Mesh, BooleanError> {
    if mesh.faces.is_empty() {
        return Err(BooleanError::EmptyInput);
    }
    let Some(target) = mesh.faces.get(face) else {
        return Err(BooleanError::InvalidFace(face));
    };
    if !depth.is_finite() || depth <= 0.0 {
        return Err(BooleanError::InvalidDepth);
    }
    let pts: Vec<Vec3> = target
        .corners
        .iter()
        .filter_map(|c| mesh.positions.get(c.position as usize).copied())
        .collect();
    if pts.len() < 3 {
        return Err(BooleanError::InvalidFace(0));
    }
    let mut outward = Vec3::ZERO;
    for i in 1..pts.len() - 1 {
        outward += (pts[i] - pts[0]).cross(pts[i + 1] - pts[0]);
    }
    let outward = outward.normalize_or_zero();
    if !outward.is_finite() || outward.length_squared() < 0.5 {
        return Err(BooleanError::InvalidFace(0));
    }
    let reach = mesh_extent(mesh) * 4.0;
    let cutter = half_space_prism(pts[0] - outward * depth, outward, reach, reach);
    subtract(mesh, &cutter)
}

/// The largest distance between two positions of `mesh`, with a floor of one
/// so a tiny model still gets a cutter comfortably bigger than itself.
fn mesh_extent(mesh: &Mesh) -> f32 {
    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    for p in &mesh.positions {
        min = min.min(*p);
        max = max.max(*p);
    }
    (max - min).length().max(1.0)
}

/// An oriented box filling the half-space in front of `origin` (on the side
/// `normal` points to). `depth` is its extent along the normal and `half` its
/// extent along the two in-plane axes; callers make both much larger than the
/// solid so only the near face can cut it.
///
/// `(u, v, n)` is right-handed: `u = n × helper`, `v = n × u`, so `u × v = n`
/// and the six quads below wind outwards.
fn half_space_prism(origin: Vec3, normal: Vec3, depth: f32, half: f32) -> Mesh {
    let n = normal.normalize_or_zero();
    let helper = if n.y.abs() < 0.9 { Vec3::Y } else { Vec3::X };
    let u = n.cross(helper).normalize_or_zero();
    let v = n.cross(u).normalize_or_zero();
    let mut m = Mesh::new();
    for face_z in [0.0, depth] {
        for (a, b) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
            m.add_vertex(origin + u * (a * half) + v * (b * half) + n * face_z);
        }
    }
    for q in [
        [0u32, 3, 2, 1], // near face, outward -n
        [4, 5, 6, 7],    // far face, outward +n
        [0, 4, 7, 3],    // -u
        [1, 2, 6, 5],    // +u
        [0, 1, 5, 4],    // -v
        [3, 7, 6, 2],    // +v
    ] {
        m.add_face(&q);
    }
    m.compute_flat_normals();
    m
}

/// Run one CSG operation over two meshes.
///
/// The solids are normalized first — centred, and scaled to an extent of about
/// one — so the engine's absolute tolerances are relative to the model rather
/// than to the world, and the finished mesh is mapped back.
fn boolean(
    a: &Mesh,
    b: &Mesh,
    op: fn(&[bsp::Polygon], &[bsp::Polygon]) -> Vec<bsp::Polygon>,
) -> Result<Mesh, BooleanError> {
    if a.faces.is_empty() || b.faces.is_empty() {
        return Err(BooleanError::EmptyInput);
    }
    let pa = mesh_to_polygons(a);
    let pb = mesh_to_polygons(b);
    if pa.is_empty() || pb.is_empty() {
        return Err(BooleanError::EmptyInput);
    }
    let fit = Fit::of(&[&pa, &pb]);
    let na = fit.polygons(&pa);
    let nb = fit.polygons(&pb);
    Ok(fit.restore(polygons_to_mesh(&op(&na, &nb))))
}

/// Signed volume of a closed mesh via the divergence theorem, returned as an
/// absolute (always non-negative) value. Meaningful only for watertight
/// meshes; open meshes give an arbitrary result.
///
/// The sum is accumulated in double precision: a result with many faces would
/// otherwise lose the small terms that decide whether the total is right.
pub fn mesh_volume(mesh: &Mesh) -> f32 {
    let mut vol6 = 0.0f64;
    for f in &mesh.faces {
        let pts: Vec<glam::DVec3> = f
            .corners
            .iter()
            .filter_map(|c| mesh.positions.get(c.position as usize).copied())
            .map(|p| p.as_dvec3())
            .collect();
        for i in 1..pts.len().saturating_sub(1) {
            // Signed volume of tetrahedron (origin, p0, pi, pi+1).
            vol6 += pts[0].dot(pts[i].cross(pts[i + 1]));
        }
    }
    (vol6 / 6.0).abs() as f32
}

/// True when every undirected edge is shared by exactly two faces — i.e. the
/// mesh encloses a closed, manifold volume with no boundary or T-junctions.
pub fn is_watertight(mesh: &Mesh) -> bool {
    use std::collections::HashMap;
    let mut counts: HashMap<(u32, u32), i32> = HashMap::new();
    for f in &mesh.faces {
        let n = f.corners.len();
        if n < 3 {
            return false;
        }
        for i in 0..n {
            let a = f.corners[i].position;
            let b = f.corners[(i + 1) % n].position;
            if a == b {
                return false; // degenerate edge
            }
            let key = if a < b { (a, b) } else { (b, a) };
            *counts.entry(key).or_insert(0) += 1;
        }
    }
    counts.values().all(|&c| c == 2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;
    use openforge_core::mesh::Mesh;
    use std::f32::consts::PI;

    /// Axis-aligned box centered at `center` with full side `size`.
    fn box_at(center: Vec3, size: f32) -> Mesh {
        let mut m = Mesh::new();
        let h = size * 0.5;
        let corners = [
            Vec3::new(-h, -h, -h),
            Vec3::new(h, -h, -h),
            Vec3::new(h, h, -h),
            Vec3::new(-h, h, -h),
            Vec3::new(-h, -h, h),
            Vec3::new(h, -h, h),
            Vec3::new(h, h, h),
            Vec3::new(-h, h, h),
        ];
        for c in corners {
            m.add_vertex(c + center);
        }
        for q in [
            [0u32, 3, 2, 1],
            [4, 5, 6, 7],
            [0, 1, 5, 4],
            [3, 7, 6, 2],
            [0, 4, 7, 3],
            [1, 2, 6, 5],
        ] {
            m.add_face(&q);
        }
        m.compute_flat_normals();
        m
    }

    #[test]
    fn volume_and_watertight_of_a_box() {
        let b = box_at(Vec3::ZERO, 2.0);
        assert!(is_watertight(&b));
        let v = mesh_volume(&b);
        assert!((v - 8.0).abs() < 1e-3, "volume {v} != 8");
    }

    #[test]
    fn open_mesh_is_not_watertight() {
        // Single quad: boundary edges have only one face.
        let mut m = Mesh::new();
        for p in [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ] {
            m.add_vertex(p);
        }
        m.add_face(&[0, 1, 2, 3]);
        assert!(!is_watertight(&m));
    }

    #[test]
    fn empty_input_is_rejected() {
        let a = box_at(Vec3::ZERO, 2.0);
        let e = Mesh::new();
        assert_eq!(union(&a, &e), Err(BooleanError::EmptyInput));
        assert_eq!(subtract(&e, &a), Err(BooleanError::EmptyInput));
        assert_eq!(intersect(&a, &e), Err(BooleanError::EmptyInput));
    }

    #[test]
    fn union_of_two_boxes_is_larger_than_either() {
        let a = box_at(Vec3::ZERO, 2.0); // -1..1, vol 8
        let b = box_at(Vec3::new(1.0, 0.0, 0.0), 2.0); // 0..2, vol 8, overlap 4
        let u = union(&a, &b).unwrap();
        let vu = mesh_volume(&u);
        // Union = 8 + 8 - 4 = 12.
        assert!((vu - 12.0).abs() < 0.2, "union volume {vu} != 12");
        assert!(is_watertight(&u), "union should be watertight");
    }

    #[test]
    fn intersect_of_two_boxes_is_the_overlap() {
        let a = box_at(Vec3::ZERO, 2.0);
        let b = box_at(Vec3::new(1.0, 0.0, 0.0), 2.0);
        let i = intersect(&a, &b).unwrap();
        let vi = mesh_volume(&i);
        // Overlap spans x 0..1, y -1..1, z -1..1 → volume 4.
        assert!((vi - 4.0).abs() < 0.2, "intersect volume {vi} != 4");
        assert!(is_watertight(&i));
    }

    #[test]
    fn subtract_removes_the_overlap() {
        let a = box_at(Vec3::ZERO, 2.0);
        let b = box_at(Vec3::new(1.0, 0.0, 0.0), 2.0);
        let d = subtract(&a, &b).unwrap();
        let vd = mesh_volume(&d);
        // 8 - 4 = 4.
        assert!((vd - 4.0).abs() < 0.2, "subtract volume {vd} != 4");
        assert!(is_watertight(&d));
    }

    #[test]
    fn subtracting_self_is_empty() {
        let a = box_at(Vec3::ZERO, 2.0);
        let d = subtract(&a, &a).unwrap();
        assert!(d.faces.is_empty(), "a - a should be empty");
        assert!(mesh_volume(&d) < 1e-3);
    }

    #[test]
    fn disjoint_boxes_intersect_to_nothing() {
        let a = box_at(Vec3::new(-5.0, 0.0, 0.0), 2.0);
        let b = box_at(Vec3::new(5.0, 0.0, 0.0), 2.0);
        let i = intersect(&a, &b).unwrap();
        assert!(i.faces.is_empty(), "disjoint boxes should not intersect");
    }

    #[test]
    fn boolean_ops_are_deterministic() {
        let a = box_at(Vec3::ZERO, 2.0);
        let b = box_at(Vec3::new(1.0, 1.0, 0.0), 2.0);
        let u1 = union(&a, &b).unwrap();
        let u2 = union(&a, &b).unwrap();
        assert_eq!(u1.vertex_count(), u2.vertex_count());
        assert_eq!(u1.face_count(), u2.face_count());
        assert!((mesh_volume(&u1) - mesh_volume(&u2)).abs() < 1e-6);
    }

    // --- robustness -------------------------------------------------------------

    /// A box with a different size on each axis.
    fn box_sized(center: Vec3, size: Vec3) -> Mesh {
        let mut m = box_at(center, 2.0);
        for p in &mut m.positions {
            *p = center + (*p - center) * (size * 0.5);
        }
        m
    }

    /// A closed sphere: a UV sphere with one vertex at each pole, so the
    /// surface is genuinely closed and a boolean over it has to stay closed.
    fn ball(center: Vec3, radius: f32, segments: u32, rings: u32) -> Mesh {
        let mut m = Mesh::new();
        let north = m.add_vertex(center + Vec3::Y * radius);
        let mut rows: Vec<Vec<u32>> = Vec::new();
        for r in 1..rings {
            let phi = PI * r as f32 / rings as f32;
            let (sp, cp) = phi.sin_cos();
            let mut row = Vec::new();
            for s in 0..segments {
                let theta = 2.0 * PI * s as f32 / segments as f32;
                let p = Vec3::new(sp * theta.cos(), cp, sp * theta.sin()) * radius;
                row.push(m.add_vertex(center + p));
            }
            rows.push(row);
        }
        let south = m.add_vertex(center - Vec3::Y * radius);
        for s in 0..segments {
            let s2 = ((s + 1) % segments) as usize;
            m.add_face(&[north, rows[0][s2], rows[0][s as usize]]);
        }
        for r in 0..rows.len() - 1 {
            for s in 0..segments {
                let s2 = ((s + 1) % segments) as usize;
                m.add_face(&[rows[r][s as usize], rows[r][s2], rows[r + 1][s2]]);
                m.add_face(&[rows[r][s as usize], rows[r + 1][s2], rows[r + 1][s as usize]]);
            }
        }
        let last = rows.len() - 1;
        for s in 0..segments {
            let s2 = ((s + 1) % segments) as usize;
            m.add_face(&[south, rows[last][s as usize], rows[last][s2]]);
        }
        m
    }

    /// How the edges of a mesh are shared: boundary (one face), manifold
    /// (two) and over-shared (three or more).
    fn edge_sharing(mesh: &Mesh) -> (usize, usize, usize) {
        use std::collections::HashMap;
        let mut counts: HashMap<(u32, u32), i32> = HashMap::new();
        for f in &mesh.faces {
            let n = f.corners.len();
            for i in 0..n {
                let a = f.corners[i].position;
                let b = f.corners[(i + 1) % n].position;
                let key = if a < b { (a, b) } else { (b, a) };
                *counts.entry(key).or_insert(0) += 1;
            }
        }
        let mut boundary = 0;
        let mut manifold = 0;
        let mut over = 0;
        for c in counts.values() {
            match c {
                1 => boundary += 1,
                2 => manifold += 1,
                _ => over += 1,
            }
        }
        (boundary, manifold, over)
    }

    /// True when no two of the mesh's vertices sit at the same place: the
    /// result of a boolean is welded, so a shared point is one index.
    fn is_welded(mesh: &Mesh, tol: f32) -> bool {
        for (i, a) in mesh.positions.iter().enumerate() {
            for b in mesh.positions.iter().skip(i + 1) {
                if (*a - *b).length() < tol {
                    return false;
                }
            }
        }
        true
    }

    /// True when every face is a real triangle: three distinct corners, and
    /// some area to it.
    fn has_no_degenerate_faces(mesh: &Mesh) -> bool {
        for f in &mesh.faces {
            if f.corners.len() < 3 {
                return false;
            }
            let mut ids: Vec<u32> = f.corners.iter().map(|c| c.position).collect();
            ids.sort_unstable();
            ids.dedup();
            if ids.len() < 3 {
                return false;
            }
            let p: Vec<Vec3> = f
                .corners
                .iter()
                .map(|c| mesh.positions[c.position as usize])
                .collect();
            for i in 1..p.len() - 1 {
                if (p[i] - p[0]).cross(p[i + 1] - p[0]).length_squared() <= 0.0 {
                    return false;
                }
            }
        }
        true
    }

    #[test]
    fn two_solids_that_share_a_face_union_into_one_solid() {
        // The two boxes touch exactly along the plane x = 1, which is the case
        // a naive split turns into slivers.
        let a = box_at(Vec3::ZERO, 2.0); // -1..1
        let b = box_at(Vec3::new(2.0, 0.0, 0.0), 2.0); // 1..3
        let u = union(&a, &b).unwrap();
        // 8 + 8, with nothing in common.
        assert!((mesh_volume(&u) - 16.0).abs() < 0.05, "volume {}", mesh_volume(&u));
        assert!(is_watertight(&u), "the union of two touching boxes is one solid");
        let (boundary, _, over) = edge_sharing(&u);
        assert_eq!(boundary, 0, "no open edges");
        assert_eq!(over, 0, "no edge shared by three faces");
    }

    #[test]
    fn subtracting_a_box_that_only_touches_leaves_the_first_alone() {
        let a = box_at(Vec3::ZERO, 2.0);
        let b = box_at(Vec3::new(2.0, 0.0, 0.0), 2.0); // touches at x = 1
        let d = subtract(&a, &b).unwrap();
        assert!((mesh_volume(&d) - 8.0).abs() < 0.05, "volume {}", mesh_volume(&d));
        assert!(is_watertight(&d), "nothing was taken off it");
    }

    #[test]
    fn boxes_that_overlap_half_way_with_coplanar_faces_give_the_right_volume() {
        // The top face of the first and the bottom face of the second are the
        // same plane, and they overlap over their whole area.
        let a = box_at(Vec3::ZERO, 2.0); // z -1..1
        let b = box_at(Vec3::new(0.0, 0.0, 1.0), 2.0); // z 0..2
        let u = union(&a, &b).unwrap();
        assert!((mesh_volume(&u) - 12.0).abs() < 0.05, "volume {}", mesh_volume(&u));
        assert!(is_watertight(&u), "coplanar faces still make one solid");
        let i = intersect(&a, &b).unwrap();
        assert!((mesh_volume(&i) - 4.0).abs() < 0.05, "volume {}", mesh_volume(&i));
        assert!(is_watertight(&i));
    }

    #[test]
    fn a_box_inside_another_leaves_a_shell_when_subtracted() {
        let outer = box_at(Vec3::ZERO, 4.0);
        let inner = box_at(Vec3::ZERO, 2.0);
        let shell = subtract(&outer, &inner).unwrap();
        // 64 - 8.
        assert!(
            (mesh_volume(&shell) - 56.0).abs() < 0.1,
            "volume {}",
            mesh_volume(&shell)
        );
        assert!(is_watertight(&shell), "the shell is closed all the way round");
        let (boundary, _, over) = edge_sharing(&shell);
        assert_eq!(boundary, 0, "the cavity is closed too");
        assert_eq!(over, 0);
        // And the intersection is just the inner box.
        let i = intersect(&outer, &inner).unwrap();
        assert!((mesh_volume(&i) - 8.0).abs() < 0.05, "volume {}", mesh_volume(&i));
    }

    #[test]
    fn a_notch_cut_out_of_a_box_leaves_the_expected_volume() {
        let a = box_at(Vec3::ZERO, 2.0);
        let notch = box_at(Vec3::new(1.0, 0.0, 0.0), 2.0); // half inside, half out
        let d = subtract(&a, &notch).unwrap();
        // 8 - 4.
        assert!((mesh_volume(&d) - 4.0).abs() < 0.05, "volume {}", mesh_volume(&d));
        assert!(is_watertight(&d));
    }

    #[test]
    fn a_boolean_of_two_spheres_is_a_closed_solid() {
        // The hard case: every face of one solid crosses the other, so the
        // split is full of slivers, and the result has to close up around
        // them.
        let a = ball(Vec3::ZERO, 1.0, 16, 8);
        let b = ball(Vec3::new(0.8, 0.0, 0.0), 1.0, 16, 8);
        let u = union(&a, &b).unwrap();
        let (boundary, _, over) = edge_sharing(&u);
        assert_eq!(boundary, 0, "the union of two spheres has no open edges");
        assert_eq!(over, 0, "and no edge shared by three faces");
        assert!(is_watertight(&u), "the union is a closed solid");
        // Two spheres of radius 1, 0.8 apart: 6.57 for the true spheres, and
        // a little under that for the polyhedral ones the boolean works on.
        let v = mesh_volume(&u);
        assert!((6.0..6.6).contains(&v), "volume {v} is not near the union's");
        assert!(has_no_degenerate_faces(&u), "no face is a point or a line");
        assert!(is_welded(&u, 1e-4), "no two vertices share a place");
    }

    #[test]
    fn a_subtraction_of_two_spheres_is_a_closed_solid() {
        let a = ball(Vec3::ZERO, 1.0, 16, 8);
        let b = ball(Vec3::new(0.8, 0.0, 0.0), 1.0, 16, 8);
        let d = subtract(&a, &b).unwrap();
        assert!(is_watertight(&d), "the difference is a closed solid");
        let (boundary, _, over) = edge_sharing(&d);
        assert_eq!(boundary, 0);
        assert_eq!(over, 0);
        assert!(has_no_degenerate_faces(&d));
    }

    #[test]
    fn the_same_geometry_gives_the_same_answer_at_any_scale() {
        // The engine's tolerances are relative to the model, so a scene in
        // millimetres and the same scene in kilometres have to come out the
        // same. An absolute tolerance would swallow the small one entirely and
        // lose accuracy in the large one.
        for scale in [1e-3f32, 1.0, 1e3] {
            let a = box_at(Vec3::ZERO, 2.0 * scale);
            let b = box_at(Vec3::new(1.0 * scale, 0.0, 0.0), 2.0 * scale);
            let u = union(&a, &b).unwrap();
            let want = 12.0 * scale * scale * scale;
            let got = mesh_volume(&u);
            assert!(
                (got - want).abs() <= want * 1e-3,
                "at scale {scale}: volume {got} is not the {want} of the union"
            );
            assert!(is_watertight(&u), "at scale {scale} the union is not closed");
        }
    }

    #[test]
    fn a_face_collapsed_to_a_line_does_not_break_the_boolean() {
        // A degenerate face in the input: two of its corners are the same
        // point, so it has no area at all (and the solid it was part of is
        // therefore open, which is why no volume is claimed here).
        let mut a = box_at(Vec3::ZERO, 2.0);
        a.positions[1] = a.positions[0];
        let b = box_at(Vec3::new(1.0, 0.0, 0.0), 2.0);
        let u = union(&a, &b).unwrap();
        assert!(is_watertight(&u), "the sliver did not open a hole");
        assert!(has_no_degenerate_faces(&u), "and left no degenerate face behind");
        assert!(u.face_count() > 0, "and the result is not empty");
    }

    #[test]
    fn a_boolean_keeps_the_solids_where_they_were() {
        // The result cannot reach outside the two inputs' own bounds.
        let a = box_at(Vec3::new(-1.0, 0.0, 0.0), 2.0);
        let b = box_at(Vec3::new(1.0, 0.0, 0.0), 2.0);
        for result in [union(&a, &b).unwrap(), intersect(&a, &b).unwrap()] {
            for p in &result.positions {
                assert!(p.x >= -2.01 && p.x <= 2.01, "x {} is outside", p.x);
                assert!(p.y >= -1.01 && p.y <= 1.01, "y {} is outside", p.y);
                assert!(p.z >= -1.01 && p.z <= 1.01, "z {} is outside", p.z);
            }
        }
    }

    #[test]
    fn a_boolean_of_a_thousand_faces_finishes() {
        // A scene big enough for the old per-edge walk over every point to
        // matter, and big enough to be worth keeping an eye on.
        let a = ball(Vec3::ZERO, 1.0, 32, 16);
        let b = ball(Vec3::new(0.5, 0.0, 0.0), 1.0, 32, 16);
        let u = union(&a, &b).unwrap();
        assert!(is_watertight(&u), "the big union is still a closed solid");
        assert!(u.face_count() > 100, "and it really is a big mesh");
    }

    // --- the texture and the material -------------------------------------------

    /// A box whose six faces each carry the unit square of UVs, so a cut
    /// across a face has somewhere to interpolate to.
    fn textured_box(center: Vec3, size: f32) -> Mesh {
        let mut m = box_at(center, size);
        for f in &mut m.faces {
            let n = f.corners.len();
            for (i, c) in f.corners.iter_mut().enumerate() {
                let u = i as f32 / (n - 1) as f32;
                c.uv = m.uvs.len() as u32;
                m.uvs.push(glam::Vec2::new(u, 1.0 - u));
            }
        }
        m
    }

    #[test]
    fn a_boolean_keeps_the_texture() {
        let a = textured_box(Vec3::ZERO, 2.0);
        let b = textured_box(Vec3::new(1.0, 0.0, 0.0), 2.0);
        let u = union(&a, &b).unwrap();
        assert!(!u.uvs.is_empty(), "the result still has UVs");
        // Every corner points at a UV that exists.
        for f in &u.faces {
            for c in &f.corners {
                assert!(
                    (c.uv as usize) < u.uvs.len(),
                    "a corner points past the UVs"
                );
            }
        }
        // And they are the *original* UVs, not a fresh pile of zeroes: the
        // unit square's corners are all there.
        let has_edge = u.uvs.iter().any(|uv| uv.x < 0.01 && uv.y > 0.99);
        let has_middle = u.uvs.iter().any(|uv| uv.x > 0.4 && uv.x < 0.6);
        assert!(has_edge, "the original UVs are still there: {:?}", u.uvs);
        assert!(
            has_middle,
            "and the cut interpolated new ones along the edge: {:?}",
            u.uvs
        );
    }

    #[test]
    fn a_cut_interpolates_the_uv_along_the_edge_it_crosses() {
        // A single quad with the unit square on it, cut across by a thin
        // box: the points where the cut crosses the edges take the UV that
        // the edge's own span puts there.
        let mut quad = Mesh::new();
        for p in [
            Vec3::new(-1.0, -1.0, 0.0),
            Vec3::new(1.0, -1.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(-1.0, 1.0, 0.0),
        ] {
            quad.add_vertex(p);
        }
        quad.add_face(&[0, 1, 2, 3]);
        for (i, uv) in [
            glam::Vec2::new(0.0, 1.0),
            glam::Vec2::new(1.0, 1.0),
            glam::Vec2::new(1.0, 0.0),
            glam::Vec2::new(0.0, 0.0),
        ]
        .iter()
        .enumerate()
        {
            quad.uvs.push(*uv);
            quad.faces[0].corners[i].uv = i as u32;
        }
        // A thin box standing across the middle of it.
        let cutter = box_sized(Vec3::ZERO, Vec3::new(0.2, 2.0, 2.0));
        let polys = bsp::union(&bsp::mesh_to_polygons(&quad), &bsp::mesh_to_polygons(&cutter));
        // The cut lands at x = ±0.1, which is u = 0.45 and u = 0.55 on the
        // quad's own span — and not at the ends, which is what an
        // uninterpolated split would give.
        let interpolated = polys
            .iter()
            .flat_map(|p| p.uvs.iter())
            .filter(|uv| uv.x > 0.4 && uv.x < 0.6)
            .count();
        assert!(
            interpolated >= 2,
            "the cut's own vertices carry the UV of the edge's span: {interpolated}"
        );
    }

    #[test]
    fn the_material_travels_with_the_face() {
        // Two faces, two materials: the result keeps both, and the face that
        // came from the second solid still points at the second material.
        let mut a = box_at(Vec3::ZERO, 2.0);
        for (i, f) in a.faces.iter_mut().enumerate() {
            f.material = i as u32 % 2;
        }
        let mut b = box_at(Vec3::new(1.0, 0.0, 0.0), 2.0);
        for f in &mut b.faces {
            f.material = 7;
        }
        let u = union(&a, &b).unwrap();
        let materials: std::collections::HashSet<u32> = u.faces.iter().map(|f| f.material).collect();
        assert!(
            materials.contains(&0) && materials.contains(&1),
            "the first solid's materials are still there: {materials:?}"
        );
        assert!(
            materials.contains(&7),
            "and the second solid's material came with it: {materials:?}"
        );
    }

    // --- normals, splitting and shaving ----------------------------------------

    /// A ball whose corners carry smooth normals: the normal index is the
    /// position index, so every face meeting at a point shades it the same.
    fn smooth_ball(center: Vec3, radius: f32) -> Mesh {
        let mut m = ball(center, radius, 16, 8);
        for c in m.faces.iter_mut().flat_map(|f| f.corners.iter_mut()) {
            c.normal = c.position;
        }
        let normals: Vec<Vec3> = m
            .positions
            .iter()
            .map(|p| (*p - center).normalize_or_zero())
            .collect();
        m.normals = normals;
        m
    }

    #[test]
    fn a_boolean_keeps_smooth_normals_instead_of_recomputing_them_flat() {
        let a = smooth_ball(Vec3::ZERO, 1.0);
        let b = smooth_ball(Vec3::new(0.8, 0.0, 0.0), 1.0);
        let u = union(&a, &b).unwrap();
        assert!(!u.normals.is_empty(), "the normals came through the cut");
        let mut smooth_corner = false;
        for f in &u.faces {
            for c in &f.corners {
                assert!(
                    (c.normal as usize) < u.normals.len(),
                    "a corner points past the normals"
                );
                assert!(
                    (u.normals[c.normal as usize].length() - 1.0).abs() < 1e-3,
                    "the carried normal is unit length"
                );
            }
            let p: Vec<Vec3> = f
                .corners
                .iter()
                .map(|c| u.positions[c.position as usize])
                .collect();
            if p.len() < 3 {
                continue;
            }
            let flat = (p[1] - p[0]).cross(p[2] - p[0]).normalize_or_zero();
            if flat.length_squared() > 0.5 {
                let n = u.normals[f.corners[0].normal as usize];
                if flat.dot(n) < 0.95 {
                    smooth_corner = true;
                }
            }
        }
        assert!(
            smooth_corner,
            "at least one result corner still shades smoothly across the cut"
        );
    }

    #[test]
    fn a_boolean_of_meshes_without_normals_still_has_usable_normals() {
        let mut a = box_at(Vec3::ZERO, 2.0);
        let mut b = box_at(Vec3::new(1.0, 0.0, 0.0), 2.0);
        a.normals.clear();
        b.normals.clear();
        for f in a.faces.iter_mut().chain(b.faces.iter_mut()) {
            for c in &mut f.corners {
                c.normal = u32::MAX;
            }
        }
        let u = union(&a, &b).unwrap();
        assert_eq!(u.normals.len(), u.faces.iter().map(|f| f.corners.len()).sum());
        assert!(u.normals.iter().all(|n| n.length() > 0.99));
    }

    #[test]
    fn splitting_a_box_at_zero_gives_two_closed_halves() {
        let box_mesh = box_at(Vec3::ZERO, 2.0);
        let (front, back) = split_by_plane(&box_mesh, Vec3::ZERO, Vec3::X).unwrap();
        assert!((mesh_volume(&front) - 4.0).abs() < 0.1, "front {}", mesh_volume(&front));
        assert!((mesh_volume(&back) - 4.0).abs() < 0.1, "back {}", mesh_volume(&back));
        assert!(is_watertight(&front), "the front half is capped");
        assert!(is_watertight(&back), "the back half is capped");
    }

    #[test]
    fn splitting_off_axis_keeps_both_halves_closed() {
        let box_mesh = box_at(Vec3::ZERO, 2.0);
        let (front, back) =
            split_by_plane(&box_mesh, Vec3::ZERO, Vec3::new(1.0, 1.0, 0.0)).unwrap();
        assert!((mesh_volume(&front) - 4.0).abs() < 0.15, "front {}", mesh_volume(&front));
        assert!((mesh_volume(&back) - 4.0).abs() < 0.15, "back {}", mesh_volume(&back));
        assert!(is_watertight(&front) && is_watertight(&back));
    }

    #[test]
    fn splitting_away_from_the_centre_gives_the_expected_volumes() {
        let box_mesh = box_at(Vec3::ZERO, 2.0);
        let (front, back) =
            split_by_plane(&box_mesh, Vec3::new(0.5, 0.0, 0.0), Vec3::X).unwrap();
        // x 0.5..1 is 0.5 thick; x -1..0.5 is 1.5 thick. Both span 2×2.
        assert!((mesh_volume(&front) - 2.0).abs() < 0.1, "front {}", mesh_volume(&front));
        assert!((mesh_volume(&back) - 6.0).abs() < 0.1, "back {}", mesh_volume(&back));
    }

    /// The index of the face whose outward normal points most towards +Y.
    fn top_face(mesh: &Mesh) -> usize {
        let mut best = (0usize, -1.0f32);
        for (i, f) in mesh.faces.iter().enumerate() {
            let p: Vec<Vec3> = f
                .corners
                .iter()
                .map(|c| mesh.positions[c.position as usize])
                .collect();
            if p.len() < 3 {
                continue;
            }
            let n = (p[1] - p[0]).cross(p[2] - p[0]).normalize_or_zero();
            if n.y > best.1 {
                best = (i, n.y);
            }
        }
        assert!(best.1 > 0.9, "the box has an upward face");
        best.0
    }

    #[test]
    fn shaving_the_top_face_removes_a_uniform_slab() {
        let box_mesh = box_at(Vec3::ZERO, 2.0);
        let top = top_face(&box_mesh);
        let shaved = shave_face(&box_mesh, top, 0.5).unwrap();
        // 2 × 1.5 × 2 = 6, down from 8.
        assert!((mesh_volume(&shaved) - 6.0).abs() < 0.1, "volume {}", mesh_volume(&shaved));
        assert!(is_watertight(&shaved), "the shaved solid is still closed");
        for p in &shaved.positions {
            assert!(p.y <= 0.51, "nothing is left above the new top: {p}");
        }
    }

    #[test]
    fn splitting_and_shaving_refuse_bad_arguments() {
        let box_mesh = box_at(Vec3::ZERO, 2.0);
        let empty = Mesh::new();
        assert_eq!(
            split_by_plane(&empty, Vec3::ZERO, Vec3::X),
            Err(BooleanError::EmptyInput)
        );
        assert_eq!(
            split_by_plane(&box_mesh, Vec3::ZERO, Vec3::ZERO),
            Err(BooleanError::DegeneratePlane)
        );
        assert_eq!(
            shave_face(&box_mesh, 999, 0.5),
            Err(BooleanError::InvalidFace(999))
        );
        assert_eq!(shave_face(&box_mesh, 0, 0.0), Err(BooleanError::InvalidDepth));
        assert_eq!(
            shave_face(&box_mesh, 0, f32::NAN),
            Err(BooleanError::InvalidDepth)
        );
    }
}
