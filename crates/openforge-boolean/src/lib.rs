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

mod bsp;

use bsp::{mesh_to_polygons, polygons_to_mesh};
use glam::Vec3;
use openforge_core::mesh::Mesh;
use thiserror::Error;

/// Errors produced by a boolean operation.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum BooleanError {
    /// One of the inputs had no faces to operate on.
    #[error("mesh has no faces")]
    EmptyInput,
}

/// Union of two meshes: everything inside `a` or `b` (or both).
pub fn union(a: &Mesh, b: &Mesh) -> Result<Mesh, BooleanError> {
    if a.faces.is_empty() || b.faces.is_empty() {
        return Err(BooleanError::EmptyInput);
    }
    let pa = mesh_to_polygons(a);
    let pb = mesh_to_polygons(b);
    Ok(polygons_to_mesh(&bsp::union(&pa, &pb)))
}

/// Difference `a - b`: everything inside `a` that is not inside `b`.
pub fn subtract(a: &Mesh, b: &Mesh) -> Result<Mesh, BooleanError> {
    if a.faces.is_empty() || b.faces.is_empty() {
        return Err(BooleanError::EmptyInput);
    }
    let pa = mesh_to_polygons(a);
    let pb = mesh_to_polygons(b);
    Ok(polygons_to_mesh(&bsp::subtract(&pa, &pb)))
}

/// Intersection of two meshes: everything inside both `a` and `b`.
pub fn intersect(a: &Mesh, b: &Mesh) -> Result<Mesh, BooleanError> {
    if a.faces.is_empty() || b.faces.is_empty() {
        return Err(BooleanError::EmptyInput);
    }
    let pa = mesh_to_polygons(a);
    let pb = mesh_to_polygons(b);
    Ok(polygons_to_mesh(&bsp::intersect(&pa, &pb)))
}

/// Signed volume of a closed mesh via the divergence theorem, returned as an
/// absolute (always non-negative) value. Meaningful only for watertight
/// meshes; open meshes give an arbitrary result.
pub fn mesh_volume(mesh: &Mesh) -> f32 {
    let mut vol6 = 0.0f32;
    for f in &mesh.faces {
        let pts: Vec<Vec3> = f
            .corners
            .iter()
            .filter_map(|c| mesh.positions.get(c.position as usize).copied())
            .collect();
        for i in 1..pts.len().saturating_sub(1) {
            // Signed volume of tetrahedron (origin, p0, pi, pi+1).
            vol6 += pts[0].dot(pts[i].cross(pts[i + 1]));
        }
    }
    (vol6 / 6.0).abs()
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
    use openforge_core::mesh::Mesh;

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
}
