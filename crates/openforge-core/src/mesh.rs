//! Indexed polygon mesh with per-corner attributes.
//!
//! Faces are n-gons (3+ corners). Each corner carries its own position index,
//! normal index and UV index so that seams (e.g. UV borders) are preserved,
//! which is the standard model for an interchange/editing mesh.

use crate::math::Aabb;
use glam::{Vec2, Vec3};
use std::collections::HashSet;

/// A vertex position plus optional source data.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Vertex {
    pub position: Vec3,
}

/// A single face corner referencing shared attribute arrays.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Corner {
    pub position: u32,
    pub normal: u32,
    pub uv: u32,
}

/// A polygon face: `corners.len() >= 3`.
#[derive(Clone, PartialEq, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Face {
    pub corners: Vec<Corner>,
    /// Index into the owning node's material slot list.
    pub material: u32,
}

/// A polygon mesh.
#[derive(Clone, PartialEq, Debug, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Mesh {
    pub positions: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    pub uvs: Vec<Vec2>,
    pub faces: Vec<Face>,
}

impl Mesh {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.positions.is_empty() || self.faces.is_empty()
    }

    pub fn face_count(&self) -> usize {
        self.faces.len()
    }

    pub fn vertex_count(&self) -> usize {
        self.positions.len()
    }

    /// Triangle count assuming fan triangulation of each n-gon.
    pub fn triangulated_count(&self) -> usize {
        self.faces.iter().map(|f| f.corners.len().saturating_sub(2)).sum()
    }

    /// Append a vertex and return its index.
    pub fn add_vertex(&mut self, p: Vec3) -> u32 {
        self.positions.push(p);
        (self.positions.len() - 1) as u32
    }

    /// Add a default normal/uv slot if arrays are empty, keeping indices valid.
    pub fn ensure_attributes(&mut self) {
        if self.normals.is_empty() {
            self.normals.push(Vec3::Y);
        }
        if self.uvs.is_empty() {
            self.uvs.push(Vec2::ZERO);
        }
    }

    /// Add a face from raw position indices, auto-creating normal/uv corners.
    pub fn add_face(&mut self, positions: &[u32]) -> Option<usize> {
        if positions.len() < 3 {
            return None;
        }
        self.ensure_attributes();
        let corners = positions
            .iter()
            .map(|&p| Corner { position: p, normal: 0, uv: 0 })
            .collect();
        self.faces.push(Face { corners, material: 0 });
        Some(self.faces.len() - 1)
    }

    /// Compute flat per-face normals into a fresh array, returning
    /// `(normals, corner_normal_indices)` where each face's corners all point
    /// at that face's normal.
    pub fn compute_flat_normals(&mut self) {
        let mut normals = Vec::with_capacity(self.faces.len());
        for fi in 0..self.faces.len() {
            normals.push(self.face_normal(fi));
        }
        for (fi, face) in self.faces.iter_mut().enumerate() {
            let idx = fi as u32;
            for c in face.corners.iter_mut() {
                c.normal = idx;
            }
        }
        self.normals = normals;
    }

    /// Area-weighted smooth normals written per position index.
    pub fn compute_smooth_normals(&mut self) {
        let mut acc = vec![Vec3::ZERO; self.positions.len()];
        for fi in 0..self.faces.len() {
            let n = self.face_normal(fi);
            let face = &self.faces[fi];
            for c in &face.corners {
                acc[c.position as usize] += n;
            }
        }
        self.normals = acc.into_iter().map(|v| v.normalize_or_zero()).collect();
        if self.normals.is_empty() {
            self.normals.push(Vec3::Y);
        }
        for face in self.faces.iter_mut() {
            for c in face.corners.iter_mut() {
                // Smooth normals index by position here.
                c.normal = c.position;
            }
        }
    }

    /// Geometric normal of a face (Newell's method, robust for n-gons).
    pub fn face_normal(&self, fi: usize) -> Vec3 {
        let face = &self.faces[fi];
        let cs = &face.corners;
        let mut n = Vec3::ZERO;
        for i in 0..cs.len() {
            let a = self.positions[cs[i].position as usize];
            let b = self.positions[cs[(i + 1) % cs.len()].position as usize];
            n += a.cross(b);
        }
        n.normalize_or_zero()
    }

    /// Center of a face's corner positions.
    pub fn face_center(&self, fi: usize) -> Vec3 {
        let face = &self.faces[fi];
        let mut s = Vec3::ZERO;
        for c in &face.corners {
            s += self.positions[c.position as usize];
        }
        s / face.corners.len() as f32
    }

    pub fn bounds(&self) -> Aabb {
        let mut b = Aabb::default();
        for p in &self.positions {
            b.expand_point(*p);
        }
        b
    }

    /// Fan-triangulate every face into `(i0, i1, i2)` position-index triples.
    pub fn triangulate(&self) -> Vec<[u32; 3]> {
        let mut out = Vec::with_capacity(self.triangulated_count());
        for face in &self.faces {
            let cs = &face.corners;
            for i in 1..cs.len().saturating_sub(1) {
                out.push([cs[0].position, cs[i].position, cs[i + 1].position]);
            }
        }
        out
    }

    /// Every edge of the mesh, as an unordered pair of position indices.
    ///
    /// The mesh keeps no edge list — an edge is simply two corners that sit
    /// next to each other on a face, and a pair shared by two faces is one
    /// edge. Faces are walked in order and each face's corners around it (the
    /// last back to the first), and a pair enters the list the first time it
    /// appears, so an edge's index is a function of the face list alone and
    /// never moves while the faces do not. Two rules keep the list honest:
    /// a face with fewer than three corners, or with a corner outside
    /// [`positions`](Self::positions), is skipped whole — the visible pass
    /// never draws it, so none of its sides exist — and a pair repeating one
    /// position index (a degenerate corner) is no edge at all. Each pair is
    /// stored with the lower index first.
    pub fn edges(&self) -> Vec<[u32; 2]> {
        let mut out: Vec<[u32; 2]> = Vec::new();
        let mut seen: HashSet<(u32, u32)> = HashSet::new();
        for face in &self.faces {
            if !self.face_is_valid(face) {
                continue;
            }
            let n = face.corners.len();
            for k in 0..n {
                let a = face.corners[k].position;
                let b = face.corners[(k + 1) % n].position;
                if a == b {
                    continue;
                }
                let key = if a < b { (a, b) } else { (b, a) };
                if seen.insert(key) {
                    out.push([key.0, key.1]);
                }
            }
        }
        out
    }

    /// Whether a face is one the visible pass draws at all.
    ///
    /// A face with fewer than three corners, or with a corner outside
    /// [`positions`](Self::positions), is skipped whole: none of its sides
    /// exist and none of its corners counts. [`edges`](Self::edges) and the
    /// selection maths ([`crate::subselect`]) both go through this, so what
    /// cannot be drawn cannot be selected either.
    pub fn face_is_valid(&self, face: &Face) -> bool {
        face.corners.len() >= 3
            && face
                .corners
                .iter()
                .all(|c| (c.position as usize) < self.positions.len())
    }

    /// Delete faces whose index is in `faces`, then drop unreferenced
    /// positions and remap indices. Returns the number of faces removed.
    pub fn delete_faces(&mut self, faces: &[usize]) -> usize {
        let mut remove = vec![false; self.faces.len()];
        for &f in faces {
            if f < remove.len() {
                remove[f] = true;
            }
        }
        let removed = remove.iter().filter(|&&r| r).count();
        let mut fi = 0;
        self.faces.retain(|_| {
            let keep = !remove[fi];
            fi += 1;
            keep
        });
        self.compact();
        removed
    }

    /// Remove unreferenced positions and remap face corners.
    pub fn compact(&mut self) {
        let mut used = vec![false; self.positions.len()];
        for face in &self.faces {
            for c in &face.corners {
                used[c.position as usize] = true;
            }
        }
        let mut remap = vec![u32::MAX; self.positions.len()];
        let mut next = 0u32;
        for (i, &u) in used.iter().enumerate() {
            if u {
                remap[i] = next;
                next += 1;
            }
        }
        let mut new_positions = Vec::with_capacity(next as usize);
        for (i, p) in self.positions.iter().enumerate() {
            if used[i] {
                new_positions.push(*p);
            }
        }
        self.positions = new_positions;
        for face in self.faces.iter_mut() {
            for c in face.corners.iter_mut() {
                c.position = remap[c.position as usize];
            }
        }
    }
}

/// Build a UV-sphere mesh, with spherical UVs. Useful as both a primitive and
/// a test fixture.
///
/// The vertex grid is one point per latitude/longitude pair — the seam column
/// is duplicated, so `u` runs a full 0..1 — and every corner's UV index is its
/// position index, so mapping the sphere is a no-op and the UV editor shows a
/// real unwrap instead of every entry stacked at the origin.
pub fn uv_sphere(radius: f32, segments: u32, rings: u32) -> Mesh {
    let mut m = Mesh::new();
    let segments = segments.max(3);
    let rings = rings.max(2);
    for r in 0..=rings {
        let v = r as f32 / rings as f32;
        let phi = v * std::f32::consts::PI;
        for s in 0..=segments {
            let u = s as f32 / segments as f32;
            let theta = u * std::f32::consts::TAU;
            let x = phi.sin() * theta.cos();
            let y = phi.cos();
            let z = phi.sin() * theta.sin();
            m.add_vertex(Vec3::new(x, y, z) * radius);
            m.uvs.push(Vec2::new(u, v));
        }
    }
    let stride = segments + 1;
    for r in 0..rings {
        for s in 0..segments {
            let a = r * stride + s;
            let b = a + 1;
            let c = a + stride;
            let d = c + 1;
            if r != 0 {
                m.add_face(&[a, c, b]);
            }
            if r != rings - 1 {
                m.add_face(&[b, c, d]);
            }
        }
    }
    for face in m.faces.iter_mut() {
        for corner in face.corners.iter_mut() {
            corner.uv = corner.position;
        }
    }
    m.compute_smooth_normals();
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quad_mesh_triangulates_to_two_tris() {
        let mut m = Mesh::new();
        m.add_vertex(Vec3::new(0.0, 0.0, 0.0));
        m.add_vertex(Vec3::new(1.0, 0.0, 0.0));
        m.add_vertex(Vec3::new(1.0, 1.0, 0.0));
        m.add_vertex(Vec3::new(0.0, 1.0, 0.0));
        m.add_face(&[0, 1, 2, 3]);
        assert_eq!(m.triangulated_count(), 2);
        assert_eq!(m.triangulate().len(), 2);
    }

    #[test]
    fn delete_faces_compacts_positions() {
        let mut m = Mesh::new();
        m.add_vertex(Vec3::ZERO);
        m.add_vertex(Vec3::X);
        m.add_vertex(Vec3::Y);
        m.add_vertex(Vec3::new(5.0, 5.0, 5.0));
        m.add_face(&[0, 1, 2]);
        m.add_face(&[0, 1, 3]);
        let removed = m.delete_faces(&[1]);
        assert_eq!(removed, 1);
        assert_eq!(m.face_count(), 1);
        // Vertex 3 was only used by the deleted face and should be dropped.
        assert_eq!(m.vertex_count(), 3);
    }

    #[test]
    fn sphere_is_watertight_enough() {
        let s = uv_sphere(1.0, 8, 6);
        assert!(s.vertex_count() > 0);
        assert!(s.face_count() > 0);
        let b = s.bounds();
        // Each axis of a unit sphere's AABB should reach ~the radius.
        assert!((b.max.x - 1.0).abs() < 1e-4);
        assert!((b.max.y - 1.0).abs() < 1e-4);
        assert!((b.max.z - 1.0).abs() < 1e-4);
        assert!((b.min.x + 1.0).abs() < 1e-4);
    }

    #[test]
    fn sphere_carries_spherical_uvs() {
        let s = uv_sphere(1.0, 8, 6);
        assert_eq!(s.uvs.len(), s.positions.len(), "one uv per grid point");
        // The seam column is duplicated, so u spans the full range.
        assert!(s.uvs.iter().any(|uv| uv.x == 0.0));
        assert!(s.uvs.iter().any(|uv| uv.x == 1.0));
        assert!(s.uvs.iter().any(|uv| uv.y == 0.0), "the north pole row");
        assert!(s.uvs.iter().any(|uv| uv.y == 1.0), "the south pole row");
        // Every corner samples its own grid point's uv.
        for face in &s.faces {
            for c in &face.corners {
                assert_eq!(c.uv, c.position);
            }
        }
    }

    /// A quad in the XY plane from four fresh positions.
    fn quad_mesh() -> Mesh {
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
        m
    }

    #[test]
    fn a_quad_has_four_edges_walked_around_its_corners() {
        let m = quad_mesh();
        // Corners in order, the last side closing back to the first, each
        // pair stored with the lower index first.
        assert_eq!(m.edges(), vec![[0, 1], [1, 2], [2, 3], [0, 3]]);
    }

    #[test]
    fn an_edge_shared_by_two_faces_is_one_edge_by_first_appearance() {
        let mut m = quad_mesh();
        // A second quad sharing the side (1, 2), built from two new positions.
        m.add_vertex(Vec3::new(2.0, 0.0, 0.0));
        m.add_vertex(Vec3::new(2.0, 1.0, 0.0));
        m.add_face(&[1, 4, 5, 2]);
        let edges = m.edges();
        // 4 sides + 3 new ones; (1, 2) is one edge, entered by the first quad.
        assert_eq!(edges.len(), 7);
        assert_eq!(edges.iter().filter(|e| **e == [1, 2]).count(), 1);
        assert_eq!(edges[0], [0, 1], "the first quad's sides keep their indices");
        assert_eq!(&edges[4..], &[[1, 4], [4, 5], [2, 5]]);
    }

    #[test]
    fn triangles_and_ngons_walk_every_side() {
        let mut tri = Mesh::new();
        tri.add_vertex(Vec3::ZERO);
        tri.add_vertex(Vec3::X);
        tri.add_vertex(Vec3::Y);
        tri.add_face(&[0, 1, 2]);
        assert_eq!(tri.edges(), vec![[0, 1], [1, 2], [0, 2]]);

        let mut pent = Mesh::new();
        for i in 0..5 {
            let a = i as f32 / 5.0 * std::f32::consts::TAU;
            pent.add_vertex(Vec3::new(a.cos(), a.sin(), 0.0));
        }
        pent.add_face(&[0, 1, 2, 3, 4]);
        assert_eq!(pent.edges().len(), 5);
    }

    #[test]
    fn a_face_the_visible_pass_would_skip_has_no_sides() {
        let mut m = quad_mesh();
        // Out of range: the face is never drawn, so none of its sides exist.
        m.add_face(&[0, 1, 99]);
        assert_eq!(m.edges().len(), 4);
        // A face with too few corners cannot even be added through add_face,
        // so build one by hand.
        m.faces.push(Face {
            corners: vec![
                Corner { position: 0, normal: 0, uv: 0 },
                Corner { position: 1, normal: 0, uv: 0 },
            ],
            material: 0,
        });
        assert_eq!(m.edges().len(), 4);
    }

    #[test]
    fn a_degenerate_corner_pair_is_no_edge() {
        let mut m = quad_mesh();
        // A sliver face that doubles a corner: only its real sides count.
        m.add_vertex(Vec3::new(2.0, 0.0, 0.0));
        m.add_face(&[1, 4, 4]);
        let edges = m.edges();
        assert_eq!(edges.len(), 5, "the quad's four sides and (1, 4)");
        assert_eq!(edges[4], [1, 4]);
        assert!(!edges.iter().any(|e| e[0] == e[1]), "no self-edges");
    }

    #[test]
    fn empty_and_loose_geometry_has_no_edges() {
        assert!(Mesh::new().edges().is_empty());
        let mut m = Mesh::new();
        m.add_vertex(Vec3::ZERO);
        m.add_vertex(Vec3::X);
        assert!(m.edges().is_empty(), "positions without faces are no edges");
    }
}
