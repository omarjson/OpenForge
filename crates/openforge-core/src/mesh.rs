//! Indexed polygon mesh with per-corner attributes.
//!
//! Faces are n-gons (3+ corners). Each corner carries its own position index,
//! normal index and UV index so that seams (e.g. UV borders) are preserved,
//! which is the standard model for an interchange/editing mesh.

use crate::math::Aabb;
use glam::{Vec2, Vec3};

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

/// Build a UV-sphere mesh. Useful as both a primitive and a test fixture.
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
}
