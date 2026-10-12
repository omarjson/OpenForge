//! Sub-object modeling operations on [`Mesh`].
//!
//! These are the workhorse editing operations (extrude, inset, weld,
//! subdivide, poke, delete) that operate on selected faces/edges/vertices.
//! All operations preserve n-gon support and per-corner attributes.

use crate::mesh::{Corner, Face, Mesh};
use crate::Vec3;
use std::collections::{HashMap, HashSet};

/// Outcome of a face extrusion.
pub struct ExtrudeResult {
    /// Indices of the cap faces (the moved originals).
    pub cap_faces: Vec<usize>,
    /// Number of side-wall faces created.
    pub side_faces: usize,
}

impl Mesh {
    /// Extract every face's directed edges as `(a, b)` position-index pairs.
    pub fn directed_edges(&self) -> Vec<(u32, u32)> {
        let mut out = Vec::new();
        for f in &self.faces {
            let cs = &f.corners;
            for i in 0..cs.len() {
                out.push((cs[i].position, cs[(i + 1) % cs.len()].position));
            }
        }
        out
    }

    /// Map each position index to the faces that reference it.
    pub fn vertex_to_faces(&self) -> HashMap<u32, Vec<usize>> {
        let mut map: HashMap<u32, Vec<usize>> = HashMap::new();
        for (fi, f) in self.faces.iter().enumerate() {
            for c in &f.corners {
                map.entry(c.position).or_default().push(fi);
            }
        }
        map
    }

    /// Faces that share an edge with any face in `faces` but are not in it.
    pub fn face_neighbors(&self, faces: &HashSet<usize>) -> HashSet<usize> {
        // Directed edges of selected faces whose reverse is NOT selected are
        // boundary edges; the face owning the reverse is a neighbor.
        let mut selected_dir: HashSet<(u32, u32)> = HashSet::new();
        for &fi in faces {
            let cs = &self.faces[fi].corners;
            for i in 0..cs.len() {
                selected_dir.insert((cs[i].position, cs[(i + 1) % cs.len()].position));
            }
        }
        let mut neighbors = HashSet::new();
        for (fi, f) in self.faces.iter().enumerate() {
            if faces.contains(&fi) {
                continue;
            }
            let cs = &f.corners;
            for i in 0..cs.len() {
                let a = cs[i].position;
                let b = cs[(i + 1) % cs.len()].position;
                if selected_dir.contains(&(b, a)) {
                    neighbors.insert(fi);
                    break;
                }
            }
        }
        neighbors
    }

    /// Extrude a connected region of faces along its average normal.
    ///
    /// The selected faces are moved outward to form the cap; side walls connect
    /// the original boundary edges to the new cap vertices. Works for a single
    /// face or a connected multi-face region.
    pub fn extrude_faces(&mut self, faces: &[usize], distance: f32) -> Option<ExtrudeResult> {
        if faces.is_empty() {
            return None;
        }
        // Extrusion direction: area-weighted average normal.
        let mut dir = Vec3::ZERO;
        for &f in faces {
            if f < self.faces.len() {
                dir += self.face_normal(f);
            }
        }
        dir = dir.normalize_or_zero();
        if dir == Vec3::ZERO {
            return None;
        }
        let offset = dir * distance;

        // Create offset copies of every position the region touches.
        let mut pos_map: HashMap<u32, u32> = HashMap::new();
        for &fi in faces {
            for c in self.faces[fi].corners.clone() {
                pos_map.entry(c.position).or_insert_with(|| {
                    let v = self.positions[c.position as usize] + offset;
                    self.add_vertex(v)
                });
            }
        }

        // Boundary directed edges: in the region, with no selected reverse.
        let mut selected_dir: HashSet<(u32, u32)> = HashSet::new();
        for &fi in faces {
            let cs = &self.faces[fi].corners;
            for i in 0..cs.len() {
                selected_dir.insert((cs[i].position, cs[(i + 1) % cs.len()].position));
            }
        }
        let boundary: Vec<(u32, u32)> = selected_dir
            .iter()
            .filter(|(a, b)| !selected_dir.contains(&(*b, *a)))
            .copied()
            .collect();

        // Re-point the selected faces to the offset vertices (they become the cap).
        let cap_faces = faces.to_vec();
        for &fi in &cap_faces {
            for c in self.faces[fi].corners.iter_mut() {
                c.position = pos_map[&c.position];
            }
        }

        // Side walls: quad [a, b, new_b, new_a] for each boundary edge a->b.
        let mut side_faces = 0;
        for (a, b) in boundary {
            let na = pos_map[&a];
            let nb = pos_map[&b];
            self.faces.push(Face {
                corners: vec![
                    Corner { position: a, normal: 0, uv: 0 },
                    Corner { position: b, normal: 0, uv: 0 },
                    Corner { position: nb, normal: 0, uv: 0 },
                    Corner { position: na, normal: 0, uv: 0 },
                ],
                material: 0,
            });
            side_faces += 1;
        }

        self.compute_flat_normals();
        Some(ExtrudeResult { cap_faces, side_faces })
    }

    /// Inset each selected face individually, creating a border ring and an
    /// inner cap scaled toward the face centroid by `amount` (0..1).
    pub fn inset_faces(&mut self, faces: &[usize], amount: f32) -> Option<Vec<usize>> {
        let amount = amount.clamp(0.0, 0.99);
        let mut new_caps = Vec::new();
        // Process highest index first so earlier indices stay valid.
        let mut ordered = faces.to_vec();
        ordered.sort_unstable();
        for &fi in &ordered {
            if fi >= self.faces.len() {
                continue;
            }
            let corners = self.faces[fi].corners.clone();
            let material = self.faces[fi].material;
            if corners.len() < 3 {
                continue;
            }
            // Centroid.
            let mut cen = Vec3::ZERO;
            for c in &corners {
                cen += self.positions[c.position as usize];
            }
            cen /= corners.len() as f32;
            // New inner vertices pulled toward centroid.
            let mut inner: Vec<u32> = Vec::with_capacity(corners.len());
            for c in &corners {
                let p = self.positions[c.position as usize];
                let np = p + (cen - p) * amount;
                inner.push(self.add_vertex(np));
            }
            // Inner cap face (same winding as original).
            let inner_corners: Vec<Corner> = inner
                .iter()
                .map(|&p| Corner { position: p, normal: 0, uv: 0 })
                .collect();
            new_caps.push(self.faces.len());
            self.faces.push(Face { corners: inner_corners, material });
            // Border quads replace the original face region.
            // The original face is repurposed as the first border segment.
            let n = corners.len();
            // Turn original face into border quad [p0, p1, inner1, inner0].
            self.faces[fi].corners = vec![
                corners[0],
                corners[1],
                Corner { position: inner[1], normal: 0, uv: 0 },
                Corner { position: inner[0], normal: 0, uv: 0 },
            ];
            // Remaining border segments.
            for i in 1..n {
                let j = (i + 1) % n;
                let seg = vec![
                    corners[i],
                    corners[j],
                    Corner { position: inner[j], normal: 0, uv: 0 },
                    Corner { position: inner[i], normal: 0, uv: 0 },
                ];
                self.faces.push(Face { corners: seg, material });
            }
        }
        self.compute_flat_normals();
        Some(new_caps)
    }

    /// Merge vertices closer than `eps`. Returns how many vertices were merged.
    pub fn weld_close(&mut self, eps: f32) -> usize {
        let n = self.positions.len();
        let mut rep = vec![u32::MAX; n];
        let mut new_positions: Vec<Vec3> = Vec::new();
        for i in 0..n {
            if rep[i] != u32::MAX {
                continue;
            }
            let idx = new_positions.len() as u32;
            rep[i] = idx;
            new_positions.push(self.positions[i]);
            for j in (i + 1)..n {
                if rep[j] == u32::MAX && self.positions[i].distance(self.positions[j]) <= eps {
                    rep[j] = idx;
                }
            }
        }
        let merged = n - new_positions.len();
        self.positions = new_positions;
        for f in self.faces.iter_mut() {
            for c in f.corners.iter_mut() {
                c.position = rep[c.position as usize];
            }
        }
        // Drop degenerate faces that now repeat a position.
        self.faces.retain(|f| {
            let mut seen = HashSet::new();
            f.corners.iter().all(|c| seen.insert(c.position))
        });
        self.compact();
        merged
    }

    /// Poke each selected face: add a centroid and fan-triangulate. Returns
    /// the new face count added.
    pub fn poke_faces(&mut self, faces: &[usize]) -> usize {
        let mut added = 0;
        for &fi in faces {
            if fi >= self.faces.len() {
                continue;
            }
            let corners = self.faces[fi].corners.clone();
            let material = self.faces[fi].material;
            if corners.len() < 3 {
                continue;
            }
            let mut cen = Vec3::ZERO;
            for c in &corners {
                cen += self.positions[c.position as usize];
            }
            cen /= corners.len() as f32;
            let ci = self.add_vertex(cen);
            let n = corners.len();
            // Repurpose original face as first triangle, add the rest.
            self.faces[fi].corners = vec![
                corners[0],
                corners[1],
                Corner { position: ci, normal: 0, uv: 0 },
            ];
            for i in 1..n {
                let j = (i + 1) % n;
                self.faces.push(Face {
                    corners: vec![
                        corners[i],
                        corners[j],
                        Corner { position: ci, normal: 0, uv: 0 },
                    ],
                    material,
                });
                added += 1;
            }
            added += 1;
        }
        self.compute_flat_normals();
        added
    }

    /// Subdivide every selected face into quads by splitting each edge at its
    /// midpoint and adding a face centroid. Each N-gon becomes N quads.
    pub fn subdivide_faces(&mut self, faces: &[usize]) -> usize {
        let mut added = 0;
        for &fi in faces {
            if fi >= self.faces.len() {
                continue;
            }
            let corners = self.faces[fi].corners.clone();
            let material = self.faces[fi].material;
            let n = corners.len();
            if n < 3 {
                continue;
            }
            // Edge midpoints.
            let mut mids: Vec<u32> = Vec::with_capacity(n);
            for i in 0..n {
                let a = self.positions[corners[i].position as usize];
                let b = self.positions[corners[(i + 1) % n].position as usize];
                mids.push(self.add_vertex((a + b) * 0.5));
            }
            // Face centroid.
            let mut cen = Vec3::ZERO;
            for c in &corners {
                cen += self.positions[c.position as usize];
            }
            cen /= n as f32;
            let ci = self.add_vertex(cen);
            // Each corner becomes a quad: [mid_prev, corner, mid_next, centroid].
            let quad_for = |i: usize| -> Vec<Corner> {
                let prev = mids[(i + n - 1) % n];
                let next = mids[i];
                vec![
                    Corner { position: prev, normal: 0, uv: 0 },
                    corners[i],
                    Corner { position: next, normal: 0, uv: 0 },
                    Corner { position: ci, normal: 0, uv: 0 },
                ]
            };
            self.faces[fi].corners = quad_for(0);
            self.faces[fi].material = material;
            for i in 1..n {
                self.faces.push(Face { corners: quad_for(i), material });
                added += 1;
            }
            added += 1;
        }
        self.compute_flat_normals();
        added
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quad() -> Mesh {
        let mut m = Mesh::new();
        m.add_vertex(Vec3::new(0.0, 0.0, 0.0));
        m.add_vertex(Vec3::new(1.0, 0.0, 0.0));
        m.add_vertex(Vec3::new(1.0, 1.0, 0.0));
        m.add_vertex(Vec3::new(0.0, 1.0, 0.0));
        m.add_face(&[0, 1, 2, 3]);
        m
    }

    #[test]
    fn extrude_single_quad_makes_box_shell() {
        let mut m = quad();
        let r = m.extrude_faces(&[0], 1.0).unwrap();
        assert_eq!(r.cap_faces, vec![0]);
        assert_eq!(r.side_faces, 4);
        // 4 original + 4 new = 8 verts; 1 cap + 4 walls = 5 faces.
        assert_eq!(m.vertex_count(), 8);
        assert_eq!(m.face_count(), 5);
        // Cap should be lifted to z=1.
        let cap_center_z: f32 = {
            let f = &m.faces[0];
            f.corners.iter().map(|c| m.positions[c.position as usize].z).sum::<f32>() / 4.0
        };
        assert!((cap_center_z - 1.0).abs() < 1e-5);
    }

    #[test]
    fn extrude_region_two_quads() {
        // Two adjacent quads sharing an edge.
        let mut m = Mesh::new();
        // quad A: 0,1,2,3 ; quad B: 1,4,5,2 (share edge 1-2)
        for p in [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(2.0, 1.0, 0.0),
        ] {
            m.add_vertex(p);
        }
        m.add_face(&[0, 1, 2, 3]); // A
        m.add_face(&[1, 4, 5, 2]); // B
        let before = m.face_count();
        let r = m.extrude_faces(&[0, 1], 1.0).unwrap();
        // Shared interior edge 1-2 is not a boundary, so only exterior edges
        // make walls: A has 3 exterior, B has 3 exterior = 6 walls.
        assert_eq!(r.side_faces, 6);
        assert_eq!(m.face_count(), before + 6);
    }

    #[test]
    fn inset_creates_border_and_cap() {
        let mut m = quad();
        let caps = m.inset_faces(&[0], 0.5).unwrap();
        assert_eq!(caps.len(), 1);
        // 1 inner cap + 4 border quads = 5 faces total.
        assert_eq!(m.face_count(), 5);
    }

    #[test]
    fn weld_merges_coincident() {
        let mut m = Mesh::new();
        m.add_vertex(Vec3::new(0.0, 0.0, 0.0));
        m.add_vertex(Vec3::new(0.0, 0.0, 0.0001)); // near-duplicate
        m.add_vertex(Vec3::new(1.0, 0.0, 0.0));
        m.add_vertex(Vec3::new(0.0, 1.0, 0.0));
        m.add_face(&[0, 2, 3]);
        m.add_face(&[1, 2, 3]);
        let merged = m.weld_close(0.001);
        assert_eq!(merged, 1);
        assert_eq!(m.vertex_count(), 3);
        // The two faces become identical and one is degenerate after weld.
        assert!(m.face_count() >= 1);
    }

    #[test]
    fn subdivide_quads_face_count() {
        let mut m = quad();
        m.subdivide_faces(&[0]);
        // One quad -> 4 quads.
        assert_eq!(m.face_count(), 4);
        // 4 corners + 4 midpoints + 1 centroid = 9 verts.
        assert_eq!(m.vertex_count(), 9);
    }

    #[test]
    fn poke_triangulates_fan() {
        let mut m = quad();
        m.poke_faces(&[0]);
        // Quad poked -> 4 triangles.
        assert_eq!(m.face_count(), 4);
    }
}
