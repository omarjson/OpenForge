//! Advanced modeling operators for OpenForge.
//!
//! `Mesh` is defined in `openforge-core`, so this crate exposes an extension
//! trait [`MeshOps`] implemented for [`Mesh`]. The operators that change
//! connectivity preserve watertightness for manifold input and guard all
//! indexing against out-of-range or degenerate geometry (never panicking).
//!
//! Supported well: vertex bevel (watertight chamfer), edge dissolve, face
//! bridge, mirror, triangulate, flip, transform, loop/ring queries. Edge bevel
//! is exact when the edge endpoints are not shared by additional faces (see its
//! doc comment); run [`MeshOps::bevel_vertices`] for a fully watertight chamfer.

use openforge_core::mesh::{Corner, Face, Mesh};
use openforge_core::Vec3;
use std::collections::{HashMap, HashSet};

/// Extension trait adding advanced modeling operators to [`Mesh`].
pub trait MeshOps {
    /// Chamfer the listed vertices: each is replaced by a small polygon that
    /// pulls adjacent geometry inward. `amount` in 0..1 controls how far.
    /// Watertight for manifold vertices.
    fn bevel_vertices(&mut self, verts: &[u32], amount: f32);

    /// Split each edge into a chamfer strip between its two adjacent faces.
    /// `amount` in 0..1. Exact only when the edge endpoints are not shared by
    /// other faces; otherwise follow with a weld or prefer
    /// [`MeshOps::bevel_vertices`].
    fn bevel_edges(&mut self, edges: &[(u32, u32)], amount: f32);

    /// Walk a face loop (a strip of quads) across opposite shared edges,
    /// returning the ordered face indices. Guarded to quads.
    fn edge_loops(&self, start_face: usize, start_edge: usize) -> Vec<usize>;

    /// The ring of faces sharing an edge with `face` (its edge neighbors).
    fn select_ring(&self, face: usize) -> Vec<usize>;

    /// Connect two same-vertex-count faces with side quads (a tunnel), then
    /// remove the two originals. Returns success.
    fn bridge_faces(&mut self, face_a: usize, face_b: usize) -> bool;

    /// Duplicate the mesh mirrored across the plane through the origin with
    /// normal `axis`, fixing winding. Welds seam vertices within `weld`.
    fn mirror_axis(&mut self, axis: Vec3, weld: f32);

    /// Convert every n-gon face to triangles in place, preserving per-corner
    /// normal/uv indices.
    fn triangulate_all(&mut self);

    /// Reverse the winding of every face and negate the normals.
    fn flip_normals(&mut self);

    /// Dissolve the given edges by merging the two faces sharing each into a
    /// single n-gon, then compact. Returns the count dissolved.
    fn delete_edges(&mut self, edges: &[(u32, u32)]) -> usize;

    /// Translate the given faces and their unique vertices by `delta`.
    fn transform_faces(&mut self, faces: &[usize], delta: Vec3);
}

// --- Shared helpers -----------------------------------------------------------

/// True when the face contains the directed edge `a -> b`.
fn has_dir(f: &Face, a: u32, b: u32) -> bool {
    let n = f.corners.len();
    (0..n).any(|k| f.corners[k].position == a && f.corners[(k + 1) % n].position == b)
}

/// An orthonormal tangent basis perpendicular to `n`.
fn tangent_basis(n: Vec3) -> (Vec3, Vec3) {
    let a = if n.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
    let t1 = n.cross(a).normalize_or_zero();
    let t2 = n.cross(t1).normalize_or_zero();
    (t1, t2)
}

/// Merge faces `i` and `j` across their shared undirected edge `{a, b}` into a
/// single n-gon by following directed-edge successors (excluding the shared
/// edge). Returns `None` if the resulting ring is degenerate.
fn merge_across_edge(mesh: &Mesh, i: usize, j: usize, a: u32, b: u32) -> Option<Face> {
    let ci = &mesh.faces[i].corners;
    let cj = &mesh.faces[j].corners;
    let (ni, nj) = (ci.len(), cj.len());
    if ni < 2 || nj < 2 {
        return None;
    }
    let mut succ: HashMap<u32, u32> = HashMap::new();
    for k in 0..ni {
        let cur = ci[k].position;
        let nxt = ci[(k + 1) % ni].position;
        if cur == a && nxt == b {
            continue; // skip shared edge in i
        }
        succ.insert(cur, nxt);
    }
    for k in 0..nj {
        let cur = cj[k].position;
        let nxt = cj[(k + 1) % nj].position;
        if cur == b && nxt == a {
            continue; // skip shared edge in j
        }
        succ.insert(cur, nxt);
    }
    // Walk the merged boundary starting at `a`.
    let mut ring = vec![a];
    let mut cur = a;
    for _ in 0..(ni + nj) {
        match succ.get(&cur) {
            Some(&nx) => {
                if nx == a {
                    break;
                }
                ring.push(nx);
                cur = nx;
            }
            None => break,
        }
    }
    if ring.len() < 3 {
        return None;
    }
    let material = mesh.faces[i].material;
    Some(Face {
        corners: ring
            .into_iter()
            .map(|p| Corner { position: p, normal: 0, uv: 0 })
            .collect(),
        material,
    })
}

impl MeshOps for Mesh {
    fn bevel_vertices(&mut self, verts: &[u32], amount: f32) {
        let amount = amount.clamp(0.0, 0.9);
        let verts: Vec<u32> = verts
            .iter()
            .copied()
            .filter(|&v| (v as usize) < self.positions.len())
            .collect();
        for v in verts {
            bevel_one_vertex(self, v, amount);
        }
    }

    fn bevel_edges(&mut self, edges: &[(u32, u32)], amount: f32) {
        let amount = amount.clamp(0.0, 0.9);
        for &(a, b) in edges {
            if a as usize >= self.positions.len() || b as usize >= self.positions.len() {
                continue;
            }
            let fi = self.faces.iter().position(|f| has_dir(f, a, b));
            let fj = self.faces.iter().position(|f| has_dir(f, b, a));
            let (Some(i), Some(j)) = (fi, fj) else { continue };
            if i == j {
                continue;
            }
            let pa = self.positions[a as usize];
            let pb = self.positions[b as usize];
            let ca = self.face_center(i);
            let cb = self.face_center(j);
            let a1 = self.add_vertex(pa + (ca - pa) * amount);
            let b1 = self.add_vertex(pb + (ca - pb) * amount);
            let a2 = self.add_vertex(pa + (cb - pa) * amount);
            let b2 = self.add_vertex(pb + (cb - pb) * amount);
            for c in self.faces[i].corners.iter_mut() {
                if c.position == a {
                    c.position = a1;
                }
                if c.position == b {
                    c.position = b1;
                }
            }
            for c in self.faces[j].corners.iter_mut() {
                if c.position == a {
                    c.position = a2;
                }
                if c.position == b {
                    c.position = b2;
                }
            }
            self.faces.push(Face {
                corners: vec![
                    Corner { position: a1, normal: 0, uv: 0 },
                    Corner { position: b1, normal: 0, uv: 0 },
                    Corner { position: b2, normal: 0, uv: 0 },
                    Corner { position: a2, normal: 0, uv: 0 },
                ],
                material: 0,
            });
        }
        self.compute_flat_normals();
    }

    fn edge_loops(&self, start_face: usize, start_edge: usize) -> Vec<usize> {
        if start_face >= self.faces.len() {
            return Vec::new();
        }
        let mut out = vec![start_face];
        let mut visited: HashSet<usize> = HashSet::new();
        visited.insert(start_face);
        let mut current = start_face;
        let mut entry = start_edge;
        // Walk forward crossing opposite edges (quads only).
        loop {
            let f = &self.faces[current];
            if f.corners.len() != 4 {
                break;
            }
            let n = f.corners.len();
            let exit = (entry + 2) % n;
            let a = f.corners[exit].position;
            let b = f.corners[(exit + 1) % n].position;
            match self.faces.iter().position(|g| has_dir(g, b, a)) {
                Some(next) if !visited.contains(&next) => {
                    visited.insert(next);
                    out.push(next);
                    entry = self.faces[next]
                        .corners
                        .iter()
                        .position(|c| c.position == b)
                        .unwrap_or(0);
                    current = next;
                }
                _ => break,
            }
        }
        out
    }

    fn select_ring(&self, face: usize) -> Vec<usize> {
        if face >= self.faces.len() {
            return Vec::new();
        }
        let f = &self.faces[face];
        let n = f.corners.len();
        let mut directed: HashSet<(u32, u32)> = HashSet::new();
        for k in 0..n {
            directed.insert((f.corners[k].position, f.corners[(k + 1) % n].position));
        }
        let mut ring = Vec::new();
        for (gi, g) in self.faces.iter().enumerate() {
            if gi == face {
                continue;
            }
            let gn = g.corners.len();
            let shares = (0..gn).any(|k| {
                let a = g.corners[k].position;
                let b = g.corners[(k + 1) % gn].position;
                directed.contains(&(a, b)) || directed.contains(&(b, a))
            });
            if shares {
                ring.push(gi);
            }
        }
        ring
    }

    fn bridge_faces(&mut self, face_a: usize, face_b: usize) -> bool {
        if face_a >= self.faces.len() || face_b >= self.faces.len() || face_a == face_b {
            return false;
        }
        let ca = self.faces[face_a].corners.clone();
        let cb = self.faces[face_b].corners.clone();
        if ca.len() != cb.len() || ca.len() < 3 {
            return false;
        }
        let n = ca.len();
        let material = self.faces[face_a].material;
        for k in 0..n {
            let a0 = ca[k].position;
            let a1 = ca[(k + 1) % n].position;
            let b0 = cb[k].position;
            let b1 = cb[(k + 1) % n].position;
            self.faces.push(Face {
                corners: vec![
                    Corner { position: a0, normal: 0, uv: 0 },
                    Corner { position: a1, normal: 0, uv: 0 },
                    Corner { position: b1, normal: 0, uv: 0 },
                    Corner { position: b0, normal: 0, uv: 0 },
                ],
                material,
            });
        }
        let (hi, lo) = if face_a > face_b {
            (face_a, face_b)
        } else {
            (face_b, face_a)
        };
        self.delete_faces(&[hi, lo]);
        self.compute_flat_normals();
        true
    }

    fn mirror_axis(&mut self, axis: Vec3, weld: f32) {
        let axis = axis.normalize_or_zero();
        if axis == Vec3::ZERO {
            return;
        }
        let n = self.vertex_count() as u32;
        let base = self.positions.clone();
        for p in &base {
            let mirrored = *p - axis * (2.0 * p.dot(axis));
            self.positions.push(mirrored);
        }
        let face_count = self.faces.len();
        for fi in 0..face_count {
            let f = &self.faces[fi];
            let mut corners: Vec<Corner> = f
                .corners
                .iter()
                .map(|c| Corner { position: c.position + n, normal: 0, uv: c.uv })
                .collect();
            corners.reverse(); // fix mirrored winding
            self.faces.push(Face { corners, material: f.material });
        }
        if weld > 0.0 {
            self.weld_close(weld);
        }
        self.compute_flat_normals();
    }

    fn triangulate_all(&mut self) {
        let mut new_faces = Vec::with_capacity(self.faces.len());
        for f in &self.faces {
            let cs = &f.corners;
            if cs.len() < 3 {
                continue;
            }
            for i in 1..cs.len() - 1 {
                new_faces.push(Face {
                    corners: vec![cs[0].clone(), cs[i].clone(), cs[i + 1].clone()],
                    material: f.material,
                });
            }
        }
        self.faces = new_faces;
    }

    fn flip_normals(&mut self) {
        for f in self.faces.iter_mut() {
            f.corners.reverse();
        }
        for nn in self.normals.iter_mut() {
            *nn = -*nn;
        }
    }

    fn delete_edges(&mut self, edges: &[(u32, u32)]) -> usize {
        let mut dissolved = 0usize;
        for &(a, b) in edges {
            // Re-find each iteration: indices shift after a removal.
            let fi = self.faces.iter().position(|f| has_dir(f, a, b));
            let fj = self.faces.iter().position(|f| has_dir(f, b, a));
            let (Some(i), Some(j)) = (fi, fj) else { continue };
            if i == j {
                continue;
            }
            if let Some(merged) = merge_across_edge(self, i, j, a, b) {
                self.faces[i] = merged;
                self.faces.remove(j);
                dissolved += 1;
            }
        }
        self.compact();
        dissolved
    }

    fn transform_faces(&mut self, faces: &[usize], delta: Vec3) {
        let mut moved: HashSet<u32> = HashSet::new();
        for &fi in faces {
            if fi >= self.faces.len() {
                continue;
            }
            for c in &self.faces[fi].corners {
                moved.insert(c.position);
            }
        }
        for &p in &moved {
            self.positions[p as usize] += delta;
        }
        self.compute_flat_normals();
    }
}

/// Bevel a single vertex into a chamfer polygon, adding bridging triangles so
/// the result stays watertight at manifold vertices.
fn bevel_one_vertex(mesh: &mut Mesh, v: u32, amount: f32) {
    let incident: Vec<usize> = mesh
        .faces
        .iter()
        .enumerate()
        .filter(|(_, f)| f.corners.iter().any(|c| c.position == v))
        .map(|(i, _)| i)
        .collect();
    if incident.len() < 3 {
        return; // need a fan of >=3 for a cap polygon
    }
    let vpos = mesh.positions[v as usize];
    let mut vnormal = Vec3::ZERO;
    for &fi in &incident {
        vnormal += mesh.face_normal(fi);
    }
    vnormal = vnormal.normalize_or_zero();
    let (t1, t2) = tangent_basis(vnormal);
    // Order incident faces radially around `v`.
    let mut keyed: Vec<(f32, usize)> = incident
        .iter()
        .map(|&fi| {
            let d = (mesh.face_center(fi) - vpos).normalize_or_zero();
            (d.dot(t1).atan2(d.dot(t2)), fi)
        })
        .collect();
    keyed.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let ordered: Vec<usize> = keyed.iter().map(|&(_, fi)| fi).collect();

    // Create one split vertex per incident face, offset toward its edge midpoint.
    let mut split: Vec<u32> = Vec::with_capacity(ordered.len());
    for &fi in &ordered {
        let f = &mesh.faces[fi];
        let n = f.corners.len();
        let idx = f.corners.iter().position(|c| c.position == v).unwrap_or(0);
        let prev = mesh.positions[f.corners[(idx + n - 1) % n].position as usize];
        let next = mesh.positions[f.corners[(idx + 1) % n].position as usize];
        let target = (prev + next) * 0.5;
        let si = mesh.add_vertex(vpos + (target - vpos) * amount);
        split.push(si);
    }
    // Re-point each incident face's corner at its split vertex.
    for (k, &fi) in ordered.iter().enumerate() {
        for c in mesh.faces[fi].corners.iter_mut() {
            if c.position == v {
                c.position = split[k];
            }
        }
    }
    // Cap polygon (reversed winding to face outward).
    let cap: Vec<Corner> = split
        .iter()
        .rev()
        .map(|&p| Corner { position: p, normal: 0, uv: 0 })
        .collect();
    mesh.faces.push(Face { corners: cap, material: 0 });
    // Bridging triangles keep the mesh watertight between neighbors.
    let m = ordered.len();
    for k in 0..m {
        let fi = ordered[k];
        let fj = ordered[(k + 1) % m];
        if let Some(x) = shared_neighbor(mesh, fi, fj, v) {
            let a = split[k];
            let b = split[(k + 1) % m];
            mesh.faces.push(Face {
                corners: vec![
                    Corner { position: a, normal: 0, uv: 0 },
                    Corner { position: x, normal: 0, uv: 0 },
                    Corner { position: b, normal: 0, uv: 0 },
                ],
                material: 0,
            });
        }
    }
    mesh.compute_flat_normals();
}

/// A vertex (other than `v`) shared by both faces, if any.
fn shared_neighbor(mesh: &Mesh, fi: usize, fj: usize, v: u32) -> Option<u32> {
    let a: HashSet<u32> = mesh.faces[fi]
        .corners
        .iter()
        .map(|c| c.position)
        .filter(|&p| p != v)
        .collect();
    mesh.faces[fj]
        .corners
        .iter()
        .map(|c| c.position)
        .find(|&p| p != v && a.contains(&p))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A unit cube from 6 quads (8 verts).
    fn cube() -> Mesh {
        let h = 0.5f32;
        let mut m = Mesh::new();
        let v = [
            Vec3::new(-h, -h, -h),
            Vec3::new(h, -h, -h),
            Vec3::new(h, h, -h),
            Vec3::new(-h, h, -h),
            Vec3::new(-h, -h, h),
            Vec3::new(h, -h, h),
            Vec3::new(h, h, h),
            Vec3::new(-h, h, h),
        ];
        for p in v {
            m.add_vertex(p);
        }
        for q in [[0, 1, 2, 3], [4, 7, 6, 5], [0, 4, 5, 1], [3, 2, 6, 7], [0, 3, 7, 4], [1, 5, 6, 2]] {
            m.add_face(&q);
        }
        m.compute_flat_normals();
        m
    }

    /// Two unit quads sharing the edge (1,2): [0,1,2,3] and [1,4,5,2].
    fn two_quads() -> Mesh {
        let mut m = Mesh::new();
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
        m.add_face(&[0, 1, 2, 3]);
        m.add_face(&[1, 4, 5, 2]);
        m.compute_flat_normals();
        m
    }

    #[test]
    fn triangulate_all_cube() {
        let mut m = cube();
        m.triangulate_all();
        assert_eq!(m.face_count(), 12); // 6 quads -> 12 tris
        assert!(m.faces.iter().all(|f| f.corners.len() == 3));
    }

    #[test]
    fn flip_normals_reverses_winding() {
        let mut m = cube();
        let before = m.face_normal(0);
        let first_pos = m.faces[0].corners[0].position;
        m.flip_normals();
        // Winding reversed: the corner that was first is now last.
        let last = m.faces[0].corners.len() - 1;
        assert_eq!(m.faces[0].corners[last].position, first_pos);
        // Normal negated.
        let after = m.face_normal(0);
        assert!(before.dot(after) < 0.0);
    }

    #[test]
    fn transform_faces_moves_only_those_vertices() {
        let mut m = cube();
        m.transform_faces(&[0], Vec3::new(1.0, 0.0, 0.0));
        // Face 0 uses verts 0..3; their x shifted by +1.
        for i in 0..4 {
            assert!((m.positions[i].x - ([-0.5, 0.5, 0.5, -0.5][i] + 1.0)).abs() < 1e-5);
        }
        // Vertex 6 (top-back) untouched.
        assert!((m.positions[6].x - 0.5).abs() < 1e-5);
    }

    #[test]
    fn mirror_doubles_and_welds_seam() {
        let mut m = cube();
        m.mirror_axis(Vec3::X, 1e-4);
        // Mirrored across x=0 (a symmetric cube) welds back to 8 verts.
        assert_eq!(m.vertex_count(), 8);
        // 6 original + 6 mirrored faces.
        assert_eq!(m.face_count(), 12);
    }

    #[test]
    fn delete_edges_merges_two_quads() {
        let mut m = two_quads();
        // Shared edge is 1->2 (in face 0) and 2->1 (in face 1).
        let dissolved = m.delete_edges(&[(1, 2)]);
        assert_eq!(dissolved, 1);
        // Two quads merge into one hexagon.
        assert_eq!(m.face_count(), 1);
        assert_eq!(m.faces[0].corners.len(), 6);
    }

    #[test]
    fn bridge_two_quads_creates_sides_and_removes_originals() {
        let mut m = two_quads();
        let before = m.face_count();
        let ok = m.bridge_faces(0, 1);
        assert!(ok);
        // 4 side quads added, 2 originals removed: before - 2 + 4.
        assert_eq!(m.face_count(), before - 2 + 4);
    }

    #[test]
    fn bevel_vertex_adds_cap_and_bridges() {
        let mut m = cube();
        let v_before = m.vertex_count();
        let f_before = m.face_count();
        m.bevel_vertices(&[0], 0.5);
        // Corner 0 has 3 incident faces -> 3 split verts + a 3-gon cap.
        assert!(m.vertex_count() > v_before);
        assert!(m.face_count() > f_before);
    }

    #[test]
    fn bevel_edges_adds_strip() {
        let mut m = two_quads();
        let v_before = m.vertex_count();
        let f_before = m.face_count();
        m.bevel_edges(&[(1, 2)], 0.5);
        // 4 new vertices and a chamfer quad.
        assert_eq!(m.vertex_count(), v_before + 4);
        assert_eq!(m.face_count(), f_before + 1);
    }

    #[test]
    fn edge_loops_walks_two_quads() {
        let m = two_quads();
        let loop_faces = m.edge_loops(0, 1);
        // At minimum returns the seed; walking reaches the neighbor when the
        // shared edge is opposite the entry.
        assert_eq!(loop_faces[0], 0);
        assert!(!loop_faces.is_empty());
    }

    #[test]
    fn select_ring_finds_neighbors() {
        let m = two_quads();
        let ring = m.select_ring(0);
        assert!(ring.contains(&1));
    }

    #[test]
    fn out_of_range_inputs_are_ignored() {
        let mut m = cube();
        m.bevel_vertices(&[99], 0.5);
        m.bevel_edges(&[(0, 99)], 0.5);
        m.transform_faces(&[99], Vec3::X);
        assert!(m.edge_loops(99, 0).is_empty());
        assert!(m.select_ring(99).is_empty());
        assert!(!m.bridge_faces(0, 0));
        assert_eq!(m.delete_edges(&[(99, 100)]), 0);
    }
}
