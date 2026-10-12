//! Catmull-Clark subdivision surfaces.
//!
//! [`catmull_clark_level`] performs one refinement pass over a polygonal
//! mesh: every face point, edge point, and vertex point is computed with the
//! classic Catmull-Clark rules, and each `k`-gon face is replaced by `k`
//! quads fanning around its face point. The result is exact for closed
//! manifold meshes (e.g. a cube goes `8 verts / 6 quads` → `26 verts / 24
//! quads`); meshes with boundaries use midpoint rules along the boundary and
//! are slightly less smooth there.

use glam::Vec3;
use openforge_core::mesh::{Corner, Face, Mesh};
use std::collections::HashMap;

/// Sorted (undirected) edge key.
fn edge_key(a: u32, b: u32) -> (u32, u32) {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

/// One level of Catmull-Clark subdivision.
pub fn catmull_clark_level(mesh: &Mesh) -> Mesh {
    let nfaces = mesh.faces.len();
    if nfaces == 0 || mesh.positions.is_empty() {
        return Mesh::new();
    }

    // Face points: centroid of each face's corner positions.
    let mut face_point: Vec<Vec3> = Vec::with_capacity(nfaces);
    for f in &mesh.faces {
        let mut sum = Vec3::ZERO;
        let mut n = 0u32;
        for c in &f.corners {
            if let Some(p) = mesh.positions.get(c.position as usize) {
                sum += *p;
                n += 1;
            }
        }
        face_point.push(if n > 0 { sum / n as f32 } else { Vec3::ZERO });
    }

    // Topology: edge -> adjacent faces; vertex -> incident edges and faces.
    let mut edges: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
    let mut vertex_edges: HashMap<u32, Vec<(u32, u32)>> = HashMap::new();
    let mut vertex_faces: HashMap<u32, Vec<usize>> = HashMap::new();
    for (fi, f) in mesh.faces.iter().enumerate() {
        let cs = &f.corners;
        let n = cs.len();
        for k in 0..n {
            let a = cs[k].position;
            let b = cs[(k + 1) % n].position;
            if a == b {
                continue;
            }
            let key = edge_key(a, b);
            edges.entry(key).or_default().push(fi);
            vertex_edges.entry(a).or_default().push(key);
            vertex_edges.entry(b).or_default().push(key);
        }
        for c in cs {
            vertex_faces.entry(c.position).or_default().push(fi);
        }
    }

    // Edge points: interior (2 faces) = (Pa+Pb+F1+F2)/4; boundary = midpoint.
    let mut edge_point: HashMap<(u32, u32), Vec3> = HashMap::new();
    for (&key, faces) in &edges {
        let (a, b) = key;
        let pa = mesh.positions[a as usize];
        let pb = mesh.positions[b as usize];
        let ep = if faces.len() >= 2 {
            (pa + pb + face_point[faces[0]] + face_point[faces[1]]) * 0.25
        } else {
            (pa + pb) * 0.5
        };
        edge_point.insert(key, ep);
    }

    // Vertex points: (F + 2R + (n-2)P) / n, with F = mean adjacent face
    // point, R = mean adjacent edge midpoint, n = valence.
    let mut used: Vec<u32> = vertex_edges.keys().copied().collect();
    used.sort_unstable();
    let mut vertex_point: HashMap<u32, Vec3> = HashMap::new();
    for &v in &used {
        let p = mesh.positions[v as usize];
        let ffaces = vertex_faces.get(&v).map(|v| v.as_slice()).unwrap_or(&[]);
        let mut fsum = Vec3::ZERO;
        for &fi in ffaces {
            fsum += face_point[fi];
        }
        let f_avg = if !ffaces.is_empty() {
            fsum / ffaces.len() as f32
        } else {
            p
        };
        let ekeys = vertex_edges.get(&v).map(|v| v.as_slice()).unwrap_or(&[]);
        let mut rsum = Vec3::ZERO;
        for &ek in ekeys {
            let (x, y) = ek;
            let other = if x == v { y } else { x };
            rsum += (p + mesh.positions[other as usize]) * 0.5;
        }
        let r_avg = if !ekeys.is_empty() {
            rsum / ekeys.len() as f32
        } else {
            p
        };
        let n = ekeys.len() as f32;
        let newp = if n >= 3.0 {
            (f_avg + r_avg * 2.0 + p * (n - 2.0)) / n
        } else {
            (p + f_avg) * 0.5 // low-valence fallback
        };
        vertex_point.insert(v, newp);
    }

    // Output layout: vertex points, then edge points, then face points.
    let mut out = Mesh::new();
    let mut vidx: HashMap<u32, u32> = HashMap::new();
    for &v in &used {
        let i = out.add_vertex(vertex_point[&v]);
        vidx.insert(v, i);
    }
    let mut eidx: HashMap<(u32, u32), u32> = HashMap::new();
    for &key in edges.keys() {
        let i = out.add_vertex(edge_point[&key]);
        eidx.insert(key, i);
    }
    let mut fidx: Vec<u32> = Vec::with_capacity(nfaces);
    for fp in &face_point {
        fidx.push(out.add_vertex(*fp));
    }

    // Each k-gon face becomes k quads: for corner i, the quad is
    // (face_point, edge_point[i-1], vertex_point[i], edge_point[i]).
    for (fi, f) in mesh.faces.iter().enumerate() {
        let cs = &f.corners;
        let n = cs.len();
        if n < 3 {
            continue;
        }
        let fp = fidx[fi];
        for i in 0..n {
            let vi = cs[i].position;
            let prev = cs[(i + n - 1) % n].position;
            let next = cs[(i + 1) % n].position;
            let (Some(&vp), Some(&ep), Some(&en)) = (
                vidx.get(&vi),
                eidx.get(&edge_key(prev, vi)),
                eidx.get(&edge_key(vi, next)),
            ) else {
                continue;
            };
            out.faces.push(Face {
                corners: vec![
                    Corner { position: fp, normal: 0, uv: 0 },
                    Corner { position: ep, normal: 0, uv: 0 },
                    Corner { position: vp, normal: 0, uv: 0 },
                    Corner { position: en, normal: 0, uv: 0 },
                ],
                material: f.material,
            });
        }
    }

    out.compute_smooth_normals();
    out
}

/// Apply `levels` rounds of Catmull-Clark subdivision.
pub fn catmull_clark(mesh: &Mesh, levels: u32) -> Mesh {
    let mut cur = mesh.clone();
    for _ in 0..levels {
        cur = catmull_clark_level(&cur);
    }
    cur
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::box_mesh;

    #[test]
    fn cube_level1_vertex_and_face_counts() {
        let cube = box_mesh(2.0, 2.0, 2.0);
        assert_eq!(cube.vertex_count(), 8);
        assert_eq!(cube.face_count(), 6);
        let s = catmull_clark_level(&cube);
        // V + E + F = 8 + 12 + 6 = 26 vertices; sum of valences = 24 quads.
        assert_eq!(s.vertex_count(), 26);
        assert_eq!(s.face_count(), 24);
        // Every new face is a quad.
        assert!(s.faces.iter().all(|f| f.corners.len() == 4));
    }

    #[test]
    fn cube_level2_counts() {
        let cube = box_mesh(2.0, 2.0, 2.0);
        let s = catmull_clark(&cube, 2);
        // Level 1: V=26 F=24 E=48; level 2: V = 26+48+24 = 98, F = 96.
        assert_eq!(s.vertex_count(), 98);
        assert_eq!(s.face_count(), 96);
    }

    #[test]
    fn subdivision_shrinks_cube_within_original_bounds() {
        let cube = box_mesh(2.0, 2.0, 2.0); // spans -1..1
        let s = catmull_clark_level(&cube);
        // Catmull-Clark is a averaging scheme: the surface stays inside the
        // original convex hull (the -1..1 cube).
        for p in &s.positions {
            assert!(p.x.abs() <= 1.0 + 1e-4 && p.y.abs() <= 1.0 + 1e-4 && p.z.abs() <= 1.0 + 1e-4);
        }
    }

    #[test]
    fn empty_mesh_is_safe() {
        let e = Mesh::new();
        assert_eq!(catmull_clark_level(&e).face_count(), 0);
        assert_eq!(catmull_clark(&e, 3).face_count(), 0);
    }
}
