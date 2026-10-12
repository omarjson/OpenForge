//! Parametric primitive mesh generators.
//!
//! Every generator returns a [`Mesh`](openforge_core::mesh::Mesh) with
//! outward-consistent winding and smoothed normals, and clamps its segment
//! counts so degenerate input (`0`, `1`, negatives) still yields a valid,
//! non-empty mesh instead of panicking.

use glam::Vec3;
use openforge_core::mesh::Mesh;

const TAU: f32 = std::f32::consts::TAU;

/// Axis-aligned box centered at the origin with the given full side lengths.
pub fn box_mesh(width: f32, height: f32, depth: f32) -> Mesh {
    let (hx, hy, hz) = (width * 0.5, height * 0.5, depth * 0.5);
    let mut m = Mesh::new();
    for p in [
        Vec3::new(-hx, -hy, -hz),
        Vec3::new(hx, -hy, -hz),
        Vec3::new(hx, hy, -hz),
        Vec3::new(-hx, hy, -hz),
        Vec3::new(-hx, -hy, hz),
        Vec3::new(hx, -hy, hz),
        Vec3::new(hx, hy, hz),
        Vec3::new(-hx, hy, hz),
    ] {
        m.add_vertex(p);
    }
    for q in [
        [0, 3, 2, 1], // -Z
        [4, 5, 6, 7], // +Z
        [0, 1, 5, 4], // -Y
        [3, 7, 6, 2], // +Y
        [0, 4, 7, 3], // -X
        [1, 2, 6, 5], // +X
    ] {
        m.add_face(&q);
    }
    m.compute_flat_normals();
    m
}

/// A flat plane on the XZ axis, subdivided into `nx` × `nz` cells.
/// Vertices: `(nx+1)*(nz+1)`; faces: `nx*nz` quads.
pub fn grid(width: f32, depth: f32, nx: u32, nz: u32) -> Mesh {
    let nx = nx.max(1);
    let nz = nz.max(1);
    let mut m = Mesh::new();
    for iz in 0..=nz {
        let z = (iz as f32 / nz as f32 - 0.5) * depth;
        for ix in 0..=nx {
            let x = (ix as f32 / nx as f32 - 0.5) * width;
            m.add_vertex(Vec3::new(x, 0.0, z));
        }
    }
    let stride = nx + 1;
    for iz in 0..nz {
        for ix in 0..nx {
            let a = iz * stride + ix;
            let b = a + 1;
            let c = b + stride;
            let d = a + stride;
            m.add_face(&[a, b, c, d]);
        }
    }
    m.compute_flat_normals();
    m
}

/// A torus in the XZ plane. Vertices/faces: `major_segments * minor_segments`.
pub fn torus(major_radius: f32, minor_radius: f32, major_segments: u32, minor_segments: u32) -> Mesh {
    let major_segments = major_segments.max(3);
    let minor_segments = minor_segments.max(3);
    let mut m = Mesh::new();
    for i in 0..major_segments {
        let u = i as f32 / major_segments as f32 * TAU;
        let (cu, su) = u.sin_cos();
        for j in 0..minor_segments {
            let v = j as f32 / minor_segments as f32 * TAU;
            let (cv, sv) = v.sin_cos();
            let ring = major_radius + minor_radius * cv;
            m.add_vertex(Vec3::new(ring * cu, minor_radius * sv, ring * su));
        }
    }
    let idx = |i: u32, j: u32| i * minor_segments + j;
    for i in 0..major_segments {
        let i2 = (i + 1) % major_segments;
        for j in 0..minor_segments {
            let j2 = (j + 1) % minor_segments;
            m.add_face(&[idx(i, j), idx(i2, j), idx(i2, j2), idx(i, j2)]);
        }
    }
    m.compute_smooth_normals();
    m
}

/// A capped cylinder along the Y axis. `capped` adds flat top/bottom discs.
pub fn cylinder(radius: f32, height: f32, segments: u32, capped: bool) -> Mesh {
    let segments = segments.max(3);
    let (h0, h1) = (-height * 0.5, height * 0.5);
    let mut m = Mesh::new();
    // Side wall: two rings of `segments` vertices.
    for ring_y in [h0, h1] {
        for i in 0..segments {
            let a = i as f32 / segments as f32 * TAU;
            let (s, c) = a.sin_cos();
            m.add_vertex(Vec3::new(radius * c, ring_y, radius * s));
        }
    }
    for i in 0..segments {
        let j = (i + 1) % segments;
        let b0 = i;
        let b1 = segments + i;
        let t0 = j;
        let t1 = segments + j;
        m.add_face(&[b0, t0, t1, b1]);
    }
    if capped {
        let bottom = m.add_vertex(Vec3::new(0.0, h0, 0.0));
        let top = m.add_vertex(Vec3::new(0.0, h1, 0.0));
        for i in 0..segments {
            let j = (i + 1) % segments;
            m.add_face(&[bottom, j, i]); // bottom faces -Y
            m.add_face(&[top, segments + i, segments + j]); // top faces +Y
        }
    }
    m.compute_flat_normals();
    m
}

/// A cone along the Y axis with a flat base at `-height/2` and apex at
/// `+height/2`.
pub fn cone(radius: f32, height: f32, segments: u32) -> Mesh {
    let segments = segments.max(3);
    let (h0, h1) = (-height * 0.5, height * 0.5);
    let mut m = Mesh::new();
    for i in 0..segments {
        let a = i as f32 / segments as f32 * TAU;
        let (s, c) = a.sin_cos();
        m.add_vertex(Vec3::new(radius * c, h0, radius * s));
    }
    let apex = m.add_vertex(Vec3::new(0.0, h1, 0.0));
    let base = m.add_vertex(Vec3::new(0.0, h0, 0.0));
    for i in 0..segments {
        let j = (i + 1) % segments;
        m.add_face(&[i, j, apex]); // side
        m.add_face(&[base, j, i]); // base faces -Y
    }
    m.compute_flat_normals();
    m
}

/// A capsule: a cylinder of the given `height` capped with hemispheres of
/// `radius`.
pub fn capsule(radius: f32, height: f32, segments: u32, rings: u32) -> Mesh {
    let segments = segments.max(3);
    let rings = rings.max(1);
    let mut m = Mesh::new();
    let half = height * 0.5;
    // latitude from -90° (bottom pole) to +90° (top pole), `2*rings+1` rows.
    let rows = 2 * rings + 1;
    for r in 0..=rows {
        let t = r as f32 / rows as f32; // 0..1
        let lat = (t - 0.5) * std::f32::consts::PI; // -π/2..π/2
        let (sy, cy) = lat.sin_cos();
        // Shift the spherical band so the straight part spans `-half..half`.
        let y = radius * sy + if sy >= 0.0 { half } else { -half };
        let ring_r = radius * cy;
        for i in 0..segments {
            let a = i as f32 / segments as f32 * TAU;
            let (s, c) = a.sin_cos();
            m.add_vertex(Vec3::new(ring_r * c, y, ring_r * s));
        }
    }
    let idx = |r: u32, i: u32| r * segments + (i % segments);
    for r in 0..rows {
        for i in 0..segments {
            let j = (i + 1) % segments;
            m.add_face(&[idx(r, i), idx(r, j), idx(r + 1, j), idx(r + 1, i)]);
        }
    }
    m.compute_smooth_normals();
    m
}

/// A geodesic icosphere: an icosahedron subdivided `subdivisions` times and
/// projected onto a sphere of `radius`.
pub fn ico_sphere(radius: f32, subdivisions: u32) -> Mesh {
    // Icosahedron: 12 vertices, 20 triangular faces.
    let t = (1.0 + 5.0f32.sqrt()) * 0.5;
    let mut verts: Vec<Vec3> = vec![
        Vec3::new(-1.0, t, 0.0),
        Vec3::new(1.0, t, 0.0),
        Vec3::new(-1.0, -t, 0.0),
        Vec3::new(1.0, -t, 0.0),
        Vec3::new(0.0, -1.0, t),
        Vec3::new(0.0, 1.0, t),
        Vec3::new(0.0, -1.0, -t),
        Vec3::new(0.0, 1.0, -t),
        Vec3::new(t, 0.0, -1.0),
        Vec3::new(t, 0.0, 1.0),
        Vec3::new(-t, 0.0, -1.0),
        Vec3::new(-t, 0.0, 1.0),
    ];
    for v in verts.iter_mut() {
        *v = v.normalize();
    }
    let mut faces: Vec<[u32; 3]> = vec![
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ];

    for _ in 0..subdivisions {
        let mut midpoints: std::collections::HashMap<(u32, u32), u32> =
            std::collections::HashMap::new();
        let mut mid = |a: u32, b: u32, verts: &mut Vec<Vec3>| -> u32 {
            let key = if a < b { (a, b) } else { (b, a) };
            if let Some(&i) = midpoints.get(&key) {
                return i;
            }
            let p = ((verts[a as usize] + verts[b as usize]) * 0.5).normalize();
            let i = verts.len() as u32;
            verts.push(p);
            midpoints.insert(key, i);
            i
        };
        let mut next = Vec::with_capacity(faces.len() * 4);
        for f in &faces {
            let (a, b, c) = (f[0], f[1], f[2]);
            let ab = mid(a, b, &mut verts);
            let bc = mid(b, c, &mut verts);
            let ca = mid(c, a, &mut verts);
            next.push([a, ab, ca]);
            next.push([b, bc, ab]);
            next.push([c, ca, bc]);
            next.push([ab, bc, ca]);
        }
        faces = next;
    }

    let mut m = Mesh::new();
    for v in &verts {
        m.add_vertex(*v * radius);
    }
    for f in &faces {
        m.add_face(f);
    }
    m.compute_smooth_normals();
    m
}

/// A superquadric: a sphere-like solid controlled by shape exponents `e1`
/// (polar) and `e2` (equatorial). Values < 1 make it boxier, > 1 pointier.
pub fn superquadric(e1: f32, e2: f32, radius: f32, segments: u32, rings: u32) -> Mesh {
    let segments = segments.max(3);
    let rings = rings.max(2);
    let sc = |x: f32, e: f32| x.signum() * x.abs().powf(e);
    let mut m = Mesh::new();
    for r in 0..=rings {
        let v = r as f32 / rings as f32; // 0..1
        let phi = (v - 0.5) * std::f32::consts::PI; // -π/2..π/2
        let (sy, cy) = phi.sin_cos();
        let y = radius * sc(sy, e1);
        let rr = radius * sc(cy, e1); // horizontal radius factor
        for i in 0..=segments {
            let u = i as f32 / segments as f32 * TAU;
            let (su, cu) = u.sin_cos();
            let x = rr * sc(cu, e2);
            let z = rr * sc(su, e2);
            m.add_vertex(Vec3::new(x, y, z));
        }
    }
    let stride = segments + 1;
    let idx = |r: u32, i: u32| r * stride + i;
    for r in 0..rings {
        for i in 0..segments {
            let (a, b, c, d) = (idx(r, i), idx(r, i + 1), idx(r + 1, i + 1), idx(r + 1, i));
            // Skip zero-area quads where a ring collapsed onto a pole.
            let (pa, pb, pc, pd) = (
                m.positions[a as usize],
                m.positions[b as usize],
                m.positions[c as usize],
                m.positions[d as usize],
            );
            if pa == pb || pb == pc || pc == pd || pd == pa {
                continue;
            }
            m.add_face(&[a, b, c, d]);
        }
    }
    m.compute_smooth_normals();
    m
}

/// A tube of radius `tube_radius` swept along a helix of the given `radius`,
/// `height` and `turns`.
pub fn helix(radius: f32, height: f32, turns: f32, tube_radius: f32, segments: u32, sides: u32) -> Mesh {
    let segments = segments.max(2);
    let sides = sides.max(3);
    // Path samples plus a parallel-transport frame.
    let mut centers: Vec<Vec3> = Vec::with_capacity((segments + 1) as usize);
    for i in 0..=segments {
        let t = i as f32 / segments as f32;
        let ang = t * turns * TAU;
        let (s, c) = ang.sin_cos();
        centers.push(Vec3::new(radius * c, (t - 0.5) * height, radius * s));
    }
    let mut m = Mesh::new();
    let up = Vec3::Y;
    for (k, center) in centers.iter().enumerate() {
        let tangent = if k == 0 {
            centers[1] - centers[0]
        } else if k == centers.len() - 1 {
            centers[k] - centers[k - 1]
        } else {
            centers[k + 1] - centers[k - 1]
        }
        .normalize_or_zero();
        // Build a stable normal/binormal around the tangent.
        let normal = if tangent.dot(up).abs() < 0.99 {
            tangent.cross(up).normalize_or_zero()
        } else {
            tangent.cross(Vec3::X).normalize_or_zero()
        };
        let binormal = tangent.cross(normal).normalize_or_zero();
        for s in 0..sides {
            let a = s as f32 / sides as f32 * TAU;
            let offset = normal * (a.cos() * tube_radius) + binormal * (a.sin() * tube_radius);
            m.add_vertex(*center + offset);
        }
    }
    let idx = |k: u32, s: u32| k * sides + (s % sides);
    for k in 0..segments as u32 {
        for s in 0..sides {
            let j = (s + 1) % sides;
            m.add_face(&[idx(k, s), idx(k, j), idx(k + 1, j), idx(k + 1, s)]);
        }
    }
    m.compute_smooth_normals();
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every face corner position index must resolve.
    fn assert_valid(m: &Mesh) {
        for f in &m.faces {
            for c in &f.corners {
                assert!((c.position as usize) < m.positions.len());
            }
        }
        assert!(!m.faces.is_empty(), "primitive must have faces");
    }

    #[test]
    fn grid_counts_and_plane() {
        let m = grid(2.0, 2.0, 4, 3);
        assert_eq!(m.vertex_count(), 5 * 4); // (nx+1)*(nz+1)
        assert_eq!(m.face_count(), 4 * 3);
        assert!(m.positions.iter().all(|p| p.y == 0.0));
        assert_valid(&m);
    }

    #[test]
    fn torus_counts_and_bounds() {
        let m = torus(2.0, 0.5, 16, 8);
        assert_eq!(m.vertex_count(), 16 * 8);
        assert_eq!(m.face_count(), 16 * 8);
        // Outer extent = major + minor.
        let max_r = m
            .positions
            .iter()
            .map(|p| (p.x * p.x + p.z * p.z).sqrt())
            .fold(0.0f32, f32::max);
        assert!((max_r - 2.5).abs() < 1e-4);
        assert_valid(&m);
    }

    #[test]
    fn cylinder_capped_counts() {
        let open = cylinder(1.0, 2.0, 12, false);
        assert_eq!(open.vertex_count(), 24);
        assert_eq!(open.face_count(), 12);
        let capped = cylinder(1.0, 2.0, 12, true);
        // +2 center vertices, +2*12 cap triangles.
        assert_eq!(capped.vertex_count(), 26);
        assert_eq!(capped.face_count(), 12 + 24);
        assert_valid(&capped);
    }

    #[test]
    fn cone_has_apex_and_base() {
        let m = cone(1.0, 2.0, 10);
        assert_eq!(m.vertex_count(), 12); // 10 rim + apex + base center
        assert_eq!(m.face_count(), 20); // 10 side + 10 base
        assert_valid(&m);
    }

    #[test]
    fn capsule_poles_and_counts() {
        let m = capsule(1.0, 2.0, 12, 3);
        // rows = 2*rings+1 bands, each `segments` vertices.
        assert_eq!(m.vertex_count(), (2 * 3 + 1 + 1) * 12);
        assert!(m
            .positions
            .iter()
            .all(|p| (p.x * p.x + p.y * p.y + p.z * p.z).sqrt() <= 1.0 + 1.0 + 1e-4));
        assert_valid(&m);
    }

    #[test]
    fn ico_sphere_subdivides_and_stays_on_sphere() {
        let m = ico_sphere(1.0, 1);
        // Level 1: 20 tris -> 80 tris; vertices = 12 + edges.
        assert_eq!(m.face_count(), 80);
        for p in &m.positions {
            assert!((p.length() - 1.0).abs() < 1e-4, "not on sphere: {p:?}");
        }
        assert_valid(&m);
    }

    #[test]
    fn superquadric_is_nonempty_and_bounded() {
        let m = superquadric(1.0, 1.0, 1.0, 16, 12);
        assert!(!m.faces.is_empty());
        for p in &m.positions {
            assert!(p.x.abs() <= 1.0 + 1e-4 && p.y.abs() <= 1.0 + 1e-4 && p.z.abs() <= 1.0 + 1e-4);
        }
        assert_valid(&m);
    }

    #[test]
    fn helix_tube_is_nonempty() {
        let m = helix(1.0, 2.0, 3.0, 0.2, 32, 8);
        assert_eq!(m.vertex_count(), 33 * 8);
        assert!(!m.faces.is_empty());
        assert_valid(&m);
    }

    #[test]
    fn degenerate_inputs_do_not_panic() {
        let _ = grid(1.0, 1.0, 0, 0);
        let _ = torus(1.0, 0.2, 0, 0);
        let _ = cylinder(1.0, 1.0, 1, true);
        let _ = cone(1.0, 1.0, 0);
        let _ = capsule(1.0, 1.0, 0, 0);
        let _ = ico_sphere(1.0, 0);
        let _ = superquadric(0.5, 0.5, 1.0, 0, 0);
        let _ = helix(1.0, 1.0, 1.0, 0.1, 0, 0);
    }
}
