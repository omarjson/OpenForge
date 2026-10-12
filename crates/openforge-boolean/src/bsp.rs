//! Binary Space Partitioning trees for constructive solid geometry.
//!
//! This is a clean-room Rust implementation of the BSP-based CSG algorithm
//! (build a tree of splitting planes, then clip one tree against another to
//! resolve union / difference / intersection). It is a direct expression of
//! the published algorithm, not a port of any specific library.

use glam::Vec3;
use openforge_core::mesh::Mesh;
use std::collections::HashMap;

const EPSILON: f32 = 1e-5;

/// A plane in the form `normal · x = w`. Points with `normal · x - w > 0` are
/// in front; `< 0` are behind.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Plane {
    pub normal: Vec3,
    pub w: f32,
}

impl Plane {
    /// Build a plane through three counter-clockwise points.
    fn from_points(a: Vec3, b: Vec3, c: Vec3) -> Option<Self> {
        let normal = (b - a).cross(c - a).normalize_or_zero();
        if normal.length_squared() < EPSILON * EPSILON {
            return None; // degenerate / collinear
        }
        Some(Self {
            normal,
            w: normal.dot(a),
        })
    }

    fn flip(&mut self) {
        self.normal = -self.normal;
        self.w = -self.w;
    }

    /// Split a polygon by this plane into four buckets: coplanar-front,
    /// coplanar-back, front, and back.
    #[allow(clippy::too_many_arguments)]
    fn split_polygon(
        &self,
        poly: &Polygon,
        coplanar_front: &mut Vec<Polygon>,
        coplanar_back: &mut Vec<Polygon>,
        front: &mut Vec<Polygon>,
        back: &mut Vec<Polygon>,
    ) {
        // Classify each vertex as FRONT (+), BACK (-), or COPLANAR (0).
        let mut types: Vec<u8> = Vec::with_capacity(poly.vertices.len());
        let mut polygon_type = 0u8;
        for v in &poly.vertices {
            let t = self.normal.dot(*v) - self.w;
            let ty = if t < -EPSILON {
                2 // BACK
            } else if t > EPSILON {
                1 // FRONT
            } else {
                0 // COPLANAR
            };
            polygon_type |= ty;
            types.push(ty);
        }

        match polygon_type {
            0 => {
                // Entirely coplanar: orient by comparing face normals.
                if self.normal.dot(poly.plane.normal) > 0.0 {
                    coplanar_front.push(poly.clone());
                } else {
                    coplanar_back.push(poly.clone());
                }
            }
            1 => front.push(poly.clone()), // entirely in front
            2 => back.push(poly.clone()),  // entirely behind
            _ => {
                // SPANNING: cut along the plane, keeping both pieces.
                let n = poly.vertices.len();
                let mut f: Vec<Vec3> = Vec::new();
                let mut b: Vec<Vec3> = Vec::new();
                for i in 0..n {
                    let j = (i + 1) % n;
                    let vi = poly.vertices[i];
                    let vj = poly.vertices[j];
                    let ti = types[i];
                    let tj = types[j];
                    if ti != 2 {
                        f.push(vi); // not strictly behind → keep on front
                    }
                    if ti != 1 {
                        b.push(vi); // not strictly in front → keep on back
                    }
                    if (ti == 2 && tj == 1) || (ti == 1 && tj == 2) {
                        // Edge crosses the plane: insert the intersection.
                        let si = self.normal.dot(vi) - self.w;
                        let sj = self.normal.dot(vj) - self.w;
                        let t = si / (si - sj);
                        let p = vi + (vj - vi) * t;
                        f.push(p);
                        b.push(p);
                    }
                }
                if f.len() >= 3 {
                    if let Some(p) = Polygon::from_points(f, poly.plane) {
                        front.push(p);
                    }
                }
                if b.len() >= 3 {
                    if let Some(p) = Polygon::from_points(b, poly.plane) {
                        back.push(p);
                    }
                }
            }
        }
    }
}

/// A convex polygon (triangle, quad, or n-gon) plus the plane it lies on.
#[derive(Clone, Debug)]
pub(crate) struct Polygon {
    pub vertices: Vec<Vec3>,
    pub plane: Plane,
}

impl Polygon {
    /// Build a polygon from counter-clockwise vertices on the given plane.
    /// Returns `None` if degenerate.
    fn from_points(vertices: Vec<Vec3>, plane: Plane) -> Option<Self> {
        if vertices.len() < 3 {
            return None;
        }
        Some(Self { vertices, plane })
    }
}

/// A BSP node: a splitting `plane` with `front`/`back` subtrees and the
/// polygons that are coplanar with (or above) this node.
#[derive(Clone, Debug, Default)]
pub(crate) struct Node {
    plane: Option<Plane>,
    front: Option<Box<Node>>,
    back: Option<Box<Node>>,
    polygons: Vec<Polygon>,
}

impl Node {
    pub fn new() -> Self {
        Self::default()
    }

    /// Recursively insert `polygons` into this tree, splitting as needed.
    pub fn build(&mut self, polygons: Vec<Polygon>) {
        if polygons.is_empty() {
            return;
        }
        if self.plane.is_none() {
            self.plane = Some(polygons[0].plane);
        }
        let plane = self.plane.unwrap();
        let mut coplanar_front = Vec::new();
        let mut coplanar_back = Vec::new();
        let mut front = Vec::new();
        let mut back = Vec::new();
        for p in &polygons {
            plane.split_polygon(p, &mut coplanar_front, &mut coplanar_back, &mut front, &mut back);
        }
        self.polygons.extend(coplanar_front);
        let mut back_polys = coplanar_back;
        back_polys.extend(back);
        if !front.is_empty() {
            self.front.get_or_insert_with(|| Box::new(Node::new())).build(front);
        }
        if !back_polys.is_empty() {
            self.back.get_or_insert_with(|| Box::new(Node::new())).build(back_polys);
        }
    }

    /// Remove the parts of `polygons` that fall inside this solid.
    pub fn clip_polygons(&self, polygons: Vec<Polygon>) -> Vec<Polygon> {
        let Some(plane) = self.plane else {
            return polygons; // leaf: nothing to clip against
        };
        let mut coplanar_front = Vec::new();
        let mut coplanar_back = Vec::new();
        let mut front = Vec::new();
        let mut back = Vec::new();
        for p in &polygons {
            plane.split_polygon(p, &mut coplanar_front, &mut coplanar_back, &mut front, &mut back);
        }
        let mut front = {
            let mut v = coplanar_front;
            v.extend(front);
            v
        };
        let mut back = {
            let mut v = coplanar_back;
            v.extend(back);
            v
        };
        if let Some(f) = &self.front {
            front = f.clip_polygons(front);
        }
        if let Some(b) = &self.back {
            back = b.clip_polygons(back);
        } else {
            back.clear(); // no back subtree → everything behind is inside
        }
        front.extend(back);
        front
    }

    /// Clip this tree's polygons against another solid `brush`.
    pub fn clip_to(&mut self, brush: &Node) {
        self.polygons = brush.clip_polygons(std::mem::take(&mut self.polygons));
        if let Some(f) = &mut self.front {
            f.clip_to(brush);
        }
        if let Some(b) = &mut self.back {
            b.clip_to(brush);
        }
    }

    /// Invert the solid (keep the complement), flipping every plane and
    /// polygon and swapping front/back.
    pub fn invert(&mut self) {
        if let Some(p) = &mut self.plane {
            p.flip();
        }
        for poly in &mut self.polygons {
            poly.plane.flip();
            poly.vertices.reverse();
        }
        if let Some(f) = &mut self.front {
            f.invert();
        }
        if let Some(b) = &mut self.back {
            b.invert();
        }
        std::mem::swap(&mut self.front, &mut self.back);
    }

    /// Collect every polygon in this tree.
    pub fn all_polygons(&self) -> Vec<Polygon> {
        let mut out = self.polygons.clone();
        if let Some(f) = &self.front {
            out.extend(f.all_polygons());
        }
        if let Some(b) = &self.back {
            out.extend(b.all_polygons());
        }
        out
    }
}

/// Triangulate a mesh's faces into a flat list of polygons (fan-triangulated,
/// so every polygon is a triangle and therefore convex and planar).
pub(crate) fn mesh_to_polygons(mesh: &Mesh) -> Vec<Polygon> {
    let mut out = Vec::new();
    for f in &mesh.faces {
        let pts: Vec<Vec3> = f
            .corners
            .iter()
            .filter_map(|c| mesh.positions.get(c.position as usize).copied())
            .collect();
        if pts.len() < 3 {
            continue;
        }
        for i in 1..pts.len() - 1 {
            let tri = vec![pts[0], pts[i], pts[i + 1]];
            if let Some(plane) = Plane::from_points(tri[0], tri[1], tri[2]) {
                out.push(Polygon {
                    vertices: tri,
                    plane,
                });
            }
        }
    }
    out
}

/// Insert any vertex that lies in the interior of a polygon edge, so that a
/// neighbour split at that point shares the same edge subdivision instead of
/// leaving a T-junction. Returns the expanded, ordered boundary loop.
fn expand_t_junctions(verts: &[Vec3], all: &[Vec3], tol: f32) -> Vec<Vec3> {
    let tol2 = tol * tol;
    let mut out: Vec<Vec3> = Vec::with_capacity(verts.len());
    let n = verts.len();
    for i in 0..n {
        let a = verts[i];
        let b = verts[(i + 1) % n];
        out.push(a);
        let ab = b - a;
        let l2 = ab.length_squared();
        if l2 < tol2 {
            continue; // degenerate edge
        }
        let mut ts: Vec<f32> = Vec::new();
        for &p in all {
            let t = (p - a).dot(ab) / l2;
            if t <= tol || t >= 1.0 - tol {
                continue; // too close to an endpoint
            }
            let proj = a + ab * t;
            if (p - proj).length() < tol {
                ts.push(t);
            }
        }
        ts.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
        let mut last = -1.0f32;
        for t in ts {
            if t - last > tol {
                out.push(a + ab * t);
                last = t;
            }
        }
    }
    out
}

/// Triangulate a planar, weakly-convex loop (possibly with collinear points
/// inserted along its edges) via ear clipping. Rejects zero-area (collinear)
/// ears so it never recreates a long chord across a split edge. `normal` gives
/// the loop's orientation for the convexity test. Returns index triples into
/// `pos`.
fn ear_clip(loop_idx: &[u32], pos: &[Vec3], normal: Vec3) -> Vec<[u32; 3]> {
    let mut idx: Vec<u32> = loop_idx.to_vec();
    let mut tris: Vec<[u32; 3]> = Vec::new();
    let mut guard = 0usize;
    while idx.len() > 3 && guard < 4096 {
        guard += 1;
        let n = idx.len();
        let mut clipped = false;
        for i in 0..n {
            let ip = idx[(i + n - 1) % n] as usize;
            let ic = idx[i] as usize;
            let inx = idx[(i + 1) % n] as usize;
            let (a, b, c) = (pos[ip], pos[ic], pos[inx]);
            let cross = (b - a).cross(c - a);
            // Skip degenerate (collinear) ears — these lie across a split edge.
            if cross.length() < 1e-9 {
                continue;
            }
            // Convex ear relative to the loop orientation.
            if cross.dot(normal) <= 0.0 {
                continue;
            }
            // Ensure no other loop vertex lies inside this ear.
            let mut occupied = false;
            for &v in &idx {
                let vi = v as usize;
                if vi == ip || vi == ic || vi == inx {
                    continue;
                }
                if point_in_tri(pos[vi], a, b, c, normal) {
                    occupied = true;
                    break;
                }
            }
            if occupied {
                continue;
            }
            tris.push([idx[(i + n - 1) % n], idx[i], idx[(i + 1) % n]]);
            idx.remove(i);
            clipped = true;
            break;
        }
        if !clipped {
            break; // fallback: whatever remains is emitted as a final fan
        }
    }
    if idx.len() == 3 {
        tris.push([idx[0], idx[1], idx[2]]);
    } else if idx.len() > 3 {
        for i in 1..idx.len() - 1 {
            tris.push([idx[0], idx[i], idx[i + 1]]);
        }
    }
    tris
}

/// Is point `p` inside triangle `(a,b,c)` (all in the plane of `normal`),
/// allowing a small tolerance on the edges?
fn point_in_tri(p: Vec3, a: Vec3, b: Vec3, c: Vec3, normal: Vec3) -> bool {
    let d1 = (b - a).cross(p - a).dot(normal);
    let d2 = (c - b).cross(p - b).dot(normal);
    let d3 = (a - c).cross(p - c).dot(normal);
    const T: f32 = 1e-9;
    let has_neg = d1 < -T || d2 < -T || d3 < -T;
    let has_pos = d1 > T || d2 > T || d3 > T;
    !(has_neg && has_pos)
}

/// Rebuild a mesh from polygons. Shared vertices are welded by quantized
/// position, T-junctions along edges are resolved so the surface stays
/// manifold, and each (now weakly-convex, possibly many-sided) polygon is
/// ear-clipped into triangles.
pub(crate) fn polygons_to_mesh(polygons: &[Polygon]) -> Mesh {
    const TOL: f32 = 1e-4;
    // Candidate T-vertices: every vertex across every polygon.
    let all: Vec<Vec3> = polygons.iter().flat_map(|p| p.vertices.iter().copied()).collect();

    let mut mesh = Mesh::new();
    let mut cache: HashMap<(i64, i64, i64), u32> = HashMap::new();
    let quant = |v: Vec3| -> (i64, i64, i64) {
        (
            (v.x * 1e4).round() as i64,
            (v.y * 1e4).round() as i64,
            (v.z * 1e4).round() as i64,
        )
    };
    let add_welded = |v: Vec3, mesh: &mut Mesh, cache: &mut HashMap<(i64, i64, i64), u32>| -> u32 {
        let key = quant(v);
        if let Some(&i) = cache.get(&key) {
            return i;
        }
        let i = mesh.add_vertex(v);
        cache.insert(key, i);
        i
    };

    for poly in polygons {
        if poly.vertices.len() < 3 {
            continue;
        }
        let loop_pts = expand_t_junctions(&poly.vertices, &all, TOL);
        let m = loop_pts.len();
        if m < 3 {
            continue;
        }
        // Weld the loop into indexed vertices.
        let idx: Vec<u32> = loop_pts
            .iter()
            .map(|&v| add_welded(v, &mut mesh, &mut cache))
            .collect();
        // Ear-clip, skipping degenerate triangles, so split edges stay split.
        for tri in ear_clip(&idx, &mesh.positions, poly.plane.normal) {
            let (p0, p1, p2) = (
                mesh.positions[tri[0] as usize],
                mesh.positions[tri[1] as usize],
                mesh.positions[tri[2] as usize],
            );
            if (p1 - p0).cross(p2 - p0).length_squared() < TOL * TOL {
                continue;
            }
            mesh.add_face(&tri);
        }
    }
    mesh.compute_flat_normals();
    mesh
}

/// The three BSP CSG operations. Each takes two polygon lists and returns the
/// resulting polygon list, leaving the inputs unchanged.
pub(crate) fn union(a: &[Polygon], b: &[Polygon]) -> Vec<Polygon> {
    let mut a = Node::from(a);
    let mut b = Node::from(b);
    a.clip_to(&b);
    b.clip_to(&a);
    b.invert();
    b.clip_to(&a);
    b.invert();
    a.build(b.all_polygons());
    a.all_polygons()
}

pub(crate) fn subtract(a: &[Polygon], b: &[Polygon]) -> Vec<Polygon> {
    let mut a = Node::from(a);
    let mut b = Node::from(b);
    a.invert();
    a.clip_to(&b);
    b.clip_to(&a);
    b.invert();
    b.clip_to(&a);
    b.invert();
    a.build(b.all_polygons());
    a.invert();
    a.all_polygons()
}

pub(crate) fn intersect(a: &[Polygon], b: &[Polygon]) -> Vec<Polygon> {
    let mut a = Node::from(a);
    let mut b = Node::from(b);
    a.invert();
    b.clip_to(&a);
    b.invert();
    a.clip_to(&b);
    b.clip_to(&a);
    a.build(b.all_polygons());
    a.invert();
    a.all_polygons()
}

impl From<&[Polygon]> for Node {
    fn from(polys: &[Polygon]) -> Self {
        let mut n = Node::new();
        n.build(polys.to_vec());
        n
    }
}
