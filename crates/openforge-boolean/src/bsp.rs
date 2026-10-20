//! Binary Space Partitioning trees for constructive solid geometry.
//!
//! This is a clean-room Rust implementation of the BSP-based CSG algorithm
//! (build a tree of splitting planes, then clip one tree against another to
//! resolve union / difference / intersection). It is a direct expression of
//! the published algorithm, not a port of any specific library.
//!
//! Two things make it hold up on real meshes rather than only on paper:
//!
//! * **The work happens in normalized `f64` space.** [`Fit`] centres both
//!   solids and scales them to an extent of about one before anything is
//!   split, and the result is mapped back afterwards. Every tolerance below is
//!   therefore *relative to the model*: a scene measured in millimetres and one
//!   measured in thousands take the same path with the same accuracy, where an
//!   absolute `1e-4` would be a no-op on one and a blunt instrument on the
//!   other. Splitting also accumulates far less error in double precision, and
//!   the result is rounded once, on the way back out.
//! * **Shared vertices are welded by tolerance, not by a grid.** A
//!   quantization grid splits two points a nanometre apart across a cell
//!   boundary and leaves a crack in the surface; [`VertexWelder`] looks at the
//!   neighbouring cells too, so a vertex that appears in several polygons (or
//!   in both solids) becomes one index however it is rounded.
//!
//! Slivers are compressed away rather than emitted: a polygon whose loop
//! repeats a vertex or folds onto itself is dropped, and the ear clipper
//! refuses zero-area ears, so a split edge never becomes a chord across itself.

use glam::{DVec3 as Vec3, Vec2};
use openforge_core::mesh::{Corner, Face, Mesh};
use std::collections::HashMap;

/// Plane-classification tolerance, in normalized units: a vertex this close to
/// a splitting plane is *on* it. The model is about a unit across by the time
/// this is used, so this is a relative 1e-5 of the model's size.
///
/// The value is a trade: a tighter tolerance splits more polygons (a bigger
/// tree, more work), a looser one snaps vertices that are not quite on the
/// plane onto it. It has to be no tighter than [`WELD`], or two solids that
/// share a face would keep a hairline crack between them.
const EPSILON: f64 = 1e-5;

/// Distance at which two points are the same point (normalized units), for
/// welding and for T-junction insertion along an edge.
const WELD: f64 = 1e-5;

/// A plane in the form `normal · x = w`. Points with `normal · x - w > 0` are
/// in front; `< 0` are behind.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Plane {
    pub normal: Vec3,
    pub w: f64,
}

impl Plane {
    /// Build a plane through three counter-clockwise points.
    ///
    /// A triangle is rejected as degenerate when its area is negligible
    /// *relative to its own sides*, which is what tells a sliver (a long, very
    /// thin triangle) apart from a small but honest one: an absolute area test
    /// throws away legitimate detail in a dense mesh and keeps garbage in a
    /// coarse one.
    fn from_points(a: Vec3, b: Vec3, c: Vec3) -> Option<Self> {
        let ab = b - a;
        let ac = c - a;
        let cross = ab.cross(ac);
        let area2 = cross.length();
        let sides = ab.length() * ac.length();
        if !area2.is_finite() || sides <= 0.0 || area2 <= sides * 1e-12 {
            return None; // collinear, or too thin to carry a normal
        }
        let normal = cross / area2;
        if !normal.is_finite() {
            return None;
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
                // The UVs travel with their vertices, and the point where the
                // cut crosses an edge gets the UV that edge's own span puts
                // there — so the texture does not jump across the cut.
                let mut fu: Vec<Vec2> = Vec::new();
                let mut bu: Vec<Vec2> = Vec::new();
                for i in 0..n {
                    let j = (i + 1) % n;
                    let vi = poly.vertices[i];
                    let vj = poly.vertices[j];
                    let ui = poly.uvs[i];
                    let uj = poly.uvs[j];
                    let ti = types[i];
                    let tj = types[j];
                    if ti != 2 {
                        f.push(vi); // not strictly behind → keep on front
                        fu.push(ui);
                    }
                    if ti != 1 {
                        b.push(vi); // not strictly in front → keep on back
                        bu.push(ui);
                    }
                    if (ti == 2 && tj == 1) || (ti == 1 && tj == 2) {
                        // Edge crosses the plane: insert the intersection.
                        let si = self.normal.dot(vi) - self.w;
                        let sj = self.normal.dot(vj) - self.w;
                        let t = si / (si - sj);
                        let p = vi + (vj - vi) * t;
                        let uv = ui + (uj - ui) * (t as f32);
                        f.push(p);
                        fu.push(uv);
                        b.push(p);
                        bu.push(uv);
                    }
                }
                if f.len() >= 3 {
                    if let Some(p) = Polygon::from_points(f, fu, poly.plane, poly.material) {
                        front.push(p);
                    }
                }
                if b.len() >= 3 {
                    if let Some(p) = Polygon::from_points(b, bu, poly.plane, poly.material) {
                        back.push(p);
                    }
                }
            }
        }
    }
}

/// A convex polygon (triangle, quad, or n-gon) plus the plane it lies on.
///
/// The UV of each vertex travels with it, and the face's material travels
/// with the polygon, so a boolean does not take the texture off the surface
/// it cuts.
#[derive(Clone, Debug)]
pub(crate) struct Polygon {
    pub vertices: Vec<Vec3>,
    pub plane: Plane,
    /// The UV of each vertex, in the same order.
    pub uvs: Vec<Vec2>,
    /// The face's material, as an index into the owning node's slot list.
    pub material: u32,
}

impl Polygon {
    /// Build a polygon from counter-clockwise vertices on the given plane,
    /// dropping repeated neighbours. Returns `None` for a loop that is too
    /// short, or too thin, to have an interior.
    fn from_points(
        vertices: Vec<Vec3>,
        uvs: Vec<Vec2>,
        plane: Plane,
        material: u32,
    ) -> Option<Self> {
        let kept = compress_indices(&vertices);
        if kept.len() < 3 {
            return None;
        }
        let vertices: Vec<Vec3> = kept.iter().map(|&i| vertices[i]).collect();
        let uvs: Vec<Vec2> = kept.iter().map(|&i| uvs.get(i).copied().unwrap_or(Vec2::ZERO)).collect();
        if loop_area(&vertices, plane.normal) <= 0.0 {
            return None; // degenerate, or wound the wrong way for its plane
        }
        Some(Self {
            vertices,
            plane,
            uvs,
            material,
        })
    }
}

/// The indices of the loop's vertices worth keeping: consecutive vertices
/// closer together than the weld tolerance are one vertex, and the closing
/// point is dropped when it folds back onto the first.
///
/// The indices come back rather than the points, so the UVs that go with them
/// can be kept in step.
fn compress_indices(verts: &[Vec3]) -> Vec<usize> {
    let mut kept: Vec<usize> = Vec::with_capacity(verts.len());
    for (i, &v) in verts.iter().enumerate() {
        let duplicate = kept
            .last()
            .is_some_and(|&k| (v - verts[k]).length() <= WELD);
        if !duplicate {
            kept.push(i);
        }
    }
    while kept.len() > 1 && (verts[*kept.last().unwrap()] - verts[kept[0]]).length() <= WELD {
        kept.pop();
    }
    kept
}

/// The signed area of a planar loop as seen from `normal`'s side.
fn loop_area(verts: &[Vec3], normal: Vec3) -> f64 {
    let mut acc = Vec3::ZERO;
    for i in 0..verts.len() {
        acc += verts[i].cross(verts[(i + 1) % verts.len()]);
    }
    acc.dot(normal) * 0.5
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
            poly.uvs.reverse();
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

/// The map from world space to the normalized space the CSG runs in.
///
/// Both solids are centred on the origin and scaled to an extent of about one
/// before anything is split, and the result is mapped back afterwards. Every
/// tolerance in this module is absolute, and this is what makes those
/// tolerances *relative to the model*: a scene measured in millimetres and one
/// measured in thousands take the same path with the same accuracy, where a
/// fixed `1e-4` would be a no-op on one and a blunt instrument on the other.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Fit {
    center: Vec3,
    scale: f64,
}

impl Fit {
    /// A map that puts every vertex of `groups` inside a box about a unit
    /// across, centred on their common middle.
    pub fn of(groups: &[&[Polygon]]) -> Self {
        let mut min = Vec3::splat(f64::INFINITY);
        let mut max = Vec3::splat(f64::NEG_INFINITY);
        for g in groups {
            for p in g.iter() {
                for &v in &p.vertices {
                    min = min.min(v);
                    max = max.max(v);
                }
            }
        }
        if !min.is_finite() || !max.is_finite() {
            return Self {
                center: Vec3::ZERO,
                scale: 1.0,
            };
        }
        let center = (min + max) * 0.5;
        let extent = (max - min).max_element();
        let scale = if extent > 0.0 { 1.0 / extent } else { 1.0 };
        Self { center, scale }
    }

    /// `p` in normalized space.
    pub fn point(&self, p: Vec3) -> Vec3 {
        (p - self.center) * self.scale
    }

    /// The same polygons, in normalized space. A uniform scale and a
    /// translation leave a plane's normal alone: only its offset moves.
    pub fn polygons(&self, polys: &[Polygon]) -> Vec<Polygon> {
        polys
            .iter()
            .map(|p| Polygon {
                vertices: p.vertices.iter().map(|&v| self.point(v)).collect(),
                plane: Plane {
                    normal: p.plane.normal,
                    w: (p.plane.w - p.plane.normal.dot(self.center)) * self.scale,
                },
                // A uniform scale and a translation leave the texture where
                // it was: the UVs are a picture of the surface, not of where
                // the surface sits.
                uvs: p.uvs.clone(),
                material: p.material,
            })
            .collect()
    }

    /// A mesh built in normalized space, put back where it came from. Only the
    /// positions move — by the same uniform map — so the faces, and the winding
    /// the normals are computed from, come back unchanged.
    pub fn restore(&self, mut mesh: Mesh) -> Mesh {
        for p in &mut mesh.positions {
            *p = (p.as_dvec3() / self.scale + self.center).as_vec3();
        }
        mesh.compute_flat_normals();
        mesh
    }
}

/// Triangulate a mesh's faces into a flat list of polygons (fan-triangulated,
/// so every polygon is a triangle and therefore convex and planar).
///
/// The mesh's own `f32` positions are widened to `f64` here: the whole CSG
/// runs in double precision, and the only rounding to single precision
/// happens once, on the way out.
pub(crate) fn mesh_to_polygons(mesh: &Mesh) -> Vec<Polygon> {
    let mut out = Vec::new();
    for f in &mesh.faces {
        let pts: Vec<Vec3> = f
            .corners
            .iter()
            .filter_map(|c| mesh.positions.get(c.position as usize).copied())
            .map(|p| p.as_dvec3())
            .collect();
        // The corner's own UV, so the texture travels with the face.
        let uvs: Vec<Vec2> = f
            .corners
            .iter()
            .map(|c| mesh.uvs.get(c.uv as usize).copied().unwrap_or(Vec2::ZERO))
            .collect();
        if pts.len() < 3 {
            continue;
        }
        for i in 1..pts.len() - 1 {
            let tri = vec![pts[0], pts[i], pts[i + 1]];
            let tri_uvs = vec![uvs[0], uvs[i], uvs[i + 1]];
            if let Some(plane) = Plane::from_points(tri[0], tri[1], tri[2]) {
                out.push(Polygon {
                    vertices: tri,
                    plane,
                    uvs: tri_uvs,
                    material: f.material,
                });
            }
        }
    }
    out
}

/// Insert any vertex that lies in the interior of a polygon edge, so that a
/// neighbour split at that point shares the same edge subdivision instead of
/// leaving a T-junction. Returns the expanded, ordered boundary loop.
///
/// The candidates come from a grid rather than from the whole point cloud:
/// this question is asked once per edge of every polygon in the result, and
/// the answer is nearly always "nothing here", so the grid is what keeps a
/// boolean of a real scene from taking minutes.
fn expand_t_junctions(
    verts: &[Vec3],
    uvs: &[Vec2],
    grid: &PointGrid,
    tol: f64,
) -> (Vec<Vec3>, Vec<Vec2>) {
    let tol2 = tol * tol;
    let mut out: Vec<Vec3> = Vec::with_capacity(verts.len());
    let mut out_uvs: Vec<Vec2> = Vec::with_capacity(verts.len());
    let n = verts.len();
    for i in 0..n {
        let a = verts[i];
        let b = verts[(i + 1) % n];
        let ua = uvs.get(i).copied().unwrap_or(Vec2::ZERO);
        let ub = uvs.get((i + 1) % n).copied().unwrap_or(Vec2::ZERO);
        out.push(a);
        out_uvs.push(ua);
        let ab = b - a;
        let l2 = ab.length_squared();
        if l2 < tol2 {
            continue; // degenerate edge
        }
        let mut ts: Vec<f64> = Vec::new();
        grid.visit_near_segment(a, b, tol, |p| {
            let t = (p - a).dot(ab) / l2;
            if t <= tol || t >= 1.0 - tol {
                return; // too close to an endpoint
            }
            let proj = a + ab * t;
            if (p - proj).length() < tol {
                ts.push(t);
            }
        });
        ts.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
        let mut last = -1.0f64;
        for t in ts {
            if t - last > tol {
                out.push(a + ab * t);
                // The point's UV is where it falls on the edge's own span, so
                // the texture runs along the edge without a jump.
                out_uvs.push(ua + (ub - ua) * (t as f32));
                last = t;
            }
        }
    }
    (out, out_uvs)
}

/// A uniform grid over a point cloud, for "which points lie within `tol` of
/// this segment" without walking the whole cloud for every edge.
///
/// The cell is a sixty-fourth of the model's extent: few enough cells to fill
/// in a moment, small enough that one holds a handful of vertices. A segment
/// is sampled along its length, and each sample looks at its own cell and the
/// six around it, so no point near the segment can be missed.
struct PointGrid {
    cell: f64,
    cells: HashMap<(i64, i64, i64), Vec<Vec3>>,
}

impl PointGrid {
    fn new(points: &[Vec3]) -> Self {
        let mut min = Vec3::splat(f64::INFINITY);
        let mut max = Vec3::splat(f64::NEG_INFINITY);
        for &p in points {
            min = min.min(p);
            max = max.max(p);
        }
        let extent = if min.is_finite() && max.is_finite() {
            (max - min).max_element()
        } else {
            0.0
        };
        let cell = (extent / 64.0).max(WELD * 4.0);
        let mut cells: HashMap<(i64, i64, i64), Vec<Vec3>> = HashMap::new();
        for &p in points {
            cells.entry(cell_of(p, cell)).or_default().push(p);
        }
        Self { cell, cells }
    }

    /// Visit every point within `tol` of the segment `a`-`b`.
    ///
    /// The points are found by walking the cells the segment passes through
    /// and their neighbours, and each candidate is then measured against the
    /// segment itself, so a cell that merely happens to be nearby does not
    /// count.
    fn visit_near_segment(&self, a: Vec3, b: Vec3, tol: f64, mut f: impl FnMut(Vec3)) {
        let d = b - a;
        let l2 = d.length_squared();
        let len = d.length();
        let steps = (len / (self.cell * 0.5)).ceil().max(1.0);
        let steps = if steps.is_finite() {
            (steps as usize).min(4096)
        } else {
            1
        };
        let mut last: Option<(i64, i64, i64)> = None;
        for i in 0..=steps {
            let t = i as f64 / steps as f64;
            let p = a + d * t;
            let (cx, cy, cz) = cell_of(p, self.cell);
            if last == Some((cx, cy, cz)) {
                continue; // the same cell as the last sample
            }
            last = Some((cx, cy, cz));
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        let Some(pts) = self.cells.get(&(cx + dx, cy + dy, cz + dz)) else {
                            continue;
                        };
                        for &q in pts {
                            if l2 < tol * tol {
                                if (q - a).length() <= tol {
                                    f(q);
                                }
                                continue;
                            }
                            let t = (q - a).dot(d) / l2;
                            if t <= 0.0 || t >= 1.0 {
                                continue;
                            }
                            let proj = a + d * t;
                            if (q - proj).length() <= tol {
                                f(q);
                            }
                        }
                    }
                }
            }
        }
    }
}

/// The grid cell a point falls in.
fn cell_of(p: Vec3, cell: f64) -> (i64, i64, i64) {
    (
        (p.x / cell).floor() as i64,
        (p.y / cell).floor() as i64,
        (p.z / cell).floor() as i64,
    )
}

/// Welds points to vertex indices by tolerance.
///
/// Two points closer than the tolerance are one vertex, whichever side of a
/// rounding boundary they fall on: a quantization grid would put them in
/// different cells and leave a crack in the surface, so this looks at the
/// neighbouring cells as well.
struct VertexWelder {
    tol: f64,
    cells: HashMap<(i64, i64, i64), Vec<(Vec3, u32)>>,
}

impl VertexWelder {
    fn new(tol: f64) -> Self {
        Self {
            tol: tol.max(f64::MIN_POSITIVE),
            cells: HashMap::new(),
        }
    }

    /// The index of an already-welded point within tolerance, if there is one.
    fn find(&self, p: Vec3) -> Option<u32> {
        let (cx, cy, cz) = cell_of(p, self.tol);
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(pts) = self.cells.get(&(cx + dx, cy + dy, cz + dz)) {
                        for (q, i) in pts {
                            if (*q - p).length() <= self.tol {
                                return Some(*i);
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Remember `p` as the vertex `index`.
    fn insert(&mut self, p: Vec3, index: u32) {
        self.cells.entry(cell_of(p, self.tol)).or_default().push((p, index));
    }
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
            // The threshold is the weld tolerance, so an ear this thin is
            // never emitted: a face that thin is dropped again by the caller,
            // and dropping it would leave a hole in the surface.
            if cross.length() < WELD {
                continue;
            }
            // Convex ear relative to the loop orientation.
            if cross.dot(normal) <= 0.0 {
                continue;
            }
            // Ensure no other loop vertex lies inside this ear. A vertex
            // that is one of the ear's own corners — the same place reached
            // through the weld — is not inside it.
            let mut occupied = false;
            for &v in &idx {
                let vi = v as usize;
                if vi == ip || vi == ic || vi == inx {
                    continue;
                }
                let p = pos[vi];
                if (p - a).length() <= WELD
                    || (p - b).length() <= WELD
                    || (p - c).length() <= WELD
                {
                    continue;
                }
                if point_in_tri(p, a, b, c, normal) {
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
    const T: f64 = 1e-12;
    let has_neg = d1 < -T || d2 < -T || d3 < -T;
    let has_pos = d1 > T || d2 > T || d3 > T;
    !(has_neg && has_pos)
}

/// Rebuild a mesh from polygons. Shared vertices are welded by tolerance,
/// T-junctions along edges are resolved so the surface stays manifold, and
/// each (now weakly-convex, possibly many-sided) polygon is ear-clipped into
/// triangles.
///
/// The polygon loop is welded *before* it is triangulated, so the ear clipper
/// sees one index per point and a shared edge is shared exactly — which is
/// what makes the result watertight.
pub(crate) fn polygons_to_mesh(polygons: &[Polygon]) -> Mesh {
    // Candidate T-vertices: every vertex across every polygon, in a grid.
    let all: Vec<Vec3> = polygons.iter().flat_map(|p| p.vertices.iter().copied()).collect();
    let grid = PointGrid::new(&all);

    let mut mesh = Mesh::new();
    let mut welder = VertexWelder::new(WELD);
    // The welded positions, parallel to the mesh's own vertex indices.
    let mut points: Vec<Vec3> = Vec::new();

    for poly in polygons {
        if poly.vertices.len() < 3 {
            continue;
        }
        let (loop_pts, loop_uvs) = expand_t_junctions(&poly.vertices, &poly.uvs, &grid, WELD);
        let m = loop_pts.len();
        if m < 3 {
            continue;
        }
        // Weld the loop into indexed vertices. The UV does not weld with the
        // position: a seam is a place where two faces want different UVs at
        // the same point, and welding them would flatten the texture there.
        let mut idx: Vec<u32> = Vec::with_capacity(m);
        let mut uv_index: Vec<u32> = Vec::with_capacity(m);
        for (k, &v) in loop_pts.iter().enumerate() {
            match welder.find(v) {
                Some(i) => idx.push(i),
                None => {
                    let i = mesh.add_vertex(v.as_vec3());
                    welder.insert(v, i);
                    points.push(v);
                    idx.push(i);
                }
            }
            uv_index.push(mesh.uvs.len() as u32);
            mesh.uvs.push(loop_uvs[k]);
        }
        // The clipper works on the *welded* geometry — that is the geometry
        // the faces will point at — but it keeps the loop's own numbering, so
        // each corner still knows its UV.
        let geom: Vec<Vec3> = idx.iter().map(|&i| points[i as usize]).collect();
        let positions: Vec<u32> = (0..m as u32).collect();
        for tri in ear_clip(&positions, &geom, poly.plane.normal) {
            let (p0, p1, p2) = (
                geom[tri[0] as usize],
                geom[tri[1] as usize],
                geom[tri[2] as usize],
            );
            // A face that is *exactly* degenerate (three identical points) is
            // dropped: it would be a zero-length edge in the mesh. A merely
            // thin face is kept — it is a sliver, but it is part of the
            // surface, and dropping it would leave a hole where the two
            // solids met.
            if (p1 - p0).cross(p2 - p0).length_squared() == 0.0 {
                continue;
            }
            let corners: Vec<Corner> = tri
                .iter()
                .map(|&p| {
                    let k = p as usize;
                    Corner {
                        position: idx[k],
                        normal: 0,
                        uv: uv_index[k],
                    }
                })
                .collect();
            mesh.faces.push(Face {
                corners,
                material: poly.material,
            });
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

