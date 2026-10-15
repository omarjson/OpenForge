//! A lightweight 3D viewport that projects the scene and draws it with the
//! egui painter.
//!
//! This keeps the renderer dependency-free and stable across egui/wgpu
//! versions on all three desktop platforms. It draws a ground grid, the
//! selected/unselected meshes (flat-shaded by face normal), and selection
//! outlines. A production build would swap this for a wgpu pipeline while
//! reusing the same camera math.

use glam::{Mat4, Vec2, Vec3, Vec4};
use openforge_core::math::Ray;
use openforge_materials::{Material, MaterialLibrary};
use std::collections::HashMap;
use openforge_core::mesh::Mesh;
use openforge_core::scene::Scene;
use openforge_core::selection::{SelectLevel, Selection};
use openforge_core::NodeId;
use std::collections::BTreeSet;

/// A perspective orbit camera.
#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub target: Vec3,
    pub distance: f32,
    pub yaw: f32,
    pub pitch: f32,
    pub fov_y: f32,
    pub near: f32,
    pub far: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            target: Vec3::ZERO,
            distance: 8.0,
            yaw: 0.6,
            pitch: 0.5,
            fov_y: 55f32.to_radians(),
            near: 0.05,
            far: 1000.0,
        }
    }
}

impl Camera {
    pub fn eye(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        self.target + Vec3::new(cy * cp, sp, sy * cp) * self.distance
    }

    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(self.eye(), self.target, Vec3::Y)
    }

    pub fn proj_matrix(&self, aspect: f32) -> Mat4 {
        Mat4::perspective_rh(self.fov_y, aspect.max(0.01), self.near, self.far)
    }

    pub fn view_proj(&self, aspect: f32) -> Mat4 {
        self.proj_matrix(aspect) * self.view_matrix()
    }

    /// Orbit by pixel deltas.
    pub fn orbit(&mut self, dx: f32, dy: f32) {
        self.yaw += dx * 0.01;
        self.pitch = (self.pitch + dy * 0.01).clamp(-1.5, 1.5);
    }

    pub fn pan(&mut self, dx: f32, dy: f32) {
        let view = self.view_matrix();
        let right = Vec3::new(view.x_axis.x, view.y_axis.x, view.z_axis.x);
        let up = Vec3::new(view.x_axis.y, view.y_axis.y, view.z_axis.y);
        let scale = self.distance * 0.002;
        self.target += (-right * dx + up * dy) * scale;
    }

    pub fn zoom(&mut self, scroll: f32) {
        self.distance = (self.distance * (1.0 - scroll * 0.1)).clamp(0.1, 5000.0);
    }

    /// Build a picking ray from normalized device coords in [-1, 1].
    pub fn ray_from_ndc(&self, ndc_x: f32, ndc_y: f32, aspect: f32) -> Ray {
        let inv = self.view_proj(aspect).inverse();
        let near = inv * Vec4::new(ndc_x, ndc_y, -1.0, 1.0);
        let far = inv * Vec4::new(ndc_x, ndc_y, 1.0, 1.0);
        let near = near.truncate() / near.w;
        let far = far.truncate() / far.w;
        Ray { origin: near, direction: (far - near).normalize_or_zero() }
    }
}

/// Project a world point to normalized device coords + depth. Returns `None`
/// when behind the camera.
pub fn project(vp: Mat4, p: Vec3, size: [f32; 2]) -> Option<(f32, f32, f32)> {
    let clip = vp * Vec4::new(p.x, p.y, p.z, 1.0);
    if clip.w <= 1e-5 {
        return None;
    }
    let ndc = clip.truncate() / clip.w;
    let sx = (ndc.x * 0.5 + 0.5) * size[0];
    let sy = (1.0 - (ndc.y * 0.5 + 0.5)) * size[1];
    Some((sx, sy, ndc.z))
}

/// A vertex ready for rasterization.
pub struct Vx {
    pub pos: Vec3,
    pub normal: Vec3,
    /// This corner's texture coordinates.
    pub uv: Vec2,
}

/// A triangle in world space with a shading normal.
pub struct Tri {
    pub v: [Vx; 3],
    pub selected: bool,
    /// True when this triangle's face is in the current face selection.
    pub face_selected: bool,
    /// The library material slot this triangle's face resolves to.
    pub material: u32,
}

/// Blend `a` towards `b` by `t` (0 keeps `a`, 1 gives `b`).
fn mix(a: u8, b: u8, t: f32) -> u8 {
    (a as f32 + (b as f32 - a as f32) * t.clamp(0.0, 1.0)).round().clamp(0.0, 255.0) as u8
}

/// sRGB bytes for a material's base colour, for the CPU viewport's shading.
///
/// The material's `effective_color` carries the diffuse colour with opacity
/// folded into alpha; a selected mesh (or a selected face) is nudged towards
/// the same highlight colours the unassigned materials use, so selection still
/// reads on a mesh that has a material of its own.
pub fn material_base(material: &Material, selected: bool, face_selected: bool) -> [u8; 4] {
    let c = material.effective_color();
    let rgb = [
        (c.x.clamp(0.0, 1.0) * 255.0).round() as u8,
        (c.y.clamp(0.0, 1.0) * 255.0).round() as u8,
        (c.z.clamp(0.0, 1.0) * 255.0).round() as u8,
    ];
    let alpha = (c.w.clamp(0.0, 1.0) * 255.0).round() as u8;
    let tint = if face_selected {
        Some([255u8, 170, 60])
    } else if selected {
        Some([80u8, 170, 255])
    } else {
        None
    };
    match tint {
        Some(t) => [
            mix(rgb[0], t[0], 0.55),
            mix(rgb[1], t[1], 0.55),
            mix(rgb[2], t[2], 0.55),
            alpha,
        ],
        None => [rgb[0], rgb[1], rgb[2], alpha],
    }
}

/// The library texture a face's material samples, if any.
///
/// A material names its diffuse map by index; a map whose texture the library
/// no longer holds is ignored, so a face can never sample past the end of the
/// texture list. Both viewports resolve through this one function — the CPU
/// painter takes the index as its texture array handle, the GPU path as the
/// vertex's layer — so they cannot disagree about which faces are textured.
pub fn face_texture(library: &MaterialLibrary, material_slot: u32) -> Option<usize> {
    let map = library.resolve(material_slot).diffuse_map;
    map.filter(|index| library.textures.get(*index).is_some())
}

/// Collect all visible mesh triangles from the scene in world space.
pub fn collect_triangles(
    scene: &mut Scene,
    selection: &Selection,
) -> Vec<Tri> {
    let mut out = Vec::new();
    let ids: Vec<NodeId> = scene.iter().filter(|n| n.is_mesh() && n.visible).map(|n| n.id).collect();
    for id in ids {
        let Ok(world) = scene.world_matrix(id) else { continue };
        let selected = selection.is_node_selected(id);
        let sel_faces = selection.selected_elements(id).cloned();
        let Ok(node) = scene.get(id) else { continue };
        let Some(mesh) = node.mesh() else { continue };
        // Resolve the node's slot list into a library slot per face.
        // Map each source face to its triangulated output.
        for (face_idx, face) in mesh.faces.iter().enumerate() {
            let face_is_sel = sel_faces
                .as_ref()
                .map(|s| s.contains(&(face_idx as u32)))
                .unwrap_or(false);
            let material = node.face_material_slot(face_idx);
            let cs = &face.corners;
            for i in 1..cs.len().saturating_sub(1) {
                let tri = [cs[0], cs[i], cs[i + 1]];
                let p = |corner: openforge_core::mesh::Corner| {
                    world.transform_point3(mesh.positions[corner.position as usize])
                };
                let a = p(tri[0]);
                let b = p(tri[1]);
                let c = p(tri[2]);
                let n = (b - a).cross(c - a).normalize_or_zero();
                let uv = |corner: openforge_core::mesh::Corner| {
                    mesh.uvs.get(corner.uv as usize).copied().unwrap_or(Vec2::ZERO)
                };
                out.push(Tri {
                    v: [
                        Vx { pos: a, normal: n, uv: uv(tri[0]) },
                        Vx { pos: b, normal: n, uv: uv(tri[1]) },
                        Vx { pos: c, normal: n, uv: uv(tri[2]) },
                    ],
                    selected,
                    face_selected: face_is_sel,
                    material,
                });
            }
        }
    }
    out
}

/// Result of a face pick: the node and the source face index.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct FaceHit {
    pub node: NodeId,
    pub face: u32,
    pub dist: f32,
}

/// Cast a ray and return the nearest visible face across all mesh nodes.
pub fn pick_face(scene: &mut Scene, ray: Ray) -> Option<FaceHit> {
    let candidates: Vec<(NodeId, Mesh)> = scene
        .iter()
        .filter(|n| n.is_mesh() && n.visible && !n.frozen)
        .filter_map(|n| n.mesh().map(|m| (n.id, m.clone())))
        .collect();

    let mut best: Option<FaceHit> = None;
    for (id, mesh) in candidates {
        let Ok(world) = scene.world_matrix(id) else { continue };
        let inv = world.inverse();
        let local_origin = inv.transform_point3(ray.origin);
        let local_dir = (inv * ray.direction.extend(0.0)).truncate().normalize_or_zero();
        let local_ray = Ray { origin: local_origin, direction: local_dir };
        for (fi, face) in mesh.faces.iter().enumerate() {
            let cs = &face.corners;
            for i in 1..cs.len().saturating_sub(1) {
                let a = mesh.positions[cs[0].position as usize];
                let b = mesh.positions[cs[i].position as usize];
                let c = mesh.positions[cs[i + 1].position as usize];
                if let Some(t) = local_ray.intersect_triangle(a, b, c) {
                    let world_hit = world.transform_point3(local_ray.at(t));
                    let dist = world_hit.distance(ray.origin);
                    let better = best.map(|b| dist < b.dist).unwrap_or(true);
                    if better {
                        best = Some(FaceHit { node: id, face: fi as u32, dist });
                    }
                }
            }
        }
    }
    best
}

/// How much edge overlay the viewport draws over the shaded meshes.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Wireframe {
    /// Shaded meshes only.
    #[default]
    Off,
    /// The edges of the selected meshes, as a selection outline.
    Selection,
    /// Every visible mesh's edges, so topology reads through the shading.
    All,
}

impl Wireframe {
    /// The next mode in the `Off → Selection → All` cycle.
    pub fn next(self) -> Self {
        match self {
            Wireframe::Off => Wireframe::Selection,
            Wireframe::Selection => Wireframe::All,
            Wireframe::All => Wireframe::Off,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Wireframe::Off => "Off",
            Wireframe::Selection => "Selection",
            Wireframe::All => "All",
        }
    }
}

/// Edge-overlay colours, authored as sRGB bytes so the CPU painter and the
/// GPU edge pass draw the same colours. The GPU path converts them to linear
/// space (the offscreen target is sRGB).
pub const EDGE_FACE_SELECTED_SRGB: [u8; 3] = [255, 170, 60];
pub const EDGE_NODE_SELECTED_SRGB: [u8; 3] = [90, 190, 255];
pub const EDGE_PLAIN_SRGB: [u8; 3] = [150, 158, 172];
/// Alpha of the plain (unselected) edge colour.
pub const EDGE_PLAIN_ALPHA: u8 = 130;

/// One world-space edge of the overlay, flagged for how it should be coloured.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct OverlayEdge {
    pub a: Vec3,
    pub b: Vec3,
    /// The owning mesh node is selected.
    pub node_selected: bool,
    /// At least one face using this edge is in the face selection.
    pub face_selected: bool,
}

/// The world-space normal of a polygon, via Newell's method.
///
/// Unlike a cross product of the first three corners this stays sane on
/// non-planar n-gons, which is exactly what an edge overlay walks.
pub fn polygon_normal(points: &[Vec3]) -> Vec3 {
    let mut n = Vec3::ZERO;
    let len = points.len();
    if len < 3 {
        return n;
    }
    for i in 0..len {
        let a = points[i];
        let b = points[(i + 1) % len];
        n.x += (a.y - b.y) * (a.z + b.z);
        n.y += (a.z - b.z) * (a.x + b.x);
        n.z += (a.x - b.x) * (a.y + b.y);
    }
    n.normalize_or_zero()
}

/// Collect the overlay edges for the visible scene.
///
/// The edges come from `wireframe`'s mode and are deduplicated per mesh, so a
/// shared edge is drawn once. An edge is dropped when *every* face using it
/// faces away from `eye`: without that, a closed mesh's wireframe would draw
/// its own far side straight through the shaded surface.
pub fn collect_edges(
    scene: &mut Scene,
    selection: &Selection,
    wireframe: Wireframe,
    eye: Vec3,
) -> Vec<OverlayEdge> {
    let mut out = Vec::new();
    if wireframe == Wireframe::Off {
        return out;
    }

    let ids: Vec<(NodeId, bool)> = scene
        .iter()
        .filter(|n| n.is_mesh() && n.visible && !n.frozen)
        .map(|n| (n.id, selection.is_node_selected(n.id)))
        .filter(|(_, selected)| wireframe == Wireframe::All || *selected)
        .collect();

    // An edge as it is discovered, before the facing test is final: later faces
    // sharing it can still reveal it as front-facing.
    struct Pending {
        a: Vec3,
        b: Vec3,
        front: bool,
        face_selected: bool,
    }

    for (id, node_selected) in ids {
        let Ok(world) = scene.world_matrix(id) else { continue };
        let sel_faces = selection.selected_elements(id).cloned();
        let Ok(node) = scene.get(id) else { continue };
        let Some(mesh) = node.mesh() else { continue };

        let mut pending: Vec<Pending> = Vec::new();
        let mut index: HashMap<(u32, u32), usize> = HashMap::new();

        for (fi, face) in mesh.faces.iter().enumerate() {
            let corners = &face.corners;
            if corners.len() < 2 {
                continue;
            }
            // Resolve the face in world space once, keeping each corner's
            // vertex index beside it; a corner whose vertex no longer exists is
            // skipped rather than trusted.
            let verts: Vec<(u32, Vec3)> = corners
                .iter()
                .filter_map(|c| {
                    mesh.positions
                        .get(c.position as usize)
                        .map(|p| (c.position, world.transform_point3(*p)))
                })
                .collect();
            if verts.len() < 2 {
                continue;
            }
            let pts: Vec<Vec3> = verts.iter().map(|(_, p)| *p).collect();
            let normal = polygon_normal(&pts);
            let front = normal.length_squared() > 0.0 && normal.dot(eye - pts[0]) > 0.0;
            let face_selected = sel_faces
                .as_ref()
                .map(|s| s.contains(&(fi as u32)))
                .unwrap_or(false);

            let len = verts.len();
            for i in 0..len {
                let (pi, pa) = verts[i];
                let (pj, pb) = verts[(i + 1) % len];
                // Two corners of one face can name the same vertex (a seam);
                // there is no edge to draw between a point and itself.
                if pi == pj {
                    continue;
                }
                let key = (pi.min(pj), pi.max(pj));
                match index.get(&key) {
                    Some(&at) => {
                        if front {
                            pending[at].front = true;
                        }
                        if face_selected {
                            pending[at].face_selected = true;
                        }
                    }
                    None => {
                        index.insert(key, pending.len());
                        pending.push(Pending { a: pa, b: pb, front, face_selected });
                    }
                }
            }
        }

        for edge in pending {
            if !edge.front {
                continue;
            }
            out.push(OverlayEdge {
                a: edge.a,
                b: edge.b,
                node_selected,
                face_selected: edge.face_selected,
            });
        }
    }
    out
}

/// The world-space centroid of the selected sub-objects, if anything is
/// selected at vertex or face level.
///
/// This is where a sub-object transform should be applied *from*: rotating a
/// face about the node's bounds centre looks like a bug, rotating it about
/// itself is what the user asked for.
pub fn element_centroid(scene: &mut Scene, selection: &Selection) -> Option<Vec3> {
    let ids: Vec<NodeId> = selection.selected_nodes().collect();
    let mut sum = Vec3::ZERO;
    let mut count = 0usize;
    for id in ids {
        let Some(elements) = selection.selected_elements(id) else { continue };
        let Ok(world) = scene.world_matrix(id) else { continue };
        let Ok(node) = scene.get(id) else { continue };
        let Some(mesh) = node.mesh() else { continue };
        for &index in elements {
            match selection.level {
                SelectLevel::Vertex => {
                    if let Some(p) = mesh.positions.get(index as usize) {
                        sum += world.transform_point3(*p);
                        count += 1;
                    }
                }
                SelectLevel::Face | SelectLevel::Edge => {
                    let Some(face) = mesh.faces.get(index as usize) else { continue };
                    for corner in &face.corners {
                        if let Some(p) = mesh.positions.get(corner.position as usize) {
                            sum += world.transform_point3(*p);
                            count += 1;
                        }
                    }
                }
                SelectLevel::Object => {}
            }
        }
    }
    (count > 0).then(|| sum / count as f32)
}

/// The vertices a sub-object drag should move, grouped by owning node, with
/// their mesh-local positions at the moment the drag began.
///
/// Corners shared by several selected faces are listed once.
pub fn selected_vertices(scene: &mut Scene, selection: &Selection) -> Vec<(NodeId, Vec<(u32, Vec3)>)> {
    let mut out = Vec::new();
    if selection.level == SelectLevel::Object {
        return out;
    }
    for id in selection.selected_nodes().collect::<Vec<_>>() {
        let Some(elements) = selection.selected_elements(id) else { continue };
        let Ok(node) = scene.get(id) else { continue };
        let Some(mesh) = node.mesh() else { continue };

        let mut indices: BTreeSet<u32> = BTreeSet::new();
        for &index in elements {
            match selection.level {
                SelectLevel::Vertex => {
                    indices.insert(index);
                }
                SelectLevel::Face | SelectLevel::Edge => {
                    let Some(face) = mesh.faces.get(index as usize) else { continue };
                    for corner in &face.corners {
                        indices.insert(corner.position);
                    }
                }
                SelectLevel::Object => {}
            }
        }

        let positions: Vec<(u32, Vec3)> = indices
            .into_iter()
            .filter_map(|vi| mesh.positions.get(vi as usize).map(|p| (vi, *p)))
            .collect();
        if !positions.is_empty() {
            out.push((id, positions));
        }
    }
    out
}

/// Result of a vertex pick.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct VertexHit {
    pub node: NodeId,
    pub vertex: u32,
    pub dist: f32,
}

/// Cast a ray and return the nearest vertex within screen-space tolerance.
pub fn pick_vertex(
    scene: &mut Scene,
    ray: Ray,
    cam_pos: Vec3,
    tol_world: f32,
) -> Option<VertexHit> {
    let candidates: Vec<(NodeId, Mesh)> = scene
        .iter()
        .filter(|n| n.is_mesh() && n.visible && !n.frozen)
        .filter_map(|n| n.mesh().map(|m| (n.id, m.clone())))
        .collect();

    let mut best: Option<VertexHit> = None;
    for (id, mesh) in candidates {
        let Ok(world) = scene.world_matrix(id) else { continue };
        for (vi, _) in mesh.positions.iter().enumerate() {
            let wp = world.transform_point3(mesh.positions[vi]);
            // Distance from the vertex to the ray.
            let to_v = wp - ray.origin;
            let along = to_v.dot(ray.direction);
            if along < 0.0 {
                continue;
            }
            let closest = ray.at(along);
            let perp = wp.distance(closest);
            if perp <= tol_world {
                let better = best.map(|b| along < b.dist).unwrap_or(true);
                if better {
                    best = Some(VertexHit { node: id, vertex: vi as u32, dist: along });
                }
            }
        }
    }
    let _ = cam_pos;
    best
}

/// A vertex a drag can snap to: where it is in the world, where it lands on
/// screen (viewport-relative pixels, the frame [`project`] returns), and its
/// depth, which breaks a screen-distance tie in favour of the nearer vertex.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct SnapCandidate {
    pub node: NodeId,
    pub vertex: u32,
    pub world: Vec3,
    pub screen: [f32; 2],
    pub depth: f32,
}

/// Every vertex a drag may snap to: the vertices of the visible, unfrozen
/// meshes, projected to screen, minus the geometry the drag itself carries.
///
/// `moving_nodes` are the nodes whose vertices all move (object-level drags);
/// `moving_vertices` are the individual vertices that move (element-level
/// drags, which may be a subset of their node, so the rest of that node stays
/// a valid target).
pub fn snap_candidates(
    scene: &mut Scene,
    moving_nodes: &[NodeId],
    moving_vertices: &[(NodeId, u32)],
    vp: Mat4,
    size: [f32; 2],
) -> Vec<SnapCandidate> {
    let mut out = Vec::new();
    let ids: Vec<NodeId> = scene
        .iter()
        .filter(|n| n.is_mesh() && n.visible && !n.frozen)
        .map(|n| n.id)
        .filter(|id| !moving_nodes.contains(id))
        .collect();
    for id in ids {
        let Ok(world) = scene.world_matrix(id) else { continue };
        let Ok(node) = scene.get(id) else { continue };
        let Some(mesh) = node.mesh() else { continue };
        for (vi, p) in mesh.positions.iter().enumerate() {
            let vi = vi as u32;
            if moving_vertices.contains(&(id, vi)) {
                continue;
            }
            let wp = world.transform_point3(*p);
            let Some((sx, sy, depth)) = project(vp, wp, size) else { continue };
            out.push(SnapCandidate { node: id, vertex: vi, world: wp, screen: [sx, sy], depth });
        }
    }
    out
}

/// The candidate closest to `cursor` (viewport-relative pixels) within `radius`
/// pixels, or `None` when nothing is close enough.
pub fn nearest_candidate(
    candidates: &[SnapCandidate],
    cursor: [f32; 2],
    radius: f32,
) -> Option<SnapCandidate> {
    let mut best: Option<(f32, SnapCandidate)> = None;
    for c in candidates {
        let dx = c.screen[0] - cursor[0];
        let dy = c.screen[1] - cursor[1];
        let dist = (dx * dx + dy * dy).sqrt();
        if dist > radius {
            continue;
        }
        let better = match &best {
            None => true,
            Some((best_dist, best_c)) => {
                dist < *best_dist - 1e-6
                    || ((dist - *best_dist).abs() <= 1e-6 && c.depth < best_c.depth)
            }
        };
        if better {
            best = Some((dist, *c));
        }
    }
    best.map(|(_, c)| c)
}

#[cfg(test)]
mod tests {
    use super::*;
    use openforge_core::mesh::Mesh;
    use openforge_core::selection::SelectLevel;
    use openforge_materials::Texture;

    /// A quad in the XY plane, wound counter-clockwise seen from `+Z`.
    fn quad(z: f32) -> Mesh {
        let mut m = Mesh::new();
        m.add_vertex(Vec3::new(-1.0, -1.0, z));
        m.add_vertex(Vec3::new(1.0, -1.0, z));
        m.add_vertex(Vec3::new(1.0, 1.0, z));
        m.add_vertex(Vec3::new(-1.0, 1.0, z));
        m.add_face(&[0, 1, 2, 3]);
        m.compute_flat_normals();
        m
    }

    /// A unit cube with outward-facing quads.
    fn cube() -> Mesh {
        let mut m = Mesh::new();
        for &(x, y, z) in &[
            (-1.0, -1.0, -1.0),
            (1.0, -1.0, -1.0),
            (1.0, 1.0, -1.0),
            (-1.0, 1.0, -1.0),
            (-1.0, -1.0, 1.0),
            (1.0, -1.0, 1.0),
            (1.0, 1.0, 1.0),
            (-1.0, 1.0, 1.0),
        ] {
            m.add_vertex(Vec3::new(x, y, z));
        }
        m.add_face(&[4, 5, 6, 7]); // +Z
        m.add_face(&[1, 0, 3, 2]); // -Z
        m.add_face(&[5, 1, 2, 6]); // +X
        m.add_face(&[0, 4, 7, 3]); // -X
        m.add_face(&[7, 6, 2, 3]); // +Y
        m.add_face(&[0, 1, 5, 4]); // -Y
        m.compute_flat_normals();
        m
    }

    #[test]
    fn wireframe_cycles_off_selection_all() {
        assert_eq!(Wireframe::default(), Wireframe::Off);
        let mut mode = Wireframe::Off;
        let mut seen = Vec::new();
        for _ in 0..3 {
            mode = mode.next();
            seen.push(mode);
        }
        assert_eq!(seen, [Wireframe::Selection, Wireframe::All, Wireframe::Off]);
        assert_eq!(Wireframe::All.label(), "All");
    }

    #[test]
    fn polygon_normal_follows_the_winding() {
        let ccw = [Vec3::ZERO, Vec3::X, Vec3::new(1.0, 1.0, 0.0), Vec3::Y];
        assert!((polygon_normal(&ccw) - Vec3::Z).length() < 1e-5);
        let mut reversed = ccw;
        reversed.reverse();
        assert!((polygon_normal(&reversed) + Vec3::Z).length() < 1e-5);
        // Degenerate input has no direction rather than a garbage one.
        assert_eq!(polygon_normal(&[Vec3::ZERO, Vec3::X]), Vec3::ZERO);
    }

    #[test]
    fn wireframe_off_collects_nothing() {
        let mut scene = Scene::new();
        scene.create_mesh("quad", quad(0.0));
        let sel = Selection::new();
        assert!(collect_edges(&mut scene, &sel, Wireframe::Off, Vec3::new(0.0, 0.0, 5.0)).is_empty());
    }

    #[test]
    fn a_quad_yields_its_four_edges() {
        let mut scene = Scene::new();
        scene.create_mesh("quad", quad(0.0));
        let sel = Selection::new();
        let edges = collect_edges(&mut scene, &sel, Wireframe::All, Vec3::new(0.0, 0.0, 5.0));
        assert_eq!(edges.len(), 4);
        for e in &edges {
            // Every edge keeps its two endpoints and no inherited flags.
            assert!((e.b - e.a).length() > 1.0);
            assert!(!e.node_selected && !e.face_selected);
        }
    }

    #[test]
    fn a_shared_edge_is_drawn_once() {
        let mut m = Mesh::new();
        for (x, y) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
            m.add_vertex(Vec3::new(x, y, 0.0));
        }
        m.add_face(&[0, 1, 2]);
        m.add_face(&[0, 2, 3]);
        m.compute_flat_normals();
        let mut scene = Scene::new();
        scene.create_mesh("two tris", m);
        let sel = Selection::new();
        let edges = collect_edges(&mut scene, &sel, Wireframe::All, Vec3::new(0.0, 0.0, 5.0));
        // Six edge slots, five distinct edges: the diagonal is shared.
        assert_eq!(edges.len(), 5);
    }

    #[test]
    fn edges_seen_only_from_behind_are_dropped() {
        let mut scene = Scene::new();
        scene.create_mesh("quad", quad(0.0));
        let sel = Selection::new();
        let front = collect_edges(&mut scene, &sel, Wireframe::All, Vec3::new(0.0, 0.0, 5.0));
        let back = collect_edges(&mut scene, &sel, Wireframe::All, Vec3::new(0.0, 0.0, -5.0));
        assert_eq!(front.len(), 4);
        assert!(back.is_empty(), "from behind, the quad's own edges are hidden");
    }

    #[test]
    fn a_closed_mesh_hides_the_edges_of_its_far_corner() {
        let mut scene = Scene::new();
        scene.create_mesh("cube", cube());
        let sel = Selection::new();
        let edges = collect_edges(&mut scene, &sel, Wireframe::All, Vec3::new(5.0, 5.0, 5.0));
        // A cube has 12 edges; the three meeting at the far corner join two
        // faces that both point away from the eye, so they are the ones hidden.
        assert_eq!(edges.len(), 9);
    }

    #[test]
    fn selection_mode_only_walks_selected_meshes() {
        let mut scene = Scene::new();
        let a = scene.create_mesh("a", quad(0.0));
        scene.create_mesh("b", quad(2.0));
        let mut sel = Selection::new();
        sel.select_node(a);

        let all = collect_edges(&mut scene, &sel, Wireframe::All, Vec3::new(0.0, 0.0, 8.0));
        let only = collect_edges(&mut scene, &sel, Wireframe::Selection, Vec3::new(0.0, 0.0, 8.0));
        assert_eq!(all.len(), 8, "both quads in All mode");
        assert_eq!(only.len(), 4, "only the selected quad in Selection mode");
        assert!(only.iter().all(|e| e.node_selected));
        assert_eq!(all.iter().filter(|e| e.node_selected).count(), 4);
    }

    #[test]
    fn face_selection_is_flagged_on_the_edges_around_it() {
        // Two quads side by side; selecting the right one flags its four edges.
        // The two meshes meet along x = 1 by *coincident* corners, not shared
        // ones, so that seam is two edges (drawn on top of each other) rather
        // than one: dedup is by vertex index, which is what the shading data
        // says the topology is.
        let mut m = Mesh::new();
        for (x, y) in [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)] {
            m.add_vertex(Vec3::new(x, y, 0.0));
        }
        for (x, y) in [(1.0, 0.0), (2.0, 0.0), (2.0, 1.0), (1.0, 1.0)] {
            m.add_vertex(Vec3::new(x, y, 0.0));
        }
        m.add_face(&[0, 1, 2, 3]);
        m.add_face(&[4, 5, 6, 7]);
        m.compute_flat_normals();

        let mut scene = Scene::new();
        let id = scene.create_mesh("strip", m);
        let mut sel = Selection::new();
        sel.level = SelectLevel::Face;
        sel.select_node(id);
        sel.select_elements(id, [1u32]);

        let edges = collect_edges(&mut scene, &sel, Wireframe::All, Vec3::new(1.0, 0.5, 5.0));
        assert_eq!(edges.len(), 8);
        assert_eq!(edges.iter().filter(|e| e.face_selected).count(), 4);
    }

    #[test]
    fn the_element_centroid_of_selected_vertices_is_their_middle() {
        let mut scene = Scene::new();
        let id = scene.create_mesh("quad", quad(0.0));
        let mut sel = Selection::new();
        sel.level = SelectLevel::Vertex;
        sel.select_node(id);
        sel.select_elements(id, [0u32, 2]);
        // Corners (-1,-1) and (1,1) average to the middle.
        let c = element_centroid(&mut scene, &sel).unwrap();
        assert!(c.length() < 1e-5, "got {c:?}");

        // Moving the node carries the centroid with it.
        scene.get_mut(id).unwrap().transform.translation = Vec3::new(3.0, 0.0, -1.0);
        let moved = element_centroid(&mut scene, &sel).unwrap();
        assert!((moved - Vec3::new(3.0, 0.0, -1.0)).length() < 1e-5, "got {moved:?}");
    }

    #[test]
    fn the_element_centroid_of_a_face_is_the_face_middle() {
        let mut scene = Scene::new();
        let id = scene.create_mesh("quad", quad(0.0));
        let mut sel = Selection::new();
        sel.level = SelectLevel::Face;
        sel.select_node(id);
        sel.select_elements(id, [0u32]);
        let c = element_centroid(&mut scene, &sel).unwrap();
        assert!(c.length() < 1e-5, "got {c:?}");

        // Two faces side by side: the centroid averages all eight corners.
        let mut m = Mesh::new();
        for (x, y) in [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)] {
            m.add_vertex(Vec3::new(x, y, 0.0));
        }
        for (x, y) in [(1.0, 0.0), (2.0, 0.0), (2.0, 1.0), (1.0, 1.0)] {
            m.add_vertex(Vec3::new(x, y, 0.0));
        }
        m.add_face(&[0, 1, 2, 3]);
        m.add_face(&[4, 5, 6, 7]);
        let mut scene = Scene::new();
        let strip = scene.create_mesh("strip", m);
        let mut sel = Selection::new();
        sel.level = SelectLevel::Face;
        sel.select_node(strip);
        sel.select_elements(strip, [0u32, 1]);
        let both = element_centroid(&mut scene, &sel).unwrap();
        assert!((both - Vec3::new(1.0, 0.5, 0.0)).length() < 1e-5, "got {both:?}");
    }

    #[test]
    fn the_element_centroid_is_none_without_a_sub_object_selection() {
        let mut scene = Scene::new();
        let id = scene.create_mesh("quad", quad(0.0));
        let mut sel = Selection::new();
        sel.level = SelectLevel::Vertex;
        sel.select_node(id);
        assert_eq!(element_centroid(&mut scene, &sel), None, "no vertices selected");

        sel.level = SelectLevel::Object;
        sel.select_elements(id, [0u32]);
        assert_eq!(element_centroid(&mut scene, &sel), None, "object level has none");
    }

    #[test]
    fn a_sub_object_selection_lists_each_corner_once() {
        let mut m = Mesh::new();
        for (x, y) in [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (2.0, 0.0), (2.0, 1.0)] {
            m.add_vertex(Vec3::new(x, y, 0.0));
        }
        m.add_face(&[0, 1, 2, 3]);
        // The second face reuses corners 2 and 3 along the shared vertical edge.
        m.add_face(&[3, 2, 4, 5]);
        let mut scene = Scene::new();
        let id = scene.create_mesh("hinge", m);
        let mut sel = Selection::new();
        sel.level = SelectLevel::Face;
        sel.select_node(id);
        sel.select_elements(id, [0u32, 1]);

        let targets = selected_vertices(&mut scene, &sel);
        assert_eq!(targets.len(), 1, "both faces belong to one node");
        let (node, positions) = &targets[0];
        assert_eq!(*node, id);
        assert_eq!(positions.len(), 6, "eight corners, two of them shared");
        // The positions are the drag-start state, i.e. the local vertices.
        assert!(positions.contains(&(2, Vec3::new(1.0, 1.0, 0.0))));

        // Vertex level lists exactly what was selected, and object level none.
        // (The app clears the element set when the level changes; here that is
        // done by hand so the two levels are not mixed in one set.)
        sel.clear_elements();
        sel.level = SelectLevel::Vertex;
        sel.select_elements(id, [5u32]);
        let only_five = selected_vertices(&mut scene, &sel);
        assert_eq!(only_five[0].1, vec![(5, Vec3::new(2.0, 1.0, 0.0))]);
        sel.level = SelectLevel::Object;
        assert!(selected_vertices(&mut scene, &sel).is_empty());
    }

    #[test]
    fn a_material_base_colour_is_its_diffuse() {
        let mut m = Material::default();
        m.diffuse = glam::Vec4::new(1.0, 0.0, 0.0, 1.0);
        assert_eq!(material_base(&m, false, false), [255, 0, 0, 255]);
    }

    #[test]
    fn a_material_base_colour_folds_opacity_into_alpha() {
        let mut m = Material::default();
        m.diffuse = glam::Vec4::new(0.0, 0.0, 1.0, 0.8);
        m.opacity = 0.5;
        let [r, g, b, a] = material_base(&m, false, false);
        assert_eq!((r, g, b), (0, 0, 255));
        assert_eq!(a, 102, "0.8 * 0.5 * 255, rounded");
    }

    #[test]
    fn a_material_base_colour_clamps_out_of_range_input() {
        let mut m = Material::default();
        m.diffuse = glam::Vec4::new(2.0, -1.0, 0.5, 4.0);
        m.opacity = 1.0;
        let [r, g, b, a] = material_base(&m, false, false);
        assert_eq!((r, g, b), (255, 0, 128));
        assert_eq!(a, 255);
    }

    #[test]
    fn selection_tints_a_material_base_colour_towards_the_highlight() {
        let mut m = Material::default();
        m.diffuse = glam::Vec4::new(1.0, 1.0, 1.0, 1.0);
        let plain = material_base(&m, false, false);
        let selected = material_base(&m, true, false);
        let face = material_base(&m, true, true);

        assert_eq!(plain, [255, 255, 255, 255]);
        // Blue-ish selection tint: blue stays highest, red is pulled down most.
        assert!(selected[2] > selected[1] && selected[1] > selected[0], "got {selected:?}");
        // Face selection is the warm highlight: red now leads.
        assert!(face[0] > face[1] && face[1] > face[2], "got {face:?}");
    }

    #[test]
    fn triangles_carry_the_slot_their_face_resolves_to() {
        let mut scene = Scene::new();
        let id = scene.create_mesh("quad", quad(0.0));
        // Unassigned: everything resolves to the default slot.
        let tris = collect_triangles(&mut scene, &Selection::new());
        assert!(tris.iter().all(|t| t.material == 0));

        // Assigned: every face of the node follows the assignment.
        scene.get_mut(id).unwrap().assign_material(5);
        let tris = collect_triangles(&mut scene, &Selection::new());
        assert!(!tris.is_empty());
        assert!(tris.iter().all(|t| t.material == 5));

        // A per-face slot list is respected face by face.
        let mut m = Mesh::new();
        for (x, y) in [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)] {
            m.add_vertex(Vec3::new(x, y, 0.0));
        }
        m.add_face(&[0, 1, 2]);
        m.add_face(&[0, 2, 3]);
        m.faces[1].material = 1;
        let mut scene = Scene::new();
        let id = scene.create_mesh("mixed", m);
        if let openforge_core::node::NodeKind::Mesh { material_slots, .. } =
            &mut scene.get_mut(id).unwrap().kind
        {
            *material_slots = vec![7, 9];
        }
        let slots: Vec<u32> = collect_triangles(&mut scene, &Selection::new())
            .iter()
            .map(|t| t.material)
            .collect();
        assert!(slots.contains(&7) && slots.contains(&9), "got {slots:?}");
    }

    #[test]
    fn hidden_and_frozen_meshes_are_left_out() {
        let mut scene = Scene::new();
        let hidden = scene.create_mesh("hidden", quad(0.0));
        let frozen = scene.create_mesh("frozen", quad(1.0));
        let shown = scene.create_mesh("shown", quad(2.0));
        scene.get_mut(hidden).unwrap().visible = false;
        scene.get_mut(frozen).unwrap().frozen = true;

        let sel = Selection::new();
        let edges = collect_edges(&mut scene, &sel, Wireframe::All, Vec3::new(0.0, 0.0, 8.0));
        assert_eq!(edges.len(), 4, "only the visible, unfrozen mesh contributes");
        let _ = shown;
    }

    // --- snapping -------------------------------------------------------------

    /// A camera five units down `+Z` looking at the origin, so a world point's
    /// screen position is easy to reason about.
    fn looking_down_z() -> Mat4 {
        Mat4::perspective_rh(45.0_f32.to_radians(), 1.0, 0.1, 100.0)
            * Mat4::look_at_rh(Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO, Vec3::Y)
    }

    #[test]
    fn snap_candidates_project_the_visible_unfrozen_vertices() {
        let mut scene = Scene::new();
        let quad_id = scene.create_mesh("quad", quad(0.0));
        let hidden = scene.create_mesh("hidden", quad(1.0));
        scene.get_mut(hidden).unwrap().visible = false;
        let frozen = scene.create_mesh("frozen", quad(1.5));
        scene.get_mut(frozen).unwrap().frozen = true;

        let size = [400.0, 400.0];
        let candidates = snap_candidates(&mut scene, &[], &[], looking_down_z(), size);
        assert_eq!(candidates.len(), 4, "only the visible, unfrozen quad offers vertices");
        assert!(candidates.iter().all(|c| c.node == quad_id));

        // Screen Y grows downwards: the corner at -Y lands below the centre,
        // the one at -X to its left.
        let corner = candidates
            .iter()
            .find(|c| c.world.x < 0.0 && c.world.y < 0.0)
            .expect("a corner towards -X, -Y exists");
        assert!(
            corner.screen[0] < 200.0 && corner.screen[1] > 200.0,
            "projected up-left of the centre: {:?}",
            corner.screen
        );
    }

    #[test]
    fn snap_candidates_leave_out_the_geometry_the_drag_carries() {
        let mut scene = Scene::new();
        let a = scene.create_mesh("a", quad(0.0));
        let b = scene.create_mesh("b", quad(1.0));
        let size = [400.0, 400.0];
        let vp = looking_down_z();

        // An object-level drag moves every vertex of its node, so that node
        // offers nothing to snap to.
        let without_a = snap_candidates(&mut scene, &[a], &[], vp, size);
        assert_eq!(without_a.len(), 4);
        assert!(without_a.iter().all(|c| c.node == b));

        // An element-level drag moves only the vertices it lists: the rest of
        // the same mesh stays snappable, which is what makes welding the
        // selected vertices onto their neighbours work.
        let some = snap_candidates(&mut scene, &[], &[(b, 0), (b, 3)], vp, size);
        assert_eq!(some.len(), 6, "four of A plus the two untouched corners of B");
        let b_vertices: Vec<u32> = some
            .iter()
            .filter(|c| c.node == b)
            .map(|c| c.vertex)
            .collect();
        assert!(!b_vertices.contains(&0) && !b_vertices.contains(&3));
        assert!(b_vertices.contains(&1) && b_vertices.contains(&2));
    }

    #[test]
    fn nearest_candidate_takes_the_closest_within_its_radius() {
        let mut scene = Scene::new();
        scene.create_mesh("quad", quad(0.0));
        let size = [400.0, 400.0];
        let candidates = snap_candidates(&mut scene, &[], &[], looking_down_z(), size);
        let corner = candidates
            .iter()
            .find(|c| c.world.x < 0.0 && c.world.y < 0.0)
            .expect("a corner towards -X, -Y exists");

        // Right on it, and just off it, are the same vertex.
        let hit = nearest_candidate(&candidates, corner.screen, 14.0).expect("the cursor is on it");
        assert_eq!(hit.vertex, corner.vertex);
        let near = [corner.screen[0] + 5.0, corner.screen[1] - 5.0];
        let hit = nearest_candidate(&candidates, near, 14.0).expect("five pixels away is close enough");
        assert_eq!(hit.vertex, corner.vertex);

        // Out of reach: nothing at all. (The viewport's corner is far from
        // every corner of the quad, which fills the middle of the frame.)
        assert!(nearest_candidate(&candidates, [0.0, 0.0], 14.0).is_none());
    }

    #[test]
    fn nearest_candidate_prefers_the_nearer_vertex_on_a_tie() {
        // Two vertices landing on exactly the same pixel, at different depths:
        // the one nearer the camera is the one meant.
        let on_the_pixel = |node, vertex, depth| SnapCandidate {
            node,
            vertex,
            world: Vec3::ZERO,
            screen: [100.0, 100.0],
            depth,
        };
        let candidates = vec![on_the_pixel(NodeId(7), 3, 0.5), on_the_pixel(NodeId(9), 0, 0.2)];
        let hit = nearest_candidate(&candidates, [100.0, 100.0], 14.0).expect("both are on the cursor");
        assert_eq!(hit.node, NodeId(9), "the nearer vertex wins the tie");
        assert_eq!(hit.vertex, 0);

        // Just an arm's length from the tie: still resolved the same way.
        let hit = nearest_candidate(&candidates, [105.0, 100.0], 14.0).expect("both are in reach");
        assert_eq!(hit.node, NodeId(9));
    }

    #[test]
    fn face_texture_resolves_the_materials_map_and_ignores_missing_ones() {
        let mut library = MaterialLibrary::new();
        let texture = library.add_texture(Texture::solid("tiny", 1, 1, [1, 2, 3, 4]));
        let mut mapped = Material::new("mapped");
        mapped.diffuse_map = Some(texture);
        let slot = library.add_material(mapped) as u32;
        // A map index the library cannot resolve is ignored rather than sampled.
        let mut orphan = Material::new("orphan");
        orphan.diffuse_map = Some(9);
        let orphan_slot = library.add_material(orphan) as u32;

        assert_eq!(face_texture(&library, slot), Some(texture));
        assert_eq!(face_texture(&library, 0), None, "the default material is untextured");
        assert_eq!(face_texture(&library, orphan_slot), None, "a dangling map is ignored");
        assert_eq!(
            face_texture(&library, 99),
            None,
            "an out-of-range slot falls back to the default material"
        );
    }

    #[test]
    fn triangles_carry_each_corners_uv() {
        let mut mesh = Mesh::new();
        for p in [Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::new(1.0, 1.0, 0.0)] {
            mesh.add_vertex(p);
        }
        mesh.uvs = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(0.0, 1.0),
        ];
        mesh.add_face(&[0, 1, 2, 3]);
        for (corner, uv) in mesh.faces[0].corners.iter_mut().zip(0..4) {
            corner.uv = uv;
        }
        let mut scene = Scene::new();
        scene.create_mesh("quad", mesh);

        let tris = collect_triangles(&mut scene, &Selection::new());
        assert_eq!(tris.len(), 2, "a quad fan-triangulates");
        assert_eq!(tris[0].v[0].uv, Vec2::new(0.0, 0.0));
        assert_eq!(tris[0].v[1].uv, Vec2::new(1.0, 0.0));
        assert_eq!(tris[0].v[2].uv, Vec2::new(1.0, 1.0));
        assert_eq!(tris[1].v[0].uv, Vec2::new(0.0, 0.0));
        assert_eq!(tris[1].v[2].uv, Vec2::new(0.0, 1.0));
    }
}

