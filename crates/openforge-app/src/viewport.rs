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
use openforge_materials::{MapKind, Material, MaterialLibrary};
use std::collections::HashMap;
use openforge_core::mesh::Mesh;
use openforge_core::scene::Scene;
use openforge_core::selection::{SelectLevel, Selection};
use openforge_core::NodeId;
use openforge_render::PickElement;
use std::collections::BTreeSet;

/// Where an orthographic view's eye sits along its view direction. The
/// projection ignores the distance (parallel rays), so the value only has to
/// keep the scene between the near and far planes.
const ORTHO_EYE: f32 = 500.0;

/// Which projection a viewport shows.
///
/// A modeller works in the six axis-aligned orthographic views as much as in
/// free perspective. Each is the same camera at a fixed pair of angles with a
/// different projection, so picking, the gizmo and the grid all keep working:
/// they only ever read [`Camera::view_proj`] and [`Camera::ray_from_ndc`].
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum ViewKind {
    /// The free orbit camera.
    #[default]
    Perspective,
    /// Looking along -Z, with +Y up and +X right: the side an object faces.
    Front,
    /// Looking along +Z.
    Back,
    /// Looking along +X, seen from -X.
    Left,
    /// Looking along -X, seen from +X.
    Right,
    /// Looking straight down, from above.
    Top,
    /// Looking straight up, from below.
    Bottom,
}

impl ViewKind {
    /// Every kind, in menu order, perspective first.
    pub const ALL: [ViewKind; 7] = [
        ViewKind::Perspective,
        ViewKind::Front,
        ViewKind::Back,
        ViewKind::Left,
        ViewKind::Right,
        ViewKind::Top,
        ViewKind::Bottom,
    ];

    /// What the menu, the viewport corner and the status bar call it.
    pub fn label(self) -> &'static str {
        match self {
            ViewKind::Perspective => "Perspective",
            ViewKind::Front => "Front",
            ViewKind::Back => "Back",
            ViewKind::Left => "Left",
            ViewKind::Right => "Right",
            ViewKind::Top => "Top",
            ViewKind::Bottom => "Bottom",
        }
    }

    /// The digit that selects it — `0` for perspective, then the six views on
    /// `1`–`6`, which is how a modeller reaches for them.
    pub fn hotkey(self) -> egui::Key {
        match self {
            ViewKind::Perspective => egui::Key::Num0,
            ViewKind::Front => egui::Key::Num1,
            ViewKind::Back => egui::Key::Num2,
            ViewKind::Left => egui::Key::Num3,
            ViewKind::Right => egui::Key::Num4,
            ViewKind::Top => egui::Key::Num5,
            ViewKind::Bottom => egui::Key::Num6,
        }
    }

    /// Whether the view is orthographic. Perspective is the only one that is
    /// not.
    pub fn is_ortho(self) -> bool {
        self != ViewKind::Perspective
    }

    /// The yaw and pitch the view sits at. `None` for perspective, which keeps
    /// wherever the user left it.
    pub fn angles(self) -> Option<(f32, f32)> {
        use std::f32::consts::{FRAC_PI_2, PI};
        match self {
            ViewKind::Perspective => None,
            // `Camera::eye` places the eye at
            // `(cos(yaw)cos(pitch), sin(pitch), sin(yaw)cos(pitch))`.
            ViewKind::Front => Some((FRAC_PI_2, 0.0)),
            ViewKind::Back => Some((-FRAC_PI_2, 0.0)),
            ViewKind::Right => Some((0.0, 0.0)),
            ViewKind::Left => Some((PI, 0.0)),
            ViewKind::Top => Some((0.0, FRAC_PI_2)),
            ViewKind::Bottom => Some((0.0, -FRAC_PI_2)),
        }
    }

    /// Which way is up on screen. A view looking straight down (or up) cannot
    /// use world up, which is parallel to the view direction.
    pub fn up(self) -> Vec3 {
        match self {
            ViewKind::Top => Vec3::NEG_Z,
            ViewKind::Bottom => Vec3::Z,
            _ => Vec3::Y,
        }
    }
}

/// A perspective orbit camera, or an axis-aligned orthographic view.
#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub target: Vec3,
    pub distance: f32,
    pub yaw: f32,
    pub pitch: f32,
    pub fov_y: f32,
    pub near: f32,
    pub far: f32,
    pub view: ViewKind,
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
            view: ViewKind::Perspective,
        }
    }
}

impl Camera {
    /// The unit direction from the target towards the eye.
    fn eye_dir(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        Vec3::new(cy * cp, sp, sy * cp)
    }

    /// How far back along the view direction the eye sits. An orthographic
    /// view puts it far enough back that the whole scene is in front of it;
    /// `distance` is then only the zoom.
    fn eye_distance(&self) -> f32 {
        if self.view.is_ortho() {
            ORTHO_EYE
        } else {
            self.distance
        }
    }

    pub fn eye(&self) -> Vec3 {
        self.target + self.eye_dir() * self.eye_distance()
    }

    /// The direction the camera looks in, from the eye towards the target.
    ///
    /// This is the normal of the screen plane, which is what the gizmo's centre
    /// handle drags in: a move there follows the cursor across the image.
    pub fn forward(&self) -> Vec3 {
        (self.target - self.eye()).normalize_or_zero()
    }

    /// How far away `p` is *as the projection sees it*, which is the depth
    /// anything drawn a constant size on screen has to be sized by.
    ///
    /// In perspective that is the distance from the eye. An orthographic view's
    /// eye stands a fixed [`ORTHO_EYE`] back — far enough to put the whole scene
    /// in front of the near plane — and its extent is set by `distance`
    /// instead, so that is the depth to size by, or the gizmo would be drawn
    /// a hundred times its size.
    pub fn eye_depth(&self, p: Vec3) -> f32 {
        if self.view.is_ortho() {
            self.distance
        } else {
            (p - self.eye()).length()
        }
    }

    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(self.eye(), self.target, self.view.up())
    }

    /// The projection: perspective, or an orthographic box with the extent the
    /// perspective view has at the same distance, so switching between them
    /// does not jump.
    pub fn proj_matrix(&self, aspect: f32) -> Mat4 {
        let aspect = aspect.max(0.01);
        if self.view.is_ortho() {
            let half_h = (self.distance * (self.fov_y * 0.5).tan()).max(0.01);
            let half_w = half_h * aspect;
            Mat4::orthographic_rh(-half_w, half_w, -half_h, half_h, 0.1, 2.0 * ORTHO_EYE)
        } else {
            Mat4::perspective_rh(self.fov_y, aspect, self.near, self.far)
        }
    }

    /// Switch the viewport to a projection. An orthographic view snaps the
    /// angles to exactly its own; going back to perspective keeps the angles,
    /// so the orbit carries on from where the view sits.
    pub fn set_view(&mut self, view: ViewKind) {
        if let Some((yaw, pitch)) = view.angles() {
            self.yaw = yaw;
            self.pitch = pitch;
        }
        self.view = view;
    }

    pub fn view_proj(&self, aspect: f32) -> Mat4 {
        self.proj_matrix(aspect) * self.view_matrix()
    }

    /// Orbit by pixel deltas. Orbiting out of an orthographic view turns the
    /// viewport back into a perspective one, from the same direction — an
    /// orthographic view has no perspective to rotate.
    pub fn orbit(&mut self, dx: f32, dy: f32) {
        self.view = ViewKind::Perspective;
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
    ///
    /// The origin is on the near plane and runs to the far one. In perspective
    /// the two points are the same view ray, and in an orthographic view they
    /// are parallel — both give the direction the cursor points in.
    pub fn ray_from_ndc(&self, ndc_x: f32, ndc_y: f32, aspect: f32) -> Ray {
        let inv = self.view_proj(aspect).inverse();
        let near = inv * Vec4::new(ndc_x, ndc_y, 0.0, 1.0);
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

/// The library texture one of a face's material's maps samples, if any.
///
/// A material names each of its maps by index; one whose texture the library
/// no longer holds is ignored, so a face can never sample past the end of the
/// texture list. Everything that reads a map goes through here — the CPU
/// painter samples through the indices, the GPU path turns them into vertex
/// layers, and `MapTexels` reads the same ones — so the two paths cannot
/// disagree about which faces are mapped.
pub fn face_map(library: &MaterialLibrary, material_slot: u32, kind: MapKind) -> Option<usize> {
    library
        .resolve(material_slot)
        .map(kind)
        .filter(|index| library.textures.get(*index).is_some())
}

/// The library texture a face's material samples as its diffuse map, if any.
///
/// Both viewports resolve through this one function — the CPU painter takes
/// the index as its texture array handle, the GPU path as the vertex's layer.
pub fn face_texture(library: &MaterialLibrary, material_slot: u32) -> Option<usize> {
    face_map(library, material_slot, MapKind::Diffuse)
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
        let sel_faces = selection.selected_faces(id).cloned();
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

/// A directly selected sub-object: a selected vertex's dot, a picked edge's
/// highlight.
///
/// Deliberately the hue a selected face's edges take, so "selected" reads as
/// one colour at every level.
pub const ELEMENT_SELECTED_SRGB: [u8; 3] = EDGE_FACE_SELECTED_SRGB;

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
        let sel_faces = selection.selected_faces(id).cloned();
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
                SelectLevel::Face => {
                    let Some(face) = mesh.faces.get(index as usize) else { continue };
                    for corner in &face.corners {
                        if let Some(p) = mesh.positions.get(corner.position as usize) {
                            sum += world.transform_point3(*p);
                            count += 1;
                        }
                    }
                }
                SelectLevel::Edge => {
                    let Some([a, b]) = mesh.edges().get(index as usize).copied() else { continue };
                    for &vi in &[a, b] {
                        if let Some(p) = mesh.positions.get(vi as usize) {
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
                SelectLevel::Face => {
                    let Some(face) = mesh.faces.get(index as usize) else { continue };
                    for corner in &face.corners {
                        indices.insert(corner.position);
                    }
                }
                SelectLevel::Edge => {
                    let Some([a, b]) = mesh.edges().get(index as usize).copied() else { continue };
                    indices.insert(a);
                    indices.insert(b);
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

/// Result of an edge pick: the node and the edge index into [`Mesh::edges`].
///
/// [`Mesh::edges`]: openforge_core::mesh::Mesh::edges
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct EdgeHit {
    pub node: NodeId,
    pub edge: u32,
    pub dist: f32,
}

/// Cast a ray and return the nearest edge within a world-space tolerance.
///
/// Edge indices are [`Mesh::edges`] indices, so a hit here names the same edge
/// the ID pass draws a line for. The distance is the ray parameter of the
/// closest point on the edge, exactly like [`VertexHit::dist`], and the nearest
/// one wins.
///
/// [`Mesh::edges`]: openforge_core::mesh::Mesh::edges
pub fn pick_edge(scene: &mut Scene, ray: Ray, tol_world: f32) -> Option<EdgeHit> {
    let candidates: Vec<(NodeId, Mesh)> = scene
        .iter()
        .filter(|n| n.is_mesh() && n.visible && !n.frozen)
        .filter_map(|n| n.mesh().map(|m| (n.id, m.clone())))
        .collect();

    let mut best: Option<EdgeHit> = None;
    for (id, mesh) in candidates {
        let Ok(world) = scene.world_matrix(id) else { continue };
        for (ei, [a, b]) in mesh.edges().iter().enumerate() {
            let pa = world.transform_point3(mesh.positions[*a as usize]);
            let pb = world.transform_point3(mesh.positions[*b as usize]);
            let seg = pb - pa;
            // The segment point closest to the ray *line*: minimise the
            // squared perpendicular distance, a quadratic in `s` whose
            // curvature `|seg|² - (seg·dir)²` vanishes only when the segment
            // is parallel to the ray. Clamping `s` to the segment and then
            // recomputing the perpendicular distance is the whole test.
            let u = pa - ray.origin;
            let along_seg = seg.dot(ray.direction);
            let denom = seg.length_squared() - along_seg * along_seg;
            let s = if denom > 0.0 {
                ((u.dot(ray.direction) * along_seg - u.dot(seg)) / denom).clamp(0.0, 1.0)
            } else {
                // Parallel: the distance varies linearly along the segment,
                // so one of its endpoints is the closer one.
                let at = |s: f32| {
                    let w = u + seg * s;
                    w.length_squared() - w.dot(ray.direction).powi(2)
                };
                if at(1.0) < at(0.0) { 1.0 } else { 0.0 }
            };
            let closest = pa + seg * s;
            let along = (closest - ray.origin).dot(ray.direction);
            if along < 0.0 {
                continue;
            }
            let perp = closest.distance(ray.at(along));
            if perp <= tol_world {
                let better = best.map(|b| along < b.dist).unwrap_or(true);
                if better {
                    best = Some(EdgeHit { node: id, edge: ei as u32, dist: along });
                }
            }
        }
    }
    best
}

/// True when two segments properly cross: each straddles the line of the
/// other, strictly on both sides.
///
/// Collinear and merely touching cases answer `false` — the marquee's
/// inclusive endpoint test is what catches a segment whose endpoint lies
/// exactly on the rectangle, and a segment that only touches a corner of it.
/// No epsilon is involved, so the answer is exact and the same on every run.
fn segments_cross(a: (f32, f32), b: (f32, f32), c: (f32, f32), d: (f32, f32)) -> bool {
    let cross = |o: (f32, f32), p: (f32, f32), q: (f32, f32)| {
        (p.0 - o.0) * (q.1 - o.1) - (p.1 - o.1) * (q.0 - o.0)
    };
    let ab_c = cross(a, b, c);
    let ab_d = cross(a, b, d);
    let cd_a = cross(c, d, a);
    let cd_b = cross(c, d, b);
    (ab_c > 0.0) != (ab_d > 0.0) && (cd_a > 0.0) != (cd_b > 0.0)
}

/// Every element of the given level whose projection falls inside a viewport
/// rectangle.
///
/// The rectangle is in the viewport's own screen space: `(0, 0)` is the
/// viewport's top-left corner and its units match `size`. This is the CPU
/// fallback for rectangle selection — it projects and tests, so unlike the GPU
/// ID pass it cannot tell occluded geometry from visible geometry and will
/// happily select through. Nodes that are hidden or frozen are skipped, like
/// every other CPU picker here.
///
/// Rules per level:
/// - `Object`: every mesh position inside the rectangle, so the caller can
///   deduplicate them into node ids.
/// - `Vertex`: every mesh position index inside the rectangle.
/// - `Face`: every face with at least one corner position inside the
///   rectangle.
/// - `Edge`: every edge whose projected segment meets the rectangle — either
///   endpoint inside (the inclusive test above), or the segment crossing one
///   of the rectangle's four sides. An endpoint that projects to nothing
///   (behind the camera) leaves its edge out, the same way a position behind
///   the camera is never inside at any level. Collinear and merely touching
///   cases may resolve either way; a segment whose endpoint sits exactly on
///   the rectangle counts through the inclusive test.
///
/// A point exactly on the rectangle's edge counts as inside. The result is
/// deterministic: nodes in scene order, indices ascending, each element once.
pub fn marquee_elements(
    scene: &mut Scene,
    vp: Mat4,
    size: [f32; 2],
    rect: egui::Rect,
    level: SelectLevel,
) -> Vec<PickElement> {
    let mut out = Vec::new();

    let candidates: Vec<(NodeId, Mesh)> = scene
        .iter()
        .filter(|n| n.is_mesh() && n.visible && !n.frozen)
        .filter_map(|n| n.mesh().map(|m| (n.id, m.clone())))
        .collect();

    for (id, mesh) in candidates {
        let Ok(world) = scene.world_matrix(id) else { continue };
        // Project every position once; all four rules below answer from this
        // scratch rather than re-projecting a position per face corner.
        let screen: Vec<Option<(f32, f32)>> = mesh
            .positions
            .iter()
            .map(|p| project(vp, world.transform_point3(*p), size).map(|(x, y, _)| (x, y)))
            .collect();
        let inside = |vi: u32| -> bool {
            match screen.get(vi as usize).copied().flatten() {
                Some((x, y)) => {
                    // Inclusive on both edges: a point exactly on the
                    // rectangle belongs to the selection.
                    x >= rect.min.x && x <= rect.max.x && y >= rect.min.y && y <= rect.max.y
                }
                None => false,
            }
        };
        match level {
            // Object level emits the positions, not one entry per node: the
            // caller folds them into node ids itself.
            SelectLevel::Object | SelectLevel::Vertex => {
                for vi in 0..mesh.positions.len() as u32 {
                    if inside(vi) {
                        out.push(PickElement::Vertex { node: id, vertex: vi });
                    }
                }
            }
            SelectLevel::Face => {
                for (fi, face) in mesh.faces.iter().enumerate() {
                    // One entry per face even when several corners are inside.
                    if face.corners.iter().any(|c| inside(c.position)) {
                        out.push(PickElement::Face { node: id, face: fi as u32 });
                    }
                }
            }
            SelectLevel::Edge => {
                // The rectangle's four sides, walked as a closed loop.
                let corners = [
                    (rect.min.x, rect.min.y),
                    (rect.max.x, rect.min.y),
                    (rect.max.x, rect.max.y),
                    (rect.min.x, rect.max.y),
                ];
                for (ei, [a, b]) in mesh.edges().iter().enumerate() {
                    let (Some(sa), Some(sb)) = (
                        screen.get(*a as usize).copied().flatten(),
                        screen.get(*b as usize).copied().flatten(),
                    ) else {
                        continue;
                    };
                    let crosses = (0..4).any(|k| {
                        segments_cross(sa, sb, corners[k], corners[(k + 1) % 4])
                    });
                    if inside(*a) || inside(*b) || crosses {
                        out.push(PickElement::Edge { node: id, edge: ei as u32 });
                    }
                }
            }
        }
    }
    out
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
    fn an_edge_selection_does_not_flag_the_edges_as_face_selected() {
        // Edge 0 and face 0 are different things: at the edge level the
        // overlay must colour the edges by the node alone.
        let mut scene = Scene::new();
        let id = scene.create_mesh("quad", quad(0.0));
        let mut sel = Selection::new();
        sel.select_node(id);
        sel.level = SelectLevel::Edge;
        sel.select_elements(id, [0u32]);

        let edges = collect_edges(&mut scene, &sel, Wireframe::All, Vec3::new(0.0, 0.0, 5.0));
        assert_eq!(edges.len(), 4);
        assert!(edges.iter().all(|e| e.node_selected), "the node is selected");
        assert!(
            edges.iter().all(|e| !e.face_selected),
            "edge 0 must not read as face 0"
        );

        // The same index at the face level does flag the outline, so the
        // assertion above is about the level and not about the number.
        sel.level = SelectLevel::Face;
        let edges = collect_edges(&mut scene, &sel, Wireframe::All, Vec3::new(0.0, 0.0, 5.0));
        assert!(edges.iter().all(|e| e.face_selected));
    }

    #[test]
    fn a_triangle_is_flagged_face_selected_only_at_the_face_level() {
        let mut scene = Scene::new();
        let id = scene.create_mesh("quad", quad(0.0));
        let mut sel = Selection::new();
        sel.select_node(id);
        sel.level = SelectLevel::Edge;
        sel.select_elements(id, [0u32]);

        let tris = collect_triangles(&mut scene, &sel);
        assert!(!tris.is_empty(), "the quad triangulates");
        assert!(tris.iter().all(|t| t.selected), "the node is selected");
        assert!(
            tris.iter().all(|t| !t.face_selected),
            "edge 0 must not tint face 0"
        );

        sel.level = SelectLevel::Face;
        let tris = collect_triangles(&mut scene, &sel);
        assert!(tris.iter().all(|t| t.face_selected));
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
    fn face_map_reads_every_kind_and_the_diffuse_shortcut_agrees() {
        let mut library = MaterialLibrary::new();
        let texture = library.add_texture(Texture::solid("tiny", 1, 1, [1, 2, 3, 4]));
        let missing = library.texture_count();
        let mut mapped = Material::new("mapped");
        mapped.diffuse_map = Some(texture);
        mapped.normal_map = Some(missing);
        mapped.specular_map = Some(texture);
        mapped.emissive_map = Some(texture);
        let slot = library.add_material(mapped) as u32;

        assert_eq!(face_map(&library, slot, MapKind::Diffuse), Some(texture));
        assert_eq!(
            face_map(&library, slot, MapKind::Normal),
            None,
            "a dangling map is ignored, not sampled"
        );
        assert_eq!(face_map(&library, slot, MapKind::Specular), Some(texture));
        assert_eq!(face_map(&library, slot, MapKind::Emissive), Some(texture));
        // The older accessor is the diffuse kind, so the UV editor and the
        // viewport agree on what \"the\" map is.
        assert_eq!(
            face_texture(&library, slot),
            face_map(&library, slot, MapKind::Diffuse)
        );
        // The default material wears nothing, whatever kind is asked for.
        for kind in MapKind::ALL {
            assert_eq!(face_map(&library, 0, kind), None, "{kind:?}");
        }
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

    // --- edge picking ---------------------------------------------------------

    /// A ray straight down `-Z` at `(x, y)`, five units up: the shortest way
    /// from the test camera to a quad lying in the XY plane.
    fn ray_down_z(x: f32, y: f32) -> Ray {
        Ray { origin: Vec3::new(x, y, 5.0), direction: Vec3::NEG_Z }
    }

    #[test]
    fn a_ray_picks_the_edge_it_meets_and_the_face_middle_picks_none() {
        let mut scene = Scene::new();
        let id = scene.create_mesh("quad", quad(0.0));
        // Edge 0 is the bottom side, from (-1, -1) to (1, -1): its midpoint
        // sits straight under this ray.
        let hit = pick_edge(&mut scene, ray_down_z(0.0, -1.0), 0.1).expect("the bottom side");
        assert_eq!((hit.node, hit.edge), (id, 0));
        assert!((hit.dist - 5.0).abs() < 1e-4, "dist {} is the ray parameter", hit.dist);

        // The middle of the quad is a whole unit from every side.
        assert_eq!(pick_edge(&mut scene, ray_down_z(0.0, 0.0), 0.5), None);
    }

    #[test]
    fn the_nearer_of_two_parallel_edges_wins() {
        let mut scene = Scene::new();
        // The far quad first, so scene order alone would answer with the
        // wrong node if distance were ignored.
        let far = scene.create_mesh("far", quad(0.0));
        scene.get_mut(far).unwrap().set_translation(Vec3::new(0.0, 0.0, -4.0));
        let near = scene.create_mesh("near", quad(0.0));

        let hit = pick_edge(&mut scene, ray_down_z(0.0, -1.0), 0.1).expect("both bottom sides");
        assert_eq!(hit.node, near, "z = 0 is nearer than z = -4");
        assert_eq!(hit.edge, 0);
        assert!((hit.dist - 5.0).abs() < 1e-4, "got {}", hit.dist);
    }

    #[test]
    fn the_closest_point_clamps_to_the_edges_endpoints() {
        let mut scene = Scene::new();
        let id = scene.create_mesh("quad", quad(0.0));
        // Edge 0 ends at (1, -1). This ray aims a little past that endpoint,
        // so the nearest point of the segment is the endpoint itself,
        // sqrt(0.02² + 0.06²) ≈ 0.063 away — inside the tolerance.
        let near = pick_edge(&mut scene, ray_down_z(1.02, -1.06), 0.1).expect("the endpoint");
        // Both sides meeting at the corner are that close, so the lower edge
        // index answers: the walk keeps the first on a tie.
        assert_eq!((near.node, near.edge), (id, 0), "got {near:?}");

        // Further out the same ray pattern is 0.5 from the endpoint, past the
        // tolerance. Without the clamp the point on the edge's *line* under
        // the ray (1.5, -1, 0) would be 0.06 away and would hit.
        assert_eq!(pick_edge(&mut scene, ray_down_z(1.5, -1.06), 0.1), None);
    }

    #[test]
    fn a_segment_behind_the_ray_origin_is_not_picked() {
        let mut scene = Scene::new();
        scene.create_mesh("quad", quad(0.0));
        // The ray starts past the quad (z = -5) and walks further away, so
        // the nearest point on every edge lies behind its origin, t < 0.
        let away = Ray { origin: Vec3::new(0.0, -1.0, -5.0), direction: Vec3::NEG_Z };
        assert_eq!(pick_edge(&mut scene, away, 0.5), None);
    }

    #[test]
    fn hidden_and_frozen_meshes_are_skipped_when_picking_edges() {
        let mut scene = Scene::new();
        // The hidden quad sits nearest the ray and the frozen one next to it,
        // so either leaking into the walk would win before the shown quad.
        let hidden = scene.create_mesh("hidden", quad(0.0));
        let node = scene.get_mut(hidden).unwrap();
        node.set_translation(Vec3::new(0.0, 0.0, 1.0));
        node.visible = false;
        let frozen = scene.create_mesh("frozen", quad(0.0));
        let node = scene.get_mut(frozen).unwrap();
        node.set_translation(Vec3::new(0.0, 0.0, 0.5));
        node.frozen = true;
        let shown = scene.create_mesh("shown", quad(0.0));

        let hit = pick_edge(&mut scene, ray_down_z(0.0, -1.0), 0.1).expect("the shown quad");
        assert_eq!((hit.node, hit.edge), (shown, 0));
        assert!((hit.dist - 5.0).abs() < 1e-4, "got {}", hit.dist);
    }

    #[test]
    fn an_empty_scene_and_a_mesh_without_faces_pick_no_edge() {
        let mut scene = Scene::new();
        let ray = ray_down_z(0.0, 0.0);
        assert_eq!(pick_edge(&mut scene, ray, 1.0), None, "nothing in the scene");

        // Positions alone are not edges: `edges()` walks faces.
        let mut bare = Mesh::new();
        bare.add_vertex(Vec3::new(-1.0, -1.0, 0.0));
        bare.add_vertex(Vec3::new(1.0, -1.0, 0.0));
        scene.create_mesh("bare", bare);
        assert_eq!(pick_edge(&mut scene, ray, 1.0), None, "a mesh with no faces");

        // A face too short for a polygon has no sides either.
        let mut sliver = Mesh::new();
        sliver.add_vertex(Vec3::new(0.0, 0.0, 0.0));
        sliver.add_vertex(Vec3::new(1.0, 0.0, 0.0));
        sliver.add_face(&[0, 1]);
        scene.create_mesh("sliver", sliver);
        assert_eq!(pick_edge(&mut scene, ray, 1.0), None, "a two-corner face is no edge");
    }

    #[test]
    fn the_tolerance_is_a_limit_and_a_ray_just_past_it_misses() {
        let mut scene = Scene::new();
        scene.create_mesh("quad", quad(0.0));
        // The bottom side is at y = -1; this ray runs about 0.1005 away.
        let ray = ray_down_z(0.0, -1.1005);
        assert_eq!(pick_edge(&mut scene, ray, 0.1), None, "just past the tolerance");
        let hit = pick_edge(&mut scene, ray, 0.2).expect("well inside a wider tolerance");
        assert_eq!(hit.edge, 0);
    }

    // --- marquee selection ----------------------------------------------------

    /// The down-`+Z` view of [`looking_down_z`], pulled back far enough that
    /// two quads side by side both fit inside the [400, 400] viewport.
    fn looking_down_z_from_afar() -> Mat4 {
        Mat4::perspective_rh(45.0_f32.to_radians(), 1.0, 0.1, 100.0)
            * Mat4::look_at_rh(Vec3::new(0.0, 0.0, 10.0), Vec3::ZERO, Vec3::Y)
    }

    /// A node's positions in world space, so a test can derive its screen box
    /// from the very points it expects inside.
    fn world_positions(scene: &mut Scene, id: NodeId) -> Vec<Vec3> {
        let Ok(world) = scene.world_matrix(id) else { return Vec::new() };
        let Ok(node) = scene.get(id) else { return Vec::new() };
        node.mesh()
            .map(|m| m.positions.iter().map(|p| world.transform_point3(*p)).collect())
            .unwrap_or_default()
    }

    /// The screen-space bounding box of `points`' projections, edges included.
    fn screen_box(vp: Mat4, size: [f32; 2], points: &[Vec3]) -> egui::Rect {
        let mut min = egui::pos2(f32::INFINITY, f32::INFINITY);
        let mut max = egui::pos2(f32::NEG_INFINITY, f32::NEG_INFINITY);
        for p in points {
            let (x, y, _) = project(vp, *p, size).expect("the points under test are in front");
            min.x = min.x.min(x);
            min.y = min.y.min(y);
            max.x = max.x.max(x);
            max.y = max.y.max(y);
        }
        egui::Rect::from_min_max(min, max)
    }

    /// Two quads side by side in the XY plane, each with its own four
    /// positions: the left one centred on x = -1.5, the right on x = 1.5.
    fn two_quads() -> (Scene, NodeId, NodeId) {
        let mut scene = Scene::new();
        let left = scene.create_mesh("left", quad(0.0));
        scene.get_mut(left).unwrap().set_translation(Vec3::new(-1.5, 0.0, 0.0));
        let right = scene.create_mesh("right", quad(0.0));
        scene.get_mut(right).unwrap().set_translation(Vec3::new(1.5, 0.0, 0.0));
        (scene, left, right)
    }

    /// Two triangles in the XY plane with a gap between them: face 0 uses
    /// positions 0..=2 on the left, face 1 uses positions 3..=5 on the right.
    fn two_gapped_triangles() -> Mesh {
        let mut m = Mesh::new();
        for (x, y) in [(-2.5, -1.0), (-1.5, -1.0), (-2.0, 1.0)] {
            m.add_vertex(Vec3::new(x, y, 0.0));
        }
        for (x, y) in [(1.5, -1.0), (2.5, -1.0), (2.0, 1.0)] {
            m.add_vertex(Vec3::new(x, y, 0.0));
        }
        m.add_face(&[0, 1, 2]);
        m.add_face(&[3, 4, 5]);
        m.compute_flat_normals();
        m
    }

    #[test]
    fn a_box_over_one_quad_selects_only_that_nodes_positions() {
        let (mut scene, _left, right) = two_quads();
        let size = [400.0, 400.0];
        let vp = looking_down_z_from_afar();
        let rect = screen_box(vp, size, &world_positions(&mut scene, right));

        let sel = marquee_elements(&mut scene, vp, size, rect, SelectLevel::Object);
        let expected: Vec<PickElement> =
            (0..4u32).map(|vi| PickElement::Vertex { node: right, vertex: vi }).collect();
        assert_eq!(sel, expected, "the right quad's four positions, once each");
    }

    #[test]
    fn vertex_level_selects_the_positions_inside_the_box() {
        let mut scene = Scene::new();
        let id = scene.create_mesh("quad", quad(0.0));
        let size = [400.0, 400.0];
        let vp = looking_down_z_from_afar();
        // A box around the quad's bottom half: corners 0 and 1 and a point
        // between them are what it must hold, corners 2 and 3 stay above it.
        let world = world_positions(&mut scene, id);
        let rect = screen_box(vp, size, &[world[0], world[1], Vec3::new(0.0, -0.5, 0.0)]);

        let sel = marquee_elements(&mut scene, vp, size, rect, SelectLevel::Vertex);
        assert_eq!(
            sel,
            vec![
                PickElement::Vertex { node: id, vertex: 0 },
                PickElement::Vertex { node: id, vertex: 1 },
            ]
        );
    }

    #[test]
    fn face_level_selects_only_the_face_under_the_box() {
        let mut scene = Scene::new();
        let id = scene.create_mesh("triangles", two_gapped_triangles());
        let size = [400.0, 400.0];
        let vp = looking_down_z_from_afar();
        // Only the right triangle's three corners: the left face's corners
        // lie far outside this box.
        let world = world_positions(&mut scene, id);
        let rect = screen_box(vp, size, &world[3..6]);

        let sel = marquee_elements(&mut scene, vp, size, rect, SelectLevel::Face);
        assert_eq!(sel, vec![PickElement::Face { node: id, face: 1 }]);
    }

    #[test]
    fn an_ngon_face_counts_once_even_when_several_corners_are_inside() {
        let mut mesh = Mesh::new();
        for i in 0..5 {
            let a = i as f32 / 5.0 * std::f32::consts::TAU;
            mesh.add_vertex(Vec3::new(a.cos(), a.sin(), 0.0));
        }
        mesh.add_face(&[0, 1, 2, 3, 4]);
        mesh.compute_flat_normals();
        let mut scene = Scene::new();
        let id = scene.create_mesh("pentagon", mesh);
        let size = [400.0, 400.0];
        let vp = looking_down_z_from_afar();
        // Four of the five corners are inside the box; the face is one hit.
        let world = world_positions(&mut scene, id);
        let rect = screen_box(vp, size, &world[0..4]);

        let sel = marquee_elements(&mut scene, vp, size, rect, SelectLevel::Face);
        assert_eq!(sel, vec![PickElement::Face { node: id, face: 0 }]);
    }

    #[test]
    fn a_box_outside_the_geometry_and_an_empty_box_select_nothing() {
        let mut scene = Scene::new();
        scene.create_mesh("quad", quad(0.0));
        let size = [400.0, 400.0];
        let vp = looking_down_z_from_afar();

        // The quad fills the middle of the frame; this box sits in the empty
        // top-left corner.
        let outside = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(10.0, 10.0));
        assert!(marquee_elements(&mut scene, vp, size, outside, SelectLevel::Object).is_empty());

        // An empty (zero-area) box contains nothing at all: its edges only
        // meet in a single point.
        let empty = egui::Rect::from_min_max(egui::pos2(300.0, 300.0), egui::pos2(300.0, 300.0));
        for level in [
            SelectLevel::Object,
            SelectLevel::Vertex,
            SelectLevel::Face,
            SelectLevel::Edge,
        ] {
            assert!(marquee_elements(&mut scene, vp, size, empty, level).is_empty());
        }
    }

    #[test]
    fn hidden_and_frozen_nodes_are_skipped_like_the_other_cpu_pickers() {
        let mut scene = Scene::new();
        // Three copies of one quad, so an unfiltered walk would see twelve
        // positions where only the visible, unfrozen node may contribute four.
        let shown = scene.create_mesh("shown", quad(0.0));
        let hidden = scene.create_mesh("hidden", quad(0.0));
        scene.get_mut(hidden).unwrap().visible = false;
        let frozen = scene.create_mesh("frozen", quad(0.0));
        scene.get_mut(frozen).unwrap().frozen = true;

        let size = [400.0, 400.0];
        let vp = looking_down_z_from_afar();
        let rect = screen_box(vp, size, &world_positions(&mut scene, shown));

        let sel = marquee_elements(&mut scene, vp, size, rect, SelectLevel::Vertex);
        let expected: Vec<PickElement> =
            (0..4u32).map(|vi| PickElement::Vertex { node: shown, vertex: vi }).collect();
        assert_eq!(sel, expected);
    }

    #[test]
    fn a_position_behind_the_camera_is_never_inside() {
        let mut scene = Scene::new();
        let front = scene.create_mesh("front", quad(0.0));
        // The camera sits at z = 10 looking towards -Z, so a quad moved to
        // z = 12 is behind the eye.
        let behind = scene.create_mesh("behind", quad(0.0));
        scene.get_mut(behind).unwrap().set_translation(Vec3::new(0.0, 0.0, 12.0));

        let size = [400.0, 400.0];
        let vp = looking_down_z_from_afar();
        for p in world_positions(&mut scene, behind) {
            assert!(project(vp, p, size).is_none(), "behind the eye: {p:?}");
        }

        // A box over the whole viewport, that would swallow those points if
        // behind-camera positions ever counted as inside.
        let rect = egui::Rect::from_min_max(egui::pos2(-100.0, -100.0), egui::pos2(500.0, 500.0));
        let sel = marquee_elements(&mut scene, vp, size, rect, SelectLevel::Object);
        let expected: Vec<PickElement> =
            (0..4u32).map(|vi| PickElement::Vertex { node: front, vertex: vi }).collect();
        assert_eq!(sel, expected, "only the quad in front of the eye");
    }

    #[test]
    fn the_result_is_deterministic_with_each_element_once_and_indices_ascending() {
        let mut scene = Scene::new();
        let id = scene.create_mesh("triangles", two_gapped_triangles());
        let size = [400.0, 400.0];
        let vp = looking_down_z_from_afar();
        let rect = screen_box(vp, size, &world_positions(&mut scene, id));

        let first = marquee_elements(&mut scene, vp, size, rect, SelectLevel::Vertex);
        let second = marquee_elements(&mut scene, vp, size, rect, SelectLevel::Vertex);
        assert_eq!(first, second, "two calls, one answer");

        assert_eq!(first.len(), 6, "all six positions, each once");
        let indices: Vec<u32> = first.iter().map(|e| e.index()).collect();
        assert_eq!(indices, (0..6).collect::<Vec<u32>>(), "indices ascending");
        assert!(first.iter().all(|e| e.node() == id));
    }

    #[test]
    fn the_endpoints_of_one_side_bring_that_side_and_the_sides_meeting_it() {
        // The box around both endpoints of the bottom side is that side's own
        // projected segment. Those endpoints are corners of the two sides
        // meeting it, so those come along through the inclusive endpoint rule;
        // the top side, the only edge with both endpoints outside and no
        // crossing, stays out.
        let mut scene = Scene::new();
        let id = scene.create_mesh("quad", quad(0.0));
        let size = [400.0, 400.0];
        let vp = looking_down_z_from_afar();

        // A box in the empty top-left of the frame selects no edge at all.
        let away = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(10.0, 10.0));
        assert!(marquee_elements(&mut scene, vp, size, away, SelectLevel::Edge).is_empty());

        let world = world_positions(&mut scene, id);
        let rect = screen_box(vp, size, &[world[0], world[1]]);
        let sel = marquee_elements(&mut scene, vp, size, rect, SelectLevel::Edge);
        assert_eq!(
            sel,
            vec![
                PickElement::Edge { node: id, edge: 0 },
                PickElement::Edge { node: id, edge: 1 },
                PickElement::Edge { node: id, edge: 3 },
            ]
        );
    }

    #[test]
    fn a_segment_crossing_the_box_selects_its_edge_with_no_endpoint_inside() {
        let mut scene = Scene::new();
        let id = scene.create_mesh("quad", quad(0.0));
        let size = [400.0, 400.0];
        let vp = looking_down_z_from_afar();
        let world = world_positions(&mut scene, id);

        // A box over the middle of the bottom side: both of the side's
        // endpoints are outside it, the segment crosses the box's left and
        // right sides, and the other three sides of the quad pass well clear.
        let (ax, ay, _) = project(vp, world[0], size).unwrap();
        let (bx, _, _) = project(vp, world[1], size).unwrap();
        let rect = egui::Rect::from_min_max(
            egui::pos2(ax + (bx - ax) * 0.4, ay - 15.0),
            egui::pos2(ax + (bx - ax) * 0.6, ay + 15.0),
        );
        assert!(ax < rect.min.x && bx > rect.max.x, "neither endpoint is inside");

        let sel = marquee_elements(&mut scene, vp, size, rect, SelectLevel::Edge);
        assert_eq!(sel, vec![PickElement::Edge { node: id, edge: 0 }]);
    }

    #[test]
    fn an_endpoint_exactly_on_the_box_edge_counts_through_the_inclusive_test() {
        let mut scene = Scene::new();
        let id = scene.create_mesh("quad", quad(0.0));
        let size = [400.0, 400.0];
        let vp = looking_down_z_from_afar();
        let world = world_positions(&mut scene, id);
        let (x, y, _) = project(vp, world[0], size).unwrap();

        // Vertex 0 sits exactly on the box's min corner: the two sides
        // meeting there are selected, each by an endpoint on the boundary.
        let on_the_edge =
            egui::Rect::from_min_max(egui::pos2(x, y), egui::pos2(x + 40.0, y + 40.0));
        let sel = marquee_elements(&mut scene, vp, size, on_the_edge, SelectLevel::Edge);
        assert_eq!(
            sel,
            vec![
                PickElement::Edge { node: id, edge: 0 },
                PickElement::Edge { node: id, edge: 3 },
            ]
        );

        // Pulled one pixel in, no endpoint is inside and the bottom side,
        // running along just under the box, crosses none of its sides.
        let pulled_in =
            egui::Rect::from_min_max(egui::pos2(x + 1.0, y + 1.0), egui::pos2(x + 41.0, y + 41.0));
        assert!(marquee_elements(&mut scene, vp, size, pulled_in, SelectLevel::Edge).is_empty());
    }

    #[test]
    fn an_endpoint_behind_the_camera_leaves_its_edges_out() {
        let mut scene = Scene::new();
        // One triangle with its last corner past the eye (the camera looks
        // from z = 10): only the edge between the two corners in front of it
        // can be selected, even by a box over the whole frame.
        let mut m = Mesh::new();
        m.add_vertex(Vec3::new(-2.0, -2.0, 0.0));
        m.add_vertex(Vec3::new(2.0, -2.0, 0.0));
        m.add_vertex(Vec3::new(0.0, 2.0, 12.0));
        m.add_face(&[0, 1, 2]);
        let id = scene.create_mesh("wedge", m);

        let size = [400.0, 400.0];
        let vp = looking_down_z_from_afar();
        assert!(project(vp, Vec3::new(0.0, 2.0, 12.0), size).is_none(), "behind the eye");
        let all = egui::Rect::from_min_max(egui::pos2(-100.0, -100.0), egui::pos2(500.0, 500.0));
        let sel = marquee_elements(&mut scene, vp, size, all, SelectLevel::Edge);
        assert_eq!(sel, vec![PickElement::Edge { node: id, edge: 0 }]);
    }

    #[test]
    fn hidden_and_frozen_nodes_are_skipped_at_edge_level() {
        let mut scene = Scene::new();
        // Three coincident quads: an unfiltered walk would report twelve
        // edges where only the visible, unfrozen node may contribute four.
        let shown = scene.create_mesh("shown", quad(0.0));
        let hidden = scene.create_mesh("hidden", quad(0.0));
        scene.get_mut(hidden).unwrap().visible = false;
        let frozen = scene.create_mesh("frozen", quad(0.0));
        scene.get_mut(frozen).unwrap().frozen = true;

        let size = [400.0, 400.0];
        let vp = looking_down_z_from_afar();
        let rect = egui::Rect::from_min_max(egui::pos2(-100.0, -100.0), egui::pos2(500.0, 500.0));
        let sel = marquee_elements(&mut scene, vp, size, rect, SelectLevel::Edge);
        let expected: Vec<PickElement> =
            (0..4u32).map(|ei| PickElement::Edge { node: shown, edge: ei }).collect();
        assert_eq!(sel, expected);
    }

    #[test]
    fn the_edge_result_is_deterministic_with_each_edge_once_and_indices_ascending() {
        let mut scene = Scene::new();
        let id = scene.create_mesh("triangles", two_gapped_triangles());
        let size = [400.0, 400.0];
        let vp = looking_down_z_from_afar();
        let rect = screen_box(vp, size, &world_positions(&mut scene, id));

        let first = marquee_elements(&mut scene, vp, size, rect, SelectLevel::Edge);
        let second = marquee_elements(&mut scene, vp, size, rect, SelectLevel::Edge);
        assert_eq!(first, second, "two calls, one answer");

        assert_eq!(first.len(), 6, "two triangles, six edges, each once");
        let indices: Vec<u32> = first.iter().map(|e| e.index()).collect();
        assert_eq!(indices, (0..6).collect::<Vec<u32>>(), "indices ascending");
        assert!(first.iter().all(|e| e.node() == id));
    }

    #[test]
    fn the_element_centroid_and_dragged_vertices_of_selected_edges() {
        // Two edges of the quad: the bottom side [0, 1] and the right side
        // [1, 2]. They share vertex 1.
        let mut scene = Scene::new();
        let id = scene.create_mesh("quad", quad(0.0));
        let mut sel = Selection::new();
        sel.level = SelectLevel::Edge;
        sel.select_node(id);
        sel.select_elements(id, [0u32, 1]);

        // The centroid averages the selected edges' four endpoint slots:
        // (-1, -1), (1, -1), (1, -1) again and (1, 1).
        let c = element_centroid(&mut scene, &sel).expect("two selected edges");
        let expected = (Vec3::new(-1.0, -1.0, 0.0)
            + Vec3::new(1.0, -1.0, 0.0)
            + Vec3::new(1.0, -1.0, 0.0)
            + Vec3::new(1.0, 1.0, 0.0))
            / 4.0;
        assert!((c - expected).length() < 1e-5, "got {c:?}");

        // A drag moves the three distinct endpoints, the shared one once.
        let targets = selected_vertices(&mut scene, &sel);
        assert_eq!(targets.len(), 1, "both edges belong to one node");
        let (node, positions) = &targets[0];
        assert_eq!(*node, id);
        assert_eq!(
            positions.iter().map(|(vi, _)| *vi).collect::<Vec<u32>>(),
            vec![0, 1, 2],
            "the shared corner is listed once"
        );
        assert!(positions.contains(&(0, Vec3::new(-1.0, -1.0, 0.0))));
    }

    #[test]
    fn a_point_exactly_on_the_box_edge_counts_as_inside() {
        let mut scene = Scene::new();
        let id = scene.create_mesh("quad", quad(0.0));
        let size = [400.0, 400.0];
        let vp = looking_down_z_from_afar();
        // Vertex 0 sits on the box's min corner — the top-left one, since
        // the box extends to the right and down from it.
        let (x, y, _) = project(vp, world_positions(&mut scene, id)[0], size).unwrap();
        let on_the_edge = egui::Rect::from_min_max(egui::pos2(x, y), egui::pos2(x + 50.0, y + 50.0));

        let sel = marquee_elements(&mut scene, vp, size, on_the_edge, SelectLevel::Vertex);
        assert_eq!(sel, vec![PickElement::Vertex { node: id, vertex: 0 }]);

        // One pixel inside the edge, the same corner is outside: the hit
        // above came from the inclusive test, not from rounding.
        let pulled_in =
            egui::Rect::from_min_max(egui::pos2(x + 1.0, y + 1.0), egui::pos2(x + 51.0, y + 51.0));
        assert!(marquee_elements(&mut scene, vp, size, pulled_in, SelectLevel::Vertex).is_empty());
    }
    #[test]
    fn every_view_names_itself_and_the_six_orthographic_ones_are_parallel() {
        assert_eq!(ViewKind::ALL.len(), 7);
        assert!(!ViewKind::Perspective.is_ortho());
        assert_eq!(ViewKind::Perspective.angles(), None, "the user keeps their angle");
        let mut labels = Vec::new();
        for view in ViewKind::ALL {
            labels.push(view.label());
            assert_eq!(view.is_ortho(), view != ViewKind::Perspective);
            assert_eq!(view.angles().is_some(), view.is_ortho());
        }
        assert_eq!(
            labels,
            vec!["Perspective", "Front", "Back", "Left", "Right", "Top", "Bottom"]
        );
        // Every orthographic view looks along one signed world axis, and no
        // two of them look the same way.
        let mut seen: Vec<[i32; 3]> = Vec::new();
        for view in ViewKind::ALL.iter().filter(|v| v.is_ortho()) {
            let mut camera = Camera::default();
            camera.set_view(*view);
            let dir = camera.forward();
            let along = dir.to_array().iter().filter(|c| c.abs() > 1e-4).count();
            assert_eq!(along, 1, "{view:?} looks along one axis, not {dir:?}");
            assert!((dir.length() - 1.0).abs() < 1e-4);
            seen.push(dir.to_array().map(|c| {
                if c > 1e-4 {
                    1
                } else if c < -1e-4 {
                    -1
                } else {
                    0
                }
            }));
        }
        seen.sort();
        seen.dedup();
        assert_eq!(seen.len(), 6, "one direction per view");
    }

    #[test]
    fn a_view_puts_the_world_axes_where_a_modeller_expects_them() {
        // Right and up on screen, read out of the view matrix, for each view.
        let cases: [(ViewKind, Vec3, Vec3); 6] = [
            (ViewKind::Front, Vec3::X, Vec3::Y),
            (ViewKind::Back, Vec3::NEG_X, Vec3::Y),
            (ViewKind::Right, Vec3::NEG_Z, Vec3::Y),
            (ViewKind::Left, Vec3::Z, Vec3::Y),
            (ViewKind::Top, Vec3::X, Vec3::NEG_Z),
            (ViewKind::Bottom, Vec3::X, Vec3::Z),
        ];
        for (view, right, up) in cases {
            let mut camera = Camera::default();
            camera.set_view(view);
            let m = camera.view_matrix();
            let screen_right = Vec3::new(m.x_axis.x, m.y_axis.x, m.z_axis.x);
            let screen_up = Vec3::new(m.x_axis.y, m.y_axis.y, m.z_axis.y);
            assert!(
                (screen_right - right).length() < 1e-4,
                "{view:?} right is {screen_right:?}, wanted {right:?}"
            );
            assert!(
                (screen_up - up).length() < 1e-4,
                "{view:?} up is {screen_up:?}, wanted {up:?}"
            );
        }
    }

    #[test]
    fn an_orthographic_view_has_no_perspective() {
        let mut camera = Camera::default();
        camera.set_view(ViewKind::Front);
        let vp = camera.view_proj(1.0);
        let size = [400.0, 400.0];

        // Two points on the same ray, one twice as far away: the same pixel
        // and a different depth. A perspective view would spread them apart.
        let (x1, y1, d1) = project(vp, Vec3::new(1.0, 0.5, -50.0), size).unwrap();
        let (x2, y2, d2) = project(vp, Vec3::new(1.0, 0.5, 50.0), size).unwrap();
        let (near_pixel, far_pixel) = ((x1, y1), (x2, y2));
        assert!(
            (x1 - x2).abs() < 1e-3 && (y1 - y2).abs() < 1e-3,
            "{near_pixel:?} vs {far_pixel:?}"
        );
        assert!((d1 - d2).abs() > 0.05, "depth still tells them apart");
        assert!(d1 > d2, "and the one further from the eye is the deeper");

        // The x axis runs across the screen with one scale, however far out.
        let (ax, _, _) = project(vp, Vec3::new(1.0, 0.0, 0.0), size).unwrap();
        let (bx, _, _) = project(vp, Vec3::new(2.0, 0.0, 0.0), size).unwrap();
        let centre = 200.0;
        assert!((ax - centre - (bx - centre) / 2.0).abs() < 1e-3, "sizes double with distance");

        // And it is flat: the whole extent is `distance * tan(fov/2)`.
        let half_h = 8.0 * (camera.fov_y * 0.5).tan();
        let (_, top, _) = project(vp, Vec3::new(0.0, half_h, 0.0), size).unwrap();
        assert!((top - 0.0).abs() < 0.5, "the top edge lands on the top pixel: {top}");
    }

    #[test]
    fn the_orthographic_picking_ray_runs_parallel_to_the_view() {
        let mut camera = Camera::default();
        camera.set_view(ViewKind::Top);
        let aspect = 1.6;
        let forward = camera.forward();
        for (ndc_x, ndc_y) in [(-0.5, 0.25), (0.0, 0.0), (0.75, -0.5)] {
            let ray = camera.ray_from_ndc(ndc_x, ndc_y, aspect);
            assert!(
                (ray.direction - forward).length() < 1e-4,
                "the ray is parallel to the view: {:?}",
                ray.direction
            );
            // The ray starts on the near plane, a hair in front of the eye.
            assert!(
                (ray.origin.y - camera.eye().y).abs() < 1.0,
                "starts at the eye: {:?}",
                ray.origin
            );
        }
        // Two rays are parallel but not the same line: x still separates them.
        let left = camera.ray_from_ndc(-0.5, 0.0, aspect);
        let right = camera.ray_from_ndc(0.5, 0.0, aspect);
        assert!((left.direction - right.direction).length() < 1e-5);
        assert!((left.origin - right.origin).length() > 0.5, "they are offset across the image");
    }

    #[test]
    fn a_view_keeps_the_scene_the_same_size_it_had_in_perspective() {
        // Switching projections at the same distance must not jump: the
        // orthographic extent is the perspective view's own extent, so the
        // point on the perspective view's top edge is on the top edge of the
        // orthographic one too.
        let mut camera = Camera::default();
        camera.yaw = std::f32::consts::FRAC_PI_2;
        camera.pitch = 0.0;
        let aspect = 1.5;
        let half_h = camera.distance * (camera.fov_y * 0.5).tan();
        let perspective = camera.view_proj(aspect);
        camera.set_view(ViewKind::Front);
        let ortho = camera.view_proj(aspect);

        // A point on the target's own plane, `half_h` above the middle: the
        // top edge of the frame at this distance and aspect.
        let on_the_top_edge = Vec3::new(0.0, half_h, camera.target.z);
        for (vp, name) in [(perspective, "perspective"), (ortho, "orthographic")] {
            let clip = vp * on_the_top_edge.extend(1.0);
            let ndc_y = clip.y / clip.w;
            assert!((ndc_y - 1.0).abs() < 1e-3, "{name} puts it at y = 1: {ndc_y}");
        }
    }

    #[test]
    fn orbiting_out_of_an_orthographic_view_returns_to_perspective() {
        let mut camera = Camera::default();
        camera.set_view(ViewKind::Top);
        assert_eq!(camera.view, ViewKind::Top);
        let eye = camera.eye();
        // Straight down by design, and far enough back for the whole scene.
        assert!((eye.x).abs() < 1e-4 && (eye.z).abs() < 1e-4);
        assert!(eye.y > 100.0, "the eye sits well above the scene: {eye:?}");

        camera.orbit(20.0, 5.0);
        assert_eq!(camera.view, ViewKind::Perspective, "orbiting leaves the fixed view");
        assert!(camera.pitch < std::f32::consts::FRAC_PI_2, "and tips off the pole");

        // Zooming an orthographic view changes the extent, not where the eye
        // is: the eye stays at the fixed distance while the extent shrinks.
        camera.set_view(ViewKind::Front);
        let before = camera.eye();
        camera.zoom(4.0);
        let after = camera.eye();
        assert!((before - after).length() < 1e-4, "the eye does not move: {before:?} {after:?}");
        let wide = {
            let mut c = camera;
            c.distance = 16.0;
            c
        };
        let size = [400.0, 400.0];
        let (narrow_x, _, _) = project(camera.view_proj(1.0), Vec3::X * 4.0, size).unwrap();
        let (wide_x, _, _) = project(wide.view_proj(1.0), Vec3::X * 4.0, size).unwrap();
        assert!(
            (narrow_x - 200.0).abs() > (wide_x - 200.0).abs(),
            "zooming in spreads the same point further from the middle"
        );
    }

}

