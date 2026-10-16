//! The OpenForge application: UI layout, viewport interaction, and the glue
//! between egui and the openforge-core scene model.

use crate::gizmo;
use crate::gpu_viewport::{self, GpuViewport};
use crate::snap::{self, SnapMode};
use crate::uv_editor::{self, UvEditor};
use crate::viewport::{self, Camera};
use eframe::egui;
use eframe::egui_wgpu::RenderState;
use glam::{Mat4, Vec2, Vec3, Vec4};
use openforge_animation::{Clip, Interp, Player, TransformTrack};
use openforge_core::math::Transform;
use openforge_core::mesh::{self, Mesh};
use openforge_core::node::NodeKind;
use openforge_core::scene::Scene;
use openforge_core::selection::{SelectLevel, Selection};
use openforge_core::subselect;
use openforge_core::tool::{
    DragState, Pointer, SelectTool, Tool, ToolContext, TransformMode,
};
use openforge_core::undo::UndoStack;
use openforge_core::NodeId;
use openforge_ops::MeshOps;
use openforge_materials::{Material, MaterialLibrary, Texture};
use openforge_plugin::PluginHost;
use openforge_render::{FaceShading, FrameStats, NodeMaterials, PickElement, TextureLayer};
use openforge_rig::{skin_positions, Bone, LocalTransform, Quat, Skeleton, SkinWeights};
use openforge_uv::UvOps;
use std::path::{Path, PathBuf};

/// The material library a new scene starts with: the crate's default grey at
/// slot 0 plus a few presets, so the materials panel is not empty on first run.
///
/// Slot `0` must stay the default (and untextured): it is what every
/// unassigned face resolves to. One preset carries a procedural checkerboard,
/// so texturing can be seen without importing an image first.
/// How long the "Spin Selected" clip runs, and how many keys carry the turn.
///
/// Four quarter-turn keys rather than a single 360° one: quaternion tracks
/// blend along the shortest arc, and a 360° end key would blend back the way
/// it came — the keys make the turn happen in the direction asked for.
const SPIN_SECONDS: f64 = 2.0;
const SPIN_STEPS: u32 = 4;

fn default_materials() -> MaterialLibrary {
    let mut library = MaterialLibrary::new();
    for (name, rgb, shininess, specular) in [
        ("Clay", [0.72, 0.70, 0.68], 16.0, 0.05),
        ("Painted Red", [0.78, 0.16, 0.14], 48.0, 0.20),
        ("Steel", [0.62, 0.65, 0.70], 160.0, 0.50),
        ("Gold", [0.86, 0.70, 0.28], 96.0, 0.60),
    ] {
        let mut material = Material::new(name);
        material.diffuse = Vec4::new(rgb[0], rgb[1], rgb[2], 1.0);
        material.shininess = shininess;
        material.specular = Vec3::splat(specular);
        library.add_material(material);
    }
    let checker = library.add_texture(Texture::checkerboard(
        "Checker",
        256,
        32,
        [236, 236, 236, 255],
        [64, 84, 116, 255],
    ));
    let mut textured = Material::new("Checker Test");
    textured.diffuse = Vec4::ONE;
    textured.shininess = 24.0;
    textured.specular = Vec3::splat(0.1);
    textured.diffuse_map = Some(checker);
    library.add_material(textured);
    library
}

/// One textured triangle as an egui mesh shape.
///
/// A mesh multiplies its texture by the vertex colour, so a caller passes the
/// face's *shaded* material colour. UVs address the texture directly: this
/// painter and the GPU both sample row 0 at `v = 0`, so neither flips.
fn textured_triangle(
    pts: [egui::Pos2; 3],
    uv: [Vec2; 3],
    color: egui::Color32,
    id: egui::TextureId,
) -> egui::Shape {
    let mut mesh = egui::Mesh::default();
    mesh.texture_id = id;
    for i in 0..3 {
        mesh.vertices.push(egui::epaint::Vertex {
            pos: pts[i],
            uv: egui::pos2(uv[i].x, uv[i].y),
            color,
        });
        mesh.indices.push(i as u32);
    }
    egui::Shape::mesh(mesh)
}

/// Which tool is active.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ActiveTool {
    Select,
    Move,
    Scale,
    Rotate,
}

/// The colour of a directly selected sub-object, as an egui colour.
fn element_selected_color() -> egui::Color32 {
    egui::Color32::from_rgb(
        viewport::ELEMENT_SELECTED_SRGB[0],
        viewport::ELEMENT_SELECTED_SRGB[1],
        viewport::ELEMENT_SELECTED_SRGB[2],
    )
}

/// The overlay mode a frame actually draws.
///
/// Edge level is where edges are picked, and a pick has a tolerance but a line
/// nobody can see is a line nobody can aim at: an *off* overlay falls back to
/// the selected meshes' outline there, the same context the vertex level gets
/// from its dots. The setting itself is untouched — the Z key still cycles it
/// and the View menu still reports it.
fn overlay_mode(level: SelectLevel, wireframe: viewport::Wireframe) -> viewport::Wireframe {
    if level == SelectLevel::Edge && wireframe == viewport::Wireframe::Off {
        viewport::Wireframe::Selection
    } else {
        wireframe
    }
}

pub struct OpenForgeApp {
    scene: Scene,
    selection: Selection,
    undo: UndoStack,
    camera: Camera,
    tool: ActiveTool,
    /// Scene state captured at the start of the current edit (for undo).
    edit_before: Option<Scene>,
    status: String,
    current_file: Option<PathBuf>,
    /// Cached triangles for the frame being painted (CPU viewport only).
    tris: Vec<viewport::Tri>,
    /// The wgpu viewport, when eframe is running the wgpu backend.
    gpu: Option<GpuViewport>,
    /// UV editor panel state.
    uv: UvEditor,
    /// The animation transport: the clips made in the app, the selected one,
    /// and the playback clock the timeline drives.
    anim: Player,
    /// Whether the timeline panel is open.
    anim_open: bool,
    /// Whether the viewport was interacted with this frame (drives repaint).
    dirty: bool,
    show_outliner: bool,
    show_properties: bool,
    /// Picks ignore occlusion ("Select Through"): the sub-object picks answer
    /// from the projection — the nearest element under the pointer, or every
    /// element a rectangle covers, geometry in front of it or not — instead of
    /// the occluding ID pass.
    select_through: bool,
    /// Deformer axis index (0 = X, 1 = Y, 2 = Z).
    deform_axis: usize,
    /// Deformer amount: taper factor, or twist/bend angle in radians.
    deform_amount: f32,
    /// Rig state for the active skinned mesh.
    rig_skeleton: Option<Skeleton>,
    rig_weights: Option<SkinWeights>,
    /// Bind (rest) pose vertex positions, skinned from on every pose change.
    rig_bind: Vec<Vec3>,
    rig_node: Option<NodeId>,
    rig_bone: usize,
    rig_angle: f32,
    rig_bones: usize,
    /// Live transform-gizmo state (hover, drag, this frame's handle layout).
    gizmo: gizmo::Gizmo,
    /// What a transform drag snaps to.
    snap: snap::Snap,
    /// Where a live snapped drag is landing, as an absolute screen position:
    /// what the marker is painted at for as long as that drag runs.
    snap_marker: Option<egui::Pos2>,
    /// A live rectangle-selection drag: its start and its current corner, in
    /// absolute screen positions. Painted as a marquee and resolved into a
    /// selection on release; `None` whenever no such drag is up.
    marquee: Option<(egui::Pos2, egui::Pos2)>,
    /// How much edge overlay the viewport draws over the shading.
    wireframe: viewport::Wireframe,
    /// The project's materials and textures.
    materials: MaterialLibrary,
    /// Bumped whenever the texture list changes, so both viewports know when
    /// their uploaded copies of it are stale.
    texture_epoch: u64,
    /// The texture list's epoch as the CPU painter last loaded it.
    cpu_texture_epoch: u64,
    /// One egui texture per library texture (lazily uploaded, `None` while
    /// unloaded), for the CPU painter and the UV editor's background.
    cpu_textures: Vec<Option<egui::TextureHandle>>,
    /// The texture list's epoch as the GPU viewport last uploaded it.
    gpu_texture_epoch: u64,
    /// Whether the UV editor paints the material's texture behind the UVs.
    uv_texture_background: bool,
    /// Loaded plugins: services, tools and format filters.
    plugins: PluginHost,
    /// The library entry the materials panel is editing.
    current_material: usize,
    /// Scene captured when a text field took focus, so committing the edit
    /// leaves one undo entry for the whole rename rather than one per keystroke.
    rename_before: Option<Scene>,
    /// In-progress free-form tool drag, held here because a drag spans frames.
    tool_drag: Option<DragState>,
}

impl OpenForgeApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        // Dark theme by default; egui respects the native theme otherwise.
        cc.egui_ctx.set_visuals(egui::Visuals::dark());
        // eframe hands us a live wgpu device when it runs the wgpu backend;
        // otherwise `gpu` stays `None` and the CPU painter takes over.
        Self::with_gpu(cc.wgpu_render_state.as_ref().map(GpuViewport::new))
    }

    /// The initial application state for a given viewport backend.
    ///
    /// Split out of [`Self::new`] because an [`eframe::CreationContext`] can
    /// only be built by eframe itself; the headless interaction tests at the
    /// bottom of this file come in through here.
    fn with_gpu(gpu: Option<GpuViewport>) -> Self {
        let mut scene = Scene::new();
        // Seed with a cube so the viewport isn't empty on first run.
        let cube = mesh::uv_sphere(1.0, 24, 16);
        scene.create_mesh("Sphere", cube);
        let backend = if gpu.is_some() { "wgpu" } else { "CPU" };
        Self {
            scene,
            selection: Selection::new(),
            undo: UndoStack::new(100),
            camera: Camera::default(),
            tool: ActiveTool::Select,
            edit_before: None,
            status: format!(
                "{backend} viewport. Left-drag selects; W/E/R show the transform gizmo; middle-drag orbits; scroll zooms."
            ),
            current_file: None,
            tris: Vec::new(),
            gpu,
            uv: UvEditor::default(),
            anim: Player::new(),
            anim_open: false,
            dirty: true,
            show_outliner: true,
            show_properties: true,
            select_through: false,
            deform_axis: 1,
            deform_amount: 0.5,
            rig_skeleton: None,
            rig_weights: None,
            rig_bind: Vec::new(),
            rig_node: None,
            rig_bone: 0,
            rig_angle: 0.0,
            rig_bones: 4,
            gizmo: gizmo::Gizmo::default(),
            snap: snap::Snap::default(),
            snap_marker: None,
            marquee: None,
            wireframe: viewport::Wireframe::default(),
            materials: default_materials(),
            texture_epoch: 0,
            cpu_texture_epoch: 0,
            cpu_textures: Vec::new(),
            gpu_texture_epoch: u64::MAX,
            uv_texture_background: true,
            plugins: openforge_plugin::builtin_host(),
            current_material: 0,
            rename_before: None,
            tool_drag: None,
        }
    }

    /// The tool for the current mode, built by the plugin host.
    ///
    /// The built-in tools are plugins like any other, so this is the seam a
    /// replacement `Move` would come through. The fallback only fires if the
    /// host somehow lacks a built-in.
    fn active_tool(&mut self) -> Box<dyn Tool> {
        let name = match self.tool {
            ActiveTool::Select => "Select",
            ActiveTool::Move => "Move",
            ActiveTool::Scale => "Scale",
            ActiveTool::Rotate => "Rotate",
        };
        self.plugins
            .make_tool(name)
            .unwrap_or_else(|| Box::new(SelectTool { additive: false }))
    }

    /// Run a closure that may edit the scene, capturing undo state around it.
    fn with_edit(&mut self, label: &str, f: impl FnOnce(&mut Self)) {
        let before = self.scene.clone();
        f(self);
        // Only record if something actually changed.
        if !self.scene.content_eq(&before) {
            self.undo.push(label, before, self.scene.clone());
            self.dirty = true;
        }
    }

    // --- Commands -----------------------------------------------------------------

    fn new_scene(&mut self) {
        self.scene = Scene::new();
        self.selection.clear();
        self.undo.clear();
        self.current_file = None;
        self.status = "New scene.".into();
        self.dirty = true;
    }

    fn create_primitive(&mut self, kind: &str) {
        let m = match kind {
            "Box" => primitive_box(1.0),
            "Sphere" => mesh::uv_sphere(1.0, 24, 16),
            "Cylinder" => primitive_cylinder(1.0, 2.0, 24),
            "Cone" => primitive_cone(1.0, 2.0, 24),
            "Plane" => primitive_plane(2.0, 2.0),
            _ => primitive_box(1.0),
        };
        let name = format!("{} {}", kind, self.scene.node_count() + 1);
        self.with_edit(&format!("Create {}", kind), |app| {
            let id = app.scene.create_mesh(name, m);
            app.selection.clear();
            app.selection.select_node(id);
        });
        self.status = format!("Created {}.", kind);
    }

    /// Duplicate every selected node, leaving the copies selected.
    ///
    /// The copy is independent (its mesh is cloned) but shallow: children are
    /// not duplicated, and a duplicated child becomes a root node.
    fn duplicate_selected(&mut self) {
        let ids: Vec<NodeId> = self.selection.selected_nodes().collect();
        if ids.is_empty() {
            self.status = "Select something to duplicate.".into();
            return;
        }
        let mut copies = Vec::new();
        self.with_edit("Duplicate", |app| {
            copies = ids
                .iter()
                .filter_map(|id| {
                    let mut copy = app.scene.get(*id).ok()?.clone();
                    // An invalid id makes the scene hand out a fresh one; keeping
                    // the original id would overwrite the node it came from.
                    copy.id = NodeId::INVALID;
                    copy.name = format!("{} copy", copy.name);
                    copy.children.clear();
                    copy.parent = None;
                    Some(app.scene.insert(copy))
                })
                .collect();
            app.selection.clear();
            for id in &copies {
                app.selection.select_node(*id);
            }
        });
        self.status = format!("Duplicated {} node(s).", copies.len());
    }

    fn delete_selected(&mut self) {
        let ids: Vec<NodeId> = self.selection.selected_nodes().collect();
        if ids.is_empty() {
            return;
        }
        self.with_edit("Delete", |app| {
            for id in ids {
                app.scene.remove_subtree(id);
            }
            app.selection.clear();
        });
        self.status = "Deleted selection.".into();
    }

    // --- Sub-object modeling operations -----------------------------------------

    /// The first selected mesh node, if any.
    fn active_mesh_node(&self) -> Option<NodeId> {
        self.selection
            .selected_nodes()
            .find(|id| self.scene.get(*id).map(|n| n.is_mesh()).unwrap_or(false))
    }

    /// The selected face indices (as `usize`) for a node.
    ///
    /// Empty unless the face level is the one being edited: element indices
    /// belong to the level that produced them (`Selection::selected_faces`),
    /// so the modeling ops that ask for faces refuse at any other level.
    fn selected_faces(&self, id: NodeId) -> Vec<usize> {
        self.selection
            .selected_faces(id)
            .map(|s| s.iter().map(|&x| x as usize).collect())
            .unwrap_or_default()
    }

    /// Run a modeling op on the active mesh's selected faces with undo. The
    /// closure receives the mesh and the selected faces and returns a status
    /// string describing what it did.
    fn with_mesh_faces(&mut self, label: &str, need_faces: bool, f: impl FnOnce(&mut Mesh, &[usize]) -> String) {
        let Some(id) = self.active_mesh_node() else {
            self.status = "Select a mesh first.".into();
            return;
        };
        let faces = self.selected_faces(id);
        if need_faces && faces.is_empty() {
            self.status = "Switch to Face level and select faces.".into();
            return;
        }
        let mut msg = String::new();
        self.with_edit(label, |app| {
            if let Ok(node) = app.scene.get_mut(id) {
                if let Some(mesh) = node.mesh_mut() {
                    msg = f(mesh, &faces);
                }
            }
        });
        if !msg.is_empty() {
            self.status = msg;
        }
    }

    fn extrude_selected(&mut self, distance: f32) {
        self.with_mesh_faces("Extrude", true, |mesh, faces| {
            match mesh.extrude_faces(faces, distance) {
                Some(r) => format!("Extruded {} face(s), {} wall(s).", r.cap_faces.len(), r.side_faces),
                None => "Extrude failed.".into(),
            }
        });
    }

    fn inset_selected(&mut self, amount: f32) {
        self.with_mesh_faces("Inset", true, |mesh, faces| {
            match mesh.inset_faces(faces, amount) {
                Some(caps) => format!("Inset {} face(s) -> {} cap(s).", faces.len(), caps.len()),
                None => "Inset failed.".into(),
            }
        });
    }

    fn poke_selected(&mut self) {
        self.with_mesh_faces("Poke", true, |mesh, faces| {
            let added = mesh.poke_faces(faces);
            format!("Poked {} face(s) -> +{} face(s).", faces.len(), added)
        });
    }

    fn subdivide_selected(&mut self) {
        self.with_mesh_faces("Subdivide", true, |mesh, faces| {
            let added = mesh.subdivide_faces(faces);
            format!("Subdivided {} face(s) -> +{} face(s).", faces.len(), added)
        });
    }

    fn delete_faces_selected(&mut self) {
        self.with_mesh_faces("Delete Faces", true, |mesh, faces| {
            let removed = mesh.delete_faces(faces);
            format!("Deleted {} face(s).", removed)
        });
    }

    /// Weld coincident vertices of the active mesh (whole-mesh, vertex level).
    fn weld_selected(&mut self, eps: f32) {
        let Some(id) = self.active_mesh_node() else {
            self.status = "Select a mesh first.".into();
            return;
        };
        let mut merged = 0usize;
        self.with_edit("Weld", |app| {
            if let Ok(node) = app.scene.get_mut(id) {
                if let Some(mesh) = node.mesh_mut() {
                    merged = mesh.weld_close(eps);
                }
            }
        });
        self.status = format!("Welded {} vertex/vertices (eps={:.3}).", merged, eps);
    }

    /// Chamfer the vertices of the selected faces (via `MeshOps`).
    fn bevel_selected(&mut self, amount: f32) {
        self.with_mesh_faces("Bevel", true, |mesh, faces| {
            let mut verts: Vec<u32> = Vec::new();
            for &fi in faces {
                if let Some(f) = mesh.faces.get(fi) {
                    for c in &f.corners {
                        if !verts.contains(&c.position) {
                            verts.push(c.position);
                        }
                    }
                }
            }
            if verts.is_empty() {
                return "Bevel: no vertices selected.".into();
            }
            let (bv, bf) = (mesh.vertex_count(), mesh.face_count());
            mesh.bevel_vertices(&verts, amount);
            format!(
                "Beveled {} vertex/vertices (+{} verts, +{} faces).",
                verts.len(),
                mesh.vertex_count() - bv,
                mesh.face_count() - bf,
            )
        });
    }

    /// Triangulate every face of the active mesh (via `MeshOps`).
    fn triangulate_selected(&mut self) {
        let Some(id) = self.active_mesh_node() else {
            self.status = "Select a mesh first.".into();
            return;
        };
        let (mut before, mut after) = (0usize, 0usize);
        self.with_edit("Triangulate", |app| {
            if let Ok(node) = app.scene.get_mut(id) {
                if let Some(mesh) = node.mesh_mut() {
                    before = mesh.face_count();
                    mesh.triangulate_all();
                    after = mesh.face_count();
                }
            }
        });
        self.status = format!("Triangulated: {} -> {} faces.", before, after);
    }

    /// Reverse the winding/normals of the active mesh (via `MeshOps`).
    fn flip_normals_selected(&mut self) {
        let Some(id) = self.active_mesh_node() else {
            self.status = "Select a mesh first.".into();
            return;
        };
        self.with_edit("Flip Normals", |app| {
            if let Ok(node) = app.scene.get_mut(id) {
                if let Some(mesh) = node.mesh_mut() {
                    mesh.flip_normals();
                }
            }
        });
        self.status = "Flipped normals.".into();
    }

    /// Duplicate the mesh mirrored across an axis plane (via `MeshOps`).
    fn mirror_selected(&mut self, axis: Vec3) {
        let Some(id) = self.active_mesh_node() else {
            self.status = "Select a mesh first.".into();
            return;
        };
        let mut verts = 0usize;
        self.with_edit("Mirror", |app| {
            if let Ok(node) = app.scene.get_mut(id) {
                if let Some(mesh) = node.mesh_mut() {
                    mesh.mirror_axis(axis, 0.001);
                    verts = mesh.vertex_count();
                }
            }
        });
        self.status = format!("Mirrored; mesh now has {} vertices.", verts);
    }

    /// Map the stored deformer axis index to a vector.
    fn deform_axis_vec(&self) -> Vec3 {
        match self.deform_axis {
            0 => Vec3::X,
            1 => Vec3::Y,
            _ => Vec3::Z,
        }
    }

    /// Apply a deformer (Taper / Twist / Bend) to the whole active mesh.
    fn deform_selected(&mut self, kind: &str) {
        let axis = self.deform_axis_vec();
        // Bend curls the length axis around a perpendicular one.
        let bend_axis = if axis == Vec3::X { Vec3::Y } else { Vec3::X };
        let amount = self.deform_amount;
        let Some(id) = self.active_mesh_node() else {
            self.status = "Select a mesh first.".into();
            return;
        };
        self.with_edit(kind, |app| {
            if let Ok(node) = app.scene.get_mut(id) {
                if let Some(mesh) = node.mesh_mut() {
                    match kind {
                        "Taper" => mesh.taper(axis, amount),
                        "Twist" => mesh.twist(axis, amount),
                        "Bend" => mesh.bend(axis, bend_axis, amount),
                        _ => {}
                    }
                }
            }
        });
        self.status = format!("{} applied along {:?} (amount {:.2}).", kind, axis, amount);
    }

    /// Combine the two first selected meshes with a boolean op (the second is
    /// consumed; the first takes the result).
    fn boolean_selected(&mut self, kind: &str) {
        let ids: Vec<NodeId> = self
            .selection
            .selected_nodes()
            .filter(|id| self.scene.get(*id).map(|n| n.is_mesh()).unwrap_or(false))
            .collect();
        if ids.len() < 2 {
            self.status = "Select two meshes for a boolean.".into();
            return;
        }
        let (a, b) = (ids[0], ids[1]);
        let (Some(ma), Some(mb)) = (
            self.scene.get(a).ok().and_then(|n| n.mesh()).cloned(),
            self.scene.get(b).ok().and_then(|n| n.mesh()).cloned(),
        ) else {
            self.status = "Could not read the two meshes.".into();
            return;
        };
        let result = match kind {
            "Union" => openforge_boolean::union(&ma, &mb),
            "Subtract" => openforge_boolean::subtract(&ma, &mb),
            "Intersect" => openforge_boolean::intersect(&ma, &mb),
            _ => return,
        };
        match result {
            Ok(mesh) => {
                let verts = mesh.vertex_count();
                self.with_edit(kind, |app| {
                    if let Ok(node) = app.scene.get_mut(a) {
                        if let Some(slot) = node.mesh_mut() {
                            *slot = mesh;
                        }
                    }
                    app.scene.remove_subtree(b);
                    app.selection.clear();
                    app.selection.select_node(a);
                });
                self.status = format!("{}: result has {} vertices.", kind, verts);
            }
            Err(e) => self.status = format!("Boolean failed: {}", e),
        }
    }

    // --- Rigging -----------------------------------------------------------------

    /// Build a vertical bone chain spanning the active mesh and auto-weight its
    /// vertices by height (smooth two-bone blend at each joint).
    fn rig_create(&mut self) {
        let Some(id) = self.active_mesh_node() else {
            self.status = "Select a mesh to rig.".into();
            return;
        };
        let Some(mesh) = self.scene.get(id).ok().and_then(|n| n.mesh()) else {
            self.status = "Could not read the mesh.".into();
            return;
        };
        let positions = mesh.positions.clone();
        if positions.is_empty() {
            self.status = "Mesh has no vertices.".into();
            return;
        }
        let (mut min_y, mut max_y) = (f32::INFINITY, f32::NEG_INFINITY);
        for p in &positions {
            min_y = min_y.min(p.y);
            max_y = max_y.max(p.y);
        }
        if !min_y.is_finite() || !max_y.is_finite() {
            min_y = 0.0;
            max_y = 0.0;
        }
        let bones = self.rig_bones.max(1);
        let span = (max_y - min_y).max(1e-4);
        let step = if bones >= 2 { span / (bones as f32 - 1.0) } else { 0.0 };

        // Chain along Y: root sits at min_y, each child advances by `step`.
        let mut sk = Skeleton::new();
        for i in 0..bones {
            let parent = if i == 0 { None } else { Some(i - 1) };
            let t = if i == 0 {
                LocalTransform::from_translation(Vec3::new(0.0, min_y, 0.0))
            } else {
                LocalTransform::from_translation(Vec3::new(0.0, step, 0.0))
            };
            let _ = sk.add_bone(Bone::with_transform(format!("bone_{i}"), parent, t));
        }

        // Auto-weight each vertex by its height along the chain.
        let mut w = SkinWeights::new(positions.len(), bones);
        for (vi, p) in positions.iter().enumerate() {
            let s = if step > 1e-6 {
                ((p.y - min_y) / step).clamp(0.0, (bones - 1) as f32)
            } else {
                0.0
            };
            let lower = (s.floor() as usize).min(bones - 1);
            let frac = (s - lower as f32).clamp(0.0, 1.0);
            if bones == 1 || frac < 1e-3 {
                let _ = w.set_weights(vi, &[(lower, 1.0)]);
            } else {
                let upper = (lower + 1).min(bones - 1);
                let _ = w.set_weights(vi, &[(lower, 1.0 - frac), (upper, frac)]);
            }
        }

        self.rig_bind = positions;
        self.rig_skeleton = Some(sk);
        self.rig_weights = Some(w);
        self.rig_node = Some(id);
        self.rig_bone = 0;
        self.rig_angle = 0.0;
        self.status = format!("Rig created: {} bones, auto-weighted.", bones);
    }

    /// Pose the selected bone and re-skin the mesh from its bind pose. This is
    /// a live preview driven by the angle slider, so it does not push undo.
    fn rig_apply_pose(&mut self) {
        let Some(node) = self.rig_node else { return };
        let Some(bone_count) = self.rig_skeleton.as_ref().map(|s| s.len()) else { return };
        if bone_count == 0 {
            return;
        }
        let bone = self.rig_bone.min(bone_count - 1);
        let angle = self.rig_angle;
        let newpos = {
            let sk = self.rig_skeleton.as_mut().unwrap();
            let _ = sk.set_pose_rotation(bone, Quat::from_rotation_x(angle));
            let matrices = sk.skinning_matrices();
            let w = self.rig_weights.as_ref().unwrap();
            skin_positions(&self.rig_bind, &matrices, w).ok()
        };
        if let Some(newpos) = newpos {
            if let Ok(n) = self.scene.get_mut(node) {
                if let Some(mesh) = n.mesh_mut() {
                    if mesh.positions.len() == newpos.len() {
                        mesh.positions = newpos;
                        mesh.compute_flat_normals();
                        self.dirty = true;
                    }
                }
            }
        }
    }

    /// Return the rig (and its mesh) to the rest pose.
    fn rig_reset(&mut self) {
        let Some(node) = self.rig_node else { return };
        self.rig_angle = 0.0;
        if let Some(sk) = self.rig_skeleton.as_mut() {
            sk.reset_pose();
        }
        let bind = self.rig_bind.clone();
        if let Ok(n) = self.scene.get_mut(node) {
            if let Some(mesh) = n.mesh_mut() {
                if mesh.positions.len() == bind.len() {
                    mesh.positions = bind;
                    mesh.compute_flat_normals();
                    self.dirty = true;
                }
            }
        }
        self.status = "Rig reset to rest pose.".into();
    }

    /// Restore the mesh to its bind pose and drop the rig.
    fn rig_clear(&mut self) {
        if let Some(node) = self.rig_node {
            let bind = self.rig_bind.clone();
            if let Ok(n) = self.scene.get_mut(node) {
                if let Some(mesh) = n.mesh_mut() {
                    if mesh.positions.len() == bind.len() {
                        mesh.positions = bind;
                        mesh.compute_flat_normals();
                        self.dirty = true;
                    }
                }
            }
        }
        self.rig_skeleton = None;
        self.rig_weights = None;
        self.rig_bind.clear();
        self.rig_node = None;
        self.rig_bone = 0;
        self.rig_angle = 0.0;
        self.status = "Rig cleared.".into();
    }

    fn undo_edit(&mut self) {
        if let Some(scene) = self.undo.undo() {
            self.scene = scene;
            self.selection.clear();
            self.dirty = true;
            self.status = format!(
                "Undo{}",
                self.undo.undo_label().map(|l| format!(": {}", l)).unwrap_or_default()
            );
        }
    }

    fn redo_edit(&mut self) {
        if let Some(scene) = self.undo.redo() {
            self.scene = scene;
            self.selection.clear();
            self.dirty = true;
            self.status = format!(
                "Redo{}",
                self.undo.redo_label().map(|l| format!(": {}", l)).unwrap_or_default()
            );
        }
    }

    /// Import one file through the plugin host's filter registry: a plugin
    /// that registers a new extension is picked up here with no app change.
    ///
    /// A path no filter knows is refused before anything is read, naming what
    /// the registry does know.
    fn import_file(&mut self, path: PathBuf) {
        let Some(label) = self.format_label(&path).map(str::to_string) else {
            self.status = format!(
                "Skipped {} — no filter knows it ({}).",
                Self::file_label(&path),
                self.known_extensions()
            );
            return;
        };
        match self.plugins.import(&path) {
            Ok(loaded) => {
                let count = loaded.mesh_count();
                self.with_edit("Import", |app| {
                    for md in loaded.meshes {
                        app.scene.create_mesh(md.name, md.mesh);
                    }
                });
                self.status = format!(
                    "Imported {} mesh(es) from {} ({label}).",
                    count,
                    Self::file_label(&path)
                );
            }
            Err(e) => self.status = format!("Import failed: {}", e),
        }
    }

    /// Import every path a drop delivered, in order.
    ///
    /// One file reports like a menu import; a pile ends on a summary of what
    /// the drop added, so the status line answers for the whole gesture.
    fn import_files(&mut self, paths: &[PathBuf]) {
        if paths.len() == 1 {
            self.import_file(paths[0].clone());
            return;
        }
        if paths.is_empty() {
            return;
        }
        let before = self.scene.node_count();
        for path in paths {
            self.import_file(path.clone());
        }
        let added = self.scene.node_count() - before;
        self.status = format!(
            "Dropped {} file(s) — imported {added} mesh node(s).",
            paths.len()
        );
    }

    /// Import whatever files the frame's input says were dropped on the
    /// window, through the same registry the File menu uses.
    ///
    /// Runs at the end of a frame, so the status a drop leaves is the frame's
    /// own — the viewport hint must not overwrite what it just did.
    fn import_dropped_files(&mut self, ctx: &egui::Context) {
        let dropped: Vec<PathBuf> = ctx.input(|i| {
            i.raw
                .dropped_files
                .iter()
                .filter_map(|file| file.path.clone())
                .collect()
        });
        if !dropped.is_empty() {
            self.import_files(&dropped);
        }
    }

    /// Export the one selected mesh node, naming the format in the status.
    ///
    /// The path's extension picks the filter, so one no filter knows is
    /// refused before anything is written; a selection holding more than one
    /// mesh node is refused too, rather than quietly writing one of them.
    fn export_selected(&mut self, path: PathBuf) {
        let selected: Vec<NodeId> = self.selection.selected_nodes().collect();
        let meshes = self.selected_mesh_nodes();
        match meshes.len() {
            1 => {}
            0 if selected.is_empty() => {
                self.status = "Select a mesh node to export.".into();
                return;
            }
            0 => {
                self.status = "Selected node has no mesh.".into();
                return;
            }
            n => {
                self.status = format!("Select one mesh to export — {n} mesh nodes selected.");
                return;
            }
        }
        let Some(label) = self.format_label(&path).map(str::to_string) else {
            self.status = format!(
                "Can't export to {} — no filter knows it ({}).",
                Self::file_label(&path),
                self.known_extensions()
            );
            return;
        };
        let id = meshes[0];
        let Some(mesh) = self.mesh_copy(id) else {
            self.status = "Selected node has no mesh.".into();
            return;
        };
        let name = self
            .scene
            .get(id)
            .map(|n| n.name.clone())
            .unwrap_or_else(|_| "mesh".into());
        let data = openforge_io::MeshData {
            name: name.clone(),
            mesh,
            material: None,
        };
        match self.plugins.export(&path, &data) {
            Ok(()) => {
                self.status = format!(
                    "Exported {name} to {} ({label}).",
                    Self::file_label(&path)
                )
            }
            Err(e) => self.status = format!("Export failed: {}", e),
        }
    }

    /// The label of the filter that would handle `path`'s extension.
    fn format_label(&self, path: &Path) -> Option<&str> {
        self.plugins.filters.for_path(path).map(|f| f.label())
    }

    /// Every extension the registry serves, as one readable list, for the
    /// status that turns a path down.
    fn known_extensions(&self) -> String {
        let mut extensions: Vec<&str> = self.plugins.filters.extensions().collect();
        extensions.sort_unstable();
        extensions.join(", ")
    }

    /// A path as the status should name it: the file's own name when it has
    /// one, the whole path otherwise.
    fn file_label(path: &Path) -> String {
        path.file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.display().to_string())
    }

    /// The registered formats as file-dialog filter groups, `(label,
    /// extensions)` per filter — what the import and export dialogs offer, so
    /// a plugin's format shows up in them too.
    fn format_groups(&self) -> Vec<(String, Vec<String>)> {
        self.plugins
            .filters
            .entries()
            .map(|(_, label, extensions)| {
                (
                    label.to_string(),
                    extensions.iter().map(|e| e.to_string()).collect(),
                )
            })
            .collect()
    }

    // --- Viewport interaction ----------------------------------------------------

    /// Normalized device coordinates (`-1..1`) for a screen position inside the
    /// viewport rect.
    fn viewport_ndc(pos: egui::Pos2, rect: egui::Rect) -> glam::Vec2 {
        glam::Vec2::new(
            ((pos.x - rect.left()) / rect.width()) * 2.0 - 1.0,
            1.0 - ((pos.y - rect.top()) / rect.height()) * 2.0,
        )
    }

    /// Per-face shading for the GPU path, resolved from the material library.
    ///
    /// `openforge-render` knows nothing about materials, so the app resolves
    /// each face's slot (plus the node/face selection tint and its texture)
    /// into a linear RGB — the offscreen target is sRGB and the shader works
    /// in linear — and hands the renderer the whole list. The GPU path
    /// therefore shades per face exactly like the CPU path, including meshes
    /// that mix several slots and textured faces.
    fn node_materials(&self) -> NodeMaterials {
        let mut materials = NodeMaterials::new();
        for node in self.scene.iter().filter(|n| n.is_mesh()) {
            let selected = self.selection.is_node_selected(node.id);
            let sel_faces = self.selection.selected_faces(node.id).cloned();
            let Some(mesh) = node.mesh() else { continue };
            let faces: Vec<FaceShading> = (0..mesh.faces.len())
                .map(|face| {
                    let face_selected = sel_faces
                        .as_ref()
                        .map(|s| s.contains(&(face as u32)))
                        .unwrap_or(false);
                    let slot = node.face_material_slot(face);
                    let base = viewport::material_base(
                        self.materials.resolve(slot),
                        selected,
                        face_selected,
                    );
                    FaceShading {
                        color: openforge_render::gpu::linear_from_srgb_u8([base[0], base[1], base[2]]),
                        layer: viewport::face_texture(&self.materials, slot),
                    }
                })
                .collect();
            materials = materials.with_faces(node.id, faces);
        }
        materials
    }

    // --- Materials -----------------------------------------------------------------

    /// The library's textures, resampled to the size the GPU's texture array
    /// uses and packed as its layers.
    ///
    /// Every layer of a texture array shares one size, and a `Texture` can be
    /// any size (a 4x4 procedural checker, a 2048px import), so the GPU copies
    /// are resampled to [`gpu_viewport::TEXTURE_LAYER_SIZE`]. Layer order is
    /// the library's texture order.
    fn texture_layers(&self) -> Vec<TextureLayer> {
        let size = gpu_viewport::TEXTURE_LAYER_SIZE;
        self.materials
            .textures
            .iter()
            .map(|texture| {
                let resized = texture.resized(size, size);
                TextureLayer::new(resized.width, resized.height, resized.pixels)
            })
            .collect()
    }

    /// Hand the library's textures to the GPU viewport, when it has not seen
    /// this revision of the library yet.
    fn sync_gpu_textures(&mut self, state: &RenderState) {
        if self.gpu_texture_epoch == self.texture_epoch {
            return;
        }
        let layers = self.texture_layers();
        if let Some(gpu) = self.gpu.as_mut() {
            gpu.set_textures(state, &layers);
        }
        self.gpu_texture_epoch = self.texture_epoch;
    }

    /// The egui texture for library texture `index`, uploaded on first use.
    ///
    /// The CPU painter and the UV editor's background both sample through
    /// these; a library edit (see [`Self::texture_epoch`]) drops the cache and
    /// the next frame re-uploads.
    fn cpu_texture(&mut self, ctx: &egui::Context, index: usize) -> Option<egui::TextureId> {
        if self.cpu_texture_epoch != self.texture_epoch {
            self.cpu_textures.clear();
            self.cpu_texture_epoch = self.texture_epoch;
        }
        self.cpu_textures.resize(self.materials.texture_count(), None);
        let slot = self.cpu_textures.get_mut(index)?;
        if slot.is_none() {
            let texture = self.materials.textures.get(index)?;
            if texture.is_empty() {
                return None;
            }
            let image = egui::ColorImage::from_rgba_unmultiplied(
                [texture.width as usize, texture.height as usize],
                &texture.pixels,
            );
            // Linear filtering and repeat addressing mirror the GPU sampler.
            let options = egui::TextureOptions {
                magnification: egui::TextureFilter::Linear,
                minification: egui::TextureFilter::Linear,
                wrap_mode: egui::TextureWrapMode::Repeat,
                ..Default::default()
            };
            *slot = Some(ctx.load_texture(texture.name.clone(), image, options));
        }
        slot.as_ref().map(|handle| handle.id())
    }

    /// Every library texture's egui handle, for a frame that draws several.
    fn cpu_texture_ids(&mut self, ctx: &egui::Context) -> Vec<Option<egui::TextureId>> {
        (0..self.materials.texture_count())
            .map(|index| self.cpu_texture(ctx, index))
            .collect()
    }

    /// Add a procedural checkerboard to the library and return its index.
    fn add_checker_texture(&mut self) -> usize {
        let name = format!("Checker {}", self.materials.texture_count() + 1);
        let index = self.materials.add_texture(Texture::checkerboard(
            name,
            256,
            32,
            [236, 236, 236, 255],
            [176, 96, 48, 255],
        ));
        self.texture_epoch += 1;
        self.dirty = true;
        index
    }

    /// Load a PNG or JPEG into the library, reporting the outcome.
    fn load_texture(&mut self, path: &std::path::Path) -> Option<usize> {
        match self.materials.try_load_texture_from_file(path) {
            Ok(index) => {
                self.texture_epoch += 1;
                self.dirty = true;
                let name = self.materials.textures[index].name.clone();
                self.status = format!("Loaded texture \"{name}\" ({} x {}).", self.materials.textures[index].width, self.materials.textures[index].height);
                Some(index)
            }
            Err(e) => {
                self.status = format!("Could not load {}: {e}", path.display());
                None
            }
        }
    }

    /// Point the material the panel is editing at a library texture.
    fn map_texture_to_current(&mut self, texture: usize) {
        let name = self.materials.textures.get(texture).map(|t| t.name.clone());
        if let Some(material) = self.materials.materials.get_mut(self.current_material) {
            material.diffuse_map = Some(texture);
            self.dirty = true;
            if let Some(name) = name {
                self.status = format!("Mapped \"{name}\" onto \"{}\".", material.name);
            }
        }
    }

    /// The texture the UV editor paints behind the UVs: the selected mesh's
    /// first face's material map, falling back to the material the panel is
    /// editing so a preset can be inspected before it is assigned.
    fn uv_background_texture(&self) -> Option<usize> {
        let slot = self
            .selection
            .selected_nodes()
            .next()
            .and_then(|id| self.scene.get(id).ok().map(|n| n.face_material_slot(0)))
            .unwrap_or(self.current_material as u32);
        viewport::face_texture(&self.materials, slot)
    }

    /// Assign a library material to every selected mesh node.
    fn assign_material_to_selection(&mut self, slot: u32) {
        let ids: Vec<NodeId> = self
            .selection
            .selected_nodes()
            .filter(|id| self.scene.get(*id).map(|n| n.is_mesh()).unwrap_or(false))
            .collect();
        if ids.is_empty() {
            self.status = "Select a mesh node to assign a material to.".into();
            return;
        }
        self.with_edit("Assign Material", |app| {
            for id in ids {
                if let Ok(node) = app.scene.get_mut(id) {
                    node.assign_material(slot);
                }
            }
        });
        let name = self.materials.resolve(slot).name.clone();
        self.status = format!("Assigned \"{name}\" to {} node(s).", self.selection.node_count());
    }

    /// Assign a library material to the selected faces of every mesh node,
    /// leaving each node's other faces on the material they already had.
    ///
    /// This is the sub-object counterpart of
    /// [`assign_material_to_selection`](Self::assign_material_to_selection): a
    /// mesh can carry several materials at once, resolved per face by both
    /// viewport paths.
    fn assign_material_to_faces(&mut self, slot: u32) {
        if self.selection.level != SelectLevel::Face || self.selection.total_element_count() == 0 {
            self.status = "Switch to Face level and select faces.".into();
            return;
        }
        // Only the faces that would actually change, so a no-op assignment
        // does not push an undo entry.
        let targets: Vec<(NodeId, Vec<u32>)> = self
            .selection
            .selected_nodes()
            .filter_map(|id| {
                let node = self.scene.get(id).ok()?;
                let faces: Vec<u32> = self
                    .selection
                    .selected_elements(id)?
                    .iter()
                    .copied()
                    .filter(|&f| node.face_material_slot(f as usize) != slot)
                    .collect();
                if faces.is_empty() {
                    None
                } else {
                    Some((id, faces))
                }
            })
            .collect();
        if targets.is_empty() {
            self.status = "Those faces already use that material.".into();
            return;
        }
        let name = self.materials.resolve(slot).name.clone();
        let mut changed = 0;
        self.with_edit("Assign Material to Faces", |app| {
            for (id, faces) in &targets {
                if let Ok(node) = app.scene.get_mut(*id) {
                    if node.assign_material_to_faces(faces, slot) {
                        changed += 1;
                    }
                }
            }
        });
        let faces = targets.iter().map(|(_, f)| f.len()).sum::<usize>();
        if changed == 0 {
            self.status = "Select a mesh node to assign a material to.".into();
        } else {
            self.status = format!("Assigned \"{name}\" to {faces} face(s).");
        }
    }

    // --- Selection tools -------------------------------------------------------

    /// The word the selection statuses use for a level's elements.
    fn element_word(level: SelectLevel) -> &'static str {
        match level {
            SelectLevel::Object => "node",
            SelectLevel::Vertex => "vertex",
            SelectLevel::Edge => "edge",
            SelectLevel::Face => "face",
        }
    }

    /// The mesh nodes a Select action applies to: every selected mesh node.
    ///
    /// Sub-object selections belong to a node, so the actions that step one
    /// run on the meshes being edited, exactly like the modeling ops.
    fn selected_mesh_nodes(&self) -> Vec<NodeId> {
        self.selection
            .selected_nodes()
            .filter(|id| self.scene.get(*id).map(|n| n.is_mesh()).unwrap_or(false))
            .collect()
    }

    /// The node's mesh, cloned, for maths that reads it while the selection is
    /// updated.
    fn mesh_copy(&self, id: NodeId) -> Option<openforge_core::mesh::Mesh> {
        self.scene.get(id).ok().and_then(|n| n.mesh().cloned())
    }

    /// "Select All — 12 face(s) selected."
    fn selection_status(&self, action: &str) -> String {
        format!(
            "{action} — {} {}(s) selected.",
            self.selection.total_element_count(),
            Self::element_word(self.selection.level)
        )
    }

    /// Select every element of the current level.
    ///
    /// At the object level that is every visible node; at a sub-object level
    /// every element of the meshes being edited, through
    /// [`openforge_core::subselect::all_elements`] — the same lists the
    /// pickers answer from, so "all" is exactly what can be picked.
    fn select_all_elements(&mut self) {
        if self.selection.level == SelectLevel::Object {
            let ids: Vec<NodeId> = self
                .scene
                .iter()
                .filter(|n| n.visible)
                .map(|n| n.id)
                .collect();
            for id in ids {
                self.selection.select_node(id);
            }
            self.status = format!("Select All — {} node(s) selected.", self.selection.node_count());
            return;
        }
        let ids = self.selected_mesh_nodes();
        if ids.is_empty() {
            self.status = "Select a mesh first.".into();
            return;
        }
        let level = self.selection.level;
        for id in ids {
            let all = self.mesh_copy(id).map(|m| subselect::all_elements(&m, level));
            if let Some(all) = all {
                self.selection.set_elements(id, all);
            }
        }
        self.status = self.selection_status("Select All");
    }

    /// Drop the element selection, keeping the nodes being edited.
    fn clear_element_selection(&mut self) {
        if self.selection.level == SelectLevel::Object {
            self.selection.clear();
            self.status = "Select None — nothing selected.".into();
            return;
        }
        self.selection.clear_elements();
        self.status = format!(
            "Select None — no {}(s) selected.",
            Self::element_word(self.selection.level)
        );
    }

    /// Flip the element selection at the current level.
    fn invert_element_selection(&mut self) {
        if self.selection.level == SelectLevel::Object {
            let ids: Vec<NodeId> = self
                .scene
                .iter()
                .filter(|n| n.visible)
                .map(|n| n.id)
                .collect();
            for id in ids {
                self.selection.toggle_node(id);
            }
            self.status = format!("Invert — {} node(s) selected.", self.selection.node_count());
            return;
        }
        let ids = self.selected_mesh_nodes();
        if ids.is_empty() {
            self.status = "Select a mesh first.".into();
            return;
        }
        let level = self.selection.level;
        for id in ids {
            let current = self.selection.selected_elements(id).cloned().unwrap_or_default();
            let flipped = self
                .mesh_copy(id)
                .map(|m| subselect::invert_selection(&m, level, &current));
            if let Some(flipped) = flipped {
                self.selection.set_elements(id, flipped);
            }
        }
        self.status = self.selection_status("Invert");
    }

    /// Grow the element selection to everything adjacent to it.
    fn grow_element_selection(&mut self) {
        self.step_element_selection(true);
    }

    /// Shrink the element selection to what no outside neighbour touches.
    fn shrink_element_selection(&mut self) {
        self.step_element_selection(false);
    }

    /// One grow or shrink step on every mesh being edited.
    ///
    /// The object level has no topology to step along, and an empty element
    /// selection has no seed, so both are status messages rather than an
    /// error: the frame simply keeps showing what it showed.
    fn step_element_selection(&mut self, grow: bool) {
        if self.selection.level == SelectLevel::Object {
            self.status = "Growing and shrinking work at the vertex, edge and face levels.".into();
            return;
        }
        let ids = self.selected_mesh_nodes();
        if ids.is_empty() {
            self.status = "Select a mesh first.".into();
            return;
        }
        let level = self.selection.level;
        for id in ids {
            let current = self.selection.selected_elements(id).cloned().unwrap_or_default();
            let next = self.mesh_copy(id).map(|m| {
                if grow {
                    subselect::grow_selection(&m, level, &current)
                } else {
                    subselect::shrink_selection(&m, level, &current)
                }
            });
            if let Some(next) = next {
                self.selection.set_elements(id, next);
            }
        }
        self.status = self.selection_status(if grow { "Grow" } else { "Shrink" });
    }

    /// Add the edge loop through every selected edge (edge level only).
    fn select_edge_loops(&mut self) {
        self.walk_edge_selection(true);
    }

    /// Add the edge ring through every selected edge.
    fn select_edge_rings(&mut self) {
        self.walk_edge_selection(false);
    }

    /// Add the loop (or ring) of every selected edge to the selection.
    ///
    /// An edge-level walk, so it refuses at the other levels; each selected
    /// edge contributes its chain and the picked edges stay picked, which
    /// makes a second press a no-op.
    fn walk_edge_selection(&mut self, loop_walk: bool) {
        let name = if loop_walk { "Loop" } else { "Ring" };
        if self.selection.level != SelectLevel::Edge {
            self.status = format!("{name} is an edge-level walk — switch to the edge level first.");
            return;
        }
        let ids = self.selected_mesh_nodes();
        if ids.is_empty() {
            self.status = "Select a mesh first.".into();
            return;
        }
        let mut walked = false;
        for id in ids {
            let current = self.selection.selected_elements(id).cloned().unwrap_or_default();
            if current.is_empty() {
                continue;
            }
            walked = true;
            if let Some(mesh) = self.mesh_copy(id) {
                let mut out = current.clone();
                for &edge in &current {
                    let chain = if loop_walk {
                        subselect::edge_loop(&mesh, edge)
                    } else {
                        subselect::edge_ring(&mesh, edge)
                    };
                    out.extend(chain);
                }
                self.selection.set_elements(id, out);
            }
        }
        if !walked {
            self.status = "Select an edge to walk from.".into();
            return;
        }
        self.status = self.selection_status(name);
    }

    /// Flip **Select Through**: with it on, the picks stop asking the ID pass
    /// and answer from the projection instead, so occluded elements can be
    /// picked.
    fn toggle_select_through(&mut self) {
        self.select_through = !self.select_through;
        self.status = if self.select_through {
            "Select Through on — picks ignore occlusion (P).".into()
        } else {
            "Select Through off — picks follow the drawn frame (P).".into()
        };
    }

    /// Show or hide one node, the outliner's visibility box.
    ///
    /// The same scene edit the H shortcut makes, scaled to one node: one undo
    /// step, and hiding drops the node from the selection the way H does.
    fn set_node_visible(&mut self, id: NodeId, visible: bool) {
        let name = match self.scene.get(id) {
            Ok(node) => node.name.clone(),
            Err(_) => return,
        };
        self.with_edit(if visible { "Show Node" } else { "Hide Node" }, |app| {
            if let Ok(node) = app.scene.get_mut(id) {
                node.visible = visible;
            }
        });
        if !visible {
            self.selection.deselect_node(id);
        }
        self.status = if visible {
            format!("Shown {name}.")
        } else {
            format!("Hidden {name}.")
        };
    }

    /// Hide every selected node and drop it from the selection.
    ///
    /// Hiding is a scene edit — the undo stack records it — and the hidden
    /// node leaves the selection, so the gizmo does not sit on something the
    /// frame no longer draws.
    fn hide_selected_nodes(&mut self) {
        let ids: Vec<NodeId> = self.selection.selected_nodes().collect();
        if ids.is_empty() {
            self.status = "Select a node to hide.".into();
            return;
        }
        self.with_edit("Hide Selected", |app| {
            for id in &ids {
                if let Ok(node) = app.scene.get_mut(*id) {
                    node.visible = false;
                }
            }
        });
        for id in &ids {
            self.selection.deselect_node(*id);
        }
        self.status = format!("Hidden {} node(s).", ids.len());
    }

    /// Show only the selected nodes: everything else is hidden.
    ///
    /// The selection is kept — this is the "work on this one piece" view — and
    /// the whole thing is one undo step.
    fn isolate_selected_nodes(&mut self) {
        let keep: Vec<NodeId> = self.selection.selected_nodes().collect();
        if keep.is_empty() {
            self.status = "Select a node to isolate.".into();
            return;
        }
        let all: Vec<NodeId> = self.scene.iter().map(|n| n.id).collect();
        self.with_edit("Isolate", |app| {
            for id in &all {
                if let Ok(node) = app.scene.get_mut(*id) {
                    node.visible = keep.contains(id);
                }
            }
        });
        let hidden = all.len() - keep.len();
        self.status = format!("Isolate — {} node(s) visible, {hidden} hidden.", keep.len());
    }

    /// Make every hidden node visible again.
    fn unhide_all_nodes(&mut self) {
        let hidden: Vec<NodeId> = self
            .scene
            .iter()
            .filter(|n| !n.visible)
            .map(|n| n.id)
            .collect();
        if hidden.is_empty() {
            self.status = "Every node is already visible.".into();
            return;
        }
        self.with_edit("Unhide All", |app| {
            for id in &hidden {
                if let Ok(node) = app.scene.get_mut(*id) {
                    node.visible = true;
                }
            }
        });
        self.status = format!("Unhidden {} node(s).", hidden.len());
    }

    /// Whether any mesh node in the scene uses `slot`.
    fn material_in_use(&self, slot: u32) -> bool {
        self.scene.iter().filter(|n| n.is_mesh()).any(|n| {
            (0..n.mesh().map(|m| m.faces.len()).unwrap_or(0)).any(|f| n.face_material_slot(f) == slot)
        })
    }

    /// Remove the library's last material, if nothing uses it.
    ///
    /// Only the last entry can be removed: deleting one in the middle would
    /// renumber every slot above it and silently reassign geometry. Material
    /// edits live outside the undo stack (which snapshots the scene), so a
    /// delete that could dangle a slot is refused rather than half-undone.
    fn delete_current_material(&mut self) {
        let index = self.current_material;
        let last = self.materials.material_count().saturating_sub(1);
        if index == 0 || index != last {
            self.status = "Only the last material can be removed.".into();
            return;
        }
        if self.material_in_use(index as u32) {
            self.status = "That material is assigned to a mesh.".into();
            return;
        }
        let name = self.materials.materials.remove(index).name;
        self.current_material = 0;
        self.status = format!("Removed material \"{name}\".");
        self.dirty = true;
    }

    /// What a new gizmo drag should move: the vertices behind a sub-object
    /// selection when there is one, otherwise the selected nodes' transforms.
    fn gizmo_targets(&mut self, nodes: &[NodeId]) -> gizmo::Targets {
        let vertices = viewport::selected_vertices(&mut self.scene, &self.selection);
        if !vertices.is_empty() {
            return gizmo::Targets::Vertices(
                vertices
                    .into_iter()
                    .map(|(node, positions)| gizmo::VertexTarget { node, positions })
                    .collect(),
            );
        }
        gizmo::Targets::Nodes(
            nodes
                .iter()
                .filter_map(|id| self.scene.get(*id).ok().map(|n| (*id, n.transform)))
                .collect(),
        )
    }

    /// The vertex a live Move drag would land on this frame: the one nearest
    /// the cursor within [`snap::SNAP_RADIUS_PIXELS`] on screen, excluding the
    /// geometry the drag itself is carrying.
    ///
    /// `None` unless vertex snapping is in force, so nothing else pays for the
    /// scene walk.
    fn snap_target(
        &mut self,
        cursor: egui::Pos2,
        rect: egui::Rect,
        vp: glam::Mat4,
        mode: SnapMode,
    ) -> Option<viewport::SnapCandidate> {
        if mode != SnapMode::Vertex {
            return None;
        }
        let targets = self.gizmo.drag.as_ref()?.targets.clone();
        let (moving_nodes, moving_vertices) = targets.moving();
        let size = [rect.width(), rect.height()];
        let candidates =
            viewport::snap_candidates(&mut self.scene, &moving_nodes, &moving_vertices, vp, size);
        // `project` yields viewport-relative pixels; the cursor is in window
        // space, so bring it into the same frame before measuring.
        let local = [cursor.x - rect.min.x, cursor.y - rect.min.y];
        viewport::nearest_candidate(&candidates, local, snap::SNAP_RADIUS_PIXELS)
    }

    /// The status bar's readout for a live snapped drag — `Move · snap grid
    /// 0.25 · Δ 1.250` — or `None` when this frame is not snapping.
    fn snap_readout(
        &self,
        snap: snap::Snap,
        ray: openforge_core::math::Ray,
        target: Option<viewport::SnapCandidate>,
    ) -> Option<String> {
        if !snap.is_on() {
            return None;
        }
        let drag = self.gizmo.drag.as_ref()?;
        let measure = drag.anchor.measure(ray)?;
        let measure = snap::snap_measure(
            snap,
            drag.axis,
            drag.anchor.origin,
            measure,
            target.map(|c| c.world),
        );
        let tool = match drag.mode {
            TransformMode::Move => "Move",
            TransformMode::Scale => "Scale",
            TransformMode::Rotate => "Rotate",
        };
        let what = match snap.mode {
            SnapMode::Grid => format!("snap grid {}", snap.grid),
            SnapMode::Vertex => "snap vertex".to_string(),
            SnapMode::Off => return None,
        };
        Some(format!("{tool} · {what} · {}", snap::format_measure(measure)))
    }

    /// A world-space picking ray for a screen position inside the viewport.
    ///
    /// Takes the camera by reference rather than `&self` so it can be called
    /// while the scene is mutably borrowed by a tool context.
    fn pick_ray(
        camera: &Camera,
        pos: egui::Pos2,
        rect: egui::Rect,
        aspect: f32,
    ) -> openforge_core::math::Ray {
        let ndc = Self::viewport_ndc(pos, rect);
        camera.ray_from_ndc(ndc.x, ndc.y, aspect)
    }

    /// Whether the picks may run through the occluding ID pass: only a backend
    /// that has one, and only while Select Through is off.
    ///
    /// Select Through is the author asking for the *other* answer — the one
    /// the projection gives on a machine without wgpu: the nearest element
    /// under the pointer, or everything a rectangle covers, hidden behind
    /// geometry or not.
    fn occluding_picks(select_through: bool, has_backend: bool) -> bool {
        has_backend && !select_through
    }

    /// What the GPU's ID pass finds under a click, at the current selection
    /// level.
    ///
    /// `None` when there is no wgpu backend, or when the patch around the
    /// click holds no element the level can use — the caller then falls back
    /// to its own raycast. With Select Through on, it answers `None` at once:
    /// the mode is exactly the request *not* to draw a line at what the image
    /// shows, so the raycast always gets the question.
    fn gpu_pick(
        &mut self,
        render_state: Option<&RenderState>,
        rect: egui::Rect,
        pos: egui::Pos2,
        scale: f32,
        level: SelectLevel,
    ) -> Option<PickElement> {
        if !Self::occluding_picks(self.select_through, render_state.is_some() && self.gpu.is_some()) {
            return None;
        }
        let Some(state) = render_state else { return None };
        let Some(gpu) = self.gpu.as_mut() else { return None };
        let camera = gpu_viewport::orbit_camera(&self.camera);
        let hits = gpu.pick(state, rect, pos, scale, &mut self.scene, &camera);
        gpu_viewport::chosen_hit(level, &hits)
    }

    /// The element a click resolves to: the GPU ID pass first, the CPU raycast
    /// as the fallback.
    ///
    /// The ID pass draws exactly what the frame drew, so a click can only land
    /// on geometry the image shows. The raycast answers when there is no wgpu
    /// backend, when the patch holds nothing the level can use, and for the
    /// levels the ID pass does not serve. With **Select Through** on, the
    /// raycast answers every time — a click may then land on geometry the
    /// frame hides.
    fn resolve_click(
        &mut self,
        render_state: Option<&RenderState>,
        rect: egui::Rect,
        ray: openforge_core::math::Ray,
        pos: egui::Pos2,
        scale: f32,
        level: SelectLevel,
    ) -> Option<PickElement> {
        self.gpu_pick(render_state, rect, pos, scale, level)
            .or_else(|| match level {
                SelectLevel::Face => viewport::pick_face(&mut self.scene, ray)
                    .map(|hit| PickElement::Face { node: hit.node, face: hit.face }),
                SelectLevel::Vertex => {
                    let tol = (self.camera.distance * 0.02).max(0.01);
                    viewport::pick_vertex(&mut self.scene, ray, self.camera.eye(), tol)
                        .map(|hit| PickElement::Vertex { node: hit.node, vertex: hit.vertex })
                }
                SelectLevel::Edge => {
                    let tol = (self.camera.distance * 0.02).max(0.01);
                    viewport::pick_edge(&mut self.scene, ray, tol)
                        .map(|hit| PickElement::Edge { node: hit.node, edge: hit.edge })
                }
                _ => None,
            })
    }

    /// Resolve a marquee rectangle into elements: the GPU's ID pass when there
    /// is one — only it can tell hidden geometry from visible — and the CPU
    /// projection otherwise.
    ///
    /// A GPU answer is trusted as it stands, empty or not: a rectangle over
    /// empty space selects nothing, and one that covers no pixel of the target
    /// at all covers no element either. Falling back to a projection that
    /// selects through geometry would contradict the picture the click path
    /// obeys — so the projection serves a machine without wgpu, and whoever
    /// turned **Select Through** on and asked for exactly that answer.
    fn marquee_hits(
        &mut self,
        render_state: Option<&RenderState>,
        rect: egui::Rect,
        box_: egui::Rect,
        vp: Mat4,
        scale: f32,
        level: SelectLevel,
    ) -> Vec<PickElement> {
        if let (Some(state), Some(gpu)) = (render_state, self.gpu.as_mut()) {
            // Select Through sends the rectangle to the projection below, so
            // the walk through the ID pass is skipped, not re-run.
            if Self::occluding_picks(self.select_through, true) {
                let camera = gpu_viewport::orbit_camera(&self.camera);
                return gpu
                    .pick_rect(
                        state,
                        rect,
                        box_.min,
                        box_.max,
                        scale,
                        &mut self.scene,
                        &camera,
                        level,
                    )
                    .unwrap_or_default();
            }
        }
        // The fallback works in viewport-local points, exactly like the
        // painter's own projection.
        let local = egui::Rect::from_min_max(
            egui::pos2(box_.min.x - rect.min.x, box_.min.y - rect.min.y),
            egui::pos2(box_.max.x - rect.min.x, box_.max.y - rect.min.y),
        );
        viewport::marquee_elements(&mut self.scene, vp, [rect.width(), rect.height()], local, level)
    }

    /// Apply what a rectangle (or a resolved click) found to the selection.
    ///
    /// A plain rectangle replaces the selection and Shift extends it; the
    /// elements go in per node, node selection included, exactly like a click.
    /// Selection only — nothing in the mesh changes, so there is nothing to
    /// undo.
    fn apply_marquee(&mut self, level: SelectLevel, hits: &[PickElement], additive: bool) {
        if !additive {
            self.selection.clear();
        }
        for hit in hits {
            if matches!(level, SelectLevel::Object) {
                self.selection.select_node(hit.node());
            } else {
                self.selection.select_elements(hit.node(), [hit.index()]);
            }
        }
        self.dirty = true;
    }

    /// Paint the live rectangle-selection drag over the viewport.
    ///
    /// A wash with a firmer edge, drawn like the gizmo overlay — on top of
    /// whichever backend painted the frame, so both show the same rectangle.
    fn paint_marquee(&self, painter: &egui::Painter) {
        let Some((a, b)) = self.marquee else { return };
        let rect = egui::Rect::from_two_pos(a, b);
        let fill = egui::Color32::from_rgba_unmultiplied(96, 156, 255, 32);
        let stroke = egui::Color32::from_rgb(120, 176, 255);
        painter.rect_filled(rect, 0.0, fill);
        painter.rect_stroke(rect, 0.0, egui::Stroke::new(1.0_f32, stroke));
    }

    /// Handle one frame of viewport interaction: camera controls, gizmo grabs,
    /// free-form tool drags and clicks.
    ///
    /// `render_state` is the wgpu device when eframe runs the wgpu backend:
    /// sub-object clicks then resolve through the renderer's ID pass first (see
    /// [`Self::gpu_pick`]), which is what makes picking match the drawn image.
    fn handle_viewport_input(
        &mut self,
        ui: &mut egui::Ui,
        rect: egui::Rect,
        render_state: Option<&RenderState>,
    ) {
        let response = ui.interact(rect, ui.id().with("viewport"), egui::Sense::click_and_drag());
        let aspect = rect.width() / rect.height().max(1.0);

        // Camera controls: middle mouse orbits, shift+middle pans, scroll zooms.
        let middle = response.dragged_by(egui::PointerButton::Middle);
        if middle {
            let d = response.drag_delta();
            if ui.input(|i| i.modifiers.shift) {
                self.camera.pan(d.x, d.y);
            } else {
                self.camera.orbit(d.x, d.y);
            }
            self.dirty = true;
        }
        if response.hovered() {
            let scroll = ui.input(|i| i.raw_scroll_delta.y);
            if scroll.abs() > 0.0 {
                self.camera.zoom(scroll * 0.01);
                self.dirty = true;
            }
        }

        // --- Gizmo ----------------------------------------------------------------
        // Lay the handles out once per frame: this same layout drives both the
        // hit-testing below and the painting later in the frame, so the arm that
        // lights up is exactly the arm that was grabbed.
        let size = [rect.width(), rect.height()];
        let vp = self.camera.view_proj(aspect);
        let gizmo_mode = match self.tool {
            ActiveTool::Select => None,
            ActiveTool::Move => Some(TransformMode::Move),
            ActiveTool::Scale => Some(TransformMode::Scale),
            ActiveTool::Rotate => Some(TransformMode::Rotate),
        };
        let ids: Vec<NodeId> = self.selection.selected_nodes().collect();
        let bounds = self.scene.world_bounds(&ids);
        // A sub-object selection is transformed about its own centroid, a node
        // selection about its bounds. While a drag is live the frame stays where
        // that drag began — otherwise the handles would chase the geometry they
        // are moving, and each frame's hit-test would be against a new origin.
        let origin = self.gizmo.drag.as_ref().map(|drag| drag.anchor.origin).or_else(|| {
            viewport::element_centroid(&mut self.scene, &self.selection)
                .or_else(|| (!bounds.is_empty()).then(|| bounds.center()))
        });
        let frame = match (gizmo_mode, origin) {
            (Some(_), Some(origin)) => Some(gizmo::Frame::for_camera(
                &self.camera,
                rect.height(),
                gizmo::HANDLE_PIXELS,
                origin,
            )),
            _ => None,
        };
        self.gizmo.layout = frame.and_then(|f| {
            gizmo::Layout::build(vp, size, rect.min.to_vec2(), &f, gizmo::RING_SEGMENTS)
        });

        // Which handle is under the cursor, and what the pointer is doing. The
        // cursor has to come from the raw pointer state rather than
        // `Response::hover_pos`: a widget stops reporting itself as hovered on
        // the very frame its press starts, which is exactly the frame a grab
        // has to be recognised on.
        let (cursor, pressed, press_pos, released) = ui.input(|i| {
            (
                i.pointer.hover_pos().filter(|p| rect.contains(*p)),
                i.pointer.primary_pressed(),
                i.pointer.press_origin().filter(|p| rect.contains(*p)),
                i.pointer.any_released(),
            )
        });
        let hovered = match (gizmo_mode, self.gizmo.layout.as_ref(), cursor) {
            (Some(TransformMode::Rotate), Some(layout), Some(pos)) => {
                layout.hit_ring(pos, gizmo::HIT_TOLERANCE)
            }
            (Some(_), Some(layout), Some(pos)) => layout.hit_axis(pos, gizmo::HIT_TOLERANCE),
            _ => None,
        };
        if !self.gizmo.is_dragging() && hovered != self.gizmo.hovered {
            self.gizmo.hovered = hovered;
            self.dirty = true;
        }

        // A handle grab owns the pointer for the whole gesture, so the free-form
        // tool drag below never competes with it. The grab is taken on the press
        // itself — by the time egui reports a drag the cursor has already left
        // the handle, so hit-testing then would always miss.
        let mut consumed = false;
        let mut gizmo_undo: Option<(String, Scene)> = None;
        // What a live snapped drag reports in the status bar, if anything.
        let mut snap_readout: Option<String> = None;
        // A snap marker only lives for the drag that shows it.
        if !self.gizmo.is_dragging() {
            self.snap_marker = None;
        }
        if self.gizmo.is_dragging() {
            consumed = true;
            if let Some(pos) = cursor {
                let ray = Self::pick_ray(&self.camera, pos, rect, aspect);
                // Snapping is resolved per frame, so a held modifier takes
                // effect the moment it goes down or comes up, mid-drag.
                let snap = if ui.input(|i| i.modifiers.ctrl) {
                    self.snap.inverted()
                } else {
                    self.snap
                };
                let target = self.snap_target(pos, rect, vp, snap.mode);
                self.snap_marker = target.map(|c| rect.min + egui::vec2(c.screen[0], c.screen[1]));
                if self.gizmo.apply(&mut self.scene, ray, snap, target.map(|c| c.world)) {
                    self.dirty = true;
                }
                snap_readout = self.snap_readout(snap, ray, target);
            }
            if released {
                self.snap_marker = None;
                if let Some(label) = self.gizmo.finish() {
                    if let Some(before) = self.edit_before.take() {
                        gizmo_undo = Some((label.to_string(), before));
                    }
                }
            }
        } else if pressed {
            if let (Some(mode), Some(frame), Some(pos)) = (gizmo_mode, frame, press_pos) {
                let axis = match self.gizmo.layout.as_ref() {
                    Some(layout) if mode == TransformMode::Rotate => {
                        layout.hit_ring(pos, gizmo::HIT_TOLERANCE)
                    }
                    Some(layout) => layout.hit_axis(pos, gizmo::HIT_TOLERANCE),
                    None => None,
                };
                if let Some(axis) = axis {
                    let ray = Self::pick_ray(&self.camera, pos, rect, aspect);
                    if let Some(anchor) = gizmo::Anchor::new(mode, axis, frame.origin, ray) {
                        let targets = self.gizmo_targets(&ids);
                        self.gizmo.begin(gizmo::Drag { mode, axis, anchor, targets });
                        if self.edit_before.is_none() {
                            self.edit_before = Some(self.scene.clone());
                        }
                        consumed = true;
                        self.dirty = true;
                    }
                }
            }
        }

        // Build the active tool and a context borrowing the scene + selection.
        let mut tool = self.active_tool();
        let level = self.selection.level;
        let mut tctx = ToolContext {
            scene: &mut self.scene,
            selection: &mut self.selection,
            // A drag spans many frames, so its state is owned by the app and
            // handed back to the tool on every frame of the gesture.
            drag: self.tool_drag.take(),
        };

        // A rectangle lives only while the primary button is held: if the
        // release never arrived as a frame of its own — the window lost the
        // pointer, say — the next frame drops it rather than leave it painted
        // over the viewport.
        if self.marquee.is_some()
            && !ui.input(|i| i.pointer.primary_down())
            && !response.drag_stopped()
        {
            self.marquee = None;
            self.dirty = true;
        }

        let mut pending_undo: Option<(String, Scene)> = None;
        // For sub-object picking we stash the click ray *and* the position it
        // landed on (the GPU pick needs the latter), and resolve both once the
        // tool context's borrows on the scene/selection are released.
        let mut click_pick: Option<(openforge_core::math::Ray, egui::Pos2)> = None;
        // A released marquee, resolved the same way once the borrows are gone.
        let mut marquee_pick: Option<(egui::Pos2, egui::Pos2)> = None;
        if let Some(pos) = response.interact_pointer_pos() {
            let ndc = Self::viewport_ndc(pos, rect);
            let ray = Self::pick_ray(&self.camera, pos, rect, aspect);
            let pointer = Pointer { ray, ndc };

            // Rectangle selection: in the Select tool a primary drag draws a
            // marquee instead of reaching the tool, which only reacts to
            // clicks — and the handle grab above has already had its say.
            if !consumed && matches!(self.tool, ActiveTool::Select) {
                if response.drag_started_by(egui::PointerButton::Primary) {
                    // egui only calls a press a drag once the pointer has moved
                    // a little, so the rectangle has to start at the press
                    // origin, not at the position that decided the drag.
                    let start = ui.input(|i| i.pointer.press_origin()).unwrap_or(pos);
                    self.marquee = Some((start, pos));
                    consumed = true;
                    self.dirty = true;
                } else if response.dragged_by(egui::PointerButton::Primary) {
                    if let Some(marquee) = self.marquee.as_mut() {
                        marquee.1 = pos;
                        consumed = true;
                        self.dirty = true;
                    }
                } else if response.drag_stopped() {
                    if let Some(marquee) = self.marquee.take() {
                        consumed = true;
                        marquee_pick = Some(marquee);
                    }
                }
            }

            if !consumed && response.drag_started_by(egui::PointerButton::Primary) {
                if self.edit_before.is_none() {
                    self.edit_before = Some(tctx.scene.clone());
                }
                tool.on_pointer_down(&mut tctx, &pointer);
                self.dirty = true;
            } else if !consumed && response.dragged_by(egui::PointerButton::Primary) {
                tool.on_pointer_move(&mut tctx, &pointer);
                self.dirty = true;
            } else if !consumed && response.drag_stopped() {
                if let Some(outcome) = tool.on_pointer_up(&mut tctx, &pointer) {
                    let before = self.edit_before.take();
                    pending_undo = before.map(|b| (outcome.label, b));
                    self.dirty = true;
                }
            } else if !consumed && response.clicked() {
                if level == SelectLevel::Object {
                    // Object level: delegate to the select tool.
                    if matches!(self.tool, ActiveTool::Select) {
                        tool.on_pointer_down(&mut tctx, &pointer);
                        tool.on_pointer_up(&mut tctx, &pointer);
                        self.dirty = true;
                    }
                } else {
                    // Face/Vertex level: stash the click and pick below.
                    click_pick = Some((ray, pos));
                }
            }
        }
        // Hand the (possibly advanced) drag state back to the app.
        self.tool_drag = tctx.drag.take();
        drop(tctx);

        // Resolve a sub-object pick now that the borrows are released. The GPU
        // answers first when there is one, the CPU raycast otherwise — see
        // [`Self::resolve_click`].
        if let Some((ray, pos)) = click_pick {
            let scale = ui.ctx().pixels_per_point();
            if let Some(element) = self.resolve_click(render_state, rect, ray, pos, scale, level) {
                self.apply_marquee(level, &[element], false);
            }
        }

        // A released marquee turns into a selection: the rectangle's own
        // answer, or — when the pointer came back to where it started, leaving
        // no rectangle behind — the click that drag really was.
        if let Some((a, b)) = marquee_pick {
            let box_ = egui::Rect::from_two_pos(a, b);
            let scale = ui.ctx().pixels_per_point();
            if box_.width() <= 0.0 && box_.height() <= 0.0 {
                let ray = Self::pick_ray(&self.camera, b, rect, aspect);
                if let Some(element) = self.resolve_click(render_state, rect, ray, b, scale, level)
                {
                    self.apply_marquee(level, &[element], false);
                }
            } else {
                let additive = ui.input(|i| i.modifiers.shift);
                let hits = self.marquee_hits(render_state, rect, box_, vp, scale, level);
                self.apply_marquee(level, &hits, additive);
            }
        }

        // Record the undo entries now that the scene borrow is released. A gizmo
        // drag and a tool drag never end on the same frame, but both feed the
        // same stack — and a gesture that moved nothing leaves no entry.
        for (label, before) in [gizmo_undo, pending_undo].into_iter().flatten() {
            if !self.scene.content_eq(&before) {
                self.undo.push(label, before, self.scene.clone());
            }
        }

        // A live snapped drag reports its own value instead of the tool hint.
        self.status = match snap_readout {
            Some(readout) => readout,
            None => match self.tool {
                ActiveTool::Select => match self.selection.level {
                    SelectLevel::Object => {
                        format!("Select — {} object(s) selected.", self.selection.node_count())
                    }
                    SelectLevel::Vertex => format!(
                        "Select — {} vertex(es) selected.",
                        self.selection.total_element_count()
                    ),
                    SelectLevel::Face => format!(
                        "Select — {} face(s) selected.",
                        self.selection.total_element_count()
                    ),
                    SelectLevel::Edge => format!(
                        "Select — {} edge(s) selected.",
                        self.selection.total_element_count()
                    ),
                },
                ActiveTool::Move => {
                    "Move — drag an axis handle, or drag the mesh in the view plane.".into()
                }
                ActiveTool::Scale => "Scale — drag an axis handle away from the centre.".into(),
                ActiveTool::Rotate => "Rotate — drag one of the axis rings.".into(),
            },
        };
    }

    fn paint_viewport(&mut self, ui: &mut egui::Ui, rect: egui::Rect) {
        let size = [rect.width(), rect.height()];
        let aspect = size[0] / size[1].max(1.0);
        let vp = self.camera.view_proj(aspect);
        let painter = ui.painter_at(rect);

        // Clip to the viewport.
        painter.rect_filled(rect, 0.0, egui::Color32::from_gray(24));

        // Ground grid.
        self.paint_grid(&painter, vp, rect);

        // Collect and depth-sort triangles (painter's algorithm).
        self.tris = viewport::collect_triangles(&mut self.scene, &self.selection);
        let cam = self.camera.eye();
        let mut order: Vec<(f32, usize)> = Vec::with_capacity(self.tris.len());
        for (i, t) in self.tris.iter().enumerate() {
            let centroid = (t.v[0].pos + t.v[1].pos + t.v[2].pos) / 3.0;
            order.push((centroid.distance(cam), i));
        }
        // Draw far-to-near.
        order.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let light = Vec3::new(0.4, 0.8, 0.5).normalize();
        // Library textures as egui handles; loaded before the loop because the
        // loop holds `self.tris` borrowed.
        let texture_ids = self.cpu_texture_ids(ui.ctx());
        for (_, i) in order {
            let t = &self.tris[i];
            let mut pts: Vec<egui::Pos2> = Vec::with_capacity(3);
            let mut behind = false;
            for v in &t.v {
                match viewport::project(vp, v.pos, size) {
                    Some((x, y, _)) => pts.push(egui::pos2(rect.left() + x, rect.top() + y)),
                    None => {
                        behind = true;
                        break;
                    }
                }
            }
            if behind || pts.len() < 3 {
                continue;
            }
            // The face's resolved material is the surface colour; selection
            // tints it towards the highlight colours.
            let base = viewport::material_base(
                self.materials.resolve(t.material),
                t.selected,
                t.face_selected,
            );
            // Phong shade using the shared materials crate (diffuse + specular).
            let centroid = (t.v[0].pos + t.v[1].pos + t.v[2].pos) / 3.0;
            let view_dir = (cam - centroid).normalize_or_zero();
            let [r, g, b, _] =
                openforge_materials::shade_phong(t.v[0].normal, light, view_dir, 24.0, base);
            let color = egui::Color32::from_rgb(r, g, b);
            // A textured face paints its texels through an egui mesh, which
            // multiplies them by the vertex colour (the shaded material colour
            // above) exactly like the GPU path's `color * texel * lit`. The
            // untextured faces keep the painter's polygon, which is cheaper.
            let texture = viewport::face_texture(&self.materials, t.material)
                .and_then(|index| texture_ids.get(index).copied().flatten());
            let shape = match texture {
                Some(id) => textured_triangle(
                    [pts[0], pts[1], pts[2]],
                    [t.v[0].uv, t.v[1].uv, t.v[2].uv],
                    color,
                    id,
                ),
                None => egui::Shape::convex_polygon(
                    pts,
                    color,
                    egui::Stroke::new(0.5_f32, color.gamma_multiply(0.6)),
                ),
            };
            painter.add(shape);
        }

        // Overlay selected vertices as small dots (vertex level editing aid).
        self.paint_vertex_dots(&painter, rect);
        self.paint_wireframe(&painter, rect);
        // The picked edges over that, in the element highlight (edge level
        // editing aid): drawn last so the bold line wins where they overlap.
        self.paint_selected_edges(&painter, rect);
        // The transform gizmo sits above the geometry so its handles stay
        // grabbable even when they overlap the mesh.
        self.paint_gizmo(&painter);
        self.paint_marquee(&painter);

        // Overlay label.
        let label = format!(
            "Perspective  |  {} tris  |  {} selected",
            self.tris.len(),
            self.selection.node_count()
        );
        painter.text(
            rect.left_top() + egui::vec2(8.0, 8.0),
            egui::Align2::LEFT_TOP,
            label,
            egui::FontId::monospace(12.0),
            egui::Color32::from_gray(200),
        );
    }

    fn paint_grid(&self, painter: &egui::Painter, vp: glam::Mat4, rect: egui::Rect) {
        let size = [rect.width(), rect.height()];
        let half = 10;
        let step = 1.0f32;
        let grid_color = egui::Color32::from_gray(45);
        let axis_x = egui::Color32::from_rgb(180, 70, 70);
        let axis_z = egui::Color32::from_rgb(70, 120, 180);
        for i in -half..=half {
            let f = i as f32 * step;
            // Line parallel to X (varying z).
            if let (Some(a), Some(b)) = (
                viewport::project(vp, Vec3::new(-half as f32, 0.0, f), size),
                viewport::project(vp, Vec3::new(half as f32, 0.0, f), size),
            ) {
                let color = if i == 0 { axis_z } else { grid_color };
                painter.line_segment(
                    [
                        egui::pos2(rect.left() + a.0, rect.top() + a.1),
                        egui::pos2(rect.left() + b.0, rect.top() + b.1),
                    ],
                    egui::Stroke::new(if i == 0 { 1.5_f32 } else { 1.0_f32 }, color),
                );
            }
            // Line parallel to Z (varying x).
            if let (Some(a), Some(b)) = (
                viewport::project(vp, Vec3::new(f, 0.0, -half as f32), size),
                viewport::project(vp, Vec3::new(f, 0.0, half as f32), size),
            ) {
                let color = if i == 0 { axis_x } else { grid_color };
                painter.line_segment(
                    [
                        egui::pos2(rect.left() + a.0, rect.top() + a.1),
                        egui::pos2(rect.left() + b.0, rect.top() + b.1),
                    ],
                    egui::Stroke::new(if i == 0 { 1.5_f32 } else { 1.0_f32 }, color),
                );
            }
        }
    }
}

impl eframe::App for OpenForgeApp {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        // The transport advances on the frame's own clock, before anything
        // paints, so the picture shows the pose it lands on. A hitch (a
        // dragged window, a sleeping machine) must not jump the clip, so the
        // step is capped.
        self.advance_animation(ctx.input(|i| i.stable_dt as f64).min(0.25));
        self.handle_menu(ctx);
        self.handle_shortcuts(ctx);
        self.handle_top_toolbar(ctx);
        // Pack the edges first (status bar lowest, then the UV editor), then
        // the side panels, and only then the central viewport, which takes what
        // is left. That order is what keeps the viewport's rect — and its mouse
        // input — clear of the bottom panels.
        self.handle_status_bar(ctx);
        self.handle_uv_editor(ctx);
        self.handle_timeline(ctx);
        self.handle_side_panels(ctx);
        self.handle_central_viewport(ctx, frame.wgpu_render_state());
        self.import_dropped_files(ctx);
        if self.dirty {
            ctx.request_repaint();
            self.dirty = false;
        }
    }
}

impl OpenForgeApp {
    fn handle_menu(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("menu").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("New").clicked() {
                        self.new_scene();
                        ui.close_menu();
                    }
                    ui.separator();
                    if ui
                        .button("Import…")
                        .on_hover_text("Or drop model files on the window")
                        .clicked()
                    {
                        if let Some(path) = rfd_pick_file(self.format_groups()) {
                            self.import_file(path);
                        }
                        ui.close_menu();
                    }
                    ui.menu_button("Export selected as", |ui| {
                        for (label, extensions) in self.format_groups() {
                            if ui.button(format!("{label}…")).clicked() {
                                if let Some(path) = rfd_save_file(&label, &extensions) {
                                    self.export_selected(path);
                                }
                                ui.close_menu();
                            }
                        }
                    });
                    ui.separator();
                    if ui.button("Quit").clicked() {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                });
                ui.menu_button("Edit", |ui| {
                    let can_undo = self.undo.can_undo();
                    let can_redo = self.undo.can_redo();
                    if ui.add_enabled(can_undo, egui::Button::new("Undo")).clicked() {
                        self.undo_edit();
                        ui.close_menu();
                    }
                    if ui.add_enabled(can_redo, egui::Button::new("Redo")).clicked() {
                        self.redo_edit();
                        ui.close_menu();
                    }
                    if ui
                        .add_enabled(self.selection.node_count() > 0, egui::Button::new("Duplicate"))
                        .clicked()
                    {
                        self.duplicate_selected();
                        ui.close_menu();
                    }
                    ui.separator();
                    if ui
                        .add_enabled(self.selection.node_count() > 0, egui::Button::new("Delete"))
                        .clicked()
                    {
                        self.delete_selected();
                        ui.close_menu();
                    }
                });
                ui.menu_button("Create", |ui| {
                    for kind in ["Box", "Sphere", "Cylinder", "Cone", "Plane"] {
                        if ui.button(kind).clicked() {
                            self.create_primitive(kind);
                            ui.close_menu();
                        }
                    }
                });
                ui.menu_button("Animate", |ui| {
                    if ui
                        .button("Spin Selected (2 s)")
                        .on_hover_text("One full turn about Y from each node's own pose")
                        .clicked()
                    {
                        self.add_spin_clip();
                        ui.close_menu();
                    }
                    ui.separator();
                    if ui.button("Play / Pause (Space)").clicked() {
                        self.toggle_playback();
                        ui.close_menu();
                    }
                    if ui.button("Stop").clicked() {
                        self.stop_playback();
                        ui.close_menu();
                    }
                    if ui
                        .button(format!(
                            "Timeline: {}",
                            if self.anim_open { "shown" } else { "hidden" }
                        ))
                        .clicked()
                    {
                        self.anim_open = !self.anim_open;
                        ui.close_menu();
                    }
                });
                ui.menu_button("Select", |ui| {
                    let sub_level = self.selection.level != SelectLevel::Object;
                    if ui.button("All (Ctrl+A)").clicked() {
                        self.select_all_elements();
                        ui.close_menu();
                    }
                    if ui.button("None (Ctrl+Shift+A)").clicked() {
                        self.clear_element_selection();
                        ui.close_menu();
                    }
                    if ui.button("Invert (Ctrl+I)").clicked() {
                        self.invert_element_selection();
                        ui.close_menu();
                    }
                    ui.separator();
                    if ui
                        .add_enabled(sub_level, egui::Button::new("Grow (Ctrl+Plus)"))
                        .on_hover_text("Add everything adjacent to the selection")
                        .clicked()
                    {
                        self.grow_element_selection();
                        ui.close_menu();
                    }
                    if ui
                        .add_enabled(sub_level, egui::Button::new("Shrink (Ctrl+Minus)"))
                        .on_hover_text("Drop what touches an unselected neighbour")
                        .clicked()
                    {
                        self.shrink_element_selection();
                        ui.close_menu();
                    }
                    ui.separator();
                    let edge_walk = self.selection.level == SelectLevel::Edge
                        && self.selection.total_element_count() > 0;
                    if ui
                        .add_enabled(edge_walk, egui::Button::new("Edge Loop (L)"))
                        .on_hover_text("Add the loop through every selected edge")
                        .clicked()
                    {
                        self.select_edge_loops();
                        ui.close_menu();
                    }
                    if ui
                        .add_enabled(edge_walk, egui::Button::new("Edge Ring (Shift+L)"))
                        .on_hover_text("Add the ring through every selected edge")
                        .clicked()
                    {
                        self.select_edge_rings();
                        ui.close_menu();
                    }
                });
                ui.menu_button("View", |ui| {
                    if ui.button("Frame All (F)").clicked() {
                        self.frame_all();
                        ui.close_menu();
                    }
                    if ui.button("UV Editor").clicked() {
                        self.uv.open = !self.uv.open;
                        ui.close_menu();
                    }
                    if ui
                        .button(format!(
                            "Select Through: {}",
                            if self.select_through { "on" } else { "off" }
                        ))
                        .on_hover_text(
                            "Picks ignore occlusion — occluded elements answer too (P)",
                        )
                        .clicked()
                    {
                        self.toggle_select_through();
                        ui.close_menu();
                    }
                    if ui
                        .button(format!("Wireframe: {}", self.wireframe.label()))
                        .on_hover_text("Cycle Off / Selection / All (Z)")
                        .clicked()
                    {
                        self.wireframe = self.wireframe.next();
                        self.dirty = true;
                        ui.close_menu();
                    }
                    if ui.button("Reset Camera").clicked() {
                        self.camera = Camera::default();
                        self.dirty = true;
                        ui.close_menu();
                    }
                    ui.separator();
                    if ui.button("Hide Selected (H)").clicked() {
                        self.hide_selected_nodes();
                        ui.close_menu();
                    }
                    if ui.button("Isolate Selected (Shift+H)").clicked() {
                        self.isolate_selected_nodes();
                        ui.close_menu();
                    }
                    if ui.button("Unhide All (U)").clicked() {
                        self.unhide_all_nodes();
                        ui.close_menu();
                    }
                });
                ui.menu_button("Help", |ui| {
                    ui.label("OpenForge — clean-room open-source 3D modeler.");
                    ui.label("A from-scratch reimplementation of a node/scene editing model.");
                });
            });
        });
    }

    fn handle_shortcuts(&mut self, ctx: &egui::Context) {
        if ctx.input_mut(|i| i.consume_key(egui::Modifiers::COMMAND, egui::Key::Z)) {
            self.undo_edit();
        }
        if ctx.input_mut(|i| i.consume_key(egui::Modifiers::COMMAND | egui::Modifiers::SHIFT, egui::Key::Z)) {
            self.redo_edit();
        }
        if ctx.input(|i| i.key_pressed(egui::Key::Delete)) {
            // Level-aware delete: element-level deletes sub-objects, object
            // level deletes whole nodes.
            match self.selection.level {
                SelectLevel::Face if self.active_mesh_node().is_some() => self.delete_faces_selected(),
                _ => self.delete_selected(),
            }
        }
        if ctx.input(|i| i.key_pressed(egui::Key::F)) {
            self.frame_all();
        }
        if ctx.input(|i| i.key_pressed(egui::Key::Z)) {
            self.wireframe = self.wireframe.next();
            self.dirty = true;
        }
        // Selection tools. Ctrl+Shift+A is checked on its own: egui's
        // `consume_key` matches modifiers exactly, so the two never collide.
        if ctx.input_mut(|i| i.consume_key(egui::Modifiers::COMMAND | egui::Modifiers::SHIFT, egui::Key::A)) {
            self.clear_element_selection();
        }
        if ctx.input_mut(|i| i.consume_key(egui::Modifiers::COMMAND, egui::Key::A)) {
            self.select_all_elements();
        }
        if ctx.input_mut(|i| i.consume_key(egui::Modifiers::COMMAND, egui::Key::I)) {
            self.invert_element_selection();
        }
        if ctx.input_mut(|i| i.consume_key(egui::Modifiers::COMMAND, egui::Key::Plus))
            || ctx.input_mut(|i| i.consume_key(egui::Modifiers::COMMAND, egui::Key::Equals))
        {
            self.grow_element_selection();
        }
        if ctx.input_mut(|i| i.consume_key(egui::Modifiers::COMMAND, egui::Key::Minus)) {
            self.shrink_element_selection();
        }
        // Edge walks: the loop through the picked edges, the ring with Shift.
        if ctx.input_mut(|i| i.consume_key(egui::Modifiers::SHIFT, egui::Key::L)) {
            self.select_edge_rings();
        }
        if ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::L)) {
            self.select_edge_loops();
        }
        if ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::P)) {
            self.toggle_select_through();
        }
        // Space is the transport: start and pause the animation.
        if ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::Space)) {
            self.toggle_playback();
        }
        // Visibility.
        if ctx.input_mut(|i| i.consume_key(egui::Modifiers::SHIFT, egui::Key::H)) {
            self.isolate_selected_nodes();
        }
        if ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::H)) {
            self.hide_selected_nodes();
        }
        if ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::U)) {
            self.unhide_all_nodes();
        }
        // Snapping: G cycles Off → Grid → Vertex.
        if ctx.input(|i| i.key_pressed(egui::Key::G)) {
            self.snap.mode = self.snap.mode.next();
            self.dirty = true;
        }
        if ctx.input_mut(|i| i.consume_key(egui::Modifiers::COMMAND, egui::Key::D)) {
            self.duplicate_selected();
        }
        // Tool hotkeys.
        if ctx.input(|i| i.key_pressed(egui::Key::Q)) {
            self.tool = ActiveTool::Select;
        }
        if ctx.input(|i| i.key_pressed(egui::Key::W)) {
            self.tool = ActiveTool::Move;
        }
        if ctx.input(|i| i.key_pressed(egui::Key::E)) {
            self.tool = ActiveTool::Scale;
        }
        if ctx.input(|i| i.key_pressed(egui::Key::R)) {
            self.tool = ActiveTool::Rotate;
        }
        // Modeling hotkeys (sub-object ops).
        if ctx.input(|i| i.key_pressed(egui::Key::X)) {
            self.extrude_selected(0.5);
        }
        if ctx.input(|i| i.key_pressed(egui::Key::I)) {
            self.inset_selected(0.3);
        }
        if ctx.input(|i| i.key_pressed(egui::Key::O)) {
            self.poke_selected();
        }
        if ctx.input(|i| i.key_pressed(egui::Key::S)) {
            self.subdivide_selected();
        }
        if ctx.input(|i| i.key_pressed(egui::Key::V)) {
            self.weld_selected(0.001);
        }
        // MeshOps hotkeys: bevel (faces), triangulate, flip, mirror.
        if ctx.input(|i| i.key_pressed(egui::Key::B)) {
            self.bevel_selected(0.5);
        }
        if ctx.input(|i| i.key_pressed(egui::Key::T)) {
            self.triangulate_selected();
        }
        if ctx.input(|i| i.key_pressed(egui::Key::N)) {
            self.flip_normals_selected();
        }
        if ctx.input(|i| i.key_pressed(egui::Key::M)) {
            self.mirror_selected(Vec3::X);
        }
    }

    fn handle_top_toolbar(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("toolbar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label("Tools:");
                tool_button(ui, &mut self.tool, ActiveTool::Select, "Select (Q)");
                tool_button(ui, &mut self.tool, ActiveTool::Move, "Move (W)");
                tool_button(ui, &mut self.tool, ActiveTool::Scale, "Scale (E)");
                tool_button(ui, &mut self.tool, ActiveTool::Rotate, "Rotate (R)");
                ui.separator();
                ui.label("Level:");
                let level_before = self.selection.level;
                ui.selectable_value(&mut self.selection.level, SelectLevel::Object, "Object");
                ui.selectable_value(&mut self.selection.level, SelectLevel::Vertex, "Vertex");
                ui.selectable_value(&mut self.selection.level, SelectLevel::Face, "Face");
                if self.selection.level != level_before {
                    // Element indices are level-specific: keeping them would read
                    // a face index as a vertex index, and the next drag would
                    // transform the wrong thing.
                    self.selection.clear_elements();
                    self.dirty = true;
                }
                ui.separator();
                if ui.button("Frame All").clicked() {
                    self.frame_all();
                }
                if ui
                    .selectable_label(self.uv.open, "UV Editor")
                    .on_hover_text("Show the UV editor panel")
                    .clicked()
                {
                    self.uv.open = !self.uv.open;
                }
                ui.separator();
                ui.label("Snap:");
                for mode in [SnapMode::Off, SnapMode::Grid, SnapMode::Vertex] {
                    let hover = match mode {
                        SnapMode::Off => "Free transforms (hold Ctrl to snap to the grid)",
                        SnapMode::Grid => {
                            "Land on grid values; rotations and scales snap to their steps (hold Ctrl to turn off)"
                        }
                        SnapMode::Vertex => {
                            "Move lands the grabbed point on the vertex under the cursor (hold Ctrl to turn off)"
                        }
                    };
                    if ui
                        .selectable_label(self.snap.mode == mode, mode.label())
                        .on_hover_text(hover)
                        .clicked()
                    {
                        self.snap.mode = mode;
                    }
                }
                if self.snap.mode == SnapMode::Grid {
                    ui.add(
                        egui::DragValue::new(&mut self.snap.grid)
                            .speed(0.01)
                            .range(0.01..=10.0)
                            .fixed_decimals(2)
                            .prefix("step "),
                    )
                    .on_hover_text("World units between grid values");
                    let mut degrees = self.snap.angle.to_degrees();
                    if ui
                        .add(
                            egui::DragValue::new(&mut degrees)
                                .speed(0.5)
                                .range(1.0..=90.0)
                                .fixed_decimals(0)
                                .suffix("°"),
                        )
                        .on_hover_text("Degrees per rotation step")
                        .changed()
                    {
                        self.snap.angle = degrees.to_radians();
                    }
                    ui.add(
                        egui::DragValue::new(&mut self.snap.factor)
                            .speed(0.01)
                            .range(0.01..=1.0)
                            .fixed_decimals(2)
                            .prefix("× "),
                    )
                    .on_hover_text("Scale factor step");
                }
            });
            // Second row: selection tools, at every level.
            ui.separator();
            ui.horizontal(|ui| {
                ui.label("Select:");
                let sub_level = self.selection.level != SelectLevel::Object;
                let has_mesh = self.active_mesh_node().is_some();
                let edge_walk = self.selection.level == SelectLevel::Edge
                    && self.selection.total_element_count() > 0;
                if ui
                    .button("All")
                    .on_hover_text("Select everything at the current level (Ctrl+A)")
                    .clicked()
                {
                    self.select_all_elements();
                }
                if ui
                    .add_enabled(!self.selection.is_empty(), egui::Button::new("None"))
                    .on_hover_text("Drop the selection (Ctrl+Shift+A)")
                    .clicked()
                {
                    self.clear_element_selection();
                }
                if ui
                    .add_enabled(!self.selection.is_empty(), egui::Button::new("Invert"))
                    .on_hover_text("Flip the selection (Ctrl+I)")
                    .clicked()
                {
                    self.invert_element_selection();
                }
                if ui
                    .add_enabled(has_mesh && sub_level, egui::Button::new("Grow"))
                    .on_hover_text("Add everything adjacent to the selection (Ctrl+Plus)")
                    .clicked()
                {
                    self.grow_element_selection();
                }
                if ui
                    .add_enabled(has_mesh && sub_level, egui::Button::new("Shrink"))
                    .on_hover_text("Drop what touches an unselected neighbour (Ctrl+Minus)")
                    .clicked()
                {
                    self.shrink_element_selection();
                }
                if ui
                    .add_enabled(edge_walk, egui::Button::new("Loop"))
                    .on_hover_text("Add the edge loop through the selected edges (L)")
                    .clicked()
                {
                    self.select_edge_loops();
                }
                if ui
                    .add_enabled(edge_walk, egui::Button::new("Ring"))
                    .on_hover_text("Add the edge ring through the selected edges (Shift+L)")
                    .clicked()
                {
                    self.select_edge_rings();
                }
            });
            // Third row: sub-object modeling operations.
            ui.separator();
            ui.horizontal(|ui| {
                ui.label("Model:");
                let has_mesh = self.active_mesh_node().is_some();
                let has_faces = has_mesh && self.selection.total_element_count() > 0
                    && self.selection.level == SelectLevel::Face;
                if ui
                    .add_enabled(has_faces, egui::Button::new("Extrude (X)"))
                    .on_hover_text("Extrude selected faces outward")
                    .clicked()
                {
                    self.extrude_selected(0.5);
                }
                if ui
                    .add_enabled(has_faces, egui::Button::new("Inset (I)"))
                    .on_hover_text("Inset selected faces")
                    .clicked()
                {
                    self.inset_selected(0.3);
                }
                if ui
                    .add_enabled(has_faces, egui::Button::new("Poke (O)"))
                    .on_hover_text("Poke selected faces to their centroid")
                    .clicked()
                {
                    self.poke_selected();
                }
                if ui
                    .add_enabled(has_faces, egui::Button::new("Subdivide (S)"))
                    .on_hover_text("Subdivide selected faces into quads")
                    .clicked()
                {
                    self.subdivide_selected();
                }
                if ui
                    .add_enabled(has_mesh, egui::Button::new("Weld (V)"))
                    .on_hover_text("Weld coincident vertices")
                    .clicked()
                {
                    self.weld_selected(0.001);
                }
                if ui
                    .add_enabled(has_faces, egui::Button::new("Delete Faces"))
                    .on_hover_text("Delete selected faces")
                    .clicked()
                {
                    self.delete_faces_selected();
                }
                if ui
                    .add_enabled(has_faces, egui::Button::new("Bevel (B)"))
                    .on_hover_text("Chamfer the vertices of the selected faces")
                    .clicked()
                {
                    self.bevel_selected(0.5);
                }
                if ui
                    .add_enabled(has_mesh, egui::Button::new("Triangulate (T)"))
                    .on_hover_text("Convert every face to triangles")
                    .clicked()
                {
                    self.triangulate_selected();
                }
                if ui
                    .add_enabled(has_mesh, egui::Button::new("Flip Normals (N)"))
                    .on_hover_text("Reverse face winding and normals")
                    .clicked()
                {
                    self.flip_normals_selected();
                }
                if ui
                    .add_enabled(has_mesh, egui::Button::new("Mirror X (M)"))
                    .on_hover_text("Duplicate the mesh mirrored across the X plane")
                    .clicked()
                {
                    self.mirror_selected(Vec3::X);
                }
                if ui
                    .add_enabled(has_mesh, egui::Button::new("Mirror Y"))
                    .on_hover_text("Duplicate the mesh mirrored across the Y plane")
                    .clicked()
                {
                    self.mirror_selected(Vec3::Y);
                }
                if ui
                    .add_enabled(has_mesh, egui::Button::new("Mirror Z"))
                    .on_hover_text("Duplicate the mesh mirrored across the Z plane")
                    .clicked()
                {
                    self.mirror_selected(Vec3::Z);
                }
            });
            // Fourth row: deformers and boolean operations.
            ui.separator();
            ui.horizontal(|ui| {
                let has_mesh = self.active_mesh_node().is_some();
                ui.label("Deform:");
                ui.label("Axis:");
                ui.selectable_value(&mut self.deform_axis, 0, "X");
                ui.selectable_value(&mut self.deform_axis, 1, "Y");
                ui.selectable_value(&mut self.deform_axis, 2, "Z");
                ui.add(
                    egui::Slider::new(&mut self.deform_amount, -3.0..=3.0)
                        .fixed_decimals(2)
                        .text("Amount"),
                );
                if ui
                    .add_enabled(has_mesh, egui::Button::new("Taper"))
                    .on_hover_text("Narrow one end along the axis (factor)")
                    .clicked()
                {
                    self.deform_selected("Taper");
                }
                if ui
                    .add_enabled(has_mesh, egui::Button::new("Twist"))
                    .on_hover_text("Twist around the axis (radians)")
                    .clicked()
                {
                    self.deform_selected("Twist");
                }
                if ui
                    .add_enabled(has_mesh, egui::Button::new("Bend"))
                    .on_hover_text("Bend into an arc (radians)")
                    .clicked()
                {
                    self.deform_selected("Bend");
                }
                ui.separator();
                ui.label("Boolean:");
                let two_meshes = self
                    .selection
                    .selected_nodes()
                    .filter(|id| self.scene.get(*id).map(|n| n.is_mesh()).unwrap_or(false))
                    .count()
                    >= 2;
                for (label, kind) in [("Union", "Union"), ("Subtract", "Subtract"), ("Intersect", "Intersect")] {
                    if ui
                        .add_enabled(two_meshes, egui::Button::new(label))
                        .on_hover_text("Combine the two selected meshes (the second is consumed)")
                        .clicked()
                    {
                        self.boolean_selected(kind);
                    }
                }
            });
            // Fourth row: rigging and skinning.
            ui.separator();
            ui.horizontal(|ui| {
                let has_mesh = self.active_mesh_node().is_some();
                let has_rig = self.rig_skeleton.is_some();
                ui.label("Rig:");
                ui.label("Bones:");
                ui.add(egui::DragValue::new(&mut self.rig_bones).range(1..=16));
                if ui
                    .add_enabled(has_mesh, egui::Button::new("Create Rig"))
                    .on_hover_text("Build a bone chain along Y and auto-weight the active mesh")
                    .clicked()
                {
                    self.rig_create();
                }
                if has_rig {
                    let bone_count = self.rig_skeleton.as_ref().map(|s| s.len()).unwrap_or(0);
                    ui.label("Bone:");
                    ui.add(
                        egui::DragValue::new(&mut self.rig_bone)
                            .range(0..=bone_count.saturating_sub(1)),
                    );
                    let posed = ui
                        .add(
                            egui::Slider::new(&mut self.rig_angle, -3.0..=3.0)
                                .fixed_decimals(2)
                                .text("Angle"),
                        )
                        .changed();
                    if posed {
                        self.rig_apply_pose();
                    }
                    if ui
                        .button("Reset")
                        .on_hover_text("Return the rig to its rest pose")
                        .clicked()
                    {
                        self.rig_reset();
                    }
                    if ui.button("Clear").clicked() {
                        self.rig_clear();
                    }
                }
            });
        });
    }

    fn handle_side_panels(&mut self, ctx: &egui::Context) {
        // Outliner on the left.
        if self.show_outliner {
            egui::SidePanel::left("outliner")
                .default_width(220.0)
                .show(ctx, |ui| {
                    ui.heading("Scene");
                    ui.separator();
                    let roots = self.scene.roots().to_vec();
                    for id in roots {
                        self.outliner_node(ui, id);
                    }
                });
        }
        // Properties on the right.
        if self.show_properties {
            egui::SidePanel::right("properties")
                .default_width(260.0)
                .show(ctx, |ui| {
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        ui.heading("Properties");
                        ui.separator();
                        self.properties_panel(ui);
                        ui.separator();
                        self.materials_panel(ui);
                        ui.separator();
                        self.plugins_panel(ui);
                    });
                });
        }
    }

    fn outliner_node(&mut self, ui: &mut egui::Ui, id: NodeId) {
        // Snapshot the node's display data so we don't hold a borrow on the
        // scene while recursively painting children.
        let (name, icon, has_children, children, visible) = match self.scene.get(id) {
            Ok(node) => {
                let icon = match &node.kind {
                    NodeKind::Group => "[] ",
                    NodeKind::Dummy => "() ",
                    NodeKind::Mesh { .. } => "/\\ ",
                };
                (node.name.clone(), icon, !node.children.is_empty(), node.children.clone(), node.visible)
            }
            Err(_) => return,
        };
        let selected = self.selection.is_node_selected(id);

        ui.push_id(id.0, |ui| {
            let mut shown = visible;
            let mut resp = None;
            ui.horizontal(|ui| {
                if ui
                    .checkbox(&mut shown, "")
                    .on_hover_text("Visible in the viewport — one undo step, like H")
                    .changed()
                {
                    self.set_node_visible(id, shown);
                }
                let line = ui.selectable_label(selected, format!("{}{}", icon, name));
                resp = Some(line);
            });
            if resp.map(|r| r.clicked()).unwrap_or(false) {
                self.selection.clear();
                self.selection.select_node(id);
            }
            if has_children {
                ui.indent(id.0, |ui| {
                    for child in children {
                        self.outliner_node(ui, child);
                    }
                });
            }
        });
    }

    /// The lines the Plugins panel shows: the loaded plugins and the services,
    /// tools and formats they contributed.
    ///
    /// Split out of the panel so the content can be asserted without a display;
    /// an empty line means "separator" to [`Self::plugins_panel`].
    fn plugin_report(&self) -> Vec<String> {
        let mut lines = Vec::new();
        for plugin in self.plugins.plugins() {
            lines.push(format!("{} {}", plugin.name, plugin.version));
            if !plugin.description.is_empty() {
                lines.push(format!("    {}", plugin.description));
            }
        }
        lines.push(String::new());
        lines.push(format!("Services ({})", self.plugins.services.len()));
        for (name, description) in self.plugins.services.entries() {
            lines.push(format!("    {name} — {description}"));
        }
        lines.push(String::new());
        lines.push(format!("Tools ({})", self.plugins.tool_count()));
        for (name, description) in self.plugins.tool_entries() {
            lines.push(format!("    {name} — {description}"));
        }
        lines.push(String::new());
        lines.push(format!("Formats ({})", self.plugins.filters.len()));
        for (id, label, extensions) in self.plugins.filters.entries() {
            lines.push(format!("    {id} — {label} ({})", extensions.join(", ")));
        }
        lines
    }

    /// What the plugin host has loaded, as reported by [`Self::plugin_report`].
    fn plugins_panel(&mut self, ui: &mut egui::Ui) {
        egui::CollapsingHeader::new("Plugins")
            .default_open(false)
            .show(ui, |ui| {
                for line in self.plugin_report() {
                    if line.is_empty() {
                        ui.separator();
                    } else if let Some(detail) = line.strip_prefix("    ") {
                        ui.label(egui::RichText::new(detail).weak().small());
                    } else {
                        ui.label(line);
                    }
                }
            });
    }

    /// The material library: pick an entry, assign it to the selection and edit
    /// it. Always visible — the library belongs to the project, not to the
    /// current selection.
    fn materials_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("Materials");
        let assigned = self
            .selection
            .selected_nodes()
            .next()
            .and_then(|id| self.scene.get(id).ok().map(|n| n.face_material_slot(0)));

        egui::ScrollArea::vertical().max_height(150.0).show(ui, |ui| {
            for index in 0..self.materials.material_count() {
                let material = self.materials.resolve(index as u32).clone();
                let swatch = viewport::material_base(&material, false, false);
                ui.horizontal(|ui| {
                    let (rect, _) =
                        ui.allocate_exact_size(egui::vec2(12.0, 12.0), egui::Sense::hover());
                    ui.painter().rect_filled(
                        rect,
                        2.0,
                        egui::Color32::from_rgb(swatch[0], swatch[1], swatch[2]),
                    );
                    let label = if Some(index as u32) == assigned {
                        format!("{} ✓", material.name)
                    } else {
                        material.name.clone()
                    };
                    if ui
                        .selectable_label(self.current_material == index, label)
                        .on_hover_text(format!("slot {index}"))
                        .clicked()
                    {
                        self.current_material = index;
                    }
                });
            }
        });

        ui.horizontal(|ui| {
            if ui
                .add_enabled(assigned.is_some(), egui::Button::new("Assign"))
                .on_hover_text("Assign the selected material to the selection")
                .clicked()
            {
                self.assign_material_to_selection(self.current_material as u32);
            }
            let face_level = self.selection.level == SelectLevel::Face
                && self.selection.total_element_count() > 0;
            if ui
                .add_enabled(face_level, egui::Button::new("Assign to Faces"))
                .on_hover_text(
                    "Assign the selected material to the selected faces, keeping the \
                     rest of the mesh on its current material",
                )
                .clicked()
            {
                self.assign_material_to_faces(self.current_material as u32);
            }
            if ui
                .button("New")
                .on_hover_text("Duplicate the selected material")
                .clicked()
            {
                let mut material = self.materials.resolve(self.current_material as u32).clone();
                material.name = format!("{} copy", material.name);
                self.current_material = self.materials.add_material(material);
                self.dirty = true;
            }
            let last = self.materials.material_count().saturating_sub(1);
            let removable = self.current_material == last
                && last > 0
                && !self.material_in_use(last as u32);
            if ui
                .add_enabled(removable, egui::Button::new("Delete"))
                .on_hover_text("Only the last, unassigned material can be removed")
                .clicked()
            {
                self.delete_current_material();
            }
        });

        let index = self.current_material;
        // Read the texture list (and upload any missing egui handles) before
        // borrowing the material itself.
        let texture_names: Vec<String> =
            self.materials.textures.iter().map(|t| t.name.clone()).collect();
        let texture_ids = self.cpu_texture_ids(ui.ctx());
        let Some(material) = self.materials.materials.get_mut(index) else { return };
        ui.separator();
        ui.label(format!("slot {index}"));
        ui.horizontal(|ui| {
            ui.label("Name");
            if ui.text_edit_singleline(&mut material.name).changed() {
                self.dirty = true;
            }
        });
        ui.horizontal(|ui| {
            ui.label("Texture");
            let mut map = material.diffuse_map;
            let selected = match map.and_then(|i| texture_names.get(i)) {
                Some(name) => name.clone(),
                None => "None".to_owned(),
            };
            egui::ComboBox::from_id_salt("material-texture")
                .selected_text(selected)
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut map, None, "None");
                    for (i, name) in texture_names.iter().enumerate() {
                        ui.selectable_value(&mut map, Some(i), name);
                    }
                });
            if map != material.diffuse_map {
                material.diffuse_map = map;
                self.dirty = true;
            }
            // A thumbnail of what the map looks like, when there is one.
            if let Some(id) = map.and_then(|i| texture_ids.get(i).copied().flatten()) {
                ui.add(egui::Image::new(egui::load::SizedTexture::new(
                    id,
                    egui::vec2(20.0, 20.0),
                )));
            }
            if texture_names.is_empty() {
                ui.label("(no textures yet)");
            }
        });
        let mut rgba = egui::Color32::from_rgba_unmultiplied(
            (material.diffuse.x.clamp(0.0, 1.0) * 255.0).round() as u8,
            (material.diffuse.y.clamp(0.0, 1.0) * 255.0).round() as u8,
            (material.diffuse.z.clamp(0.0, 1.0) * 255.0).round() as u8,
            (material.diffuse.w.clamp(0.0, 1.0) * 255.0).round() as u8,
        );
        if ui.color_edit_button_srgba(&mut rgba).changed() {
            material.diffuse = Vec4::new(
                rgba.r() as f32 / 255.0,
                rgba.g() as f32 / 255.0,
                rgba.b() as f32 / 255.0,
                rgba.a() as f32 / 255.0,
            );
            self.dirty = true;
        }
        if ui
            .add(egui::Slider::new(&mut material.shininess, 1.0..=256.0).text("Shininess"))
            .changed()
        {
            self.dirty = true;
        }
        if ui
            .add(egui::Slider::new(&mut material.opacity, 0.0..=1.0).text("Opacity"))
            .changed()
        {
            self.dirty = true;
        }
        ui.horizontal(|ui| {
            if ui
                .button("Load Texture…")
                .on_hover_text("Import a PNG or JPEG into the library and map it onto this material")
                .clicked()
            {
                if let Some(path) = rfd_pick_texture() {
                    if let Some(texture) = self.load_texture(&path) {
                        self.map_texture_to_current(texture);
                    }
                }
            }
            if ui
                .button("Checker")
                .on_hover_text("Add a procedural checkerboard and map it onto this material")
                .clicked()
            {
                let texture = self.add_checker_texture();
                self.map_texture_to_current(texture);
            }
            if ui
                .checkbox(&mut self.uv_texture_background, "UV background")
                .on_hover_text("Paint this texture behind the UVs in the UV editor")
                .changed()
            {
                self.dirty = true;
            }
        });
    }

    fn properties_panel(&mut self, ui: &mut egui::Ui) {
        let ids: Vec<NodeId> = self.selection.selected_nodes().collect();
        if ids.is_empty() {
            ui.label("Nothing selected.");
            ui.label(format!("Nodes: {}", self.scene.node_count()));
            ui.label(format!("Mesh nodes: {}", self.scene.mesh_names().len()));
            return;
        }
        for &id in &ids {
            let Ok(node) = self.scene.get(id) else { continue };
            ui.label(format!("Node: {}", node.name));
            ui.label(format!("Id: {}", id.0));
            if let NodeKind::Mesh { mesh, .. } = &node.kind {
                ui.label(format!("Vertices: {}", mesh.vertex_count()));
                ui.label(format!("Faces: {}", mesh.face_count()));
                ui.label(format!("Triangles: {}", mesh.triangulated_count()));
                let slot = node.face_material_slot(0);
                ui.label(format!("Material: {} (slot {slot})", self.materials.resolve(slot).name));
            }
            ui.separator();
        }

        // Edit the transform of a single selected node.
        if ids.len() == 1 {
            let id = ids[0];
            let mut name = self.scene.get(id).map(|n| n.name.clone()).unwrap_or_default();
            ui.horizontal(|ui| {
                ui.label("Name");
                let response = ui.text_edit_singleline(&mut name);
                // One undo entry for the whole rename, snapshotted when the
                // field took focus — not one entry per keystroke.
                if response.gained_focus() && self.rename_before.is_none() {
                    self.rename_before = Some(self.scene.clone());
                }
                if response.changed() {
                    if let Ok(node) = self.scene.get_mut(id) {
                        node.name = name.clone();
                    }
                    self.dirty = true;
                }
                if response.lost_focus() {
                    if let Some(before) = self.rename_before.take() {
                        if !self.scene.content_eq(&before) {
                            self.undo.push("Rename", before, self.scene.clone());
                        }
                    }
                }
            });
            let before = self.scene.clone();
            let mut changed = false;
            if let Ok(node) = self.scene.get_mut(id) {
                ui.label("Transform");
                let mut t = node.transform.translation;
                let resp = ui.add(egui::DragValue::new(&mut t.x).prefix("X ").speed(0.05))
                    .changed()
                    | ui.add(egui::DragValue::new(&mut t.y).prefix("Y ").speed(0.05)).changed()
                    | ui.add(egui::DragValue::new(&mut t.z).prefix("Z ").speed(0.05)).changed();
                if resp {
                    node.transform.translation = t;
                    changed = true;
                }
            }
            if changed {
                self.undo.push("Move", before, self.scene.clone());
                self.dirty = true;
            }
        }
    }

    fn handle_central_viewport(&mut self, ctx: &egui::Context, render_state: Option<&RenderState>) {
        egui::CentralPanel::default()
            .frame(egui::Frame::none())
            .show(ctx, |ui| {
                let rect = ui.available_rect_before_wrap();
                if rect.width() < 1.0 || rect.height() < 1.0 {
                    return;
                }
                self.handle_viewport_input(ui, rect, render_state);

                // Render offscreen with wgpu when a device is available, and
                // fall back to the CPU painter when it is not. Both paths show
                // the same scene through the same camera state, so the only
                // difference is shading quality and depth correctness.
                // The per-face colours have to be resolved before `gpu` is
                // borrowed mutably, and only the GPU path needs them.
                // Textures upload on the library's revision, not per frame.
                if let Some(state) = render_state {
                    self.sync_gpu_textures(state);
                }
                let colors = match (render_state, self.gpu.is_some()) {
                    (Some(_), true) => self.node_materials(),
                    _ => NodeMaterials::new(),
                };
                let rendered = match (render_state, self.gpu.as_mut()) {
                    (Some(state), Some(gpu)) => {
                        let scale = ui.ctx().pixels_per_point();
                        let size = [rect.width() * scale, rect.height() * scale];
                        let camera = gpu_viewport::orbit_camera(&self.camera);
                        let time = ui.input(|i| i.time) as f32;
                        // The edge overlay is part of the GPU frame itself:
                        // the renderer draws it as depth-tested lines, so the
                        // app does not repaint it on top (the CPU fallback
                        // below still does).
                        let mode = overlay_mode(self.selection.level, self.wireframe);
                        let edge_vertices = if mode == viewport::Wireframe::Off {
                            Vec::new()
                        } else {
                            let eye = self.camera.eye();
                            let edges =
                                viewport::collect_edges(&mut self.scene, &self.selection, mode, eye);
                            gpu_viewport::edge_vertices(&edges)
                        };
                        Some(gpu.render(
                            state,
                            size,
                            &mut self.scene,
                            &self.selection,
                            &colors,
                            &camera,
                            time,
                            &edge_vertices,
                        ))
                    }
                    _ => None,
                };

                match rendered {
                    Some((texture_id, stats)) => {
                        self.paint_gpu_viewport(ui, rect, texture_id, stats)
                    }
                    None => self.paint_viewport(ui, rect),
                }
            });
    }

    /// Present the wgpu-rendered image and draw the CPU overlays on top of it:
    /// the vertex-level dots, the transform gizmo and the stats label. The
    /// wireframe / selection outline is not repainted here — the renderer
    /// already drew it as depth-tested line geometry inside the image.
    fn paint_gpu_viewport(
        &mut self,
        ui: &mut egui::Ui,
        rect: egui::Rect,
        texture_id: egui::TextureId,
        stats: FrameStats,
    ) {
        let painter = ui.painter_at(rect);
        // The offscreen image covers the whole viewport. egui samples user
        // textures with normalized uv and a top-left origin, which is exactly
        // how wgpu's targets are laid out, so no flip is needed.
        let uv = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));
        painter.image(texture_id, rect, uv, egui::Color32::WHITE);

        self.paint_vertex_dots(&painter, rect);
        self.paint_selected_edges(&painter, rect);
        self.paint_gizmo(&painter);
        self.paint_marquee(&painter);

        let label = format!(
            "Perspective  |  wgpu  |  {} tris  |  {} draws  |  {} selected",
            stats.indices / 3,
            stats.draws,
            self.selection.node_count()
        );
        painter.text(
            rect.left_top() + egui::vec2(8.0, 8.0),
            egui::Align2::LEFT_TOP,
            label,
            egui::FontId::monospace(12.0),
            egui::Color32::from_gray(200),
        );
    }

    /// Vertex-level editing aid: a dot per vertex of each selected mesh,
    /// highlighted when that vertex is part of the selection.
    ///
    /// Shared by both viewports — each projects to the same normalized device
    /// coordinates, so the dots land in the same place either way.
    fn paint_vertex_dots(&mut self, painter: &egui::Painter, rect: egui::Rect) {
        if self.selection.level != SelectLevel::Vertex {
            return;
        }
        let dot = 3.0f32;
        let size = [rect.width(), rect.height()];
        let aspect = size[0] / size[1].max(1.0);
        let vp = self.camera.view_proj(aspect);
        for id in self.selection.selected_nodes() {
            let Ok(world) = self.scene.world_matrix(id) else { continue };
            let selected = self.selection.selected_elements(id).cloned();
            let Ok(node) = self.scene.get(id) else { continue };
            let Some(mesh) = node.mesh() else { continue };
            for (vi, p) in mesh.positions.iter().enumerate() {
                let wp = world.transform_point3(*p);
                let Some((x, y, _)) = viewport::project(vp, wp, size) else { continue };
                let is_sel = selected
                    .as_ref()
                    .map(|s| s.contains(&(vi as u32)))
                    .unwrap_or(false);
                let center = egui::pos2(rect.left() + x, rect.top() + y);
                let (rad, color) = if is_sel {
                    (dot + 1.0, element_selected_color())
                } else {
                    (dot, egui::Color32::from_rgb(120, 200, 255))
                };
                painter.circle_filled(center, rad, color);
            }
        }
    }

    /// Edge-level editing aid: a bold line over every edge in the selection.
    ///
    /// Shared by both viewports, like the vertex dots: each projects to the
    /// same normalized device coordinates, so the line lands in the same place
    /// either way. It is drawn over the shading, unoccluded — an editing aid
    /// rather than part of the shaded image, exactly like the dots.
    fn paint_selected_edges(&mut self, painter: &egui::Painter, rect: egui::Rect) {
        if self.selection.level != SelectLevel::Edge {
            return;
        }
        let size = [rect.width(), rect.height()];
        let aspect = size[0] / size[1].max(1.0);
        let vp = self.camera.view_proj(aspect);
        for id in self.selection.selected_nodes() {
            let selected = self.selection.selected_elements(id).cloned();
            let Some(selected) = selected else { continue };
            let Ok(world) = self.scene.world_matrix(id) else { continue };
            let Ok(node) = self.scene.get(id) else { continue };
            let Some(mesh) = node.mesh() else { continue };
            for index in selected {
                let Some([a, b]) = mesh.edges().get(index as usize).copied() else {
                    continue;
                };
                let (Some(pa), Some(pb)) = (
                    mesh.positions.get(a as usize),
                    mesh.positions.get(b as usize),
                ) else {
                    continue;
                };
                let (Some(pa), Some(pb)) = (
                    viewport::project(vp, world.transform_point3(*pa), size),
                    viewport::project(vp, world.transform_point3(*pb), size),
                ) else {
                    continue;
                };
                painter.line_segment(
                    [
                        egui::pos2(rect.left() + pa.0, rect.top() + pa.1),
                        egui::pos2(rect.left() + pb.0, rect.top() + pb.1),
                    ],
                    egui::Stroke::new(3.0_f32, element_selected_color()),
                );
            }
        }
    }

    /// Draw the edge overlay (wireframe and/or selection outline) on the CPU
    /// path.
    ///
    /// Edges are built from the scene in world space and projected through the
    /// same camera the GPU path uses. They are drawn on top of the shading —
    /// the CPU path has no depth buffer to consult — which is why
    /// `collect_edges` drops the ones only ever seen from behind. The GPU path
    /// draws the same collected edges as depth-tested line geometry instead
    /// (see `gpu_viewport::edge_vertices`).
    fn paint_wireframe(&mut self, painter: &egui::Painter, rect: egui::Rect) {
        let mode = overlay_mode(self.selection.level, self.wireframe);
        if mode == viewport::Wireframe::Off {
            return;
        }
        let size = [rect.width(), rect.height()];
        let aspect = size[0] / size[1].max(1.0);
        let vp = self.camera.view_proj(aspect);
        let eye = self.camera.eye();
        let edges = viewport::collect_edges(&mut self.scene, &self.selection, mode, eye);
        for edge in edges {
            let (Some(a), Some(b)) = (
                viewport::project(vp, edge.a, size),
                viewport::project(vp, edge.b, size),
            ) else {
                continue;
            };
            let (width, color) = if edge.face_selected {
                (2.0_f32, egui::Color32::from_rgb(
                    viewport::EDGE_FACE_SELECTED_SRGB[0],
                    viewport::EDGE_FACE_SELECTED_SRGB[1],
                    viewport::EDGE_FACE_SELECTED_SRGB[2],
                ))
            } else if edge.node_selected {
                (1.6_f32, egui::Color32::from_rgb(
                    viewport::EDGE_NODE_SELECTED_SRGB[0],
                    viewport::EDGE_NODE_SELECTED_SRGB[1],
                    viewport::EDGE_NODE_SELECTED_SRGB[2],
                ))
            } else {
                (1.0_f32, egui::Color32::from_rgba_unmultiplied(
                    viewport::EDGE_PLAIN_SRGB[0],
                    viewport::EDGE_PLAIN_SRGB[1],
                    viewport::EDGE_PLAIN_SRGB[2],
                    viewport::EDGE_PLAIN_ALPHA,
                ))
            };
            painter.line_segment(
                [
                    egui::pos2(rect.left() + a.0, rect.top() + a.1),
                    egui::pos2(rect.left() + b.0, rect.top() + b.1),
                ],
                egui::Stroke::new(width, color),
            );
        }
    }

    /// Draw the transform gizmo over the viewport.
    ///
    /// Shared by both viewport backends: the handle layout was computed in
    /// screen space by the input pass, so what is drawn here is exactly what
    /// can be grabbed.
    fn paint_gizmo(&self, painter: &egui::Painter) {
        // A live snapped drag shows exactly where it is landing, above the
        // handles that are being dragged.
        if let Some(p) = self.snap_marker {
            let color = egui::Color32::from_rgb(255, 210, 90);
            painter.circle_stroke(p, 5.0, egui::Stroke::new(1.6_f32, color));
            painter.line_segment(
                [p - egui::vec2(9.0, 0.0), p + egui::vec2(9.0, 0.0)],
                egui::Stroke::new(1.2_f32, color),
            );
            painter.line_segment(
                [p - egui::vec2(0.0, 9.0), p + egui::vec2(0.0, 9.0)],
                egui::Stroke::new(1.2_f32, color),
            );
        }
        let Some(layout) = self.gizmo.layout.as_ref() else { return };
        let mode = match self.tool {
            ActiveTool::Select => return,
            ActiveTool::Move => TransformMode::Move,
            ActiveTool::Scale => TransformMode::Scale,
            ActiveTool::Rotate => TransformMode::Rotate,
        };
        let hot = self.gizmo.highlight();

        if mode == TransformMode::Rotate {
            for (axis, pts) in &layout.rings {
                if pts.len() < 2 {
                    continue;
                }
                let on = Some(*axis) == hot;
                let mut closed = pts.clone();
                closed.push(pts[0]);
                painter.add(egui::Shape::line(
                    closed,
                    egui::Stroke::new(if on { 3.0_f32 } else { 1.6_f32 }, axis.color(on)),
                ));
            }
            return;
        }

        for (axis, tip) in &layout.arms {
            let on = Some(*axis) == hot;
            let color = axis.color(on);
            painter.line_segment(
                [layout.origin, *tip],
                egui::Stroke::new(if on { 4.0_f32 } else { 2.4_f32 }, color),
            );
            if mode == TransformMode::Scale {
                // A square cap reads as "scale" at a glance, a round one as
                // "move".
                painter.rect_filled(
                    egui::Rect::from_center_size(*tip, egui::vec2(9.0, 9.0)),
                    1.0,
                    color,
                );
            } else {
                painter.circle_filled(*tip, if on { 5.0_f32 } else { 4.0_f32 }, color);
            }
            if on {
                painter.text(
                    *tip + egui::vec2(9.0, -9.0),
                    egui::Align2::LEFT_BOTTOM,
                    axis.label(),
                    egui::FontId::monospace(11.0),
                    egui::Color32::WHITE,
                );
            }
        }
        painter.circle_filled(layout.origin, 3.5, egui::Color32::from_gray(235));
    }

    fn handle_status_bar(&self, ctx: &egui::Context) {
        egui::TopBottomPanel::bottom("status").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(&self.status);
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let file = self
                        .current_file
                        .as_ref()
                        .map(|p| p.display().to_string())
                        .unwrap_or_else(|| "untitled".into());
                    ui.label(file);
                });
            });
        });
    }

    // --- Animation --------------------------------------------------------------

    /// Build a spin clip for the selected nodes: one full turn about Y over
    /// [`SPIN_SECONDS`], in quarter-turn keys starting from each node's own
    /// rotation.
    ///
    /// The clip is named around "spin" without colliding with the clips the
    /// transport already holds, selected for playback and shown in the
    /// timeline; the nodes themselves are untouched until it runs.
    fn add_spin_clip(&mut self) {
        let ids: Vec<NodeId> = self.selection.selected_nodes().collect();
        if ids.is_empty() {
            self.status = "Select a node to animate.".into();
            return;
        }
        let name = self.unique_clip_name("spin");
        let mut clip = Clip::new(name.clone());
        for id in &ids {
            let start = self
                .scene
                .get(*id)
                .map(|n| n.transform.rotation)
                .unwrap_or(Quat::IDENTITY);
            let mut track = TransformTrack::targeting(*id);
            for step in 0..=SPIN_STEPS {
                let time = f64::from(step) * SPIN_SECONDS / f64::from(SPIN_STEPS);
                let turn = Quat::from_rotation_y(
                    std::f32::consts::TAU * step as f32 / SPIN_STEPS as f32,
                );
                track.add_rotation_key(time, turn * start, Interp::Linear);
            }
            clip.add_track(track);
        }
        self.anim.library_mut().add_clip(clip);
        self.anim.select_clip(&name);
        self.anim_open = true;
        self.status = format!(
            "Created {name} — {SPIN_SECONDS:.2} s over {} node(s).",
            ids.len()
        );
    }

    /// A clip name around `base` that the transport does not hold yet.
    fn unique_clip_name(&self, base: &str) -> String {
        let mut name = base.to_string();
        let mut n = 2;
        while self.anim.library().clip(&name).is_some() {
            name = format!("{base} {n}");
            n += 1;
        }
        name
    }

    /// Every key time in the selected clip, sorted and deduplicated — what the
    /// timeline draws its ticks from.
    fn clip_key_times(&self) -> Vec<f32> {
        let Some(clip) = self.anim.current_clip() else {
            return Vec::new();
        };
        let mut times: Vec<f32> = clip
            .tracks()
            .iter()
            .flat_map(|track| {
                track
                    .position
                    .keys()
                    .iter()
                    .map(|key| key.time)
                    .chain(track.rotation.keys().iter().map(|key| key.time))
                    .chain(track.scale.keys().iter().map(|key| key.time))
            })
            .map(|time| time as f32)
            .collect();
        times.sort_by(f32::total_cmp);
        times.dedup();
        times
    }

    /// Start or pause the transport.
    fn toggle_playback(&mut self) {
        let Some(name) = self.anim.current_clip_name().map(str::to_string) else {
            self.status = "No clip — Animate ▸ Spin Selected to make one.".into();
            return;
        };
        if self.anim.playing {
            self.anim.pause();
            self.status = format!("Paused at {:.2} s.", self.anim.time);
        } else {
            self.anim.play();
            self.status = format!("Playing {name}.");
        }
    }

    /// Stop the transport, rewind, and put the scene back at the clip's start.
    fn stop_playback(&mut self) {
        self.anim.stop();
        self.pose_current();
        self.status = "Stopped.".into();
    }

    /// Seek the transport and leave the scene posed there.
    fn scrub_to(&mut self, time: f64) {
        self.anim.seek(time);
        self.pose_current();
        self.status = self.transport_readout();
    }

    /// What a scrub or a stop reports: where the transport now stands.
    fn transport_readout(&self) -> String {
        match self.anim.current_clip_name() {
            Some(name) => format!(
                "{name} — {:.2} s of {:.2} s.",
                self.anim.time,
                self.anim.duration()
            ),
            None => "No clip to scrub.".into(),
        }
    }

    /// Advance the transport by one frame and pose the scene with the result.
    ///
    /// Playback is a view over the clip, not an edit: it writes node
    /// transforms so the frame shows the animation, and it leaves no undo
    /// entry — whatever pose it stops at is simply where the scene stands.
    fn advance_animation(&mut self, delta: f64) {
        let poses = self.anim.step(delta);
        if poses.is_empty() {
            return;
        }
        self.apply_poses(&poses);
    }

    /// Pose the scene at the transport's current time without running the
    /// clock.
    fn pose_current(&mut self) {
        let Some(poses) = self
            .anim
            .current_clip()
            .map(|clip| clip.sample_all(self.anim.time))
        else {
            return;
        };
        self.apply_poses(&poses);
    }

    /// Write sampled transforms onto their nodes.
    ///
    /// A node the clip names may be gone; its pose is skipped.
    fn apply_poses(&mut self, poses: &[(NodeId, Transform)]) {
        for (id, transform) in poses {
            if let Ok(node) = self.scene.get_mut(*id) {
                node.transform = *transform;
            }
        }
        self.dirty = true;
    }

    /// The timeline panel: the clip picker, the transport buttons, and the
    /// scrub slider with the selected clip's key times under it.
    fn handle_timeline(&mut self, ctx: &egui::Context) {
        if !self.anim_open {
            return;
        }
        egui::TopBottomPanel::bottom("timeline").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("Timeline");
                if ui.button("Close").clicked() {
                    self.anim_open = false;
                }
                ui.separator();

                let names: Vec<String> = self
                    .anim
                    .library()
                    .clip_names()
                    .iter()
                    .map(|name| name.to_string())
                    .collect();
                let current = self
                    .anim
                    .current_clip_name()
                    .unwrap_or("(no clip)")
                    .to_string();
                egui::ComboBox::from_label("Clip")
                    .selected_text(current)
                    .show_ui(ui, |ui| {
                        for name in &names {
                            if ui.selectable_label(false, name).clicked() {
                                self.anim.select_clip(name);
                                self.pose_current();
                            }
                        }
                    });
                ui.separator();

                let playing = self.anim.playing;
                if ui.button(if playing { "Pause" } else { "Play" }).clicked() {
                    self.toggle_playback();
                }
                if ui.button("Stop").clicked() {
                    self.stop_playback();
                }
                let mut looping = self.anim.looping;
                if ui.checkbox(&mut looping, "Loop").changed() {
                    self.anim.looping = looping;
                }
                ui.separator();
                ui.label(self.transport_readout());
            });

            let duration = self.anim.duration() as f32;
            let mut time = self.anim.time as f32;
            let changed = ui
                .add_enabled(
                    duration > 0.0,
                    egui::Slider::new(&mut time, 0.0..=duration.max(0.001)).suffix(" s"),
                )
                .changed();
            if changed {
                self.scrub_to(time as f64);
            }
            let keys = self.clip_key_times();
            if !keys.is_empty() {
                let ticks: Vec<String> = keys.iter().map(|t| format!("{t:.2}")).collect();
                ui.label(format!("keys: {}", ticks.join(", ")));
            }
        });
    }

    // --- UV editor --------------------------------------------------------------

    /// Number of UV entries in a node's mesh (0 when it has no mesh).
    fn uv_count(&self, id: NodeId) -> usize {
        self.scene
            .get(id)
            .ok()
            .and_then(|n| n.mesh().map(|m| m.uvs.len()))
            .unwrap_or(0)
    }

    /// Whether a node has genuinely projected UVs.
    ///
    /// A mesh that has never been projected still carries a single placeholder
    /// UV slot (core's `Mesh::ensure_attributes` installs one so corner indices
    /// stay valid), so "more than one entry" is what separates real UVs from
    /// that placeholder.
    fn uv_has_real_uvs(&self, id: NodeId) -> bool {
        self.uv_count(id) > 1
    }

    fn handle_uv_editor(&mut self, ctx: &egui::Context) {
        if !self.uv.open {
            return;
        }
        egui::TopBottomPanel::bottom("uv_editor")
            .resizable(true)
            .default_height(420.0)
            .min_height(200.0)
            .show(ctx, |ui| self.uv_panel(ui));
    }

    fn uv_panel(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.heading("UV Editor");
            if ui.button("Close").clicked() {
                self.uv.open = false;
            }
        });
        ui.separator();

        let Some(id) = self.active_mesh_node() else {
            self.uv.retarget(None, 0);
            ui.label("Select a mesh to edit its UVs.");
            return;
        };
        self.uv.retarget(Some(id), self.uv_count(id));

        self.uv_controls(ui, id);
        ui.separator();
        self.uv_canvas(ui, id);

        let selected = self.uv.selected.len();
        let status = if self.uv_has_real_uvs(id) {
            let bounds = match self
                .scene
                .get(id)
                .ok()
                .and_then(|n| n.mesh())
                .and_then(|m| m.uv_bounds())
            {
                Some(b) => format!(
                    "u {:.2}..{:.2}   v {:.2}..{:.2}",
                    b.min.x, b.max.x, b.min.y, b.max.y
                ),
                None => "no finite UVs".to_string(),
            };
            format!("{} UV(s)   {selected} selected   {bounds}", self.uv_count(id))
        } else {
            format!("not projected yet   {selected} selected   (use a Project button)")
        };
        ui.label(status);
    }

    /// The UV editor's button rows: selection, projection and editing.
    fn uv_controls(&mut self, ui: &mut egui::Ui, id: NodeId) {
        let uv_count = self.uv_count(id);
        // Projections need geometry, not UVs; every other edit needs real UVs.
        let has_uvs = self.uv_has_real_uvs(id);
        let has_faces = self
            .scene
            .get(id)
            .ok()
            .and_then(|n| n.mesh())
            .map(|m| !m.faces.is_empty())
            .unwrap_or(false);
        let selected = self.uv.selected.len();
        let faces_selected = self.selection.level == SelectLevel::Face
            && self
                .selection
                .selected_elements(id)
                .map(|s| !s.is_empty())
                .unwrap_or(false);

        ui.horizontal(|ui| {
            ui.label("Selection:");
            if ui
                .add_enabled(has_uvs, egui::Button::new("All"))
                .on_hover_text("Select every UV")
                .clicked()
            {
                self.uv.select_only(0..uv_count as u32);
            }
            if ui
                .add_enabled(selected > 0, egui::Button::new("None"))
                .clicked()
            {
                self.uv.deselect_all();
            }
            if ui
                .add_enabled(faces_selected, egui::Button::new("From Faces"))
                .on_hover_text("Select the UVs of the faces selected in the viewport")
                .clicked()
            {
                self.uv_select_from_faces(id);
            }
            if ui
                .add_enabled(selected > 0, egui::Button::new("Island"))
                .on_hover_text("Select every UV of the islands the selection touches")
                .clicked()
            {
                self.uv_select_island(id);
            }
        });

        ui.horizontal(|ui| {
            ui.label("Project:");
            for (label, axis) in [
                ("Planar X", Vec3::X),
                ("Planar Y", Vec3::Y),
                ("Planar Z", Vec3::Z),
            ] {
                if ui
                    .add_enabled(has_faces, egui::Button::new(label))
                    .on_hover_text("Project the mesh along this axis (replaces every UV)")
                    .clicked()
                {
                    self.uv_apply(id, label, move |m| m.uv_project_planar(axis));
                }
            }
            if ui
                .add_enabled(has_faces, egui::Button::new("Cylindrical"))
                .clicked()
            {
                self.uv_apply(id, "UV Cylindrical", |m| m.uv_project_cylindrical(Vec3::Y));
            }
            if ui.add_enabled(has_faces, egui::Button::new("Spherical")).clicked() {
                self.uv_apply(id, "UV Spherical", |m| m.uv_project_spherical(Vec3::Y));
            }
            if ui.add_enabled(has_faces, egui::Button::new("Box")).clicked() {
                self.uv_apply(id, "UV Box", |m| m.uv_project_box());
            }
        });

        ui.horizontal(|ui| {
            ui.label("Edit:");
            if ui.add_enabled(has_uvs, egui::Button::new("Normalize")).clicked() {
                self.uv_apply(id, "UV Normalize", |m| m.uv_normalize());
            }
            if ui.add_enabled(has_uvs, egui::Button::new("Flip V")).clicked() {
                self.uv_apply(id, "UV Flip V", |m| m.uv_flip_v());
            }
            ui.add(
                egui::DragValue::new(&mut self.uv.grid)
                    .range(1..=8)
                    .prefix("grid "),
            );
            // The two packers share this grid but group differently: "Pack"
            // welds faces by coincident positions (uv_repack), "Pack Islands"
            // by UV continuity across shared mesh edges (uv_pack_islands).
            if ui
                .add_enabled(has_uvs, egui::Button::new("Pack"))
                .on_hover_text("Copy-position grid repack")
                .clicked()
            {
                let grid = self.uv.grid;
                self.uv_apply(id, "UV Pack", move |m| m.uv_repack(grid));
            }
            if ui
                .add_enabled(has_uvs, egui::Button::new("Pack Islands"))
                .on_hover_text("Pack each UV island into its own atlas cell, preserving its layout")
                .clicked()
            {
                let grid = self.uv.grid.max(1);
                self.uv_apply(id, "UV Pack Islands", move |m| {
                    m.uv_pack_islands(grid, uv_editor::ISLAND_MARGIN);
                });
            }
            ui.separator();
            let can_transform = selected > 0;
            ui.add_enabled(
                can_transform,
                egui::Slider::new(&mut self.uv.scale, 0.1..=4.0).text("Scale"),
            );
            ui.add_enabled(
                can_transform,
                egui::Slider::new(&mut self.uv.rotate_deg, -180.0..=180.0).text("Rotate"),
            );
            if ui
                .add_enabled(can_transform, egui::Button::new("Apply"))
                .on_hover_text("Scale then rotate the selection about its centre")
                .clicked()
            {
                let scale = Vec2::splat(self.uv.scale);
                let rotate = self.uv.rotate_deg.to_radians();
                self.uv_transform(id, scale, rotate);
            }
            if ui
                .add_enabled(can_transform, egui::Button::new("Reset"))
                .clicked()
            {
                self.uv.scale = 1.0;
                self.uv.rotate_deg = 0.0;
            }
        });
    }

    /// The UV canvas: draw the mesh's UVs, then handle pick/drag/band input.
    fn uv_canvas(&mut self, ui: &mut egui::Ui, id: NodeId) {
        let avail = ui.available_size();
        let side = uv_editor::CANVAS_SIZE.min(avail.x).min(avail.y).max(80.0);
        let (rect, response) =
            ui.allocate_exact_size(egui::vec2(side, side), egui::Sense::click_and_drag());
        let painter = ui.painter_at(rect);

        let hovered = match (response.hover_pos(), self.uv_mesh_uvs(id)) {
            (Some(pos), Some(mesh)) => uv_editor::pick_uv(mesh, pos, rect),
            _ => None,
        };
        let background = if self.uv_texture_background {
            self.uv_background_texture()
                .and_then(|index| self.cpu_texture(ui.ctx(), index))
        } else {
            None
        };
        if let Ok(node) = self.scene.get(id) {
            if let Some(mesh) = node.mesh() {
                uv_editor::draw_canvas(&painter, rect, mesh, &self.uv, hovered, background);
            }
        }

        let shift = ui.input(|i| i.modifiers.shift);
        let pointer = response.interact_pointer_pos();
        if response.drag_started() {
            if let Some(pos) = pointer {
                match self.uv_pick(id, pos, rect) {
                    Some(uv) => {
                        if shift {
                            self.uv.selected.insert(uv);
                        } else if !self.uv.selected.contains(&uv) {
                            self.uv.select_only([uv]);
                        }
                        self.uv.drag = Some(uv_editor::UvDrag {
                            before: self.scene.clone(),
                            last: pos,
                        });
                    }
                    None => {
                        if !shift {
                            self.uv.deselect_all();
                        }
                        self.uv.band = Some((pos, pos));
                    }
                }
            }
        } else if response.dragged() {
            if let Some(pos) = pointer {
                if self.uv.drag.is_some() {
                    // Scope the mutable borrow so the mesh can be edited below.
                    let step = {
                        let drag = self.uv.drag.as_mut().expect("checked above");
                        let last = drag.last;
                        drag.last = pos;
                        uv_editor::canvas_to_uv(pos, rect) - uv_editor::canvas_to_uv(last, rect)
                    };
                    self.uv_apply_delta(id, step);
                } else if let Some(band) = self.uv.band.as_mut() {
                    band.1 = pos;
                }
            }
        } else if response.drag_stopped() {
            // A finished move becomes exactly one undo entry.
            if let Some(drag) = self.uv.drag.take() {
                if !self.scene.content_eq(&drag.before) {
                    self.undo.push("UV Move", drag.before, self.scene.clone());
                    self.dirty = true;
                    self.status = "Moved UVs.".to_string();
                }
            }
            if let Some((a, b)) = self.uv.band.take() {
                let picked = self
                    .uv_mesh_uvs(id)
                    .map(|m| uv_editor::band_select(m, rect, a, b))
                    .unwrap_or_default();
                let count = picked.len();
                if shift {
                    self.uv.select_more(picked);
                } else {
                    self.uv.select_only(picked);
                }
                self.status = format!("{} UV(s) in the band.", count);
            }
        } else if response.clicked() {
            if let Some(pos) = pointer {
                match self.uv_pick(id, pos, rect) {
                    Some(uv) => {
                        if shift {
                            self.uv.selected.insert(uv);
                        } else {
                            self.uv.select_only([uv]);
                        }
                    }
                    None => {
                        if !shift {
                            self.uv.deselect_all();
                        }
                    }
                }
            }
        }

        // Drawn after the update so the band tracks the pointer in-frame.
        if let Some((a, b)) = self.uv.band {
            let band = egui::Rect::from_two_pos(a, b);
            painter.rect_filled(
                band,
                0.0,
                egui::Color32::from_rgba_unmultiplied(90, 140, 220, 40),
            );
            painter.rect_stroke(
                band,
                0.0,
                egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(120, 170, 240)),
            );
        }
        if !self.uv_has_real_uvs(id) {
            painter.text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                "No UVs yet — pick a projection above",
                egui::FontId::proportional(13.0),
                egui::Color32::from_gray(170),
            );
        }
    }

    /// The active mesh's UVs, if it has a mesh.
    fn uv_mesh_uvs(&self, id: NodeId) -> Option<&Mesh> {
        self.scene.get(id).ok().and_then(|n| n.mesh())
    }

    /// The UV entry under `pos`, if any.
    fn uv_pick(&self, id: NodeId, pos: egui::Pos2, canvas: egui::Rect) -> Option<u32> {
        self.uv_mesh_uvs(id)
            .and_then(|m| uv_editor::pick_uv(m, pos, canvas))
    }

    /// The pivot for selection transforms: the selection's UV centre, or the
    /// middle of the tile when nothing is selected.
    fn uv_pivot(&self, id: NodeId) -> Vec2 {
        self.uv_mesh_uvs(id)
            .map(|m| self.uv.pivot(m))
            .unwrap_or_else(|| Vec2::splat(0.5))
    }

    /// Run a committed UV operation on the active mesh as one undo entry.
    fn uv_apply(&mut self, id: NodeId, label: &str, f: impl FnOnce(&mut Mesh)) {
        let mut ran = false;
        self.with_edit(label, |app| {
            if let Ok(node) = app.scene.get_mut(id) {
                if let Some(mesh) = node.mesh_mut() {
                    f(mesh);
                    ran = true;
                }
            }
        });
        self.status = if ran {
            format!("{}: {} UV(s).", label, self.uv_count(id))
        } else {
            "That node has no mesh.".to_string()
        };
    }

    /// Scale then rotate the selected UVs about their centre, as one undo entry.
    fn uv_transform(&mut self, id: NodeId, scale: Vec2, rotate: f32) {
        let indices = self.uv.indices();
        if indices.is_empty() {
            self.status = "Select UVs first (click the canvas or use a band).".to_string();
            return;
        }
        let count = indices.len();
        let pivot = self.uv_pivot(id);
        self.with_edit("UV Transform", move |app| {
            if let Ok(node) = app.scene.get_mut(id) {
                if let Some(mesh) = node.mesh_mut() {
                    mesh.uv_transform_selected(&indices, Vec2::ZERO, scale, rotate, pivot);
                }
            }
        });
        self.status = format!(
            "Transformed {} UV(s): scale {:.2}, {:+.0} deg.",
            count,
            scale.x,
            rotate.to_degrees()
        );
    }

    /// Apply a live drag delta to the selected UVs.
    ///
    /// Deliberately bypasses `with_edit`: the drag pushes a single undo entry
    /// when the pointer is released, so the stack is not flooded per frame.
    fn uv_apply_delta(&mut self, id: NodeId, delta: Vec2) {
        let indices = self.uv.indices();
        if indices.is_empty() || delta == Vec2::ZERO {
            return;
        }
        if let Ok(node) = self.scene.get_mut(id) {
            if let Some(mesh) = node.mesh_mut() {
                mesh.uv_transform_selected(&indices, delta, Vec2::ONE, 0.0, Vec2::ZERO);
                self.dirty = true;
            }
        }
    }

    /// Select the UVs of the faces selected in the viewport.
    fn uv_select_from_faces(&mut self, id: NodeId) {
        let faces = self.selected_faces(id);
        let mut uvs: Vec<u32> = Vec::new();
        if let Ok(node) = self.scene.get(id) {
            if let Some(mesh) = node.mesh() {
                for f in &faces {
                    for (uv, _) in mesh.uv_face_uvs(*f) {
                        if !uvs.contains(&uv) {
                            uvs.push(uv);
                        }
                    }
                }
            }
        }
        if uvs.is_empty() {
            self.status = "The selected faces have no UVs.".to_string();
            return;
        }
        self.status = format!(
            "Selected {} UV(s) from {} face(s).",
            uvs.len(),
            faces.len()
        );
        self.uv.select_only(uvs);
    }

    /// Grow the UV selection to the whole islands the current entries sit in.
    ///
    /// Selection only: no undo entry, because nothing in the mesh changes.
    fn uv_select_island(&mut self, id: NodeId) {
        let indices = self.uv.indices();
        let Some(mesh) = self.uv_mesh_uvs(id) else { return };
        let island = uv_editor::island_selection(mesh, &indices, uv_editor::ISLAND_TOLERANCE);
        if island.is_empty() {
            self.status = "Select a UV to grow into its island.".into();
            return;
        }
        self.status = format!("{} UVs in the selected island(s).", island.len());
        self.uv.select_only(island);
    }

    fn frame_all(&mut self) {
        let ids = self.scene.all_ids();
        let b = self.scene.world_bounds(&ids);
        if !b.is_empty() {
            self.camera.target = b.center();
            let radius = b.size().length() * 0.5;
            self.camera.distance = (radius * 2.5).clamp(1.0, 500.0);
            self.dirty = true;
        }
    }
}

fn tool_button(ui: &mut egui::Ui, current: &mut ActiveTool, value: ActiveTool, label: &str) {
    let selected = *current == value;
    if ui.selectable_label(selected, label).clicked() {
        *current = value;
    }
}

// --- Primitive builders ---------------------------------------------------------

fn primitive_box(size: f32) -> Mesh {
    let h = size * 0.5;
    let mut m = Mesh::new();
    // 8 corners.
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
    let quads = [
        [0, 1, 2, 3], // back
        [4, 7, 6, 5], // front
        [0, 4, 5, 1], // bottom
        [3, 2, 6, 7], // top
        [0, 3, 7, 4], // left
        [1, 5, 6, 2], // right
    ];
    for q in quads {
        m.add_face(&q);
    }
    m.compute_flat_normals();
    m
}

fn primitive_plane(w: f32, d: f32) -> Mesh {
    let mut m = Mesh::new();
    m.add_vertex(Vec3::new(-w * 0.5, 0.0, -d * 0.5));
    m.add_vertex(Vec3::new(w * 0.5, 0.0, -d * 0.5));
    m.add_vertex(Vec3::new(w * 0.5, 0.0, d * 0.5));
    m.add_vertex(Vec3::new(-w * 0.5, 0.0, d * 0.5));
    m.add_face(&[0, 1, 2, 3]);
    m.compute_flat_normals();
    m
}

fn primitive_cylinder(radius: f32, height: f32, segments: u32) -> Mesh {
    let mut m = Mesh::new();
    let segments = segments.max(3);
    let h = height * 0.5;
    // Side vertices (two rings).
    for ring_y in [-h, h] {
        for s in 0..segments {
            let a = (s as f32 / segments as f32) * std::f32::consts::TAU;
            m.add_vertex(Vec3::new(a.cos() * radius, ring_y, a.sin() * radius));
        }
    }
    // Side quads.
    for s in 0..segments {
        let s2 = (s + 1) % segments;
        let a = s;
        let b = s2;
        let c = segments + s2;
        let d = segments + s;
        m.add_face(&[a, b, c, d]);
    }
    // Caps as triangle fans.
    let top_center = m.add_vertex(Vec3::new(0.0, h, 0.0));
    let bot_center = m.add_vertex(Vec3::new(0.0, -h, 0.0));
    for s in 0..segments {
        let s2 = (s + 1) % segments;
        m.add_face(&[top_center, segments + s, segments + s2]);
        m.add_face(&[bot_center, s2, s]);
    }
    m.compute_flat_normals();
    m
}

fn primitive_cone(radius: f32, height: f32, segments: u32) -> Mesh {
    let mut m = Mesh::new();
    let segments = segments.max(3);
    let h = height * 0.5;
    let apex = m.add_vertex(Vec3::new(0.0, h, 0.0));
    let base_start = m.vertex_count() as u32;
    for s in 0..segments {
        let a = (s as f32 / segments as f32) * std::f32::consts::TAU;
        m.add_vertex(Vec3::new(a.cos() * radius, -h, a.sin() * radius));
    }
    let base_center = m.add_vertex(Vec3::new(0.0, -h, 0.0));
    for s in 0..segments {
        let s2 = (s + 1) % segments;
        m.add_face(&[apex, base_start + s, base_start + s2]);
        m.add_face(&[base_center, base_start + s2, base_start + s]);
    }
    m.compute_flat_normals();
    m
}

// --- File dialogs (rfd) ---------------------------------------------------------

/// The open dialog: one filter group per registered format.
fn rfd_pick_file(groups: Vec<(String, Vec<String>)>) -> Option<PathBuf> {
    let mut dialog = rfd::FileDialog::new();
    for (label, extensions) in groups {
        let exts: Vec<&str> = extensions.iter().map(|e| e.as_str()).collect();
        dialog = dialog.add_filter(label, &exts);
    }
    dialog.pick_file()
}

/// The save dialog for one format: its own filter and a matching default file
/// name, so the file the user picks carries the format they chose.
fn rfd_save_file(label: &str, extensions: &[String]) -> Option<PathBuf> {
    let exts: Vec<&str> = extensions.iter().map(|e| e.as_str()).collect();
    let first = exts.first().copied().unwrap_or("obj");
    rfd::FileDialog::new()
        .add_filter(label, &exts)
        .set_file_name(format!("export.{first}"))
        .save_file()
}

fn rfd_pick_texture() -> Option<PathBuf> {
    rfd::FileDialog::new()
        .add_filter("Images", &["png", "jpg", "jpeg"])
        .pick_file()
}

#[cfg(test)]
mod tests {
    use super::*;
    use openforge_io::{IoResult, LoadedScene, MeshData};
    use openforge_plugin::{FormatFilter, Plugin, PluginError};

    const SCREEN: [f32; 2] = [1000.0, 700.0];

    /// One frame of raw input: where the pointer is, and whether the primary
    /// button went down or up on this frame.
    fn frame(time: f64, pointer: egui::Pos2, click: Option<bool>) -> egui::RawInput {
        let mut input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(SCREEN[0], SCREEN[1]),
            )),
            time: Some(time),
            ..Default::default()
        };
        input.events.push(egui::Event::PointerMoved(pointer));
        if let Some(pressed) = click {
            input.events.push(egui::Event::PointerButton {
                pos: pointer,
                button: egui::PointerButton::Primary,
                pressed,
                modifiers: egui::Modifiers::NONE,
            });
        }
        input
    }

    /// A frame carrying one key press.
    fn key_frame(time: f64, key: egui::Key) -> egui::RawInput {
        let mut input = frame(time, egui::pos2(500.0, 350.0), None);
        input.events.push(egui::Event::Key {
            key,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        });
        input
    }

    /// A frame carrying one key press with modifiers (egui's `consume_key`
    /// matches modifiers exactly, so Ctrl+A and Ctrl+Shift+A arrive apart).
    fn key_frame_mods(time: f64, key: egui::Key, modifiers: egui::Modifiers) -> egui::RawInput {
        let mut input = frame(time, egui::pos2(500.0, 350.0), None);
        input.modifiers = modifiers;
        input.events.push(egui::Event::Key {
            key,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers,
        });
        input
    }

    /// A frame with Ctrl held down, for the snapping tests.
    fn ctrl_frame(time: f64, pointer: egui::Pos2, click: Option<bool>) -> egui::RawInput {
        let mut input = frame(time, pointer, click);
        input.modifiers = egui::Modifiers::CTRL;
        input
    }

    /// Drive one frame including the global shortcuts, the way `update` does.
    fn step_with_keys(ctx: &egui::Context, app: &mut OpenForgeApp, input: egui::RawInput) {
        let _ = ctx.run(input, |ctx| {
            app.handle_shortcuts(ctx);
            app.handle_central_viewport(ctx, None);
        });
    }

    /// Drive one frame of the app's viewport path (input + paint), headlessly,
    /// returning the shapes the frame painted.
    fn paint_once(
        ctx: &egui::Context,
        app: &mut OpenForgeApp,
        input: egui::RawInput,
    ) -> Vec<egui::epaint::ClippedShape> {
        ctx.run(input, |ctx| app.handle_central_viewport(ctx, None))
            .shapes
    }

    /// Drive one frame and discard its output.
    fn step(ctx: &egui::Context, app: &mut OpenForgeApp, input: egui::RawInput) {
        let _ = paint_once(ctx, app, input);
    }

    /// How many stroked segments and filled paths a frame's shapes contain.
    ///
    /// The gizmo's axis arms are segments and its rotation rings are paths, so
    /// counting them is how the tests see what was actually drawn.
    fn stroked(shapes: &[egui::epaint::ClippedShape]) -> (usize, usize) {
        fn walk(shape: &egui::Shape, lines: &mut usize, paths: &mut usize) {
            match shape {
                egui::Shape::LineSegment { .. } => *lines += 1,
                egui::Shape::Path(_) => *paths += 1,
                egui::Shape::Vec(inner) => {
                    for s in inner {
                        walk(s, lines, paths);
                    }
                }
                _ => {}
            }
        }
        let (mut lines, mut paths) = (0, 0);
        for clipped in shapes {
            walk(&clipped.shape, &mut lines, &mut paths);
        }
        (lines, paths)
    }

    /// The app as it starts up, with the seed sphere selected and the Move tool
    /// active — the state a user lands in after clicking the sphere and
    /// pressing W.
    fn app_ready_to_move() -> (OpenForgeApp, NodeId) {
        let mut app = OpenForgeApp::with_gpu(None);
        let id = app
            .scene
            .iter()
            .find(|n| n.is_mesh())
            .map(|n| n.id)
            .expect("the seed sphere exists");
        app.selection.select_node(id);
        app.tool = ActiveTool::Move;
        (app, id)
    }

    /// Where an axis arm ends on screen, as the app laid it out this frame.
    fn arm_tip(app: &OpenForgeApp, axis: gizmo::Axis) -> egui::Pos2 {
        app.gizmo
            .layout
            .as_ref()
            .expect("a transform tool with a selection lays the handles out")
            .arms
            .iter()
            .find(|(a, _)| *a == axis)
            .map(|(_, p)| *p)
            .expect("the axis projects on screen")
    }

    #[test]
    fn the_gizmo_appears_only_for_a_transform_tool_over_a_selection() {
        let ctx = egui::Context::default();
        let (mut app, _id) = app_ready_to_move();
        step(&ctx, &mut app, frame(0.0, egui::pos2(500.0, 350.0), None));
        assert!(app.gizmo.layout.is_some(), "Move tool + selection draws handles");

        app.tool = ActiveTool::Select;
        step(&ctx, &mut app, frame(1.0, egui::pos2(500.0, 350.0), None));
        assert!(app.gizmo.layout.is_none(), "the Select tool has no handles");

        app.tool = ActiveTool::Move;
        app.selection.clear();
        step(&ctx, &mut app, frame(2.0, egui::pos2(500.0, 350.0), None));
        assert!(app.gizmo.layout.is_none(), "nothing selected means nothing to move");
    }

    #[test]
    fn hovering_an_arm_highlights_it_and_only_it() {
        let ctx = egui::Context::default();
        let (mut app, _id) = app_ready_to_move();
        step(&ctx, &mut app, frame(0.0, egui::pos2(500.0, 350.0), None));

        let tip = arm_tip(&app, gizmo::Axis::X);
        step(&ctx, &mut app, frame(1.0, tip, None));
        assert_eq!(app.gizmo.hovered, Some(gizmo::Axis::X));

        // A corner of the viewport is nowhere near a handle.
        step(&ctx, &mut app, frame(2.0, egui::pos2(20.0, 660.0), None));
        assert_eq!(app.gizmo.hovered, None);
    }

    #[test]
    fn dragging_an_arm_moves_along_that_axis_and_undoes_in_one_step() {
        let ctx = egui::Context::default();
        let (mut app, id) = app_ready_to_move();
        step(&ctx, &mut app, frame(0.0, egui::pos2(500.0, 350.0), None));

        let tip = arm_tip(&app, gizmo::Axis::X);
        let before = app.scene.get(id).unwrap().transform;

        // Press on the handle: the grab has to be taken here, because by the
        // time the pointer has moved egui calls it a drag, not a hover.
        step(&ctx, &mut app, frame(1.0, tip, Some(true)));
        assert!(app.gizmo.is_dragging(), "pressing a handle starts a drag");
        assert_eq!(app.gizmo.active, Some(gizmo::Axis::X));

        // Drag clear of the handle and the selection follows the axis.
        let dragged = tip + egui::vec2(140.0, 0.0);
        step(&ctx, &mut app, frame(2.0, dragged, None));
        let mid = app.scene.get(id).unwrap().transform;
        assert!(app.gizmo.is_dragging());

        // Release.
        step(&ctx, &mut app, frame(3.0, dragged, Some(false)));
        assert!(!app.gizmo.is_dragging(), "releasing ends the gesture");
        assert_eq!(app.gizmo.active, None);

        let after = app.scene.get(id).unwrap().transform;
        assert!(
            after.translation.x > before.translation.x + 0.05,
            "moved along +X: {:?} -> {:?}",
            before.translation,
            after.translation
        );
        assert!((after.translation.y - before.translation.y).abs() < 1e-5, "Y is untouched");
        assert!((after.translation.z - before.translation.z).abs() < 1e-5, "Z is untouched");
        assert!(
            (after.translation.x - mid.translation.x).abs() < 1e-5,
            "the released position is the one that sticks"
        );

        // One undo step restores the whole gesture.
        assert!(app.undo.can_undo(), "the gesture left an undo entry");
        app.undo_edit();
        let restored = app.scene.get(id).unwrap().transform;
        assert!(
            (restored.translation - before.translation).length() < 1e-5,
            "undo restored the start: {:?}",
            restored.translation
        );
    }

    /// One frame of raw input with Shift held, for the additive-rectangle test.
    fn shift_frame(time: f64, pointer: egui::Pos2, click: Option<bool>) -> egui::RawInput {
        let mut input = frame(time, pointer, click);
        input.modifiers = egui::Modifiers::SHIFT;
        input
    }

    /// Drag a rectangle from `a` to `b` with the Select tool, then let go.
    ///
    /// A hover to settle, a press, a move past egui's click threshold, a move
    /// to the far corner, a release: the frames a user's drag is made of.
    /// `shift` holds the additive modifier for the whole gesture.
    fn drag_rectangle(
        ctx: &egui::Context,
        app: &mut OpenForgeApp,
        t: f64,
        a: egui::Pos2,
        b: egui::Pos2,
        shift: bool,
    ) {
        let at = |time: f64, pointer: egui::Pos2, click: Option<bool>| {
            if shift {
                shift_frame(time, pointer, click)
            } else {
                frame(time, pointer, click)
            }
        };
        step(ctx, app, at(t, a, None));
        step(ctx, app, at(t + 0.1, a, Some(true)));
        let mid = egui::pos2((a.x + b.x) / 2.0, (a.y + b.y) / 2.0);
        step(ctx, app, at(t + 0.2, mid, None));
        step(ctx, app, at(t + 0.3, b, None));
        step(ctx, app, at(t + 0.4, b, Some(false)));
    }

    /// Where the selection's gizmo sits on screen: the app's own projection of
    /// the mesh's centre, which is what a test needs to aim a rectangle at
    /// geometry it cannot see.
    fn gizmo_origin(ctx: &egui::Context, app: &mut OpenForgeApp, id: NodeId, t: f64) -> egui::Pos2 {
        let tool = app.tool;
        app.selection.select_node(id);
        app.tool = ActiveTool::Move;
        step(ctx, app, frame(t, egui::pos2(500.0, 350.0), None));
        let origin = app
            .gizmo
            .layout
            .as_ref()
            .expect("a selection lays the gizmo out")
            .origin;
        app.tool = tool;
        app.selection.clear();
        origin
    }

    /// How many elements of `id` the selection holds.
    fn selected_elements(app: &OpenForgeApp, id: NodeId) -> usize {
        app.selection.selected_elements(id).map(|s| s.len()).unwrap_or(0)
    }

    /// The selected element indices of `id`, ascending.
    fn selected_indices(app: &OpenForgeApp, id: NodeId) -> Vec<u32> {
        app.selection
            .selected_elements(id)
            .map(|s| s.iter().copied().collect())
            .unwrap_or_default()
    }

    /// A 2×2 quad flat on the XY plane, centred on the origin and facing +Z.
    ///
    /// With the camera turned to look straight down -Z it projects as an
    /// axis-aligned square around the gizmo origin: its bottom edge runs
    /// below the centre, its top edge above it, its left and right edges to
    /// the sides, which makes every edge's screen position predictable. Its
    /// four edges walk the corners in order, so the bottom edge is edge 0,
    /// the right 1, the top 2 and the left 3.
    fn centred_quad() -> Mesh {
        let mut m = Mesh::new();
        m.add_vertex(Vec3::new(-1.0, -1.0, 0.0));
        m.add_vertex(Vec3::new(1.0, -1.0, 0.0));
        m.add_vertex(Vec3::new(1.0, 1.0, 0.0));
        m.add_vertex(Vec3::new(-1.0, 1.0, 0.0));
        m.add_face(&[0, 1, 2, 3]);
        m
    }

    /// A 2×2 grid of quads on the XY plane: 9 vertices and four faces — 0
    /// lower left, 1 lower right, 2 upper left, 3 upper right — so face 0's
    /// edge neighbours are 1 and 2 while 3 touches it only at a corner.
    fn face_grid() -> Mesh {
        let mut m = Mesh::new();
        for y in 0..3 {
            for x in 0..3 {
                m.add_vertex(Vec3::new(x as f32, y as f32, 0.0));
            }
        }
        m.add_face(&[0, 1, 4, 3]);
        m.add_face(&[1, 2, 5, 4]);
        m.add_face(&[3, 4, 7, 6]);
        m.add_face(&[4, 5, 8, 7]);
        m
    }

    /// Point the camera straight down -Z at the origin and return where the
    /// quad's centre lands on screen.
    fn face_quad(ctx: &egui::Context, app: &mut OpenForgeApp, id: NodeId) -> egui::Pos2 {
        app.camera.yaw = std::f32::consts::FRAC_PI_2;
        app.camera.pitch = 0.0;
        gizmo_origin(ctx, app, id, 0.0)
    }

    /// Whether a frame painted a rectangle with exactly this rect.
    fn paints_rect(shapes: &[egui::epaint::ClippedShape], rect: egui::Rect) -> bool {
        fn walk(shape: &egui::Shape, rect: egui::Rect) -> bool {
            match shape {
                egui::Shape::Rect(r) => r.rect == rect,
                egui::Shape::Vec(inner) => inner.iter().any(|s| walk(s, rect)),
                _ => false,
            }
        }
        shapes.iter().any(|clipped| walk(&clipped.shape, rect))
    }

    #[test]
    fn a_dragged_rectangle_selects_the_faces_inside_it() {
        let ctx = egui::Context::default();
        let mut app = OpenForgeApp::with_gpu(None);
        let id = app
            .scene
            .iter()
            .find(|n| n.is_mesh())
            .map(|n| n.id)
            .expect("the seed sphere exists");
        let faces = app.scene.get(id).unwrap().mesh().unwrap().faces.len();
        let centre = gizmo_origin(&ctx, &mut app, id, 0.0);
        app.tool = ActiveTool::Select;
        app.selection.level = SelectLevel::Face;

        // A rectangle over the middle of the sphere: what the frame shows
        // there is what a drag there selects.
        let a = centre + egui::vec2(-70.0, -50.0);
        let b = centre + egui::vec2(70.0, 50.0);
        step(&ctx, &mut app, frame(1.0, a, None));
        step(&ctx, &mut app, frame(1.1, a, Some(true)));
        let shapes = paint_once(&ctx, &mut app, frame(1.2, b, None));

        let live = app.marquee.expect("a drag draws a rectangle");
        assert_eq!(live.0, a, "the rectangle starts where the button went down");
        assert_eq!(live.1, b, "and follows the pointer while it drags");
        let box_ = egui::Rect::from_two_pos(a, b);
        assert!(paints_rect(&shapes, box_), "the live rectangle is painted");

        step(&ctx, &mut app, frame(1.3, b, Some(false)));
        assert!(app.marquee.is_none(), "letting go takes the rectangle down");
        let selected = selected_elements(&app, id);
        assert!(selected > 0, "the rectangle selected faces");
        assert!(
            selected < faces,
            "and left the faces outside it alone: {selected} of {faces}"
        );
        assert!(app.selection.is_node_selected(id), "the node comes with its faces");
    }

    #[test]
    fn shift_extends_the_selection_a_rectangle_found_and_a_plain_one_replaces_it() {
        let ctx = egui::Context::default();
        let mut app = OpenForgeApp::with_gpu(None);
        let id = app
            .scene
            .iter()
            .find(|n| n.is_mesh())
            .map(|n| n.id)
            .expect("the seed sphere exists");
        let centre = gizmo_origin(&ctx, &mut app, id, 0.0);
        app.tool = ActiveTool::Select;
        app.selection.level = SelectLevel::Face;

        let near = (
            centre + egui::vec2(-70.0, -50.0),
            centre + egui::vec2(70.0, 50.0),
        );
        let wide = (
            centre + egui::vec2(-130.0, -130.0),
            centre + egui::vec2(130.0, 130.0),
        );

        drag_rectangle(&ctx, &mut app, 1.0, near.0, near.1, false);
        let first = selected_elements(&app, id);
        assert!(first > 0, "the first rectangle found faces");

        // Shift keeps what was there and adds what the wider rectangle found.
        drag_rectangle(&ctx, &mut app, 2.0, wide.0, wide.1, true);
        let extended = selected_elements(&app, id);
        assert!(
            extended > first,
            "shift extended the selection: {first} -> {extended}"
        );

        // Without Shift the same rectangle replaces it instead.
        drag_rectangle(&ctx, &mut app, 3.0, near.0, near.1, false);
        assert_eq!(
            selected_elements(&app, id),
            first,
            "a plain rectangle replaces the selection with its own faces"
        );
    }

    #[test]
    fn an_object_rectangle_selects_the_nodes_it_holds_and_nothing_else() {
        let ctx = egui::Context::default();
        let mut app = OpenForgeApp::with_gpu(None);
        // A second node off to the side, so a rectangle can tell the two apart.
        let far = app.scene.create_mesh("Far", mesh::uv_sphere(0.2, 8, 6));
        app.scene.get_mut(far).unwrap().transform.translation = Vec3::new(6.0, 0.0, 0.0);
        let near = app
            .scene
            .iter()
            .find(|n| n.is_mesh() && n.id != far)
            .map(|n| n.id)
            .expect("the seed sphere exists");
        let centre = gizmo_origin(&ctx, &mut app, near, 0.0);
        app.tool = ActiveTool::Select;
        app.selection.level = SelectLevel::Object;

        // A rectangle over the seed sphere takes its node.
        drag_rectangle(
            &ctx,
            &mut app,
            1.0,
            centre + egui::vec2(-20.0, -20.0),
            centre + egui::vec2(20.0, 20.0),
            false,
        );
        let selected: Vec<NodeId> = app.selection.selected_nodes().collect();
        assert_eq!(selected, vec![near], "only the node under the rectangle");

        // A rectangle over empty space takes nothing, and lets go of what was
        // selected before.
        drag_rectangle(
            &ctx,
            &mut app,
            2.0,
            egui::pos2(20.0, 640.0),
            egui::pos2(60.0, 672.0),
            false,
        );
        assert_eq!(app.selection.node_count(), 0, "empty space selects nothing");
    }

    #[test]
    fn an_edge_click_selects_the_edge_under_the_pointer() {
        let ctx = egui::Context::default();
        let mut app = OpenForgeApp::with_gpu(None);
        app.scene = Scene::new();
        let id = app.scene.create_mesh("Quad", centred_quad());
        let centre = face_quad(&ctx, &mut app, id);
        app.tool = ActiveTool::Select;
        app.selection.level = SelectLevel::Edge;

        // The screen position of the quad's bottom edge, from the same
        // projection the app picks with — confirmed, like the snapping tests
        // do, against the gizmo the frame drew.
        let size = [SCREEN[0], SCREEN[1]];
        let vp = app.camera.view_proj(size[0] / size[1]);
        let (ex, ey, _) = viewport::project(vp, Vec3::new(0.0, -1.0, 0.0), size)
            .expect("the edge midpoint is in front");
        let (cx, cy, _) = viewport::project(vp, Vec3::ZERO, size).expect("the centre is in front");
        assert!(
            (egui::pos2(cx, cy) - centre).length() < 1.0,
            "the projection and the gizmo agree: ({cx}, {cy}) vs {centre:?}"
        );

        let on_edge = egui::pos2(ex, ey);
        step(&ctx, &mut app, frame(1.0, on_edge, None));
        step(&ctx, &mut app, frame(1.1, on_edge, Some(true)));
        step(&ctx, &mut app, frame(1.2, on_edge, Some(false)));
        assert_eq!(
            selected_indices(&app, id),
            vec![0],
            "the click must take the edge under it"
        );
        assert!(app.selection.is_node_selected(id), "the node comes with its edge");

        // The middle of the face is most of a world unit clear of every edge:
        // a click there takes no edge, and an edge level has nothing else.
        app.selection.clear();
        let inside = egui::pos2(cx, cy);
        step(&ctx, &mut app, frame(2.0, inside, None));
        step(&ctx, &mut app, frame(2.1, inside, Some(true)));
        step(&ctx, &mut app, frame(2.2, inside, Some(false)));
        assert_eq!(
            app.selection.total_element_count(),
            0,
            "no edge runs under the middle of the face"
        );
    }

    #[test]
    fn an_edge_rectangle_selects_the_edges_it_covers() {
        let ctx = egui::Context::default();
        let mut app = OpenForgeApp::with_gpu(None);
        app.scene = Scene::new();
        let id = app.scene.create_mesh("Quad", centred_quad());
        let centre = face_quad(&ctx, &mut app, id);
        app.tool = ActiveTool::Select;
        app.selection.level = SelectLevel::Edge;

        // The quad's sides sit about 84 px from its centre at the default
        // distance, so a rectangle over the upper-left corner covers the top
        // edge (2 → 3) and the left edge (0 → 3) and neither of the other two.
        drag_rectangle(
            &ctx,
            &mut app,
            1.0,
            centre + egui::vec2(-110.0, -110.0),
            centre + egui::vec2(-10.0, -10.0),
            false,
        );
        assert_eq!(
            selected_indices(&app, id),
            vec![2, 3],
            "only the edges the rectangle covers"
        );
        assert!(app.selection.is_node_selected(id), "the node comes with its edges");

        // A rectangle over the whole quad takes all four edges.
        drag_rectangle(
            &ctx,
            &mut app,
            2.0,
            centre + egui::vec2(-110.0, -110.0),
            centre + egui::vec2(110.0, 110.0),
            false,
        );
        assert_eq!(
            selected_indices(&app, id),
            vec![0, 1, 2, 3],
            "the whole quad is all four edges"
        );
        assert_eq!(app.status, "Select — 4 edge(s) selected.", "the status counts them");
    }

    #[test]
    fn the_overlay_falls_back_to_the_outline_only_at_the_edge_level() {
        use viewport::Wireframe;
        assert_eq!(
            overlay_mode(SelectLevel::Edge, Wireframe::Off),
            Wireframe::Selection,
            "an off overlay still shows what an edge pick aims at"
        );
        assert_eq!(overlay_mode(SelectLevel::Face, Wireframe::Off), Wireframe::Off);
        assert_eq!(overlay_mode(SelectLevel::Vertex, Wireframe::Off), Wireframe::Off);
        assert_eq!(
            overlay_mode(SelectLevel::Edge, Wireframe::All),
            Wireframe::All,
            "a chosen mode is drawn as chosen"
        );
        assert_eq!(
            overlay_mode(SelectLevel::Edge, Wireframe::Selection),
            Wireframe::Selection
        );
    }

    #[test]
    fn edge_level_draws_the_outline_and_marks_the_picked_edges() {
        let ctx = egui::Context::default();
        let mut app = OpenForgeApp::with_gpu(None);
        app.scene = Scene::new();
        let id = app.scene.create_mesh("Quad", centred_quad());
        let centre = face_quad(&ctx, &mut app, id);
        app.tool = ActiveTool::Select;
        app.selection.level = SelectLevel::Edge;
        let highlight = egui::Color32::from_rgb(
            viewport::ELEMENT_SELECTED_SRGB[0],
            viewport::ELEMENT_SELECTED_SRGB[1],
            viewport::ELEMENT_SELECTED_SRGB[2],
        );
        let outline = egui::Color32::from_rgb(
            viewport::EDGE_NODE_SELECTED_SRGB[0],
            viewport::EDGE_NODE_SELECTED_SRGB[1],
            viewport::EDGE_NODE_SELECTED_SRGB[2],
        );

        // Nothing picked yet: no node is being edited, so the fallback
        // outline has nothing to outline and nothing is highlighted.
        let shapes = paint_once(&ctx, &mut app, frame(0.0, egui::pos2(60.0, 60.0), None));
        assert!(painted_strokes(&shapes, highlight).is_empty());
        assert!(painted_strokes(&shapes, outline).is_empty());

        // A rectangle over the upper-left corner takes the top and left edges.
        drag_rectangle(
            &ctx,
            &mut app,
            1.0,
            centre + egui::vec2(-110.0, -110.0),
            centre + egui::vec2(-10.0, -10.0),
            false,
        );
        assert_eq!(selected_indices(&app, id), vec![2, 3]);
        assert_eq!(
            app.wireframe,
            viewport::Wireframe::Off,
            "the fallback is the frame's, not the setting's"
        );

        let shapes = paint_once(&ctx, &mut app, frame(2.0, egui::pos2(60.0, 60.0), None));
        assert_eq!(
            painted_strokes(&shapes, outline).len(),
            4,
            "the quad's edges, outlined through the off overlay"
        );
        let picked = painted_strokes(&shapes, highlight);
        assert_eq!(picked.len(), 2, "the two picked edges are marked");
        assert!(picked.iter().all(|(_, width)| *width == 3.0), "boldly: {picked:?}");

        // And the marks land on the edges themselves: the top edge joins the
        // corners (1,1,0) and (-1,1,0), the left edge (-1,-1,0) and (-1,1,0).
        let size = [SCREEN[0], SCREEN[1]];
        let vp = app.camera.view_proj(size[0] / size[1]);
        let at = |p: Vec3| {
            let (x, y, _) = viewport::project(vp, p, size).expect("the quad faces the camera");
            egui::pos2(x, y)
        };
        let fits = |seg: &[egui::Pos2; 2], want: [egui::Pos2; 2]| {
            (seg[0].distance(want[0]) < 1.5 && seg[1].distance(want[1]) < 1.5)
                || (seg[0].distance(want[1]) < 1.5 && seg[1].distance(want[0]) < 1.5)
        };
        let top = [at(Vec3::new(1.0, 1.0, 0.0)), at(Vec3::new(-1.0, 1.0, 0.0))];
        let left = [at(Vec3::new(-1.0, -1.0, 0.0)), at(Vec3::new(-1.0, 1.0, 0.0))];
        assert!(
            picked.iter().any(|(seg, _)| fits(seg, top)),
            "the top edge is marked: {picked:?}"
        );
        assert!(
            picked.iter().any(|(seg, _)| fits(seg, left)),
            "the left edge is marked: {picked:?}"
        );

        // Face level has no edge highlight to draw: the aid belongs to the
        // level that picked the edges.
        app.selection.level = SelectLevel::Face;
        let shapes = paint_once(&ctx, &mut app, frame(3.0, egui::pos2(60.0, 60.0), None));
        assert!(
            painted_strokes(&shapes, highlight).is_empty(),
            "no edge highlight away from the edge level"
        );
    }

    #[test]
    fn grid_snapping_lands_a_move_on_a_grid_value() {
        let ctx = egui::Context::default();

        // The same drag twice: once free, to learn where it would have landed,
        // and once snapped, to see it land on the grid instead.
        let drag = egui::vec2(163.0, 0.0);
        let free = {
            let (mut app, id) = app_ready_to_move();
            step(&ctx, &mut app, frame(0.0, egui::pos2(500.0, 350.0), None));
            let tip = arm_tip(&app, gizmo::Axis::X);
            step(&ctx, &mut app, frame(1.0, tip, Some(true)));
            step(&ctx, &mut app, frame(2.0, tip + drag, None));
            step(&ctx, &mut app, frame(3.0, tip + drag, Some(false)));
            app.scene.get(id).unwrap().transform.translation.x
        };
        assert!(free > 0.05, "the free drag moved ({free})");
        assert!(
            (free / 0.25 - (free / 0.25).round()).abs() > 1e-3,
            "and did not land on a grid value by luck ({free})"
        );

        let (mut app, id) = app_ready_to_move();
        app.snap = snap::Snap { mode: SnapMode::Grid, grid: 0.25, ..app.snap };
        step(&ctx, &mut app, frame(4.0, egui::pos2(500.0, 350.0), None));
        let tip = arm_tip(&app, gizmo::Axis::X);
        step(&ctx, &mut app, frame(5.0, tip, Some(true)));
        step(&ctx, &mut app, frame(6.0, tip + drag, None));
        assert!(
            app.status.contains("snap grid 0.25") && app.status.contains("Move"),
            "the readout says what it snapped to: {}",
            app.status
        );
        step(&ctx, &mut app, frame(7.0, tip + drag, Some(false)));

        let x = app.scene.get(id).unwrap().transform.translation.x;
        assert!(
            (x - snap::snap_to_step(free, 0.25)).abs() < 1e-3,
            "the same drag came to rest on the nearest grid value: {free} -> {x}"
        );
        assert!((x / 0.25 - (x / 0.25).round()).abs() < 1e-4, "x = {x} sits exactly on the grid");

        // One undo entry covers the whole snapped gesture.
        assert!(app.undo.can_undo());
        app.undo_edit();
        assert_eq!(app.scene.get(id).unwrap().transform.translation.x, 0.0);
    }

    #[test]
    fn holding_ctrl_inverts_snapping() {
        let ctx = egui::Context::default();
        let drag = egui::vec2(150.0, 0.0);

        // Snapping off + Ctrl: the drag snaps to the grid.
        let (mut app, id) = app_ready_to_move();
        assert_eq!(app.snap.mode, SnapMode::Off);
        step(&ctx, &mut app, frame(0.0, egui::pos2(500.0, 350.0), None));
        let tip = arm_tip(&app, gizmo::Axis::X);
        step(&ctx, &mut app, ctrl_frame(1.0, tip, Some(true)));
        step(&ctx, &mut app, ctrl_frame(2.0, tip + drag, None));
        assert!(app.status.contains("snap grid"), "{}", app.status);
        step(&ctx, &mut app, ctrl_frame(3.0, tip + drag, Some(false)));
        let x = app.scene.get(id).unwrap().transform.translation.x;
        let steps = x / app.snap.grid;
        assert!((steps - steps.round()).abs() < 1e-4, "x = {x} landed on the grid");

        // Snapping on + Ctrl: the drag is free again.
        let (mut app, id) = app_ready_to_move();
        app.snap = snap::Snap { mode: SnapMode::Grid, grid: 0.5, ..app.snap };
        step(&ctx, &mut app, frame(4.0, egui::pos2(500.0, 350.0), None));
        let tip = arm_tip(&app, gizmo::Axis::X);
        step(&ctx, &mut app, ctrl_frame(5.0, tip, Some(true)));
        step(&ctx, &mut app, ctrl_frame(6.0, tip + drag, None));
        assert!(!app.status.contains("snap"), "no readout while inverted: {}", app.status);
        step(&ctx, &mut app, ctrl_frame(7.0, tip + drag, Some(false)));
        let x = app.scene.get(id).unwrap().transform.translation.x;
        assert!(x > 0.05, "it still moved ({x})");
        let steps = x / 0.5;
        assert!((steps - steps.round()).abs() > 1e-3, "x = {x} is off the grid when inverted");
    }

    #[test]
    fn vertex_snapping_lands_the_grab_point_on_the_vertex_under_the_cursor() {
        let ctx = egui::Context::default();
        let (mut app, id) = app_ready_to_move();

        // A marker mesh with a lone vertex out at +X, and nothing else near it.
        let mut marker = Mesh::new();
        marker.add_vertex(Vec3::new(3.0, 0.0, 0.0));
        marker.add_vertex(Vec3::new(3.0, 1.0, 0.0));
        marker.add_vertex(Vec3::new(3.0, 0.0, 1.0));
        marker.add_face(&[0, 1, 2]);
        app.scene.create_mesh("Marker", marker);
        app.snap.mode = SnapMode::Vertex;

        step(&ctx, &mut app, frame(0.0, egui::pos2(500.0, 350.0), None));

        // The world -> screen mapping, confirmed against the gizmo itself: the
        // handles sit on the projected bounds centre.
        let size = [SCREEN[0], SCREEN[1]];
        let vp = app.camera.view_proj(size[0] / size[1]);
        let ids: Vec<NodeId> = app.selection.selected_nodes().collect();
        let centre = app.scene.world_bounds(&ids).center();
        let layout_origin = app.gizmo.layout.as_ref().expect("handles are laid out").origin;
        let (cx, cy, _) = viewport::project(vp, centre, size).expect("the centre is in front");
        assert!(
            (layout_origin - egui::pos2(cx, cy)).length() < 1.0,
            "the gizmo sits on the projected centre: {layout_origin:?} vs ({cx}, {cy})"
        );

        let target = Vec3::new(3.0, 0.0, 0.0);
        let (vx, vy, _) = viewport::project(vp, target, size).expect("the marker is in front");
        let tip = arm_tip(&app, gizmo::Axis::X);
        step(&ctx, &mut app, frame(1.0, tip, Some(true)));
        assert_eq!(app.gizmo.active, Some(gizmo::Axis::X));

        // Drag with the cursor over the marker's vertex.
        let over = egui::pos2(vx, vy);
        step(&ctx, &mut app, frame(2.0, over, None));
        assert!(app.snap_marker.is_some(), "the target is marked while the drag runs");
        assert!(app.status.contains("snap vertex"), "{}", app.status);
        step(&ctx, &mut app, frame(3.0, over, Some(false)));
        assert!(app.snap_marker.is_none(), "the marker goes when the drag does");

        // The grabbed point — the selection's centre — landed on the vertex.
        let moved = app.scene.world_bounds(&ids).center();
        assert!(
            (moved.x - target.x).abs() < 1e-3,
            "the grab point landed on the vertex: {} vs {}",
            moved.x,
            target.x
        );
        assert!((moved.y - centre.y).abs() < 1e-4, "nothing moved off the dragged axis");
        assert!(
            app.scene.get(id).unwrap().transform.translation.x > 1.5,
            "and the node went with the grab point"
        );
    }

    #[test]
    fn the_g_key_cycles_the_snap_mode() {
        let ctx = egui::Context::default();
        let (mut app, _id) = app_ready_to_move();
        assert_eq!(app.snap.mode, SnapMode::Off);

        step_with_keys(&ctx, &mut app, key_frame(0.0, egui::Key::G));
        assert_eq!(app.snap.mode, SnapMode::Grid);
        step_with_keys(&ctx, &mut app, key_frame(1.0, egui::Key::G));
        assert_eq!(app.snap.mode, SnapMode::Vertex);
        step_with_keys(&ctx, &mut app, key_frame(2.0, egui::Key::G));
        assert_eq!(app.snap.mode, SnapMode::Off, "the cycle comes round again");
    }

    #[test]
    fn a_vertex_level_drag_moves_only_the_selected_vertex() {
        let ctx = egui::Context::default();
        let (mut app, id) = app_ready_to_move();

        // Pick a vertex that faces the camera, so its axis arm is grabbable.
        let eye = app.camera.eye();
        let mesh = app.scene.get(id).unwrap().mesh().unwrap().clone();
        let front = (0..mesh.positions.len())
            .max_by(|a, b| {
                mesh.positions[*a].dot(eye).partial_cmp(&mesh.positions[*b].dot(eye)).unwrap()
            })
            .unwrap();
        app.selection.level = SelectLevel::Vertex;
        app.selection.select_node(id);
        app.selection.select_elements(id, [front as u32]);

        let before: Vec<Vec3> = mesh.positions.clone();
        let start = app.scene.get(id).unwrap().transform;
        step(&ctx, &mut app, frame(0.0, egui::pos2(500.0, 350.0), None));

        // The gizmo sits on the vertex itself, not on the whole node.
        let pivot = app.gizmo.layout.as_ref().expect("handles are laid out").origin;
        let tip = arm_tip(&app, gizmo::Axis::X);
        assert!(
            (tip - pivot).length() > 20.0,
            "the X arm runs away from the pivot"
        );

        step(&ctx, &mut app, frame(1.0, tip, Some(true)));
        assert_eq!(app.gizmo.active, Some(gizmo::Axis::X), "the arm was grabbed");
        step(&ctx, &mut app, frame(2.0, tip + egui::vec2(120.0, 0.0), None));
        step(&ctx, &mut app, frame(3.0, tip + egui::vec2(120.0, 0.0), Some(false)));
        assert!(!app.gizmo.is_dragging());

        let after: Vec<Vec3> = app.scene.get(id).unwrap().mesh().unwrap().positions.clone();
        assert_eq!(after.len(), before.len());
        let moved = after[front] - before[front];
        assert!(moved.length() > 0.05, "the picked vertex moved: {moved:?}");
        for (i, (a, b)) in after.iter().zip(&before).enumerate() {
            if i == front {
                continue;
            }
            assert!(
                (*a - *b).length() < 1e-5,
                "vertex {i} should not have moved: {:?} -> {:?}",
                b,
                a
            );
        }
        // Sub-object editing never touches the node's own transform.
        assert_eq!(app.scene.get(id).unwrap().transform, start);
        assert!(app.undo.can_undo(), "the vertex drag is undoable");
    }

    #[test]
    fn a_press_that_misses_the_handles_falls_through_to_the_free_drag() {
        let ctx = egui::Context::default();
        let (mut app, id) = app_ready_to_move();
        step(&ctx, &mut app, frame(0.0, egui::pos2(500.0, 350.0), None));
        let before = app.scene.get(id).unwrap().transform;

        // Press in an empty corner, well away from the gizmo, and drag away.
        let empty = egui::pos2(940.0, 40.0);
        let moved = empty + egui::vec2(-120.0, 90.0);
        step(&ctx, &mut app, frame(1.0, empty, Some(true)));
        assert!(!app.gizmo.is_dragging(), "the handles were not grabbed");
        step(&ctx, &mut app, frame(2.0, moved, None));
        step(&ctx, &mut app, frame(3.0, moved + egui::vec2(-20.0, 20.0), None));
        step(&ctx, &mut app, frame(4.0, moved + egui::vec2(-20.0, 20.0), Some(false)));

        // The gizmo never claimed the gesture, so no axis state is left behind.
        assert_eq!(app.gizmo.active, None);
        assert!(!app.gizmo.is_dragging());

        // The gesture went to the free-form tool instead, which slides the
        // selection through the view plane: moved, but not off the ground.
        let delta = app.scene.get(id).unwrap().transform.translation - before.translation;
        assert!(delta.length() > 0.05, "the free drag moved the node: {delta:?}");
        assert!(delta.y.abs() < 1e-4, "view-plane motion keeps the height: {delta:?}");
        assert!(app.undo.can_undo(), "the free drag is undoable too");
    }

    #[test]
    fn the_rotate_tool_uses_rings_instead_of_arms() {
        let ctx = egui::Context::default();
        let (mut app, _id) = app_ready_to_move();
        app.tool = ActiveTool::Rotate;
        step(&ctx, &mut app, frame(0.0, egui::pos2(500.0, 350.0), None));

        // A point on the Z ring is grabbed as the Z axis, not as an arm tip.
        let ring: Vec<egui::Pos2> = app
            .gizmo
            .layout
            .as_ref()
            .expect("rings are laid out")
            .rings
            .iter()
            .find(|(a, pts)| *a == gizmo::Axis::Z && pts.len() > 4)
            .map(|(_, pts)| pts.clone())
            .expect("the Z ring projects");
        let on_ring = ring[2];
        step(&ctx, &mut app, frame(1.0, on_ring, None));
        assert_eq!(app.gizmo.hovered, Some(gizmo::Axis::Z));

        let before = app.scene.get(_id).unwrap().transform;
        step(&ctx, &mut app, frame(2.0, on_ring, Some(true)));
        assert_eq!(app.gizmo.active, Some(gizmo::Axis::Z));
        // Sweep a quarter of the way round the ring and release.
        let quarter = ring[ring.len() / 4];
        step(&ctx, &mut app, frame(3.0, quarter, None));
        step(&ctx, &mut app, frame(4.0, quarter, Some(false)));

        let after = app.scene.get(_id).unwrap().transform;
        assert!(
            before.rotation.angle_between(after.rotation) > 0.05,
            "the node spun: {:?} -> {:?}",
            before.rotation,
            after.rotation
        );
        assert!(
            (after.translation - before.translation).length() < 1e-5,
            "rotating does not move the node"
        );
    }

    #[test]
    fn scaling_an_arm_scales_the_selection() {
        let ctx = egui::Context::default();
        let (mut app, id) = app_ready_to_move();
        app.tool = ActiveTool::Scale;
        step(&ctx, &mut app, frame(0.0, egui::pos2(500.0, 350.0), None));

        let tip = arm_tip(&app, gizmo::Axis::Y);
        let before = app.scene.get(id).unwrap().transform;
        step(&ctx, &mut app, frame(1.0, tip, Some(true)));
        assert_eq!(app.gizmo.active, Some(gizmo::Axis::Y));
        // Push the handle further from the gizmo origin.
        let out = tip + (tip - app.gizmo.layout.as_ref().unwrap().origin) * 0.5;
        step(&ctx, &mut app, frame(2.0, out, None));
        step(&ctx, &mut app, frame(3.0, out, Some(false)));

        let after = app.scene.get(id).unwrap().transform;
        assert!(
            after.scale.y > before.scale.y * 1.05,
            "scaled up: {:?} -> {:?}",
            before.scale,
            after.scale
        );
        assert!(
            (after.translation - before.translation).length() < 1e-5,
            "scaling does not move the node"
        );
    }

    #[test]
    fn the_move_gizmo_paints_one_stroke_per_axis() {
        let ctx = egui::Context::default();
        let (mut app, _id) = app_ready_to_move();

        // With no gizmo drawn, the frame is just the grid and the mesh.
        app.tool = ActiveTool::Select;
        let (plain, _) = stroked(&paint_once(&ctx, &mut app, frame(0.0, egui::pos2(500.0, 350.0), None)));

        app.tool = ActiveTool::Move;
        let (armed, _) = stroked(&paint_once(&ctx, &mut app, frame(1.0, egui::pos2(500.0, 350.0), None)));
        assert_eq!(
            armed,
            plain + 3,
            "the Move gizmo adds exactly three axis arms"
        );
    }

    #[test]
    fn the_rotate_gizmo_paints_rings_instead_of_arms() {
        let ctx = egui::Context::default();
        let (mut app, _id) = app_ready_to_move();

        app.tool = ActiveTool::Move;
        let (move_lines, _) = stroked(&paint_once(&ctx, &mut app, frame(0.0, egui::pos2(500.0, 350.0), None)));

        app.tool = ActiveTool::Rotate;
        let (rotate_lines, rotate_paths) =
            stroked(&paint_once(&ctx, &mut app, frame(1.0, egui::pos2(500.0, 350.0), None)));

        assert!(
            rotate_paths >= 3,
            "one closed ring per axis, got {rotate_paths}"
        );
        assert_eq!(
            rotate_lines + 3,
            move_lines,
            "the arms are replaced by rings, not drawn as well"
        );
    }

    #[test]
    fn the_wireframe_cycles_from_the_keyboard() {
        let ctx = egui::Context::default();
        let (mut app, _id) = app_ready_to_move();
        assert_eq!(app.wireframe, viewport::Wireframe::Off, "shaded only by default");

        step_with_keys(&ctx, &mut app, key_frame(0.0, egui::Key::Z));
        assert_eq!(app.wireframe, viewport::Wireframe::Selection);
        step_with_keys(&ctx, &mut app, key_frame(1.0, egui::Key::Z));
        assert_eq!(app.wireframe, viewport::Wireframe::All);
        step_with_keys(&ctx, &mut app, key_frame(2.0, egui::Key::Z));
        assert_eq!(app.wireframe, viewport::Wireframe::Off, "the cycle closes");
    }

    #[test]
    fn the_wireframe_overlay_adds_edges_for_the_selection_and_then_the_rest() {
        let ctx = egui::Context::default();
        let (mut app, _id) = app_ready_to_move();
        // A second, *unselected* mesh, so Selection and All have to differ.
        app.scene.create_mesh("Box", primitive_box(1.0));

        let stroke_count = |app: &mut OpenForgeApp, ctx: &egui::Context, mode: viewport::Wireframe| {
            app.wireframe = mode;
            let shapes = paint_once(ctx, app, frame(0.0, egui::pos2(500.0, 350.0), None));
            stroked(&shapes).0
        };

        let off = stroke_count(&mut app, &ctx, viewport::Wireframe::Off);
        let selection = stroke_count(&mut app, &ctx, viewport::Wireframe::Selection);
        let all = stroke_count(&mut app, &ctx, viewport::Wireframe::All);

        assert!(selection > off, "the outline strokes the sphere's edges: {off} -> {selection}");
        // From this camera three of the box's six faces look at it, so nine of
        // its twelve edges survive the facing test — and only All reaches them.
        assert_eq!(
            all - selection,
            9,
            "All adds the unselected box's visible edges: {selection} -> {all}"
        );
    }

    #[test]
    fn select_all_fills_every_level_and_none_empties_it() {
        let (mut app, id) = app_ready_to_move();
        let mesh = app.scene.get(id).unwrap().mesh().unwrap().clone();
        for (level, count) in [
            (SelectLevel::Vertex, mesh.positions.len() as u32),
            (SelectLevel::Edge, mesh.edges().len() as u32),
            (SelectLevel::Face, mesh.faces.len() as u32),
        ] {
            app.selection.level = level;
            app.selection.clear_elements();
            app.select_all_elements();
            assert_eq!(
                app.selection.total_element_count(),
                count as usize,
                "{level:?} level takes every one of its own elements"
            );
            assert_eq!(
                app.status,
                format!(
                    "Select All — {count} {}(s) selected.",
                    OpenForgeApp::element_word(level)
                )
            );
            assert!(app.selection.is_node_selected(id), "the node stays the one edited");

            app.clear_element_selection();
            assert_eq!(app.selection.total_element_count(), 0, "{level:?} cleared");
            assert!(app.selection.is_node_selected(id), "clearing elements keeps the node");
        }

        // The object level selects every visible node, and clears them too.
        app.selection.level = SelectLevel::Object;
        app.selection.clear();
        app.scene.create_mesh("Box", primitive_box(1.0));
        app.select_all_elements();
        assert_eq!(app.selection.node_count(), 2, "both meshes");
        app.clear_element_selection();
        assert_eq!(app.selection.node_count(), 0, "and none clears the nodes");
        assert_eq!(app.status, "Select None — nothing selected.");
    }

    #[test]
    fn the_select_tools_ask_for_a_mesh_and_the_level() {
        let mut app = OpenForgeApp::with_gpu(None);
        app.selection.level = SelectLevel::Face;
        app.selection.clear();
        app.select_all_elements();
        assert_eq!(app.status, "Select a mesh first.");
        assert_eq!(app.selection.total_element_count(), 0);
        app.invert_element_selection();
        assert_eq!(app.status, "Select a mesh first.");
        app.grow_element_selection();
        assert_eq!(app.status, "Select a mesh first.");
        app.hide_selected_nodes();
        assert_eq!(app.status, "Select a node to hide.");
        app.isolate_selected_nodes();
        assert_eq!(app.status, "Select a node to isolate.");
        app.unhide_all_nodes();
        assert_eq!(app.status, "Every node is already visible.");

        // Growing works at the sub-object levels only: there is no topology
        // between nodes.
        let (mut app, _id) = app_ready_to_move();
        app.selection.level = SelectLevel::Object;
        app.grow_element_selection();
        assert!(
            app.status.contains("vertex, edge and face"),
            "the object level is told where growing works: {}",
            app.status
        );
    }

    #[test]
    fn invert_flips_the_element_selection() {
        let mut app = OpenForgeApp::with_gpu(None);
        app.scene = Scene::new();
        let id = app.scene.create_mesh("Grid", face_grid());
        app.selection.level = SelectLevel::Face;
        app.selection.select_node(id);
        app.selection.set_elements(id, [0u32]);

        app.invert_element_selection();
        assert_eq!(selected_indices(&app, id), vec![1, 2, 3]);
        assert_eq!(app.status, "Invert — 3 face(s) selected.");
        app.invert_element_selection();
        assert_eq!(selected_indices(&app, id), vec![0], "inverting twice is where it began");
    }

    #[test]
    fn grow_and_shrink_step_the_face_selection_on_the_edited_mesh() {
        let mut app = OpenForgeApp::with_gpu(None);
        app.scene = Scene::new();
        let id = app.scene.create_mesh("Grid", face_grid());
        app.selection.level = SelectLevel::Face;
        app.selection.select_node(id);
        app.selection.set_elements(id, [0u32]);

        app.grow_element_selection();
        assert_eq!(
            selected_indices(&app, id),
            vec![0, 1, 2],
            "the edge neighbours join; the corner one does not"
        );
        assert_eq!(app.status, "Grow — 3 face(s) selected.");

        app.shrink_element_selection();
        assert_eq!(selected_indices(&app, id), vec![0], "shrink gives the seed back");
        assert_eq!(app.status, "Shrink — 1 face(s) selected.");

        // Two neighbours of each other still touch unselected faces, so one
        // step cuts both away.
        app.selection.set_elements(id, [0u32, 1]);
        app.shrink_element_selection();
        assert_eq!(app.selection.total_element_count(), 0);
    }

    #[test]
    fn the_edge_walks_extend_the_selection_to_the_loop_and_the_ring() {
        let ctx = egui::Context::default();
        let mut app = OpenForgeApp::with_gpu(None);
        app.scene = Scene::new();
        let id = app.scene.create_mesh("Grid", face_grid());
        app.selection.level = SelectLevel::Edge;
        app.selection.select_node(id);
        app.selection.set_elements(id, [1u32]);

        // The loop carries on straight through four-edge vertices, so from the
        // middle vertical it is the middle column.
        app.select_edge_loops();
        assert_eq!(selected_indices(&app, id), vec![1, 7]);
        assert_eq!(app.status, "Loop — 2 edge(s) selected.");
        app.select_edge_loops();
        assert_eq!(selected_indices(&app, id), vec![1, 7], "a second press changes nothing");

        // The ring is the parallel band the walk crosses quads into.
        app.selection.set_elements(id, [1u32]);
        app.select_edge_rings();
        assert_eq!(selected_indices(&app, id), vec![1, 3, 5]);
        assert_eq!(app.status, "Ring — 3 edge(s) selected.");

        // Two edges contribute both bands.
        app.selection.set_elements(id, [1u32, 7]);
        app.select_edge_rings();
        assert_eq!(selected_indices(&app, id), vec![1, 3, 5, 7, 9, 10]);

        // The hotkeys drive the same walks.
        app.selection.set_elements(id, [1u32]);
        step_with_keys(&ctx, &mut app, key_frame_mods(0.0, egui::Key::L, egui::Modifiers::NONE));
        assert_eq!(selected_indices(&app, id), vec![1, 7], "L takes the loop");
        step_with_keys(&ctx, &mut app, key_frame_mods(1.0, egui::Key::L, egui::Modifiers::SHIFT));
        assert_eq!(
            selected_indices(&app, id),
            vec![1, 3, 5, 7, 9, 10],
            "Shift+L adds the rings of everything picked"
        );

        // It is an edge-level walk, and it needs an edge to walk from.
        app.selection.level = SelectLevel::Face;
        app.select_edge_loops();
        assert!(app.status.contains("edge-level"), "{}", app.status);
        app.selection.level = SelectLevel::Edge;
        app.selection.clear_elements();
        app.select_edge_rings();
        assert_eq!(app.status, "Select an edge to walk from.");
    }

    #[test]
    fn select_through_makes_the_picks_ignore_occlusion() {
        // The gate first: the occluding pass serves only when there is a
        // backend that has one, and only while Select Through is off.
        assert!(OpenForgeApp::occluding_picks(false, true));
        assert!(!OpenForgeApp::occluding_picks(true, true));
        assert!(!OpenForgeApp::occluding_picks(false, false));
        assert!(!OpenForgeApp::occluding_picks(true, false));

        let ctx = egui::Context::default();
        let mut app = OpenForgeApp::with_gpu(None);
        assert!(!app.select_through, "off by default");
        step_with_keys(&ctx, &mut app, key_frame(0.0, egui::Key::P));
        assert!(app.select_through, "P turns Select Through on");
        step_with_keys(&ctx, &mut app, key_frame(1.0, egui::Key::P));
        assert!(!app.select_through, "and P turns it off again");
        // The viewport rewrites the status line every frame, so the messages
        // are pinned on the action itself, like the Select menu's.
        app.toggle_select_through();
        assert_eq!(app.status, "Select Through on — picks ignore occlusion (P).");
        app.toggle_select_through();
        assert_eq!(app.status, "Select Through off — picks follow the drawn frame (P).");

        // This machine has no ID pass, so the picks answer from the projection
        // either way — the fallback the mode asks for on machines that have
        // one. A vertex behind a surface takes the click that lands on it.
        app.scene = Scene::new();
        let mut m = Mesh::new();
        for (x, y) in [(-1.0f32, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
            m.add_vertex(Vec3::new(x, y, 0.0));
        }
        let hidden = m.add_vertex(Vec3::new(0.0, 0.0, -1.0));
        m.add_face(&[0, 1, 2, 3]);
        let id = app.scene.create_mesh("Quad", m);
        app.selection.level = SelectLevel::Vertex;
        app.select_through = true;

        let size = [SCREEN[0], SCREEN[1]];
        let vp = app.camera.view_proj(size[0] / size[1]);
        let (x, y, _) = viewport::project(vp, Vec3::new(0.0, 0.0, -1.0), size)
            .expect("behind the quad, still in front of the camera");
        let pos = egui::pos2(x, y);
        step(&ctx, &mut app, frame(2.0, pos, None));
        step(&ctx, &mut app, frame(2.1, pos, Some(true)));
        step(&ctx, &mut app, frame(2.2, pos, Some(false)));
        assert_eq!(
            selected_indices(&app, id),
            vec![hidden],
            "the occluded vertex takes the click"
        );
        assert!(app.selection.is_node_selected(id), "and the node comes with it");
    }

    #[test]
    fn the_outliner_visibility_toggle_is_an_undoable_edit() {
        let mut app = OpenForgeApp::with_gpu(None);
        app.scene = Scene::new();
        let id = app.scene.create_mesh("Grid", face_grid());
        app.selection.select_node(id);

        app.set_node_visible(id, false);
        assert!(!app.scene.get(id).unwrap().visible, "the box hid the node");
        assert_eq!(app.status, "Hidden Grid.");
        assert!(!app.selection.is_node_selected(id), "hiding drops the selection");
        app.undo_edit();
        assert!(app.scene.get(id).unwrap().visible, "and it is one undo step");

        app.set_node_visible(id, false);
        app.set_node_visible(id, true);
        assert!(app.scene.get(id).unwrap().visible, "the box shows it again");
        assert_eq!(app.status, "Shown Grid.");
    }

    #[test]
    fn the_selection_hotkeys_drive_the_same_actions() {
        let ctx = egui::Context::default();
        let (mut app, id) = app_ready_to_move();
        let faces = app.scene.get(id).unwrap().mesh().unwrap().faces.len() as u32;
        app.selection.level = SelectLevel::Face;
        app.selection.clear_elements();

        step_with_keys(&ctx, &mut app, key_frame_mods(0.0, egui::Key::A, egui::Modifiers::COMMAND));
        assert_eq!(app.selection.total_element_count(), faces as usize, "Ctrl+A selects all");

        step_with_keys(
            &ctx,
            &mut app,
            key_frame_mods(1.0, egui::Key::A, egui::Modifiers::COMMAND | egui::Modifiers::SHIFT),
        );
        assert_eq!(app.selection.total_element_count(), 0, "Ctrl+Shift+A clears");

        step_with_keys(&ctx, &mut app, key_frame_mods(2.0, egui::Key::I, egui::Modifiers::COMMAND));
        assert_eq!(
            app.selection.total_element_count(),
            faces as usize,
            "Ctrl+I flips the empty set into the whole mesh"
        );

        // One face, grown and shrunk by key.
        app.selection.set_elements(id, [0u32]);
        step_with_keys(&ctx, &mut app, key_frame_mods(3.0, egui::Key::Plus, egui::Modifiers::COMMAND));
        let grown = app.selection.total_element_count();
        assert!(grown > 1, "Ctrl+Plus adds the neighbours ({grown})");
        step_with_keys(&ctx, &mut app, key_frame_mods(4.0, egui::Key::Minus, egui::Modifiers::COMMAND));
        assert_eq!(selected_indices(&app, id), vec![0], "Ctrl+Minus gives the seed back");
    }

    #[test]
    fn hiding_drops_the_node_from_the_selection_and_undo_brings_it_back() {
        let ctx = egui::Context::default();
        let (mut app, _id) = app_ready_to_move();
        let box_id = app
            .scene
            .create_mesh("Box", primitive_box(1.0));
        app.selection.clear();
        app.selection.select_node(box_id);
        app.wireframe = viewport::Wireframe::All;
        let before = stroked(&paint_once(&ctx, &mut app, frame(0.0, egui::pos2(60.0, 60.0), None))).0;

        app.hide_selected_nodes();
        assert!(!app.scene.get(box_id).unwrap().visible, "the box is hidden");
        assert!(!app.selection.is_node_selected(box_id), "and out of the selection");
        assert_eq!(app.status, "Hidden 1 node(s).");
        let after = stroked(&paint_once(&ctx, &mut app, frame(1.0, egui::pos2(60.0, 60.0), None))).0;
        assert!(
            after < before,
            "the frame stops drawing its edges: {before} -> {after}"
        );

        app.undo_edit();
        assert!(app.scene.get(box_id).unwrap().visible, "undo shows it again");
        // Hiding took the node out of the selection (and with it the gizmo);
        // select it again so the two frames differ in nothing else.
        app.selection.select_node(box_id);
        let restored = stroked(&paint_once(&ctx, &mut app, frame(2.0, egui::pos2(60.0, 60.0), None))).0;
        assert_eq!(restored, before, "and draws it exactly as before");
    }

    #[test]
    fn isolate_shows_only_the_selection_and_unhide_restores_everything() {
        let (mut app, id) = app_ready_to_move();
        let box_id = app.scene.create_mesh("Box", primitive_box(1.0));
        app.selection.clear();
        app.selection.select_node(id);

        app.isolate_selected_nodes();
        assert!(app.scene.get(id).unwrap().visible, "the selection stays shown");
        assert!(!app.scene.get(box_id).unwrap().visible, "everything else hides");
        assert_eq!(app.status, "Isolate — 1 node(s) visible, 1 hidden.");

        app.unhide_all_nodes();
        assert!(app.scene.get(box_id).unwrap().visible);
        assert_eq!(app.status, "Unhidden 1 node(s).");

        // Both steps are undoable scene edits.
        app.undo_edit();
        assert!(
            !app.scene.get(box_id).unwrap().visible,
            "undo puts the isolation back"
        );
        app.undo_edit();
        assert!(app.scene.get(box_id).unwrap().visible, "and undo shows it again");
    }

    #[test]
    fn the_face_selection_colours_the_edges_around_the_selected_face() {
        let ctx = egui::Context::default();
        let (mut app, id) = app_ready_to_move();
        app.selection.level = SelectLevel::Face;
        app.wireframe = viewport::Wireframe::Selection;

        // Rank the sphere's faces by how squarely they look at the camera: the
        // facing one's edges all survive, the one turned furthest away keeps
        // none of them.
        let eye = app.camera.eye();
        let mesh = app.scene.get(id).unwrap().mesh().unwrap().clone();
        let facing = |i: usize| {
            let face = &mesh.faces[i];
            let n = mesh.normals[face.corners[0].normal as usize];
            let mut centre = Vec3::ZERO;
            for c in &face.corners {
                centre += mesh.positions[c.position as usize];
            }
            n.dot(eye - centre / face.corners.len() as f32)
        };
        let toward = (0..mesh.faces.len()).max_by(|a, b| facing(*a).partial_cmp(&facing(*b)).unwrap()).unwrap();
        let away = (0..mesh.faces.len()).min_by(|a, b| facing(*a).partial_cmp(&facing(*b)).unwrap()).unwrap();
        let corners = mesh.faces[toward].corners.len();
        assert!(facing(away) < 0.0, "the sphere has faces turned away");

        let select_only = |app: &mut OpenForgeApp, face: usize| {
            app.selection.clear();
            app.selection.select_node(id);
            app.selection.select_elements(id, [face as u32]);
        };

        select_only(&mut app, toward);
        assert_eq!(
            painted_face_edge_count(&ctx, &mut app),
            corners,
            "every edge of the facing face is stroked in the highlight colour"
        );

        select_only(&mut app, away);
        assert_eq!(
            painted_face_edge_count(&ctx, &mut app),
            0,
            "a face turned away keeps its edges hidden"
        );
    }

    /// Drive one frame of the *whole* UI the way `update` does (minus the eframe
    /// frame): menu bar, shortcuts, toolbars, side panels, status bar, viewport.
    fn step_full(ctx: &egui::Context, app: &mut OpenForgeApp, input: egui::RawInput) {
        let _ = ctx.run(input, |ctx| {
            app.handle_menu(ctx);
            app.handle_shortcuts(ctx);
            app.handle_top_toolbar(ctx);
            app.handle_status_bar(ctx);
            app.handle_uv_editor(ctx);
            app.handle_side_panels(ctx);
            app.handle_central_viewport(ctx, None);
        });
    }

    /// Every fill colour the frame painted.
    fn painted_fills(shapes: &[egui::epaint::ClippedShape]) -> Vec<egui::Color32> {
        fn walk(shape: &egui::Shape, out: &mut Vec<egui::Color32>) {
            match shape {
                egui::Shape::Path(p) => out.push(p.fill),
                egui::Shape::Vec(inner) => {
                    for s in inner {
                        walk(s, out);
                    }
                }
                _ => {}
            }
        }
        let mut out = Vec::new();
        for clipped in shapes {
            walk(&clipped.shape, &mut out);
        }
        out
    }

    /// The stroked segments of one colour a frame painted, with their widths.
    fn painted_strokes(
        shapes: &[egui::epaint::ClippedShape],
        color: egui::Color32,
    ) -> Vec<([egui::Pos2; 2], f32)> {
        fn walk(
            shape: &egui::Shape,
            color: egui::Color32,
            out: &mut Vec<([egui::Pos2; 2], f32)>,
        ) {
            match shape {
                egui::Shape::LineSegment { points, stroke } => {
                    if stroke.color == egui::epaint::ColorMode::Solid(color) {
                        out.push((*points, stroke.width));
                    }
                }
                egui::Shape::Vec(inner) => {
                    for s in inner {
                        walk(s, color, out);
                    }
                }
                _ => {}
            }
        }
        let mut out = Vec::new();
        for clipped in shapes {
            walk(&clipped.shape, color, &mut out);
        }
        out
    }

    /// How many stroked segments the frame drew in the face-highlight colour.
    fn painted_face_edge_count(ctx: &egui::Context, app: &mut OpenForgeApp) -> usize {
        let highlight = egui::Color32::from_rgb(255, 170, 60);
        let shapes = paint_once(ctx, app, frame(0.0, egui::pos2(500.0, 350.0), None));
        let mut count = 0;
        fn walk(shape: &egui::Shape, highlight: egui::Color32, count: &mut usize) {
            match shape {
                egui::Shape::LineSegment { stroke, .. } => {
                    if stroke.color == egui::epaint::ColorMode::Solid(highlight) {
                        *count += 1;
                    }
                }
                egui::Shape::Vec(inner) => {
                    for s in inner {
                        walk(s, highlight, count);
                    }
                }
                _ => {}
            }
        }
        for clipped in &shapes {
            walk(&clipped.shape, highlight, &mut count);
        }
        count
    }

    #[test]
    fn the_whole_ui_paints_a_frame() {
        let ctx = egui::Context::default();
        let (mut app, _id) = app_ready_to_move();
        // With a selection...
        step_full(&ctx, &mut app, frame(0.0, egui::pos2(500.0, 350.0), None));
        // ...and with none, which takes a different branch in the panels.
        app.selection.clear();
        step_full(&ctx, &mut app, frame(1.0, egui::pos2(500.0, 350.0), None));
        assert_eq!(app.scene.node_count(), 1, "nothing was changed by painting");
    }

    #[test]
    fn the_material_library_starts_with_the_default_and_the_presets() {
        let library = default_materials();
        assert_eq!(library.material_count(), 6);
        // Slot 0 stays the default: it is what an unassigned face resolves to.
        assert_eq!(library.resolve(0).name, "Default");
        assert_eq!(library.resolve(2).name, "Painted Red");
        assert_eq!(library.resolve(4).name, "Gold");
        assert!(library.resolve(99).name == "Default", "out of range falls back");
        // One preset carries a procedural checkerboard, so texturing can be
        // seen without importing anything.
        assert_eq!(library.texture_count(), 1);
        assert_eq!(library.resolve_texture(Some(0)).unwrap().name, "Checker");
        assert_eq!(library.resolve(5).name, "Checker Test");
        assert_eq!(library.resolve(5).diffuse_map, Some(0));
        // The default material stays untextured.
        assert_eq!(library.resolve(0).diffuse_map, None);
    }

    /// Every mesh shape a frame painted, in paint order.
    fn painted_meshes(shapes: &[egui::epaint::ClippedShape]) -> Vec<egui::Mesh> {
        fn walk(shape: &egui::Shape, out: &mut Vec<egui::Mesh>) {
            match shape {
                egui::Shape::Mesh(mesh) => out.push(mesh.clone()),
                egui::Shape::Vec(inner) => {
                    for s in inner {
                        walk(s, out);
                    }
                }
                _ => {}
            }
        }
        let mut out = Vec::new();
        for clipped in shapes {
            walk(&clipped.shape, &mut out);
        }
        out
    }

    #[test]
    fn a_textured_material_reaches_the_cpu_and_gpu_paths() {
        let ctx = egui::Context::default();
        let (mut app, id) = app_ready_to_move();
        app.selection.clear();

        // Untextured: the painter draws shaded polygons, no textured meshes.
        let shapes = paint_once(&ctx, &mut app, frame(0.0, egui::pos2(500.0, 350.0), None));
        assert!(painted_meshes(&shapes).is_empty(), "nothing is textured yet");
        assert!(!painted_fills(&shapes).is_empty(), "the mesh still shades");

        // Assign the checker preset, and deselect so the tints stay out of it.
        app.selection.select_node(id);
        app.assign_material_to_selection(5);
        app.selection.deselect_node(id);

        // The GPU path names the library texture per face.
        let layers: Vec<Option<usize>> = app
            .node_materials()
            .faces(id)
            .expect("the node's faces")
            .iter()
            .map(|shading| shading.layer)
            .collect();
        assert!(!layers.is_empty());
        assert!(layers.iter().all(|layer| *layer == Some(0)), "{layers:?}");

        // The CPU path paints each textured face as a mesh over the library
        // texture, with the mesh's own UVs.
        let shapes = paint_once(&ctx, &mut app, frame(1.0, egui::pos2(500.0, 350.0), None));
        let meshes = painted_meshes(&shapes);
        assert!(!meshes.is_empty(), "no textured triangle was painted");
        let texture = app.cpu_texture(&ctx, 0).expect("the checker is in the library");
        assert!(meshes.iter().all(|m| m.texture_id == texture));
        assert!(meshes.iter().all(|m| m.vertices.len() == 3 && m.indices == [0, 1, 2]));
        assert!(
            meshes
                .iter()
                .any(|m| m.vertices.iter().any(|v| v.uv != egui::pos2(0.0, 0.0))),
            "the UVs never left the mesh"
        );
    }

    #[test]
    fn adding_textures_bumps_the_epoch_and_keeps_the_gpu_layers_uniform() {
        let (mut app, _id) = app_ready_to_move();
        assert_eq!(app.texture_epoch, 0, "the seeded library is the first revision");
        assert_eq!(app.materials.texture_count(), 1);

        let index = app.add_checker_texture();
        assert_eq!(index, 1, "it lands after the seeded checker");
        assert_eq!(app.texture_epoch, 1);

        // A small texture still reaches the GPU at the array's one size.
        app.materials.add_texture(Texture::checkerboard("tiny", 8, 2, [255, 0, 0, 255], [0, 0, 255, 255]));
        app.texture_epoch += 1;
        let layers = app.texture_layers();
        assert_eq!(layers.len(), 3);
        assert!(layers.iter().all(|layer| {
            layer.width == gpu_viewport::TEXTURE_LAYER_SIZE
                && layer.height == gpu_viewport::TEXTURE_LAYER_SIZE
                && layer.is_complete()
        }));
        // The resample kept the pattern: both of the tiny checker's colours are
        // in the resized layer.
        let tiny = &layers[2];
        let has = |rgb: [u8; 3]| {
            tiny.rgba
                .chunks_exact(4)
                .any(|t| t[0] == rgb[0] && t[1] == rgb[1] && t[2] == rgb[2])
        };
        assert!(has([255, 0, 0]) && has([0, 0, 255]), "the resize lost the checker");
    }

    #[test]
    fn a_failed_texture_load_is_reported_and_changes_nothing() {
        let (mut app, _id) = app_ready_to_move();
        assert!(app.load_texture(std::path::Path::new("/no/such/texture.png")).is_none());
        assert!(app.status.contains("Could not load"), "status: {}", app.status);
        assert_eq!(app.materials.texture_count(), 1);
        assert_eq!(app.texture_epoch, 0, "a failed load is not a new revision");
    }

    #[test]
    fn mapping_a_texture_onto_a_material_feeds_the_uv_editor() {
        let (mut app, id) = app_ready_to_move();
        app.current_material = 1;
        assert_eq!(app.uv_background_texture(), None, "clay has no map");

        let index = app.add_checker_texture();
        app.map_texture_to_current(index);
        assert_eq!(app.materials.resolve(1).diffuse_map, Some(index));
        assert!(app.status.contains("Mapped"), "status: {}", app.status);

        // The UV editor follows the selected mesh's material when there is one.
        app.selection.select_node(id);
        assert_eq!(app.uv_background_texture(), None, "the node is on the default slot");
        app.assign_material_to_selection(1);
        assert_eq!(app.uv_background_texture(), Some(index));
        // With nothing selected it falls back to the material the panel edits.
        app.selection.clear();
        assert_eq!(app.uv_background_texture(), Some(index));
    }

    /// Two quads sharing the position edge `{1, 2}` whose shared-edge UVs
    /// differ, so they are two separate UV islands: entries `0..4` and `4..8`.
    fn two_island_quads() -> Mesh {
        let mut m = Mesh::new();
        for p in [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0), // shared
            (1.0, 1.0, 0.0), // shared
            (0.0, 1.0, 0.0),
            (2.0, 0.0, 0.0),
            (2.0, 1.0, 0.0),
        ] {
            m.add_vertex(Vec3::new(p.0, p.1, p.2));
        }
        m.uvs = vec![
            Vec2::new(0.0, 0.0), // 0
            Vec2::new(1.0, 0.0), // 1
            Vec2::new(1.0, 1.0), // 2
            Vec2::new(0.0, 1.0), // 3
            Vec2::new(0.8, 0.0), // 4: shifted off face 0's (1, 0)
            Vec2::new(2.0, 0.0), // 5
            Vec2::new(2.0, 1.0), // 6
            Vec2::new(0.8, 1.0), // 7: shifted off face 0's (1, 1)
        ];
        m.add_face(&[0, 1, 2, 3]);
        m.add_face(&[1, 4, 5, 2]);
        for (i, c) in m.faces[0].corners.iter_mut().enumerate() {
            c.uv = i as u32;
        }
        for (i, c) in m.faces[1].corners.iter_mut().enumerate() {
            c.uv = 4 + i as u32;
        }
        m
    }

    #[test]
    fn the_island_button_grows_the_selection_to_the_whole_island() {
        let mut app = OpenForgeApp::with_gpu(None);
        let id = app.scene.create_mesh("Two Island Quads", two_island_quads());
        // One entry of the first island selected, as a canvas click leaves it.
        app.uv.select_only([2]);
        app.uv_select_island(id);
        // The whole first island, and nothing of the second.
        assert_eq!(app.uv.indices(), vec![0, 1, 2, 3]);
        assert!(app.status.contains("4 UVs in the selected island"), "status: {}", app.status);
        assert!(!app.undo.can_undo(), "a selection change is no undo entry");
    }

    #[test]
    fn a_textured_triangle_maps_uvs_to_the_texture_and_keeps_the_colour() {
        let color = egui::Color32::from_rgb(10, 20, 30);
        let id = egui::TextureId::Managed(7);
        let shape = textured_triangle(
            [egui::pos2(1.0, 2.0), egui::pos2(3.0, 4.0), egui::pos2(5.0, 6.0)],
            [
                Vec2::new(0.0, 0.0),
                Vec2::new(1.0, 0.5),
                Vec2::new(0.25, 1.0),
            ],
            color,
            id,
        );
        let egui::Shape::Mesh(mesh) = shape else {
            panic!("a textured triangle is a mesh shape");
        };
        assert_eq!(mesh.texture_id, id);
        assert_eq!(mesh.indices, vec![0, 1, 2]);
        assert_eq!(mesh.vertices[0].pos, egui::pos2(1.0, 2.0));
        assert_eq!(mesh.vertices[1].uv, egui::pos2(1.0, 0.5));
        assert_eq!(mesh.vertices[2].uv, egui::pos2(0.25, 1.0));
        assert!(mesh.vertices.iter().all(|v| v.color == color));
    }

    #[test]
    fn assigning_a_material_shades_the_mesh_with_it() {
        let ctx = egui::Context::default();
        let (mut app, id) = app_ready_to_move();
        let pos = egui::pos2(500.0, 350.0);
        // Unselected, so the painted colour is the material's own.
        app.selection.clear();
        let grey = painted_fills(&paint_once(&ctx, &mut app, frame(0.0, pos, None)));
        assert!(
            !grey.iter().any(|c| c.r() as i32 > c.b() as i32 + 30),
            "the default material is neutral"
        );

        app.selection.select_node(id);
        app.assign_material_to_selection(2);
        assert_eq!(app.scene.get(id).unwrap().face_material_slot(0), 2);
        app.selection.clear();
        let red = painted_fills(&paint_once(&ctx, &mut app, frame(1.0, pos, None)));
        assert!(
            red.iter().any(|c| c.r() as i32 > c.b() as i32 + 30),
            "\"Painted Red\" should read red"
        );

        // The assignment is one undo step.
        assert!(app.undo.can_undo());
        app.undo_edit();
        assert_eq!(app.scene.get(id).unwrap().face_material_slot(0), 0);
    }

    #[test]
    fn an_edge_selection_does_not_shade_a_face_as_selected() {
        // Element indices are per level. At the edge level the whole-node tint
        // is the strongest the mesh may take, however its edges are numbered;
        // at the face level the same index really does tint.
        let (mut app, id) = app_ready_to_move();
        let node_only = viewport::material_base(app.materials.resolve(0), true, false);
        let face_too = viewport::material_base(app.materials.resolve(0), true, true);
        assert_ne!(node_only, face_too, "the two tints must differ to be told apart");

        let expected = openforge_render::gpu::linear_from_srgb_u8([
            node_only[0],
            node_only[1],
            node_only[2],
        ]);
        app.selection.level = SelectLevel::Edge;
        app.selection.select_node(id);
        app.selection.select_elements(id, [0u32]);
        for shading in app.node_materials().faces(id).expect("the node's colours") {
            for (a, b) in shading.color.iter().zip(&expected) {
                assert!((a - b).abs() < 1e-6, "edge 0 shaded face 0: {shading:?}");
            }
        }

        app.selection.level = SelectLevel::Face;
        let face_tint = openforge_render::gpu::linear_from_srgb_u8([
            face_too[0],
            face_too[1],
            face_too[2],
        ]);
        assert_eq!(
            app.node_materials().faces(id).expect("the node's colours")[0].color,
            face_tint,
            "face 0 at the face level carries the face tint"
        );
    }

    #[test]
    fn node_colors_go_to_the_renderer_in_linear_space() {
        let (mut app, id) = app_ready_to_move();
        app.selection.select_node(id);
        app.assign_material_to_selection(2);
        let srgba = viewport::material_base(app.materials.resolve(2), true, false);
        let expected = openforge_render::gpu::linear_from_srgb_u8([srgba[0], srgba[1], srgba[2]]);
        let materials = app.node_materials();
        assert_eq!(materials.len(), 1, "one entry per mesh node");
        let got = materials.faces(id).expect("the node's colours");
        assert!(!got.is_empty(), "one colour per face");
        for shading in got {
            for (a, b) in shading.color.iter().zip(&expected) {
                assert!((a - b).abs() < 1e-6, "{shading:?} vs {expected:?}");
            }
        }
    }

    /// The GPU path's linear colour for a library slot, untinted.
    fn slot_linear(app: &OpenForgeApp, slot: u32) -> [f32; 4] {
        let srgb = viewport::material_base(app.materials.resolve(slot), false, false);
        openforge_render::gpu::linear_from_srgb_u8([srgb[0], srgb[1], srgb[2]])
    }

    #[test]
    fn per_face_materials_reach_the_gpu_path_per_face() {
        let (mut app, id) = app_ready_to_move();
        let face_count = app.scene.get(id).unwrap().mesh().unwrap().faces.len();
        assert!(face_count >= 2, "the seed mesh has several faces");
        // A whole-node assignment, then one face overridden with another slot.
        app.selection.select_node(id);
        app.assign_material_to_selection(1);
        app.selection.level = SelectLevel::Face;
        app.selection.select_elements(id, [1u32]);
        app.assign_material_to_faces(3);
        // Drop the selection so the colours below are the plain materials (a
        // selected node is tinted towards the blue highlight).
        app.selection.deselect_node(id);

        let clay = slot_linear(&app, 1);
        let steel = slot_linear(&app, 3);
        assert_ne!(clay, steel, "the two slots must be visibly different");

        let materials = app.node_materials();
        let colors: Vec<[f32; 4]> = materials
            .faces(id)
            .expect("the node's colours")
            .iter()
            .map(|shading| shading.color)
            .collect();
        assert_eq!(colors.len(), face_count);
        assert_eq!(colors[0], clay);
        assert_eq!(colors[1], steel);
        assert_eq!(colors[2], clay);
        // The override left a per-face slot on the node, not a whole-node one.
        assert_eq!(app.scene.get(id).unwrap().face_material_slot(0), 1);
        assert_eq!(app.scene.get(id).unwrap().face_material_slot(1), 3);

        // Undo puts the node back on the single slot.
        app.undo_edit();
        assert_eq!(app.scene.get(id).unwrap().face_material_slot(1), 1);
        assert_eq!(app.node_materials().faces(id).unwrap()[1].color, clay);
    }

    #[test]
    fn assigning_a_face_material_needs_faces_and_skips_no_ops() {
        let (mut app, id) = app_ready_to_move();
        app.selection.select_node(id);
        app.assign_material_to_selection(2);
        let undo_len = app.undo.len();

        // Object level: the element indices are vertices, not faces, so refuse.
        app.selection.level = SelectLevel::Object;
        app.selection.select_elements(id, [0u32]);
        app.assign_material_to_faces(4);
        assert_eq!(app.scene.get(id).unwrap().face_material_slot(0), 2);
        assert!(app.status.contains("Face level"), "{}", app.status);

        // Face level, but the faces already carry that material: no undo entry.
        app.selection.level = SelectLevel::Face;
        app.assign_material_to_faces(2);
        assert_eq!(app.undo.len(), undo_len, "a no-op must not push undo");
        assert_eq!(app.undo.undo_label(), Some("Assign Material"));
        assert!(app.status.contains("already use"), "{}", app.status);
    }

    #[test]
    fn a_material_that_is_in_use_cannot_be_removed() {
        let (mut app, id) = app_ready_to_move();
        app.selection.select_node(id);
        app.assign_material_to_selection(5);
        assert!(app.material_in_use(5));
        assert!(!app.material_in_use(3));

        app.current_material = 5;
        app.delete_current_material();
        assert_eq!(app.materials.material_count(), 6, "refused while assigned");

        // Unassign it and the last entry can go.
        app.assign_material_to_selection(0);
        app.current_material = 5;
        app.delete_current_material();
        assert_eq!(app.materials.material_count(), 5);
        assert_eq!(app.current_material, 0, "the selection falls back to the default");
    }

    #[test]
    fn a_material_in_the_middle_of_the_library_cannot_be_removed() {
        let (mut app, _id) = app_ready_to_move();
        app.current_material = 2;
        app.delete_current_material();
        assert_eq!(app.materials.material_count(), 6, "removing it would renumber slots");
        assert!(app.status.contains("last material"), "status: {}", app.status);
    }

    #[test]
    fn duplicating_a_node_leaves_an_independent_copy() {
        let ctx = egui::Context::default();
        let (mut app, id) = app_ready_to_move();
        let original = app.scene.get(id).unwrap().mesh().unwrap().vertex_count();

        step_full(&ctx, &mut app, frame(0.0, egui::pos2(500.0, 350.0), None));
        app.duplicate_selected();

        assert_eq!(app.scene.node_count(), 2);
        let copy = app.selection.selected_nodes().next().expect("the copy is selected");
        assert_ne!(copy, id, "the copy has its own id");
        assert!(app.scene.get(copy).unwrap().name.ends_with(" copy"));
        assert_eq!(app.scene.get(copy).unwrap().mesh().unwrap().vertex_count(), original);

        // Editing the copy must not touch the node it came from.
        app.scene.get_mut(copy).unwrap().mesh_mut().unwrap().positions[0] = Vec3::new(9.0, 9.0, 9.0);
        assert_ne!(app.scene.get(id).unwrap().mesh().unwrap().positions[0], Vec3::new(9.0, 9.0, 9.0));

        assert!(app.undo.can_undo(), "one entry for the duplicate");
        app.undo_edit();
        assert_eq!(app.scene.node_count(), 1);
    }

    #[test]
    fn the_handles_stay_a_constant_size_on_screen() {
        let ctx = egui::Context::default();
        let (mut app, _id) = app_ready_to_move();
        step(&ctx, &mut app, frame(0.0, egui::pos2(500.0, 350.0), None));

        let span = |app: &OpenForgeApp, axis: gizmo::Axis| {
            let layout = app.gizmo.layout.as_ref().unwrap();
            let tip = arm_tip(app, axis);
            (tip - layout.origin).length()
        };
        let near = span(&app, gizmo::Axis::X);

        // Pull the camera back and the handle keeps its pixel length.
        app.camera.distance *= 4.0;
        step(&ctx, &mut app, frame(1.0, egui::pos2(500.0, 350.0), None));
        let far = span(&app, gizmo::Axis::X);
        assert!(
            (near - far).abs() < 0.02 * near.max(1.0),
            "the arm keeps its screen length: {near} vs {far}"
        );
    }

    // --- the plugin host --------------------------------------------------------

    /// A unique temp path, so parallel tests do not collide.
    fn temp_path(extension: &str) -> PathBuf {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("openforge-app-{}-{n}.{extension}", std::process::id()))
    }

    /// A format the engine does not ship: a fixed triangle in, a marker out.
    struct DotFilter;

    impl FormatFilter for DotFilter {
        fn id(&self) -> &str {
            "dot"
        }

        fn label(&self) -> &str {
            "Triangle dot file"
        }

        fn extensions(&self) -> &[&str] {
            &["dot"]
        }

        fn import(&self, _path: &std::path::Path) -> IoResult<LoadedScene> {
            let mut mesh = Mesh::new();
            mesh.add_vertex(Vec3::ZERO);
            mesh.add_vertex(Vec3::X);
            mesh.add_vertex(Vec3::Y);
            mesh.add_face(&[0, 1, 2]);
            mesh.compute_flat_normals();
            Ok(LoadedScene {
                meshes: vec![MeshData { name: "Dot".into(), mesh, material: None }],
                materials: Vec::new(),
            })
        }

        fn export(&self, path: &std::path::Path, _data: &MeshData) -> IoResult<()> {
            std::fs::write(path, b"dot")?;
            Ok(())
        }
    }

    /// The plugin that registers [`DotFilter`].
    struct DotPlugin;

    impl Plugin for DotPlugin {
        fn name(&self) -> &str {
            "test.dot-format"
        }

        fn description(&self) -> &str {
            "imports .dot files"
        }

        fn register(&self, host: &mut PluginHost) -> Result<(), PluginError> {
            host.filters.insert(Box::new(DotFilter))
        }
    }

    #[test]
    fn the_plugins_panel_reports_what_the_host_loaded() {
        let app = OpenForgeApp::with_gpu(None);
        let report = app.plugin_report().join("\n");
        assert!(report.contains("openforge.core-tools"), "{report}");
        assert!(report.contains("openforge.interchange"), "{report}");
        assert!(report.contains("Services (3)"), "{report}");
        assert!(report.contains("Tools (4)"), "{report}");
        assert!(report.contains("Formats (4)"), "{report}");
        for tool in ["Select", "Move", "Scale", "Rotate"] {
            assert!(report.contains(tool), "`{tool}` is missing from:\n{report}");
        }
        for format in ["obj", "glb", "ply", "stl"] {
            assert!(report.contains(format), "`{format}` is missing from:\n{report}");
        }
    }

    #[test]
    fn the_side_panels_draw_without_a_display() {
        let ctx = egui::Context::default();
        let mut app = OpenForgeApp::with_gpu(None);
        let output = ctx.run(frame(0.0, egui::pos2(500.0, 350.0), None), |ctx| {
            app.handle_side_panels(ctx);
        });
        assert!(!output.shapes.is_empty(), "the panels painted something");
    }

    #[test]
    fn import_and_export_run_through_the_plugin_host() {
        let mut app = OpenForgeApp::with_gpu(None);
        let id = app.scene.iter().find(|n| n.is_mesh()).map(|n| n.id).expect("a seed mesh");
        app.selection.select_node(id);

        let path = temp_path("obj");
        app.export_selected(path.clone());
        assert!(app.status.contains("Exported"), "status: {}", app.status);
        assert!(path.exists(), "the host filter wrote the file");

        let before = app.scene.node_count();
        app.import_file(path.clone());
        assert!(app.status.contains("Imported"), "status: {}", app.status);
        assert_eq!(app.scene.node_count(), before + 1, "the file came back as a node");
        assert!(app.undo.can_undo(), "an import is undoable");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn exporting_with_nothing_selected_writes_nothing() {
        let mut app = OpenForgeApp::with_gpu(None);
        let path = temp_path("obj");
        app.export_selected(path.clone());
        assert!(app.status.contains("Select a mesh node"), "status: {}", app.status);
        assert!(!path.exists(), "an empty selection writes no file");
    }

    #[test]
    fn import_names_the_format_and_the_refusal_names_what_is_known() {
        let mut app = OpenForgeApp::with_gpu(None);
        app.scene = Scene::new();
        let obj = temp_path("obj");
        std::fs::write(&obj, "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n").expect("the temp obj writes");
        let name = obj.file_name().unwrap().to_string_lossy().into_owned();

        app.import_file(obj.clone());
        assert_eq!(app.scene.node_count(), 1, "the triangle landed");
        assert_eq!(
            app.status,
            format!("Imported 1 mesh(es) from {name} (Wavefront OBJ).")
        );
        app.undo_edit();
        assert_eq!(app.scene.node_count(), 0, "and it is one undo step");

        // A path no filter knows is turned down before anything is read, and
        // the refusal names every extension that would have worked.
        let unknown = temp_path("wtf");
        let unknown_name = unknown.file_name().unwrap().to_string_lossy().into_owned();
        app.import_file(unknown);
        assert_eq!(
            app.status,
            format!("Skipped {unknown_name} — no filter knows it (glb, gltf, obj, ply, stl).")
        );
        assert_eq!(app.scene.node_count(), 0, "and nothing came in");
        let _ = std::fs::remove_file(&obj);
    }

    #[test]
    fn export_refuses_ambiguity_and_names_the_format() {
        let mut app = OpenForgeApp::with_gpu(None);
        app.scene = Scene::new();
        let id = app.scene.create_mesh("Cube", primitive_box(1.0));
        let path = temp_path("obj");
        let name = path.file_name().unwrap().to_string_lossy().into_owned();

        app.export_selected(path.clone());
        assert_eq!(app.status, "Select a mesh node to export.");
        assert!(!path.exists(), "an empty selection writes no file");

        app.selection.select_node(id);
        app.export_selected(path.clone());
        assert!(path.exists(), "the host filter wrote the file");
        assert_eq!(
            app.status,
            format!("Exported Cube to {name} (Wavefront OBJ).")
        );

        // A second selected mesh makes the answer ambiguous: refuse and write
        // nothing rather than quietly picking one.
        let second = app.scene.create_mesh("Box", primitive_box(1.0));
        app.selection.select_node(second);
        let two = temp_path("obj");
        app.export_selected(two.clone());
        assert_eq!(
            app.status,
            "Select one mesh to export — 2 mesh nodes selected."
        );
        assert!(!two.exists(), "the ambiguous export writes nothing");

        // An extension no filter knows is refused before the write.
        app.selection.clear();
        app.selection.select_node(id);
        let unknown = temp_path("wtf");
        let unknown_name = unknown.file_name().unwrap().to_string_lossy().into_owned();
        app.export_selected(unknown.clone());
        assert_eq!(
            app.status,
            format!("Can't export to {unknown_name} — no filter knows it (glb, gltf, obj, ply, stl).")
        );
        assert!(!unknown.exists());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_drop_imports_through_the_same_registry() {
        let ctx = egui::Context::default();
        let mut app = OpenForgeApp::with_gpu(None);
        app.scene = Scene::new();
        let obj = temp_path("obj");
        std::fs::write(&obj, "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n").expect("the temp obj writes");
        let name = obj.file_name().unwrap().to_string_lossy().into_owned();

        let mut input = frame(0.0, egui::pos2(500.0, 350.0), None);
        input.dropped_files.push(egui::DroppedFile {
            path: Some(obj.clone()),
            ..Default::default()
        });
        let _ = ctx.run(input, |ctx| {
            app.handle_central_viewport(ctx, None);
            app.import_dropped_files(ctx);
        });
        assert_eq!(app.scene.node_count(), 1, "the drop imported it");
        assert_eq!(
            app.status,
            format!("Imported 1 mesh(es) from {name} (Wavefront OBJ)."),
            "the frame reports the import, not the viewport hint"
        );
        app.undo_edit();
        assert_eq!(app.scene.node_count(), 0, "the drop is undoable");

        // A pile, half of it unknown: the summary answers for the gesture.
        let mut input = frame(1.0, egui::pos2(500.0, 350.0), None);
        input.dropped_files.push(egui::DroppedFile {
            path: Some(obj.clone()),
            ..Default::default()
        });
        input.dropped_files.push(egui::DroppedFile {
            path: Some(temp_path("wtf")),
            ..Default::default()
        });
        let _ = ctx.run(input, |ctx| {
            app.handle_central_viewport(ctx, None);
            app.import_dropped_files(ctx);
        });
        assert_eq!(app.scene.node_count(), 1, "one of the two files landed");
        assert_eq!(app.status, "Dropped 2 file(s) — imported 1 mesh node(s).");
        let _ = std::fs::remove_file(&obj);
    }

    #[test]
    fn a_spin_clip_turns_the_selected_node_and_plays_it() {
        let mut app = OpenForgeApp::with_gpu(None);
        app.scene = Scene::new();
        let id = app.scene.create_mesh("Cube", primitive_box(1.0));
        app.selection.select_node(id);

        app.add_spin_clip();
        assert!(app.anim_open, "the timeline opens with the clip");
        assert_eq!(app.anim.current_clip_name(), Some("spin"));
        assert_eq!(app.status, "Created spin — 2.00 s over 1 node(s).");
        assert_eq!(
            app.scene.get(id).unwrap().transform.rotation,
            Quat::IDENTITY,
            "the clip leaves the scene where it stood"
        );

        // A scrub is a still frame of the clip: half way is half a turn.
        app.scrub_to(1.0);
        assert_eq!(app.status, "spin — 1.00 s of 2.00 s.");
        let half = app.scene.get(id).unwrap().transform.rotation;
        assert!(
            half.w.abs() < 1e-4 && (half.y.abs() - 1.0).abs() < 1e-4,
            "half a turn: {half:?}"
        );

        // The end lands back on the start (up to the quaternion double cover).
        app.scrub_to(2.0);
        let end = app.scene.get(id).unwrap().transform.rotation;
        assert!((end.w.abs() - 1.0) < 1e-4, "a full turn: {end:?}");

        // The transport turns it on the clock: a quarter second of a two
        // second turn is 45°.
        app.scrub_to(0.0);
        app.toggle_playback();
        assert_eq!(app.status, "Playing spin.");
        app.advance_animation(0.25);
        let turned = app.scene.get(id).unwrap().transform.rotation * Vec3::X;
        let want = Quat::from_rotation_y(std::f32::consts::TAU / 8.0) * Vec3::X;
        assert!((turned - want).length() < 1e-4, "{turned:?} vs {want:?}");

        // Running past the end pauses at the end.
        app.advance_animation(1.0);
        app.advance_animation(1.0);
        assert!(!app.anim.playing, "the transport pauses at the end");
        assert_eq!(app.anim.time, 2.0);

        app.stop_playback();
        assert_eq!(app.status, "Stopped.");
        assert_eq!(
            app.scene.get(id).unwrap().transform.rotation,
            Quat::IDENTITY,
            "a stop rewinds the scene to the clip's start"
        );
    }

    #[test]
    fn playback_is_not_an_undo_step_and_clips_do_not_collide() {
        let mut app = OpenForgeApp::with_gpu(None);
        app.scene = Scene::new();

        app.toggle_playback();
        assert_eq!(app.status, "No clip — Animate ▸ Spin Selected to make one.");
        app.add_spin_clip();
        assert_eq!(app.status, "Select a node to animate.");

        let id = app.scene.create_mesh("Cube", primitive_box(1.0));
        app.selection.select_node(id);
        app.add_spin_clip();
        assert_eq!(app.anim.current_clip_name(), Some("spin"));
        assert_eq!(
            app.clip_key_times(),
            vec![0.0, 0.5, 1.0, 1.5, 2.0],
            "the turn is carried by five keys"
        );
        app.add_spin_clip();
        assert_eq!(
            app.anim.current_clip_name(),
            Some("spin 2"),
            "a second clip gets its own name"
        );

        // Playback writes the pose and leaves the undo stack alone.
        app.anim.select_clip("spin");
        app.toggle_playback();
        app.advance_animation(0.5);
        assert_ne!(
            app.scene.get(id).unwrap().transform.rotation,
            Quat::IDENTITY,
            "the clip turned the node"
        );
        assert!(!app.undo.can_undo(), "playback is not an edit");
    }

    #[test]
    fn a_plugin_filter_extends_import_without_touching_the_app() {
        let mut app = OpenForgeApp::with_gpu(None);
        let before = app.scene.node_count();
        let path = temp_path("dot");

        // The engine ships no `.dot` filter, so this has to fail first — the
        // refusal names the extensions that would have worked.
        app.import_file(path.clone());
        assert!(
            app.status.contains("no filter knows it"),
            "status: {}",
            app.status
        );
        assert_eq!(app.scene.node_count(), before);

        // Registering one plugin is the whole change to the application.
        app.plugins.register(&DotPlugin).expect("the dot filter loads");
        app.import_file(path);
        assert!(app.status.contains("Imported"), "status: {}", app.status);
        assert_eq!(app.scene.node_count(), before + 1);
        assert!(app.plugins.filters.for_extension("dot").is_some());
    }

    #[test]
    fn a_plugin_cannot_steal_a_builtin_extension() {
        let mut app = OpenForgeApp::with_gpu(None);
        // An id nobody uses, claiming an extension `obj` already serves.
        struct Hijacker;
        impl FormatFilter for Hijacker {
            fn id(&self) -> &str {
                "obj2"
            }
            fn label(&self) -> &str {
                "Also OBJ"
            }
            fn extensions(&self) -> &[&str] {
                &["obj"]
            }
            fn import(&self, _path: &std::path::Path) -> IoResult<LoadedScene> {
                Err(openforge_io::IoError::Unsupported("no"))
            }
            fn export(&self, _path: &std::path::Path, _data: &MeshData) -> IoResult<()> {
                Err(openforge_io::IoError::Unsupported("no"))
            }
        }
        struct HijackPlugin;
        impl Plugin for HijackPlugin {
            fn name(&self) -> &str {
                "test.hijack"
            }
            fn register(&self, host: &mut PluginHost) -> Result<(), PluginError> {
                host.filters.insert(Box::new(Hijacker))
            }
        }

        let err = app.plugins.register(&HijackPlugin).unwrap_err();
        assert!(matches!(err, PluginError::Duplicate { kind: "extension", .. }), "{err:?}");
        assert_eq!(app.plugins.filters.len(), 4, "the rejected plugin left nothing behind");
        assert_eq!(app.plugins.filters.for_extension("obj").unwrap().id(), "obj");
        assert_eq!(app.plugins.plugins().len(), 2, "not reported as loaded");
    }

    #[test]
    fn the_active_tool_comes_from_the_plugin_host() {
        let mut app = OpenForgeApp::with_gpu(None);
        for (mode, name) in [
            (ActiveTool::Select, "Select"),
            (ActiveTool::Move, "Move"),
            (ActiveTool::Scale, "Scale"),
            (ActiveTool::Rotate, "Rotate"),
        ] {
            app.tool = mode;
            assert_eq!(app.active_tool().name(), name, "the host built {name}");
        }
    }
}
