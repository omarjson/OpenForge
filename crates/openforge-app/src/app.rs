//! The OpenForge application: UI layout, viewport interaction, and the glue
//! between egui and the openforge-core scene model.

use crate::gpu_viewport::{self, GpuViewport};
use crate::uv_editor::{self, UvEditor};
use crate::viewport::{self, Camera};
use eframe::egui;
use eframe::egui_wgpu::RenderState;
use glam::{Vec2, Vec3};
use openforge_core::mesh::{self, Mesh};
use openforge_core::node::NodeKind;
use openforge_core::scene::Scene;
use openforge_core::selection::{SelectLevel, Selection};
use openforge_core::tool::{Pointer, SelectTool, Tool, ToolContext, TransformMode, TransformTool};
use openforge_core::undo::UndoStack;
use openforge_core::NodeId;
use openforge_ops::MeshOps;
use openforge_render::FrameStats;
use openforge_rig::{skin_positions, Bone, LocalTransform, Quat, Skeleton, SkinWeights};
use openforge_uv::UvOps;
use std::path::PathBuf;

/// Which tool is active.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ActiveTool {
    Select,
    Move,
    Scale,
    Rotate,
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
    /// Whether the viewport was interacted with this frame (drives repaint).
    dirty: bool,
    show_outliner: bool,
    show_properties: bool,
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
}

impl OpenForgeApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        // Dark theme by default; egui respects the native theme otherwise.
        cc.egui_ctx.set_visuals(egui::Visuals::dark());
        let mut scene = Scene::new();
        // Seed with a cube so the viewport isn't empty on first run.
        let cube = mesh::uv_sphere(1.0, 24, 16);
        scene.create_mesh("Sphere", cube);
        // eframe hands us a live wgpu device when it runs the wgpu backend;
        // otherwise `gpu` stays `None` and the CPU painter takes over.
        let gpu = cc.wgpu_render_state.as_ref().map(GpuViewport::new);
        let backend = if gpu.is_some() { "wgpu" } else { "CPU" };
        Self {
            scene,
            selection: Selection::new(),
            undo: UndoStack::new(100),
            camera: Camera::default(),
            tool: ActiveTool::Select,
            edit_before: None,
            status: format!(
                "{backend} viewport. Left-drag selects; middle-drag orbits; scroll zooms. Face level: X extrude, I inset, O poke, S subdivide, Del delete."
            ),
            current_file: None,
            tris: Vec::new(),
            gpu,
            uv: UvEditor::default(),
            dirty: true,
            show_outliner: true,
            show_properties: true,
            deform_axis: 1,
            deform_amount: 0.5,
            rig_skeleton: None,
            rig_weights: None,
            rig_bind: Vec::new(),
            rig_node: None,
            rig_bone: 0,
            rig_angle: 0.0,
            rig_bones: 4,
        }
    }

    fn active_tool(&mut self) -> Box<dyn Tool> {
        match self.tool {
            ActiveTool::Select => Box::new(SelectTool { additive: false }),
            ActiveTool::Move => Box::new(TransformTool::new(TransformMode::Move)),
            ActiveTool::Scale => Box::new(TransformTool::new(TransformMode::Scale)),
            ActiveTool::Rotate => Box::new(TransformTool::new(TransformMode::Rotate)),
        }
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
    fn selected_faces(&self, id: NodeId) -> Vec<usize> {
        self.selection
            .selected_elements(id)
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

    fn import_file(&mut self, path: PathBuf) {
        match openforge_io::import_path(&path) {
            Ok(loaded) => {
                let count = loaded.mesh_count();
                self.with_edit("Import", |app| {
                    for md in loaded.meshes {
                        app.scene.create_mesh(md.name, md.mesh);
                    }
                });
                self.status = format!("Imported {} mesh(es) from {}.", count, path.display());
            }
            Err(e) => self.status = format!("Import failed: {}", e),
        }
    }

    fn export_selected(&mut self, path: PathBuf) {
        let Some(id) = self.selection.selected_nodes().next() else {
            self.status = "Select a mesh node to export.".into();
            return;
        };
        let Some(mesh) = self.scene.get(id).ok().and_then(|n| n.mesh()).cloned() else {
            self.status = "Selected node has no mesh.".into();
            return;
        };
        let name = self.scene.get(id).map(|n| n.name.clone()).unwrap_or_else(|_| "mesh".into());
        let data = openforge_io::MeshData { name, mesh, material: None };
        match openforge_io::export_path(&path, &data) {
            Ok(()) => self.status = format!("Exported to {}.", path.display()),
            Err(e) => self.status = format!("Export failed: {}", e),
        }
    }

    // --- Viewport interaction ----------------------------------------------------

    fn handle_viewport_input(&mut self, ui: &mut egui::Ui, rect: egui::Rect) {
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

        // Build the active tool and a context borrowing the scene + selection.
        let mut tool = self.active_tool();
        let level = self.selection.level;
        let mut tctx = ToolContext {
            scene: &mut self.scene,
            selection: &mut self.selection,
            drag: None,
        };

        // Primary (left) button drives the active tool.
        let primary = response.dragged_by(egui::PointerButton::Primary)
            || response.drag_started_by(egui::PointerButton::Primary);
        let mut pending_undo: Option<(String, Scene)> = None;
        // For sub-object picking we stash the click ray and resolve it after the
        // tool context's borrows on the scene/selection are released.
        let mut click_pick: Option<openforge_core::math::Ray> = None;
        if let Some(pos) = response.interact_pointer_pos() {
            let ndc_x = ((pos.x - rect.left()) / rect.width()) * 2.0 - 1.0;
            let ndc_y = 1.0 - ((pos.y - rect.top()) / rect.height()) * 2.0;
            let ray = self.camera.ray_from_ndc(ndc_x, ndc_y, aspect);
            let pointer = Pointer { ray, ndc: glam::Vec2::new(ndc_x, ndc_y) };

            if response.drag_started_by(egui::PointerButton::Primary) {
                if self.edit_before.is_none() {
                    self.edit_before = Some(tctx.scene.clone());
                }
                tool.on_pointer_down(&mut tctx, &pointer);
                self.dirty = true;
            } else if response.dragged_by(egui::PointerButton::Primary) {
                tool.on_pointer_move(&mut tctx, &pointer);
                self.dirty = true;
            } else if response.drag_stopped() && primary {
                if let Some(outcome) = tool.on_pointer_up(&mut tctx, &pointer) {
                    let before = self.edit_before.take();
                    pending_undo = before.map(|b| (outcome.label, b));
                    self.dirty = true;
                }
            } else if response.clicked() {
                if level == SelectLevel::Object {
                    // Object level: delegate to the select tool.
                    if matches!(self.tool, ActiveTool::Select) {
                        tool.on_pointer_down(&mut tctx, &pointer);
                        tool.on_pointer_up(&mut tctx, &pointer);
                        self.dirty = true;
                    }
                } else {
                    // Face/Vertex level: stash the ray and pick below.
                    click_pick = Some(ray);
                }
            }
        }
        drop(tctx);

        // Resolve a sub-object pick now that the borrows are released.
        if let Some(ray) = click_pick {
            match level {
                SelectLevel::Face => {
                    if let Some(hit) = viewport::pick_face(&mut self.scene, ray) {
                        self.selection.clear();
                        self.selection.select_node(hit.node);
                        self.selection.select_elements(hit.node, [hit.face]);
                        self.dirty = true;
                    }
                }
                SelectLevel::Vertex => {
                    let tol = (self.camera.distance * 0.02).max(0.01);
                    if let Some(hit) =
                        viewport::pick_vertex(&mut self.scene, ray, self.camera.eye(), tol)
                    {
                        self.selection.clear();
                        self.selection.select_node(hit.node);
                        self.selection.select_elements(hit.node, [hit.vertex]);
                        self.dirty = true;
                    }
                }
                _ => {}
            }
        }

        // Record the undo entry now that the scene borrow is released.
        if let Some((label, before)) = pending_undo {
            self.undo.push(label, before, self.scene.clone());
        }

        self.status = match self.tool {
            ActiveTool::Select => {
                format!("Select — {} object(s) selected.", self.selection.node_count())
            }
            ActiveTool::Move => "Move — drag to translate selection.".into(),
            ActiveTool::Scale => "Scale — drag to scale selection.".into(),
            ActiveTool::Rotate => "Rotate — drag to rotate selection.".into(),
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
            let base = if t.face_selected {
                // Selected sub-object face: warm highlight.
                [255u8, 170, 60, 255]
            } else if t.selected {
                [80u8, 170, 255, 255]
            } else {
                [170u8, 170, 178, 255]
            };
            // Phong shade using the shared materials crate (diffuse + specular).
            let centroid = (t.v[0].pos + t.v[1].pos + t.v[2].pos) / 3.0;
            let view_dir = (cam - centroid).normalize_or_zero();
            let [r, g, b, _] =
                openforge_materials::shade_phong(t.v[0].normal, light, view_dir, 24.0, base);
            let color = egui::Color32::from_rgb(r, g, b);
            let shape = egui::Shape::convex_polygon(pts, color, egui::Stroke::new(0.5_f32, color.gamma_multiply(0.6)));
            painter.add(shape);
        }

        // Overlay selected vertices as small dots (vertex level editing aid).
        self.paint_vertex_dots(&painter, rect);

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
        self.handle_menu(ctx);
        self.handle_shortcuts(ctx);
        self.handle_top_toolbar(ctx);
        // Pack the edges first (status bar lowest, then the UV editor), then
        // the side panels, and only then the central viewport, which takes what
        // is left. That order is what keeps the viewport's rect — and its mouse
        // input — clear of the bottom panels.
        self.handle_status_bar(ctx);
        self.handle_uv_editor(ctx);
        self.handle_side_panels(ctx);
        self.handle_central_viewport(ctx, frame.wgpu_render_state());
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
                    if ui.button("Import…").clicked() {
                        if let Some(path) = rfd_pick_file() {
                            self.import_file(path);
                        }
                        ui.close_menu();
                    }
                    if ui.button("Export selected…").clicked() {
                        if let Some(path) = rfd_save_file() {
                            self.export_selected(path);
                        }
                        ui.close_menu();
                    }
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
                ui.menu_button("View", |ui| {
                    if ui.button("Frame All (F)").clicked() {
                        self.frame_all();
                        ui.close_menu();
                    }
                    if ui.button("UV Editor").clicked() {
                        self.uv.open = !self.uv.open;
                        ui.close_menu();
                    }
                    if ui.button("Reset Camera").clicked() {
                        self.camera = Camera::default();
                        self.dirty = true;
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
                ui.selectable_value(&mut self.selection.level, SelectLevel::Object, "Object");
                ui.selectable_value(&mut self.selection.level, SelectLevel::Vertex, "Vertex");
                ui.selectable_value(&mut self.selection.level, SelectLevel::Face, "Face");
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
            });
            // Second row: sub-object modeling operations.
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
            // Third row: deformers and boolean operations.
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
                    ui.heading("Properties");
                    ui.separator();
                    self.properties_panel(ui);
                });
        }
    }

    fn outliner_node(&mut self, ui: &mut egui::Ui, id: NodeId) {
        // Snapshot the node's display data so we don't hold a borrow on the
        // scene while recursively painting children.
        let (name, icon, has_children, children) = match self.scene.get(id) {
            Ok(node) => {
                let icon = match &node.kind {
                    NodeKind::Group => "[] ",
                    NodeKind::Dummy => "() ",
                    NodeKind::Mesh { .. } => "/\\ ",
                };
                (node.name.clone(), icon, !node.children.is_empty(), node.children.clone())
            }
            Err(_) => return,
        };
        let selected = self.selection.is_node_selected(id);

        ui.push_id(id.0, |ui| {
            let resp = ui.selectable_label(selected, format!("{}{}", icon, name));
            if resp.clicked() {
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

    fn properties_panel(&mut self, ui: &mut egui::Ui) {
        let ids: Vec<NodeId> = self.selection.selected_nodes().collect();
        if ids.is_empty() {
            ui.label("Nothing selected.");
            ui.separator();
            ui.label(format!("Nodes: {}", self.scene.node_count()));
            let meshes = self.scene.mesh_names().len();
            ui.label(format!("Mesh nodes: {}", meshes));
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
            }
            ui.separator();
        }

        // Edit the transform of a single selected node.
        if ids.len() == 1 {
            let id = ids[0];
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
                self.handle_viewport_input(ui, rect);

                // Render offscreen with wgpu when a device is available, and
                // fall back to the CPU painter when it is not. Both paths show
                // the same scene through the same camera state, so the only
                // difference is shading quality and depth correctness.
                let rendered = match (render_state, self.gpu.as_mut()) {
                    (Some(state), Some(gpu)) => {
                        let scale = ui.ctx().pixels_per_point();
                        let size = [rect.width() * scale, rect.height() * scale];
                        let camera = gpu_viewport::orbit_camera(&self.camera);
                        let time = ui.input(|i| i.time) as f32;
                        Some(gpu.render(state, size, &mut self.scene, &self.selection, &camera, time))
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
    /// the vertex-level dots and the stats label.
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
                    (dot + 1.0, egui::Color32::from_rgb(255, 170, 60))
                } else {
                    (dot, egui::Color32::from_rgb(120, 200, 255))
                };
                painter.circle_filled(center, rad, color);
            }
        }
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
            if ui
                .add_enabled(has_uvs, egui::Button::new("Pack"))
                .on_hover_text("Repack UV islands into a grid atlas")
                .clicked()
            {
                let grid = self.uv.grid;
                self.uv_apply(id, "UV Pack", move |m| m.uv_repack(grid));
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
        if let Ok(node) = self.scene.get(id) {
            if let Some(mesh) = node.mesh() {
                uv_editor::draw_canvas(&painter, rect, mesh, &self.uv, hovered);
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

fn rfd_pick_file() -> Option<PathBuf> {
    rfd::FileDialog::new()
        .add_filter("3D models", &["obj", "glb", "gltf", "ply", "stl"])
        .pick_file()
}

fn rfd_save_file() -> Option<PathBuf> {
    rfd::FileDialog::new()
        .add_filter("3D models", &["obj", "glb", "ply", "stl"])
        .set_file_name("export.obj")
        .save_file()
}
