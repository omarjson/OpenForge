//! Interactive tools: selection and transform (move/scale/rotate).
//!
//! Tools operate on the scene through a narrow [`ToolContext`] so they stay
//! decoupled from the GUI. This mirrors the common editor pattern of a
//! "current tool" that receives pointer/drag events.

use crate::math::Ray;
use crate::mesh::Mesh;
use crate::scene::Scene;
use crate::selection::{SelectLevel, Selection};
use crate::NodeId;
use glam::Vec3;

/// A 3D manipulation mode.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum TransformMode {
    #[default]
    Move,
    Scale,
    Rotate,
}

/// A drag interaction with the mouse in the viewport.
#[derive(Clone, Copy, Debug)]
pub struct Pointer {
    /// World-space ray under the cursor.
    pub ray: Ray,
    /// Normalized viewport position (0..1) for UI affordances.
    pub ndc: glam::Vec2,
}

/// The result of applying a tool operation to the scene.
pub struct ToolOutcome {
    /// A human-readable label for the undo stack.
    pub label: String,
}

/// An interactive tool.
pub trait Tool {
    fn name(&self) -> &'static str;
    /// Called when the primary pointer goes down.
    fn on_pointer_down(&mut self, ctx: &mut ToolContext<'_>, p: &Pointer);
    /// Called when the pointer moves (with or without a button held).
    fn on_pointer_move(&mut self, ctx: &mut ToolContext<'_>, p: &Pointer);
    /// Called when the primary pointer is released.
    fn on_pointer_up(&mut self, ctx: &mut ToolContext<'_>, p: &Pointer) -> Option<ToolOutcome>;
}

/// Borrowed bundle of scene state handed to tools.
pub struct ToolContext<'a> {
    pub scene: &'a mut Scene,
    pub selection: &'a mut Selection,
    /// Transform applied to the current selection, in world space.
    pub drag: Option<DragState>,
}

/// Tracks an in-progress transform drag.
#[derive(Clone, Debug)]
pub struct DragState {
    pub mode: TransformMode,
    /// World-space origin captured at drag start.
    pub start_point: Vec3,
    /// Per-node transform captured at drag start.
    pub start_transforms: Vec<(NodeId, crate::math::Transform)>,
    /// World-space axis for the active drag (for gizmo highlight).
    pub axis: Vec3,
}

impl ToolContext<'_> {
    /// Current world-space bounds of the selection (mesh nodes only).
    pub fn selection_bounds(&mut self) -> crate::math::Aabb {
        let ids: Vec<NodeId> = self.selection.selected_nodes().collect();
        self.scene.world_bounds(&ids)
    }

    /// Snapshot the selected nodes' local transforms for a drag.
    pub fn snapshot_transforms(&self) -> Vec<(NodeId, crate::math::Transform)> {
        self.selection
            .selected_nodes()
            .filter_map(|id| self.scene.get(id).ok().map(|n| (id, n.transform)))
            .collect()
    }
}

/// The built-in select tool: click to select objects.
#[derive(Default)]
pub struct SelectTool {
    /// If true, a click toggles instead of replacing the selection.
    pub additive: bool,
}

impl Tool for SelectTool {
    fn name(&self) -> &'static str {
        "Select"
    }

    fn on_pointer_down(&mut self, ctx: &mut ToolContext<'_>, p: &Pointer) {
        if let Some(hit) = pick_node(ctx.scene, p.ray) {
            if self.additive {
                ctx.selection.toggle_node(hit);
            } else {
                ctx.selection.clear();
                ctx.selection.select_node(hit);
            }
        } else if !self.additive {
            ctx.selection.clear();
        }
    }

    fn on_pointer_move(&mut self, _ctx: &mut ToolContext<'_>, _p: &Pointer) {}

    fn on_pointer_up(&mut self, _ctx: &mut ToolContext<'_>, _p: &Pointer) -> Option<ToolOutcome> {
        None
    }
}

/// The built-in transform tool.
pub struct TransformTool {
    pub mode: TransformMode,
    dragging: bool,
}

impl TransformTool {
    pub fn new(mode: TransformMode) -> Self {
        Self { mode, dragging: false }
    }
}

impl Tool for TransformTool {
    fn name(&self) -> &'static str {
        match self.mode {
            TransformMode::Move => "Move",
            TransformMode::Scale => "Scale",
            TransformMode::Rotate => "Rotate",
        }
    }

    fn on_pointer_down(&mut self, ctx: &mut ToolContext<'_>, p: &Pointer) {
        let bounds = ctx.selection_bounds();
        if bounds.is_empty() {
            return;
        }
        let start = closest_point_on_bounds(p.ray, bounds);
        let axis = dominant_axis(p.ray.direction);
        ctx.drag = Some(DragState {
            mode: self.mode,
            start_point: start,
            start_transforms: ctx.snapshot_transforms(),
            axis,
        });
        self.dragging = true;
    }

    fn on_pointer_move(&mut self, ctx: &mut ToolContext<'_>, p: &Pointer) {
        let Some(drag) = ctx.drag.clone() else { return };
        let delta = projected_delta(&drag, p.ray);
        // Reset each node to its drag-start transform, then apply the delta.
        for (id, start) in &drag.start_transforms {
            let Ok(node) = ctx.scene.get_mut(*id) else { continue };
            node.transform = *start;
            match drag.mode {
                TransformMode::Move => {
                    node.transform.translation += delta;
                }
                TransformMode::Scale => {
                    let s = 1.0 + delta.dot(drag.axis);
                    node.transform.scale *= s.max(0.01);
                }
                TransformMode::Rotate => {
                    let angle = delta.dot(perpendicular(drag.axis));
                    node.transform.rotation =
                        glam::Quat::from_axis_angle(drag.axis.normalize_or_zero(), angle)
                            * start.rotation;
                }
            }
        }
    }

    fn on_pointer_up(&mut self, ctx: &mut ToolContext<'_>, _p: &Pointer) -> Option<ToolOutcome> {
        self.dragging = false;
        let had_drag = ctx.drag.take().is_some();
        if had_drag {
            Some(ToolOutcome { label: self.name().to_string() })
        } else {
            None
        }
    }
}

/// Cast a ray and return the nearest visible, unfrozen mesh node.
pub fn pick_node(scene: &mut Scene, ray: Ray) -> Option<NodeId> {
    // Gather candidate meshes first to avoid conflicting borrows.
    let candidates: Vec<(NodeId, Mesh)> = scene
        .iter()
        .filter(|n| n.is_mesh() && n.visible && !n.frozen)
        .filter_map(|n| n.mesh().map(|m| (n.id, m.clone())))
        .collect();

    let mut best: Option<(f32, NodeId)> = None;
    for (id, mesh) in candidates {
        let Ok(world) = scene.world_matrix(id) else { continue };
        let inv = world.inverse();
        let local_origin = inv.transform_point3(ray.origin);
        let local_dir = (inv * ray.direction.extend(0.0)).truncate().normalize_or_zero();
        let local_ray = Ray { origin: local_origin, direction: local_dir };
        for tri in mesh.triangulate() {
            let a = mesh.positions[tri[0] as usize];
            let b = mesh.positions[tri[1] as usize];
            let c = mesh.positions[tri[2] as usize];
            if let Some(t) = local_ray.intersect_triangle(a, b, c) {
                let world_hit = world.transform_point3(local_ray.at(t));
                let dist = world_hit.distance(ray.origin);
                if best.map(|(d, _)| dist < d).unwrap_or(true) {
                    best = Some((dist, id));
                }
            }
        }
    }
    best.map(|(_, id)| id)
}

fn closest_point_on_bounds(ray: Ray, b: crate::math::Aabb) -> Vec3 {
    // Cheap approximation: project the box center onto the ray.
    let t = (b.center() - ray.origin).dot(ray.direction);
    ray.at(t.max(0.0))
}

fn dominant_axis(v: Vec3) -> Vec3 {
    let a = v.abs();
    if a.x >= a.y && a.x >= a.z {
        Vec3::X
    } else if a.y >= a.z {
        Vec3::Y
    } else {
        Vec3::Z
    }
}

fn perpendicular(v: Vec3) -> Vec3 {
    let n = v.normalize_or_zero();
    let other = if n.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
    n.cross(other).normalize_or_zero()
}

fn projected_delta(drag: &DragState, ray: Ray) -> Vec3 {
    // Intersect the ray with a plane through the start point facing the camera-ish.
    let n = drag.axis.cross(Vec3::Y).cross(drag.axis).normalize_or_zero();
    let denom = ray.direction.dot(n);
    if denom.abs() < 1e-6 {
        return Vec3::ZERO;
    }
    let t = (drag.start_point - ray.origin).dot(n) / denom;
    let hit = ray.at(t);
    hit - drag.start_point
}

/// Switch the current selection to a given level (thin convenience wrapper).
pub fn set_level(selection: &mut Selection, level: SelectLevel) {
    selection.level = level;
    selection.clear();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::Mesh;

    #[test]
    fn pick_hits_nearest_mesh() {
        let mut scene = Scene::new();
        let mut cube = Mesh::new();
        // A unit quad in the XY plane.
        cube.add_vertex(Vec3::new(-1.0, -1.0, 0.0));
        cube.add_vertex(Vec3::new(1.0, -1.0, 0.0));
        cube.add_vertex(Vec3::new(1.0, 1.0, 0.0));
        cube.add_vertex(Vec3::new(-1.0, 1.0, 0.0));
        cube.add_face(&[0, 1, 2, 3]);
        let id = scene.create_mesh("quad", cube);
        let ray = Ray { origin: Vec3::new(0.0, 0.0, 5.0), direction: Vec3::new(0.0, 0.0, -1.0) };
        assert_eq!(pick_node(&mut scene, ray), Some(id));
    }

    #[test]
    fn pick_misses_outside() {
        let mut scene = Scene::new();
        let mut cube = Mesh::new();
        cube.add_vertex(Vec3::new(-1.0, -1.0, 0.0));
        cube.add_vertex(Vec3::new(1.0, -1.0, 0.0));
        cube.add_vertex(Vec3::new(1.0, 1.0, 0.0));
        cube.add_face(&[0, 1, 2]);
        scene.create_mesh("tri", cube);
        let ray = Ray { origin: Vec3::new(50.0, 50.0, 5.0), direction: Vec3::new(0.0, 0.0, -1.0) };
        assert_eq!(pick_node(&mut scene, ray), None);
    }
}
