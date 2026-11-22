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

/// Step sizes for snapping a *free-form* (non-gizmo) transform drag.
///
/// The gizmo snaps through `openforge-app`'s grid/geometry machinery; a plain
/// view-plane drag never reaches that machinery, so the app hands these steps
/// to the tool with the pointer. `None` means the drag stays free.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct FreeSnap {
    /// World units between translation values.
    pub grid: f32,
    /// Radians between rotation values.
    pub angle: f32,
    /// Multiples between scale factors.
    pub factor: f32,
}

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
    /// Step sizes when this drag should snap, `None` for a free drag.
    pub snap: Option<FreeSnap>,
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
        // A free drag never sees the gizmo's snap machinery, so the pointer
        // carries the steps here. Snapping lands the *destination* on a step,
        // not the distance travelled, matching the gizmo's grid mode.
        let snap = p.snap.filter(|s| {
            s.grid.is_finite() && s.angle.is_finite() && s.factor.is_finite()
        });
        for (id, start) in &drag.start_transforms {
            let Ok(node) = ctx.scene.get_mut(*id) else { continue };
            node.transform = *start;
            match drag.mode {
                TransformMode::Move => {
                    let mut target = start.translation + delta;
                    if let Some(s) = snap {
                        target = snap_vec(target, s.grid);
                    }
                    node.transform.translation = target;
                }
                TransformMode::Scale => {
                    let mut factor = 1.0 + delta.dot(drag.axis);
                    if let Some(s) = snap {
                        factor = snap_scalar(factor, s.factor).max(s.factor.max(1e-3));
                    }
                    node.transform.scale *= factor.max(0.01);
                }
                TransformMode::Rotate => {
                    let mut angle = delta.dot(perpendicular(drag.axis));
                    if let Some(s) = snap {
                        angle = snap_scalar(angle, s.angle);
                    }
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

/// The multiple of `step` nearest `value`, or `value` for a step that cannot
/// snap. Shared by the free-drag transform modes so all three agree on what a
/// "step" means.
fn snap_scalar(value: f32, step: f32) -> f32 {
    if !(step > 0.0) || !value.is_finite() {
        return value;
    }
    (value / step).round() * step
}

/// [`snap_scalar`] on each component of a world position.
fn snap_vec(value: Vec3, step: f32) -> Vec3 {
    Vec3::new(
        snap_scalar(value.x, step),
        snap_scalar(value.y, step),
        snap_scalar(value.z, step),
    )
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

    // --- free-drag snapping ----------------------------------------------------

    fn one_node(at: Vec3) -> (Scene, crate::NodeId) {
        let mut scene = Scene::new();
        let mut quad = Mesh::new();
        quad.add_vertex(Vec3::ZERO);
        quad.add_vertex(Vec3::X);
        quad.add_vertex(Vec3::X + Vec3::Y);
        quad.add_face(&[0, 1, 2]);
        let id = scene.create_mesh("node", quad);
        scene.get_mut(id).unwrap().transform.translation = at;
        (scene, id)
    }

    fn pointer(origin: Vec3, snap: Option<FreeSnap>) -> Pointer {
        Pointer {
            ray: Ray { origin, direction: Vec3::NEG_Y },
            ndc: glam::Vec2::ZERO,
            snap,
        }
    }

    #[test]
    fn a_free_move_can_land_its_destination_on_the_grid() {
        let start = Vec3::new(0.1, 0.2, 0.3);
        let step = FreeSnap { grid: 0.25, angle: 0.1, factor: 0.1 };
        let (mut free, id) = one_node(start);
        let (mut snapped, _) = one_node(start);
        let drag = DragState {
            mode: TransformMode::Move,
            start_point: start,
            start_transforms: vec![(id, free.get(id).unwrap().transform)],
            axis: Vec3::X,
        };
        // The ray meets the drag plane at the origin, so the unsnapped
        // destination is (0, 0.2, 0): only y is off a quarter-unit step.
        let at = pointer(Vec3::new(0.0, 5.0, 0.0), None);
        let mut tool = TransformTool::new(TransformMode::Move);
        let mut ctx = ToolContext {
            scene: &mut free,
            selection: &mut Selection::new(),
            drag: Some(drag.clone()),
        };
        tool.on_pointer_move(&mut ctx, &at);
        assert!((free.get(id).unwrap().transform.translation.y - 0.2).abs() < 1e-6);

        let at = pointer(Vec3::new(0.0, 5.0, 0.0), Some(step));
        let mut ctx = ToolContext {
            scene: &mut snapped,
            selection: &mut Selection::new(),
            drag: Some(drag),
        };
        tool.on_pointer_move(&mut ctx, &at);
        assert!(
            (snapped.get(id).unwrap().transform.translation.y - 0.25).abs() < 1e-6,
            "the destination lands on the next grid step: {:?}",
            snapped.get(id).unwrap().transform.translation
        );
    }

    #[test]
    fn a_free_scale_snaps_its_factor_and_never_collapses_it() {
        let step = FreeSnap { grid: 0.25, angle: 0.1, factor: 0.1 };
        let (mut scene, id) = one_node(Vec3::ZERO);
        let drag = DragState {
            mode: TransformMode::Scale,
            start_point: Vec3::ZERO,
            start_transforms: vec![(id, scene.get(id).unwrap().transform)],
            axis: Vec3::X,
        };
        // delta.x = 0.37, so the factor is 1.37 and snaps to 1.4.
        let at = pointer(Vec3::new(0.37, 5.0, 0.0), Some(step));
        let mut tool = TransformTool::new(TransformMode::Scale);
        let mut ctx = ToolContext {
            scene: &mut scene,
            selection: &mut Selection::new(),
            drag: Some(drag),
        };
        tool.on_pointer_move(&mut ctx, &at);
        let scale = scene.get(id).unwrap().transform.scale;
        assert!((scale.x - 1.4).abs() < 1e-5, "factor {scale}");
        // A free scale is uniform: the axis decides the measure, the factor
        // scales the whole node (the existing TransformTool contract).
        assert!((scale.y - 1.4).abs() < 1e-5 && (scale.z - 1.4).abs() < 1e-5);
    }

    #[test]
    fn a_free_rotation_snaps_its_angle() {
        let step = FreeSnap { grid: 0.25, angle: 0.1, factor: 0.1 };
        let (mut scene, id) = one_node(Vec3::ZERO);
        let drag = DragState {
            mode: TransformMode::Rotate,
            start_point: Vec3::ZERO,
            start_transforms: vec![(id, scene.get(id).unwrap().transform)],
            axis: Vec3::X,
        };
        // With axis X the drag plane is horizontal and the rotation runs
        // along Z: a ray landing at z = 0.23 gives an angle of 0.23 radians,
        // which snaps to 0.2.
        let at = pointer(Vec3::new(0.0, 5.0, 0.23), Some(step));
        let mut tool = TransformTool::new(TransformMode::Rotate);
        let mut ctx = ToolContext {
            scene: &mut scene,
            selection: &mut Selection::new(),
            drag: Some(drag),
        };
        tool.on_pointer_move(&mut ctx, &at);
        let got = scene.get(id).unwrap().transform.rotation;
        let expected = glam::Quat::from_axis_angle(Vec3::X, 0.2);
        assert!(
            (got.x - expected.x).abs() < 1e-5 && (got.w - expected.w).abs() < 1e-5,
            "rotation {got:?} vs {expected:?}"
        );
    }
}
