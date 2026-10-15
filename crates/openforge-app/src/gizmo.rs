//! The interactive transform gizmo.
//!
//! The gizmo is drawn directly by the egui painter (on top of either viewport
//! backend), so its handles are *screen-space* geometry: a constant number of
//! pixels regardless of how far away the object is. Hit-testing therefore works
//! on projected 2D points, while the actual edit is resolved in world space
//! against the picked axis — a drag slides the selection along that axis, or
//! spins it around it, instead of the loose view-plane motion of a free drag.
//!
//! Everything here except the painting is pure math, which is what the unit
//! tests at the bottom of this file exercise.

use crate::viewport::{self, Camera};
use eframe::egui;
use glam::{Mat4, Quat, Vec3};
use openforge_core::math::{Ray, Transform};
use openforge_core::scene::Scene;
use openforge_core::tool::TransformMode;
use openforge_core::NodeId;

/// How long an axis arm is on screen, in pixels.
pub const HANDLE_PIXELS: f32 = 90.0;
/// How close the cursor must come to a handle to grab it, in pixels.
pub const HIT_TOLERANCE: f32 = 8.0;
/// Number of segments used to draw and hit-test a rotation ring.
pub const RING_SEGMENTS: usize = 48;

/// One of the three world axes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Axis {
    X,
    Y,
    Z,
}

impl Axis {
    pub const ALL: [Axis; 3] = [Axis::X, Axis::Y, Axis::Z];

    pub fn dir(self) -> Vec3 {
        match self {
            Axis::X => Vec3::X,
            Axis::Y => Vec3::Y,
            Axis::Z => Vec3::Z,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Axis::X => "X",
            Axis::Y => "Y",
            Axis::Z => "Z",
        }
    }

    /// Handle colour, brightened when the handle is hovered or being dragged.
    pub fn color(self, hot: bool) -> egui::Color32 {
        let (r, g, b) = match (self, hot) {
            (Axis::X, false) => (198, 78, 78),
            (Axis::X, true) => (255, 126, 110),
            (Axis::Y, false) => (122, 198, 96),
            (Axis::Y, true) => (168, 240, 130),
            (Axis::Z, false) => (86, 140, 232),
            (Axis::Z, true) => (130, 184, 255),
        };
        egui::Color32::from_rgb(r, g, b)
    }
}

/// Two unit vectors spanning the plane perpendicular to `axis`, in a fixed
/// right-handed order so that angles are stable frame to frame.
pub fn ring_basis(axis: Axis) -> (Vec3, Vec3) {
    match axis {
        Axis::X => (Vec3::Y, Vec3::Z),
        Axis::Y => (Vec3::Z, Vec3::X),
        Axis::Z => (Vec3::X, Vec3::Y),
    }
}

/// World length that spans `pixels` of a `height`-pixel viewport at `distance`.
pub fn pixel_size_to_world(distance: f32, fov_y: f32, height: f32, pixels: f32) -> f32 {
    if height <= 0.0 {
        return 0.0;
    }
    2.0 * distance.max(1e-3) * (fov_y * 0.5).tan() * (pixels / height)
}

/// The gizmo's world-space frame: where it sits and how big its handles are.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Frame {
    pub origin: Vec3,
    /// Length of one axis arm, in world units.
    pub arm: f32,
    /// Radius of the rotation rings, in world units.
    pub radius: f32,
}

impl Frame {
    /// A frame centred on `origin` whose handles are `pixels` tall on screen.
    ///
    /// The size is derived from the distance between the eye and `origin`, not
    /// from the camera's orbit radius, so a selection that has been moved away
    /// from the camera's target keeps the same handle size on screen.
    pub fn for_camera(camera: &Camera, viewport_height: f32, pixels: f32, origin: Vec3) -> Self {
        let distance = (origin - camera.eye()).length();
        let size = pixel_size_to_world(distance, camera.fov_y, viewport_height, pixels).max(1e-3);
        Self { origin, arm: size, radius: size }
    }

    pub fn tip(&self, axis: Axis) -> Vec3 {
        self.origin + axis.dir() * self.arm
    }

    /// The rotation ring around `axis`, closed and sampled at `segments` points.
    pub fn ring(&self, axis: Axis, segments: usize) -> Vec<Vec3> {
        let (u, v) = ring_basis(axis);
        let n = segments.max(3);
        (0..n)
            .map(|i| {
                let t = i as f32 / n as f32 * std::f32::consts::TAU;
                self.origin + (u * t.cos() + v * t.sin()) * self.radius
            })
            .collect()
    }
}

/// Distance from `p` to the segment `a`–`b`.
pub fn distance_to_segment(p: egui::Pos2, a: egui::Pos2, b: egui::Pos2) -> f32 {
    let ab = b - a;
    let len_sq = ab.length_sq();
    if len_sq <= f32::EPSILON {
        return (p - a).length();
    }
    let t = ((p - a).dot(ab) / len_sq).clamp(0.0, 1.0);
    (p - (a + ab * t)).length()
}

/// The gizmo projected to screen space for one frame.
#[derive(Clone, Debug)]
pub struct Layout {
    pub origin: egui::Pos2,
    /// One screen-space segment per axis, for the arms of Move/Scale.
    pub arms: Vec<(Axis, egui::Pos2)>,
    /// One closed screen-space loop per axis, for the rings of Rotate.
    pub rings: Vec<(Axis, Vec<egui::Pos2>)>,
}

impl Layout {
    /// Project a frame into `size`-pixel viewport coordinates offset by
    /// `offset` (the viewport rect's top-left, so the result is screen space).
    ///
    /// Returns `None` when the gizmo origin is behind the camera.
    pub fn build(vp: Mat4, size: [f32; 2], offset: egui::Vec2, frame: &Frame, segments: usize) -> Option<Self> {
        let project = |p: Vec3| {
            viewport::project(vp, p, size).map(|(x, y, _)| egui::pos2(x, y) + offset)
        };
        let origin = project(frame.origin)?;
        let arms = Axis::ALL
            .iter()
            .filter_map(|&a| project(frame.tip(a)).map(|p| (a, p)))
            .collect();
        let rings = Axis::ALL
            .iter()
            .map(|&a| {
                let pts = frame
                    .ring(a, segments)
                    .into_iter()
                    .filter_map(project)
                    .collect::<Vec<_>>();
                (a, pts)
            })
            .collect();
        Some(Self { origin, arms, rings })
    }

    /// The nearest axis whose arm is within `tol` pixels of `cursor`.
    pub fn hit_axis(&self, cursor: egui::Pos2, tol: f32) -> Option<Axis> {
        let mut best: Option<(f32, Axis)> = None;
        for &(axis, tip) in &self.arms {
            let d = distance_to_segment(cursor, self.origin, tip);
            if d <= tol && best.map(|(bd, _)| d < bd).unwrap_or(true) {
                best = Some((d, axis));
            }
        }
        best.map(|(_, axis)| axis)
    }

    /// The nearest rotation ring within `tol` pixels of `cursor`.
    pub fn hit_ring(&self, cursor: egui::Pos2, tol: f32) -> Option<Axis> {
        let mut best: Option<(f32, Axis)> = None;
        for (axis, pts) in &self.rings {
            if pts.len() < 2 {
                continue;
            }
            let mut d = f32::INFINITY;
            for w in pts.windows(2) {
                d = d.min(distance_to_segment(cursor, w[0], w[1]));
            }
            d = d.min(distance_to_segment(cursor, pts[pts.len() - 1], pts[0]));
            if d <= tol && best.map(|(bd, _)| d < bd).unwrap_or(true) {
                best = Some((d, *axis));
            }
        }
        best.map(|(_, axis)| axis)
    }
}

/// Closest point between `ray` and the infinite line `origin + t * dir`.
///
/// Because this returns the point *on the line*, its parameter doubles as the
/// handle position the cursor is over.
pub fn closest_point_on_line(ray: Ray, origin: Vec3, dir: Vec3) -> Option<Vec3> {
    let v = dir.normalize_or_zero();
    if v.length_squared() < 0.5 {
        return None;
    }
    let u = ray.direction.normalize_or_zero();
    if u.length_squared() < 0.5 {
        return None;
    }
    let w0 = ray.origin - origin;
    let a = u.dot(u);
    let b = u.dot(v);
    let c = v.dot(v);
    let d = u.dot(w0);
    let e = v.dot(w0);
    let denom = a * c - b * b;
    if denom.abs() < 1e-6 {
        // The ray runs parallel to the axis: no stable single closest point.
        return None;
    }
    let t = (a * e - b * d) / denom;
    Some(origin + v * t)
}

/// Intersect `ray` with the plane through `origin` with normal `normal`.
pub fn ray_plane(ray: Ray, origin: Vec3, normal: Vec3) -> Option<Vec3> {
    let n = normal.normalize_or_zero();
    if n.length_squared() < 0.5 {
        return None;
    }
    let denom = ray.direction.dot(n);
    if denom.abs() < 1e-6 {
        return None;
    }
    let t = (origin - ray.origin).dot(n) / denom;
    if t < 0.0 {
        return None;
    }
    Some(ray.at(t))
}

/// Signed angle of `p` around `axis`, measured in the axis' ring basis.
pub fn plane_angle(origin: Vec3, axis: Axis, p: Vec3) -> f32 {
    let (u, v) = ring_basis(axis);
    let r = p - origin;
    r.dot(v).atan2(r.dot(u))
}

/// Shortest signed difference between two angles, in `(-pi, pi]`.
pub fn angle_delta(from: f32, to: f32) -> f32 {
    use std::f32::consts::{PI, TAU};
    let mut d = (to - from) % TAU;
    if d > PI {
        d -= TAU;
    }
    if d <= -PI {
        d += TAU;
    }
    d
}

/// How far a drag has travelled, in the unit its mode consumes.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Measure {
    /// Move: world-space distance along the axis.
    Translate(f32),
    /// Scale: multiplier relative to the drag start.
    Scale(f32),
    /// Rotate: radians around the axis.
    Rotate(f32),
}

/// Apply a drag measure to one node's drag-start transform.
pub fn apply_measure(axis: Axis, start: &Transform, measure: Measure) -> Transform {
    let mut out = *start;
    match measure {
        Measure::Translate(d) => out.translation = start.translation + axis.dir() * d,
        Measure::Scale(f) => out.scale = start.scale * f.clamp(0.01, 1000.0),
        Measure::Rotate(a) => out.rotation = Quat::from_axis_angle(axis.dir(), a) * start.rotation,
    }
    out
}

/// Where a gizmo drag started, so each frame's cursor ray can be turned back
/// into a measure relative to that moment.
#[derive(Clone, Copy, Debug)]
pub struct Anchor {
    pub mode: TransformMode,
    pub axis: Axis,
    pub origin: Vec3,
    /// World point on the axis line where the drag began (Move/Scale).
    start_point: Vec3,
    /// Angle around the axis where the drag began (Rotate).
    start_angle: f32,
}

impl Anchor {
    /// Resolve a drag that is starting now, from the ray under the cursor.
    ///
    /// `None` when the ray cannot be resolved against the axis (parallel, or
    /// pointing away from the rotation plane).
    pub fn new(mode: TransformMode, axis: Axis, origin: Vec3, ray: Ray) -> Option<Self> {
        match mode {
            TransformMode::Rotate => {
                let p = ray_plane(ray, origin, axis.dir())?;
                Some(Self {
                    mode,
                    axis,
                    origin,
                    start_point: origin,
                    start_angle: plane_angle(origin, axis, p),
                })
            }
            _ => {
                let p = closest_point_on_line(ray, origin, axis.dir())?;
                Some(Self { mode, axis, origin, start_point: p, start_angle: 0.0 })
            }
        }
    }

    pub fn measure(&self, ray: Ray) -> Option<Measure> {
        let dir = self.axis.dir();
        match self.mode {
            TransformMode::Rotate => {
                let p = ray_plane(ray, self.origin, dir)?;
                let angle = angle_delta(self.start_angle, plane_angle(self.origin, self.axis, p));
                Some(Measure::Rotate(angle))
            }
            TransformMode::Move => {
                let p = closest_point_on_line(ray, self.origin, dir)?;
                Some(Measure::Translate((p - self.start_point).dot(dir)))
            }
            TransformMode::Scale => {
                let p = closest_point_on_line(ray, self.origin, dir)?;
                let start = (self.start_point - self.origin).dot(dir);
                let now = (p - self.origin).dot(dir);
                // Grabbing the arm away from the centre gives a true ratio;
                // near the centre that ratio explodes, so fall back to an
                // additive factor there.
                let factor = if start.abs() > 1e-3 { now / start } else { 1.0 + (now - start) };
                Some(Measure::Scale(factor))
            }
        }
    }
}

/// The world-space transform a measure describes, about `pivot`.
pub fn measure_matrix(axis: Axis, measure: Measure, pivot: Vec3) -> Mat4 {
    let about_pivot = |m: Mat4| Mat4::from_translation(pivot) * m * Mat4::from_translation(-pivot);
    match measure {
        Measure::Translate(d) => Mat4::from_translation(axis.dir() * d),
        Measure::Scale(f) => about_pivot(Mat4::from_scale(Vec3::splat(f))),
        Measure::Rotate(a) => about_pivot(Mat4::from_axis_angle(axis.dir(), a)),
    }
}

/// One node's vertices, as they were when the drag began.
#[derive(Clone, Debug)]
pub struct VertexTarget {
    pub node: NodeId,
    /// Mesh-local positions, keyed by vertex index.
    pub positions: Vec<(u32, Vec3)>,
}

/// What a drag moves.
#[derive(Clone, Debug)]
pub enum Targets {
    /// Whole nodes, re-derived from the transform each held when the drag
    /// began.
    Nodes(Vec<(NodeId, Transform)>),
    /// The vertices behind a sub-object selection, re-derived from the local
    /// positions each held when the drag began.
    Vertices(Vec<VertexTarget>),
}

impl Targets {
    /// What snapping must ignore, so a drag can never snap to the geometry it
    /// is carrying: the nodes whose vertices all move, and the individual
    /// vertices that move (which may be a subset of their node).
    ///
    /// A whole node is excluded at object level because its own pivot sits on
    /// its geometry: without that, the nearest vertex under the cursor would be
    /// the one the drag started on and the node could never move at all.
    pub fn moving(&self) -> (Vec<NodeId>, Vec<(NodeId, u32)>) {
        match self {
            Targets::Nodes(nodes) => (nodes.iter().map(|(id, _)| *id).collect(), Vec::new()),
            Targets::Vertices(targets) => (
                Vec::new(),
                targets
                    .iter()
                    .flat_map(|t| t.positions.iter().map(|(vi, _)| (t.node, *vi)))
                    .collect(),
            ),
        }
    }
}

/// An in-progress axis-constrained drag.
#[derive(Clone, Debug)]
pub struct Drag {
    pub mode: TransformMode,
    pub axis: Axis,
    pub anchor: Anchor,
    pub targets: Targets,
}

impl Drag {
    pub fn undo_label(&self) -> &'static str {
        match self.mode {
            TransformMode::Move => "Move",
            TransformMode::Scale => "Scale",
            TransformMode::Rotate => "Rotate",
        }
    }
}

/// The gizmo's live state: what is hovered, what is being dragged, and where
/// its handles were laid out for the frame being drawn.
#[derive(Default)]
pub struct Gizmo {
    pub hovered: Option<Axis>,
    /// The axis being dragged, held for the whole gesture.
    pub active: Option<Axis>,
    pub drag: Option<Drag>,
    /// This frame's screen layout, refreshed by the input pass and consumed by
    /// the painting pass so both agree on where the handles are.
    pub layout: Option<Layout>,
}

impl Gizmo {
    pub fn is_dragging(&self) -> bool {
        self.drag.is_some()
    }

    /// The axis to highlight: the dragged one while dragging, otherwise hover.
    pub fn highlight(&self) -> Option<Axis> {
        self.active.or(self.hovered)
    }

    pub fn begin(&mut self, drag: Drag) {
        self.active = Some(drag.axis);
        self.hovered = None;
        self.drag = Some(drag);
    }

    /// End the drag, returning the undo label for it.
    pub fn finish(&mut self) -> Option<&'static str> {
        self.active = None;
        self.drag.take().map(|d| d.undo_label())
    }

    /// Move the dragged targets along the dragged axis, using the cursor ray.
    ///
    /// `snap` and `vertex` come from the app, which resolves the vertex under
    /// the cursor (if any) against the scene's other geometry: [`apply_measure`]
    /// gets the snapped measure, so the drag lands where the settings say.
    ///
    /// Every target is re-derived from its drag-start state, so the edit is
    /// idempotent and never accumulates rounding error. Returns whether the
    /// measure could be resolved this frame.
    pub fn apply(
        &self,
        scene: &mut Scene,
        ray: Ray,
        snap: crate::snap::Snap,
        vertex: Option<Vec3>,
    ) -> bool {
        let Some(drag) = self.drag.as_ref() else { return false };
        let Some(measure) = drag.anchor.measure(ray) else { return false };
        let measure = crate::snap::snap_measure(snap, drag.axis, drag.anchor.origin, measure, vertex);
        match &drag.targets {
            Targets::Nodes(nodes) => {
                for (id, start) in nodes {
                    let Ok(node) = scene.get_mut(*id) else { continue };
                    node.transform = apply_measure(drag.axis, start, measure);
                }
            }
            Targets::Vertices(targets) => {
                // The measure is world-space; the vertices are mesh-local, so
                // the transform is carried through each node's world matrix.
                let world_space = measure_matrix(drag.axis, measure, drag.anchor.origin);
                for target in targets {
                    let Ok(world) = scene.world_matrix(target.node) else { continue };
                    if world.determinant().abs() < 1e-12 {
                        continue;
                    }
                    let local = world.inverse() * world_space * world;
                    let Ok(node) = scene.get_mut(target.node) else { continue };
                    let Some(mesh) = node.mesh_mut() else { continue };
                    for (vi, start) in &target.positions {
                        if let Some(p) = mesh.positions.get_mut(*vi as usize) {
                            *p = local.transform_point3(*start);
                        }
                    }
                }
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::snap::Snap;
    use openforge_core::mesh::Mesh;

    /// Apply a drag with snapping off — the app's default, and what these pure
    /// gizmo tests exercise. Snapped drags are covered by `snap.rs` and the
    /// headless interaction tests in `app.rs`.
    fn gizmo_apply(gizmo: &Gizmo, scene: &mut Scene, ray: Ray) -> bool {
        gizmo.apply(scene, ray, Snap::default(), None)
    }

    fn camera() -> Camera {
        Camera { distance: 8.0, ..Default::default() }
    }

    fn layout_for(frame: &Frame) -> Layout {
        let cam = camera();
        let vp = cam.view_proj(1.6);
        Layout::build(vp, [800.0, 500.0], egui::Vec2::ZERO, frame, RING_SEGMENTS)
            .expect("gizmo origin is in front of the camera")
    }

    #[test]
    fn pixel_size_grows_with_distance_and_pixels() {
        let near = pixel_size_to_world(4.0, 1.0, 500.0, 90.0);
        let far = pixel_size_to_world(16.0, 1.0, 500.0, 90.0);
        assert!(far > near);
        assert!((far - near * 4.0).abs() < 1e-3, "size is linear in distance");

        let wide = pixel_size_to_world(4.0, 1.0, 500.0, 180.0);
        assert!((wide - near * 2.0).abs() < 1e-3, "size is linear in pixels");

        // A camera at distance d with a square-ish viewport: the fraction of
        // the view the handle spans matches the fraction of the viewport.
        let fraction = near / (2.0 * 4.0 * (0.5f32).tan());
        assert!((fraction - 90.0 / 500.0).abs() < 1e-4);
    }

    #[test]
    fn pixel_size_is_zero_for_an_empty_viewport() {
        assert_eq!(pixel_size_to_world(4.0, 1.0, 0.0, 90.0), 0.0);
    }

    #[test]
    fn handle_size_follows_the_gizmo_distance_not_the_orbit_radius() {
        let cam = camera();
        let at_target = Frame::for_camera(&cam, 500.0, 90.0, cam.target);
        // Halfway from the eye to the target: half the world size, which is the
        // same number of pixels on screen.
        let eye = cam.eye();
        let half_way = eye + (cam.target - eye) * 0.5;
        let nearer = Frame::for_camera(&cam, 500.0, 90.0, half_way);
        assert!(
            (nearer.arm - at_target.arm * 0.5).abs() < 1e-3,
            "{} vs {}",
            nearer.arm,
            at_target.arm
        );
    }

    #[test]
    fn ring_points_lie_in_the_axis_plane() {
        let frame = Frame { origin: Vec3::new(1.0, 2.0, 3.0), arm: 2.0, radius: 2.0 };
        for axis in Axis::ALL {
            let ring = frame.ring(axis, 16);
            assert_eq!(ring.len(), 16);
            for p in ring {
                let offset = p - frame.origin;
                assert!(offset.dot(axis.dir()).abs() < 1e-4, "{axis:?} ring is flat");
                assert!((offset.length() - frame.radius).abs() < 1e-4, "{axis:?} ring radius");
            }
        }
    }

    #[test]
    fn tips_sit_along_their_axis() {
        let frame = Frame { origin: Vec3::ZERO, arm: 2.0, radius: 2.0 };
        assert_eq!(frame.tip(Axis::Y), Vec3::new(0.0, 2.0, 0.0));
    }

    #[test]
    fn hit_axis_finds_the_arm_under_the_cursor() {
        let frame = Frame { origin: Vec3::ZERO, arm: 1.0, radius: 1.0 };
        let layout = layout_for(&frame);
        let tip = layout
            .arms
            .iter()
            .find(|(a, _)| *a == Axis::X)
            .map(|(_, p)| *p)
            .expect("X arm projects");

        assert_eq!(layout.hit_axis(tip, HIT_TOLERANCE), Some(Axis::X));
        // Mid-shaft grabs count too.
        let mid = layout.origin + (tip - layout.origin) * 0.5;
        assert_eq!(layout.hit_axis(mid, HIT_TOLERANCE), Some(Axis::X));
        // Well clear of every handle: nothing.
        assert_eq!(layout.hit_axis(layout.origin + egui::vec2(200.0, 200.0), HIT_TOLERANCE), None);
    }

    #[test]
    fn hit_axis_reports_the_nearest_arm() {
        let frame = Frame { origin: Vec3::ZERO, arm: 1.0, radius: 1.0 };
        let layout = layout_for(&frame);
        // A cursor a couple of pixels off the Y arm picks Y, not a neighbour,
        // as long as no other arm is closer.
        let y_tip = layout.arms.iter().find(|(a, _)| *a == Axis::Y).map(|(_, p)| *p).unwrap();
        let nudge = y_tip + egui::vec2(3.0, 3.0);
        assert_eq!(layout.hit_axis(nudge, HIT_TOLERANCE), Some(Axis::Y));
    }

    #[test]
    fn hit_ring_finds_a_ring_edge() {
        let frame = Frame { origin: Vec3::ZERO, arm: 1.5, radius: 1.5 };
        let layout = layout_for(&frame);
        let (axis, pts) = layout
            .rings
            .iter()
            .find(|(a, pts)| *a == Axis::Y && pts.len() >= 8)
            .expect("Y ring projects");
        let on_ring = pts[3];
        assert_eq!(layout.hit_ring(on_ring, HIT_TOLERANCE), Some(*axis));
        // The ring's centre is far from any ring edge.
        assert_eq!(layout.hit_ring(layout.origin, 2.0), None);
    }

    #[test]
    fn closest_point_slides_along_the_axis() {
        // A ray perpendicular to the X axis, sweeping down at x = 5. The point
        // of the X axis closest to it is therefore the origin of that sweep.
        let ray = Ray { origin: Vec3::new(5.0, 3.0, 4.0), direction: -Vec3::Z };
        let p = closest_point_on_line(ray, Vec3::ZERO, Vec3::X).unwrap();
        assert!((p - Vec3::new(5.0, 0.0, 0.0)).length() < 1e-4, "got {p:?}");

        // An offset origin shifts the answer along the axis with it.
        let moved = closest_point_on_line(ray, Vec3::new(2.0, 0.0, 0.0), Vec3::X).unwrap();
        assert!((moved - Vec3::new(5.0, 0.0, 0.0)).length() < 1e-4, "got {moved:?}");
    }

    #[test]
    fn closest_point_rejects_parallel_rays() {
        let ray = Ray { origin: Vec3::new(0.0, 1.0, 0.0), direction: Vec3::X };
        assert_eq!(closest_point_on_line(ray, Vec3::ZERO, Vec3::X), None);
    }

    #[test]
    fn ray_plane_hits_and_misses() {
        let ray = Ray { origin: Vec3::new(0.0, 4.0, 0.0), direction: -Vec3::Y };
        let hit = ray_plane(ray, Vec3::ZERO, Vec3::Y).unwrap();
        assert!(hit.length() < 1e-5, "lands on the plane origin, got {hit:?}");
        // A ray parallel to the plane never meets it.
        let parallel = Ray { origin: Vec3::new(0.0, 4.0, 0.0), direction: Vec3::X };
        assert_eq!(ray_plane(parallel, Vec3::ZERO, Vec3::Y), None);
    }

    #[test]
    fn plane_angle_follows_the_ring_basis() {
        let frame = Frame { origin: Vec3::new(2.0, -1.0, 0.5), arm: 1.0, radius: 1.0 };
        let (u, v) = ring_basis(Axis::Z);
        let on_u = frame.origin + u;
        let on_v = frame.origin + v;
        assert!(plane_angle(frame.origin, Axis::Z, on_u).abs() < 1e-5);
        assert!((plane_angle(frame.origin, Axis::Z, on_v) - std::f32::consts::FRAC_PI_2).abs() < 1e-5);
    }

    #[test]
    fn angle_delta_takes_the_short_way_round() {
        use std::f32::consts::PI;
        // Wrapping across ±pi stays a small step, not a near-full turn.
        assert!((angle_delta(PI - 0.1, -PI + 0.1) - 0.2).abs() < 1e-4);
        assert!((angle_delta(0.0, 0.5) - 0.5).abs() < 1e-5);
        assert!((angle_delta(0.5, 0.0) + 0.5).abs() < 1e-5);
    }

    #[test]
    fn measure_translates_along_the_axis() {
        let anchor = Anchor::new(TransformMode::Move, Axis::X, Vec3::ZERO, Ray {
            origin: Vec3::new(2.0, 5.0, 0.0),
            direction: -Vec3::Y,
        })
        .unwrap();
        // Move the cursor to X = 4 and the drag has travelled 2 units.
        let m = anchor
            .measure(Ray { origin: Vec3::new(4.0, 5.0, 0.0), direction: -Vec3::Y })
            .unwrap();
        assert_eq!(m, Measure::Translate(2.0));

        let start = Transform::from_translation(Vec3::new(1.0, 1.0, 1.0));
        let moved = apply_measure(Axis::X, &start, m);
        assert!((moved.translation - Vec3::new(3.0, 1.0, 1.0)).length() < 1e-5);
    }

    #[test]
    fn measure_scales_relative_to_the_grab_point() {
        let origin = Vec3::ZERO;
        let anchor = Anchor::new(TransformMode::Scale, Axis::Y, origin, Ray {
            origin: Vec3::new(6.0, 1.0, 0.0),
            direction: -Vec3::X,
        })
        .unwrap();
        // Grabbed at Y = 1, now over Y = 2: a 2x scale.
        let m = anchor
            .measure(Ray { origin: Vec3::new(6.0, 2.0, 0.0), direction: -Vec3::X })
            .unwrap();
        assert_eq!(m, Measure::Scale(2.0));

        let start = Transform::identity();
        let scaled = apply_measure(Axis::Y, &start, m);
        assert!((scaled.scale - Vec3::splat(2.0)).length() < 1e-5);

        // Shrinking towards the centre is clamped, never inverted.
        let tiny = apply_measure(Axis::Y, &start, Measure::Scale(-3.0));
        assert!(tiny.scale.y > 0.0);
    }

    #[test]
    fn measure_rotates_around_the_axis() {
        // Look straight down -Z at the XY plane, where a Z rotation happens.
        let ray_at = |x: f32, y: f32| Ray { origin: Vec3::new(x, y, 5.0), direction: -Vec3::Z };
        let anchor = Anchor::new(TransformMode::Rotate, Axis::Z, Vec3::ZERO, ray_at(1.0, 0.0)).unwrap();
        // Sweeping from +X to +Y is a quarter turn about Z.
        let m = anchor.measure(ray_at(0.0, 1.0)).unwrap();
        match m {
            Measure::Rotate(a) => assert!((a - std::f32::consts::FRAC_PI_2).abs() < 1e-4, "got {a}"),
            other => panic!("expected a rotation, got {other:?}"),
        }

        let start = Transform::identity();
        let spun = apply_measure(Axis::Z, &start, m);
        let rotated = spun.rotation * Vec3::X;
        assert!((rotated - Vec3::Y).length() < 1e-4, "X rotated into Y, got {rotated:?}");
    }

    #[test]
    fn apply_re_derives_from_the_start_transform_each_frame() {
        // Two successive drags from the same start must not compound.
        let mut scene = Scene::new();
        let mut mesh = Mesh::new();
        mesh.add_vertex(Vec3::new(-1.0, -1.0, 0.0));
        mesh.add_vertex(Vec3::new(1.0, -1.0, 0.0));
        mesh.add_vertex(Vec3::new(1.0, 1.0, 0.0));
        mesh.add_face(&[0, 1, 2]);
        let id = scene.create_mesh("tri", mesh);

        let mut gizmo = Gizmo::default();
        let ray = Ray { origin: Vec3::new(0.0, 0.0, 5.0), direction: -Vec3::Z };
        let anchor = Anchor::new(TransformMode::Move, Axis::X, Vec3::ZERO, ray).unwrap();
        let start = scene.get(id).unwrap().transform;
        gizmo.begin(Drag {
            mode: TransformMode::Move,
            axis: Axis::X,
            anchor,
            targets: Targets::Nodes(vec![(id, start)]),
        });

        let ray_at = |x: f32| Ray { origin: Vec3::new(x, 0.0, 5.0), direction: -Vec3::Z };
        assert!(gizmo_apply(&gizmo, &mut scene, ray_at(0.0)));
        assert!(gizmo_apply(&gizmo, &mut scene, ray_at(3.0)));
        assert!(gizmo_apply(&gizmo, &mut scene, ray_at(1.0)));

        let t = scene.get(id).unwrap().transform.translation;
        assert!((t - Vec3::new(1.0, 0.0, 0.0)).length() < 1e-5, "got {t:?}");
        assert_eq!(gizmo.finish(), Some("Move"));
        assert!(!gizmo.is_dragging());
    }

    #[test]
    fn measure_matrix_moves_scales_and_spins_about_the_pivot() {
        let pivot = Vec3::new(1.0, 0.0, 0.0);
        let p = Vec3::new(2.0, 0.0, 0.0);

        let moved = measure_matrix(Axis::X, Measure::Translate(3.0), pivot).transform_point3(p);
        assert!((moved - Vec3::new(5.0, 0.0, 0.0)).length() < 1e-5, "got {moved:?}");

        // Scaling about the pivot: x = 1 + (2 - 1) * 2.
        let scaled = measure_matrix(Axis::Y, Measure::Scale(2.0), pivot).transform_point3(p);
        assert!((scaled - Vec3::new(3.0, 0.0, 0.0)).length() < 1e-5, "got {scaled:?}");

        // A quarter turn about Z around the pivot sends +X to +Y.
        let spun = measure_matrix(Axis::Z, Measure::Rotate(std::f32::consts::FRAC_PI_2), pivot)
            .transform_point3(p);
        assert!((spun - Vec3::new(1.0, 1.0, 0.0)).length() < 1e-4, "got {spun:?}");
    }

    /// A unit square in the XY plane whose corners are vertices 0..3.
    fn square_scene() -> (Scene, NodeId) {
        let mut mesh = Mesh::new();
        mesh.add_vertex(Vec3::new(0.0, 0.0, 0.0));
        mesh.add_vertex(Vec3::new(1.0, 0.0, 0.0));
        mesh.add_vertex(Vec3::new(1.0, 1.0, 0.0));
        mesh.add_vertex(Vec3::new(0.0, 1.0, 0.0));
        mesh.add_face(&[0, 1, 2, 3]);
        let mut scene = Scene::new();
        let id = scene.create_mesh("square", mesh);
        (scene, id)
    }

    fn vertex_drag(node: NodeId, positions: Vec<(u32, Vec3)>) -> (Gizmo, Vec3) {
        let ray = Ray { origin: Vec3::new(0.0, 5.0, 0.0), direction: -Vec3::Y };
        let anchor = Anchor::new(TransformMode::Move, Axis::X, Vec3::ZERO, ray).unwrap();
        let mut gizmo = Gizmo::default();
        gizmo.begin(Drag {
            mode: TransformMode::Move,
            axis: Axis::X,
            anchor,
            targets: Targets::Vertices(vec![VertexTarget { node, positions }]),
        });
        (gizmo, Vec3::ZERO)
    }

    fn drag_ray(x: f32) -> Ray {
        Ray { origin: Vec3::new(x, 5.0, 0.0), direction: -Vec3::Y }
    }

    #[test]
    fn a_vertex_drag_moves_only_the_vertices_it_was_given() {
        let (mut scene, id) = square_scene();
        let (gizmo, _) = vertex_drag(
            id,
            vec![(0, Vec3::new(0.0, 0.0, 0.0)), (2, Vec3::new(1.0, 1.0, 0.0))],
        );
        assert!(gizmo_apply(&gizmo, &mut scene, drag_ray(2.0)));

        let mesh = scene.get(id).unwrap().mesh().unwrap();
        assert!((mesh.positions[0] - Vec3::new(2.0, 0.0, 0.0)).length() < 1e-5);
        assert!((mesh.positions[2] - Vec3::new(3.0, 1.0, 0.0)).length() < 1e-5);
        // The corners left out of the drag stay exactly where they were.
        assert!((mesh.positions[1] - Vec3::new(1.0, 0.0, 0.0)).length() < 1e-5);
        assert!((mesh.positions[3] - Vec3::new(0.0, 1.0, 0.0)).length() < 1e-5);
        // Sub-object editing leaves the node's own transform alone.
        assert_eq!(scene.get(id).unwrap().transform, Transform::identity());
    }

    #[test]
    fn a_vertex_drag_is_re_derived_from_the_start_positions() {
        let (mut scene, id) = square_scene();
        let (gizmo, _) = vertex_drag(id, vec![(1, Vec3::new(1.0, 0.0, 0.0))]);
        // Sweeping the cursor around must not accumulate: the last position is
        // one drag measured from the start, i.e. x = 1 + 4.
        for x in [0.5, 3.0, -2.0, 4.0] {
            gizmo_apply(&gizmo, &mut scene, drag_ray(x));
        }
        let mesh = scene.get(id).unwrap().mesh().unwrap();
        assert!(
            (mesh.positions[1] - Vec3::new(5.0, 0.0, 0.0)).length() < 1e-5,
            "only the last ray counts: {:?}",
            mesh.positions[1]
        );
    }

    #[test]
    fn a_vertex_drag_is_carried_through_the_node_transform() {
        let (mut scene, id) = square_scene();
        {
            // The node sits at x = 5 and is scaled twice over, so one world
            // unit of drag is half a unit of mesh-local movement.
            let node = scene.get_mut(id).unwrap();
            node.transform.translation = Vec3::new(5.0, 0.0, 0.0);
            node.transform.scale = Vec3::splat(2.0);
        }
        let (gizmo, _) = vertex_drag(id, vec![(0, Vec3::ZERO)]);
        assert!(gizmo_apply(&gizmo, &mut scene, drag_ray(1.0)));

        let local = scene.get(id).unwrap().mesh().unwrap().positions[0];
        let world = scene.world_matrix(id).unwrap().transform_point3(local);
        assert!(
            (world - Vec3::new(6.0, 0.0, 0.0)).length() < 1e-4,
            "one world unit along +X from x = 5: got {world:?}"
        );
    }

    #[test]
    fn a_vertex_scale_drag_pulls_the_vertices_about_the_pivot() {
        let (mut scene, id) = square_scene();
        // Grab the +Y arm at y = 1 (the pivot is the origin), then drag out to
        // y = 2 for a 2x scale.
        let anchor = Anchor::new(
            TransformMode::Scale,
            Axis::Y,
            Vec3::ZERO,
            Ray { origin: Vec3::new(5.0, 1.0, 0.0), direction: -Vec3::X },
        )
        .unwrap();
        let mut gizmo = Gizmo::default();
        gizmo.begin(Drag {
            mode: TransformMode::Scale,
            axis: Axis::Y,
            anchor,
            targets: Targets::Vertices(vec![VertexTarget {
                node: id,
                positions: vec![(2, Vec3::new(1.0, 1.0, 0.0)), (3, Vec3::new(0.0, 1.0, 0.0))],
            }]),
        });
        assert!(gizmo_apply(
            &gizmo,
            &mut scene,
            Ray { origin: Vec3::new(5.0, 2.0, 0.0), direction: -Vec3::X }
        ));

        let mesh = scene.get(id).unwrap().mesh().unwrap();
        assert!((mesh.positions[2] - Vec3::new(2.0, 2.0, 0.0)).length() < 1e-4, "got {:?}", mesh.positions[2]);
        assert!((mesh.positions[3] - Vec3::new(0.0, 2.0, 0.0)).length() < 1e-4, "got {:?}", mesh.positions[3]);
    }

    #[test]
    fn a_lone_axis_grab_does_not_move_the_other_axes() {
        let start = Transform::from_translation(Vec3::new(4.0, 4.0, 4.0));
        let moved = apply_measure(Axis::Z, &start, Measure::Translate(-2.0));
        assert!((moved.translation - Vec3::new(4.0, 4.0, 2.0)).length() < 1e-5);
    }

    #[test]
    fn moving_reports_the_geometry_a_drag_carries() {
        let (_scene, id) = square_scene();

        // An object-level drag carries every vertex of the nodes it moves.
        let nodes = Targets::Nodes(vec![(id, Transform::identity())]);
        let (moving_nodes, moving_vertices) = nodes.moving();
        assert_eq!(moving_nodes, vec![id]);
        assert!(moving_vertices.is_empty());

        // An element-level drag carries only the vertices it lists, so the rest
        // of the mesh stays a valid thing to snap to.
        let vertices = Targets::Vertices(vec![VertexTarget {
            node: id,
            positions: vec![(0, Vec3::ZERO), (2, Vec3::ONE)],
        }]);
        let (moving_nodes, moving_vertices) = vertices.moving();
        assert!(moving_nodes.is_empty());
        assert_eq!(moving_vertices, vec![(id, 0), (id, 2)]);
    }
}
