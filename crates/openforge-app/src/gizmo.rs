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
/// Where a plane handle's square sits along its two axes, as a fraction of the
/// arm length (its centre, measured from the gizmo origin on both axes).
pub const PLANE_CENTRE: f32 = 0.38;
/// Half the side of a plane handle's square, as a fraction of the arm length.
pub const PLANE_HALF: f32 = 0.13;
/// Half the side of the centre handle's square, in pixels (it faces the camera,
/// so a constant screen size is what keeps it usable at any distance).
pub const CENTRE_PIXELS: f32 = 5.0;

/// Which way a gizmo's handles point.
///
/// The handles are always labelled X, Y and Z, and always coloured red, green
/// and blue; what changes is what those three axes *are*. A modeller moves a
/// turned object along its own axes as often as along the world's.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Axes {
    /// The world's own axes, whatever the camera and the object do.
    #[default]
    World,
    /// The selected object's axes, taken from its rotation: what a modeller
    /// reaches for to move something along the way it is already turned.
    Local,
    /// The screen's axes — right, up and the view direction — so the handles
    /// sit the same way however the view and the object are set.
    Screen,
}

impl Axes {
    /// Every mode, in menu order.
    pub const ALL: [Axes; 3] = [Axes::World, Axes::Local, Axes::Screen];

    /// What the toolbar and the menu call it.
    pub fn label(self) -> &'static str {
        match self {
            Axes::World => "World",
            Axes::Local => "Local",
            Axes::Screen => "Screen",
        }
    }
}

/// Three orthonormal world-space unit vectors: what a gizmo's X, Y and Z
/// handles point along.
///
/// A drag's [`Measure`] is written in this basis' own components, so a handle
/// that moves "one axis" moves one *of these* axes, and the world-space
/// transform is built from it at the very end.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Basis {
    pub x: Vec3,
    pub y: Vec3,
    pub z: Vec3,
}

impl Default for Basis {
    fn default() -> Self {
        Self::WORLD
    }
}

impl Basis {
    /// The world's own axes.
    pub const WORLD: Basis = Basis { x: Vec3::X, y: Vec3::Y, z: Vec3::Z };

    /// The direction handle `axis` points along.
    pub fn axis(self, axis: Axis) -> Vec3 {
        match axis {
            Axis::X => self.x,
            Axis::Y => self.y,
            Axis::Z => self.z,
        }
    }

    /// A world-space direction as this basis' components (a projection: exact,
    /// because the three vectors are orthonormal).
    pub fn components(self, world: Vec3) -> Vec3 {
        Vec3::new(world.dot(self.x), world.dot(self.y), world.dot(self.z))
    }

    /// This basis' components back as a world-space direction.
    pub fn to_world(self, components: Vec3) -> Vec3 {
        self.x * components.x + self.y * components.y + self.z * components.z
    }

    /// A transform's own axes, so a local gizmo lines up with the object it is
    /// on. A scaled object's axes are normalized: the gizmo points the way the
    /// object does, not as far.
    pub fn of_matrix(matrix: Mat4) -> Basis {
        Basis {
            x: matrix.x_axis.truncate().normalize_or_zero(),
            y: matrix.y_axis.truncate().normalize_or_zero(),
            z: matrix.z_axis.truncate().normalize_or_zero(),
        }
    }

    /// The screen's axes: right, up, and the view direction pointing out of
    /// the screen towards the viewer, which is the order a right-handed view
    /// basis uses.
    pub fn of_camera(camera: &Camera) -> Basis {
        let view = camera.view_matrix();
        // The columns of the view rotation are the camera's axes in world
        // space; the third is the direction *out* of the screen.
        Basis {
            x: Vec3::new(view.x_axis.x, view.y_axis.x, view.z_axis.x).normalize_or_zero(),
            y: Vec3::new(view.x_axis.y, view.y_axis.y, view.z_axis.y).normalize_or_zero(),
            z: Vec3::new(view.x_axis.z, view.y_axis.z, view.z_axis.z).normalize_or_zero(),
        }
    }
}

/// One of the three world axes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Axis {
    X,
    Y,
    Z,
}

impl Axis {
    pub const ALL: [Axis; 3] = [Axis::X, Axis::Y, Axis::Z];

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

/// One of the three coordinate planes, spanned by two axes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Plane {
    XY,
    YZ,
    ZX,
}

impl Plane {
    pub const ALL: [Plane; 3] = [Plane::XY, Plane::YZ, Plane::ZX];

    /// The two axes the plane spans, in the order its handle's square is built.
    pub fn axes(self) -> (Axis, Axis) {
        match self {
            Plane::XY => (Axis::X, Axis::Y),
            Plane::YZ => (Axis::Y, Axis::Z),
            Plane::ZX => (Axis::Z, Axis::X),
        }
    }

    /// The axis the plane is perpendicular to, which is also the normal a drag
    /// in the plane is resolved against.
    pub fn normal(self) -> Axis {
        match self {
            Plane::XY => Axis::Z,
            Plane::YZ => Axis::X,
            Plane::ZX => Axis::Y,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Plane::XY => "XY",
            Plane::YZ => "YZ",
            Plane::ZX => "ZX",
        }
    }

    /// Handle colour: the two axis colours of the plane it spans, mixed.
    pub fn color(self, hot: bool) -> egui::Color32 {
        let (a, b) = self.axes();
        let (ca, cb) = (a.color(hot), b.color(hot));
        egui::Color32::from_rgb(
            ((ca.r() as u16 + cb.r() as u16) / 2) as u8,
            ((ca.g() as u16 + cb.g() as u16) / 2) as u8,
            ((ca.b() as u16 + cb.b() as u16) / 2) as u8,
        )
    }
}

/// Which of the gizmo's handles a gesture is on.
///
/// The three kinds resolve differently: an axis handle slides along (or spins
/// about) one world axis, a plane handle moves or scales in two, and the centre
/// handle works in the plane facing the camera — the screen plane, which has no
/// world axis to snap to.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Handle {
    /// An axis arm, for a move/scale along it or a spin about it.
    Axis(Axis),
    /// A square in one coordinate plane: a move or scale in its two axes.
    Plane(Plane),
    /// The square at the middle, which works in the plane facing the camera —
    /// the screen plane, which no world axis spans.
    Centre,
}

impl Handle {
    /// The world axes this handle can move along. The centre handle has none:
    /// its plane faces the camera, so no world axis is privileged.
    pub fn axes(self) -> Vec<Axis> {
        match self {
            Handle::Axis(a) => vec![a],
            Handle::Plane(p) => {
                let (u, v) = p.axes();
                vec![u, v]
            }
            Handle::Centre => Vec::new(),
        }
    }

    /// What the handle is called where a user reads it: the axis it follows,
    /// the pair of axes its plane spans, or `screen` for the centre.
    pub fn label(self) -> &'static str {
        match self {
            Handle::Axis(a) => a.label(),
            Handle::Plane(p) => p.label(),
            Handle::Centre => "screen",
        }
    }
}

/// Two orthonormal unit vectors spanning the plane perpendicular to `axis`,
/// in a fixed right-handed order (`u × v` points along `axis`) so that angles
/// are stable from frame to frame.
///
/// The companions of a world axis are the same two the gizmo has always used;
/// any other direction takes the world pair it is closest to and projects the
/// axis out of it.
pub fn ring_basis(axis: Vec3) -> (Vec3, Vec3) {
    let n = axis.normalize_or_zero();
    let (a, b) = {
        let (ax, ay, az) = (n.x.abs(), n.y.abs(), n.z.abs());
        if ax >= ay && ax >= az {
            (Vec3::Y, Vec3::Z)
        } else if ay >= az {
            (Vec3::Z, Vec3::X)
        } else {
            (Vec3::X, Vec3::Y)
        }
    };
    let u = (a - n * a.dot(n)).normalize_or_zero();
    let v = (b - n * b.dot(n) - u * b.dot(u)).normalize_or_zero();
    // A negative direction (or a turned one) can come out the wrong way round;
    // the pair has to stay right-handed about the axis it spans.
    if u.cross(v).dot(n) < 0.0 {
        (u, -v)
    } else {
        (u, v)
    }
}

/// World length that spans `pixels` of a `height`-pixel viewport at `distance`.
pub fn pixel_size_to_world(distance: f32, fov_y: f32, height: f32, pixels: f32) -> f32 {
    if height <= 0.0 {
        return 0.0;
    }
    2.0 * distance.max(1e-3) * (fov_y * 0.5).tan() * (pixels / height)
}

/// The gizmo's world-space frame: where it sits, which way it points, and how
/// big its handles are.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Frame {
    pub origin: Vec3,
    /// Which way the three handles point.
    pub basis: Basis,
    /// Length of one axis arm, in world units.
    pub arm: f32,
    /// Radius of the rotation rings, in world units.
    pub radius: f32,
}

impl Frame {
    /// A frame centred on `origin` whose handles are `pixels` tall on screen.
    ///
    /// The size is derived from the depth the projection gives `origin` — the
    /// distance from the eye in perspective, and the view's own depth in an
    /// orthographic view — so a selection that has been moved away from the
    /// camera's target keeps the same handle size on screen.
    pub fn for_camera(
        camera: &Camera,
        viewport_height: f32,
        pixels: f32,
        origin: Vec3,
        basis: Basis,
    ) -> Self {
        let distance = camera.eye_depth(origin);
        let size = pixel_size_to_world(distance, camera.fov_y, viewport_height, pixels).max(1e-3);
        Self { origin, basis, arm: size, radius: size }
    }

    /// The world direction handle `axis` points along.
    pub fn dir(&self, axis: Axis) -> Vec3 {
        self.basis.axis(axis)
    }

    pub fn tip(&self, axis: Axis) -> Vec3 {
        self.origin + self.dir(axis) * self.arm
    }

    /// The rotation ring around `axis`, closed and sampled at `segments` points.
    pub fn ring(&self, axis: Axis, segments: usize) -> Vec<Vec3> {
        let (u, v) = ring_basis(self.dir(axis));
        let n = segments.max(3);
        (0..n)
            .map(|i| {
                let t = i as f32 / n as f32 * std::f32::consts::TAU;
                self.origin + (u * t.cos() + v * t.sin()) * self.radius
            })
            .collect()
    }

    /// The square of the handle for `plane`, in the plane's own two axes,
    /// offset from the origin along both of them.
    ///
    /// Corners run counter-clockwise as seen from the plane's normal side.
    pub fn plane_quad(&self, plane: Plane) -> [Vec3; 4] {
        let (u, v) = plane.axes();
        let (du, dv) = (self.dir(u), self.dir(v));
        let centre = self.origin + (du + dv) * (self.arm * PLANE_CENTRE);
        let half = self.arm * PLANE_HALF;
        [
            centre - du * half - dv * half,
            centre + du * half - dv * half,
            centre + du * half + dv * half,
            centre - du * half + dv * half,
        ]
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
    /// One screen-space quad per coordinate plane, for the plane handles of
    /// Move/Scale.
    pub planes: Vec<(Plane, [egui::Pos2; 4])>,
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
        // A plane handle is dropped when any of its corners is off the camera,
        // so a partly-behind quad can never be grabbed where it is not drawn.
        let planes = Plane::ALL
            .iter()
            .filter_map(|&p| {
                let corners = frame.plane_quad(p);
                let pts: Vec<egui::Pos2> = corners.iter().filter_map(|c| project(*c)).collect();
                (pts.len() == 4).then(|| (p, [pts[0], pts[1], pts[2], pts[3]]))
            })
            .collect();
        Some(Self { origin, arms, rings, planes })
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

    /// The plane handle whose square is under `cursor`, if any.
    pub fn hit_plane(&self, cursor: egui::Pos2, tol: f32) -> Option<Plane> {
        self.planes
            .iter()
            .find(|(_, quad)| quad_contains(quad, cursor, tol))
            .map(|(plane, _)| *plane)
    }

    /// Whether the cursor is over the centre square (`tol` pixels of slack).
    pub fn hit_centre(&self, cursor: egui::Pos2, tol: f32) -> bool {
        let reach = CENTRE_PIXELS + tol;
        let d = cursor - self.origin;
        d.x.abs() <= reach && d.y.abs() <= reach
    }

    /// The handle under `cursor` for `mode`.
    ///
    /// The order is the order the handles are painted — the centre square over
    /// the plane squares over the arms — so what the cursor picks is what it
    /// appears to be over. Rotate has only its rings.
    pub fn hit(&self, mode: TransformMode, cursor: egui::Pos2, tol: f32) -> Option<Handle> {
        match mode {
            TransformMode::Rotate => self.hit_ring(cursor, tol).map(Handle::Axis),
            _ => {
                if self.hit_centre(cursor, tol) {
                    return Some(Handle::Centre);
                }
                if let Some(plane) = self.hit_plane(cursor, tol) {
                    return Some(Handle::Plane(plane));
                }
                self.hit_axis(cursor, tol).map(Handle::Axis)
            }
        }
    }
}

/// Whether `p` is inside the convex quad `quad`, or within `tol` of an edge.
///
/// A plane handle seen nearly edge-on projects to a sliver, so the tolerance is
/// what keeps it grabbable. A quad that has collapsed to a line or a point has
/// no inside at all, which is why the edge distance is checked as well as the
/// side of each edge.
fn quad_contains(quad: &[egui::Pos2; 4], p: egui::Pos2, tol: f32) -> bool {
    let mut side = 0.0f32;
    let mut usable = 0;
    let mut same_side = true;
    let mut edge_distance = f32::INFINITY;
    for i in 0..4 {
        let a = quad[i];
        let b = quad[(i + 1) % 4];
        edge_distance = edge_distance.min(distance_to_segment(p, a, b));
        let ab = b - a;
        let len = ab.length();
        if len <= 1e-6 {
            continue;
        }
        usable += 1;
        // Positive on one side of the edge, negative on the other.
        let d = (ab.x * (p.y - a.y) - ab.y * (p.x - a.x)) / len;
        if d.abs() > tol {
            if side == 0.0 {
                side = d.signum();
            } else if d.signum() != side {
                same_side = false;
            }
        }
    }
    (usable > 0 && same_side) || edge_distance <= tol
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

/// Signed angle of `p` around the direction `axis`, measured in its ring basis.
pub fn plane_angle(origin: Vec3, axis: Vec3, p: Vec3) -> f32 {
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
///
/// A translation carries its vector and a scale its per-axis factors, both
/// written in the *gizmo's own* components (see [`Basis`]), so a handle that
/// works in one axis and a handle that works in a plane are the same shape to
/// everything downstream: `1` in the factor of an axis a scale handle does not
/// touch, `0` in the component a move does not.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Measure {
    /// Move: the translation, in the gizmo's components.
    Translate(Vec3),
    /// Scale: a multiplier per gizmo axis, relative to the drag start.
    Scale(Vec3),
    /// Rotate: radians around the gizmo axis.
    Rotate(Axis, f32),
}

/// Per-axis scale factors that leave every axis alone except those in `axes`,
/// which all take `f`.
fn factors(axes: &[Axis], f: f32) -> Vec3 {
    let f = f.clamp(0.01, 1000.0);
    let mut out = Vec3::ONE;
    for &axis in axes {
        match axis {
            Axis::X => out.x = f,
            Axis::Y => out.y = f,
            Axis::Z => out.z = f,
        }
    }
    out
}

/// The component of `v` that `axis` names.
pub(crate) fn component(v: Vec3, axis: Axis) -> f32 {
    match axis {
        Axis::X => v.x,
        Axis::Y => v.y,
        Axis::Z => v.z,
    }
}

/// `out` with `value` written into the component `axis` names, leaving the
/// other two exactly as they were.
pub(crate) fn set_component(out: Vec3, axis: Axis, value: f32) -> Vec3 {
    match axis {
        Axis::X => Vec3::new(value, out.y, out.z),
        Axis::Y => Vec3::new(out.x, value, out.z),
        Axis::Z => Vec3::new(out.x, out.y, value),
    }
}

/// Apply a drag measure to one node's drag-start transform, about `origin` —
/// the gizmo's own centre, which is the selection's or a pivot the user placed.
///
/// `basis` is the gizmo's: the measure's components are its axes, so a local
/// gizmo moves the node along the way it is turned. Translation and rotation
/// are exact in every basis; a scale is exact along the world axes and along a
/// node's own axes (which is what a local gizmo asks for), and is applied to
/// the node's own scale axes elsewhere — a node transform cannot carry the
/// shear that scaling a turned object along a screen axis would need. Vertex
/// targets do not go through here: they take [`measure_matrix`], which is
/// exact in any basis.
///
/// A rotation or a scale moves the node's *position* about `origin` as well as
/// its own transform, which is what makes a placed pivot mean something: with
/// the pivot somewhere else, a turn swings the selection round it instead of
/// spinning it in place.
pub fn apply_measure(start: &Transform, basis: Basis, origin: Vec3, measure: Measure) -> Transform {
    let mut out = *start;
    match measure {
        Measure::Translate(d) => out.translation = start.translation + basis.to_world(d),
        Measure::Scale(f) => {
            let s = Vec3::new(
                f.x.clamp(0.01, 1000.0),
                f.y.clamp(0.01, 1000.0),
                f.z.clamp(0.01, 1000.0),
            );
            out.scale = start.scale * s;
            // The node's own position swings with the same stretch, exactly:
            // the map is a linear one however the basis is turned.
            let b = glam::Mat3::from_cols(basis.x, basis.y, basis.z);
            let linear = b * glam::Mat3::from_diagonal(s) * b.transpose();
            out.translation = origin + linear * (start.translation - origin);
        }
        Measure::Rotate(axis, a) => {
            let spin = Quat::from_axis_angle(basis.axis(axis), a);
            out.rotation = spin * start.rotation;
            out.translation = origin + spin * (start.translation - origin);
        }
    }
    out
}

/// Where a gizmo drag started, so each frame's cursor ray can be turned back
/// into a measure relative to that moment.
#[derive(Clone, Copy, Debug)]
pub struct Anchor {
    pub mode: TransformMode,
    pub handle: Handle,
    pub origin: Vec3,
    /// Which way the handles point, fixed for the whole gesture: a gizmo whose
    /// basis moved with the object it is dragging would chase itself.
    pub basis: Basis,
    /// The plane a non-axis drag is resolved in: the plane's own normal for a
    /// plane handle, the camera's view direction for the centre handle.
    plane_normal: Vec3,
    /// World point on the axis (or in the plane) where the drag began.
    start_point: Vec3,
    /// Angle around the axis where the drag began (Rotate).
    start_angle: f32,
}

impl Anchor {
    /// Resolve a drag that is starting now, from the ray under the cursor.
    ///
    /// `view` is the camera's view direction, which is the plane the centre
    /// handle works in. `None` when the ray cannot be resolved against the
    /// handle (parallel to the axis, or pointing away from its plane).
    pub fn new(
        mode: TransformMode,
        handle: Handle,
        origin: Vec3,
        basis: Basis,
        ray: Ray,
        view: Vec3,
    ) -> Option<Self> {
        let dir = |axis: Axis| basis.axis(axis);
        let plane_normal = match handle {
            Handle::Axis(axis) => dir(axis),
            Handle::Plane(plane) => dir(plane.normal()),
            Handle::Centre => view.normalize_or_zero(),
        };
        match (mode, handle) {
            (TransformMode::Rotate, Handle::Axis(axis)) => {
                let p = ray_plane(ray, origin, dir(axis))?;
                Some(Self {
                    mode,
                    handle,
                    origin,
                    basis,
                    plane_normal,
                    start_point: origin,
                    start_angle: plane_angle(origin, dir(axis), p),
                })
            }
            (TransformMode::Rotate, _) => None,
            (_, Handle::Axis(axis)) => {
                let p = closest_point_on_line(ray, origin, dir(axis))?;
                Some(Self {
                    mode,
                    handle,
                    origin,
                    basis,
                    plane_normal,
                    start_point: p,
                    start_angle: 0.0,
                })
            }
            (_, _) => {
                let p = ray_plane(ray, origin, plane_normal)?;
                Some(Self {
                    mode,
                    handle,
                    origin,
                    basis,
                    plane_normal,
                    start_point: p,
                    start_angle: 0.0,
                })
            }
        }
    }

    pub fn measure(&self, ray: Ray) -> Option<Measure> {
        let dir = |axis: Axis| self.basis.axis(axis);
        match (self.mode, self.handle) {
            (TransformMode::Rotate, Handle::Axis(axis)) => {
                let p = ray_plane(ray, self.origin, dir(axis))?;
                let angle = angle_delta(
                    self.start_angle,
                    plane_angle(self.origin, dir(axis), p),
                );
                Some(Measure::Rotate(axis, angle))
            }
            (TransformMode::Move, Handle::Axis(axis)) => {
                let d = dir(axis);
                let p = closest_point_on_line(ray, self.origin, d)?;
                // The measure is written in the gizmo's components, so only the
                // axis that moved carries a value and the other two are exactly
                // zero however the basis is turned.
                let travel = (p - self.start_point).dot(d);
                Some(Measure::Translate(set_component(Vec3::ZERO, axis, travel)))
            }
            (TransformMode::Move, _) => {
                // A plane or centre handle: the cursor's own point in the
                // drag's plane, relative to where the grab began. Both points
                // lie in that plane, so a plane's delta has no share of its own
                // normal, which is exactly what its measure leaves at zero.
                let p = ray_plane(ray, self.origin, self.plane_normal)?;
                let world = p - self.start_point;
                let d = match self.handle {
                    Handle::Plane(plane) => {
                        let (u, v) = plane.axes();
                        let mut c = set_component(Vec3::ZERO, u, world.dot(dir(u)));
                        c = set_component(c, v, world.dot(dir(v)));
                        c
                    }
                    _ => self.basis.components(world),
                };
                Some(Measure::Translate(d))
            }
            (TransformMode::Scale, Handle::Axis(axis)) => {
                let d = dir(axis);
                let p = closest_point_on_line(ray, self.origin, d)?;
                let start = (self.start_point - self.origin).dot(d);
                let now = (p - self.origin).dot(d);
                // Grabbing the arm away from the centre gives a true ratio;
                // near the centre that ratio explodes, so fall back to an
                // additive factor there.
                let factor = if start.abs() > 1e-3 { now / start } else { 1.0 + (now - start) };
                Some(Measure::Scale(factors(&[axis], factor)))
            }
            (TransformMode::Scale, Handle::Plane(plane)) => {
                // Both axes of the plane scale together, by how much further
                // from the origin the cursor has travelled within the plane.
                let p = ray_plane(ray, self.origin, self.plane_normal)?;
                let start = (self.start_point - self.origin).length();
                let now = (p - self.origin).length();
                let factor = if start > 1e-3 { now / start } else { 1.0 };
                let (u, v) = plane.axes();
                Some(Measure::Scale(factors(&[u, v], factor)))
            }
            (TransformMode::Scale, Handle::Centre) => {
                let p = ray_plane(ray, self.origin, self.plane_normal)?;
                let start = (self.start_point - self.origin).length();
                let now = (p - self.origin).length();
                let factor = if start > 1e-3 { now / start } else { 1.0 };
                Some(Measure::Scale(Vec3::splat(factor.clamp(0.01, 1000.0))))
            }
            // Rotate has only its rings: no plane square, no centre.
            (TransformMode::Rotate, _) => None,
        }
    }
}

/// The world-space transform a measure describes, about `pivot`.
///
/// Every mode is exact here, whatever the basis: a scale along the gizmo's
/// axes maps each of them to `factor * axis`, which is a full linear map and
/// not a diagonal one — that is why vertex targets take this route.
pub fn measure_matrix(measure: Measure, basis: Basis, pivot: Vec3) -> Mat4 {
    let about_pivot = |m: Mat4| Mat4::from_translation(pivot) * m * Mat4::from_translation(-pivot);
    match measure {
        Measure::Translate(d) => Mat4::from_translation(basis.to_world(d)),
        Measure::Scale(f) => {
            let s = Vec3::new(
                f.x.clamp(0.01, 1000.0),
                f.y.clamp(0.01, 1000.0),
                f.z.clamp(0.01, 1000.0),
            );
            // Each of the gizmo's axes is stretched by its own factor: the map
            // is `B · S · B⁻¹`, which for the world basis is the plain diagonal
            // scale and for a turned one is the same stretch about that axis.
            let b = glam::Mat3::from_cols(basis.x, basis.y, basis.z);
            about_pivot(Mat4::from_mat3(b * glam::Mat3::from_diagonal(s) * b.transpose()))
        }
        Measure::Rotate(axis, a) => {
            about_pivot(Mat4::from_axis_angle(basis.axis(axis), a))
        }
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

/// An in-progress handle drag.
#[derive(Clone, Debug)]
pub struct Drag {
    pub mode: TransformMode,
    pub handle: Handle,
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
    pub hovered: Option<Handle>,
    /// The handle being dragged, held for the whole gesture.
    pub active: Option<Handle>,
    pub drag: Option<Drag>,
    /// This frame's screen layout, refreshed by the input pass and consumed by
    /// the painting pass so both agree on where the handles are.
    pub layout: Option<Layout>,
}

impl Gizmo {
    pub fn is_dragging(&self) -> bool {
        self.drag.is_some()
    }

    /// The handle to highlight: the dragged one while dragging, otherwise
    /// whatever the cursor is over.
    pub fn highlight(&self) -> Option<Handle> {
        self.active.or(self.hovered)
    }

    pub fn begin(&mut self, drag: Drag) {
        self.active = Some(drag.handle);
        self.hovered = None;
        self.drag = Some(drag);
    }

    /// End the drag, returning the undo label for it.
    pub fn finish(&mut self) -> Option<&'static str> {
        self.active = None;
        self.drag.take().map(|d| d.undo_label())
    }

    /// Move the dragged targets by the dragged handle, using the cursor ray.
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
        let measure = crate::snap::snap_measure_in(
            snap,
            drag.anchor.basis,
            &drag.handle.axes(),
            drag.anchor.origin,
            measure,
            vertex,
        );
        match &drag.targets {
            Targets::Nodes(nodes) => {
                for (id, start) in nodes {
                    let Ok(node) = scene.get_mut(*id) else { continue };
                    node.transform =
                        apply_measure(start, drag.anchor.basis, drag.anchor.origin, measure);
                }
            }
            Targets::Vertices(targets) => {
                // The measure is world-space; the vertices are mesh-local, so
                // the transform is carried through each node's world matrix.
                let world_space = measure_matrix(measure, drag.anchor.basis, drag.anchor.origin);
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

    /// A view direction for the tests that do not exercise the centre handle:
    /// looking straight down -Z, so screen space is the XY plane.
    const VIEW: Vec3 = Vec3::NEG_Z;

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
        let at_target = Frame::for_camera(&cam, 500.0, 90.0, cam.target, Basis::WORLD);
        // Halfway from the eye to the target: half the world size, which is the
        // same number of pixels on screen.
        let eye = cam.eye();
        let half_way = eye + (cam.target - eye) * 0.5;
        let nearer = Frame::for_camera(&cam, 500.0, 90.0, half_way, Basis::WORLD);
        assert!(
            (nearer.arm - at_target.arm * 0.5).abs() < 1e-3,
            "{} vs {}",
            nearer.arm,
            at_target.arm
        );
    }

    #[test]
    fn ring_points_lie_in_the_axis_plane() {
        let frame = Frame { origin: Vec3::new(1.0, 2.0, 3.0), arm: 2.0, radius: 2.0, basis: Basis::WORLD };
        for axis in Axis::ALL {
            let ring = frame.ring(axis, 16);
            assert_eq!(ring.len(), 16);
            for p in ring {
                let offset = p - frame.origin;
                assert!(offset.dot(Basis::WORLD.axis(axis)).abs() < 1e-4, "{axis:?} ring is flat");
                assert!((offset.length() - frame.radius).abs() < 1e-4, "{axis:?} ring radius");
            }
        }
    }

    #[test]
    fn tips_sit_along_their_axis() {
        let frame = Frame { origin: Vec3::ZERO, arm: 2.0, radius: 2.0, basis: Basis::WORLD };
        assert_eq!(frame.tip(Axis::Y), Vec3::new(0.0, 2.0, 0.0));
    }

    #[test]
    fn hit_axis_finds_the_arm_under_the_cursor() {
        let frame = Frame { origin: Vec3::ZERO, arm: 1.0, radius: 1.0, basis: Basis::WORLD };
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
        let frame = Frame { origin: Vec3::ZERO, arm: 1.0, radius: 1.0, basis: Basis::WORLD };
        let layout = layout_for(&frame);
        // A cursor a couple of pixels off the Y arm picks Y, not a neighbour,
        // as long as no other arm is closer.
        let y_tip = layout.arms.iter().find(|(a, _)| *a == Axis::Y).map(|(_, p)| *p).unwrap();
        let nudge = y_tip + egui::vec2(3.0, 3.0);
        assert_eq!(layout.hit_axis(nudge, HIT_TOLERANCE), Some(Axis::Y));
    }

    #[test]
    fn hit_ring_finds_a_ring_edge() {
        let frame = Frame { origin: Vec3::ZERO, arm: 1.5, radius: 1.5, basis: Basis::WORLD };
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
    fn plane_and_handle_names_describe_their_axes() {
        for plane in Plane::ALL {
            let (u, v) = plane.axes();
            let normal = plane.normal();
            assert_ne!(u, v);
            assert_ne!(u, normal);
            assert_ne!(v, normal, "{plane:?} is spanned by two axes and misses the third");
            // The normal is the one axis the plane does not span.
            let spanned = [u, v];
            assert!(!spanned.contains(&normal));
            assert_eq!(plane.label().len(), 2);
            // The handle that grabs it reports the plane's own two axes.
            assert_eq!(Handle::Plane(plane).axes(), vec![u, v]);
            assert_eq!(Handle::Axis(normal).axes(), vec![normal]);
            assert!(Handle::Centre.axes().is_empty(), "the centre has no world axis");
            assert_eq!(Handle::Plane(plane).label(), plane.label());
        }
        assert_eq!(Handle::Centre.label(), "screen");
        assert_eq!(Handle::Axis(Axis::Y).label(), "Y");
    }

    #[test]
    fn a_plane_quad_sits_between_its_two_axes_and_off_the_third() {
        let frame = Frame { origin: Vec3::new(1.0, 0.0, -2.0), arm: 2.0, radius: 2.0, basis: Basis::WORLD };
        for plane in Plane::ALL {
            let (u, v) = plane.axes();
            let quad = frame.plane_quad(plane);
            let centre: Vec3 = quad.iter().sum::<Vec3>() / 4.0;
            let offset = centre - frame.origin;
            // Halfway out along both axes of the plane, and nowhere else.
            let expected = (Basis::WORLD.axis(u) + Basis::WORLD.axis(v)) * (frame.arm * PLANE_CENTRE);
            assert!((offset - expected).length() < 1e-5, "{plane:?}: {offset:?}");
            assert!(
                offset.dot(Basis::WORLD.axis(plane.normal())).abs() < 1e-5,
                "{plane:?} stays out of its own normal"
            );

            // Every corner lies in the plane, and the sides are square.
            for corner in quad {
                let off = corner - frame.origin;
                assert!(off.dot(Basis::WORLD.axis(plane.normal())).abs() < 1e-5);
            }
            let side = (quad[1] - quad[0]).length();
            assert!((side - frame.arm * PLANE_HALF * 2.0).abs() < 1e-5, "{plane:?} side {side}");
            assert!((quad[0] - quad[3]).length() < (side + 1e-5), "{plane:?} is a square");
            assert!((quad[2] - quad[1]).length() < (side + 1e-5));
        }
    }

    #[test]
    fn the_cursor_picks_the_handle_it_appears_to_be_over() {
        let frame = Frame { origin: Vec3::ZERO, arm: 1.0, radius: 1.0, basis: Basis::WORLD };
        let layout = layout_for(&frame);
        let tol = HIT_TOLERANCE;

        // The centre square wins over everything, even though the plane quads
        // and the arms all start under it.
        assert_eq!(layout.hit(TransformMode::Move, layout.origin, tol), Some(Handle::Centre));
        assert_eq!(layout.hit(TransformMode::Scale, layout.origin, tol), Some(Handle::Centre));

        // A plane square wins over an arm that runs through it.
        let (plane, quad) = layout.planes.first().expect("a plane square projects");
        let middle = egui::pos2(
            (quad[0].x + quad[1].x + quad[2].x + quad[3].x) / 4.0,
            (quad[0].y + quad[1].y + quad[2].y + quad[3].y) / 4.0,
        );
        assert_eq!(layout.hit(TransformMode::Move, middle, tol), Some(Handle::Plane(*plane)));
        // Dragging the plane square in the Scale tool too.
        assert_eq!(layout.hit(TransformMode::Scale, middle, tol), Some(Handle::Plane(*plane)));

        // Well away from every handle, nothing is grabbed.
        let away = layout.origin + egui::vec2(400.0, 400.0);
        assert_eq!(layout.hit(TransformMode::Move, away, tol), None);
        assert_eq!(layout.hit(TransformMode::Scale, away, tol), None);
    }

    #[test]
    fn a_rotate_gizmo_offers_only_its_rings() {
        let frame = Frame { origin: Vec3::ZERO, arm: 1.0, radius: 1.0, basis: Basis::WORLD };
        let layout = layout_for(&frame);
        // Rotate has no plane square and no centre handle: the origin is free,
        // even though it is the middle of every ring.
        assert_eq!(layout.hit(TransformMode::Rotate, layout.origin, HIT_TOLERANCE), None);
        // A point on a ring is still grabbed.
        let (axis, pts) = layout
            .rings
            .iter()
            .find(|(a, pts)| *a == Axis::Z && pts.len() >= 8)
            .expect("the Z ring projects");
        assert_eq!(
            layout.hit(TransformMode::Rotate, pts[2], HIT_TOLERANCE),
            Some(Handle::Axis(*axis))
        );
    }

    #[test]
    fn a_plane_square_is_grab_able_even_seen_nearly_edge_on() {
        // A square whose corners all project to the same pixel is degenerate,
        // and must not swallow the whole screen.
        let degenerate = [egui::pos2(10.0, 10.0); 4];
        assert!(!quad_contains(&degenerate, egui::pos2(500.0, 500.0), HIT_TOLERANCE));
        // A cursor right on the collapsed square still counts.
        assert!(quad_contains(&degenerate, egui::pos2(10.0, 10.0), HIT_TOLERANCE));

        // A normal square: inside, just outside an edge within tolerance, and
        // far outside.
        let square = [
            egui::pos2(0.0, 0.0),
            egui::pos2(20.0, 0.0),
            egui::pos2(20.0, 20.0),
            egui::pos2(0.0, 20.0),
        ];
        assert!(quad_contains(&square, egui::pos2(10.0, 10.0), HIT_TOLERANCE));
        assert!(quad_contains(&square, egui::pos2(26.0, 10.0), HIT_TOLERANCE), "6px off the edge");
        assert!(!quad_contains(&square, egui::pos2(60.0, 10.0), HIT_TOLERANCE), "far off the edge");
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
        let frame = Frame { origin: Vec3::new(2.0, -1.0, 0.5), arm: 1.0, radius: 1.0, basis: Basis::WORLD };
        let (u, v) = ring_basis(Basis::WORLD.axis(Axis::Z));
        let on_u = frame.origin + u;
        let on_v = frame.origin + v;
        assert!(plane_angle(frame.origin, Basis::WORLD.axis(Axis::Z), on_u).abs() < 1e-5);
        assert!((plane_angle(frame.origin, Basis::WORLD.axis(Axis::Z), on_v) - std::f32::consts::FRAC_PI_2).abs() < 1e-5);
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
        let anchor = Anchor::new(TransformMode::Move, Handle::Axis(Axis::X), Vec3::ZERO, Basis::WORLD, Ray {
            origin: Vec3::new(2.0, 5.0, 0.0),
            direction: -Vec3::Y,
        }, VIEW)
        .unwrap();
        // Move the cursor to X = 4 and the drag has travelled 2 units.
        let m = anchor
            .measure(Ray { origin: Vec3::new(4.0, 5.0, 0.0), direction: -Vec3::Y })
            .unwrap();
        assert_eq!(m, Measure::Translate(Vec3::X * 2.0));

        let start = Transform::from_translation(Vec3::new(1.0, 1.0, 1.0));
        let moved = apply_measure(&start, Basis::WORLD, Vec3::ZERO, m);
        assert!((moved.translation - Vec3::new(3.0, 1.0, 1.0)).length() < 1e-5);
    }

    #[test]
    fn measure_scales_relative_to_the_grab_point() {
        let origin = Vec3::ZERO;
        let anchor = Anchor::new(TransformMode::Scale, Handle::Axis(Axis::Y), origin, Basis::WORLD, Ray {
            origin: Vec3::new(6.0, 1.0, 0.0),
            direction: -Vec3::X,
        }, VIEW)
        .unwrap();
        // Grabbed at Y = 1, now over Y = 2: a 2x scale.
        let m = anchor
            .measure(Ray { origin: Vec3::new(6.0, 2.0, 0.0), direction: -Vec3::X })
            .unwrap();
        assert_eq!(m, Measure::Scale(Vec3::new(1.0, 2.0, 1.0)));

        let start = Transform::identity();
        let scaled = apply_measure(&start, Basis::WORLD, Vec3::ZERO, m);
        assert!(
            (scaled.scale - Vec3::new(1.0, 2.0, 1.0)).length() < 1e-5,
            "only the grabbed axis scales: {:?}",
            scaled.scale
        );

        // Shrinking towards the centre is clamped, never inverted.
        let tiny = apply_measure(&start, Basis::WORLD, Vec3::ZERO, Measure::Scale(Vec3::new(1.0, -3.0, 1.0)));
        assert!(tiny.scale.y > 0.0);
    }

    #[test]
    fn measure_rotates_around_the_axis() {
        // Look straight down -Z at the XY plane, where a Z rotation happens.
        let ray_at = |x: f32, y: f32| Ray { origin: Vec3::new(x, y, 5.0), direction: -Vec3::Z };
        let anchor = Anchor::new(
            TransformMode::Rotate,
            Handle::Axis(Axis::Z),
            Vec3::ZERO, Basis::WORLD,
            ray_at(1.0, 0.0),
            VIEW,
        )
        .unwrap();
        // Sweeping from +X to +Y is a quarter turn about Z.
        let m = anchor.measure(ray_at(0.0, 1.0)).unwrap();
        match m {
            Measure::Rotate(Axis::Z, a) => {
                assert!((a - std::f32::consts::FRAC_PI_2).abs() < 1e-4, "got {a}")
            }
            other => panic!("expected a Z rotation, got {other:?}"),
        }

        let start = Transform::identity();
        let spun = apply_measure(&start, Basis::WORLD, Vec3::ZERO, m);
        let rotated = spun.rotation * Vec3::X;
        assert!((rotated - Vec3::Y).length() < 1e-4, "X rotated into Y, got {rotated:?}");
    }

    #[test]
    fn a_plane_drag_moves_in_the_plane_and_leaves_its_normal_alone() {
        // Looking down -Z at the origin, which is where an XY plane move is at
        // its most visible: the plane faces the camera.
        let ray_at = |x: f32, y: f32| Ray { origin: Vec3::new(x, y, 5.0), direction: -Vec3::Z };
        let anchor = Anchor::new(
            TransformMode::Move,
            Handle::Plane(Plane::XY),
            Vec3::ZERO, Basis::WORLD,
            ray_at(1.0, 1.0),
            VIEW,
        )
        .unwrap();
        let m = anchor.measure(ray_at(3.0, -1.0)).unwrap();
        assert_eq!(m, Measure::Translate(Vec3::new(2.0, -2.0, 0.0)));

        let moved = apply_measure(&Transform::from_translation(Vec3::new(0.0, 0.0, 7.0)), Basis::WORLD, Vec3::ZERO, m);
        assert!((moved.translation - Vec3::new(2.0, -2.0, 7.0)).length() < 1e-5);
    }

    #[test]
    fn a_plane_scale_scales_its_two_axes_and_not_the_third() {
        let anchor = Anchor::new(
            TransformMode::Scale,
            Handle::Plane(Plane::ZX),
            Vec3::ZERO, Basis::WORLD,
            Ray { origin: Vec3::new(1.0, 6.0, 0.0), direction: -Vec3::Y },
            VIEW,
        )
        .unwrap();
        // Grabbed one unit out in the ZX plane, now three: a 3x scale.
        let m = anchor
            .measure(Ray { origin: Vec3::new(3.0, 6.0, 0.0), direction: -Vec3::Y })
            .unwrap();
        assert_eq!(m, Measure::Scale(Vec3::new(3.0, 1.0, 3.0)));

        let scaled = apply_measure(&Transform::identity(), Basis::WORLD, Vec3::ZERO, m);
        assert!((scaled.scale - Vec3::new(3.0, 1.0, 3.0)).length() < 1e-5);
    }

    #[test]
    fn the_centre_handle_works_in_the_plane_facing_the_camera() {
        // Looking down -Z, so the centre handle's plane *is* the XY plane, and
        // its view normal is +Z: the same maths as the XY plane handle.
        let ray_at = |x: f32, y: f32| Ray { origin: Vec3::new(x, y, 5.0), direction: -Vec3::Z };
        let move_anchor = Anchor::new(
            TransformMode::Move,
            Handle::Centre,
            Vec3::ZERO, Basis::WORLD,
            ray_at(1.0, 0.0),
            VIEW,
        )
        .unwrap();
        let m = move_anchor.measure(ray_at(1.0, 4.0)).unwrap();
        assert_eq!(m, Measure::Translate(Vec3::new(0.0, 4.0, 0.0)));

        // A centre scale is uniform, in all three axes at once.
        let scale_anchor = Anchor::new(
            TransformMode::Scale,
            Handle::Centre,
            Vec3::ZERO, Basis::WORLD,
            ray_at(2.0, 0.0),
            VIEW,
        )
        .unwrap();
        let m = scale_anchor.measure(ray_at(4.0, 0.0)).unwrap();
        assert_eq!(m, Measure::Scale(Vec3::splat(2.0)));

        // Rotate has no centre handle: the grab cannot even be anchored.
        assert!(Anchor::new(
            TransformMode::Rotate,
            Handle::Centre,
            Vec3::ZERO, Basis::WORLD,
            ray_at(1.0, 0.0),
            VIEW
        )
        .is_none());
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
        let anchor =
            Anchor::new(TransformMode::Move, Handle::Axis(Axis::X), Vec3::ZERO, Basis::WORLD, ray, VIEW).unwrap();
        let start = scene.get(id).unwrap().transform;
        gizmo.begin(Drag {
            mode: TransformMode::Move,
            handle: Handle::Axis(Axis::X),
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

        let moved = measure_matrix(Measure::Translate(Vec3::X * 3.0), Basis::WORLD, pivot).transform_point3(p);
        assert!((moved - Vec3::new(5.0, 0.0, 0.0)).length() < 1e-5, "got {moved:?}");

        // Scaling about the pivot: x = 1 + (2 - 1) * 2.
        let scaled =
            measure_matrix(Measure::Scale(Vec3::new(2.0, 1.0, 1.0)), Basis::WORLD, pivot).transform_point3(p);
        assert!((scaled - Vec3::new(3.0, 0.0, 0.0)).length() < 1e-5, "got {scaled:?}");

        // A quarter turn about Z around the pivot sends +X to +Y.
        let spun = measure_matrix(
            Measure::Rotate(Axis::Z, std::f32::consts::FRAC_PI_2), Basis::WORLD,
            pivot,
        )
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
        let anchor =
            Anchor::new(TransformMode::Move, Handle::Axis(Axis::X), Vec3::ZERO, Basis::WORLD, ray, VIEW).unwrap();
        let mut gizmo = Gizmo::default();
        gizmo.begin(Drag {
            mode: TransformMode::Move,
            handle: Handle::Axis(Axis::X),
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
            Handle::Axis(Axis::Y),
            Vec3::ZERO, Basis::WORLD,
            Ray { origin: Vec3::new(5.0, 1.0, 0.0), direction: -Vec3::X },
            VIEW,
        )
        .unwrap();
        let mut gizmo = Gizmo::default();
        gizmo.begin(Drag {
            mode: TransformMode::Scale,
            handle: Handle::Axis(Axis::Y),
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

        // The Y arm scales the vertices along Y only, exactly as an axis scale
        // does on a node's own transform: the other axes keep their size.
        let mesh = scene.get(id).unwrap().mesh().unwrap();
        assert!((mesh.positions[2] - Vec3::new(1.0, 2.0, 0.0)).length() < 1e-4, "got {:?}", mesh.positions[2]);
        assert!((mesh.positions[3] - Vec3::new(0.0, 2.0, 0.0)).length() < 1e-4, "got {:?}", mesh.positions[3]);
    }

    #[test]
    fn a_lone_axis_grab_does_not_move_the_other_axes() {
        let start = Transform::from_translation(Vec3::new(4.0, 4.0, 4.0));
        let moved = apply_measure(&start, Basis::WORLD, Vec3::ZERO, Measure::Translate(Vec3::new(0.0, 0.0, -2.0)));
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
    #[test]
    fn the_axes_modes_name_themselves_and_the_world_basis_is_the_default() {
        let names: Vec<&str> = Axes::ALL.iter().map(|a| a.label()).collect();
        assert_eq!(names, vec!["World", "Local", "Screen"]);
        assert_eq!(Axes::default(), Axes::World);
        assert_eq!(Basis::default(), Basis::WORLD);
        for (axis, dir) in [(Axis::X, Vec3::X), (Axis::Y, Vec3::Y), (Axis::Z, Vec3::Z)] {
            assert_eq!(Basis::WORLD.axis(axis), dir);
        }
        // Components and back is the identity for an orthonormal basis, and it
        // is exact on the world's own axes.
        let v = Vec3::new(1.0, -2.0, 3.0);
        assert_eq!(Basis::WORLD.components(v), v);
        assert_eq!(Basis::WORLD.to_world(Basis::WORLD.components(v)), v);
    }

    #[test]
    fn a_local_basis_takes_a_turned_transform_its_own_axes() {
        // A quarter turn about Y sends X to -Z and Z to +X.
        let basis = Basis::of_matrix(Mat4::from_rotation_y(std::f32::consts::FRAC_PI_2));
        assert!((basis.axis(Axis::X) - -Vec3::Z).length() < 1e-5, "{:?}", basis.x);
        assert!((basis.axis(Axis::Y) - Vec3::Y).length() < 1e-5, "{:?}", basis.y);
        assert!((basis.axis(Axis::Z) - Vec3::X).length() < 1e-5, "{:?}", basis.z);
        // A scaled node's axes stay unit: the gizmo points the way the object
        // does, not as far.
        let basis = Basis::of_matrix(Mat4::from_scale(Vec3::new(5.0, 0.5, 2.0)));
        for axis in Axis::ALL {
            assert!((basis.axis(axis).length() - 1.0).abs() < 1e-5, "{axis:?}");
        }
        assert!((basis.axis(Axis::X) - Vec3::X).length() < 1e-5);
        // An untransformed node is the world.
        assert_eq!(Basis::of_matrix(Mat4::IDENTITY), Basis::WORLD);
    }

    #[test]
    fn a_screen_basis_is_right_up_and_out_of_the_screen() {
        // Looking down -Z: right is +X, up is +Y, and the third axis points out
        // of the screen, back at the viewer.
        let camera = Camera { yaw: std::f32::consts::FRAC_PI_2, pitch: 0.0, ..Camera::default() };
        let basis = Basis::of_camera(&camera);
        assert!((basis.axis(Axis::X) - Vec3::X).length() < 1e-5, "{:?}", basis.x);
        assert!((basis.axis(Axis::Y) - Vec3::Y).length() < 1e-5, "{:?}", basis.y);
        assert!((basis.axis(Axis::Z) - Vec3::Z).length() < 1e-5, "{:?}", basis.z);
        assert!(basis.z.dot(camera.forward()) < -0.99, "out of the screen");

        // Whatever the orbit, the three stay orthonormal and right-handed, and
        // the third is still the way out of the screen.
        let camera = Camera { yaw: 0.7, pitch: -0.4, ..Camera::default() };
        let basis = Basis::of_camera(&camera);
        for (a, b) in [
            (basis.x, basis.y),
            (basis.y, basis.z),
            (basis.z, basis.x),
        ] {
            assert!(a.dot(b).abs() < 1e-5, "square to each other");
            assert!((a.length() - 1.0).abs() < 1e-5, "unit");
        }
        assert!((basis.x.cross(basis.y) - basis.z).length() < 1e-5, "right-handed");
        assert!(basis.z.dot(camera.forward()) < -0.99);
    }

    #[test]
    fn ring_basis_spans_the_plane_around_any_direction() {
        // The world axes keep the companions they have always had, in the same
        // order, so the angle a ring reports has not moved.
        for (dir, expected) in [
            (Vec3::X, (Vec3::Y, Vec3::Z)),
            (Vec3::Y, (Vec3::Z, Vec3::X)),
            (Vec3::Z, (Vec3::X, Vec3::Y)),
        ] {
            let (u, v) = ring_basis(dir);
            assert!((u - expected.0).length() < 1e-5 && (v - expected.1).length() < 1e-5);
        }
        // Any other direction: the two companions stay unit, square to the axis
        // and to each other, and right-handed about it.
        for dir in [Vec3::new(1.0, 2.0, 3.0), Vec3::new(-1.0, 0.3, 0.0), -Vec3::Y] {
            let n = dir.normalize();
            let (u, v) = ring_basis(dir);
            assert!(u.dot(n).abs() < 1e-5 && v.dot(n).abs() < 1e-5, "square to the axis");
            assert!(u.dot(v).abs() < 1e-5, "square to each other");
            assert!((u.length() - 1.0).abs() < 1e-5 && (v.length() - 1.0).abs() < 1e-5);
            assert!((u.cross(v) - n).length() < 1e-4, "right-handed about {n:?}");
        }
    }

    #[test]
    fn a_move_along_a_turned_axis_follows_it_and_leaves_the_rest_exactly_zero() {
        // A basis whose X handle points along world +Z: a rigid quarter turn
        // about Y the other way.
        let basis = Basis::of_matrix(Mat4::from_rotation_y(-std::f32::consts::FRAC_PI_2));
        assert!((basis.axis(Axis::X) - Vec3::Z).length() < 1e-5);
        // A ray that slides along world Z, so it meets the turned X handle,
        // which is the world Z axis.
        let ray = |z: f32| Ray { origin: Vec3::new(2.0, 0.0, z), direction: -Vec3::X };
        let anchor = Anchor::new(
            TransformMode::Move,
            Handle::Axis(Axis::X),
            Vec3::ZERO,
            basis,
            ray(5.0),
            VIEW,
        )
        .expect("the ray meets the axis");
        let m = anchor.measure(ray(8.0)).expect("a measure each frame");
        assert_eq!(
            m,
            Measure::Translate(Vec3::new(3.0, 0.0, 0.0)),
            "three along the gizmo's own X, and nothing else at all"
        );

        // The node follows that turned axis — world +Z, not the world axis its
        // handle is named after.
        let start = Transform::from_translation(Vec3::new(1.0, 1.0, 1.0));
        let moved = apply_measure(&start, basis, Vec3::ZERO, m);
        assert!(
            (moved.translation - Vec3::new(1.0, 1.0, 4.0)).length() < 1e-5,
            "got {:?}",
            moved.translation
        );
    }

    #[test]
    fn a_plane_handle_in_a_turned_frame_stays_in_its_own_plane() {
        let basis = Basis::of_matrix(Mat4::from_rotation_y(-std::f32::consts::FRAC_PI_2));
        // The XY square spans the gizmo's X (world Z) and Y (world Y), and its
        // normal is the gizmo's Z (world -X): its plane is x = 0.
        let ray_to = |p: Vec3| Ray {
            origin: p + Vec3::X * 10.0,
            direction: -Vec3::X,
        };
        let anchor = Anchor::new(
            TransformMode::Move,
            Handle::Plane(Plane::XY),
            Vec3::ZERO,
            basis,
            ray_to(Vec3::new(0.0, 1.0, 4.0)),
            VIEW,
        )
        .expect("the ray meets the plane");
        let m = anchor
            .measure(ray_to(Vec3::new(0.0, 2.0, 5.0)))
            .expect("a measure each frame");
        let Measure::Translate(d) = m else { panic!("a move") };
        assert_eq!(d.z, 0.0, "the normal's own component is left exactly zero");
        assert!((d.x - 1.0).abs() < 1e-4, "one along the turned X: {d:?}");
        assert!((d.y - 1.0).abs() < 1e-4, "one along Y: {d:?}");
        // And the world translation lies in the plane: nothing along its normal.
        let world = basis.to_world(d);
        assert!(world.dot(basis.axis(Axis::Z)).abs() < 1e-4, "{world:?}");
        assert!((world - Vec3::new(0.0, 1.0, 1.0)).length() < 1e-3, "{world:?}");
    }

    #[test]
    fn a_scale_in_a_turned_basis_stretches_along_its_own_axes() {
        let basis = Basis::of_matrix(Mat4::from_rotation_y(-std::f32::consts::FRAC_PI_2));
        // Vertex targets take the whole matrix, so this is exact however the
        // basis is turned: a point one along the gizmo's X (world Z) doubles
        // along that axis...
        let m = measure_matrix(Measure::Scale(Vec3::new(2.0, 1.0, 1.0)), basis, Vec3::ZERO);
        let stretched = m.transform_point3(Vec3::Z);
        assert!((stretched - Vec3::new(0.0, 0.0, 2.0)).length() < 1e-5, "got {stretched:?}");
        // ...and one along a gizmo axis that does not scale stays put.
        let same = m.transform_point3(Vec3::X);
        assert!((same - Vec3::X).length() < 1e-5, "got {same:?}");
        // About a pivot, so a scale does not drag the geometry to the origin.
        let about = measure_matrix(
            Measure::Scale(Vec3::new(2.0, 1.0, 1.0)),
            basis,
            Vec3::new(0.0, 0.0, 1.0),
        );
        let p = about.transform_point3(Vec3::new(0.0, 0.0, 2.0));
        assert!((p - Vec3::new(0.0, 0.0, 3.0)).length() < 1e-5, "got {p:?}");
    }

    #[test]
    fn a_rotation_in_a_turned_basis_turns_about_the_turned_axis() {
        let basis = Basis::of_matrix(Mat4::from_rotation_y(-std::f32::consts::FRAC_PI_2));
        // The gizmo's Z handle is world -X here, so a quarter turn about it is
        // a quarter turn about world -X.
        let turned = apply_measure(
            &Transform::identity(),
            basis, Vec3::ZERO,
            Measure::Rotate(Axis::Z, std::f32::consts::FRAC_PI_2),
        );
        let (axis, angle) = turned.rotation.to_axis_angle();
        assert!((axis - -Vec3::X).length() < 1e-4, "got {axis:?}");
        assert!((angle - std::f32::consts::FRAC_PI_2).abs() < 1e-4, "got {angle}");

        // Vertices take the same turn, from the same axis, about the pivot.
        let m = measure_matrix(
            Measure::Rotate(Axis::Z, std::f32::consts::FRAC_PI_2),
            basis,
            Vec3::new(0.0, 1.0, 0.0),
        );
        // A point one above the pivot swings a quarter turn about world -X,
        // which is towards -Z.
        let p = m.transform_point3(Vec3::new(0.0, 2.0, 0.0));
        assert!((p - Vec3::new(0.0, 1.0, -1.0)).length() < 1e-4, "got {p:?}");
        // And the pivot itself is the one point that does not move.
        let still = m.transform_point3(Vec3::new(0.0, 1.0, 0.0));
        assert!((still - Vec3::new(0.0, 1.0, 0.0)).length() < 1e-5);
    }

    #[test]
    fn a_view_sizes_the_handles_by_the_depth_its_projection_uses() {
        let origin = Vec3::ZERO;
        let size = [800.0, 800.0];
        // How many pixels an arm of this length comes out as on screen.
        let arm_pixels = |camera: &Camera, frame: &Frame| {
            let vp = camera.view_proj(size[0] / size[1]);
            let (ax, ay, _) = viewport::project(vp, frame.origin, size).expect("the origin projects");
            let tip = frame.origin + frame.dir(Axis::X) * frame.arm;
            let (bx, by, _) = viewport::project(vp, tip, size).expect("the tip projects");
            ((bx - ax).powi(2) + (by - ay).powi(2)).sqrt()
        };

        // Looking straight down -Z, the arm is across the screen, so the pixel
        // length is the arm's own size on screen.
        let mut camera = Camera { target: origin, yaw: std::f32::consts::FRAC_PI_2, pitch: 0.0, ..Camera::default() };
        let frame = Frame::for_camera(&camera, size[1], HANDLE_PIXELS, origin, Basis::WORLD);
        assert!(
            (arm_pixels(&camera, &frame) - HANDLE_PIXELS).abs() < 1.0,
            "in perspective the arm is {} pixels",
            arm_pixels(&camera, &frame)
        );

        // The same from an orthographic front view: its eye stands a fixed way
        // back, so sizing by the eye distance would draw a handle thousands of
        // pixels long. The projection's own depth is what counts.
        camera.set_view(viewport::ViewKind::Front);
        let frame = Frame::for_camera(&camera, size[1], HANDLE_PIXELS, origin, Basis::WORLD);
        assert!(
            (arm_pixels(&camera, &frame) - HANDLE_PIXELS).abs() < 1.0,
            "and in an axis view too: {} pixels",
            arm_pixels(&camera, &frame)
        );
    }

}
