//! Snapping: where a transform lands.
//!
//! Like `gizmo.rs`, the maths is pure and unit-tested here while the app owns
//! the state and the wiring. Snapping applies to the gizmo's axis-constrained
//! drags — the free-form view-plane drag is unsnapped.
//!
//! `Grid` mode snaps the *destination* of the grabbed point, not the distance
//! travelled, so a drag comes to rest on a grid value wherever it started;
//! `Vertex` mode lands the grabbed point on the vertex under the cursor.

use crate::gizmo::{Axis, Measure};
use glam::Vec3;

/// The default grid step, in world units.
pub const DEFAULT_GRID_STEP: f32 = 0.25;
/// The default rotation step, in degrees.
pub const DEFAULT_ANGLE_STEP: f32 = 15.0;
/// The default scale factor step.
pub const DEFAULT_FACTOR_STEP: f32 = 0.1;
/// How close (in screen pixels) the cursor must come to a vertex for a vertex
/// snap to take it.
pub const SNAP_RADIUS_PIXELS: f32 = 14.0;

/// What a drag snaps its measure to.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum SnapMode {
    /// Free: the cursor decides.
    #[default]
    Off,
    /// Multiples of the step: the dragged point lands on grid values, a
    /// rotation on angle steps and a scale on factor steps.
    Grid,
    /// The nearest vertex under the cursor (Move only).
    Vertex,
}

impl SnapMode {
    pub fn label(self) -> &'static str {
        match self {
            SnapMode::Off => "Off",
            SnapMode::Grid => "Grid",
            SnapMode::Vertex => "Vertex",
        }
    }

    /// The next mode in the toolbar's cycle.
    pub fn next(self) -> Self {
        match self {
            SnapMode::Off => SnapMode::Grid,
            SnapMode::Grid => SnapMode::Vertex,
            SnapMode::Vertex => SnapMode::Off,
        }
    }
}

/// The snapping settings: a mode plus the step each transform snaps to.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Snap {
    pub mode: SnapMode,
    /// World units between grid values (Move).
    pub grid: f32,
    /// Radians per rotation step (Rotate).
    pub angle: f32,
    /// Scale factor step (Scale).
    pub factor: f32,
}

impl Default for Snap {
    /// Snapping off, with a quarter-unit grid, 15° rotations and tenth-step
    /// scales ready for when it is switched on.
    fn default() -> Self {
        Self {
            mode: SnapMode::Off,
            grid: DEFAULT_GRID_STEP,
            angle: DEFAULT_ANGLE_STEP.to_radians(),
            factor: DEFAULT_FACTOR_STEP,
        }
    }
}

impl Snap {
    /// Whether anything is snapped at all.
    pub fn is_on(self) -> bool {
        self.mode != SnapMode::Off
    }

    /// The same settings with snapping inverted, for a held modifier: a mode
    /// that is on turns off and `Off` becomes `Grid`. The steps are kept, so
    /// releasing the modifier restores exactly what was set.
    pub fn inverted(self) -> Self {
        let mode = match self.mode {
            SnapMode::Off => SnapMode::Grid,
            SnapMode::Grid | SnapMode::Vertex => SnapMode::Off,
        };
        Self { mode, ..self }
    }
}

/// The multiple of `step` nearest to `value`, or `value` untouched when `step`
/// is not a positive number.
pub fn snap_to_step(value: f32, step: f32) -> f32 {
    if !(step > 0.0) || !value.is_finite() {
        return value;
    }
    (value / step).round() * step
}

/// Snap `measure` for a drag whose grabbed point is `grab` on `axis`.
///
/// `vertex` is the world position of the vertex under the cursor, when the app
/// found one within its radius; that is what [`SnapMode::Vertex`] lands on.
/// Only the dragged axis moves, so only its component of a target is used.
pub fn snap_measure(
    snap: Snap,
    axis: Axis,
    grab: Vec3,
    measure: Measure,
    vertex: Option<Vec3>,
) -> Measure {
    let dir = axis.dir();
    match (snap.mode, measure) {
        (SnapMode::Off, m) => m,
        (SnapMode::Grid, Measure::Translate(d)) => {
            let along = grab.dot(dir);
            Measure::Translate(snap_to_step(along + d, snap.grid) - along)
        }
        (SnapMode::Vertex, Measure::Translate(d)) => match vertex {
            Some(v) => Measure::Translate((v - grab).dot(dir)),
            None => Measure::Translate(d),
        },
        (SnapMode::Grid, Measure::Rotate(a)) => Measure::Rotate(snap_to_step(a, snap.angle)),
        (SnapMode::Grid, Measure::Scale(f)) => {
            // Never snap a scale away entirely: the step is the floor.
            let step = snap.factor.max(1e-3);
            Measure::Scale(snap_to_step(f, step).max(step))
        }
        // Vertex snapping has nothing to say about rotating or scaling.
        (SnapMode::Vertex, m) => m,
    }
}

/// A measure as the status bar reads it out: a signed distance, a multiplier,
/// or degrees.
pub fn format_measure(measure: Measure) -> String {
    match measure {
        Measure::Translate(d) => format!("Δ {d:.3}"),
        Measure::Scale(f) => format!("× {f:.3}"),
        Measure::Rotate(a) => format!("{:.1}°", a.to_degrees()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    #[test]
    fn snap_to_step_finds_the_nearest_multiple() {
        assert_eq!(snap_to_step(1.27, 0.25), 1.25);
        assert_eq!(snap_to_step(1.13, 0.25), 1.25);
        assert_eq!(snap_to_step(0.0, 0.25), 0.0);
        assert_eq!(snap_to_step(-0.37, 0.25), -0.25, "negative values too");
        assert_eq!(snap_to_step(0.4, 0.5), 0.5);
        // A step that cannot snap leaves the value alone.
        assert_eq!(snap_to_step(1.27, 0.0), 1.27);
        assert_eq!(snap_to_step(1.27, -1.0), 1.27);
        assert!(snap_to_step(f32::NAN, 0.25).is_nan(), "a NaN stays a NaN");
    }

    #[test]
    fn off_snapping_changes_nothing() {
        let snap = Snap { mode: SnapMode::Off, ..Snap::default() };
        let grab = Vec3::new(0.37, 0.0, 0.0);
        let m = Measure::Translate(0.9);
        assert_eq!(snap_measure(snap, Axis::X, grab, m, None), m);
        assert!(!snap.is_on());
    }

    #[test]
    fn grid_move_lands_on_a_grid_value_wherever_it_started() {
        let snap = Snap { mode: SnapMode::Grid, grid: 0.25, ..Snap::default() };
        let grab = Vec3::new(0.37, 2.0, 0.0);
        let Measure::Translate(d) = snap_measure(snap, Axis::X, grab, Measure::Translate(0.9), None)
        else {
            panic!("still a translation")
        };
        let landed = grab.x + d;
        assert!((landed - 1.25).abs() < 1e-6, "landed on {landed}");

        // Idempotent: feeding the snapped delta back changes nothing.
        let Measure::Translate(again) = snap_measure(snap, Axis::X, grab, Measure::Translate(d), None)
        else {
            panic!("still a translation")
        };
        assert!((again - d).abs() < 1e-6, "{again} vs {d}");

        // The dragged axis is the one that snaps: a Y drag lands on a Y grid
        // value too.
        let Measure::Translate(dy) = snap_measure(snap, Axis::Y, grab, Measure::Translate(0.3), None)
        else {
            panic!("still a translation")
        };
        assert!((grab.y + dy - 2.25).abs() < 1e-6, "y landed on {}", grab.y + dy);
    }

    #[test]
    fn vertex_move_lands_the_grabbed_point_on_the_vertex() {
        let snap = Snap { mode: SnapMode::Vertex, ..Snap::default() };
        let grab = Vec3::new(-1.0, 0.5, 0.0);
        let vertex = Vec3::new(2.3, 9.0, 4.0);
        let Measure::Translate(d) =
            snap_measure(snap, Axis::X, grab, Measure::Translate(0.2), Some(vertex))
        else {
            panic!("still a translation")
        };
        assert!((grab.x + d - 2.3).abs() < 1e-6, "landed on {}", grab.x + d);

        // No vertex under the cursor: the drag stays free.
        let free = snap_measure(snap, Axis::X, grab, Measure::Translate(0.2), None);
        assert_eq!(free, Measure::Translate(0.2));
    }

    #[test]
    fn vertex_snapping_leaves_rotate_and_scale_alone() {
        let snap = Snap { mode: SnapMode::Vertex, ..Snap::default() };
        assert_eq!(
            snap_measure(snap, Axis::Y, Vec3::ZERO, Measure::Rotate(0.4), Some(Vec3::X)),
            Measure::Rotate(0.4)
        );
        assert_eq!(
            snap_measure(snap, Axis::Y, Vec3::ZERO, Measure::Scale(1.23), Some(Vec3::X)),
            Measure::Scale(1.23)
        );
    }

    #[test]
    fn grid_snaps_rotation_to_the_angle_step() {
        let snap = Snap { mode: SnapMode::Grid, angle: (15.0f32).to_radians(), ..Snap::default() };
        let Measure::Rotate(a) =
            snap_measure(snap, Axis::Z, Vec3::ZERO, Measure::Rotate(0.4), None)
        else {
            panic!("still a rotation")
        };
        let step = 15.0f32.to_radians();
        assert!((a - 2.0 * step).abs() < 1e-5, "{a} is not two steps");

        // Exactly PI/2 snaps to itself with a 45 degree step.
        let snap = Snap { mode: SnapMode::Grid, angle: PI / 4.0, ..Snap::default() };
        let Measure::Rotate(a) =
            snap_measure(snap, Axis::Z, Vec3::ZERO, Measure::Rotate(PI / 2.0 + 0.01), None)
        else {
            panic!("still a rotation")
        };
        assert!((a - PI / 2.0).abs() < 1e-5, "{a}");
    }

    #[test]
    fn grid_snaps_scale_to_the_factor_step_without_collapsing() {
        let snap = Snap { mode: SnapMode::Grid, factor: 0.1, ..Snap::default() };
        let Measure::Scale(f) = snap_measure(snap, Axis::X, Vec3::ZERO, Measure::Scale(1.23), None)
        else {
            panic!("still a scale")
        };
        assert!((f - 1.2).abs() < 1e-6, "{f}");

        // A tiny or negative factor lands on the step rather than on zero.
        let Measure::Scale(f) = snap_measure(snap, Axis::X, Vec3::ZERO, Measure::Scale(0.001), None)
        else {
            panic!("still a scale")
        };
        assert!((f - 0.1).abs() < 1e-6, "{f}");
    }

    #[test]
    fn inverting_swaps_off_for_grid_and_keeps_the_steps() {
        let off = Snap::default();
        let on = off.inverted();
        assert_eq!(on.mode, SnapMode::Grid);
        assert_eq!(on.grid, off.grid);
        assert_eq!(on.angle, off.angle);
        assert_eq!(on.factor, off.factor);
        assert_eq!(on.inverted(), off, "inverting twice is the identity");

        let vertex = Snap { mode: SnapMode::Vertex, grid: 0.5, ..Snap::default() };
        assert_eq!(vertex.inverted().mode, SnapMode::Off);
        assert_eq!(vertex.inverted().grid, 0.5, "the step survives");
    }

    #[test]
    fn measures_read_out_the_way_the_status_bar_shows_them() {
        assert_eq!(format_measure(Measure::Translate(1.25)), "Δ 1.250");
        assert_eq!(format_measure(Measure::Scale(2.0)), "× 2.000");
        assert_eq!(format_measure(Measure::Rotate(90.0f32.to_radians())), "90.0°");
    }

    #[test]
    fn the_mode_cycle_is_off_grid_vertex() {
        assert_eq!(SnapMode::Off.next(), SnapMode::Grid);
        assert_eq!(SnapMode::Grid.next(), SnapMode::Vertex);
        assert_eq!(SnapMode::Vertex.next(), SnapMode::Off);
        assert_eq!(SnapMode::Grid.label(), "Grid");
    }
}
