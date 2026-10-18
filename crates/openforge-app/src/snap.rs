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

/// Snap `measure` for a drag whose grabbed point is `grab`.
///
/// `axes` are the world axes the dragged handle can move along — one for an
/// axis arm, two for a plane square, none for the centre handle, whose plane
/// faces the camera and so lines up with no world axis. Cutting the measure
/// down to those axes is what keeps a snapped drag from nudging a component the
/// handle was never allowed to touch.
///
/// `vertex` is the world position of the vertex under the cursor, when the app
/// found one within its radius; that is what [`SnapMode::Vertex`] lands on.
pub fn snap_measure(
    snap: Snap,
    axes: &[Axis],
    grab: Vec3,
    measure: Measure,
    vertex: Option<Vec3>,
) -> Measure {
    match (snap.mode, measure) {
        (SnapMode::Off, m) => m,
        // With no world axis to snap to (the centre handle) the point comes
        // back exactly as it went in, so the drag stays free.
        (SnapMode::Grid, Measure::Translate(d)) => {
            Measure::Translate(snap_point(grab + d, axes, snap.grid) - grab)
        }
        (SnapMode::Vertex, Measure::Translate(d)) if !axes.is_empty() => match vertex {
            Some(v) => {
                // The grabbed point lands on the vertex, but only along the
                // axes this handle moves: the rest of the vertex is dropped.
                let mut delta = v - grab;
                for axis in Axis::ALL {
                    if !axes.contains(&axis) {
                        let dir = axis.dir();
                        delta -= dir * delta.dot(dir);
                    }
                }
                Measure::Translate(delta)
            }
            None => Measure::Translate(d),
        },
        (SnapMode::Grid, Measure::Rotate(axis, a)) => {
            Measure::Rotate(axis, snap_to_step(a, snap.angle))
        }
        (SnapMode::Grid, Measure::Scale(f)) => {
            // Never snap a scale away entirely: the step is the floor. Only the
            // axes the handle scales are snapped, so an axis it leaves alone
            // stays at exactly 1 rather than drifting to the step's multiple.
            let step = snap.factor.max(1e-3);
            let mut out = f;
            for &axis in axes {
                let dir = axis.dir();
                let v = f.dot(dir);
                out += dir * (snap_to_step(v, step).max(step) - v);
            }
            Measure::Scale(out)
        }
        // Vertex snapping has nothing to say about rotating or scaling.
        (SnapMode::Vertex, m) => m,
    }
}

/// `p` with the components on `axes` snapped to the nearest multiple of
/// `step`, and every other component left exactly as it was.
fn snap_point(p: Vec3, axes: &[Axis], step: f32) -> Vec3 {
    let mut out = p;
    for &axis in axes {
        let dir = axis.dir();
        let v = out.dot(dir);
        out += dir * (snap_to_step(v, step) - v);
    }
    out
}

/// Whether a factor leaves its axis alone.
fn is_unity(f: f32) -> bool {
    (f - 1.0).abs() < 1e-4
}

/// A measure as the status bar reads it out: a distance, a multiplier, or
/// degrees.
///
/// A handle that moves a single axis or scales a single axis prints just that
/// number with its axis named; a plane handle prints the pair; the centre
/// handle, which moves in no world axis, prints all three components.
pub fn format_measure(measure: Measure) -> String {
    match measure {
        Measure::Translate(d) => {
            // Only the axes the drag is actually moving get named, so a plane
            // drag does not read out the axis it is not allowed to touch.
            let moved: Vec<String> = [(Axis::X, d.x), (Axis::Y, d.y), (Axis::Z, d.z)]
                .into_iter()
                .filter(|(_, v)| *v != 0.0)
                .map(|(axis, v)| format!("{} {:.3}", axis.label(), v))
                .collect();
            match moved.len() {
                0 => "Δ 0.000".to_string(),
                1 | 2 => format!("Δ {}", moved.join(" ")),
                // Three axes at once is what the centre handle does: in the
                // screen plane every world axis can move, so the vector itself
                // is the clearest thing to show.
                _ => format!("Δ ({:.3}, {:.3}, {:.3})", d.x, d.y, d.z),
            }
        }
        Measure::Scale(f) => {
            if (f.x - f.y).abs() < 1e-4 && (f.y - f.z).abs() < 1e-4 {
                format!("× {:.3}", f.x)
            } else if is_unity(f.y) && is_unity(f.z) {
                format!("× X {:.3}", f.x)
            } else if is_unity(f.x) && is_unity(f.z) {
                format!("× Y {:.3}", f.y)
            } else if is_unity(f.x) && is_unity(f.y) {
                format!("× Z {:.3}", f.z)
            } else if is_unity(f.x) {
                format!("× YZ {:.3}", f.y)
            } else if is_unity(f.y) {
                format!("× ZX {:.3}", f.z)
            } else if is_unity(f.z) {
                format!("× XY {:.3}", f.x)
            } else {
                format!("× ({:.3}, {:.3}, {:.3})", f.x, f.y, f.z)
            }
        }
        Measure::Rotate(_, a) => format!("{:.1}°", a.to_degrees()),
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
        let m = Measure::Translate(Vec3::new(0.9, 0.0, 0.0));
        assert_eq!(snap_measure(snap, &[Axis::X], grab, m, None), m);
        assert!(!snap.is_on());
    }

    #[test]
    fn grid_move_lands_on_a_grid_value_wherever_it_started() {
        let snap = Snap { mode: SnapMode::Grid, grid: 0.25, ..Snap::default() };
        let grab = Vec3::new(0.37, 2.0, 0.0);
        let delta = |d: f32| Measure::Translate(Vec3::X * d);
        let Measure::Translate(d) =
            snap_measure(snap, &[Axis::X], grab, delta(0.9), None)
        else {
            panic!("still a translation")
        };
        let landed = grab.x + d.x;
        assert!((landed - 1.25).abs() < 1e-6, "landed on {landed}");
        assert_eq!(d.y, 0.0, "the other axes are not touched");

        // Idempotent: feeding the snapped delta back changes nothing.
        let Measure::Translate(again) =
            snap_measure(snap, &[Axis::X], grab, Measure::Translate(d), None)
        else {
            panic!("still a translation")
        };
        assert!((again - d).length() < 1e-6, "{again} vs {d}");

        // The dragged axis is the one that snaps: a Y drag lands on a Y grid
        // value too.
        let Measure::Translate(dy) =
            snap_measure(snap, &[Axis::Y], grab, Measure::Translate(Vec3::Y * 0.3), None)
        else {
            panic!("still a translation")
        };
        assert!((grab.y + dy.y - 2.25).abs() < 1e-6, "y landed on {}", grab.y + dy.y);
    }

    #[test]
    fn a_plane_move_snaps_both_of_its_axes_and_leaves_the_third_alone() {
        let snap = Snap { mode: SnapMode::Grid, grid: 0.25, ..Snap::default() };
        let grab = Vec3::new(0.37, 0.6, 5.0);
        // An XY plane drag: the delta has no Z component at all.
        let measure = Measure::Translate(Vec3::new(0.9, 0.7, 0.0));
        let Measure::Translate(d) = snap_measure(snap, &[Axis::X, Axis::Y], grab, measure, None)
        else {
            panic!("still a translation")
        };
        assert!((grab.x + d.x - 1.25).abs() < 1e-6, "x landed on {}", grab.x + d.x);
        assert!((grab.y + d.y - 1.25).abs() < 1e-6, "y landed on {}", grab.y + d.y);
        assert_eq!(d.z, 0.0, "the plane's normal axis is left alone");
    }

    #[test]
    fn a_centre_move_has_no_world_axis_to_snap_to() {
        // The centre handle works in the plane facing the camera, which lines
        // up with no world axis, so nothing is snapped in any mode.
        let grab = Vec3::new(0.37, 0.6, -0.1);
        let measure = Measure::Translate(Vec3::new(0.13, -0.27, 0.41));
        for mode in [SnapMode::Grid, SnapMode::Vertex] {
            let snap = Snap { mode, ..Snap::default() };
            assert_eq!(
                snap_measure(snap, &[], grab, measure, Some(Vec3::new(9.0, 9.0, 9.0))),
                measure,
                "{mode:?} leaves a centre drag free"
            );
        }
    }

    #[test]
    fn vertex_move_lands_the_grabbed_point_on_the_vertex() {
        let snap = Snap { mode: SnapMode::Vertex, ..Snap::default() };
        let grab = Vec3::new(-1.0, 0.5, 0.0);
        let vertex = Vec3::new(2.3, 9.0, 4.0);
        let Measure::Translate(d) =
            snap_measure(snap, &[Axis::X], grab, Measure::Translate(Vec3::X * 0.2), Some(vertex))
        else {
            panic!("still a translation")
        };
        assert!((grab.x + d.x - 2.3).abs() < 1e-6, "landed on {}", grab.x + d.x);
        assert_eq!(d.y, 0.0, "only the dragged axis follows the vertex");
        assert_eq!(d.z, 0.0);

        // No vertex under the cursor: the drag stays free.
        let free = snap_measure(snap, &[Axis::X], grab, Measure::Translate(Vec3::new(0.2, 0.3, 0.0)), None);
        assert_eq!(free, Measure::Translate(Vec3::new(0.2, 0.3, 0.0)));
    }

    #[test]
    fn vertex_snapping_leaves_rotate_and_scale_alone() {
        let snap = Snap { mode: SnapMode::Vertex, ..Snap::default() };
        assert_eq!(
            snap_measure(snap, &[Axis::Y], Vec3::ZERO, Measure::Rotate(Axis::Y, 0.4), Some(Vec3::X)),
            Measure::Rotate(Axis::Y, 0.4)
        );
        assert_eq!(
            snap_measure(
                snap,
                &[Axis::Y],
                Vec3::ZERO,
                Measure::Scale(Vec3::new(1.0, 1.23, 1.0)),
                Some(Vec3::X)
            ),
            Measure::Scale(Vec3::new(1.0, 1.23, 1.0))
        );
    }

    #[test]
    fn grid_snaps_rotation_to_the_angle_step() {
        let snap = Snap { mode: SnapMode::Grid, angle: (15.0f32).to_radians(), ..Snap::default() };
        let Measure::Rotate(axis, a) =
            snap_measure(snap, &[Axis::Z], Vec3::ZERO, Measure::Rotate(Axis::Z, 0.4), None)
        else {
            panic!("still a rotation")
        };
        assert_eq!(axis, Axis::Z, "the rotation keeps its axis");
        let step = 15.0f32.to_radians();
        assert!((a - 2.0 * step).abs() < 1e-5, "{a} is not two steps");

        // Exactly PI/2 snaps to itself with a 45 degree step.
        let snap = Snap { mode: SnapMode::Grid, angle: PI / 4.0, ..Snap::default() };
        let Measure::Rotate(_, a) =
            snap_measure(snap, &[Axis::Z], Vec3::ZERO, Measure::Rotate(Axis::Z, PI / 2.0 + 0.01), None)
        else {
            panic!("still a rotation")
        };
        assert!((a - PI / 2.0).abs() < 1e-5, "{a}");
    }

    #[test]
    fn grid_snaps_scale_to_the_factor_step_without_collapsing() {
        let snap = Snap { mode: SnapMode::Grid, factor: 0.1, ..Snap::default() };
        let x_only = |f: f32| Measure::Scale(Vec3::new(f, 1.0, 1.0));
        let Measure::Scale(f) = snap_measure(snap, &[Axis::X], Vec3::ZERO, x_only(1.23), None) else {
            panic!("still a scale")
        };
        assert!((f.x - 1.2).abs() < 1e-6, "{}", f.x);
        assert_eq!(f.y, 1.0, "an axis the handle does not scale stays at 1");

        // A tiny or negative factor lands on the step rather than on zero.
        let Measure::Scale(f) = snap_measure(snap, &[Axis::X], Vec3::ZERO, x_only(0.001), None)
        else {
            panic!("still a scale")
        };
        assert!((f.x - 0.1).abs() < 1e-6, "{}", f.x);

        // A plane handle scales its two axes by the same factor.
        let Measure::Scale(f) = snap_measure(
            snap,
            &[Axis::X, Axis::Y],
            Vec3::ZERO,
            Measure::Scale(Vec3::new(1.23, 1.23, 1.0)),
            None,
        ) else {
            panic!("still a scale")
        };
        assert!((f.x - 1.2).abs() < 1e-6 && (f.y - 1.2).abs() < 1e-6, "{f}");
        assert!((f.z - 1.0).abs() < 1e-6, "the third axis is left at 1");
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
        assert_eq!(
            format_measure(Measure::Translate(Vec3::new(1.25, 0.0, 0.0))),
            "Δ X 1.250"
        );
        // A plane move names both of its axes, leaves the normal axis out, and
        // a move of nothing at all still reads as a number.
        assert_eq!(
            format_measure(Measure::Translate(Vec3::new(1.25, -0.5, 0.0))),
            "Δ X 1.250 Y -0.500"
        );
        assert_eq!(format_measure(Measure::Translate(Vec3::ZERO)), "Δ 0.000");
        // The centre handle's view-plane move has a share of all three.
        assert_eq!(
            format_measure(Measure::Translate(Vec3::new(0.5, -0.25, 0.125))),
            "Δ (0.500, -0.250, 0.125)"
        );
        assert_eq!(format_measure(Measure::Scale(Vec3::splat(2.0))), "× 2.000");
        assert_eq!(
            format_measure(Measure::Scale(Vec3::new(2.0, 1.0, 1.0))),
            "× X 2.000"
        );
        assert_eq!(
            format_measure(Measure::Scale(Vec3::new(1.0, 1.0, 3.0))),
            "× Z 3.000"
        );
        assert_eq!(
            format_measure(Measure::Scale(Vec3::new(2.0, 2.0, 1.0))),
            "× XY 2.000"
        );
        assert_eq!(format_measure(Measure::Rotate(Axis::Z, 90.0f32.to_radians())), "90.0°");
    }

    #[test]
    fn the_mode_cycle_is_off_grid_vertex() {
        assert_eq!(SnapMode::Off.next(), SnapMode::Grid);
        assert_eq!(SnapMode::Grid.next(), SnapMode::Vertex);
        assert_eq!(SnapMode::Vertex.next(), SnapMode::Off);
        assert_eq!(SnapMode::Grid.label(), "Grid");
    }
}
