//! Interpolation modes and value blending.

use glam::{Quat, Vec2, Vec3, Vec4};

/// How a keyframe's value blends toward the next key.
///
/// The mode is stored on the *source* key: it describes the segment that
/// starts at that key and ends at the following one.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum Interp {
    /// Hold the key's value until the next key is reached (constant).
    Step,
    /// Straight-line blend at constant speed.
    #[default]
    Linear,
    /// Hermite smoothstep: eases in and out with zero velocity at both keys.
    Smooth,
    /// Cubic ease-in-out: gentle at the ends, fastest through the middle.
    Ease,
}

impl Interp {
    /// Map a normalized time `t` (clamped to `[0, 1]`) through this mode's
    /// easing curve.
    pub fn apply(self, t: f32) -> f32 {
        let t = t.clamp(0.0, 1.0);
        match self {
            Interp::Step => {
                if t < 1.0 {
                    0.0
                } else {
                    1.0
                }
            }
            Interp::Linear => t,
            Interp::Smooth => smoothstep(t),
            Interp::Ease => ease_in_out_cubic(t),
        }
    }
}

/// Classic smoothstep, `3t² − 2t³`, for `t` in `[0, 1]`.
fn smoothstep(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}

/// Cubic ease-in-out for `t` in `[0, 1]`: `4t³` below the midpoint, mirrored
/// above it, so both ends meet with zero velocity.
fn ease_in_out_cubic(t: f32) -> f32 {
    if t < 0.5 {
        4.0 * t * t * t
    } else {
        let u = -2.0 * t + 2.0;
        1.0 - u * u * u / 2.0
    }
}

/// A value that can be blended between two keyframes.
///
/// Implementors provide a plain linear blend; the crate applies easing on top
/// of it via [`Interpolatable::interpolate`].
pub trait Interpolatable: Copy + Sized {
    /// Blend `a` toward `b` by `t` in `[0, 1]` with no easing applied.
    ///
    /// Quaternion implementations must blend along the shortest arc rather
    /// than per-component, so the result stays on the unit hypersphere.
    fn lerp(a: Self, b: Self, t: f32) -> Self;

    /// Blend `a` toward `b`, applying `interp`'s easing curve to `t` first.
    fn interpolate(a: Self, b: Self, t: f32, interp: Interp) -> Self {
        Self::lerp(a, b, interp.apply(t))
    }
}

/// Shortest-arc spherical linear interpolation between two quaternions.
///
/// Both inputs are expected to be normalized. When `b` lies in the opposite
/// hemisphere of `a` (`dot < 0`) it is negated first, so the blend never
/// travels the long way around the sphere. Nearly-parallel inputs fall back to
/// a normalized linear blend to avoid dividing by a near-zero sine.
pub fn slerp(a: Quat, b: Quat, t: f32) -> Quat {
    let mut b = b;
    let mut dot = a.dot(b);
    if dot < 0.0 {
        // `b` and `-b` are the same rotation; flip to the nearer one.
        dot = -dot;
        b = Quat::from_xyzw(-b.x, -b.y, -b.z, -b.w);
    }

    // Above this threshold the arc is so short that a normalized lerp is both
    // stable and visually identical to a true slerp.
    const NEAR_PARALLEL: f32 = 0.9995;
    if dot > NEAR_PARALLEL {
        let lerped = Quat::from_xyzw(
            a.x + (b.x - a.x) * t,
            a.y + (b.y - a.y) * t,
            a.z + (b.z - a.z) * t,
            a.w + (b.w - a.w) * t,
        );
        return lerped.normalize();
    }

    // `dot` is in `(0, NEAR_PARALLEL]` here, so the angle is never 0 or π
    // and its sine is safely non-zero.
    let theta = dot.min(1.0).max(-1.0).acos();
    let sin_theta = theta.sin();
    let scale_a = ((1.0 - t) * theta).sin() / sin_theta;
    let scale_b = (t * theta).sin() / sin_theta;
    Quat::from_xyzw(
        a.x * scale_a + b.x * scale_b,
        a.y * scale_a + b.y * scale_b,
        a.z * scale_a + b.z * scale_b,
        a.w * scale_a + b.w * scale_b,
    )
}

impl Interpolatable for f32 {
    fn lerp(a: f32, b: f32, t: f32) -> f32 {
        a + (b - a) * t
    }
}

impl Interpolatable for f64 {
    fn lerp(a: f64, b: f64, t: f32) -> f64 {
        a + (b - a) * f64::from(t)
    }
}

impl Interpolatable for Vec2 {
    fn lerp(a: Vec2, b: Vec2, t: f32) -> Vec2 {
        a.lerp(b, t)
    }
}

impl Interpolatable for Vec3 {
    fn lerp(a: Vec3, b: Vec3, t: f32) -> Vec3 {
        a.lerp(b, t)
    }
}

impl Interpolatable for Vec4 {
    fn lerp(a: Vec4, b: Vec4, t: f32) -> Vec4 {
        a.lerp(b, t)
    }
}

impl Interpolatable for Quat {
    /// Blends along the shortest arc; quaternions are never blended
    /// component-wise because that leaves unit length.
    fn lerp(a: Quat, b: Quat, t: f32) -> Quat {
        slerp(a, b, t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_apply_is_identity() {
        assert_eq!(Interp::default(), Interp::Linear);
        assert_eq!(Interp::Linear.apply(0.0), 0.0);
        assert_eq!(Interp::Linear.apply(0.37), 0.37);
        assert_eq!(Interp::Linear.apply(1.0), 1.0);
    }

    #[test]
    fn step_holds_until_the_next_key() {
        assert_eq!(Interp::Step.apply(0.0), 0.0);
        assert_eq!(Interp::Step.apply(0.5), 0.0);
        assert_eq!(Interp::Step.apply(0.999), 0.0);
        assert_eq!(Interp::Step.apply(1.0), 1.0);
    }

    #[test]
    fn smoothstep_midpoint_and_quarter() {
        assert_eq!(Interp::Smooth.apply(0.0), 0.0);
        assert_eq!(Interp::Smooth.apply(1.0), 1.0);
        assert!((Interp::Smooth.apply(0.5) - 0.5).abs() < 1e-6);
        // 3(0.25)² − 2(0.25)³ = 0.15625: slower than linear early on.
        assert!((Interp::Smooth.apply(0.25) - 0.15625).abs() < 1e-6);
    }

    #[test]
    fn ease_in_out_cubic_midpoint_and_quarter() {
        assert_eq!(Interp::Ease.apply(0.0), 0.0);
        assert_eq!(Interp::Ease.apply(1.0), 1.0);
        assert!((Interp::Ease.apply(0.5) - 0.5).abs() < 1e-6);
        // 4(0.25)³ = 0.0625: much slower than linear early on.
        assert!((Interp::Ease.apply(0.25) - 0.0625).abs() < 1e-6);
    }

    #[test]
    fn apply_clamps_out_of_range_time() {
        assert_eq!(Interp::Linear.apply(-3.0), 0.0);
        assert_eq!(Interp::Linear.apply(4.0), 1.0);
        assert_eq!(Interp::Step.apply(2.0), 1.0);
    }

    #[test]
    fn slerp_endpoints_and_midpoint_angle() {
        let a = Quat::IDENTITY;
        let b = Quat::from_rotation_y(1.0);
        assert!(a.dot(b) > 0.0);

        let start = slerp(a, b, 0.0);
        assert!(start.angle_between(a) < 1e-5);

        let mid = slerp(a, b, 0.5);
        assert!((mid.angle_between(a) - 0.5).abs() < 1e-4);

        let end = slerp(a, b, 1.0);
        assert!(end.abs_diff_eq(b, 1e-6));
        assert!((end.length() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn slerp_takes_shortest_path_for_negative_dot() {
        let a = Quat::IDENTITY;
        // `-q` is the same rotation as `q` but sits in the opposite hemisphere.
        let b = -Quat::from_rotation_x(0.5);
        assert!(a.dot(b) < 0.0, "precondition: target in opposite hemisphere");

        let mid = slerp(a, b, 0.5);
        // Short arc: half of 0.5 rad, not the ~1.46 rad long way around.
        assert!(mid.angle_between(Quat::from_rotation_x(0.25)) < 1e-4);
        assert!(mid.angle_between(a) < 0.3);
        assert!((mid.length() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn slerp_near_parallel_inputs_stay_normalized() {
        let a = Quat::from_rotation_z(0.3);
        let b = Quat::from_rotation_z(0.3000001);
        for t in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let m = slerp(a, b, t);
            assert!((m.length() - 1.0).abs() < 1e-5, "not unit length at t={t}");
        }
    }

    #[test]
    fn quat_interpolate_eases_then_slerps() {
        // A Linear segment is a plain slerp: quarter turn at the midpoint.
        let a = Quat::IDENTITY;
        let b = Quat::from_rotation_x(1.0);
        let mid = Quat::interpolate(a, b, 0.5, Interp::Linear);
        assert!((mid.angle_between(a) - 0.5).abs() < 1e-4);

        // Step holds the source rotation for the whole segment.
        let held = Quat::interpolate(a, b, 0.9, Interp::Step);
        assert!(held.angle_between(a) < 1e-5);
    }
}
