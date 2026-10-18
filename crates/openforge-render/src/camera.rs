//! A self-contained orbit camera for the GPU viewport.
//!
//! The camera looks at a target point from a spherical position defined by
//! yaw/pitch/distance and produces the view/projection matrices a GPU
//! backend consumes. It is deliberately independent from the app's CPU
//! painter camera so either can evolve without breaking the other.

use glam::{Mat4, Vec3, Vec4};
use openforge_core::math::{Aabb, Ray};

/// Maximum pitch, kept just short of the poles so the view basis never
/// degenerates at exactly ±90°.
const MAX_PITCH: f32 = std::f32::consts::PI * 89.0 / 180.0;
/// How far back along the view direction an orthographic view's eye sits. The
/// projection ignores it; it only has to keep the scene in front of the eye,
/// between the near and far planes.
const ORTHO_EYE: f32 = 500.0;
/// Near clip plane of an orthographic view. The rest of the depth range is
/// filled to just past the far side of [`ORTHO_EYE`].
const ORTHO_NEAR: f32 = 0.1;
/// Nearest the camera may dolly in.
const MIN_DISTANCE: f32 = 0.1;
/// Farthest the camera may dolly out.
const MAX_DISTANCE: f32 = 5000.0;
/// Yaw/pitch sensitivity per pixel of drag in [`orbit`](Self::orbit).
const ORBIT_SPEED: f32 = 0.01;
/// Pan sensitivity, as a fraction of distance per pixel of drag.
const PAN_SPEED: f32 = 0.002;
/// Zoom factor applied per unit of scroll.
const ZOOM_STEP: f32 = 0.1;

/// A perspective orbit ("turntable") camera.
///
/// `yaw`/`pitch` place the eye on a sphere of radius `distance` around
/// [`target`](Self::target). Projection is right-handed with a `[0, 1]`
/// depth range, matching wgpu's clip space.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct OrbitCamera {
    /// World point the camera looks at.
    pub target: Vec3,
    /// Distance from the eye to the target.
    pub distance: f32,
    /// Horizontal orbit angle in radians.
    pub yaw: f32,
    /// Vertical orbit angle in radians, clamped to ±89°.
    pub pitch: f32,
    /// Vertical field of view in radians.
    pub fov_y: f32,
    /// Near clip plane distance.
    pub near: f32,
    /// Far clip plane distance.
    pub far: f32,
    /// Draw with a parallel (orthographic) projection instead. `distance` is
    /// then the zoom — how much of the world fits the frame — rather than how
    /// far the eye sits from the target.
    pub ortho: bool,
    /// Which way is up on screen. An axis-aligned view looking straight down
    /// or straight up cannot use world up, which is parallel to its view.
    pub up: Vec3,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self {
            target: Vec3::ZERO,
            distance: 8.0,
            yaw: 0.6,
            pitch: 0.5,
            fov_y: 55f32.to_radians(),
            near: 0.05,
            far: 1000.0,
            ortho: false,
            up: Vec3::Y,
        }
    }
}

impl OrbitCamera {
    /// How far the eye sits back along the view direction. An orthographic
    /// view stands well clear of the scene so the whole of it is in front of
    /// the near plane.
    fn eye_distance(&self) -> f32 {
        if self.ortho {
            ORTHO_EYE
        } else {
            self.distance
        }
    }

    /// World-space position of the eye.
    pub fn eye(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        self.target + Vec3::new(cy * cp, sp, sy * cp) * self.eye_distance()
    }

    /// Right-handed view matrix (world → view space).
    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(self.eye(), self.target, self.up)
    }

    /// Right-handed projection with a `[0, 1]` depth range: perspective, or an
    /// orthographic box with the extent this camera's perspective view has at
    /// the same distance, so switching between the two does not jump.
    pub fn proj_matrix(&self, aspect: f32) -> Mat4 {
        let aspect = aspect.max(0.01);
        if self.ortho {
            let half_h = (self.distance * (self.fov_y * 0.5).tan()).max(0.01);
            let half_w = half_h * aspect;
            return Mat4::orthographic_rh(
                -half_w,
                half_w,
                -half_h,
                half_h,
                ORTHO_NEAR,
                2.0 * ORTHO_EYE,
            );
        }
        let near = self.near.max(1e-4);
        let far = self.far.max(near + 1e-3);
        Mat4::perspective_rh(self.fov_y, aspect, near, far)
    }

    /// Combined `proj * view` matrix.
    pub fn view_proj(&self, aspect: f32) -> Mat4 {
        self.proj_matrix(aspect) * self.view_matrix()
    }

    /// Orbit around the target by pixel deltas, clamping pitch to ±89°.
    pub fn orbit(&mut self, dx: f32, dy: f32) {
        self.yaw += dx * ORBIT_SPEED;
        self.pitch = (self.pitch + dy * ORBIT_SPEED).clamp(-MAX_PITCH, MAX_PITCH);
    }

    /// Pan the target within the camera's right/up plane (drag in pixels).
    pub fn pan(&mut self, dx: f32, dy: f32) {
        let view = self.view_matrix();
        // Rows of the view rotation are the camera axes in world space.
        let right = Vec3::new(view.x_axis.x, view.y_axis.x, view.z_axis.x);
        let up = Vec3::new(view.x_axis.y, view.y_axis.y, view.z_axis.y);
        let scale = self.distance * PAN_SPEED;
        self.target += (-right * dx + up * dy) * scale;
    }

    /// Dolly in/out by scroll amount, clamping the distance to a sane range.
    pub fn zoom(&mut self, scroll: f32) {
        let next = self.distance * (1.0 - scroll * ZOOM_STEP);
        self.distance = next.clamp(MIN_DISTANCE, MAX_DISTANCE);
    }

    /// Frame a world-space bounding box: center the target on it and set the
    /// distance so the box fits the view. An empty box leaves the camera
    /// unchanged.
    pub fn frame_bounds(&mut self, bounds: &Aabb) {
        if bounds.is_empty() {
            return;
        }
        self.target = bounds.center();
        // Half-diagonal of the box, kept positive so a point-sized box still
        // produces a usable (minimum) distance.
        let radius = (bounds.size() * 0.5).length().max(1e-3);
        let fov = self.fov_y.clamp(0.01, std::f32::consts::PI - 0.01);
        let fit = radius / (fov * 0.5).sin();
        self.distance = (fit * 1.1).clamp(MIN_DISTANCE, MAX_DISTANCE);
    }

    /// Build a picking ray from normalized device coords in `[-1, 1]` by
    /// unprojecting the near/far points through the inverse view-projection.
    pub fn ray_from_ndc(&self, ndc_x: f32, ndc_y: f32, aspect: f32) -> Ray {
        let inv = self.view_proj(aspect).inverse();
        let near = inv * Vec4::new(ndc_x, ndc_y, -1.0, 1.0);
        let far = inv * Vec4::new(ndc_x, ndc_y, 1.0, 1.0);
        let near = near.truncate() / near.w;
        let far = far.truncate() / far.w;
        Ray {
            origin: near,
            direction: (far - near).normalize_or_zero(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-4;

    fn approx(a: Vec3, b: Vec3) -> bool {
        (a - b).length() < EPS
    }

    #[test]
    fn default_camera_values() {
        let c = OrbitCamera::default();
        assert_eq!(c.target, Vec3::ZERO);
        assert_eq!(c.distance, 8.0);
        assert!(c.near > 0.0 && c.far > c.near);
    }

    #[test]
    fn eye_lies_at_target_plus_spherical_offset() {
        let c = OrbitCamera::default();
        let (sy, cy) = c.yaw.sin_cos();
        let (sp, cp) = c.pitch.sin_cos();
        let expected = c.target + Vec3::new(cy * cp, sp, sy * cp) * c.distance;
        assert!(approx(c.eye(), expected));
        assert!((c.eye() - c.target).length() - c.distance < EPS);
    }

    #[test]
    fn view_matrix_places_target_on_neg_z_axis() {
        let c = OrbitCamera::default();
        let view = c.view_matrix();
        let p = view.transform_point3(c.target);
        assert!(p.x.abs() < EPS && p.y.abs() < EPS);
        assert!((p.z + c.distance).abs() < EPS);
    }

    #[test]
    fn proj_matrix_maps_near_and_far_planes() {
        let c = OrbitCamera::default();
        let proj = c.proj_matrix(16.0 / 9.0);
        let at = |depth: f32| {
            let clip = proj * Vec4::new(0.0, 0.0, -depth, 1.0);
            clip.z / clip.w
        };
        assert!(at(c.near).abs() < EPS);
        assert!((at(c.far) - 1.0).abs() < EPS);
    }

    #[test]
    fn view_proj_projects_target_to_screen_center() {
        let c = OrbitCamera::default();
        let vp = c.view_proj(1.5);
        let clip = vp * Vec4::new(0.0, 0.0, 0.0, 1.0);
        assert!(clip.w > 0.0);
        let ndc = clip.truncate() / clip.w;
        assert!(ndc.x.abs() < EPS && ndc.y.abs() < EPS);
        assert!((0.0..=1.0).contains(&ndc.z));
    }

    #[test]
    fn orbit_changes_yaw_and_clamps_pitch() {
        let mut c = OrbitCamera::default();
        let yaw0 = c.yaw;
        c.orbit(10.0, 0.0);
        assert!((c.yaw - yaw0 - 0.1).abs() < EPS);
        // Far past the top pole: clamped to +89°.
        c.orbit(0.0, 1e6);
        assert!((c.pitch - 89f32.to_radians()).abs() < EPS);
        // Far past the bottom pole: clamped to -89°.
        c.orbit(0.0, -2e6);
        assert!((c.pitch + 89f32.to_radians()).abs() < EPS);
    }

    #[test]
    fn orbit_preserves_distance_to_target() {
        let mut c = OrbitCamera::default();
        c.orbit(3.5, -1.25);
        assert!((c.eye() - c.target).length() - c.distance < EPS);
    }

    #[test]
    fn zoom_scales_then_clamps_distance() {
        let mut c = OrbitCamera::default();
        c.zoom(1.0);
        assert!((c.distance - 7.2).abs() < EPS);
        c.zoom(1e6);
        assert_eq!(c.distance, 0.1);
        c.zoom(-1e6);
        assert_eq!(c.distance, 5000.0);
    }

    #[test]
    fn pan_moves_target_within_view_plane() {
        let mut c = OrbitCamera::default();
        let view = c.view_matrix();
        let forward = (c.target - c.eye()).normalize();
        let right = Vec3::new(view.x_axis.x, view.y_axis.x, view.z_axis.x);
        let before = c.target;
        c.pan(10.0, 0.0);
        let delta = c.target - before;
        // The target moves perpendicular to the view direction...
        assert!(delta.dot(forward).abs() < EPS);
        // ...to the left when dragging right.
        assert!(delta.dot(-right) > 0.0);
    }

    #[test]
    fn frame_bounds_centers_target_and_fits_box() {
        let mut c = OrbitCamera::default();
        let bounds = Aabb {
            min: Vec3::new(-1.0, -1.0, -1.0),
            max: Vec3::new(1.0, 1.0, 1.0),
        };
        c.frame_bounds(&bounds);
        assert!(approx(c.target, bounds.center()));
        let vp = c.view_proj(16.0 / 9.0);
        // Every corner of the box must land inside the frustum.
        for x in [-1.0f32, 1.0] {
            for y in [-1.0f32, 1.0] {
                for z in [-1.0f32, 1.0] {
                    let clip = vp * Vec4::new(x, y, z, 1.0);
                    assert!(clip.w > 0.0);
                    let ndc = clip.truncate() / clip.w;
                    assert!(ndc.x.abs() <= 1.001 && ndc.y.abs() <= 1.001);
                    assert!((-0.001..=1.001).contains(&ndc.z));
                }
            }
        }
    }

    #[test]
    fn frame_bounds_handles_empty_and_point_boxes() {
        let mut c = OrbitCamera::default();
        let before = c;
        c.frame_bounds(&Aabb::default());
        assert_eq!(c, before); // empty box: no change, no panic
        let point = Aabb { min: Vec3::splat(3.0), max: Vec3::splat(3.0) };
        c.frame_bounds(&point);
        assert!(approx(c.target, Vec3::splat(3.0)));
        assert!(c.distance >= 0.1 && c.distance.is_finite());
    }

    #[test]
    fn ray_from_ndc_center_points_from_eye_to_target() {
        let c = OrbitCamera::default();
        let ray = c.ray_from_ndc(0.0, 0.0, 16.0 / 9.0);
        let to_target = (c.target - c.eye()).normalize();
        assert!(ray.direction.dot(to_target) > 0.999);
        // Origin sits at the near plane, just in front of the eye.
        assert!(ray.origin.distance(c.eye()) < 0.5);
        // The ray reaches the target's depth plane at (almost) the target.
        let t = (c.target - ray.origin).dot(ray.direction);
        assert!(ray.at(t).distance(c.target) < 0.01);
    }

    #[test]
    fn ray_from_ndc_corners_differ_and_are_normalized() {
        let c = OrbitCamera::default();
        let center = c.ray_from_ndc(0.0, 0.0, 1.5);
        let corner = c.ray_from_ndc(1.0, 1.0, 1.5);
        assert!((corner.direction.length() - 1.0).abs() < EPS);
        assert!(corner.direction.dot(center.direction) < 0.999);
        assert!(corner.origin.distance(c.eye()) < 1.0);
    }

    #[test]
    fn ray_from_ndc_survives_degenerate_aspect() {
        let c = OrbitCamera::default();
        let ray = c.ray_from_ndc(0.5, -0.5, 0.0); // aspect guarded to 0.01
        assert!(ray.direction.length() <= 1.0 + EPS);
    }
}
