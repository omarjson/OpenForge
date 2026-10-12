//! App-side glue between egui and the wgpu viewport backend.
//!
//! [`GpuViewport`] owns an [`GpuRenderer`] plus the egui texture handle that
//! presents its offscreen image. The renderer itself knows nothing about egui;
//! this module is the whole bridge, and it is only ever constructed when
//! eframe is running its `wgpu` backend. When it cannot be constructed the app
//! keeps painting through the CPU path in [`crate::viewport`], so the viewport
//! degrades instead of failing.

use crate::viewport::Camera;
use eframe::egui;
use eframe::egui_wgpu::RenderState;
use openforge_core::scene::Scene;
use openforge_core::selection::Selection;
use openforge_render::gpu::{FrameStats, GpuRenderer};
use openforge_render::OrbitCamera;
use wgpu::FilterMode;

/// The wgpu viewport and the egui handle its image is drawn through.
pub struct GpuViewport {
    renderer: GpuRenderer,
    texture_id: egui::TextureId,
}

impl GpuViewport {
    /// Build the backend on `state`'s device and register its (1x1) image with
    /// egui. The real size arrives with the first [`render`](Self::render).
    pub fn new(state: &RenderState) -> Self {
        let renderer = GpuRenderer::new(&state.device);
        let texture_id = state.renderer.write().register_native_texture(
            &state.device,
            renderer.color_view(),
            FilterMode::Linear,
        );
        Self { renderer, texture_id }
    }

    /// Render one frame of `scene` at `size` in physical pixels, returning the
    /// egui image handle holding that frame plus [`FrameStats`] describing it.
    ///
    /// The handle is returned rather than exposed as an accessor on purpose:
    /// resizing re-registers the egui texture (egui holds a bind group over the
    /// old texture view), so a caller that read the handle *before* rendering
    /// would draw a texture id that has already been freed.
    pub fn render(
        &mut self,
        state: &RenderState,
        size: [f32; 2],
        scene: &mut Scene,
        selection: &Selection,
        camera: &OrbitCamera,
        time: f32,
    ) -> (egui::TextureId, FrameStats) {
        let size = target_size_px(size);
        if size != self.renderer.size() {
            self.renderer.resize(&state.device, size);
            let mut egui_renderer = state.renderer.write();
            egui_renderer.free_texture(&self.texture_id);
            self.texture_id = egui_renderer.register_native_texture(
                &state.device,
                self.renderer.color_view(),
                FilterMode::Linear,
            );
        }
        let stats = self
            .renderer
            .render(&state.device, &state.queue, scene, selection, camera, time);
        (self.texture_id, stats)
    }
}

/// Convert a viewport size in points to offscreen pixels, never returning a
/// degenerate (zero) target — wgpu rejects zero-sized textures.
pub fn target_size_px(points: [f32; 2]) -> [u32; 2] {
    let w = (points[0].round() as i64).max(1).min(u32::MAX as i64) as u32;
    let h = (points[1].round() as i64).max(1).min(u32::MAX as i64) as u32;
    [w, h]
}

/// Map the app's CPU painter camera onto the GPU camera.
///
/// The two are intentionally separate types (the GPU camera is tuned for clip
/// space and picking), so this keeps the viewport's orbit/pan/zoom state in
/// one place while the renderer works with its own copy.
pub fn orbit_camera(camera: &Camera) -> OrbitCamera {
    OrbitCamera {
        target: camera.target,
        distance: camera.distance,
        yaw: camera.yaw,
        pitch: camera.pitch,
        fov_y: camera.fov_y,
        near: camera.near,
        far: camera.far,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;
    use openforge_render::gpu::COLOR_FORMAT;

    /// egui_wgpu only accepts this format for `register_native_texture`.
    #[test]
    fn offscreen_format_is_the_one_egui_requires() {
        assert_eq!(COLOR_FORMAT, wgpu::TextureFormat::Rgba8UnormSrgb);
    }

    #[test]
    fn target_size_rounds_and_never_collapses_to_zero() {
        assert_eq!(target_size_px([640.0, 480.0]), [640, 480]);
        assert_eq!(target_size_px([640.4, 480.6]), [640, 481]);
        assert_eq!(target_size_px([0.0, 0.0]), [1, 1]);
        assert_eq!(target_size_px([-5.0, 12.0]), [1, 12]);
        assert_eq!(target_size_px([0.2, 0.2]), [1, 1]);
    }

    #[test]
    fn orbit_camera_mirrors_the_cpu_camera() {
        let camera = Camera {
            target: Vec3::new(1.0, -2.0, 3.0),
            distance: 12.5,
            yaw: 1.1,
            pitch: -0.4,
            ..Camera::default()
        };
        let gpu = orbit_camera(&camera);
        assert_eq!(gpu.target, camera.target);
        assert_eq!(gpu.distance, camera.distance);
        assert_eq!(gpu.yaw, camera.yaw);
        assert_eq!(gpu.pitch, camera.pitch);
        // The derived matrices must agree, or picking and rendering would
        // disagree about where things are.
        assert_eq!(gpu.eye(), camera.eye());
        assert_eq!(gpu.view_matrix(), camera.view_matrix());
        assert_eq!(gpu.view_proj(16.0 / 9.0), camera.view_proj(16.0 / 9.0));
    }
}
