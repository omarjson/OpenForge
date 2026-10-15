//! App-side glue between egui and the wgpu viewport backend.
//!
//! [`GpuViewport`] owns an [`GpuRenderer`] plus the egui texture handle that
//! presents its offscreen image. The renderer itself knows nothing about egui;
//! this module is the whole bridge, and it is only ever constructed when
//! eframe is running its `wgpu` backend. When it cannot be constructed the app
//! keeps painting through the CPU path in [`crate::viewport`], so the viewport
//! degrades instead of failing.

use crate::viewport::{self, Camera, OverlayEdge};
use eframe::egui;
use eframe::egui_wgpu::RenderState;
use openforge_core::scene::Scene;
use openforge_core::selection::{SelectLevel, Selection};
use openforge_render::gpu::{
    linear_from_srgb_u8, supported_samples, FrameStats, GpuRenderer, COLOR_FORMAT, KEY_LIGHT,
    PREFERRED_SAMPLES,
};
use openforge_render::{GpuVertex, NodeMaterials, OrbitCamera, PickElement, TextureLayer};
use wgpu::FilterMode;

/// Edge length, in texels, of every layer of the GPU texture array.
///
/// A texture array's layers share one size, so the app resamples each library
/// texture to this before handing it over. 256 keeps a library of a dozen
/// textures under a few megabytes while staying sharp at viewport sizes.
pub const TEXTURE_LAYER_SIZE: u32 = 256;

/// Cursor radius, in pixels, a GPU pick resolves within.
///
/// The ID pass draws mesh vertices as single pixels, so this radius is what
/// makes a click *near* a vertex land on it; it doubles as the tolerance for
/// face picks at grazing angles.
pub const PICK_RADIUS: u32 = 12;

/// The wgpu viewport and the egui handle its image is drawn through.
pub struct GpuViewport {
    renderer: GpuRenderer,
    texture_id: egui::TextureId,
}

impl GpuViewport {
    /// Build the backend on `state`'s device and register its (1x1) image with
    /// egui. The real size arrives with the first [`render`](Self::render).
    ///
    /// The offscreen target runs multi-sampled whenever the adapter supports
    /// the preferred count for the color format egui requires.
    pub fn new(state: &RenderState) -> Self {
        let samples = supported_samples(&state.adapter, COLOR_FORMAT, PREFERRED_SAMPLES);
        let renderer = GpuRenderer::new(&state.device, &state.queue, samples);
        let texture_id = state.renderer.write().register_native_texture(
            &state.device,
            renderer.color_view(),
            FilterMode::Linear,
        );
        Self { renderer, texture_id }
    }

    /// Upload the library's textures as the scene pass's texture array.
    ///
    /// Layers are taken in order from array layer 1; layer 0 stays the white
    /// layer untextured faces sample (see
    /// [`openforge_render::WHITE_LAYER`]), which is why a face's texture index
    /// is its layer minus one.
    pub fn set_textures(&mut self, state: &RenderState, layers: &[TextureLayer]) {
        self.renderer.set_textures(&state.device, &state.queue, layers);
    }

    /// Render one frame of `scene` at `size` in physical pixels, returning the
    /// egui image handle holding that frame plus [`FrameStats`] describing it.
    /// `edges` are the wireframe / selection-outline lines to draw inside the
    /// frame (see [`edge_vertices`]); an empty slice skips the line pass.
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
        materials: &NodeMaterials,
        camera: &OrbitCamera,
        time: f32,
        edges: &[GpuVertex],
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
        let stats = self.renderer.render(
            &state.device,
            &state.queue,
            scene,
            selection,
            materials,
            camera,
            time,
            edges,
        );
        (self.texture_id, stats)
    }

    /// Ask the GPU what sits under the pointer, nearest first.
    ///
    /// `pos` and `rect` are in egui points, and `scale` is the context's
    /// `pixels_per_point`: the renderer re-draws the scene the last frame drew
    /// as element ids, so the answer is exactly what the image shows — an
    /// occluded vertex, or a face whose back is turned, resolves to nothing.
    /// An empty result means "nothing there": the caller falls back to its own
    /// raycast.
    pub fn pick(
        &mut self,
        state: &RenderState,
        rect: egui::Rect,
        pos: egui::Pos2,
        scale: f32,
        scene: &mut Scene,
        camera: &OrbitCamera,
    ) -> Vec<PickElement> {
        let pixel = pick_pixel(pos, rect, scale, self.renderer.size());
        self.renderer.pick(
            &state.device,
            &state.queue,
            scene,
            camera,
            [pixel[0] as f32, pixel[1] as f32],
            PICK_RADIUS,
        )
    }
}

/// Convert world-space overlay edges into GPU line vertices for
/// [`GpuRenderer::render`].
///
/// Two vertices per edge (a line-list pair), coloured exactly like the CPU
/// painter's edge overlay (`viewport::EDGE_*`, converted from sRGB to the
/// linear space the offscreen target expects). The normal points at the key
/// light, so the shader's `N·L` term is 1 and each edge keeps its authored
/// colour.
pub fn edge_vertices(edges: &[OverlayEdge]) -> Vec<GpuVertex> {
    let normal = KEY_LIGHT.normalize();
    let mut out = Vec::with_capacity(edges.len() * 2);
    for edge in edges {
        let (srgb, alpha) = if edge.face_selected {
            (viewport::EDGE_FACE_SELECTED_SRGB, 1.0)
        } else if edge.node_selected {
            (viewport::EDGE_NODE_SELECTED_SRGB, 1.0)
        } else {
            (
                viewport::EDGE_PLAIN_SRGB,
                viewport::EDGE_PLAIN_ALPHA as f32 / 255.0,
            )
        };
        let mut color = linear_from_srgb_u8(srgb);
        color[3] = alpha;
        out.push(GpuVertex::new(edge.a, normal, [0.0, 0.0], color));
        out.push(GpuVertex::new(edge.b, normal, [0.0, 0.0], color));
    }
    out
}

/// Convert a viewport size in points to offscreen pixels, never returning a
/// degenerate (zero) target — wgpu rejects zero-sized textures.
pub fn target_size_px(points: [f32; 2]) -> [u32; 2] {
    let w = (points[0].round() as i64).max(1).min(u32::MAX as i64) as u32;
    let h = (points[1].round() as i64).max(1).min(u32::MAX as i64) as u32;
    [w, h]
}

/// The offscreen pixel a pointer position lands on, clamped into the target.
///
/// The frame is rendered at `rect.size() * scale` physical pixels with a
/// top-left origin, so a pointer's offset from the viewport's own top-left
/// corner scales straight onto a pixel — no flipping, since both put y down.
/// A click that lands outside (a stale rect, say) clamps rather than
/// addressing a texel the pick pass never drew.
pub fn pick_pixel(pos: egui::Pos2, rect: egui::Rect, scale: f32, size: [u32; 2]) -> [u32; 2] {
    let scale = if scale.is_finite() && scale > 0.0 { scale } else { 1.0 };
    let x = ((pos.x - rect.min.x) * scale).floor().max(0.0) as u32;
    let y = ((pos.y - rect.min.y) * scale).floor().max(0.0) as u32;
    [x.min(size[0].saturating_sub(1)), y.min(size[1].saturating_sub(1))]
}

/// The element a click at `level` should take from a pick's hits.
///
/// Hits come back nearest first, but a level only accepts what it edits: the
/// pass draws vertices as points over the faces they lie on, so a face-level
/// click has to skip past a nearer vertex and a vertex-level click has to skip
/// the face covering it. Object and edge levels pick nothing here — object
/// clicks go through the select tool, which owns node picking.
pub fn chosen_hit(level: SelectLevel, hits: &[PickElement]) -> Option<PickElement> {
    hits.iter().copied().find(|hit| match (level, hit) {
        (SelectLevel::Face, PickElement::Face { .. }) => true,
        (SelectLevel::Vertex, PickElement::Vertex { .. }) => true,
        _ => false,
    })
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
    use openforge_core::NodeId;

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

    #[test]
    fn edge_vertices_pair_the_endpoints_with_the_overlay_colours() {
        let edges = [
            OverlayEdge { a: Vec3::ZERO, b: Vec3::X, node_selected: true, face_selected: true },
            OverlayEdge { a: Vec3::Y, b: Vec3::Z, node_selected: true, face_selected: false },
            OverlayEdge { a: Vec3::X, b: Vec3::Y, node_selected: false, face_selected: false },
        ];
        let verts = edge_vertices(&edges);
        assert_eq!(verts.len(), edges.len() * 2);

        // Each pair shares its colour, serves as one line, and carries a unit
        // normal pointed at the key light (`N·L == 1`, so the shader returns
        // the authored colour unlit).
        let light = KEY_LIGHT.normalize();
        for pair in verts.chunks(2) {
            assert_eq!(pair[0].color, pair[1].color);
            assert_eq!(pair[0].uv, [0.0, 0.0]);
            for v in pair {
                let n = Vec3::from(v.normal);
                assert!((n.length() - 1.0).abs() < 1e-5);
                assert!((n.dot(light) - 1.0).abs() < 1e-5);
            }
        }
        assert_eq!(Vec3::from(verts[0].position), Vec3::ZERO);
        assert_eq!(Vec3::from(verts[1].position), Vec3::X);

        // Face-selected wins over node-selected; plain edges keep their alpha.
        assert_eq!(verts[0].color, linear_from_srgb_u8(viewport::EDGE_FACE_SELECTED_SRGB));
        assert_eq!(verts[2].color, linear_from_srgb_u8(viewport::EDGE_NODE_SELECTED_SRGB));
        let mut plain = linear_from_srgb_u8(viewport::EDGE_PLAIN_SRGB);
        plain[3] = viewport::EDGE_PLAIN_ALPHA as f32 / 255.0;
        assert_eq!(verts[4].color, plain);
    }

    #[test]
    fn edge_vertices_are_empty_without_edges() {
        assert!(edge_vertices(&[]).is_empty());
    }

    #[test]
    fn pick_pixel_scales_the_pointer_offset_onto_the_target() {
        let rect = egui::Rect::from_min_size(egui::pos2(100.0, 50.0), egui::vec2(320.0, 240.0));
        // 1:1: the offset from the viewport's corner is the pixel.
        assert_eq!(pick_pixel(egui::pos2(110.0, 70.0), rect, 1.0, [320, 240]), [10, 20]);
        // A 2x display doubles it.
        assert_eq!(pick_pixel(egui::pos2(110.0, 70.0), rect, 2.0, [640, 480]), [20, 40]);
        // Sub-pixel positions floor into the pixel they are inside.
        assert_eq!(pick_pixel(egui::pos2(110.9, 70.9), rect, 1.0, [320, 240]), [10, 20]);
    }

    #[test]
    fn pick_pixel_clamps_into_the_target() {
        let rect = egui::Rect::from_min_size(egui::pos2(100.0, 50.0), egui::vec2(320.0, 240.0));
        // Outside the viewport (left/above): the first texel.
        assert_eq!(pick_pixel(egui::pos2(0.0, 0.0), rect, 1.0, [320, 240]), [0, 0]);
        // Outside on the far side: the last texel, never past the target.
        assert_eq!(pick_pixel(egui::pos2(900.0, 900.0), rect, 1.0, [320, 240]), [319, 239]);
        // A scale of zero or NaN must not address a texel outside the target.
        for scale in [0.0, -3.0, f32::NAN, f32::INFINITY] {
            let p = pick_pixel(egui::pos2(110.0, 70.0), rect, scale, [320, 240]);
            assert_eq!(p, [10, 20], "scale {scale}");
        }
        // A degenerate target still answers with something addressable.
        assert_eq!(pick_pixel(egui::pos2(100.0, 50.0), rect, 1.0, [0, 0]), [0, 0]);
    }

    #[test]
    fn chosen_hit_skips_elements_the_level_cannot_use() {
        let face = PickElement::Face { node: NodeId(2), face: 7 };
        let vertex = PickElement::Vertex { node: NodeId(1), vertex: 3 };

        // A vertex point sits on top of its face: a face click must skip it.
        assert_eq!(chosen_hit(SelectLevel::Face, &[vertex, face]), Some(face));
        assert_eq!(chosen_hit(SelectLevel::Vertex, &[face, vertex]), Some(vertex));
        // A level with no usable hit falls through to the caller's raycast.
        assert_eq!(chosen_hit(SelectLevel::Vertex, &[face]), None);
        assert_eq!(chosen_hit(SelectLevel::Face, &[vertex]), None);
        assert_eq!(chosen_hit(SelectLevel::Face, &[]), None);
        // Node clicks go through the select tool, and edges are not pickable
        // yet: neither takes anything from a pick.
        assert_eq!(chosen_hit(SelectLevel::Object, &[vertex, face]), None);
        assert_eq!(chosen_hit(SelectLevel::Edge, &[vertex, face]), None);
    }
}
