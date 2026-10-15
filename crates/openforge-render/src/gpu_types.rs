//! GPU-ready POD types shared with the wgpu backend.
//!
//! Both types are `#[repr(C)]` and derive `bytemuck::Pod`, so they can be
//! written straight into GPU buffers. Their byte layouts match the WGSL
//! declarations in [`crate::pipeline`] exactly.

use crate::camera::OrbitCamera;
use glam::Vec3;

/// Array layer that samples a white texel.
///
/// It is reserved in every texture array a [`GpuRenderer`](crate::gpu::GpuRenderer)
/// uploads, so an untextured vertex can name *a* layer and the shader never
/// needs a branch: `color * white == color`.
pub const WHITE_LAYER: f32 = 0.0;

/// The array layer that samples the caller's texture `index` (0-based).
///
/// Textures start at layer 1 because layer 0 is the reserved
/// [`WHITE_LAYER`]; the renderer takes plain texture indices so callers never
/// do the offset arithmetic themselves.
pub fn texture_layer_index(index: usize) -> f32 {
    (index + 1) as f32
}

/// One layer of the material texture array the scene pass samples.
///
/// The renderer knows nothing about materials: a caller that has a material
/// library resamples each of its textures to a common size and hands the
/// pixels in. Layer order is the caller's texture order, starting at array
/// layer 1 (layer 0 is the reserved [`WHITE_LAYER`]).
#[derive(Clone, PartialEq, Debug)]
pub struct TextureLayer {
    /// Width in texels (all layers of one array share this).
    pub width: u32,
    /// Height in texels.
    pub height: u32,
    /// RGBA8 texels, row-major, `width * height * 4` bytes.
    pub rgba: Vec<u8>,
}

impl TextureLayer {
    /// A layer from raw RGBA8 pixels. An empty (or inconsistent) pixel buffer
    /// yields a degenerate layer the renderer fills with white instead.
    pub fn new(width: u32, height: u32, rgba: Vec<u8>) -> Self {
        Self { width, height, rgba }
    }

    /// A layer of one solid color.
    pub fn solid(width: u32, height: u32, color: [u8; 4]) -> Self {
        let mut rgba = Vec::with_capacity((width as usize) * (height as usize) * 4);
        for _ in 0..(width as usize) * (height as usize) {
            rgba.extend_from_slice(&color);
        }
        Self { width, height, rgba }
    }

    /// True when the pixel buffer is the right size for the dimensions.
    pub fn is_complete(&self) -> bool {
        self.rgba.len() == (self.width as usize) * (self.height as usize) * 4
    }
}

/// One vertex of the GPU vertex buffer.
///
/// The field order matches the `@location`s declared by
/// [`vertex_shader_wgsl`](crate::pipeline::vertex_shader_wgsl):
/// position = 0, normal = 1, uv = 2, color = 3, texture layer = 4.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuVertex {
    /// World-space position.
    pub position: [f32; 3],
    /// World-space shading normal (unit length).
    pub normal: [f32; 3],
    /// Texture coordinates.
    pub uv: [f32; 2],
    /// Linear-space vertex color.
    pub color: [f32; 4],
    /// Texture-array layer to sample; [`WHITE_LAYER`] means "no texture".
    pub tex_layer: f32,
}

impl GpuVertex {
    /// Assemble a vertex from glam vectors, sampling the white layer.
    pub fn new(position: Vec3, normal: Vec3, uv: [f32; 2], color: [f32; 4]) -> Self {
        Self {
            position: position.to_array(),
            normal: normal.to_array(),
            uv,
            color,
            tex_layer: WHITE_LAYER,
        }
    }

    /// Sample the caller's texture `index` (0-based) at this vertex, chaining.
    #[must_use]
    pub fn with_texture_layer(mut self, index: usize) -> Self {
        self.tex_layer = texture_layer_index(index);
        self
    }
}

/// The per-frame uniform block, shared by the vertex and fragment stages.
///
/// The byte layout mirrors the `FrameUniforms` struct in the WGSL sources so
/// the whole block can be uploaded with a single `queue.write_buffer`:
///
/// | field        | offset | size |
/// |--------------|--------|------|
/// | `view_proj`  | 0      | 64   |
/// | `camera_pos` | 64     | 12   |
/// | `_pad0`      | 76     | 4    |
/// | `light_dir`  | 80     | 12   |
/// | `time`       | 92     | 4    |
///
/// Total: 96 bytes. `camera_pos`/`light_dir` are `vec3<f32>` in WGSL, which
/// are 16-byte aligned; the explicit `_pad0` keeps this struct free of
/// implicit padding (required by `bytemuck::Pod`).
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FrameUniforms {
    /// Column-major `proj * view`.
    pub view_proj: [[f32; 4]; 4],
    /// World-space camera position.
    pub camera_pos: [f32; 3],
    /// Padding: keeps `light_dir` at offset 80 like the WGSL `vec3`.
    pub _pad0: f32,
    /// Direction pointing from the scene *towards* the key light.
    pub light_dir: [f32; 3],
    /// Seconds since renderer start (reserved for shader animation).
    pub time: f32,
}

impl FrameUniforms {
    /// Build the block for a camera at the given aspect ratio, with the key
    /// light direction (normalized internally) and time.
    pub fn new(camera: &OrbitCamera, aspect: f32, light_dir: Vec3, time: f32) -> Self {
        Self {
            view_proj: camera.view_proj(aspect).to_cols_array_2d(),
            camera_pos: camera.eye().to_array(),
            _pad0: 0.0,
            light_dir: light_dir.normalize_or_zero().to_array(),
            time,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Mat4;
    use std::mem::{align_of, offset_of, size_of};

    #[test]
    fn gpu_vertex_layout_is_52_bytes() {
        assert_eq!(size_of::<GpuVertex>(), 52);
        assert_eq!(align_of::<GpuVertex>(), 4);
        assert_eq!(offset_of!(GpuVertex, position), 0);
        assert_eq!(offset_of!(GpuVertex, normal), 12);
        assert_eq!(offset_of!(GpuVertex, uv), 24);
        assert_eq!(offset_of!(GpuVertex, color), 32);
        assert_eq!(offset_of!(GpuVertex, tex_layer), 48);
    }

    #[test]
    fn a_new_vertex_samples_the_white_layer() {
        let v = GpuVertex::new(Vec3::ZERO, Vec3::Y, [0.5, 0.25], [1.0; 4]);
        assert_eq!(v.tex_layer, WHITE_LAYER);
        // Textures are offset past the reserved white layer.
        assert_eq!(v.with_texture_layer(0).tex_layer, 1.0);
        assert_eq!(v.with_texture_layer(3).tex_layer, 4.0);
        // Everything else survives the chaining.
        assert_eq!(v.with_texture_layer(2).uv, [0.5, 0.25]);
        assert_eq!(v.with_texture_layer(2).color, [1.0; 4]);
    }

    #[test]
    fn texture_layers_check_their_pixel_buffer() {
        let layer = TextureLayer::solid(2, 3, [10, 20, 30, 255]);
        assert_eq!(layer.width, 2);
        assert_eq!(layer.height, 3);
        assert_eq!(layer.rgba.len(), 2 * 3 * 4);
        assert!(layer.is_complete());
        assert!(!TextureLayer::new(2, 2, vec![0; 4]).is_complete());
        assert!(!TextureLayer::new(2, 2, Vec::new()).is_complete());
    }

    #[test]
    fn frame_uniforms_layout_is_96_bytes() {
        assert_eq!(size_of::<FrameUniforms>(), 96);
        assert_eq!(offset_of!(FrameUniforms, view_proj), 0);
        assert_eq!(offset_of!(FrameUniforms, camera_pos), 64);
        assert_eq!(offset_of!(FrameUniforms, light_dir), 80);
        assert_eq!(offset_of!(FrameUniforms, time), 92);
    }

    #[test]
    fn frame_uniforms_from_camera_match_camera_matrices() {
        let camera = OrbitCamera::default();
        let light = Vec3::new(0.0, 2.0, 0.0);
        let u = FrameUniforms::new(&camera, 16.0 / 9.0, light, 0.25);
        assert_eq!(u.view_proj, camera.view_proj(16.0 / 9.0).to_cols_array_2d());
        assert_eq!(u.camera_pos, camera.eye().to_array());
        // Light direction is normalized even when given unnormalized input.
        assert!((Vec3::from(u.light_dir).length() - 1.0).abs() < 1e-6);
        assert_eq!(u.time, 0.25);
        assert_eq!(u._pad0, 0.0);
    }

    #[test]
    fn view_proj_is_column_major_for_wgsl() {
        let camera = OrbitCamera::default();
        let u = FrameUniforms::new(&camera, 1.0, Vec3::Y, 0.0);
        let m = Mat4::from_cols_array_2d(&u.view_proj);
        assert_eq!(m, camera.view_proj(1.0));
    }
}
