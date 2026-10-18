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
/// needs a branch: `color * white == color`. White is also the identity of the
/// specular map, whose texel *scales* the material's highlight.
pub const WHITE_LAYER: f32 = 0.0;

/// Array layer that samples the neutral normal `(0.5, 0.5, 1.0)`.
///
/// A tangent-space normal map stores a direction in `0.0..=1.0` per channel,
/// so its identity is not white but "straight out of the surface": sampling
/// this layer leaves the geometric normal alone, the same value the CPU path
/// uses for a missing normal map.
pub const NEUTRAL_NORMAL_LAYER: f32 = 1.0;

/// Array layer that samples black.
///
/// The identity of the emissive map, whose texel is *added* to the lit colour:
/// nothing to add is black, not white.
pub const BLACK_LAYER: f32 = 2.0;

/// How many layers a texture array reserves for those identities ahead of the
/// caller's own textures.
pub const RESERVED_LAYERS: usize = 3;

/// The array layer that samples the caller's texture `index` (0-based).
///
/// Textures start after the reserved identity layers; the renderer takes plain
/// texture indices so callers never do the offset arithmetic themselves.
pub fn texture_layer_index(index: usize) -> f32 {
    (index + RESERVED_LAYERS) as f32
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
/// position = 0, normal = 1, uv = 2, color = 3, texture layer = 4, the other
/// three map layers = 5..=7, and the material's own highlight (RGB plus the
/// exponent in `.w`) and glow = 8..=9.
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
    /// Diffuse texture-array layer; [`WHITE_LAYER`] means "no texture".
    pub tex_layer: f32,
    /// Normal-map array layer; [`NEUTRAL_NORMAL_LAYER`] means "no normal map".
    pub normal_layer: f32,
    /// Specular-map array layer; [`WHITE_LAYER`] means "no specular map".
    pub specular_layer: f32,
    /// Emissive-map array layer; [`BLACK_LAYER`] means "no emissive map".
    pub emissive_layer: f32,
    /// Linear-space highlight color, with the Phong exponent in `.w`; a
    /// non-positive exponent means the face has no highlight.
    pub specular: [f32; 4],
    /// Linear-space self-illumination, added to the lit color.
    pub emissive: [f32; 3],
}

impl GpuVertex {
    /// Assemble a vertex from glam vectors, sampling the identity layers and
    /// carrying no highlight or glow of its own.
    pub fn new(position: Vec3, normal: Vec3, uv: [f32; 2], color: [f32; 4]) -> Self {
        Self {
            position: position.to_array(),
            normal: normal.to_array(),
            uv,
            color,
            tex_layer: WHITE_LAYER,
            normal_layer: NEUTRAL_NORMAL_LAYER,
            specular_layer: WHITE_LAYER,
            emissive_layer: BLACK_LAYER,
            specular: [0.0; 4],
            emissive: [0.0; 3],
        }
    }

    /// Sample the caller's texture `index` (0-based) as the diffuse map, chaining.
    #[must_use]
    pub fn with_texture_layer(mut self, index: usize) -> Self {
        self.tex_layer = texture_layer_index(index);
        self
    }

    /// Sample the caller's texture `index` (0-based) as the normal map, chaining.
    #[must_use]
    pub fn with_normal_layer(mut self, index: usize) -> Self {
        self.normal_layer = texture_layer_index(index);
        self
    }

    /// Sample the caller's texture `index` (0-based) as the specular map, chaining.
    #[must_use]
    pub fn with_specular_layer(mut self, index: usize) -> Self {
        self.specular_layer = texture_layer_index(index);
        self
    }

    /// Sample the caller's texture `index` (0-based) as the emissive map, chaining.
    #[must_use]
    pub fn with_emissive_layer(mut self, index: usize) -> Self {
        self.emissive_layer = texture_layer_index(index);
        self
    }

    /// Carry the material's own highlight (RGB plus exponent in `.w`) and
    /// self-illumination, chaining.
    #[must_use]
    pub fn with_surface(mut self, specular: [f32; 4], emissive: [f32; 3]) -> Self {
        self.specular = specular;
        self.emissive = emissive;
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
    fn gpu_vertex_layout_is_92_bytes() {
        assert_eq!(size_of::<GpuVertex>(), 92);
        assert_eq!(align_of::<GpuVertex>(), 4);
        assert_eq!(offset_of!(GpuVertex, position), 0);
        assert_eq!(offset_of!(GpuVertex, normal), 12);
        assert_eq!(offset_of!(GpuVertex, uv), 24);
        assert_eq!(offset_of!(GpuVertex, color), 32);
        assert_eq!(offset_of!(GpuVertex, tex_layer), 48);
        assert_eq!(offset_of!(GpuVertex, normal_layer), 52);
        assert_eq!(offset_of!(GpuVertex, specular_layer), 56);
        assert_eq!(offset_of!(GpuVertex, emissive_layer), 60);
        assert_eq!(offset_of!(GpuVertex, specular), 64);
        assert_eq!(offset_of!(GpuVertex, emissive), 80);
        // The last attribute ends exactly at the stride.
        assert_eq!(offset_of!(GpuVertex, emissive) + 3 * 4, size_of::<GpuVertex>());
    }

    #[test]
    fn a_new_vertex_samples_the_identity_layers() {
        let v = GpuVertex::new(Vec3::ZERO, Vec3::Y, [0.5, 0.25], [1.0; 4]);
        // No map at all: white for the diffuse, the neutral normal, white for
        // the specular scale, black to add for the emissive.
        assert_eq!(v.tex_layer, WHITE_LAYER);
        assert_eq!(v.normal_layer, NEUTRAL_NORMAL_LAYER);
        assert_eq!(v.specular_layer, WHITE_LAYER);
        assert_eq!(v.emissive_layer, BLACK_LAYER);
        // And no highlight or glow of its own.
        assert_eq!(v.specular, [0.0; 4]);
        assert_eq!(v.emissive, [0.0; 3]);
        // Textures are offset past the reserved identity layers.
        assert_eq!(v.with_texture_layer(0).tex_layer, 3.0);
        assert_eq!(v.with_texture_layer(3).tex_layer, 6.0);
        assert_eq!(v.with_normal_layer(0).normal_layer, 3.0);
        assert_eq!(v.with_specular_layer(1).specular_layer, 4.0);
        assert_eq!(v.with_emissive_layer(2).emissive_layer, 5.0);
        // The reserved layers and the offset follow one rule.
        assert_eq!(RESERVED_LAYERS, 3);
        assert_eq!(texture_layer_index(0), RESERVED_LAYERS as f32);
        assert_eq!(NEUTRAL_NORMAL_LAYER, 1.0);
        assert_eq!(BLACK_LAYER, 2.0);
        // Everything else survives the chaining.
        assert_eq!(v.with_texture_layer(2).uv, [0.5, 0.25]);
        assert_eq!(v.with_texture_layer(2).color, [1.0; 4]);
        let mapped = v.with_surface([0.5, 0.25, 0.0, 32.0], [1.0, 0.0, 0.5]);
        assert_eq!(mapped.specular, [0.5, 0.25, 0.0, 32.0]);
        assert_eq!(mapped.emissive, [1.0, 0.0, 0.5]);
        assert_eq!(mapped.tex_layer, WHITE_LAYER, "the other fields are untouched");
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
