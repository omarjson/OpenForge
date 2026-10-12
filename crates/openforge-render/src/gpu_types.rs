//! GPU-ready POD types shared with the wgpu backend.
//!
//! Both types are `#[repr(C)]` and derive `bytemuck::Pod`, so they can be
//! written straight into GPU buffers. Their byte layouts match the WGSL
//! declarations in [`crate::pipeline`] exactly.

use crate::camera::OrbitCamera;
use glam::Vec3;

/// One vertex of the GPU vertex buffer.
///
/// The field order matches the `@location`s declared by
/// [`vertex_shader_wgsl`](crate::pipeline::vertex_shader_wgsl):
/// position = 0, normal = 1, uv = 2, color = 3.
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
}

impl GpuVertex {
    /// Assemble a vertex from glam vectors.
    pub fn new(position: Vec3, normal: Vec3, uv: [f32; 2], color: [f32; 4]) -> Self {
        Self {
            position: position.to_array(),
            normal: normal.to_array(),
            uv,
            color,
        }
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
    fn gpu_vertex_layout_is_48_bytes() {
        assert_eq!(size_of::<GpuVertex>(), 48);
        assert_eq!(align_of::<GpuVertex>(), 4);
        assert_eq!(offset_of!(GpuVertex, position), 0);
        assert_eq!(offset_of!(GpuVertex, normal), 12);
        assert_eq!(offset_of!(GpuVertex, uv), 24);
        assert_eq!(offset_of!(GpuVertex, color), 32);
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
