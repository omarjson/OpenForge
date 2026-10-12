//! GPU viewport renderer backend for OpenForge.
//!
//! This crate prepares everything a GPU renderer needs — camera matrices,
//! GPU-ready vertex/uniform data, mesh uploads and pass/shader descriptions
//! — without depending on `wgpu`, so all of it compiles and tests headless.
//! A wgpu backend (wired by the app) consumes:
//!
//! - [`OrbitCamera`] + [`FrameUniforms`] for per-frame uniforms,
//! - [`MeshBuffers`] for vertex/index buffer uploads,
//! - [`RenderPassDesc`], the WGSL sources and format strings in [`pipeline`]
//!   to configure the actual pipeline and render pass.

pub mod camera;
pub mod gpu_types;
pub mod mesh_upload;
pub mod pipeline;

pub use camera::OrbitCamera;
pub use gpu_types::{FrameUniforms, GpuVertex};
pub use mesh_upload::MeshBuffers;
pub use pipeline::{
    color_format, default_pass, depth_format, fragment_shader_wgsl, vertex_shader_wgsl, CullMode,
    RenderPassDesc,
};
