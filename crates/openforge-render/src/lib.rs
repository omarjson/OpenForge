//! GPU viewport renderer backend for OpenForge.
//!
//! The crate has two layers:
//!
//! - **Renderer-neutral, always available.** Camera matrices, GPU-ready
//!   vertex/uniform data, mesh uploads and pass/shader descriptions that
//!   compile and test headless without any graphics dependency:
//!   [`OrbitCamera`] + [`FrameUniforms`] for per-frame uniforms,
//!   [`MeshBuffers`] for vertex/index uploads, and the WGSL sources and format
//!   constants in [`pipeline`].
//! - **The actual wgpu backend**, behind the `gpu` feature: [`gpu::GpuRenderer`]
//!   builds the pipelines, owns an offscreen color + depth target, and draws
//!   the grid, the scene and the corner axis gizmo. It is the only part that
//!   needs a GPU.

pub mod camera;
pub mod gpu_types;
pub mod mesh_upload;
pub mod pipeline;

#[cfg(feature = "gpu")]
pub mod gpu;

pub use camera::OrbitCamera;
pub use gpu_types::{FrameUniforms, GpuVertex};
pub use mesh_upload::{
    build_scene, build_scene_materials, build_scene_with, MeshBuffers, NodeColors, NodeMaterials,
};
pub use pipeline::{
    color_format, default_pass, depth_format, fragment_shader_wgsl, vertex_shader_wgsl, CullMode,
    RenderPassDesc,
};

#[cfg(feature = "gpu")]
pub use gpu::{FrameStats, GpuRenderer};
