//! Renderer-neutral description of the opaque-geometry draw pass, plus the
//! WGSL shader source and texture-format constants a wgpu backend needs.
//!
//! None of this depends on wgpu types, so it stays unit-testable headless;
//! the app translates these values into actual pipeline and encoder calls.

/// Face culling mode for the rasterizer.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum CullMode {
    /// Cull nothing.
    None,
    /// Cull back faces (the usual choice for closed meshes).
    #[default]
    Back,
    /// Cull front faces.
    Front,
}

/// What one render pass needs, in renderer-neutral terms.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct RenderPassDesc {
    /// RGBA clear color for the color attachment.
    pub clear_color: [f32; 4],
    /// Whether the pass writes and tests depth.
    pub depth_test: bool,
    /// Rasterizer face culling.
    pub cull: CullMode,
}

impl Default for RenderPassDesc {
    fn default() -> Self {
        default_pass()
    }
}

/// The standard opaque-geometry pass: dark viewport clear, depth test on,
/// back faces culled.
pub fn default_pass() -> RenderPassDesc {
    RenderPassDesc {
        clear_color: [0.11, 0.12, 0.14, 1.0],
        depth_test: true,
        cull: CullMode::Back,
    }
}

/// Vertex stage: MVP transform with the [`FrameUniforms`] uniform block at
/// `@group(0) @binding(0)`, matching [`crate::gpu_types::FrameUniforms`] and
/// [`crate::gpu_types::GpuVertex`] byte-for-byte.
///
/// [`FrameUniforms`]: crate::gpu_types::FrameUniforms
pub fn vertex_shader_wgsl() -> &'static str {
    r#"
struct FrameUniforms {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _pad0: f32,
    light_dir: vec3<f32>,
    time: f32,
};

@group(0) @binding(0) var<uniform> frame: FrameUniforms;

struct VsIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec4<f32>,
};

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) color: vec4<f32>,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    out.clip_pos = frame.view_proj * vec4<f32>(in.position, 1.0);
    out.world_normal = in.normal;
    out.color = in.color;
    return out;
}
"#
}

/// Fragment stage: N·L Lambert diffuse plus a small ambient term, modulated
/// by the interpolated vertex color. `frame.light_dir` points from the scene
/// toward the light.
pub fn fragment_shader_wgsl() -> &'static str {
    r#"
struct FrameUniforms {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _pad0: f32,
    light_dir: vec3<f32>,
    time: f32,
};

@group(0) @binding(0) var<uniform> frame: FrameUniforms;

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) color: vec4<f32>,
};

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let n = normalize(in.world_normal);
    let l = normalize(frame.light_dir);
    let diffuse = max(dot(n, l), 0.0);
    let ambient = 0.18;
    let lit = ambient + (1.0 - ambient) * diffuse;
    return vec4<f32>(in.color.rgb * lit, in.color.a);
}
"#
}

/// Texture format for the depth attachment, as a wgpu format name.
pub fn depth_format() -> &'static str {
    "Depth32Float"
}

/// Texture format for the color attachment, as a wgpu format name. Shaders
/// output linear color; the `_Srgb` suffix makes the hardware convert.
pub fn color_format() -> &'static str {
    "Rgba8UnormSrgb"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_pass_configuration() {
        let p = default_pass();
        assert!(p.depth_test);
        assert_eq!(p.cull, CullMode::Back);
        assert_eq!(p.clear_color.len(), 4);
        assert!(p.clear_color[3] > 0.0); // opaque clear
        assert_eq!(p, RenderPassDesc::default());
    }

    #[test]
    fn cull_mode_default_is_back() {
        assert_eq!(CullMode::default(), CullMode::Back);
    }

    #[test]
    fn vertex_wgsl_has_expected_structure() {
        let src = vertex_shader_wgsl();
        assert!(src.contains("@vertex"));
        assert!(src.contains("fn vs_main"));
        assert!(src.contains("@builtin(position)"));
        assert!(src.contains("view_proj"));
        assert!(src.contains("@group(0) @binding(0)"));
        assert!(src.contains("@location(0) position"));
        assert!(src.contains("@location(3) color"));
    }

    #[test]
    fn fragment_wgsl_has_expected_structure() {
        let src = fragment_shader_wgsl();
        assert!(src.contains("@fragment"));
        assert!(src.contains("fn fs_main"));
        assert!(src.contains("@location(0) vec4<f32>"));
        assert!(src.contains("dot(")); // N·L diffuse
        assert!(src.contains("ambient"));
    }

    #[test]
    fn both_stages_declare_the_same_uniform_block() {
        let vs = vertex_shader_wgsl();
        let fs = fragment_shader_wgsl();
        // The FrameUniforms declaration must be identical in both stages so
        // the binding layout matches.
        let decl = "light_dir: vec3<f32>,";
        assert!(vs.contains(decl) && fs.contains(decl));
        assert!(vs.contains("struct FrameUniforms"));
        assert!(fs.contains("struct FrameUniforms"));
    }

    #[test]
    fn format_constants_are_wgpu_names() {
        assert_eq!(depth_format(), "Depth32Float");
        assert_eq!(color_format(), "Rgba8UnormSrgb");
    }
}
