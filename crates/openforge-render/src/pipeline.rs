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
/// The fragment stage also samples the texture array at `@binding(1)` through
/// the sampler at `@binding(2)`; see [`fragment_shader_wgsl`].
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
    @location(4) layer: f32,
    @location(5) normal_layer: f32,
    @location(6) specular_layer: f32,
    @location(7) emissive_layer: f32,
    @location(8) specular: vec4<f32>,
    @location(9) emissive: vec3<f32>,
};

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) color: vec4<f32>,
    @location(3) uv: vec2<f32>,
    @location(4) layers: vec4<f32>,
    @location(5) specular: vec4<f32>,
    @location(6) emissive: vec3<f32>,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    out.clip_pos = frame.view_proj * vec4<f32>(in.position, 1.0);
    out.world_pos = in.position;
    out.world_normal = in.normal;
    out.color = in.color;
    out.uv = in.uv;
    out.layers = vec4<f32>(in.layer, in.normal_layer, in.specular_layer, in.emissive_layer);
    out.specular = in.specular;
    out.emissive = in.emissive;
    return out;
}
"#
}

/// Fragment stage: N·L Lambert diffuse plus a small ambient term, the
/// material's own Phong highlight and self-illumination, all modulated by the
/// interpolated vertex color and the face's four map texels.
/// `frame.light_dir` points from the scene toward the light.
///
/// Each map has a reserved identity layer — white for the diffuse and specular
/// maps (one multiplies a colour, the other scales the highlight), the neutral
/// normal `(0.5, 0.5, 1.0)` for the normal map and black for the emissive map
/// (which is *added*, so nothing to add is black) — so a face without a map
/// samples its identity and no branch is needed to decide that. The normal
/// map's frame comes from the position and UV derivatives of the fragment
/// itself, so the mesh needs no tangent attribute.
///
/// [`WHITE_LAYER`]: crate::gpu_types::WHITE_LAYER
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
@group(0) @binding(1) var face_texture: texture_2d_array<f32>;
@group(0) @binding(2) var face_sampler: sampler;

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) color: vec4<f32>,
    @location(3) uv: vec2<f32>,
    @location(4) layers: vec4<f32>,
    @location(5) specular: vec4<f32>,
    @location(6) emissive: vec3<f32>,
};

/// Undo the sampling of an sRGB map array.
///
/// The array is an sRGB format so that *colour* maps (diffuse, emissive) arrive
/// as linear light. A normal or specular map is data, not colour: byte 128
/// means 0.5, not the 0.22 linear the decode gives it. This puts the texel back
/// where the artist stored it, so the two paths read a map the same way.
fn raw_texel(c: vec3<f32>) -> vec3<f32> {
    let low = c * 12.92;
    let high = 1.055 * pow(c, vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(low, high, c > vec3<f32>(0.0031308));
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // `textureSampleLevel` rather than `textureSample`: the layer varies per
    // face, and an explicit level keeps the fetch legal in any control flow.
    let base_texel = textureSampleLevel(face_texture, face_sampler, in.uv, i32(in.layers.x + 0.5), 0.0);
    let normal_texel = textureSampleLevel(face_texture, face_sampler, in.uv, i32(in.layers.y + 0.5), 0.0);
    let spec_texel = textureSampleLevel(face_texture, face_sampler, in.uv, i32(in.layers.z + 0.5), 0.0);
    let emissive_texel = textureSampleLevel(face_texture, face_sampler, in.uv, i32(in.layers.w + 0.5), 0.0);

    // The normal map stores a tangent-space direction. The surface's frame is
    // read back from the fragment's own position and UV derivatives, so the
    // mesh needs no tangent attribute; a triangle with no UV area keeps its
    // geometric normal.
    let n = normalize(in.world_normal);
    var normal = n;
    let dp1 = dpdx(in.world_pos);
    let dp2 = dpdy(in.world_pos);
    let duv1 = dpdx(in.uv);
    let duv2 = dpdy(in.uv);
    let det = duv1.x * duv2.y - duv2.x * duv1.y;
    if (abs(det) > 1e-8) {
        let scaled = 1.0 / det;
        let raw = (dp1 * duv2.y - dp2 * duv1.y) * scaled;
        let leaning = raw - n * dot(n, raw);
        // A triangle whose UVs give no frame, or a tangent parallel to the
        // normal, has nothing to hang the map on; the same three guards the
        // CPU path applies, so a degenerate map cannot produce a NaN normal.
        if (length(leaning) > 1e-6) {
            let tangent = normalize(leaning);
            let bitangent = cross(n, tangent);
            let map_n = raw_texel(normal_texel.rgb) * 2.0 - vec3<f32>(1.0);
            let perturbed = tangent * map_n.x + bitangent * map_n.y + n * map_n.z;
            if (length(perturbed) > 1e-6) {
                normal = normalize(perturbed);
            }
        }
    }

    let l = normalize(frame.light_dir);
    let v = normalize(frame.camera_pos - in.world_pos);
    let half = normalize(l + v);
    let diffuse = max(dot(normal, l), 0.0);
    let ambient = 0.18;
    let lit = ambient + (1.0 - ambient) * diffuse;

    // The specular map scales the material's own highlight — the mean of its
    // texel's channels, as the CPU path reads it — and the exponent rides in
    // `.w`, where a non-positive one means the face has no highlight.
    let spec_strength = dot(raw_texel(spec_texel.rgb), vec3<f32>(1.0 / 3.0));
    var specular = vec3<f32>(0.0);
    if (in.specular.w > 0.0) {
        specular = in.specular.rgb * spec_strength * pow(max(dot(normal, half), 0.0), in.specular.w);
    }
    let emissive = in.emissive + emissive_texel.rgb;
    return vec4<f32>(in.color.rgb * base_texel.rgb * lit + specular + emissive, in.color.a * base_texel.a);
}
"#
}

/// Both stages of the ID ("pick") pass, as one module.
///
/// The pass draws the scene a second time into a single-channel unsigned
/// integer target, writing each vertex's element id instead of a color: no
/// lighting, no texturing, no blending — and the id rides flat, so every
/// fragment of a triangle reports the same element however it interpolates.
/// Pair it with [`crate::pick::PickVertex`], whose `@location`s and 16-byte
/// stride match this module.
pub fn pick_shader_wgsl() -> &'static str {
    r#"
struct PickUniforms {
    view_proj: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> pick: PickUniforms;

struct VsIn {
    @location(0) position: vec3<f32>,
    @location(1) id: u32,
};

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) @interpolate(flat) id: u32,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    out.clip_pos = pick.view_proj * vec4<f32>(in.position, 1.0);
    out.id = in.id;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) u32 {
    return in.id;
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

/// Texture format for the material texture array, as a wgpu format name.
///
/// The same `_Srgb` encoding as the color target, so sampled texels and
/// vertex colors are both linear by the time the shader multiplies them.
pub fn texture_format() -> &'static str {
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
        // Every map layer and the two surface terms ride through untouched.
        assert!(src.contains("@location(4) layer: f32"));
        assert!(src.contains("@location(5) normal_layer: f32"));
        assert!(src.contains("@location(6) specular_layer: f32"));
        assert!(src.contains("@location(7) emissive_layer: f32"));
        assert!(src.contains("@location(8) specular: vec4<f32>"));
        assert!(src.contains("@location(9) emissive: vec3<f32>"));
        assert!(src.contains("@location(4) layers: vec4<f32>"));
        assert!(src.contains("out.layers = vec4<f32>(in.layer, in.normal_layer, in.specular_layer, in.emissive_layer);"));
        assert!(src.contains("out.specular = in.specular;"));
        assert!(src.contains("out.emissive = in.emissive;"));
        assert!(src.contains("out.world_pos = in.position;"));
        assert!(src.contains("out.uv = in.uv;"));
    }

    #[test]
    fn fragment_wgsl_has_expected_structure() {
        let src = fragment_shader_wgsl();
        assert!(src.contains("@fragment"));
        assert!(src.contains("fn fs_main"));
        assert!(src.contains("@location(0) vec4<f32>"));
        assert!(src.contains("dot(")); // N·L diffuse
        assert!(src.contains("ambient"));
        // All four maps come from the one array the bind group carries.
        assert!(src.contains("@group(0) @binding(1) var face_texture: texture_2d_array<f32>;"));
        assert!(src.contains("@group(0) @binding(2) var face_sampler: sampler;"));
        assert!(src.contains("textureSampleLevel("));
        assert!(!src.contains("textureSample("), "a level keeps the fetch legal in any control flow");
        // A normal map is put into the surface's own derivative-built frame.
        assert!(src.contains("dpdx(") && src.contains("dpdy("));
        assert!(src.contains("cross(n, tangent)"));
        // The specular map scales the material's own highlight, and the
        // exponent is the one the material carries.
        assert!(src.contains("spec_strength"));
        assert!(src.contains("in.specular.w"));
        assert!(src.contains("pow("));
        // Self-illumination is added, not multiplied.
        assert!(src.contains("in.emissive + emissive_texel.rgb"));
        assert!(src.contains(
            "in.color.rgb * base_texel.rgb * lit + specular + emissive"
        ));
        // A normal or specular map is data, not colour: the sRGB decode the
        // array applies to every texel is undone for those two, so byte 128
        // means 0.5 the way the CPU path reads it.
        assert!(src.contains("fn raw_texel("));
        assert!(src.contains("raw_texel(normal_texel.rgb) * 2.0"));
        assert!(src.contains("dot(raw_texel(spec_texel.rgb)"));
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
    fn both_stages_agree_on_the_varying_locations() {
        // The fragment stage reads what the vertex stage wrote; a location
        // mismatch would silently drop a map or a surface term.
        let vs = vertex_shader_wgsl();
        let fs = fragment_shader_wgsl();
        for location in [
            "@location(0) world_pos: vec3<f32>",
            "@location(1) world_normal: vec3<f32>",
            "@location(2) color: vec4<f32>",
            "@location(3) uv: vec2<f32>",
            "@location(4) layers: vec4<f32>",
            "@location(5) specular: vec4<f32>",
            "@location(6) emissive: vec3<f32>",
        ] {
            assert!(vs.contains(location), "vertex stage lost {location}");
            assert!(fs.contains(location), "fragment stage lost {location}");
        }
    }

    #[test]
    fn pick_wgsl_carries_the_id_through_flat() {
        let src = pick_shader_wgsl();
        assert!(src.contains("@vertex") && src.contains("@fragment"));
        assert!(src.contains("fn vs_main") && src.contains("fn fs_main"));
        assert!(src.contains("@location(1) id: u32"));
        // Flat: an element id may not be blended between vertices.
        assert!(src.contains("@location(0) @interpolate(flat) id: u32"));
        assert!(src.contains("out.id = in.id;"));
        // The target is an integer image, so the fragment stage writes u32.
        assert!(src.contains("-> @location(0) u32"));
        assert!(!src.contains("textureSample"), "an id pass samples nothing");
        assert!(!src.contains("light"), "an id pass is unlit");
    }

    #[test]
    fn format_constants_are_wgpu_names() {
        assert_eq!(depth_format(), "Depth32Float");
        assert_eq!(color_format(), "Rgba8UnormSrgb");
        assert_eq!(texture_format(), "Rgba8UnormSrgb");
    }
}
