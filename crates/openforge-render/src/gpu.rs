//! The wgpu backend for the OpenForge viewport.
//!
//! [`GpuRenderer`] owns an offscreen color + depth target, the pipelines built
//! from the WGSL in [`crate::pipeline`], and the per-frame geometry buffers.
//! [`GpuRenderer::render`] draws into that target, in order:
//!
//! 1. the ground grid (depth-tested, **no** depth write, so meshes always
//!    composite over it — matching the old CPU painter behaviour),
//! 2. the visible scene meshes (depth-tested *and* depth-written, which is the
//!    whole point of moving off the painter's-algorithm CPU path),
//! 3. the caller's wireframe / selection-outline edges as lines (depth-tested
//!    with `LessEqual` and no depth write, so an edge lying exactly on a shaded
//!    surface still draws while geometry behind it occludes it), and
//! 4. a corner axis gizmo drawn with a rotation-only ortho camera, so it
//!    tracks the orbit without ever being clipped by scene geometry.
//!
//! [`GpuRenderer::pick`] re-draws the scene into a single-channel integer
//! target (an "ID pass": one element id per texel, same camera, same depth
//! test, same culling) and reads a small patch back around the cursor. That is
//! what makes sub-object picking agree with the drawn frame: nothing the image
//! occludes or culls can be picked.
//!
//! The target can run multi-sampled: construct with a sample count the color
//! format supports (see [`supported_samples`]) and every pass resolves into
//! the single-sampled [`GpuRenderer::color_texture`] the app presents.
//!
//! The app then presents the offscreen color texture as an egui image; see
//! `openforge-app`'s `gpu_viewport` module. Nothing here depends on egui.
//!
//! The module sits behind the crate's `gpu` feature so the camera/geometry
//! half of the crate keeps building and testing without wgpu in the graph.

use crate::camera::OrbitCamera;
use crate::gpu_types::{FrameUniforms, GpuVertex, TextureLayer};
use crate::mesh_upload::{build_scene_materials, MeshBuffers, NodeMaterials};
use crate::pick::{resolve_patch, PickElement, PickGeometry, PickVertex};
use crate::pipeline::{self, CullMode};
use glam::{Mat4, Vec3, Vec4};
use openforge_core::scene::Scene;
use openforge_core::selection::{SelectLevel, Selection};
use wgpu::util::DeviceExt;

/// Depth format of the offscreen pass.
pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// Color format of the offscreen target.
///
/// `Rgba8UnormSrgb` is required by
/// [`egui_wgpu::Renderer::register_native_texture`] and matches what the
/// shaders assume: they write linear color and let the hardware encode it.
///
/// [`egui_wgpu::Renderer::register_native_texture`]:
///     https://docs.rs/egui-wgpu/0.29/egui_wgpu/struct.Renderer.html
pub const COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

/// Format of the material texture array, matching [`pipeline::texture_format`].
pub const TEXTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

/// Most texture layers one scene pass uploads (layer 0 is the white layer).
pub const MAX_TEXTURE_LAYERS: u32 = 64;

/// Color format of the ID ("pick") pass: one unsigned 32-bit element id per
/// texel, zero where nothing was drawn.
pub const PICK_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Uint;

/// Row alignment wgpu demands of every texture→buffer copy.
const PICK_ROW_ALIGNMENT: u32 = 256;

/// Largest cursor radius a pick resolves over, in pixels. A patch this wide
/// is already 129x129 texels: more than any click needs, and small enough to
/// map back synchronously.
const MAX_PICK_RADIUS: u32 = 64;

/// The multi-sample count [`GpuRenderer`] prefers when the adapter supports it.
pub const PREFERRED_SAMPLES: u32 = 4;

/// The sample count to build a [`GpuRenderer`] with: `requested` when the
/// adapter supports that exact count for `format`, otherwise 1.
///
/// There is deliberately no silent downgrade — a caller that wants a fallback
/// chain should try `PREFERRED_SAMPLES`, then smaller counts, and finally 1.
pub fn supported_samples(
    adapter: &wgpu::Adapter,
    format: wgpu::TextureFormat,
    requested: u32,
) -> u32 {
    if requested >= 2
        && adapter
            .get_texture_format_features(format)
            .flags
            .sample_count_supported(requested)
    {
        requested
    } else {
        1
    }
}

/// Largest offscreen target we allocate (anything bigger is clamped).
const MAX_TARGET_DIM: u32 = 8192;
/// Smallest offscreen target (wgpu rejects zero-sized textures).
const MIN_TARGET_DIM: u32 = 1;

/// Ground-grid half extent, in world units.
pub const GRID_HALF_EXTENT: i32 = 10;
/// Corner-gizmo edge length, in pixels.
pub const GIZMO_PIXELS: f32 = 108.0;
/// Gap between the gizmo and the viewport corner, in pixels.
pub const GIZMO_MARGIN: f32 = 12.0;
/// Half extent of the axis triad in gizmo space (axes are one unit long).
pub const GIZMO_HALF_EXTENT: f32 = 1.4;
/// Direction pointing from the scene *towards* the key light.
pub const KEY_LIGHT: Vec3 = Vec3::new(0.4, 0.8, 0.5);

/// Grid line color, authored as sRGB bytes for visual parity with the old CPU
/// painter. Converted to linear before it reaches the shader.
const GRID_SRGB: [u8; 3] = [45, 45, 45];
/// The X axis grid line.
const AXIS_X_SRGB: [u8; 3] = [180, 70, 70];
/// The Z axis grid line.
const AXIS_Z_SRGB: [u8; 3] = [70, 120, 180];
/// Axis-triad colors for the corner gizmo.
const GIZMO_X_SRGB: [u8; 3] = [226, 72, 72];
const GIZMO_Y_SRGB: [u8; 3] = [122, 200, 96];
const GIZMO_Z_SRGB: [u8; 3] = [84, 148, 226];

// --- Color helpers ---------------------------------------------------------------

/// Convert one sRGB-encoded channel in `0..=1` to linear light.
///
/// The sRGB transfer function; the inverse of the encoding the hardware
/// applies when writing linear color into an `Rgba8UnormSrgb` attachment.
pub fn srgb_channel_to_linear(c: f32) -> f32 {
    let c = c.clamp(0.0, 1.0);
    if c <= 0.040_45 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Convert opaque 8-bit sRGB components into a linear RGBA color suitable for
/// [`GpuVertex::color`].
pub fn linear_from_srgb_u8(rgb: [u8; 3]) -> [f32; 4] {
    [
        srgb_channel_to_linear(rgb[0] as f32 / 255.0),
        srgb_channel_to_linear(rgb[1] as f32 / 255.0),
        srgb_channel_to_linear(rgb[2] as f32 / 255.0),
        1.0,
    ]
}

/// The pass clear color in linear space.
///
/// [`pipeline::default_pass`] stores the intended *display* (sRGB) color, so
/// it is converted here; clearing an `Rgba8UnormSrgb` attachment with the raw
/// linear value would look washed out.
pub fn clear_color() -> wgpu::Color {
    let [r, g, b, a] = pipeline::default_pass().clear_color;
    wgpu::Color {
        r: srgb_channel_to_linear(r) as f64,
        g: srgb_channel_to_linear(g) as f64,
        b: srgb_channel_to_linear(b) as f64,
        a: a as f64,
    }
}

// --- Static geometry -------------------------------------------------------------

/// The unit normal shared by "unlit-looking" helper geometry.
///
/// The fragment stage always applies `ambient + (1 - ambient) * N·L`. Pointing
/// a helper vertex's normal straight at the key light makes `N·L == 1`, so the
/// grid and gizmo render at exactly their authored color without needing a
/// second shader.
fn unlit_normal() -> Vec3 {
    KEY_LIGHT.normalize()
}

fn grid_line(out: &mut Vec<GpuVertex>, a: Vec3, b: Vec3, color: [f32; 4]) {
    let n = unlit_normal();
    out.push(GpuVertex::new(a, n, [0.0, 0.0], color));
    out.push(GpuVertex::new(b, n, [0.0, 0.0], color));
}

/// Build the ground grid as a `LineList` vertex buffer: `2 * (2 * (half*2+1))`
/// vertices spanning `[-half, half]` on the y = 0 plane. The two lines through
/// the origin carry the axis colors.
pub fn grid_vertices(half: i32) -> Vec<GpuVertex> {
    let half = half.max(0);
    let e = half as f32;
    let grid = linear_from_srgb_u8(GRID_SRGB);
    let axis_x = linear_from_srgb_u8(AXIS_X_SRGB);
    let axis_z = linear_from_srgb_u8(AXIS_Z_SRGB);
    let mut out = Vec::with_capacity((2 * (2 * half as usize + 1)) * 2);
    for i in -half..=half {
        let f = i as f32;
        // Line along X (constant z): the z == 0 one is the X axis.
        let color = if i == 0 { axis_x } else { grid };
        grid_line(&mut out, Vec3::new(-e, 0.0, f), Vec3::new(e, 0.0, f), color);
        // Line along Z (constant x): the x == 0 one is the Z axis.
        let color = if i == 0 { axis_z } else { grid };
        grid_line(&mut out, Vec3::new(f, 0.0, -e), Vec3::new(f, 0.0, e), color);
    }
    out
}

/// Build the corner axis triad as a `LineList`: one line per world axis, from
/// the origin to +1 on that axis.
pub fn gizmo_vertices() -> Vec<GpuVertex> {
    let mut out = Vec::with_capacity(6);
    grid_line(&mut out, Vec3::ZERO, Vec3::X, linear_from_srgb_u8(GIZMO_X_SRGB));
    grid_line(&mut out, Vec3::ZERO, Vec3::Y, linear_from_srgb_u8(GIZMO_Y_SRGB));
    grid_line(&mut out, Vec3::ZERO, Vec3::Z, linear_from_srgb_u8(GIZMO_Z_SRGB));
    out
}

/// The corner gizmo's `[x, y, edge]` rect in pixels for a viewport of
/// `size`.
///
/// Guaranteed to fit entirely inside the viewport (wgpu rejects a viewport
/// that overflows the attachment), so it shrinks next to the margin when the
/// viewport is small and collapses to a 1px square when it is tiny.
/// `y` starts at the top edge because wgpu's framebuffer origin is top-left.
///
/// The gizmo sits in the **top-right** corner: the app draws its status text
/// in the top-left.
pub fn gizmo_viewport(size: [u32; 2]) -> [f32; 3] {
    let w = size[0] as f32;
    let h = size[1] as f32;
    let room = |extent: f32| (extent - 2.0 * GIZMO_MARGIN).max(1.0);
    let edge = GIZMO_PIXELS.min(room(w)).min(room(h));
    let x = (w - edge - GIZMO_MARGIN).clamp(0.0, (w - edge).max(0.0));
    let y = GIZMO_MARGIN.clamp(0.0, (h - edge).max(0.0));
    [x, y, edge]
}

/// Rotation-only ortho view-projection for the corner gizmo.
///
/// The camera's translation is dropped so the triad always stays centered, and
/// the ortho box uses a negative near plane so no axis can be clipped as the
/// triad spins.
pub fn gizmo_view_proj(camera: &OrbitCamera) -> Mat4 {
    let mut view = camera.view_matrix();
    view.w_axis = Vec4::new(0.0, 0.0, 0.0, 1.0);
    let h = GIZMO_HALF_EXTENT;
    Mat4::orthographic_rh(-h, h, -h, h, -10.0, 10.0) * view
}

// --- Frame statistics ------------------------------------------------------------

/// What the last frame drew, for the app's status overlay.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FrameStats {
    /// Vertices in the merged scene buffer.
    pub vertices: usize,
    /// Indices (three per triangle) in the merged scene buffer.
    pub indices: usize,
    /// Draw calls recorded for the frame (grid + scene + edges + gizmo).
    pub draws: usize,
}

// --- Vertex layout ---------------------------------------------------------------

/// Attribute layout matching [`GpuVertex`]'s `@location`s and byte offsets.
const VERTEX_ATTRIBUTES: [wgpu::VertexAttribute; 5] = [
    wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0, shader_location: 0 },
    wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 12, shader_location: 1 },
    wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 24, shader_location: 2 },
    wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 32, shader_location: 3 },
    wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32, offset: 48, shader_location: 4 },
];

fn vertex_layout() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<GpuVertex>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &VERTEX_ATTRIBUTES,
    }
}

/// Attribute layout matching [`PickVertex`]'s `@location`s and byte offsets.
const PICK_VERTEX_ATTRIBUTES: [wgpu::VertexAttribute; 2] = [
    wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0, shader_location: 0 },
    wgpu::VertexAttribute { format: wgpu::VertexFormat::Uint32, offset: 12, shader_location: 1 },
];

fn pick_vertex_layout() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<PickVertex>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &PICK_VERTEX_ATTRIBUTES,
    }
}

fn face_from_cull(cull: CullMode) -> Option<wgpu::Face> {
    match cull {
        CullMode::None => None,
        CullMode::Back => Some(wgpu::Face::Back),
        CullMode::Front => Some(wgpu::Face::Front),
    }
}

// --- Renderer ---------------------------------------------------------------------

/// Offscreen wgpu renderer for one viewport.
///
/// Build it once from the app's `wgpu::Device`, call [`resize`](Self::resize)
/// whenever the viewport's pixel size changes, and [`render`](Self::render)
/// once per frame. The completed image lives in [`color_texture`](Self::color_texture)
/// / [`color_view`](Self::color_view).
pub struct GpuRenderer {
    size: [u32; 2],
    samples: u32,
    color_texture: wgpu::Texture,
    color_view: wgpu::TextureView,
    depth_view: wgpu::TextureView,
    msaa_view: Option<wgpu::TextureView>,
    bind_group_layout: wgpu::BindGroupLayout,
    frame_uniforms: wgpu::Buffer,
    frame_bind_group: wgpu::BindGroup,
    gizmo_uniforms: wgpu::Buffer,
    gizmo_bind_group: wgpu::BindGroup,
    face_texture: wgpu::Texture,
    face_texture_view: wgpu::TextureView,
    face_sampler: wgpu::Sampler,
    texture_size: [u32; 2],
    layer_count: u32,
    scene_pipeline: wgpu::RenderPipeline,
    grid_pipeline: wgpu::RenderPipeline,
    edge_pipeline: wgpu::RenderPipeline,
    gizmo_pipeline: wgpu::RenderPipeline,
    /// ID-pass pipelines: faces as triangles, mesh vertices as points.
    pick_pipeline: wgpu::RenderPipeline,
    pick_point_pipeline: wgpu::RenderPipeline,
    /// The ID pass's own uniform block (just `view_proj`) and bind group,
    /// kept apart from the frame's so a pick never disturbs a rendered frame.
    pick_uniforms: wgpu::Buffer,
    pick_bind_group: wgpu::BindGroup,
    /// The ID pass's attachments, built on the first pick and rebuilt when the
    /// viewport is resized.
    pick_target: Option<PickTarget>,
    grid_vertices: wgpu::Buffer,
    grid_vertex_count: u32,
    gizmo_vertices: wgpu::Buffer,
    gizmo_vertex_count: u32,
    scene_vertices: wgpu::Buffer,
    scene_indices: wgpu::Buffer,
    scene_vertex_capacity: u64,
    scene_index_capacity: u64,
    scene_index_count: u32,
    edge_vertices: wgpu::Buffer,
    edge_vertex_capacity: u64,
    edge_vertex_count: u32,
    stats: FrameStats,
}

impl GpuRenderer {
    /// Build the pipelines, the bind groups and the static grid/gizmo geometry
    /// for a `samples`-sampled target (`samples` is clamped to at least 1; use
    /// [`supported_samples`] to pick a value the adapter supports).
    ///
    /// The material texture array starts as a single white layer, so a scene
    /// rendered before [`set_textures`](Self::set_textures) looks untextured
    /// rather than black. The offscreen target starts at 1x1; call
    /// [`resize`](Self::resize) with the real viewport size before the first
    /// frame.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, samples: u32) -> Self {
        let samples = samples.max(1);
        let vertex_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("openforge-scene-vertex"),
            source: wgpu::ShaderSource::Wgsl(pipeline::vertex_shader_wgsl().into()),
        });
        let fragment_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("openforge-scene-fragment"),
            source: wgpu::ShaderSource::Wgsl(pipeline::fragment_shader_wgsl().into()),
        });

        let pass = pipeline::default_pass();
        let cull = face_from_cull(pass.cull);

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("openforge-frame-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(
                            std::mem::size_of::<FrameUniforms>() as u64
                        ),
                    },
                    count: None,
                },
                // The material texture array: one layer per caller texture,
                // with layer 0 reserved for white (see `WHITE_LAYER`).
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("openforge-pipeline-layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let uniform_buffer = |label: &str| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::bytes_of(&FrameUniforms {
                    view_proj: Mat4::IDENTITY.to_cols_array_2d(),
                    camera_pos: [0.0; 3],
                    _pad0: 0.0,
                    light_dir: KEY_LIGHT.normalize().to_array(),
                    time: 0.0,
                }),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
        };
        let bind_group = |label: &str,
                          buffer: &wgpu::Buffer,
                          texture: &wgpu::TextureView,
                          sampler: &wgpu::Sampler| {
            face_bind_group(device, &bind_group_layout, label, buffer, texture, sampler)
        };

        // Repeat addressing matches how a `Texture`'s UVs wrap; linear
        // filtering matches the CPU painter's sampler.
        let face_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("openforge-face-sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let face_texture =
            create_face_texture(device, MIN_TARGET_DIM, MIN_TARGET_DIM, 1);
        write_face_layer(
            queue,
            &face_texture,
            [MIN_TARGET_DIM, MIN_TARGET_DIM],
            0,
            &[255, 255, 255, 255],
        );
        let face_texture_view = face_array_view(&face_texture);

        let frame_uniforms = uniform_buffer("openforge-frame-uniforms");
        let frame_bind_group = bind_group(
            "openforge-frame-bind-group",
            &frame_uniforms,
            &face_texture_view,
            &face_sampler,
        );
        let gizmo_uniforms = uniform_buffer("openforge-gizmo-uniforms");
        let gizmo_bind_group = bind_group(
            "openforge-gizmo-bind-group",
            &gizmo_uniforms,
            &face_texture_view,
            &face_sampler,
        );

        let build_pipeline = |label: &str,
                              topology: wgpu::PrimitiveTopology,
                              cull_mode: Option<wgpu::Face>,
                              depth_write: bool,
                              depth_compare: wgpu::CompareFunction| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &vertex_module,
                    entry_point: "vs_main",
                    compilation_options: Default::default(),
                    buffers: &[vertex_layout()],
                },
                primitive: wgpu::PrimitiveState {
                    topology,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode,
                    unclipped_depth: false,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    conservative: false,
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: depth_write,
                    depth_compare,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: samples,
                    ..Default::default()
                },
                fragment: Some(wgpu::FragmentState {
                    module: &fragment_module,
                    entry_point: "fs_main",
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: COLOR_FORMAT,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview: None,
                cache: None,
            })
        };

        // Opaque scene: back-face culled, depth tested and written.
        let scene_pipeline = build_pipeline(
            "openforge-scene",
            wgpu::PrimitiveTopology::TriangleList,
            cull,
            true,
            wgpu::CompareFunction::Less,
        );
        // Edge overlay: depth tested so meshes occlude it, but no depth write,
        // and `LessEqual` so an edge lying exactly on its own shaded surface
        // still passes the test.
        let edge_pipeline = build_pipeline(
            "openforge-edges",
            wgpu::PrimitiveTopology::LineList,
            None,
            false,
            wgpu::CompareFunction::LessEqual,
        );
        // Grid: drawn first, depth tested but not written, so meshes composite
        // over it exactly like the CPU painter did.
        let grid_pipeline = build_pipeline(
            "openforge-grid",
            wgpu::PrimitiveTopology::LineList,
            None,
            false,
            wgpu::CompareFunction::Less,
        );
        // Gizmo: last, always wins the depth test, never writes depth.
        let gizmo_pipeline = build_pipeline(
            "openforge-gizmo",
            wgpu::PrimitiveTopology::LineList,
            None,
            false,
            wgpu::CompareFunction::Always,
        );

        // --- The ID ("pick") pass ---------------------------------------------
        // A pick is a second draw of the same scene into a single-channel
        // integer target. It has its own shader, layout and attachments and
        // shares no state with the frame, so a pick can run at any time —
        // multi-sampled frame or not — without disturbing what is on screen.
        let pick_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("openforge-pick"),
            source: wgpu::ShaderSource::Wgsl(pipeline::pick_shader_wgsl().into()),
        });
        let pick_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("openforge-pick-layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(64),
                },
                count: None,
            }],
        });
        let pick_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("openforge-pick-pipeline-layout"),
            bind_group_layouts: &[&pick_layout],
            push_constant_ranges: &[],
        });
        let pick_uniforms = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("openforge-pick-uniforms"),
            contents: bytemuck::bytes_of(&Mat4::IDENTITY.to_cols_array()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let pick_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("openforge-pick-bind-group"),
            layout: &pick_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: pick_uniforms.as_entire_binding(),
            }],
        });
        let build_pick_pipeline = |label: &str,
                                   topology: wgpu::PrimitiveTopology,
                                   cull_mode: Option<wgpu::Face>,
                                   depth_compare: wgpu::CompareFunction| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pick_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &pick_module,
                    entry_point: "vs_main",
                    compilation_options: Default::default(),
                    buffers: &[pick_vertex_layout()],
                },
                primitive: wgpu::PrimitiveState {
                    topology,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode,
                    unclipped_depth: false,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    conservative: false,
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: true,
                    depth_compare,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                // Ids must survive a readback unblended, so the ID target is
                // never multi-sampled — whatever the frame runs at.
                multisample: wgpu::MultisampleState { count: 1, ..Default::default() },
                fragment: Some(wgpu::FragmentState {
                    module: &pick_module,
                    entry_point: "fs_main",
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: PICK_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview: None,
                cache: None,
            })
        };
        // Faces: exactly the visible pass's culling and depth test.
        let pick_pipeline = build_pick_pipeline(
            "openforge-pick-faces",
            wgpu::PrimitiveTopology::TriangleList,
            cull,
            wgpu::CompareFunction::Less,
        );
        // Vertices: points drawn after the faces, one pixel each, so a vertex
        // wins over the surface it lies on (`LessEqual` for the coincident
        // depth) without ever beating a face genuinely in front of it.
        let pick_point_pipeline = build_pick_pipeline(
            "openforge-pick-vertices",
            wgpu::PrimitiveTopology::PointList,
            None,
            wgpu::CompareFunction::LessEqual,
        );

        let grid = grid_vertices(GRID_HALF_EXTENT);
        let grid_vertex_count = grid.len() as u32;
        let grid_vertices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("openforge-grid-vertices"),
            contents: bytemuck::cast_slice(&grid),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let gizmo = gizmo_vertices();
        let gizmo_vertex_count = gizmo.len() as u32;
        let gizmo_vertices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("openforge-gizmo-vertices"),
            contents: bytemuck::cast_slice(&gizmo),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let scene_vertex_capacity = 1024;
        let scene_index_capacity = 1024;
        let scene_vertices = create_dynamic_buffer(
            device,
            "openforge-scene-vertices",
            scene_vertex_capacity,
            wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        );
        let scene_indices = create_dynamic_buffer(
            device,
            "openforge-scene-indices",
            scene_index_capacity,
            wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
        );

        let edge_vertices = create_dynamic_buffer(
            device,
            "openforge-edge-vertices",
            1024,
            wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        );

        let (color_texture, color_view, depth_view, msaa_view) =
            create_targets(device, MIN_TARGET_DIM, MIN_TARGET_DIM, samples);

        Self {
            size: [MIN_TARGET_DIM, MIN_TARGET_DIM],
            samples,
            color_texture,
            color_view,
            depth_view,
            msaa_view,
            bind_group_layout,
            frame_uniforms,
            frame_bind_group,
            gizmo_uniforms,
            gizmo_bind_group,
            face_texture,
            face_texture_view,
            face_sampler,
            texture_size: [MIN_TARGET_DIM, MIN_TARGET_DIM],
            layer_count: 1,
            scene_pipeline,
            grid_pipeline,
            edge_pipeline,
            gizmo_pipeline,
            pick_pipeline,
            pick_point_pipeline,
            pick_uniforms,
            pick_bind_group,
            pick_target: None,
            grid_vertices,
            grid_vertex_count,
            gizmo_vertices,
            gizmo_vertex_count,
            scene_vertices,
            scene_indices,
            scene_vertex_capacity,
            scene_index_capacity,
            scene_index_count: 0,
            edge_vertices,
            edge_vertex_capacity: 1024,
            edge_vertex_count: 0,
            stats: FrameStats::default(),
        }
    }

    /// Current offscreen size in pixels.
    pub fn size(&self) -> [u32; 2] {
        self.size
    }

    /// Size of the uploaded material texture array, in texels.
    pub fn texture_size(&self) -> [u32; 2] {
        self.texture_size
    }

    /// Layer count of the material texture array: the reserved white layer
    /// plus the textures last handed to [`set_textures`](Self::set_textures).
    pub fn texture_layers(&self) -> u32 {
        self.layer_count
    }

    /// Upload the caller's textures as the scene pass's texture array,
    /// replacing any previous set.
    ///
    /// Layer order is the slice order offset by one: array layer 0 stays the
    /// reserved white layer an untextured vertex names (see
    /// [`WHITE_LAYER`]). Every layer is written through as white before the
    /// caller's pixels go in, so a layer whose buffer is missing or the wrong
    /// size shades like an untextured face instead of reading leftovers, and
    /// every layer of one array shares the first layer's dimensions. Layers
    /// past [`MAX_TEXTURE_LAYERS`] are ignored; an empty slice leaves a single
    /// white layer.
    pub fn set_textures(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layers: &[TextureLayer],
    ) {
        let layers = &layers[..layers.len().min(MAX_TEXTURE_LAYERS as usize)];
        let (width, height) = match layers.first() {
            Some(l) => (l.width.max(1), l.height.max(1)),
            None => (MIN_TARGET_DIM, MIN_TARGET_DIM),
        };
        let layer_count = 1 + layers.len() as u32;

        if [width, height] != self.texture_size || layer_count != self.layer_count {
            self.face_texture = create_face_texture(device, width, height, layer_count);
            self.face_texture_view = face_array_view(&self.face_texture);
            self.texture_size = [width, height];
            self.layer_count = layer_count;
            // Both bind groups hold the texture view, so they are rebuilt
            // whenever the array itself changes.
            self.frame_bind_group = face_bind_group(
                device,
                &self.bind_group_layout,
                "openforge-frame-bind-group",
                &self.frame_uniforms,
                &self.face_texture_view,
                &self.face_sampler,
            );
            self.gizmo_bind_group = face_bind_group(
                device,
                &self.bind_group_layout,
                "openforge-gizmo-bind-group",
                &self.gizmo_uniforms,
                &self.face_texture_view,
                &self.face_sampler,
            );
        }

        let white = vec![255u8; (width as usize) * (height as usize) * 4];
        for layer in 0..layer_count {
            write_face_layer(queue, &self.face_texture, self.texture_size, layer, &white);
        }
        for (i, layer) in layers.iter().enumerate() {
            if layer.width == width && layer.height == height && layer.is_complete() {
                write_face_layer(
                    queue,
                    &self.face_texture,
                    self.texture_size,
                    i as u32 + 1,
                    &layer.rgba,
                );
            }
        }
    }

    /// The target's multi-sample count (1 when single-sampled).
    pub fn samples(&self) -> u32 {
        self.samples
    }

    /// The rendered image. Register this with the egui renderer (or read it
    /// back in tests).
    pub fn color_texture(&self) -> &wgpu::Texture {
        &self.color_texture
    }

    /// The rendered image's view; this is what
    /// `egui_wgpu::Renderer::register_native_texture` wants.
    pub fn color_view(&self) -> &wgpu::TextureView {
        &self.color_view
    }

    /// Statistics for the last rendered frame.
    pub fn stats(&self) -> FrameStats {
        self.stats
    }

    /// Resize the offscreen target, clamping to `[1, 8192]`. Cheap no-op when
    /// the size is unchanged.
    pub fn resize(&mut self, device: &wgpu::Device, size: [u32; 2]) {
        let size = [
            size[0].clamp(MIN_TARGET_DIM, MAX_TARGET_DIM),
            size[1].clamp(MIN_TARGET_DIM, MAX_TARGET_DIM),
        ];
        if size == self.size {
            return;
        }
        let (color_texture, color_view, depth_view, msaa_view) =
            create_targets(device, size[0], size[1], self.samples);
        self.size = size;
        self.color_texture = color_texture;
        self.color_view = color_view;
        self.depth_view = depth_view;
        self.msaa_view = msaa_view;
    }

    /// Draw the grid, the visible meshes of `scene`, the caller's `edges`
    /// (wireframe / selection outline as line pairs) and the corner gizmo into
    /// the offscreen target, then submit the work. `time` feeds the shader's
    /// reserved clock (currently unused); an empty `edges` skips the line pass.
    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scene: &mut Scene,
        selection: &Selection,
        materials: &NodeMaterials,
        camera: &OrbitCamera,
        time: f32,
        edges: &[GpuVertex],
    ) -> FrameStats {
        self.upload(device, queue, scene, selection, materials);
        self.upload_edges(device, queue, edges);
        self.write_uniforms(queue, camera, time);

        let gizmo = gizmo_viewport(self.size);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("openforge-viewport"),
        });
        self.encode(&mut encoder, gizmo);
        queue.submit(Some(encoder.finish()));

        self.stats
    }

    /// Re-draw `scene` as element ids and resolve what sits under `cursor`
    /// (offscreen pixels, top-left origin), nearest first within `radius`.
    ///
    /// The pass draws the same geometry through the same camera with the same
    /// depth test and back-face culling as the visible frame, so what comes
    /// back matches what the image shows: an occluded vertex, or a face whose
    /// back faces the camera, cannot be picked. Ids are read back
    /// synchronously, so call this on a click rather than every frame.
    ///
    /// `radius` is clamped to [`MAX_PICK_RADIUS`]; an empty scene — or one
    /// whose every mesh is hidden or frozen — resolves to nothing without
    /// touching the GPU.
    pub fn pick(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scene: &mut Scene,
        camera: &OrbitCamera,
        cursor: [f32; 2],
        radius: u32,
    ) -> Vec<PickElement> {
        // The patch around the cursor, clipped to the target: at the edges it
        // shrinks, and the cursor is no longer at its middle.
        let cx = (cursor[0].floor().max(0.0) as u32).min(self.size[0] - 1);
        let cy = (cursor[1].floor().max(0.0) as u32).min(self.size[1] - 1);
        let radius = radius.min(MAX_PICK_RADIUS);
        let rect = [
            cx.saturating_sub(radius),
            cy.saturating_sub(radius),
            (cx + radius + 1).min(self.size[0]),
            (cy + radius + 1).min(self.size[1]),
        ];
        let Some((geometry, ids)) = self.read_id_rect(device, queue, scene, camera, rect, true)
        else {
            return Vec::new();
        };
        resolve_patch(
            &ids,
            rect[2] - rect[0],
            rect[3] - rect[1],
            [cx - rect[0], cy - rect[1]],
        )
        .into_iter()
        .filter_map(|hit| geometry.element(hit.id))
        .collect()
    }

    /// Re-draw `scene` as element ids and resolve every element inside an
    /// offscreen pixel rectangle `[x0, y0, x1, y1]` (top-left origin,
    /// half-open like every wgpu extent).
    ///
    /// This is the rectangle-selection counterpart of [`Self::pick`]: the same
    /// pass with the same depth test and culling, but what comes back is every
    /// distinct element anywhere in the rectangle, in element-table order —
    /// distance to the middle means nothing to a rectangle. The result is
    /// filtered to what the level can select, exactly like [`Self::pick`]'s
    /// hits are: vertex elements for `Vertex`, face elements (and the vertex
    /// pixels that put a node in the rectangle) for `Object`, face elements
    /// alone for `Face`, nothing for `Edge`.
    ///
    /// A face rectangle draws no vertex points: they are 1-pixel dots sitting
    /// on their own face, and a pixel hole in the face at every vertex would
    /// be the one way the picture and the pick could disagree.
    pub fn pick_rect(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scene: &mut Scene,
        camera: &OrbitCamera,
        rect: [u32; 4],
        level: SelectLevel,
    ) -> Vec<PickElement> {
        // No edge pass exists, so an edge rectangle is empty by definition.
        if matches!(level, SelectLevel::Edge) {
            return Vec::new();
        }
        let rect = [
            rect[0].min(self.size[0]),
            rect[1].min(self.size[1]),
            rect[2].min(self.size[0]),
            rect[3].min(self.size[1]),
        ];
        let with_points = !matches!(level, SelectLevel::Face);
        let Some((geometry, ids)) =
            self.read_id_rect(device, queue, scene, camera, rect, with_points)
        else {
            return Vec::new();
        };
        let elements = geometry.elements_in_patch(&ids, rect[2] - rect[0], rect[3] - rect[1]);
        elements
            .into_iter()
            .filter(|element| match level {
                SelectLevel::Vertex => matches!(element, PickElement::Vertex { .. }),
                // An object rectangle keeps both kinds: the app only needs the
                // node, and the CPU fallback answers with positions too.
                SelectLevel::Object => true,
                SelectLevel::Face => matches!(element, PickElement::Face { .. }),
                SelectLevel::Edge => false,
            })
            .collect()
    }

    /// Draw the ID pass for `scene` once and read the half-open rectangle
    /// `[x0, y0, x1, y1]` back out of it as tightly packed ids.
    ///
    /// `None` when the scene has nothing to pick or the rectangle is empty:
    /// neither is worth touching the GPU for.
    fn read_id_rect(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scene: &mut Scene,
        camera: &OrbitCamera,
        rect: [u32; 4],
        with_points: bool,
    ) -> Option<(PickGeometry, Vec<u32>)> {
        let (x0, y0, x1, y1) = (rect[0], rect[1], rect[2], rect[3]);
        if x1 <= x0 || y1 <= y0 {
            return None;
        }
        let geometry = PickGeometry::build(scene);
        if geometry.is_empty() {
            return None;
        }

        let size = self.size;
        let aspect = size[0] as f32 / size[1].max(1) as f32;
        queue.write_buffer(
            &self.pick_uniforms,
            0,
            bytemuck::bytes_of(&camera.view_proj(aspect).to_cols_array()),
        );

        let patch = [x1 - x0, y1 - y0];
        let row_bytes = pick_row_bytes(patch[0]);
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("openforge-pick-readback"),
            size: row_bytes as u64 * patch[1] as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // A scene of nothing but loose vertices has no triangles, and wgpu
        // rejects a zero-sized buffer.
        let triangles = (!geometry.triangles.is_empty()).then(|| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("openforge-pick-face-vertices"),
                contents: bytemuck::cast_slice(&geometry.triangles),
                usage: wgpu::BufferUsages::VERTEX,
            })
        });
        let points = with_points.then(|| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("openforge-pick-point-vertices"),
                contents: bytemuck::cast_slice(&geometry.points),
                usage: wgpu::BufferUsages::VERTEX,
            })
        });

        self.ensure_pick_target(device);
        let target = self.pick_target.as_ref().expect("pick target was just built");
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("openforge-pick"),
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("openforge-pick"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target.view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &target.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_bind_group(0, &self.pick_bind_group, &[]);
            if let Some(buffer) = &triangles {
                pass.set_pipeline(&self.pick_pipeline);
                pass.set_vertex_buffer(0, buffer.slice(..));
                pass.draw(0..geometry.triangles.len() as u32, 0..1);
            }
            // Points come last, so a vertex pixel wins over the face under it.
            if let Some(buffer) = &points {
                pass.set_pipeline(&self.pick_point_pipeline);
                pass.set_vertex_buffer(0, buffer.slice(..));
                pass.draw(0..geometry.points.len() as u32, 0..1);
            }
        }
        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &target.texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x: x0, y: y0, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &readback,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(row_bytes),
                    rows_per_image: Some(patch[1]),
                },
            },
            wgpu::Extent3d {
                width: patch[0],
                height: patch[1],
                depth_or_array_layers: 1,
            },
        );
        queue.submit(Some(encoder.finish()));

        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::Maintain::Wait);
        let data = slice.get_mapped_range();
        // Copies pad every row to 256 bytes; strip the padding on the way out.
        let mut ids: Vec<u32> = Vec::with_capacity((patch[0] * patch[1]) as usize);
        for row in 0..patch[1] as usize {
            let start = row * row_bytes as usize;
            let end = start + patch[0] as usize * 4;
            ids.extend_from_slice(bytemuck::cast_slice(&data[start..end]));
        }
        drop(data);
        readback.unmap();

        Some((geometry, ids))
    }

    /// Make sure the ID pass's attachments match the current viewport size,
    /// building them on the first pick and after every resize.
    fn ensure_pick_target(&mut self, device: &wgpu::Device) {
        if self.pick_target.as_ref().map(|t| t.size) != Some(self.size) {
            self.pick_target = Some(PickTarget::new(device, self.size));
        }
    }

    /// Merge the scene's visible meshes into the dynamic vertex/index buffers.
    fn upload(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scene: &mut Scene,
        selection: &Selection,
        materials: &NodeMaterials,
    ) {
        let (parts, _bounds) = build_scene_materials(scene, selection, materials);
        let mut merged = MeshBuffers::default();
        for part in &parts {
            merged.append(part);
        }

        self.scene_index_count = merged.indices.len() as u32;
        // Grid + optional scene + gizmo; `upload_edges` adds the edge pass.
        self.stats = FrameStats {
            vertices: merged.vertices.len(),
            indices: merged.indices.len(),
            draws: if merged.indices.is_empty() { 2 } else { 3 },
        };

        if !merged.vertices.is_empty() {
            write_dynamic(
                device,
                queue,
                &mut self.scene_vertices,
                &mut self.scene_vertex_capacity,
                wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                "openforge-scene-vertices",
                bytemuck::cast_slice(&merged.vertices),
            );
            write_dynamic(
                device,
                queue,
                &mut self.scene_indices,
                &mut self.scene_index_capacity,
                wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                "openforge-scene-indices",
                bytemuck::cast_slice(&merged.indices),
            );
        }
    }

    /// Upload the edge-overlay line vertices (two per edge). An empty slice
    /// records no edge draw at all.
    fn upload_edges(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, edges: &[GpuVertex]) {
        self.edge_vertex_count = edges.len() as u32;
        if edges.is_empty() {
            return;
        }
        write_dynamic(
            device,
            queue,
            &mut self.edge_vertices,
            &mut self.edge_vertex_capacity,
            wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            "openforge-edge-vertices",
            bytemuck::cast_slice(edges),
        );
        self.stats.draws += 1;
    }

    /// Upload the per-frame uniforms (scene camera + gizmo camera).
    fn write_uniforms(&mut self, queue: &wgpu::Queue, camera: &OrbitCamera, time: f32) {
        let aspect = self.size[0] as f32 / self.size[1].max(1) as f32;
        let frame = FrameUniforms::new(camera, aspect, KEY_LIGHT, time);
        queue.write_buffer(&self.frame_uniforms, 0, bytemuck::bytes_of(&frame));

        let gizmo = FrameUniforms {
            view_proj: gizmo_view_proj(camera).to_cols_array_2d(),
            camera_pos: [0.0; 3],
            _pad0: 0.0,
            light_dir: unlit_normal().to_array(),
            time,
        };
        queue.write_buffer(&self.gizmo_uniforms, 0, bytemuck::bytes_of(&gizmo));
    }

    /// Record the draw passes (grid, scene, edges, gizmo). `gizmo` is
    /// `[x, y, size]` in pixels.
    fn encode(&self, encoder: &mut wgpu::CommandEncoder, gizmo: [f32; 3]) {
        let clear = clear_color();
        // With MSAA the pass renders into the multi-sampled texture and
        // resolves into the single-sampled one the app presents.
        let (color_view, resolve_target) = match &self.msaa_view {
            Some(msaa) => (msaa, Some(&self.color_view)),
            None => (&self.color_view, None),
        };
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("openforge-viewport"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: color_view,
                resolve_target,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(clear),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        // 1. Ground grid: no depth write, so meshes composite over it.
        pass.set_pipeline(&self.grid_pipeline);
        pass.set_bind_group(0, &self.frame_bind_group, &[]);
        pass.set_vertex_buffer(0, self.grid_vertices.slice(..));
        pass.draw(0..self.grid_vertex_count, 0..1);

        // 2. Scene meshes: depth tested *and* written.
        if self.scene_index_count > 0 {
            pass.set_pipeline(&self.scene_pipeline);
            pass.set_bind_group(0, &self.frame_bind_group, &[]);
            pass.set_vertex_buffer(0, self.scene_vertices.slice(..));
            pass.set_index_buffer(self.scene_indices.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..self.scene_index_count, 0, 0..1);
        }

        // 3. Edge overlay: depth tested (no write) so hidden edges are
        //    occluded by the meshes in front of them.
        if self.edge_vertex_count > 0 {
            pass.set_pipeline(&self.edge_pipeline);
            pass.set_bind_group(0, &self.frame_bind_group, &[]);
            pass.set_vertex_buffer(0, self.edge_vertices.slice(..));
            pass.draw(0..self.edge_vertex_count, 0..1);
        }

        // 4. Corner gizmo: its own ortho camera, always drawn on top.
        let [x, y, size] = gizmo;
        if size >= 1.0 {
            pass.set_viewport(x, y, size, size, 0.0, 1.0);
            pass.set_scissor_rect(x as u32, y as u32, size as u32, size as u32);
            pass.set_pipeline(&self.gizmo_pipeline);
            pass.set_bind_group(0, &self.gizmo_bind_group, &[]);
            pass.set_vertex_buffer(0, self.gizmo_vertices.slice(..));
            pass.draw(0..self.gizmo_vertex_count, 0..1);
        }
    }
}

/// Create the material texture array: `layer_count` layers of `width` x
/// `height`, in the shader's texture format.
fn create_face_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    layer_count: u32,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("openforge-face-textures"),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: layer_count.max(1),
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: TEXTURE_FORMAT,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    })
}

/// A 2D-array view over the whole material texture array.
fn face_array_view(texture: &wgpu::Texture) -> wgpu::TextureView {
    texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some("openforge-face-view"),
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    })
}

/// The bind group every pass uses: the frame uniform plus the texture array
/// and its sampler.
fn face_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    label: &str,
    uniforms: &wgpu::Buffer,
    texture: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniforms.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(texture),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    })
}

/// Write one whole array layer of RGBA8 texels.
fn write_face_layer(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    size: [u32; 2],
    layer: u32,
    rgba: &[u8],
) {
    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d { x: 0, y: 0, z: layer },
            aspect: wgpu::TextureAspect::All,
        },
        rgba,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(4 * size[0]),
            rows_per_image: Some(size[1]),
        },
        wgpu::Extent3d {
            width: size[0],
            height: size[1],
            depth_or_array_layers: 1,
        },
    );
}

/// Create the offscreen target for a `width` x `height` viewport: the
/// single-sampled color texture (also the MSAA resolve target), a depth
/// texture, and — when `samples > 1` — the multi-sampled color view the passes
/// actually render into.
fn create_targets(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    samples: u32,
) -> (
    wgpu::Texture,
    wgpu::TextureView,
    wgpu::TextureView,
    Option<wgpu::TextureView>,
) {
    let samples = samples.max(1);
    let extent = wgpu::Extent3d {
        width: width.max(MIN_TARGET_DIM),
        height: height.max(MIN_TARGET_DIM),
        depth_or_array_layers: 1,
    };
    let color = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("openforge-color"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: COLOR_FORMAT,
        // COPY_SRC keeps the target readable for the headless smoke test.
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let msaa_view = (samples > 1).then(|| {
        let msaa = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("openforge-msaa"),
            size: extent,
            mip_level_count: 1,
            sample_count: samples,
            dimension: wgpu::TextureDimension::D2,
            format: COLOR_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        msaa.create_view(&wgpu::TextureViewDescriptor::default())
    });
    let depth = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("openforge-depth"),
        size: extent,
        mip_level_count: 1,
        sample_count: samples,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let color_view = color.create_view(&wgpu::TextureViewDescriptor::default());
    let depth_view = depth.create_view(&wgpu::TextureViewDescriptor::default());
    (color, color_view, depth_view, msaa_view)
}

/// The ID pass's attachments: the integer image its ids land in and a
/// single-sampled depth buffer with the visible pass's depth semantics, plus
/// the size they were built for.
struct PickTarget {
    size: [u32; 2],
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    depth_view: wgpu::TextureView,
}

impl PickTarget {
    /// Build the attachments for a `size` viewport.
    fn new(device: &wgpu::Device, size: [u32; 2]) -> Self {
        let extent = wgpu::Extent3d {
            width: size[0].max(MIN_TARGET_DIM),
            height: size[1].max(MIN_TARGET_DIM),
            depth_or_array_layers: 1,
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("openforge-pick-ids"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: PICK_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let depth = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("openforge-pick-depth"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        Self {
            size,
            view: texture.create_view(&wgpu::TextureViewDescriptor::default()),
            texture,
            depth_view: depth.create_view(&wgpu::TextureViewDescriptor::default()),
        }
    }
}

/// Row pitch for the ID readback: wgpu requires texture→buffer rows to be
/// padded to 256 bytes.
fn pick_row_bytes(width: u32) -> u32 {
    (width * 4).div_ceil(PICK_ROW_ALIGNMENT) * PICK_ROW_ALIGNMENT
}

fn create_dynamic_buffer(
    device: &wgpu::Device,
    label: &str,
    size: u64,
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: size.max(4),
        usage,
        mapped_at_creation: false,
    })
}

/// Grow `buffer` to fit `bytes` (rounding up to a power of two) and upload.
fn write_dynamic(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer: &mut wgpu::Buffer,
    capacity: &mut u64,
    usage: wgpu::BufferUsages,
    label: &str,
    bytes: &[u8],
) {
    if bytes.is_empty() {
        return;
    }
    let needed = bytes.len() as u64;
    if needed > *capacity {
        let grown = needed.next_power_of_two().max(4);
        *buffer = create_dynamic_buffer(device, label, grown, usage);
        *capacity = grown;
    }
    queue.write_buffer(buffer, 0, bytes);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh_upload::FaceShading;

    const EPS: f32 = 1e-4;

    #[test]
    fn srgb_transfer_function_matches_known_anchors() {
        assert!(srgb_channel_to_linear(0.0).abs() < EPS);
        assert!((srgb_channel_to_linear(1.0) - 1.0).abs() < EPS);
        // The linear segment below the sRGB knee is exactly /12.92.
        assert!((srgb_channel_to_linear(0.02) - 0.02 / 12.92).abs() < EPS);
        // Mid grey: sRGB 0.5 is noticeably darker in linear light.
        let mid = srgb_channel_to_linear(0.5);
        assert!(mid > 0.2 && mid < 0.22, "mid grey was {mid}");
        // Monotonic, and clamps out-of-range input.
        assert!(srgb_channel_to_linear(0.2) < srgb_channel_to_linear(0.8));
        assert!((srgb_channel_to_linear(-1.0)).abs() < EPS);
        assert!((srgb_channel_to_linear(2.0) - 1.0).abs() < EPS);
    }

    #[test]
    fn linear_from_srgb_u8_converts_each_channel_and_is_opaque() {
        let white = linear_from_srgb_u8([255, 255, 255]);
        for c in &white[..3] {
            assert!((c - 1.0).abs() < EPS);
        }
        assert_eq!(white[3], 1.0);
        let black = linear_from_srgb_u8([0, 0, 0]);
        assert_eq!(black, [0.0, 0.0, 0.0, 1.0]);
        // Channels are independent and ordered.
        let c = linear_from_srgb_u8([10, 128, 240]);
        assert!(c[0] < c[1] && c[1] < c[2]);
    }

    #[test]
    fn clear_color_is_dark_and_opaque() {
        let c = clear_color();
        assert_eq!(c.a, 1.0);
        for v in [c.r, c.g, c.b] {
            assert!((0.0..0.05).contains(&v), "clear channel was {v}");
            assert!(v < 0.11); // darker than the sRGB value it came from
        }
    }

    #[test]
    fn grid_has_two_lines_per_grid_value_and_axis_colors() {
        let half = 2;
        let grid = grid_vertices(half);
        let lines = (2 * half as usize + 1) * 2;
        assert_eq!(grid.len(), lines * 2);
        // Every vertex carries the unlit normal and unit-length shading data.
        for v in &grid {
            assert!((Vec3::from(v.normal).length() - 1.0).abs() < EPS);
            assert_eq!(v.uv, [0.0, 0.0]);
        }
        let axis_x = linear_from_srgb_u8(AXIS_X_SRGB);
        let axis_z = linear_from_srgb_u8(AXIS_Z_SRGB);
        let grid_color = linear_from_srgb_u8(GRID_SRGB);
        // Exactly one line in each direction is drawn in an axis color; the
        // rest are the plain grid color.
        let colored = grid.iter().filter(|v| v.color == axis_x || v.color == axis_z).count();
        assert_eq!(colored, 4);
        assert_eq!(grid.iter().filter(|v| v.color == grid_color).count(), grid.len() - 4);
        // Endpoints span the full extent.
        let max = grid
            .iter()
            .map(|v| Vec3::from(v.position).abs().max_element())
            .fold(0.0f32, f32::max);
        assert!((max - half as f32).abs() < EPS);
    }

    #[test]
    fn grid_handles_a_zero_extent() {
        let grid = grid_vertices(0);
        assert_eq!(grid.len(), 4); // one line per direction, both axes
        for v in &grid {
            assert_eq!(v.position, [0.0, 0.0, 0.0]);
        }
    }

    #[test]
    fn gizmo_is_three_axis_lines_of_unit_length() {
        let g = gizmo_vertices();
        assert_eq!(g.len(), 6);
        // Vertex pairs: origin then +1 along the matching world axis.
        for (axis, pair) in g.chunks(2).enumerate() {
            let [origin, tip] = [pair[0].position, pair[1].position];
            assert_eq!(origin, [0.0, 0.0, 0.0]);
            let mut expected = [0.0f32; 3];
            expected[axis] = 1.0;
            assert_eq!(tip, expected);
        }
    }

    #[test]
    fn gizmo_viewport_always_fits_inside_the_target() {
        for size in [
            [1, 1],
            [2, 640],
            [20, 20],
            [64, 64],
            [200, 100],
            [800, 500],
            [3840, 2160],
        ] {
            let [x, y, edge] = gizmo_viewport(size);
            let (w, h) = (size[0] as f32, size[1] as f32);
            assert!(edge >= 1.0, "{size:?} gave edge {edge}");
            assert!(x >= 0.0 && y >= 0.0, "{size:?} gave origin {x},{y}");
            assert!(x + edge <= w + 1e-3, "{size:?} overflows x: {x}+{edge}>{w}");
            assert!(y + edge <= h + 1e-3, "{size:?} overflows y: {y}+{edge}>{h}");
        }
        // On a normal-sized viewport it is the full 108px next to the margin,
        // pinned to the right edge and the top edge.
        let [x, y, edge] = gizmo_viewport([800, 500]);
        assert_eq!(edge, GIZMO_PIXELS);
        assert_eq!(y, GIZMO_MARGIN);
        assert!((x + edge + GIZMO_MARGIN - 800.0).abs() < 1e-3);
    }

    #[test]
    fn gizmo_view_proj_keeps_the_triad_centered_and_inside_clip_space() {
        let camera = OrbitCamera::default();
        let vp = gizmo_view_proj(&camera);
        // The origin maps to the middle of the gizmo (no translation leaks in).
        let origin = vp * Vec4::new(0.0, 0.0, 0.0, 1.0);
        assert!(origin.w > 0.0);
        let ndc = origin.truncate() / origin.w;
        assert!(ndc.x.abs() < EPS && ndc.y.abs() < EPS);
        assert!((0.0..=1.0).contains(&ndc.z));
        // Every axis tip stays inside the gizmo viewport.
        for axis in [Vec3::X, Vec3::Y, Vec3::Z] {
            let clip = vp * axis.extend(1.0);
            assert!(clip.w > 0.0);
            let ndc = clip.truncate() / clip.w;
            assert!(ndc.x.abs() <= 1.0 && ndc.y.abs() <= 1.0, "ndc {ndc:?} for {axis:?}");
            assert!((0.0..=1.0).contains(&ndc.z));
        }
    }

    #[test]
    fn gizmo_view_proj_rotates_with_the_camera_and_ignores_distance() {
        let mut near = OrbitCamera::default();
        let mut far = near;
        far.distance = 500.0;
        // Distance must not change the gizmo (rotation-only view).
        let a = gizmo_view_proj(&near) * Vec3::X.extend(1.0);
        let b = gizmo_view_proj(&far) * Vec3::X.extend(1.0);
        assert!((a - b).length() < EPS);

        let before = gizmo_view_proj(&near) * Vec3::X.extend(1.0);
        near.orbit(120.0, 0.0);
        let after = gizmo_view_proj(&near) * Vec3::X.extend(1.0);
        assert!((before - after).length() > 0.05);
    }

    #[test]
    fn vertex_layout_matches_gpu_vertex_size() {
        let layout = vertex_layout();
        assert_eq!(layout.array_stride, std::mem::size_of::<GpuVertex>() as u64);
        assert_eq!(layout.attributes.len(), 5);
        assert_eq!(layout.attributes[0].shader_location, 0);
        assert_eq!(layout.attributes[3].offset + 16, layout.attributes[4].offset);
        assert_eq!(layout.attributes[4].shader_location, 4);
        assert_eq!(layout.attributes[4].format, wgpu::VertexFormat::Float32);
        // The last attribute lands exactly at the end of the stride: no gaps a
        // shader could read garbage from.
        assert_eq!(layout.attributes[4].offset + 4, layout.array_stride);
    }

    #[test]
    fn cull_modes_map_to_wgpu_faces() {
        assert_eq!(face_from_cull(CullMode::None), None);
        assert_eq!(face_from_cull(CullMode::Back), Some(wgpu::Face::Back));
        assert_eq!(face_from_cull(CullMode::Front), Some(wgpu::Face::Front));
        // The default pass culls back faces, which is the scene pipeline.
        assert_eq!(face_from_cull(pipeline::default_pass().cull), Some(wgpu::Face::Back));
    }

    #[test]
    fn pick_vertex_layout_matches_pick_vertex_size() {
        let layout = pick_vertex_layout();
        assert_eq!(layout.array_stride, std::mem::size_of::<PickVertex>() as u64);
        assert_eq!(layout.attributes.len(), 2);
        assert_eq!(layout.attributes[0].shader_location, 0);
        assert_eq!(layout.attributes[0].format, wgpu::VertexFormat::Float32x3);
        assert_eq!(layout.attributes[1].shader_location, 1);
        assert_eq!(layout.attributes[1].offset, 12);
        assert_eq!(layout.attributes[1].format, wgpu::VertexFormat::Uint32);
        // The id lands exactly at the end of the stride: no gap between the
        // position and the id a shader could read garbage from.
        assert_eq!(layout.attributes[1].offset + 4, layout.array_stride);
    }

    #[test]
    fn pick_rows_are_padded_to_the_copy_alignment() {
        assert_eq!(pick_row_bytes(1), 256);
        assert_eq!(pick_row_bytes(64), 256);
        // A 65-pixel row is 260 bytes: the next multiple of 256.
        assert_eq!(pick_row_bytes(65), 512);
        assert_eq!(pick_row_bytes(128), 512);
        for width in [1u32, 7, 32, 63, 64, 65, 128, 129, 256] {
            let bytes = pick_row_bytes(width);
            assert_eq!(bytes % PICK_ROW_ALIGNMENT, 0, "width {width}");
            assert!(bytes >= width * 4, "width {width}");
        }
    }

    #[test]
    fn pick_format_is_an_integer_target() {
        assert_eq!(PICK_FORMAT, wgpu::TextureFormat::R32Uint);
        // Not a depth format and not blendable: an id must survive as-is.
        assert!(!PICK_FORMAT.is_depth_stencil_format());
        assert!(!PICK_FORMAT.is_srgb());
    }

    #[test]
    fn pick_radius_stays_sane() {
        // The clamp exists so a patch cannot grow into a full-screen readback.
        assert!(MAX_PICK_RADIUS >= 8 && MAX_PICK_RADIUS <= 128);
    }

    #[test]
    fn shaders_are_valid_wgsl() {
        for (name, src) in [
            ("vertex", pipeline::vertex_shader_wgsl()),
            ("fragment", pipeline::fragment_shader_wgsl()),
            ("pick", pipeline::pick_shader_wgsl()),
        ] {
            let module = naga::front::wgsl::parse_str(src).unwrap_or_else(|e| {
                panic!("{name} shader failed to parse: {}", e.emit_to_string(src))
            });
            let mut validator = naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            );
            validator
                .validate(&module)
                .unwrap_or_else(|e| panic!("{name} shader failed validation: {e:?}"));
        }
    }

    // --- Headless GPU smoke test -------------------------------------------------
    //
    // Real wgpu work, so it needs a graphics adapter. On machines without one
    // (a headless CI box with no Vulkan/Metal/D3D12 driver) it reports the skip
    // and returns instead of failing.

    const TARGET: u32 = 64;
    /// Offscreen rows must be 256-byte aligned for texture→buffer copies.
    const ROW_BYTES: u32 = 256;

    /// An adapter plus its device, or `None` when no GPU is available. The
    /// adapter stays around so tests can query format support.
    fn headless_device() -> Option<(wgpu::Adapter, wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            force_fallback_adapter: false,
            compatible_surface: None,
        }))?;
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("openforge-smoke"),
                ..Default::default()
            },
            None,
        ))
        .ok()?;
        Some((adapter, device, queue))
    }

    /// A unit cube centred on the origin, with flat normals.
    fn unit_cube() -> openforge_core::mesh::Mesh {
        let mut m = openforge_core::mesh::Mesh::new();
        for p in [
            Vec3::new(-0.5, -0.5, -0.5),
            Vec3::new(0.5, -0.5, -0.5),
            Vec3::new(0.5, 0.5, -0.5),
            Vec3::new(-0.5, 0.5, -0.5),
            Vec3::new(-0.5, -0.5, 0.5),
            Vec3::new(0.5, -0.5, 0.5),
            Vec3::new(0.5, 0.5, 0.5),
            Vec3::new(-0.5, 0.5, 0.5),
        ] {
            m.add_vertex(p);
        }
        for face in [
            [0u32, 1, 2, 3],
            [4, 7, 6, 5],
            [0, 4, 5, 1],
            [3, 2, 6, 7],
            [0, 3, 7, 4],
            [1, 5, 6, 2],
        ] {
            m.add_face(&face);
        }
        m.compute_flat_normals();
        m
    }

    fn read_pixels(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        renderer: &GpuRenderer,
    ) -> Vec<u8> {
        let size = renderer.size();
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("openforge-readback"),
            size: (ROW_BYTES * size[1]).max(ROW_BYTES) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("openforge-readback"),
        });
        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: renderer.color_texture(),
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(ROW_BYTES),
                    rows_per_image: Some(size[1]),
                },
            },
            wgpu::Extent3d {
                width: size[0],
                height: size[1],
                depth_or_array_layers: 1,
            },
        );
        queue.submit(Some(encoder.finish()));

        let slice = buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::Maintain::Wait);
        let data = slice.get_mapped_range().to_vec();
        buffer.unmap();
        data
    }

    /// Pixels that are neutral and bright: only lit mesh faces can produce
    /// them (the grid stays below 70 in its weakest channel).
    fn bright_neutral_pixels(pixels: &[u8], size: [u32; 2]) -> usize {
        (0..(size[0] * size[1]) as usize)
            .filter(|i| {
                let p = &pixels[i * 4..i * 4 + 4];
                p[0].min(p[1]).min(p[2]) > 120 && p[3] == 255
            })
            .count()
    }

    /// Pixels that only a selection-tinted face can produce (orange).
    fn orange_pixels(pixels: &[u8], size: [u32; 2]) -> usize {
        (0..(size[0] * size[1]) as usize)
            .filter(|i| {
                let p = &pixels[i * 4..i * 4 + 4];
                p[0] > 140 && p[0] > p[2].saturating_add(40) && p[1] > 80
            })
            .count()
    }

    #[test]
    fn renders_scene_offscreen_and_reflects_selection() {
        let Some((_adapter, device, queue)) = headless_device() else {
            eprintln!("skipping GPU smoke test: no wgpu adapter available");
            return;
        };

        let mut scene = Scene::new();
        let cube = scene.create_mesh("Cube", unit_cube());
        let selection = Selection::new();
        let camera = OrbitCamera::default();

        let mut renderer = GpuRenderer::new(&device, &queue, 1);
        renderer.resize(&device, [TARGET, TARGET]);
        assert_eq!(renderer.size(), [TARGET, TARGET]);

        // 1. The cube renders: some pixels are lit geometry, the rest are the
        //    dark background (plus the grid, which never reaches this
        //    brightness).
        let stats = renderer.render(&device, &queue, &mut scene, &selection, &NodeMaterials::default(), &camera, 0.0, &[]);
        assert!(stats.indices > 0 && stats.indices.is_multiple_of(3), "{stats:?}");
        assert_eq!(stats.draws, 3);
        let pixels = read_pixels(&device, &queue, &renderer);
        let lit = bright_neutral_pixels(&pixels, renderer.size());
        let total = (TARGET * TARGET) as usize;
        assert!(lit > 0, "nothing rendered: the offscreen target is all background");
        assert!(lit < total, "the background is missing: every pixel is geometry");
        assert_eq!(orange_pixels(&pixels, renderer.size()), 0, "unselected cube is tinted");

        // 2. Hiding the only mesh leaves an empty (but still drawn) frame.
        if let Ok(node) = scene.get_mut(cube) {
            node.visible = false;
        }
        renderer.render(&device, &queue, &mut scene, &selection, &NodeMaterials::default(), &camera, 0.0, &[]);
        let hidden = read_pixels(&device, &queue, &renderer);
        assert_eq!(
            bright_neutral_pixels(&hidden, renderer.size()),
            0,
            "geometry was still drawn for a hidden node"
        );

        // 3. Selecting it changes the shading, which proves the per-vertex
        //    selection tint reaches the GPU.
        if let Ok(node) = scene.get_mut(cube) {
            node.visible = true;
        }
        let mut selection = Selection::new();
        selection.select_node(cube);
        renderer
            .render(&device, &queue, &mut scene, &selection, &NodeMaterials::default(), &camera, 0.0, &[]);
        let selected = read_pixels(&device, &queue, &renderer);
        assert!(
            orange_pixels(&selected, renderer.size()) > 0,
            "selected mesh was not tinted"
        );
        assert_ne!(selected, hidden);

        // 4. The edge overlay draws as a line pass of its own: a warm line
        //    floating clear of the cube shows up, and the stats count the
        //    extra draw.
        let unselected = Selection::new();
        renderer.render(&device, &queue, &mut scene, &unselected, &NodeMaterials::default(), &camera, 0.0, &[]);
        let plain = read_pixels(&device, &queue, &renderer);
        assert_eq!(orange_pixels(&plain, renderer.size()), 0, "edge test baseline is tinted");
        // Below the cube and clear of its body from the default camera, so
        // only the line itself can paint these pixels.
        let n = KEY_LIGHT.normalize();
        let warm = linear_from_srgb_u8([255, 170, 60]);
        let line = [
            GpuVertex::new(Vec3::new(-0.5, -1.5, 0.0), n, [0.0, 0.0], warm),
            GpuVertex::new(Vec3::new(0.5, -1.5, 0.0), n, [0.0, 0.0], warm),
        ];
        let stats = renderer
            .render(&device, &queue, &mut scene, &unselected, &NodeMaterials::default(), &camera, 0.0, &line);
        assert_eq!(stats.draws, 4, "the edge pass is its own draw");
        let edged = read_pixels(&device, &queue, &renderer);
        assert!(
            orange_pixels(&edged, renderer.size()) > 0,
            "the edge overlay never reached the target"
        );
    }

    /// Pixels dominated by a two-channel hue: both `high` channels bright
    /// against the `low` one. Only a face coloured with a saturated material can
    /// produce them — the background, the grid's axis lines and the corner gizmo
    /// are all single-channel (red/green/blue).
    fn hue_pixels(pixels: &[u8], size: [u32; 2], high: (usize, usize), low: usize) -> usize {
        (0..(size[0] * size[1]) as usize)
            .filter(|&i| {
                let p = &pixels[i * 4..i * 4 + 4];
                p[3] == 255
                    && p[high.0] > p[low].saturating_add(40)
                    && p[high.1] > p[low].saturating_add(40)
            })
            .count()
    }

    /// Pixels a single channel owns outright: `channel` lit, the other two
    /// within a texel of black. Only a face sampling a saturated texel can
    /// produce them — the grid's axis lines and the corner gizmo keep their
    /// off-channels above 70.
    fn pure_channel_pixels(pixels: &[u8], size: [u32; 2], channel: usize) -> usize {
        (0..(size[0] * size[1]) as usize)
            .filter(|&i| {
                let p = &pixels[i * 4..i * 4 + 4];
                let others = (0..3)
                    .filter(|c| *c != channel)
                    .map(|c| p[c])
                    .max()
                    .unwrap_or(0);
                p[3] == 255 && p[channel] > 40 && others <= 8
            })
            .count()
    }

    #[test]
    fn per_face_material_colours_reach_the_gpu_per_face() {
        let Some((_adapter, device, queue)) = headless_device() else {
            eprintln!("skipping GPU per-face material test: no wgpu adapter available");
            return;
        };
        let mut scene = Scene::new();
        let cube = scene.create_mesh("Cube", unit_cube());
        let selection = Selection::new();
        let camera = OrbitCamera::default();
        let mut renderer = GpuRenderer::new(&device, &queue, 1);
        renderer.resize(&device, [TARGET, TARGET]);
        let size = renderer.size();

        let magenta = linear_from_srgb_u8([255, 20, 255]);
        let yellow = linear_from_srgb_u8([255, 255, 20]);
        // Magenta on the ±Z and ±Y faces, yellow on the ±X pair: a cube shows one
        // face per axis pair however it is viewed, so both colours are in view
        // from any angle. Neither hue exists anywhere else in the frame, so an
        // implementation that shaded the whole node with one colour could paint
        // one of them and never both.
        let mut faces: Vec<crate::mesh_upload::FaceShading> = vec![magenta.into(); 6];
        faces[4] = yellow.into();
        faces[5] = yellow.into();
        renderer.render(
            &device,
            &queue,
            &mut scene,
            &selection,
            &NodeMaterials::new().with_faces(cube, faces),
            &camera,
            0.0,
            &[],
        );
        let pixels = read_pixels(&device, &queue, &renderer);
        assert!(
            hue_pixels(&pixels, size, (0, 2), 1) > 0,
            "no magenta face reached the target"
        );
        assert!(
            hue_pixels(&pixels, size, (0, 1), 2) > 0,
            "no yellow face reached the target: the node was shaded with one colour"
        );

        // Control: one colour for the whole node paints no yellow at all, which
        // is what makes the assertion above about the per-face colours.
        let all_magenta = NodeMaterials::new().with_faces(cube, vec![magenta.into(); 6]);
        renderer.render(&device, &queue, &mut scene, &selection, &all_magenta, &camera, 0.0, &[]);
        let pixels = read_pixels(&device, &queue, &renderer);
        assert_eq!(
            hue_pixels(&pixels, size, (0, 1), 2),
            0,
            "yellow came from somewhere other than the per-face colours"
        );
    }

    #[test]
    fn a_faces_texture_layer_reaches_the_sampler() {
        let Some((_adapter, device, queue)) = headless_device() else {
            eprintln!("skipping GPU texture test: no wgpu adapter available");
            return;
        };

        let mut scene = Scene::new();
        let cube = scene.create_mesh("Cube", unit_cube());
        let selection = Selection::new();
        let camera = OrbitCamera::default();
        let size = [TARGET, TARGET];

        let mut renderer = GpuRenderer::new(&device, &queue, 1);
        renderer.resize(&device, size);
        // Two 1x1 textures: layer 1 is pure red, layer 2 pure green. With four
        // texels the test would be at the sampler's mercy; one texel per layer
        // is the whole layer however it is filtered.
        renderer.set_textures(
            &device,
            &queue,
            &[
                TextureLayer::solid(1, 1, [255, 0, 0, 255]),
                TextureLayer::solid(1, 1, [0, 255, 0, 255]),
            ],
        );
        assert_eq!(renderer.texture_size(), [1, 1]);
        assert_eq!(renderer.texture_layers(), 3, "white + two textures");

        // White material colours, so every hue in the frame comes from a texel.
        // The two faces of the Y axis pair carry the green texture: one of them
        // faces the camera from any orbit, so both hues are always in view.
        let white = [1.0, 1.0, 1.0, 1.0];
        let mut faces = vec![FaceShading::textured(white, 0); 6];
        faces[4] = FaceShading::textured(white, 1);
        faces[5] = FaceShading::textured(white, 1);
        let textured = NodeMaterials::new().with_faces(cube, faces);
        renderer.render(&device, &queue, &mut scene, &selection, &textured, &camera, 0.0, &[]);
        let pixels = read_pixels(&device, &queue, &renderer);
        assert!(
            pure_channel_pixels(&pixels, size, 0) > 0,
            "no red texel reached the target: texture 0 was not sampled"
        );
        assert!(
            pure_channel_pixels(&pixels, size, 1) > 0,
            "no green texel reached the target: the per-face layer was ignored"
        );

        // Control: the same colours with the texture off paint only neutral
        // pixels, so the hues really came from sampling.
        let flat = NodeMaterials::new().with_faces(cube, vec![white.into(); 6]);
        renderer.render(&device, &queue, &mut scene, &selection, &flat, &camera, 0.0, &[]);
        let pixels = read_pixels(&device, &queue, &renderer);
        assert_eq!(pure_channel_pixels(&pixels, size, 0), 0, "red without a texture");
        assert_eq!(pure_channel_pixels(&pixels, size, 1), 0, "green without a texture");
        assert!(
            bright_neutral_pixels(&pixels, size) > 0,
            "the untextured cube should still render"
        );
    }

    #[test]
    fn set_textures_rebuilds_the_array_and_falls_back_to_white() {
        let Some((_adapter, device, queue)) = headless_device() else {
            eprintln!("skipping GPU texture test: no wgpu adapter available");
            return;
        };
        let mut renderer = GpuRenderer::new(&device, &queue, 1);
        renderer.resize(&device, [TARGET, TARGET]);

        // A green layer whose pixel buffer is far too short: the renderer fills
        // it white rather than leaving the layer as it was.
        let mut scene = Scene::new();
        let cube = scene.create_mesh("Cube", unit_cube());
        let selection = Selection::new();
        let camera = OrbitCamera::default();
        renderer.set_textures(
            &device,
            &queue,
            &[TextureLayer::solid(4, 4, [255, 0, 0, 255]), TextureLayer::new(4, 4, vec![0, 255, 0, 255])],
        );
        assert_eq!(renderer.texture_size(), [4, 4]);
        assert_eq!(renderer.texture_layers(), 3);

        let white = [1.0, 1.0, 1.0, 1.0];
        let materials =
            NodeMaterials::new().with_faces(cube, vec![FaceShading::textured(white, 1); 6]);
        renderer.render(&device, &queue, &mut scene, &selection, &materials, &camera, 0.0, &[]);
        let pixels = read_pixels(&device, &queue, &renderer);
        assert_eq!(
            pure_channel_pixels(&pixels, [TARGET, TARGET], 1),
            0,
            "the short green layer should have been filled with white"
        );
        assert!(bright_neutral_pixels(&pixels, [TARGET, TARGET]) > 0);

        // An empty set goes back to a single white layer, and the bind groups
        // that hold the old array are rebuilt with it.
        renderer.set_textures(&device, &queue, &[]);
        assert_eq!(renderer.texture_size(), [MIN_TARGET_DIM, MIN_TARGET_DIM]);
        assert_eq!(renderer.texture_layers(), 1);
        renderer.render(&device, &queue, &mut scene, &selection, &materials, &camera, 0.0, &[]);
        let pixels = read_pixels(&device, &queue, &renderer);
        assert!(bright_neutral_pixels(&pixels, [TARGET, TARGET]) > 0);
    }

    #[test]
    fn resize_is_clamped_and_a_no_op_when_unchanged() {
        let Some((_adapter, device, queue)) = headless_device() else {
            eprintln!("skipping GPU resize test: no wgpu adapter available");
            return;
        };
        let mut renderer = GpuRenderer::new(&device, &queue, 1);
        assert_eq!(renderer.size(), [MIN_TARGET_DIM, MIN_TARGET_DIM]);
        renderer.resize(&device, [0, 0]);
        assert_eq!(renderer.size(), [MIN_TARGET_DIM, MIN_TARGET_DIM]);
        renderer.resize(&device, [200, 100]);
        assert_eq!(renderer.size(), [200, 100]);
        // Clamped to the maximum.
        renderer.resize(&device, [u32::MAX, 10]);
        assert_eq!(renderer.size(), [MAX_TARGET_DIM, 10]);
        // Idempotent.
        let before = renderer.color_texture().width();
        renderer.resize(&device, [MAX_TARGET_DIM, 10]);
        assert_eq!(renderer.color_texture().width(), before);
    }

    #[test]
    fn multisampled_target_resolves_into_the_presented_texture() {
        let Some((adapter, device, queue)) = headless_device() else {
            eprintln!("skipping GPU MSAA test: no wgpu adapter available");
            return;
        };
        // Ask for the preferred count the way the app does; a device that
        // cannot do 4x reports 1 and is exercised single-sampled below.
        assert_eq!(supported_samples(&adapter, COLOR_FORMAT, 0), 1);
        let samples = supported_samples(&adapter, COLOR_FORMAT, PREFERRED_SAMPLES);
        assert!(samples == 1 || samples == PREFERRED_SAMPLES, "got {samples}");

        let mut scene = Scene::new();
        scene.create_mesh("Cube", unit_cube());
        let camera = OrbitCamera::default();
        let mut renderer = GpuRenderer::new(&device, &queue, samples);
        assert_eq!(renderer.samples(), samples);
        renderer.resize(&device, [TARGET, TARGET]);
        assert_eq!(renderer.size(), [TARGET, TARGET]);

        let stats = renderer.render(
            &device,
            &queue,
            &mut scene,
            &Selection::new(),
            &NodeMaterials::default(),
            &camera,
            0.0,
            &[],
        );
        assert!(stats.draws >= 3, "{stats:?}");
        // The readback goes through the resolve: a broken MSAA configuration
        // leaves this black (or fails validation before we get here).
        let pixels = read_pixels(&device, &queue, &renderer);
        assert!(
            bright_neutral_pixels(&pixels, renderer.size()) > 0,
            "nothing resolved into the color target with {samples}x MSAA"
        );
    }

    /// A 2x2 quad in the XY plane facing +Z, sharing its four corners.
    fn unit_quad() -> openforge_core::mesh::Mesh {
        let mut m = openforge_core::mesh::Mesh::new();
        m.add_vertex(Vec3::new(-1.0, -1.0, 0.0));
        m.add_vertex(Vec3::new(1.0, -1.0, 0.0));
        m.add_vertex(Vec3::new(1.0, 1.0, 0.0));
        m.add_vertex(Vec3::new(-1.0, 1.0, 0.0));
        m.add_face(&[0, 1, 2, 3]);
        m
    }

    /// A mesh holding a single loose vertex, and no faces at all.
    fn lone_vertex() -> openforge_core::mesh::Mesh {
        let mut m = openforge_core::mesh::Mesh::new();
        m.add_vertex(Vec3::ZERO);
        m
    }

    /// A camera looking down -Z from +Z, so "in front" is +Z.
    fn front_camera() -> OrbitCamera {
        OrbitCamera {
            yaw: std::f32::consts::FRAC_PI_2,
            pitch: 0.0,
            distance: 4.0,
            ..OrbitCamera::default()
        }
    }

    /// The offscreen pixel a world point projects to, using the same mapping
    /// the app's `viewport::project` uses (top-left origin, y down).
    fn pixel_of(camera: &OrbitCamera, size: [u32; 2], p: Vec3) -> [f32; 2] {
        let aspect = size[0] as f32 / size[1].max(1) as f32;
        let clip = camera.view_proj(aspect) * p.extend(1.0);
        let ndc = clip.truncate() / clip.w;
        [
            (ndc.x * 0.5 + 0.5) * size[0] as f32,
            (0.5 - ndc.y * 0.5) * size[1] as f32,
        ]
    }

    #[test]
    fn the_pick_pass_resolves_what_the_frame_shows() {
        let Some((_adapter, device, queue)) = headless_device() else {
            eprintln!("skipping GPU pick test: no wgpu adapter available");
            return;
        };
        let mut scene = Scene::new();
        let front = scene.create_mesh("front", unit_quad());
        let back = scene.create_mesh("back", unit_quad());
        scene.get_mut(front).unwrap().set_translation(Vec3::new(0.0, 0.0, 0.5));
        scene.get_mut(back).unwrap().set_translation(Vec3::new(0.0, 0.0, -0.5));
        let camera = front_camera();

        let mut renderer = GpuRenderer::new(&device, &queue, 1);
        renderer.resize(&device, [TARGET, TARGET]);
        let centre = [TARGET as f32 / 2.0, TARGET as f32 / 2.0];
        let front_face = PickElement::Face { node: front, face: 0 };
        let back_face = PickElement::Face { node: back, face: 0 };

        // The nearer quad's face is the only thing under a cursor at the
        // centre: the other is behind it, and the quad corners are ~15 px away
        // so no vertex element can reach the patch.
        let hits = renderer.pick(&device, &queue, &mut scene, &camera, centre, 4);
        assert_eq!(hits, vec![front_face], "the near face must win");

        // A cursor clear of every quad resolves nothing at all.
        let hits = renderer.pick(&device, &queue, &mut scene, &camera, [2.0, 2.0], 4);
        assert!(hits.is_empty(), "empty space picked {hits:?}");

        // Hiding the near node lets the pick fall through to the visible one.
        scene.get_mut(front).unwrap().visible = false;
        let hits = renderer.pick(&device, &queue, &mut scene, &camera, centre, 4);
        assert_eq!(hits, vec![back_face], "the pick must follow visibility");
        scene.get_mut(front).unwrap().visible = true;

        // Reversing the near quad's winding back-culls it from the frame, so
        // the pick must not find it either.
        scene.get_mut(front).unwrap().mesh_mut().unwrap().faces[0].corners.reverse();
        let hits = renderer.pick(&device, &queue, &mut scene, &camera, centre, 4);
        assert_eq!(hits, vec![back_face], "a back-culled face must not be pickable");
    }

    #[test]
    fn a_vertex_is_picked_at_its_pixel_unless_occluded() {
        let Some((_adapter, device, queue)) = headless_device() else {
            eprintln!("skipping GPU vertex pick test: no wgpu adapter available");
            return;
        };
        let mut scene = Scene::new();
        let quad = scene.create_mesh("quad", unit_quad());
        // A loose vertex (a node with no faces at all), right behind the quad's
        // middle to start with.
        let loose = scene.create_mesh("loose", lone_vertex());
        scene.get_mut(loose).unwrap().set_translation(Vec3::new(0.0, 0.0, -1.0));
        let camera = front_camera();

        let mut renderer = GpuRenderer::new(&device, &queue, 1);
        renderer.resize(&device, [TARGET, TARGET]);
        let size = renderer.size();

        // A cursor on the quad's (-1,-1) corner picks that vertex, not the face
        // covering the same pixel.
        let corner = pixel_of(&camera, size, Vec3::new(-1.0, -1.0, 0.0));
        let hits = renderer.pick(&device, &queue, &mut scene, &camera, corner, 2);
        assert_eq!(
            hits.first(),
            Some(&PickElement::Vertex { node: quad, vertex: 0 }),
            "a vertex on its own pixel must win over its face: {hits:?}"
        );

        // The loose vertex projects to the centre pixel, but it sits behind the
        // quad there: the frame does not show it, so the pick must not find it.
        let centre = pixel_of(&camera, size, Vec3::ZERO);
        let hits = renderer.pick(&device, &queue, &mut scene, &camera, centre, 2);
        assert!(
            hits.iter().all(|e| !matches!(e, PickElement::Vertex { node, .. } if *node == loose)),
            "an occluded vertex was picked: {hits:?}"
        );
        assert_eq!(hits.first(), Some(&PickElement::Face { node: quad, face: 0 }));

        // Move it in front of the quad: now the point pass draws it over the
        // face there, so it resolves — while the same vertex behind the quad
        // did not.
        scene.get_mut(loose).unwrap().set_translation(Vec3::new(0.0, 0.0, 1.0));
        let hits = renderer.pick(&device, &queue, &mut scene, &camera, centre, 2);
        assert!(
            hits.contains(&PickElement::Vertex { node: loose, vertex: 0 }),
            "a vertex in front of the surface must be picked: {hits:?}"
        );
        assert!(
            hits.contains(&PickElement::Face { node: quad, face: 0 }),
            "the face behind the vertex should still resolve: {hits:?}"
        );
    }

    #[test]
    fn a_face_rectangle_finds_what_the_frame_shows() {
        let Some((_adapter, device, queue)) = headless_device() else {
            eprintln!("skipping GPU rectangle test: no wgpu adapter available");
            return;
        };
        let mut scene = Scene::new();
        let front = scene.create_mesh("front", unit_quad());
        let back = scene.create_mesh("back", unit_quad());
        scene.get_mut(front).unwrap().set_translation(Vec3::new(0.0, 0.0, 0.5));
        scene.get_mut(back).unwrap().set_translation(Vec3::new(0.0, 0.0, -0.5));
        let camera = front_camera();

        let mut renderer = GpuRenderer::new(&device, &queue, 1);
        renderer.resize(&device, [TARGET, TARGET]);
        let all = [0, 0, TARGET, TARGET];

        // Over the whole viewport the frame shows one of the two quads, so
        // one face is all a rectangle may select.
        let hits = renderer.pick_rect(&device, &queue, &mut scene, &camera, all, SelectLevel::Face);
        assert_eq!(hits, vec![PickElement::Face { node: front, face: 0 }]);

        // A corner rectangle, clear of both quads, selects nothing.
        let corner = [0, 0, 8, 8];
        let hits = renderer.pick_rect(&device, &queue, &mut scene, &camera, corner, SelectLevel::Face);
        assert!(hits.is_empty(), "empty space selected {hits:?}");

        // A rectangle of no width, or one inside out, never reaches the pass.
        let hits = renderer.pick_rect(&device, &queue, &mut scene, &camera, [4, 4, 4, 20], SelectLevel::Face);
        assert!(hits.is_empty());
        let hits = renderer.pick_rect(&device, &queue, &mut scene, &camera, [20, 20, 4, 4], SelectLevel::Face);
        assert!(hits.is_empty());

        // Hiding the near node lets the rectangle see the far one instead.
        scene.get_mut(front).unwrap().visible = false;
        let hits = renderer.pick_rect(&device, &queue, &mut scene, &camera, all, SelectLevel::Face);
        assert_eq!(hits, vec![PickElement::Face { node: back, face: 0 }]);

        // There is no edge pass, so an edge rectangle is empty by definition.
        let hits = renderer.pick_rect(&device, &queue, &mut scene, &camera, all, SelectLevel::Edge);
        assert!(hits.is_empty());
    }

    #[test]
    fn a_face_rectangle_is_not_punched_through_by_its_own_vertices() {
        let Some((_adapter, device, queue)) = headless_device() else {
            eprintln!("skipping GPU rectangle test: no wgpu adapter available");
            return;
        };
        let mut scene = Scene::new();
        let quad = scene.create_mesh("quad", unit_quad());
        let camera = front_camera();

        let mut renderer = GpuRenderer::new(&device, &queue, 1);
        renderer.resize(&device, [TARGET, TARGET]);
        let size = renderer.size();

        // A small rectangle around the quad's (-1,-1) corner. The vertex point
        // pass draws one pixel there; a face rectangle must still find the
        // face, not a one-pixel hole in it.
        let corner = pixel_of(&camera, size, Vec3::new(-1.0, -1.0, 0.0));
        let x = corner[0].floor().max(0.0) as u32;
        let y = corner[1].floor().max(0.0) as u32;
        let around = [x.saturating_sub(1), y.saturating_sub(1), x + 2, y + 2];
        let hits = renderer.pick_rect(&device, &queue, &mut scene, &camera, around, SelectLevel::Face);
        assert_eq!(
            hits,
            vec![PickElement::Face { node: quad, face: 0 }],
            "the vertex point must not punch a hole in the face"
        );

        // The same rectangle at vertex level finds the corner vertex, and
        // answers with vertices alone: the face pixels are only there to hide
        // the vertices the frame does not show.
        let hits = renderer.pick_rect(&device, &queue, &mut scene, &camera, around, SelectLevel::Vertex);
        assert!(
            hits.contains(&PickElement::Vertex { node: quad, vertex: 0 }),
            "the corner vertex is not in {hits:?}"
        );
        assert!(
            hits.iter().all(|e| matches!(e, PickElement::Vertex { .. })),
            "a vertex rectangle must not answer with faces: {hits:?}"
        );
    }

    #[test]
    fn a_vertex_rectangle_finds_only_the_vertices_the_frame_shows() {
        let Some((_adapter, device, queue)) = headless_device() else {
            eprintln!("skipping GPU rectangle test: no wgpu adapter available");
            return;
        };
        let mut scene = Scene::new();
        let quad = scene.create_mesh("quad", unit_quad());
        let loose = scene.create_mesh("loose", lone_vertex());
        scene.get_mut(loose).unwrap().set_translation(Vec3::new(0.0, 0.0, -1.0));
        let camera = front_camera();

        let mut renderer = GpuRenderer::new(&device, &queue, 1);
        renderer.resize(&device, [TARGET, TARGET]);
        let all = [0, 0, TARGET, TARGET];

        // The quad's four vertices are inside the rectangle; the loose one
        // sits behind the quad's face, which the frame draws over it.
        let hits = renderer.pick_rect(&device, &queue, &mut scene, &camera, all, SelectLevel::Vertex);
        for vertex in 0..4 {
            assert!(
                hits.contains(&PickElement::Vertex { node: quad, vertex }),
                "vertex {vertex} is missing from {hits:?}"
            );
        }
        assert!(
            hits.iter().all(|e| !matches!(e, PickElement::Vertex { node, .. } if *node == loose)),
            "an occluded vertex was selected: {hits:?}"
        );

        // In front of the quad it is drawn, so it is inside the rectangle.
        scene.get_mut(loose).unwrap().set_translation(Vec3::new(0.0, 0.0, 1.0));
        let hits = renderer.pick_rect(&device, &queue, &mut scene, &camera, all, SelectLevel::Vertex);
        assert!(
            hits.contains(&PickElement::Vertex { node: loose, vertex: 0 }),
            "a visible vertex must be selected: {hits:?}"
        );
    }
}
