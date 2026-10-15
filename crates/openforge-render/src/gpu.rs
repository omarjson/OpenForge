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
use crate::gpu_types::{FrameUniforms, GpuVertex};
use crate::mesh_upload::{build_scene_with, MeshBuffers, NodeColors};
use crate::pipeline::{self, CullMode};
use glam::{Mat4, Vec3, Vec4};
use openforge_core::scene::Scene;
use openforge_core::selection::Selection;
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
const VERTEX_ATTRIBUTES: [wgpu::VertexAttribute; 4] = [
    wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0, shader_location: 0 },
    wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 12, shader_location: 1 },
    wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 24, shader_location: 2 },
    wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 32, shader_location: 3 },
];

fn vertex_layout() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<GpuVertex>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &VERTEX_ATTRIBUTES,
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
    frame_uniforms: wgpu::Buffer,
    frame_bind_group: wgpu::BindGroup,
    gizmo_uniforms: wgpu::Buffer,
    gizmo_bind_group: wgpu::BindGroup,
    scene_pipeline: wgpu::RenderPipeline,
    grid_pipeline: wgpu::RenderPipeline,
    edge_pipeline: wgpu::RenderPipeline,
    gizmo_pipeline: wgpu::RenderPipeline,
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
    /// The offscreen target starts at 1x1; call [`resize`](Self::resize) with
    /// the real viewport size before the first frame.
    pub fn new(device: &wgpu::Device, samples: u32) -> Self {
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
            entries: &[wgpu::BindGroupLayoutEntry {
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
            }],
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
        let bind_group = |label: &str, buffer: &wgpu::Buffer| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                }],
            })
        };

        let frame_uniforms = uniform_buffer("openforge-frame-uniforms");
        let frame_bind_group = bind_group("openforge-frame-bind-group", &frame_uniforms);
        let gizmo_uniforms = uniform_buffer("openforge-gizmo-uniforms");
        let gizmo_bind_group = bind_group("openforge-gizmo-bind-group", &gizmo_uniforms);

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
            frame_uniforms,
            frame_bind_group,
            gizmo_uniforms,
            gizmo_bind_group,
            scene_pipeline,
            grid_pipeline,
            edge_pipeline,
            gizmo_pipeline,
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
        colors: &NodeColors,
        camera: &OrbitCamera,
        time: f32,
        edges: &[GpuVertex],
    ) -> FrameStats {
        self.upload(device, queue, scene, selection, colors);
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

    /// Merge the scene's visible meshes into the dynamic vertex/index buffers.
    fn upload(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scene: &mut Scene,
        selection: &Selection,
        colors: &NodeColors,
    ) {
        let (parts, _bounds) = build_scene_with(scene, selection, colors);
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
        assert_eq!(layout.attributes.len(), 4);
        assert_eq!(layout.attributes[0].shader_location, 0);
        assert_eq!(layout.attributes[3].offset + 16, layout.array_stride);
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
    fn shaders_are_valid_wgsl() {
        for (name, src) in [
            ("vertex", pipeline::vertex_shader_wgsl()),
            ("fragment", pipeline::fragment_shader_wgsl()),
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

        let mut renderer = GpuRenderer::new(&device, 1);
        renderer.resize(&device, [TARGET, TARGET]);
        assert_eq!(renderer.size(), [TARGET, TARGET]);

        // 1. The cube renders: some pixels are lit geometry, the rest are the
        //    dark background (plus the grid, which never reaches this
        //    brightness).
        let stats = renderer.render(&device, &queue, &mut scene, &selection, &NodeColors::default(), &camera, 0.0, &[]);
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
        renderer.render(&device, &queue, &mut scene, &selection, &NodeColors::default(), &camera, 0.0, &[]);
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
            .render(&device, &queue, &mut scene, &selection, &NodeColors::default(), &camera, 0.0, &[]);
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
        renderer.render(&device, &queue, &mut scene, &unselected, &NodeColors::default(), &camera, 0.0, &[]);
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
            .render(&device, &queue, &mut scene, &unselected, &NodeColors::default(), &camera, 0.0, &line);
        assert_eq!(stats.draws, 4, "the edge pass is its own draw");
        let edged = read_pixels(&device, &queue, &renderer);
        assert!(
            orange_pixels(&edged, renderer.size()) > 0,
            "the edge overlay never reached the target"
        );
    }

    #[test]
    fn resize_is_clamped_and_a_no_op_when_unchanged() {
        let Some((_adapter, device, _queue)) = headless_device() else {
            eprintln!("skipping GPU resize test: no wgpu adapter available");
            return;
        };
        let mut renderer = GpuRenderer::new(&device, 1);
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
        let mut renderer = GpuRenderer::new(&device, samples);
        assert_eq!(renderer.samples(), samples);
        renderer.resize(&device, [TARGET, TARGET]);
        assert_eq!(renderer.size(), [TARGET, TARGET]);

        let stats = renderer.render(
            &device,
            &queue,
            &mut scene,
            &Selection::new(),
            &NodeColors::default(),
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
}
