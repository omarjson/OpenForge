# OpenForge — Handoff for the Next Agent

This file is the onboarding doc for an AI agent (or human) continuing work on
**OpenForge**, a clean-room, open-source, cross-platform 3D modeler written in
Rust. Read it fully before touching code.

---

## 1. What this project is

OpenForge is an original 3D modeler **inspired by** (not a clone of) ZModeler3,
which was analyzed with the REA tool (`npx rea-agents@latest setup`) to learn its
*architecture* (a plugin host with 23 services, 9 tool plugins, 15 format
filters, native `.z3d` format) — **not** to copy any code.

It is a growing feature-by-feature reimplementation. The engine layer is largely
complete and the plugin host now exists as a first cut; the remaining work is
mostly application surface (GUI, GPU viewport, interactive editors) and
deepening the host (dynamic plugin loading, behaviour-carrying services).

## 2. HARD CONSTRAINTS — do not violate

- **Clean-room only.** Never decompile, reverse, or bypass the `zmStoneGuard.dll`
  DRM. Never copy proprietary ZModeler code. Inspiration from public behavior/
  architecture is fine; transliterating protected code is not.
- **Standard formats only.** Support OBJ, glTF/GLB, PLY, STL (and other open
  formats). Do **not** implement the proprietary `.z3d` format.
- **Cross-platform.** Must build and test on Windows, macOS, and Linux. CI runs
  all three.
- Push to `https://github.com/omarjson/OpenForge.git` (private repo, default
  branch `main`).

## 3. Environment setup (macOS host)

```bash
export PATH="$HOME/.cargo/bin:$PATH"   # rustup / cargo / rustc (Rust 1.99.0)
export PATH="$HOME/bin:$PATH"          # gh CLI (v2.102.0), already authed as omarjson
```

- Working dir: `/Users/omar/Documents/zm/openforge`
- Shell is `zsh`: **never** use `$status` as a variable name (it's reserved).
- CI: `gh run list --repo omarjson/OpenForge`; each run is ~3–5 min across 7
  jobs. Watch with `gh run watch <id> --repo omarjson/OpenForge --exit-status`.

## 4. Repo layout — 12 crates (571 `#[test]` fns; 573 tests pass, 2 doc-tests ignored)

| Crate | Tests | Purpose |
|-------|-------|---------|
| `openforge-core` | 50 | `Mesh`, `Scene`, nodes, selection, undo, tools. The data model. `uv_sphere` carries spherical UVs; **`Mesh::edges()` is the one edge enumeration** the whole app numbers edges by, `Selection::selected_faces` is the one face-set read (it answers only at the face level — an edge index is not a face index), and **`subselect` holds the Select menu's maths** (all/invert/grow/shrink plus the edge loop/ring walks) on that same edge list. `Node::lod` is the one game tag (untagged by default). |
| `openforge-io` | 13 | Import/export: OBJ, glTF/GLB, PLY, STL. |
| `openforge-ops` | 17 | `MeshOps` extension trait: bevel, bridge, mirror, triangulate, flip, delete_edges, edge_loops, ring, **taper/twist/bend deformers**. |
| `openforge-materials` | 35 | Phong/Lambert materials, colours, `Texture` (+ `resized` for shared-size texture arrays). |
| `openforge-render` | 102 | Camera/vertex/uniform plumbing **plus the wgpu viewport backend** (`gpu` feature): offscreen color+depth target (multi-sampled when supported), scene/grid/edge/gizmo pipelines, **a material texture array the scene pass samples**, **an ID pass that resolves exactly what the frame draws when picking — faces as triangles, edges as lines, vertices as points, through a click patch or a whole rectangle** (`pick.rs` holds the geometry/element table/patch maths), frame stats. |
| `openforge-animation` | 50 | Keyframe tracks, interpolation, clips, player. |
| `openforge-algorithms` | 24 | Primitives, Catmull-Clark, Perlin/fBm noise, L-systems. |
| `openforge-uv` | 53 | Planar/cylindrical/spherical/box projections, unwrap, pack, **UV islands by continuity with per-island atlas packing** (`islands.rs`), plus per-selection affine edits for the UV editor. |
| `openforge-boolean` | 9 | BSP CSG: union/subtract/intersect, watertight output, volume/watertight checks. |
| `openforge-rig` | 22 | Skeletal rigging + linear-blend skinning: bones, hierarchy, bind/pose, `SkinWeights`, `skin_mesh`. |
| `openforge-plugin` | 18 | The plugin host: a service registry, a tool registry (factories, not shared instances) and a format-filter registry that dispatches import/export on an extension. The engine's own tools and formats are ordinary plugins registered here. |
| `openforge-app` | 178 (bin) | eframe/egui GUI. `app.rs` (~6000 lines) + `gizmo.rs` (screen-space transform gizmo) + `snap.rs` (snapping settings + pure snap maths) + `viewport.rs` (CPU painter + picking, edge picking included, + the rectangle projection fallback) + `gpu_viewport.rs` (wgpu bridge + GPU pick mapping, click and rectangle) + `uv_editor.rs` (UV panel, texture background, island selection). The Select menu, toolbar row and hotkeys drive `core::subselect` (all/none/invert, grow/shrink, edge loop/ring); hide/isolate/unhide and the outliner's visibility box are undoable scene edits, **Select Through** (`occluding_picks`) is the one gate deciding whether a pick asks the ID pass or the projection, import/export (menu, dialogs, drops) all go through the plugin filter registry with the format named in every status, the **Timeline** plays `openforge-animation` clips over the scene (spin clips, Space, scrub), the **Game menu** tags LOD levels, places dummies and reports the triangle budget (`LodReport`), and the **Plugins panel** enables/disables each built-in and says what it brings (a disable rebuilds the host from the rest). |

## 5. Architecture & conventions (follow these exactly)

- **A plugin registration either applies completely or not at all.** A plugin
  fills a *staged* `PluginHost` of its own and only a clean registration is
  merged in, so a plugin that clashes on its second entry cannot leave its
  first behind. Names (service, tool, filter id, extension) are unique and a
  clash is an error naming the offender — never a silent replacement.
- **Extension traits, one impl per type-pair.** Rust allows only ONE
  `impl Trait for Type` per pair. Ops like `MeshOps` (openforge-ops) and `UvOps`
  (openforge-uv) are implemented for core's `Mesh` from *separate* crates. Their
  method bodies live as `pub(crate)` free functions + a single delegating impl
  block. Do not split into multiple `impl MeshOps for Mesh` blocks.
- **No inherent `impl` on foreign types.** You cannot write `impl Mesh { ... }`
  outside openforge-core. Use free functions or an extension trait.
- **Undo is snapshot-based.** `Scene::clone()` before/after; `Scene::content_eq`
  decides whether to record. In the app, wrap edits in
  `self.with_edit(label, |app| { ... })`. Live/previews (e.g. a slider dragging)
  should mutate directly **without** `with_edit` to avoid flooding the undo stack.
- **The viewport has two paths, picked at runtime.** When eframe runs its
  `wgpu` backend the app renders through `openforge-render`'s `gpu` feature:
  `gpu_viewport.rs` sizes/owns the offscreen target, registers its texture with
  egui and draws it via `painter.image`. With no GPU device the app falls back
  to the CPU painter in `viewport.rs`. Both share one camera state (`Camera`),
  bridged by `gpu_viewport::orbit_camera`; picking is always CPU-side because
  NDC math is identical for either path.
- **Colour space matters.** The offscreen target must be `Rgba8UnormSrgb` (the
  only format `egui_wgpu` accepts for native textures) and the shaders write
  *linear* colour. Anything authored as sRGB bytes must go through
  `openforge_render::gpu::linear_from_srgb_u8`; clear colours likewise.
- **Materials are resolved by the app, never by the renderer.** A mesh carries a
  slot list (`NodeKind::Mesh::material_slots`) and each face indexes it
  (`Face::material`); `Node::face_material_slot(face)` is the one place that
  resolves a face to a library slot (an out-of-range index falls back to slot 0).
  Assign a whole node with `Node::assign_material`, named faces with
  `Node::assign_material_to_faces` (it appends the slot to the node's list when
  new and leaves other faces alone). The renderer only ever receives shading —
  `NodeColors` (one colour per node) with `build_scene_with`, or `NodeMaterials`
  (one `FaceShading` — colour plus optional texture layer — per face) with
  `build_scene_materials`. Both viewport paths must resolve identically:
  `viewport::collect_triangles` for the CPU painter, the app's `node_materials()`
  for the GPU path, both through `viewport::material_base` (which folds in the
  node- and face-selection tints) and `viewport::face_texture` (which resolves a
  face's map, ignoring one the library cannot find).
- **Textures travel as array layers, and layer 0 is white.** `GpuRenderer`
  owns one `texture_2d_array` (repeat addressing, linear filtering) plus a
  sampler; the app hands it `TextureLayer`s through `set_textures`, in library
  order, starting at array layer 1. Layer 0 is a reserved white texel
  (`openforge_render::WHITE_LAYER`) that untextured faces name, so the fragment
  shader has no branch and no derivative non-uniformity: `colour * white` is the
  old flat shade. A vertex's `tex_layer` (a `f32`, from
  `FaceShading::layer` via `MeshBuffers`) joins the dedup key, so a textured and
  an untextured face meeting at a corner keep their own sampling. **Array layers
  must share one size**, so the app resamples every library texture to
  `gpu_viewport::TEXTURE_LAYER_SIZE` with `Texture::resized` before upload, on a
  library *revision* (`texture_epoch`), not per frame.
- **The CPU painter shades texture the same way, through egui.** A textured face
  is one `egui::Mesh` (`textured_triangle`) over an `egui::TextureHandle` cached
  per library texture; egui multiplies the texel by the vertex colour, which is
  the shaded `material_base` colour, exactly like the shader's
  `colour * texel * lit`. UVs are passed straight through: egui's texture origin
  (0,0) is the image's top-left, which is `Texture` row 0, which is `v = 0` —
  the same convention wgpu samples. Only the **UV editor** flips (`v = 0` at the
  *bottom* of the tile), because it draws the UV space v-up.
- **Sub-object picking is a GPU ID pass first, the CPU raycast second.** A
  face/vertex/edge-level click in the wgpu viewport goes through
  `GpuRenderer::pick` (`pick.rs` owns the geometry, the element table and the
  patch maths — no wgpu types, so all of it is headless-testable; the pass
  itself lives in `gpu.rs`) and only falls back to `viewport::pick_face` /
  `viewport::pick_vertex` / `viewport::pick_edge` when there is no wgpu backend
  or the patch holds nothing the current level can use
  (`gpu_viewport::chosen_hit` scans the nearest-first hits for the first one
  the level accepts, so a vertex point sitting on its face never stops a face
  click). The pass draws the frame's own geometry with the frame's culling and
  depth test into a separate, single-sampled `R32Uint` target, so a click can
  never land on something the image does not show — an occluded vertex stays
  unpicked and a back-culled face is gone. Ids are 1-based (`0` is the clear
  value) and **per node run face, then vertices, then edges** — edges were
  appended last on purpose, so ids older streams pinned never moved. A patch is
  clipped at the viewport edges, and its readback rows are padded to wgpu's
  256-byte copy alignment. See §8.
- **Rectangle selection is the Select tool's primary drag** (`app.rs`). The
  tool's own work is click-driven, so a primary drag never reaches it: the app
  keeps a live `marquee` rectangle in *absolute screen* positions, paints it
  over whichever backend drew the frame (`paint_marquee`, beside the gizmo
  overlay) and resolves it on release — the GPU's ID pass first
  (`GpuRenderer::pick_rect`, which answers per level), the projection in
  `viewport::marquee_elements` only on a machine without wgpu. A plain drag
  replaces the selection and **Shift** extends it; a rectangle the pointer came
  back from, with no area at all, is resolved as the click it ended as. It
  starts at the pointer's **press origin**, because egui only calls a press a
  drag after movement. Nothing here changes the scene, so nothing here is
  undoable. See §8.
- **Edges are numbered by `Mesh::edges()` and nothing else** (`openforge-core`).
  The mesh keeps no edge list: `edges()` walks the faces in order, each face's
  corners around it (the last back to the first), a pair enters the list the
  first time it appears, each pair is stored with the lower position index
  first, a face the visible pass would skip contributes no sides, and a pair
  repeating one position index is no edge. An edge index is therefore a
  function of the face list alone — stable while the faces do not move — and
  every edge picker answers from that one list: `viewport::pick_edge`, the
  `SelectLevel::Edge` arm of `viewport::marquee_elements`,
  `PickGeometry::build`'s line geometry and the endpoint lookups in
  `element_centroid` / `selected_vertices`. In the ID pass a level draws what it
  has to answer with **on top of the faces** — vertex points, or edge lines, or
  both for a click (`PickDraw`) — because a point or a line sitting exactly on
  its own face would otherwise punch a one-pixel hole in the face's pixels; the
  lines are a `LineList` pass with `CullMode::None` and `LessEqual` drawn
  between the faces and the points, so an edge wins the pixels it lies on and
  still loses to anything genuinely in front of it. See §8.
- **Element indices belong to their level, and the frame reads them only
  there** (`Selection::selected_faces` in `openforge-core`). The element set is
  plain `u32`s, and edge 0 is not face 0 (nor vertex 0) however much the number
  looks like one, so every path that shades a *face* by the selection asks
  `selected_faces`, which answers only while the face level is the one being
  edited: `viewport::collect_triangles` (the CPU tint), `collect_edges` (the
  `face_selected` flag) and the app's `node_materials` (the GPU per-face
  colours). The app's own `selected_faces` helper (the modeling ops' face list,
  and the UV panel's "faces selected in the viewport") asks the same question,
  so those ops refuse at any other level instead of acting on edge indices.
- **Edge level draws its own aids, and the frame's overlay falls back on it.**
  A pick has a tolerance but a line nobody can see is a line nobody can aim at,
  so an *off* overlay is drawn as the selected meshes' outline at the edge
  level (`overlay_mode` in `app.rs`, feeding both backends); the setting itself
  is untouched — `Z` still cycles it and the View menu still reports it. On top
  of that, `paint_selected_edges` marks every picked edge with a bold
  `ELEMENT_SELECTED_SRGB` line, the same hue a selected vertex's dot and a
  selected face's edges take, so "selected" reads as one colour at every level.
  Both viewports draw it the way they draw the vertex dots — projected through
  the shared camera, on top, unoccluded — while the GPU overlay itself stays
  depth-tested inside the frame (see §8).
- **The Select tools are pure maths on one mesh** (`openforge_core::subselect`).
  All / Invert / Grow / Shrink take a level plus an element set and return a
  new set; the app owns which nodes they run on (the selected mesh nodes, like
  the modeling ops) and the status line. Adjacency is the app's own topology
  rule: faces are neighbours through a shared mesh *edge* (never a bare
  corner), edges through a shared position, vertices through an edge, all of
  it from `Mesh::edges()` — so an element the visible pass skips is neither
  selectable nor anybody's neighbour. Shrink is grow's exact mirror
  (`shrink(grow(set))` is the set it started from, which is a test), and "All"
  is exactly the list a picker can answer with (`all_elements`). At the object
  level All / None / Invert work on the *nodes* — every visible one — while
  grow and shrink report that they belong to the sub-object levels. Hide /
  Isolate / Unhide are scene edits, one undo step each through `with_edit`:
  hiding drops the node from the selection too (the gizmo must not sit on
  something the frame no longer draws), isolate keeps it, and both show up in
  every path immediately because the collectors already skip hidden nodes.
  The **edge-level loop and ring walks** are the same idea in the other
  direction: `edge_loop` carries on *straight* through a vertex only when
  exactly four edges meet there, along the one that is neither the current
  edge nor one of its two fan neighbours (the other edges of the faces the
  current edge borders); `edge_ring` crosses a quad face to its opposite side
  and comes back through the face it did not just cross, so it runs *around* a
  mesh rather than back through itself and stops at a face that is not a quad
  or at a boundary edge (one face). Both answer in `Mesh::edges()` ids, both
  return sets, and the app adds the walk of *every* selected edge (L /
  Shift+L), so a press can only grow the selection — clearing is what "None"
  is for. One more switch sits over the picks: **Select Through** (P) turns
  the sub-object answers from the occluding ID pass over to the projection —
  the nearest element under the pointer, everything a rectangle covers,
  hidden geometry included — through the one gate both pick paths ask,
  `occluding_picks(select_through, has_backend)`. Without a wgpu backend the
  switch has nothing to change, because the projection is already the answer
  there.
- **The UV editor is its own panel, not a viewport mode** (`uv_editor.rs` owns
  the canvas math and panel state, `app.rs` owns the wiring).
  `uv_editor::island_selection` grows a UV selection to whole islands through
  the uv crate's partition, and the **Pack Islands** button calls
  `uv_pack_islands` (islands by UV *continuity*) while **Pack** is the older
  `uv_repack` (groups by coincident *positions*): the two group faces
  differently and are deliberately both offered. See §8 for its gotchas: canvas
  rect construction, the placeholder UV slot, and one undo entry per drag.
- **The transform gizmo is screen-space geometry** (`gizmo.rs`): the frame is
  projected through the same `Camera` as the viewport, so handles are a constant
  pixel length and picking is a 2D distance test. The layout is computed once by
  the input pass and stored on the gizmo for the paint pass to reuse; the grab
  is taken on the *press*, not on egui's later `drag_started`. See §8.
- **Snapping is decided per frame, in the app, and applied to the drag's
  measure** (`snap.rs` holds the settings and the pure maths). `Gizmo::apply`
  takes a `snap::Snap` plus an optional vertex *target*: the app resolves that
  target (the vertex nearest the cursor within `SNAP_RADIUS_PIXELS`, excluding
  the geometry the drag carries) because it needs the camera and the scene, and
  `snap_measure` turns both into the measure the drag uses. Grid mode snaps the
  *destination* of the grabbed point (so a drag lands on a grid value wherever
  it began), rotations and scales snap to the angle/factor steps, and a held
  **Ctrl** inverts whatever the toolbar says (Off becomes Grid, anything else
  becomes Off). Snapping covers the gizmo's axis-constrained drags; the
  free-form view-plane tool drag is deliberately unsnapped.
- **Animation is a view over a clip, not an edit.** The transport
  (`openforge_animation::Player`) owns the clock; the app advances it once per
  frame before anything paints (`advance_animation`, with `stable_dt` capped at
  0.25 s so a dragged window cannot jump the clip) and writes the sampled node
  transforms straight into the scene. No undo entry is pushed — whatever pose
  playback stops at is simply where the scene stands — and a scrub or a stop
  poses the scene through the same `pose_current`. Quaternion tracks blend
  along the shortest arc, so a 360° turn needs intermediate keys: the "Spin
  Selected" clip carries four quarter-turn keys, not one full-turn key.
  The timeline is another bottom panel, added after the UV editor in `update`.
- **Game tags ride the node; the LOD preview reuses visibility.** `Node::lod`
  is the one game tag: an `Option<u8>` (0 = highest detail), part of node
  equality, so a tag travels through undo and serialization like any other node
  data, and the default is untagged — an ordinary scene is unaffected. The Game
  menu's commands are ordinary `with_edit` edits: **Set LOD 0–3 / Clear** writes
  the tag, **Isolate LOD 0–3** hides everything the level does not hold through
  the same `node.visible` path as Isolate (so the viewport, the picks and `U`
  keep agreeing — no second visibility filter), and **Add Dummy** creates the
  attach point game formats carry beside their geometry, at the selection's
  centre (the average of the selected nodes' world positions; the origin with
  nothing selected). The model report is `LodReport` — a fold over mesh nodes,
  triangles per level with the untagged pile first and the totals last — shown
  in the Properties panel and held nowhere else.
- **The Plugins panel is the plugin host's one control, and a rebuild is all
  or nothing.** Every built-in carries a checkbox; turning one off calls
  `rebuild_plugins`, which builds a fresh `PluginHost` from the remaining
  built-ins — so a disabled plugin's formats, tools and services leave in one
  step and nothing half-registered is left behind. Each row's "what it brings"
  line is measured, not guessed: `plugin_infos` registers that plugin alone
  into a scratch host and reads its registries, so the panel cannot drift from
  the host. The refusals follow the active set: `unknown_format_line` names the
  formats that are known, or says every format plugin is off; and a disabled
  `openforge.core-tools` degrades `active_tool` to the Select fallback that was
  already there for a host without a built-in.
- **Bottom-panel order is load-bearing.** Add the status bar first, then other
  bottom panels, then the side panels, and the central viewport last: that is
  what keeps the viewport's rect clear of the panels below it.
- **`wgpu` is pinned once**, in `[workspace.dependencies]`, and must stay in
  lockstep with the wgpu eframe re-exports — otherwise the device/texture types
  crossing between eframe and `openforge-render` stop being the same types.
- **`.cargo/config.toml` sets `future-incompat-report.frequency = "never"`.**
  wgpu's Metal backend pulls in `block v0.1.6`, whose notice Cargo prints as a
  `warning:` line and would otherwise break the zero-warning bar. Inspect it
  again with `cargo report future-incompatibilities`.
- **Tests use an atomic-counter temp dir** helper where temp files are needed, to
  survive parallel `cargo test`.
- **Zero warnings is the bar.** `cargo build --workspace 2>&1 | grep -c warning`
  must be `0`. Run clippy-level clean code; no `#[allow]` unless justified.
- **The app's `app.rs` pattern for wiring a new engine:**
  1. Add crate dep to `crates/openforge-app/Cargo.toml`.
  2. Import it in `app.rs`.
  3. Add command methods (`fn *_selected(&mut self, ...)`) near the other
     commands, using `with_edit` for committed edits.
  4. Add a toolbar row in `handle_top_toolbar` (see Deform/Boolean/Rig rows).
  5. `cargo build -p openforge-app` until clean, then full workspace + CI.

## 6. Definition of done (for every change)

1. `cargo build --workspace` → **0 warnings**.
2. `cargo test --workspace` → **all pass, 0 failures**.
3. `git add -A && git commit` with a descriptive message; `git push origin main`.
4. Confirm CI green: **all 7 jobs** on Windows/macOS/Linux.
5. For a new crate: add to workspace `members` in the root `Cargo.toml` and give
   it tests (aim for 10–25 meaningful tests).

## 7. Current state & what's next

**State:** every commit since baseline `2a201c8` is landed on `main`. Full
workspace builds with **0 warnings**, **441 tests pass**, CI green on all 3
platforms. Eleven work-streams are done — the **wgpu viewport** (#1), the
**interactive UV editor** (#2), the **transform gizmo** (#3), the **edge
overlay** (#4), the **sub-object transforms** (#5), **materials** (#6), the
**plugin host** (#7), the **GPU edge overlay + MSAA** (#8), **per-face
materials** (#9), **snapping** (#10) and **textures end to end** (#11) — plus a
progress report:

- **`scripts/progress.sh`** draws a progress bar per parity area and can
  rewrite [`PROGRESS.md`](PROGRESS.md) (`--write`). It keeps *measured* counts
  (crates, test fns, lines of Rust) separate from *curated* per-area scores,
  because parity has no machine-checkable definition. Run it after every
  work-stream and re-score the area you touched.

- **`openforge-render` grew a real GPU backend** behind the `gpu` feature.
  `gpu::GpuRenderer` builds pipelines from the existing WGSL, keeps a growable
  scene vertex/index buffer, writes per-frame uniforms, and draws into an
  offscreen `Rgba8UnormSrgb` target **with a `Depth32Float` depth buffer** — the
  depth test replaces the old painter's-algorithm sort. It draws three passes:
  the ground grid (depth-tested, no depth write, so meshes always composite over
  it), the scene (depth test *and* write), and a corner axis gizmo using a
  rotation-only ortho camera that always draws on top.
- **The app presents that image through egui** (`gpu_viewport.rs`) and keeps the
  CPU painter as a fallback when no GPU device exists. Selection highlight comes
  from the per-vertex colours `MeshBuffers::build_scene` already produced.
- **19 new tests**: 16 in `openforge-render` (pure sRGB/grid/gizmo/viewport
  helpers, `MeshBuffers::append`, plus an adapter-guarded **headless GPU smoke
  test** that renders offscreen and reads pixels back) and 3 in `openforge-app`
  (format/rounding/camera-bridge). The WGSL is also validated with **naga** in a
  test that needs no GPU, so CI checks shader correctness on GPU-less runners.
- `.cargo/config.toml` and one `lsystem.rs` test rename keep both the build and
  the test build warning-free.
- **A working interactive UV editor.** `openforge-uv` gained per-selection ops —
  `uv_transform_selected` (scale, then rotate about a pivot, then translate),
  `uv_face_uvs` and `uv_bounds_selected` — because the existing `UvOps` could
  only transform *every* UV. `openforge-app` gained `uv_editor.rs` (canvas math
  + panel state + drawing) and an `app.rs` panel: a 2D canvas with the `0..1`
  tile, a quarter grid, the UV wireframe and a dot per UV; click and
  rubber-band selection (Shift adds); drag to move (one undo entry per drag);
  and buttons for the six projections, Normalize, Flip V, Pack (grid 1..8) and
  a scale/rotate transform about the selection's centre.
- **15 more tests** (6 in `openforge-uv`, 9 in `openforge-app`), all pure: the
  affine selection maths and subset queries, plus the canvas mapping, picking,
  band selection and selection-retarget rules.

- **A working transform gizmo**, and with it the transform tools actually
  started working. `gizmo.rs` (new, `openforge-app`) holds the whole thing: the
  world-space frame (arm/ring size derived from the camera so handles subtend a
  constant ~90 px), the projection to screen, `Layout::hit_axis`/`hit_ring`
  (2D distance to a segment / to a sampled ring), and the drag maths
  (`closest_point_on_line`, `ray_plane`, `plane_angle`, `Anchor::measure` →
  `Translate`/`Scale`/`Rotate` → `apply_measure`). `app.rs` wires it: `W`/`E`/`R`
  put handles on the selection, the hovered or grabbed axis lights up and is
  labelled, the drag is axis-constrained (it re-derives every node from its
  drag-start transform each frame, so it is idempotent), and the gesture lands
  as **one** undo entry on release.
- **Two real bugs fell out of that work.** `ToolContext::drag` was rebuilt as
  `None` every frame, so `TransformTool::on_pointer_move` returned immediately
  and *Move/Scale/Rotate drags never moved anything*; and the release branch was
  gated on `dragged_by(Primary)`, which is false on the release frame, so
  `on_pointer_up` never ran and drags produced no undo entry. The app now owns
  the drag state (`self.tool_drag`) and gates the release on `drag_stopped()`
  alone. A press that misses the handles therefore falls through to the
  free-form view-plane drag, which now works too.
- **27 tests**: 18 in `gizmo.rs` (pure math — pixel size, ring geometry,
  hit-testing, the three drag measures, transform application) and 9 in
  `app.rs`, which drive `handle_central_viewport` headlessly on a plain
  `egui::Context` with synthetic pointer events: hit the arm and only that arm,
  grab → drag → release moving the node along one axis and *not* Y/Z, one undo
  step restoring the start, the miss falling through to the free-form drag, the
  rotate rings spinning without translating, the scale ratio, and a shape-count
  check that the Move gizmo paints exactly three extra strokes while Rotate
  paints paths instead.

- **A wireframe / selection-outline overlay** over both viewport backends.
  `viewport::collect_edges` walks the visible meshes and returns deduplicated
  world-space edges, dropping any edge whose faces all point away from the eye
  (Newell normals, so non-planar n-gons behave) — without that cull a closed
  mesh would draw its own far side through the shading, since the CPU painter
  has no depth buffer to consult. `Wireframe::Off/Selection/All` cycles from
  `Z` and the View menu, and one paint routine serves both backends. Edges of
  selected faces are drawn in the warm highlight, of selected nodes in blue,
  and everything else dim.
- **13 tests**: 10 in `viewport.rs` (edge dedup, the facing cull — including a
  cube hiding exactly the three edges meeting at its far corner — mode and
  shortcut cycling, the visibility/frozen filters, face-selection flags) and 3
  in `app.rs` driving the overlay headlessly: the `Z` cycle, the `Selection`
  vs `All` stroke counts (exactly nine more for an unselected box), and the
  highlight appearing on the facing face's edges only.

- **The gizmo edits sub-objects, not just nodes.** A drag now carries a
  `gizmo::Targets`: whole node transforms at object level, or the vertices
  behind a vertex/face selection at element level. `viewport::element_centroid`
  puts the handles on the selected elements' centroid (rotating a face about
  the node's bounds centre looks like a bug) and `viewport::selected_vertices`
  lists the affected corners, deduplicated, with their drag-start local
  positions. The measure is world-space, so the app carries it through each
  node's world matrix (`world⁻¹ · M · world`) before touching mesh-local
  positions — one world unit of drag stays one world unit whatever the node's
  own transform is. Idempotence and the single undo entry work exactly as for
  nodes. `Selection::clear_elements` (new in core) fires when the user changes
  level, because face 3 and vertex 3 are different things and keeping the index
  would silently transform the wrong geometry.
- **11 tests**: 1 in `openforge-core` (`clear_elements` keeps the nodes), 5 in
  `gizmo.rs` (the measure matrix about a pivot; a vertex drag moving only its
  own vertices and leaving the node transform alone; sweeping the cursor
  without accumulating; the drag carried through the node's transform; a scale
  about the pivot), 4 in `viewport.rs` (element centroids for vertices and
  faces, the empty cases, corner dedup) and 1 end-to-end in `app.rs` — a
  vertex-level drag that asserts every other vertex is untouched.

- **A plugin host, and the app goes through it.** New crate `openforge-plugin`
  is the seam ZModeler's architecture is built around: a `ServiceRegistry`
  (named capabilities), a tool registry of **factories** (`fn() -> Box<dyn Tool>`,
  because a `Tool` carries an in-progress drag and sharing it would leak that
  state) and a `FilterRegistry` that dispatches import/export on a file's
  extension. The engine's own four tools and four formats are ordinary plugins
  (`CoreToolsPlugin`, `InterchangePlugin`) registered by `builtin_host()`, so
  the app never names a tool or a format again: `active_tool()` asks
  `host.make_tool(name)` and `import_file`/`export_selected` call
  `host.import`/`host.export`. Three rules keep it honest — names are unique
  (a duplicate service, tool, filter id *or* extension is an error naming the
  clash), registration is atomic (a plugin registers into a staged host and is
  merged only if it succeeds, so a rejected plugin leaves nothing behind), and
  the app's Properties panel gained a **Plugins** section listing the loaded
  plugins and the services, tools and formats they contributed. Import/export
  goes through it end to end (work-stream #19): the open dialog carries one
  filter group per format, **Export selected as** names them, files dropped on
  the window import through `import_files`, and `for_path` / `extensions`
  answer the refusals and the format labels every status rides on — a path no
  filter knows is turned down before anything is read or written, naming every
  extension that would have worked.
- **18 tests** in `openforge-plugin`: extension dispatch and its
  case/dot handling, the unique-name rules for all four kinds, the atomic
  rollback of a plugin that clashes on its *second* registration,
  import-only filters (with the `can_import` gate), the factory handing back a
  fresh tool per request, and an empty host staying usable.
- **7 more in `app.rs`**, which drive the seam end to end with no display:
  import and export round-tripping a real file through the host and leaving an
  undo entry, an empty selection writing no file, a `.dot` filter that fails to
  import until one plugin is registered and works afterwards, a plugin trying to
  steal `obj` being rejected with the host unchanged, the active tool coming
  from `make_tool`, and the Plugins panel's report listing exactly the two
  built-ins, three services, four tools and four formats.

- **Materials are wired through to the pixels.** `openforge-materials` already
  had `Material`, `MaterialLibrary` and textures; nothing used them, and
  `Face::material` / `NodeKind::Mesh::material_slots` were dead weight. Core now
  has `Node::assign_material(slot)` (whole node, returns whether anything
  changed) and `Node::face_material_slot(face)` (a face's index through the
  node's slot list, falling back to slot 0). The app owns a library seeded with
  the default grey plus Clay / Painted Red / Steel / Gold, and the Properties
  panel gained a **Materials** section: pick an entry, edit name, colour,
  shininess and opacity, Assign it to the selection (undoable), or duplicate it.
  `collect_triangles` resolves each face's slot, so the **CPU path shades per
  face**; `viewport::material_base` turns a material into viewport bytes and
  tints it towards the selection colours, so selection still reads on a meshed
  material. The **GPU path** gets a colour per node: `openforge-render` gained
  `NodeColors` and `build_scene_with`, because the renderer knows nothing about
  materials — the app resolves them and passes linear colours in. (Superseded by
  work-stream #9 below: the GPU path now takes a colour *per face* through
  `NodeMaterials`/`build_scene_materials`, and `NodeColors` remains for callers
  that want one colour for a whole node.)
- **The panels do more than report.** Rename (one undo step for the whole edit,
  snapshotted when the field takes focus, not one per keystroke), a visibility
  checkbox per outliner row, the material read-out in Properties, and **Ctrl+D**
  to duplicate the selection (an independent copy — the mesh is cloned — but a
  shallow one: children are not duplicated and the copy goes to the root).
- **18 tests**: 5 in `openforge-core` (material assignment, stale slots, the
  non-mesh no-op), 3 in `openforge-render` (`NodeColors` addressing,
  `build_scene_with` overriding and falling back), 5 in `viewport.rs`
  (`material_base` colour/alpha/clamping/tinting, and triangles carrying their
  face's resolved slot) and 7 in `app.rs` — including assigning a material and
  asserting the painted fills actually turn red, the linear colour handed to the
  renderer, refusing to delete an in-use or non-final material, and duplicating
  without aliasing the original mesh.

- **The GPU viewport grew MSAA and draws the wireframe itself.** The offscreen
  target now runs 4x multi-sampled when the adapter supports it: the app asks
  `openforge_render::gpu::supported_samples` (which never downgrades silently),
  `GpuRenderer::new` takes the sample count, the pipelines and the depth
  attachment match it, and every pass resolves into the single-sampled
  `Rgba8UnormSrgb` texture egui presents and the smoke test reads back. The
  wireframe / selection outline is no longer an egui overlay on the GPU path:
  the app converts `viewport::collect_edges`' world-space edges into line
  vertices (`gpu_viewport::edge_vertices`, the same `EDGE_*` colours the CPU
  painter uses) and the renderer draws them as a depth-tested `LessEqual` line
  pass of its own, so meshes occlude hidden edges while an edge lying exactly on
  its own shaded surface still draws. `FrameStats::draws` counts that pass
  (grid + scene + edges + gizmo). The CPU fallback keeps painting edges on top
  as before.
- **3 new tests**: 1 in `openforge-render` (an adapter-guarded MSAA smoke test
  that builds a 4x renderer, renders, and reads the resolved target back) and 2
  in `openforge-app` (`edge_vertices` pair/colour/normal rules — including the
  unlit `N·L == 1` normal that keeps edge colours authored — and the empty
  case). The existing edge-overlay smoke test also asserts the new line pass: a
  warm line floats clear of the cube, `draws` goes to 4 and the pixels arrive.

- **Materials shade per face on the GPU too, and faces can now be assigned
  one.** Work-stream #6 could *resolve* a face's slot but nothing could ever
  produce a multi-slot node, and the GPU path ignored the slot entirely (it
  shaded the node with its first face's material). Both halves are closed:
  - `MeshBuffers::build_with_face_colors(mesh, world, face_color)` resolves a
    colour per face and puts it in the dedup key — corners shared by faces of
    different colours become separate vertices, coincident corners of one colour
    still collapse, so `build` (one colour) keeps its old output exactly.
    `NodeMaterials` + `build_scene_materials` are the scene-level seam, mirroring
    `NodeColors`/`build_scene_with`; both now share one private walk
    (`build_scene_per_face`) that resolves a colour per (node, face) and falls
    back to the selection tint / default grey. The app's
    `node_materials()` mirrors `collect_triangles`: it resolves each face's slot,
    node selection and **face selection** tint, and hands the renderer one linear
    RGBA per face — so the GPU path now matches the CPU path face for face.
  - `Node::assign_material_to_faces(faces, slot)` points *named* faces at a slot
    and leaves the rest alone, appending the slot to the node's list when it is
    new and reusing the entry when it is not; it returns whether anything moved
    (out-of-range faces, empty lists and already-correct faces are all no-ops).
    The app exposes it as **Assign to Faces** in the Materials panel, enabled
    only at Face level with a face selection, as one undo entry ("Assign Material
    to Faces") and none at all for a no-op.
- **11 new tests**: 3 in `openforge-core` (per-face assignment leaves the other
  faces alone and appends the slot once; the no-change cases — unassigned faces
  on slot 0, an out-of-range face, an empty list; non-mesh nodes are no-ops),
  6 in `openforge-render` (per-face colours reach the vertices while one colour
  still dedups; `NodeMaterials` addressing; scene resolution per face; the
  per-face and per-node fallbacks; and an **adapter-guarded GPU test** that
  renders a cube with two saturated hues per face — magenta on the ±Z/±Y pairs,
  yellow on ±X, so every viewing angle shows both — and asserts both hues reach
  the target pixels while a single-colour control paints exactly one, which is
  the regression test for the old "first face's material" shading) and 2 in
  `openforge-app` (`node_materials()` resolves per face with the slot's linear
  colour untouched by selection, and a face override shows up as a second
  colour, survives undo; the command refuses at Object level and pushes no undo
  for a no-op).

- **Transforms snap now, and a snapped drag says where it is landing.** New
  `openforge-app/src/snap.rs` holds `SnapMode` (Off / Grid / Vertex) and `Snap`
  (mode + a grid step in world units, an angle step and a scale factor step),
  with the maths pure and unit-tested: `snap_to_step` rounds to the nearest
  multiple, `snap_measure` turns any `Measure` into its snapped one, and
  `format_measure` is the status-bar readout (`Δ 1.250`, `× 1.200`, `45.0°`).
  Grid mode snaps the **destination** of the grabbed point rather than the
  distance it travelled — a drag comes to rest *on* a grid value wherever it
  started — and clamps a snapping scale to its step so it can never collapse to
  zero. Vertex mode lands the grabbed point on the vertex under the cursor:
  `viewport::snap_candidates` projects every vertex of the visible, unfrozen
  meshes and `nearest_candidate` takes the closest within
  `snap::SNAP_RADIUS_PIXELS` (14 px), preferring the nearer vertex on a tie;
  `gizmo::Targets::moving` is what keeps a drag from snapping to the geometry it
  is carrying — the whole node at object level, because its own pivot sits on
  its vertices, and just the moving vertices at element level, so the rest of
  the same mesh stays a valid target.
- `Gizmo::apply` now takes the frame's `Snap` and the resolved vertex target, so
  the app decides what a drag lands on while the gizmo stays pure math. The app
  side: a **Snap:** group in the toolbar's first row (mode + the step fields,
  which appear only in Grid mode), `G` cycles the mode, holding **Ctrl**
  inverts it for as long as it is held (Off → Grid, anything → Off) — decided
  *per frame*, so it takes effect mid-drag — and **the live value is painted and
  reported**: a gold cross-hair marks the vertex being snapped onto, and the
  status bar shows `Move · snap grid 0.25 · Δ 1.250` instead of the tool hint
  until the release. The gesture still lands as **one** undo entry, and a drag
  that snaps back to where it started leaves none.
- **19 new tests**: 10 in `snap.rs` (nearest multiple — negatives and a
  non-positive step included; grid landing on a value *and* being idempotent
  under a repeated measure; vertex landing and its free fallback with no vertex;
  vertex leaving rotate/scale alone; angle and factor stepping; the factor floor;
  Ctrl inversion keeping the steps; the readout format; the mode cycle),
  4 in `viewport.rs` (`snap_candidates` projecting only visible, unfrozen
  meshes and dropping exactly what a drag carries — whole nodes at object level,
  named vertices at element level — plus the radius, the nearest-wins rule and
  the nearer-vertex tie-break), 1 in `gizmo.rs` (`Targets::moving` for both
  kinds of drag) and 4 in `app.rs`, driving real gestures headlessly: the same
  drag once free and once snapped to prove it lands on the nearest grid value
  with the readout showing and one undo restoring it, Ctrl inverting both ways
  (off→snapped, grid→free), a vertex-snapped drag that walks the grab point onto
  a marked vertex of another mesh (checking the marker comes and goes with the
  drag), and the `G` cycle.

- **Textures, end to end** (work-stream #11). `openforge-materials` gained
  `Texture::resized` (clamped bilinear, same-size copies bit-exact, name kept),
  and `openforge-core`'s `uv_sphere` now fills its `uvs` with the spherical
  coordinates it was already computing (with every corner's UV index its
  position index, so mapping the sphere is a no-op) — before this the default
  scene had no UVs at all.
  `openforge-render`: `GpuVertex` grew `tex_layer` (**48 → 52 bytes**, the
  attribute joins the layout at offset 48 / `@location(4)`, and the *dedup key*,
  so faces on different maps do not merge); `GpuRenderer::new` now takes the
  `queue` and reserves **array layer 0 as white**; new
  `TextureLayer { width, height, rgba }` + `GpuRenderer::set_textures` rebuild
  the array (and both bind groups that hold its view) and fill every layer
  white before the caller's pixels go in, so a layer with a missing or
  mis-sized buffer shades as if it were untextured; the scene pass binds
  `texture_2d_array<f32>` + a filtering sampler and samples with
  **`textureSampleLevel(..., 0.0)`**, because a per-face layer index is
  non-uniform control flow and naga rejects `textureSample` there. `NodeMaterials`
  now stores a `FaceShading { color, layer: Option<usize> }` per face
  (`Texture::flat`/`textured`, `From<[f32; 4]>`), and `MeshBuffers` gained
  `build_with_face_shading` (the colour-only `build_with_face_colors` delegates
  to it).
  The app resolves each face's map through one shared rule
  (`viewport::face_texture`, which ignores a map pointing at a texture the
  library no longer holds), resamples the library to 256² layers on a
  `texture_epoch` revision and uploads them, paints textured faces in the CPU
  viewport as `egui::Mesh` triangles (`textured_triangle`, one cached
  `TextureHandle` per library texture), paints the material's texture behind
  the UV editor's tile (`v = 0` at the tile's *bottom*, matching the v-up UV
  space, with a light quarter-grid over it), and grew the Materials panel a
  texture picker (with a 20px thumbnail), **Load Texture…** (rfd, PNG/JPEG) and
  **Checker** (a procedural checkerboard), plus a "Checker Test" preset that
  ships a 256px checkerboard so texturing can be seen without importing
  anything. Slot 0 stays untextured.
- **26 new tests**: 1 in `openforge-core` (the sphere's UVs),
  8 in `openforge-materials` (`resized`: upscale corners/palette, bit-exact
  same-size, checkerboard downscale, stretch, solid, zero size, empty source,
  name), 9 in `openforge-render` (the layer offset and its dedup behaviour,
  `WHITE_LAYER` for flat faces, per-face layers through `build_scene_materials`,
  the 52-byte vertex, the two stages' varying locations, plus an
  adapter-guarded **GPU readback test**: a cube with two 1×1 textures — a red
  and a green layer — shows *both* pure channels and neither hue without the
  maps), and 8 in `openforge-app`: the shared `face_texture` rule (including a
  dangling map and an out-of-range slot), UVs on collected triangles, both paths
  seeing the texture, the epoch/layer-size invariants, a failed import changing
  nothing, mapping onto the panel material feeding the UV editor,
  `textured_triangle`'s mesh, and the UV editor's background quad with its v
  orientation.

- **Sub-object picking goes through a GPU ID pass, and UVs have islands**
  (work-stream #12). Two halves:

  - `openforge-render/src/pick.rs` (no wgpu types anywhere, so it is fully
    testable headless): `PickVertex` (a world position plus the u32 element id
    its fragments report — 16 bytes, `Pod`, layout matched by the pick shader's
    two `@location`s), `PickElement::{Face, Vertex}` (a node plus the mesh-local
    face/vertex index), `PickGeometry::build(scene)` — visible, unfrozen mesh
    nodes only; faces fan-triangulated exactly like the visible pass with
    invalid ones skipped; every mesh position also emitted as a point; one
    shared 1-based element table, so `0` stays the clear value — and
    `resolve_patch(patch, width, height, centre)`, which turns a read-back
    integer patch into nearest-first, de-duplicated `PickHit`s (squared pixel
    distance, ties broken by id, centre given *inside* the patch because a patch
    clipped at the viewport edge is not centred).
  - `GpuRenderer::pick(device, queue, scene, camera, cursor, radius)` draws that
    geometry into its own `R32Uint` target (`PICK_FORMAT`) with a
    **single-sampled** depth buffer and its own pipelines, shader
    (`pipeline::pick_shader_wgsl`, flat `u32` varying, integer fragment output —
    validated by the same naga test as the other two) and uniform buffer, so a
    pick never disturbs a rendered frame and works whatever MSAA the frame runs
    at. Faces are drawn with the frame's culling and `Less` depth test; mesh
    vertices follow as 1-px `PointList`s with `LessEqual`, so a vertex pixel wins
    over the face it lies on but never over a face genuinely in front of it. The
    patch around the cursor (radius clamped to 64 px, clipped at the viewport
    edges) is copied out at the alignment wgpu demands (256-byte row pitch),
    mapped and polled synchronously — this is a click path, not a frame path.
  - The app asks the GPU first: `gpu_viewport::pick` maps a pointer position
    onto the OFFSCREEN pixel (`pick_pixel`: offset from the viewport rect ×
    `pixels_per_point`, floored, clamped) and `gpu_viewport::chosen_hit` picks
    the first hit the current level can use — a face click must skip a nearer
    vertex (the point pass draws vertices on top of their faces) and a vertex
    click must skip the covering face. When there is no wgpu backend, or the
    patch holds nothing usable, the CPU raycast runs exactly as before.
    Object-level clicks still go through `SelectTool`: the raycast there is
    occlusion-correct already.
  - `openforge-uv/src/islands.rs` (via a subagent, verified here):
    `uv_face_islands(mesh, tolerance)` welds two faces sharing a mesh edge into
    one island only when the UVs they assign at **both** shared endpoints match
    within the tolerance, then orders islands by smallest face with every face
    in exactly one island; `uv_island_of_face` and `uv_island_uv_indices`
    (sorted, de-duplicated, out-of-range indices dropped) sit beside it; and
    `uv_pack_islands(mesh, grid, margin)` packs each island into its own cell of
    a `grid × grid` atlas at one uniform scale (aspect preserved) in two phases
    (measure every island, then apply) so the result never depends on the order,
    treats an entry shared by several islands as the first island's, clamps the
    margin into `[0, 0.5)`, and returns the island count. The crate's own
    `UnionFind`/`Pack` were widened to `pub(crate)` for the reuse, nothing else.
  - One honest finding from that work: `uv_project_box` gives **numerically
    identical** UVs to faces that meet at a cube corner across two different
    axes, so a box-projected cube welds into **2** islands (three sides each),
    not six. The rule is UV continuity and those faces genuinely carry matching
    UVs at their shared edges; six islands would need a different projection.
  - The UV editor gets both operations as buttons: **Island** (grow the UV
    selection to every entry of the islands it touches —
    `uv_editor::island_selection`, one partition per call,
    `ISLAND_TOLERANCE = 1e-4`) and **Pack Islands** next to the existing
    position-welding **Pack**.

- **47 new tests**, essentially all runnable without a display: 16 in
  `openforge-uv` (`islands.rs`: welding across a shared edge, island order and
  determinism, tolerance edges, packing bounds/grid/margin and an entry shared
  by two islands, `u32::MAX` guards, empty meshes), 15 in `openforge-render`
  (14 pure in `pick.rs` — fan triangulation, 1-based ids and the element table,
  world transforms, hidden/frozen exclusion, invalid faces, patch resolution
  with ties/dedup/edge clipping — plus the naga validation of the pick shader),
  6 more in `gpu.rs` behind the adapter guard (the pick pass resolves what the
  frame shows; a vertex is picked at its pixel unless occluded; layout/format/
  radius/copy-pitch invariants), 3 in the app's `gpu_viewport`
  (`pick_pixel` cursor mapping and clamping, `chosen_hit`'s per-level rules),
  and 7 in the app (6 `uv_editor::island_selection` cases and the Island
  button's selection path).

- **Rectangle (marquee) selection, through the same ID pass** (work-stream
  #13). In the **Select** tool a primary drag now draws a rectangle instead of
  reaching the tool (whose own work is click-driven anyway): the app records the
  drag's start from egui's **`press_origin`** — egui only calls a press a drag
  once the pointer has moved, and the position that decided the drag is already
  a few pixels in — paints the live rectangle over whichever backend drew the
  frame and resolves it on release. A **plain** rectangle replaces the
  selection, **Shift** extends it, and a rectangle the pointer came back from
  (no area at all) is resolved as the click it ended as. Selection is not scene
  state, so there is nothing to undo.

  - `GpuRenderer::pick_rect(device, queue, scene, camera, rect_px, level)` is
    the rectangle counterpart of `pick`: the same single-sampled `R32Uint` pass
    with the same culling and depth test, but the whole half-open pixel
    rectangle `[x0, y0, x1, y1]` is read back and **every distinct element**
    inside it comes back — `pick.rs`'s new `PickGeometry::elements_in_patch`
    takes each id once and answers in element-table order (ascending ids), with
    `0` and ids past the table skipped and short or empty patches empty. Both
    `pick` and `pick_rect` now share one `read_id_rect` driver, and the level
    decides what the pass draws and what it answers: a **face** rectangle leaves
    the vertex points out entirely — they are 1-px dots sitting on their own
    face, and a pixel hole in the face at every vertex would be exactly the
    disagreement the pass exists to prevent — while `Vertex` and `Object`
    rectangles draw them and keep vertex hits alone (`Vertex`) or both kinds
    (`Object`, whose caller only needs the node).
  - The app asks the GPU first and **trusts its answer even when it is empty**:
    a rectangle over empty space selects nothing, and one that covers no pixel
    of the target covers no element either — falling back to a projection that
    selects *through* geometry would contradict the picture the click path
    obeys. Only a machine with no wgpu backend falls back to
    `viewport::marquee_elements(scene, vp, size, rect, level)` (a subagent's
    work, verified here): project each node's positions once and answer per
    level — `Object` = every position inside, `Vertex` = every position index
    inside, `Face` = every face with at least one corner inside, `Edge` =
    nothing — skipping hidden and frozen nodes, inclusive on the rectangle's
    edges, deterministic. It cannot tell occluded geometry from visible, which
    is exactly why it is only the fallback.
  - The status bar now reads out **elements** at sub-object levels
    ("Select — 12 face(s) selected.") rather than always the node count.
  - **One test failed honestly and was fixed**: the first multi-node rectangle
    test assumed `Scene::iter()` yields nodes in insertion order, but
    `Scene::nodes` is a `HashMap`, so `PickGeometry::build`'s ids follow hash
    order — stable within a run, not across runs. It was rewritten to pin
    ascending ids on a single node, where the table order is unambiguous. See
    §8.

- **23 new tests**: 4 in `openforge-render`'s `pick.rs` (a rectangle finds each
  element once and in table order; clear values and unknown ids fall out; short
  or empty patches resolve to nothing, an empty geometry too), 3 adapter-guarded
  in `gpu.rs` (a face rectangle finds what the frame shows — the near quad, not
  the one it hides, the far one once the near is hidden, and nothing over empty
  space, a zero-width rectangle or an `Edge` level; a face rectangle is **not**
  punched through by its own vertices while the same rectangle at vertex level
  finds the corner vertex; a vertex rectangle finds only the vertices the frame
  shows, a loose vertex behind a face excluded until it moves in front), 3 in
  the app's `gpu_viewport` (`marquee_pixel_rect`: pixel coverage through
  floor/ceil with the drag's direction normalised, clamping and bad scales,
  `None` for a click, a NaN corner or a degenerate target), the **10 in
  `viewport.rs`** the subagent wrote (per-level rules, n-gon faces, hidden and
  frozen nodes, a position behind the camera, determinism, the edge level,
  inclusive rectangle edges), and 3 in `app.rs` driving the real gesture with
  synthetic pointer events: a drag paints its rectangle and selects the faces
  inside it but not the ones outside, Shift extends while a plain rectangle
  replaces, and an object-level rectangle takes exactly the node it holds (and
  lets go of it over empty space).

- **Edge-level picking** (work-stream #14). The one level neither the click nor
  the rectangle could answer is now first class, in the same three layers:
  - **`openforge-core` grew `Mesh::edges()`** (see §5) — the single edge
    enumeration every layer numbers edges by. `PickGeometry` gained a `lines`
    list (two vertices per edge, both carrying the edge's id), appended per node
    **after** its face and vertex elements so the older ids never moved.
  - **The ID pass draws lines.** `gpu.rs` gained a third pick pipeline
    (`LineList`, `CullMode::None`, `LessEqual`), and `PickDraw { lines, points }`
    now decides what each level draws on top of the faces: faces alone at Face
    level, faces + points for Vertex and Object, faces + lines for Edge, and
    everything for a click — whose hits the caller filters (`chosen_hit` grew
    its `(Edge, Edge)` arm). `pick_rect`'s "no edge pass" early return is gone;
    an Edge rectangle draws faces and lines and answers edge elements like any
    other level.
  - **The app picks edges by click and by rectangle.** `viewport::pick_edge` is
    a ray-vs-segment test — the exact closed form, `s` clamped to the segment,
    `t < 0` skipped, nearest wins — and is the CPU fallback `resolve_click` uses
    at Edge level; the marquee's Edge rule projects both endpoints of every
    `mesh.edges()` pair and selects an edge with an endpoint inside the
    rectangle or a segment crossing one of its four sides (an endpoint behind
    the camera leaves its edge out, like every position behind it);
    `element_centroid` and `selected_vertices` read the endpoints, so a selected
    edge and a gizmo drag move exactly its own two vertices. `apply_marquee` no
    longer refuses the level, and the status bar counts the edges.

- **25 new tests**: 6 in `openforge-core` (`mesh.rs`: corner order, the shared
  edge entered once, triangles and n-gons, the valid-face rule, degenerate
  corner pairs, loose positions), 2 in `pick.rs` (a quad's four edges as eight
  line vertices with ids 6–9, and edge ids resolving out of a patch in table
  order), 2 adapter-guarded in `gpu.rs` (an edge rectangle finds the near
  quad's four edges over the far quad's — which is also the proof that a line
  wins the pixels it lies on, the face underneath being at the same depth — the
  far one's once the near is hidden, vertices alone at Vertex level, and
  nothing over empty space or over a zero-width rectangle; a click on a quad's
  top side finds that edge first with its face still behind it, and the quad's
  middle is its face, not an edge), 13 net in the **`viewport.rs`** the
  subagent wrote (7 for `pick_edge`: midpoint hit, face middle miss, nearest of
  two parallel edges, endpoint clamping, a segment behind the origin, hidden
  and frozen nodes, an empty scene and a mesh without faces, the tolerance
  boundary; 6 for the rectangle: the endpoint rule and the neighbours it
  brings, crossing without an endpoint inside, an endpoint exactly on the box,
  an endpoint behind the camera, hidden/frozen, determinism with each edge once
  and indices ascending; plus the centroid and dragged-vertices test, which
  replaced the old "edge level selects nothing" test), and 2 in `app.rs`
  driving the real gestures: a click on the projected bottom edge selects edge
  0 (and the middle of the face selects nothing at Edge level), and a rectangle
  over the quad's upper-left corner selects exactly the top and left edges, the
  whole quad all four of them, with the status counting them.

- **Showing the selection in the frame** (work-stream #15). #14's last gap: the
  picture did not show a picked edge, only the status bar counted it.
  - **The frame's overlay falls back to the outline at edge level.**
    `overlay_mode` draws an *off* overlay as the selected meshes' edges there,
    so the lines a pick can answer from are on screen the moment a node is
    being edited — the same context the vertex level gets from its dots. The
    setting itself is untouched: `Z` still cycles it, the View menu still
    reports it.
  - **A picked edge is marked.** `paint_selected_edges` draws a bold
    `ELEMENT_SELECTED_SRGB` line over every selected edge — the same hue a
    selected vertex's dot and a selected face's edges take, so "selected" reads
    as one colour at every level — from the same `Mesh::edges()` indices the
    pick answered with, projected through the shared camera on both paths, on
    top and unoccluded exactly like the dots. The GPU overlay itself stays
    depth-tested inside the frame; the aid is an overlay, not part of the
    shaded image.
  - **A latent misread fell out of the same review.** The display paths read the
    element set as *face* indices without checking the level, so at edge level
    a selected edge 0 tinted face 0 — in three places: the CPU triangle tint,
    the overlay's `face_selected` flag and the GPU path's per-face colours. All
    three now ask `Selection::selected_faces`, which answers only while the
    face level is the one being edited; the app's modeling-op and UV face lists
    use it too, so a face op at another level refuses instead of acting on edge
    indices.

- **6 new tests**: 1 in `openforge-core` (`selected_faces` answers at the face
  level, refuses at the edge level, and leaves the indices themselves alone),
  2 in `viewport.rs` (`collect_edges` flags no `face_selected` at edge level
  while the node flag stays and the face level still flags the same index;
  `collect_triangles` tints only at the face level), and 3 in `app.rs`:
  `overlay_mode`'s table (the fallback only at edge level *and* an off overlay,
  a chosen mode drawn as chosen), the frame test driving a real corner
  rectangle — four outline strokes through the off overlay, exactly two bold
  highlights landing within 1.5 px of the projected top and left edges of the
  quad, none at face level — and the GPU face colours (an edge selection keeps
  every face on the node tint, while the same index at the face level carries
  the face tint).

- **Selection tools** (work-stream #16). The tools *around* the levels, all of
  them on the one edge list #14 established:
  - **`openforge_core::subselect`** — pure maps from one element set to
    another at a level: `all_elements`, `invert_selection`, `grow_selection`,
    `shrink_selection`, over `Mesh::edges()`. Faces are neighbours through a
    shared edge, edges through a shared position, vertices through an edge; an
    undrawable face is no element and joins nothing (the `face_is_valid` rule
    `edges()` now shares).
  - **The app drives them per mesh node**: a **Select** menu, a "Select:"
    toolbar row and hotkeys — Ctrl+A, Ctrl+Shift+A, Ctrl+I, Ctrl+Plus (or
    Ctrl+Equals), Ctrl+Minus — each action leaving a status that counts the
    result. At the object level All / None / Invert work on the visible nodes;
    grow and shrink say where they work instead of doing nothing quietly.
  - **Hide / Isolate / Unhide** (**H**, **Shift+H**, **U**, and the View
    menu): one undo step each through the same `with_edit` every scene edit
    takes. Hiding drops the node from the selection, isolate keeps it, and the
    frame follows at once because the collectors already skip hidden nodes.

- **13 new tests**: 6 in `openforge-core::subselect` (the per-level element
  counts; an undrawable face is no selectable face, no side and no neighbour;
  invert flips only the level's own list; face grow takes the edge neighbours
  and not the corner one; shrink cuts exactly the boundary grow would have
  reached, so growing then shrinking gives back the seed; the vertex and edge
  walks, with the diagonal staying out two steps in), 1 in `selection.rs`
  (`set_elements` replaces, never adds) and 7 in `app.rs`: Select All / None on
  every level with their statuses; the "no mesh" and "wrong level" messages;
  invert round-tripping; grow and shrink through the app on a 2×2 grid; the
  hotkeys driven as real key events; and hide (the visible flag, the dropped
  selection, undo, and the frame's stroke count following it) plus
  isolate/unhide with their undo steps.

- **Edge loops and rings** (work-stream #17). The classic selection moves, at
  the edge level the ID pass already answers for:
  - **`subselect::edge_ring`** crosses each quad face to its opposite side and
    continues through the face it did not just cross, so it walks *around* a
    mesh: every one of a cube's twelve edges rings in exactly four closed
    edges (tested from all twelve), a grid edge rings along its parallel band
    and stops where the grid does, and a face that is not a quad stops it too.
  - **`subselect::edge_loop`** carries on straight through a vertex that has
    exactly four edges, along the one that is neither the current edge nor one
    of its two fan neighbours: a grid's middle column is one loop, the border
    (three edges) ends it, and a cube corner ends it at once.
  - **The app walks every picked edge** — L, Shift+L, the Select menu or the
    toolbar row — adding the union to the selection, so a second press changes
    nothing and the status counts what is selected now. The walk is
    edge-level: another level is told so, and no picked edge says to pick one.

- **5 new tests**: 4 in `subselect` (the parallel-band rings with their closure
  — the ring of any member is the same ring — the straight loops where the
  four-edge vertices carry and the border vertices stop, every cube edge
  ringing in four closed edges, and a ring stopping at a non-quad face sharing
  the edge) and 1 in `app.rs` driving both walks through the actions, the
  hotkeys, and the two refusals.

- **Select Through and the outliner's visibility box** (work-stream #18). Two
  loose ends from the picking and visibility work:
  - **View ▸ Select Through (P)** sends the sub-object picks to the projection
    instead of the ID pass: a click takes the nearest element under the
    pointer, a rectangle everything it covers, whether the frame shows it or
    not. `OpenForgeApp::occluding_picks(select_through, has_backend)` is the
    one gate, asked by `gpu_pick` and the GPU branch of `marquee_hits` alike;
    on a machine without wgpu the switch changes nothing, because the raycast
    and the projection already answer there.
  - **The outliner's visibility box** now runs through `set_node_visible`:
    one undo step (it used to edit the scene with no undo entry at all), a
    status naming the node ("Hidden Grid." / "Shown Grid."), and hiding drops
    the node from the selection exactly the way H does.

- **2 new tests**: one drives the gate's truth table, flips the mode by the P
  key and lands a click on a vertex hidden behind the face in front of it
  (the projection's own answer, pinned because it is what the mode falls back
  to); the other takes the outliner's box through hide, undo and show.

- **An import/export surface that says what it will do** (work-stream #19).
  The file plumbing existed; the *feedback* did not:
  - **Formats come from the registry** in the UI as well as in the plumbing:
    the open dialog carries one filter group per registered format, **Export
    selected as ▸** lists them by label (Wavefront OBJ, glTF 2.0, Stanford PLY,
    STL) and saves with a matching default file name, so a plugin's format
    appears in the menus with no app change.
  - **Refusals name what would have worked**: a path no filter knows is turned
    down before anything is read or written — "Skipped a.foo — no filter knows
    it (glb, gltf, obj, ply, stl)." / "Can't export to a.foo — …" — and every
    status names the file and the format ("Imported 2 mesh(es) from scene.obj
    (Wavefront OBJ).", "Exported Cube to cube.obj (Wavefront OBJ).").
  - **Export refuses ambiguity**: more than one selected mesh node is "Select
    one mesh to export — 2 mesh nodes selected." and writes nothing, instead
    of quietly writing the first one; an empty selection and a selected
    non-mesh keep their own messages.
  - **Files dropped on the window import** through the same registry, at the
    end of the frame so the status survives the viewport hint: one file reports
    like a menu import (own file, own undo step), a pile ends on "Dropped 3
    file(s) — imported 2 mesh node(s)."

- **3 new tests**: the format-naming import with its refusal list (and its undo
  step), the export refusals (empty, ambiguous, unknown extension) around a
  real write whose status names the format, and a drop driven through a frame —
  one file, then a mixed pile, with the summary counted.

- **A timeline that plays the scene** (work-stream #20). `openforge-animation`
  had tracks, shortest-arc interpolation, clips and a transport; the app had
  never touched it. Now:
  - **Animate ▸ Spin Selected (2 s)** builds a clip for every selected node:
    one full turn about Y from the node's own pose, carried by **four
    quarter-turn keys** — a single 360° end key would blend back the way it
    came, because quaternion tracks blend along the shortest arc. The clip is
    named around "spin" without colliding ("spin 2"), selected, and the
    timeline opens.
  - **The Timeline panel** is the transport: clip picker, Play/Pause
    (**Space**), Stop, Loop, and a scrub slider with the clip's key times
    listed under it (`clip_key_times` merges the position/rotation/scale keys,
    sorted and deduplicated).
  - **Playback is a view over the clip, not an edit.** `update` advances the
    transport on the frame clock before anything paints (`stable_dt`, capped at
    0.25 s so a dragged window cannot jump the clip) and writes the sampled
    transforms into the scene; no undo entry is pushed, and whatever pose it
    stops at is where the scene stands. A scrub poses the scene through the
    same `pose_current`; Stop rewinds to the clip's start.

- **2 new tests**: the spin clip scrubbed to the half turn and the full turn
  (the double cover included), then played a quarter second — 45° of a two
  second turn — past the end to the pause, and stopped back to the start; and
  the refusals, the name collision, the five key times and the untouched undo
  stack. Both run headless, like the rest: the frame seam (`advance_animation`)
  is a method, because the painting helpers stop at the central viewport.

- **Game asset prep: LOD tags, dummies, a triangle budget** (work-stream #21).
  A generic, format-neutral take on the game half of the tool:
  - **`Node.lod: Option<u8>`** (core): 0 = highest detail, `None` = untagged.
    Part of node `PartialEq`, so the tag rides the existing undo snapshots and
    the serde round-trips without new plumbing.
  - **Game ▸ Add Dummy**: `Scene::create_dummy` existed in core; nothing had
    ever made one. The command puts it at the selection's centre (the average
    of the selected nodes' world positions — the origin when nothing is
    selected), selects it, and is one undo step.
  - **Game ▸ Set LOD 0–3 / Clear** tags the selection in one undo step, with a
    refusal when nothing is selected ("Select a node to tag.").
  - **Game ▸ Isolate LOD 0–3** is the preview a game model needs while one
    detail level is worked on: every node the level does not hold is hidden
    through the ordinary `visible` flag — one undo step, exactly like Isolate,
    and `U` shows everything again. An untagged level is refused with the
    reason ("Nothing is tagged LOD n — set it first.").
  - **The model report** (`LodReport`): triangles per LOD level over a set of
    nodes — untagged first, then levels ascending, then the total — shown in
    the Properties panel for the selection and for the whole scene, with each
    node's tag also printed on its row and an `LOD n` suffix in the outliner.

- **3 new tests**: the tag / isolate / clear round (one undo step for the
  isolate; the refusals for an empty selection and an untagged level); the
  report's per-level counts and panel lines over three boxes (a dummy holds no
  triangles and is simply absent); and the dummy landing on the origin with
  nothing selected and at the selection's centre (2, 1, 0) otherwise — kind
  checked, undo entry counted.

- **The plugin host gets its controls** (work-stream #22). The Plugins panel
  was a read-only report; now it is the host's one control surface:
  - **A checkbox per built-in** (`openforge.core-tools`, `openforge.interchange`
    today). Toggling calls `set_plugin_enabled`, which records the name and
    calls `rebuild_plugins`: a fresh `PluginHost` registers every remaining
    built-in, so a plugin's formats, tools and services leave together and a
    rejected registration cannot leave half of itself behind.
  - **"What it brings" is measured, not guessed.** `plugin_infos` registers
    each built-in alone into a scratch host and reads its registries, so every
    row lists its formats by label plus service/tool counts — and keeps listing
    them while the plugin is off, so a user sees what re-enabling restores.
  - **The refusals follow the active set.** `unknown_format_line` (shared by
    the import and the export path) names the known formats ("no filter knows
    it (glb, gltf, obj, ply, stl).") or, with every format plugin off, says
    exactly that ("Skipped x.stl — every format plugin is disabled (see the
    Plugins panel).").

- **2 new tests**: the interchange plugin's formats leaving and coming back —
  an import refused while it is off, the same file imported while it is on —
  and the panel rows adding up to what the host serves (4 tools, 3 services,
  4 filters, the OBJ filter present; a disabled plugin still lists what it
  would bring).

**Verified (build/tests):** `cargo build --release -p openforge-app` (the CI
build job) → 0 warnings; `cargo build --workspace` → 0 warnings (and the test
build too); `cargo test --workspace` → **573 pass, 0 failures**
(`cargo build --workspace ... | grep -c warning` → 0). The gizmo, the wiring
behind the UV canvas, the side panels, the snapping gestures, the plugin
seam and now texture sampling on both paths are all covered *without* a
display: the `app.rs` test module runs the real
`handle_central_viewport` / `handle_side_panels` against a plain
`egui::Context` with synthetic pointer events and asserts on both the gesture
(hover, grab, axis-constrained drag, release, undo) and the shapes the frame
painted, and on what the host did (import/export through a filter, a plugin's
filter taking effect, a rejected plugin changing nothing). The texture sampling
itself is proven by a headless GPU readback (red and green layers on a cube
show both hues, neither without the maps) rather than by eye. Sub-object
picking is proven the same way: `pick.rs` resolves patches with no GPU at all,
a headless readback witnesses a cube's front face while a vertex hidden behind
it never appears in the patch, and the app's per-level rules (`chosen_hit`) are
asserted directly. Rectangle selection is covered the same way: the pure
resolution in `pick.rs`, the pass itself through adapter-guarded readbacks
(including that a face rectangle is not punched through by its own vertex
points and that a vertex rectangle excludes what a face hides), the pixel
mapping and per-level filtering in `gpu_viewport`, and the gesture end to end
in `app.rs` — press, drag, painted rectangle, release, selection — on the CPU
path. Edge picking is covered the same way: the enumeration and the line
element table headless in `pick.rs`, the line pass through adapter-guarded
readbacks (an edge rectangle finds exactly the near quad's four edges — which
is also the proof that a line beats the face at the same depth — and a click on
a quad's top side finds that edge first while the face still resolves behind
it), the ray-vs-segment and segment-rectangle rules in `viewport.rs`, and both
gestures end to end in `app.rs` — plus what the edge level *draws*: the frames
are asserted on the shapes painted (four outline strokes through an off
overlay, two bold highlight segments matched against the projected edge
endpoints), and `overlay_mode`'s table is tested directly. The selection tools
are covered at both ends: the set maths headless in `subselect` (including
that grow and shrink are inverses and that a corner is not a neighbour) and
the app actions with their statuses, hotkeys and undo steps — hide is even
followed into the painted frame, whose stroke count drops and returns. The
edge walks are pinned in core too (the grid's parallel bands and middle
column, the cube's closed four-edge rings, the non-quad stop) and driven in
the app through the same keys. Select Through is pinned at the gate itself
(its truth table and the P key) and by a click landing on a vertex hidden
behind the face in front of it, and the outliner's box by its single undo
step. Import and export are pinned headless too — a real file written and
read back through the registry, the refusals with their extension lists, and
a drop driven through a frame — so only the native dialogs themselves are
left to the eye. The animation path is pinned at the transport: the spin
clip's key times, the half turn, the full turn (the double cover included),
45° after a quarter second of play, the pause at the end and the rewind on
stop, with the undo stack provably untouched. The game-prep surface is pinned
the same way: the tag / isolate / clear round with its single undo step and its
refusals, the dummy at the origin and at the selection's centre (its kind
checked), and the report's counts and lines per level. The plugin controls are
pinned through the host itself: the formats leaving and returning, the refusal
wording when none is left, and the panel's rows summing to the live registry
counts.
**Not verified:** how the window actually looks, and the gizmo, the UV canvas,
the new Plugins panel, the GPU edge overlay by eye (its depth-tested look
and the MSAA smoothing), the **per-face material shading on the GPU
path** (assign one face a different material and confirm the two paths agree
visually), the **snapping feedback** (the gold marker on the vertex a drag
is about to land on, the live status readout, and Ctrl switching snapping on and
off mid-drag), and the **textured look** (the checker preset on a sphere, the
texture picker and its thumbnail, the UV editor's texture background and
whether the two viewports agree on orientation), plus the **GPU pick** (click a
vertex hidden behind a face: the face behind should take the click, never the
hidden vertex; face clicks on a vertex must still take the face), the
**Island / Pack Islands buttons** (Island should light up a whole island, Pack
Islands should leave each island in its own cell with a gutter) and the
**rectangle selection in the live app** (drag a box in the Select tool: the
rectangle should follow the pointer, Shift should add to the selection rather
than replace it, and the wgpu path's answer — which the headless tests can only
prove through a readback — should match what the image shows), and the
**edge-level picking and its highlight in the live app** (click an edge, drag a
box at the Edge level: the pass answers from one-pixel lines, the frame should
outline the edited mesh through an off overlay and draw a bold line over every
picked edge, and the highlight is projected rather than depth-tested — so a
picked edge behind a face glows through, exactly like the vertex dots), the
**Select menu / toolbar row / hotkeys and hide-isolate by eye** (Ctrl+A should
fill the level, Ctrl+Plus/Ctrl+Minus should step along the topology with the
status counting, H/Shift+H/U should hide, isolate and reveal with undo
stepping back), the **edge loop / ring walk** by eye (pick one edge of a
quad grid, L should take its column and Shift+L its band of parallels), the
**select-through switch** by eye (P, then click a vertex behind the surface or
drag a box over hidden edges: the occluded element should answer, and turning
it off should bring the image's own answer back) and the **outliner's Visible
box** by eye (it should hide and show one node with its status and one undo
step) and the **File menu's dialogs** by eye (the import dialog's filter
groups per format, the export submenu's default file names, and a drop of a
real file onto the window) and the **timeline** by eye (Animate ▸ Spin
Selected: the panel should appear with the clip's five key times under the
slider, Play and Space should turn the node smoothly, and Stop should snap it
back to the start), and the **Game menu** by eye (Set LOD / Isolate LOD / Add
Dummy: outliner rows should carry their `LOD n` tag, Isolate LOD 1 should
leave only the level-1 nodes standing, and `U` should bring the rest back),
and the **Plugins panel** by eye (uncheck `openforge.interchange`: the row
should stay with its formats still listed, the report below should shrink to
the core-tools counts, and importing a `.obj` should refuse with the
all-disabled message) — the agent has no
display access, so a human should look at all of those once. (An earlier session did
launch the app on the wgpu path and saw `eframe: ... Using wgpu.` with a Metal
adapter, no `egui_wgpu` warnings and no panic over a 12 s run.)
`rustfmt` is not installed for this toolchain, so formatting was matched by hand.
`cargo clippy -- -D warnings` still trips on **pre-existing** lints
in `openforge-core` (`needless_range_loop` in `edit.rs`) and `openforge-io`
(`collapsible_if`, …): untouched here, and not part of CI.

**Progress toward full ZModeler parity: see [`PROGRESS.md`](PROGRESS.md)** for
the per-area bars (`scripts/progress.sh --write` regenerates it; the mean of the
16 curated areas is **67.1%**, shown as 67%). The engine
layer is essentially done, and the largest application-surface gaps — the GPU
viewport, the UV editor and the editing UX around the gizmo — now have real
implementations, with materials shading per face on both paths as of #9, the
gizmo snapping as of #10, textures sampling on both paths as of #11,
sub-object picking that matches the drawn frame (plus UV islands) as of #12,
rectangle selection through that same pass as of #13, edge-level picking by
click and rectangle as of #14, the visible sub-object selection as of #15,
the selection tools (all / none / invert, grow / shrink, hide / isolate /
unhide) as of #16, the edge loop / ring walks as of #17, the
select-through switch with the outliner's undoable visibility box as of #18,
the format-named import/export surface (dialogs, drops, refusals) as of #19
and the animation timeline (spin clips, playback, scrub, Space) as of #20 and
the game-prep surface (LOD tags and isolate, dummies, the triangle budget) as
of #21 and the plugin host controls (per-built-in enable/disable with measured
attribution) as of #22.
Prioritized next steps:

1. **Deepen the GPU viewport.** MSAA, the depth-tested edge overlay, per-face
   material shading (#9), texture sampling (#11), occlusion-exact sub-object
   picking (#12), rectangle selection (#13), edge-level picking (#14), the
   visible sub-object selection (#15), the selection tools (#16: all / none /
   invert, grow / shrink, hide / isolate / unhide), the edge loop / ring walks
   (#17) and the select-through switch (#18: the ID pass or the projection, P)
   are in. What remains of the viewport surface: shadowing / ambient occlusion
   and a configurable shading mode.
2. **Extend the gizmo further** (`gizmo.rs`, all additive): plane and centre
   handles, and numeric entry for rotation/scale (translation already has
   DragValues in Properties). Snapping landed in #10 but is deliberately partial:
   only the gizmo's axis drags snap (the free-form view-plane drag does not) and
   only the grid and *vertices* are targets — ZModeler also snaps to edges,
   faces, midpoints and the grid of the *screen plane*, and its snapping works
   during free drags too. (Sub-object pivots are done: a vertex/face-level drag
   moves only its own geometry, via `gizmo::Targets` +
   `viewport::element_centroid`.)
3. **Deepen the UV editor.** The texture background (#11) and island selection
   plus per-island packing (#12) landed; still missing: island weld/split/mirror,
   snapping, per-face map assignment from the canvas, and the 3D face selection
   staying in sync with the UV selection rather than being pushed on demand.
4. **Deepen the plugin host.** The host is in and the app goes through it, but
   three pieces are still hard-wired: tools are chosen through the app's
   `ActiveTool` enum rather than by walking `host.tool_entries()`, a `Service`
   carries a name and a description rather than behaviour, and plugins are
   compiled in rather than *loaded* — a `cdylib`/WASM filter discovered at
   runtime is the real prize. That is the long road to true parity.
5. **IO expansion.** Add COLLADA/DAE, OFF, or ASCII FBX (open subsets) to widen
   format coverage beyond the current 4.
6. **Materials node editor.** Replace basic Phong/Lambert with a node graph.
   The GPU scene pass now samples a **diffuse** map (#11); normal, specular and
   emissive maps, per-face map assignment and mipmapped/anisotropic sampling are
   the remaining pieces. Per-face material assignment — the reason slots exist
   per face — landed in #9.

**Honest gap to parity:** format-specific game tooling (vehicles, TXD
dictionaries, per-game export paths — the generic half, LOD tags, dummies and
their triangle budget, landed in #21), the full panel/dialog surface, animation
*editing* (the timeline plays clips; keys are not yet drawn, dragged or
recorded), and exact format breadth remain the largest deltas and are many
work-streams, not one push.

## 8. Known gotchas when using AI subagents here

- **Free-model concurrency cap.** Running multiple subagents in parallel stalls
  them (they choke on the shared model). **Run ONE subagent at a time.** A single
  background agent works but is slow to report — it is usually *working*, not
  stuck.
- **Always verify a subagent's output on disk** (`cargo test -p <crate>`,
  warnings check) before trusting its summary. It may have left compile errors.
- **Large-file writes can time out** (the `write` tool / single-shot shell). For
  big files, write in chunks or use a shell heredoc (`cat > file <<'EOF'`).
- **The wgpu viewport breaks in three easy-to-hit ways.** (a) Read the egui
  texture id *after* `GpuViewport::render` — it re-registers the texture on
  resize, and a stale id logs `egui_wgpu ... Missing texture: User(n)`. (b) wgpu
  rejects a viewport/scissor rect that overflows the attachment; use
  `gpu::gizmo_viewport`, which clamps to fit. (c) Author sRGB colours, not raw
  linear bytes, or the image is washed out.
- **MSAA turns the render target into two textures.** With a sample count > 1
  the pass renders into the multi-sampled color texture and resolves into the
  single-sampled one — that single-sampled texture is what egui registers and
  what the smoke test reads back. The depth attachment and *every* pipeline's
  `multisample` state must use the same count, and `GpuRenderer::new` takes the
  count as a parameter: the app asks
  `supported_samples(adapter, COLOR_FORMAT, PREFERRED_SAMPLES)` rather than
  assuming 4x (no silent downgrade — a device that cannot do it reports 1).
- **The ID ("pick") pass is single-sampled on purpose, and its ids must stay
  integers.** It has its own `R32Uint` target, its own *single-sampled* depth
  buffer and 1-sample pipelines even when the frame runs MSAA — a resolve would
  average ids into nonsense. The point *and line* pipelines need `LessEqual` (a
  vertex and an edge lie exactly on the surface they belong to, so `Less` would
  drop every vertex on its own face and every edge on the face it borders)
  while the face pipeline keeps the frame's `Less` and culling (so a face
  genuinely in front still hides a vertex or an edge behind it); the pass draws
  **faces, then lines, then points** — a vertex pixel wins over the two lines
  that end at it. An empty buffer is a wgpu error, so every per-kind buffer is
  `Option`-guarded: a scene can have points and no lines (a visible mesh whose
  every face is invalid has no edges). The readback patch must be at least 256
  bytes per row, so `resolve_patch` is fed row by row with the padding
  stripped; and ids are 1-based because the target clears to `0`.
- **A level draws only what it must answer with, on top of the faces.** A
  vertex *point* or an edge *line* sitting exactly on its own face punches a
  one-pixel hole or notch into that face's pixels — the one way the picture and
  the pick could disagree. So a rectangle pass draws per level (`PickDraw`):
  faces alone for Face, faces + points for Vertex/Object, faces + lines for
  Edge — and only the *click* pass draws everything, because its hits are
  filtered by level afterwards (`chosen_hit` skips any element the level cannot
  use). An Edge rectangle therefore draws no points: a point pixel could only
  eat a pixel of an edge that ends there.
- **Edge ids come from `Mesh::edges()` and nothing else** — the CPU pickers,
  the ID pass's line table and the marquee all number edges by that one list
  (see §5). Its rules bite: a face with an out-of-range corner is skipped
  *whole* (the visible pass never draws it, so none of its sides exist), a pair
  repeating one position index is no edge, and each pair is stored lower index
  first — so a line's two vertices run in *index* order, not the order the face
  walks its corners (a test here once asserted the walk direction and was
  wrong). The list is derived from the face list, so editing faces renumbers
  edges: an Edge-level selection index only means anything while the faces it
  was picked from do not move.
- **Element indices are per level — never read one level's set as another's.**
  The element set holds plain `u32`s with no level attached, so a face-shading
  path that looks up `selected_elements` directly will tint whatever face
  shares a selected *edge's* (or vertex's) index — three display paths did
  exactly that until #15, where a selected edge 0 tinted face 0. Ask
  `Selection::selected_faces` for faces (it answers only at the face level);
  the indices themselves stay where they were put, so `element_count` and the
  status bar count them at any level.
- **Element sets replace, they do not merge** (`Selection::set_elements`).
  `select_elements` extends — it is what a pick does when Shift adds — while
  invert, grow, shrink and select-all compute a whole new set, and using the
  extending call there would leave the old indices behind (a grown set would
  look right only by accident, and a shrink could never remove anything).
  `clear_elements` keeps the nodes being edited; `Selection::clear` drops both,
  which is the object level's "None".
- **Selection state is not scene state — except visibility, which is.**
  The Select tools touch only `Selection`, so they never push an undo entry
  (and undoing an edit must not resurrect an old selection). Hide / Isolate /
  Unhide *are* scene edits: `Scene::content_eq` compares whole `Node`s with
  `visible` in them, which is what lets `with_edit` record them — hiding an
  already-hidden node pushes nothing. Hiding also deselects the node, so undo
  does not bring the selection back with it; the test that compares painted
  frames around a hide re-selects first, because a selected node drags its
  gizmo into the picture and changes the stroke count.
- **egui's `consume_key` matches modifiers exactly.** `Ctrl+A` and
  `Ctrl+Shift+A` (as `H` and `Shift+H`) are separate patterns, so no ordering
  is needed and a pattern that omits `SHIFT` will not fire while Shift is
  held; `Modifiers::NONE` means *no* modifier, which is how the plain-key
  hotkeys (H, U) are read.
- **In `app.rs` tests the viewport rect is the whole `SCREEN` (1000×700).**
  `CentralPanel` gets no margin in that path, so
  `viewport::project(app.camera.view_proj(SCREEN[0] / SCREEN[1]), world, SCREEN)`
  is a world point's exact screen position — check it against the gizmo origin
  before relying on it (the pattern the snapping tests use). At the default
  camera (distance 8, fov 55°) one world unit is about **84 px**, and the
  Edge/Vertex click tolerance is `distance * 0.02` world units (~13 px). The
  first edge tests here guessed ±40 px for a ±1 quad — half the real scale — and
  clicked the middle of the face instead of its edge.
- **The GPU edge overlay replaces the egui one, it does not stack with it.**
  When the GPU backend is active the edges are line geometry inside the frame
  (depth-tested, `LessEqual`, no depth write); `paint_gpu_viewport` therefore
  must not also call `paint_wireframe`, or every edge draws twice. The CPU
  fallback still paints the overlay on top, and both paths share the
  `viewport::EDGE_*` colours through `gpu_viewport::edge_vertices`. The frame
  must ask `overlay_mode`, not `self.wireframe` directly: it is what turns an
  *off* overlay into the selected meshes' outline at the edge level (the
  setting itself stays as the user left it), so a new call site that reads the
  field would draw nothing there. The **edge-level highlight is deliberately
  not built from that overlay**: `paint_selected_edges` projects each picked
  edge on top, unoccluded, on both paths — an editing aid exactly like the
  vertex dots — so it reads through whatever the depth-tested frame shows.
- **eframe picks its backend from `renderer: eframe::Renderer::Wgpu`** in
  `main.rs`. eframe's default features still include `glow`, so without that line
  the GPU path would silently never activate and the app would quietly stay on
  the CPU fallback. (`glow` could be dropped from `eframe` to trim CI/build
  time; the defaults are kept for safety.)
- **The UV editor is its own panel, not a viewport mode.** `uv_editor.rs`
  owns the canvas math (`v` grows *upward*; UV↔canvas mapping, picking, rubber
  band) and the panel state; `app.rs` owns the wiring and pushes committed
  edits through `with_edit`. A live drag mutates directly and pushes **one**
  undo entry on release, the same rule the 3D transforms follow. Its texture
  background follows the selected mesh's material, falling back to the material
  the Materials panel is editing, and `draw_canvas` takes it as an
  `Option<egui::TextureId>` — a canvas without one paints the plain dark tile.
- **UV islands weld on UV *continuity*, and packing never splits entries.**
  Two faces sharing a mesh edge join one island only when the UVs they assign
  at **both** shared endpoints match within the tolerance; `uv_pack_islands`
  then moves only the entries an island *owns* (an entry shared by several
  islands follows the first island that uses it in island order), so it never
  creates or re-points a uv index — a UV-editor selection survives a pack.
  `uv_repack` is a different grouping (coincident *positions*) and the two
  disagree on any mesh with a seam. A box-projected cube is **2** islands under
  the continuity rule, not 6: `uv_project_box` hands numerically identical UVs
  to faces that meet at a corner across axes.
- **A texture array's layers share one size, and its sampler needs a level.**
  Three things bite when wiring textures into the GPU pass: (a) array layers
  must all be the same dimensions, so the app resamples through
  `Texture::resized` before `set_textures` — a mismatched layer is filled white
  rather than failing; (b) `textureSample` (no level) inside a branch or with a
  per-face array index is a **derivative-uniformity** error in naga, which
  rejects the shader at pipeline creation — use `textureSampleLevel(..., 0.0)`;
  (c) `set_textures` changes the array's *view*, so every bind group holding it
  must be rebuilt (the uniform buffers stay), and the app must not upload per
  frame — it uploads on a `texture_epoch` revision.
- **One UV convention, two places that flip it.** `Texture` row 0 is `v = 0`;
  wgpu samples uv `(0, 0)` as texel `(0, 0)`, and egui's texture space has
  `(0, 0)` at the *top-left* — which is also row 0. So the CPU painter and the
  GPU pass both pass UVs straight through, and only the **UV editor** flips
  (`Rect::from_min_max(pos2(0.0, 1.0), pos2(1.0, 0.0))`), because it draws the
  UV space `v`-up like every modelling app. `painter.image` and `Shape::Mesh`
  are both `Shape::Mesh` under the hood — that is how the tests see them.
- **`GpuVertex` changes ripple into every vertex constructor and the dedup
  key.** `tex_layer` (offset 48, 4 bytes, `@location(4)`) is read by *all four*
  pipelines from one shared vertex buffer, so grid/edge/gizmo vertices simply
  name `WHITE_LAYER` (0.0) and are unaffected; forgetting a `size_of` assertion
  is what the layout test guards. Material layers join the dedup key, or two
  faces on different maps merge their shared corners and one of them samples
  the wrong texture.
- **The gizmo is hit-tested in *screen* space and grabbed on the *press*.**
  Two consequences, both easy to get wrong: (a) take the grab on the frame the
  button goes down — by the time egui reports `drag_started` the cursor has
  already left the handle, so hit-testing there always misses; and read the
  cursor from `ui.input(|i| i.pointer...)`, not `Response::hover_pos()`, which
  goes `None` on exactly the frame a press starts. (b) The layout the input pass
  computed is stored on the gizmo (`Gizmo::layout`) and reused by `paint_gizmo`;
  recomputing it in the paint pass would let what is drawn drift from what can
  be grabbed.
- **Snapping must never offer the geometry a drag is carrying.** A node's pivot
  sits on its own vertices, so at object level the whole node is excluded from
  the candidates (`gizmo::Targets::moving`) — otherwise the nearest vertex is
  the one the drag started on and the node can never move at all. At element
  level only the moving vertices are excluded, which is what makes welding a
  vertex onto its neighbour work. And the modifier test is per frame: read
  `modifiers.ctrl` inside the drag branch, not once when the drag begins, or
  holding Ctrl mid-drag does nothing.
- **An `egui::Context` keeps its input history across `ctx.run`.** Driving two
  apps through one context in a test is fine, but the frame times must never go
  backwards (a second app starting at `time: 0.0` trips emath's
  `debug_assert!(*last_time <= now)` and panics the test). Continue the clock
  instead: 0–3 for the first app, 4–7 for the second.
- **A drag spans frames, so its state must live on the app.** `ToolContext::drag`
  is constructed fresh each frame; `app.rs` keeps it in `self.tool_drag` and
  hands it to the tool and back. Leave either half out and Move/Scale/Rotate
  silently do nothing (or nothing is undoable). Likewise reach `on_pointer_up`
  on `drag_stopped()` alone — gating it on `dragged_by(Primary)` never fires on
  the release frame.
- **A drag's rectangle starts at the pointer's `press_origin`, not at the
  position that started the drag.** egui only calls a press a drag once the
  pointer has moved past `max_click_dist`, so on that frame `drag_started` is
  true but the pointer is already several pixels from where the button went
  down — a marquee built from it misses the corner the user drew. Read
  `ui.input(|i| i.pointer.press_origin())` inside the `drag_started` branch
  (falling back to the current position); it is `Some` for as long as a button
  is down. And a rectangle only lives while the button does: drop it on the
  first frame where no button is down and no `drag_stopped` arrived, or a lost
  release leaves it painted forever.
- **`Scene::nodes` is a `HashMap`, so `Scene::iter()` is not insertion order.**
  Anything built by walking a scene (`PickGeometry::build`'s element ids, the
  CPU painter's draw order) follows hash order, which is stable for the life of
  the map but **differs between runs**. Never assert node order in a test
  across two nodes; assert per-node order (ascending ids/indices) or compare
  sets. A test that got this wrong failed one run in three.
- **A face rectangle must not draw the vertex points.** The ID pass normally
  draws 1-px point sprites for vertices on top of their faces (since a vertex
  lies exactly on the surface, its pipeline uses `LessEqual`); a rectangle
  asking for *faces* has to skip the point pass entirely, or every vertex
  punches a one-pixel hole in its own face and a rectangle over that pixel
  could miss the face the image plainly shows. `pick_rect` does that off the
  `level`. The app also **trusts an empty GPU answer**: a rectangle over empty
  space selects nothing, and falling back to the select-through CPU projection
  there would contradict the picture.
- **`Pos2 - Pos2` is a `Vec2` in emath 0.29, and `Rect - Vec2` is not a thing**
  (the `Sub` impl on `Rect` takes a `Margin`, so it shrinks rather than moves).
  To move a rect into viewport-local space, build both corners with `pos2(x -
  offset.x, y - offset.y)`, and note that the CPU fallback wants *local* points
  while the GPU path wants *offscreen pixels* — the scale and the viewport rect
  are separate conversions (`marquee_pixel_rect`).
- **`Stroke::color` is a `ColorMode`, not a `Color32`, in egui 0.29.** A test
  that inspects painted shapes must compare against
  `egui::epaint::ColorMode::Solid(colour)`.
- **The undo stack stores `Scene`s only.** Material-library edits (creating a
  material, changing its colour) are therefore *not* undoable, while assigning
  one to a node (which changes the scene) is. That is also why deleting a
  material is restricted to the last, unassigned entry: a delete in the middle
  would renumber slots with no way to undo it. Making the stack generic over the
  document state is the fix, and a good next work-stream.
- **`Scene::insert` keeps a valid `NodeId`.** A duplicate that clones a node
  without clearing `copy.id` overwrites the node it came from; set
  `copy.id = NodeId::INVALID` to make the scene allocate a fresh one.
- **`Scene::iter()` is hash-ordered.** Tests that walk a scene's buffers or
  nodes must not assume creation order — count or look up instead.
- **Element indices are only meaningful at the level they were picked at.**
  `Selection` stores vertex/face indices in one set, so anything that changes
  `level` must call `Selection::clear_elements()` (the toolbar does). Skip it
  and a face index is read as a vertex index: the gizmo then pivots and drags
  the wrong geometry.
- **A live drag must pin its gizmo frame.** While dragging, the handles are
  drawn at `drag.anchor.origin` rather than at the recomputed centroid/bounds —
  otherwise the handles chase the geometry they are moving and every frame's
  hit-test is against a new pivot.
- **Panel order in `update()` matters.** The status bar is added first so it
  owns the bottom edge, then the UV editor, then the side panels, and the
  central viewport last. That is what keeps the viewport's rect — and its mouse
  input — clear of the bottom panels.
- **An unprojected mesh still has one UV slot.** `Mesh::ensure_attributes`
  installs a placeholder `Vec2::ZERO` so corner indices stay valid, so "has real
  UVs" is `uvs.len() > 1`, not `> 0`.
- **Use `egui::Rect::from_two_pos`, not `from_min_max`, for canvas rects.**
  `from_min_max` is a raw constructor that does *not* normalize; feeding it
  bottom-left/top-right (which is what a `v`-up canvas produces) yields an
  inverted rect with a negative height.
- **Build a `GpuRenderer` from the device eframe hands you**
  (`cc.wgpu_render_state` / `frame.wgpu_render_state()`), not from a wgpu
  `Instance` of your own. The headless test is the one place that creates its
  own instance, and it must skip gracefully when no adapter exists.
- **`scene.get()` returns `Result`** (not `Option`) — use `.ok()` / `if let Ok`.
- **`selection.selected_nodes()`** yields `NodeId` by value; in `.filter(|id|)`
  the binding is `&NodeId`, so deref with `*id` (not `**id`).
- **A Plugins-panel test asserts on `plugin_report()`, not on painted text.**
  The `CollapsingHeader` is closed by default and a closed body is never laid
  out, so its labels are not in the frame's shapes at all; the panel is a thin
  loop over the report so the content can be checked without a display.
- **Tools come from the plugin host, not `openforge-core` directly.**
  `OpenForgeApp::active_tool()` calls `host.make_tool("Move")` and only falls
  back to a bare `SelectTool` if the host somehow lacks a built-in, so
  importing `TransformTool` into `app.rs` is now an unused import (a warning,
  and the zero-warning bar). Same for formats: `import_file`/`export_selected`
  go through `host.import`/`host.export`, and `openforge_io` is reached for its
  data types (`MeshData`) only.

---

*When you finish a work-stream, update §7 (state + progress) and this file so the
next agent inherits an accurate picture.*
