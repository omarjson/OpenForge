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

## 4. Repo layout — 12 crates (413 `#[test]` fns; 415 tests pass, 2 doc-tests ignored)

| Crate | Tests | Purpose |
|-------|-------|---------|
| `openforge-core` | 31 | `Mesh`, `Scene`, nodes, selection, undo, tools. The data model. |
| `openforge-io` | 13 | Import/export: OBJ, glTF/GLB, PLY, STL. |
| `openforge-ops` | 17 | `MeshOps` extension trait: bevel, bridge, mirror, triangulate, flip, delete_edges, edge_loops, ring, **taper/twist/bend deformers**. |
| `openforge-materials` | 27 | Phong/Lambert materials, colors. |
| `openforge-render` | 61 | Camera/vertex/uniform plumbing **plus the wgpu viewport backend** (`gpu` feature): offscreen color+depth target (multi-sampled when supported), scene/grid/edge/gizmo pipelines, frame stats. |
| `openforge-animation` | 50 | Keyframe tracks, interpolation, clips, player. |
| `openforge-algorithms` | 24 | Primitives, Catmull-Clark, Perlin/fBm noise, L-systems. |
| `openforge-uv` | 37 | Planar/cylindrical/spherical/box projections, unwrap, pack, plus per-selection affine edits for the UV editor. |
| `openforge-boolean` | 9 | BSP CSG: union/subtract/intersect, watertight output, volume/watertight checks. |
| `openforge-rig` | 22 | Skeletal rigging + linear-blend skinning: bones, hierarchy, bind/pose, `SkinWeights`, `skin_mesh`. |
| `openforge-plugin` | 18 | The plugin host: a service registry, a tool registry (factories, not shared instances) and a format-filter registry that dispatches import/export on an extension. The engine's own tools and formats are ordinary plugins registered here. |
| `openforge-app` | 104 (bin) | eframe/egui GUI. `app.rs` (~4100 lines) + `gizmo.rs` (screen-space transform gizmo) + `snap.rs` (snapping settings + pure snap maths) + `viewport.rs` (CPU painter + picking) + `gpu_viewport.rs` (wgpu bridge) + `uv_editor.rs` (UV panel). |

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
  new and leaves other faces alone). The renderer only ever receives colours —
  `NodeColors` (one per node) with `build_scene_with`, or `NodeMaterials` (one
  per face) with `build_scene_materials`. Both viewport paths must resolve
  identically: `viewport::collect_triangles` for the CPU painter, the app's
  `node_materials()` for the GPU path, both through `viewport::material_base`
  (which folds in the node- and face-selection tints).
- **The UV editor is its own panel, not a viewport mode** (`uv_editor.rs` owns
  the canvas math and panel state, `app.rs` owns the wiring). See §8 for its
  gotchas: canvas rect construction, the placeholder UV slot, and one undo entry
  per drag.
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
workspace builds with **0 warnings**, **415 tests pass**, CI green on all 3
platforms. Ten work-streams are done — the **wgpu viewport** (#1), the
**interactive UV editor** (#2), the **transform gizmo** (#3), the **edge
overlay** (#4), the **sub-object transforms** (#5), **materials** (#6), the
**plugin host** (#7), the **GPU edge overlay + MSAA** (#8), **per-face
materials** (#9) and **snapping** (#10) — plus a progress report:

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
  plugins and the services, tools and formats they contributed.
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

**Verified (build/tests):** `cargo build --release -p openforge-app` (the CI
build job) → 0 warnings; `cargo build --workspace` → 0 warnings (and the test
build too); `cargo test --workspace` → **415 pass, 0 failures**
(`cargo build --workspace ... | grep -c warning` → 0). The gizmo, the wiring
behind the UV canvas, the side panels, the snapping gestures and now the plugin
seam are all covered *without* a display: the `app.rs` test module runs the real
`handle_central_viewport` / `handle_side_panels` against a plain
`egui::Context` with synthetic pointer events and asserts on both the gesture
(hover, grab, axis-constrained drag, release, undo) and the shapes the frame
painted, and on what the host did (import/export through a filter, a plugin's
filter taking effect, a rejected plugin changing nothing).
**Not verified:** how the window actually looks, and the gizmo, the UV canvas,
the new Plugins panel, the GPU edge overlay by eye (its depth-tested look
and the MSAA smoothing), the **per-face material shading on the GPU
path** (assign one face a different material and confirm the two paths agree
visually), and the **snapping feedback** (the gold marker on the vertex a drag
is about to land on, the live status readout, and Ctrl switching snapping on and
off mid-drag) — the agent has no display access, so a human should
look at all of those once. (An earlier session did launch the app on the wgpu
path and saw `eframe: ... Using wgpu.` with a Metal adapter, no `egui_wgpu`
warnings and no panic over a 12 s run.)
`rustfmt` is not installed for this toolchain, so formatting was matched by hand.
`cargo clippy -- -D warnings` still trips on **pre-existing** lints
in `openforge-core` (`needless_range_loop` in `edit.rs`) and `openforge-io`
(`collapsible_if`, …): untouched here, and not part of CI.

**Progress toward full ZModeler parity: see [`PROGRESS.md`](PROGRESS.md)** for
the per-area bars (`scripts/progress.sh --write` regenerates it; the mean of the
16 curated areas is **58.9%**, which the overall bar shows as 58%). The engine
layer is essentially done, and the largest application-surface gaps — the GPU
viewport, the UV editor and the editing UX around the gizmo — now have real
implementations, with materials shading per face on both paths as of #9 and the
gizmo snapping as of #10. Prioritized next steps:

1. **Deepen the GPU viewport** (still the best next value). MSAA, the
   depth-tested edge overlay and **per-face material shading** (work-stream #9)
   are in; what remains of the editor surface is **depth-buffer picking** so
   sub-object picking matches exactly what is drawn (an ID pass, or a depth read
   aligned with the CPU raycast). The shading itself is still flat colour +
   Lambert with no texture sampling — that is the material node graph in item 6.
   All additive to `GpuRenderer`.
2. **Extend the gizmo further** (`gizmo.rs`, all additive): plane and centre
   handles, and numeric entry for rotation/scale (translation already has
   DragValues in Properties). Snapping landed in #10 but is deliberately partial:
   only the gizmo's axis drags snap (the free-form view-plane drag does not) and
   only the grid and *vertices* are targets — ZModeler also snaps to edges,
   faces, midpoints and the grid of the *screen plane*, and its snapping works
   during free drags too. (Sub-object pivots are done: a vertex/face-level drag
   moves only its own geometry, via `gizmo::Targets` +
   `viewport::element_centroid`.)
3. **Deepen the UV editor.** No texture background yet, no island-level ops
   (weld/split/mirror), no snapping, and the 3D face selection is pushed into
   the UV selection on demand rather than staying in sync.
4. **Deepen the plugin host.** The host is in and the app goes through it, but
   three pieces are still hard-wired: tools are chosen through the app's
   `ActiveTool` enum rather than by walking `host.tool_entries()`, a `Service`
   carries a name and a description rather than behaviour, and plugins are
   compiled in rather than *loaded* — a `cdylib`/WASM filter discovered at
   runtime is the real prize. That is the long road to true parity.
5. **IO expansion.** Add COLLADA/DAE, OFF, or ASCII FBX (open subsets) to widen
   format coverage beyond the current 4.
6. **Materials node editor.** Replace basic Phong/Lambert with a node graph, and
   sample textures in the shader (the library already has textures; neither
   viewport samples them yet). Per-face material assignment — the reason slots
   exist per face — landed in #9.

**Honest gap to parity:** game-specific tooling (vehicles, LOD, dummies/TXD),
the full panel/dialog surface, and exact format breadth remain the largest
deltas and are many work-streams, not one push.

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
- **The GPU edge overlay replaces the egui one, it does not stack with it.**
  When the GPU backend is active the edges are line geometry inside the frame
  (depth-tested, `LessEqual`, no depth write); `paint_gpu_viewport` therefore
  must not also call `paint_wireframe`, or every edge draws twice. The CPU
  fallback still paints the overlay on top, and both paths share the
  `viewport::EDGE_*` colours through `gpu_viewport::edge_vertices`.
- **eframe picks its backend from `renderer: eframe::Renderer::Wgpu`** in
  `main.rs`. eframe's default features still include `glow`, so without that line
  the GPU path would silently never activate and the app would quietly stay on
  the CPU fallback. (`glow` could be dropped from `eframe` to trim CI/build
  time; the defaults are kept for safety.)
- **The UV editor is its own panel, not a viewport mode.** `uv_editor.rs`
  owns the canvas math (`v` grows *upward*; UV↔canvas mapping, picking, rubber
  band) and the panel state; `app.rs` owns the wiring and pushes committed
  edits through `with_edit`. A live drag mutates directly and pushes **one**
  undo entry on release, the same rule the 3D transforms follow.
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
