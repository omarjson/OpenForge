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
complete; the remaining work is mostly application surface (GUI, GPU viewport,
interactive editors) and the plugin host.

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

## 4. Repo layout — 11 crates (270 tests total, all green)

| Crate | Tests | Purpose |
|-------|-------|---------|
| `openforge-core` | 22 | `Mesh`, `Scene`, nodes, selection, undo, tools. The data model. |
| `openforge-io` | 13 | Import/export: OBJ, glTF/GLB, PLY, STL. |
| `openforge-ops` | 17 | `MeshOps` extension trait: bevel, bridge, mirror, triangulate, flip, delete_edges, edge_loops, ring, **taper/twist/bend deformers**. |
| `openforge-materials` | 27 | Phong/Lambert materials, colors. |
| `openforge-render` | 51 | Camera/vertex/uniform plumbing **plus the wgpu viewport backend** (`gpu` feature): offscreen color+depth target, scene/grid/gizmo pipelines, frame stats. |
| `openforge-animation` | 51 | Keyframe tracks, interpolation, clips, player. |
| `openforge-algorithms` | 24 | Primitives, Catmull-Clark, Perlin/fBm noise, L-systems. |
| `openforge-uv` | 31 | Planar/cylindrical/spherical/box projections, unwrap, pack. |
| `openforge-boolean` | 9 | BSP CSG: union/subtract/intersect, watertight output, volume/watertight checks. |
| `openforge-rig` | 22 | Skeletal rigging + linear-blend skinning: bones, hierarchy, bind/pose, `SkinWeights`, `skin_mesh`. |
| `openforge-app` | 3 (bin) | eframe/egui GUI. `app.rs` (~1520 lines) + `viewport.rs` (CPU painter + picking) + `gpu_viewport.rs` (wgpu bridge). |

## 5. Architecture & conventions (follow these exactly)

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

**State:** baseline commit `2a201c8`. Full workspace builds with **0 warnings**,
**270 tests pass**, CI green on all 3 platforms. This session took work-stream
#1 — the wgpu viewport — to done:

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

**Verified this session:** `cargo build --workspace` → 0 warnings;
`cargo test --workspace` → 270 pass; `cargo build --release -p openforge-app`
(the CI build job) succeeds; the app was launched locally and logged
`eframe: ... Using wgpu.` with a Metal adapter and **no** `egui_wgpu` warnings.
**Not verified:** how the window actually looks (the agent has no display
access) — the offscreen pixel-readback test is the stand-in for that, so a human
should eyeball the viewport once.

**Progress toward full ZModeler parity: ~45%.** The engine layer is essentially
done and the viewport is no longer the bottleneck; what remains is still mostly
application surface. Prioritized next steps:

1. **Interactive UV editor** (best next value). The UV math (`openforge-uv`)
   exists; build a click-drag panel (select UVs, move/scale/rotate, pack, live
   preview). The GPU viewport makes a UV overlay cheap to add.
2. **Deepen the GPU viewport.** It renders shaded geometry; the editor surface
   around it is still missing — GPU selection outlines/wireframe, MSAA,
   materials-driven shaders instead of the single hard-coded Lambert pass, and
   depth-buffer picking. All additive to `GpuRenderer`.
3. **Plugin/service host.** ZModeler's core idea: 23 services + pluggable tools.
   Design an `openforge-plugin` trait + registry so tools/formats become
   pluggable instead of hard-wired. This is the long road to true parity.
4. **IO expansion.** Add COLLADA/DAE, OFF, or ASCII FBX (open subsets) to widen
   format coverage beyond the current 4.
5. **Materials node editor.** Replace basic Phong/Lambert with a node graph.

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
- **eframe picks its backend from `renderer: eframe::Renderer::Wgpu`** in
  `main.rs`. eframe's default features still include `glow`, so without that line
  the GPU path would silently never activate and the app would quietly stay on
  the CPU fallback. (`glow` could be dropped from `eframe` to trim CI/build
  time; the defaults are kept for safety.)
- **Build a `GpuRenderer` from the device eframe hands you**
  (`cc.wgpu_render_state` / `frame.wgpu_render_state()`), not from a wgpu
  `Instance` of your own. The headless test is the one place that creates its
  own instance, and it must skip gracefully when no adapter exists.
- **`scene.get()` returns `Result`** (not `Option`) — use `.ok()` / `if let Ok`.
- **`selection.selected_nodes()`** yields `NodeId` by value; in `.filter(|id|)`
  the binding is `&NodeId`, so deref with `*id` (not `**id`).

---

*When you finish a work-stream, update §7 (state + progress) and this file so the
next agent inherits an accurate picture.*
