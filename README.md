# OpenForge

A clean-room, open-source 3D modeling application written in Rust, targeting
**Windows, macOS, and Linux** from a single codebase.

> **Provenance & license note.** OpenForge is an *original* implementation. It was
> inspired by studying the **architecture** of a commercial plugin-based modeler
> (its node/scene-graph design, service/plugin split, and tool paradigms), but it
> copies **no proprietary code**, assets, or file-format secrets. Concepts and UI
> layouts are not copyrightable; the code here is written from scratch. It uses
> only **standard, openly-documented interchange formats** (OBJ, glTF 2.0, PLY,
> STL). It does **not** implement any proprietary/DRM-protected game formats and
> does not interact with any anti-tamper system.

## Architecture

The project is a Cargo workspace of eleven crates; these are the main layers:

| Crate | Role | Notes |
|-------|------|-------|
| `openforge-core` | Scene graph, geometry, editing model | GUI- and renderer-agnostic. Nodes, meshes, selection, undo/redo, interactive tools, ray picking. |
| `openforge-io` | Import/export | OBJ, glTF/GLB, PLY, STL. Clean-room readers and writers. |
| `openforge-render` | GPU viewport backend | Camera/uniform/vertex plumbing plus a `wgpu` renderer behind the `gpu` feature: offscreen color + depth target, and the scene, grid and axis-gizmo passes. |
| `openforge-app` | Desktop application | `eframe`/`egui` UI over `winit`/`wgpu`. Viewport, outliner, properties, toolbar, hotkeys. |

### Core concepts

- **`Scene` / `Node`** — a hierarchy of `Group`, `Dummy`, and `Mesh` nodes with
  TRS transforms and derived world matrices.
- **`Mesh`** — indexed n-gon polygon mesh with per-corner position/normal/UV
  attributes, so seams are preserved. Fan-triangulates for rendering and export.
- **`Selection`** — object/vertex/edge/face selection levels.
- **`UndoStack`** — snapshot-based undo/redo with labels.
- **`Tool`** — an interactive-tool trait. Ships with `SelectTool` and a
  `TransformTool` (move / scale / rotate) driven by viewport pointer rays.
- **`Camera`** — a perspective orbit camera with pan/zoom and NDC→ray picking.

## Building

Requires a stable Rust toolchain ([rustup](https://rustup.rs)).

```bash
# Build everything
cargo build

# Run the application (host platform)
cargo run -p openforge-app --release

# Run all tests
cargo test
```

### Packaging an installable app

`cargo build` produces a bare binary. To get a real, double-clickable app
bundle, use the packaging script:

```bash
./scripts/build-app.sh --release
```

- **macOS** → `dist/OpenForge.app` (ad-hoc signed). Install with
  `cp -R dist/OpenForge.app /Applications/`, or launch with `open dist/OpenForge.app`.
- **Windows** → `dist/OpenForge.exe` (wrap in an installer such as Inno Setup or MSIX).
- **Linux** → `dist/OpenForge` (package as an AppImage or `.deb`).

### Cross-platform targets

The `openforge-core` and `openforge-io` crates are pure Rust and cross-compile
to every desktop target with no extra dependencies:

```bash
rustup target add x86_64-pc-windows-msvc x86_64-unknown-linux-gnu x86_64-apple-darwin

# Type-check the portable crates for each OS
cargo check -p openforge-core -p openforge-io --target x86_64-pc-windows-msvc
cargo check -p openforge-core -p openforge-io --target x86_64-unknown-linux-gnu
cargo check -p openforge-core -p openforge-io --target x86_64-apple-darwin
```

The `openforge-app` GUI crate links against each platform's native windowing and
graphics stack (via `winit`/`wgpu`), so build it **on** (or with a full SDK/CI
runner for) the target OS:

- **Windows** — `cargo build --release --target x86_64-pc-windows-msvc` (MSVC toolchain).
- **macOS** — `cargo build --release` (universal or `--target aarch64-apple-darwin`).
- **Linux** — `cargo build --release --target x86_64-unknown-linux-gnu` (needs X11/Wayland dev libs for `winit`).

A sample GitHub Actions matrix is shown below.

## Using the app

- **Left-drag** selects (with the Select tool), or drags the selection through
  the view plane (with a transform tool).
- **Q / W / E / R** switch Select / Move / Scale / Rotate. With a transform tool
  active a **transform gizmo** sits on the selection: drag an axis arm to move or
  scale along that axis, or an axis ring to rotate around it. The grabbed handle
  highlights and is labelled while the drag is live, and the whole gesture is a
  single undo step. Handles keep a constant size on screen whatever the camera
  distance.
- **Z** cycles the edge overlay: off, then the selected meshes' edges as a
  selection outline, then every visible mesh's wireframe (**View → Wireframe**
  shows the current mode). Edges of the selected faces are highlighted, and
  edges seen only from behind are dropped so a closed mesh doesn't draw its own
  far side through the shading.
- **Vertex / Face level** editing: select sub-objects and the gizmo moves to their
  centroid, where a drag moves *only* those vertices — the node's own transform
  is left alone. Changing level clears the element selection, because a face
  index and a vertex index are not the same thing.
- **Middle-drag** orbits the camera; **Shift + middle-drag** pans; **scroll** zooms.
- **Ctrl+Z** undoes, **Ctrl+Shift+Z** redoes, **Ctrl+D** duplicates the selection,
  **Delete** removes, **F** frames all.
- The **Properties** panel renames a node (one undo step per rename) and the
  **Materials** panel lists the library: pick an entry to edit its colour,
  shininess and opacity, **Assign** it to the selection, or **New** to duplicate
  it. Assigned meshes are shaded with that material in both viewports. The
  outliner's checkbox hides a node from the viewport.
- **File → Import/Export** loads and saves OBJ, glTF/GLB, PLY and STL.
- **Create** menu adds primitives (box, sphere, cylinder, cone, plane).

## Testing

```bash
cargo test                 # unit + integration tests
```

`openforge-core` and `openforge-io` carry unit tests for geometry, the scene
graph, undo, picking, and every format. `openforge-io/tests/roundtrip.rs` writes
real files to disk and re-imports them for all four formats.

## Roadmap

This is a deliberately solid *foundation*, not a feature-complete clone of a
mature commercial modeler. Sub-element editing, deformers, boolean CSG,
skeletal skinning, materials, and a **depth-tested `wgpu` viewport** are in;
what remains is mostly application surface. Natural next steps:

- Wireframe and selection outlines in the viewport, and gizmo polish: plane and
  centre handles, snapping, numeric entry, and per-face gizmo placement.
- A deeper GPU viewport: MSAA, a materials-driven shader instead of one Lambert
  pass, and depth-buffer picking so picking matches exactly what is drawn.
- More standard formats (FBX-ascii, USDZ, glTF extensions) behind the same `openforge-io` traits.
- Dynamic plugin loading (`cdylib` filters) mirroring the study architecture.

## License

MIT OR Apache-2.0.
