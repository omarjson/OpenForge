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

The project is a Cargo workspace of three crates, mirroring the layered design of
modern editors:

| Crate | Role | Notes |
|-------|------|-------|
| `openforge-core` | Scene graph, geometry, editing model | GUI- and renderer-agnostic. Nodes, meshes, selection, undo/redo, interactive tools, ray picking. |
| `openforge-io` | Import/export | OBJ, glTF/GLB, PLY, STL. Clean-room readers and writers. |
| `openforge-app` | Desktop application | `eframe`/`egui` UI (which bundles `winit` for windowing and `wgpu` for GPU). Viewport, outliner, properties, toolbar, hotkeys. |

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

- **Left-drag** selects (with the Select tool) or manipulates the selection
  (Move/Scale/Rotate).
- **Middle-drag** orbits the camera; **Shift + middle-drag** pans; **scroll** zooms.
- **Q / W / E / R** switch Select / Move / Scale / Rotate.
- **Ctrl+Z** undoes, **Ctrl+Shift+Z** redoes, **Delete** removes, **F** frames all.
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
mature commercial modeler. Natural next steps:

- Sub-element (vertex/edge/face) editing and selection highlighting in the viewport.
- Materials, textures, and a shading model beyond flat headlight shading.
- A GPU (`wgpu`) viewport backend for large scenes.
- Skeletal animation and skinning.
- More standard formats (FBX-ascii, USDZ, glTF extensions) behind the same `openforge-io` traits.
- Dynamic plugin loading (`cdylib` filters) mirroring the study architecture.

## License

MIT OR Apache-2.0.
