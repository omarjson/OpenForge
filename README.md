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

The project is a Cargo workspace of twelve crates; these are the main layers:

| Crate | Role | Notes |
|-------|------|-------|
| `openforge-core` | Scene graph, geometry, editing model | GUI- and renderer-agnostic. Nodes, meshes, selection, undo/redo, interactive tools, ray picking. |
| `openforge-io` | Import/export | OBJ, glTF/GLB, PLY, STL. Clean-room readers and writers. |
| `openforge-render` | GPU viewport backend | Camera/uniform/vertex plumbing plus a `wgpu` renderer behind the `gpu` feature: offscreen color + depth target (multi-sampled when the adapter supports it), the scene, grid, edge-overlay and axis-gizmo passes, a material texture array the scene pass samples per face, and an ID pass that resolves a click — or a whole rectangle — against exactly what the frame drew, down to single faces, edges and vertices. |
| `openforge-plugin` | Plugin host | Services, tools and format filters registered by name. The engine's own tools and formats are ordinary plugins here, so a replacement tool or a new format is a registration rather than an app change. |
| `openforge-app` | Desktop application | `eframe`/`egui` UI over `winit`/`wgpu`. Viewport, UV editor, outliner, properties, toolbar, hotkeys. |

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
- **`PluginHost`** — the seam plugins register into: a service registry, a tool
  registry (tools are *factories*, since a tool carries drag state) and a format
  filter registry that dispatches import/export on a file extension. Names are
  unique and a rejected plugin leaves the host exactly as it was.

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
  far side through the shading; at the **Edge** level an *off* overlay is
  drawn as the selected meshes' outline anyway, so the lines a pick can answer
  from are on screen while the setting itself stays where the user left it. On
  the wgpu backend the overlay is drawn inside the frame (depth-tested, so
  meshes occlude hidden edges) and the viewport is anti-aliased when the GPU
  supports it; the CPU fallback paints the edges on top.
- **Vertex / Edge / Face level** editing: select sub-objects and the gizmo moves to
  their centroid, where a drag moves *only* those vertices — the node's own
  transform is left alone. Edges are numbered by the mesh's own edge
  enumeration, one list every picker answers from, and a selected edge carries
  its endpoints into the drag like a corner carries a vertex, and is drawn as a
  bold highlight so what is selected reads in the picture, not just the status
  bar. Changing level
  clears the element selection, because a face index and a vertex index are not
  the same thing — and the viewport refuses to read one level's indices as
  another's, so a selected edge never tints a face that happens to share its
  number.
- **Selection tools** (the **Select** menu, the "Select:" toolbar row, or
  hotkeys): **Ctrl+A** takes everything at the current level — every visible
  node at the object level, every vertex/edge/face of the meshes being edited
  at a sub-object level — **Ctrl+Shift+A** clears it, **Ctrl+I** inverts it,
  and **Ctrl+Plus** / **Ctrl+Minus** grow or shrink it along the topology
  (faces through shared edges, edges through shared vertices, vertices through
  edges), the result counted in the status bar. **L** adds the edge **loop**
  through the selected edges and **Shift+L** the **ring** (Select menu or
  toolbar row): the loop carries straight on through vertices where exactly
  four edges meet, the ring crosses quads into the parallel band, and both add
  to the selection, so a second press changes nothing. **H** hides the selected
  nodes, **Shift+H** isolates them (everything else hidden) and **U** shows
  everything again; all three are undo steps, and hidden nodes drop out of the
  viewport and the picks at once — the outliner's per-node **Visible** box
  makes the same edit for one node, undo included.
- **Select Through** (**View → Select Through**, or **P**): the sub-object
  picks stop asking the depth-tested ID pass and answer from the projection
  instead, so a click takes the nearest element under the pointer and a
  rectangle everything it covers, whether the frame shows it or not. On a
  machine without wgpu the picks already answer that way, so the switch has
  nothing to change there.
- **Animate ▸ Spin Selected (2 s)** builds a one-turn clip over the selected
  nodes' current poses and opens the **Timeline**: the clip picker, **Space**
  to play/pause, Stop, Loop, and a scrub slider with the clip's key times
  listed under it. Playback poses the scene on the frame clock; it is a view
  over the clip, not an edit, so nothing lands on the undo stack — Stop
  rewinds to the start.
- **Game ▸ Set LOD 0–3 / Clear** tags the selected nodes with a detail level
  (0 = highest), **Isolate LOD n** shows only that level (one undo step, like
  Isolate; **U** shows everything again), and **Add Dummy** drops an
  attach-point dummy at the selection's centre. The **Properties panel** reports
  the triangle budget per LOD for the selection and the whole scene, and the
  outliner rows carry their `LOD n` tag.
- **Picking matches the picture.** In the wgpu viewport a Vertex-, Edge- or
  Face-level click resolves through an ID pass drawn with the same geometry,
  camera, depth test and back-face culling as the frame, so only what the image
  shows can be picked: a vertex hidden behind a surface stays unpicked, a face
  whose back is turned is skipped, and an edge wins the pixels it lies on
  without beating a face in front of it. The CPU raycast takes over when there
  is no GPU backend, or when nothing was under the cursor.
- **Rectangle selection.** With the Select tool, dragging draws a rectangle
  (the same overlay style as the gizmo) and on release everything inside it is
  selected — through the same ID pass, so only what the frame shows there is
  taken and occluded geometry is left out. A plain drag replaces the selection
  and **Shift + drag** adds to it; the status bar reads out the element count at
  Vertex, Edge and Face level. An edge counts as inside when either endpoint is
  or its line crosses the box. A rectangle dragged out and back with no area at
  all counts as the click it ended as. Off the wgpu backend the fallback
  projects the geometry instead — and, unlike the pass, it cannot tell what is
  hidden.
- **Snapping** (toolbar **Snap:**, or **G** to cycle Off → Grid → Vertex): a
  gizmo drag then lands on grid values — where the grabbed point *ends up*, not
  how far it travelled — with rotations and scales stepping too; in **Vertex**
  mode a Move drag lands the grabbed point on the vertex nearest the cursor
  (marked with a gold cross-hair while the drag is live). Hold **Ctrl** to
  invert snapping for as long as it is held, even mid-drag, and the status bar
  shows the value the drag is snapping to.
- **Textures.** A material can carry a **diffuse map**: the **Materials** panel
  picks one from the library (with a thumbnail and a **None** entry), **Load
  Texture…** imports a PNG or JPEG and maps it onto the material being edited,
  and **Checker** generates a procedural checkerboard. A **Checker Test** preset
  ships with a 256px checkerboard, so assigning it to a mesh shows texturing
  without importing anything. Textured faces are shaded as `texel × material
  colour` in *both* viewports — on the wgpu path through a texture array the
  scene pass samples per face, in the CPU fallback through an egui mesh that
  multiplies the texel by the shaded vertex colour — and the **UV editor** paints
  the map behind the UVs (toggle it in the Materials panel) with `v = 0` at the
  bottom of the tile. UVs come from the mesh: the sphere primitive carries
  spherical UVs, and imports bring their own.
- **The UV editor** (bottom panel) picks, rubber-band selects and drags UVs, and
  projects, packs and scales/rotates a selection. **Island** grows the selection
  to every UV of the islands it touches — islands are welded across mesh edges
  whose UVs match on both endpoints — and **Pack Islands** puts each island in
  its own cell of the `grid × grid` atlas with its aspect preserved (the other
  **Pack** groups by coincident positions instead).
- **Middle-drag** orbits the camera; **Shift + middle-drag** pans; **scroll** zooms.
- **Ctrl+Z** undoes, **Ctrl+Shift+Z** redoes, **Ctrl+D** duplicates the selection,
  **Delete** removes, **F** frames all.
- The **Properties** panel renames a node (one undo step per rename) and the
  **Materials** panel lists the library: pick an entry to edit its colour,
  shininess, opacity and texture, **Assign** it to the selection, or **New** to
  duplicate it. Assigned meshes are shaded with that material in both viewports.
  The outliner's **Visible** box hides or shows one node — one undo step, the
  same edit H makes.
- **File → Import/Export** loads and saves OBJ, glTF/GLB, PLY and STL. Both go
  through the plugin host's filter registry, so a third-party filter for another
  extension needs no change to the app: the open dialog is grouped by format,
  **Export selected as ▸** picks the format by name, and dropping files on the
  window imports them the same way. A path no filter knows is turned down with
  the list of extensions that would have worked, exporting is refused when the
  selection holds more than one mesh node rather than quietly writing one, and
  the statuses name the file and the format.
- The **Plugins** section of the Properties panel lists the loaded plugins and
  the services, tools and formats they contributed.
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
skeletal skinning, materials, **texture mapping end to end**,
**occlusion-exact sub-object picking — faces, edges and vertices, by click or
by rectangle —** and a **depth-tested `wgpu` viewport** are in; what remains is
mostly application surface. Natural next steps:

- Gizmo polish: plane and centre handles, and numeric entry for rotation and
  scale (translation already has fields). Snapping is in for the grid and for
  vertices; snapping to edges, faces and midpoints, and snapping during the
  free-form view-plane drag, are still to come.
- A deeper GPU viewport: shading depth on top of the ID pass that now resolves
  faces, edges and vertices — and rectangles of them — exactly as drawn (the
  select-through switch answers from the projection instead), with a picked
  edge already highlighted in the frame and the selection tools (all / none /
  invert, grow / shrink, edge loop / ring, hide / isolate) in place: shadowing
  and a configurable shading mode are the honest gaps left, then the
  materials node graph — normal/specular maps and mipmapped sampling on top
  of the diffuse map that samples today.
- A deeper UV editor: island weld/split/mirror, snapping, and live sync with the
  3D face selection (island selection and per-island packing are in).
- More standard formats (FBX-ascii, USDZ, glTF extensions) behind the same `openforge-io` traits.
- Dynamic plugin loading (`cdylib` filters) behind the host seam, which already
  mirrors the study architecture's service/tool/filter split.

## License

MIT OR Apache-2.0.
