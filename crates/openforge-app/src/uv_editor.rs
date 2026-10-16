//! The interactive UV editor panel.
//!
//! A 2D canvas over the active mesh's UVs: pick, rubber-band select, and drag
//! UVs, plus buttons for the engine's projection/pack/affine operations. The
//! 3D viewport keeps showing the same mesh, so an edit is visible in both
//! halves at once.
//!
//! The canvas is painted with egui rather than through the GPU renderer: it is
//! 2D line work with no depth, lighting or picking, so a GPU pass would add a
//! target, a bind group and a readback for no visual gain.
//!
//! This module owns the pure canvas math and the panel's state; [`crate::app`]
//! owns the wiring (undo, the scene, the buttons).

use eframe::egui;
use glam::Vec2;
use openforge_core::mesh::Mesh;
use openforge_core::scene::Scene;
use openforge_core::NodeId;
use openforge_uv::UvOps;
use std::collections::BTreeSet;

/// Smallest shown `u`/`v` (the canvas covers `VIEW_MIN .. VIEW_MIN + VIEW_SPAN`).
///
/// The range is wider than `0..1` so UVs that fall outside the tile — which
/// projections and packs regularly produce — stay visible and pickable rather
/// than piling up on the edge.
pub const VIEW_MIN: f32 = -0.2;
/// Width of the shown UV range.
pub const VIEW_SPAN: f32 = 1.4;
/// Click radius, in pixels, for picking a UV entry.
pub const PICK_RADIUS: f32 = 8.0;

/// Tolerance the editor's island operations use to weld UVs across a shared
/// mesh edge: wide enough to absorb the float noise a projection leaves
/// behind, far below any meaningful UV distance.
pub const ISLAND_TOLERANCE: f32 = 1e-4;

/// Gutter left around every island by the Pack Islands button, in UV units.
pub const ISLAND_MARGIN: f32 = 0.02;

/// The canvas size in points that the panel aims for, clamped to what is free.
pub const CANVAS_SIZE: f32 = 300.0;

/// Map a UV coordinate onto the canvas.
///
/// `u` grows to the right and `v` grows **up** (the usual UV convention), so
/// `v = 0` sits at the bottom of the canvas.
pub fn uv_to_canvas(uv: Vec2, canvas: egui::Rect) -> egui::Pos2 {
    let f = |v: f32| (v - VIEW_MIN) / VIEW_SPAN;
    egui::pos2(
        canvas.left() + f(uv.x) * canvas.width(),
        canvas.bottom() - f(uv.y) * canvas.height(),
    )
}

/// Inverse of [`uv_to_canvas`].
pub fn canvas_to_uv(pos: egui::Pos2, canvas: egui::Rect) -> Vec2 {
    let w = canvas.width().max(1e-6);
    let h = canvas.height().max(1e-6);
    Vec2::new(
        (pos.x - canvas.left()) / w * VIEW_SPAN + VIEW_MIN,
        (canvas.bottom() - pos.y) / h * VIEW_SPAN + VIEW_MIN,
    )
}

/// The UV-space grid step a snapped drag lands on.
pub const SNAP_STEP: f32 = 0.05;

/// How close two UVs must be for the Weld tool to pull them together.
pub const WELD_TOLERANCE: f32 = 0.005;

/// Round a UV-space delta to the snap grid.
///
/// The drag applies the difference of *rounded totals* through this (see
/// [`UvDrag::total`]), so a slow drag still lands exactly on the grid once it
/// has moved far enough, instead of rounding each frame's step to nothing.
/// A non-positive step passes the delta through unchanged.
pub fn snap_uv(delta: Vec2, step: f32) -> Vec2 {
    if !delta.is_finite() || !step.is_finite() || step <= 0.0 {
        return delta;
    }
    Vec2::new(
        (delta.x / step).round() * step,
        (delta.y / step).round() * step,
    )
}

/// The `0..1` tile, inset inside the canvas.
///
/// Uses [`egui::Rect::from_two_pos`], not `from_min_max`: `v` grows upwards, so
/// the tile's corners arrive in bottom-left/top-right order and `from_min_max`
/// (which is a raw constructor and does *not* normalize) would hand back an
/// inverted rect with a negative height.
pub fn tile_rect(canvas: egui::Rect) -> egui::Rect {
    egui::Rect::from_two_pos(uv_to_canvas(Vec2::ZERO, canvas), uv_to_canvas(Vec2::ONE, canvas))
}

/// The UV entry nearest to `pos`, within [`PICK_RADIUS`] pixels.
///
/// `None` when nothing is close enough (or the mesh has no usable UVs).
pub fn pick_uv(mesh: &Mesh, pos: egui::Pos2, canvas: egui::Rect) -> Option<u32> {
    let mut best: Option<(f32, u32)> = None;
    for (i, uv) in mesh.uvs.iter().enumerate() {
        if !uv.is_finite() {
            continue;
        }
        let d = uv_to_canvas(*uv, canvas).distance(pos);
        if d <= PICK_RADIUS && best.map(|(bd, _)| d < bd).unwrap_or(true) {
            best = Some((d, i as u32));
        }
    }
    best.map(|(_, i)| i)
}

/// UV indices whose canvas position falls inside the screen-space rectangle
/// spanned by `a` and `b` — a rubber-band selection.
pub fn band_select(mesh: &Mesh, canvas: egui::Rect, a: egui::Pos2, b: egui::Pos2) -> Vec<u32> {
    let band = egui::Rect::from_two_pos(a, b);
    mesh.uvs
        .iter()
        .enumerate()
        .filter(|(_, uv)| uv.is_finite() && band.contains(uv_to_canvas(**uv, canvas)))
        .map(|(i, _)| i as u32)
        .collect()
}

/// Every UV entry of every island the given entries touch.
///
/// A UV index can be shared by several faces, so this walks the mesh's island
/// partition (via [`UvOps::uv_face_islands`]) and unions the entries of each
/// island that uses at least one of `indices`. The result is sorted and
/// deduplicated like every other index list the panel produces; unknown or
/// out-of-range indices contribute nothing.
pub fn island_selection(mesh: &Mesh, indices: &[u32], tolerance: f32) -> Vec<u32> {
    let wanted: BTreeSet<u32> = indices.iter().copied().collect();
    let mut selected: BTreeSet<u32> = BTreeSet::new();
    for faces in mesh.uv_face_islands(tolerance) {
        let entries = mesh.uv_island_uv_indices(&faces);
        if entries.iter().any(|e| wanted.contains(e)) {
            selected.extend(entries);
        }
    }
    selected.into_iter().collect()
}

/// The face-outline segments to draw, as UV pairs.
///
/// One segment per face edge (in corner order, wrapping), skipping faces with
/// fewer than two corners. Kept separate from the painter so it is testable
/// without a UI.
pub fn face_edges(mesh: &Mesh) -> Vec<(Vec2, Vec2)> {
    let mut out = Vec::new();
    for face in &mesh.faces {
        let n = face.corners.len();
        if n < 2 {
            continue;
        }
        for i in 0..n {
            let a = mesh.uvs.get(face.corners[i].uv as usize);
            let b = mesh.uvs.get(face.corners[(i + 1) % n].uv as usize);
            if let (Some(a), Some(b)) = (a, b) {
                if a.is_finite() && b.is_finite() {
                    out.push((*a, *b));
                }
            }
        }
    }
    out
}

/// A move-drag in progress.
pub struct UvDrag {
    /// Scene snapshot from the moment the drag began, for one undo entry.
    pub before: Scene,
    /// Canvas position of the previous frame.
    pub last: egui::Pos2,
    /// Total UV-space delta since the drag began, before snapping — the snap
    /// rounds this total, not each frame's step, so a slow drag still reaches
    /// the grid instead of rounding every step to nothing.
    pub total: Vec2,
}

/// Persistent state of the UV editor panel.
pub struct UvEditor {
    /// Whether the panel is shown.
    pub open: bool,
    /// Node whose mesh [`selected`](Self::selected) indexes into.
    pub node: Option<NodeId>,
    /// Selected UV indices.
    pub selected: BTreeSet<u32>,
    /// Move-drag in progress.
    pub drag: Option<UvDrag>,
    /// Rubber band in progress, as its two canvas corners.
    pub band: Option<(egui::Pos2, egui::Pos2)>,
    /// Grid size used by the Pack button.
    pub grid: u32,
    /// Scale factor applied to the selection by the transform button.
    pub scale: f32,
    /// Rotation in degrees applied to the selection by the transform button.
    pub rotate_deg: f32,
    /// Whether drag translations land on the [`SNAP_STEP`] grid.
    pub snap: bool,
}

impl Default for UvEditor {
    fn default() -> Self {
        Self {
            open: false,
            node: None,
            selected: BTreeSet::new(),
            drag: None,
            band: None,
            grid: 2,
            scale: 1.0,
            rotate_deg: 0.0,
            snap: false,
        }
    }
}

impl UvEditor {
    /// Point the editor at `node`'s mesh: drop a selection that belongs to a
    /// different mesh and prune indices that no longer exist.
    pub fn retarget(&mut self, node: Option<NodeId>, uv_count: usize) {
        if self.node != node {
            self.node = node;
            self.selected.clear();
            self.drag = None;
            self.band = None;
        }
        self.selected.retain(|&i| (i as usize) < uv_count);
    }

    /// The selection as the engine's index slice wants it (sorted, `u32`).
    pub fn indices(&self) -> Vec<u32> {
        self.selected.iter().copied().collect()
    }

    /// Pivot for selection transforms: the center of the selection's UV bounds,
    /// or the middle of the tile when nothing (usable) is selected.
    pub fn pivot(&self, mesh: &Mesh) -> Vec2 {
        match mesh.uv_bounds_selected(&self.indices()) {
            Some(b) => (b.min + b.max) * 0.5,
            None => Vec2::splat(0.5),
        }
    }

    /// Replace the selection with `indices`.
    pub fn select_only(&mut self, indices: impl IntoIterator<Item = u32>) {
        self.selected = indices.into_iter().collect();
    }

    /// Add `indices` to the selection.
    pub fn select_more(&mut self, indices: impl IntoIterator<Item = u32>) {
        self.selected.extend(indices);
    }

    /// Drop the whole selection.
    pub fn deselect_all(&mut self) {
        self.selected.clear();
    }
}

/// Draw the UV canvas: the tile (with the material's texture behind it when
/// `background` names one), a quarter grid, every face outline and a dot per
/// UV entry (selected entries highlighted, `hovered` ringed).
pub fn draw_canvas(
    painter: &egui::Painter,
    canvas: egui::Rect,
    mesh: &Mesh,
    editor: &UvEditor,
    hovered: Option<u32>,
    background: Option<egui::TextureId>,
) {
    let tile = tile_rect(canvas);
    // Dim the out-of-tile area so the 0..1 region reads as the UV space proper.
    painter.rect_filled(canvas, 0.0, egui::Color32::from_gray(20));
    match background {
        // The image's last row lands at the tile's top: UV `v = 0` is the
        // texture's row 0, which sits at the *bottom* of the UV space.
        Some(id) => {
            painter.image(
                id,
                tile,
                egui::Rect::from_min_max(egui::pos2(0.0, 1.0), egui::pos2(1.0, 0.0)),
                egui::Color32::WHITE,
            );
        }
        None => {
            painter.rect_filled(tile, 0.0, egui::Color32::from_rgb(38, 40, 46));
        }
    }

    let grid_stroke = if background.is_some() {
        // Over a texture the dark grid disappears; a light veil keeps the
        // quarter divisions readable without hiding the texels.
        egui::Stroke::new(1.0_f32, egui::Color32::from_white_alpha(48))
    } else {
        egui::Stroke::new(1.0_f32, egui::Color32::from_gray(58))
    };
    for i in 1..4 {
        let t = i as f32 / 4.0;
        let x = tile.left() + t * tile.width();
        let y = tile.top() + t * tile.height();
        painter.line_segment(
            [egui::pos2(x, tile.top()), egui::pos2(x, tile.bottom())],
            grid_stroke,
        );
        painter.line_segment(
            [egui::pos2(tile.left(), y), egui::pos2(tile.right(), y)],
            grid_stroke,
        );
    }

    let edge_stroke = egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(104, 136, 176));
    for (a, b) in face_edges(mesh) {
        painter.line_segment([uv_to_canvas(a, canvas), uv_to_canvas(b, canvas)], edge_stroke);
    }

    for (i, uv) in mesh.uvs.iter().enumerate() {
        if !uv.is_finite() {
            continue;
        }
        let p = uv_to_canvas(*uv, canvas);
        if editor.selected.contains(&(i as u32)) {
            painter.circle_filled(p, 3.5, egui::Color32::from_rgb(255, 170, 60));
        } else {
            painter.circle_filled(p, 2.0, egui::Color32::from_rgb(150, 190, 230));
        }
    }

    if let Some(uv) = hovered.and_then(|h| mesh.uvs.get(h as usize)) {
        if uv.is_finite() {
            painter.circle_stroke(
                uv_to_canvas(*uv, canvas),
                6.0,
                egui::Stroke::new(1.5_f32, egui::Color32::WHITE),
            );
        }
    }

    // The tile border goes on last so it stays readable over the wireframe.
    painter.rect_stroke(tile, 0.0, egui::Stroke::new(1.0_f32, egui::Color32::from_gray(120)));
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    fn canvas() -> egui::Rect {
        egui::Rect::from_min_max(egui::pos2(100.0, 50.0), egui::pos2(500.0, 450.0))
    }

    fn mesh_with_uvs(uvs: &[Vec2]) -> Mesh {
        let mut m = Mesh::new();
        m.uvs = uvs.to_vec();
        m
    }

    /// Two quads sharing the position edge `{1, 2}`, each corner on its own uv
    /// entry and matching UVs along the shared edge: one island, entries `0..8`.
    fn two_quads() -> Mesh {
        let mut m = mesh_with_uvs(&[
            Vec2::new(0.0, 0.0), // 0: face 0, pos 0
            Vec2::new(1.0, 0.0), // 1: face 0, pos 1 (shared)
            Vec2::new(1.0, 1.0), // 2: face 0, pos 2 (shared)
            Vec2::new(0.0, 1.0), // 3: face 0, pos 3
            Vec2::new(1.0, 0.0), // 4: face 1, pos 1 (shared, matches 1)
            Vec2::new(2.0, 0.0), // 5: face 1, pos 4
            Vec2::new(2.0, 1.0), // 6: face 1, pos 5
            Vec2::new(1.0, 1.0), // 7: face 1, pos 2 (shared, matches 2)
        ]);
        for p in [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0), // shared
            (1.0, 1.0, 0.0), // shared
            (0.0, 1.0, 0.0),
            (2.0, 0.0, 0.0),
            (2.0, 1.0, 0.0),
        ] {
            m.add_vertex(Vec3::new(p.0, p.1, p.2));
        }
        m.add_face(&[0, 1, 2, 3]);
        m.add_face(&[1, 4, 5, 2]);
        for (i, c) in m.faces[0].corners.iter_mut().enumerate() {
            c.uv = i as u32;
        }
        for (i, c) in m.faces[1].corners.iter_mut().enumerate() {
            c.uv = 4 + i as u32;
        }
        m
    }

    #[test]
    fn uv_to_canvas_puts_v_up_and_u_right() {
        let c = canvas();
        let tile = tile_rect(c);
        // (0,0) is the tile's bottom-left, (1,1) its top-right.
        let zero = uv_to_canvas(Vec2::ZERO, c);
        let one = uv_to_canvas(Vec2::ONE, c);
        assert!((zero - tile.left_bottom()).length() < 1e-3, "{zero:?}");
        assert!((one - tile.right_top()).length() < 1e-3, "{one:?}");
        // v grows upwards: a bigger v is a smaller screen y.
        assert!(uv_to_canvas(Vec2::new(0.0, 0.9), c).y < uv_to_canvas(Vec2::new(0.0, 0.1), c).y);
        // u grows rightwards.
        assert!(uv_to_canvas(Vec2::new(0.9, 0.0), c).x > uv_to_canvas(Vec2::new(0.1, 0.0), c).x);
    }

    #[test]
    fn tile_is_a_square_inside_the_canvas() {
        let c = canvas();
        let tile = tile_rect(c);
        assert!(c.contains_rect(tile));
        assert!((tile.width() - tile.height()).abs() < 1e-3);
        // The padding on each side is equal, so the tile is centered.
        assert!((tile.left() - c.left() - (c.right() - tile.right())).abs() < 1e-3);
    }

    #[test]
    fn canvas_round_trips_uvs() {
        let c = canvas();
        for uv in [
            Vec2::ZERO,
            Vec2::ONE,
            Vec2::splat(0.5),
            Vec2::new(-0.2, 1.2),
            Vec2::new(0.13, 0.87),
        ] {
            let back = canvas_to_uv(uv_to_canvas(uv, c), c);
            assert!((back - uv).length() < 1e-4, "{uv:?} -> {back:?}");
        }
        // A degenerate canvas must not produce NaN.
        let tiny = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(0.0, 0.0));
        assert!(canvas_to_uv(egui::pos2(0.0, 0.0), tiny).is_finite());
    }

    #[test]
    fn pick_uv_finds_the_nearest_entry_within_radius() {
        let c = canvas();
        let mesh = mesh_with_uvs(&[Vec2::new(0.2, 0.2), Vec2::new(0.8, 0.8)]);
        let on_first = uv_to_canvas(Vec2::new(0.2, 0.2), c);
        assert_eq!(pick_uv(&mesh, on_first, c), Some(0));
        // A few pixels off still hits the same entry.
        assert_eq!(pick_uv(&mesh, on_first + egui::vec2(3.0, -3.0), c), Some(0));
        let on_second = uv_to_canvas(Vec2::new(0.8, 0.8), c);
        assert_eq!(pick_uv(&mesh, on_second, c), Some(1));
        // Empty space and an empty mesh miss.
        assert_eq!(pick_uv(&mesh, c.center(), c), None);
        assert_eq!(pick_uv(&Mesh::new(), c.center(), c), None);
        // A non-finite entry is never picked.
        let ghost = mesh_with_uvs(&[Vec2::new(f32::NAN, 0.0)]);
        assert_eq!(pick_uv(&ghost, uv_to_canvas(Vec2::ZERO, c), c), None);
    }

    #[test]
    fn band_select_takes_only_contained_entries() {
        let c = canvas();
        let mesh = mesh_with_uvs(&[
            Vec2::new(0.1, 0.1),
            Vec2::new(0.2, 0.2),
            Vec2::new(0.9, 0.9),
        ]);
        // A band over the lower-left quadrant covers the first two only.
        let a = uv_to_canvas(Vec2::ZERO, c);
        let b = uv_to_canvas(Vec2::splat(0.5), c);
        assert_eq!(band_select(&mesh, c, a, b), vec![0, 1]);
        // The band is direction-agnostic.
        assert_eq!(band_select(&mesh, c, b, a), vec![0, 1]);
        // A band over everything covers every finite entry.
        assert_eq!(
            band_select(&mesh, c, uv_to_canvas(Vec2::new(-1.0, -1.0), c), uv_to_canvas(Vec2::splat(2.0), c)),
            vec![0, 1, 2]
        );
        // A zero-size band selects nothing.
        assert!(band_select(&mesh, c, c.center(), c.center()).is_empty());
    }

    #[test]
    fn selecting_an_entry_of_a_welded_pair_returns_its_whole_island() {
        let m = two_quads();
        // Entry 2 belongs to face 0; both faces weld into one island.
        assert_eq!(
            island_selection(&m, &[2], ISLAND_TOLERANCE),
            vec![0, 1, 2, 3, 4, 5, 6, 7]
        );
    }

    #[test]
    fn a_shifted_shared_edge_keeps_each_island_to_itself() {
        let mut m = two_quads();
        m.uvs[4] = Vec2::new(0.8, 0.0); // no longer matches face 0's (1, 0)
        // Face 0's entries come back, face 1's stay out...
        assert_eq!(island_selection(&m, &[2], ISLAND_TOLERANCE), vec![0, 1, 2, 3]);
        // ...and the other island grown from its own entry does the reverse.
        assert_eq!(island_selection(&m, &[5], ISLAND_TOLERANCE), vec![4, 5, 6, 7]);
    }

    #[test]
    fn entries_from_several_islands_unite_sorted_and_deduplicated() {
        let mut m = two_quads();
        m.uvs[4] = Vec2::new(0.8, 0.0); // face 1 is now its own island
        // One entry per island: the union of both islands, sorted.
        assert_eq!(
            island_selection(&m, &[5, 2], ISLAND_TOLERANCE),
            vec![0, 1, 2, 3, 4, 5, 6, 7]
        );
        // Several entries of one island still contribute it exactly once.
        assert_eq!(island_selection(&m, &[2, 2, 3], ISLAND_TOLERANCE), vec![0, 1, 2, 3]);
    }

    #[test]
    fn unnamed_empty_and_out_of_range_entries_select_nothing() {
        let mut m = two_quads();
        m.uvs.push(Vec2::new(0.5, 0.5)); // entry 8: no corner uses it
        assert!(island_selection(&m, &[8], ISLAND_TOLERANCE).is_empty());
        assert!(island_selection(&m, &[], ISLAND_TOLERANCE).is_empty());
        assert!(island_selection(&m, &[99], ISLAND_TOLERANCE).is_empty());
        // A dead index next to a live one is ignored, not fatal.
        assert_eq!(
            island_selection(&m, &[2, 99], ISLAND_TOLERANCE),
            vec![0, 1, 2, 3, 4, 5, 6, 7]
        );
    }

    #[test]
    fn the_tolerance_decides_whether_shared_edge_uvs_weld() {
        // 1e-2 apart: split at the editor's tolerance, one island above it.
        let mut apart = two_quads();
        apart.uvs[4] = Vec2::new(1.01, 0.0);
        assert_eq!(island_selection(&apart, &[2], ISLAND_TOLERANCE), vec![0, 1, 2, 3]);
        assert_eq!(
            island_selection(&apart, &[2], 0.02),
            vec![0, 1, 2, 3, 4, 5, 6, 7]
        );
        // A 1e-6 gap is float noise: welded at the editor's tolerance.
        let mut noise = two_quads();
        noise.uvs[4] = Vec2::new(1.0 + 1e-6, 0.0);
        assert_eq!(
            island_selection(&noise, &[2], ISLAND_TOLERANCE),
            vec![0, 1, 2, 3, 4, 5, 6, 7]
        );
    }

    #[test]
    fn a_mesh_without_faces_has_no_islands() {
        // UVs but no faces: nothing to walk.
        let m = mesh_with_uvs(&[Vec2::ZERO, Vec2::ONE]);
        assert!(island_selection(&m, &[0], ISLAND_TOLERANCE).is_empty());
        // A truly empty mesh stays empty for any index.
        assert!(island_selection(&Mesh::new(), &[0, 1], ISLAND_TOLERANCE).is_empty());
    }

    #[test]
    fn face_edges_walks_corners_and_wraps() {
        let mut m = Mesh::new();
        m.uvs = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(1.0, 1.0),
        ];
        // One triangle: three edges, the last wrapping back to the first corner.
        m.faces.push(openforge_core::mesh::Face {
            corners: vec![
                openforge_core::mesh::Corner { position: 0, normal: 0, uv: 0 },
                openforge_core::mesh::Corner { position: 0, normal: 0, uv: 1 },
                openforge_core::mesh::Corner { position: 0, normal: 0, uv: 2 },
            ],
            material: 0,
        });
        let edges = face_edges(&m);
        assert_eq!(edges.len(), 3);
        assert_eq!(edges[2], (Vec2::new(1.0, 1.0), Vec2::new(0.0, 0.0)));
        // A dangling uv index contributes no edge instead of panicking.
        m.faces[0].corners[1].uv = 99;
        assert_eq!(face_edges(&m).len(), 1);
    }

    #[test]
    fn retarget_clears_on_mesh_change_and_prunes_dead_indices() {
        let mut e = UvEditor::default();
        let a = NodeId(1);
        let b = NodeId(2);
        e.retarget(Some(a), 4);
        e.select_only([0, 1, 2, 3]);
        // Same mesh, fewer UVs: the dead indices go, the rest stay.
        e.retarget(Some(a), 2);
        assert_eq!(e.indices(), vec![0, 1]);
        // Different mesh: the selection is dropped entirely.
        e.retarget(Some(b), 8);
        assert!(e.selected.is_empty());
        // Nothing selected at all.
        e.retarget(None, 0);
        assert!(e.selected.is_empty());
        assert_eq!(e.node, None);
    }

    #[test]
    fn pivot_is_the_selection_center_or_the_tile_center() {
        let mesh = mesh_with_uvs(&[
            Vec2::new(0.0, 0.0),
            Vec2::new(0.5, 0.5),
            Vec2::new(1.0, 1.0),
        ]);
        let mut e = UvEditor::default();
        // No selection: the middle of the tile.
        assert_eq!(e.pivot(&mesh), Vec2::splat(0.5));
        e.select_only([0, 2]);
        assert_eq!(e.pivot(&mesh), Vec2::splat(0.5));
        e.select_only([0]);
        assert_eq!(e.pivot(&mesh), Vec2::ZERO);
        // Out-of-range indices fall back to the tile center.
        e.select_only([99]);
        assert_eq!(e.pivot(&mesh), Vec2::splat(0.5));
    }

    #[test]
    fn select_helpers_replace_and_extend() {
        let mut e = UvEditor::default();
        e.select_only([2, 1]);
        // Stored sorted, so the engine gets a deterministic index slice.
        assert_eq!(e.indices(), vec![1, 2]);
        e.select_more([5]);
        assert_eq!(e.indices(), vec![1, 2, 5]);
        // Re-selecting replaces rather than merging.
        e.select_only([7]);
        assert_eq!(e.indices(), vec![7]);
    }

    /// Paint one canvas frame and return the shapes it produced.
    fn painted(
        mesh: &Mesh,
        editor: &UvEditor,
        background: Option<egui::TextureId>,
    ) -> Vec<egui::Shape> {
        let ctx = egui::Context::default();
        let output = ctx.run(egui::RawInput::default(), |ctx| {
            let painter =
                ctx.layer_painter(egui::LayerId::new(egui::Order::Middle, egui::Id::new("uv")));
            draw_canvas(&painter, canvas(), mesh, editor, None, background);
        });
        output.shapes.into_iter().map(|clipped| clipped.shape).collect()
    }

    #[test]
    fn a_texture_background_paints_the_tile_behind_the_uvs() {
        let mesh = mesh_with_uvs(&[Vec2::new(0.1, 0.2), Vec2::new(0.9, 0.8)]);
        let editor = UvEditor::default();
        let id = egui::TextureId::Managed(2);
        let image = |shapes: &[egui::Shape]| {
            shapes.iter().find_map(|s| match s {
                egui::Shape::Mesh(m) if m.texture_id == id => Some(m.clone()),
                _ => None,
            })
        };

        // Without a background the tile is a plain fill, not an image.
        assert!(image(&painted(&mesh, &editor, None)).is_none());

        // With one, a textured quad covers the tile.
        let shapes = painted(&mesh, &editor, Some(id));
        let quad = image(&shapes).expect("the background texture is painted");
        assert_eq!(quad.vertices.len(), 4);
        let tile = tile_rect(canvas());
        assert!(quad.vertices.iter().all(|v| tile.contains(v.pos)));
        let xs: Vec<f32> = quad.vertices.iter().map(|v| v.pos.x).collect();
        let ys: Vec<f32> = quad.vertices.iter().map(|v| v.pos.y).collect();
        assert!((xs.iter().cloned().fold(f32::INFINITY, f32::min) - tile.left()).abs() < 1e-3);
        assert!((xs.iter().cloned().fold(f32::NEG_INFINITY, f32::max) - tile.right()).abs() < 1e-3);
        assert!((ys.iter().cloned().fold(f32::INFINITY, f32::min) - tile.top()).abs() < 1e-3);
        assert!((ys.iter().cloned().fold(f32::NEG_INFINITY, f32::max) - tile.bottom()).abs() < 1e-3);

        // The texel rows follow the UV space: row 0 (v = 0) is the tile's
        // *bottom*, because the editor draws v upwards.
        let top = quad
            .vertices
            .iter()
            .min_by(|a, b| a.pos.y.total_cmp(&b.pos.y))
            .expect("four vertices");
        let bottom = quad
            .vertices
            .iter()
            .max_by(|a, b| a.pos.y.total_cmp(&b.pos.y))
            .expect("four vertices");
        assert_eq!(top.uv.y, 1.0, "the texture's last row is the UV space's top");
        assert_eq!(bottom.uv.y, 0.0, "row 0 is v = 0, at the bottom");
    }

    #[test]
    fn snap_uv_rounds_to_the_grid_and_passes_bad_input_through() {
        assert_eq!(snap_uv(Vec2::ZERO, SNAP_STEP), Vec2::ZERO);
        let rounded = snap_uv(Vec2::new(0.07, -0.03), SNAP_STEP);
        assert!(
            (rounded - Vec2::new(0.05, -0.05)).length() < 1e-6,
            "{rounded:?}"
        );
        let rounded = snap_uv(Vec2::new(0.073, 0.024), SNAP_STEP);
        assert!(
            (rounded - Vec2::new(0.05, 0.0)).length() < 1e-6,
            "{rounded:?}"
        );

        // A non-positive step and non-finite deltas pass through unchanged.
        assert_eq!(
            snap_uv(Vec2::new(0.07, 0.0), 0.0),
            Vec2::new(0.07, 0.0)
        );
        assert!(snap_uv(Vec2::new(f32::NAN, 0.0), SNAP_STEP).x.is_nan());
    }
}
