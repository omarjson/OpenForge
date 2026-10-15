//! UV-island detection and per-island atlas packing.
//!
//! Faces are grouped into *UV islands* by UV continuity: two faces that share
//! a mesh edge belong to the same island when the UVs they assign at both
//! shared endpoints coincide within a tolerance. The
//! [`UvOps`](crate::UvOps) methods that call into this module live in the
//! crate root's single impl block.

use std::collections::HashMap;

use glam::Vec2;
use openforge_core::mesh::Mesh;

use crate::unwrap::{Pack, UnionFind};
use crate::uv_at;

/// Tolerance used by [`uv_pack_islands`] to partition faces into islands:
/// wide enough to absorb the float noise a projection leaves behind, far
/// below any meaningful UV distance.
pub(crate) const ISLAND_TOLERANCE: f32 = 1e-5;

/// The UV face `fi` assigns at position `p`: the value of the first corner
/// visiting `p`, or `None` when no corner does or its uv index does not
/// resolve.
fn uv_at_position(mesh: &Mesh, fi: usize, p: u32) -> Option<Vec2> {
    let face = mesh.faces.get(fi)?;
    let uv = face.corners.iter().find(|c| c.position == p)?.uv as usize;
    mesh.uvs.get(uv).copied()
}

/// Whether faces `fx` and `fy` assign matching UVs (within `tolerance`) at
/// both endpoints of the edge `(a, b)` they share. A missing or out-of-range
/// UV on either side means the edge cannot weld.
fn uvs_match_across_edge(
    mesh: &Mesh,
    fx: usize,
    fy: usize,
    a: u32,
    b: u32,
    tolerance: f32,
) -> bool {
    let close = |p: u32| match (uv_at_position(mesh, fx, p), uv_at_position(mesh, fy, p)) {
        (Some(ux), Some(uy)) => (ux - uy).length() <= tolerance,
        _ => false,
    };
    close(a) && close(b)
}

/// Partition face indices into UV islands: faces connected through a shared
/// mesh edge whose UVs match at both shared endpoints, welded transitively.
///
/// Every face index appears in exactly one island; islands are ordered by
/// their smallest face index and hold ascending face indices. Unconnectable
/// faces are singleton islands, an empty mesh yields an empty vector, and
/// faces with fewer than three corners never connect.
pub(crate) fn uv_face_islands(mesh: &Mesh, tolerance: f32) -> Vec<Vec<u32>> {
    let n_faces = mesh.faces.len();
    if n_faces == 0 {
        return Vec::new();
    }
    // `max` ignores NaN and keeps the lower bound at zero.
    let tolerance = tolerance.max(0.0);
    let n_pos = mesh.positions.len();

    // Undirected position-index edges -> the faces using them, one entry per
    // face even when a face visits the same edge twice.
    let mut edges: HashMap<(u32, u32), Vec<u32>> = HashMap::new();
    for (fi, face) in mesh.faces.iter().enumerate() {
        let cs = &face.corners;
        if cs.len() < 3 {
            continue;
        }
        for i in 0..cs.len() {
            let a = cs[i].position;
            let b = cs[(i + 1) % cs.len()].position;
            if a == b || a as usize >= n_pos || b as usize >= n_pos {
                continue;
            }
            let key = if a < b { (a, b) } else { (b, a) };
            let users = edges.entry(key).or_default();
            if users.last() != Some(&(fi as u32)) {
                users.push(fi as u32);
            }
        }
    }

    // Weld every pair of faces meeting on an edge whose UVs agree.
    let mut uf = UnionFind::new(n_faces);
    for (&(a, b), users) in &edges {
        for i in 0..users.len() {
            for j in i + 1..users.len() {
                let (fx, fy) = (users[i] as usize, users[j] as usize);
                if fx != fy && uvs_match_across_edge(mesh, fx, fy, a, b, tolerance) {
                    uf.union(fx as u32, fy as u32);
                }
            }
        }
    }

    // Collect in face order, which orders islands by their smallest face
    // index and each island's faces ascending.
    let mut root_to_island: HashMap<u32, usize> = HashMap::new();
    let mut islands: Vec<Vec<u32>> = Vec::new();
    for fi in 0..n_faces {
        let root = uf.find(fi as u32);
        if let Some(&island) = root_to_island.get(&root) {
            islands[island].push(fi as u32);
        } else {
            root_to_island.insert(root, islands.len());
            islands.push(vec![fi as u32]);
        }
    }
    islands
}

/// The island of [`uv_face_islands`] that contains `face`, or `None` when
/// `face` is out of range.
pub(crate) fn uv_island_of_face(mesh: &Mesh, tolerance: f32, face: u32) -> Option<Vec<u32>> {
    if face as usize >= mesh.faces.len() {
        return None;
    }
    uv_face_islands(mesh, tolerance)
        .into_iter()
        .find(|island| island.binary_search(&face).is_ok())
}

/// Sorted, de-duplicated indices of the UV entries used by the corners of
/// `faces`. Out-of-range face indices are skipped and only indices that
/// resolve into `mesh.uvs` are reported.
pub(crate) fn uv_island_uv_indices(mesh: &Mesh, faces: &[u32]) -> Vec<u32> {
    let mut indices: Vec<u32> = Vec::new();
    for &fi in faces {
        let Some(face) = mesh.faces.get(fi as usize) else {
            continue;
        };
        for c in &face.corners {
            if (c.uv as usize) < mesh.uvs.len() {
                indices.push(c.uv);
            }
        }
    }
    indices.sort_unstable();
    indices.dedup();
    indices
}

/// One island's measured placement: the UV entries it owns plus the [`Pack`]
/// that moves them into their atlas cell.
struct IslandPack {
    entries: Vec<u32>,
    pack: Pack,
}

/// Pack every UV island into its own cell of a `grid x grid` atlas,
/// preserving each island's internal layout with a uniform scale.
///
/// Islands come from the UV-continuity partition at [`ISLAND_TOLERANCE`].
/// Only `mesh.uvs` values move: `positions`, `normals` and `faces` are
/// untouched, and no uv index is created or removed, so corners keep their uv
/// indices. A uv entry used by several islands belongs to the first island
/// (in island order) that uses it, and only that owner moves it; the bounds
/// fitted into a cell cover exactly the entries an island owns.
///
/// Returns the number of islands; `grid == 0` and meshes without faces change
/// nothing and return `0`.
pub(crate) fn uv_pack_islands(mesh: &mut Mesh, grid: u32, margin: f32) -> usize {
    if grid == 0 || mesh.faces.is_empty() {
        return 0;
    }
    let islands = uv_face_islands(mesh, ISLAND_TOLERANCE);
    if islands.is_empty() {
        return 0;
    }

    // Margin is a UV distance; NaN reads as 0 and the exclusive upper bound
    // keeps the padded cell strictly positive.
    let margin = if margin > 0.0 { margin.min(0.5 - f32::EPSILON) } else { 0.0 };
    let g = grid as f32;
    let cell = 1.0 / g;
    let inner = cell - 2.0 * margin; // > 0

    // Phase 1 (measure): owned uv entries, bounds and placement per island.
    let mut claimed = vec![false; mesh.uvs.len()];
    let total_cells = (grid as u64).saturating_mul(grid as u64);
    let mut packs: Vec<IslandPack> = Vec::with_capacity(islands.len());
    for (island, faces) in islands.iter().enumerate() {
        let mut entries: Vec<u32> = Vec::new();
        let (mut min, mut max) = (Vec2::ZERO, Vec2::ZERO);
        let mut any = false;
        for &fi in faces {
            for c in &mesh.faces[fi as usize].corners {
                let u = c.uv as usize;
                if u >= mesh.uvs.len() || claimed[u] {
                    continue;
                }
                claimed[u] = true;
                entries.push(c.uv);
                let uv = mesh.uvs[u];
                if !uv.is_finite() {
                    continue;
                }
                if any {
                    min = min.min(uv);
                    max = max.max(uv);
                } else {
                    min = uv;
                    max = uv;
                    any = true;
                }
            }
        }
        // Islands beyond the grid wrap around to earlier cells.
        let cell_i = (island as u64 % total_cells) as u32;
        let (col, row) = (cell_i % grid, cell_i / grid);
        let cell_center = Vec2::new((col as f32 + 0.5) / g, (row as f32 + 0.5) / g);
        let (center, scale) = if any {
            let center = (min + max) * 0.5;
            let span = max - min;
            let sx = if span.x > 0.0 { inner / span.x } else { f32::INFINITY };
            let sy = if span.y > 0.0 { inner / span.y } else { f32::INFINITY };
            let fit = sx.min(sy);
            (center, if fit.is_finite() && fit > 0.0 { fit } else { 1.0 })
        } else {
            // No finite entry: the island collapses to the cell center.
            (Vec2::ZERO, 1.0)
        };
        packs.push(IslandPack { entries, pack: Pack { center, scale, cell_center } });
    }

    // Phase 2 (apply): move each island's owned entries into its cell, every
    // one strictly inside the margin-padded cell.
    let half = Vec2::splat(inner * 0.5);
    for island in &packs {
        let pack = island.pack;
        for &e in &island.entries {
            let old = uv_at(&mesh.uvs, e);
            let moved = pack.cell_center + (old - pack.center) * pack.scale;
            mesh.uvs[e as usize] = moved.clamp(pack.cell_center - half, pack.cell_center + half);
        }
    }
    islands.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::{assert_uv_indices_valid, cube, open_cylinder, unit_quad};
    use crate::UvOps;
    use glam::Vec3;
    use openforge_core::mesh::{Corner, Face};

    /// Two quads sharing the position edge `{1, 2}`, each corner on its own
    /// uv entry, with matching UV values along the shared edge.
    fn two_quads() -> Mesh {
        let mut m = Mesh::new();
        m.add_vertex(Vec3::new(0.0, 0.0, 0.0)); // 0
        m.add_vertex(Vec3::new(1.0, 0.0, 0.0)); // 1, shared
        m.add_vertex(Vec3::new(1.0, 1.0, 0.0)); // 2, shared
        m.add_vertex(Vec3::new(0.0, 1.0, 0.0)); // 3
        m.add_vertex(Vec3::new(2.0, 0.0, 0.0)); // 4
        m.add_vertex(Vec3::new(2.0, 1.0, 0.0)); // 5
        m.add_face(&[0, 1, 2, 3]);
        m.add_face(&[1, 4, 5, 2]);
        m.uvs = vec![
            Vec2::new(0.0, 0.0), // 0: pos 0
            Vec2::new(1.0, 0.0), // 1: pos 1
            Vec2::new(1.0, 1.0), // 2: pos 2
            Vec2::new(0.0, 1.0), // 3: pos 3
            Vec2::new(1.0, 0.0), // 4: pos 1 of face 1, matches entry 1
            Vec2::new(2.0, 0.0), // 5: pos 4
            Vec2::new(2.0, 1.0), // 6: pos 5
            Vec2::new(1.0, 1.0), // 7: pos 2 of face 1, matches entry 2
        ];
        for (i, c) in m.faces[0].corners.iter_mut().enumerate() {
            c.uv = i as u32;
        }
        for (i, c) in m.faces[1].corners.iter_mut().enumerate() {
            c.uv = 4 + i as u32;
        }
        m
    }

    /// Four faces in two welded pairs: faces 0 and 2 share the edge `{2, 3}`,
    /// faces 1 and 3 share the edge `{4, 5}`. Every corner reads the single
    /// zero UV entry, so welding is decided purely by shared edges.
    fn two_welded_pairs() -> Mesh {
        let mut m = Mesh::new();
        for p in [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (5.0, 5.0, 0.0),
            (6.0, 5.0, 0.0),
            (5.0, 6.0, 0.0),
            (1.0, 2.0, 0.0),
            (0.0, 2.0, 0.0),
            (6.0, 6.0, 0.0),
        ] {
            m.add_vertex(Vec3::new(p.0, p.1, p.2));
        }
        m.add_face(&[0, 1, 2, 3]); // 0
        m.add_face(&[4, 5, 6]); // 1
        m.add_face(&[3, 2, 7, 8]); // 2, shares edge {2, 3} with face 0
        m.add_face(&[5, 4, 9]); // 3, shares edge {4, 5} with face 1
        m
    }

    /// A quad on the left and a quad on the right with no shared positions,
    /// so they form two islands.
    fn two_disjoint_quads() -> Mesh {
        let mut m = Mesh::new();
        m.add_vertex(Vec3::new(0.0, 0.0, 0.0));
        m.add_vertex(Vec3::new(1.0, 0.0, 0.0));
        m.add_vertex(Vec3::new(1.0, 1.0, 0.0));
        m.add_vertex(Vec3::new(0.0, 1.0, 0.0));
        m.add_vertex(Vec3::new(5.0, 5.0, 0.0));
        m.add_vertex(Vec3::new(6.0, 5.0, 0.0));
        m.add_vertex(Vec3::new(6.0, 6.0, 0.0));
        m.add_vertex(Vec3::new(5.0, 6.0, 0.0));
        m.add_face(&[0, 1, 2, 3]);
        m.add_face(&[4, 5, 6, 7]);
        m
    }

    #[test]
    fn two_faces_sharing_an_edge_with_matching_uvs_form_one_island() {
        let m = two_quads();
        assert_eq!(uv_face_islands(&m, 0.0), vec![vec![0, 1]]);
    }

    #[test]
    fn shifting_shared_edge_uvs_splits_faces_into_two_islands() {
        let mut m = two_quads();
        m.uvs[4] = Vec2::new(0.8, 0.0); // no longer matches face 0's (1, 0)
        assert_eq!(uv_face_islands(&m, 1e-4), vec![vec![0], vec![1]]);
    }

    #[test]
    fn near_identical_uvs_stay_welded_within_tolerance() {
        let mut m = two_quads();
        m.uvs[4] = Vec2::new(1.0 + 1e-6, 0.0);
        assert_eq!(uv_face_islands(&m, 1e-4), vec![vec![0, 1]]);
        // The 1e-6 gap still separates the faces at zero tolerance.
        assert_eq!(uv_face_islands(&m, 0.0), vec![vec![0], vec![1]]);
    }

    #[test]
    fn uvs_apart_beyond_the_tolerance_split_islands() {
        let mut m = two_quads();
        m.uvs[4] = Vec2::new(1.01, 0.0);
        assert_eq!(uv_face_islands(&m, 1e-4), vec![vec![0], vec![1]]);
        // A tolerance above the 0.01 gap welds them again.
        assert_eq!(uv_face_islands(&m, 0.02), vec![vec![0, 1]]);
    }

    #[test]
    fn box_projected_cube_welds_faces_that_keep_matching_uvs() {
        let mut m = cube();
        m.uv_project_box();
        let islands = m.uv_face_islands(1e-4);
        // `uv_project_box` normalizes the X-, Y- and Z-facing groups
        // independently, so the three faces meeting around (-1,-1,-1) keep
        // matching UVs along their shared edges and weld (likewise around
        // (1,1,1)): two islands, not six.
        assert_eq!(islands, vec![vec![0, 2, 4], vec![1, 3, 5]]);
        let mut covered: Vec<u32> = islands.iter().flatten().copied().collect();
        covered.sort_unstable();
        assert_eq!(covered, (0..6).collect::<Vec<u32>>());
        // A repeated call partitions identically.
        assert_eq!(m.uv_face_islands(1e-4), islands);
    }

    #[test]
    fn open_cylinder_faces_partition_into_one_ordered_island() {
        let mut m = open_cylinder(8);
        m.uv_project_cylindrical(Vec3::Y);
        // The unrolled wall is continuous with one seam at the wrap; the
        // faces chain around it, so everything is a single island.
        let islands = m.uv_face_islands(1e-4);
        assert_eq!(islands, vec![(0..8).collect::<Vec<u32>>()]);
        let total: usize = islands.iter().map(|i| i.len()).sum();
        assert_eq!(total, m.face_count(), "every face appears exactly once");
    }

    #[test]
    fn hand_built_mesh_islands_are_ordered_by_smallest_face() {
        let m = two_welded_pairs();
        let islands = uv_face_islands(&m, 1e-4);
        // Faces 0 and 2 weld (smallest index 0), then faces 1 and 3 (1).
        assert_eq!(islands, vec![vec![0, 2], vec![1, 3]]);
        let mut covered: Vec<u32> = islands.iter().flatten().copied().collect();
        covered.sort_unstable();
        assert_eq!(covered, vec![0, 1, 2, 3], "every face appears exactly once");
        // Empty mesh: no islands.
        assert_eq!(uv_face_islands(&Mesh::new(), 1e-4), Vec::<Vec<u32>>::new());
    }

    #[test]
    fn island_of_face_finds_the_island_and_none_past_the_end() {
        let m = two_welded_pairs();
        assert_eq!(m.uv_island_of_face(1e-4, 0), Some(vec![0, 2]));
        assert_eq!(m.uv_island_of_face(1e-4, 3), Some(vec![1, 3]));
        assert_eq!(m.uv_island_of_face(1e-4, 4), None); // past the end
        assert_eq!(Mesh::new().uv_island_of_face(1e-4, 0), None);
    }

    #[test]
    fn island_uv_indices_are_sorted_deduplicated_and_in_range() {
        let mut m = unit_quad();
        m.uvs = vec![Vec2::ZERO; 4];
        for (i, c) in m.faces[0].corners.iter_mut().enumerate() {
            c.uv = [3u32, 1, 1, 999][i];
        }
        assert_eq!(m.uv_island_uv_indices(&[0]), vec![1, 3]);
        // Out-of-range face indices are skipped, never panicking.
        assert_eq!(m.uv_island_uv_indices(&[0, 99]), vec![1, 3]);
        assert_eq!(m.uv_island_uv_indices(&[99]), Vec::<u32>::new());
        assert_eq!(m.uv_island_uv_indices(&[]), Vec::<u32>::new());
    }

    #[test]
    fn packing_places_each_island_inside_its_own_cell() {
        let mut m = two_disjoint_quads();
        m.uv_project_planar(Vec3::Z);
        let margin = 0.05;
        assert_eq!(m.uv_pack_islands(2, margin), 2);
        assert_uv_indices_valid(&m);
        for uv in &m.uvs {
            assert!((margin..=1.0 - margin).contains(&uv.x), "u={}", uv.x);
            assert!((margin..=1.0 - margin).contains(&uv.y), "v={}", uv.y);
        }
        // Island 0 lands in cell (0, 0), island 1 in cell (1, 0), each with
        // the margin free on every side.
        let cell_of = |face: usize| -> (bool, bool) {
            let (mut hi_x, mut hi_y) = (false, false);
            for c in &m.faces[face].corners {
                let uv = m.uvs[c.uv as usize];
                hi_x |= uv.x >= 0.5;
                hi_y |= uv.y >= 0.5;
            }
            (hi_x, hi_y)
        };
        assert_eq!(cell_of(0), (false, false));
        assert_eq!(cell_of(1), (true, false));
    }

    #[test]
    fn packing_preserves_aspect_and_leaves_geometry_alone() {
        // A wide, flat island (4:1 in UV space) must stay 4:1 after packing.
        let mut m = Mesh::new();
        m.add_vertex(Vec3::new(0.0, 0.0, 0.0));
        m.add_vertex(Vec3::new(10.0, 0.0, 0.0));
        m.add_vertex(Vec3::new(10.0, 1.0, 0.0));
        m.add_vertex(Vec3::new(0.0, 1.0, 0.0));
        m.add_face(&[0, 1, 2, 3]);
        m.uv_project_planar(Vec3::Z);
        m.uv_scale_translate(Vec2::new(1.0, 0.25), Vec2::ZERO);
        let positions = m.positions.clone();
        let faces = m.faces.clone();
        assert_eq!(m.uv_pack_islands(2, 0.02), 1);
        assert_eq!(m.positions, positions, "positions must not move");
        assert_eq!(m.faces, faces, "faces must not change");
        assert_eq!(m.uvs.len(), 4, "no uv index is created or removed");
        let size = m.uv_bounds().unwrap().size();
        assert!(size.x > size.y, "a wide island stays wider than tall");
        let ratio = size.x / size.y;
        assert!((4.0 - ratio).abs() < 0.05, "expected ~4:1, got {ratio}");
    }

    #[test]
    fn packing_empty_mesh_and_grid_zero_are_noops() {
        // Empty mesh: returns 0 and changes nothing.
        let mut empty = Mesh::new();
        assert_eq!(empty.uv_pack_islands(4, 0.1), 0);
        assert!(empty.uvs.is_empty());
        assert!(empty.faces.is_empty());
        // A mesh with vertices but no faces: still nothing to pack.
        let mut faceless = Mesh::new();
        faceless.add_vertex(Vec3::ZERO);
        assert_eq!(faceless.uv_pack_islands(2, 0.0), 0);
        // grid == 0: returns 0 and leaves the UVs alone (no divide by zero).
        let mut m = unit_quad();
        m.uv_project_planar(Vec3::Z);
        let before = m.uvs.clone();
        assert_eq!(m.uv_pack_islands(0, 0.1), 0);
        assert_eq!(m.uvs, before);
        assert_uv_indices_valid(&m);
    }

    #[test]
    fn packing_nan_margin_behaves_like_zero_margin() {
        let mut zero = cube();
        zero.uv_project_box();
        let mut nan = zero.clone();
        let mut negative = zero.clone();
        assert_eq!(zero.uv_pack_islands(2, 0.0), 2);
        assert_eq!(nan.uv_pack_islands(2, f32::NAN), 2);
        assert_eq!(negative.uv_pack_islands(2, -1.0), 2);
        assert_eq!(nan.uvs, zero.uvs);
        assert_eq!(negative.uvs, zero.uvs);
    }

    #[test]
    fn packing_keeps_a_shared_uv_entry_with_the_first_island() {
        // Two disjoint quads in different islands; face 1's first corner
        // reuses face 0's uv entry 0, so island 0 owns it.
        let mut m = two_disjoint_quads();
        m.uvs = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(0.0, 1.0),
            Vec2::new(0.1, 0.1),
            Vec2::new(0.9, 0.1),
            Vec2::new(0.9, 0.9),
        ];
        for (i, c) in m.faces[0].corners.iter_mut().enumerate() {
            c.uv = i as u32;
        }
        let shared = [0u32, 4, 5, 6];
        for (i, c) in m.faces[1].corners.iter_mut().enumerate() {
            c.uv = shared[i];
        }
        assert_eq!(m.uv_pack_islands(2, 0.05), 2);
        assert_eq!(m.uvs.len(), 7, "no uv index is created or removed");
        assert_uv_indices_valid(&m);
        // Entry 0 moved with island 0 (cell (0,0)); island 1's own entries
        // (4, 5, 6) moved with island 1 (cell (1,0)).
        for uv in &m.uvs[0..4] {
            assert!(uv.x < 0.5 && uv.y < 0.5, "shared entry left its cell: {uv:?}");
        }
        for uv in &m.uvs[4..7] {
            assert!(uv.x > 0.5 && uv.y < 0.5, "island 1 entry left its cell: {uv:?}");
        }
    }

    #[test]
    fn packing_cube_and_cylinder_keeps_uvs_valid_and_finite() {
        let mut c = cube();
        c.uv_project_box();
        assert_eq!(c.uv_pack_islands(3, 0.02), 2);
        assert_uv_indices_valid(&c);
        for uv in &c.uvs {
            assert!(uv.is_finite(), "non-finite uv after packing: {uv:?}");
            assert!((0.0..=1.0).contains(&uv.x) && (0.0..=1.0).contains(&uv.y));
        }
        let mut cyl = open_cylinder(8);
        cyl.uv_project_cylindrical(Vec3::Y);
        assert_eq!(cyl.uv_pack_islands(3, 0.02), 1);
        assert_uv_indices_valid(&cyl);
        for uv in &cyl.uvs {
            assert!(uv.is_finite(), "non-finite uv after packing: {uv:?}");
            assert!((0.0..=1.0).contains(&uv.x) && (0.0..=1.0).contains(&uv.y));
        }
    }

    #[test]
    fn faces_with_too_few_corners_or_dangling_indices_stay_isolated() {
        // Two 2-corner faces share the edge {0, 1} with identical UVs, but
        // faces with fewer than three corners never connect.
        let mut m = Mesh::new();
        m.add_vertex(Vec3::ZERO);
        m.add_vertex(Vec3::new(1.0, 0.0, 0.0));
        for _ in 0..2 {
            m.faces.push(Face {
                corners: vec![
                    Corner { position: 0, normal: 0, uv: 0 },
                    Corner { position: 1, normal: 0, uv: 0 },
                ],
                material: 0,
            });
        }
        m.uvs = vec![Vec2::ZERO];
        assert_eq!(uv_face_islands(&m, 1.0), vec![vec![0], vec![1]]);
        // A dangling position index isolates its face without panicking.
        m.faces[1].corners[0].position = 999;
        assert_eq!(uv_face_islands(&m, 1.0), vec![vec![0], vec![1]]);
        // A dangling uv index cannot weld either.
        m.faces[1].corners[0].position = 0;
        m.faces[0].corners[0].uv = 7;
        assert_eq!(uv_face_islands(&m, 1.0), vec![vec![0], vec![1]]);
        m.faces[0].corners[0].uv = 0;
        // Packing such a mesh stays safe and keeps uv indices valid.
        assert_eq!(uv_pack_islands(&mut m, 2, 0.05), 2);
        assert_uv_indices_valid(&m);
        for uv in &m.uvs {
            assert!(uv.is_finite(), "non-finite uv after packing: {uv:?}");
        }
    }
}
