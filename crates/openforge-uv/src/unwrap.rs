//! Unwrapping support: seam detection by dihedral angle and a simple
//! grid-atlas UV repack for connected UV islands.
//!
//! The [`UvOps`](crate::UvOps) methods that call into this module live in the
//! crate root's single impl block.

use std::collections::HashMap;

use glam::{Vec2, Vec3};
use openforge_core::mesh::Mesh;

use crate::{face_normal_opt, uv_at};

/// Minimal disjoint-set (union-find) forest over position indices.
pub(crate) struct UnionFind {
    parent: Vec<u32>,
    size: Vec<u32>,
}

impl UnionFind {
    pub(crate) fn new(n: usize) -> Self {
        Self {
            parent: (0..n as u32).collect(),
            size: vec![1; n],
        }
    }

    /// Find the representative of `x`, compressing the path.
    /// Returns `u32::MAX` when `x` is out of range.
    pub(crate) fn find(&mut self, x: u32) -> u32 {
        let mut i = x as usize;
        if i >= self.parent.len() {
            return u32::MAX;
        }
        let mut root = i;
        while self.parent[root] != root as u32 {
            root = self.parent[root] as usize;
        }
        while self.parent[i] != i as u32 {
            let next = self.parent[i] as usize;
            self.parent[i] = root as u32;
            i = next;
        }
        root as u32
    }

    /// Merge the sets containing `a` and `b` (union by size).
    pub(crate) fn union(&mut self, a: u32, b: u32) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra == rb || ra == u32::MAX || rb == u32::MAX {
            return;
        }
        let (ra, rb) = if self.size[ra as usize] >= self.size[rb as usize] {
            (ra, rb)
        } else {
            (rb, ra)
        };
        self.parent[rb as usize] = ra;
        self.size[ra as usize] += self.size[rb as usize];
    }
}

/// Per-island precomputed placement inside one atlas cell.
#[derive(Clone, Copy)]
pub(crate) struct Pack {
    /// Center of the island's current UV bounds.
    pub(crate) center: Vec2,
    /// Uniform scale fitting the island into the cell (aspect preserved).
    pub(crate) scale: f32,
    /// Center of the destination atlas cell.
    pub(crate) cell_center: Vec2,
}

/// Detect seam edges: boundary edges, edges whose adjacent face-normal angle
/// exceeds `angle_deg`, and non-manifold edges. Sorted, de-duplicated pairs.
pub(crate) fn uv_seams_from_angle(mesh: &Mesh, angle_deg: f32) -> Vec<(u32, u32)> {
    if mesh.faces.is_empty() || mesh.positions.is_empty() || !angle_deg.is_finite() {
        return Vec::new();
    }
    let threshold = angle_deg.max(0.0);
    // Degenerate faces (zero area, too few corners, invalid indices) are
    // skipped entirely: they contribute no trustworthy edge or normal.
    let normals: Vec<Option<Vec3>> = (0..mesh.faces.len())
        .map(|fi| face_normal_opt(mesh, fi))
        .collect();

    // Collect undirected position-index edges with their user faces.
    let mut edges: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
    for (fi, face) in mesh.faces.iter().enumerate() {
        if normals[fi].is_none() {
            continue;
        }
        let cs = &face.corners;
        if cs.len() < 2 {
            continue;
        }
        for i in 0..cs.len() {
            let a = cs[i].position;
            let b = cs[(i + 1) % cs.len()].position;
            if a == b
                || a as usize >= mesh.positions.len()
                || b as usize >= mesh.positions.len()
            {
                continue;
            }
            let key = if a < b { (a, b) } else { (b, a) };
            let users = edges.entry(key).or_default();
            if users.last() != Some(&fi) {
                users.push(fi);
            }
        }
    }

    let mut seams: Vec<(u32, u32)> = Vec::new();
    for (key, users) in edges {
        let seam = match users.len() {
            // Used by a single face: a boundary edge is always a seam.
            1 => true,
            // Two faces: a seam when the dihedral crease exceeds the
            // threshold (0 = coplanar, 90 = cube corner).
            2 => match (normals[users[0]], normals[users[1]]) {
                (Some(n1), Some(n2)) => {
                    let d = n1.dot(n2).clamp(-1.0, 1.0);
                    d.acos().to_degrees() > threshold
                }
                _ => false,
            },
            // Non-manifold edge: force a cut to stay on the safe side.
            _ => true,
        };
        if seam {
            seams.push(key);
        }
    }
    seams.sort_unstable();
    seams
}

/// Repack UVs into a `grid x grid` atlas: islands (faces grouped by shared
/// position edges via union-find) are uniformly scaled to fit their own cell
/// and centered in it; islands beyond `grid * grid` wrap to earlier cells.
pub(crate) fn uv_repack(mesh: &mut Mesh, grid: u32) {
    if grid == 0 || mesh.faces.is_empty() {
        return;
    }
    let n_pos = mesh.positions.len();

    // 1. Islands: faces connected through shared position edges.
    let mut uf = UnionFind::new(n_pos);
    for face in &mesh.faces {
        let cs = &face.corners;
        for i in 0..cs.len() {
            let a = cs[i].position;
            let b = cs[(i + 1) % cs.len()].position;
            if a != b && (a as usize) < n_pos && (b as usize) < n_pos {
                uf.union(a, b);
            }
        }
    }

    // 2. One component id per island, assigned in face order (deterministic).
    let mut root_to_comp: HashMap<u32, usize> = HashMap::new();
    let mut face_comp: Vec<usize> = Vec::with_capacity(mesh.faces.len());
    for face in &mesh.faces {
        let root = face
            .corners
            .iter()
            .map(|c| c.position)
            .find(|&p| (p as usize) < n_pos)
            .map(|p| uf.find(p))
            .unwrap_or(u32::MAX);
        let comp = match root_to_comp.get(&root) {
            Some(&c) => c,
            None => {
                let c = root_to_comp.len();
                root_to_comp.insert(root, c);
                c
            }
        };
        face_comp.push(comp);
    }
    let n_comp = root_to_comp.len();

    // 3. UV bounds per island (defensive reads, non-finite treated as zero).
    let mut c_min = vec![Vec2::ZERO; n_comp];
    let mut c_max = vec![Vec2::ZERO; n_comp];
    let mut c_seen = vec![false; n_comp];
    for (fi, face) in mesh.faces.iter().enumerate() {
        let comp = face_comp[fi];
        for c in &face.corners {
            let uv = uv_at(&mesh.uvs, c.uv);
            if !c_seen[comp] {
                c_min[comp] = uv;
                c_max[comp] = uv;
                c_seen[comp] = true;
            } else {
                c_min[comp] = c_min[comp].min(uv);
                c_max[comp] = c_max[comp].max(uv);
            }
        }
    }

    // 4. Placement: each island uniformly scaled (aspect preserved) and
    // centered in its cell, with a 2% cell padding against bleeding.
    let g = grid as f32;
    let cell = 1.0 / g;
    let pad = cell * 0.02;
    let inner = cell - 2.0 * pad;
    let total_cells = (grid as u64).saturating_mul(grid as u64).max(1);
    let mut packs: Vec<Pack> = Vec::with_capacity(n_comp);
    for comp in 0..n_comp {
        let span = c_max[comp] - c_min[comp];
        let center = (c_min[comp] + c_max[comp]) * 0.5;
        let sx = if span.x > 0.0 && span.x.is_finite() {
            inner / span.x
        } else {
            f32::INFINITY
        };
        let sy = if span.y > 0.0 && span.y.is_finite() {
            inner / span.y
        } else {
            f32::INFINITY
        };
        let scale = sx.min(sy);
        let scale = if scale.is_finite() && scale > 0.0 {
            scale
        } else {
            1.0
        };
        // Islands beyond the grid wrap around to earlier cells.
        let cell_i = (comp as u64 % total_cells) as u32;
        let (col, row) = (cell_i % grid, cell_i / grid);
        let origin = Vec2::new(col as f32, row as f32) * cell;
        packs.push(Pack {
            center,
            scale,
            cell_center: origin + Vec2::splat(cell * 0.5),
        });
    }

    // 5. Write fresh per-corner UVs; every corner gets a valid index. The
    // shared slice borrow of `mesh.uvs` coexists with the `mesh.faces`
    // mutable borrow because they are disjoint fields.
    let uvs = &mesh.uvs;
    let mut new_uvs: Vec<Vec2> =
        Vec::with_capacity(mesh.faces.iter().map(|f| f.corners.len()).sum());
    for (fi, face) in mesh.faces.iter_mut().enumerate() {
        let pack = packs[face_comp[fi]];
        for c in face.corners.iter_mut() {
            let old = uv_at(uvs, c.uv);
            let packed = (pack.cell_center + (old - pack.center) * pack.scale)
                .clamp(Vec2::ZERO, Vec2::ONE);
            c.uv = new_uvs.len() as u32;
            new_uvs.push(packed);
        }
    }
    mesh.uvs = new_uvs;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::{assert_uv_indices_valid, cube, open_cylinder, unit_quad};
    use crate::UvOps;

    #[test]
    fn seams_watertight_cube_empty_above_crease_angle() {
        let m = cube();
        // Closed mesh: no boundary edges, and its creases are exactly 90°, so
        // a threshold above 90° leaves nothing to cut.
        assert!(m.uv_seams_from_angle(90.5).is_empty());
        assert!(m.uv_seams_from_angle(180.0).is_empty());
    }

    #[test]
    fn seams_cube_low_threshold_finds_all_creases() {
        let m = cube();
        let seams = m.uv_seams_from_angle(45.0);
        // A cube has exactly 12 edges; every one is a 90° crease.
        assert_eq!(seams.len(), 12);
        assert!(seams.windows(2).all(|w| w[0] < w[1]), "must be sorted");
        for (a, b) in &seams {
            assert!((*a as usize) < m.positions.len());
            assert!((*b as usize) < m.positions.len());
        }
    }

    #[test]
    fn seams_boundary_edges_returned_at_any_threshold() {
        // An open quad has 4 boundary edges and no interior edges.
        let m = unit_quad();
        let low = m.uv_seams_from_angle(1.0);
        let high = m.uv_seams_from_angle(179.0);
        assert_eq!(low, vec![(0, 1), (0, 3), (1, 2), (2, 3)]);
        assert_eq!(low, high);
    }

    #[test]
    fn seams_open_cylinder_reports_rims() {
        let m = open_cylinder(8);
        let seams = m.uv_seams_from_angle(180.0);
        // Only the two boundary rims: 8 bottom + 8 top position pairs.
        assert_eq!(seams.len(), 16);
        // Each rim vertex pair connects y=0 to y=0 or y=1 to y=1.
        for (a, b) in &seams {
            let ya = m.positions[*a as usize].y;
            let yb = m.positions[*b as usize].y;
            assert_eq!(ya, yb);
        }
    }

    #[test]
    fn seams_degenerate_and_invalid_input_return_empty() {
        // Empty mesh.
        let empty = Mesh::new();
        assert!(empty.uv_seams_from_angle(30.0).is_empty());
        // Non-finite threshold.
        let q = unit_quad();
        assert!(q.uv_seams_from_angle(f32::NAN).is_empty());
        assert!(q.uv_seams_from_angle(f32::INFINITY).is_empty());
        // A zero-area face contributes no edges at all.
        let mut degenerate = Mesh::new();
        degenerate.add_vertex(Vec3::new(0.0, 0.0, 0.0));
        degenerate.add_vertex(Vec3::new(1.0, 0.0, 0.0));
        degenerate.add_vertex(Vec3::new(2.0, 0.0, 0.0));
        degenerate.add_face(&[0, 1, 2]);
        assert!(degenerate.uv_seams_from_angle(10.0).is_empty());
        // Out-of-range corner positions are skipped without panicking.
        let mut broken = unit_quad();
        broken.faces[0].corners[0].position = 999;
        assert!(broken.uv_seams_from_angle(10.0).is_empty());
    }

    #[test]
    fn repack_two_islands_land_in_distinct_cells() {
        // Two disjoint triangles: separate islands by shared positions.
        let mut m = Mesh::new();
        m.add_vertex(Vec3::new(0.0, 0.0, 0.0));
        m.add_vertex(Vec3::new(1.0, 0.0, 0.0));
        m.add_vertex(Vec3::new(0.0, 1.0, 0.0));
        m.add_vertex(Vec3::new(5.0, 5.0, 0.0));
        m.add_vertex(Vec3::new(6.0, 5.0, 0.0));
        m.add_vertex(Vec3::new(5.0, 6.0, 0.0));
        m.add_face(&[0, 1, 2]);
        m.add_face(&[3, 4, 5]);
        m.uv_project_planar(Vec3::Z);
        m.uv_repack(2);
        assert_uv_indices_valid(&m);
        assert_eq!(m.uvs.len(), 6); // one UV per corner
        // Which atlas half (low/high) each face's UVs occupy.
        let cell_of = |face: usize, m: &Mesh| -> (u32, u32) {
            let (mut hi_x, mut hi_y) = (false, false);
            for c in &m.faces[face].corners {
                let uv = m.uvs[c.uv as usize];
                assert!((0.0..=1.0).contains(&uv.x) && (0.0..=1.0).contains(&uv.y));
                hi_x |= uv.x >= 0.5;
                hi_y |= uv.y >= 0.5;
            }
            (hi_x as u32, hi_y as u32)
        };
        // Island 0 goes to cell (0,0), island 1 to cell (1,0).
        assert_eq!(cell_of(0, &m), (0, 0));
        assert_eq!(cell_of(1, &m), (1, 0));
    }

    #[test]
    fn repack_keeps_uvs_inside_the_atlas() {
        let mut m = cube();
        m.uv_project_box();
        m.uv_repack(2);
        assert_uv_indices_valid(&m);
        assert_eq!(m.uvs.len(), 24); // 6 faces x 4 corners
        for uv in &m.uvs {
            assert!((0.0..=1.0).contains(&uv.x), "u out of range: {uv:?}");
            assert!((0.0..=1.0).contains(&uv.y), "v out of range: {uv:?}");
        }
        // The cube is one connected island, so everything lands in one cell.
        let c0 = m.uvs[0];
        for uv in &m.uvs {
            assert_eq!((uv.x >= 0.5) as u32, (c0.x >= 0.5) as u32);
            assert_eq!((uv.y >= 0.5) as u32, (c0.y >= 0.5) as u32);
        }
    }

    #[test]
    fn repack_grid_one_fills_the_unit_square() {
        let mut m = unit_quad();
        m.uv_project_planar(Vec3::Z);
        m.uv_repack(1);
        assert_uv_indices_valid(&m);
        let b = m.uv_bounds().unwrap();
        assert!(b.min.x >= 0.0 && b.min.y >= 0.0);
        assert!(b.max.x <= 1.0 && b.max.y <= 1.0);
        // A grid of one cell still uses a meaningful part of the space.
        assert!(b.size().x > 0.5 && b.size().y > 0.5);
    }

    #[test]
    fn repack_degenerate_inputs_are_noop_or_safe() {
        // Empty mesh: no-op.
        let mut empty = Mesh::new();
        empty.uv_repack(4);
        assert!(empty.uvs.is_empty());
        // grid == 0: no-op (does not divide by zero).
        let mut m = unit_quad();
        m.uv_project_planar(Vec3::Z);
        let before = m.uvs.clone();
        m.uv_repack(0);
        assert_eq!(m.uvs, before);
        // A mesh whose faces reference nothing: still produces valid indices.
        let mut broken = unit_quad();
        broken.faces[0].corners[0].position = 999;
        broken.uv_repack(2);
        assert_uv_indices_valid(&broken);
    }

    #[test]
    fn repack_preserves_island_shape_by_uniform_scale() {
        // A wide, flat island must keep its aspect ratio inside the cell.
        let mut m = Mesh::new();
        m.add_vertex(Vec3::new(0.0, 0.0, 0.0));
        m.add_vertex(Vec3::new(10.0, 0.0, 0.0));
        m.add_vertex(Vec3::new(10.0, 1.0, 0.0));
        m.add_vertex(Vec3::new(0.0, 1.0, 0.0));
        m.add_face(&[0, 1, 2, 3]);
        m.uv_project_planar(Vec3::Z);
        assert_eq!(m.uv_bounds().unwrap().size(), Vec2::new(1.0, 1.0));
        // Squash the island to a 4:1 rectangle before repacking.
        m.uv_scale_translate(Vec2::new(1.0, 0.25), Vec2::ZERO);
        m.uv_repack(2);
        let b = m.uv_bounds().unwrap();
        let size = b.size();
        let ratio = size.x / size.y;
        assert!((4.0 - ratio).abs() < 0.05, "expected ~4:1, got {ratio}");
    }
}
