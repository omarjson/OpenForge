//! Sub-object selection maths: the Select menu's operations.
//!
//! Each function is a pure map from one element set to another at a level. The
//! app owns *which* node it applies to; the mesh owns what "adjacent" means,
//! and adjacency follows the topology the rest of the app answers from: faces
//! are neighbours when they share a mesh edge (never a bare corner), edges
//! when they share a position, vertices when an edge joins them — and the edge
//! list is [`Mesh::edges`], the one enumeration every picker numbers by, so an
//! element the visible pass skips contributes no adjacency either.

use crate::mesh::{Face, Mesh};
use crate::selection::SelectLevel;
use std::collections::{BTreeSet, HashMap};

/// Every element index `level` can select in one mesh, ascending.
///
/// Empty at the object level: nodes belong to the scene, not to a mesh.
pub fn all_elements(mesh: &Mesh, level: SelectLevel) -> BTreeSet<u32> {
    match level {
        SelectLevel::Object => BTreeSet::new(),
        SelectLevel::Vertex => (0..mesh.positions.len() as u32).collect(),
        SelectLevel::Edge => (0..mesh.edges().len() as u32).collect(),
        SelectLevel::Face => mesh
            .faces
            .iter()
            .enumerate()
            .filter(|(_, face)| mesh.face_is_valid(face))
            .map(|(i, _)| i as u32)
            .collect(),
    }
}

/// `current` with every element of the level flipped: what was not selected
/// is, and what was is not.
pub fn invert_selection(mesh: &Mesh, level: SelectLevel, current: &BTreeSet<u32>) -> BTreeSet<u32> {
    all_elements(mesh, level)
        .difference(current)
        .copied()
        .collect()
}

/// `current` plus everything adjacent to it.
pub fn grow_selection(mesh: &Mesh, level: SelectLevel, current: &BTreeSet<u32>) -> BTreeSet<u32> {
    let adj = adjacency(mesh, level);
    let mut out = current.clone();
    for &element in current {
        if let Some(neighbours) = adj.get(element as usize) {
            out.extend(neighbours.iter().copied());
        }
    }
    out
}

/// Keep only the elements every one of whose neighbours is selected.
///
/// The mirror of [`grow_selection`]: what grow would have reached from outside
/// the set is exactly the boundary this cuts away, so shrinking a set that was
/// just grown gives back what it grew from.
pub fn shrink_selection(mesh: &Mesh, level: SelectLevel, current: &BTreeSet<u32>) -> BTreeSet<u32> {
    let adj = adjacency(mesh, level);
    let grown = grow_selection(mesh, level, current);
    let boundary: BTreeSet<u32> = grown.difference(current).copied().collect();
    if boundary.is_empty() {
        return current.clone();
    }
    current
        .iter()
        .filter(|&&element| {
            adj.get(element as usize)
                .map(|neighbours| neighbours.iter().all(|n| !boundary.contains(n)))
                .unwrap_or(true)
        })
        .copied()
        .collect()
}

/// The ring through `start`: the edges parallel to it, walked across the quads
/// that share it.
///
/// A step crosses a quad face to its opposite side; from there the walk
/// carries on through the face it did not just cross — so it runs around the
/// mesh rather than straight back — and it stops at a face that is not a quad
/// or an edge with no further side to cross (a boundary edge has one face).
pub fn edge_ring(mesh: &Mesh, start: u32) -> BTreeSet<u32> {
    let edges = mesh.edges();
    let id_of = edge_ids(&edges);
    let users = edge_users(mesh, &edges, &id_of);
    let mut out = BTreeSet::new();
    let mut frontier: Vec<(u32, Option<usize>)> = vec![(start, None)];
    while let Some((id, came_from)) = frontier.pop() {
        if !out.insert(id) {
            continue;
        }
        for &(fi, k) in users.get(id as usize).into_iter().flatten() {
            if Some(fi) == came_from {
                continue;
            }
            let face = &mesh.faces[fi];
            if face.corners.len() != 4 {
                continue;
            }
            if let Some(next) = face_edge_id(face, k + 2, &id_of) {
                frontier.push((next, Some(fi)));
            }
        }
    }
    out
}

/// The loop through `start`: the chain that carries on straight through
/// four-edge vertices.
///
/// At an endpoint of the current edge the walk continues only when exactly
/// four edges meet there, along the one that is neither the current edge nor
/// one of its two fan neighbours (the other edges of the faces the current
/// edge borders) — the straight continuation. Any other valence ends the walk,
/// so a grid's border stops a loop and a cube's three-edge corner does too.
pub fn edge_loop(mesh: &Mesh, start: u32) -> BTreeSet<u32> {
    let edges = mesh.edges();
    let id_of = edge_ids(&edges);
    let users = edge_users(mesh, &edges, &id_of);
    let mut by_vertex: HashMap<u32, Vec<u32>> = HashMap::new();
    for (i, [a, b]) in edges.iter().enumerate() {
        by_vertex.entry(*a).or_default().push(i as u32);
        by_vertex.entry(*b).or_default().push(i as u32);
    }
    let mut out = BTreeSet::new();
    let mut frontier = vec![start];
    while let Some(id) = frontier.pop() {
        if !out.insert(id) {
            continue;
        }
        let [a, b] = edges[id as usize];
        for v in [a, b] {
            let Some(incident) = by_vertex.get(&v) else {
                continue;
            };
            if incident.len() != 4 {
                continue;
            }
            // The current edge's fan neighbours at `v`: the other edges of the
            // faces it borders that also touch `v`.
            let mut blocked: BTreeSet<u32> = BTreeSet::new();
            for &(fi, k) in users.get(id as usize).into_iter().flatten() {
                let face = &mesh.faces[fi];
                let n = face.corners.len();
                let p = face.corners[k].position;
                let q = face.corners[(k + 1) % n].position;
                let neighbour = if p == v {
                    face_edge_id(face, (k + n - 1) % n, &id_of)
                } else if q == v {
                    face_edge_id(face, (k + 1) % n, &id_of)
                } else {
                    None
                };
                if let Some(neighbour) = neighbour {
                    blocked.insert(neighbour);
                }
            }
            for &other in incident {
                if other != id && !blocked.contains(&other) {
                    frontier.push(other);
                }
            }
        }
    }
    out
}

/// The edge id of every canonical position pair, so a side found by walking a
/// face comes back as the id [`Mesh::edges`] gave it.
fn edge_ids(edges: &[[u32; 2]]) -> HashMap<(u32, u32), u32> {
    edges
        .iter()
        .enumerate()
        .map(|(i, [a, b])| ((*a, *b), i as u32))
        .collect()
}

/// The side at corner `k` of a face (from corner `k` to `k + 1`, wrapping) as
/// an edge id — `None` for a degenerate pair or one `edges` does not hold.
fn face_edge_id(face: &Face, k: usize, id_of: &HashMap<(u32, u32), u32>) -> Option<u32> {
    let n = face.corners.len();
    let a = face.corners[k % n].position;
    let b = face.corners[(k + 1) % n].position;
    if a == b {
        return None;
    }
    let key = if a < b { (a, b) } else { (b, a) };
    id_of.get(&key).copied()
}

/// Every `(face index, corner index)` whose side is that edge: the faces an
/// edge borders and where in each the side sits.
fn edge_users(
    mesh: &Mesh,
    edges: &[[u32; 2]],
    id_of: &HashMap<(u32, u32), u32>,
) -> Vec<Vec<(usize, usize)>> {
    let mut users = vec![Vec::new(); edges.len()];
    for (fi, face) in mesh.faces.iter().enumerate() {
        if !mesh.face_is_valid(face) {
            continue;
        }
        let n = face.corners.len();
        for k in 0..n {
            if let Some(id) = face_edge_id(face, k, id_of) {
                users[id as usize].push((fi, k));
            }
        }
    }
    users
}

/// The neighbour list of every element at a level, indexed the way the level
/// indexes its elements.
fn adjacency(mesh: &Mesh, level: SelectLevel) -> Vec<Vec<u32>> {
    match level {
        SelectLevel::Object => Vec::new(),
        SelectLevel::Vertex => {
            let mut adj = vec![Vec::new(); mesh.positions.len()];
            for [a, b] in mesh.edges() {
                adj[a as usize].push(b);
                adj[b as usize].push(a);
            }
            dedup(&mut adj);
            adj
        }
        SelectLevel::Edge => {
            let edges = mesh.edges();
            let mut by_vertex: HashMap<u32, Vec<u32>> = HashMap::new();
            for (i, [a, b]) in edges.iter().enumerate() {
                by_vertex.entry(*a).or_default().push(i as u32);
                by_vertex.entry(*b).or_default().push(i as u32);
            }
            let mut adj = vec![Vec::new(); edges.len()];
            for (i, [a, b]) in edges.iter().enumerate() {
                for v in [a, b] {
                    for &other in by_vertex.get(v).into_iter().flatten() {
                        if other as usize != i {
                            adj[i].push(other);
                        }
                    }
                }
            }
            dedup(&mut adj);
            adj
        }
        SelectLevel::Face => {
            // Faces are neighbours when they share a mesh edge, so key each
            // face's sides by the same lower-first pair `Mesh::edges` stores.
            let mut by_side: HashMap<(u32, u32), Vec<u32>> = HashMap::new();
            for (i, face) in mesh.faces.iter().enumerate() {
                if !mesh.face_is_valid(face) {
                    continue;
                }
                let n = face.corners.len();
                for k in 0..n {
                    let a = face.corners[k].position;
                    let b = face.corners[(k + 1) % n].position;
                    if a == b {
                        continue;
                    }
                    let key = if a < b { (a, b) } else { (b, a) };
                    by_side.entry(key).or_default().push(i as u32);
                }
            }
            let mut adj = vec![Vec::new(); mesh.faces.len()];
            for users in by_side.values() {
                for &i in users {
                    for &j in users {
                        if j != i {
                            adj[i as usize].push(j);
                        }
                    }
                }
            }
            dedup(&mut adj);
            adj
        }
    }
}

/// Sort and dedup each neighbour list (two faces can share more than one side).
fn dedup(adj: &mut [Vec<u32>]) {
    for list in adj {
        list.sort_unstable();
        list.dedup();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::Corner;
    use glam::Vec3;

    /// A 2×2 grid of quads on the XY plane: 9 vertices (`y * 3 + x`), four
    /// faces — 0 lower left, 1 lower right, 2 upper left, 3 upper right — and
    /// the 12 edges [`Mesh::edges`] walks: sides 0..=3 belong to face 0, 4..=6
    /// to face 1, 7..=9 to face 2, 10 and 11 to face 3.
    fn grid() -> Mesh {
        let mut m = Mesh::new();
        for y in 0..3 {
            for x in 0..3 {
                m.add_vertex(Vec3::new(x as f32, y as f32, 0.0));
            }
        }
        m.add_face(&[0, 1, 4, 3]);
        m.add_face(&[1, 2, 5, 4]);
        m.add_face(&[3, 4, 7, 6]);
        m.add_face(&[4, 5, 8, 7]);
        m
    }

    fn set(items: impl IntoIterator<Item = u32>) -> BTreeSet<u32> {
        items.into_iter().collect()
    }

    #[test]
    fn all_elements_counts_each_level() {
        let m = grid();
        assert!(all_elements(&m, SelectLevel::Object).is_empty(), "nodes are the scene's");
        assert_eq!(all_elements(&m, SelectLevel::Vertex).len(), 9);
        assert_eq!(all_elements(&m, SelectLevel::Edge).len(), 12);
        assert_eq!(all_elements(&m, SelectLevel::Face).len(), 4);
    }

    #[test]
    fn an_undrawable_face_is_not_selectable_and_joins_nothing() {
        let mut m = grid();
        // A face referencing a position that is not there: the visible pass
        // skips it whole, so it is no selectable face and no adjacency.
        m.faces.push(Face {
            corners: vec![
                Corner { position: 0, normal: 0, uv: 0 },
                Corner { position: 1, normal: 0, uv: 0 },
                Corner { position: 42, normal: 0, uv: 0 },
            ],
            material: 0,
        });
        assert_eq!(all_elements(&m, SelectLevel::Face).len(), 4, "the bad face is not one");
        assert_eq!(all_elements(&m, SelectLevel::Edge).len(), 12, "nor extra sides");
        // Vertex 8 is the far corner of face 3; growing from 7 (its neighbour)
        // reaches it through face 3's side, and the bad face changes nothing.
        assert_eq!(grow_selection(&m, SelectLevel::Vertex, &set([7])), set([4, 6, 7, 8]));
    }

    #[test]
    fn invert_flips_the_levels_own_elements() {
        let m = grid();
        assert_eq!(
            invert_selection(&m, SelectLevel::Face, &set([0, 1])),
            set([2, 3])
        );
        assert_eq!(
            invert_selection(&m, SelectLevel::Vertex, &set([0])),
            set([1, 2, 3, 4, 5, 6, 7, 8])
        );
        assert!(
            invert_selection(&m, SelectLevel::Face, &all_elements(&m, SelectLevel::Face)).is_empty()
        );
    }

    #[test]
    fn grow_takes_face_neighbours_across_edges_but_not_a_bare_corner() {
        let m = grid();
        // Face 0 shares sides with 1 (right) and 2 (above); face 3 touches it
        // only at vertex 4, which is not adjacency.
        assert_eq!(
            grow_selection(&m, SelectLevel::Face, &set([0])),
            set([0, 1, 2])
        );
        // Growing the whole selection is the whole selection.
        let all = all_elements(&m, SelectLevel::Face);
        assert_eq!(grow_selection(&m, SelectLevel::Face, &all), all);
    }

    #[test]
    fn shrink_cuts_the_boundary_grow_would_have_reached() {
        let m = grid();
        // Growing one face then shrinking gives it back.
        let grown = grow_selection(&m, SelectLevel::Face, &set([0]));
        assert_eq!(shrink_selection(&m, SelectLevel::Face, &grown), set([0]));
        // Two faces side by side: each still touches an unselected neighbour,
        // so one step shrinks them away entirely.
        assert_eq!(
            shrink_selection(&m, SelectLevel::Face, &set([0, 1])),
            BTreeSet::new()
        );
        // Everything selected has no boundary to cut.
        let all = all_elements(&m, SelectLevel::Face);
        assert_eq!(shrink_selection(&m, SelectLevel::Face, &all), all);
        assert!(shrink_selection(&m, SelectLevel::Face, &BTreeSet::new()).is_empty());
    }

    #[test]
    fn grow_walks_vertices_and_edges_by_their_shared_parts() {
        let m = grid();
        // Vertex 0 joins vertices 1 and 3.
        assert_eq!(grow_selection(&m, SelectLevel::Vertex, &set([0])), set([0, 1, 3]));
        // Edge 0 is [0, 1]: the edges sharing 0 or 1 are 1 ([1, 4]),
        // 3 ([0, 3]) and 4 ([1, 2]).
        assert_eq!(
            grow_selection(&m, SelectLevel::Edge, &set([0])),
            set([0, 1, 3, 4])
        );
        // Two steps from the corner vertex: the four edge-neighbours of the
        // corner (1, 3, 4, 6) land in reach, while the diagonal ones (5 and 7)
        // and the far corner (8) are still further out — grow never jumps a
        // corner.
        let one = grow_selection(&m, SelectLevel::Vertex, &set([0]));
        let two = grow_selection(&m, SelectLevel::Vertex, &one);
        assert_eq!(two, set([0, 1, 2, 3, 4, 6]));
        assert!(!two.contains(&5), "the diagonal stays out");
        let three = grow_selection(&m, SelectLevel::Vertex, &two);
        assert_eq!(three, set([0, 1, 2, 3, 4, 5, 6, 7]), "the far corner is four edges away");
        let four = grow_selection(&m, SelectLevel::Vertex, &three);
        assert_eq!(four, all_elements(&m, SelectLevel::Vertex), "four steps cover the grid");
    }

    /// A unit cube: eight corners (index `z * 4 + y * 2 + x`) and six quads,
    /// one per side, so every edge borders exactly two faces and every corner
    /// carries three edges.
    fn cube() -> Mesh {
        let mut m = Mesh::new();
        for z in [0.0f32, 1.0] {
            for y in [0.0f32, 1.0] {
                for x in [0.0f32, 1.0] {
                    m.add_vertex(Vec3::new(x, y, z));
                }
            }
        }
        m.add_face(&[0, 1, 3, 2]); // bottom (z = 0)
        m.add_face(&[4, 6, 7, 5]); // top (z = 1)
        m.add_face(&[0, 4, 5, 1]); // front (y = 0)
        m.add_face(&[2, 3, 7, 6]); // back (y = 1)
        m.add_face(&[0, 2, 6, 4]); // left (x = 0)
        m.add_face(&[1, 5, 7, 3]); // right (x = 1)
        m
    }

    #[test]
    fn a_ring_walks_the_parallel_edges_across_quads() {
        let m = grid();
        // Edge 1 is [1, 4], the middle vertical of the lower band: its ring is
        // that whole band — x = 0, 1, 2 between y = 0 and 1 — because the walk
        // runs perpendicular to the edge and stops where the grid does.
        let band = set([1, 3, 5]);
        assert_eq!(edge_ring(&m, 1), band);
        // Every member of a ring gives the same ring back: the walk closes.
        for &e in &[1u32, 3, 5] {
            assert_eq!(edge_ring(&m, e), band, "from edge {e}");
        }
        // Edge 2 ([3, 4]) is a horizontal of the left band: the horizontals at
        // x = 0..1, and the upper verticals ring the same way.
        assert_eq!(edge_ring(&m, 2), set([0, 2, 8]));
        assert_eq!(edge_ring(&m, 7), set([7, 9, 10]));
    }

    #[test]
    fn a_loop_runs_straight_through_four_edge_vertices_only() {
        let m = grid();
        // Edge 1 ([1, 4]) has four-edge vertex 4 at one end — straight on to
        // edge 7 ([4, 7]) — and border vertex 1 at the other, which stops it.
        assert_eq!(edge_loop(&m, 1), set([1, 7]));
        assert_eq!(edge_loop(&m, 7), set([1, 7]));
        // The horizontal middle row through vertices 4 and 5.
        assert_eq!(edge_loop(&m, 2), set([2, 6]));
        // A border edge runs off the mesh at both ends.
        assert_eq!(edge_loop(&m, 0), set([0]));
        // A triangle with one four-edge vertex still ends the walk: the chain
        // is a property of the vertex fan, not of the edge count alone.
        assert_eq!(edge_loop(&cube(), 0).len(), 1, "a cube corner carries three edges");
    }

    #[test]
    fn every_cube_edge_rings_around_the_cube_and_closes() {
        let m = cube();
        let edges = m.edges();
        assert_eq!(edges.len(), 12, "a cube has twelve edges");
        for e in 0..edges.len() as u32 {
            let ring = edge_ring(&m, e);
            assert_eq!(ring.len(), 4, "a cube ring is four edges: {ring:?} from {e}");
            assert!(ring.contains(&e));
            for &member in &ring {
                assert_eq!(edge_ring(&m, member), ring, "the walk closes through {member}");
            }
        }
    }

    #[test]
    fn a_ring_stops_at_a_face_that_is_not_a_quad() {
        let mut m = Mesh::new();
        for (x, y) in [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (0.0, 1.0), (1.0, 1.0)] {
            m.add_vertex(Vec3::new(x, y, 0.0));
        }
        m.add_face(&[0, 1, 4, 3]); // a quad
        m.add_face(&[1, 2, 4]); // a triangle sharing the side [1, 4]
        let edges = m.edges();
        let side = |pair: [u32; 2]| {
            edges
                .iter()
                .position(|e| *e == pair)
                .expect("the side exists") as u32
        };
        let shared = side([1, 4]);
        let opposite = side([0, 3]);
        // Across the quad the walk lands on the opposite side and stops at the
        // triangle's other face; through the triangle it never starts.
        assert_eq!(edge_ring(&m, shared), set([shared, opposite]));
    }
}
