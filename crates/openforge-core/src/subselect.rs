//! Sub-object selection maths: the Select menu's operations.
//!
//! Each function is a pure map from one element set to another at a level. The
//! app owns *which* node it applies to; the mesh owns what "adjacent" means,
//! and adjacency follows the topology the rest of the app answers from: faces
//! are neighbours when they share a mesh edge (never a bare corner), edges
//! when they share a position, vertices when an edge joins them — and the edge
//! list is [`Mesh::edges`], the one enumeration every picker numbers by, so an
//! element the visible pass skips contributes no adjacency either.

use crate::mesh::Mesh;
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
    use crate::mesh::{Corner, Face};
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
}
