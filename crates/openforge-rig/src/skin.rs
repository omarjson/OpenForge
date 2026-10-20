//! Per-vertex skin weights and linear blend skinning.

use crate::error::RigError;
use glam::{Mat4, Vec3};
use openforge_core::mesh::Mesh;
use std::collections::BTreeMap;

/// Default cap on how many bones may influence one vertex.
pub const DEFAULT_MAX_INFLUENCES: usize = 4;

/// One bone influence on a vertex: a bone index and a weight in `0..=1`.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Influence {
    /// Index of the influencing bone.
    pub bone: usize,
    /// Blend weight (non-negative; per-vertex weights sum to 1).
    pub weight: f32,
}

/// Skinning data for a whole mesh: for each vertex, the (up to
/// [`DEFAULT_MAX_INFLUENCES`]) bones that move it and their blend weights.
///
/// Weights are stored normalized: [`SkinWeights::set_weights`] validates the
/// input, merges repeated bone indices, clamps to the top-N influences and
/// renormalizes so they sum to exactly 1. A vertex with no influences is left
/// at its original (bind) position when skinned.
#[derive(Clone, PartialEq, Debug)]
pub struct SkinWeights {
    vertex_count: usize,
    bone_count: usize,
    max_influences: usize,
    influences: Vec<Vec<Influence>>,
}

impl SkinWeights {
    /// All-zero weights for a mesh with `vertex_count` vertices deformed by a
    /// skeleton with `bone_count` bones, allowing up to
    /// [`DEFAULT_MAX_INFLUENCES`] influences per vertex.
    pub fn new(vertex_count: usize, bone_count: usize) -> Self {
        Self::with_max_influences(vertex_count, bone_count, DEFAULT_MAX_INFLUENCES)
    }

    /// Like [`SkinWeights::new`] with a custom per-vertex influence cap
    /// (clamped to at least 1).
    pub fn with_max_influences(vertex_count: usize, bone_count: usize, max: usize) -> Self {
        Self {
            vertex_count,
            bone_count,
            max_influences: max.max(1),
            influences: vec![Vec::new(); vertex_count],
        }
    }

    /// Number of vertices covered.
    pub fn vertex_count(&self) -> usize {
        self.vertex_count
    }

    /// Number of bones the weights may reference.
    pub fn bone_count(&self) -> usize {
        self.bone_count
    }

    /// Maximum number of influences kept per vertex.
    pub fn max_influences(&self) -> usize {
        self.max_influences
    }

    /// The stored (normalized) influences of a vertex, heaviest first.
    pub fn influences(&self, vertex: usize) -> Result<&[Influence], RigError> {
        self.influences
            .get(vertex)
            .map(|v| v.as_slice())
            .ok_or(RigError::IndexOutOfBounds(vertex))
    }

    /// Assign a vertex's influences from `(bone, weight)` pairs.
    ///
    /// Weights must be finite and non-negative and bone indices must be valid;
    /// repeated bones are merged, the top [`SkinWeights::max_influences`]
    /// weights are kept, and the result is renormalized to sum to 1. Fails
    /// with [`RigError::NoWeights`] when the pairs carry no positive weight.
    pub fn set_weights(&mut self, vertex: usize, pairs: &[(usize, f32)]) -> Result<(), RigError> {
        if vertex >= self.vertex_count {
            return Err(RigError::IndexOutOfBounds(vertex));
        }
        // Merge duplicates, validating every pair.
        let mut merged: Vec<Influence> = Vec::with_capacity(pairs.len());
        for &(bone, weight) in pairs {
            if bone >= self.bone_count {
                return Err(RigError::InvalidBone { index: bone, count: self.bone_count });
            }
            if !weight.is_finite() || weight < 0.0 {
                return Err(RigError::InvalidWeight { bone, weight });
            }
            match merged.iter_mut().find(|inf| inf.bone == bone) {
                Some(existing) => existing.weight += weight,
                None => merged.push(Influence { bone, weight }),
            }
        }
        // Heaviest-first, then clamp to the top-N.
        merged.sort_by(|a, b| {
            b.weight.partial_cmp(&a.weight).unwrap_or(std::cmp::Ordering::Equal)
        });
        merged.truncate(self.max_influences);
        let total: f32 = merged.iter().map(|inf| inf.weight).sum();
        if total <= 0.0 {
            return Err(RigError::NoWeights(vertex));
        }
        for inf in &mut merged {
            inf.weight /= total;
        }
        self.influences[vertex] = merged;
        Ok(())
    }

    /// Bind every listed vertex to `bone` at full weight, replacing whatever
    /// influences it carried.
    ///
    /// Vertices outside the mesh and bones the skeleton does not have are
    /// skipped, and a vertex already bound exactly there is not counted as a
    /// change; the return value is how many vertices actually moved.
    pub fn assign(&mut self, vertices: &[u32], bone: usize) -> usize {
        let mut changed = 0;
        for &vertex in vertices {
            let vertex = vertex as usize;
            let Ok(existing) = self.influences(vertex) else {
                continue;
            };
            if existing.len() == 1 && existing[0] == (Influence { bone, weight: 1.0 }) {
                continue;
            }
            if self.set_weights(vertex, &[(bone, 1.0)]).is_ok() {
                changed += 1;
            }
        }
        changed
    }

    /// Move a signed amount of each vertex's influence onto `bone`.
    ///
    /// This is what a weight brush does: a stroke adds to the bone under the
    /// cursor and takes the same share from the others, so the vertex's total
    /// stays one. A negative amount takes the bone's share away instead, and
    /// the others take it back.
    ///
    /// `amounts` is one signed amount per vertex — the brush's falloff folded
    /// into the stroke's strength — so a stroke can fade out with the
    /// distance from the cursor. The amount is clamped to what the vertex can
    /// carry: a vertex already fully on the bone cannot take more, and one
    /// with nothing on it cannot give any away.
    ///
    /// Returns how many vertices changed.
    pub fn brush(&mut self, amounts: &[(u32, f32)], bone: usize) -> usize {
        let mut changed = 0;
        for &(vertex, amount) in amounts {
            if !amount.is_finite() || amount == 0.0 {
                continue;
            }
            let vertex = vertex as usize;
            let Ok(existing) = self.influences(vertex) else {
                continue;
            };
            let current: f32 = existing
                .iter()
                .find(|influence| influence.bone == bone)
                .map(|influence| influence.weight)
                .unwrap_or(0.0);
            let others: f32 = existing
                .iter()
                .filter(|influence| influence.bone != bone)
                .map(|influence| influence.weight)
                .sum();
            // Nothing to take from, or nowhere to give it: a vertex's
            // influences always add up to one, so weight can only move
            // between the bones that are there.
            if (amount > 0.0 && current >= 1.0 - 1e-6) || (amount < 0.0 && others <= 1e-6) {
                continue;
            }
            let new = (current + amount).clamp(0.0, 1.0);
            let delta = new - current;
            if delta.abs() < 1e-6 {
                continue;
            }
            // The other bones keep their proportions of what is left.
            let scale = if current >= 1.0 - 1e-6 {
                0.0
            } else {
                (1.0 - new) / (1.0 - current).max(1e-6)
            };
            let mut pairs: Vec<(usize, f32)> = existing
                .iter()
                .filter(|influence| influence.bone != bone)
                .map(|influence| (influence.bone, influence.weight * scale))
                .filter(|&(_, weight)| weight > 1e-6)
                .collect();
            pairs.push((bone, new));
            if self.set_weights(vertex, &pairs).is_ok() {
                changed += 1;
            }
        }
        changed
    }

    /// Move `weight` of every listed vertex's influence onto `bone`: the
    /// influences the vertex already carried keep their proportions of the
    /// remaining `1 - weight`, and the bone takes the rest.
    ///
    /// A vertex with no influences yet ends fully on the bone, exactly as one
    /// already fully there; `weight` outside `0..=1` and non-finite values are
    /// clamped or refused. Returns how many vertices changed.
    pub fn blend(&mut self, vertices: &[u32], bone: usize, weight: f32) -> usize {
        if !weight.is_finite() {
            return 0;
        }
        let weight = weight.clamp(0.0, 1.0);
        if weight <= 0.0 {
            return 0;
        }
        let mut changed = 0;
        for &vertex in vertices {
            let vertex = vertex as usize;
            let Ok(existing) = self.influences(vertex) else {
                continue;
            };
            let before: Vec<Influence> = existing.to_vec();
            let mut pairs: Vec<(usize, f32)> = existing
                .iter()
                .filter(|influence| influence.bone != bone)
                .map(|influence| (influence.bone, influence.weight * (1.0 - weight)))
                .filter(|&(_, weight)| weight > 0.0)
                .collect();
            pairs.push((bone, weight));
            if self.set_weights(vertex, &pairs).is_err() {
                continue;
            }
            match self.influences(vertex) {
                Ok(now) if now != before.as_slice() => changed += 1,
                _ => {}
            }
        }
        changed
    }

    /// Smooth the weights over `edges`, `iterations` times: each affected
    /// vertex averages its influences with those of its edge neighbours — the
    /// vertex itself counting once alongside every neighbour, whichever way
    /// the edge points — and the average is renormalized into a fresh set.
    ///
    /// Vertices outside `affect` keep their weights but still count as
    /// neighbours; an affected vertex with no neighbours keeps its own. The
    /// result is a new, independent `SkinWeights`; `self` is not written to.
    pub fn smooth(&self, edges: &[[u32; 2]], affect: &[u32], iterations: usize) -> SkinWeights {
        let mut out = self.clone();
        if iterations == 0 {
            return out;
        }
        let mut neighbours: Vec<Vec<usize>> = vec![Vec::new(); self.vertex_count];
        for &[a, b] in edges {
            let (a, b) = (a as usize, b as usize);
            if a == b || a >= self.vertex_count || b >= self.vertex_count {
                continue;
            }
            neighbours[a].push(b);
            neighbours[b].push(a);
        }
        let affected: Vec<usize> = affect
            .iter()
            .map(|&vertex| vertex as usize)
            .filter(|&vertex| vertex < self.vertex_count)
            .collect();
        for _ in 0..iterations {
            // Every pass reads the previous pass's weights, so the result does
            // not depend on the order the affected vertices are listed in.
            let mut next: Vec<Option<Vec<(usize, f32)>>> = vec![None; self.vertex_count];
            for &vertex in &affected {
                let around = &neighbours[vertex];
                if around.is_empty() {
                    continue;
                }
                let mut merged: BTreeMap<usize, f32> = BTreeMap::new();
                for influence in out.influences(vertex).unwrap_or(&[]) {
                    *merged.entry(influence.bone).or_insert(0.0) += influence.weight;
                }
                for &neighbour in around {
                    for influence in out.influences(neighbour).unwrap_or(&[]) {
                        *merged.entry(influence.bone).or_insert(0.0) += influence.weight;
                    }
                }
                let scale = 1.0 / (1 + around.len()) as f32;
                next[vertex] = Some(
                    merged
                        .iter()
                        .map(|(&bone, &weight)| (bone, weight * scale))
                        .collect(),
                );
            }
            for &vertex in &affected {
                if let Some(pairs) = &next[vertex] {
                    let _ = out.set_weights(vertex, pairs);
                }
            }
        }
        out
    }
}

/// Deform `positions` by the linear-blend-skinning matrices `matrices`
/// (one per bone, from [`crate::Skeleton::skinning_matrices`]) using
/// `weights`.
///
/// Each vertex becomes `sum_i weight_i * (matrix_i * vec4(pos, 1.0))`. A
/// vertex with no influences is returned unchanged. Fails when the weight
/// data and the position list disagree on length, or when an influence
/// references a missing matrix.
pub fn skin_positions(
    positions: &[Vec3],
    matrices: &[Mat4],
    weights: &SkinWeights,
) -> Result<Vec<Vec3>, RigError> {
    if weights.vertex_count() != positions.len() {
        return Err(RigError::MismatchedVertexCounts {
            positions: positions.len(),
            weights: weights.vertex_count(),
        });
    }
    let mut out = Vec::with_capacity(positions.len());
    for (vi, &pos) in positions.iter().enumerate() {
        let mut acc = Vec3::ZERO;
        for inf in weights.influences(vi)? {
            let m = matrices
                .get(inf.bone)
                .ok_or(RigError::InvalidBone { index: inf.bone, count: matrices.len() })?;
            let transformed = (*m * pos.extend(1.0)).truncate();
            acc += transformed * inf.weight;
        }
        // A vertex with no influences keeps its bind position.
        out.push(if weights.influences(vi)?.is_empty() { pos } else { acc });
    }
    Ok(out)
}

/// Deform `mesh` in place with the linear-blend-skinning matrices; see
/// [`skin_positions`] for the semantics.
pub fn skin_mesh(mesh: &mut Mesh, matrices: &[Mat4], weights: &SkinWeights) -> Result<(), RigError> {
    mesh.positions = skin_positions(&mesh.positions, matrices, weights)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bone::{Bone, LocalTransform};
    use crate::skeleton::Skeleton;
    use glam::Quat;
    use std::f32::consts::FRAC_PI_2;

    fn total(infs: &[Influence]) -> f32 {
        infs.iter().map(|i| i.weight).sum()
    }

    #[test]
    fn weights_are_normalized_to_one() {
        let mut w = SkinWeights::new(2, 4);
        w.set_weights(0, &[(0, 1.0), (1, 3.0)]).unwrap();
        let infs = w.influences(0).unwrap();
        assert_eq!(infs.len(), 2);
        assert!((total(infs) - 1.0).abs() < 1e-6);
        assert!((infs[0].bone == 1).then_some(()).is_some()); // heaviest first
        assert!((infs[0].weight - 0.75).abs() < 1e-6);
        assert!((infs[1].weight - 0.25).abs() < 1e-6);
    }

    #[test]
    fn influences_clamp_to_top_n_and_renormalize() {
        let mut w = SkinWeights::with_max_influences(1, 6, 4);
        let pairs: Vec<(usize, f32)> = (0..6).map(|i| (i, (i + 1) as f32)).collect();
        w.set_weights(0, &pairs).unwrap();
        let infs = w.influences(0).unwrap();
        assert_eq!(infs.len(), 4);
        // Bones 5..2 survive (weights 6,5,4,3) in descending order.
        assert_eq!(infs.iter().map(|i| i.bone).collect::<Vec<_>>(), vec![5, 4, 3, 2]);
        assert!((total(infs) - 1.0).abs() < 1e-6);
        assert!((infs[0].weight - 6.0 / 18.0).abs() < 1e-6);
    }

    #[test]
    fn duplicate_bones_are_merged() {
        let mut w = SkinWeights::new(1, 4);
        w.set_weights(0, &[(1, 0.2), (1, 0.4), (2, 0.4)]).unwrap();
        let infs = w.influences(0).unwrap();
        assert_eq!(infs.len(), 2);
        assert!((infs[0].weight - 0.6).abs() < 1e-6);
        assert!((infs[1].weight - 0.4).abs() < 1e-6);
    }

    #[test]
    fn invalid_weights_and_bones_are_rejected() {
        let mut w = SkinWeights::new(1, 4);
        // Negative weight.
        assert!(matches!(
            w.set_weights(0, &[(0, -0.5)]),
            Err(RigError::InvalidWeight { bone: 0, .. })
        ));
        // Non-finite weight.
        assert!(matches!(
            w.set_weights(0, &[(0, f32::NAN)]),
            Err(RigError::InvalidWeight { .. })
        ));
        // Out-of-range bone index.
        assert!(matches!(
            w.set_weights(0, &[(4, 1.0)]),
            Err(RigError::InvalidBone { index: 4, count: 4 })
        ));
        // Out-of-range vertex index.
        assert!(matches!(w.set_weights(9, &[(0, 1.0)]), Err(RigError::IndexOutOfBounds(9))));
        // All-zero pairs carry no weight at all.
        assert!(matches!(w.set_weights(0, &[(0, 0.0)]), Err(RigError::NoWeights(0))));
        // The failed writes left the vertex untouched.
        assert!(w.influences(0).unwrap().is_empty());
    }

    /// A single bone at the identity rest pose.
    fn one_bone() -> Skeleton {
        let mut sk = Skeleton::new();
        sk.add_bone(Bone::new("root", None)).unwrap();
        sk
    }

    #[test]
    fn rigid_binding_follows_the_bone_exactly() {
        let mut sk = one_bone();
        let pos = vec![Vec3::new(2.0, 0.0, 0.0), Vec3::ZERO, Vec3::new(0.0, 3.0, 0.0)];
        let mut w = SkinWeights::new(pos.len(), 1);
        for v in 0..pos.len() {
            w.set_weights(v, &[(0, 1.0)]).unwrap();
        }
        // Bind first: skinning must be a no-op.
        let mats = sk.skinning_matrices();
        let at_bind = skin_positions(&pos, &mats, &w).unwrap();
        for (a, b) in at_bind.iter().zip(&pos) {
            assert!(a.abs_diff_eq(*b, 1e-5), "{a:?} != {b:?}");
        }
        // Now rotate the bone 90° about Z and translate +X. The bone's world
        // matrix is exactly the posed matrix, and weight 1.0 must follow it.
        let posed_local = LocalTransform::from_rotation(Quat::from_rotation_z(FRAC_PI_2))
            .with_translation(Vec3::new(1.0, 0.0, 0.0));
        sk.set_pose_local(0, posed_local).unwrap();
        let mats = sk.skinning_matrices();
        let posed = skin_positions(&pos, &mats, &w).unwrap();
        for (a, b) in posed.iter().zip(&pos) {
            let expected = posed_local.matrix() * b.extend(1.0);
            assert!(a.abs_diff_eq(expected.truncate(), 1e-5), "{a:?}");
        }
    }

    #[test]
    fn two_bone_blend_lands_at_the_midpoint() {
        // Two bones: bone 0 at the identity, bone 1 translated to +X.
        let mut sk = Skeleton::new();
        sk.add_bone(Bone::new("a", None)).unwrap();
        sk.add_bone(Bone::with_transform(
            "b",
            None,
            LocalTransform::from_translation(Vec3::new(4.0, 0.0, 0.0)),
        ))
        .unwrap();
        // Pose each bone differently: bone 0 stays put, bone 1 moves to +X=8.
        sk.set_pose_translation(1, Vec3::new(8.0, 0.0, 0.0)).unwrap();
        let mut w = SkinWeights::new(1, 2);
        w.set_weights(0, &[(0, 0.5), (1, 0.5)]).unwrap();
        let pos = vec![Vec3::ZERO];
        let out = skin_positions(&pos, &sk.skinning_matrices(), &w).unwrap();
        // Bone 0's skin matrix leaves the vertex at 0; bone 1's moves it from
        // its bind spot (0) to 4. The 0.5/0.5 blend is the midpoint: 2.
        assert!(out[0].abs_diff_eq(Vec3::new(2.0, 0.0, 0.0), 1e-5), "{:?}", out[0]);
    }

    #[test]
    fn rest_pose_skinning_leaves_a_mesh_unchanged() {
        let sk = Skeleton::chain("bone", 3, Vec3::new(0.0, 1.0, 0.0));
        let mut mesh = Mesh::new();
        for p in [
            Vec3::new(0.5, 0.0, 0.0),
            Vec3::new(-0.5, 1.5, 0.25),
            Vec3::new(0.0, 2.0, -1.0),
        ] {
            mesh.add_vertex(p);
        }
        let mut w = SkinWeights::new(3, 3);
        w.set_weights(0, &[(0, 1.0)]).unwrap();
        w.set_weights(1, &[(1, 0.5), (2, 0.5)]).unwrap();
        w.set_weights(2, &[(2, 1.0)]).unwrap();
        let original = mesh.positions.clone();
        skin_mesh(&mut mesh, &sk.skinning_matrices(), &w).unwrap();
        for (a, b) in mesh.positions.iter().zip(&original) {
            assert!(a.abs_diff_eq(*b, 1e-5), "{a:?} != {b:?}");
        }
    }

    #[test]
    fn pose_skin_reset_round_trip() {
        let mut sk = Skeleton::chain("bone", 3, Vec3::new(0.0, 1.0, 0.0));
        let mut mesh = Mesh::new();
        for p in [
            Vec3::new(0.5, 0.0, 0.0),
            Vec3::new(0.5, 1.0, 0.0),
            Vec3::new(0.5, 2.0, 0.0),
            Vec3::new(-0.5, 3.0, 0.0),
        ] {
            mesh.add_vertex(p);
        }
        let mut w = SkinWeights::new(4, 3);
        w.set_weights(0, &[(0, 1.0)]).unwrap();
        w.set_weights(1, &[(0, 0.5), (1, 0.5)]).unwrap();
        w.set_weights(2, &[(1, 1.0)]).unwrap();
        w.set_weights(3, &[(2, 1.0)]).unwrap();
        let original = mesh.positions.clone();

        // Pose the rig, skin: the mesh must actually move.
        sk.set_pose_rotation(1, Quat::from_rotation_z(FRAC_PI_2)).unwrap();
        skin_mesh(&mut mesh, &sk.skinning_matrices(), &w).unwrap();
        assert!(mesh.positions[3] != original[3]);

        // Reset to bind pose and skin the ORIGINAL rest geometry again: the
        // original positions are recovered.
        sk.reset_pose();
        mesh.positions = original.clone();
        skin_mesh(&mut mesh, &sk.skinning_matrices(), &w).unwrap();
        for (a, b) in mesh.positions.iter().zip(&original) {
            assert!(a.abs_diff_eq(*b, 1e-5), "{a:?} != {b:?}");
        }
    }

    #[test]
    fn uninfluenced_vertices_are_left_alone() {
        let mut sk = Skeleton::chain("bone", 2, Vec3::Y);
        let pos = vec![Vec3::new(1.0, 2.0, 3.0), Vec3::ZERO];
        let mut w = SkinWeights::new(2, 2);
        w.set_weights(0, &[(1, 1.0)]).unwrap();
        // Vertex 1 never gets weights.
        sk.set_pose_translation(0, Vec3::new(5.0, 0.0, 0.0)).unwrap();
        let out = skin_positions(&pos, &sk.skinning_matrices(), &w).unwrap();
        assert!(out[0].abs_diff_eq(Vec3::new(6.0, 2.0, 3.0), 1e-5));
        assert_eq!(out[1], pos[1]);
    }

    #[test]
    fn skinning_detects_mismatched_counts_and_matrices() {
        let sk = Skeleton::chain("bone", 2, Vec3::Y);
        let w = SkinWeights::new(3, 2);
        // Two positions, weights for three vertices.
        assert!(matches!(
            skin_positions(&[Vec3::ZERO; 2], &sk.skinning_matrices(), &w),
            Err(RigError::MismatchedVertexCounts { positions: 2, weights: 3 })
        ));
        // One influence references a matrix that is not there.
        let mut w1 = SkinWeights::new(1, 2);
        w1.set_weights(0, &[(1, 1.0)]).unwrap();
        assert!(matches!(
            skin_positions(&[Vec3::ZERO], &[], &w1),
            Err(RigError::InvalidBone { index: 1, count: 0 })
        ));
    }

    #[test]
    fn assigning_binds_a_vertex_to_one_bone() {
        let mut w = SkinWeights::new(3, 2);
        w.set_weights(0, &[(0, 0.5), (1, 0.5)]).unwrap();

        assert_eq!(w.assign(&[0, 2], 1), 2, "both vertices changed");
        assert_eq!(w.influences(0).unwrap(), &[Influence { bone: 1, weight: 1.0 }]);
        assert_eq!(w.influences(2).unwrap(), &[Influence { bone: 1, weight: 1.0 }]);
        assert_eq!(w.assign(&[0], 1), 0, "already exactly that");
        assert_eq!(w.assign(&[7], 0), 0, "outside the mesh");
        assert_eq!(w.assign(&[1], 5), 0, "no such bone");
    }

    #[test]
    fn blending_moves_a_share_onto_the_bone() {
        let mut w = SkinWeights::new(2, 3);
        w.set_weights(0, &[(0, 1.0)]).unwrap();

        // Half the influence moves over; the rest keeps its proportion.
        assert_eq!(w.blend(&[0], 1, 0.5), 1);
        let inf = w.influences(0).unwrap();
        assert_eq!(inf.len(), 2, "{inf:?}");
        assert!((inf.iter().find(|i| i.bone == 0).unwrap().weight - 0.5).abs() < 1e-6);
        assert!((inf.iter().find(|i| i.bone == 1).unwrap().weight - 0.5).abs() < 1e-6);

        // An unbound vertex simply lands on the bone, and repeating a blend
        // that is already in effect counts nothing.
        assert_eq!(w.blend(&[1], 2, 0.75), 1);
        assert_eq!(w.influences(1).unwrap(), &[Influence { bone: 2, weight: 1.0 }]);
        assert_eq!(w.blend(&[1], 2, 0.75), 0, "already fully there");
        assert_eq!(w.blend(&[0], 1, 0.0), 0, "nothing to move");
        assert_eq!(w.blend(&[0], 1, f32::NAN), 0);
        assert_eq!(w.blend(&[0], 9, 0.5), 0, "no such bone");
    }

    #[test]
    fn smoothing_averages_over_the_edges() {
        let mut w = SkinWeights::new(3, 2);
        w.set_weights(0, &[(0, 1.0)]).unwrap();
        w.set_weights(1, &[(0, 1.0)]).unwrap();
        w.set_weights(2, &[(1, 1.0)]).unwrap();

        // A line 0 — 1 — 2; smoothing only the middle one averages its own
        // influence with both neighbours', pulling a third of bone 1 into it.
        let smoothed = w.smooth(&[[0, 1], [1, 2]], &[1], 1);
        let inf = smoothed.influences(1).unwrap();
        let weight = |bone: usize| {
            inf.iter().find(|i| i.bone == bone).map(|i| i.weight).unwrap_or(0.0)
        };
        assert!((weight(0) - 2.0 / 3.0).abs() < 1e-6, "{}", weight(0));
        assert!((weight(1) - 1.0 / 3.0).abs() < 1e-6, "{}", weight(1));
        assert!((total(inf) - 1.0).abs() < 1e-6, "still normalized");
        // The untouched vertices are the same as before, and the source was
        // not written to.
        assert_eq!(smoothed.influences(0).unwrap(), &[Influence { bone: 0, weight: 1.0 }]);
        assert_eq!(smoothed.influences(2).unwrap(), &[Influence { bone: 1, weight: 1.0 }]);
        assert_eq!(w.influences(1).unwrap(), &[Influence { bone: 0, weight: 1.0 }]);

        // A second pass pulls more of the far end in; zero passes is a copy,
        // and an edge that names no real neighbour leaves the vertex alone.
        let twice = w.smooth(&[[0, 1], [1, 2]], &[1], 2);
        let inf = twice.influences(1).unwrap();
        let far = inf.iter().find(|i| i.bone == 1).map(|i| i.weight).unwrap_or(0.0);
        assert!(far > 1.0 / 3.0, "{far}");
        assert_eq!(
            w.smooth(&[[0, 1]], &[1], 0).influences(1).unwrap(),
            w.influences(1).unwrap()
        );
        assert_eq!(
            w.smooth(&[[1, 9], [1, 1]], &[1], 1).influences(1).unwrap(),
            w.influences(1).unwrap()
        );
    }
    #[test]
    fn a_brush_stroke_moves_weight_onto_the_bone() {
        let mut w = SkinWeights::new(1, 2);
        w.set_weights(0, &[(0, 1.0)]).unwrap();
        // Half of the vertex's weight moves to the other bone.
        let changed = w.brush(&[(0, 0.5)], 1);
        assert_eq!(changed, 1);
        let infs = w.influences(0).unwrap();
        assert!((total(&infs) - 1.0).abs() < 1e-6, "the total stays one: {infs:?}");
        let on_bone: f32 = infs
            .iter()
            .find(|i| i.bone == 1)
            .map(|i| i.weight)
            .unwrap_or(0.0);
        assert!((on_bone - 0.5).abs() < 1e-6, "the bone took the stroke: {infs:?}");
        let on_other: f32 = infs
            .iter()
            .find(|i| i.bone == 0)
            .map(|i| i.weight)
            .unwrap_or(0.0);
        assert!((on_other - 0.5).abs() < 1e-6, "and the other gave it up: {infs:?}");
    }

    #[test]
    fn a_brush_stroke_can_take_weight_away_again() {
        let mut w = SkinWeights::new(1, 2);
        w.set_weights(0, &[(0, 0.5), (1, 0.5)]).unwrap();
        // A negative stroke takes the weight off the bone, and the other bone
        // takes it back.
        let changed = w.brush(&[(0, -0.25)], 1);
        assert_eq!(changed, 1);
        let infs = w.influences(0).unwrap();
        let on_bone: f32 = infs
            .iter()
            .find(|i| i.bone == 1)
            .map(|i| i.weight)
            .unwrap_or(0.0);
        assert!((on_bone - 0.25).abs() < 1e-6, "{infs:?}");
        let on_other: f32 = infs
            .iter()
            .find(|i| i.bone == 0)
            .map(|i| i.weight)
            .unwrap_or(0.0);
        assert!((on_other - 0.75).abs() < 1e-6, "{infs:?}");
        assert!((total(&infs) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn a_brush_stroke_stops_at_what_a_vertex_can_carry() {
        let mut w = SkinWeights::new(1, 2);
        w.set_weights(0, &[(0, 1.0)]).unwrap();
        // More than the vertex can hold is clamped: it ends fully on the bone.
        assert_eq!(w.brush(&[(0, 5.0)], 1), 1);
        let on_bone: f32 = w
            .influences(0)
            .unwrap()
            .iter()
            .find(|i| i.bone == 1)
            .map(|i| i.weight)
            .unwrap_or(0.0);
        assert!((on_bone - 1.0).abs() < 1e-6, "the vertex ended fully on the bone");
        // A vertex that is fully on a bone has nowhere to give the weight to.
        let mut w = SkinWeights::new(1, 2);
        w.set_weights(0, &[(1, 1.0)]).unwrap();
        assert_eq!(w.brush(&[(0, -0.5)], 1), 0);
        // A non-finite or empty stroke is refused.
        let mut w = SkinWeights::new(1, 2);
        w.set_weights(0, &[(0, 1.0)]).unwrap();
        assert_eq!(w.brush(&[(0, f32::NAN)], 1), 0);
        assert_eq!(w.brush(&[(0, 0.0)], 1), 0);
    }

    #[test]
    fn a_brush_stroke_fades_with_its_amounts() {
        let mut w = SkinWeights::new(2, 2);
        w.set_weights(0, &[(0, 1.0)]).unwrap();
        w.set_weights(1, &[(0, 1.0)]).unwrap();
        // The first vertex is under the cursor and takes the whole stroke; the
        // second is at the brush's edge and takes a tenth of it.
        let changed = w.brush(&[(0, 0.4), (1, 0.04)], 1);
        assert_eq!(changed, 2);
        let near: f32 = w
            .influences(0)
            .unwrap()
            .iter()
            .find(|i| i.bone == 1)
            .map(|i| i.weight)
            .unwrap_or(0.0);
        let far: f32 = w
            .influences(1)
            .unwrap()
            .iter()
            .find(|i| i.bone == 1)
            .map(|i| i.weight)
            .unwrap_or(0.0);
        assert!((near - 0.4).abs() < 1e-6, "{near}");
        assert!((far - 0.04).abs() < 1e-6, "{far}");
    }

}
