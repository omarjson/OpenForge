//! Per-vertex skin weights and linear blend skinning.

use crate::error::RigError;
use glam::{Mat4, Vec3};
use openforge_core::mesh::Mesh;

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
}
