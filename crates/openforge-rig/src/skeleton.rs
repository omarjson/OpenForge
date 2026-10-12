//! A bone hierarchy with a bind (rest) pose and an editable animated pose.

use crate::bone::{Bone, LocalTransform};
use crate::error::RigError;
use glam::{Mat4, Quat, Vec3};

/// An ordered bone hierarchy plus its animated pose.
///
/// **Ordering invariant:** a bone's parent (when `Some`) must be an *earlier*
/// bone in the list. [`Skeleton::add_bone`] enforces this, which makes the
/// hierarchy acyclic by construction and lets world matrices be composed in a
/// single forward pass (`child_world = parent_world * child_local`).
///
/// The skeleton keeps two poses:
///
/// * the **bind (rest) pose**, stored on each [`Bone`]'s `transform`, fixed
///   when the bone is added, and
/// * the **current (animated) pose**, a parallel list of local transforms
///   initialized to the bind pose and edited via the `set_pose_*` methods.
///
/// The classic linear-blend-skinning matrix for each bone is
/// `inverse(bind_world) * current_world` (see [`Skeleton::skinning_matrices`]).
#[derive(Clone, PartialEq, Debug, Default)]
pub struct Skeleton {
    bones: Vec<Bone>,
    pose: Vec<LocalTransform>,
}

impl Skeleton {
    /// An empty skeleton.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a bone and return its index.
    ///
    /// Fails with [`RigError::InvalidParent`] when the bone declares a parent
    /// that is not an earlier bone in the skeleton.
    pub fn add_bone(&mut self, bone: Bone) -> Result<usize, RigError> {
        if let Some(p) = bone.parent {
            if p >= self.bones.len() {
                return Err(RigError::InvalidParent {
                    child: self.bones.len(),
                    parent: p,
                });
            }
        }
        self.pose.push(bone.transform);
        self.bones.push(bone);
        Ok(self.bones.len() - 1)
    }

    /// Build a straight chain of `count` bones: the first is a root at the
    /// identity and each next bone is translated by `step` in its parent's
    /// local space. Bone `i` sits at `i * step` in model space at rest.
    pub fn chain(name: &str, count: usize, step: Vec3) -> Skeleton {
        let mut sk = Skeleton::new();
        for i in 0..count {
            let parent = if i == 0 { None } else { Some(i - 1) };
            let transform = if i == 0 {
                LocalTransform::IDENTITY
            } else {
                LocalTransform::from_translation(step)
            };
            let bone = Bone::with_transform(format!("{name}_{i}"), parent, transform);
            // Invariant holds by construction; ignore the unreachable error.
            let _ = sk.add_bone(bone);
        }
        sk
    }

    /// All bones, in hierarchy order (parents before children).
    pub fn bones(&self) -> &[Bone] {
        &self.bones
    }

    /// The number of bones.
    pub fn len(&self) -> usize {
        self.bones.len()
    }

    /// `true` when the skeleton has no bones.
    pub fn is_empty(&self) -> bool {
        self.bones.is_empty()
    }

    /// The bone at `index`, or [`RigError::InvalidBone`].
    pub fn bone(&self, index: usize) -> Result<&Bone, RigError> {
        self.bones
            .get(index)
            .ok_or(RigError::InvalidBone { index, count: self.bones.len() })
    }

    /// The bone's **rest (bind) pose** local transform.
    pub fn bind_local(&self, index: usize) -> Result<&LocalTransform, RigError> {
        Ok(&self.bone(index)?.transform)
    }

    /// The bone's **current pose** local transform.
    pub fn pose_local(&self, index: usize) -> Result<&LocalTransform, RigError> {
        self.pose
            .get(index)
            .ok_or(RigError::InvalidBone { index, count: self.bones.len() })
    }

    /// Replace the bone's current-pose local transform.
    pub fn set_pose_local(&mut self, index: usize, transform: LocalTransform) -> Result<(), RigError> {
        let slot = self
            .pose
            .get_mut(index)
            .ok_or(RigError::InvalidBone { index, count: self.bones.len() })?;
        *slot = transform;
        Ok(())
    }

    /// Set the bone's current-pose local translation.
    pub fn set_pose_translation(&mut self, index: usize, translation: Vec3) -> Result<(), RigError> {
        let slot = self.pose_slot(index)?;
        slot.translation = translation;
        Ok(())
    }

    /// Set the bone's current-pose local rotation.
    pub fn set_pose_rotation(&mut self, index: usize, rotation: Quat) -> Result<(), RigError> {
        let slot = self.pose_slot(index)?;
        slot.rotation = rotation;
        Ok(())
    }

    /// Reset every bone's current pose back to its rest (bind) pose.
    pub fn reset_pose(&mut self) {
        for (slot, bone) in self.pose.iter_mut().zip(&self.bones) {
            *slot = bone.transform;
        }
    }

    /// The bone's world (model-space) matrix in the **current pose**, found by
    /// walking up the parent chain: `... * grandparent_local * parent_local *
    /// bone_local`.
    pub fn world_matrix(&self, index: usize) -> Result<Mat4, RigError> {
        self.world_from(&self.pose, index)
    }

    /// The bone's world (model-space) matrix in the **bind (rest) pose**.
    pub fn bind_world_matrix(&self, index: usize) -> Result<Mat4, RigError> {
        let bind: Vec<LocalTransform> = self.bones.iter().map(|b| b.transform).collect();
        self.world_from(&bind, index)
    }

    /// World matrices of the bind (rest) pose, one per bone in order.
    pub fn bind_world_matrices(&self) -> Vec<Mat4> {
        compose_world(&self.bones, &self.bones.iter().map(|b| b.transform).collect::<Vec<_>>())
    }

    /// World matrices of the current pose, one per bone in order.
    pub fn pose_world_matrices(&self) -> Vec<Mat4> {
        compose_world(&self.bones, &self.pose)
    }

    /// The linear-blend-skinning matrices `inverse(bind_world) * current_world`,
    /// one per bone in order.
    ///
    /// With the current pose equal to the bind pose every matrix is the
    /// identity, so a skinned mesh is left exactly at its rest positions.
    pub fn skinning_matrices(&self) -> Vec<Mat4> {
        let bind = self.bind_world_matrices();
        let current = self.pose_world_matrices();
        bind.iter()
            .zip(&current)
            .map(|(b, c)| b.inverse() * *c)
            .collect()
    }

    fn pose_slot(&mut self, index: usize) -> Result<&mut LocalTransform, RigError> {
        self.pose
            .get_mut(index)
            .ok_or(RigError::InvalidBone { index, count: self.bones.len() })
    }

    /// World matrix of `index` from the given per-bone local transforms, by
    /// walking up the parent chain and multiplying from the root down.
    fn world_from(&self, locals: &[LocalTransform], index: usize) -> Result<Mat4, RigError> {
        if index >= self.bones.len() {
            return Err(RigError::InvalidBone { index, count: self.bones.len() });
        }
        // Collect the chain root..=index; parents always precede children, so
        // this walk terminates.
        let mut chain = Vec::new();
        let mut i = index;
        loop {
            chain.push(i);
            match self.bones[i].parent {
                Some(p) => i = p,
                None => break,
            }
        }
        Ok(compose_chain(locals, &chain))
    }
}

/// Compose a chain given leaf..=root into one world matrix (multiplied
/// root-first).
fn compose_chain(locals: &[LocalTransform], chain: &[usize]) -> Mat4 {
    let mut world = Mat4::IDENTITY;
    for &i in chain.iter().rev() {
        world = world * locals[i].matrix();
    }
    world
}

/// World matrix of every bone in one forward pass.
fn compose_world(bones: &[Bone], locals: &[LocalTransform]) -> Vec<Mat4> {
    let mut out = Vec::with_capacity(bones.len());
    for (i, bone) in bones.iter().enumerate() {
        let local = locals[i].matrix();
        let world = match bone.parent {
            Some(p) => out[p] * local,
            None => local,
        };
        out.push(world);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::FRAC_PI_2;

    /// Root at identity with a child translated by +Y.
    fn two_bone() -> Skeleton {
        let mut sk = Skeleton::new();
        assert_eq!(sk.add_bone(Bone::new("root", None)).unwrap(), 0);
        let child = Bone::with_transform(
            "child",
            Some(0),
            LocalTransform::from_translation(Vec3::new(0.0, 2.0, 0.0)),
        );
        assert_eq!(sk.add_bone(child).unwrap(), 1);
        sk
    }

    #[test]
    fn add_bone_rejects_forward_or_missing_parent() {
        let mut sk = Skeleton::new();
        sk.add_bone(Bone::new("root", None)).unwrap();
        // Parent index 1 does not exist yet (and would be a child).
        let bad = Bone::new("bad", Some(1));
        assert!(matches!(
            sk.add_bone(bad),
            Err(RigError::InvalidParent { child: 1, parent: 1 })
        ));
        assert_eq!(sk.len(), 1);
    }

    #[test]
    fn child_world_is_parent_world_times_local() {
        let sk = two_bone();
        let root_world = sk.world_matrix(0).unwrap();
        let child_world = sk.world_matrix(1).unwrap();
        let expected = root_world
            * Mat4::from_translation(Vec3::new(0.0, 2.0, 0.0));
        assert!(child_world.abs_diff_eq(expected, 1e-5));
        assert!(child_world
            .transform_point3(Vec3::ZERO)
            .abs_diff_eq(Vec3::new(0.0, 2.0, 0.0), 1e-5));
    }

    #[test]
    fn world_matrix_walks_the_whole_chain() {
        let mut sk = Skeleton::chain("bone", 3, Vec3::new(0.0, 1.0, 0.0));
        // Pose the middle bone with a +X translation; the tip's world matrix
        // must include it.
        sk.set_pose_translation(1, Vec3::new(5.0, 0.0, 0.0)).unwrap();
        let tip_world = sk.world_matrix(2).unwrap();
        assert!(tip_world
            .transform_point3(Vec3::ZERO)
            .abs_diff_eq(Vec3::new(5.0, 1.0, 0.0), 1e-5));
    }

    #[test]
    fn bind_pose_gives_identity_skinning_matrices() {
        // The critical correctness property: at rest, skin matrices are the
        // identity, so skinning leaves the mesh exactly where it is.
        let sk = two_bone();
        for m in sk.skinning_matrices() {
            assert!(m.abs_diff_eq(Mat4::IDENTITY, 1e-5), "{m:?}");
        }
    }

    #[test]
    fn skinning_matrix_tracks_pose_changes() {
        let mut sk = two_bone();
        // Pose the root: translate +X and rotate 90° about Z.
        let rot = Quat::from_rotation_z(FRAC_PI_2);
        sk.set_pose_local(
            0,
            LocalTransform::from_rotation(rot).with_translation(Vec3::new(1.0, 0.0, 0.0)),
        )
        .unwrap();
        let mats = sk.skinning_matrices();
        // Root: bind world was identity, so its skin matrix is the pose local.
        assert!(mats[0].abs_diff_eq(Mat4::from_translation(Vec3::new(1.0, 0.0, 0.0)) * Mat4::from_quat(rot), 1e-5));
        // Child: skin = inverse(bind_world) * current_world, where bind world
        // was T(0,2,0) and current world is root_pose * T(0,2,0).
        let root_pose = Mat4::from_translation(Vec3::new(1.0, 0.0, 0.0)) * Mat4::from_quat(rot);
        let expected = Mat4::from_translation(Vec3::new(0.0, -2.0, 0.0))
            * root_pose
            * Mat4::from_translation(Vec3::new(0.0, 2.0, 0.0));
        assert!(mats[1].abs_diff_eq(expected, 1e-5));
    }

    #[test]
    fn reset_pose_restores_identity_skinning() {
        let mut sk = two_bone();
        sk.set_pose_translation(0, Vec3::new(3.0, 0.0, 0.0)).unwrap();
        assert!(sk.skinning_matrices()[0] != Mat4::IDENTITY);
        sk.reset_pose();
        for m in sk.skinning_matrices() {
            assert!(m.abs_diff_eq(Mat4::IDENTITY, 1e-5));
        }
    }

    #[test]
    fn chain_builds_a_parented_hierarchy() {
        let sk = Skeleton::chain("bone", 4, Vec3::new(0.0, 1.0, 0.0));
        assert_eq!(sk.len(), 4);
        assert_eq!(sk.bones()[0].parent, None);
        assert_eq!(sk.bones()[1].parent, Some(0));
        assert_eq!(sk.bones()[3].parent, Some(2));
        assert_eq!(sk.bones()[2].name, "bone_2");
        for (i, world) in sk.bind_world_matrices().iter().enumerate() {
            let expected = Vec3::new(0.0, i as f32, 0.0);
            assert!(world.transform_point3(Vec3::ZERO).abs_diff_eq(expected, 1e-5));
        }
    }

    #[test]
    fn out_of_range_bone_index_errors() {
        let sk = two_bone();
        assert!(matches!(
            sk.world_matrix(7),
            Err(RigError::InvalidBone { index: 7, count: 2 })
        ));
        assert!(sk.bone(7).is_err());
        assert!(sk.pose_local(7).is_err());
        assert!(sk.bind_local(7).is_err());
        let mut sk = sk;
        assert!(sk.set_pose_translation(7, Vec3::ZERO).is_err());
        assert!(sk.set_pose_rotation(7, Quat::IDENTITY).is_err());
        assert!(sk.set_pose_local(7, LocalTransform::IDENTITY).is_err());
    }

    #[test]
    fn pose_defaults_to_bind_pose() {
        let sk = two_bone();
        for i in 0..sk.len() {
            assert_eq!(sk.pose_local(i).unwrap(), sk.bind_local(i).unwrap());
        }
        assert_eq!(sk.pose_world_matrices(), sk.bind_world_matrices());
    }
}
