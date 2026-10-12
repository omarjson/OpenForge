//! A single bone: a name, a parent link, and a local (rest/bind) transform.

use glam::{Mat4, Quat, Vec3};

/// A TRS transform in the space of a bone's parent.
///
/// Stored as separate translation/rotation/scale components so poses can be
/// edited (and later interpolated) component-wise before being composed into
/// a matrix.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct LocalTransform {
    pub translation: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

impl LocalTransform {
    /// The identity transform.
    pub const IDENTITY: LocalTransform = LocalTransform {
        translation: Vec3::ZERO,
        rotation: Quat::IDENTITY,
        scale: Vec3::ONE,
    };

    /// The identity transform.
    pub fn identity() -> Self {
        Self::IDENTITY
    }

    /// A pure translation.
    pub fn from_translation(translation: Vec3) -> Self {
        Self { translation, ..Self::IDENTITY }
    }

    /// A pure rotation.
    pub fn from_rotation(rotation: Quat) -> Self {
        Self { rotation, ..Self::IDENTITY }
    }

    /// This transform with its translation replaced (builder style).
    pub fn with_translation(mut self, translation: Vec3) -> Self {
        self.translation = translation;
        self
    }

    /// A translation, rotation and scale.
    pub fn from_scale_rotation_translation(scale: Vec3, rotation: Quat, translation: Vec3) -> Self {
        Self { translation, rotation, scale }
    }

    /// The matrix form `T * R * S` (rotation and scale applied first, in the
    /// parent bone's space).
    pub fn matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.translation)
    }

    /// Transform a point by this TRS.
    pub fn transform_point(&self, p: Vec3) -> Vec3 {
        self.matrix().transform_point3(p)
    }
}

impl Default for LocalTransform {
    fn default() -> Self {
        // Not derived: a default scale of zero would collapse the bone.
        Self::IDENTITY
    }
}

/// One bone of a [`crate::Skeleton`].
///
/// `transform` is the bone's local **rest (bind) pose** transform relative to
/// its parent; the animated pose lives on the skeleton, not on the bone.
/// `parent` is an index into the skeleton's bone list; by the skeleton's
/// invariant it is always `None` (a root) or refers to an *earlier* bone.
#[derive(Clone, PartialEq, Debug)]
pub struct Bone {
    /// Human-readable bone name (not required to be unique).
    pub name: String,
    /// Index of the parent bone, or `None` for a root.
    pub parent: Option<usize>,
    /// Local rest (bind) pose transform relative to the parent.
    pub transform: LocalTransform,
}

impl Bone {
    /// A bone at the identity local transform.
    pub fn new(name: impl Into<String>, parent: Option<usize>) -> Self {
        Self {
            name: name.into(),
            parent,
            transform: LocalTransform::IDENTITY,
        }
    }

    /// A bone with a local transform relative to its parent.
    pub fn with_transform(
        name: impl Into<String>,
        parent: Option<usize>,
        transform: LocalTransform,
    ) -> Self {
        Self {
            name: name.into(),
            parent,
            transform,
        }
    }

    /// The bone's local rest (bind) pose matrix.
    pub fn local_matrix(&self) -> Mat4 {
        self.transform.matrix()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::FRAC_PI_2;

    #[test]
    fn default_bone_is_identity() {
        let b = Bone::new("root", None);
        assert_eq!(b.name, "root");
        assert_eq!(b.parent, None);
        assert_eq!(b.transform, LocalTransform::IDENTITY);
        assert_eq!(b.local_matrix(), Mat4::IDENTITY);
    }

    #[test]
    fn local_matrix_is_translation_times_rotation_times_scale() {
        let t = LocalTransform::from_scale_rotation_translation(
            Vec3::splat(2.0),
            Quat::from_rotation_z(FRAC_PI_2),
            Vec3::new(1.0, 0.0, 0.0),
        );
        // Scale first (X -> 2X), then rotate 90° about Z (2X -> 2Y), then
        // translate by +X.
        let p = t.transform_point(Vec3::X);
        assert!(p.abs_diff_eq(Vec3::new(1.0, 2.0, 0.0), 1e-5), "{p:?}");
    }

    #[test]
    fn from_translation_and_from_rotation_are_pure() {
        let tr = LocalTransform::from_translation(Vec3::new(0.0, 3.0, 0.0));
        assert_eq!(tr.transform_point(Vec3::ZERO), Vec3::new(0.0, 3.0, 0.0));
        let rot = LocalTransform::from_rotation(Quat::from_rotation_x(FRAC_PI_2));
        assert!(rot
            .transform_point(Vec3::Y)
            .abs_diff_eq(Vec3::new(0.0, 0.0, 1.0), 1e-5));
    }
}
