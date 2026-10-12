//! Skeletal rigging and linear blend skinning for OpenForge.
//!
//! This crate turns a mesh + skeleton pair into deformed geometry:
//!
//! 1. Build a [`Skeleton`] out of [`Bone`]s (each with a name, an optional
//!    parent index that must refer to an *earlier* bone, and a local rest
//!    transform). [`Skeleton::chain`] builds a simple bone chain in one call.
//! 2. Pose the rig per frame with the `set_pose_*` methods (or
//!    [`Skeleton::reset_pose`] to return to rest) and fetch
//!    [`Skeleton::skinning_matrices`] — the classic linear-blend-skinning
//!    matrices `inverse(bind_world) * current_world`. At rest they are the
//!    identity, so an unposed mesh is never touched.
//! 3. Assign per-vertex influences with [`SkinWeights`] (normalized, top-N
//!    clamped, validated), then deform positions with [`skin_positions`] or a
//!    whole [`openforge_core::mesh::Mesh`] with [`skin_mesh`].
//!
//! Everything fallible returns a [`RigError`] instead of panicking.

pub mod bone;
pub mod error;
pub mod skeleton;
pub mod skin;

pub use bone::{Bone, LocalTransform};
pub use error::RigError;
pub use skeleton::Skeleton;
pub use skin::{skin_mesh, skin_positions, Influence, SkinWeights, DEFAULT_MAX_INFLUENCES};

// Re-exported for convenience so callers can pose and skin without importing
// the sibling crates directly.
pub use glam::{Mat4, Quat, Vec3};
pub use openforge_core::mesh::Mesh;
