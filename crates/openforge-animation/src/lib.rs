//! Keyframe animation tracks, interpolation, and sampling for OpenForge.
//!
//! The crate is self-contained: it defines its own keyframe types rather than
//! reaching into the scene graph, so the application stays in charge of
//! driving it. The typical flow is:
//!
//! 1. Build [`Clip`]s out of [`TransformTrack`]s (each holding [`Track`]s of
//!    [`Key`]s with an [`Interp`] mode),
//! 2. register them in a [`ClipLibrary`],
//! 3. hand the library to a [`Player`], select a clip by name, call
//!    [`Player::play`], and call [`Player::step`] once per frame to get the
//!    `(NodeId, Transform)` pairs to apply to the scene.

pub mod clip;
pub mod interpolation;
pub mod keyframe;
pub mod player;
pub mod track;

pub use clip::{Clip, ClipLibrary, TransformTrack};
pub use interpolation::{slerp, Interp, Interpolatable};
pub use keyframe::Key;
pub use player::Player;
pub use track::Track;

// Re-exported for convenience so callers can drive animation without
// importing the sibling crates directly.
pub use openforge_core::NodeId;
pub use openforge_core::math::Transform;
