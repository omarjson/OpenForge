//! OpenForge core: a clean-room scene graph and geometry model.
//!
//! This crate is deliberately GUI- and renderer-agnostic. It mirrors the
//! *concepts* of a node/scene editing model (group nodes, mesh nodes, a
//! selection set, an undo stack, interactive tools) without copying any
//! proprietary implementation.

pub mod edit;
pub mod math;
pub mod mesh;
pub mod node;
pub mod scene;
pub mod selection;
pub mod subselect;
pub mod undo;
pub mod tool;

pub use glam::{Vec2, Vec3, Vec4, Quat, Mat4};

/// A stable identifier for a node in a [`scene::Scene`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct NodeId(pub u32);

impl NodeId {
    /// An identifier that never refers to a real node.
    pub const INVALID: NodeId = NodeId(u32::MAX);
    pub fn is_invalid(self) -> bool {
        self == NodeId::INVALID
    }
}

/// Generic result alias used across the core.
pub type CoreResult<T> = Result<T, CoreError>;

/// Errors produced by core scene operations.
#[derive(Debug, thiserror::Error)]
pub enum CoreError {
    #[error("node {0:?} does not exist")]
    MissingNode(NodeId),
    #[error("operation would create a cycle in the scene graph")]
    Cycle,
    #[error("index {0} out of bounds")]
    IndexOutOfBounds(usize),
    #[error("invalid operation: {0}")]
    Invalid(&'static str),
}
