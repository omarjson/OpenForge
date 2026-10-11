//! Scene graph nodes: groups, dummies and mesh nodes.

use crate::math::Transform;
use crate::mesh::Mesh;
use crate::NodeId;
use glam::Vec3;

/// What a node holds data-wise.
#[derive(Clone, PartialEq, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum NodeKind {
    /// A pure transform/grouping node with children.
    Group,
    /// A non-rendered placeholder used for pivots and attach points.
    Dummy,
    /// A renderable polygon mesh.
    Mesh { mesh: Mesh, material_slots: Vec<u32> },
}

/// A node in the scene graph.
#[derive(Clone, PartialEq, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Node {
    pub id: NodeId,
    pub name: String,
    pub kind: NodeKind,
    pub transform: Transform,
    pub parent: Option<NodeId>,
    pub children: Vec<NodeId>,
    pub visible: bool,
    /// When frozen, the node is not selectable in the viewport.
    pub frozen: bool,
}

impl Node {
    pub fn new(id: NodeId, name: impl Into<String>, kind: NodeKind) -> Self {
        Self {
            id,
            name: name.into(),
            kind,
            transform: Transform::identity(),
            parent: None,
            children: Vec::new(),
            visible: true,
            frozen: false,
        }
    }

    pub fn group(id: NodeId, name: impl Into<String>) -> Self {
        Node::new(id, name, NodeKind::Group)
    }

    pub fn dummy(id: NodeId, name: impl Into<String>) -> Self {
        Node::new(id, name, NodeKind::Dummy)
    }

    pub fn mesh_node(id: NodeId, name: impl Into<String>, mesh: Mesh) -> Self {
        Node::new(
            id,
            name,
            NodeKind::Mesh { mesh, material_slots: Vec::new() },
        )
    }

    pub fn is_mesh(&self) -> bool {
        matches!(self.kind, NodeKind::Mesh { .. })
    }

    pub fn mesh(&self) -> Option<&Mesh> {
        match &self.kind {
            NodeKind::Mesh { mesh, .. } => Some(mesh),
            _ => None,
        }
    }

    pub fn mesh_mut(&mut self) -> Option<&mut Mesh> {
        match &mut self.kind {
            NodeKind::Mesh { mesh, .. } => Some(mesh),
            _ => None,
        }
    }

    pub fn local_bounds(&self) -> crate::math::Aabb {
        match &self.kind {
            NodeKind::Mesh { mesh, .. } => mesh.bounds(),
            _ => crate::math::Aabb::default(),
        }
    }

    pub fn set_translation(&mut self, t: Vec3) {
        self.transform.translation = t;
    }
}
