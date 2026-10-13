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

    /// The library slot a face of this node resolves to.
    ///
    /// A face stores an index into the node's slot list, which is what lets one
    /// mesh carry several materials; a face whose index is out of range (or a
    /// node with no slots yet) falls back to slot `0`, the default material.
    pub fn face_material_slot(&self, face: usize) -> u32 {
        let NodeKind::Mesh { mesh, material_slots } = &self.kind else {
            return 0;
        };
        match mesh.faces.get(face) {
            Some(f) => material_slots.get(f.material as usize).copied().unwrap_or(0),
            None => 0,
        }
    }

    /// Assign one library material slot to the whole node.
    ///
    /// Every face is pointed at the single slot, so the mesh renders with that
    /// material whatever slots it carried before. Returns whether anything
    /// changed, so callers can avoid an undo entry for a no-op.
    pub fn assign_material(&mut self, slot: u32) -> bool {
        let NodeKind::Mesh { mesh, material_slots } = &mut self.kind else {
            return false;
        };
        let mut changed = *material_slots != vec![slot];
        *material_slots = vec![slot];
        for face in &mut mesh.faces {
            if face.material != 0 {
                face.material = 0;
                changed = true;
            }
        }
        changed
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

#[cfg(test)]
mod tests {
    use super::*;

    fn triangle() -> Mesh {
        let mut m = Mesh::new();
        m.add_vertex(Vec3::ZERO);
        m.add_vertex(Vec3::X);
        m.add_vertex(Vec3::Y);
        m.add_face(&[0, 1, 2]);
        m
    }

    fn two_faces() -> Mesh {
        let mut m = Mesh::new();
        m.add_vertex(Vec3::ZERO);
        m.add_vertex(Vec3::X);
        m.add_vertex(Vec3::X + Vec3::Y);
        m.add_vertex(Vec3::Y);
        m.add_face(&[0, 1, 2]);
        m.add_face(&[0, 2, 3]);
        m
    }

    #[test]
    fn a_fresh_mesh_node_resolves_to_the_default_slot() {
        let node = Node::mesh_node(NodeId(1), "tri", triangle());
        assert_eq!(node.face_material_slot(0), 0);
        // No faces carry a slot index yet, so the node is "unassigned".
        if let NodeKind::Mesh { material_slots, .. } = &node.kind {
            assert!(material_slots.is_empty());
        }
    }

    #[test]
    fn assigning_a_material_points_every_face_at_it() {
        let mut node = Node::mesh_node(NodeId(1), "quad", two_faces());
        if let NodeKind::Mesh { mesh, material_slots } = &mut node.kind {
            material_slots.push(7); // a stale slot the assignment must replace
            mesh.faces[1].material = 5; // ditto for a face's index
        }

        assert!(node.assign_material(3), "the assignment changed something");
        for face in 0..2 {
            assert_eq!(node.face_material_slot(face), 3, "face {face}");
        }
        if let NodeKind::Mesh { mesh, material_slots } = &node.kind {
            assert_eq!(material_slots, &vec![3]);
            assert!(mesh.faces.iter().all(|f| f.material == 0));
        }
    }

    #[test]
    fn re_assigning_the_same_material_reports_no_change() {
        let mut node = Node::mesh_node(NodeId(1), "tri", triangle());
        assert!(node.assign_material(2));
        assert!(!node.assign_material(2), "nothing left to change");
    }

    #[test]
    fn an_out_of_range_face_falls_back_to_the_default_slot() {
        let mut node = Node::mesh_node(NodeId(1), "tri", triangle());
        node.assign_material(9);
        assert_eq!(node.face_material_slot(99), 0, "no such face");
    }

    #[test]
    fn assigning_to_a_node_without_a_mesh_is_a_no_op() {
        let mut group = Node::group(NodeId(2), "group");
        assert!(!group.assign_material(4));
        assert_eq!(group.face_material_slot(0), 0);

        let mut dummy = Node::dummy(NodeId(3), "dummy");
        assert!(!dummy.assign_material(4));
    }
}
