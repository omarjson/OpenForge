//! Selection state: which nodes and which sub-elements are selected.

use crate::NodeId;
use std::collections::{BTreeSet, HashMap};

/// The granularity the user is selecting at.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum SelectLevel {
    Object,
    Vertex,
    Edge,
    #[default]
    Face,
}

/// Element selections keyed by owning mesh node.
#[derive(Clone, Debug, Default)]
pub struct Selection {
    pub level: SelectLevel,
    nodes: BTreeSet<NodeId>,
    /// Per-node selected element indices (vertex/edge/face depending on level).
    elements: HashMap<NodeId, BTreeSet<u32>>,
}

impl Selection {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn clear(&mut self) {
        self.nodes.clear();
        self.elements.clear();
    }

    /// Drop the element selection but keep the nodes.
    ///
    /// Element indices only mean anything at the level they were picked at —
    /// face 3 and vertex 3 are different things — so anything that changes the
    /// level has to call this rather than reinterpret the index.
    pub fn clear_elements(&mut self) {
        self.elements.clear();
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty() && self.elements.is_empty()
    }

    pub fn selected_nodes(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.nodes.iter().copied()
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn select_node(&mut self, id: NodeId) {
        self.nodes.insert(id);
    }

    pub fn deselect_node(&mut self, id: NodeId) {
        self.nodes.remove(&id);
        self.elements.remove(&id);
    }

    pub fn toggle_node(&mut self, id: NodeId) {
        if !self.nodes.remove(&id) {
            self.nodes.insert(id);
        } else {
            self.elements.remove(&id);
        }
    }

    pub fn is_node_selected(&self, id: NodeId) -> bool {
        self.nodes.contains(&id)
    }

    pub fn select_elements(&mut self, id: NodeId, idx: impl IntoIterator<Item = u32>) {
        self.nodes.insert(id);
        self.elements.entry(id).or_default().extend(idx);
    }

    pub fn selected_elements(&self, id: NodeId) -> Option<&BTreeSet<u32>> {
        self.elements.get(&id)
    }

    /// The selected *face* indices of `id`, when the face level is the one
    /// being edited.
    ///
    /// Element indices only mean anything at the level they were picked at, so
    /// a caller asking about faces gets nothing while another level is active:
    /// an edge index is not a face index, however much the number looks like
    /// one.
    pub fn selected_faces(&self, id: NodeId) -> Option<&BTreeSet<u32>> {
        if self.level == SelectLevel::Face {
            self.selected_elements(id)
        } else {
            None
        }
    }

    pub fn element_count(&self, id: NodeId) -> usize {
        self.elements.get(&id).map(|s| s.len()).unwrap_or(0)
    }

    pub fn total_element_count(&self) -> usize {
        self.elements.values().map(|s| s.len()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn toggle_node() {
        let mut s = Selection::new();
        let id = NodeId(1);
        s.toggle_node(id);
        assert!(s.is_node_selected(id));
        s.toggle_node(id);
        assert!(!s.is_node_selected(id));
        assert!(s.is_empty());
    }

    #[test]
    fn element_selection() {
        let mut s = Selection::new();
        let id = NodeId(7);
        s.select_elements(id, [0u32, 2, 4]);
        assert_eq!(s.element_count(id), 3);
        assert_eq!(s.total_element_count(), 3);
    }

    #[test]
    fn face_indices_only_read_as_faces_at_the_face_level() {
        let mut s = Selection::new();
        let id = NodeId(7);
        assert!(s.selected_faces(id).is_none(), "nothing is selected yet");
        s.select_elements(id, [0u32, 2, 4]);
        assert_eq!(
            s.selected_faces(id).map(|f| f.len()),
            Some(3),
            "the default level is the face level"
        );
        s.level = SelectLevel::Edge;
        assert!(
            s.selected_faces(id).is_none(),
            "an edge index must not read as a face"
        );
        assert_eq!(s.element_count(id), 3, "the indices themselves are untouched");
    }

    #[test]
    fn clearing_elements_keeps_the_nodes() {
        let mut s = Selection::new();
        let id = NodeId(7);
        s.select_elements(id, [0u32, 1]);
        s.clear_elements();
        assert_eq!(s.total_element_count(), 0);
        assert!(s.is_node_selected(id), "the node is still the one being edited");
        assert_eq!(s.node_count(), 1);
    }
}
