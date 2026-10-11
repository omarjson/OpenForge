//! The [`Scene`] owns all nodes and derives world transforms.

use crate::math::{Aabb, Transform};
use crate::node::{Node, NodeKind};
use crate::{CoreError, CoreResult, NodeId};
use glam::Mat4;
use std::collections::HashMap;

/// A scene graph of nodes with a single logical root set.
#[derive(Clone, Debug, Default)]
pub struct Scene {
    nodes: HashMap<NodeId, Node>,
    next_id: u32,
    /// Top-level nodes (no parent).
    roots: Vec<NodeId>,
    world_cache: HashMap<NodeId, Mat4>,
}

impl Scene {
    pub fn new() -> Self {
        Self::default()
    }

    fn alloc_id(&mut self) -> NodeId {
        let id = NodeId(self.next_id);
        self.next_id += 1;
        id
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn roots(&self) -> &[NodeId] {
        &self.roots
    }

    pub fn contains(&self, id: NodeId) -> bool {
        self.nodes.contains_key(&id)
    }

    pub fn get(&self, id: NodeId) -> CoreResult<&Node> {
        self.nodes.get(&id).ok_or(CoreError::MissingNode(id))
    }

    pub fn get_mut(&mut self, id: NodeId) -> CoreResult<&mut Node> {
        self.world_cache.clear();
        self.nodes.get_mut(&id).ok_or(CoreError::MissingNode(id))
    }

    pub fn iter(&self) -> impl Iterator<Item = &Node> {
        self.nodes.values()
    }

    /// Insert a fully-formed node, wiring up its parent links.
    pub fn insert(&mut self, mut node: Node) -> NodeId {
        let id = if node.id.is_invalid() {
            let id = self.alloc_id();
            node.id = id;
            id
        } else {
            node.id
        };
        self.world_cache.clear();
        let parent = node.parent;
        self.nodes.insert(id, node);
        match parent {
            Some(p) => {
                if let Some(parent) = self.nodes.get_mut(&p) {
                    if !parent.children.contains(&id) {
                        parent.children.push(id);
                    }
                }
            }
            None => {
                if !self.roots.contains(&id) {
                    self.roots.push(id);
                }
            }
        }
        id
    }

    pub fn create_group(&mut self, name: impl Into<String>) -> NodeId {
        let node = Node::group(NodeId::INVALID, name);
        self.insert(node)
    }

    pub fn create_dummy(&mut self, name: impl Into<String>) -> NodeId {
        let node = Node::dummy(NodeId::INVALID, name);
        self.insert(node)
    }

    pub fn create_mesh(&mut self, name: impl Into<String>, mesh: crate::mesh::Mesh) -> NodeId {
        let node = Node::mesh_node(NodeId::INVALID, name, mesh);
        self.insert(node)
    }

    /// Re-parent `child` under `new_parent` (or a root if `None`).
    pub fn reparent(&mut self, child: NodeId, new_parent: Option<NodeId>) -> CoreResult<()> {
        if let Some(p) = new_parent {
            if self.is_ancestor(child, p) {
                return Err(CoreError::Cycle);
            }
        }
        // Detach from old parent.
        if let Some(old) = self.nodes.get(&child).and_then(|n| n.parent) {
            if let Some(op) = self.nodes.get_mut(&old) {
                op.children.retain(|&c| c != child);
            }
        } else {
            self.roots.retain(|&r| r != child);
        }
        // Attach to new.
        if let Some(p) = new_parent {
            if let Some(np) = self.nodes.get_mut(&p) {
                np.children.push(child);
            }
        } else {
            self.roots.push(child);
        }
        if let Some(n) = self.nodes.get_mut(&child) {
            n.parent = new_parent;
        }
        self.world_cache.clear();
        Ok(())
    }

    /// True if `maybe_ancestor` is an ancestor of (or equal to) `node`.
    pub fn is_ancestor(&self, maybe_ancestor: NodeId, node: NodeId) -> bool {
        let mut cur = Some(node);
        while let Some(c) = cur {
            if c == maybe_ancestor {
                return true;
            }
            cur = self.nodes.get(&c).and_then(|n| n.parent);
        }
        false
    }

    /// Remove a node and its whole subtree. Returns removed ids.
    pub fn remove_subtree(&mut self, id: NodeId) -> Vec<NodeId> {
        let mut stack = vec![id];
        let mut order = Vec::new();
        while let Some(cur) = stack.pop() {
            if let Some(node) = self.nodes.get(&cur) {
                stack.extend(node.children.iter().copied());
            }
            order.push(cur);
        }
        if let Some(parent) = self.nodes.get(&id).and_then(|n| n.parent) {
            if let Some(p) = self.nodes.get_mut(&parent) {
                p.children.retain(|&c| c != id);
            }
        } else {
            self.roots.retain(|&r| r != id);
        }
        for removed in &order {
            self.nodes.remove(removed);
        }
        self.world_cache.clear();
        order
    }

    /// World matrix for a node, composing ancestors' transforms.
    pub fn world_matrix(&mut self, id: NodeId) -> CoreResult<Mat4> {
        if let Some(m) = self.world_cache.get(&id) {
            return Ok(*m);
        }
        let (local, parent) = {
            let n = self.nodes.get(&id).ok_or(CoreError::MissingNode(id))?;
            (n.transform.to_matrix(), n.parent)
        };
        let world = match parent {
            Some(p) => self.world_matrix(p)? * local,
            None => local,
        };
        self.world_cache.insert(id, world);
        Ok(world)
    }

    pub fn world_transform(&mut self, id: NodeId) -> CoreResult<Transform> {
        let m = self.world_matrix(id)?;
        Ok(Transform::from_matrix(m))
    }

    /// World-space AABB over a set of nodes (meshes only contribute geometry).
    pub fn world_bounds(&mut self, ids: &[NodeId]) -> Aabb {
        let mut out = Aabb::default();
        for &id in ids {
            if let Ok(m) = self.world_matrix(id) {
                let local = self.nodes.get(&id).map(|n| n.local_bounds()).unwrap_or_default();
                if !local.is_empty() {
                    out.merge(&local.transform(m));
                }
            }
        }
        out
    }

    /// Convenience: every node id in the scene.
    pub fn all_ids(&self) -> Vec<NodeId> {
        self.nodes.keys().copied().collect()
    }

    /// Names of mesh-bearing nodes, useful for the outliner and tests.
    pub fn mesh_names(&self) -> Vec<String> {
        let mut v: Vec<String> = self
            .nodes
            .values()
            .filter(|n| matches!(n.kind, NodeKind::Mesh { .. }))
            .map(|n| n.name.clone())
            .collect();
        v.sort();
        v
    }

    /// Compare the meaningful content (nodes + roots), ignoring the derived
    /// world-matrix cache. Used to detect whether an edit changed anything.
    pub fn content_eq(&self, other: &Scene) -> bool {
        self.roots == other.roots && self.nodes == other.nodes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    #[test]
    fn insert_and_parent() {
        let mut s = Scene::new();
        let g = s.create_group("root");
        let m = s.create_mesh("cube", crate::mesh::Mesh::new());
        s.reparent(m, Some(g)).unwrap();
        assert_eq!(s.roots(), &[g]);
        assert_eq!(s.get(g).unwrap().children, vec![m]);
    }

    #[test]
    fn cycle_rejected() {
        let mut s = Scene::new();
        let a = s.create_group("a");
        let b = s.create_group("b");
        s.reparent(b, Some(a)).unwrap();
        assert!(matches!(s.reparent(a, Some(b)), Err(CoreError::Cycle)));
    }

    #[test]
    fn world_matrix_composes() {
        let mut s = Scene::new();
        let g = s.create_group("g");
        let m = s.create_mesh("m", crate::mesh::Mesh::new());
        s.reparent(m, Some(g)).unwrap();
        s.get_mut(g).unwrap().transform.translation = Vec3::new(10.0, 0.0, 0.0);
        s.get_mut(m).unwrap().transform.translation = Vec3::new(0.0, 5.0, 0.0);
        let w = s.world_matrix(m).unwrap();
        assert!((w.w_axis.x - 10.0).abs() < 1e-5);
        assert!((w.w_axis.y - 5.0).abs() < 1e-5);
    }

    #[test]
    fn remove_subtree_removes_children() {
        let mut s = Scene::new();
        let g = s.create_group("g");
        let m = s.create_mesh("m", crate::mesh::Mesh::new());
        s.reparent(m, Some(g)).unwrap();
        let removed = s.remove_subtree(g);
        assert_eq!(removed.len(), 2);
        assert_eq!(s.node_count(), 0);
    }
}
