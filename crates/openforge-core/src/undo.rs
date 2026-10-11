//! Undo/redo via captured scene snapshots.
//!
//! Editing commands capture the scene state before and after they run. This is
//! simple, robust, and adequate for a foundation; a production editor would
//! switch to fine-grained commands for memory efficiency.

use crate::scene::Scene;

/// A recorded undo step.
pub struct UndoEntry {
    pub label: String,
    before: Scene,
    after: Scene,
}

/// An undo/redo stack.
pub struct UndoStack {
    entries: Vec<UndoEntry>,
    /// Number of entries that are currently applied (the "redo" tail starts here).
    applied: usize,
    limit: usize,
}

impl Default for UndoStack {
    fn default() -> Self {
        Self::new(100)
    }
}

impl UndoStack {
    pub fn new(limit: usize) -> Self {
        Self { entries: Vec::new(), applied: 0, limit: limit.max(1) }
    }

    /// Record a completed edit. `before` is the state prior to the edit and
    /// `after` is the current state.
    pub fn push(&mut self, label: impl Into<String>, before: Scene, after: Scene) {
        // Drop the redo tail.
        self.entries.truncate(self.applied);
        self.entries.push(UndoEntry { label: label.into(), before, after });
        if self.entries.len() > self.limit {
            let overflow = self.entries.len() - self.limit;
            self.entries.drain(0..overflow);
        }
        self.applied = self.entries.len();
    }

    pub fn can_undo(&self) -> bool {
        self.applied > 0
    }

    pub fn can_redo(&self) -> bool {
        self.applied < self.entries.len()
    }

    pub fn undo_label(&self) -> Option<&str> {
        if self.can_undo() {
            Some(&self.entries[self.applied - 1].label)
        } else {
            None
        }
    }

    pub fn redo_label(&self) -> Option<&str> {
        self.entries.get(self.applied).map(|e| e.label.as_str())
    }

    /// Step backward, returning the scene to restore.
    pub fn undo(&mut self) -> Option<Scene> {
        if !self.can_undo() {
            return None;
        }
        self.applied -= 1;
        Some(self.entries[self.applied].before.clone())
    }

    /// Step forward, returning the scene to restore.
    pub fn redo(&mut self) -> Option<Scene> {
        if !self.can_redo() {
            return None;
        }
        let scene = self.entries[self.applied].after.clone();
        self.applied += 1;
        Some(scene)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.applied = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn undo_redo_round_trip() {
        let mut stack = UndoStack::new(10);
        let a = Scene::new();
        let mut b = Scene::new();
        b.create_mesh("x", crate::mesh::Mesh::new());

        stack.push("add x", a.clone(), b.clone());
        assert!(stack.can_undo());
        assert!(!stack.can_redo());

        let restored = stack.undo().unwrap();
        assert_eq!(restored.node_count(), a.node_count());
        assert!(stack.can_redo());

        let redone = stack.redo().unwrap();
        assert_eq!(redone.node_count(), b.node_count());
        assert!(!stack.can_redo());
    }

    #[test]
    fn redo_tail_cleared_on_new_push() {
        let mut stack = UndoStack::new(10);
        let s0 = Scene::new();
        let s1 = Scene::new();
        stack.push("one", s0.clone(), s1.clone());
        stack.undo();
        assert!(stack.can_redo());
        stack.push("two", s0.clone(), s1.clone());
        assert!(!stack.can_redo());
    }
}
