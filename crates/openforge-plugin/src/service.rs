//! Services: the named capabilities a plugin publishes.
//!
//! This mirrors the study architecture, where a plugin host exposes a fixed set
//! of *services* (selection, undo, materials, …) that plugins consume and add
//! to. A service is deliberately opaque here — the host only needs its name and
//! a description for the UI. Anything richer belongs in a typed registry of the
//! consumer's own, keyed off the service name.

use crate::error::PluginError;
use std::collections::BTreeMap;

/// A named capability provided by a plugin.
pub trait Service: Send + Sync {
    /// The lookup name, unique within its registry (e.g. `"selection"`).
    fn name(&self) -> &str;
    /// A one-line description for the plugins panel.
    fn description(&self) -> &str {
        ""
    }
}

/// The smallest useful service: a name and a description.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SimpleService {
    name: String,
    description: String,
}

impl SimpleService {
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self { name: name.into(), description: description.into() }
    }
}

impl Service for SimpleService {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }
}

/// Services keyed by name, in a stable (alphabetical) order.
#[derive(Default)]
pub struct ServiceRegistry {
    services: BTreeMap<String, Box<dyn Service>>,
}

impl ServiceRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a service. A second service with the same name is rejected rather
    /// than silently replacing the first: two plugins claiming `"materials"`
    /// is a bug, and the error names the offender.
    pub fn insert(&mut self, service: Box<dyn Service>) -> Result<(), PluginError> {
        let name = service.name().to_string();
        if name.is_empty() {
            return Err(PluginError::EmptyName { kind: "service" });
        }
        if self.services.contains_key(&name) {
            return Err(PluginError::duplicate("service", name));
        }
        self.services.insert(name, service);
        Ok(())
    }

    pub fn get(&self, name: &str) -> Option<&dyn Service> {
        self.services.get(name).map(|s| s.as_ref())
    }

    pub fn contains(&self, name: &str) -> bool {
        self.services.contains_key(name)
    }

    /// Every service name, alphabetically.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.services.keys().map(|k| k.as_str())
    }

    /// `(name, description)` for the UI, alphabetically.
    pub fn entries(&self) -> impl Iterator<Item = (&str, &str)> {
        self.services.iter().map(|(k, v)| (k.as_str(), v.description()))
    }

    /// Every service, unsorted, for merging one registry into another.
    pub fn values(&self) -> impl Iterator<Item = &dyn Service> {
        self.services.values().map(|s| s.as_ref())
    }

    /// Remove and hand back every service (used when merging registries).
    pub fn drain(&mut self) -> Vec<Box<dyn Service>> {
        std::mem::take(&mut self.services).into_values().collect()
    }

    pub fn len(&self) -> usize {
        self.services.len()
    }

    pub fn is_empty(&self) -> bool {
        self.services.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A service with a description, to check both come back.
    struct Labeled;

    impl Service for Labeled {
        fn name(&self) -> &str {
            "labeled"
        }
        fn description(&self) -> &str {
            "a described service"
        }
    }

    #[test]
    fn a_service_is_looked_up_by_name() {
        let mut registry = ServiceRegistry::new();
        registry.insert(Box::new(Labeled)).unwrap();
        assert!(registry.contains("labeled"));
        assert!(!registry.contains("missing"));
        assert_eq!(registry.get("labeled").unwrap().description(), "a described service");
        assert_eq!(registry.len(), 1);
        assert!(!registry.is_empty());
    }

    #[test]
    fn names_come_back_in_a_stable_order() {
        let mut registry = ServiceRegistry::new();
        for name in ["undo", "selection", "materials"] {
            registry.insert(Box::new(SimpleService::new(name, ""))).unwrap();
        }
        assert_eq!(
            registry.names().collect::<Vec<_>>(),
            ["materials", "selection", "undo"],
            "BTreeMap keeps the panel's list from reshuffling every frame"
        );
    }

    #[test]
    fn a_duplicate_name_is_rejected_describing_the_clash() {
        let mut registry = ServiceRegistry::new();
        registry.insert(Box::new(SimpleService::new("undo", "first"))).unwrap();
        let err = registry.insert(Box::new(SimpleService::new("undo", "second"))).unwrap_err();
        assert_eq!(
            err,
            PluginError::Duplicate { kind: "service", name: "undo".into() }
        );
        // The original is untouched: no silent replacement.
        assert_eq!(registry.get("undo").unwrap().description(), "first");
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn an_empty_name_is_rejected() {
        let mut registry = ServiceRegistry::new();
        let err = registry.insert(Box::new(SimpleService::new("", ""))).unwrap_err();
        assert_eq!(err, PluginError::EmptyName { kind: "service" });
        assert!(registry.is_empty());
    }

    #[test]
    fn a_new_registry_is_empty() {
        let registry = ServiceRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
        assert!(registry.get("anything").is_none());
        assert_eq!(registry.entries().count(), 0);
    }
}
