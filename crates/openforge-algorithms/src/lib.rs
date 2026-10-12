//! Procedural mesh generation and mesh algorithms.
//!
//! Placeholder crate filled in by a focused work stream. Depends only on
//! `openforge-core` for isolated development.

/// Returns the crate name, used by a smoke test to confirm linkage.
pub fn name() -> &'static str {
    "openforge-algorithms"
}

#[cfg(test)]
mod tests {
    #[test]
    fn links() {
        assert_eq!(super::name(), "openforge-algorithms");
    }
}
