//! Errors produced by rigging and skinning operations.

/// Errors produced by rigging and skinning operations.
///
/// Fallible operations (adding bones with bogus parents, posing or querying
/// bones by index, assigning skin weights) return this type rather than
/// panicking.
#[derive(Debug, thiserror::Error)]
pub enum RigError {
    #[error("bone index {index} is out of bounds for a skeleton with {count} bones")]
    InvalidBone { index: usize, count: usize },
    #[error("bone {child} declares parent {parent}, which is not an earlier bone in the skeleton")]
    InvalidParent { child: usize, parent: usize },
    #[error("weight {weight} assigned to bone {bone} must be finite and non-negative")]
    InvalidWeight { bone: usize, weight: f32 },
    #[error("vertex {0} has no positive bone weights")]
    NoWeights(usize),
    #[error("index {0} out of bounds")]
    IndexOutOfBounds(usize),
    #[error("mesh has {positions} vertices but weight data covers {weights}")]
    MismatchedVertexCounts { positions: usize, weights: usize },
}
