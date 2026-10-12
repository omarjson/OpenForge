//! Materials, textures, and shading for OpenForge.
//!
//! This crate is clean-room: original code implementing standard, textbook
//! material concepts (Phong-style material descriptions, RGBA8 textures with
//! wrap-addressed sampling, Lambertian/Phong CPU shading). It depends only on
//! `openforge-core` for shared math and error types.
//!
//! The main pieces are:
//!
//! - [`Material`] — colors, optional texture maps and flags for a surface.
//! - [`Texture`] — an RGBA8 image with `solid`/`checkerboard` constructors
//!   and wrap-addressed nearest sampling.
//! - [`MaterialLibrary`] — owns the project's materials and textures, and
//!   resolves mesh material slots safely.
//! - [`shade_lambert`] / [`shade_phong`] — small CPU shading helpers for the
//!   viewport preview.

pub mod error;
pub mod library;
pub mod material;
pub mod shading;
pub mod texture;

pub use error::MaterialError;
pub use library::MaterialLibrary;
pub use material::Material;
pub use shading::{shade_lambert, shade_phong};
pub use texture::{Texture, MISSING_TEXEL};

/// Returns the crate name, used by a smoke test to confirm linkage.
pub fn name() -> &'static str {
    "openforge-materials"
}

#[cfg(test)]
mod tests {
    #[test]
    fn links() {
        assert_eq!(super::name(), "openforge-materials");
    }
}
