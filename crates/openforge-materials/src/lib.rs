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
//! - [`MapKind`] — the four maps a material can carry, so code loops over them
//!   instead of branching on each field.
//! - [`Texture`] — an RGBA8 image with `solid`/`checkerboard` constructors
//!   and wrap-addressed nearest sampling.
//! - [`MaterialLibrary`] — owns the project's materials and textures, and
//!   resolves mesh material slots safely.
//! - [`shade_lambert`] / [`shade_phong`] / [`shade_surface`] — small CPU
//!   shading helpers for the viewport preview.
//! - [`maps::MapTexels`] — what the maps contribute at one sample point, the
//!   same identity values the GPU path gets from its reserved white layer.

pub mod error;
pub mod library;
pub mod maps;
pub mod material;
pub mod shading;
pub mod texture;

pub use error::MaterialError;
pub use library::MaterialLibrary;
pub use maps::{apply_normal_map, tangent_frame, MapTexels};
pub use material::{MapKind, Material};
pub use shading::{shade_lambert, shade_phong, shade_surface};
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
