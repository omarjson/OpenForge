//! Procedural mesh generation and mesh algorithms for OpenForge.
//!
//! This crate turns parameters and rules into ready-to-edit [`Mesh`]es and
//! provides a few classic mesh algorithms:
//!
//! * [`primitives`] — parametric primitives: box, grid, torus, cylinder,
//!   cone, capsule, icosphere, superquadric, and helix tube.
//! * [`subdivision`] — Catmull-Clark subdivision surfaces.
//! * [`noise`] — seeded gradient noise, fBm, and normal-space displacement.
//! * [`lsystem`] — L-system string rewriting with a 3D turtle that emits line
//!   segments and tube meshes.
//!
//! Everything here is deterministic and defensive: degenerate parameters are
//! clamped so no generator or algorithm panics on bad input.
//!
//! ```ignore
//! use openforge_algorithms::{primitives, subdivision};
//! let sphere = primitives::ico_sphere(1.0, 2);
//! let smooth = subdivision::catmull_clark(&primitives::box_mesh(1.0, 1.0, 1.0), 2);
//! ```

pub mod lsystem;
pub mod noise;
pub mod primitives;
pub mod subdivision;

pub use lsystem::{segments_to_mesh, turtle_segments, LSystem};
pub use noise::{displace_along_normals, fbm2, fbm3, Perlin};
pub use primitives::{
    box_mesh, capsule, cone, cylinder, grid, helix, ico_sphere, superquadric, torus,
};
pub use subdivision::{catmull_clark, catmull_clark_level};
