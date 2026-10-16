//! The maths behind the import/export options dialog: the up-axis a file
//! follows and a uniform scale, applied between the file's data and the
//! world's Y-up space.

use glam::Vec3;
use openforge_core::mesh::Mesh;
use openforge_io::LoadedScene;

/// The up-axis a file follows.
///
/// The world is Y-up, so [`UpAxis::Y`] is the as-is case and the other two
/// turn the data as it crosses the file boundary: on import into world space,
/// on export back into file space.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UpAxis {
    /// Y up — the world's own convention, so nothing turns.
    Y,
    /// Z up: import reads `(x, y, z)` as `(x, z, -y)`.
    Z,
    /// X up: import reads `(x, y, z)` as `(-y, x, z)`.
    X,
}

impl UpAxis {
    /// The label the options dialog shows.
    pub fn label(self) -> &'static str {
        match self {
            UpAxis::Y => "Y up (as-is)",
            UpAxis::Z => "Z up",
            UpAxis::X => "X up",
        }
    }

    /// Turn a file-space direction into world space: the import direction.
    fn to_world(self, v: Vec3) -> Vec3 {
        match self {
            UpAxis::Y => v,
            UpAxis::Z => Vec3::new(v.x, v.z, -v.y),
            UpAxis::X => Vec3::new(-v.y, v.x, v.z),
        }
    }

    /// Turn a world-space direction into file space: the export direction,
    /// the exact inverse of [`Self::to_world`].
    fn from_world(self, v: Vec3) -> Vec3 {
        match self {
            UpAxis::Y => v,
            UpAxis::Z => Vec3::new(v.x, -v.z, v.y),
            UpAxis::X => Vec3::new(v.y, -v.x, v.z),
        }
    }
}

/// What the import/export options dialog applies to a file's data.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FileOptions {
    /// The up-axis the file follows; the world is Y-up.
    pub up: UpAxis,
    /// Uniform scale applied after the axes turn. Non-finite and
    /// non-positive values fall back to `1.0` when applied.
    pub scale: f32,
}

impl Default for FileOptions {
    fn default() -> Self {
        Self {
            up: UpAxis::Y,
            scale: 1.0,
        }
    }
}

impl FileOptions {
    /// Whether these options leave a mesh exactly as it is — what the status
    /// line leaves unsaid.
    pub fn is_identity(self) -> bool {
        self.up == UpAxis::Y && (self.scale - 1.0).abs() < 1e-9
    }

    /// Apply the options to one mesh, into world space (`to_world`) or back
    /// into file space.
    ///
    /// Positions turn and scale; normals only turn — a uniform scale does not
    /// change a direction. A scale that is zero, negative or non-finite is
    /// ignored rather than flattening the mesh.
    pub fn apply(self, mesh: &mut Mesh, to_world: bool) {
        let scale = if self.scale.is_finite() && self.scale > 0.0 {
            self.scale
        } else {
            1.0
        };
        let turn = |v: Vec3| {
            if to_world {
                self.up.to_world(v)
            } else {
                self.up.from_world(v)
            }
        };
        for position in &mut mesh.positions {
            *position = turn(*position) * scale;
        }
        for normal in &mut mesh.normals {
            *normal = turn(*normal);
        }
    }
}

/// Apply the options to every mesh a filter loaded, on the way in.
pub fn apply_loaded(scene: &mut LoadedScene, options: FileOptions) {
    if options.is_identity() {
        return;
    }
    for data in &mut scene.meshes {
        options.apply(&mut data.mesh, true);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use openforge_io::MeshData;

    /// A mesh with a couple of positions and normals to turn.
    fn probe() -> Mesh {
        let mut m = Mesh::new();
        m.positions = vec![Vec3::new(1.0, 2.0, 3.0), Vec3::new(0.0, 0.0, 1.0)];
        m.normals = vec![Vec3::new(0.0, 1.0, 0.0)];
        m
    }

    #[test]
    fn the_z_up_turn_moves_the_up_axis_onto_y() {
        let options = FileOptions {
            up: UpAxis::Z,
            scale: 1.0,
        };
        let mut m = probe();
        options.apply(&mut m, true);
        // File (x, y, z) reads as world (x, z, -y): the file's +Z comes up.
        assert_eq!(m.positions[0], Vec3::new(1.0, 3.0, -2.0));
        assert_eq!(m.positions[1], Vec3::new(0.0, 1.0, 0.0));
        assert_eq!(m.normals[0], Vec3::new(0.0, 0.0, -1.0));
    }

    #[test]
    fn the_x_up_turn_moves_the_up_axis_onto_y() {
        let options = FileOptions {
            up: UpAxis::X,
            scale: 1.0,
        };
        let mut m = probe();
        options.apply(&mut m, true);
        assert_eq!(m.positions[0], Vec3::new(-2.0, 1.0, 3.0));
        assert_eq!(m.normals[0], Vec3::new(-1.0, 0.0, 0.0));
    }

    #[test]
    fn export_turns_are_the_exact_inverse_of_import_turns() {
        for up in [UpAxis::Y, UpAxis::Z, UpAxis::X] {
            let mut m = probe();
            let original = m.clone();
            // A file imported at one scale is written back out at its
            // reciprocal: the round trip lands on the original data.
            FileOptions { up, scale: 2.0 }.apply(&mut m, true);
            FileOptions { up, scale: 0.5 }.apply(&mut m, false);
            for (got, want) in m.positions.iter().zip(&original.positions) {
                assert!((*got - *want).length() < 1e-5, "{up:?}: {got:?} vs {want:?}");
            }
            for (got, want) in m.normals.iter().zip(&original.normals) {
                assert!((*got - *want).length() < 1e-5, "{up:?}: {got:?} vs {want:?}");
            }
        }
    }

    #[test]
    fn the_scale_multiplies_positions_but_not_normals() {
        let options = FileOptions {
            up: UpAxis::Y,
            scale: 2.5,
        };
        let mut m = probe();
        options.apply(&mut m, true);
        assert_eq!(m.positions[0], Vec3::new(2.5, 5.0, 7.5));
        assert_eq!(m.normals[0], Vec3::new(0.0, 1.0, 0.0), "a direction does not scale");
    }

    #[test]
    fn identity_options_and_bad_scales_leave_the_mesh_alone() {
        let mut m = probe();
        let original = m.clone();
        FileOptions::default().apply(&mut m, true);
        assert_eq!(m, original);
        assert!(FileOptions::default().is_identity());

        // Zero, negative and non-finite scales fall back to 1.0.
        for scale in [0.0, -3.0, f32::NAN] {
            let mut m = probe();
            FileOptions {
                up: UpAxis::Y,
                scale,
            }
            .apply(&mut m, true);
            assert_eq!(m.positions, original.positions, "scale {scale}");
        }
    }

    #[test]
    fn apply_loaded_turns_every_mesh_of_a_scene() {
        let mut scene = LoadedScene {
            meshes: vec![MeshData {
                name: "part".into(),
                mesh: probe(),
                material: None,
            }],
            materials: Vec::new(),
        };
        apply_loaded(
            &mut scene,
            FileOptions {
                up: UpAxis::Z,
                scale: 1.0,
            },
        );
        assert_eq!(scene.meshes[0].mesh.positions[0], Vec3::new(1.0, 3.0, -2.0));

        // Identity options are a no-op.
        let mut scene = LoadedScene {
            meshes: vec![MeshData {
                name: "part".into(),
                mesh: probe(),
                material: None,
            }],
            materials: Vec::new(),
        };
        apply_loaded(&mut scene, FileOptions::default());
        assert_eq!(scene.meshes[0].mesh.positions[0], Vec3::new(1.0, 2.0, 3.0));
    }
}
