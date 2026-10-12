//! CPU-side geometry preparation: turns scene [`Mesh`]es into GPU vertex
//! and index data a wgpu backend uploads verbatim.
//!
//! Nothing here touches a GPU device, so the whole module is unit-testable
//! headless. [`MeshBuffers::build`] is the vertex-soup builder; the
//! [`build_scene`] convenience walks a whole [`Scene`].

use crate::gpu_types::GpuVertex;
use glam::{Mat4, Vec2, Vec3};
use openforge_core::math::Aabb;
use openforge_core::mesh::Mesh;
use openforge_core::scene::Scene;
use openforge_core::selection::Selection;
use openforge_core::NodeId;
use std::collections::HashMap;

/// Base color applied to nodes that are not selected.
pub const DEFAULT_BASE_COLOR: [f32; 4] = [0.72, 0.76, 0.82, 1.0];
/// Highlight tint applied to selected nodes.
pub const SELECTION_COLOR: [f32; 4] = [1.0, 0.62, 0.15, 1.0];

/// Prepared CPU-side geometry for one mesh: a vertex buffer plus an index
/// buffer ready for GPU upload.
#[derive(Clone, PartialEq, Debug, Default)]
pub struct MeshBuffers {
    /// Deduplicated vertices (shared at seams: position/normal/uv triples).
    pub vertices: Vec<GpuVertex>,
    /// Triangle list indices into `vertices`.
    pub indices: Vec<u32>,
}

impl MeshBuffers {
    /// True when there is nothing to draw.
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    /// World-space AABB over the prepared vertices (empty when there are none).
    pub fn bounds(&self) -> Aabb {
        let mut b = Aabb::default();
        for v in &self.vertices {
            b.expand_point(Vec3::from(v.position));
        }
        b
    }

    /// Build GPU geometry for `mesh`, transformed into world space by
    /// `world` and colored with `base_color`.
    ///
    /// Faces are fan-triangulated; vertices are deduplicated by their
    /// (position, normal, uv) corner triple, so a quad authored with shared
    /// corners stays four vertices. Authored normals are transformed by the
    /// inverse-transpose of `world` (correct under non-uniform scale); faces
    /// without usable normals get flat per-face normals computed from the
    /// transformed positions. Invalid faces (out-of-range indices, fewer
    /// than three corners) are skipped rather than panicking.
    pub fn build(mesh: &Mesh, world: Mat4, base_color: [f32; 4]) -> MeshBuffers {
        if mesh.is_empty() {
            return MeshBuffers::default();
        }
        // Inverse-transpose for normals; fall back to identity if `world` is
        // singular (its inverse would be NaN).
        let normal_mat = if world.determinant().abs() > 1e-12 {
            world.inverse().transpose()
        } else {
            Mat4::IDENTITY
        };

        let mut out = MeshBuffers::default();
        let mut corner_map: HashMap<(u32, u32, u32), u32> = HashMap::new();

        for face in &mesh.faces {
            let cs = &face.corners;
            if cs.len() < 3 {
                continue;
            }
            if cs.iter().any(|c| c.position as usize >= mesh.positions.len()) {
                continue;
            }
            // Flat fallback: Newell normal over the face's transformed corners.
            let mut flat = Vec3::ZERO;
            for i in 0..cs.len() {
                let a = world.transform_point3(mesh.positions[cs[i].position as usize]);
                let b =
                    world.transform_point3(mesh.positions[cs[(i + 1) % cs.len()].position as usize]);
                flat += a.cross(b);
            }
            let flat = flat.normalize_or_zero();

            for i in 1..cs.len() - 1 {
                for &c in &[cs[0], cs[i], cs[i + 1]] {
                    let key = (c.position, c.normal, c.uv);
                    let idx = *corner_map.entry(key).or_insert_with(|| {
                        let position = world.transform_point3(mesh.positions[c.position as usize]);
                        let normal = match mesh.normals.get(c.normal as usize) {
                            Some(n) => normal_mat.transform_vector3(*n).normalize_or_zero(),
                            None => flat,
                        };
                        let uv = mesh.uvs.get(c.uv as usize).copied().unwrap_or(Vec2::ZERO);
                        out.vertices.push(GpuVertex::new(position, normal, uv.to_array(), base_color));
                        (out.vertices.len() - 1) as u32
                    });
                    out.indices.push(idx);
                }
            }
        }
        out
    }
}

/// Build buffers for every visible mesh node in `scene`, tinting selected
/// nodes with [`SELECTION_COLOR`] and everything else with
/// [`DEFAULT_BASE_COLOR`]. Returns the buffers plus the combined world-space
/// bounds of all built geometry (empty bounds when nothing is visible).
pub fn build_scene(scene: &mut Scene, selection: &Selection) -> (Vec<MeshBuffers>, Aabb) {
    let ids: Vec<NodeId> = scene
        .iter()
        .filter(|n| n.is_mesh() && n.visible)
        .map(|n| n.id)
        .collect();
    let mut buffers = Vec::with_capacity(ids.len());
    let mut bounds = Aabb::default();
    for id in ids {
        let Ok(world) = scene.world_matrix(id) else { continue };
        let Ok(node) = scene.get(id) else { continue };
        let Some(mesh) = node.mesh() else { continue };
        let color = if selection.is_node_selected(id) {
            SELECTION_COLOR
        } else {
            DEFAULT_BASE_COLOR
        };
        let buf = MeshBuffers::build(mesh, world, color);
        bounds.merge(&buf.bounds());
        buffers.push(buf);
    }
    (buffers, bounds)
}

#[cfg(test)]
mod tests {
    use super::*;
    use openforge_core::mesh::{Corner, Face};

    /// A unit quad in the XY plane (facing +Z) with shared corners.
    fn unit_quad() -> Mesh {
        let mut m = Mesh::new();
        m.add_vertex(Vec3::new(0.0, 0.0, 0.0));
        m.add_vertex(Vec3::new(1.0, 0.0, 0.0));
        m.add_vertex(Vec3::new(1.0, 1.0, 0.0));
        m.add_vertex(Vec3::new(0.0, 1.0, 0.0));
        m.add_face(&[0, 1, 2, 3]);
        m
    }

    #[test]
    fn quad_builds_four_verts_and_six_indices() {
        let buf = MeshBuffers::build(&unit_quad(), Mat4::IDENTITY, DEFAULT_BASE_COLOR);
        assert_eq!(buf.vertices.len(), 4);
        assert_eq!(buf.indices, vec![0, 1, 2, 0, 2, 3]);
        assert!(!buf.is_empty());
    }

    #[test]
    fn quad_vertex_attributes_are_filled() {
        let color = [1.0, 0.0, 0.0, 1.0];
        let buf = MeshBuffers::build(&unit_quad(), Mat4::IDENTITY, color);
        for v in &buf.vertices {
            assert!((Vec3::from(v.normal).length() - 1.0).abs() < 1e-5);
            assert_eq!(v.color, color);
            assert_eq!(v.uv, [0.0, 0.0]); // quad authored with uv slot 0
        }
        // Corners were deduplicated down to the quad's four positions.
        let mut positions: Vec<[f32; 3]> = buf.vertices.iter().map(|v| v.position).collect();
        positions.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(positions[0], [0.0, 0.0, 0.0]);
        assert_eq!(positions[3], [1.0, 1.0, 0.0]);
    }

    #[test]
    fn build_applies_world_transform_to_positions() {
        let world = Mat4::from_scale_rotation_translation(
            Vec3::splat(2.0),
            glam::Quat::from_rotation_z(std::f32::consts::FRAC_PI_2),
            Vec3::new(10.0, 0.0, 0.0),
        );
        let mesh = unit_quad();
        let buf = MeshBuffers::build(&mesh, world, DEFAULT_BASE_COLOR);
        for (i, v) in buf.vertices.iter().enumerate() {
            // Quad corner order: v0..v3.
            let expected = world.transform_point3(mesh.positions[i]);
            assert!((Vec3::from(v.position) - expected).length() < 1e-5);
        }
    }

    #[test]
    fn normals_use_inverse_transpose_under_non_uniform_scale() {
        let mut mesh = unit_quad();
        // A diagonal (non-axis) normal makes a naive scale-transform wrong.
        let n = Vec3::new(1.0, 1.0, 0.0).normalize();
        mesh.normals = vec![n]; // corners all reference normal slot 0
        let world = Mat4::from_scale(Vec3::new(2.0, 1.0, 1.0));
        let buf = MeshBuffers::build(&mesh, world, DEFAULT_BASE_COLOR);
        // inv-transpose of diag(2,1,1) is diag(0.5,1,1): n' ∝ (0.3536, 0.7071, 0).
        let expected = Vec3::new(0.35355339, 0.70710678, 0.0).normalize();
        for v in &buf.vertices {
            assert!((Vec3::from(v.normal) - expected).length() < 1e-5);
        }
    }

    #[test]
    fn build_uses_flat_normals_when_mesh_has_none() {
        let mut mesh = unit_quad();
        mesh.normals = Vec::new(); // authored normals stripped
        let buf = MeshBuffers::build(&mesh, Mat4::IDENTITY, DEFAULT_BASE_COLOR);
        for v in &buf.vertices {
            // Flat normal of an XY-plane quad facing +Z.
            assert!((Vec3::from(v.normal) - Vec3::Z).length() < 1e-5);
        }
    }

    #[test]
    fn build_skips_invalid_faces_without_panicking() {
        let mut mesh = unit_quad();
        // Out-of-range position index: skipped.
        mesh.faces.push(Face {
            corners: vec![
                Corner { position: 99, normal: 0, uv: 0 },
                Corner { position: 100, normal: 0, uv: 0 },
                Corner { position: 101, normal: 0, uv: 0 },
            ],
            material: 0,
        });
        // Fewer than three corners: skipped.
        mesh.faces.push(Face {
            corners: vec![Corner { position: 0, normal: 0, uv: 0 }, Corner { position: 1, normal: 0, uv: 0 }],
            material: 0,
        });
        // Out-of-range uv index: falls back to (0, 0) but still renders.
        mesh.faces.push(Face {
            corners: vec![
                Corner { position: 0, normal: 0, uv: 42 },
                Corner { position: 1, normal: 0, uv: 42 },
                Corner { position: 2, normal: 0, uv: 42 },
            ],
            material: 0,
        });
        let buf = MeshBuffers::build(&mesh, Mat4::IDENTITY, DEFAULT_BASE_COLOR);
        // 4 quad verts + the extra tri (new uv keys, reused positions);
        // indices: 6 from the quad + 3 from the extra tri.
        assert_eq!(buf.vertices.len(), 7);
        assert_eq!(buf.indices.len(), 9);
    }

    #[test]
    fn build_empty_mesh_is_empty() {
        let buf = MeshBuffers::build(&Mesh::new(), Mat4::IDENTITY, DEFAULT_BASE_COLOR);
        assert!(buf.is_empty());
        assert!(buf.vertices.is_empty());
        assert!(buf.bounds().is_empty());
    }

    #[test]
    fn build_scene_two_meshes_returns_two_buffers_and_combined_bounds() {
        let mut scene = Scene::new();
        let a = scene.create_mesh("a", unit_quad());
        let mut b = unit_quad();
        // Give the second quad an authored smooth normal path too.
        b.compute_smooth_normals();
        let b = scene.create_mesh("b", b);
        scene.get_mut(b).unwrap().set_translation(Vec3::new(5.0, 0.0, 0.0));
        let selection = Selection::new();

        let (buffers, bounds) = build_scene(&mut scene, &selection);
        assert_eq!(buffers.len(), 2);
        assert!(!bounds.is_empty());
        // Quad a spans [0,1]², quad b (translated +5 in x) spans [5,6]×[1].
        assert!((bounds.min - Vec3::ZERO).length() < 1e-5);
        assert!((bounds.max - Vec3::new(6.0, 1.0, 0.0)).length() < 1e-5);
        assert!(a != b);
    }

    #[test]
    fn build_scene_respects_visibility_and_selection_color() {
        let mut scene = Scene::new();
        let a = scene.create_mesh("a", unit_quad());
        let b = scene.create_mesh("b", unit_quad());
        scene.get_mut(a).unwrap().visible = false; // hidden: skipped

        let mut selection = Selection::new();
        selection.select_node(b);

        let (buffers, bounds) = build_scene(&mut scene, &selection);
        assert_eq!(buffers.len(), 1);
        for v in &buffers[0].vertices {
            assert_eq!(v.color, SELECTION_COLOR);
        }
        assert!(!bounds.is_empty());
    }

    #[test]
    fn build_scene_on_empty_scene_is_empty() {
        let mut scene = Scene::new();
        let (buffers, bounds) = build_scene(&mut scene, &Selection::new());
        assert!(buffers.is_empty());
        assert!(bounds.is_empty());
    }

    #[test]
    fn build_scene_skips_empty_mesh_nodes() {
        let mut scene = Scene::new();
        scene.create_mesh("empty", Mesh::new());
        let (buffers, bounds) = build_scene(&mut scene, &Selection::new());
        assert_eq!(buffers.len(), 1);
        assert!(buffers[0].is_empty());
        assert!(bounds.is_empty());
    }
}
