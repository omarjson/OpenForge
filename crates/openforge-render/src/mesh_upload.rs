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

    /// Append `other`'s geometry to this buffer, rebasing its indices so the
    /// merged buffer stays drawable as one `draw_indexed` call.
    pub fn append(&mut self, other: &MeshBuffers) {
        let base = self.vertices.len() as u32;
        self.vertices.extend_from_slice(&other.vertices);
        self.indices.extend(other.indices.iter().map(|i| i + base));
    }

    /// Build GPU geometry for `mesh`, transformed into world space by
    /// `world` and colored with `base_color`.
    ///
    /// Faces are fan-triangulated; vertices are deduplicated by their
    /// (position, normal, uv, colour) corner quadruple, so a quad authored with
    /// shared corners stays four vertices. Authored normals are transformed by
    /// the inverse-transpose of `world` (correct under non-uniform scale);
    /// faces without usable normals get flat per-face normals computed from the
    /// transformed positions. Invalid faces (out-of-range indices, fewer
    /// than three corners) are skipped rather than panicking.
    pub fn build(mesh: &Mesh, world: Mat4, base_color: [f32; 4]) -> MeshBuffers {
        Self::build_with_face_colors(mesh, world, |_| base_color)
    }

    /// [`MeshBuffers::build`] with the colour resolved per face: `face_color(i)`
    /// returns the RGBA for face `i`.
    ///
    /// The colour joins the dedup key, so corners shared by faces of different
    /// colours become separate vertices and each face keeps its shade, while
    /// coincident corners of the same colour still collapse.
    pub fn build_with_face_colors(
        mesh: &Mesh,
        world: Mat4,
        mut face_color: impl FnMut(usize) -> [f32; 4],
    ) -> MeshBuffers {
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
        let mut corner_map: HashMap<(u32, u32, u32, [u32; 4]), u32> = HashMap::new();

        for (face_idx, face) in mesh.faces.iter().enumerate() {
            let cs = &face.corners;
            if cs.len() < 3 {
                continue;
            }
            if cs.iter().any(|c| c.position as usize >= mesh.positions.len()) {
                continue;
            }
            let color = face_color(face_idx);
            let color_key = [
                color[0].to_bits(),
                color[1].to_bits(),
                color[2].to_bits(),
                color[3].to_bits(),
            ];
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
                    let key = (c.position, c.normal, c.uv, color_key);
                    let idx = *corner_map.entry(key).or_insert_with(|| {
                        let position = world.transform_point3(mesh.positions[c.position as usize]);
                        let normal = match mesh.normals.get(c.normal as usize) {
                            Some(n) => normal_mat.transform_vector3(*n).normalize_or_zero(),
                            None => flat,
                        };
                        let uv = mesh.uvs.get(c.uv as usize).copied().unwrap_or(Vec2::ZERO);
                        out.vertices.push(GpuVertex::new(position, normal, uv.to_array(), color));
                        (out.vertices.len() - 1) as u32
                    });
                    out.indices.push(idx);
                }
            }
        }
        out
    }
}

/// Per-node base colours for [`build_scene_with`].
///
/// The renderer knows nothing about materials — a caller that has a material
/// library resolves each node's colour and hands the result in. Nodes with no
/// entry keep the selection/default colours.
#[derive(Clone, Debug, Default)]
pub struct NodeColors {
    colors: HashMap<NodeId, [f32; 4]>,
}

impl NodeColors {
    pub fn new() -> Self {
        Self::default()
    }

    /// Set (or replace) a node's base colour, returning `self` for chaining.
    #[must_use]
    pub fn with(mut self, id: NodeId, color: [f32; 4]) -> Self {
        self.colors.insert(id, color);
        self
    }

    pub fn get(&self, id: NodeId) -> Option<[f32; 4]> {
        self.colors.get(&id).copied()
    }

    pub fn len(&self) -> usize {
        self.colors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.colors.is_empty()
    }
}

/// Per-node, per-face colours for [`build_scene_materials`]: the resolved RGBA
/// of each face, in face order.
///
/// Like [`NodeColors`] the renderer stays material-agnostic — a caller that has
/// a material library resolves each face (its material slot plus any selection
/// tint) and hands the list in. Nodes with no entry, and faces past the end of
/// an entry, keep the selection/default colours.
#[derive(Clone, Debug, Default)]
pub struct NodeMaterials {
    faces: HashMap<NodeId, Vec<[f32; 4]>>,
}

impl NodeMaterials {
    pub fn new() -> Self {
        Self::default()
    }

    /// Set (or replace) a node's per-face colours, returning `self` for
    /// chaining.
    #[must_use]
    pub fn with_faces(mut self, id: NodeId, colors: Vec<[f32; 4]>) -> Self {
        self.faces.insert(id, colors);
        self
    }

    /// The colour of one face, when the node has an entry that reaches it.
    pub fn face_color(&self, id: NodeId, face: usize) -> Option<[f32; 4]> {
        self.faces.get(&id).and_then(|v| v.get(face)).copied()
    }

    /// A node's whole per-face list, when it has one.
    pub fn faces(&self, id: NodeId) -> Option<&[[f32; 4]]> {
        self.faces.get(&id).map(Vec::as_slice)
    }

    pub fn len(&self) -> usize {
        self.faces.len()
    }

    pub fn is_empty(&self) -> bool {
        self.faces.is_empty()
    }
}

/// Build buffers for every visible mesh node in `scene`, tinting selected
/// nodes with [`SELECTION_COLOR`] and everything else with
/// [`DEFAULT_BASE_COLOR`]. Returns the buffers plus the combined world-space
/// bounds of all built geometry (empty bounds when nothing is visible).
pub fn build_scene(scene: &mut Scene, selection: &Selection) -> (Vec<MeshBuffers>, Aabb) {
    build_scene_with(scene, selection, &NodeColors::default())
}

/// [`build_scene`] with a caller-supplied colour per node (e.g. resolved from a
/// material library). A node missing from `colors` falls back to the selection
/// tint when it is selected and [`DEFAULT_BASE_COLOR`] otherwise.
pub fn build_scene_with(
    scene: &mut Scene,
    selection: &Selection,
    colors: &NodeColors,
) -> (Vec<MeshBuffers>, Aabb) {
    build_scene_per_face(scene, selection, |id, _face| colors.get(id))
}

/// [`build_scene`] with a caller-supplied colour per face (e.g. each face's
/// material slot resolved through a library). A face missing from `materials`
/// falls back to the selection tint when its node is selected and
/// [`DEFAULT_BASE_COLOR`] otherwise.
pub fn build_scene_materials(
    scene: &mut Scene,
    selection: &Selection,
    materials: &NodeMaterials,
) -> (Vec<MeshBuffers>, Aabb) {
    build_scene_per_face(scene, selection, |id, face| materials.face_color(id, face))
}

/// Walk `scene`'s visible mesh nodes and build one [`MeshBuffers`] each,
/// resolving a colour per (node, face). A `None` falls back to the selection
/// tint for selected nodes and [`DEFAULT_BASE_COLOR`] otherwise.
fn build_scene_per_face(
    scene: &mut Scene,
    selection: &Selection,
    mut color_of: impl FnMut(NodeId, usize) -> Option<[f32; 4]>,
) -> (Vec<MeshBuffers>, Aabb) {
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
        let fallback = if selection.is_node_selected(id) {
            SELECTION_COLOR
        } else {
            DEFAULT_BASE_COLOR
        };
        let colors: Vec<[f32; 4]> = (0..mesh.faces.len())
            .map(|face| color_of(id, face).unwrap_or(fallback))
            .collect();
        let buf = MeshBuffers::build_with_face_colors(mesh, world, |face| colors[face]);
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
        // inv-transpose of diag(2,1,1) is diag(0.5,1,1), which maps n = (1,1,0)
        // to (0.5, 1, 0), i.e. (1,2,0) once normalized.
        let expected = Vec3::new(0.5, 1.0, 0.0).normalize();
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
    fn append_rebases_indices_and_keeps_geometry() {
        let a = MeshBuffers::build(&unit_quad(), Mat4::IDENTITY, DEFAULT_BASE_COLOR);
        let b = MeshBuffers::build(&unit_quad(), Mat4::from_translation(Vec3::new(3.0, 0.0, 0.0)), SELECTION_COLOR);
        let mut merged = MeshBuffers::default();
        merged.append(&a);
        merged.append(&b);
        assert_eq!(merged.vertices.len(), a.vertices.len() + b.vertices.len());
        assert_eq!(merged.indices.len(), a.indices.len() + b.indices.len());
        // The first quad's indices are untouched, the second's are offset by
        // 4 (the quad's vertex count) and stay in range.
        assert_eq!(&merged.indices[..6], &a.indices[..]);
        assert!(merged.indices[6..].iter().all(|&i| i >= 4));
        assert!(merged.indices.iter().all(|&i| (i as usize) < merged.vertices.len()));
        // Vertex colors survive the merge.
        assert_eq!(merged.vertices[4].color, SELECTION_COLOR);
        assert!(!merged.is_empty());
    }

    #[test]
    fn append_into_empty_buffer_is_a_copy() {
        let a = MeshBuffers::build(&unit_quad(), Mat4::IDENTITY, DEFAULT_BASE_COLOR);
        let mut merged = MeshBuffers::default();
        merged.append(&a);
        assert_eq!(merged, a);
        // Appending nothing changes nothing.
        let before = merged.clone();
        merged.append(&MeshBuffers::default());
        assert_eq!(merged, before);
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
    fn node_colors_are_addressable_and_chain() {
        let mut colors = NodeColors::new();
        assert!(colors.is_empty());
        colors = colors.with(NodeId(1), [1.0, 0.0, 0.0, 1.0]).with(NodeId(2), [0.0, 1.0, 0.0, 1.0]);
        assert_eq!(colors.len(), 2);
        assert!(!colors.is_empty());
        assert_eq!(colors.get(NodeId(2)), Some([0.0, 1.0, 0.0, 1.0]));
        assert_eq!(colors.get(NodeId(3)), None);
        // Overwriting keeps one entry per node.
        let colors = colors.with(NodeId(1), [0.0, 0.0, 1.0, 1.0]);
        assert_eq!(colors.len(), 2);
        assert_eq!(colors.get(NodeId(1)), Some([0.0, 0.0, 1.0, 1.0]));
    }

    #[test]
    fn build_scene_with_uses_the_supplied_colour() {
        let mut scene = Scene::new();
        let id = scene.create_mesh("quad", unit_quad());
        let black = [0.0, 0.0, 0.0, 1.0];
        let (buffers, _) = build_scene_with(
            &mut scene,
            &Selection::new(),
            &NodeColors::new().with(id, black),
        );
        assert!(buffers.iter().flat_map(|b| &b.vertices).all(|v| v.color == black));
    }

    #[test]
    fn build_scene_with_falls_back_for_nodes_it_was_not_given() {
        let mut scene = Scene::new();
        let plain = scene.create_mesh("plain", unit_quad());
        let picked = scene.create_mesh("picked", unit_quad());
        let mut selection = Selection::new();
        selection.select_node(picked);

        // An empty table: the old behaviour, selection tint and default grey.
        // The scene iterates in hash order, so count the two cases rather than
        // assuming which buffer is which.
        let (buffers, _) = build_scene_with(&mut scene, &selection, &NodeColors::new());
        let all = |color: [f32; 4]| {
            buffers
                .iter()
                .filter(|b| !b.vertices.is_empty() && b.vertices.iter().all(|v| v.color == color))
                .count()
        };
        assert_eq!(all(DEFAULT_BASE_COLOR), 1, "the unselected node");
        assert_eq!(all(SELECTION_COLOR), 1, "the selected node");

        // A table entry wins over both, including for the selected node.
        let gold = [0.9, 0.7, 0.3, 1.0];
        let colors = NodeColors::new().with(picked, gold).with(plain, gold);
        let (buffers, _) = build_scene_with(&mut scene, &selection, &colors);
        assert!(buffers.iter().flat_map(|b| &b.vertices).all(|v| v.color == gold));
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

    /// A quad plus a triangle that reuses its first three corners, so the two
    /// faces share every corner of the triangle.
    fn quad_and_triangle() -> Mesh {
        let mut m = unit_quad();
        m.add_face(&[0, 1, 2]);
        m
    }

    const RED: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
    const BLUE: [f32; 4] = [0.0, 0.0, 1.0, 1.0];

    #[test]
    fn face_colors_reach_each_face() {
        let mesh = quad_and_triangle();
        let buf = MeshBuffers::build_with_face_colors(&mesh, Mat4::IDENTITY, |face| {
            if face == 0 {
                RED
            } else {
                BLUE
            }
        });
        // Faces of different colours may not share vertices, so the quad's four
        // corners come first and the triangle's three follow.
        assert_eq!(buf.vertices.len(), 7);
        assert_eq!(buf.indices.len(), 9);
        assert!(buf.vertices[..4].iter().all(|v| v.color == RED));
        assert!(buf.vertices[4..].iter().all(|v| v.color == BLUE));
    }

    #[test]
    fn one_face_color_still_deduplicates_shared_corners() {
        let mesh = quad_and_triangle();
        let buf = MeshBuffers::build_with_face_colors(&mesh, Mat4::IDENTITY, |_| SELECTION_COLOR);
        assert_eq!(buf.vertices.len(), 4, "one colour: shared corners collapse");
        assert_eq!(buf.indices.len(), 9);
    }

    #[test]
    fn node_materials_are_addressable_and_chain() {
        let mut mats = NodeMaterials::new();
        assert!(mats.is_empty());
        mats = mats
            .with_faces(NodeId(1), vec![RED, BLUE])
            .with_faces(NodeId(2), vec![BLUE]);
        assert_eq!(mats.len(), 2);
        assert!(!mats.is_empty());
        assert_eq!(mats.face_color(NodeId(1), 1), Some(BLUE));
        assert_eq!(mats.face_color(NodeId(1), 2), None, "past the end");
        assert_eq!(mats.face_color(NodeId(3), 0), None, "unknown node");
        assert_eq!(mats.faces(NodeId(2)), Some(&[BLUE][..]));
        // Overwriting keeps one entry per node.
        let mats = mats.with_faces(NodeId(1), vec![BLUE]);
        assert_eq!(mats.len(), 2);
        assert_eq!(mats.face_color(NodeId(1), 0), Some(BLUE));
        assert_eq!(mats.face_color(NodeId(1), 1), None);
    }

    #[test]
    fn build_scene_materials_resolves_each_face() {
        let mut scene = Scene::new();
        let id = scene.create_mesh("two", quad_and_triangle());
        let (buffers, bounds) = build_scene_materials(
            &mut scene,
            &Selection::new(),
            &NodeMaterials::new().with_faces(id, vec![RED, BLUE]),
        );
        let colors: Vec<[f32; 4]> = buffers[0].vertices.iter().map(|v| v.color).collect();
        assert_eq!(colors.iter().filter(|c| **c == RED).count(), 4, "the quad");
        assert_eq!(colors.iter().filter(|c| **c == BLUE).count(), 3, "the triangle");
        assert!(!bounds.is_empty());
    }

    #[test]
    fn build_scene_materials_falls_back_per_face_and_per_node() {
        let mut scene = Scene::new();
        let id = scene.create_mesh("two", quad_and_triangle());
        // Only face 0 is in the table: face 1 keeps the default colour.
        let (buffers, _) = build_scene_materials(
            &mut scene,
            &Selection::new(),
            &NodeMaterials::new().with_faces(id, vec![RED]),
        );
        let colors: Vec<[f32; 4]> = buffers[0].vertices.iter().map(|v| v.color).collect();
        assert_eq!(colors.iter().filter(|c| **c == RED).count(), 4);
        assert_eq!(
            colors.iter().filter(|c| **c == DEFAULT_BASE_COLOR).count(),
            3
        );

        // A selected node with no entry at all keeps the selection tint.
        let mut selection = Selection::new();
        selection.select_node(id);
        let (buffers, _) = build_scene_materials(&mut scene, &selection, &NodeMaterials::new());
        assert!(buffers[0].vertices.iter().all(|v| v.color == SELECTION_COLOR));
    }
}
