//! CPU-side preparation for ID-buffer picking.
//!
//! A pick pass draws the scene a second time into a single-channel integer
//! target, writing one *element id* per texel instead of a shaded color. The
//! ids are then read back around the cursor, which makes sub-object picking
//! agree with the drawn frame exactly: the pass runs with the same camera,
//! the same depth buffer semantics and the same back-face culling as the
//! visible pass, so a click can never hit geometry the image does not show,
//! and an occluded vertex stays unpicked.
//!
//! Nothing here touches a GPU: the geometry, the element table and the patch
//! resolution are plain data, so the whole module is unit-testable headless.
//! [`crate::gpu::GpuRenderer::pick`] is the wgpu front end.

use glam::Vec3;
use openforge_core::scene::Scene;
use openforge_core::NodeId;

/// One vertex of the ID pass: a world-space position plus the element id its
/// fragments report.
///
/// The layout matches the pick shader's `@location`s — position = 0,
/// id = 1 — and is uploaded verbatim, so it is `#[repr(C)]` and 16 bytes.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PickVertex {
    /// World-space position.
    pub position: [f32; 3],
    /// The element id a fragment covered by this vertex reports.
    pub id: u32,
}

impl PickVertex {
    /// A vertex from a glam position and an element id.
    pub fn new(position: Vec3, id: u32) -> Self {
        Self { position: position.to_array(), id }
    }
}

/// What a nonzero texel of an ID pass refers to.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PickElement {
    /// One face of a node's mesh, as the mesh's own face index.
    Face { node: NodeId, face: u32 },
    /// One vertex of a node's mesh, as an index into `mesh.positions`.
    Vertex { node: NodeId, vertex: u32 },
}

impl PickElement {
    /// The node this element belongs to.
    pub fn node(&self) -> NodeId {
        match *self {
            PickElement::Face { node, .. } | PickElement::Vertex { node, .. } => node,
        }
    }

    /// The element's index inside its node's mesh — a face index or a vertex
    /// index, depending on the variant.
    pub fn index(&self) -> u32 {
        match *self {
            PickElement::Face { face, .. } => face,
            PickElement::Vertex { vertex, .. } => vertex,
        }
    }
}

/// The ID pass for a whole scene: face triangles, vertex points, and the
/// element table their ids index into.
///
/// Vertices are deliberately *not* deduplicated and triangles are a plain
/// list (no index buffer): the pass runs once per click, and keeping every
/// face's ids on its own vertices means no texture, material or shading
/// difference can ever leak into a pick.
#[derive(Clone, PartialEq, Debug, Default)]
pub struct PickGeometry {
    /// Triangle-list vertices, three per triangle.
    pub triangles: Vec<PickVertex>,
    /// Point-list vertices, one per mesh vertex.
    pub points: Vec<PickVertex>,
    /// The element table: id `i + 1` is `elements[i]` (`0` is "nothing").
    pub elements: Vec<PickElement>,
}

impl PickGeometry {
    /// Build the ID geometry for every visible, unfrozen mesh node of
    /// `scene`.
    ///
    /// Faces are fan-triangulated like the visible pass and invalid ones
    /// (fewer than three corners, out-of-range positions) are skipped, so the
    /// triangles are exactly the ones the frame drew. A frozen node is left
    /// out even though it renders: it is not selectable, and the CPU pickers
    /// already let a click fall through it.
    pub fn build(scene: &mut Scene) -> Self {
        let ids: Vec<NodeId> = scene
            .iter()
            .filter(|n| n.is_mesh() && n.visible && !n.frozen)
            .map(|n| n.id)
            .collect();
        let mut geo = Self::default();
        for id in ids {
            let Ok(world) = scene.world_matrix(id) else { continue };
            let Ok(node) = scene.get(id) else { continue };
            let Some(mesh) = node.mesh() else { continue };
            for (fi, face) in mesh.faces.iter().enumerate() {
                let cs = &face.corners;
                if cs.len() < 3 {
                    continue;
                }
                if cs.iter().any(|c| c.position as usize >= mesh.positions.len()) {
                    continue;
                }
                let element = geo.push_element(PickElement::Face { node: id, face: fi as u32 });
                for i in 1..cs.len() - 1 {
                    for &c in &[cs[0], cs[i], cs[i + 1]] {
                        let p = world.transform_point3(mesh.positions[c.position as usize]);
                        geo.triangles.push(PickVertex::new(p, element));
                    }
                }
            }
            for vi in 0..mesh.positions.len() {
                let element = geo.push_element(PickElement::Vertex { node: id, vertex: vi as u32 });
                let p = world.transform_point3(mesh.positions[vi]);
                geo.points.push(PickVertex::new(p, element));
            }
        }
        geo
    }

    /// The element an id names; `None` for `0` (the clear value) and for ids
    /// past the table.
    pub fn element(&self, id: u32) -> Option<PickElement> {
        if id == 0 {
            return None;
        }
        self.elements.get(id as usize - 1).copied()
    }

    /// True when there is nothing to pick.
    pub fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }

    /// Every element found in a read-back ID patch, for rectangle selection.
    ///
    /// Unlike [`resolve_patch`], which answers "what is under the cursor", a
    /// rectangle asks "what is anywhere under it": distances do not matter and
    /// neither does where in the patch an id showed up. Ids are taken once
    /// each and answered in element-table order — ascending ids, the order
    /// [`build`] wrote them in — which makes the result deterministic whatever
    /// the pass rasterized and in whatever order the patch holds the ids.
    ///
    /// `patch` is row-major and tightly packed: `width * height` ids, `0` for
    /// nothing. A zero-sized or short patch resolves to nothing, and so does
    /// an id past the table (`0` is the clear value, never an element).
    ///
    /// [`build`]: PickGeometry::build
    pub fn elements_in_patch(&self, patch: &[u32], width: u32, height: u32) -> Vec<PickElement> {
        let (w, h) = (width as usize, height as usize);
        if w == 0 || h == 0 || patch.len() < w * h {
            return Vec::new();
        }
        // One slot per id: index 0 is the clear value, nothing marks it.
        let mut seen = vec![false; self.elements.len() + 1];
        let mut ids: Vec<u32> = Vec::new();
        for &id in &patch[..w * h] {
            let slot = id as usize;
            if slot == 0 || slot >= seen.len() || seen[slot] {
                continue;
            }
            seen[slot] = true;
            ids.push(id);
        }
        ids.sort_unstable();
        ids.into_iter().filter_map(|id| self.element(id)).collect()
    }

    /// Append an element and return its 1-based id, so `0` stays reserved for
    /// "no element here".
    fn push_element(&mut self, element: PickElement) -> u32 {
        self.elements.push(element);
        self.elements.len() as u32
    }
}

/// One element found in a read-back ID patch: its id and how far it was from
/// the patch's centre.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PickHit {
    /// The element id, as written by the pass.
    pub id: u32,
    /// Squared Euclidean distance, in pixels, from the patch's centre pixel.
    pub distance: u32,
}

/// Resolve a read-back ID patch into hits, nearest first.
///
/// `patch` is row-major, `width * height` ids; `0` is the pass's clear value
/// and never a hit. The same id found on several pixels keeps its nearest one.
/// `centre` is the cursor's pixel *inside the patch* — the patch is clipped to
/// the viewport, so the cursor is not always at the patch's middle.
///
/// Ties break by id so a given patch always resolves to the same list.
pub fn resolve_patch(patch: &[u32], width: u32, height: u32, centre: [u32; 2]) -> Vec<PickHit> {
    let (w, h) = (width as usize, height as usize);
    if w == 0 || h == 0 || patch.len() < w * h {
        return Vec::new();
    }
    let cx = centre[0].min(width - 1) as i64;
    let cy = centre[1].min(height - 1) as i64;
    let mut hits: Vec<PickHit> = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let id = patch[y * w + x];
            if id == 0 {
                continue;
            }
            let dx = x as i64 - cx;
            let dy = y as i64 - cy;
            let distance = (dx * dx + dy * dy) as u32;
            match hits.iter_mut().find(|hit| hit.id == id) {
                Some(hit) => hit.distance = hit.distance.min(distance),
                None => hits.push(PickHit { id, distance }),
            }
        }
    }
    hits.sort_by_key(|hit| (hit.distance, hit.id));
    hits
}

#[cfg(test)]
mod tests {
    use super::*;
    use openforge_core::mesh::Mesh;
    use openforge_core::selection::Selection;

    /// A unit quad in the XY plane facing +Z, sharing its four corners.
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
    fn pick_vertex_is_sixteen_bytes_and_plain_data() {
        assert_eq!(std::mem::size_of::<PickVertex>(), 16);
        let v = PickVertex::new(Vec3::new(1.0, 2.0, 3.0), 7);
        assert_eq!(v.position, [1.0, 2.0, 3.0]);
        assert_eq!(v.id, 7);
        // Plain old data: it uploads as bytes, with no hidden padding.
        assert_eq!(bytemuck::bytes_of(&v).len(), 16);
        assert_eq!(PickVertex::default().id, 0);
    }

    #[test]
    fn a_quad_gives_two_triangles_four_points_and_five_elements() {
        let mut scene = Scene::new();
        let node = scene.create_mesh("quad", unit_quad());
        let geo = PickGeometry::build(&mut scene);

        // Fan-triangulated: two triangles of three vertices.
        assert_eq!(geo.triangles.len(), 6);
        // One point per mesh position.
        assert_eq!(geo.points.len(), 4);
        // One face element plus one vertex element per position.
        assert_eq!(geo.elements.len(), 5);
        assert_eq!(geo.elements[0], PickElement::Face { node, face: 0 });
        for (vi, element) in geo.elements[1..].iter().enumerate() {
            assert_eq!(*element, PickElement::Vertex { node, vertex: vi as u32 });
        }
        assert!(!geo.is_empty());
    }

    #[test]
    fn every_vertex_carries_its_elements_one_based_id() {
        let mut scene = Scene::new();
        scene.create_mesh("quad", unit_quad());
        let geo = PickGeometry::build(&mut scene);

        assert!(geo.triangles.iter().all(|v| v.id == 1), "the only face");
        for (i, v) in geo.points.iter().enumerate() {
            assert_eq!(v.id, i as u32 + 2, "vertices follow the face, from id 2");
        }
        for v in geo.triangles.iter().chain(&geo.points) {
            assert!(v.id > 0, "zero is the clear value, never an element");
            assert!(geo.element(v.id).is_some());
        }
        assert_eq!(geo.element(0), None);
        assert_eq!(geo.element(99), None);
    }

    #[test]
    fn geometry_is_transformed_into_world_space() {
        let mut scene = Scene::new();
        let node = scene.create_mesh("quad", unit_quad());
        scene.get_mut(node).unwrap().set_translation(Vec3::new(5.0, -2.0, 1.0));
        let geo = PickGeometry::build(&mut scene);

        // Quad corner (1, 1, 0) lands at (6, -1, 1).
        assert!(geo.points.iter().any(|v| v.position == [6.0, -1.0, 1.0]));
        assert!(geo.triangles.iter().all(|v| v.position[0] >= 5.0));
    }

    #[test]
    fn hidden_and_frozen_nodes_contribute_nothing() {
        let mut scene = Scene::new();
        let hidden = scene.create_mesh("hidden", unit_quad());
        let frozen = scene.create_mesh("frozen", unit_quad());
        let visible = scene.create_mesh("visible", unit_quad());
        scene.get_mut(hidden).unwrap().visible = false;
        scene.get_mut(frozen).unwrap().frozen = true;

        let geo = PickGeometry::build(&mut scene);
        assert_eq!(geo.elements.len(), 5, "only the visible node");
        assert!(geo.elements.iter().all(|e| e.node() == visible));
    }

    #[test]
    fn invalid_faces_are_skipped_like_the_visible_pass_skips_them() {
        let mut scene = Scene::new();
        let node = scene.create_mesh("quad", unit_quad());
        {
            let mesh = scene.get_mut(node).unwrap().mesh_mut().unwrap();
            // Out-of-range position: not drawn, so not picked either.
            mesh.add_face(&[0, 1, 99]);
        }
        let geo = PickGeometry::build(&mut scene);
        // Still one face element and four vertices: the bogus face added none.
        assert_eq!(geo.elements.len(), 5);
        assert_eq!(geo.triangles.len(), 6);
    }

    #[test]
    fn an_ngon_is_fan_triangulated_with_one_id_for_the_face() {
        let mut scene = Scene::new();
        let mut mesh = Mesh::new();
        for i in 0..5 {
            let a = i as f32 / 5.0 * std::f32::consts::TAU;
            mesh.add_vertex(Vec3::new(a.cos(), a.sin(), 0.0));
        }
        mesh.add_face(&[0, 1, 2, 3, 4]);
        scene.create_mesh("pentagon", mesh);

        let geo = PickGeometry::build(&mut scene);
        assert_eq!(geo.triangles.len(), 9, "three fan triangles");
        assert!(geo.triangles.iter().all(|v| v.id == 1));
        assert_eq!(geo.elements.len(), 6, "one face + five vertices");
    }

    #[test]
    fn non_mesh_nodes_and_empty_scenes_pick_nothing() {
        let mut scene = Scene::new();
        assert!(PickGeometry::build(&mut scene).is_empty());
        scene.create_dummy("dummy");
        scene.create_group("group");
        let geo = PickGeometry::build(&mut scene);
        assert!(geo.is_empty());
        assert!(geo.triangles.is_empty() && geo.points.is_empty());
    }

    #[test]
    fn build_is_deterministic_for_one_node() {
        let mut scene = Scene::new();
        scene.create_mesh("quad", unit_quad());
        let a = PickGeometry::build(&mut scene);
        let b = PickGeometry::build(&mut scene);
        assert_eq!(a, b);
        // ... and the selection plays no part in it: picking never sees a tint.
        let mut selection = Selection::new();
        selection.select_node(NodeId(1));
        assert_eq!(PickGeometry::build(&mut scene), a);
    }

    #[test]
    fn an_empty_patch_resolves_to_nothing() {
        assert!(resolve_patch(&[], 0, 0, [0, 0]).is_empty());
        assert!(resolve_patch(&[], 3, 3, [1, 1]).is_empty(), "short buffer");
        assert!(resolve_patch(&[0; 9], 3, 3, [1, 1]).is_empty(), "all clear");
    }

    #[test]
    fn hits_are_nearest_first_and_deduplicated() {
        // 5x5, centre (2,2). Id 7 on the centre and further out; id 9 one
        // pixel up; id 4 at a corner.
        let mut patch = [0u32; 25];
        patch[2 * 5 + 2] = 7; // centre, distance 0
        patch[0] = 7; // recomputed to the nearest pixel (0), kept once
        patch[1 * 5 + 2] = 9; // distance 1
        patch[4 * 5 + 4] = 4; // distance 8
        let hits = resolve_patch(&patch, 5, 5, [2, 2]);
        assert_eq!(hits.len(), 3, "id 7 is one hit, not two");
        assert_eq!(hits[0], PickHit { id: 7, distance: 0 });
        assert_eq!(hits[1], PickHit { id: 9, distance: 1 });
        assert_eq!(hits[2], PickHit { id: 4, distance: 4 + 4 });
    }

    #[test]
    fn an_id_seen_twice_keeps_its_nearest_pixel() {
        let mut patch = [0u32; 25];
        patch[4 * 5 + 4] = 3; // further
        patch[2 * 5 + 3] = 3; // nearer, must win
        let hits = resolve_patch(&patch, 5, 5, [2, 2]);
        assert_eq!(hits, vec![PickHit { id: 3, distance: 1 }]);
    }

    #[test]
    fn ties_break_by_id() {
        let mut patch = [0u32; 25];
        patch[2 * 5 + 1] = 9; // left of the centre
        patch[2 * 5 + 3] = 4; // right of the centre, same distance
        let hits = resolve_patch(&patch, 5, 5, [2, 2]);
        assert_eq!(hits.iter().map(|h| h.id).collect::<Vec<_>>(), vec![4, 9]);
    }

    #[test]
    fn the_centre_is_a_patch_coordinate_not_the_middle() {
        // A patch clipped at the viewport's top-left corner: the cursor is
        // pixel (0, 0) of the patch.
        let mut patch = [0u32; 4];
        patch[3] = 2; // bottom-right, distance 1 + 1
        patch[1] = 5; // top-right, distance 0 + 1
        let hits = resolve_patch(&patch, 2, 2, [0, 0]);
        assert_eq!(hits[0].id, 5);
        assert_eq!(hits[0].distance, 1);
        assert_eq!(hits[1], PickHit { id: 2, distance: 2 });

        // An out-of-bounds centre is clamped, not trusted.
        let hits = resolve_patch(&patch, 2, 2, [9, 9]);
        assert_eq!(hits[0].id, 2);
    }

    #[test]
    fn a_rectangle_finds_every_element_under_it_once() {
        let mut scene = Scene::new();
        let node = scene.create_mesh("quad", unit_quad());
        let geo = PickGeometry::build(&mut scene);

        // Row-major, so the vertex id is seen before the face id: the result
        // still comes back in table order (face first, it was pushed first).
        let patch = [
            0, 4, 0,
            0, 1, 0,
            1, 0, 0,
        ];
        let found = geo.elements_in_patch(&patch, 3, 3);
        assert_eq!(found, vec![
            PickElement::Face { node, face: 0 },
            PickElement::Vertex { node, vertex: 2 },
        ]);
    }

    #[test]
    fn clear_values_and_unknown_ids_fall_out_of_a_rectangle() {
        let mut scene = Scene::new();
        scene.create_mesh("quad", unit_quad());
        let geo = PickGeometry::build(&mut scene);

        // 0 is the clear value; 99 is past the five-element table.
        let patch = [0u32; 9];
        assert!(geo.elements_in_patch(&patch, 3, 3).is_empty());
        let patch = [99, 0, 99, 0, 99, 0, 99, 0, 99];
        assert!(geo.elements_in_patch(&patch, 3, 3).is_empty());
    }

    #[test]
    fn a_short_or_empty_rectangle_resolves_to_nothing() {
        let mut scene = Scene::new();
        scene.create_mesh("quad", unit_quad());
        let geo = PickGeometry::build(&mut scene);

        assert!(geo.elements_in_patch(&[], 0, 0).is_empty());
        assert!(geo.elements_in_patch(&[], 2, 2).is_empty(), "short buffer");
        assert!(geo.elements_in_patch(&[1, 1, 1], 2, 2).is_empty(), "short buffer");
        assert!(PickGeometry::default().elements_in_patch(&[1], 1, 1).is_empty());
    }

    #[test]
    fn a_rectangle_answers_in_element_table_order_not_patch_order() {
        let mut scene = Scene::new();
        let node = scene.create_mesh("quad", unit_quad());
        let geo = PickGeometry::build(&mut scene);

        // Vertex ids 5 (index 3), 3 (index 1), 2 (index 0) and the face id 1
        // appear in that order in the patch; they come back indices ascending,
        // with the face that was pushed before any of them first of all.
        let patch = [0u32, 1, 5, 0, 3, 2];
        let found = geo.elements_in_patch(&patch, 3, 2);
        assert_eq!(found, vec![
            PickElement::Face { node, face: 0 },
            PickElement::Vertex { node, vertex: 0 },
            PickElement::Vertex { node, vertex: 1 },
            PickElement::Vertex { node, vertex: 3 },
        ]);
    }
}
