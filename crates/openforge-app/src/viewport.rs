//! A lightweight 3D viewport that projects the scene and draws it with the
//! egui painter.
//!
//! This keeps the renderer dependency-free and stable across egui/wgpu
//! versions on all three desktop platforms. It draws a ground grid, the
//! selected/unselected meshes (flat-shaded by face normal), and selection
//! outlines. A production build would swap this for a wgpu pipeline while
//! reusing the same camera math.

use glam::{Mat4, Vec3, Vec4};
use openforge_core::math::Ray;
use openforge_core::mesh::Mesh;
use openforge_core::scene::Scene;
use openforge_core::selection::Selection;
use openforge_core::NodeId;

/// A perspective orbit camera.
#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub target: Vec3,
    pub distance: f32,
    pub yaw: f32,
    pub pitch: f32,
    pub fov_y: f32,
    pub near: f32,
    pub far: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            target: Vec3::ZERO,
            distance: 8.0,
            yaw: 0.6,
            pitch: 0.5,
            fov_y: 55f32.to_radians(),
            near: 0.05,
            far: 1000.0,
        }
    }
}

impl Camera {
    pub fn eye(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        self.target + Vec3::new(cy * cp, sp, sy * cp) * self.distance
    }

    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(self.eye(), self.target, Vec3::Y)
    }

    pub fn proj_matrix(&self, aspect: f32) -> Mat4 {
        Mat4::perspective_rh(self.fov_y, aspect.max(0.01), self.near, self.far)
    }

    pub fn view_proj(&self, aspect: f32) -> Mat4 {
        self.proj_matrix(aspect) * self.view_matrix()
    }

    /// Orbit by pixel deltas.
    pub fn orbit(&mut self, dx: f32, dy: f32) {
        self.yaw += dx * 0.01;
        self.pitch = (self.pitch + dy * 0.01).clamp(-1.5, 1.5);
    }

    pub fn pan(&mut self, dx: f32, dy: f32) {
        let view = self.view_matrix();
        let right = Vec3::new(view.x_axis.x, view.y_axis.x, view.z_axis.x);
        let up = Vec3::new(view.x_axis.y, view.y_axis.y, view.z_axis.y);
        let scale = self.distance * 0.002;
        self.target += (-right * dx + up * dy) * scale;
    }

    pub fn zoom(&mut self, scroll: f32) {
        self.distance = (self.distance * (1.0 - scroll * 0.1)).clamp(0.1, 5000.0);
    }

    /// Build a picking ray from normalized device coords in [-1, 1].
    pub fn ray_from_ndc(&self, ndc_x: f32, ndc_y: f32, aspect: f32) -> Ray {
        let inv = self.view_proj(aspect).inverse();
        let near = inv * Vec4::new(ndc_x, ndc_y, -1.0, 1.0);
        let far = inv * Vec4::new(ndc_x, ndc_y, 1.0, 1.0);
        let near = near.truncate() / near.w;
        let far = far.truncate() / far.w;
        Ray { origin: near, direction: (far - near).normalize_or_zero() }
    }
}

/// Project a world point to normalized device coords + depth. Returns `None`
/// when behind the camera.
pub fn project(vp: Mat4, p: Vec3, size: [f32; 2]) -> Option<(f32, f32, f32)> {
    let clip = vp * Vec4::new(p.x, p.y, p.z, 1.0);
    if clip.w <= 1e-5 {
        return None;
    }
    let ndc = clip.truncate() / clip.w;
    let sx = (ndc.x * 0.5 + 0.5) * size[0];
    let sy = (1.0 - (ndc.y * 0.5 + 0.5)) * size[1];
    Some((sx, sy, ndc.z))
}

/// A vertex ready for rasterization.
pub struct Vx {
    pub pos: Vec3,
    pub normal: Vec3,
}

/// A triangle in world space with a shading normal.
pub struct Tri {
    pub v: [Vx; 3],
    pub selected: bool,
    /// True when this triangle's face is in the current face selection.
    pub face_selected: bool,
}

/// Collect all visible mesh triangles from the scene in world space.
pub fn collect_triangles(
    scene: &mut Scene,
    selection: &Selection,
) -> Vec<Tri> {
    let mut out = Vec::new();
    let ids: Vec<NodeId> = scene.iter().filter(|n| n.is_mesh() && n.visible).map(|n| n.id).collect();
    for id in ids {
        let Ok(world) = scene.world_matrix(id) else { continue };
        let selected = selection.is_node_selected(id);
        let sel_faces = selection.selected_elements(id).cloned();
        let Ok(node) = scene.get(id) else { continue };
        let Some(mesh) = node.mesh() else { continue };
        // Map each source face to its triangulated output.
        for (face_idx, face) in mesh.faces.iter().enumerate() {
            let face_is_sel = sel_faces
                .as_ref()
                .map(|s| s.contains(&(face_idx as u32)))
                .unwrap_or(false);
            let cs = &face.corners;
            for i in 1..cs.len().saturating_sub(1) {
                let tri = [cs[0].position, cs[i].position, cs[i + 1].position];
                let p = |pi: u32| world.transform_point3(mesh.positions[pi as usize]);
                let a = p(tri[0]);
                let b = p(tri[1]);
                let c = p(tri[2]);
                let n = (b - a).cross(c - a).normalize_or_zero();
                out.push(Tri {
                    v: [
                        Vx { pos: a, normal: n },
                        Vx { pos: b, normal: n },
                        Vx { pos: c, normal: n },
                    ],
                    selected,
                    face_selected: face_is_sel,
                });
            }
        }
    }
    out
}

/// Result of a face pick: the node and the source face index.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct FaceHit {
    pub node: NodeId,
    pub face: u32,
    pub dist: f32,
}

/// Cast a ray and return the nearest visible face across all mesh nodes.
pub fn pick_face(scene: &mut Scene, ray: Ray) -> Option<FaceHit> {
    let candidates: Vec<(NodeId, Mesh)> = scene
        .iter()
        .filter(|n| n.is_mesh() && n.visible && !n.frozen)
        .filter_map(|n| n.mesh().map(|m| (n.id, m.clone())))
        .collect();

    let mut best: Option<FaceHit> = None;
    for (id, mesh) in candidates {
        let Ok(world) = scene.world_matrix(id) else { continue };
        let inv = world.inverse();
        let local_origin = inv.transform_point3(ray.origin);
        let local_dir = (inv * ray.direction.extend(0.0)).truncate().normalize_or_zero();
        let local_ray = Ray { origin: local_origin, direction: local_dir };
        for (fi, face) in mesh.faces.iter().enumerate() {
            let cs = &face.corners;
            for i in 1..cs.len().saturating_sub(1) {
                let a = mesh.positions[cs[0].position as usize];
                let b = mesh.positions[cs[i].position as usize];
                let c = mesh.positions[cs[i + 1].position as usize];
                if let Some(t) = local_ray.intersect_triangle(a, b, c) {
                    let world_hit = world.transform_point3(local_ray.at(t));
                    let dist = world_hit.distance(ray.origin);
                    let better = best.map(|b| dist < b.dist).unwrap_or(true);
                    if better {
                        best = Some(FaceHit { node: id, face: fi as u32, dist });
                    }
                }
            }
        }
    }
    best
}

/// Result of a vertex pick.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct VertexHit {
    pub node: NodeId,
    pub vertex: u32,
    pub dist: f32,
}

/// Cast a ray and return the nearest vertex within screen-space tolerance.
pub fn pick_vertex(
    scene: &mut Scene,
    ray: Ray,
    cam_pos: Vec3,
    tol_world: f32,
) -> Option<VertexHit> {
    let candidates: Vec<(NodeId, Mesh)> = scene
        .iter()
        .filter(|n| n.is_mesh() && n.visible && !n.frozen)
        .filter_map(|n| n.mesh().map(|m| (n.id, m.clone())))
        .collect();

    let mut best: Option<VertexHit> = None;
    for (id, mesh) in candidates {
        let Ok(world) = scene.world_matrix(id) else { continue };
        for (vi, _) in mesh.positions.iter().enumerate() {
            let wp = world.transform_point3(mesh.positions[vi]);
            // Distance from the vertex to the ray.
            let to_v = wp - ray.origin;
            let along = to_v.dot(ray.direction);
            if along < 0.0 {
                continue;
            }
            let closest = ray.at(along);
            let perp = wp.distance(closest);
            if perp <= tol_world {
                let better = best.map(|b| along < b.dist).unwrap_or(true);
                if better {
                    best = Some(VertexHit { node: id, vertex: vi as u32, dist: along });
                }
            }
        }
    }
    let _ = cam_pos;
    best
}
