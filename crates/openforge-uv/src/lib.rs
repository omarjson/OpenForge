//! UV projection, unwrapping, and UV editing operations for OpenForge.
//!
//! This crate extends [`Mesh`](openforge_core::mesh::Mesh) with UV workflows:
//!
//! * [`project`] — planar, cylindrical, spherical, and box (triplanar)
//!   projection of vertex positions into UV space.
//! * [`edit`] — bounds queries and affine UV edits (normalize, flip, scale,
//!   and per-selection transforms for an interactive editor).
//! * [`unwrap`] — seam detection by dihedral angle and a simple grid atlas
//!   repack for UV islands.
//!
//! All operations are defensive: empty meshes, out-of-range indices and
//! degenerate geometry never panic, and after any mutating call every face
//! corner's `uv` index is guaranteed to be a valid index into `mesh.uvs`.
//!
//! The API is exposed through the [`UvOps`] extension trait implemented for
//! `Mesh`:
//!
//! ```ignore
//! use glam::{Vec2, Vec3};
//! use openforge_core::mesh::Mesh;
//! use openforge_uv::UvOps;
//!
//! let mut mesh = Mesh::new();
//! // ... build geometry ...
//! mesh.uv_project_cylindrical(Vec3::Y);
//! mesh.uv_normalize();
//! mesh.uv_flip_v();
//! ```

mod edit;
mod project;
mod unwrap;

use glam::{Vec2, Vec3};
use openforge_core::mesh::Mesh;

/// Axis-aligned bounding box of a set of UV coordinates.
///
/// Non-finite entries are ignored; `None` is produced when no finite entry
/// exists (for example on an empty mesh).
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct UvBounds {
    /// Smallest corner of the box.
    pub min: Vec2,
    /// Largest corner of the box.
    pub max: Vec2,
}

impl UvBounds {
    /// Width (`x`) and height (`y`) of the box. Each component is `>= 0`.
    pub fn size(&self) -> Vec2 {
        self.max - self.min
    }
}

/// Extension operations for [`Mesh`](openforge_core::mesh::Mesh): projection
/// into UV space, affine UV edits, atlas repacking, and seam detection.
///
/// Every mutating method keeps all corner `uv` indices valid, even for empty
/// or malformed input, so it can be applied to any mesh at any time.
pub trait UvOps {
    /// Project every vertex onto the plane perpendicular to `axis` and write
    /// one UV per vertex. The two axes orthogonal to `axis` become U and V;
    /// the resulting range is then stretched into `0..1` per axis.
    ///
    /// A zero-length `axis` falls back to `Vec3::Y`.
    fn uv_project_planar(&mut self, axis: Vec3);

    /// Wrap the mesh around `axis`: `U = angle / 2π` (shifted to `0..1`)
    /// measured in the plane perpendicular to the axis, `V` = height along
    /// the axis normalized to `0..1`.
    ///
    /// The axis line passes through the center of the mesh bounds.
    fn uv_project_cylindrical(&mut self, axis: Vec3);

    /// Map the mesh onto a sphere around `axis` (line through the center of
    /// the mesh bounds): `U` = longitude and `V` = latitude measured from the
    /// axis, both inherently normalized to `0..1`.
    fn uv_project_spherical(&mut self, axis: Vec3);

    /// Box (triplanar) projection: for each face, pick the dominant axis of
    /// its geometric normal and drop that coordinate, giving every face its
    /// own UVs. Projections are normalized `0..1` per dominant-axis group
    /// (X-, Y-, and Z-facing faces are normalized together), which keeps
    /// texel density consistent between the two faces of each axis pair.
    fn uv_project_box(&mut self);

    /// Rescale all UVs so their bounding box maps into `0..1`.
    ///
    /// Normalization is **per axis** (aspect is *not* preserved): the U range
    /// and the V range are each stretched independently. A degenerate axis
    /// (zero extent) is mapped to `0.5`. Non-finite entries are reset to
    /// `Vec2::ZERO` first.
    fn uv_normalize(&mut self);

    /// Flip the V coordinate of every UV (`v = 1 - v`), the common
    /// GL-vs-UV texture convention fix.
    fn uv_flip_v(&mut self);

    /// Apply the affine transform `uv' = uv * scale + offset` to every UV.
    fn uv_scale_translate(&mut self, scale: Vec2, offset: Vec2);

    /// 2D area of a face in UV space, summing the fan-triangulated corner
    /// triangles (absolute values). Useful for spotting distortion.
    ///
    /// Returns `0.0` for an out-of-range face, faces with fewer than three
    /// corners, or corners whose `uv` index does not resolve.
    fn uv_face_area_uv(&self, face: usize) -> f32;

    /// Repack UVs into a `grid x grid` atlas to reduce overlap.
    ///
    /// Faces are grouped into islands by shared position edges (union-find),
    /// each island's existing UVs are uniformly scaled (aspect preserved) to
    /// fit its own atlas cell with a small padding, and centered in the cell.
    /// Islands beyond `grid * grid` cells wrap around to earlier cells (so
    /// they overlap); `grid == 0` is a no-op.
    fn uv_repack(&mut self, grid: u32);

    /// Detect seam edges: edges on a boundary (used by a single face), edges
    /// whose adjacent face-normal angle exceeds `angle_deg`, and
    /// non-manifold edges. Returned as sorted, de-duplicated position-index
    /// pairs.
    ///
    /// The normal angle is `0` for coplanar faces and `90` for the perpendicular
    /// faces of a cube; edges of degenerate faces (zero area or invalid
    /// indices) are ignored. Returns an empty vector for an empty mesh or a
    /// non-finite threshold.
    fn uv_seams_from_angle(&self, angle_deg: f32) -> Vec<(u32, u32)>;

    /// Bounding box of all UV entries, or `None` when there are no finite
    /// entries (e.g. an empty mesh).
    fn uv_bounds(&self) -> Option<UvBounds>;

    /// `(uv index, value)` for every corner of `face`, in corner order.
    ///
    /// Empty for an out-of-range face. Values read defensively (a dangling or
    /// non-finite entry reads as `Vec2::ZERO`), so the result is always safe to
    /// display or feed back into a selection.
    fn uv_face_uvs(&self, face: usize) -> Vec<(u32, Vec2)>;

    /// Bounding box over a subset of UV entries, by index.
    ///
    /// Out-of-range and non-finite entries are skipped; `None` when nothing
    /// usable remains. This is the pivot source for selection transforms.
    fn uv_bounds_selected(&self, indices: &[u32]) -> Option<UvBounds>;

    /// Apply `uv' = R(rotate) * ((uv - pivot) * scale) + pivot + translate` to
    /// a subset of UV entries, leaving every other entry untouched.
    ///
    /// Scale is applied before the rotation, both about `pivot`. This is what
    /// an interactive editor drags: the caller re-applies a per-frame delta for
    /// a live preview. Out-of-range indices are skipped, and a non-finite
    /// parameter (or an empty index list) makes the whole call a no-op.
    fn uv_transform_selected(
        &mut self,
        indices: &[u32],
        translate: Vec2,
        scale: Vec2,
        rotate: f32,
        pivot: Vec2,
    );
}

// A single impl block (Rust allows only one per trait/type pair); the bodies
// live in their feature modules as `pub(crate)` free functions.
impl UvOps for Mesh {
    fn uv_project_planar(&mut self, axis: Vec3) {
        project::uv_project_planar(self, axis);
    }

    fn uv_project_cylindrical(&mut self, axis: Vec3) {
        project::uv_project_cylindrical(self, axis);
    }

    fn uv_project_spherical(&mut self, axis: Vec3) {
        project::uv_project_spherical(self, axis);
    }

    fn uv_project_box(&mut self) {
        project::uv_project_box(self);
    }

    fn uv_normalize(&mut self) {
        edit::uv_normalize(self);
    }

    fn uv_flip_v(&mut self) {
        edit::uv_flip_v(self);
    }

    fn uv_scale_translate(&mut self, scale: Vec2, offset: Vec2) {
        edit::uv_scale_translate(self, scale, offset);
    }

    fn uv_face_area_uv(&self, face: usize) -> f32 {
        edit::uv_face_area_uv(self, face)
    }

    fn uv_repack(&mut self, grid: u32) {
        unwrap::uv_repack(self, grid);
    }

    fn uv_seams_from_angle(&self, angle_deg: f32) -> Vec<(u32, u32)> {
        unwrap::uv_seams_from_angle(self, angle_deg)
    }

    fn uv_bounds(&self) -> Option<UvBounds> {
        edit::uv_bounds(self)
    }

    fn uv_face_uvs(&self, face: usize) -> Vec<(u32, Vec2)> {
        edit::uv_face_uvs(self, face)
    }

    fn uv_bounds_selected(&self, indices: &[u32]) -> Option<UvBounds> {
        edit::uv_bounds_selected(self, indices)
    }

    fn uv_transform_selected(
        &mut self,
        indices: &[u32],
        translate: Vec2,
        scale: Vec2,
        rotate: f32,
        pivot: Vec2,
    ) {
        edit::uv_transform_selected(self, indices, translate, scale, rotate, pivot);
    }
}

/// Build an orthonormal basis `(n, u, v)` where `n` is the normalized
/// `axis` (falling back to `Vec3::Y` when it is zero or non-finite) and
/// `u`, `v` span the plane perpendicular to it.
pub(crate) fn projection_basis(axis: Vec3) -> (Vec3, Vec3, Vec3) {
    let n = axis.normalize_or_zero();
    let n = if n == Vec3::ZERO { Vec3::Y } else { n };
    // Project a cardinal axis onto the plane perpendicular to `n`.
    let mut u = Vec3::ZERO;
    for helper in [Vec3::X, Vec3::Y, Vec3::Z] {
        let candidate = (helper - n * helper.dot(n)).normalize_or_zero();
        if candidate != Vec3::ZERO {
            u = candidate;
            break;
        }
    }
    let v = n.cross(u);
    (n, u, v)
}

/// Bounds of the finite entries of `uvs`, or `None` if there are none.
pub(crate) fn uv_bounds_of(uvs: &[Vec2]) -> Option<UvBounds> {
    let mut min = Vec2::ZERO;
    let mut max = Vec2::ZERO;
    let mut first = true;
    for v in uvs {
        if !v.is_finite() {
            continue;
        }
        if first {
            min = *v;
            max = *v;
            first = false;
        } else {
            min = min.min(*v);
            max = max.max(*v);
        }
    }
    if first {
        None
    } else {
        Some(UvBounds { min, max })
    }
}

/// Replace non-finite entries with `Vec2::ZERO`.
pub(crate) fn sanitize_uvs(uvs: &mut [Vec2]) {
    for v in uvs.iter_mut() {
        if !v.is_finite() {
            *v = Vec2::ZERO;
        }
    }
}

/// Stretch `uvs` per axis so the bounding box spans `0..1`
/// (aspect is not preserved); a degenerate axis maps to `0.5`.
pub(crate) fn normalize_in_place(uvs: &mut [Vec2]) {
    sanitize_uvs(uvs);
    let Some(b) = uv_bounds_of(uvs) else {
        return;
    };
    let span = b.size();
    for v in uvs.iter_mut() {
        v.x = if span.x > 0.0 && span.x.is_finite() {
            (v.x - b.min.x) / span.x
        } else {
            0.5
        };
        v.y = if span.y > 0.0 && span.y.is_finite() {
            (v.y - b.min.y) / span.y
        } else {
            0.5
        };
    }
}

/// Geometric face normal via Newell's method, guarded against out-of-range
/// indices. Returns `None` for faces with fewer than three valid corners or
/// zero/non-finite area.
pub(crate) fn face_normal_opt(mesh: &Mesh, fi: usize) -> Option<Vec3> {
    let face = mesh.faces.get(fi)?;
    let cs = &face.corners;
    if cs.len() < 3 {
        return None;
    }
    let corner = |i: usize| -> Option<Vec3> {
        let idx = cs[i].position as usize;
        if idx < mesh.positions.len() {
            Some(mesh.positions[idx])
        } else {
            None
        }
    };
    let mut n = Vec3::ZERO;
    for i in 0..cs.len() {
        let a = corner(i)?;
        let b = corner((i + 1) % cs.len())?;
        n += a.cross(b);
    }
    let n = n.normalize_or_zero();
    if n == Vec3::ZERO {
        None
    } else {
        Some(n)
    }
}

/// Install `raw` as the new UV array with **one UV entry per vertex**
/// (`corner.uv = corner.position`), sanitizing values and repairing any
/// out-of-range corner position so every corner `uv` index stays valid.
pub(crate) fn commit_vertex_uvs(mesh: &mut Mesh, mut raw: Vec<Vec2>) {
    sanitize_uvs(&mut raw);
    if raw.is_empty() && !mesh.faces.is_empty() {
        raw.push(Vec2::ZERO);
    }
    let len = raw.len();
    for face in mesh.faces.iter_mut() {
        for c in face.corners.iter_mut() {
            c.uv = if (c.position as usize) < len { c.position } else { 0 };
        }
    }
    mesh.uvs = raw;
}

/// Install `raw` as the new UV array with **one UV entry per face corner**
/// (in face/corner order, so every face gets its own UVs), guaranteeing that
/// every corner `uv` index is valid afterwards.
pub(crate) fn commit_corner_uvs(mesh: &mut Mesh, mut raw: Vec<Vec2>) {
    let total: usize = mesh.faces.iter().map(|f| f.corners.len()).sum();
    if raw.len() != total {
        // Defensive: keep indices valid even if the caller mis-sized `raw`.
        raw.resize(total, Vec2::ZERO);
    }
    sanitize_uvs(&mut raw);
    let mut idx = 0u32;
    for face in mesh.faces.iter_mut() {
        for c in face.corners.iter_mut() {
            c.uv = idx;
            idx += 1;
        }
    }
    mesh.uvs = raw;
}

/// Read a UV defensively, returning `Vec2::ZERO` when the index is
/// out of range or the value is non-finite.
pub(crate) fn uv_at(uvs: &[Vec2], uv_index: u32) -> Vec2 {
    match uvs.get(uv_index as usize) {
        Some(v) if v.is_finite() => *v,
        _ => Vec2::ZERO,
    }
}

#[cfg(test)]
pub(crate) mod test_util {
    use super::*;

    /// Assert that every corner's `uv` index resolves into `mesh.uvs`.
    pub(crate) fn assert_uv_indices_valid(mesh: &Mesh) {
        for (fi, face) in mesh.faces.iter().enumerate() {
            for (ci, c) in face.corners.iter().enumerate() {
                assert!(
                    (c.uv as usize) < mesh.uvs.len(),
                    "face {fi} corner {ci} has uv index {} but mesh has {} uvs",
                    c.uv,
                    mesh.uvs.len()
                );
            }
        }
    }

    /// A unit quad in the XY plane with vertices `(0,0) (1,0) (1,1) (0,1)`.
    pub(crate) fn unit_quad() -> Mesh {
        let mut m = Mesh::new();
        m.add_vertex(Vec3::new(0.0, 0.0, 0.0));
        m.add_vertex(Vec3::new(1.0, 0.0, 0.0));
        m.add_vertex(Vec3::new(1.0, 1.0, 0.0));
        m.add_vertex(Vec3::new(0.0, 1.0, 0.0));
        m.add_face(&[0, 1, 2, 3]);
        m
    }

    /// A watertight unit cube (side length 2, centered at the origin) built
    /// from six quads with shared corner positions.
    pub(crate) fn cube() -> Mesh {
        let mut m = Mesh::new();
        m.add_vertex(Vec3::new(-1.0, -1.0, -1.0)); // 0
        m.add_vertex(Vec3::new(1.0, -1.0, -1.0)); // 1
        m.add_vertex(Vec3::new(1.0, 1.0, -1.0)); // 2
        m.add_vertex(Vec3::new(-1.0, 1.0, -1.0)); // 3
        m.add_vertex(Vec3::new(-1.0, -1.0, 1.0)); // 4
        m.add_vertex(Vec3::new(1.0, -1.0, 1.0)); // 5
        m.add_vertex(Vec3::new(1.0, 1.0, 1.0)); // 6
        m.add_vertex(Vec3::new(-1.0, 1.0, 1.0)); // 7
        m.add_face(&[0, 3, 2, 1]); // -Z
        m.add_face(&[4, 5, 6, 7]); // +Z
        m.add_face(&[0, 1, 5, 4]); // -Y
        m.add_face(&[3, 7, 6, 2]); // +Y
        m.add_face(&[0, 4, 7, 3]); // -X
        m.add_face(&[1, 2, 6, 5]); // +X
        m
    }

    /// An open cylinder side wall (`segments` quads) with `y` in `0..1` and
    /// radius 1 around the Y axis. No caps, so the top and bottom rims are
    /// boundary loops.
    pub(crate) fn open_cylinder(segments: u32) -> Mesh {
        let mut m = Mesh::new();
        let segments = segments.max(3);
        for i in 0..segments {
            let a = i as f32 / segments as f32 * std::f32::consts::TAU;
            let (s, c) = a.sin_cos();
            m.add_vertex(Vec3::new(c, 0.0, s));
            m.add_vertex(Vec3::new(c, 1.0, s));
        }
        for i in 0..segments {
            let j = (i + 1) % segments;
            let bot_i = i * 2;
            let top_i = bot_i + 1;
            let bot_j = j * 2;
            let top_j = bot_j + 1;
            m.add_face(&[bot_i, bot_j, top_j, top_i]);
        }
        m
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_mesh_all_operations_are_safe() {
        let mut m = Mesh::new();
        m.uv_project_planar(Vec3::Z);
        m.uv_project_cylindrical(Vec3::Y);
        m.uv_project_spherical(Vec3::Y);
        m.uv_project_box();
        m.uv_normalize();
        m.uv_flip_v();
        m.uv_scale_translate(Vec2::splat(2.0), Vec2::splat(0.25));
        m.uv_repack(2);
        assert_eq!(m.uv_face_area_uv(0), 0.0);
        assert!(m.uv_seams_from_angle(30.0).is_empty());
        assert_eq!(m.uv_bounds(), None);
        test_util::assert_uv_indices_valid(&m);
    }

    #[test]
    fn uv_bounds_ignores_non_finite_and_matches_content() {
        let mut m = Mesh::new();
        assert_eq!(m.uv_bounds(), None);
        m.uvs = vec![Vec2::new(-1.0, 2.0), Vec2::new(3.0, 6.0), Vec2::new(f32::NAN, 0.0)];
        let b = m.uv_bounds().expect("finite entries exist");
        assert_eq!(b.min, Vec2::new(-1.0, 2.0));
        assert_eq!(b.max, Vec2::new(3.0, 6.0));
        assert_eq!(b.size(), Vec2::new(4.0, 4.0));
    }

    #[test]
    fn projection_basis_is_orthonormal() {
        for axis in [Vec3::X, Vec3::Y, Vec3::Z, Vec3::new(1.0, 2.0, -3.0), Vec3::ZERO] {
            let (n, u, v) = projection_basis(axis);
            assert!((n.length() - 1.0).abs() < 1e-6);
            assert!(n.dot(u).abs() < 1e-6, "u must be perpendicular to axis");
            assert!(n.dot(v).abs() < 1e-6, "v must be perpendicular to axis");
            assert!(u.dot(v).abs() < 1e-6);
            assert!((u.length() - 1.0).abs() < 1e-6);
        }
    }
}
