//! Small math helpers layered on top of `glam`.

use glam::{Mat4, Quat, Vec3};

/// A TRS (translate/rotate/scale) transform, the unit most scene nodes use.
#[derive(Clone, Copy, PartialEq, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Transform {
    pub translation: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            translation: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        }
    }
}

impl Transform {
    pub fn identity() -> Self {
        Self::default()
    }

    pub fn from_translation(t: Vec3) -> Self {
        Self { translation: t, ..Default::default() }
    }

    /// Compose this transform into a single matrix.
    pub fn to_matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.translation)
    }

    /// Build a transform from a matrix, discarding shear (best-effort).
    pub fn from_matrix(m: Mat4) -> Self {
        let (scale, rotation, translation) = m.to_scale_rotation_translation();
        Self { translation, rotation, scale }
    }

    /// The inverse, suitable for transforming normals (ignores non-uniform scale shear).
    pub fn inverse_matrix(&self) -> Mat4 {
        self.to_matrix().inverse()
    }
}

/// An axis-aligned bounding box.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Default for Aabb {
    fn default() -> Self {
        Self { min: Vec3::splat(f32::INFINITY), max: Vec3::splat(f32::NEG_INFINITY) }
    }
}

impl Aabb {
    pub fn is_empty(&self) -> bool {
        self.min.x > self.max.x || self.min.y > self.max.y || self.min.z > self.max.z
    }

    pub fn expand_point(&mut self, p: Vec3) {
        self.min = self.min.min(p);
        self.max = self.max.max(p);
    }

    pub fn merge(&mut self, other: &Aabb) {
        if !other.is_empty() {
            self.min = self.min.min(other.min);
            self.max = self.max.max(other.max);
        }
    }

    pub fn center(&self) -> Vec3 {
        if self.is_empty() {
            Vec3::ZERO
        } else {
            (self.min + self.max) * 0.5
        }
    }

    pub fn size(&self) -> Vec3 {
        if self.is_empty() {
            Vec3::ZERO
        } else {
            self.max - self.min
        }
    }

    pub fn transform(&self, m: Mat4) -> Aabb {
        if self.is_empty() {
            return *self;
        }
        let mut out = Aabb::default();
        for x in [self.min.x, self.max.x] {
            for y in [self.min.y, self.max.y] {
                for z in [self.min.z, self.max.z] {
                    out.expand_point(m.transform_point3(Vec3::new(x, y, z)));
                }
            }
        }
        out
    }
}

/// A ray in world space.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
}

impl Ray {
    pub fn at(&self, t: f32) -> Vec3 {
        self.origin + self.direction * t
    }

    /// Möller–Trumbore intersection. Returns the distance `t` along the ray.
    pub fn intersect_triangle(&self, a: Vec3, b: Vec3, c: Vec3) -> Option<f32> {
        const EPS: f32 = 1e-7;
        let e1 = b - a;
        let e2 = c - a;
        let p = self.direction.cross(e2);
        let det = e1.dot(p);
        if det.abs() < EPS {
            return None;
        }
        let inv = 1.0 / det;
        let tvec = self.origin - a;
        let u = tvec.dot(p) * inv;
        if !(0.0..=1.0).contains(&u) {
            return None;
        }
        let q = tvec.cross(e1);
        let v = self.direction.dot(q) * inv;
        if v < 0.0 || u + v > 1.0 {
            return None;
        }
        let t = e2.dot(q) * inv;
        if t > EPS {
            Some(t)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aabb_grows_and_centers() {
        let mut b = Aabb::default();
        assert!(b.is_empty());
        b.expand_point(Vec3::new(-1.0, -2.0, -3.0));
        b.expand_point(Vec3::new(1.0, 2.0, 3.0));
        assert!(!b.is_empty());
        assert_eq!(b.center(), Vec3::ZERO);
        assert_eq!(b.size(), Vec3::new(2.0, 4.0, 6.0));
    }

    #[test]
    fn trs_round_trip() {
        let t = Transform {
            translation: Vec3::new(1.0, 2.0, 3.0),
            rotation: Quat::from_rotation_y(0.5),
            scale: Vec3::new(2.0, 2.0, 2.0),
        };
        let back = Transform::from_matrix(t.to_matrix());
        assert!(back.translation.distance(t.translation) < 1e-5);
        assert!(back.scale.distance(t.scale) < 1e-5);
    }

    #[test]
    fn ray_hits_triangle() {
        let ray = Ray { origin: Vec3::new(0.25, 0.25, 1.0), direction: Vec3::new(0.0, 0.0, -1.0) };
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(1.0, 0.0, 0.0);
        let c = Vec3::new(0.0, 1.0, 0.0);
        let t = ray.intersect_triangle(a, b, c).expect("should hit");
        assert!((t - 1.0).abs() < 1e-5);
    }
}
