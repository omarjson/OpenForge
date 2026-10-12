//! Lindenmayer systems (L-systems): parametric-free string rewriting plus a
//! 3D turtle interpreter that turns the derived string into line segments and
//! optionally a tube mesh. Useful for procedural plants, branches, and
//! coral-like structures.

use glam::Vec3;
use openforge_core::mesh::Mesh;
use std::collections::HashMap;

/// A context-free L-system: an axiom string and a set of single-character
/// rewrite rules.
#[derive(Clone, Debug, Default)]
pub struct LSystem {
    pub axiom: String,
    pub rules: HashMap<char, String>,
}

impl LSystem {
    /// Start a system whose initial string is `axiom`.
    pub fn new(axiom: &str) -> Self {
        Self {
            axiom: axiom.to_string(),
            rules: HashMap::new(),
        }
    }

    /// Add (or replace) the rule `from -> to`. Builder-style.
    pub fn rule(mut self, from: char, to: &str) -> Self {
        self.rules.insert(from, to.to_string());
        self
    }

    /// Derive the string after `iterations` rounds of parallel rewriting.
    /// Characters without a rule are copied unchanged.
    pub fn derive(&self, iterations: u32) -> String {
        let mut cur = self.axiom.clone();
        for _ in 0..iterations {
            let mut next = String::with_capacity(cur.len() * 2);
            for ch in cur.chars() {
                match self.rules.get(&ch) {
                    Some(replacement) => next.push_str(replacement),
                    None => next.push(ch),
                }
            }
            cur = next;
        }
        cur
    }
}

/// Interpret an L-system string with a 3D turtle and return the drawn line
/// segments `(start, end)`.
///
/// Commands: `F`/`G` draw forward, `f` moves without drawing, `+`/`-` yaw,
/// `&`/`^` pitch, `|` turns around, `[`/`]` push/pop branch state. Any other
/// character is ignored.
pub fn turtle_segments(instructions: &str, angle_deg: f32, step: f32) -> Vec<(Vec3, Vec3)> {
    let angle = angle_deg.to_radians();
    let mut segs: Vec<(Vec3, Vec3)> = Vec::new();
    let mut stack: Vec<(Vec3, Vec3, Vec3)> = Vec::new();
    let mut pos = Vec3::ZERO;
    let mut forward = Vec3::Z;
    let mut up = Vec3::Y;
    for ch in instructions.chars() {
        match ch {
            'F' | 'G' => {
                let end = pos + forward * step;
                segs.push((pos, end));
                pos = end;
            }
            'f' => pos += forward * step,
            '+' => forward = glam::Quat::from_axis_angle(up, angle) * forward,
            '-' => forward = glam::Quat::from_axis_angle(up, -angle) * forward,
            '&' => {
                let right = forward.cross(up).normalize_or_zero();
                let q = glam::Quat::from_axis_angle(right, angle);
                forward = q * forward;
                up = q * up;
            }
            '^' => {
                let right = forward.cross(up).normalize_or_zero();
                let q = glam::Quat::from_axis_angle(right, -angle);
                forward = q * forward;
                up = q * up;
            }
            '|' => forward = -forward,
            '[' => stack.push((pos, forward, up)),
            ']' => {
                if let Some((p, f, u)) = stack.pop() {
                    pos = p;
                    forward = f;
                    up = u;
                }
            }
            _ => {}
        }
    }
    segs
}

/// Build a mesh of tubes, one cylindrical segment per `(start, end)` pair.
pub fn segments_to_mesh(segments: &[(Vec3, Vec3)], radius: f32, sides: u32) -> Mesh {
    let sides = sides.max(3);
    let mut m = Mesh::new();
    for &(a, b) in segments {
        let dir = (b - a).normalize_or_zero();
        if dir == Vec3::ZERO {
            continue;
        }
        let helper = if dir.dot(Vec3::Y).abs() < 0.99 { Vec3::Y } else { Vec3::Z };
        let n = dir.cross(helper).normalize_or_zero();
        let bn = dir.cross(n).normalize_or_zero();
        let base = m.vertex_count() as u32;
        for &end in [&a, &b] {
            for s in 0..sides {
                let ang = s as f32 / sides as f32 * std::f32::consts::TAU;
                let off = n * (ang.cos() * radius) + bn * (ang.sin() * radius);
                m.add_vertex(end + off);
            }
        }
        for s in 0..sides {
            let j = (s + 1) % sides;
            m.add_face(&[base + s, base + j, base + sides + j, base + sides + s]);
        }
    }
    m.compute_flat_normals();
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derive_grows_deterministically() {
        let ls = LSystem::new("F").rule('F', "F+F");
        assert_eq!(ls.derive(0), "F");
        assert_eq!(ls.derive(1), "F+F");
        assert_eq!(ls.derive(2), "F+F+F+F");
        // Deterministic across calls.
        assert_eq!(ls.derive(3), ls.derive(3));
    }

    #[test]
    fn chars_without_rules_are_copied() {
        let ls = LSystem::new("A+B").rule('A', "AB");
        assert_eq!(ls.derive(1), "AB+B");
    }

    #[test]
    fn turtle_draws_one_segment_per_F() {
        let segs = turtle_segments("F", 90.0, 1.0);
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].0, Vec3::ZERO);
        assert!((segs[0].1 - Vec3::Z).length() < 1e-5);
    }

    #[test]
    fn turtle_branches_push_and_pop() {
        // "F[F]F": the branch draws from the trunk tip, then state pops back so
        // the final F retraces — 3 segments total.
        let segs = turtle_segments("F[F]F", 90.0, 1.0);
        assert_eq!(segs.len(), 3);
        // A branching plant: main trunk plus two side branches.
        let plant = turtle_segments("F[+F][-F]", 45.0, 1.0);
        assert_eq!(plant.len(), 3);
    }

    #[test]
    fn segments_to_mesh_builds_tubes() {
        let segs = turtle_segments("F", 90.0, 2.0);
        let m = segments_to_mesh(&segs, 0.1, 6);
        assert_eq!(m.vertex_count(), 12); // 2 rings x 6
        assert_eq!(m.face_count(), 6);
    }

    #[test]
    fn degenerate_segment_is_skipped() {
        let m = segments_to_mesh(&[(Vec3::ZERO, Vec3::ZERO)], 0.1, 6);
        assert_eq!(m.face_count(), 0);
    }
}
