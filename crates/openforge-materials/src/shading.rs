//! Simple CPU shading helpers used by the viewport preview.

use glam::Vec3;

/// Basic Lambertian (diffuse) shading of an RGBA8 color.
///
/// `light_dir` points from the surface *toward* the light. The result is
/// `base` scaled by `max(N·L, 0)` with the alpha channel preserved.
/// Degenerate (zero-length) vectors shade to black rather than panicking.
pub fn shade_lambert(normal: Vec3, light_dir: Vec3, base: [u8; 4]) -> [u8; 4] {
    let n_dot_l = normal
        .normalize_or_zero()
        .dot(light_dir.normalize_or_zero())
        .clamp(0.0, 1.0);
    scale_rgb(base, n_dot_l)
}

/// Lambertian diffuse plus a Phong specular highlight.
///
/// `view_dir` points from the surface *toward* the viewer. The specular term
/// is a white highlight of intensity `max(H·N, 0)^shininess` where
/// `H = normalize(light_dir + view_dir)`, added on top of the diffuse term
/// and clamped per channel. Alpha is preserved; a non-positive or
/// non-finite `shininess` is treated as `0.0`.
pub fn shade_phong(
    normal: Vec3,
    light_dir: Vec3,
    view_dir: Vec3,
    shininess: f32,
    base: [u8; 4],
) -> [u8; 4] {
    let n = normal.normalize_or_zero();
    let l = light_dir.normalize_or_zero();
    let diffuse = n.dot(l).clamp(0.0, 1.0);
    let half = (l + view_dir).normalize_or_zero();
    let spec = if shininess.is_finite() && shininess > 0.0 {
        n.dot(half).clamp(0.0, 1.0).powf(shininess)
    } else {
        0.0
    };
    let rgb = [
        base[0] as f32 * diffuse + 255.0 * spec,
        base[1] as f32 * diffuse + 255.0 * spec,
        base[2] as f32 * diffuse + 255.0 * spec,
    ];
    [rgb[0].round() as u8, rgb[1].round() as u8, rgb[2].round() as u8, base[3]]
}

/// Multiply the RGB channels of an RGBA8 color by a scalar in `0.0..=1.0`,
/// preserving alpha.
fn scale_rgb(base: [u8; 4], factor: f32) -> [u8; 4] {
    let f = factor.clamp(0.0, 1.0);
    [
        (base[0] as f32 * f).round() as u8,
        (base[1] as f32 * f).round() as u8,
        (base[2] as f32 * f).round() as u8,
        base[3],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    const BASE: [u8; 4] = [200, 100, 50, 128];

    #[test]
    fn lambert_facing_light_is_full_brightness() {
        assert_eq!(shade_lambert(Vec3::Z, Vec3::Z, BASE), BASE);
    }

    #[test]
    fn lambert_facing_away_is_black() {
        assert_eq!(shade_lambert(Vec3::Z, Vec3::NEG_Z, BASE), [0, 0, 0, 128]);
    }

    #[test]
    fn lambert_preserves_alpha_and_scales_rgb() {
        // 45 degrees gives N·L = cos(45) ~= 0.7071.
        let l = Vec3::new(1.0, 0.0, 1.0).normalize();
        let out = shade_lambert(Vec3::Z, l, BASE);
        assert_eq!(out[3], 128); // alpha untouched
        assert_eq!(out[0], 141); // round(200 * 0.7071)
        assert_eq!(out[1], 71);
        assert_eq!(out[2], 35);
    }

    #[test]
    fn lambert_handles_zero_vectors() {
        assert_eq!(shade_lambert(Vec3::ZERO, Vec3::Z, BASE), [0, 0, 0, 128]);
        assert_eq!(shade_lambert(Vec3::Z, Vec3::ZERO, BASE), [0, 0, 0, 128]);
    }

    #[test]
    fn phong_adds_specular_highlight() {
        let lambert = shade_lambert(Vec3::Z, Vec3::Z, BASE);
        let phong = shade_phong(Vec3::Z, Vec3::Z, Vec3::Z, 32.0, BASE);
        // Head-on: diffuse is full plus a strong white highlight, clamped.
        assert_eq!(phong[0], 255);
        assert!(phong[0] > lambert[0]);
        assert_eq!(phong[3], 128);
    }

    #[test]
    fn phong_facing_away_is_black() {
        let out = shade_phong(Vec3::Z, Vec3::NEG_Z, Vec3::NEG_Z, 32.0, BASE);
        assert_eq!(out, [0, 0, 0, 128]);
    }
}
