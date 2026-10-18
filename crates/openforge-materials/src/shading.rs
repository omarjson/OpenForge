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

/// Lambertian diffuse plus a Phong highlight of the material's own specular
/// colour, plus self-illumination.
///
/// `specular` is the material's highlight colour, already scaled by any
/// specular map; `emissive` is its self-illumination plus any emissive map.
/// `view_dir` points from the surface *toward* the viewer, and `H =
/// normalize(light_dir + view_dir)` gives a highlight of
/// `max(H·N, 0)^shininess`, added on top of the diffuse term and clamped per
/// channel. Alpha is preserved. A non-positive or non-finite `shininess`
/// contributes no highlight; degenerate vectors shade to black rather than
/// panicking, and a zero `specular`/`emissive` adds nothing.
pub fn shade_surface(
    base: [u8; 4],
    specular: Vec3,
    emissive: Vec3,
    shininess: f32,
    normal: Vec3,
    light_dir: Vec3,
    view_dir: Vec3,
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
    let mut rgb = [0.0f32; 3];
    for (i, channel) in rgb.iter_mut().enumerate() {
        let highlight = specular[i].clamp(0.0, 1.0) * spec;
        let glow = emissive[i].clamp(0.0, 1.0);
        *channel = base[i] as f32 * diffuse + 255.0 * (highlight + glow);
    }
    [
        rgb[0].round().clamp(0.0, 255.0) as u8,
        rgb[1].round().clamp(0.0, 255.0) as u8,
        rgb[2].round().clamp(0.0, 255.0) as u8,
        base[3],
    ]
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
    shade_surface(
        base,
        Vec3::ONE,
        Vec3::ZERO,
        shininess,
        normal,
        light_dir,
        view_dir,
    )
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

    #[test]
    fn the_specular_colour_scales_the_highlight() {
        let white = shade_surface(BASE, Vec3::ONE, Vec3::ZERO, 32.0, Vec3::Z, Vec3::Z, Vec3::Z);
        let half = shade_surface(
            BASE,
            Vec3::splat(0.5),
            Vec3::ZERO,
            32.0,
            Vec3::Z,
            Vec3::Z,
            Vec3::Z,
        );
        let none = shade_surface(
            BASE,
            Vec3::ZERO,
            Vec3::ZERO,
            32.0,
            Vec3::Z,
            Vec3::Z,
            Vec3::Z,
        );
        // The highlight is the material's own colour: full white clamps at
        // 255, half of it still brightens, and none is the plain lambert. The
        // green channel is the one that has room to show it (red saturates).
        assert_eq!(white, shade_phong(Vec3::Z, Vec3::Z, Vec3::Z, 32.0, BASE));
        assert!(half[1] < white[1] && half[1] > none[1], "{half:?}");
        assert_eq!(none, shade_lambert(Vec3::Z, Vec3::Z, BASE));
        assert_eq!(half[3], 128, "alpha is preserved throughout");
    }

    #[test]
    fn emissive_glows_even_where_no_light_reaches() {
        let dark = shade_surface(
            [0, 0, 0, 255],
            Vec3::ZERO,
            Vec3::new(1.0, 0.5, 0.0),
            32.0,
            Vec3::NEG_Z,
            Vec3::Z,
            Vec3::NEG_Z,
        );
        assert_eq!(dark, [255, 128, 0, 255], "self-illumination survives facing away");

        // And it adds to a lit face rather than replacing it.
        let lit = shade_surface(
            BASE,
            Vec3::ZERO,
            Vec3::X,
            32.0,
            Vec3::Z,
            Vec3::Z,
            Vec3::Z,
        );
        let plain = shade_surface(
            BASE,
            Vec3::ZERO,
            Vec3::ZERO,
            32.0,
            Vec3::Z,
            Vec3::Z,
            Vec3::Z,
        );
        assert_eq!(lit[0], 255, "red saturates");
        assert_eq!((lit[1], lit[2]), (plain[1], plain[2]), "the other channels are untouched");
    }

    #[test]
    fn a_zero_specular_and_emissive_are_exactly_lambert() {
        let out = shade_surface(BASE, Vec3::ZERO, Vec3::ZERO, 0.0, Vec3::Z, Vec3::Z, Vec3::Z);
        assert_eq!(out, shade_lambert(Vec3::Z, Vec3::Z, BASE));
    }
}
