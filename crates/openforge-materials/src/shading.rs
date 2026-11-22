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

/// How the viewport presents a shaded surface.
///
/// The GPU fragment shader mirrors these four modes through the numeric code
/// from [`as_u32`](Self::as_u32); keep the two in step.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ShadingMode {
    /// The full Phong surface: ambient + diffuse + specular + emissive (default).
    #[default]
    #[serde(alias = "Lit")]
    Lit,
    /// The resolved material colour with no lighting at all — useful while
    /// editing geometry, where changing the camera must not change the colour.
    #[serde(alias = "Unlit")]
    Unlit,
    /// One neutral clay colour with the full lighting model, so the eye reads
    /// form and silhouette without material or texture distractions.
    #[serde(alias = "Clay")]
    Clay,
    /// World-space normals encoded as RGB, a debugging view of the shading
    /// normals both paths actually use.
    #[serde(alias = "Normals")]
    Normals,
}

impl ShadingMode {
    /// All modes, in the order the View menu lists them.
    pub const ALL: [Self; 4] = [Self::Lit, Self::Unlit, Self::Clay, Self::Normals];

    /// The spelling stored in the settings file.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Lit => "lit",
            Self::Unlit => "unlit",
            Self::Clay => "clay",
            Self::Normals => "normals",
        }
    }

    /// The label shown in the View menu.
    pub fn label(self) -> &'static str {
        match self {
            Self::Lit => "Lit",
            Self::Unlit => "Unlit",
            Self::Clay => "Clay",
            Self::Normals => "Normals",
        }
    }

    /// Parse the stored spelling, case-insensitively.
    pub fn from_str(value: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|mode| mode.as_str().eq_ignore_ascii_case(value))
    }

    /// The numeric code handed to the shader.
    pub fn as_u32(self) -> u32 {
        match self {
            Self::Lit => 0,
            Self::Unlit => 1,
            Self::Clay => 2,
            Self::Normals => 3,
        }
    }

    /// Decode a shader code; an unknown code falls back to [`Lit`](Self::Lit).
    pub fn from_u32(code: u32) -> Self {
        match code {
            1 => Self::Unlit,
            2 => Self::Clay,
            3 => Self::Normals,
            _ => Self::Lit,
        }
    }
}

/// Viewport-wide shading options shared by the CPU painter and GPU shader.
///
/// `ambient` is the share of the light that reaches a surface facing away from
/// the key light. `ao_strength` and `shadow_strength` scale the two occlusion
/// terms in [`occlusion_terms`]; `contact_radius` is the height over the ground
/// plane over which the contact shadow fades out.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct ShadeOptions {
    pub ambient: f32,
    pub ao_strength: f32,
    pub shadow_strength: f32,
    pub contact_radius: f32,
}

impl Default for ShadeOptions {
    fn default() -> Self {
        Self {
            ambient: 0.18,
            ao_strength: 1.0,
            shadow_strength: 1.0,
            contact_radius: 0.75,
        }
    }
}

impl ShadeOptions {
    /// Clamp every field to a safe, shader-friendly range. An invalid input
    /// (NaN, negative) degrades to the default for that field rather than
    /// producing NaN pixels.
    pub fn clamped(self) -> Self {
        let finite_or = |value: f32, fallback: f32| {
            if value.is_finite() { value } else { fallback }
        };
        let defaults = Self::default();
        Self {
            ambient: finite_or(self.ambient, defaults.ambient).clamp(0.0, 1.0),
            ao_strength: finite_or(self.ao_strength, defaults.ao_strength).clamp(0.0, 1.0),
            shadow_strength: finite_or(self.shadow_strength, defaults.shadow_strength)
                .clamp(0.0, 1.0),
            contact_radius: finite_or(self.contact_radius, defaults.contact_radius).clamp(0.01, 100.0),
        }
    }
}

/// Hemisphere ambient occlusion and ground-contact shadowing at one surface
/// point, both returned in `0.0..=1.0` where `1.0` means "unoccluded".
///
/// This is the cheap, geometry-free term both viewport paths use: an upward
/// normal sees the whole sky, a downward one sees none, and a point near the
/// ground plane loses light to the surface it is resting on. It is deliberately
/// analytic so it is identical on the CPU painter and in the fragment shader —
/// a full shadow map / SSAO pass would be a separate, larger work-stream.
pub fn occlusion_terms(normal: Vec3, world_y: f32, light_dir: Vec3, opts: ShadeOptions) -> (f32, f32) {
    let opts = opts.clamped();
    let n = normal.normalize_or_zero();
    let l = light_dir.normalize_or_zero();
    let sky = (0.5 + 0.5 * n.y).clamp(0.0, 1.0);
    let ao = 1.0 - opts.ao_strength * (1.0 - sky);
    // The lower a point sits and the more it faces away from the key light,
    // the deeper the contact shadow. Above the radius there is none.
    let grounded = 1.0 - (world_y / opts.contact_radius).clamp(0.0, 1.0);
    let away = (0.5 - 0.5 * n.dot(l)).clamp(0.0, 1.0);
    let shadow = 1.0 - opts.shadow_strength * grounded * away;
    (ao.clamp(0.0, 1.0), shadow.clamp(0.0, 1.0))
}

/// [`shade_surface`] with a viewport [`ShadingMode`], occlusion terms and an
/// ambient term, all mirrored by the GPU fragment shader.
///
/// In [`Lit`](ShadingMode::Lit) the result is `base * (ambient*ao +
/// (1-ambient)*N·L*shadow) + specular + emissive`, clamped per channel.
/// [`Unlit`](ShadingMode::Unlit) returns `base` unchanged,
/// [`Clay`](ShadingMode::Clay) substitutes a neutral clay colour before the
/// same lighting, and [`Normals`](ShadingMode::Normals) paints the normal as
/// `(n*0.5+0.5)*255`. Alpha is preserved in every mode.
pub fn shade_surface_configured(
    base: [u8; 4],
    specular: Vec3,
    emissive: Vec3,
    shininess: f32,
    normal: Vec3,
    light_dir: Vec3,
    view_dir: Vec3,
    world_y: f32,
    opts: ShadeOptions,
    mode: ShadingMode,
) -> [u8; 4] {
    match mode {
        ShadingMode::Unlit => base,
        ShadingMode::Normals => {
            let n = normal.normalize_or_zero();
            let enc = |c: f32| ((c * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0).round() as u8;
            [enc(n.x), enc(n.y), enc(n.z), base[3]]
        }
        ShadingMode::Lit | ShadingMode::Clay => {
            let opts = opts.clamped();
            let surface = if mode == ShadingMode::Clay {
                [176, 176, 176, base[3]]
            } else {
                base
            };
            let n = normal.normalize_or_zero();
            let l = light_dir.normalize_or_zero();
            let v = view_dir.normalize_or_zero();
            let half = (l + v).normalize_or_zero();
            let diffuse = n.dot(l).clamp(0.0, 1.0);
            let (ao, shadow) = occlusion_terms(normal, world_y, light_dir, opts);
            let lit = opts.ambient * ao + (1.0 - opts.ambient) * diffuse * shadow;
            let spec = if shininess.is_finite() && shininess > 0.0 {
                n.dot(half).clamp(0.0, 1.0).powf(shininess)
            } else {
                0.0
            };
            let mut rgb = [0.0f32; 3];
            for (i, channel) in rgb.iter_mut().enumerate() {
                let highlight = specular[i].clamp(0.0, 1.0) * spec * shadow;
                let glow = emissive[i].clamp(0.0, 1.0);
                *channel = surface[i] as f32 * lit + 255.0 * (highlight + glow);
            }
            [
                rgb[0].round().clamp(0.0, 255.0) as u8,
                rgb[1].round().clamp(0.0, 255.0) as u8,
                rgb[2].round().clamp(0.0, 255.0) as u8,
                base[3],
            ]
        }
    }
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

    // --- configurable viewport shading ---------------------------------------

    fn lit(mode: ShadingMode) -> [u8; 4] {
        shade_surface_configured(
            BASE,
            Vec3::ZERO,
            Vec3::ZERO,
            0.0,
            Vec3::Y,
            Vec3::Y,
            Vec3::Y,
            2.0,
            ShadeOptions::default(),
            mode,
        )
    }

    #[test]
    fn shading_modes_round_trip_through_text_and_shader_codes() {
        for mode in ShadingMode::ALL {
            assert_eq!(ShadingMode::from_str(mode.as_str()), Some(mode));
            assert_eq!(ShadingMode::from_str(mode.label()), Some(mode));
            assert_eq!(ShadingMode::from_u32(mode.as_u32()), mode);
        }
        assert_eq!(ShadingMode::from_str("LIT"), Some(ShadingMode::Lit));
        assert_eq!(ShadingMode::from_str("dusk"), None);
        assert_eq!(ShadingMode::from_u32(99), ShadingMode::Lit);
    }

    #[test]
    fn unlit_returns_the_exact_base_colour() {
        assert_eq!(lit(ShadingMode::Unlit), BASE);
    }

    #[test]
    fn normals_encodes_the_world_normal() {
        let out = shade_surface_configured(
            BASE,
            Vec3::ZERO,
            Vec3::ZERO,
            0.0,
            Vec3::Z,
            Vec3::Y,
            Vec3::Y,
            2.0,
            ShadeOptions::default(),
            ShadingMode::Normals,
        );
        assert_eq!(out, [128, 128, 255, 128]);
    }

    #[test]
    fn clay_ignores_the_material_colour_but_keeps_alpha() {
        let clay = lit(ShadingMode::Clay);
        assert_eq!(clay[3], BASE[3]);
        assert_eq!(clay[0], clay[1]);
        assert_eq!(clay[1], clay[2]);
        assert_ne!(clay, lit(ShadingMode::Lit), "clay replaces the material colour");
    }

    #[test]
    fn hemisphere_ao_darkens_downward_normals_only() {
        let opts = ShadeOptions::default();
        let (up, _) = occlusion_terms(Vec3::Y, 2.0, Vec3::Y, opts);
        let (down, _) = occlusion_terms(Vec3::NEG_Y, 2.0, Vec3::Y, opts);
        let (side, _) = occlusion_terms(Vec3::X, 2.0, Vec3::Y, opts);
        assert_eq!(up, 1.0);
        assert!((side - 0.5).abs() < 1e-6, "{side}");
        assert_eq!(down, 0.0);
    }

    #[test]
    fn ao_strength_zero_disables_ambient_occlusion() {
        let opts = ShadeOptions { ao_strength: 0.0, ..ShadeOptions::default() };
        assert_eq!(occlusion_terms(Vec3::NEG_Y, 2.0, Vec3::Y, opts).0, 1.0);
    }

    #[test]
    fn contact_shadow_fades_out_above_the_radius() {
        let opts = ShadeOptions::default();
        let (_, near) = occlusion_terms(Vec3::NEG_Y, 0.0, Vec3::Y, opts);
        let (_, far) = occlusion_terms(Vec3::NEG_Y, 2.0, Vec3::Y, opts);
        assert!(near < 1.0, "a point on the ground is shadowed: {near}");
        assert_eq!(far, 1.0, "above the radius there is no contact shadow");
    }

    #[test]
    fn shadow_strength_zero_disables_the_contact_shadow() {
        let opts = ShadeOptions { shadow_strength: 0.0, ..ShadeOptions::default() };
        assert_eq!(occlusion_terms(Vec3::NEG_Y, 0.0, Vec3::Y, opts).1, 1.0);
    }

    #[test]
    fn invalid_options_clamp_to_safe_values() {
        let opts = ShadeOptions {
            ambient: f32::NAN,
            ao_strength: -1.0,
            shadow_strength: 5.0,
            contact_radius: 0.0,
        }
        .clamped();
        assert_eq!(opts.ambient, ShadeOptions::default().ambient);
        assert_eq!(opts.ao_strength, 0.0);
        assert_eq!(opts.shadow_strength, 1.0);
        assert_eq!(opts.contact_radius, 0.01);
    }

    #[test]
    fn ambient_lifts_a_face_pointing_away_from_the_light() {
        let opts = ShadeOptions { ao_strength: 0.0, shadow_strength: 0.0, ..ShadeOptions::default() };
        let out = shade_surface_configured(
            BASE,
            Vec3::ZERO,
            Vec3::ZERO,
            0.0,
            Vec3::NEG_Z,
            Vec3::Z,
            Vec3::Z,
            2.0,
            opts,
            ShadingMode::Lit,
        );
        let expected = (BASE[0] as f32 * opts.ambient).round() as u8;
        assert_eq!(out[0], expected, "ambient, not pure black");
        assert_eq!(out[3], BASE[3]);
    }
}
