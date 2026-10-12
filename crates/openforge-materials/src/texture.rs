//! RGBA8 image textures and CPU-side sampling.

use crate::error::MaterialError;

/// The color returned for missing or unreadable texels (classic magenta
/// "missing texture" marker, so UV problems are obvious in the viewport).
pub const MISSING_TEXEL: [u8; 4] = [255, 0, 255, 255];

/// An uncompressed RGBA8 image texture.
///
/// Pixels are stored row-major, `4` bytes per texel (`R, G, B, A`). Row `0`
/// corresponds to texture coordinate `v = 0.0`. UV coordinates in
/// `0.0..=1.0` address the texture with wrap (repeat) addressing.
#[derive(Clone, PartialEq, Debug)]
pub struct Texture {
    /// Human-readable name (usually the source file name).
    pub name: String,
    /// Width in texels.
    pub width: u32,
    /// Height in texels.
    pub height: u32,
    /// RGBA8 texel data; `len() == width * height * 4` for constructed textures.
    pub pixels: Vec<u8>,
}

impl Texture {
    /// A texture of the given size filled with a single color.
    ///
    /// A zero width or height (or a size whose texel count would overflow)
    /// yields an empty texture rather than panicking.
    pub fn solid(name: impl Into<String>, width: u32, height: u32, rgba: [u8; 4]) -> Self {
        let len = texel_bytes(width, height);
        let pixels = match len {
            Some(n) => {
                let mut pixels = Vec::with_capacity(n);
                for _ in 0..n / 4 {
                    pixels.extend_from_slice(&rgba);
                }
                pixels
            }
            None => Vec::new(),
        };
        Texture { name: name.into(), width, height, pixels }
    }

    /// A square procedural checkerboard, great for UV debugging and tests.
    ///
    /// `size` is the texture extent in texels, `cell` the extent of one
    /// checker square (clamped to at least `1`). Alternating squares are
    /// filled with `color_a` and `color_b`; the texel at `(0, 0)` is `color_a`.
    pub fn checkerboard(
        name: impl Into<String>,
        size: u32,
        cell: u32,
        color_a: [u8; 4],
        color_b: [u8; 4],
    ) -> Self {
        let cell = cell.max(1);
        let len = texel_bytes(size, size);
        let pixels = match len {
            Some(n) => {
                let mut pixels = Vec::with_capacity(n);
                for y in 0..size {
                    for x in 0..size {
                        let c = if (x / cell + y / cell) % 2 == 0 { color_a } else { color_b };
                        pixels.extend_from_slice(&c);
                    }
                }
                pixels
            }
            None => Vec::new(),
        };
        Texture { name: name.into(), width: size, height: size, pixels }
    }

    /// Decode encoded image bytes (PNG or JPEG) into an RGBA8 texture.
    ///
    /// `source` labels the origin (typically a file path) in error messages.
    pub fn from_encoded(source: impl Into<String>, encoded: &[u8]) -> Result<Self, MaterialError> {
        let source = source.into();
        let img = image::load_from_memory(encoded)
            .map_err(|e| MaterialError::decode(source.clone(), e))?;
        let rgba = img.to_rgba8();
        let (width, height) = rgba.dimensions();
        Ok(Texture { name: source, width, height, pixels: rgba.into_raw() })
    }

    /// True when the texture has no texels to sample.
    pub fn is_empty(&self) -> bool {
        self.width == 0 || self.height == 0 || self.pixels.is_empty()
    }

    /// The texel at integer coordinates, or [`MISSING_TEXEL`] when the
    /// coordinates are out of range or the pixel data is short/empty.
    pub fn pixel_at(&self, x: u32, y: u32) -> [u8; 4] {
        if x >= self.width || y >= self.height || self.pixels.is_empty() {
            return MISSING_TEXEL;
        }
        let i = (y as usize) * (self.width as usize) + (x as usize);
        let i = i.saturating_mul(4);
        if i + 4 > self.pixels.len() {
            return MISSING_TEXEL;
        }
        let p = &self.pixels[i..i + 4];
        [p[0], p[1], p[2], p[3]]
    }

    /// Sample the texture at UV coordinates in `0.0..=1.0` with nearest-texel
    /// filtering and wrap (repeat) addressing.
    ///
    /// `u = 1.0` is equivalent to `u = 0.0` (it wraps), and negative or
    /// out-of-range coordinates repeat rather than clamping. Non-finite
    /// coordinates sample texel `(0, 0)`. Empty textures return
    /// [`MISSING_TEXEL`]. Row `0` (texel `y = 0`) is at `v = 0.0`.
    pub fn sample_nearest(&self, u: f32, v: f32) -> [u8; 4] {
        if self.is_empty() {
            return MISSING_TEXEL;
        }
        let x = wrap_coord(u * self.width as f32, self.width);
        let y = wrap_coord(v * self.height as f32, self.height);
        self.pixel_at(x, y)
    }
}

/// Byte length of `width * height` RGBA8 texels, or `None` on overflow/zero.
fn texel_bytes(width: u32, height: u32) -> Option<usize> {
    let w = width as usize;
    let h = height as usize;
    w.checked_mul(h)?.checked_mul(4)
}

/// Map a scaled texture coordinate to a wrapped texel index in `0..extent`.
///
/// Uses the floor convention (texel `i` covers `[i/extent, (i+1)/extent)`),
/// so `value == extent` wraps back to `0`. NaN maps to `0`; the casts
/// saturate, so infinities wrap instead of overflowing.
fn wrap_coord(value: f32, extent: u32) -> u32 {
    if extent == 0 {
        return 0;
    }
    let e = extent as i64;
    let i = value.floor() as i64;
    (((i % e) + e) % e) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    const RED: [u8; 4] = [255, 0, 0, 255];
    const GREEN: [u8; 4] = [0, 255, 0, 255];
    const BLUE: [u8; 4] = [0, 0, 255, 128];

    #[test]
    fn solid_has_expected_size_and_color() {
        let t = Texture::solid("red", 4, 3, RED);
        assert_eq!(t.width, 4);
        assert_eq!(t.height, 3);
        assert_eq!(t.pixels.len(), 4 * 3 * 4);
        assert!(t.pixels.chunks_exact(4).all(|p| p == RED));
        assert_eq!(t.pixel_at(0, 0), RED);
        assert_eq!(t.pixel_at(3, 2), RED);
    }

    #[test]
    fn solid_with_zero_size_is_empty() {
        let t = Texture::solid("empty", 0, 4, RED);
        assert!(t.is_empty());
        assert!(t.pixels.is_empty());
        // Sampling an empty texture is graceful, not a panic.
        assert_eq!(t.sample_nearest(0.5, 0.5), MISSING_TEXEL);
    }

    #[test]
    fn checkerboard_alternates_at_expected_coordinates() {
        let t = Texture::checkerboard("check", 4, 2, RED, BLUE);
        assert_eq!(t.width, 4);
        assert_eq!(t.height, 4);
        assert_eq!(t.pixels.len(), 4 * 4 * 4);
        assert_eq!(t.pixel_at(0, 0), RED);
        assert_eq!(t.pixel_at(1, 1), RED); // still inside the first cell
        assert_eq!(t.pixel_at(2, 0), BLUE);
        assert_eq!(t.pixel_at(3, 1), BLUE);
        assert_eq!(t.pixel_at(0, 2), BLUE);
        assert_eq!(t.pixel_at(2, 2), RED); // second row of cells wraps back
        assert_eq!(t.pixel_at(3, 3), RED);
    }

    #[test]
    fn checkerboard_zero_cell_does_not_panic() {
        let t = Texture::checkerboard("check", 4, 0, RED, GREEN);
        assert_eq!(t.pixels.len(), 4 * 4 * 4);
        // cell is clamped to 1, giving a 1-texel checker.
        assert_eq!(t.pixel_at(0, 0), RED);
        assert_eq!(t.pixel_at(1, 0), GREEN);
        assert_eq!(t.pixel_at(0, 1), GREEN);
    }

    #[test]
    fn sample_nearest_at_texel_centers_is_exact() {
        // Four distinct texels; centers are at (i + 0.5) / 2.
        let mut t = Texture::solid("q", 2, 2, RED);
        t.pixels[4..8].copy_from_slice(&GREEN);
        t.pixels[8..12].copy_from_slice(&BLUE);
        t.pixels[12..16].copy_from_slice(&[9, 9, 9, 9]);
        assert_eq!(t.sample_nearest(0.25, 0.25), RED);
        assert_eq!(t.sample_nearest(0.75, 0.25), GREEN);
        assert_eq!(t.sample_nearest(0.25, 0.75), BLUE);
        assert_eq!(t.sample_nearest(0.75, 0.75), [9, 9, 9, 9]);
    }

    #[test]
    fn sample_nearest_wraps_at_upper_boundary() {
        let mut t = Texture::solid("q", 2, 1, RED);
        t.pixels[4..8].copy_from_slice(&GREEN);
        // u = 1.0 is equivalent to u = 0.0 under wrap addressing.
        assert_eq!(t.sample_nearest(1.0, 0.5), RED);
        assert_eq!(t.sample_nearest(0.0, 0.5), RED);
        // Just below 1.0 still lands in the last column.
        assert_eq!(t.sample_nearest(0.999, 0.5), GREEN);

        // v = 1.0 wraps to row 0.
        let mut t2 = Texture::solid("s", 1, 2, RED);
        t2.pixels[4..8].copy_from_slice(&GREEN);
        assert_eq!(t2.sample_nearest(0.5, 0.75), GREEN);
        assert_eq!(t2.sample_nearest(0.5, 1.0), RED);
    }

    #[test]
    fn sample_nearest_wraps_negative_coordinates() {
        let mut t = Texture::solid("q", 2, 2, RED);
        t.pixels[4..8].copy_from_slice(&GREEN);
        // u = -0.1 wraps to the last column.
        assert_eq!(t.sample_nearest(-0.1, 0.25), GREEN);
        // Large positive coordinates repeat rather than clamping.
        assert_eq!(t.sample_nearest(2.25, 0.25), RED);
    }

    #[test]
    fn sample_nearest_handles_nonfinite_uv() {
        // Texel (0, 0) is RED; the rest are GREEN.
        let mut t = Texture::solid("q", 2, 2, GREEN);
        t.pixels[0..4].copy_from_slice(&RED);
        // NaN samples texel (0, 0); infinities wrap via saturating casts:
        // +inf maps to the last column, -inf to index 0.
        assert_eq!(t.sample_nearest(f32::NAN, f32::NAN), RED);
        assert_eq!(t.sample_nearest(f32::INFINITY, 0.25), GREEN);
        assert_eq!(t.sample_nearest(0.25, f32::NEG_INFINITY), RED);
    }

    #[test]
    fn pixel_at_out_of_range_returns_missing_texel() {
        let t = Texture::solid("s", 2, 2, RED);
        assert_eq!(t.pixel_at(2, 0), MISSING_TEXEL);
        assert_eq!(t.pixel_at(0, 2), MISSING_TEXEL);
    }

    #[test]
    fn from_encoded_rejects_garbage_bytes() {
        let err = Texture::from_encoded("bogus", &[1, 2, 3, 4]).unwrap_err();
        assert!(matches!(err, MaterialError::Decode { .. }));
    }
}
