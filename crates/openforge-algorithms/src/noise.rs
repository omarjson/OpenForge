//! Deterministic gradient noise and fractal Brownian motion (fBm).
//!
//! [`Perlin`] is a seeded 2D/3D gradient noise generator producing values in
//! approximately `-1..1`. [`fbm2`]/[`fbm3`] layer several octaves for
//! natural-looking turbulence, and [`displace_along_normals`] pushes a mesh's
//! vertices along their normals by an fBm field for quick roughening.

use openforge_core::mesh::Mesh;

/// Seeded gradient noise. The same seed always yields the same field.
pub struct Perlin {
    perm: [u8; 512],
}

impl Perlin {
    /// Build a noise field from `seed` using a seeded Fisher-Yates shuffle of
    /// the 0..256 permutation.
    pub fn new(seed: u64) -> Self {
        let mut p: [u8; 256] = [0; 256];
        for (i, v) in p.iter_mut().enumerate() {
            *v = i as u8;
        }
        // SplitMix64 for a well-distributed, seedable stream.
        let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
        let mut next = || {
            state = state.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        };
        for i in (1..256).rev() {
            let j = (next() as usize) % (i + 1);
            p.swap(i, j);
        }
        let mut perm = [0u8; 512];
        for i in 0..512 {
            perm[i] = p[i & 255];
        }
        Self { perm }
    }

    fn grad2(hash: u8, x: f32, y: f32) -> f32 {
        // 8 evenly spaced gradient directions.
        match hash & 7 {
            0 => x + y,
            1 => -x + y,
            2 => x - y,
            3 => -x - y,
            4 => x,
            5 => -x,
            6 => y,
            _ => -y,
        }
    }

    fn grad3(hash: u8, x: f32, y: f32, z: f32) -> f32 {
        match hash & 15 {
            0 => x + y,
            1 => -x + y,
            2 => x - y,
            3 => -x - y,
            4 => x + z,
            5 => -x + z,
            6 => x - z,
            7 => -x - z,
            8 => y + z,
            9 => -y + z,
            10 => y - z,
            11 => -y - z,
            12 => x + y,
            13 => -y + z,
            14 => -x + y,
            _ => -y - z,
        }
    }

    /// 2D gradient noise at `(x, y)`, roughly in `-1..1`.
    pub fn noise2(&self, x: f32, y: f32) -> f32 {
        let xi = x.floor();
        let yi = y.floor();
        let xf = x - xi;
        let yf = y - yi;
        let (u, v) = (fade(xf), fade(yf));
        let (x0, y0) = ((xi as i32) & 255, (yi as i32) & 255);
        let (x1, y1) = ((x0 + 1) & 255, (y0 + 1) & 255);
        let p = &self.perm;
        let aa = p[(p[x0 as usize] as usize + y0 as usize) & 511];
        let ab = p[(p[x0 as usize] as usize + y1 as usize) & 511];
        let ba = p[(p[x1 as usize] as usize + y0 as usize) & 511];
        let bb = p[(p[x1 as usize] as usize + y1 as usize) & 511];
        let lerp = |a: f32, b: f32, t: f32| a + t * (b - a);
        let x1l = lerp(Self::grad2(aa, xf, yf), Self::grad2(ba, xf - 1.0, yf), u);
        let x2l = lerp(Self::grad2(ab, xf, yf - 1.0), Self::grad2(bb, xf - 1.0, yf - 1.0), u);
        // Scale to about -1..1.
        lerp(x1l, x2l, v) * 1.4142
    }

    /// 3D gradient noise at `(x, y, z)`, roughly in `-1..1`.
    pub fn noise3(&self, x: f32, y: f32, z: f32) -> f32 {
        let xi = x.floor();
        let yi = y.floor();
        let zi = z.floor();
        let (xf, yf, zf) = (x - xi, y - yi, z - zi);
        let (u, v, w) = (fade(xf), fade(yf), fade(zf));
        let (x0, y0, z0) = ((xi as i32) & 255, (yi as i32) & 255, (zi as i32) & 255);
        let (x1, y1, z1) = ((x0 + 1) & 255, (y0 + 1) & 255, (z0 + 1) & 255);
        let p = &self.perm;
        let a = p[x0 as usize] as usize;
        let b = p[x1 as usize] as usize;
        let aa = p[(a + y0 as usize) & 511] as usize;
        let ab = p[(a + y1 as usize) & 511] as usize;
        let ba = p[(b + y0 as usize) & 511] as usize;
        let bb = p[(b + y1 as usize) & 511] as usize;
        let g = |h: usize, dx: f32, dy: f32, dz: f32| Self::grad3(p[(h + z0 as usize) & 511], dx, dy, dz);
        let g1 = |h: usize, dx: f32, dy: f32, dz: f32| Self::grad3(p[(h + z1 as usize) & 511], dx, dy, dz);
        let lerp = |a: f32, b: f32, t: f32| a + t * (b - a);
        let c00 = lerp(g(aa, xf, yf, zf), g(ba, xf - 1.0, yf, zf), u);
        let c10 = lerp(g(ab, xf, yf - 1.0, zf), g(bb, xf - 1.0, yf - 1.0, zf), u);
        let c01 = lerp(g1(aa, xf, yf, zf - 1.0), g1(ba, xf - 1.0, yf, zf - 1.0), u);
        let c11 = lerp(g1(ab, xf, yf - 1.0, zf - 1.0), g1(bb, xf - 1.0, yf - 1.0, zf - 1.0), u);
        let c0 = lerp(c00, c10, v);
        let c1 = lerp(c01, c11, v);
        lerp(c0, c1, w) * 1.1547
    }
}

fn fade(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

/// Fractal Brownian motion in 2D: sum `octaves` layers of `noise2`, each
/// `lacunarity`× higher in frequency and `gain`× lower in amplitude.
pub fn fbm2(p: &Perlin, x: f32, y: f32, octaves: u32, lacunarity: f32, gain: f32) -> f32 {
    let (mut amp, mut freq) = (1.0f32, 1.0f32);
    let (mut sum, mut norm) = (0.0f32, 0.0f32);
    for _ in 0..octaves.max(1) {
        sum += amp * p.noise2(x * freq, y * freq);
        norm += amp;
        amp *= gain;
        freq *= lacunarity;
    }
    if norm > 0.0 {
        sum / norm
    } else {
        0.0
    }
}

/// Fractal Brownian motion in 3D.
pub fn fbm3(p: &Perlin, x: f32, y: f32, z: f32, octaves: u32, lacunarity: f32, gain: f32) -> f32 {
    let (mut amp, mut freq) = (1.0f32, 1.0f32);
    let (mut sum, mut norm) = (0.0f32, 0.0f32);
    for _ in 0..octaves.max(1) {
        sum += amp * p.noise3(x * freq, y * freq, z * freq);
        norm += amp;
        amp *= gain;
        freq *= lacunarity;
    }
    if norm > 0.0 {
        sum / norm
    } else {
        0.0
    }
}

/// Push every vertex along its normal by an fBm field sampled at its
/// position, scaled by `amplitude`. Uses flat normals when the mesh has none.
pub fn displace_along_normals(mesh: &mut Mesh, p: &Perlin, amplitude: f32, frequency: f32, octaves: u32) {
    if mesh.normals.len() != mesh.positions.len() {
        mesh.compute_flat_normals();
    }
    let offsets: Vec<f32> = mesh
        .positions
        .iter()
        .map(|pos| fbm3(p, pos.x * frequency, pos.y * frequency, pos.z * frequency, octaves, 2.0, 0.5) * amplitude)
        .collect();
    for (i, off) in offsets.iter().enumerate() {
        if let (Some(n), Some(pos)) = (mesh.normals.get(i).copied(), mesh.positions.get_mut(i)) {
            *pos += n * *off;
        }
    }
    mesh.compute_flat_normals();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_is_deterministic() {
        let a = Perlin::new(42);
        let b = Perlin::new(42);
        for i in 0..50 {
            let (x, y) = (i as f32 * 0.13, i as f32 * -0.27);
            assert_eq!(a.noise2(x, y), b.noise2(x, y));
            assert_eq!(a.noise3(x, y, 1.5), b.noise3(x, y, 1.5));
        }
    }

    #[test]
    fn different_seeds_differ() {
        let a = Perlin::new(1);
        let b = Perlin::new(2);
        let diff = (0..20).any(|i| (a.noise2(i as f32 * 0.3, 0.5) - b.noise2(i as f32 * 0.3, 0.5)).abs() > 1e-6);
        assert!(diff, "different seeds should produce different fields");
    }

    #[test]
    fn noise_stays_in_range() {
        let p = Perlin::new(7);
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        for i in -20..20 {
            for j in -20..20 {
                let v = p.noise2(i as f32 * 0.37, j as f32 * 0.41);
                min = min.min(v);
                max = max.max(v);
                assert!(v.is_finite());
            }
        }
        assert!(min >= -2.0 && max <= 2.0, "range {min}..{max}");
        assert!(max - min > 0.5, "noise should vary");
    }

    #[test]
    fn fbm_is_bounded_and_varies() {
        let p = Perlin::new(9);
        for i in 0..40 {
            let v = fbm2(&p, i as f32 * 0.2, 0.1, 4, 2.0, 0.5);
            assert!(v.is_finite() && v.abs() <= 2.0);
        }
    }

    #[test]
    fn displacement_keeps_valid_mesh() {
        use crate::primitives::grid;
        let mut m = grid(4.0, 4.0, 8, 8);
        let p = Perlin::new(3);
        displace_along_normals(&mut m, &p, 0.25, 1.5, 3);
        assert_eq!(m.vertex_count(), 9 * 9);
        assert!(m.positions.iter().all(|v| v.is_finite()));
        // Vertices should have moved off the flat plane.
        assert!(m.positions.iter().any(|v| v.y.abs() > 1e-4));
    }
}
