//! STL import/export (binary and ASCII).
//!
//! STL stores flat triangles only, so imported meshes are triangulated and
//! carry per-face normals (no UVs).

use crate::{finalize_mesh, IoError, IoResult, LoadedScene, MeshData};
use openforge_core::mesh::{Corner, Face, Mesh};
use openforge_core::Vec3;
use std::fs;
use std::path::Path;

pub fn import(path: &Path) -> IoResult<LoadedScene> {
    let bytes = fs::read(path)?;
    import_bytes(&bytes)
}

pub fn import_bytes(bytes: &[u8]) -> IoResult<LoadedScene> {
    // Heuristic: if it begins with "solid" AND contains "facet", treat as ASCII.
    let looks_ascii = bytes.starts_with(b"solid") && find_ascii_marker(bytes);
    let mesh = if looks_ascii {
        parse_ascii(bytes)?
    } else {
        parse_binary(bytes)?
    };
    Ok(LoadedScene {
        meshes: vec![MeshData { name: "mesh".into(), mesh: finalize_mesh(mesh), material: None }],
        materials: Vec::new(),
    })
}

fn find_ascii_marker(bytes: &[u8]) -> bool {
    // Search the first 1KB for "facet normal".
    let window = &bytes[..bytes.len().min(1024)];
    window.windows(12).any(|w| w == b"facet normal")
}

fn parse_binary(bytes: &[u8]) -> IoResult<Mesh> {
    if bytes.len() < 84 {
        return Err(IoError::Parse { format: "stl", message: "too short for binary STL".into() });
    }
    let tri_count = u32::from_le_bytes([bytes[80], bytes[81], bytes[82], bytes[83]]) as usize;
    let expected = 84 + tri_count * 50;
    if bytes.len() < expected {
        return Err(IoError::Parse {
            format: "stl",
            message: format!("expected {} bytes for {} triangles, got {}", expected, tri_count, bytes.len()),
        });
    }
    let mut mesh = Mesh::new();
    let mut cursor = 84usize;
    for _ in 0..tri_count {
        let normal = read_vec3(bytes, &mut cursor);
        let a = read_vec3(bytes, &mut cursor);
        let b = read_vec3(bytes, &mut cursor);
        let c = read_vec3(bytes, &mut cursor);
        cursor += 2; // attribute byte count
        let n = if normal.length_squared() > 0.0 { normal.normalize() } else { Vec3::Y };
        let ni = mesh.normals.len() as u32;
        mesh.normals.push(n);
        let ia = mesh.add_vertex(a);
        let ib = mesh.add_vertex(b);
        let ic = mesh.add_vertex(c);
        mesh.faces.push(Face {
            corners: vec![
                Corner { position: ia, normal: ni, uv: 0 },
                Corner { position: ib, normal: ni, uv: 0 },
                Corner { position: ic, normal: ni, uv: 0 },
            ],
            material: 0,
        });
    }
    mesh.uvs.push(openforge_core::Vec2::ZERO);
    Ok(mesh)
}

fn read_vec3(bytes: &[u8], cursor: &mut usize) -> Vec3 {
    let mut read = || {
        let v = f32::from_le_bytes([bytes[*cursor], bytes[*cursor + 1], bytes[*cursor + 2], bytes[*cursor + 3]]);
        *cursor += 4;
        v
    };
    Vec3::new(read(), read(), read())
}

fn parse_ascii(bytes: &[u8]) -> IoResult<Mesh> {
    let text = std::str::from_utf8(bytes)
        .map_err(|_| IoError::Parse { format: "stl", message: "non-utf8 ascii STL".into() })?;
    let mut mesh = Mesh::new();
    let mut normal = Vec3::Y;
    let mut verts: Vec<Vec3> = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("facet normal") {
            let nums: Vec<f32> = rest.split_whitespace().filter_map(|s| s.parse().ok()).collect();
            if nums.len() == 3 {
                normal = Vec3::new(nums[0], nums[1], nums[2]);
            }
            verts.clear();
        } else if let Some(rest) = line.strip_prefix("vertex") {
            let nums: Vec<f32> = rest.split_whitespace().filter_map(|s| s.parse().ok()).collect();
            if nums.len() == 3 {
                verts.push(Vec3::new(nums[0], nums[1], nums[2]));
            }
        } else if line.starts_with("endfacet") {
            if verts.len() >= 3 {
                let n = if normal.length_squared() > 0.0 { normal.normalize() } else { Vec3::Y };
                let ni = mesh.normals.len() as u32;
                mesh.normals.push(n);
                let idx: Vec<u32> = verts.iter().map(|v| mesh.add_vertex(*v)).collect();
                mesh.faces.push(Face {
                    corners: idx.iter().map(|&p| Corner { position: p, normal: ni, uv: 0 }).collect(),
                    material: 0,
                });
            }
        }
    }
    mesh.uvs.push(openforge_core::Vec2::ZERO);
    if mesh.faces.is_empty() {
        return Err(IoError::Parse { format: "stl", message: "no facets found".into() });
    }
    Ok(mesh)
}

pub fn export(path: &Path, data: &MeshData) -> IoResult<()> {
    fs::write(path, export_binary(data))?;
    Ok(())
}

/// Export as binary STL (compact, universally supported).
pub fn export_binary(data: &MeshData) -> Vec<u8> {
    let tris = data.mesh.triangulate();
    let mut out = Vec::with_capacity(84 + tris.len() * 50);
    let mut header = [0u8; 80];
    let tag = b"OpenForge binary STL";
    header[..tag.len()].copy_from_slice(tag);
    out.extend_from_slice(&header);
    out.extend_from_slice(&(tris.len() as u32).to_le_bytes());
    for t in &tris {
        let a = data.mesh.positions[t[0] as usize];
        let b = data.mesh.positions[t[1] as usize];
        let c = data.mesh.positions[t[2] as usize];
        let n = (b - a).cross(c - a).normalize_or_zero();
        push_vec3(&mut out, n);
        push_vec3(&mut out, a);
        push_vec3(&mut out, b);
        push_vec3(&mut out, c);
        out.extend_from_slice(&[0u8, 0]); // attribute byte count
    }
    out
}

fn push_vec3(out: &mut Vec<u8>, v: Vec3) {
    out.extend_from_slice(&v.x.to_le_bytes());
    out.extend_from_slice(&v.y.to_le_bytes());
    out.extend_from_slice(&v.z.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_round_trip() {
        let mut mesh = Mesh::new();
        mesh.add_vertex(Vec3::new(0.0, 0.0, 0.0));
        mesh.add_vertex(Vec3::new(1.0, 0.0, 0.0));
        mesh.add_vertex(Vec3::new(0.0, 1.0, 0.0));
        mesh.add_face(&[0, 1, 2]);
        let data = MeshData { name: "tri".into(), mesh, material: None };
        let bytes = export_binary(&data);
        let loaded = import_bytes(&bytes).unwrap();
        assert_eq!(loaded.meshes[0].mesh.face_count(), 1);
        assert_eq!(loaded.meshes[0].mesh.vertex_count(), 3);
    }
}
