//! Wavefront OBJ import/export.
//!
//! Handles `v`, `vt`, `vn`, `f`, `g`/`o` (grouping) and `usemtl`. Face
//! corners may be `v`, `v/t`, `v//n` or `v/t/n` and negative (relative)
//! indices are supported.

use crate::{finalize_mesh, IoError, IoResult, LoadedScene, MeshData};
use openforge_core::mesh::{Corner, Face, Mesh};
use openforge_core::{Vec2, Vec3};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

pub fn import(path: &Path) -> IoResult<LoadedScene> {
    let text = fs::read_to_string(path)?;
    import_str(&text)
}

pub fn import_str(text: &str) -> IoResult<LoadedScene> {
    let mut positions: Vec<Vec3> = Vec::new();
    let mut normals: Vec<Vec3> = Vec::new();
    let mut uvs: Vec<Vec2> = Vec::new();

    let mut meshes: Vec<MeshData> = Vec::new();
    let mut materials: Vec<String> = Vec::new();

    // Current open mesh being filled.
    let mut current_name = String::from("default");
    let mut current_material: Option<String> = None;
    let mut current = Mesh::new();
    // De-dup (pos,uv,normal) corners into a single index per mesh.
    let mut corner_map: HashMap<(i64, i64, i64), u32> = HashMap::new();
    let mut has_geometry = false;

    for (lineno, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut parts = line.split_whitespace();
        let tag = parts.next().unwrap_or("");
        match tag {
            "v" => {
                let v = parse_vec3(parts, lineno, "v")?;
                positions.push(v);
            }
            "vn" => {
                let v = parse_vec3(parts, lineno, "vn")?;
                normals.push(v);
            }
            "vt" => {
                let vals = parse_f32_list(&mut parts, 2, lineno, "vt")?;
                uvs.push(Vec2::new(vals[0], vals[1]));
            }
            "g" | "o" => {
                if has_geometry {
                    meshes.push(finish(&current, &current_name, current_material.clone()));
                    current = Mesh::new();
                    corner_map.clear();
                    has_geometry = false;
                }
                current_name = parts.next().unwrap_or("default").to_string();
            }
            "usemtl" => {
                let name = parts.next().unwrap_or("default").to_string();
                if !materials.contains(&name) {
                    materials.push(name.clone());
                }
                current_material = Some(name);
            }
            "f" => {
                let mut idxs: Vec<u32> = Vec::new();
                for tok in parts {
                    let key = parse_corner(tok, positions.len(), normals.len(), uvs.len(), lineno)?;
                    let ci = *corner_map.entry(key).or_insert_with(|| {
                        let p = if key.0 < 0 { (positions.len() as i64 + key.0) as usize } else { (key.0 - 1) as usize };
                        let n = if key.1 < 0 { (normals.len() as i64 + key.1) as usize } else { (key.1 - 1) as usize };
                        let t = if key.2 < 0 { (uvs.len() as i64 + key.2) as usize } else { (key.2 - 1) as usize };
                        let idx = current.positions.len() as u32;
                        current.positions.push(positions[p]);
                        if key.1 != 0 && n < normals.len() {
                            current.normals.push(normals[n]);
                        }
                        if key.2 != 0 && t < uvs.len() {
                            current.uvs.push(uvs[t]);
                        }
                        idx
                    });
                    idxs.push(ci);
                }
                if idxs.len() >= 3 {
                    let corners: Vec<Corner> = idxs
                        .iter()
                        .map(|&i| {
                            let p = &current.positions[i as usize];
                            let _ = p;
                            Corner {
                                position: i,
                                normal: if current.normals.is_empty() { 0 } else { i.min(current.normals.len() as u32 - 1) },
                                uv: if current.uvs.is_empty() { 0 } else { i.min(current.uvs.len() as u32 - 1) },
                            }
                        })
                        .collect();
                    current.faces.push(Face { corners, material: 0 });
                    has_geometry = true;
                }
            }
            _ => {}
        }
    }
    if has_geometry {
        meshes.push(finish(&current, &current_name, current_material));
    }

    if meshes.is_empty() {
        return Err(IoError::Parse { format: "obj", message: "no faces found".into() });
    }
    Ok(LoadedScene { meshes, materials })
}

fn finish(mesh: &Mesh, name: &str, material: Option<String>) -> MeshData {
    let mut m = mesh.clone();
    if m.normals.is_empty() {
        m.compute_smooth_normals();
    }
    MeshData { name: name.to_string(), mesh: finalize_mesh(m), material }
}

fn parse_vec3<'a>(mut parts: impl Iterator<Item = &'a str>, lineno: usize, tag: &str) -> IoResult<Vec3> {
    let vals = parse_f32_list(&mut parts, 3, lineno, tag)?;
    Ok(Vec3::new(vals[0], vals[1], vals[2]))
}

fn parse_f32_list<'a>(
    parts: &mut impl Iterator<Item = &'a str>,
    n: usize,
    lineno: usize,
    tag: &str,
) -> IoResult<Vec<f32>> {
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        match parts.next() {
            Some(s) => out.push(s.parse::<f32>().map_err(|_| IoError::Parse {
                format: "obj",
                message: format!("line {}: bad number in {}", lineno + 1, tag),
            })?),
            None => {
                return Err(IoError::Parse {
                    format: "obj",
                    message: format!("line {}: {} needs {} numbers", lineno + 1, tag, n),
                })
            }
        }
    }
    Ok(out)
}

/// Parse a face-corner token into `(position, uv, normal)` 0-based-or-relative
/// keys. A zero means "absent".
fn parse_corner(tok: &str, _np: usize, _nn: usize, _nu: usize, lineno: usize) -> IoResult<(i64, i64, i64)> {
    let mut it = tok.split('/');
    let p = it.next().unwrap_or("").parse::<i64>().map_err(|_| IoError::Parse {
        format: "obj",
        message: format!("line {}: bad face index", lineno + 1),
    })?;
    let t = match it.next() {
        Some("") | None => 0,
        Some(s) => s.parse::<i64>().map_err(|_| IoError::Parse {
            format: "obj",
            message: format!("line {}: bad uv index", lineno + 1),
        })?,
    };
    let n = match it.next() {
        Some("") | None => 0,
        Some(s) => s.parse::<i64>().map_err(|_| IoError::Parse {
            format: "obj",
            message: format!("line {}: bad normal index", lineno + 1),
        })?,
    };
    Ok((p, t, n))
}

pub fn export(path: &Path, data: &MeshData) -> IoResult<()> {
    let text = export_str(data);
    fs::write(path, text)?;
    Ok(())
}

pub fn export_str(data: &MeshData) -> String {
    let m = &data.mesh;
    let mut out = String::new();
    out.push_str("# Exported by OpenForge\n");
    out.push_str(&format!("o {}\n", data.name));
    if let Some(mat) = &data.material {
        out.push_str(&format!("usemtl {}\n", mat));
    }
    for p in &m.positions {
        out.push_str(&format!("v {} {} {}\n", p.x, p.y, p.z));
    }
    for uv in &m.uvs {
        out.push_str(&format!("vt {} {}\n", uv.x, uv.y));
    }
    for n in &m.normals {
        out.push_str(&format!("vn {} {} {}\n", n.x, n.y, n.z));
    }
    let has_uv = !m.uvs.is_empty();
    let has_n = !m.normals.is_empty();
    for face in &m.faces {
        out.push('f');
        for c in &face.corners {
            let p = c.position + 1;
            match (has_uv, has_n) {
                (true, true) => out.push_str(&format!(" {}/{}/{}", p, c.uv + 1, c.normal + 1)),
                (true, false) => out.push_str(&format!(" {}/{}", p, c.uv + 1)),
                (false, true) => out.push_str(&format!(" {}//{}", p, c.normal + 1)),
                (false, false) => out.push_str(&format!(" {}", p)),
            }
        }
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_quad() {
        let mut mesh = Mesh::new();
        mesh.add_vertex(Vec3::new(0.0, 0.0, 0.0));
        mesh.add_vertex(Vec3::new(1.0, 0.0, 0.0));
        mesh.add_vertex(Vec3::new(1.0, 1.0, 0.0));
        mesh.add_vertex(Vec3::new(0.0, 1.0, 0.0));
        mesh.add_face(&[0, 1, 2, 3]);
        mesh.compute_smooth_normals();
        let data = MeshData { name: "quad".into(), mesh, material: None };

        let text = export_str(&data);
        let loaded = import_str(&text).unwrap();
        assert_eq!(loaded.mesh_count(), 1);
        let m = &loaded.meshes[0].mesh;
        assert_eq!(m.vertex_count(), 4);
        assert_eq!(m.face_count(), 1);
    }

    #[test]
    fn negative_indices_supported() {
        let text = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf -3 -2 -1\n";
        let loaded = import_str(text).unwrap();
        assert_eq!(loaded.meshes[0].mesh.face_count(), 1);
    }
}
