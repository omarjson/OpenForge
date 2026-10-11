//! Stanford PLY import/export (ASCII and binary little-endian).
//!
//! Reads vertex `x y z` plus optional `nx ny nz` / `s t` (uv) and builds faces
//! from a vertex-index list. Writes the same layout.

use crate::{finalize_mesh, IoError, IoResult, LoadedScene, MeshData};
use openforge_core::mesh::Mesh;
use openforge_core::{Vec2, Vec3};
use std::fs;
use std::path::Path;

pub fn import(path: &Path) -> IoResult<LoadedScene> {
    let bytes = fs::read(path)?;
    import_bytes(&bytes)
}

pub fn import_bytes(bytes: &[u8]) -> IoResult<LoadedScene> {
    // Find the end of the header.
    let header_end = find_header_end(bytes).ok_or(IoError::Parse {
        format: "ply",
        message: "missing end_header".into(),
    })?;
    let header = std::str::from_utf8(&bytes[..header_end])
        .map_err(|_| IoError::Parse { format: "ply", message: "non-utf8 header".into() })?;

    let mut binary_le = false;
    let mut vertex_count = 0usize;
    let mut face_count = 0usize;
    // Property layout for the vertex element.
    let mut vertex_props: Vec<String> = Vec::new();
    let mut current_element = String::new();

    for line in header.lines() {
        let line = line.trim();
        let mut it = line.split_whitespace();
        match it.next() {
            Some("format") => {
                if let Some(f) = it.next() {
                    binary_le = f == "binary_little_endian";
                }
            }
            Some("element") => {
                current_element = it.next().unwrap_or("").to_string();
                let count: usize = it.next().unwrap_or("0").parse().unwrap_or(0);
                match current_element.as_str() {
                    "vertex" => vertex_count = count,
                    "face" => face_count = count,
                    _ => {}
                }
            }
            Some("property") => {
                if current_element == "vertex" {
                    // "property <type> <name>" (skip list properties, which have 4 tokens).
                    let toks: Vec<&str> = line.split_whitespace().collect();
                    if toks.len() == 3 {
                        vertex_props.push(toks[2].to_string());
                    }
                }
            }
            _ => {}
        }
    }

    let body = &bytes[header_end..];
    let mesh = if binary_le {
        parse_binary_body(body, vertex_count, face_count, &vertex_props)?
    } else {
        parse_ascii_body(body, vertex_count, face_count, &vertex_props)?
    };

    Ok(LoadedScene {
        meshes: vec![MeshData { name: "mesh".into(), mesh: finalize_mesh(mesh), material: None }],
        materials: Vec::new(),
    })
}

fn find_header_end(bytes: &[u8]) -> Option<usize> {
    let marker = b"end_header";
    let mut i = 0;
    while i + marker.len() <= bytes.len() {
        if &bytes[i..i + marker.len()] == marker {
            // Skip to end of that line.
            let mut j = i + marker.len();
            while j < bytes.len() && bytes[j] != b'\n' {
                j += 1;
            }
            return Some((j + 1).min(bytes.len()));
        }
        i += 1;
    }
    None
}

fn prop_index(props: &[String], name: &str) -> Option<usize> {
    props.iter().position(|p| p == name)
}

fn parse_ascii_body(
    body: &[u8],
    vertex_count: usize,
    face_count: usize,
    props: &[String],
) -> IoResult<Mesh> {
    let text = std::str::from_utf8(body)
        .map_err(|_| IoError::Parse { format: "ply", message: "non-utf8 body".into() })?;
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let mut mesh = Mesh::new();
    let ix = prop_index(props, "x");
    let iy = prop_index(props, "y");
    let iz = prop_index(props, "z");
    let inx = prop_index(props, "nx");
    let iny = prop_index(props, "ny");
    let inz = prop_index(props, "nz");
    let is = prop_index(props, "s").or_else(|| prop_index(props, "u"));
    let it = prop_index(props, "t").or_else(|| prop_index(props, "v"));

    for _ in 0..vertex_count {
        let line = lines.next().ok_or(IoError::Parse { format: "ply", message: "short vertex data".into() })?;
        let nums: Vec<f32> = line
            .split_whitespace()
            .filter_map(|s| s.parse::<f32>().ok())
            .collect();
        let (ix, iy, iz) = (ix.ok_or_parse()?, iy.ok_or_parse()?, iz.ok_or_parse()?);
        mesh.add_vertex(Vec3::new(nums[ix], nums[iy], nums[iz]));
        if let (Some(a), Some(b), Some(c)) = (inx, iny, inz) {
            if nums.len() > c {
                mesh.normals.push(Vec3::new(nums[a], nums[b], nums[c]));
            }
        }
        if let (Some(a), Some(b)) = (is, it) {
            if nums.len() > b {
                mesh.uvs.push(Vec2::new(nums[a], nums[b]));
            }
        }
    }
    mesh.ensure_attributes();
    // Fix normal/uv corner indices to point at per-vertex attributes when present.
    for _ in 0..face_count {
        let line = lines.next().ok_or(IoError::Parse { format: "ply", message: "short face data".into() })?;
        let mut it = line.split_whitespace();
        let n: usize = it.next().unwrap_or("0").parse().unwrap_or(0);
        let idx: Vec<u32> = it.filter_map(|s| s.parse::<u32>().ok()).take(n).collect();
        if idx.len() >= 3 {
            mesh.add_face(&idx);
        }
    }
    // Map corners: if normals/uvs are per-vertex, index them by position.
    if mesh.normals.len() == mesh.positions.len() && mesh.normals.len() > 1 {
        for f in mesh.faces.iter_mut() {
            for c in f.corners.iter_mut() {
                c.normal = c.position;
            }
        }
    }
    if mesh.uvs.len() == mesh.positions.len() && mesh.uvs.len() > 1 {
        for f in mesh.faces.iter_mut() {
            for c in f.corners.iter_mut() {
                c.uv = c.position;
            }
        }
    }
    Ok(mesh)
}

// Small helper trait to convert a missing property into a parse error.
trait OkOrParse<T> {
    fn ok_or_parse(self) -> IoResult<T>;
}
impl<T> OkOrParse<T> for Option<T> {
    fn ok_or_parse(self) -> IoResult<T> {
        self.ok_or_else(|| IoError::Parse { format: "ply", message: "missing x/y/z property".into() })
    }
}

fn parse_binary_body(
    body: &[u8],
    vertex_count: usize,
    face_count: usize,
    props: &[String],
) -> IoResult<Mesh> {
    use openforge_core::mesh::Corner;
    // Determine vertex stride from property types (assume all float32 for x/y/z/n/s/t).
    let mut mesh = Mesh::new();
    let mut cursor = 0usize;
    let read_f32 = |buf: &[u8], cur: &mut usize| -> IoResult<f32> {
        if *cur + 4 > buf.len() {
            return Err(IoError::Parse { format: "ply", message: "truncated binary".into() });
        }
        let v = f32::from_le_bytes([buf[*cur], buf[*cur + 1], buf[*cur + 2], buf[*cur + 3]]);
        *cur += 4;
        Ok(v)
    };
    // For binary we assume each vertex property is float32 (common case).
    let stride_props = props.len();
    for _ in 0..vertex_count {
        let mut vals = Vec::with_capacity(stride_props);
        for _ in 0..stride_props {
            vals.push(read_f32(body, &mut cursor)?);
        }
        let ix = prop_index(props, "x").unwrap_or(0);
        let iy = prop_index(props, "y").unwrap_or(1);
        let iz = prop_index(props, "z").unwrap_or(2);
        mesh.add_vertex(Vec3::new(vals[ix], vals[iy], vals[iz]));
        if let (Some(a), Some(b), Some(c)) =
            (prop_index(props, "nx"), prop_index(props, "ny"), prop_index(props, "nz"))
        {
            if vals.len() > c {
                mesh.normals.push(Vec3::new(vals[a], vals[b], vals[c]));
            }
        }
        if let (Some(a), Some(b)) = (prop_index(props, "s"), prop_index(props, "t")) {
            if vals.len() > b {
                mesh.uvs.push(Vec2::new(vals[a], vals[b]));
            }
        }
    }
    mesh.ensure_attributes();
    for _ in 0..face_count {
        if cursor >= body.len() {
            return Err(IoError::Parse { format: "ply", message: "truncated faces".into() });
        }
        let n = body[cursor] as usize;
        cursor += 1;
        let mut idx = Vec::with_capacity(n);
        for _ in 0..n {
            if cursor + 4 > body.len() {
                return Err(IoError::Parse { format: "ply", message: "truncated face index".into() });
            }
            let v = u32::from_le_bytes([body[cursor], body[cursor + 1], body[cursor + 2], body[cursor + 3]]);
            cursor += 4;
            idx.push(v);
        }
        if idx.len() >= 3 {
            let corners: Vec<Corner> = idx
                .iter()
                .map(|&i| Corner { position: i, normal: i.min(mesh.normals.len() as u32 - 1), uv: i.min(mesh.uvs.len() as u32 - 1) })
                .collect();
            mesh.faces.push(openforge_core::mesh::Face { corners, material: 0 });
        }
    }
    Ok(mesh)
}

pub fn export(path: &Path, data: &MeshData) -> IoResult<()> {
    fs::write(path, export_str(data))?;
    Ok(())
}

pub fn export_str(data: &MeshData) -> String {
    let m = &data.mesh;
    let has_n = !m.normals.is_empty();
    let has_uv = !m.uvs.is_empty();
    let mut out = String::new();
    out.push_str("ply\nformat ascii 1.0\ncomment OpenForge\n");
    out.push_str(&format!("element vertex {}\n", m.positions.len()));
    out.push_str("property float x\nproperty float y\nproperty float z\n");
    if has_n {
        out.push_str("property float nx\nproperty float ny\nproperty float nz\n");
    }
    if has_uv {
        out.push_str("property float s\nproperty float t\n");
    }
    out.push_str(&format!("element face {}\n", m.faces.len()));
    out.push_str("property list uchar int vertex_indices\nend_header\n");
    for (i, p) in m.positions.iter().enumerate() {
        out.push_str(&format!("{} {} {}", p.x, p.y, p.z));
        if has_n {
            let n = m.normals.get(i).copied().unwrap_or(Vec3::Y);
            out.push_str(&format!(" {} {} {}", n.x, n.y, n.z));
        }
        if has_uv {
            let t = m.uvs.get(i).copied().unwrap_or(Vec2::ZERO);
            out.push_str(&format!(" {} {}", t.x, t.y));
        }
        out.push('\n');
    }
    for f in &m.faces {
        out.push_str(&format!("{}", f.corners.len()));
        for c in &f.corners {
            out.push_str(&format!(" {}", c.position));
        }
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ascii_round_trip() {
        let mut mesh = Mesh::new();
        mesh.add_vertex(Vec3::new(0.0, 0.0, 0.0));
        mesh.add_vertex(Vec3::new(1.0, 0.0, 0.0));
        mesh.add_vertex(Vec3::new(0.0, 1.0, 0.0));
        mesh.add_face(&[0, 1, 2]);
        mesh.compute_smooth_normals();
        let data = MeshData { name: "tri".into(), mesh, material: None };
        let text = export_str(&data);
        let loaded = import_bytes(text.as_bytes()).unwrap();
        assert_eq!(loaded.meshes[0].mesh.vertex_count(), 3);
        assert_eq!(loaded.meshes[0].mesh.face_count(), 1);
    }
}
