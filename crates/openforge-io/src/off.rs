//! OFF import/export (Object File Format).
//!
//! OFF is plain text: an `OFF` header, one vertex per line, then one face per
//! line as a corner count followed by vertex indices. Faces may be n-gons and
//! the format carries no normals or UVs (smooth normals are computed on
//! import). Parsing is line-based, so the color values some writers append
//! (`COFF`-style vertices, per-face colors) are ignored rather than fatal.

use crate::{finalize_mesh, IoError, IoResult, LoadedScene, MeshData};
use openforge_core::mesh::Mesh;
use openforge_core::Vec3;
use std::fs;
use std::path::Path;

fn parse_err(message: impl Into<String>) -> IoError {
    IoError::Parse { format: "off", message: message.into() }
}

pub fn import(path: &Path) -> IoResult<LoadedScene> {
    let text = fs::read_to_string(path)?;
    import_str(&text)
}

pub fn import_str(text: &str) -> IoResult<LoadedScene> {
    let mut lines = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'));

    let header = lines.next().ok_or_else(|| parse_err("empty file"))?;
    let mut header = header.split_whitespace();
    let magic = header.next().unwrap_or("");
    if !magic.eq_ignore_ascii_case("off") {
        return Err(parse_err("missing OFF header"));
    }
    // The counts sit on the magic line or on exactly one line of their own.
    let mut counts: Vec<i64> = header.filter_map(|token| token.parse::<i64>().ok()).collect();
    if counts.len() < 3 {
        let line = lines.next().ok_or_else(|| parse_err("truncated header"))?;
        counts.extend(line.split_whitespace().filter_map(|token| token.parse::<i64>().ok()));
    }
    if counts.len() != 3 {
        return Err(parse_err(format!("expected three counts, found {}", counts.len())));
    }
    if counts.iter().any(|&count| count < 0) {
        return Err(parse_err("negative count"));
    }
    let vertex_count = counts[0] as usize;
    let face_count = counts[1] as usize;

    let mut mesh = Mesh::new();
    for vertex in 0..vertex_count {
        let line = lines.next().ok_or_else(|| {
            parse_err(format!("expected {vertex_count} vertices, found {vertex}"))
        })?;
        let mut parts = line.split_whitespace();
        let mut coordinate = || parts.next().and_then(|token| token.parse::<f32>().ok());
        match (coordinate(), coordinate(), coordinate()) {
            (Some(x), Some(y), Some(z)) => {
                mesh.add_vertex(Vec3::new(x, y, z));
            }
            _ => return Err(parse_err(format!("vertex {} is not three numbers", vertex + 1))),
        }
    }

    for face in 0..face_count {
        let line = lines.next().ok_or_else(|| {
            parse_err(format!("expected {face_count} faces, found {face}"))
        })?;
        let mut parts = line.split_whitespace();
        let corners: usize = parts
            .next()
            .and_then(|token| token.parse().ok())
            .ok_or_else(|| parse_err(format!("face {} has no corner count", face + 1)))?;
        if corners < 3 {
            return Err(parse_err(format!(
                "face {} has {corners} corners, need at least 3",
                face + 1
            )));
        }
        let mut indices = Vec::with_capacity(corners);
        for _ in 0..corners {
            let index: u32 = parts
                .next()
                .and_then(|token| token.parse().ok())
                .ok_or_else(|| parse_err(format!("face {} is missing indices", face + 1)))?;
            if index as usize >= vertex_count {
                return Err(parse_err(format!(
                    "face {} names vertex {index} of {vertex_count}",
                    face + 1
                )));
            }
            indices.push(index);
        }
        // add_face keeps the corner order and turns any n-gon into one face.
        mesh.add_face(&indices);
    }

    if mesh.faces.is_empty() {
        return Err(parse_err("no faces"));
    }

    Ok(LoadedScene {
        meshes: vec![MeshData {
            name: "mesh".into(),
            mesh: finalize_mesh(mesh),
            material: None,
        }],
        materials: Vec::new(),
    })
}

pub fn export(path: &Path, data: &MeshData) -> IoResult<()> {
    fs::write(path, export_str(data))?;
    Ok(())
}

pub fn export_str(data: &MeshData) -> String {
    let mesh = &data.mesh;
    let mut out = String::from("OFF\n");
    out.push_str(&format!("{} {} 0\n", mesh.positions.len(), mesh.faces.len()));
    for position in &mesh.positions {
        out.push_str(&format!("{} {} {}\n", position.x, position.y, position.z));
    }
    for face in &mesh.faces {
        out.push_str(&face.corners.len().to_string());
        for corner in &face.corners {
            out.push_str(&format!(" {}", corner.position));
        }
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The refusal a document earns, as a string.
    fn refusal(text: &str) -> String {
        match import_str(text) {
            Ok(loaded) => panic!("expected a refusal, got {} mesh(es)", loaded.mesh_count()),
            Err(error) => error.to_string(),
        }
    }

    #[test]
    fn round_trip_keeps_ngons() {
        let mut mesh = Mesh::new();
        for (x, y) in [(0.0, 0.0), (1.0, 0.0), (1.5, 1.0), (0.5, 1.5), (-0.5, 1.0)] {
            mesh.add_vertex(Vec3::new(x, y, 0.25));
        }
        mesh.add_face(&[0, 1, 2, 3, 4]);
        let data = MeshData { name: "pentagon".into(), mesh, material: None };

        let text = export_str(&data);
        assert!(text.starts_with("OFF\n5 1 0\n"), "{text}");
        let loaded = import_str(&text).unwrap();
        assert_eq!(loaded.mesh_count(), 1);
        let mesh = &loaded.meshes[0].mesh;
        assert_eq!(mesh.vertex_count(), 5);
        assert_eq!(mesh.face_count(), 1);
        assert_eq!(mesh.faces[0].corners.len(), 5, "the pentagon stayed one face");
        assert_eq!(mesh.positions[3], Vec3::new(0.5, 1.5, 0.25));
        assert_eq!(mesh.normals.len(), 5, "normals were computed on import");
    }

    #[test]
    fn comments_counts_and_color_extras_are_tolerated() {
        let text = "# a colored OFF\nOFF 3 1 0\n\
                    0 0 0 255 0 0 255\n\
                    1 0 0 0 255 0 255\n\
                    0 1 0 0 0 255 255\n\
                    3 0 1 2 64 64 64\n";
        let loaded = import_str(text).unwrap();
        let mesh = &loaded.meshes[0].mesh;
        assert_eq!(mesh.vertex_count(), 3);
        assert_eq!(mesh.face_count(), 1);
        assert_eq!(mesh.positions[1], Vec3::new(1.0, 0.0, 0.0));
    }

    #[test]
    fn broken_documents_are_refused() {
        let cases = [
            ("", "empty file"),
            ("PLY\n3 1 0\n", "missing OFF header"),
            ("OFF\n", "truncated header"),
            ("OFF\n2 1\n", "expected three counts, found 2"),
            ("OFF\n8 6 0 1\n", "expected three counts, found 4"),
            ("OFF 3 1 0\n0 0 0\n1 0 0\n", "expected 3 vertices, found 2"),
            ("OFF 3 1 0\n0 0 0\n1 0 0\nnope\n3 0 1 2\n", "not three numbers"),
            ("OFF 3 1 0\n0 0 0\n1 0 0\n0 1 0\n2 0 1\n", "need at least 3"),
            ("OFF 3 1 0\n0 0 0\n1 0 0\n0 1 0\n3 0 1 9\n", "names vertex 9 of 3"),
            ("OFF 3 0 0\n0 0 0\n1 0 0\n0 1 0\n", "no faces"),
        ];
        for (text, message) in cases {
            let error = refusal(text);
            assert!(error.contains(message), "{error:?} should mention {message:?}");
        }
    }
}
