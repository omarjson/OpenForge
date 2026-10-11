//! End-to-end format round-trip tests through the real filesystem.

use openforge_core::mesh::Mesh;
use openforge_core::Vec3;
use openforge_io::{export_path, import_path, MeshData};
use std::fs;

fn temp_dir() -> std::path::PathBuf {
    let mut d = std::env::temp_dir();
    // Unique per call so parallel tests don't clobber each other.
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    d.push(format!("openforge_test_{}_{}", std::process::id(), nanos));
    fs::create_dir_all(&d).expect("create temp dir");
    d
}

fn sample_mesh() -> Mesh {
    let mut m = Mesh::new();
    // A unit tetrahedron.
    m.add_vertex(Vec3::new(0.0, 0.0, 0.0));
    m.add_vertex(Vec3::new(1.0, 0.0, 0.0));
    m.add_vertex(Vec3::new(0.0, 1.0, 0.0));
    m.add_vertex(Vec3::new(0.0, 0.0, 1.0));
    m.add_face(&[0, 2, 1]);
    m.add_face(&[0, 1, 3]);
    m.add_face(&[0, 3, 2]);
    m.add_face(&[1, 2, 3]);
    m.compute_flat_normals();
    m
}

fn round_trip(ext: &str) {
    let dir = temp_dir();
    let path = dir.join(format!("mesh.{}", ext));
    let original = sample_mesh();
    let data = MeshData { name: "tetra".into(), mesh: original.clone(), material: None };

    export_path(&path, &data).expect("export");
    assert!(path.exists(), "file should exist for {}", ext);

    let loaded = import_path(&path).expect("import");
    assert!(!loaded.meshes.is_empty(), "no meshes loaded for {}", ext);
    let got = &loaded.meshes[0].mesh;

    // Vertex count should match. (STL and PLY may duplicate for seams, so we
    // only require the face/triangle count to be preserved for all formats.)
    let expected_tris = original.triangulated_count();
    assert_eq!(
        got.triangulated_count(),
        expected_tris,
        "triangle count mismatch for {}",
        ext
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn round_trip_obj() {
    round_trip("obj")
}

#[test]
fn round_trip_glb() {
    round_trip("glb")
}

#[test]
fn round_trip_gltf() {
    round_trip("gltf")
}

#[test]
fn round_trip_ply() {
    round_trip("ply")
}

#[test]
fn round_trip_stl() {
    round_trip("stl")
}

#[test]
fn unknown_extension_errors() {
    let dir = temp_dir();
    let path = dir.join("mesh.xyz");
    fs::write(&path, b"junk").unwrap();
    assert!(import_path(&path).is_err());
    let _ = fs::remove_dir_all(&dir);
}
