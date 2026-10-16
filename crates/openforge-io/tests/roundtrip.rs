//! End-to-end format round-trip tests through the real filesystem.

use openforge_core::mesh::Mesh;
use openforge_core::Vec3;
use openforge_io::{export_path, import_path, MeshData};
use std::fs;

use std::sync::atomic::{AtomicU64, Ordering};

/// Monotonic counter so concurrent tests never share a directory, even within
/// the same clock tick.
static DIR_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Create a fresh, guaranteed-unique temp directory.
///
/// A timestamp alone is not enough: on fast machines two parallel tests can
/// read the same clock value, land on the same path, and then one test's
/// cleanup deletes the other's files mid-read. We combine an atomic counter
/// with a create-or-retry loop so `create_dir` (which fails if the directory
/// already exists) is the source of truth for uniqueness.
fn temp_dir() -> std::path::PathBuf {
    let base = std::env::temp_dir();
    let pid = std::process::id();
    for attempt in 0..u32::MAX {
        let n = DIR_COUNTER.fetch_add(1, Ordering::Relaxed);
        let candidate = base.join(format!("openforge_test_{}_{}", pid, n));
        // `create_dir` errors if it already exists — exactly the guarantee we
        // want. Retry with the next counter value on any collision.
        if fs::create_dir(&candidate).is_ok() {
            return candidate;
        }
        let _ = attempt;
    }
    // Unreachable in practice; fall back to a create_all path.
    let fallback = base.join(format!("openforge_test_{}_{}", pid, DIR_COUNTER.load(Ordering::Relaxed)));
    fs::create_dir_all(&fallback).expect("create temp dir");
    fallback
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
fn round_trip_dae() {
    round_trip("dae")
}

#[test]
fn round_trip_off() {
    round_trip("off")
}

#[test]
fn unknown_extension_errors() {
    let dir = temp_dir();
    let path = dir.join("mesh.xyz");
    fs::write(&path, b"junk").unwrap();
    assert!(import_path(&path).is_err());
    let _ = fs::remove_dir_all(&dir);
}
