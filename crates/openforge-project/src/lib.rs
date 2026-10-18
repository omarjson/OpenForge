//! The OpenForge project file: your work in one document.
//!
//! A project file is a single JSON document holding what the app has open:
//!
//! - the **scene** — groups, dummies and meshes, with every node's name,
//!   transform, visibility, freeze flag, LOD tag and per-face material slots
//!   (the geometry itself rides along inside the mesh node's kind),
//! - the **material library** — the materials, field for field, and the
//!   textures they point at (RGBA8 pixels, base64-encoded so the file stays
//!   plain readable text), and
//! - the **animation clips** — each clip's tracks, with their targets, keys
//!   and interpolation modes.
//!
//! The document is versioned and self-describing:
//!
//! ```json
//! { "format": "openforge-project", "version": 1, "nodes": [], "materials": [], "textures": [], "clips": [] }
//! ```
//!
//! so a file that is not JSON, a JSON file that is not an OpenForge project,
//! and a project from a newer build are each refused with a sentence that says
//! which one it was. Scene node ids never reach the file: nodes are listed
//! parent-first and references are positions in that list (see [`doc`] for the
//! two places the document deliberately differs from memory).
//!
//! ```no_run
//! use openforge_project::Project;
//!
//! # fn main() -> Result<(), openforge_project::ProjectError> {
//! let project = Project::load("scene.ofproj".as_ref())?;
//! println!("{}", project.summary().describe());
//! # Ok(())
//! # }
//! ```
//!
//! Version 1 files still load — they simply carry no rig. What this build
//! writes is version 2, which adds the skinned rig: the skeleton with its
//! current pose, the per-vertex weights, the bind positions and the mesh
//! node the rig drives. The application's preferences are still not part
//! of the file.

mod base64;
mod doc;

pub use base64::{decode as decode_base64, encode as encode_base64};

use doc::Doc;
use openforge_animation::ClipLibrary;
use openforge_core::scene::Scene;
use openforge_core::{NodeId, Vec3};
use openforge_materials::MaterialLibrary;
use openforge_rig::{Skeleton, SkinWeights};
use std::path::Path;

/// The `format` field every project file starts with.
pub const FORMAT: &str = "openforge-project";
/// The format version this build reads and writes.
pub const FORMAT_VERSION: u32 = 2;
/// The file extension the app's dialogs offer (without the dot).
pub const EXTENSION: &str = "ofproj";

/// Why a project file could not be saved or opened.
#[derive(Debug, thiserror::Error)]
pub enum ProjectError {
    /// The file could not be read or written.
    #[error("could not read or write the file: {0}")]
    Io(#[from] std::io::Error),
    /// The text is not JSON at all (an OBJ or GLB opened as a project, say).
    #[error("the file is not JSON: {0}")]
    Json(#[from] serde_json::Error),
    /// The JSON is fine but the `format` field is not ours.
    #[error("not an OpenForge project file (its format field is {found})")]
    NotAProject { found: String },
    /// The file says it needs a newer OpenForge than this build.
    #[error("the project needs a newer OpenForge: file version {found}, this build writes version {FORMAT_VERSION}")]
    NewerVersion { found: u64 },
    /// The document is ours but broken: a forward parent reference, a clip
    /// driving a node the file does not define, a texture whose pixel count
    /// disagrees with its size, a missing version field.
    #[error("the project file is malformed: {0}")]
    Malformed(String),
}

/// An in-memory document: exactly the pieces a project file holds.
///
/// The clones are deliberate — a save must not disturb the app's live scene,
/// and an open hands the app a document it owns outright.
#[derive(Clone, Debug)]
pub struct Project {
    pub scene: Scene,
    pub materials: MaterialLibrary,
    pub clips: ClipLibrary,
    /// The skinned rig, when the document has one (since version 2).
    pub rig: Option<ProjectRig>,
}

/// The skinned rig a project carries: the skeleton with both its poses, the
/// per-vertex weights, the bind (rest) positions and the node the rig drives.
///
/// The pieces are `openforge-rig`'s own types — the document stores a
/// spelled-out form of them, so nothing here depends on how they serialize.
#[derive(Clone, Debug, PartialEq)]
pub struct ProjectRig {
    pub skeleton: Skeleton,
    pub weights: SkinWeights,
    /// One bind (rest) position per vertex of the driven mesh.
    pub bind: Vec<Vec3>,
    /// The mesh node the rig drives, when it still exists in the scene.
    pub node: Option<NodeId>,
}

impl Project {
    /// Capture the three pieces as a document (with no rig; set
    /// [`Project::rig`] to save one).
    pub fn from_parts(scene: &Scene, materials: &MaterialLibrary, clips: &ClipLibrary) -> Self {
        Self {
            scene: scene.clone(),
            materials: materials.clone(),
            clips: clips.clone(),
            rig: None,
        }
    }

    /// The document as pretty-printed JSON, the exact text [`Project::save`]
    /// writes (with a trailing newline).
    pub fn to_json(&self) -> Result<String, ProjectError> {
        Ok(serde_json::to_string_pretty(&Doc::from_parts(
            &self.scene,
            &self.materials,
            &self.clips,
            self.rig.as_ref(),
        ))?)
    }

    /// Read a document back from JSON text.
    pub fn from_json(text: &str) -> Result<Self, ProjectError> {
        // Two phases: a plain value first, so a file that is valid JSON but
        // not ours is refused by its own field with a clear sentence, rather
        // than by whichever field serde happens to miss first.
        let value: serde_json::Value = serde_json::from_str(text)?;
        let found = match value.get("format") {
            Some(serde_json::Value::String(format)) => format.clone(),
            Some(other) => other.to_string(),
            None => "(missing)".to_string(),
        };
        if found != FORMAT {
            return Err(ProjectError::NotAProject { found });
        }
        let Some(version) = value.get("version").and_then(serde_json::Value::as_u64) else {
            return Err(ProjectError::Malformed(
                "the version field is missing".into(),
            ));
        };
        if version == 0 {
            return Err(ProjectError::Malformed("version 0 does not exist".into()));
        }
        if version > u64::from(FORMAT_VERSION) {
            return Err(ProjectError::NewerVersion { found: version });
        }
        let doc: Doc = serde_json::from_value(value)
            .map_err(|error| ProjectError::Malformed(error.to_string()))?;
        let (scene, materials, clips, rig) = doc.into_parts()?;
        Ok(Self {
            scene,
            materials,
            clips,
            rig,
        })
    }

    /// Write the document to `path` as pretty-printed JSON.
    pub fn save(&self, path: &Path) -> Result<(), ProjectError> {
        std::fs::write(path, format!("{}\n", self.to_json()?))?;
        Ok(())
    }

    /// Read the document at `path`.
    pub fn load(path: &Path) -> Result<Self, ProjectError> {
        Self::from_json(&std::fs::read_to_string(path)?)
    }

    /// What the file holds, as counts — what the app's status line says
    /// after a save or an open.
    pub fn summary(&self) -> ProjectSummary {
        ProjectSummary {
            nodes: self.scene.node_count(),
            materials: self.materials.materials.len(),
            textures: self.materials.textures.len(),
            clips: self.clips.len(),
            bones: self
                .rig
                .as_ref()
                .map_or(0, |rig| rig.skeleton.bones().len()),
            keys: self
                .clips
                .clips()
                .iter()
                .flat_map(|clip| clip.tracks())
                .map(|track| track.position.len() + track.rotation.len() + track.scale.len())
                .sum(),
        }
    }
}

/// The counts a [`Project`] holds, for statuses and tests.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ProjectSummary {
    pub nodes: usize,
    pub materials: usize,
    pub textures: usize,
    pub clips: usize,
    /// Bones in the document's rig (`0` when there is none).
    pub bones: usize,
    pub keys: usize,
}

impl ProjectSummary {
    /// The counts in the app's status style, e.g. `"3 node(s), 4 material(s),
    /// 1 clip(s)"`. Textures and keys stay part of the struct for callers
    /// that want them; the sentence keeps to what a user asked about.
    pub fn describe(&self) -> String {
        format!(
            "{} node(s), {} material(s), {} clip(s)",
            self.nodes, self.materials, self.clips
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use openforge_animation::{Clip, Interp, TransformTrack};
    use openforge_core::mesh::{Corner, Face, Mesh};
    use openforge_core::node::NodeKind;
    use openforge_core::{NodeId, Quat, Vec2, Vec3, Vec4};
    use openforge_rig::{Bone, LocalTransform};

    /// A two-triangle quad in the XY plane, with normals, UVs and a material
    /// slot on the face.
    fn quad_mesh() -> Mesh {
        Mesh {
            positions: vec![
                Vec3::new(-0.5, -0.5, 0.0),
                Vec3::new(0.5, -0.5, 0.0),
                Vec3::new(0.5, 0.5, 0.0),
                Vec3::new(-0.5, 0.5, 0.0),
            ],
            normals: vec![Vec3::Z; 4],
            uvs: vec![
                Vec2::new(0.0, 0.0),
                Vec2::new(1.0, 0.0),
                Vec2::new(1.0, 1.0),
                Vec2::new(0.0, 1.0),
            ],
            faces: vec![
                Face {
                    corners: (0..3)
                        .map(|i| Corner {
                            position: i,
                            normal: i,
                            uv: i,
                        })
                        .collect(),
                    material: 1,
                },
                Face {
                    corners: [0, 2, 3]
                        .into_iter()
                        .map(|i| Corner {
                            position: i,
                            normal: i,
                            uv: i,
                        })
                        .collect(),
                    material: 1,
                },
            ],
        }
    }

    /// A document with something of everything: a nested group, a tagged and
    /// frozen mesh, a dummy, two materials (one textured and two-sided), a
    /// texture with real pixels, and a clip whose track spans all four
    /// interpolation modes.
    fn rich_project() -> Project {
        let mut scene = Scene::new();
        let root = scene.create_group("Hull");
        let body = scene.create_mesh("Plates", quad_mesh());
        scene.reparent(body, Some(root)).unwrap();
        let dummy = scene.create_dummy("Attach Point");
        scene.reparent(dummy, Some(root)).unwrap();
        {
            let node = scene.get_mut(body).unwrap();
            node.transform.translation = Vec3::new(1.0, 2.0, 3.0);
            node.visible = false;
            node.frozen = true;
            node.lod = Some(2);
        }

        let mut materials = MaterialLibrary::new();
        let mut painted = openforge_materials::Material::new("Painted Steel");
        painted.diffuse = Vec4::new(0.4, 0.1, 0.1, 1.0);
        painted.specular = Vec3::new(0.9, 0.9, 0.9);
        painted.shininess = 128.0;
        painted.two_sided = true;
        painted.diffuse_map = Some(0);
        painted.normal_map = Some(0);
        painted.emissive = Vec3::new(0.2, 0.0, 0.1);
        materials.materials.push(painted);
        materials
            .materials
            .push(openforge_materials::Material::new("Glass"));
        materials.textures.push(openforge_materials::Texture::solid(
            "checker",
            2,
            2,
            [255, 0, 255, 255],
        ));

        let mut clips = ClipLibrary::new();
        let mut clip = Clip::new("Hover");
        let mut track = TransformTrack::targeting(body);
        track.add_position_key(0.0, Vec3::ZERO, Interp::Step);
        track.add_position_key(1.5, Vec3::new(0.0, 1.0, 0.0), Interp::Linear);
        track.add_rotation_key(0.5, Quat::IDENTITY, Interp::Smooth);
        track.add_scale_key(0.25, Vec3::splat(2.0), Interp::Ease);
        clip.add_track(track);
        // And one inert track, keys kept, no target.
        let mut inert = TransformTrack::new(None);
        inert.add_position_key(0.0, Vec3::X, Interp::Linear);
        clip.add_track(inert);
        clips.add_clip(clip);

        Project {
            scene,
            materials,
            clips,
            rig: None,
        }
    }

    /// `rich_project` with a rig over the "Plates" node: three bones (the
    /// second one posed), two influences on one corner vertex, and the quad's
    /// own positions as the bind list.
    fn rigged_project() -> Project {
        let mut project = rich_project();
        let body = project
            .scene
            .iter()
            .find(|n| n.name == "Plates")
            .unwrap()
            .id;
        let mut skeleton = Skeleton::new();
        for i in 0..3 {
            let parent = if i == 0 { None } else { Some(i - 1) };
            skeleton
                .add_bone(Bone::with_transform(
                    format!("bone_{i}"),
                    parent,
                    LocalTransform::from_translation(Vec3::new(0.0, 1.0, 0.0)),
                ))
                .unwrap();
        }
        skeleton
            .set_pose_rotation(1, Quat::from_rotation_z(0.5))
            .unwrap();
        let mut weights = SkinWeights::new(4, 3);
        weights.set_weights(0, &[(0, 1.0)]).unwrap();
        weights.set_weights(3, &[(1, 0.25), (2, 0.75)]).unwrap();
        let bind = project
            .scene
            .get(body)
            .unwrap()
            .mesh()
            .unwrap()
            .positions
            .clone();
        project.rig = Some(ProjectRig {
            skeleton,
            weights,
            bind,
            node: Some(body),
        });
        project
    }

    fn temp_path(extension: &str) -> std::path::PathBuf {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "openforge-project-{}-{n}.{extension}",
            std::process::id()
        ))
    }

    #[test]
    fn a_project_round_trips_through_its_json() {
        let original = rich_project();
        let text = original.to_json().unwrap();
        let loaded = Project::from_json(&text).unwrap();

        assert!(
            original.scene.content_eq(&loaded.scene),
            "the scene changed"
        );
        // The library's default grey at slot 0 is part of the document too.
        assert_eq!(original.materials.materials.len(), 3);
        assert_eq!(loaded.materials.materials.len(), 3);
        assert_eq!(loaded.materials.materials[1].name, "Painted Steel");
        assert_eq!(
            loaded.materials.materials[1].diffuse,
            Vec4::new(0.4, 0.1, 0.1, 1.0)
        );
        assert!(loaded.materials.materials[1].two_sided);
        assert_eq!(loaded.materials.materials[1].diffuse_map, Some(0));
        assert_eq!(loaded.materials.materials[1].normal_map, Some(0));
        assert_eq!(
            loaded.materials.materials[1].emissive,
            Vec3::new(0.2, 0.0, 0.1)
        );
        assert_eq!(loaded.materials.textures.len(), 1);
        assert_eq!(loaded.materials.textures[0].pixels.len(), 2 * 2 * 4);
        assert_eq!(loaded.clips.clip_names(), vec!["Hover"]);

        let clip = loaded.clips.clip("Hover").unwrap();
        assert_eq!(clip.tracks().len(), 2);
        let track = &clip.tracks()[0];
        assert!(track.target.is_some());
        assert_eq!(track.position.len(), 2);
        assert_eq!(track.rotation.len(), 1);
        assert_eq!(track.scale.len(), 1);
        let keys = track.position.keys();
        assert_eq!(keys[0].interp, Interp::Step);
        assert_eq!(keys[1].interp, Interp::Linear);
        assert_eq!(track.scale.keys()[0].interp, Interp::Ease);
        assert!(clip.tracks()[1].target.is_none());

        // The rebuilt scene numbers nodes freshly but keeps the shape.
        let restored = loaded.scene.iter().find(|n| n.name == "Plates").unwrap();
        assert_eq!(restored.transform.translation, Vec3::new(1.0, 2.0, 3.0));
        assert!(!restored.visible && restored.frozen);
        assert_eq!(restored.lod, Some(2));
        let root = loaded.scene.iter().find(|n| n.name == "Hull").unwrap();
        assert!(root.children.contains(&restored.id));
        assert!(matches!(restored.kind, NodeKind::Mesh { .. }));

        assert_eq!(loaded.summary(), original.summary());
        assert_eq!(loaded.summary().keys, 5);
    }

    #[test]
    fn an_emissive_map_saves_and_a_file_without_the_field_still_loads() {
        let mut project = rich_project();
        project.materials.materials[1].emissive_map = Some(0);
        let text = project.to_json().unwrap();
        assert!(text.contains("\"emissive_map\": 0"), "{text}");
        let loaded = Project::from_json(&text).unwrap();
        assert_eq!(loaded.materials.materials[1].emissive_map, Some(0));

        // A file written before the field existed: the line is simply absent,
        // and the map is absent rather than the whole file being refused.
        let older: String = text
            .lines()
            .filter(|line| !line.contains("\"emissive_map\""))
            .collect::<Vec<_>>()
            .join("\n");
        assert_ne!(older, text, "the field should have been written");
        let loaded = Project::from_json(&older).unwrap();
        assert_eq!(loaded.materials.materials[1].emissive_map, None);
        // Nothing else moved: the other maps are still there.
        assert_eq!(loaded.materials.materials[1].diffuse_map, Some(0));
    }

    #[test]
    fn the_json_names_its_format_and_version() {
        let text = rich_project().to_json().unwrap();
        assert!(text.contains("\"format\": \"openforge-project\""));
        assert!(text.contains("\"version\": 2"));
        // Ids are never written; positions are.
        assert!(
            text.contains("\"parent\": 0"),
            "parent positions should reach the file"
        );
        assert!(
            text.contains("\"interp\": \"ease\""),
            "interpolation is spelled out"
        );
    }

    #[test]
    fn a_dangling_clip_target_saves_as_an_inert_track() {
        let mut project = rich_project();
        let mut clip = Clip::new("Orphaned");
        let mut track = TransformTrack::targeting(NodeId(9999));
        track.add_rotation_key(0.0, Quat::IDENTITY, Interp::Linear);
        clip.add_track(track);
        project.clips.add_clip(clip);

        // Saving succeeds — the keys are the user's work — and the broken
        // reference comes back as an inert track.
        let loaded = Project::from_json(&project.to_json().unwrap()).unwrap();
        let track = &loaded.clips.clip("Orphaned").unwrap().tracks()[0];
        assert!(track.target.is_none());
        assert_eq!(track.rotation.len(), 1);
    }

    #[test]
    fn a_rig_round_trips_through_its_json() {
        let project = rigged_project();
        let loaded = Project::from_json(&project.to_json().unwrap()).unwrap();
        let rig = loaded.rig.as_ref().expect("the rig came back");

        assert_eq!(rig.skeleton.bones().len(), 3);
        assert_eq!(rig.skeleton.bones()[1].name, "bone_1");
        assert_eq!(rig.skeleton.bones()[1].parent, Some(0));
        // The pose, not just the bind pose: the saved document opens posed.
        assert_eq!(
            rig.skeleton.pose_local(1).unwrap().rotation,
            Quat::from_rotation_z(0.5)
        );
        assert_eq!(rig.weights.vertex_count(), 4);
        assert_eq!(rig.weights.bone_count(), 3);
        assert_eq!(rig.weights.influences(0).unwrap().len(), 1);
        assert!(
            rig.weights.influences(1).unwrap().is_empty(),
            "an unweighted vertex stays unweighted"
        );
        let influences = rig.weights.influences(3).unwrap();
        assert_eq!(influences.len(), 2);
        // Heaviest-first, the way live weights are stored.
        assert_eq!(influences[0].bone, 2);
        assert_eq!(influences[1].bone, 1);
        assert_eq!(rig.bind.len(), 4);

        // The driven node is mapped to the rebuilt scene's fresh id.
        let node = loaded.scene.get(rig.node.expect("a driven node")).unwrap();
        assert_eq!(node.name, "Plates");
        assert_eq!(loaded.summary().bones, 3);
        assert_eq!(loaded.summary(), project.summary());
    }

    #[test]
    fn a_version_one_file_loads_without_a_rig() {
        let project = rigged_project();
        let mut value: serde_json::Value =
            serde_json::from_str(&project.to_json().unwrap()).unwrap();
        value.as_object_mut().unwrap().remove("rig");
        value["version"] = serde_json::json!(1);

        // Version 1 files are still read: the rig is simply absent.
        let loaded = Project::from_json(&value.to_string()).unwrap();
        assert!(loaded.rig.is_none());
        assert!(loaded.scene.content_eq(&project.scene));
        assert_eq!(loaded.summary().bones, 0);
    }

    #[test]
    fn a_broken_rig_is_refused_not_guessed() {
        let text = rigged_project().to_json().unwrap();

        // A bone whose parent is listed after it.
        let mut value: serde_json::Value = serde_json::from_str(&text).unwrap();
        value["rig"]["bones"][0]["parent"] = serde_json::json!(2);
        let error = Project::from_json(&value.to_string()).unwrap_err();
        assert!(matches!(error, ProjectError::Malformed(_)), "{error}");
        assert!(error.to_string().contains("rig bone 0"), "{error}");

        // An influence on a bone the skeleton does not have.
        let mut value: serde_json::Value = serde_json::from_str(&text).unwrap();
        value["rig"]["weights"]["vertices"][3][0]["bone"] = serde_json::json!(9);
        let error = Project::from_json(&value.to_string()).unwrap_err();
        assert!(
            error.to_string().contains("rig weights for vertex 3"),
            "{error}"
        );

        // A bind list that does not cover the weight rows.
        let mut value: serde_json::Value = serde_json::from_str(&text).unwrap();
        value["rig"]["bind"] = serde_json::json!([[0.0, 0.0, 0.0]]);
        let error = Project::from_json(&value.to_string()).unwrap_err();
        assert!(error.to_string().contains("binds 1 vertex"), "{error}");

        // A rig driving a node the file does not define.
        let mut value: serde_json::Value = serde_json::from_str(&text).unwrap();
        value["rig"]["node"] = serde_json::json!(99);
        let error = Project::from_json(&value.to_string()).unwrap_err();
        assert!(error.to_string().contains("drives node 99"), "{error}");
    }

    #[test]
    fn foreign_files_are_refused_by_name() {
        // Not JSON at all (an OBJ opened as a project, say).
        let error = Project::from_json("o Cube\nv 0 0 0\n").unwrap_err();
        assert!(matches!(error, ProjectError::Json(_)), "{error}");
        assert!(error.to_string().starts_with("the file is not JSON"));

        // JSON, but not ours.
        let error = Project::from_json("{\"hello\": 1}").unwrap_err();
        assert!(matches!(error, ProjectError::NotAProject { .. }), "{error}");
        assert!(error.to_string().contains("(missing)"));

        let error =
            Project::from_json("{\"format\": \"something-else\", \"version\": 1}").unwrap_err();
        assert!(error.to_string().contains("something-else"));
    }

    #[test]
    fn a_newer_version_is_refused_with_its_number() {
        let error =
            Project::from_json("{\"format\": \"openforge-project\", \"version\": 3}").unwrap_err();
        match error {
            ProjectError::NewerVersion { found } => assert_eq!(found, 3),
            other => panic!("expected NewerVersion, got {other}"),
        }
        assert!(error.to_string().contains("newer OpenForge"));
    }

    #[test]
    fn broken_references_are_refused_not_guessed() {
        let text = rich_project().to_json().unwrap();

        // A parent listed at or after its child.
        let mut value: serde_json::Value = serde_json::from_str(&text).unwrap();
        value["nodes"][0]["parent"] = serde_json::json!(1);
        let error = Project::from_json(&value.to_string()).unwrap_err();
        assert!(matches!(error, ProjectError::Malformed(_)), "{error}");
        assert!(error.to_string().contains("not listed before it"));

        // A clip track driving a node the file does not define.
        let mut value: serde_json::Value = serde_json::from_str(&text).unwrap();
        value["clips"][0]["tracks"][0]["target"] = serde_json::json!(99);
        let error = Project::from_json(&value.to_string()).unwrap_err();
        assert!(error.to_string().contains("does not define"));

        // A texture whose pixel count disagrees with its size.
        let mut value: serde_json::Value = serde_json::from_str(&text).unwrap();
        value["textures"][0]["width"] = serde_json::json!(99);
        let error = Project::from_json(&value.to_string()).unwrap_err();
        assert!(error.to_string().contains("but carries"));

        // A texture whose pixels are not base64.
        let mut value: serde_json::Value = serde_json::from_str(&text).unwrap();
        value["textures"][0]["pixels"] = serde_json::json!("not base64!!");
        let error = Project::from_json(&value.to_string()).unwrap_err();
        assert!(error.to_string().contains("not base64"));

        // A missing version field.
        let mut value: serde_json::Value = serde_json::from_str(&text).unwrap();
        value.as_object_mut().unwrap().remove("version");
        let error = Project::from_json(&value.to_string()).unwrap_err();
        assert!(error.to_string().contains("version field is missing"));
    }

    #[test]
    fn a_project_saves_and_loads_through_a_real_file() {
        let path = temp_path("ofproj");
        let original = rich_project();
        original.save(&path).unwrap();
        let loaded = Project::load(&path).unwrap();
        assert!(loaded.scene.content_eq(&original.scene));
        assert_eq!(loaded.summary(), original.summary());
        std::fs::remove_file(&path).ok();

        // A path that is not there reads as an error, not a panic.
        let error = Project::load(&temp_path("ofproj")).unwrap_err();
        assert!(matches!(error, ProjectError::Io(_)), "{error}");
    }

    #[test]
    fn saving_the_same_scene_twice_writes_the_same_text() {
        let project = rich_project();
        assert_eq!(project.to_json().unwrap(), project.to_json().unwrap());

        // Capturing from parts clones rather than borrows: the file does not
        // move with the scene afterwards.
        let mut scene = Scene::new();
        let id = scene.create_group("G");
        let captured = Project::from_parts(&scene, &MaterialLibrary::new(), &ClipLibrary::new());
        scene.get_mut(id).unwrap().name = "Renamed".into();
        scene.remove_subtree(id);
        assert_eq!(captured.scene.node_count(), 1);
        assert!(captured.to_json().unwrap().contains("\"G\""));
    }
}
