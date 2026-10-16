//! The on-disk document and its bridge to the in-memory types.
//!
//! The document is deliberately *not* the in-memory types with derives bolted
//! on: the file format has its own shape, and the two places it differs from
//! memory are the load-bearing ones.
//!
//! - **Scene ids never reach the file.** The in-memory scene numbers nodes by
//!   allocation order; the document lists them depth-first (every parent
//!   before its children) and writes references — a node's parent, a clip
//!   track's target — as *positions in that list*. A file therefore loads
//!   into whatever ids the fresh scene allocates, and two saves of the same
//!   scene are byte-identical.
//! - **A clip track whose node is gone saves as an inert track.** Deleting a
//!   node does not reach into the clip library, so a track can outlive its
//!   target; the document keeps its keys and drops the broken reference
//!   (which is exactly what a `target: None` track means), rather than
//!   refusing to save the user's work over a dangling pointer.

use openforge_animation::{Clip, ClipLibrary, Interp, TransformTrack};
use openforge_core::math::Transform;
use openforge_core::node::{Node, NodeKind};
use openforge_core::scene::Scene;
use openforge_core::NodeId;
use openforge_materials::{Material, MaterialLibrary, Texture};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use crate::base64;
use crate::{ProjectError, FORMAT, FORMAT_VERSION};

/// The JSON document, exactly as it sits in the file.
#[derive(Serialize, Deserialize)]
pub(crate) struct Doc {
    pub format: String,
    pub version: u32,
    pub nodes: Vec<DocNode>,
    pub materials: Vec<DocMaterial>,
    pub textures: Vec<DocTexture>,
    pub clips: Vec<DocClip>,
}

/// One scene node: its own fields, its kind (`openforge-core`'s `NodeKind`,
/// so a mesh carries its geometry and material slots), and its parent as a
/// position in `Doc::nodes`.
#[derive(Serialize, Deserialize)]
pub(crate) struct DocNode {
    pub name: String,
    /// Position of the parent node in this list; parents always come first.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent: Option<usize>,
    pub transform: Transform,
    pub visible: bool,
    pub frozen: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lod: Option<u8>,
    pub kind: NodeKind,
}

/// One material, field for field (`openforge-materials::Material`).
#[derive(Serialize, Deserialize)]
pub(crate) struct DocMaterial {
    pub name: String,
    pub diffuse: openforge_core::Vec4,
    pub specular: openforge_core::Vec3,
    pub emissive: openforge_core::Vec3,
    pub shininess: f32,
    pub opacity: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub diffuse_map: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub normal_map: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub specular_map: Option<usize>,
    pub two_sided: bool,
}

impl From<&Material> for DocMaterial {
    fn from(m: &Material) -> Self {
        Self {
            name: m.name.clone(),
            diffuse: m.diffuse,
            specular: m.specular,
            emissive: m.emissive,
            shininess: m.shininess,
            opacity: m.opacity,
            diffuse_map: m.diffuse_map,
            normal_map: m.normal_map,
            specular_map: m.specular_map,
            two_sided: m.two_sided,
        }
    }
}

impl DocMaterial {
    fn to_material(&self) -> Material {
        Material {
            name: self.name.clone(),
            diffuse: self.diffuse,
            specular: self.specular,
            emissive: self.emissive,
            shininess: self.shininess,
            opacity: self.opacity,
            diffuse_map: self.diffuse_map,
            normal_map: self.normal_map,
            specular_map: self.specular_map,
            two_sided: self.two_sided,
        }
    }
}

/// One texture: its size and its RGBA8 pixels, base64-encoded.
#[derive(Serialize, Deserialize)]
pub(crate) struct DocTexture {
    pub name: String,
    pub width: u32,
    pub height: u32,
    /// RGBA8 texels, row-major, standard base64 — see [`crate::base64`].
    pub pixels: String,
}

/// One clip: a name and its transform tracks.
#[derive(Serialize, Deserialize)]
pub(crate) struct DocClip {
    pub name: String,
    pub tracks: Vec<DocTrack>,
}

/// One transform track: its target as a position in `Doc::nodes`, and its
/// three sub-tracks' keys.
#[derive(Serialize, Deserialize)]
pub(crate) struct DocTrack {
    /// Position of the driven node in `Doc::nodes`; absent for an inert track.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target: Option<usize>,
    pub position: Vec<DocKey<openforge_core::Vec3>>,
    pub rotation: Vec<DocKey<openforge_core::Quat>>,
    pub scale: Vec<DocKey<openforge_core::Vec3>>,
}

/// One keyframe: time, value and outgoing interpolation.
#[derive(Serialize, Deserialize)]
pub(crate) struct DocKey<T> {
    pub time: f64,
    pub value: T,
    pub interp: DocInterp,
}

/// The interpolation modes, spelled out for the file (`step`, `linear`,
/// `smooth`, `ease`) so a reader never depends on `Interp`'s variant order.
#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Debug)]
#[serde(rename_all = "lowercase")]
pub(crate) enum DocInterp {
    Step,
    Linear,
    Smooth,
    Ease,
}

impl From<Interp> for DocInterp {
    fn from(interp: Interp) -> Self {
        match interp {
            Interp::Step => Self::Step,
            Interp::Linear => Self::Linear,
            Interp::Smooth => Self::Smooth,
            Interp::Ease => Self::Ease,
        }
    }
}

impl From<DocInterp> for Interp {
    fn from(interp: DocInterp) -> Self {
        match interp {
            DocInterp::Step => Self::Step,
            DocInterp::Linear => Self::Linear,
            DocInterp::Smooth => Self::Smooth,
            DocInterp::Ease => Self::Ease,
        }
    }
}

impl Doc {
    /// Capture the in-memory document.
    pub(crate) fn from_parts(
        scene: &Scene,
        library: &MaterialLibrary,
        clips: &ClipLibrary,
    ) -> Self {
        // Depth-first, parents before children; entries the scene cannot
        // resolve are left out of *both* this list and the name map, so the
        // positions written into the file always line up with the nodes.
        let listed: Vec<(NodeId, &Node)> = node_order(scene)
            .into_iter()
            .filter_map(|id| scene.get(id).ok().map(|node| (id, node)))
            .collect();
        let index_of: HashMap<NodeId, usize> = listed
            .iter()
            .enumerate()
            .map(|(i, (id, _))| (*id, i))
            .collect();

        let nodes = listed
            .iter()
            .map(|(_, node)| DocNode {
                name: node.name.clone(),
                parent: node.parent.and_then(|p| index_of.get(&p).copied()),
                transform: node.transform,
                visible: node.visible,
                frozen: node.frozen,
                lod: node.lod,
                kind: node.kind.clone(),
            })
            .collect();

        let materials = library.materials.iter().map(DocMaterial::from).collect();
        let textures = library
            .textures
            .iter()
            .map(|texture| DocTexture {
                name: texture.name.clone(),
                width: texture.width,
                height: texture.height,
                pixels: base64::encode(&texture.pixels),
            })
            .collect();
        let clips = clips
            .clips()
            .iter()
            .map(|clip| DocClip {
                name: clip.name().to_string(),
                tracks: clip
                    .tracks()
                    .iter()
                    .map(|track| DocTrack {
                        // A target node that is no longer in the scene saves as
                        // an inert track — the keys are the user's work, the
                        // broken pointer is not.
                        target: track.target.and_then(|id| index_of.get(&id).copied()),
                        position: track.position.keys().iter().map(key_of).collect(),
                        rotation: track.rotation.keys().iter().map(key_of).collect(),
                        scale: track.scale.keys().iter().map(key_of).collect(),
                    })
                    .collect(),
            })
            .collect();

        Doc {
            format: FORMAT.to_string(),
            version: FORMAT_VERSION,
            nodes,
            materials,
            textures,
            clips,
        }
    }

    /// Rebuild the in-memory document, refusing references the file got wrong.
    pub(crate) fn into_parts(self) -> Result<(Scene, MaterialLibrary, ClipLibrary), ProjectError> {
        let mut scene = Scene::new();
        let mut ids: Vec<NodeId> = Vec::with_capacity(self.nodes.len());
        for (index, doc) in self.nodes.iter().enumerate() {
            let parent = match doc.parent {
                Some(p) if p < index => Some(ids[p]),
                Some(p) => {
                    return Err(ProjectError::Malformed(format!(
                        "node {index} ({:?}) names parent {p}, which is not listed before it",
                        doc.name
                    )));
                }
                None => None,
            };
            let mut node = Node::new(NodeId::INVALID, doc.name.clone(), doc.kind.clone());
            node.transform = doc.transform;
            node.visible = doc.visible;
            node.frozen = doc.frozen;
            node.lod = doc.lod;
            node.parent = parent;
            ids.push(scene.insert(node));
        }

        // Empty on purpose: the file lists the *whole* library, including the
        // default grey material at index 0 that `MaterialLibrary::new` would
        // also seed — seeding again here would shift every slot by one.
        let mut materials = MaterialLibrary {
            materials: Vec::new(),
            textures: Vec::new(),
        };
        for doc in &self.materials {
            materials.materials.push(doc.to_material());
        }
        for doc in &self.textures {
            let pixels = base64::decode(&doc.pixels).ok_or_else(|| {
                ProjectError::Malformed(format!(
                    "texture {:?} carries characters that are not base64",
                    doc.name
                ))
            })?;
            let expected = u64::from(doc.width) * u64::from(doc.height) * 4;
            if pixels.len() as u64 != expected {
                return Err(ProjectError::Malformed(format!(
                    "texture {:?} says {}×{} but carries {} byte(s)",
                    doc.name,
                    doc.width,
                    doc.height,
                    pixels.len()
                )));
            }
            materials.textures.push(Texture {
                name: doc.name.clone(),
                width: doc.width,
                height: doc.height,
                pixels,
            });
        }

        let mut clips = ClipLibrary::new();
        for doc in &self.clips {
            let mut clip = Clip::new(doc.name.clone());
            for track in &doc.tracks {
                let target = match track.target {
                    Some(i) => Some(*ids.get(i).ok_or_else(|| {
                        ProjectError::Malformed(format!(
                            "clip {:?} drives node {i}, which the file does not define",
                            doc.name
                        ))
                    })?),
                    None => None,
                };
                let mut rebuilt = TransformTrack::new(target);
                for key in &track.position {
                    rebuilt.add_position_key(key.time, key.value, key.interp.into());
                }
                for key in &track.rotation {
                    rebuilt.add_rotation_key(key.time, key.value, key.interp.into());
                }
                for key in &track.scale {
                    rebuilt.add_scale_key(key.time, key.value, key.interp.into());
                }
                clip.add_track(rebuilt);
            }
            clips.add_clip(clip);
        }

        Ok((scene, materials, clips))
    }
}

fn key_of<T: Copy>(key: &openforge_animation::Key<T>) -> DocKey<T> {
    DocKey {
        time: key.time,
        value: key.value,
        interp: key.interp.into(),
    }
}

/// The nodes depth-first, every parent before its children — the order the
/// file lists them in and the positions in it refer to.
fn node_order(scene: &Scene) -> Vec<NodeId> {
    let mut order = Vec::with_capacity(scene.node_count());
    let mut seen: HashSet<NodeId> = HashSet::with_capacity(scene.node_count());
    for root in scene.roots() {
        push_subtree(scene, *root, &mut seen, &mut order);
    }
    // Defensive: a node that hangs under no root (no scene API leaves one)
    // still needs a place in the file. A second DFS from each leftover keeps
    // parents before children there too, so the file it writes stays loadable.
    for id in scene.all_ids() {
        push_subtree(scene, id, &mut seen, &mut order);
    }
    order
}

fn push_subtree(scene: &Scene, id: NodeId, seen: &mut HashSet<NodeId>, order: &mut Vec<NodeId>) {
    if !seen.insert(id) {
        return;
    }
    order.push(id);
    if let Ok(node) = scene.get(id) {
        for child in &node.children {
            push_subtree(scene, *child, seen, order);
        }
    }
}
