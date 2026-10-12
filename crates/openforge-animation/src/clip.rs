//! Animation clips: per-node transform tracks grouped under a name.

use crate::interpolation::Interp;
use crate::track::Track;
use glam::{Quat, Vec3};
use openforge_core::{math::Transform, NodeId};

/// The position, rotation and scale tracks that drive a single node.
///
/// Any sub-track may be empty: a missing sub-track falls back to the identity
/// component, so a track that only animates position leaves rotation and
/// scale untouched by the sample result.
#[derive(Clone, Debug, PartialEq)]
pub struct TransformTrack {
    /// The scene node this track drives. `None` marks the track as inert:
    /// it still samples, but a [`Clip`] skips it when producing results.
    pub target: Option<NodeId>,
    /// Local-space translation over time.
    pub position: Track<Vec3>,
    /// Local-space rotation over time.
    pub rotation: Track<Quat>,
    /// Local-space scale over time.
    pub scale: Track<Vec3>,
}

impl TransformTrack {
    /// Create a track for an optional target node, with all sub-tracks empty.
    pub fn new(target: Option<NodeId>) -> Self {
        Self {
            target,
            position: Track::new(),
            rotation: Track::new(),
            scale: Track::new(),
        }
    }

    /// Create a track that drives `target`.
    pub fn targeting(target: NodeId) -> Self {
        Self::new(Some(target))
    }

    /// Whether none of the sub-tracks hold any keys.
    pub fn is_empty(&self) -> bool {
        self.position.is_empty() && self.rotation.is_empty() && self.scale.is_empty()
    }

    /// Longest duration across the three sub-tracks (`0.0` when empty).
    pub fn duration(&self) -> f64 {
        self.position
            .duration()
            .max(self.rotation.duration())
            .max(self.scale.duration())
    }

    /// Add a key to the position sub-track; returns its index.
    pub fn add_position_key(&mut self, time: f64, value: Vec3, interp: Interp) -> usize {
        self.position.add_key(time, value, interp)
    }

    /// Add a key to the rotation sub-track; returns its index.
    pub fn add_rotation_key(&mut self, time: f64, value: Quat, interp: Interp) -> usize {
        self.rotation.add_key(time, value, interp)
    }

    /// Add a key to the scale sub-track; returns its index.
    pub fn add_scale_key(&mut self, time: f64, value: Vec3, interp: Interp) -> usize {
        self.scale.add_key(time, value, interp)
    }

    /// Sample the combined transform at `time`.
    ///
    /// Returns `None` only when every sub-track is empty. Empty sub-tracks
    /// contribute their identity component (translation `ZERO`, rotation
    /// `IDENTITY`, scale `ONE`).
    pub fn sample(&self, time: f64) -> Option<Transform> {
        let mut transform = Transform::identity();
        let mut any = false;
        if let Some(position) = self.position.sample(time) {
            transform.translation = position;
            any = true;
        }
        if let Some(rotation) = self.rotation.sample(time) {
            transform.rotation = rotation;
            any = true;
        }
        if let Some(scale) = self.scale.sample(time) {
            transform.scale = scale;
            any = true;
        }
        any.then_some(transform)
    }
}

/// A named animation clip: a set of [`TransformTrack`]s sampled together.
#[derive(Clone, Debug, PartialEq)]
pub struct Clip {
    name: String,
    tracks: Vec<TransformTrack>,
}

impl Clip {
    /// Create an empty clip with the given name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            tracks: Vec::new(),
        }
    }

    /// The clip's name (unique within a [`ClipLibrary`]).
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The clip's length in seconds: the longest track duration, or `0.0`
    /// when the clip has no tracks.
    pub fn duration(&self) -> f64 {
        self.tracks
            .iter()
            .map(TransformTrack::duration)
            .fold(0.0, f64::max)
    }

    /// All transform tracks in the clip, in insertion order.
    pub fn tracks(&self) -> &[TransformTrack] {
        &self.tracks
    }

    /// The transform track at `index`, or `None` if out of range.
    pub fn track(&self, index: usize) -> Option<&TransformTrack> {
        self.tracks.get(index)
    }

    /// Append a transform track to the clip.
    pub fn add_track(&mut self, track: TransformTrack) {
        self.tracks.push(track);
    }

    /// Remove the transform track at `index`, returning it, or `None` if out
    /// of range.
    pub fn remove_track(&mut self, index: usize) -> Option<TransformTrack> {
        if index < self.tracks.len() {
            Some(self.tracks.remove(index))
        } else {
            None
        }
    }

    /// Sample every track that targets a node at `time`.
    ///
    /// Tracks without a target (or whose sub-tracks are all empty) are
    /// skipped. When several tracks target the same node, the last one in
    /// the clip wins.
    pub fn sample_all(&self, time: f64) -> Vec<(NodeId, Transform)> {
        let mut out = Vec::with_capacity(self.tracks.len());
        for track in &self.tracks {
            let Some(id) = track.target else {
                continue;
            };
            if let Some(transform) = track.sample(time) {
                out.push((id, transform));
            }
        }
        out
    }
}

/// A collection of uniquely named [`Clip`]s.
///
/// Clips keep their insertion order. Adding a clip whose name already exists
/// replaces the previous one in place.
#[derive(Clone, Debug, Default)]
pub struct ClipLibrary {
    clips: Vec<Clip>,
}

impl ClipLibrary {
    /// Create an empty library.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of clips in the library.
    pub fn len(&self) -> usize {
        self.clips.len()
    }

    /// Whether the library holds no clips.
    pub fn is_empty(&self) -> bool {
        self.clips.is_empty()
    }

    /// Insert a clip, replacing any clip with the same name. Returns the
    /// index the clip occupies.
    pub fn add_clip(&mut self, clip: Clip) -> usize {
        match self.clips.iter().position(|c| c.name() == clip.name()) {
            Some(index) => {
                self.clips[index] = clip;
                index
            }
            None => {
                self.clips.push(clip);
                self.clips.len() - 1
            }
        }
    }

    /// Look up a clip by name.
    pub fn clip(&self, name: &str) -> Option<&Clip> {
        self.clips.iter().find(|c| c.name() == name)
    }

    /// Mutably look up a clip by name.
    pub fn clip_mut(&mut self, name: &str) -> Option<&mut Clip> {
        self.clips.iter_mut().find(|c| c.name() == name)
    }

    /// Remove a clip by name, returning it, or `None` if it wasn't present.
    pub fn remove_clip(&mut self, name: &str) -> Option<Clip> {
        let index = self.clips.iter().position(|c| c.name() == name)?;
        Some(self.clips.remove(index))
    }

    /// Remove every clip from the library.
    pub fn clear(&mut self) {
        self.clips.clear();
    }

    /// Names of every clip, in insertion order.
    pub fn clip_names(&self) -> Vec<&str> {
        self.clips.iter().map(|c| c.name()).collect()
    }

    /// All clips, in insertion order.
    pub fn clips(&self) -> &[Clip] {
        &self.clips
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(id: u32) -> NodeId {
        NodeId(id)
    }

    #[test]
    fn empty_transform_track_samples_none() {
        let track = TransformTrack::targeting(node(0));
        assert!(track.is_empty());
        assert_eq!(track.duration(), 0.0);
        assert_eq!(track.sample(0.0), None);
        assert_eq!(track.sample(10.0), None);
    }

    #[test]
    fn position_only_track_keeps_identity_rotation_and_scale() {
        let mut track = TransformTrack::targeting(node(0));
        track.add_position_key(0.0, Vec3::ZERO, Interp::Linear);
        track.add_position_key(1.0, Vec3::new(4.0, 0.0, 0.0), Interp::Linear);

        let transform = track.sample(0.5).expect("position track samples");
        assert_eq!(transform.translation, Vec3::new(2.0, 0.0, 0.0));
        assert_eq!(transform.rotation, Quat::IDENTITY);
        assert_eq!(transform.scale, Vec3::ONE);
        assert_eq!(track.duration(), 1.0);
    }

    #[test]
    fn transform_track_combines_all_sub_tracks() {
        let mut track = TransformTrack::targeting(node(3));
        track.add_position_key(0.0, Vec3::ZERO, Interp::Linear);
        track.add_position_key(1.0, Vec3::new(2.0, 4.0, 6.0), Interp::Linear);
        track.add_rotation_key(0.0, Quat::IDENTITY, Interp::Linear);
        track.add_rotation_key(2.0, Quat::from_rotation_y(1.0), Interp::Linear);
        track.add_scale_key(0.0, Vec3::ONE, Interp::Linear);
        track.add_scale_key(3.0, Vec3::splat(2.0), Interp::Linear);

        assert_eq!(track.duration(), 3.0);

        // Before the keys everything clamps to the first key's pose.
        let transform = track.sample(0.0).expect("samples before keys");
        assert_eq!(transform.translation, Vec3::ZERO);
        assert_eq!(transform.rotation, Quat::IDENTITY);
        assert_eq!(transform.scale, Vec3::ONE);

        // After the keys everything clamps to the last key's pose.
        let transform = track.sample(10.0).expect("samples after keys");
        assert_eq!(transform.translation, Vec3::new(2.0, 4.0, 6.0));
        assert!(transform.rotation.angle_between(Quat::from_rotation_y(1.0)) < 1e-4);
        assert_eq!(transform.scale, Vec3::splat(2.0));

        // Midway, each sub-track contributes its own blend.
        let transform = track.sample(0.5).expect("samples midway");
        assert_eq!(transform.translation, Vec3::new(1.0, 2.0, 3.0));
        assert!(transform.rotation.angle_between(Quat::from_rotation_y(0.25)) < 1e-3);
        assert!(transform.scale.distance(Vec3::splat(7.0 / 6.0)) < 1e-5);
    }

    #[test]
    fn clip_duration_is_longest_track() {
        let mut clip = Clip::new("idle");
        assert_eq!(clip.duration(), 0.0);

        let mut short = TransformTrack::targeting(node(0));
        short.add_position_key(0.5, Vec3::ZERO, Interp::Linear);
        let mut long = TransformTrack::targeting(node(1));
        long.add_scale_key(2.5, Vec3::ONE, Interp::Linear);
        clip.add_track(short);
        clip.add_track(long);

        assert_eq!(clip.name(), "idle");
        assert_eq!(clip.duration(), 2.5);
        assert_eq!(clip.tracks().len(), 2);
        assert!(clip.track(1).is_some());
        assert!(clip.track(2).is_none());

        let removed = clip.remove_track(1);
        assert!(removed.is_some());
        assert_eq!(clip.duration(), 0.5);
        assert!(clip.remove_track(5).is_none());
    }

    #[test]
    fn clip_sample_all_returns_targeted_transforms_only() {
        let mut clip = Clip::new("walk");

        let mut driven = TransformTrack::targeting(node(7));
        driven.add_position_key(0.0, Vec3::ZERO, Interp::Linear);
        driven.add_position_key(2.0, Vec3::new(0.0, 0.0, 4.0), Interp::Linear);
        clip.add_track(driven);

        // No target: never appears in the sampled results.
        let mut orphan = TransformTrack::new(None);
        orphan.add_position_key(0.0, Vec3::ONE, Interp::Linear);
        clip.add_track(orphan);

        // Targeted but empty: also skipped.
        clip.add_track(TransformTrack::targeting(node(9)));

        let poses = clip.sample_all(1.0);
        assert_eq!(poses.len(), 1);
        assert_eq!(poses[0].0, node(7));
        assert_eq!(poses[0].1.translation, Vec3::new(0.0, 0.0, 2.0));

        // An empty clip samples to no poses at all.
        let empty = Clip::new("nothing");
        assert!(empty.sample_all(0.0).is_empty());
    }

    #[test]
    fn library_add_lookup_and_names_preserve_insertion_order() {
        let mut library = ClipLibrary::new();
        assert!(library.is_empty());
        assert!(library.clip("missing").is_none());
        assert!(library.clip_mut("missing").is_none());

        library.add_clip(Clip::new("second"));
        library.add_clip(Clip::new("first"));
        library.add_clip(Clip::new("third"));

        assert_eq!(library.len(), 3);
        assert_eq!(library.clip_names(), vec!["second", "first", "third"]);
        assert_eq!(library.clips().len(), 3);
        assert_eq!(library.clip("first").map(Clip::name), Some("first"));
        assert!(library.clip_mut("first").is_some());
    }

    #[test]
    fn library_replaces_clip_with_the_same_name() {
        let mut library = ClipLibrary::new();
        library.add_clip(Clip::new("spin"));
        let mut with_track = Clip::new("spin");
        with_track.add_track(TransformTrack::targeting(node(1)));

        let index = library.add_clip(with_track);
        assert_eq!(index, 0);
        assert_eq!(library.len(), 1);
        assert_eq!(library.clip("spin").map(Clip::duration), Some(0.0));
        assert_eq!(library.clip("spin").map(|c| c.tracks().len()), Some(1));
    }

    #[test]
    fn library_remove_clip() {
        let mut library = ClipLibrary::new();
        library.add_clip(Clip::new("a"));
        library.add_clip(Clip::new("b"));

        assert!(library.remove_clip("a").is_some());
        assert_eq!(library.len(), 1);
        assert!(library.remove_clip("a").is_none());
        assert_eq!(library.clip_names(), vec!["b"]);

        library.clear();
        assert!(library.is_empty());
    }
}
