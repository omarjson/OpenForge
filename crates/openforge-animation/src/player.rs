//! Playback transport for animation clips.

use crate::clip::{Clip, ClipLibrary};
use openforge_core::{math::Transform, NodeId};

/// Playback state machine over a [`ClipLibrary`].
///
/// The application selects a clip by name, starts playback, and calls
/// [`Player::step`] once per frame; each step advances [`Player::time`] by
/// `delta * speed` and returns the transforms to apply to the scene.
///
/// ```
/// use openforge_animation::{Clip, ClipLibrary, Player, TransformTrack};
/// use openforge_core::NodeId;
/// use glam::Vec3;
///
/// let mut track = TransformTrack::targeting(NodeId(0));
/// track.add_position_key(0.0, Vec3::ZERO, openforge_animation::Interp::Linear);
/// track.add_position_key(1.0, Vec3::new(1.0, 0.0, 0.0), openforge_animation::Interp::Linear);
/// let mut clip = Clip::new("slide");
/// clip.add_track(track);
///
/// let mut library = ClipLibrary::new();
/// library.add_clip(clip);
///
/// let mut player = Player::with_library(library);
/// assert!(player.select_clip("slide"));
/// player.play();
/// let poses = player.step(0.5);
/// assert_eq!(poses.len(), 1);
/// assert_eq!(poses[0].0, NodeId(0));
/// assert_eq!(poses[0].1.translation, Vec3::new(0.5, 0.0, 0.0));
/// ```
#[derive(Clone, Debug)]
pub struct Player {
    /// Current playback position in seconds.
    pub time: f64,
    /// Playback rate; negative values play in reverse.
    pub speed: f32,
    /// Whether the transport is currently advancing time.
    pub playing: bool,
    /// Whether time wraps at the clip's ends instead of stopping there.
    pub looping: bool,
    library: ClipLibrary,
    current: Option<String>,
}

impl Default for Player {
    fn default() -> Self {
        Self::new()
    }
}

impl Player {
    /// Create a stopped player with an empty clip library, a speed of `1.0`
    /// and looping disabled.
    pub fn new() -> Self {
        Self {
            time: 0.0,
            speed: 1.0,
            playing: false,
            looping: false,
            library: ClipLibrary::new(),
            current: None,
        }
    }

    /// Create a stopped player over the given clip library.
    pub fn with_library(library: ClipLibrary) -> Self {
        Self {
            library,
            ..Self::new()
        }
    }

    /// Start or resume playback.
    ///
    /// If a non-looping clip has already finished, playback restarts from
    /// the beginning.
    pub fn play(&mut self) {
        if !self.looping {
            let duration = self.duration();
            if duration > 0.0 && self.time >= duration {
                self.time = 0.0;
            }
        }
        self.playing = true;
    }

    /// Pause playback, keeping the current time.
    pub fn pause(&mut self) {
        self.playing = false;
    }

    /// Stop playback and rewind to time `0.0`.
    pub fn stop(&mut self) {
        self.playing = false;
        self.time = 0.0;
    }

    /// Jump to an absolute time, clamped to `0.0..=duration()` of the
    /// selected clip (or `0.0` when no clip is selected). A NaN time seeks
    /// to the start.
    pub fn seek(&mut self, time: f64) {
        let time = if time.is_nan() { 0.0 } else { time };
        self.time = time.clamp(0.0, self.duration());
    }

    /// Advance playback by `delta_seconds * speed` and sample the selected
    /// clip at the new time.
    ///
    /// Returns an empty vector when the transport is paused or no clip is
    /// selected (in which case time does not advance either). When looping,
    /// time wraps around the clip's duration; otherwise it clamps at the
    /// end (or the start, for negative speeds) and pauses. A clip with zero
    /// duration holds at its start.
    pub fn step(&mut self, delta_seconds: f64) -> Vec<(NodeId, Transform)> {
        if !self.playing {
            return Vec::new();
        }
        // Borrow the duration first so `time` can be mutated below.
        let duration = match self.current_clip() {
            Some(clip) => clip.duration(),
            None => return Vec::new(),
        };

        if !self.time.is_finite() {
            self.time = 0.0;
        }
        let advance = delta_seconds * f64::from(self.speed);
        let advance = if advance.is_finite() { advance } else { 0.0 };
        self.time += advance;

        if duration > 0.0 {
            if self.looping {
                self.time = self.time.rem_euclid(duration);
            } else if self.time >= duration {
                // At or past the end: clamp into range, and finish when
                // moving forward (or standing still at the end).
                self.time = duration;
                if advance >= 0.0 {
                    self.playing = false;
                }
            } else if self.time <= 0.0 {
                // At or before the start: clamp into range, and finish only
                // when actually moving backward.
                self.time = 0.0;
                if advance < 0.0 {
                    self.playing = false;
                }
            }
        } else {
            // No span to play through: hold at the clip's start.
            self.time = 0.0;
            if !self.looping {
                self.playing = false;
            }
        }

        match self.current_clip() {
            Some(clip) => clip.sample_all(self.time),
            None => Vec::new(),
        }
    }

    /// Select the clip named `name` for playback and rewind to `0.0`.
    ///
    /// Returns `false` and leaves the current selection untouched when no
    /// clip has that name.
    pub fn select_clip(&mut self, name: &str) -> bool {
        if self.library.clip(name).is_some() {
            self.current = Some(name.to_owned());
            self.time = 0.0;
            true
        } else {
            false
        }
    }

    /// Deselect the current clip and rewind to `0.0`.
    pub fn clear_clip(&mut self) {
        self.current = None;
        self.time = 0.0;
    }

    /// The clip currently selected for playback, if any.
    pub fn current_clip(&self) -> Option<&Clip> {
        self.current.as_ref().and_then(|name| self.library.clip(name))
    }

    /// Name of the currently selected clip, if any.
    pub fn current_clip_name(&self) -> Option<&str> {
        self.current.as_deref()
    }

    /// Duration of the selected clip, or `0.0` when none is selected.
    pub fn duration(&self) -> f64 {
        self.current_clip().map_or(0.0, Clip::duration)
    }

    /// The clip library this player draws clips from.
    pub fn library(&self) -> &ClipLibrary {
        &self.library
    }

    /// Mutable access to the clip library (to add, edit or remove clips).
    pub fn library_mut(&mut self) -> &mut ClipLibrary {
        &mut self.library
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clip::TransformTrack;
    use crate::interpolation::Interp;
    use glam::Vec3;

    /// A clip named "slide" that moves node 0 along +X over two seconds.
    fn demo_player() -> Player {
        let mut track = TransformTrack::targeting(NodeId(0));
        track.add_position_key(0.0, Vec3::ZERO, Interp::Linear);
        track.add_position_key(2.0, Vec3::new(2.0, 0.0, 0.0), Interp::Linear);
        let mut clip = Clip::new("slide");
        clip.add_track(track);

        let mut library = ClipLibrary::new();
        library.add_clip(clip);
        Player::with_library(library)
    }

    #[test]
    fn new_player_starts_stopped() {
        let mut player = Player::new();
        assert_eq!(player.time, 0.0);
        assert_eq!(player.speed, 1.0);
        assert!(!player.playing);
        assert!(!player.looping);
        assert!(player.current_clip().is_none());
        assert_eq!(player.current_clip_name(), None);
        assert_eq!(player.duration(), 0.0);
        assert_eq!(player.step(0.1), Vec::new());
    }

    #[test]
    fn play_pause_stop_transport_controls() {
        let mut player = demo_player();
        player.select_clip("slide");
        player.seek(1.0);

        player.play();
        assert!(player.playing);
        assert_eq!(player.time, 1.0);

        player.pause();
        assert!(!player.playing);
        assert_eq!(player.time, 1.0, "pause keeps the current time");

        player.play();
        player.stop();
        assert!(!player.playing);
        assert_eq!(player.time, 0.0, "stop rewinds");
    }

    #[test]
    fn select_clip_by_name_resets_time_and_reports_misses() {
        let mut player = demo_player();
        player.seek(1.5);

        assert!(player.select_clip("slide"));
        assert_eq!(player.current_clip_name(), Some("slide"));
        assert_eq!(player.current_clip().map(Clip::name), Some("slide"));
        assert_eq!(player.time, 0.0, "selecting rewinds");
        assert_eq!(player.duration(), 2.0);

        // A missing name leaves the previous selection alone.
        assert!(!player.select_clip("nope"));
        assert_eq!(player.current_clip_name(), Some("slide"));
        assert_eq!(player.time, 0.0);

        player.clear_clip();
        assert!(player.current_clip().is_none());
        assert_eq!(player.duration(), 0.0);
        assert_eq!(player.time, 0.0);
    }

    #[test]
    fn step_does_not_advance_while_paused() {
        let mut player = demo_player();
        assert!(player.select_clip("slide"));
        // Never started: no advance, no samples.
        assert_eq!(player.step(0.5), Vec::new());
        assert_eq!(player.time, 0.0);

        player.play();
        player.pause();
        assert_eq!(player.step(0.5), Vec::new());
        assert_eq!(player.time, 0.0);
    }

    #[test]
    fn step_without_a_clip_is_empty_and_freezes_time() {
        let mut player = Player::new();
        player.looping = true;
        player.play();

        assert_eq!(player.step(1.0), Vec::new());
        assert_eq!(player.time, 0.0, "no clip means nothing to play through");
    }

    #[test]
    fn step_advances_time_and_returns_sampled_poses() {
        let mut player = demo_player();
        player.select_clip("slide");
        player.play();

        let poses = player.step(0.5);
        assert_eq!(player.time, 0.5);
        assert_eq!(poses.len(), 1);
        assert_eq!(poses[0].0, NodeId(0));
        assert_eq!(poses[0].1.translation, Vec3::new(0.5, 0.0, 0.0));
        assert_eq!(poses[0].1.rotation, glam::Quat::IDENTITY);
        assert_eq!(poses[0].1.scale, Vec3::ONE);

        let poses = player.step(0.5);
        assert_eq!(player.time, 1.0);
        assert_eq!(poses[0].1.translation, Vec3::new(1.0, 0.0, 0.0));
    }

    #[test]
    fn speed_scales_how_fast_time_advances() {
        let mut player = demo_player();
        player.select_clip("slide");
        player.play();
        player.speed = 2.0;

        player.step(0.5);
        assert_eq!(player.time, 1.0);

        player.speed = 0.5;
        player.step(1.0);
        assert_eq!(player.time, 1.5);
    }

    #[test]
    fn non_looping_step_clamps_at_the_end_and_pauses() {
        let mut player = demo_player();
        player.select_clip("slide");
        player.play();

        let poses = player.step(10.0);
        assert_eq!(player.time, 2.0, "clamped to the clip's end");
        assert!(!player.playing, "clamping pauses playback");
        assert_eq!(poses.len(), 1);
        assert_eq!(poses[0].1.translation, Vec3::new(2.0, 0.0, 0.0));

        // Stays paused and frozen afterwards.
        assert_eq!(player.step(1.0), Vec::new());
        assert_eq!(player.time, 2.0);
    }

    #[test]
    fn non_looping_negative_speed_stops_at_the_start() {
        let mut player = demo_player();
        player.select_clip("slide");
        player.seek(1.0);
        player.speed = -1.0;
        player.play();

        player.step(0.5);
        assert_eq!(player.time, 0.5);
        assert!(player.playing);

        player.step(10.0);
        assert_eq!(player.time, 0.0);
        assert!(!player.playing);
    }

    #[test]
    fn looping_step_wraps_time_and_keeps_playing() {
        let mut player = demo_player();
        player.select_clip("slide");
        player.looping = true;
        player.play();

        // Exactly one full duration wraps back to the start.
        player.step(2.0);
        assert_eq!(player.time, 0.0);
        assert!(player.playing);

        // Stepping one second lands mid-clip; three more (4.0 total) wraps.
        player.step(1.0);
        assert_eq!(player.time, 1.0);
        player.step(3.0);
        assert_eq!(player.time, 0.0);
        assert!(player.playing);

        // A shorter overstep lands at the wrapped offset.
        player.step(2.75);
        assert_eq!(player.time, 0.75);
        let poses = player.step(0.0);
        assert_eq!(poses[0].1.translation, Vec3::new(0.75, 0.0, 0.0));
    }

    #[test]
    fn looping_negative_speed_wraps_from_the_start() {
        let mut player = demo_player();
        player.select_clip("slide");
        player.looping = true;
        player.speed = -1.0;
        player.play();

        player.step(0.5);
        assert_eq!(player.time, 1.5, "wraps into the tail of the clip");
        assert!(player.playing);
    }

    #[test]
    fn play_after_finishing_restarts_from_the_start() {
        let mut player = demo_player();
        player.select_clip("slide");
        player.play();
        player.step(10.0);
        assert_eq!(player.time, 2.0);
        assert!(!player.playing);

        player.play();
        assert!(player.playing);
        assert_eq!(player.time, 0.0, "restarts rather than staying at the end");

        let poses = player.step(0.5);
        assert_eq!(poses[0].1.translation, Vec3::new(0.5, 0.0, 0.0));
    }

    #[test]
    fn seek_clamps_to_the_clip_range() {
        let mut player = demo_player();
        player.select_clip("slide");

        player.seek(1.0);
        assert_eq!(player.time, 1.0);
        player.seek(99.0);
        assert_eq!(player.time, 2.0);
        player.seek(-4.0);
        assert_eq!(player.time, 0.0);
        player.seek(f64::NAN);
        assert_eq!(player.time, 0.0);

        // With no clip selected there is nowhere to seek.
        player.clear_clip();
        player.seek(1.0);
        assert_eq!(player.time, 0.0);
    }

    #[test]
    fn zero_duration_clip_holds_at_start_without_panicking() {
        let mut track = TransformTrack::targeting(NodeId(5));
        track.add_position_key(0.0, Vec3::ONE, Interp::Linear);
        let mut clip = Clip::new("frozen");
        clip.add_track(track);

        let mut player = Player::new();
        player.library_mut().add_clip(clip);
        player.select_clip("frozen");
        player.play();

        // Non-looping: a zero-length clip finishes on the first step.
        let poses = player.step(0.1);
        assert_eq!(player.time, 0.0);
        assert!(!player.playing);
        assert_eq!(poses.len(), 1);
        assert_eq!(poses[0].0, NodeId(5));
        assert_eq!(poses[0].1.translation, Vec3::ONE);

        // Looping: it holds at the start and keeps running.
        player.looping = true;
        player.play();
        let poses = player.step(0.1);
        assert_eq!(player.time, 0.0);
        assert!(player.playing);
        assert_eq!(poses.len(), 1);
    }

    #[test]
    fn step_ignores_non_finite_deltas() {
        let mut player = demo_player();
        player.select_clip("slide");
        player.play();

        player.step(f64::NAN);
        assert_eq!(player.time, 0.0);
        player.step(f64::INFINITY);
        assert_eq!(player.time, 0.0);
        assert!(player.playing);
    }
}
