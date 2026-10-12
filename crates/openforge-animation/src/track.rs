//! Ordered keyframe tracks with generic interpolation.

use crate::interpolation::{Interp, Interpolatable};
use crate::keyframe::Key;

/// An ordered sequence of [`Key`]s for a value of type `T`.
///
/// Keys are kept sorted by time, which lets [`Track::sample`] binary-search
/// the segment covering a given time. Supported value types include `f32`,
/// `Vec3` and `Quat` (see [`Interpolatable`]).
#[derive(Clone, Debug, PartialEq)]
pub struct Track<T> {
    keys: Vec<Key<T>>,
}

impl<T> Default for Track<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Track<T> {
    /// Create an empty track.
    pub fn new() -> Self {
        Self { keys: Vec::new() }
    }

    /// Whether the track holds no keys at all.
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// Number of keys in the track.
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    /// Time of the last key, or `0.0` when the track is empty.
    pub fn duration(&self) -> f64 {
        self.keys.last().map_or(0.0, |k| k.time)
    }

    /// All keys, sorted by ascending time.
    pub fn keys(&self) -> &[Key<T>] {
        &self.keys
    }

    /// The key at `index`, or `None` when out of range.
    pub fn key(&self, index: usize) -> Option<&Key<T>> {
        self.keys.get(index)
    }

    /// Insert a key at `time`, keeping the track sorted by time.
    ///
    /// If a key already exists at exactly `time` it is replaced in place
    /// (which also guarantees segments never have zero length). NaN times are
    /// treated as `0.0`. Returns the index the key occupies.
    pub fn add_key(&mut self, time: f64, value: T, interp: Interp) -> usize {
        let time = if time.is_nan() { 0.0 } else { time };
        // Index just past any keys already at `time`.
        let at = self.keys.partition_point(|k| k.time <= time);
        if at > 0 && self.keys[at - 1].time == time {
            self.keys[at - 1] = Key::new(time, value, interp);
            at - 1
        } else {
            self.keys.insert(at, Key::new(time, value, interp));
            at
        }
    }

    /// Remove the key at `index`, returning it, or `None` if out of range.
    pub fn remove_key(&mut self, index: usize) -> Option<Key<T>> {
        if index < self.keys.len() {
            Some(self.keys.remove(index))
        } else {
            None
        }
    }

    /// Remove every key from the track.
    pub fn clear(&mut self) {
        self.keys.clear();
    }
}

impl<T: Interpolatable> Track<T> {
    /// Evaluate the track at `time`.
    ///
    /// Times at or before the first key clamp to the first key's value, and
    /// times at or after the last key clamp to the last key's value, so the
    /// track always holds a defined pose outside its own range. An empty
    /// track returns `None`. Non-finite times fall back to the first key.
    ///
    /// The blend across each segment is driven by the *left* key's
    /// [`Interp`] mode.
    pub fn sample(&self, time: f64) -> Option<T> {
        let first = self.keys.first()?;
        let last = self.keys.last()?;
        if self.keys.len() == 1 || time <= first.time {
            return Some(first.value);
        }
        if time >= last.time {
            return Some(last.value);
        }

        // `first.time < time < last.time`, so the split point lands strictly
        // between two keys; the checked arithmetic guards against a key list
        // that somehow violates that (e.g. a non-finite time argument).
        let next = self.keys.partition_point(|k| k.time <= time);
        let prev = match next.checked_sub(1) {
            Some(p) => p,
            None => return Some(first.value),
        };
        let (a, b) = match (self.keys.get(prev), self.keys.get(next)) {
            (Some(a), Some(b)) => (a, b),
            _ => return Some(last.value),
        };

        let span = b.time - a.time;
        let t = if span > 0.0 {
            ((time - a.time) / span) as f32
        } else {
            0.0
        };
        Some(T::interpolate(
            a.value,
            b.value,
            t.clamp(0.0, 1.0),
            a.interp,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Quat, Vec3};

    fn scalar_track(interp: Interp) -> Track<f32> {
        let mut track = Track::new();
        track.add_key(0.0, 0.0, interp);
        track.add_key(1.0, 10.0, Interp::Linear);
        track
    }

    #[test]
    fn new_track_is_empty_with_zero_duration() {
        let mut track: Track<f32> = Track::new();
        assert!(track.is_empty());
        assert_eq!(track.len(), 0);
        assert_eq!(track.duration(), 0.0);
        assert_eq!(track.sample(5.0), None);
        assert_eq!(track.remove_key(0), None);
    }

    #[test]
    fn add_key_keeps_keys_sorted_by_time() {
        let mut track: Track<f32> = Track::new();
        assert_eq!(track.add_key(1.0, 1.0, Interp::Linear), 0);
        assert_eq!(track.add_key(0.0, 0.0, Interp::Linear), 0);
        assert_eq!(track.add_key(2.0, 2.0, Interp::Linear), 2);
        assert_eq!(track.add_key(1.5, 1.5, Interp::Linear), 2);

        let times: Vec<f64> = track.keys().iter().map(|k| k.time).collect();
        assert_eq!(times, vec![0.0, 1.0, 1.5, 2.0]);
        let values: Vec<f32> = track.keys().iter().map(|k| k.value).collect();
        assert_eq!(values, vec![0.0, 1.0, 1.5, 2.0]);
        assert_eq!(track.key(2).map(|k| k.value), Some(1.5));
        assert_eq!(track.key(99), None);
    }

    #[test]
    fn add_key_at_existing_time_replaces_in_place() {
        let mut track: Track<f32> = Track::new();
        track.add_key(0.0, 0.0, Interp::Linear);
        track.add_key(1.0, 10.0, Interp::Step);
        let index = track.add_key(1.0, 99.0, Interp::Smooth);

        assert_eq!(index, 1);
        assert_eq!(track.len(), 2);
        assert_eq!(track.key(1).map(|k| k.value), Some(99.0));
        assert_eq!(track.key(1).map(|k| k.interp), Some(Interp::Smooth));
        // No NaN segment: sampling across the replaced key still works.
        assert_eq!(track.sample(0.5), Some(49.5));
    }

    #[test]
    fn add_key_treats_nan_time_as_zero() {
        let mut track: Track<f32> = Track::new();
        track.add_key(1.0, 1.0, Interp::Linear);
        track.add_key(f64::NAN, 5.0, Interp::Linear);
        assert_eq!(track.keys()[0].time, 0.0);
        assert_eq!(track.keys()[0].value, 5.0);
    }

    #[test]
    fn duration_is_time_of_last_key() {
        let mut track: Track<f32> = Track::new();
        track.add_key(0.5, 0.0, Interp::Linear);
        track.add_key(3.25, 1.0, Interp::Linear);
        track.add_key(2.0, 0.5, Interp::Linear);
        assert_eq!(track.duration(), 3.25);

        // Dropping the last key moves the duration back.
        let removed = track.remove_key(2);
        assert_eq!(removed.map(|k| k.time), Some(3.25));
        assert_eq!(track.duration(), 2.0);
    }

    #[test]
    fn remove_key_reports_out_of_range() {
        let mut track = scalar_track(Interp::Linear);
        assert_eq!(track.len(), 2);

        let removed = track.remove_key(0);
        assert_eq!(removed.map(|k| k.value), Some(0.0));
        assert_eq!(track.len(), 1);

        assert!(track.remove_key(7).is_none());
        assert_eq!(track.len(), 1);

        track.clear();
        assert!(track.is_empty());
    }

    #[test]
    fn single_key_track_samples_that_key_everywhere() {
        let mut track: Track<f32> = Track::new();
        track.add_key(2.0, 42.0, Interp::Linear);

        assert_eq!(track.sample(-100.0), Some(42.0));
        assert_eq!(track.sample(0.0), Some(42.0));
        assert_eq!(track.sample(2.0), Some(42.0));
        assert_eq!(track.sample(1000.0), Some(42.0));
        assert_eq!(track.sample(f64::NAN), Some(42.0));
    }

    #[test]
    fn sample_clamps_before_first_and_after_last_key() {
        let mut track: Track<f32> = Track::new();
        track.add_key(1.0, 10.0, Interp::Linear);
        track.add_key(3.0, 30.0, Interp::Linear);

        assert_eq!(track.sample(-5.0), Some(10.0));
        assert_eq!(track.sample(1.0), Some(10.0));
        assert_eq!(track.sample(3.0), Some(30.0));
        assert_eq!(track.sample(500.0), Some(30.0));
        assert_eq!(track.sample(f64::NEG_INFINITY), Some(10.0));
        assert_eq!(track.sample(f64::INFINITY), Some(30.0));
    }

    #[test]
    fn linear_sample_is_straight_line_at_midpoint() {
        let track = scalar_track(Interp::Linear);
        assert_eq!(track.sample(0.5), Some(5.0));
        assert_eq!(track.sample(0.25), Some(2.5));
        // Endpoints hit the keys exactly.
        assert_eq!(track.sample(0.0), Some(0.0));
        assert_eq!(track.sample(1.0), Some(10.0));
    }

    #[test]
    fn interp_modes_agree_at_midpoint_and_differ_earlier() {
        let linear = scalar_track(Interp::Linear).sample(0.5);
        let smooth = scalar_track(Interp::Smooth).sample(0.5);
        let ease = scalar_track(Interp::Ease).sample(0.5);
        let step = scalar_track(Interp::Step).sample(0.5);

        // Both symmetric curves pass through the midpoint, like linear.
        assert!((linear.unwrap() - 5.0).abs() < 1e-5);
        assert!((smooth.unwrap() - 5.0).abs() < 1e-5);
        assert!((ease.unwrap() - 5.0).abs() < 1e-5);
        // Step holds the first key for the whole segment.
        assert_eq!(step, Some(0.0));

        // A quarter of the way in, all four modes differ.
        let linear = scalar_track(Interp::Linear).sample(0.25).unwrap();
        let smooth = scalar_track(Interp::Smooth).sample(0.25).unwrap();
        let ease = scalar_track(Interp::Ease).sample(0.25).unwrap();
        let step = scalar_track(Interp::Step).sample(0.25).unwrap();
        assert!((linear - 2.5).abs() < 1e-5);
        assert!((smooth - 1.5625).abs() < 1e-5);
        assert!((ease - 0.625).abs() < 1e-5);
        assert_eq!(step, 0.0);
    }

    #[test]
    fn step_holds_first_value_until_the_next_key_time() {
        let mut track: Track<f32> = Track::new();
        track.add_key(0.0, 10.0, Interp::Step);
        track.add_key(1.0, 20.0, Interp::Step);

        assert_eq!(track.sample(0.0), Some(10.0));
        assert_eq!(track.sample(0.999), Some(10.0));
        // At and past the last key the value snaps to it.
        assert_eq!(track.sample(1.0), Some(20.0));
        assert_eq!(track.sample(2.0), Some(20.0));
    }

    #[test]
    fn segment_uses_the_left_keys_interp_mode() {
        let mut track: Track<f32> = Track::new();
        track.add_key(0.0, 0.0, Interp::Linear);
        // The right key's Step mode belongs to the *next* segment.
        track.add_key(1.0, 10.0, Interp::Step);
        assert_eq!(track.sample(0.5), Some(5.0));
    }

    #[test]
    fn vec3_track_lerps_each_component() {
        let mut track: Track<Vec3> = Track::new();
        track.add_key(0.0, Vec3::ZERO, Interp::Linear);
        track.add_key(1.0, Vec3::new(2.0, 4.0, 6.0), Interp::Linear);

        assert_eq!(track.sample(0.5), Some(Vec3::new(1.0, 2.0, 3.0)));
        assert_eq!(track.sample(-1.0), Some(Vec3::ZERO));
        assert_eq!(track.sample(2.0), Some(Vec3::new(2.0, 4.0, 6.0)));
    }

    #[test]
    fn quat_track_slerps_rather_than_component_lerp() {
        let mut track: Track<Quat> = Track::new();
        track.add_key(0.0, Quat::IDENTITY, Interp::Linear);
        track.add_key(1.0, Quat::from_rotation_x(2.0), Interp::Linear);

        let mid = track.sample(0.5).expect("midpoint exists");
        // A true slerp lands exactly halfway in angle space (1.0 rad); a
        // per-component lerp would be off by a wide margin.
        assert!(mid.angle_between(Quat::from_rotation_x(1.0)) < 1e-3);
        assert!((mid.length() - 1.0).abs() < 1e-5);

        // Endpoint returns the last key bit-for-bit.
        let end = track.sample(1.0).expect("end exists");
        assert!(end.abs_diff_eq(Quat::from_rotation_x(2.0), 1e-6));
    }

    #[test]
    fn quat_track_takes_shortest_path_when_dot_is_negative() {
        let target = -Quat::from_rotation_x(2.0);
        assert!(Quat::IDENTITY.dot(target) < 0.0, "precondition");

        let mut track: Track<Quat> = Track::new();
        track.add_key(0.0, Quat::IDENTITY, Interp::Linear);
        track.add_key(1.0, target, Interp::Linear);

        let mid = track.sample(0.5).expect("midpoint exists");
        // Short arc: half of 2.0 rad, even though the stored key points at
        // the opposite hemisphere representative of the same rotation.
        assert!(mid.angle_between(Quat::from_rotation_x(1.0)) < 1e-3);
    }

    #[test]
    fn quat_step_holds_rotation() {
        let mut track: Track<Quat> = Track::new();
        let start = Quat::from_rotation_y(0.4);
        track.add_key(0.0, start, Interp::Step);
        track.add_key(1.0, Quat::from_rotation_y(1.4), Interp::Linear);

        let held = track.sample(0.75).expect("value exists");
        assert!(held.angle_between(start) < 1e-5);
    }
}
