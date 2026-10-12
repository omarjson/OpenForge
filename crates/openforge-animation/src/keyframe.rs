//! A single keyframe.

use crate::interpolation::Interp;

/// One keyframe: the value of a track at a specific point in time.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Key<T> {
    /// Position on the timeline, in seconds from the clip's start.
    pub time: f64,
    /// The value this key holds at `time`.
    pub value: T,
    /// How to blend from this key toward the **next** key in its track.
    pub interp: Interp,
}

impl<T> Key<T> {
    /// Create a keyframe at `time` with the given value and outgoing
    /// interpolation mode.
    pub fn new(time: f64, value: T, interp: Interp) -> Self {
        Self { time, value, interp }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_stores_its_fields() {
        let key = Key::new(1.25, 7.0f32, Interp::Smooth);
        assert_eq!(key.time, 1.25);
        assert_eq!(key.value, 7.0);
        assert_eq!(key.interp, Interp::Smooth);
    }

    #[test]
    fn keys_compare_by_all_fields() {
        let a = Key::new(1.0, 2.0f32, Interp::Linear);
        let b = Key::new(1.0, 2.0f32, Interp::Linear);
        let c = Key::new(1.0, 3.0f32, Interp::Linear);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }
}
