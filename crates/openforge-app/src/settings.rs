//! Preferences: the choices the Preferences window edits, and the pure data
//! the frame code reads from them.
//!
//! Everything here is plain state plus small maths — the ground grid's line
//! layout, the camera speed multipliers, the theme name and the fixed
//! keyboard-shortcut table — so the frame and the tests read the same
//! numbers, and nothing here needs an event loop to be checked.

use eframe::egui;
use glam::Vec3;

/// Which egui theme the app applies.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Theme {
    Dark,
    Light,
}

impl Theme {
    pub fn label(self) -> &'static str {
        match self {
            Theme::Dark => "Dark",
            Theme::Light => "Light",
        }
    }

    /// Both choices, in the order the Preferences window offers them.
    pub fn all() -> [Theme; 2] {
        [Theme::Dark, Theme::Light]
    }
}

/// One ground-grid line, in world space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GridLine {
    pub start: Vec3,
    pub end: Vec3,
    /// The line at the origin (drawn heavier, in the axis's own colour).
    pub axis: bool,
    /// True when the line runs along X (so its origin line *is* the Z axis).
    pub along_x: bool,
}

/// What the Preferences window edits. `Default` is what the app starts with
/// and what "Reset to defaults" restores.
#[derive(Clone, PartialEq, Debug)]
pub struct Settings {
    pub theme: Theme,
    pub show_grid: bool,
    /// World units between grid lines.
    pub grid_step: f32,
    /// Lines drawn each side of the origin.
    pub grid_half: u32,
    /// Multipliers on the middle-drag orbit/pan and the scroll zoom.
    pub orbit_speed: f32,
    pub pan_speed: f32,
    pub zoom_speed: f32,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            theme: Theme::Dark,
            show_grid: true,
            grid_step: 1.0,
            grid_half: 10,
            orbit_speed: 1.0,
            pan_speed: 1.0,
            zoom_speed: 1.0,
        }
    }
}

impl Settings {
    /// The ground grid's lines, centred on the origin.
    ///
    /// A step the window could not mean (zero, negative, non-finite) falls
    /// back to `1.0` and the line count is capped, so a typed-in value can
    /// never ask the painter for an unbounded number of lines.
    pub fn grid_lines(&self) -> Vec<GridLine> {
        let step = if self.grid_step.is_finite() && self.grid_step > 0.0 {
            self.grid_step
        } else {
            1.0
        };
        let half = self.grid_half.min(64) as i32;
        let extent = half as f32 * step;
        let mut lines = Vec::with_capacity((half as usize * 2 + 1) * 2);
        for i in -half..=half {
            let f = i as f32 * step;
            lines.push(GridLine {
                start: Vec3::new(-extent, 0.0, f),
                end: Vec3::new(extent, 0.0, f),
                axis: i == 0,
                along_x: true,
            });
            lines.push(GridLine {
                start: Vec3::new(f, 0.0, -extent),
                end: Vec3::new(f, 0.0, extent),
                axis: i == 0,
                along_x: false,
            });
        }
        lines
    }

    /// A multiplier the camera can use: finite, positive and within a range a
    /// DragValue could be typed into. Anything else reads as the default.
    pub fn sane_speed(speed: f32) -> f32 {
        if speed.is_finite() && speed > 0.0 {
            speed.clamp(0.05, 10.0)
        } else {
            1.0
        }
    }

    /// The orbit multiplier the drag handler applies.
    pub fn orbit_step(&self) -> f32 {
        Self::sane_speed(self.orbit_speed)
    }

    /// The pan multiplier the drag handler applies.
    pub fn pan_step(&self) -> f32 {
        Self::sane_speed(self.pan_speed)
    }

    /// The zoom multiplier the scroll handler applies.
    pub fn zoom_step(&self) -> f32 {
        Self::sane_speed(self.zoom_speed)
    }

    /// Put the chosen theme on the egui context.
    pub fn apply(&self, ctx: &egui::Context) {
        let visuals = match self.theme {
            Theme::Dark => egui::Visuals::dark(),
            Theme::Light => egui::Visuals::light(),
        };
        ctx.set_visuals(visuals);
    }

    /// Back to the startup choices.
    pub fn reset(&mut self) {
        *self = Settings::default();
    }
}

/// The fixed global keybindings, as the Keyboard Shortcuts window lists them.
///
/// The tool hotkeys are *not* here: they come from `ActiveTool` (which is what
/// the frame, the toolbar and the registry's names all share), so the window
/// and the handlers cannot drift apart.
pub const SHORTCUTS: &[(&str, &str)] = &[
    ("Ctrl+Z / Ctrl+Shift+Z", "Undo / redo"),
    ("Ctrl+A / Ctrl+Shift+A", "Select all elements / clear the selection"),
    ("Ctrl+I", "Invert the element selection"),
    ("Ctrl+Plus / Ctrl+Minus", "Grow / shrink the element selection"),
    ("Ctrl+D", "Duplicate the selected nodes"),
    ("Delete", "Delete the selection (level-aware)"),
    ("G", "Cycle snapping: off → grid → vertex"),
    ("P", "Toggle select-through (picks ignore occlusion)"),
    ("Z", "Cycle wireframe: off → selection → all"),
    ("H / Shift+H / U", "Hide / isolate / unhide"),
    ("F", "Frame all"),
    ("L / Shift+L", "Edge loop / edge ring (edge level)"),
    ("Space", "Start or pause the animation"),
    ("X / I / O", "Extrude / inset / poke (face level)"),
    ("S / V", "Subdivide / weld"),
    ("B / T / N / M", "Bevel / triangulate / flip normals / mirror X"),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_grid_is_the_old_painter_constant() {
        let settings = Settings::default();
        assert!(settings.show_grid);
        assert_eq!(settings.theme, Theme::Dark);
        for step in [settings.orbit_step(), settings.pan_step(), settings.zoom_step()] {
            assert_eq!(step, 1.0);
        }
        let lines = settings.grid_lines();
        assert_eq!(lines.len(), 42, "21 lines each way, ±10 around the origin");
        let first = lines[0];
        assert_eq!(first.start, Vec3::new(-10.0, 0.0, -10.0));
        assert_eq!(first.end, Vec3::new(10.0, 0.0, -10.0));
        assert!(first.along_x && !first.axis);
        let axes: Vec<&GridLine> = lines.iter().filter(|line| line.axis).collect();
        assert_eq!(axes.len(), 2, "the X and Z axles");
        assert_eq!(axes[0].end, Vec3::new(10.0, 0.0, 0.0), "the X axle runs along X");
        assert_eq!(axes[1].end, Vec3::new(0.0, 0.0, 10.0), "and the Z axle along Z");
    }

    #[test]
    fn the_step_spaces_the_lines_and_impossible_values_fall_back() {
        let mut settings = Settings { grid_step: 2.5, grid_half: 2, ..Settings::default() };
        let lines = settings.grid_lines();
        assert_eq!(lines.len(), 10);
        assert_eq!(lines[0].start, Vec3::new(-5.0, 0.0, -5.0));
        assert_eq!(lines[2].start, Vec3::new(-5.0, 0.0, -2.5), "one step up in Z");
        assert!(lines[4].axis, "the middle pair sits on the origin");

        settings.grid_half = 10;
        settings.grid_step = 0.0;
        assert_eq!(settings.grid_lines().len(), 42, "zero falls back to 1.0");
        settings.grid_step = f32::NAN;
        assert_eq!(settings.grid_lines().len(), 42, "and so does a NaN");
        settings.grid_step = -3.0;
        assert_eq!(settings.grid_lines().len(), 42, "and a negative step");
        settings.grid_step = 1.0;
        settings.grid_half = 1000;
        assert_eq!(settings.grid_lines().len(), (64 * 2 + 1) * 2, "the count is capped at 64");
    }

    #[test]
    fn speeds_are_clamped_and_non_finite_ones_are_ignored() {
        let mut settings = Settings::default();
        settings.orbit_speed = 0.0;
        assert_eq!(settings.orbit_step(), 1.0);
        settings.orbit_speed = f32::INFINITY;
        assert_eq!(settings.orbit_step(), 1.0);
        settings.orbit_speed = -2.0;
        assert_eq!(settings.orbit_step(), 1.0);
        settings.pan_speed = 100.0;
        assert_eq!(settings.pan_step(), 10.0, "clamped to what a DragValue offers");
        settings.zoom_speed = 0.001;
        assert_eq!(settings.zoom_step(), 0.05);
        settings.reset();
        assert_eq!(settings, Settings::default());
    }

    #[test]
    fn the_shortcut_table_is_filled_in() {
        for (keys, action) in SHORTCUTS {
            assert!(!keys.is_empty() && !action.is_empty());
        }
        for expected in ["Ctrl+Z", "Delete", "G", "P", "Space", "F"] {
            assert!(
                SHORTCUTS.iter().any(|(keys, _)| keys.contains(expected)),
                "`{expected}` is missing"
            );
        }
    }
}
