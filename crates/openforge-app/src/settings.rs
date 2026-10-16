//! Preferences: the choices the Preferences window edits, the pure data the
//! frame code reads from them, and the one file they come back from.
//!
//! Everything here is plain state plus small maths — the ground grid's line
//! layout, the camera speed multipliers, the theme name and the fixed
//! keyboard-shortcut table — so the frame and the tests read the same
//! numbers, and nothing here needs an event loop to be checked. The file
//! side follows the same rule: the platform rule is a pure function of an
//! environment lookup, and load/save take a path, so every branch is
//! testable from any host.

use eframe::egui;
use glam::Vec3;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Which egui theme the app applies.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Theme {
    /// The file spells the themes the way the window labels them; the
    /// lowercase names are what gets written, and both spellings read.
    #[serde(alias = "Dark")]
    Dark,
    #[serde(alias = "Light")]
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

/// What the Preferences window edits. `Default` is what the app starts with,
/// what "Reset to defaults" restores, and — through `serde(default)` — what
/// a file that names only some of the fields keeps for the rest.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
#[serde(default)]
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

    /// The file name the preferences live under, inside the platform's
    /// OpenForge config directory.
    pub const FILE_NAME: &'static str = "settings.json";

    /// The settings as the JSON document the file holds. Pretty-printed
    /// with a closing newline: the file is the user's to read and to
    /// hand-edit.
    pub fn to_json(&self) -> String {
        let mut json = serde_json::to_string_pretty(self).expect("settings always serialise");
        json.push('\n');
        json
    }

    /// Parse a settings document.
    ///
    /// A field the document does not name keeps its default and a key it
    /// does not know is ignored, so an older build reads a newer file; but a
    /// document that does not parse is refused whole — the caller keeps its
    /// settings rather than half-believes a broken file.
    pub fn from_json(text: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(text)
    }

    /// Read the settings from `path`. A file that is missing or unreadable
    /// in any way reads as `None` (and is logged when it exists but is
    /// broken); the caller keeps its defaults.
    pub fn load(path: &Path) -> Option<Self> {
        let text = match std::fs::read_to_string(path) {
            Ok(text) => text,
            Err(error) => {
                if error.kind() != std::io::ErrorKind::NotFound {
                    log::warn!("could not read settings at {}: {error}", path.display());
                }
                return None;
            }
        };
        match Self::from_json(&text) {
            Ok(settings) => Some(settings),
            Err(error) => {
                log::warn!("could not parse settings at {}: {error}", path.display());
                None
            }
        }
    }

    /// Write the settings to `path`, making the config directory chain when
    /// it is not there yet.
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        if let Some(parent) = path.parent().filter(|parent| !parent.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, self.to_json())
    }
}

/// Which platform's config-directory convention [`platform_settings_path`]
/// follows. A value rather than `cfg` alone, so the tests can walk every
/// convention from any host.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Platform {
    Windows,
    MacOs,
    /// Linux, the BSDs, and anything else that speaks XDG.
    Unix,
}

/// The settings file's path for a platform, from a `getenv`-style lookup:
///
/// * Windows — `%APPDATA%\OpenForge\settings.json`, falling back to
///   `%LOCALAPPDATA%` when `APPDATA` is unset.
/// * macOS — `$HOME/Library/Application Support/OpenForge/settings.json`.
/// * elsewhere — `$XDG_CONFIG_HOME/openforge/settings.json`, falling back to
///   `$HOME/.config/openforge/settings.json`.
///
/// `None` when the variables the convention needs are unset: the caller then
/// keeps the preferences for the session only.
pub fn platform_settings_path(
    platform: Platform,
    env: impl Fn(&str) -> Option<String>,
) -> Option<PathBuf> {
    match platform {
        Platform::Windows => {
            let base = env("APPDATA").or_else(|| env("LOCALAPPDATA"))?;
            Some(PathBuf::from(base).join("OpenForge").join(Settings::FILE_NAME))
        }
        Platform::MacOs => {
            let home = env("HOME")?;
            Some(
                PathBuf::from(home)
                    .join("Library")
                    .join("Application Support")
                    .join("OpenForge")
                    .join(Settings::FILE_NAME),
            )
        }
        Platform::Unix => {
            let base = match env("XDG_CONFIG_HOME") {
                Some(xdg) if !xdg.is_empty() => PathBuf::from(xdg),
                _ => PathBuf::from(env("HOME")?).join(".config"),
            };
            Some(base.join("openforge").join(Settings::FILE_NAME))
        }
    }
}

/// Which convention this build follows. Written with `cfg!` rather than
/// `#[cfg]` so every variant stays constructed in every build (the branches
/// fold away at compile time) and no platform's build warns about the
/// others.
pub fn host_platform() -> Platform {
    if cfg!(target_os = "windows") {
        Platform::Windows
    } else if cfg!(target_os = "macos") {
        Platform::MacOs
    } else {
        Platform::Unix
    }
}

/// The settings file for this build: [`host_platform`] plus the process
/// environment. `None` only when the environment names no config directory.
pub fn default_settings_path() -> Option<PathBuf> {
    platform_settings_path(host_platform(), |key| std::env::var(key).ok())
}

/// The fixed global keybindings, as the Keyboard Shortcuts window lists them.
///
/// The tool hotkeys are *not* here: they come from `ActiveTool` (which is what
/// the frame, the toolbar and the registry's names all share), so the window
/// and the handlers cannot drift apart.
pub const SHORTCUTS: &[(&str, &str)] = &[
    ("Ctrl+O / Ctrl+S", "Open a project / save the project"),
    ("Ctrl+Shift+S", "Save the project under a new name"),
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

    /// A `getenv` stand-in over fixed pairs, for the platform-path tests.
    fn env_from<'a>(pairs: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
        move |key| pairs.iter().find(|(k, _)| *k == key).map(|(_, value)| (*value).to_string())
    }

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

    #[test]
    fn settings_survive_a_json_round_trip() {
        let settings = Settings {
            theme: Theme::Light,
            show_grid: false,
            grid_step: 2.5,
            grid_half: 4,
            orbit_speed: 1.5,
            pan_speed: 0.5,
            zoom_speed: 2.0,
        };
        let json = settings.to_json();
        assert!(json.ends_with('\n'), "the file ends its last line");
        assert!(json.contains("\"light\""), "the theme is stored by name: {json}");
        assert!(json.contains("\"grid_step\": 2.5"), "{json}");
        let back = Settings::from_json(&json).expect("the writer's own output parses");
        assert_eq!(back, settings);
        assert_eq!(
            Settings::from_json(&Settings::default().to_json()).unwrap(),
            Settings::default()
        );
    }

    #[test]
    fn a_partial_document_keeps_the_other_defaults() {
        let settings = Settings::from_json("{\"theme\": \"Light\"}").expect("labels parse too");
        assert_eq!(settings.theme, Theme::Light);
        assert_eq!(settings.grid_step, Settings::default().grid_step);
        assert!(settings.show_grid);
        let empty = Settings::from_json("{}").expect("an empty object is all defaults");
        assert_eq!(empty, Settings::default());
        let extra = Settings::from_json("{\"theme\": \"dark\", \"from_the_future\": true}")
            .expect("unknown keys from newer builds are ignored");
        assert_eq!(extra, Settings::default());
    }

    #[test]
    fn malformed_documents_are_refused_rather_than_guessed() {
        assert!(Settings::from_json("{\"theme\":").is_err(), "truncated JSON");
        assert!(Settings::from_json("\"Dark\"").is_err(), "not an object");
        assert!(Settings::from_json("{\"grid_step\": \"wide\"}").is_err(), "wrong type");
        assert!(
            Settings::from_json("{\"theme\": \"dusk\"}").is_err(),
            "a theme that does not exist"
        );
    }

    #[test]
    fn settings_files_load_save_and_fail_softly() {
        let dir = std::env::temp_dir().join(format!("openforge-settings-io-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let nested = dir.join("config").join("settings.json");

        assert_eq!(Settings::load(&nested), None, "a missing file reads as nothing");

        let mut settings = Settings::default();
        settings.theme = Theme::Light;
        settings.grid_step = 3.5;
        settings.save(&nested).expect("save makes the directory chain");
        assert_eq!(
            Settings::load(&nested).as_ref(),
            Some(&settings),
            "and reads back exactly"
        );

        std::fs::write(&nested, "openforge, but not JSON").unwrap();
        assert_eq!(Settings::load(&nested), None, "a corrupt file is refused, not guessed");
        let _ = std::fs::remove_dir_all(&dir);

        // A plain file name with no directory part is a legal target too.
        let flat = std::env::temp_dir()
            .join(format!("openforge-settings-flat-{}.json", std::process::id()));
        settings.save(&flat).expect("a bare name needs no directory made");
        assert_eq!(Settings::load(&flat), Some(settings));
        let _ = std::fs::remove_file(&flat);
    }

    #[test]
    fn the_settings_path_follows_each_platforms_convention() {
        let name = Settings::FILE_NAME;

        let windows = platform_settings_path(
            Platform::Windows,
            env_from(&[("APPDATA", "C:\\Users\\omar\\AppData\\Roaming")]),
        )
        .expect("APPDATA names the roaming directory");
        assert_eq!(
            windows,
            PathBuf::from("C:\\Users\\omar\\AppData\\Roaming").join("OpenForge").join(name)
        );
        let fallback = platform_settings_path(
            Platform::Windows,
            env_from(&[("LOCALAPPDATA", "C:\\Users\\omar\\AppData\\Local")]),
        )
        .expect("LOCALAPPDATA stands in when APPDATA is unset");
        assert!(
            fallback.starts_with(PathBuf::from("C:\\Users\\omar\\AppData\\Local").join("OpenForge"))
        );
        assert_eq!(platform_settings_path(Platform::Windows, env_from(&[])), None);

        let mac = platform_settings_path(Platform::MacOs, env_from(&[("HOME", "/Users/omar")]))
            .expect("HOME is all macOS needs");
        assert_eq!(
            mac,
            PathBuf::from("/Users/omar/Library/Application Support/OpenForge").join(name)
        );
        assert_eq!(platform_settings_path(Platform::MacOs, env_from(&[])), None);

        let xdg = platform_settings_path(
            Platform::Unix,
            env_from(&[("XDG_CONFIG_HOME", "/home/omar/.config2"), ("HOME", "/home/omar")]),
        )
        .expect("XDG_CONFIG_HOME comes first");
        assert_eq!(xdg, PathBuf::from("/home/omar/.config2/openforge").join(name));
        let home = platform_settings_path(Platform::Unix, env_from(&[("HOME", "/home/omar")]))
            .expect("HOME/.config is the fallback");
        assert_eq!(home, PathBuf::from("/home/omar/.config/openforge").join(name));
        let blank = platform_settings_path(
            Platform::Unix,
            env_from(&[("XDG_CONFIG_HOME", ""), ("HOME", "/home/omar")]),
        )
        .expect("an empty XDG_CONFIG_HOME falls back to HOME");
        assert_eq!(blank, PathBuf::from("/home/omar/.config/openforge").join(name));
        assert_eq!(platform_settings_path(Platform::Unix, env_from(&[])), None);

        // The host's own answer only has to be shaped right, env or no env.
        if let Some(path) = default_settings_path() {
            assert_eq!(path.file_name().unwrap(), name);
        }
    }
}
