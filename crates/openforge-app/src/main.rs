//! OpenForge desktop application entry point.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod app;
mod file_options;
mod gizmo;
mod gpu_viewport;
mod settings;
mod snap;
mod uv_editor;
mod viewport;

fn main() -> eframe::Result {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    // The window comes back the size the user left it, when the settings file
    // says so — the workspace record the app keeps.
    let size = settings::default_settings_path()
        .and_then(|path| settings::Settings::load(&path))
        .map(|s| s.window_size)
        .filter(|s| s[0] >= 800.0 && s[1] >= 500.0 && s[0].is_finite() && s[1].is_finite())
        .unwrap_or([1280.0, 800.0]);

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size(size)
            .with_min_inner_size([800.0, 500.0])
            .with_title("OpenForge"),
        // Render the viewport through wgpu. If no adapter can be created the
        // app still runs and falls back to the CPU painter viewport.
        renderer: eframe::Renderer::Wgpu,
        ..Default::default()
    };

    eframe::run_native(
        "OpenForge",
        options,
        Box::new(|cc| Ok(Box::new(app::OpenForgeApp::new(cc)))),
    )
}
