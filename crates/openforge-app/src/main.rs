//! OpenForge desktop application entry point.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod app;
mod gpu_viewport;
mod uv_editor;
mod viewport;

fn main() -> eframe::Result {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 800.0])
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
