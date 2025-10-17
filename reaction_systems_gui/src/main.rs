#![cfg_attr(not(debug_assertions), deny(warnings))]
#![warn(clippy::all, rust_2018_idioms)]
// hide console window on Windows in release
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use reaction_systems_gui::AppHandle;

// When compiling natively:
#[cfg(not(target_arch = "wasm32"))]
fn main() {
    use eframe::egui::Visuals;

    eframe::run_native(
        "Reaction Systems",
        eframe::NativeOptions::default(),
        Box::new(|cc| {
            cc.egui_ctx.set_visuals(Visuals::dark());
            #[cfg(feature = "persistence")]
            {
                Ok(Box::new(AppHandle::new(cc)))
            }
            #[cfg(not(feature = "persistence"))]
            {
                Ok(Box::<AppHandle>::default())
            }
        }),
    )
    .expect("Failed to run native example");
}
