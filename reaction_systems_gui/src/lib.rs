#![forbid(unsafe_code)]
#![warn(clippy::all, rust_2018_idioms)]
// Forbid warnings in release builds
#![cfg_attr(not(debug_assertions), deny(warnings))]

mod app;
mod app_logic;
mod helper;
mod svg;

pub use app::AppHandle;

// If compiling for web
#[cfg(target_arch = "wasm32")]
mod web;

// Export endpoints for wasm
#[cfg(target_arch = "wasm32")]
pub use web::*;
