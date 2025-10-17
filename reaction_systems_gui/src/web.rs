#![allow(clippy::mem_forget)] // False positives from #[wasm_bindgen] macro

use eframe::wasm_bindgen::prelude::*;
use eframe::wasm_bindgen::{self};

use crate::app::NodeGraphExample;

#[derive(Clone)]
#[wasm_bindgen()]
pub struct WebHandle {
    runner: eframe::WebRunner,
}

#[wasm_bindgen()]
impl WebHandle {
    /// Installs a panic hook, then returns.
    #[allow(clippy::new_without_default, clippy::allow_attributes)]
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        // Redirect [`log`] message to `console.log` and friends:
        let log_level = if cfg!(debug_assertions) {
            log::LevelFilter::Trace
        } else {
            log::LevelFilter::Debug
        };
        eframe::WebLogger::init(log_level).ok();

        Self {
            runner: eframe::WebRunner::new(),
        }
    }

    /// Call this once from JavaScript to start the app.
    #[wasm_bindgen]
    pub async fn start(
        &self,
        canvas: eframe::web_sys::HtmlCanvasElement,
    ) -> Result<(), wasm_bindgen::JsValue> {
        self.runner
            .start(
                canvas,
                eframe::WebOptions::default(),
                Box::new(|_cc| {
                    #[cfg(feature = "persistence")]
                    {
                        Ok(Box::new(NodeGraphExample::new(_cc)))
                    }
                    #[cfg(not(feature = "persistence"))]
                    {
                        Ok(Box::<NodeGraphExample>::default())
                    }
                }),
            )
            .await
    }

    #[wasm_bindgen]
    pub fn destroy(&self) {
        self.runner.destroy();
    }

    /// The JavaScript can check whether or not the app has crashed:
    #[wasm_bindgen]
    pub fn has_panicked(&self) -> bool {
        self.runner.has_panicked()
    }

    #[wasm_bindgen]
    pub fn panic_message(&self) -> Option<String> {
        self.runner.panic_summary().map(|s| s.message())
    }

    #[wasm_bindgen]
    pub fn panic_callstack(&self) -> Option<String> {
        self.runner.panic_summary().map(|s| s.callstack())
    }
}
