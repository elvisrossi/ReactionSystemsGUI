use std::{fmt::Debug, hash::Hash, sync::{Arc, Mutex}};

use layout::{backends::svg::SVGWriter, gv::{self, GraphBuilder}};
use eframe::egui;

#[cfg_attr(
    feature = "persistence",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Clone, Default)]
pub(crate) struct Svg {
    image: egui::ColorImage,
    /// original size of the svg
    svg_size: egui::Vec2,

    original: String,

    #[cfg_attr(feature = "persistence", serde(skip))]
    svg_texture: Arc<Mutex<Option<egui::TextureHandle>>>,
}

impl Svg {
    pub(crate) fn parse_dot_string(dot_str: &str) -> Result<Svg, String> {
        let mut parser = gv::DotParser::new(dot_str);
        let g = match parser.process() {
            Ok(g) => g,
            Err(_) =>
                // errors are printed to sdtout so we ignore them
                return Err("Could not parse dot string.".into()),
        };

        let mut gb = GraphBuilder::new();
        gb.visit_graph(&g);
        let mut graph = gb.get();
        let mut svg = SVGWriter::new();
        graph.do_it(
            false,
            false,
            false,
            &mut svg,
        );
        let content = svg.finalize();

        let svg = match nsvg::parse_str(&content, nsvg::Units::Pixel, 96.0) {
            Ok(svg) => svg,
            Err(nsvg_err) => return Err(format!("{}", nsvg_err)),
        };

        let svg_size = egui::vec2(svg.width(), svg.height());

        let (w, h, data) = match svg.rasterize_to_raw_rgba(1.) {
            Ok(o) => o,
            Err(e) => return Err(format!("{}", e)),
        };

        let image = egui::ColorImage::from_rgba_unmultiplied([w as _, h as _], &data);

        let svg = Svg { image,
                        original: content,
                        svg_size,
                        svg_texture: Arc::new(Mutex::new(None)) };

        Ok(svg)
    }

    pub(crate) fn get_texture(&self, ctx: &egui::Context) -> egui::TextureHandle {
        let tx = self.svg_texture.lock().expect("Poisoned");
        if tx.is_some() {
            (*tx).clone().unwrap()
        } else {
            std::mem::drop(tx);
            let svg_texture = ctx.load_texture("svg", self.image.clone(), Default::default());
            *self.svg_texture.lock().expect("Poisoned") = Some(svg_texture.clone());
            svg_texture
        }
    }

    pub(crate) fn rasterize(&self) -> Result<nsvg::image::ImageBuffer<nsvg::image::Rgba<u8>, Vec<u8>>, String> {
        let svg = match nsvg::parse_str(&self.original, nsvg::Units::Pixel, 96.0) {
            Ok(svg) => svg,
            Err(nsvg_err) => return Err(format!("{}", nsvg_err)),
        };
        let data = match svg.rasterize(1.) {
            Ok(o) => o,
            Err(e) => return Err(format!("{}", e)),
        };

        Ok(data)
    }
}

impl Debug for Svg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[image: {:?}, svg_size: {:?}, svg_texture: {}",
               self.image,
               self.svg_size,
               if self.svg_texture.lock().expect("Poisoned").is_some() {
                   "Some(...)"
               } else {
                   "None"
               }
        )
    }
}

impl Hash for Svg {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        macro_rules! hash_float {
            ($name:expr) => (
                let bits = if $name.is_nan() {
                    // "Canonical" NaN.
                    0x7fc00000
                } else {
                    // A trick taken from the `ordered-float` crate: -0.0 + 0.0 == +0.0.
                    // https://github.com/reem/rust-ordered-float/blob/1841f0541ea0e56779cbac03de2705149e020675/src/lib.rs#L2178-L2181
                    ($name + 0.0).to_bits()
                };
                bits.hash(state);
            );
        }

        hash_float!(self.svg_size.x);
        hash_float!(self.svg_size.y);
        self.image.pixels.hash(state);
        self.image.size.hash(state);
        self.original.hash(state);
        hash_float!(self.image.source_size.x);
        hash_float!(self.image.source_size.y);
        self.svg_texture.lock().expect("Poisoned").hash(state);
    }
}
