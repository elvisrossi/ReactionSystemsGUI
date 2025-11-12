use std::fmt::Debug;
use std::hash::Hash;
use std::sync::{Arc, Mutex};

use eframe::egui;
use layout::backends::svg::SVGWriter;
use layout::gv::{self, GraphBuilder};

#[cfg_attr(
    feature = "persistence",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Clone, Default)]
pub(crate) struct Svg {
    image:    egui::ColorImage,
    /// original size of the svg
    svg_size: egui::Vec2,

    original: String,

    #[cfg_attr(feature = "persistence", serde(skip))]
    svg_texture: Arc<Mutex<Option<egui::TextureHandle>>>,
}

impl Svg {
    pub(crate) fn parse_dot_string(dot_str: &str) -> Result<Svg, String> {
        let mut fontdb = fontdb::Database::new();
        fontdb.load_system_fonts();

        let mut parser = gv::DotParser::new(dot_str);
        let g = match parser.process() {
            | Ok(g) => g,
            | Err(_) =>
            // errors are printed to sdtout so we ignore them
                return Err("Could not parse dot string.".into()),
        };

        let mut gb = GraphBuilder::new();
        gb.visit_graph(&g);
        let mut graph = gb.get();
        let mut svg = SVGWriter::new();
        graph.do_it(false, false, false, &mut svg);
        let content = svg.finalize();

        let svg_tree =
            match resvg::usvg::Tree::from_str(&content, &resvg::usvg::Options {
                dpi: 92.,
                font_family: "Andale Mono".into(),
                fontdb: Arc::new(fontdb),
                ..Default::default()
            }) {
                | Ok(svg) => svg,
                | Err(err) => return Err(format!("{}", err)),
            };

        let svg_size =
            egui::vec2(svg_tree.size().width(), svg_tree.size().height());

        let mut pixmap =
            resvg::tiny_skia::Pixmap::new(svg_size.x as _, svg_size.y as _)
                .expect("Could not allocate svg");
        let pixmap_mut = &mut pixmap.as_mut();
        resvg::render(&svg_tree, Default::default(), pixmap_mut);
        let pixmap = pixmap_mut.to_owned();

        let image = egui::ColorImage::from_rgba_unmultiplied(
            [pixmap.width() as _, pixmap.height() as _],
            pixmap.data(),
        );

        let svg = Svg {
            image,
            original: content,
            svg_size,
            svg_texture: Arc::new(Mutex::new(None)),
        };

        Ok(svg)
    }

    pub(crate) fn get_texture(
        &self,
        ctx: &egui::Context,
    ) -> egui::TextureHandle {
        let tx = self.svg_texture.lock().expect("Poisoned");
        if tx.is_some() {
            (*tx).clone().unwrap()
        } else {
            std::mem::drop(tx);
            let svg_texture =
                ctx.load_texture("svg", self.image.clone(), Default::default());
            *self.svg_texture.lock().expect("Poisoned") =
                Some(svg_texture.clone());
            svg_texture
        }
    }

    pub(crate) fn rasterize(&self) -> Result<Vec<u8>, String> {
        let svg_tree = match resvg::usvg::Tree::from_str(
            &self.original,
            &resvg::usvg::Options {
                dpi: 92.,
                font_family: "Andale Mono".into(),
                ..Default::default()
            },
        ) {
            | Ok(svg) => svg,
            | Err(err) => return Err(format!("{}", err)),
        };
        let mut pixmap = resvg::tiny_skia::Pixmap::new(
            self.svg_size.x as _,
            self.svg_size.y as _,
        )
        .expect("Could not allocate svg");
        let pixmap_mut = &mut pixmap.as_mut();
        resvg::render(&svg_tree, Default::default(), pixmap_mut);
        let pixmap = pixmap_mut.to_owned();

        match pixmap.encode_png() {
            | Ok(png) => Ok(png),
            | Err(e) => Err(format!("{}", e)),
        }
    }
}

impl Debug for Svg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[image: {:?}, svg_size: {:?}, svg_texture: {}",
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
