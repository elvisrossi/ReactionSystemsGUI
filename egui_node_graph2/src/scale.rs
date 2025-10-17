use egui::epaint::{CornerRadiusF32, Shadow};
use egui::style::WidgetVisuals;
use egui::{CornerRadius, Margin, Stroke, Style, Vec2};

// Copied from https://github.com/gzp-crey/shine

pub trait Scale {
    fn scale(&mut self, amount: f32);

    fn scaled(&self, amount: f32) -> Self
    where
        Self: Clone,
    {
        let mut scaled = self.clone();
        scaled.scale(amount);
        scaled
    }
}

impl Scale for Vec2 {
    fn scale(&mut self, amount: f32) {
        self.x *= amount;
        self.y *= amount;
    }
}

impl Scale for Margin {
    fn scale(&mut self, amount: f32) {
        self.left = (self.leftf() * amount).round() as _;
        self.right = (self.rightf() * amount).round() as _;
        self.top = (self.topf() * amount).round() as _;
        self.bottom = (self.bottomf() * amount).round() as _;
    }
}

impl Scale for CornerRadiusF32 {
    fn scale(&mut self, amount: f32) {
        self.ne *= amount;
        self.nw *= amount;
        self.se *= amount;
        self.sw *= amount;
    }
}

impl Scale for CornerRadius {
    fn scale(&mut self, amount: f32) {
        self.nw = (self.nw as f32 * amount) as u8;
        self.ne = (self.ne as f32 * amount) as u8;
        self.sw = (self.sw as f32 * amount) as u8;
        self.se = (self.se as f32 * amount) as u8;
    }
}

impl Scale for Stroke {
    fn scale(&mut self, amount: f32) {
        self.width *= amount;
    }
}

impl Scale for Shadow {
    fn scale(&mut self, amount: f32) {
        self.spread =
            ((self.spread as f32) * amount.clamp(0.4, 1.)).round() as _;
    }
}

impl Scale for WidgetVisuals {
    fn scale(&mut self, amount: f32) {
        self.bg_stroke.scale(amount);
        self.fg_stroke.scale(amount);
        self.corner_radius.scale(amount);
        self.expansion *= amount.clamp(0.4, 1.);
    }
}

impl Scale for Style {
    fn scale(&mut self, amount: f32) {
        if let Some(ov_font_id) = &mut self.override_font_id {
            ov_font_id.size *= amount;
        }

        for text_style in self.text_styles.values_mut() {
            text_style.size *= amount;
        }

        self.spacing.item_spacing.scale(amount);
        self.spacing.window_margin.scale(amount);
        self.spacing.button_padding.scale(amount);
        self.spacing.menu_margin *= amount;
        self.spacing.indent *= amount;
        self.spacing.interact_size.scale(amount);
        self.spacing.slider_width *= amount;
        self.spacing.slider_rail_height *= amount;
        self.spacing.combo_width *= amount;
        self.spacing.text_edit_width *= amount;
        self.spacing.icon_width *= amount;
        self.spacing.icon_width_inner *= amount;
        self.spacing.icon_spacing *= amount;
        self.spacing.default_area_size *= amount;
        self.spacing.tooltip_width *= amount;
        self.spacing.menu_width *= amount;
        self.spacing.menu_spacing *= amount;
        self.spacing.combo_height *= amount;

        self.spacing.scroll.bar_width *= amount;
        self.spacing.scroll.floating_allocated_width *= amount;
        self.spacing.scroll.floating_width *= amount;

        self.interaction.resize_grab_radius_side *= amount;
        self.interaction.resize_grab_radius_corner *= amount;

        self.visuals.widgets.noninteractive.scale(amount);
        self.visuals.widgets.inactive.scale(amount);
        self.visuals.widgets.hovered.scale(amount);
        self.visuals.widgets.active.scale(amount);
        self.visuals.widgets.open.scale(amount);

        self.visuals.selection.stroke.scale(amount);

        self.visuals.menu_corner_radius *= amount;
        self.visuals.resize_corner_size *= amount;
        self.visuals.text_cursor.stroke.width *= amount;
        self.visuals.clip_rect_margin *= amount;
        self.visuals.window_corner_radius.scale(amount);
        self.visuals.window_shadow.scale(amount);
        self.visuals.popup_shadow.scale(amount);
    }
}
