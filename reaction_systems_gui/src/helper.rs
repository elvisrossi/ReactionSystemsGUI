use std::fmt::Display;

use eframe::egui::{self, Color32, TextFormat, TextStyle};
use egui::text::LayoutJob;
use grammar_separated::user_error::{UserError, UserErrorTypes};
use lalrpop_util::ParseError;

fn create_error<T>(
    input_str: &str,
    l: usize,
    t: T,
    r: usize,
    expected: Option<Vec<String>>,
    error: Option<UserErrorTypes>,
    ctx: &eframe::egui::Context,
) -> LayoutJob
where
    T: Display,
{
    let style = ctx.style();
    let monospace_text = TextStyle::Monospace.resolve(&style);
    let monospace = TextFormat {
        font_id: monospace_text.clone(),
        ..Default::default()
    };
    let monospace_red = TextFormat {
        font_id: monospace_text.clone(),
        color: Color32::RED,
        ..Default::default()
    };
    let monospace_blue = TextFormat {
        font_id: monospace_text.clone(),
        color: Color32::BLUE,
        ..Default::default()
    };
    let monospace_green = TextFormat {
        font_id: monospace_text,
        color: Color32::GREEN,
        ..Default::default()
    };

    let mut err = LayoutJob::default();
    if let Some(error) = error {
        err.append(&format!("{error} "), 0., Default::default());
        err.append(&format!("\"{t}\""), 0., monospace_red.clone());
        err.append(
            &format!(" between positions {l} and {r}."),
            0.,
            Default::default(),
        );
    } else {
        err.append("Unrecognized token ", 0., Default::default());
        err.append(&format!("\"{t}\""), 0., monospace_red.clone());
        err.append(
            &format!(" between positions {l} and {r}."),
            0.,
            Default::default(),
        );
    }

    {
        if let Some(expected) = expected {
            // Temporary debug.
            err.append("\nExpected: ", 0., Default::default());
            let mut it = expected.iter().peekable();
            while let Some(s) = it.next() {
                err.append("(", 0., monospace.clone());
                err.append(s, 0., monospace_green.clone());
                err.append(")", 0., monospace.clone());
                if it.peek().is_some() {
                    err.append(", ", 0., monospace.clone());
                }
            }
        }
    }
    let right_new_line = input_str[l..]
        .find("\n")
        .map(|pos| pos + l)
        .unwrap_or(input_str.len());
    let left_new_line = input_str[..r]
        .rfind("\n")
        .map(|pos| pos + 1)
        .unwrap_or_default();

    let line_number = input_str[..l].match_indices('\n').count() + 1;
    let pre = format!("{line_number} |");

    let line_pos_l = l - left_new_line;
    let line_pos_r = r - left_new_line;

    err.append(
        &format!(
            "\nLine {} position {} to {}:\n",
            line_number, line_pos_l, line_pos_r,
        ),
        0.,
        Default::default(),
    );
    err.append(&pre, 0., monospace_blue.clone());
    err.append(&input_str[left_new_line..l], 0., monospace_green);
    err.append(&input_str[l..r], 0., monospace_red.clone());
    err.append(&input_str[r..right_new_line], 0., monospace.clone());

    err.append("\n", 0., monospace.clone());
    err.append(&" ".repeat(pre.len() - 1), 0., monospace.clone());
    err.append("|", 0., monospace_blue);
    err.append(&" ".repeat(l - left_new_line), 0., monospace.clone());
    err.append("^", 0., monospace_red.clone());

    if r - l > 1 {
        err.append(&" ".repeat(r - l - 2), 0., monospace);
        err.append("^", 0., monospace_red);
    }

    err
}

pub fn reformat_error<T>(
    e: ParseError<usize, T, UserError>,
    input_str: &str,
    ctx: &eframe::egui::Context,
) -> LayoutJob
where
    T: Display,
{
    let mut job = LayoutJob::default();
    match e {
        | ParseError::ExtraToken { token: (l, t, r) } => job.append(
            &format!(
                "Unexpected extra token \"{t}\" between positions {l} \
                                and {r}."
            ),
            0.,
            Default::default(),
        ),
        | ParseError::UnrecognizedEof {
            location: _,
            expected: _,
        } => job.append(
            "End of file encountered while parsing.",
            0.,
            Default::default(),
        ),
        | ParseError::InvalidToken { location } => job.append(
            &format!("Invalid token at position {location}."),
            0.,
            Default::default(),
        ),
        | ParseError::UnrecognizedToken {
            token: (l, t, r),
            expected,
        } => job = create_error(input_str, l, t, r, Some(expected), None, ctx),
        | ParseError::User {
            error:
                UserError {
                    token: (l, t, r),
                    error,
                },
        } => job = create_error(input_str, l, t, r, None, Some(error), ctx),
    };

    job
}
