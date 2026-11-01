//! Output layer for the CLI.
//!
//! Every command reports through a [`Ui`] rather than writing directly, so the
//! text and JSON forms of a command can never drift apart: each command builds
//! one serializable report and hands it to [`Ui::emit`], which either renders
//! it for humans or prints it as JSON.
//!
//! Human output goes to stdout; progress bars go to stderr, so
//! `--format json | jq` stays clean. `colored` suppresses styling on its own
//! when stdout is not a terminal or `NO_COLOR` is set.

use std::fmt::Display;

use clap::ValueEnum;
use colored::Colorize as _;
use comfy_table::{ContentArrangement, Table, presets};
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use serde::Serialize;

/// Width of the label column in key/value rows.
const LABEL_WIDTH: usize = 13;

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum Format {
    /// Styled, human-readable output.
    Text,
    /// A single JSON document per command, for scripting.
    Json,
}

pub struct Ui {
    format: Format,
}

impl Ui {
    pub fn new(format: Format) -> Self {
        Self { format }
    }

    pub fn is_json(&self) -> bool {
        self.format == Format::Json
    }

    /// Render `value`: as JSON, or via `human` in text mode.
    ///
    /// Call this once per command, with everything the command has to say.
    pub fn emit<T: Serialize>(
        &self,
        value: &T,
        human: impl FnOnce(&Self, &T),
    ) -> anyhow::Result<()> {
        match self.format {
            Format::Text => human(self, value),
            Format::Json => println!("{}", serde_json::to_string_pretty(value)?),
        }
        Ok(())
    }

    /// A bold section header, preceded by a blank line.
    pub fn section(&self, title: &str) {
        println!("\n{}", title.cyan().bold());
    }

    /// A `label   value` row with a dimmed label.
    pub fn kv(&self, label: &str, value: impl Display) {
        println!("  {} {}", format!("{label:<LABEL_WIDTH$}").dimmed(), value);
    }

    pub fn success(&self, msg: impl Display) {
        println!("{} {msg}", "✓".green().bold());
    }

    /// A no-op outcome: nothing needed doing.
    pub fn skip(&self, msg: impl Display) {
        println!(
            "{} {} {}",
            "•".dimmed(),
            "Skipped".yellow().bold(),
            msg.to_string().dimmed(),
        );
    }

    pub fn warn(&self, msg: impl Display) {
        println!("{} {msg}", "!".yellow().bold());
    }

    /// A bordered table. Columns size themselves to the terminal width.
    pub fn table(&self, headers: &[&str], rows: Vec<Vec<String>>) {
        let mut table = Table::new();
        table
            .load_preset(presets::UTF8_HORIZONTAL_ONLY)
            .set_content_arrangement(ContentArrangement::Dynamic)
            .set_header(headers.iter().map(|h| h.cyan().bold().to_string()));
        for row in rows {
            table.add_row(row);
        }
        println!("{table}");
    }

    /// A spinner on stderr for work with no known total, hidden in JSON mode
    /// and when stderr is not a terminal (so redirected output does not fill
    /// up with control codes).
    ///
    /// Promote it to a real bar with [`into_bar`] once a total is known.
    pub fn spinner(&self, label: &str) -> ProgressBar {
        if self.is_json() {
            return ProgressBar::hidden();
        }
        let bar = ProgressBar::with_draw_target(None, ProgressDrawTarget::stderr());
        bar.set_style(
            ProgressStyle::with_template("{spinner:.green} [{elapsed_precise}] {msg}")
                .expect("valid spinner template"),
        );
        bar.set_message(label.to_owned());
        bar.enable_steady_tick(std::time::Duration::from_millis(120));
        bar
    }
}

/// Re-style a spinner from [`Ui::spinner`] into a determinate progress bar.
pub fn into_bar(bar: &ProgressBar, len: u64, label: &str) {
    bar.disable_steady_tick();
    bar.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg} ({eta})",
        )
        .expect("valid progress template")
        .progress_chars("=>-"),
    );
    bar.set_length(len);
    bar.set_message(label.to_owned());
}

/// Format an integer with thousands separators (e.g. `1,234,567`).
pub fn group_digits(value: u64) -> String {
    let digits = value.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    let bytes = digits.as_bytes();
    for (i, b) in bytes.iter().enumerate() {
        if i > 0 && (bytes.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(*b as char);
    }
    out
}
