use std::{io, time::Duration};

use anyhow::Context as _;
use crossterm::{
    event::{self, Event, KeyCode},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::{
    Terminal,
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, ListState, Paragraph, StatefulWidget, Widget, Wrap},
};
use ratatui_image::{StatefulImage, picker::Picker, protocol::StatefulProtocol};

use trql::storage::datasets::SubDataset;

use crate::app::{App, ConvItemStatus, Pane, TrqlStatus, View};
use crate::peek;

/// UI-only state that can't live in `App` (terminal image protocol handles).
struct Ui {
    /// Image protocol picker, queried from the terminal on first use.
    picker: Option<Picker>,
    /// The currently displayed frame preview, if in the Peek view.
    peek: Option<PeekView>,
}

struct PeekView {
    protocol: StatefulProtocol,
    caption: String,
}

impl Ui {
    fn new() -> Self {
        Self {
            picker: None,
            peek: None,
        }
    }

    /// Extract and load a random frame for the selected entry, switching to the
    /// Peek view on success.
    fn peek_selected(&mut self, app: &mut App) {
        let Some(entry) = app.selected_entry() else {
            return;
        };
        let dir = entry.dir().to_path_buf();
        let fps = entry.metadata.as_ref().and_then(|m| m.fps);
        let duration = entry.metadata.as_ref().and_then(|m| m.duration_s);
        let label = entry.rel_path.display().to_string();

        match self.load_frame(&dir, fps, duration, &label) {
            Ok(view) => {
                self.peek = Some(view);
                app.view = View::Peek;
            }
            Err(err) => {
                app.log.push(format!("Peek failed for {label}: {err:#}"));
                self.peek = None;
                app.view = View::Log;
            }
        }
    }

    fn load_frame(
        &mut self,
        dir: &std::path::Path,
        fps: Option<f64>,
        duration: Option<f64>,
        label: &str,
    ) -> anyhow::Result<PeekView> {
        let frame = peek::extract_random_frame(dir, fps, duration)?;
        if self.picker.is_none() {
            self.picker = Some(
                Picker::from_query_stdio()
                    .map_err(|e| anyhow::anyhow!("terminal image query failed: {e}"))?,
            );
        }
        let picker = self.picker.as_mut().expect("picker initialized above");
        let dyn_img = image::ImageReader::open(&frame.image_path)?
            .with_guessed_format()?
            .decode()?;
        let protocol = picker.new_resize_protocol(dyn_img);
        let caption = format!(
            "{label}   frame {}   t={:.1}s / {:.1}s   (p: reroll, q: back)",
            frame
                .frame_index
                .map(|i| i.to_string())
                .unwrap_or_else(|| "?".to_owned()),
            frame.timestamp_s,
            frame.duration_s,
        );
        Ok(PeekView { protocol, caption })
    }

    fn clear_peek(&mut self) {
        self.peek = None;
    }
}

pub fn run(mut app: App) -> anyhow::Result<()> {
    enable_raw_mode().context("failed to enable raw mode")?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let result = run_loop(&mut terminal, &mut app);

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    result
}

fn run_loop(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    app: &mut App,
) -> anyhow::Result<()> {
    let mut ui = Ui::new();
    loop {
        terminal.draw(|f| draw(f.area(), f.buffer_mut(), app, &mut ui))?;

        if app.conversion.is_some() {
            // Active conversion: poll so we can refresh progress without
            // blocking on input, then drain any worker updates.
            if event::poll(Duration::from_millis(100))?
                && let Event::Key(key) = event::read()?
                && handle_key(app, &mut ui, key.code)
            {
                break;
            }
            app.poll_conversion();
        } else if let Event::Key(key) = event::read()? {
            if handle_key(app, &mut ui, key.code) {
                break;
            }
        }
    }
    Ok(())
}

/// Handle a key press. Returns `true` when the app should quit.
fn handle_key(app: &mut App, ui: &mut Ui, code: KeyCode) -> bool {
    // The detector-config menu is modal: it captures navigation keys.
    if app.view == View::DetectMenu {
        match code {
            KeyCode::Char('q') | KeyCode::Esc => app.view = View::List,
            KeyCode::Down | KeyCode::Char('j') => app.detect_menu_move(1),
            KeyCode::Up | KeyCode::Char('k') => app.detect_menu_move(-1),
            KeyCode::Enter => app.confirm_detect(),
            _ => {}
        }
        return false;
    }

    match code {
        KeyCode::Char('q') | KeyCode::Esc => {
            if app.view == View::List {
                return true;
            }
            ui.clear_peek();
            app.view = View::List;
        }
        KeyCode::Char('?') => {
            app.view = if app.view == View::Help {
                View::List
            } else {
                View::Help
            };
        }
        KeyCode::Char('p') => ui.peek_selected(app),
        KeyCode::Down | KeyCode::Char('j') => app.select_next(),
        KeyCode::Up | KeyCode::Char('k') => app.select_prev(),
        KeyCode::Left | KeyCode::Char('h') => app.focus_dataset(),
        KeyCode::Right | KeyCode::Char('l') => app.focus_subdataset(),
        KeyCode::Enter => app.view = View::Detail,
        KeyCode::Char('s') => app.open_schema(),
        KeyCode::Char('L') => app.view = View::Log,
        KeyCode::Char('r') => {
            if let Err(err) = app.refresh() {
                app.log.push(format!("Refresh failed: {err:#}"));
            }
        }
        KeyCode::Char('c') => app.convert_selected(),
        KeyCode::Char(' ') => app.toggle_mark(),
        KeyCode::Char('a') => app.toggle_mark_group(),
        KeyCode::Char('x') => app.clear_marks(),
        KeyCode::Char('C') => app.convert_marked(),
        KeyCode::Char('m') => app.merge_marked(),
        KeyCode::Char('d') => app.open_detect_menu(),
        KeyCode::Char('[') => app.adjust_max_gap(-1),
        KeyCode::Char(']') => app.adjust_max_gap(1),
        KeyCode::Char('-') | KeyCode::Char('_') => app.adjust_max_error(-1.0),
        KeyCode::Char('+') | KeyCode::Char('=') => app.adjust_max_error(1.0),
        _ => {}
    }
    false
}

fn draw(area: Rect, buf: &mut ratatui::buffer::Buffer, app: &App, ui: &mut Ui) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(3), Constraint::Length(3)])
        .split(area);
    let main = chunks[0];
    let help = chunks[1];

    match app.view {
        View::List => draw_list(main, buf, app),
        View::Detail => draw_detail(main, buf, app),
        View::Schema => draw_schema(main, buf, app),
        View::Log => draw_log(main, buf, app),
        View::Peek => draw_peek(main, buf, ui),
        View::Help => draw_help_view(main, buf),
        View::DetectMenu => draw_detect_menu(main, buf, app),
    }
    draw_status(help, buf, app);
}

fn draw_peek(area: Rect, buf: &mut ratatui::buffer::Buffer, ui: &mut Ui) {
    let block = Block::default()
        .title("Peek")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan));
    let inner = block.inner(area);
    block.render(area, buf);

    let Some(view) = ui.peek.as_mut() else {
        Paragraph::new(Line::from(Span::styled(
            "No frame loaded",
            Style::default().fg(Color::DarkGray),
        )))
        .render(inner, buf);
        return;
    };

    // Image fills the area above a one-line caption.
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(1), Constraint::Length(1)])
        .split(inner);
    StatefulWidget::render(
        StatefulImage::<StatefulProtocol>::default(),
        rows[0],
        buf,
        &mut view.protocol,
    );
    Paragraph::new(Line::from(Span::styled(
        view.caption.clone(),
        Style::default().fg(Color::Gray),
    )))
    .render(rows[1], buf);
}

fn draw_list(area: Rect, buf: &mut ratatui::buffer::Buffer, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(20),
            Constraint::Percentage(38),
            Constraint::Percentage(42),
        ])
        .split(area);

    draw_dataset_pane(chunks[0], buf, app);
    draw_subdataset_pane(chunks[1], buf, app);
    draw_entry_summary(chunks[2], buf, app, app.selected_entry());
}

/// Border style for a pane that does or does not have keyboard focus.
fn pane_border(focused: bool) -> Style {
    if focused {
        Style::default().fg(Color::Cyan)
    } else {
        Style::default().fg(Color::DarkGray)
    }
}

/// Highlight style for the selected row; brighter when its pane is focused.
fn selection_style(focused: bool) -> Style {
    if focused {
        Style::default().fg(Color::Black).bg(Color::Cyan)
    } else {
        Style::default().add_modifier(Modifier::REVERSED | Modifier::DIM)
    }
}

fn draw_dataset_pane(area: Rect, buf: &mut ratatui::buffer::Buffer, app: &App) {
    let focused = app.focus == Pane::Dataset;
    let items: Vec<ListItem> = app
        .datasets
        .datasets()
        .iter()
        .map(|group| {
            ListItem::new(Line::from(format!(
                "{} ({})",
                group.name,
                group.entries.len()
            )))
        })
        .collect();

    let mut state = ListState::default();
    if !app.datasets.datasets().is_empty() {
        state.select(Some(app.selected_group));
    }
    StatefulWidget::render(
        List::new(items)
            .block(
                Block::default()
                    .title("Dataset")
                    .borders(Borders::ALL)
                    .border_style(pane_border(focused)),
            )
            .highlight_style(selection_style(focused)),
        area,
        buf,
        &mut state,
    );
}

fn draw_subdataset_pane(area: Rect, buf: &mut ratatui::buffer::Buffer, app: &App) {
    let focused = app.focus == Pane::Subdataset;
    let group_entries = app.current_group_entries();
    let items: Vec<ListItem> = group_entries
        .iter()
        .map(|&idx| {
            let entry = &app.entries()[idx];
            // During a conversion, the first column shows live job status;
            // otherwise it's the mark checkbox.
            let lead = match app.item_status(entry) {
                Some(status) => conv_status_span(status),
                None => {
                    let marked = app.is_marked(entry);
                    Span::styled(
                        if marked { "[x] " } else { "[ ] " },
                        if marked {
                            Style::default()
                                .fg(Color::Green)
                                .add_modifier(Modifier::BOLD)
                        } else {
                            Style::default().fg(Color::DarkGray)
                        },
                    )
                }
            };
            ListItem::new(Line::from(vec![
                lead,
                Span::styled(
                    status_icon(&app.trql_status(entry)),
                    status_style(&app.trql_status(entry)),
                ),
                Span::raw(" "),
                Span::raw(entry.name.clone()),
            ]))
        })
        .collect();

    let marked = app.marked_count();
    let title = if marked > 0 {
        format!("Sub-Dataset ({marked} marked)")
    } else {
        "Sub-Dataset".to_owned()
    };

    let mut state = ListState::default();
    if !group_entries.is_empty() {
        state.select(Some(app.selected_sub));
    }
    StatefulWidget::render(
        List::new(items)
            .block(
                Block::default()
                    .title(title)
                    .borders(Borders::ALL)
                    .border_style(pane_border(focused)),
            )
            .highlight_style(selection_style(focused)),
        area,
        buf,
        &mut state,
    );
}

fn draw_detail(area: Rect, buf: &mut ratatui::buffer::Buffer, app: &App) {
    let Some(entry) = app.selected_entry() else {
        return;
    };
    let mut lines = base_lines(app, entry);
    if let Some(meta) = &entry.metadata {
        lines.push(Line::raw(""));
        lines.push(Line::raw(format!(
            "source: {}",
            meta.source.as_deref().unwrap_or("-")
        )));
        lines.push(Line::raw(format!(
            "scene: {}",
            meta.scene.as_deref().unwrap_or("-")
        )));
        lines.push(Line::raw(format!(
            "clip: {}",
            meta.clip.as_deref().unwrap_or("-")
        )));
        lines.push(Line::raw(format!(
            "video_id: {}",
            meta.video_id.as_deref().unwrap_or("-")
        )));
        lines.push(Line::raw(format!(
            "fps: {}  size: {}x{}  frames: {}  duration: {}s",
            opt_f64(meta.fps),
            opt_u32(meta.width),
            opt_u32(meta.height),
            opt_u32(meta.num_frames),
            opt_f64(meta.duration_s),
        )));
        if let Some(counts) = &meta.row_counts {
            lines.push(Line::raw(format!(
                "metadata counts: objects={} bbox={}",
                counts
                    .objects
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "-".into()),
                counts
                    .bbox
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "-".into())
            )));
        }
        if !meta.obj_classes.is_empty() {
            lines.push(Line::raw(format!(
                "object classes: {}",
                meta.obj_classes.join(", ")
            )));
        }
        if !meta.activity_classes.is_empty() {
            lines.push(Line::raw(format!(
                "activity classes: {}",
                meta.activity_classes.join(", ")
            )));
        }
        if let Some(notes) = &meta.notes {
            lines.push(Line::raw(format!("notes: {notes}")));
        }
    }
    Paragraph::new(lines)
        .block(
            Block::default()
                .title("Dataset Detail")
                .borders(Borders::ALL),
        )
        .wrap(Wrap { trim: false })
        .render(area, buf);
}

fn draw_schema(area: Rect, buf: &mut ratatui::buffer::Buffer, app: &App) {
    let Some(entry) = app.selected_entry() else {
        return;
    };
    let app_trql = app.trql_name(entry);
    let mut lines = Vec::new();
    if let Some(db) = &entry.duckdb {
        lines.push(Line::raw("objects columns"));
        for col in &db.object_columns {
            lines.push(Line::raw(format!(
                "  {}: {}{}",
                col.name,
                col.ty,
                if col.nullable { " nullable" } else { "" }
            )));
        }
        lines.push(Line::raw(""));
        lines.push(Line::raw("bbox columns"));
        for col in &db.bbox_columns {
            lines.push(Line::raw(format!(
                "  {}: {}{}",
                col.name,
                col.ty,
                if col.nullable { " nullable" } else { "" }
            )));
        }
        lines.push(Line::raw(""));
        lines.push(Line::raw("inferred TrQL schema"));
        for field in &db.schema_fields {
            lines.push(Line::raw(format!(
                "  {} <- {}: {}",
                field.name, field.column, field.type_desc
            )));
        }
    } else {
        lines.push(Line::raw("DuckDB metadata unavailable"));
        if let Some(err) = &entry.error {
            lines.push(Line::raw(err));
        }
    }

    lines.push(Line::raw(""));
    lines.push(section("TRQL stats"));
    match &entry.trql_stats {
        Some(s) => {
            lines.push(kv("file", app_trql.to_owned()));
            lines.push(kv("objects", group_digits(s.num_objs as u64)));
            lines.push(kv(
                "avg len",
                format!("{:.1} frames covered", s.avg_traj_len),
            ));
            lines.push(kv(
                "avg span",
                format!("{:.1} frames (incl. gaps)", s.avg_span),
            ));
            lines.push(kv(
                "pieces",
                format!("{} total ({:.2}/obj)", s.total_segments, s.avg_segments),
            ));
            let mut seg = format!("{} poly · {} direct", s.poly_segments, s.direct_segments);
            if s.mixed_segments > 0 {
                seg.push_str(&format!(" · {} mixed", s.mixed_segments));
            }
            lines.push(kv("segments", seg));
            lines.push(kv("canonical", group_digits(s.total_canonical)));
        }
        None if app.trql_status(entry) != TrqlStatus::Missing => {
            lines.push(Line::from(Span::styled(
                "  (loading failed — see log)",
                Style::default().fg(Color::Red),
            )))
        }
        None => lines.push(Line::from(Span::styled(
            format!("  no {app_trql} built (press c)"),
            Style::default()
                .fg(Color::DarkGray)
                .add_modifier(Modifier::ITALIC),
        ))),
    }

    Paragraph::new(lines)
        .block(Block::default().title("Schema").borders(Borders::ALL))
        .wrap(Wrap { trim: false })
        .render(area, buf);
}

/// A small status glyph for a per-item conversion state.
fn conv_status_span(status: ConvItemStatus) -> Span<'static> {
    let (glyph, color) = match status {
        ConvItemStatus::Pending => ("\u{00b7}   ", Color::DarkGray), // ·
        ConvItemStatus::Running => ("\u{2026}   ", Color::Cyan),     // …
        ConvItemStatus::Done => ("\u{2713}   ", Color::Green),       // ✓
        ConvItemStatus::Failed => ("\u{2717}   ", Color::Red),       // ✗
    };
    Span::styled(
        glyph,
        Style::default().fg(color).add_modifier(Modifier::BOLD),
    )
}

/// A textual progress bar like `[####----] 4/8`.
fn progress_bar(done: usize, total: usize, width: usize) -> Line<'static> {
    let width = width.max(1);
    let filled = if total == 0 {
        0
    } else {
        (done * width) / total
    };
    let bar: String = std::iter::repeat_n('#', filled)
        .chain(std::iter::repeat_n('-', width - filled))
        .collect();
    Line::from(vec![
        Span::raw("["),
        Span::styled(bar, Style::default().fg(Color::Cyan)),
        Span::raw(format!("] {done}/{total}")),
    ])
}

fn draw_log(area: Rect, buf: &mut ratatui::buffer::Buffer, app: &App) {
    let mut lines: Vec<Line> = Vec::new();

    if let Some(conv) = &app.conversion {
        lines.push(Line::from(Span::styled(
            format!("{} {}/{}", conv.kind.verb(), conv.completed, conv.total),
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )));
        let bar_width = (area.width.saturating_sub(4) as usize).min(40);
        lines.push(progress_bar(conv.completed, conv.total, bar_width));
        lines.push(Line::from(vec![
            Span::styled(
                format!("ok {}", conv.succeeded),
                Style::default().fg(Color::Green),
            ),
            Span::raw("   "),
            Span::styled(
                format!("failed {}", conv.failed),
                Style::default().fg(if conv.failed > 0 {
                    Color::Red
                } else {
                    Color::DarkGray
                }),
            ),
        ]));
        if let Some(current) = &conv.current {
            lines.push(Line::from(vec![
                Span::styled("current: ", Style::default().fg(Color::DarkGray)),
                Span::raw(current.clone()),
            ]));
        }
        lines.push(Line::raw(""));
    }

    let budget = (area.height.saturating_sub(2) as usize).saturating_sub(lines.len());
    lines.extend(
        app.log
            .iter()
            .rev()
            .take(budget)
            .rev()
            .map(|line| Line::raw(line.clone())),
    );

    let title = match &app.conversion {
        Some(conv) => match conv.kind {
            crate::app::JobKind::Convert => "Log (converting…)",
            crate::app::JobKind::Merge => "Log (merging…)",
            crate::app::JobKind::Detect => "Log (detecting…)",
        },
        None => "Log",
    };
    Paragraph::new(lines)
        .block(Block::default().title(title).borders(Borders::ALL))
        .wrap(Wrap { trim: false })
        .render(area, buf);
}

fn draw_entry_summary(
    area: Rect,
    buf: &mut ratatui::buffer::Buffer,
    app: &App,
    entry: Option<&SubDataset>,
) {
    let block = Block::default().title("Summary").borders(Borders::ALL);
    let lines = match entry {
        Some(entry) => summary_lines(app, entry),
        None => vec![Line::from(Span::styled(
            "No dataset selected",
            Style::default()
                .fg(Color::DarkGray)
                .add_modifier(Modifier::ITALIC),
        ))],
    };
    Paragraph::new(lines)
        .block(block)
        .wrap(Wrap { trim: false })
        .render(area, buf);
}

/// Width of the label column in summary key/value rows.
const LABEL_WIDTH: usize = 9;

/// A `label   value` row with a dimmed label.
fn kv(label: &str, value: impl Into<String>) -> Line<'static> {
    Line::from(vec![
        Span::styled(
            format!("{label:<LABEL_WIDTH$}"),
            Style::default().fg(Color::DarkGray),
        ),
        Span::raw(value.into()),
    ])
}

/// A bold section header.
fn section(title: &'static str) -> Line<'static> {
    Line::from(Span::styled(
        title,
        Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD),
    ))
}

/// A colored badge for the .trql build status.
fn status_badge(status: &TrqlStatus) -> Span<'static> {
    let (text, color) = match status {
        TrqlStatus::Missing => (" missing ", Color::Red),
        TrqlStatus::Generated => (" generated ", Color::Green),
        TrqlStatus::Stale => (" stale ", Color::Yellow),
    };
    Span::styled(
        text,
        Style::default()
            .fg(Color::Black)
            .bg(color)
            .add_modifier(Modifier::BOLD),
    )
}

/// Format an integer with thousands separators (e.g. `1,234,567`).
fn group_digits(value: u64) -> String {
    let digits = value.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    let bytes = digits.as_bytes();
    for (i, b) in bytes.iter().enumerate() {
        if i > 0 && (bytes.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(*b as char);
    }
    out
}

/// Read the config label from a predicted dataset's `metadata.json` (the
/// `notes` field is `"Predicted by <config> (<id>)"`).
fn read_prediction_config(pred_dir: &std::path::Path) -> Option<String> {
    let raw = std::fs::read_to_string(pred_dir.join("metadata.json")).ok()?;
    let value: serde_json::Value = serde_json::from_str(&raw).ok()?;
    let notes = value.get("notes")?.as_str()?;
    Some(
        notes
            .strip_prefix("Predicted by ")
            .unwrap_or(notes)
            .to_owned(),
    )
}

/// Every prediction under `pred_parent` (one subdir per config that contains a
/// `duckdb.db`) as `(label, dir)`, sorted by label for stable display.
fn prediction_configs(pred_parent: &std::path::Path) -> Vec<(String, std::path::PathBuf)> {
    let Ok(read_dir) = std::fs::read_dir(pred_parent) else {
        return Vec::new();
    };
    let mut configs: Vec<(String, std::path::PathBuf)> = read_dir
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.join("duckdb.db").is_file())
        .map(|p| {
            let label = read_prediction_config(&p).unwrap_or_else(|| {
                p.file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default()
            });
            (label, p)
        })
        .collect();
    configs.sort_by(|a, b| a.0.cmp(&b.0));
    configs
}

/// Read `(f1, precision, recall)` from a prediction's `metrics.json`, if it has
/// been evaluated.
fn read_metrics(dir: &std::path::Path) -> Option<(f64, f64, f64)> {
    let raw = std::fs::read_to_string(dir.join("metrics.json")).ok()?;
    let v: serde_json::Value = serde_json::from_str(&raw).ok()?;
    Some((
        v.get("f1")?.as_f64()?,
        v.get("precision")?.as_f64()?,
        v.get("recall")?.as_f64()?,
    ))
}

/// Read `(matched, pred_objects)` from a prediction's `alignment.json`.
fn read_alignment(dir: &std::path::Path) -> Option<(u64, u64)> {
    let raw = std::fs::read_to_string(dir.join("alignment.json")).ok()?;
    let v: serde_json::Value = serde_json::from_str(&raw).ok()?;
    Some((
        v.get("matched")?.as_u64()?,
        v.get("pred_objects")?.as_u64()?,
    ))
}

fn summary_lines(app: &App, entry: &SubDataset) -> Vec<Line<'static>> {
    let current_trql = app.trql_name(entry);
    let data_dir = app.data_dir.as_path();
    let mut lines = Vec::new();

    // Header: subdataset name + dataset family.
    lines.push(Line::from(Span::styled(
        entry.name.clone(),
        Style::default()
            .fg(Color::White)
            .add_modifier(Modifier::BOLD),
    )));
    lines.push(Line::from(vec![
        Span::styled(
            format!("{:<LABEL_WIDTH$}", "trql"),
            Style::default().fg(Color::DarkGray),
        ),
        status_badge(&app.trql_status(entry)),
    ]));
    lines.push(Line::raw(""));

    // Stats: prefer live DuckDB metadata, fall back to recorded counts.
    if let Some(db) = &entry.duckdb {
        lines.push(section("Stats"));
        lines.push(kv("objects", group_digits(db.object_count)));
        lines.push(kv("bbox", group_digits(db.bbox_count)));
        lines.push(kv(
            "frames",
            match (db.frame_min, db.frame_max) {
                (Some(a), Some(b)) => format!("{a} .. {b}"),
                _ => "-".to_owned(),
            },
        ));
        lines.push(kv("schema", format!("{} fields", db.schema_fields.len())));
    } else if let Some(counts) = entry.metadata.as_ref().and_then(|m| m.row_counts.as_ref()) {
        lines.push(section("Stats (recorded)"));
        lines.push(kv(
            "objects",
            counts
                .objects
                .map(group_digits)
                .unwrap_or_else(|| "-".into()),
        ));
        lines.push(kv(
            "bbox",
            counts.bbox.map(group_digits).unwrap_or_else(|| "-".into()),
        ));
    }

    // Video metadata.
    if let Some(meta) = &entry.metadata {
        let has_video = meta.fps.is_some()
            || meta.width.is_some()
            || meta.num_frames.is_some()
            || meta.duration_s.is_some();
        if has_video {
            lines.push(Line::raw(""));
            lines.push(section("Video"));
            if meta.width.is_some() || meta.height.is_some() {
                lines.push(kv(
                    "size",
                    format!("{} x {}", opt_u32(meta.width), opt_u32(meta.height)),
                ));
            }
            if meta.fps.is_some() {
                lines.push(kv("fps", opt_f64(meta.fps)));
            }
            if meta.num_frames.is_some() {
                lines.push(kv("frames", opt_u32(meta.num_frames)));
            }
            if meta.duration_s.is_some() {
                lines.push(kv("duration", format!("{}s", opt_f64(meta.duration_s))));
            }
        }
    }

    // Generated .trql artifacts in this entry's directory.
    lines.push(Line::raw(""));
    lines.push(section("Generated trql"));
    let artifacts = &entry.trql_files;
    if artifacts.is_empty() {
        lines.push(Line::from(Span::styled(
            "  (none yet)",
            Style::default()
                .fg(Color::DarkGray)
                .add_modifier(Modifier::ITALIC),
        )));
    } else {
        for art in artifacts {
            let name = art.file_name();
            let is_current = name == current_trql;
            let marker = if is_current { "> " } else { "  " };
            let mut name_style = Style::default().fg(if art.stale {
                Color::Yellow
            } else {
                Color::Green
            });
            if is_current {
                name_style = name_style.add_modifier(Modifier::BOLD);
            }
            let mut spans = vec![
                Span::styled(marker, Style::default().fg(Color::Cyan)),
                Span::styled(name.clone(), name_style),
            ];
            if art.stale {
                spans.push(Span::styled(
                    "  (stale)",
                    Style::default().fg(Color::Yellow),
                ));
            }
            lines.push(Line::from(spans));
        }
    }

    // Prediction: provenance for a predicted dataset, or whether one exists for
    // a ground-truth dataset (output of the `d` detect action).
    lines.push(Line::raw(""));
    lines.push(section("Prediction"));
    if entry.rel_path.starts_with("predicted") {
        // This entry IS a prediction; show its source and the config used.
        if let Some(meta) = &entry.metadata {
            if let Some(src) = meta.source.as_deref().filter(|s| !s.is_empty()) {
                lines.push(kv("of", src.to_owned()));
            }
            if let Some(notes) = &meta.notes {
                let cfg = notes.strip_prefix("Predicted by ").unwrap_or(notes);
                lines.push(kv("config", cfg.to_owned()));
            }
        }
        match read_metrics(entry.dir()) {
            Some((f1, p, r)) => {
                lines.push(Line::from(vec![
                    Span::styled(
                        format!("{:<LABEL_WIDTH$}", "F1"),
                        Style::default().fg(Color::DarkGray),
                    ),
                    Span::styled(
                        format!("{f1:.3}"),
                        Style::default()
                            .fg(Color::Green)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(
                        format!("   (P {p:.3}  R {r:.3}  @IoU .5)"),
                        Style::default().fg(Color::DarkGray),
                    ),
                ]));
            }
            None => lines.push(Line::from(Span::styled(
                "  (no metrics)",
                Style::default()
                    .fg(Color::DarkGray)
                    .add_modifier(Modifier::ITALIC),
            ))),
        }
        if let Some((matched, total)) = read_alignment(entry.dir()) {
            lines.push(kv(
                "ids",
                format!("{matched}/{total} matched to GT (continuous)"),
            ));
        }
    } else {
        // Each config produced a `predicted/<rel_path>/<config>/` subdataset.
        let pred_parent = data_dir.join("predicted").join(&entry.rel_path);
        let configs = prediction_configs(&pred_parent);
        if configs.is_empty() {
            lines.push(Line::from(Span::styled(
                "  (none — press d to detect)",
                Style::default()
                    .fg(Color::DarkGray)
                    .add_modifier(Modifier::ITALIC),
            )));
        } else {
            lines.push(Line::from(vec![
                Span::styled(
                    format!("{:<LABEL_WIDTH$}", "labels"),
                    Style::default().fg(Color::DarkGray),
                ),
                Span::styled(
                    format!(" {} config(s) ", configs.len()),
                    Style::default()
                        .fg(Color::Black)
                        .bg(Color::Green)
                        .add_modifier(Modifier::BOLD),
                ),
            ]));
            for (cfg, dir) in configs {
                let mut spans = vec![
                    Span::styled("  • ", Style::default().fg(Color::Cyan)),
                    Span::styled(cfg, Style::default().fg(Color::Green)),
                ];
                if let Some((f1, _, _)) = read_metrics(&dir) {
                    spans.push(Span::styled(
                        format!("  F1 {f1:.3}"),
                        Style::default().fg(Color::Gray),
                    ));
                }
                lines.push(Line::from(spans));
            }
        }
    }

    // Paths.
    lines.push(Line::raw(""));
    lines.push(section("Paths"));
    lines.push(kv("dir", entry.rel_path.display().to_string()));
    lines.push(kv("trql", app.trql_path(entry).display().to_string()));

    if let Some(err) = &entry.error {
        lines.push(Line::raw(""));
        lines.push(Line::from(Span::styled(
            format!("error: {err}"),
            Style::default().fg(Color::Red),
        )));
    }

    lines
}

fn base_lines(app: &App, entry: &SubDataset) -> Vec<Line<'static>> {
    let mut lines = vec![
        Line::raw(format!("path: {}", entry.rel_path.display())),
        Line::raw(format!("duckdb: {}", entry.duckdb_path.display())),
        Line::raw(format!("trql: {}", app.trql_path(entry).display())),
        Line::raw(format!("trql status: {}", app.trql_status(entry).label())),
    ];
    if let Some(path) = &entry.metadata_path {
        lines.push(Line::raw(format!("metadata: {}", path.display())));
    }
    if let Some(db) = &entry.duckdb {
        lines.push(Line::raw(format!(
            "objects: {}  bbox rows: {}  frames: {}..{}",
            db.object_count,
            db.bbox_count,
            db.frame_min
                .map(|v| v.to_string())
                .unwrap_or_else(|| "-".into()),
            db.frame_max
                .map(|v| v.to_string())
                .unwrap_or_else(|| "-".into())
        )));
        lines.push(Line::raw(format!(
            "inferred schema fields: {}",
            db.schema_fields.len()
        )));
    } else if let Some(meta) = &entry.metadata
        && let Some(counts) = &meta.row_counts
    {
        lines.push(Line::raw(format!(
            "metadata counts: objects={} bbox={}",
            counts
                .objects
                .map(|v| v.to_string())
                .unwrap_or_else(|| "-".into()),
            counts
                .bbox
                .map(|v| v.to_string())
                .unwrap_or_else(|| "-".into())
        )));
    }
    if let Some(err) = &entry.error {
        lines.push(Line::raw(format!("error: {err}")));
    }
    lines
}

/// The slim bottom bar: live status only, plus a hint to open the help view.
fn draw_status(area: Rect, buf: &mut ratatui::buffer::Buffer, app: &App) {
    let mut spans = vec![
        Span::styled(
            "TrQL TUI",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw("  "),
        Span::styled(
            format!("gap {}  err {}", app.max_gap, app.max_error_px),
            Style::default().fg(Color::Gray),
        ),
    ];

    let marked = app.marked_count();
    if marked > 0 {
        spans.push(Span::raw("  "));
        spans.push(Span::styled(
            format!("{marked} marked"),
            Style::default()
                .fg(Color::Green)
                .add_modifier(Modifier::BOLD),
        ));
    }

    if let Some(conv) = &app.conversion {
        spans.push(Span::raw("  "));
        spans.push(Span::styled(
            format!("{} {}/{}", conv.kind.verb(), conv.completed, conv.total),
            Style::default().fg(Color::Yellow),
        ));
    }

    // Right-aligned help hint.
    let hint = "? help";
    let used: usize = spans.iter().map(|s| s.content.chars().count()).sum();
    let pad = (area.width as usize)
        .saturating_sub(2) // borders
        .saturating_sub(used)
        .saturating_sub(hint.len());
    spans.push(Span::raw(" ".repeat(pad)));
    spans.push(Span::styled(hint, Style::default().fg(Color::DarkGray)));

    Paragraph::new(Line::from(spans))
        .block(Block::default().borders(Borders::ALL))
        .render(area, buf);
}

/// Full-screen list of every key binding, opened with `?`.
fn draw_help_view(area: Rect, buf: &mut ratatui::buffer::Buffer) {
    // (key, description), grouped by blank-line section headers.
    let rows: &[(&str, &str)] = &[
        ("Navigation", ""),
        ("j / k  ↓ / ↑", "move selection"),
        ("h / l  ← / →", "switch dataset / sub-dataset pane"),
        ("Enter", "open dataset detail"),
        ("s", "show inferred schema"),
        ("L", "show log"),
        ("p", "peek a random video frame"),
        ("", ""),
        ("Marks & actions", ""),
        ("space", "toggle mark on selection"),
        ("a", "toggle mark on whole group"),
        ("x", "clear all marks"),
        ("c", "convert selection to .trql"),
        ("C", "convert all marked"),
        ("m", "merge all marked into one dataset"),
        ("d", "detect+track → predicted dataset (auto-scored vs GT)"),
        ("", ""),
        ("Parameters", ""),
        ("[ / ]", "decrease / increase max gap"),
        ("- / +", "decrease / increase max error (px)"),
        ("r", "refresh / rescan datasets"),
        ("", ""),
        ("General", ""),
        ("?", "toggle this help"),
        ("q / Esc", "back, or quit from the list"),
    ];

    let mut lines: Vec<Line> = Vec::with_capacity(rows.len());
    for (key, desc) in rows {
        if desc.is_empty() {
            if key.is_empty() {
                lines.push(Line::raw(""));
            } else {
                // Section header.
                lines.push(Line::from(Span::styled(
                    *key,
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                )));
            }
        } else {
            lines.push(Line::from(vec![
                Span::styled(
                    format!("  {key:<14}"),
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::raw(*desc),
            ]));
        }
    }

    Paragraph::new(lines)
        .block(
            Block::default()
                .title("Key Bindings")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Cyan)),
        )
        .wrap(Wrap { trim: false })
        .render(area, buf);
}

/// Modal picker for the detector+tracker configuration, opened with `d`.
fn draw_detect_menu(area: Rect, buf: &mut ratatui::buffer::Buffer, app: &App) {
    let targets = app.detect_target_count();
    let mut lines: Vec<Line> = Vec::new();
    lines.push(Line::from(Span::styled(
        format!(
            "Run detect+track on {target} subdataset(s). Output → predicted/. Enter to run, Esc to cancel.",
            target = targets
        ),
        Style::default().fg(Color::Gray),
    )));
    lines.push(Line::raw(""));

    for (i, cfg) in app.detect_configs.iter().enumerate() {
        let selected = i == app.detect_menu_sel;
        let marker = if selected { "❯ " } else { "  " };
        let name_style = if selected {
            Style::default()
                .fg(Color::Black)
                .bg(Color::Cyan)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD)
        };
        lines.push(Line::from(vec![
            Span::styled(marker, Style::default().fg(Color::Cyan)),
            Span::styled(format!(" {} ", cfg.name), name_style),
        ]));
        lines.push(Line::from(vec![
            Span::raw("    "),
            Span::styled(cfg.description.clone(), Style::default().fg(Color::DarkGray)),
        ]));
        lines.push(Line::from(vec![
            Span::raw("    "),
            Span::styled(
                format!(
                    "model {}  ·  tracker {}  ·  imgsz {}  ·  conf {}  ·  {}",
                    cfg.model,
                    cfg.tracker.trim_end_matches(".yaml"),
                    cfg.imgsz,
                    cfg.conf,
                    if cfg.classes.is_empty() {
                        "all classes".to_owned()
                    } else {
                        format!("{} classes", cfg.classes.len())
                    },
                ),
                Style::default().fg(Color::DarkGray),
            ),
        ]));
        lines.push(Line::raw(""));
    }

    Paragraph::new(lines)
        .block(
            Block::default()
                .title("Detect — choose a configuration")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Cyan)),
        )
        .wrap(Wrap { trim: false })
        .render(area, buf);
}

fn status_icon(status: &TrqlStatus) -> &'static str {
    match status {
        TrqlStatus::Missing => "M",
        TrqlStatus::Generated => "G",
        TrqlStatus::Stale => "S",
    }
}

fn status_style(status: &TrqlStatus) -> Style {
    match status {
        TrqlStatus::Missing => Style::default().fg(Color::Red),
        TrqlStatus::Generated => Style::default().fg(Color::Green),
        TrqlStatus::Stale => Style::default().fg(Color::Yellow),
    }
}

fn opt_u32(value: Option<u32>) -> String {
    value
        .map(|v| v.to_string())
        .unwrap_or_else(|| "-".to_owned())
}

fn opt_f64(value: Option<f64>) -> String {
    value
        .map(|v| format!("{v:.2}"))
        .unwrap_or_else(|| "-".to_owned())
}
