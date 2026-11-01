mod app;
// `merge` lives in `trql-storage` so `trql-cli` can drive the same code.
mod peek;
mod trql_stats;
mod tui;

use std::path::PathBuf;

use anyhow::Context as _;
use clap::Parser;

#[derive(Parser)]
#[command(about = "Explore processed TrQL datasets and build .trql files")]
struct Args {
    /// Directory containing processed dataset leaves with duckdb.db files.
    #[arg(long, default_value = "datasets/processed")]
    data_dir: PathBuf,

    /// Max frame gap before splitting a trajectory into separate polynomial segments.
    #[arg(long, default_value_t = 10)]
    max_gap: u32,

    /// Max allowed polynomial fitting error in pixels.
    #[arg(long, default_value_t = 0.0)]
    max_error_px: f64,

    /// Python interpreter used to run the detect+track pipeline.
    #[arg(long, default_value = "python3")]
    python: String,

    /// Path to the detect+track pipeline script.
    #[arg(long, default_value = "scripts/detect_track.py")]
    detect_script: PathBuf,

    /// Print discovered datasets and exit instead of launching the TUI.
    #[arg(long)]
    list: bool,

    /// Convert the dataset at this discovered list index and exit.
    #[arg(long)]
    convert_index: Option<usize>,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let mut app = app::App::load(
        args.data_dir.clone(),
        args.max_gap,
        args.max_error_px,
        args.python.clone(),
        args.detect_script.clone(),
    )
    .with_context(|| format!("failed to scan {}", args.data_dir.display()))?;
    if let Some(index) = args.convert_index {
        if index >= app.entries().len() {
            anyhow::bail!(
                "convert index {index} is out of range for {} entries",
                app.entries().len()
            );
        }
        app.selected = index;
        app.convert_selected();
        // Conversion runs on a worker thread; drain progress until it finishes.
        while app.conversion.is_some() {
            app.poll_conversion();
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        for line in &app.log {
            println!("{line}");
        }
        Ok(())
    } else if args.list {
        for (idx, entry) in app.entries().iter().enumerate() {
            let (objects, bboxes) = entry
                .duckdb
                .as_ref()
                .map(|db| (db.object_count, db.bbox_count))
                .or_else(|| {
                    entry.metadata.as_ref().and_then(|meta| {
                        let counts = meta.row_counts.as_ref()?;
                        Some((counts.objects.unwrap_or(0), counts.bbox.unwrap_or(0)))
                    })
                })
                .unwrap_or((0, 0));
            println!(
                "{idx:4}  {:9}  objects={objects:<6} bbox={bboxes:<8} {}",
                app.trql_status(entry).label(),
                entry.rel_path.display()
            );
        }
        Ok(())
    } else {
        tui::run(app)
    }
}
