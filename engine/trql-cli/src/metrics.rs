//! Runtime/resource measurement for `bench`, mirroring `evaluator/src/utils.rs`
//! so the CLI reports the same metrics as the evaluator.

use std::{
    fs::File,
    io::Seek as _,
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    thread,
    time::Duration,
};

use procfs::process::Process;

/// Size of `path` on disk, in whole megabytes.
pub fn get_file_size_mb(path: &Path) -> anyhow::Result<u32> {
    let mut file = File::open(path)?;
    let size = file.seek(std::io::SeekFrom::End(0))?;
    Ok((size / (1024 * 1024)) as u32)
}

/// This process's resident set size, in kilobytes.
fn current_rss_kb() -> u64 {
    Process::myself()
        .ok()
        .and_then(|p| p.status().ok())
        .and_then(|s| s.vmrss)
        .unwrap_or(0)
}

/// Run `f`, sampling this process's peak RSS (in KB) on a background thread for
/// its whole duration. Returns `f`'s result alongside the peak.
///
/// Process-wide, so the peak includes the loaded database already resident in
/// memory — the same measurement the evaluator records.
pub fn measure_peak_rss_kb<T, F: FnOnce() -> T>(f: F) -> (T, u64) {
    let (result, peak, _) = measure_rss_kb(f);
    (result, peak)
}

/// Like [`measure_peak_rss_kb`], but also reports the RSS as `f` was entered.
///
/// The difference is what `f` added to the process. That is the only honest
/// figure for work sharing a process with unrelated resident data — DuckDB runs
/// in-process here, after the engine's `.trql` is already loaded, so its
/// *absolute* RSS would be dominated by an artifact it never touches (11.7 GB on
/// i24). The growth is attributable to it; the total is not.
///
/// It can still under-report: memory the allocator has freed but not returned to
/// the kernel is reused without RSS moving. So treat it as a lower bound.
pub fn measure_rss_kb<T, F: FnOnce() -> T>(f: F) -> (T, u64, u64) {
    let baseline = current_rss_kb();
    let peak = Arc::new(AtomicU64::new(baseline));
    let done = Arc::new(AtomicBool::new(false));
    let peak2 = Arc::clone(&peak);
    let done2 = Arc::clone(&done);

    let sampler = thread::spawn(move || {
        while !done2.load(Ordering::Relaxed) {
            peak2.fetch_max(current_rss_kb(), Ordering::Relaxed);
            thread::sleep(Duration::from_millis(5));
        }
    });

    let result = f();
    done.store(true, Ordering::Relaxed);
    let _ = sampler.join();
    (result, peak.load(Ordering::Relaxed), baseline)
}
