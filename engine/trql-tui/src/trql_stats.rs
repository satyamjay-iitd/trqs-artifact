use std::path::Path;

use anyhow::Context as _;
use trql::{db::Trqs, storage::stats::TrqsStats};

/// Read the trajectory/segment statistics stored in a `.trql` header.
///
/// The stats are persisted at build time, so this reads only the file header
/// rather than decoding the whole database.
pub fn compute(trql_path: &Path) -> anyhow::Result<TrqsStats> {
    let stats = Trqs::read_stats(trql_path)
        .with_context(|| format!("reading stats from {}", trql_path.display()))?;

    Ok(TrqsStats {
        num_objs: stats.num_objs,
        avg_traj_len: stats.avg_traj_len,
        avg_span: stats.avg_span,
        total_segments: stats.total_segments,
        avg_segments: stats.avg_segments,
        poly_segments: stats.poly_segments,
        direct_segments: stats.direct_segments,
        mixed_segments: stats.mixed_segments,
        total_canonical: stats.total_canonical,
    })
}
