use bincode::{Decode, Encode};
use serde::Serialize;
use trql_poly::{PolyTraj, ScalarSegmentData2};

/// Aggregate statistics about a built `.trql`'s polynomial trajectories.
#[derive(Clone, Debug, Encode, Decode, Serialize)]
pub struct TrqsStats {
    pub num_objs: usize,
    /// Covered frames per object (sum of segment lengths / objects).
    pub avg_traj_len: f64,
    /// First-to-last frame span per object (includes gaps).
    pub avg_span: f64,
    /// Total polynomial pieces (segments) across all trajectories.
    pub total_segments: u64,
    /// Segments per object.
    pub avg_segments: f64,
    /// Segments whose four channels are all fitted polynomials.
    pub poly_segments: u64,
    /// Segments whose four channels all store raw sampled values.
    pub direct_segments: u64,
    /// Segments mixing polynomial and direct channels.
    pub mixed_segments: u64,
    /// Total canonical (analysis) sub-segments across all trajectories.
    pub total_canonical: u64,
}

impl TrqsStats {
    pub fn empty() -> TrqsStats {
        TrqsStats {
            num_objs: 0,
            avg_traj_len: 0.0,
            avg_span: 0.0,
            total_segments: 0,
            avg_segments: 0.0,
            poly_segments: 0,
            direct_segments: 0,
            mixed_segments: 0,
            total_canonical: 0,
        }
    }

    pub fn from_trajs(trajs: &[PolyTraj]) -> TrqsStats {
        let num_objs = trajs.len();

        let mut covered_frames = 0u64;
        let mut total_span = 0u64;
        let mut total_segments = 0u64;
        let mut poly_segments = 0u64;
        let mut direct_segments = 0u64;
        let mut mixed_segments = 0u64;
        let mut total_canonical = 0u64;

        for tr in trajs {
            let segs = tr.segments();
            total_segments += segs.len() as u64;
            total_canonical += tr.canonical_segments().len() as u64;
            if let (Some(first), Some(last)) = (segs.first(), segs.last()) {
                total_span += u64::from(last.end - first.start + 1);
            }
            for s in segs {
                covered_frames += u64::from(s.end - s.start + 1);
                let n_poly = [&s.cx1, &s.cy1, &s.cx2, &s.cy2]
                    .into_iter()
                    .filter(|c| matches!(c, ScalarSegmentData2::Poly(_)))
                    .count();
                match n_poly {
                    4 => poly_segments += 1,
                    0 => direct_segments += 1,
                    _ => mixed_segments += 1,
                }
            }
        }

        let denom = num_objs.max(1) as f64;
        TrqsStats {
            num_objs,
            avg_traj_len: covered_frames as f64 / denom,
            avg_span: total_span as f64 / denom,
            total_segments,
            avg_segments: total_segments as f64 / denom,
            poly_segments,
            direct_segments,
            mixed_segments,
            total_canonical,
        }
    }
}
