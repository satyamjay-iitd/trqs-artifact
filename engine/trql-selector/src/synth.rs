//! Synthetic database generation for tests and benchmarks.

use trql_storage::db::{ObjWithBBox, Trqs, TrqsBuilder};
use trql_types::intervals::Interval;
use trql_types::obj::{BoundingBox, ExactTraj};
use trql_types::prop::Schema;

/// xorshift64* — deterministic, dependency-free RNG for workload generation.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed.max(1))
    }

    pub fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 32) as u32
    }

    /// Uniform in `[lo, hi]` (inclusive). Requires `lo <= hi`.
    pub fn gen_range(&mut self, lo: u32, hi: u32) -> u32 {
        lo + self.next_u32() % (hi - lo + 1)
    }
}

#[derive(Clone, Debug)]
pub struct SynthConfig {
    pub num_objs: usize,
    /// Segment starts are drawn uniformly from `[0, timeline)`.
    pub timeline: u32,
    /// Inclusive range for the number of canonical segments per object.
    pub segs_per_obj: Interval,
    /// Inclusive range for segment length in frames.
    pub seg_len: (u32, u32),
    pub seed: u64,
}

/// Builds a `Trqs` of random multi-segment trajectories (gap_threshold = 1).
pub fn gen_db(cfg: &SynthConfig) -> Trqs {
    gen_db_heavy_tail(cfg, 0, 0)
}

/// Like [`gen_db`], plus `long_lived` extra objects each covering a single
/// segment of `long_span` frames.
///
/// Uniform generation gives every object roughly the same span; real trajectory
/// data has a heavy right tail, and the tail is what a candidate-scan bound has
/// to survive. On i24, `pickup` spans run to a median of 230 frames against a
/// maximum of 16,892 — one stalled vehicle — so a bound derived from the slot's
/// widest candidate is ~73x looser than the typical one it is applied to.
pub fn gen_db_heavy_tail(cfg: &SynthConfig, long_lived: usize, long_span: u32) -> Trqs {
    let mut rng = Rng::new(cfg.seed);
    let schema = Schema::new(vec![]);
    let mut builder = TrqsBuilder::new(&schema);

    for i in 0..cfg.num_objs + long_lived {
        let mut ivs: Vec<Interval> = if i >= cfg.num_objs {
            let start = rng.gen_range(0, cfg.timeline.saturating_sub(long_span).max(1));
            vec![(start, start + long_span)]
        } else {
            let nsegs = rng.gen_range(cfg.segs_per_obj.0, cfg.segs_per_obj.1);
            (0..nsegs)
                .map(|_| {
                    let len = rng.gen_range(cfg.seg_len.0, cfg.seg_len.1);
                    let start = rng.gen_range(0, cfg.timeline.saturating_sub(len).max(1));
                    (start, start + len)
                })
                .collect()
        };
        ivs.sort_unstable();

        // Keep a gap of at least 2 frames between segments so the builder
        // (gap_threshold = 1) keeps them as separate canonical segments.
        let mut kept: Vec<Interval> = Vec::new();
        for (s, e) in ivs {
            match kept.last() {
                Some(&(_, pe)) if s <= pe.saturating_add(2) => {}
                _ => kept.push((s, e)),
            }
        }

        let mut frames = Vec::new();
        let mut bboxes = Vec::new();
        for &(s, e) in &kept {
            for f in s..=e {
                frames.push(f);
                bboxes.push(BoundingBox {
                    x1: 0,
                    y1: 0,
                    x2: 10,
                    y2: 10,
                });
            }
        }

        builder
            .ingest(ObjWithBBox {
                props: vec![],
                exact_traj: ExactTraj { frames, bboxes },
            })
            .unwrap();
    }

    builder.build(1, 0.0)
}
