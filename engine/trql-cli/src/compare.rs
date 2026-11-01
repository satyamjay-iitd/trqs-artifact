//! Compare a query's output against ground truth.
//!
//! A [`QueryOutput`] is the set of matched object-id tuples, each with the
//! frame intervals it holds over. Both the live query result and the
//! ground-truth CSV are lifted into a `QueryOutput`, then compared: tuple-level
//! precision/recall/F1 (did we find the right object combinations) plus the
//! mean temporal IoU over the matched tuples (did we get the right time
//! windows).
//!
//! The ground-truth CSV has one row per `(tuple, interval)`, with a column per
//! slot named `oid0`, `oid1`, … and inclusive-frame `start` / `end` columns:
//!
//! ```csv
//! oid0,oid1,start,end
//! 7,12,10,45
//! 7,12,90,120
//! 3,9,5,30
//! ```

use std::{collections::BTreeMap, path::Path};

use anyhow::{Context as _, anyhow};
use serde::Serialize;

/// How many false-positive / false-negative tuples to keep as samples.
const SAMPLE_LIMIT: usize = 20;

/// A query result: each matched tuple (object ids per slot) mapped to the
/// merged, inclusive `[start, end]` frame intervals it holds over.
pub struct QueryOutput {
    rows: BTreeMap<Vec<u32>, Vec<(u32, u32)>>,
}

/// The outcome of comparing a query output against ground truth.
#[derive(Serialize)]
pub struct CompareMetrics {
    /// Distinct tuples in the query output.
    pub query_tuples: usize,
    /// Distinct tuples in the ground truth.
    pub gt_tuples: usize,
    /// Tuples in both (true positives).
    pub tp: usize,
    /// Tuples only in the query output (false positives).
    pub fp: usize,
    /// Tuples only in the ground truth (false negatives).
    pub fn_: usize,
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
    /// Mean temporal IoU over the `tp` matched tuples; `0.0` when none match.
    pub mean_iou: f64,
    /// Matched tuples whose interval sets are not identical. `mean_iou` can sit
    /// just under 1.0 for a single frame's disagreement across thousands of
    /// tuples, which rounds away in a report; this counts them instead.
    pub interval_mismatches: usize,
    /// Up to [`SAMPLE_LIMIT`] tuples present only in the query output.
    pub sample_fp: Vec<Vec<u32>>,
    /// Up to [`SAMPLE_LIMIT`] tuples present only in the ground truth.
    pub sample_fn: Vec<Vec<u32>>,
}

impl QueryOutput {
    /// Lift raw query rows (`tuple`, its intervals) into a `QueryOutput`,
    /// grouping by tuple and merging intervals.
    pub fn from_rows(rows: impl IntoIterator<Item = (Vec<u32>, Vec<(u32, u32)>)>) -> Self {
        let mut map: BTreeMap<Vec<u32>, Vec<(u32, u32)>> = BTreeMap::new();
        for (tuple, intervals) in rows {
            map.entry(tuple).or_default().extend(intervals);
        }
        for intervals in map.values_mut() {
            *intervals = merge(std::mem::take(intervals));
        }
        Self { rows: map }
    }

    /// Write these rows as the `oid0,oid1,…,start,end` CSV [`Self::from_csv`] reads.
    ///
    /// Answers only round-trip through this: the engine reports interval *sets*
    /// whose 5-tuples can describe overlapping spans, and it is the merged spans
    /// [`Self::from_rows`] produced that mean something. Writing the raw sets and
    /// merging them somewhere else is how the same query's answer ends up
    /// disagreeing with itself.
    ///
    /// `slots` is passed rather than read off the rows: a query that matched
    /// nothing still has an arity, and a header of bare `start,end` is not a
    /// file [`Self::from_csv`] will accept. An empty answer is an answer.
    pub fn write_csv(&self, path: &Path, slots: usize) -> anyhow::Result<()> {
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
        debug_assert!(
            self.rows.keys().all(|t| t.len() == slots),
            "every tuple should have the query's arity"
        );
        let mut w = csv::Writer::from_path(path)
            .with_context(|| format!("failed to write {}", path.display()))?;
        let mut header: Vec<String> = (0..slots).map(|i| format!("oid{i}")).collect();
        header.push("start".to_owned());
        header.push("end".to_owned());
        w.write_record(&header)?;
        for (tuple, intervals) in &self.rows {
            for (s, e) in intervals {
                let mut rec: Vec<String> = tuple.iter().map(u32::to_string).collect();
                rec.push(s.to_string());
                rec.push(e.to_string());
                w.write_record(&rec)?;
            }
        }
        w.flush()?;
        Ok(())
    }

    /// Read ground truth from a CSV with `oid0,oid1,…,start,end` columns.
    pub fn from_csv(path: &Path) -> anyhow::Result<Self> {
        let mut reader = csv::Reader::from_path(path)
            .with_context(|| format!("failed to open {}", path.display()))?;
        let headers = reader
            .headers()
            .with_context(|| format!("failed to read header of {}", path.display()))?
            .clone();

        // Slot columns are `oid<N>`, ordered by N; then the interval columns.
        let mut oid_cols: Vec<(u32, usize)> = headers
            .iter()
            .enumerate()
            .filter_map(|(i, h)| h.strip_prefix("oid")?.parse::<u32>().ok().map(|n| (n, i)))
            .collect();
        oid_cols.sort_unstable();
        let column = |name: &str| headers.iter().position(|h| h == name);
        let (start_col, end_col) = match (column("start"), column("end")) {
            (Some(s), Some(e)) => (s, e),
            _ => {
                return Err(anyhow!(
                    "{} is missing `start`/`end` columns; expected `oid0,oid1,…,start,end`",
                    path.display()
                ));
            }
        };
        if oid_cols.is_empty() {
            return Err(anyhow!(
                "{} has no `oid<N>` slot columns; expected `oid0,oid1,…,start,end`",
                path.display()
            ));
        }

        let mut map: BTreeMap<Vec<u32>, Vec<(u32, u32)>> = BTreeMap::new();
        for (line, record) in reader.records().enumerate() {
            let record =
                record.with_context(|| format!("reading {} row {}", path.display(), line + 1))?;
            let cell = |idx: usize, what: &str| -> anyhow::Result<u32> {
                record
                    .get(idx)
                    .ok_or_else(|| anyhow!("row {}: missing {what}", line + 1))?
                    .trim()
                    .parse::<u32>()
                    .with_context(|| format!("row {}: invalid {what}", line + 1))
            };
            let tuple = oid_cols
                .iter()
                .map(|&(_, i)| cell(i, "oid"))
                .collect::<anyhow::Result<Vec<u32>>>()?;
            let interval = (cell(start_col, "start")?, cell(end_col, "end")?);
            map.entry(tuple).or_default().push(interval);
        }
        for intervals in map.values_mut() {
            *intervals = merge(std::mem::take(intervals));
        }
        Ok(Self { rows: map })
    }

    /// Compare this output (the query) against `gt` (ground truth).
    pub fn compare(&self, gt: &QueryOutput) -> CompareMetrics {
        let mut tp = 0usize;
        let mut iou_sum = 0.0;
        let mut interval_mismatches = 0usize;
        let mut sample_fp = Vec::new();
        let mut sample_fn = Vec::new();

        for (tuple, q_intervals) in &self.rows {
            match gt.rows.get(tuple) {
                Some(g_intervals) => {
                    tp += 1;
                    iou_sum += iou(q_intervals, g_intervals);
                    if q_intervals != g_intervals {
                        interval_mismatches += 1;
                    }
                }
                None if sample_fp.len() < SAMPLE_LIMIT => sample_fp.push(tuple.clone()),
                None => {}
            }
        }
        for tuple in gt.rows.keys() {
            if !self.rows.contains_key(tuple) && sample_fn.len() < SAMPLE_LIMIT {
                sample_fn.push(tuple.clone());
            }
        }

        let query_tuples = self.rows.len();
        let gt_tuples = gt.rows.len();
        let fp = query_tuples - tp;
        let fn_ = gt_tuples - tp;

        let ratio = |num: usize, den: usize| {
            if den == 0 {
                0.0
            } else {
                num as f64 / den as f64
            }
        };
        let precision = ratio(tp, query_tuples);
        let recall = ratio(tp, gt_tuples);
        let f1 = if precision + recall == 0.0 {
            0.0
        } else {
            2.0 * precision * recall / (precision + recall)
        };
        let mean_iou = if tp == 0 { 0.0 } else { iou_sum / tp as f64 };

        CompareMetrics {
            query_tuples,
            gt_tuples,
            tp,
            fp,
            fn_,
            precision,
            recall,
            f1,
            mean_iou,
            interval_mismatches,
            sample_fp,
            sample_fn,
        }
    }
}

/// Merge inclusive `[start, end]` ranges into sorted, non-overlapping ranges,
/// dropping invalid (`start > end`) ones.
fn merge(mut ranges: Vec<(u32, u32)>) -> Vec<(u32, u32)> {
    ranges.retain(|&(s, e)| s <= e);
    ranges.sort_unstable();
    let mut out: Vec<(u32, u32)> = Vec::with_capacity(ranges.len());
    for (s, e) in ranges {
        match out.last_mut() {
            // Overlapping or touching (inclusive) — extend the current range.
            Some(last) if s <= last.1.saturating_add(1) => last.1 = last.1.max(e),
            _ => out.push((s, e)),
        }
    }
    out
}

/// Total covered frames of merged, inclusive ranges.
fn frames(ranges: &[(u32, u32)]) -> u64 {
    ranges.iter().map(|&(s, e)| u64::from(e - s + 1)).sum()
}

/// Frames covered by both merged range sets.
fn intersect_frames(a: &[(u32, u32)], b: &[(u32, u32)]) -> u64 {
    let (mut i, mut j, mut total) = (0, 0, 0u64);
    while i < a.len() && j < b.len() {
        let (s1, e1) = a[i];
        let (s2, e2) = b[j];
        let lo = s1.max(s2);
        let hi = e1.min(e2);
        if lo <= hi {
            total += u64::from(hi - lo + 1);
        }
        // Advance the range that ends first.
        if e1 < e2 {
            i += 1;
        } else {
            j += 1;
        }
    }
    total
}

/// Temporal intersection-over-union of two merged range sets.
fn iou(a: &[(u32, u32)], b: &[(u32, u32)]) -> f64 {
    let inter = intersect_frames(a, b);
    let union = frames(a) + frames(b) - inter;
    if union == 0 {
        0.0
    } else {
        inter as f64 / union as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn out(rows: &[(&[u32], &[(u32, u32)])]) -> QueryOutput {
        QueryOutput::from_rows(rows.iter().map(|&(t, iv)| (t.to_vec(), iv.to_vec())))
    }

    #[test]
    fn identical_outputs_score_perfectly() {
        let q = out(&[(&[7], &[(0, 9)]), (&[3, 9], &[(5, 30)])]);
        let g = out(&[(&[7], &[(0, 9)]), (&[3, 9], &[(5, 30)])]);
        let m = q.compare(&g);
        assert_eq!((m.tp, m.fp, m.fn_), (2, 0, 0));
        assert_eq!(m.precision, 1.0);
        assert_eq!(m.recall, 1.0);
        assert_eq!(m.f1, 1.0);
        assert_eq!(m.mean_iou, 1.0);
    }

    #[test]
    fn tuple_fp_and_fn_counted() {
        let q = out(&[(&[1], &[(0, 10)]), (&[2], &[(0, 10)])]); // 2 has no GT
        let g = out(&[(&[1], &[(0, 10)]), (&[3], &[(0, 10)])]); // 3 missed
        let m = q.compare(&g);
        assert_eq!((m.tp, m.fp, m.fn_), (1, 1, 1));
        assert_eq!(m.sample_fp, vec![vec![2]]);
        assert_eq!(m.sample_fn, vec![vec![3]]);
        assert!((m.precision - 0.5).abs() < 1e-9);
        assert!((m.recall - 0.5).abs() < 1e-9);
    }

    #[test]
    fn partial_interval_overlap_is_iou() {
        // Same tuple; query [0,9] (10 frames), gt [5,14] (10 frames).
        // intersection [5,9] = 5, union = 15 -> IoU 1/3.
        let q = out(&[(&[1], &[(0, 9)])]);
        let g = out(&[(&[1], &[(5, 14)])]);
        let m = q.compare(&g);
        assert_eq!(m.tp, 1);
        assert!(
            (m.mean_iou - 1.0 / 3.0).abs() < 1e-9,
            "iou was {}",
            m.mean_iou
        );
    }

    #[test]
    fn overlapping_input_ranges_are_merged() {
        // [0,4] and [3,9] merge to [0,9]; matched exactly by gt [0,9].
        let q = out(&[(&[1], &[(0, 4), (3, 9)])]);
        let g = out(&[(&[1], &[(0, 9)])]);
        assert!((q.compare(&g).mean_iou - 1.0).abs() < 1e-9);
    }
}
