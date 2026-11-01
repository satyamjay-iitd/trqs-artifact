use std::{collections::BTreeMap, fs, path::Path};

use anyhow::Context as _;
use serde::Serialize;

use crate::duckdb_meta::open_read_only;

/// Detection accuracy of a prediction against ground truth, computed by greedy
/// IoU matching per frame. Class-agnostic: it measures localization (did we put
/// a box in the right place), not classification, because the predictor's COCO
/// taxonomy generally differs from each dataset's label set.
#[derive(Serialize, Clone, Debug)]
pub struct Metrics {
    pub iou_threshold: f64,
    pub frames: u64,
    pub gt_boxes: u64,
    pub pred_boxes: u64,
    pub tp: u64,
    pub fp: u64,
    #[serde(rename = "fn")]
    pub fn_: u64,
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
    /// Predicted boxes discarded before matching because their centre fell in a
    /// region the ground truth declares un-annotated. Zero when the dataset
    /// declares none.
    pub pred_ignored: u64,
}

#[derive(Clone, Copy)]
struct BBox {
    x1: i64,
    y1: i64,
    x2: i64,
    y2: i64,
}

/// Compare the `pred_db` predictions against `gt_db` ground truth. Both must
/// have a `bbox(fid, x1, y1, x2, y2, …)` table on the same frame timeline and
/// pixel coordinate space (predictions run on the source video, so they align).
pub fn evaluate(gt_db: &Path, pred_db: &Path, iou_threshold: f64) -> anyhow::Result<Metrics> {
    let gt = read_boxes(gt_db).with_context(|| format!("reading {}", gt_db.display()))?;
    let pred = read_boxes(pred_db).with_context(|| format!("reading {}", pred_db.display()))?;

    // Some datasets mark areas of the frame as ignored during scoring —
    // UA-DETRAC's `ignored_region` covers ~15% of a typical frame. A detection
    // there is not a mistake, and counting it a false positive is most of why
    // DETRAC precision reads ~0.37 instead of ~0.53.
    //
    // Applied *after* matching, and only to unmatched predictions. DETRAC's
    // ground truth puts 23.5% of its boxes inside these regions, so discarding
    // every prediction in one throws away real matches: doing that cost 771 true
    // positives on MVI_20011 and took recall from 0.80 to 0.70. A detection that
    // found a real object is a success wherever it happened.
    let ignored = read_ignored_regions(gt_db)
        .with_context(|| format!("reading ignored regions from {}", gt_db.display()))?;

    let gt_boxes: u64 = gt.values().map(|v| v.len() as u64).sum();
    let pred_boxes: u64 = pred.values().map(|v| v.len() as u64).sum();

    let mut tp = 0u64;
    let mut fp = 0u64;
    let mut fn_ = 0u64;

    // Walk the union of frames present in either side.
    let mut fids: Vec<i64> = gt.keys().chain(pred.keys()).copied().collect();
    fids.sort_unstable();
    fids.dedup();
    let mut pred_ignored = 0u64;
    for fid in &fids {
        let g = gt.get(fid).map(Vec::as_slice).unwrap_or(&[]);
        let p = pred.get(fid).map(Vec::as_slice).unwrap_or(&[]);
        let pred_matched = match_frame(g, p, iou_threshold);
        let matched = pred_matched.iter().filter(|m| **m).count() as u64;
        // An unmatched prediction inside an ignored region is neither right nor
        // wrong: it is not scored at all.
        let excused = if ignored.is_empty() {
            0
        } else {
            p.iter()
                .zip(&pred_matched)
                .filter(|(b, m)| !**m && ignored.covers(*fid, b))
                .count() as u64
        };
        // Excusing a prediction that already matched would count it twice and
        // underflow the false-positive tally — an earlier version did exactly
        // that and reported precision above 1.
        debug_assert!(
            matched + excused <= p.len() as u64,
            "frame {fid}: {matched} matched + {excused} excused exceeds {} predictions",
            p.len()
        );
        tp += matched;
        fp += p.len() as u64 - matched - excused;
        fn_ += g.len() as u64 - matched;
        pred_ignored += excused;
    }

    let precision = ratio(tp, tp + fp);
    let recall = ratio(tp, tp + fn_);
    let f1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };

    Ok(Metrics {
        iou_threshold,
        frames: fids.len() as u64,
        gt_boxes,
        pred_boxes,
        tp,
        fp,
        fn_,
        precision,
        recall,
        f1,
        pred_ignored,
    })
}

/// The rectangles a dataset declares ignored, whole-sequence and per-frame.
///
/// Both forms exist because both are annotated: UA-DETRAC lists a handful of
/// rectangles for a whole sequence, VisDrone marks them per frame (category 0).
/// A row without a frame applies to every frame.
#[derive(Default)]
struct IgnoredRegions {
    global: Vec<IgnoredRegion>,
    per_frame: BTreeMap<i64, Vec<IgnoredRegion>>,
}

impl IgnoredRegions {
    fn is_empty(&self) -> bool {
        self.global.is_empty() && self.per_frame.is_empty()
    }

    /// Whether `b`'s centre falls in a region ignored on frame `fid`.
    fn covers(&self, fid: i64, b: &BBox) -> bool {
        self.global.iter().any(|r| r.contains_centre(b))
            || self
                .per_frame
                .get(&fid)
                .is_some_and(|rs| rs.iter().any(|r| r.contains_centre(b)))
    }
}

/// A rectangle the ground truth declares un-annotated.
struct IgnoredRegion {
    x1: f64,
    y1: f64,
    x2: f64,
    y2: f64,
}

impl IgnoredRegion {
    /// Centre-in-region, the test UA-DETRAC's own toolkit applies. A box
    /// overlapping the edge of an ignored area still counts, which is the
    /// conservative reading: only detections *of* the ignored thing are dropped.
    fn contains_centre(&self, b: &BBox) -> bool {
        let (cx, cy) = (
            (b.x1 + b.x2) as f64 / 2.0,
            (b.y1 + b.y2) as f64 / 2.0,
        );
        cx >= self.x1 && cx <= self.x2 && cy >= self.y1 && cy <= self.y2
    }
}

/// The `ignored_regions` table, or empty when the dataset declares none.
///
/// Optional by design: most datasets annotate the whole frame and have no such
/// table, and scoring must work for them unchanged. An optional `fid` column
/// makes a row apply to one frame instead of all of them — also optional, so a
/// table written before per-frame regions existed still reads.
fn read_ignored_regions(db: &Path) -> anyhow::Result<IgnoredRegions> {
    let conn = open_read_only(db)?;
    let present: bool = conn
        .prepare("SELECT 1 FROM duckdb_tables() WHERE table_name = 'ignored_regions'")?
        .exists([])?;
    if !present {
        return Ok(IgnoredRegions::default());
    }
    let has_fid: bool = conn
        .prepare(
            "SELECT 1 FROM duckdb_columns()
             WHERE table_name = 'ignored_regions' AND column_name = 'fid'",
        )?
        .exists([])?;

    let fid_col = if has_fid { "fid" } else { "NULL" };
    let sql = format!(
        "SELECT CAST({fid_col} AS BIGINT), CAST(x1 AS DOUBLE), CAST(y1 AS DOUBLE),
                CAST(x2 AS DOUBLE), CAST(y2 AS DOUBLE) FROM ignored_regions"
    );
    let mut stmt = conn.prepare(&sql)?;
    let mut rows = stmt.query([])?;
    let mut out = IgnoredRegions::default();
    while let Some(row) = rows.next()? {
        let fid: Option<i64> = row.get(0)?;
        let region = IgnoredRegion {
            x1: row.get(1)?,
            y1: row.get(2)?,
            x2: row.get(3)?,
            y2: row.get(4)?,
        };
        match fid {
            Some(f) => out.per_frame.entry(f).or_default().push(region),
            None => out.global.push(region),
        }
    }
    Ok(out)
}

/// Write `metrics.json` into `dir` and return the metrics for logging.
pub fn evaluate_to_file(
    gt_db: &Path,
    pred_db: &Path,
    dir: &Path,
    iou_threshold: f64,
) -> anyhow::Result<Metrics> {
    let metrics = evaluate(gt_db, pred_db, iou_threshold)?;
    let path = dir.join("metrics.json");
    fs::write(&path, serde_json::to_vec_pretty(&metrics)?)
        .with_context(|| format!("failed to write {}", path.display()))?;
    Ok(metrics)
}

fn ratio(num: u64, den: u64) -> f64 {
    if den == 0 {
        0.0
    } else {
        num as f64 / den as f64
    }
}

/// One-to-one greedy IoU matching: take the highest-IoU pairs first, each box
/// used once. Returns, per predicted box, whether it matched a ground-truth one.
///
/// Per-prediction rather than a count because the caller has to tell an unmatched
/// prediction from a matched one to apply ignored regions to the former only.
fn match_frame(gt: &[BBox], pred: &[BBox], thr: f64) -> Vec<bool> {
    let mut pairs: Vec<(f64, usize, usize)> = Vec::new();
    for (i, g) in gt.iter().enumerate() {
        for (j, p) in pred.iter().enumerate() {
            let v = iou(g, p);
            if v >= thr {
                pairs.push((v, i, j));
            }
        }
    }
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut gt_used = vec![false; gt.len()];
    let mut pred_used = vec![false; pred.len()];
    for (_, i, j) in pairs {
        if !gt_used[i] && !pred_used[j] {
            gt_used[i] = true;
            pred_used[j] = true;
        }
    }
    pred_used
}

fn iou(a: &BBox, b: &BBox) -> f64 {
    let ix1 = a.x1.max(b.x1);
    let iy1 = a.y1.max(b.y1);
    let ix2 = a.x2.min(b.x2);
    let iy2 = a.y2.min(b.y2);
    let iw = (ix2 - ix1).max(0) as f64;
    let ih = (iy2 - iy1).max(0) as f64;
    let inter = iw * ih;
    let area = |b: &BBox| ((b.x2 - b.x1).max(0) as f64) * ((b.y2 - b.y1).max(0) as f64);
    let union = area(a) + area(b) - inter;
    if union <= 0.0 { 0.0 } else { inter / union }
}

/// Read all boxes from a `bbox` table, bucketed by frame id. Casts to BIGINT so
/// INTEGER/BIGINT columns are read uniformly.
fn read_boxes(db: &Path) -> anyhow::Result<BTreeMap<i64, Vec<BBox>>> {
    let conn = open_read_only(db)?;
    let mut stmt = conn
        .prepare("SELECT fid::BIGINT, x1::BIGINT, y1::BIGINT, x2::BIGINT, y2::BIGINT FROM bbox")?;
    let mut rows = stmt.query([])?;
    let mut out: BTreeMap<i64, Vec<BBox>> = BTreeMap::new();
    while let Some(row) = rows.next()? {
        let fid: i64 = row.get(0)?;
        out.entry(fid).or_default().push(BBox {
            x1: row.get(1)?,
            y1: row.get(2)?,
            x2: row.get(3)?,
            y2: row.get(4)?,
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iou_basics() {
        let a = BBox {
            x1: 0,
            y1: 0,
            x2: 10,
            y2: 10,
        };
        assert!((iou(&a, &a) - 1.0).abs() < 1e-9);
        let b = BBox {
            x1: 100,
            y1: 100,
            x2: 110,
            y2: 110,
        };
        assert_eq!(iou(&a, &b), 0.0);
        let c = BBox {
            x1: 5,
            y1: 0,
            x2: 15,
            y2: 10,
        }; // half overlap
        assert!((iou(&a, &c) - (50.0 / 150.0)).abs() < 1e-9);
    }

    #[test]
    fn greedy_one_to_one() {
        let g = vec![
            BBox {
                x1: 0,
                y1: 0,
                x2: 10,
                y2: 10,
            },
            BBox {
                x1: 20,
                y1: 20,
                x2: 30,
                y2: 30,
            },
        ];
        // One good match, one box with no counterpart.
        let p = vec![BBox {
            x1: 0,
            y1: 0,
            x2: 10,
            y2: 10,
        }];
        assert_eq!(match_frame(&g, &p, 0.5), vec![true]);
        // Two preds overlapping one gt: only one may match.
        let p2 = vec![
            BBox {
                x1: 0,
                y1: 0,
                x2: 10,
                y2: 10,
            },
            BBox {
                x1: 1,
                y1: 1,
                x2: 11,
                y2: 11,
            },
        ];
        // The better-overlapping prediction wins; the other is left unmatched, and
        // it is that distinction the caller needs to apply ignored regions.
        assert_eq!(match_frame(&g[..1], &p2, 0.5), vec![true, false]);
    }
}
