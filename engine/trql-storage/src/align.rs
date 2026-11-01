use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    fs,
    path::Path,
};

use anyhow::Context as _;
use duckdb::Connection;
use serde::Serialize;

use crate::duckdb_meta::open_read_only;

/// Summary of an id-alignment pass.
#[derive(Serialize, Clone, Debug)]
pub struct AlignStats {
    pub gt_objects: u64,
    pub pred_objects: u64,
    pub matched: u64,
    /// The starting id the predicted objects were renumbered from (GT's base).
    pub base: i64,
}

#[derive(Clone, Copy)]
struct BBox {
    x1: i64,
    y1: i64,
    x2: i64,
    y2: i64,
}

/// Align the predicted track ids in `pred_db` to the ground-truth ids in
/// `gt_db`, rewriting `pred_db` in place, and write an `alignment.json` summary
/// into `dir`.
///
/// Predicted objects are renumbered into a **continuous** range starting at the
/// GT base id: tracks matched to a GT object come first (ordered by GT id, so a
/// perfect prediction reproduces the GT ids), then unmatched tracks. The
/// `predicted -> GT` correspondence is stored in an `id_map` table in `pred_db`.
///
/// Errors if the GT `oid`s are not continuous.
pub fn align_to_file(
    gt_db: &Path,
    pred_db: &Path,
    dir: &Path,
    iou_threshold: f64,
) -> anyhow::Result<AlignStats> {
    let stats = align_ids(gt_db, pred_db, iou_threshold)?;
    let path = dir.join("alignment.json");
    fs::write(&path, serde_json::to_vec_pretty(&stats)?)
        .with_context(|| format!("failed to write {}", path.display()))?;
    Ok(stats)
}

pub fn align_ids(gt_db: &Path, pred_db: &Path, thr: f64) -> anyhow::Result<AlignStats> {
    let gt = read_frames(gt_db).with_context(|| format!("reading {}", gt_db.display()))?;
    let pred = read_frames(pred_db).with_context(|| format!("reading {}", pred_db.display()))?;

    // GT ids must form a gap-free range.
    let gt_oids = sorted_oids(&gt);
    if let (Some(&min), Some(&max)) = (gt_oids.first(), gt_oids.last()) {
        if (max - min + 1) as usize != gt_oids.len() {
            anyhow::bail!(
                "GT oids are not continuous: {} distinct in range [{min}, {max}]",
                gt_oids.len()
            );
        }
    }
    let base = gt_oids.first().copied().unwrap_or(0);

    // Every predicted id must be remapped, including objects with no boxes.
    let mut pred_oids: BTreeSet<i64> = sorted_oids(&pred).into_iter().collect();
    pred_oids.extend(object_oids(pred_db)?);

    // Track affinity = number of frames where a box of the predicted track is
    // one-to-one matched (IoU >= thr) to a box of the GT track.
    let mut counts: HashMap<(i64, i64), u32> = HashMap::new();
    let mut fids: Vec<i64> = gt.keys().chain(pred.keys()).copied().collect();
    fids.sort_unstable();
    fids.dedup();
    for fid in &fids {
        if let (Some(g), Some(p)) = (gt.get(fid), pred.get(fid)) {
            for (gt_oid, pred_oid) in frame_matches(g, p, thr) {
                *counts.entry((pred_oid, gt_oid)).or_insert(0) += 1;
            }
        }
    }

    // Greedy one-to-one matching over the affinity counts, best first.
    let mut pairs: Vec<(u32, i64, i64)> = counts
        .into_iter()
        .map(|((pred_oid, gt_oid), c)| (c, pred_oid, gt_oid))
        .collect();
    pairs.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
    let mut matched: HashMap<i64, i64> = HashMap::new(); // pred_oid -> gt_oid
    let mut gt_used: HashSet<i64> = HashSet::new();
    for (_, pred_oid, gt_oid) in pairs {
        if !matched.contains_key(&pred_oid) && !gt_used.contains(&gt_oid) {
            matched.insert(pred_oid, gt_oid);
            gt_used.insert(gt_oid);
        }
    }

    // Build the continuous remap: matched (ordered by GT id) first, then the
    // rest by old id, numbered densely from `base`.
    let mut matched_pairs: Vec<(i64, i64)> = matched.iter().map(|(&p, &g)| (p, g)).collect();
    matched_pairs.sort_by_key(|&(_, g)| g);
    let mut remap: Vec<(i64, i64, Option<i64>)> = Vec::with_capacity(pred_oids.len());
    let mut next = base;
    for (pred_oid, gt_oid) in &matched_pairs {
        remap.push((*pred_oid, next, Some(*gt_oid)));
        next += 1;
    }
    for &pred_oid in &pred_oids {
        if !matched.contains_key(&pred_oid) {
            remap.push((pred_oid, next, None));
            next += 1;
        }
    }

    rewrite(pred_db, &remap)?;

    Ok(AlignStats {
        gt_objects: gt_oids.len() as u64,
        pred_objects: pred_oids.len() as u64,
        matched: matched_pairs.len() as u64,
        base,
    })
}

/// Rewrite `objects`/`bbox` oids via `remap` and record the predicted->GT
/// mapping as an `id_map` table, all in one transaction.
fn rewrite(pred_db: &Path, remap: &[(i64, i64, Option<i64>)]) -> anyhow::Result<()> {
    let conn = Connection::open(pred_db)
        .with_context(|| format!("failed to open {}", pred_db.display()))?;

    let mut values = String::new();
    for (i, (old, new, gt)) in remap.iter().enumerate() {
        if i > 0 {
            values.push(',');
        }
        match gt {
            Some(g) => values.push_str(&format!("({old},{new},{g})")),
            None => values.push_str(&format!("({old},{new},NULL)")),
        }
    }

    conn.execute_batch("BEGIN;")?;
    conn.execute_batch("CREATE TEMP TABLE _remap(old BIGINT, new INTEGER, gt INTEGER);")?;
    if !values.is_empty() {
        conn.execute_batch(&format!("INSERT INTO _remap VALUES {values};"))?;
    }
    conn.execute_batch(
        "CREATE TABLE _objects_new AS
             SELECT r.new AS oid, o.* EXCLUDE (oid)
             FROM objects o JOIN _remap r ON o.oid = r.old;
         DROP TABLE objects;
         ALTER TABLE _objects_new RENAME TO objects;

         CREATE TABLE _bbox_new AS
             SELECT r.new AS oid, b.* EXCLUDE (oid)
             FROM bbox b JOIN _remap r ON b.oid = r.old;
         DROP TABLE bbox;
         ALTER TABLE _bbox_new RENAME TO bbox;

         CREATE OR REPLACE TABLE id_map AS
             SELECT new AS pred_oid, gt AS gt_oid FROM _remap ORDER BY new;
         DROP TABLE _remap;",
    )?;
    conn.execute_batch("COMMIT;")?;
    Ok(())
}

/// One-to-one greedy IoU matching within a frame, returning `(gt_oid, pred_oid)`
/// for each matched pair.
fn frame_matches(gt: &[(i64, BBox)], pred: &[(i64, BBox)], thr: f64) -> Vec<(i64, i64)> {
    let mut pairs: Vec<(f64, usize, usize)> = Vec::new();
    for (i, (_, g)) in gt.iter().enumerate() {
        for (j, (_, p)) in pred.iter().enumerate() {
            let v = iou(g, p);
            if v >= thr {
                pairs.push((v, i, j));
            }
        }
    }
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut gt_used = vec![false; gt.len()];
    let mut pred_used = vec![false; pred.len()];
    let mut out = Vec::new();
    for (_, i, j) in pairs {
        if !gt_used[i] && !pred_used[j] {
            gt_used[i] = true;
            pred_used[j] = true;
            out.push((gt[i].0, pred[j].0));
        }
    }
    out
}

fn iou(a: &BBox, b: &BBox) -> f64 {
    let iw = (a.x2.min(b.x2) - a.x1.max(b.x1)).max(0) as f64;
    let ih = (a.y2.min(b.y2) - a.y1.max(b.y1)).max(0) as f64;
    let inter = iw * ih;
    let area = |b: &BBox| ((b.x2 - b.x1).max(0) as f64) * ((b.y2 - b.y1).max(0) as f64);
    let union = area(a) + area(b) - inter;
    if union <= 0.0 { 0.0 } else { inter / union }
}

fn sorted_oids(frames: &BTreeMap<i64, Vec<(i64, BBox)>>) -> Vec<i64> {
    let set: BTreeSet<i64> = frames.values().flatten().map(|(oid, _)| *oid).collect();
    set.into_iter().collect()
}

/// All oids in the `objects` table (may include objects with no boxes).
fn object_oids(db: &Path) -> anyhow::Result<Vec<i64>> {
    let conn = open_read_only(db)?;
    let mut stmt = conn.prepare("SELECT oid::BIGINT FROM objects")?;
    let mut rows = stmt.query([])?;
    let mut out = Vec::new();
    while let Some(row) = rows.next()? {
        out.push(row.get(0)?);
    }
    Ok(out)
}

fn read_frames(db: &Path) -> anyhow::Result<BTreeMap<i64, Vec<(i64, BBox)>>> {
    let conn = open_read_only(db)?;
    let mut stmt = conn.prepare(
        "SELECT fid::BIGINT, oid::BIGINT, x1::BIGINT, y1::BIGINT, x2::BIGINT, y2::BIGINT FROM bbox",
    )?;
    let mut rows = stmt.query([])?;
    let mut out: BTreeMap<i64, Vec<(i64, BBox)>> = BTreeMap::new();
    while let Some(row) = rows.next()? {
        let fid: i64 = row.get(0)?;
        let oid: i64 = row.get(1)?;
        out.entry(fid).or_default().push((
            oid,
            BBox {
                x1: row.get(2)?,
                y1: row.get(3)?,
                x2: row.get(4)?,
                y2: row.get(5)?,
            },
        ));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a duckdb with `objects(oid,type)` + `bbox(oid,fid,x1,y1,x2,y2)`.
    /// A box is placed for every (oid, frame) at a per-oid location.
    fn make_db(path: &Path, oids: &[i64], frames: &[i64], loc: impl Fn(i64) -> (i64, i64)) {
        let conn = Connection::open(path).unwrap();
        conn.execute_batch(
            "CREATE TABLE objects(oid INTEGER, type VARCHAR);
             CREATE TABLE bbox(oid INTEGER, fid INTEGER, x1 INTEGER, y1 INTEGER, x2 INTEGER, y2 INTEGER);",
        )
        .unwrap();
        for &oid in oids {
            conn.execute(&format!("INSERT INTO objects VALUES ({oid}, 'car')"), [])
                .unwrap();
            let (x, y) = loc(oid);
            for &f in frames {
                conn.execute(
                    &format!(
                        "INSERT INTO bbox VALUES ({oid},{f},{x},{y},{},{})",
                        x + 10,
                        y + 10
                    ),
                    [],
                )
                .unwrap();
            }
        }
    }

    #[test]
    fn aligns_and_renumbers_continuously() {
        let dir = std::env::temp_dir().join(format!("trql-align-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let gt = dir.join("gt.db");
        let pred = dir.join("pred.db");

        // GT ids 0,1,2 (continuous), each at a distinct location.
        let locs = |oid: i64| ((oid % 3) * 100, 0);
        make_db(&gt, &[0, 1, 2], &[0, 1], locs);
        // Pred tracker ids 5,9,12: 5 sits on GT0's spot, 9 on GT2's spot, 12 is
        // a false positive far away.
        make_db(&pred, &[5, 9, 12], &[0, 1], |oid| match oid {
            5 => (0, 0),     // GT0
            9 => (200, 0),   // GT2
            _ => (999, 999), // no overlap
        });

        let stats = align_ids(&gt, &pred, 0.5).unwrap();
        assert_eq!((stats.matched, stats.pred_objects, stats.base), (2, 3, 0));

        let conn = open_read_only(&pred).unwrap();
        // Predicted oids are continuous 0..2.
        let ids: Vec<i64> = {
            let mut s = conn
                .prepare("SELECT oid::BIGINT FROM objects ORDER BY oid")
                .unwrap();
            let mut r = s.query([]).unwrap();
            let mut v = Vec::new();
            while let Some(row) = r.next().unwrap() {
                v.push(row.get(0).unwrap());
            }
            v
        };
        assert_eq!(ids, vec![0, 1, 2]);

        // id_map: matched preds carry their GT id (0->GT0, 1->GT2); FP -> NULL.
        let map: Vec<(i64, Option<i64>)> = {
            let mut s = conn
                .prepare("SELECT pred_oid::BIGINT, gt_oid::BIGINT FROM id_map ORDER BY pred_oid")
                .unwrap();
            let mut r = s.query([]).unwrap();
            let mut v = Vec::new();
            while let Some(row) = r.next().unwrap() {
                v.push((row.get(0).unwrap(), row.get(1).unwrap()));
            }
            v
        };
        assert_eq!(map, vec![(0, Some(0)), (1, Some(2)), (2, None)]);

        // The GT0-located box now has oid 0, the GT2-located box oid 1.
        let oid_at = |x: i64| -> i64 {
            conn.query_row(
                &format!("SELECT DISTINCT oid::BIGINT FROM bbox WHERE x1 = {x}"),
                [],
                |r| r.get(0),
            )
            .unwrap()
        };
        assert_eq!(oid_at(0), 0);
        assert_eq!(oid_at(200), 1);
        assert_eq!(oid_at(999), 2);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn errors_on_non_continuous_gt() {
        let dir = std::env::temp_dir().join(format!("trql-align-nc-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let gt = dir.join("gt.db");
        let pred = dir.join("pred.db");
        make_db(&gt, &[0, 2], &[0], |oid| (oid * 100, 0)); // gap at 1
        make_db(&pred, &[7], &[0], |_| (0, 0));

        let err = align_ids(&gt, &pred, 0.5).unwrap_err();
        assert!(err.to_string().contains("not continuous"), "{err}");

        let _ = fs::remove_dir_all(&dir);
    }
}
