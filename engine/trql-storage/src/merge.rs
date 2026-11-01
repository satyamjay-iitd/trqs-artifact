use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::Context as _;
use duckdb::Connection;

/// One dataset to fold into a merge, in merge order.
#[derive(Clone, Debug)]
pub struct MergeSource {
    /// Stable identity for UI status tracking (the source `duckdb.db` path).
    pub key: PathBuf,
    /// Human label shown in the log/progress (the relative path).
    pub label: String,
    pub duckdb_path: PathBuf,
    /// The source `video.mp4`, if one exists; symlinked into the merge output.
    pub video_path: Option<PathBuf>,
    /// Frame count from metadata, used to lay out the continuous frame timeline.
    /// Falls back to `max(fid) + 1` when absent.
    pub num_frames: Option<u32>,
    /// This source's `metadata.json` (or `world-metadata.json`), if it has one.
    ///
    /// Read as raw JSON rather than [`DatasetMetadata`] on purpose: the fields
    /// that most need carrying into a merge — `coordinate_scale`, `x_delta`,
    /// `y_delta` — are not modelled by that struct, and are read straight from
    /// the JSON by `trql-server` and the paper's query tooling.
    ///
    /// [`DatasetMetadata`]: crate::datasets::DatasetMetadata
    pub metadata_path: Option<PathBuf>,
}

/// Progress events emitted while merging, one Started/Finished pair per source.
pub enum MergeEvent {
    Started {
        key: PathBuf,
        label: String,
    },
    Finished {
        key: PathBuf,
        label: String,
        /// Objects contributed by this source, or an error message.
        result: Result<u32, String>,
    },
}

pub struct MergeSummary {
    pub out_dir: PathBuf,
    pub num_objs: u32,
    pub num_sources: usize,
    pub total_frames: u32,
}

/// The directory of symlinked source videos within a merged dataset.
pub const VIDEO_DIR: &str = "videos";

/// Merge `sources` into a fresh dataset at `out_dir`.
///
/// `oid` and `fid` are offset per source so that object and frame ids stay
/// continuous and non-overlapping across the whole merge. Videos are not
/// copied: each source `video.mp4` is symlinked into `out_dir/videos/` under
/// the name `<start_fid>_<end_fid>.mp4`, the merged-frame range it covers.
pub fn merge_datasets(
    sources: &[MergeSource],
    out_dir: &Path,
    progress: &mut dyn FnMut(MergeEvent),
) -> anyhow::Result<MergeSummary> {
    if sources.len() < 2 {
        anyhow::bail!("merge needs at least 2 datasets");
    }

    fs::create_dir_all(out_dir)
        .with_context(|| format!("failed to create {}", out_dir.display()))?;
    let out_duckdb = out_dir.join("duckdb.db");
    if out_duckdb.exists() {
        fs::remove_file(&out_duckdb)
            .with_context(|| format!("failed to remove existing {}", out_duckdb.display()))?;
    }
    let conn = Connection::open(&out_duckdb)
        .with_context(|| format!("failed to create {}", out_duckdb.display()))?;

    let video_dir = out_dir.join(VIDEO_DIR);
    fs::create_dir_all(&video_dir)
        .with_context(|| format!("failed to create {}", video_dir.display()))?;

    let mut frame_cursor: i64 = 0;
    let mut obj_cursor: i64 = 0;
    let mut total_objs: u32 = 0;
    let mut tables_created = false;

    for src in sources {
        progress(MergeEvent::Started {
            key: src.key.clone(),
            label: src.label.clone(),
        });
        match merge_one(
            &conn,
            src,
            &video_dir,
            &mut frame_cursor,
            &mut obj_cursor,
            &mut tables_created,
        ) {
            Ok(objs) => {
                total_objs = total_objs.saturating_add(objs);
                progress(MergeEvent::Finished {
                    key: src.key.clone(),
                    label: src.label.clone(),
                    result: Ok(objs),
                });
            }
            Err(err) => {
                let msg = format!("{err:#}");
                progress(MergeEvent::Finished {
                    key: src.key.clone(),
                    label: src.label.clone(),
                    result: Err(msg.clone()),
                });
                return Err(err.context(format!("merging {}", src.label)));
            }
        }
    }

    let total_frames = frame_cursor.max(0) as u32;
    write_metadata(out_dir, sources, total_frames)?;

    Ok(MergeSummary {
        out_dir: out_dir.to_path_buf(),
        num_objs: total_objs,
        num_sources: sources.len(),
        total_frames,
    })
}

/// Fold a single source into the output connection, advancing the id cursors.
/// Returns the number of objects contributed.
fn merge_one(
    conn: &Connection,
    src: &MergeSource,
    video_dir: &Path,
    frame_cursor: &mut i64,
    obj_cursor: &mut i64,
    tables_created: &mut bool,
) -> anyhow::Result<u32> {
    let attach = format!(
        "ATTACH '{}' AS src (READ_ONLY);",
        sql_escape(&src.duckdb_path.to_string_lossy())
    );
    conn.execute_batch(&attach)
        .with_context(|| format!("failed to attach {}", src.duckdb_path.display()))?;

    // Done inside a closure so we always DETACH, even on error.
    let result = (|| -> anyhow::Result<u32> {
        // Take table layout from the first source.
        if !*tables_created {
            conn.execute_batch(
                "CREATE TABLE objects AS SELECT * FROM src.objects LIMIT 0;
                 CREATE TABLE bbox AS SELECT * FROM src.bbox LIMIT 0;",
            )?;
            *tables_created = true;
        }

        let max_fid: Option<i64> =
            conn.query_row("SELECT MAX(fid) FROM src.bbox", [], |r| r.get(0))?;
        let max_oid: Option<i64> =
            conn.query_row("SELECT MAX(oid) FROM src.objects", [], |r| r.get(0))?;
        let obj_count: i64 =
            conn.query_row("SELECT COUNT(*) FROM src.objects", [], |r| r.get(0))?;

        // The frame span this source occupies on the merged timeline. Prefer the
        // declared frame count; never let it fall below the observed detections.
        let fid_span = max_fid.map(|m| m + 1).unwrap_or(0);
        let span = (src.num_frames.map(i64::from).unwrap_or(0)).max(fid_span);

        conn.execute(
            &format!(
                "INSERT INTO objects BY NAME
                 SELECT * EXCLUDE (oid), (oid + {offset}) AS oid FROM src.objects",
                offset = *obj_cursor
            ),
            [],
        )?;
        conn.execute(
            &format!(
                "INSERT INTO bbox BY NAME
                 SELECT * EXCLUDE (oid, fid),
                        (oid + {oid_off}) AS oid,
                        (fid + {fid_off}) AS fid
                 FROM src.bbox",
                oid_off = *obj_cursor,
                fid_off = *frame_cursor
            ),
            [],
        )?;

        // Symlink the source video under its merged-frame range.
        if let Some(video) = &src.video_path {
            let start = *frame_cursor as u32;
            let end = (*frame_cursor + span.max(1) - 1) as u32;
            link_video(video, video_dir, start, end)?;
        }

        *frame_cursor += span;
        *obj_cursor += max_oid.map(|m| m + 1).unwrap_or(0);
        Ok(obj_count.max(0) as u32)
    })();

    conn.execute_batch("DETACH src;").ok();
    result
}

/// Create `video_dir/<start>_<end>.<ext>` as a symlink to `video`.
fn link_video(video: &Path, video_dir: &Path, start: u32, end: u32) -> anyhow::Result<()> {
    let target = fs::canonicalize(video).unwrap_or_else(|_| video.to_path_buf());
    let ext = target.extension().and_then(|e| e.to_str()).unwrap_or("mp4");
    let link = video_dir.join(format!("{start}_{end}.{ext}"));
    if link.exists() || fs::symlink_metadata(&link).is_ok() {
        fs::remove_file(&link).ok();
    }
    std::os::unix::fs::symlink(&target, &link).with_context(|| {
        format!(
            "failed to symlink {} -> {}",
            link.display(),
            target.display()
        )
    })?;
    Ok(())
}

/// Fields naming something specific to *one* source, which must never be carried
/// into a merge even if every source happens to agree on the value.
///
/// `world_view` is the motivating case: every nuScenes scene calls its map
/// raster `world-view.png`, so the values agree while pointing at 850 different
/// files. Identity and provenance fields are here for the same reason — a merged
/// dataset is not scene 3, was not captured on one date, and has no single
/// `frame0_epoch`.
const PER_SOURCE_FIELDS: &[&str] = &[
    "video_id", "scene", "clip", "subset", "world", "world_view", "scene_token",
    "map_token", "log", "date_captured", "location", "scene_description",
    "frame0_epoch", "fid_range", "cameras", "event_stats", "team_assignment",
    "dropped", "annotation_size", "frames_without_boxes", "map_px", "view_px",
    "map_extent_m", "no_video",
];

/// Fields this function computes itself, so a source's value must not win.
const RECOMPUTED_FIELDS: &[&str] = &[
    "num_frames", "notes", "merged_from", "row_counts", "obj_classes",
    "activity_classes", "duration_s", "merged_inconsistent",
];

/// Write the merge's `metadata.json`.
///
/// Three kinds of field, handled differently:
///
/// * **Invariant** — carried through only when *every* source has the field and
///   all agree. Unanimity is the safety rule: if two sources disagree on
///   `coordinate_scale` then the merged database has no single coordinate frame,
///   and writing one source's value would be worse than writing none. The
///   disagreeing names are listed under `merged_inconsistent` so their absence
///   is legible rather than mysterious.
/// * **Additive** — recomputed here: `row_counts` summed, class lists unioned,
///   `duration_s` summed, `num_frames` from the merged timeline.
/// * **Per-source** — dropped, per [`PER_SOURCE_FIELDS`].
///
/// Carrying anything at all matters because these fields are *load-bearing*
/// downstream: without `coordinate_scale`/`x_delta`/`y_delta` a query cannot be
/// written in natural units against a merged world dataset, and without `fps` no
/// consumer can convert a frame count to seconds.
fn write_metadata(
    out_dir: &Path,
    sources: &[MergeSource],
    total_frames: u32,
) -> anyhow::Result<()> {
    let names: Vec<String> = sources.iter().map(|s| s.label.clone()).collect();

    // Read what each source recorded. A source with no metadata (or unreadable
    // metadata) is not fatal — the merge still works — but it does mean no field
    // can be unanimous, so nothing is carried.
    let per_source: Vec<Option<serde_json::Map<String, serde_json::Value>>> = sources
        .iter()
        .map(|s| {
            s.metadata_path
                .as_deref()
                .and_then(|p| fs::read_to_string(p).ok())
                .and_then(|raw| serde_json::from_str::<serde_json::Value>(&raw).ok())
                .and_then(|v| match v {
                    serde_json::Value::Object(m) => Some(m),
                    _ => None,
                })
        })
        .collect();
    let all_present: Option<Vec<&serde_json::Map<String, serde_json::Value>>> =
        per_source.iter().map(Option::as_ref).collect();

    let mut meta = serde_json::Map::new();
    let mut inconsistent: Vec<String> = Vec::new();

    if let Some(maps) = &all_present {
        // Every field any source mentions is a candidate; unanimity decides.
        let mut candidates: Vec<&str> = maps
            .iter()
            .flat_map(|m| m.keys().map(String::as_str))
            .filter(|k| !PER_SOURCE_FIELDS.contains(k) && !RECOMPUTED_FIELDS.contains(k))
            .collect();
        candidates.sort_unstable();
        candidates.dedup();

        for key in candidates {
            let first = maps[0].get(key);
            match first {
                Some(v) if maps.iter().all(|m| m.get(key) == first) => {
                    meta.insert(key.to_owned(), v.clone());
                }
                // Present somewhere but not agreed everywhere: say so rather
                // than silently picking one.
                _ => inconsistent.push(key.to_owned()),
            }
        }

        // Additive fields.
        let sum_u64 = |field: &str, sub: &str| -> Option<u64> {
            maps.iter()
                .map(|m| m.get(field)?.get(sub)?.as_u64())
                .try_fold(0u64, |acc, v| v.map(|v| acc + v))
        };
        let mut counts = serde_json::Map::new();
        for sub in ["objects", "bbox", "activities"] {
            if let Some(total) = sum_u64("row_counts", sub) {
                counts.insert(sub.to_owned(), total.into());
            }
        }
        if !counts.is_empty() {
            meta.insert("row_counts".to_owned(), counts.into());
        }
        for field in ["obj_classes", "activity_classes"] {
            let mut union: Vec<String> = maps
                .iter()
                .filter_map(|m| m.get(field)?.as_array())
                .flatten()
                .filter_map(|v| v.as_str().map(str::to_owned))
                .collect();
            union.sort();
            union.dedup();
            if !union.is_empty() || maps.iter().all(|m| m.contains_key(field)) {
                meta.insert(field.to_owned(), union.into());
            }
        }
        if let Some(total) = maps
            .iter()
            .map(|m| m.get("duration_s")?.as_f64())
            .try_fold(0.0f64, |acc, v| v.map(|v| acc + v))
        {
            meta.insert("duration_s".to_owned(), total.into());
        }
    }

    // The merge's own fields always win, so a source's `dataset` or `notes`
    // cannot mislabel the result.
    meta.insert(
        "dataset".to_owned(),
        meta.get("dataset").cloned().unwrap_or_else(|| "merged".into()),
    );
    meta.insert("num_frames".to_owned(), total_frames.into());
    meta.insert(
        "notes".to_owned(),
        format!(
            "Merged from {} datasets: {}",
            sources.len(),
            names.join(", ")
        )
        .into(),
    );
    meta.insert("merged_from".to_owned(), names.into());
    if !inconsistent.is_empty() {
        meta.insert("merged_inconsistent".to_owned(), inconsistent.into());
    }

    let path = out_dir.join("metadata.json");
    fs::write(&path, serde_json::to_vec_pretty(&serde_json::Value::Object(meta))?)
        .with_context(|| format!("failed to write {}", path.display()))?;
    Ok(())
}

fn sql_escape(s: &str) -> String {
    s.replace('\'', "''")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a tiny dataset dir with a duckdb.db and a dummy video, returning
    /// the duckdb path. `oids`/`fids` are 0-based and contiguous.
    fn make_dataset(dir: &Path, n_objs: i64, n_frames: i64) -> PathBuf {
        fs::create_dir_all(dir).unwrap();
        let db = dir.join("duckdb.db");
        let conn = Connection::open(&db).unwrap();
        conn.execute_batch(
            "CREATE TABLE objects (oid INTEGER, cls VARCHAR);
             CREATE TABLE bbox (oid INTEGER, fid INTEGER, x1 DOUBLE, y1 DOUBLE, x2 DOUBLE, y2 DOUBLE);",
        )
        .unwrap();
        for o in 0..n_objs {
            conn.execute(&format!("INSERT INTO objects VALUES ({o}, 'car')"), [])
                .unwrap();
            for f in 0..n_frames {
                conn.execute(
                    &format!("INSERT INTO bbox VALUES ({o}, {f}, 0, 0, 10, 10)"),
                    [],
                )
                .unwrap();
            }
        }
        drop(conn);
        fs::write(dir.join("video.mp4"), b"fake").unwrap();
        db
    }

    /// Write `metadata.json` next to a fixture dataset and return its path.
    fn with_metadata(dir: &Path, json: &str) -> PathBuf {
        let p = dir.join("metadata.json");
        fs::write(&p, json).unwrap();
        p
    }

    #[test]
    fn merge_carries_invariant_metadata_and_flags_disagreement() {
        let root = std::env::temp_dir().join(format!("trql-merge-meta-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let a = make_dataset(&root.join("a"), 2, 3);
        let b = make_dataset(&root.join("b"), 3, 4);

        // Agreed: `fps`, `coordinate_scale`, `dataset`. Disagreed: `width`.
        // Per-source: `world_view` agrees in value but names a different file in
        // each source, so it must be dropped rather than carried.
        // Additive: `row_counts`, `obj_classes`, `duration_s`.
        let ma = with_metadata(
            &root.join("a"),
            r#"{"dataset":"soccertrack","fps":25.0,"width":1000,"coordinate_scale":100.0,
                "world_view":"world-view.png","obj_classes":["ball","playerA"],
                "row_counts":{"objects":2,"bbox":6},"duration_s":1.5,
                "scene":"a","num_frames":3}"#,
        );
        let mb = with_metadata(
            &root.join("b"),
            r#"{"dataset":"soccertrack","fps":25.0,"width":2000,"coordinate_scale":100.0,
                "world_view":"world-view.png","obj_classes":["playerB","ball"],
                "row_counts":{"objects":3,"bbox":12},"duration_s":2.5,
                "scene":"b","num_frames":4}"#,
        );

        let sources = vec![
            MergeSource {
                key: a.clone(),
                label: "a".into(),
                duckdb_path: a.clone(),
                video_path: None,
                num_frames: Some(3),
                metadata_path: Some(ma),
            },
            MergeSource {
                key: b.clone(),
                label: "b".into(),
                duckdb_path: b.clone(),
                video_path: None,
                num_frames: Some(4),
                metadata_path: Some(mb),
            },
        ];
        let out = root.join("out");
        merge_datasets(&sources, &out, &mut |_| {}).unwrap();
        let m: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(out.join("metadata.json")).unwrap()).unwrap();

        // Unanimous invariants carried through.
        assert_eq!(m["fps"], 25.0);
        assert_eq!(m["coordinate_scale"], 100.0);
        assert_eq!(m["dataset"], "soccertrack", "agreed dataset beats \"merged\"");

        // Disagreement is reported, not resolved by picking one.
        assert!(m.get("width").is_none(), "width disagreed, must be absent");
        let inc: Vec<&str> = m["merged_inconsistent"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        assert!(inc.contains(&"width"), "{inc:?}");

        // Per-source fields dropped even when the values agree.
        assert!(m.get("world_view").is_none(), "world_view names a per-source file");
        assert!(m.get("scene").is_none());

        // Additive fields recomputed, not copied.
        assert_eq!(m["row_counts"]["objects"], 5);
        assert_eq!(m["row_counts"]["bbox"], 18);
        assert_eq!(m["duration_s"], 4.0);
        assert_eq!(m["obj_classes"], serde_json::json!(["ball", "playerA", "playerB"]));

        // The merge's own fields always win.
        assert_eq!(m["num_frames"], 7);
        assert_eq!(m["merged_from"], serde_json::json!(["a", "b"]));
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn merge_without_source_metadata_carries_nothing() {
        // A source with no metadata means no field can be unanimous, so the
        // output falls back to exactly what the old minimal writer produced.
        let root = std::env::temp_dir().join(format!("trql-merge-nometa-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let a = make_dataset(&root.join("a"), 1, 2);
        let b = make_dataset(&root.join("b"), 1, 2);
        let ma = with_metadata(&root.join("a"), r#"{"fps":25.0}"#);
        let sources = vec![
            MergeSource {
                key: a.clone(),
                label: "a".into(),
                duckdb_path: a.clone(),
                video_path: None,
                num_frames: Some(2),
                metadata_path: Some(ma),
            },
            MergeSource {
                key: b.clone(),
                label: "b".into(),
                duckdb_path: b.clone(),
                video_path: None,
                num_frames: Some(2),
                metadata_path: None, // no metadata at all
            },
        ];
        let out = root.join("out");
        merge_datasets(&sources, &out, &mut |_| {}).unwrap();
        let m: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(out.join("metadata.json")).unwrap()).unwrap();
        assert!(m.get("fps").is_none(), "one source lacking metadata blocks carry-through");
        assert_eq!(m["dataset"], "merged");
        assert_eq!(m["num_frames"], 4);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn merge_keeps_ids_continuous_and_symlinks_videos() {
        let root = std::env::temp_dir().join(format!("trql-merge-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let a = make_dataset(&root.join("a"), 2, 3); // oids 0..1, fids 0..2
        let b = make_dataset(&root.join("b"), 3, 4); // oids 0..2, fids 0..3

        let sources = vec![
            MergeSource {
                key: a.clone(),
                label: "a".into(),
                duckdb_path: a.clone(),
                video_path: Some(root.join("a/video.mp4")),
                num_frames: Some(3),
                metadata_path: None,
            },
            MergeSource {
                key: b.clone(),
                label: "b".into(),
                duckdb_path: b.clone(),
                video_path: Some(root.join("b/video.mp4")),
                num_frames: Some(4),
                metadata_path: None,
            },
        ];

        let out = root.join("out");
        let mut events = 0;
        let summary = merge_datasets(&sources, &out, &mut |_| events += 1).unwrap();

        assert_eq!(events, 4); // Started+Finished per source
        assert_eq!(summary.num_objs, 5);
        assert_eq!(summary.total_frames, 7); // 3 + 4

        let conn = Connection::open(out.join("duckdb.db")).unwrap();
        let (min_oid, max_oid, n_obj): (i64, i64, i64) = conn
            .query_row(
                "SELECT MIN(oid), MAX(oid), COUNT(*) FROM objects",
                [],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .unwrap();
        // Source a: oids 0..1; source b offset by max_oid_a+1 = 2 -> 2..4.
        assert_eq!((min_oid, max_oid, n_obj), (0, 4, 5));

        let (min_fid, max_fid): (i64, i64) = conn
            .query_row("SELECT MIN(fid), MAX(fid) FROM bbox", [], |r| {
                Ok((r.get(0)?, r.get(1)?))
            })
            .unwrap();
        // a frames 0..2, b frames offset by 3 -> 3..6.
        assert_eq!((min_fid, max_fid), (0, 6));

        // Every bbox row references an existing object after offsetting.
        let orphans: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM bbox b LEFT JOIN objects o USING (oid) WHERE o.oid IS NULL",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(orphans, 0);

        // Videos symlinked under their merged-frame ranges.
        let vids = out.join(VIDEO_DIR);
        assert!(vids.join("0_2.mp4").exists());
        assert!(vids.join("3_6.mp4").exists());
        assert!(
            fs::symlink_metadata(vids.join("0_2.mp4"))
                .unwrap()
                .file_type()
                .is_symlink()
        );

        let _ = fs::remove_dir_all(&root);
    }
}
