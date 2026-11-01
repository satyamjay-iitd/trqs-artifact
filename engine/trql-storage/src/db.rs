use std::{
    collections::HashMap,
    fs::{self, File},
    io::{Read as _, Write as _},
    path::{Path, PathBuf},
};

use anyhow::anyhow;
use bincode::{Decode, Encode, config};
use trql_poly::PolyTraj;
use trql_types::{
    obj::ExactTraj,
    prop::{Schema, TypeDesc, Value},
};

use crate::{
    prop::{AttrPred, IntPred, Prop, UIntPred, validate_props},
    stats::TrqsStats,
};

/// An ingestion-time object: property values paired with its exact trajectory.
#[derive(Clone)]
pub struct ObjWithBBox {
    pub props: Vec<Value>,
    pub exact_traj: ExactTraj,
}

/// Maps logical frame IDs to image file paths for visualization.
///
/// Persisted as a JSON sidecar alongside the `.trql` file. If the database is
/// saved to `"foo/bar.trql"`, the sidecar is `"foo/bar.frames.json"`.
/// Returns an empty index (not an error) when the sidecar is absent.
pub struct FramePathIndex(pub HashMap<u32, String>);

impl FramePathIndex {
    pub fn new() -> Self {
        Self(HashMap::new())
    }

    /// Returns the image file path for `frame_id`, or `None` if not registered.
    pub fn get(&self, frame_id: u32) -> Option<&str> {
        self.0.get(&frame_id).map(String::as_str)
    }

    /// Saves the index as a JSON sidecar next to `trql_path`.
    pub fn save(&self, trql_path: &Path) -> anyhow::Result<()> {
        let sidecar = trql_path.with_extension("frames.json");
        let json = serde_json::to_string_pretty(&self.0)?;
        fs::write(sidecar, json)?;
        Ok(())
    }

    /// Loads the sidecar from alongside `trql_path`.
    ///
    /// Returns an empty index if the sidecar does not exist.
    pub fn load(trql_path: &Path) -> anyhow::Result<Self> {
        let sidecar = trql_path.with_extension("frames.json");
        if !sidecar.exists() {
            return Ok(Self::new());
        }
        let json = fs::read_to_string(&sidecar)?;
        let map: HashMap<u32, String> = serde_json::from_str(&json)?;
        Ok(Self(map))
    }
}

/// Accumulates objects and builds a [`Trqs`] with polynomial trajectories and indexes.
#[derive(Clone)]
pub struct TrqsBuilder {
    schema: Schema,
    prop_idx_to_prop: Vec<Prop>,
    pub exact_trajs: Vec<ExactTraj>,
    enable_query_out: bool,
    frame_paths: HashMap<u32, String>,
}

/// Columnar database with polynomial trajectories, property indexes, and frame-to-object mapping.
///
/// Persisted as a bincode-encoded `.trql` file with magic bytes and versioning.
#[derive(Clone, Encode, Decode)]
pub struct Trqs {
    schema: Schema,
    pub prop_idx_to_prop: Vec<Prop>,
    poly_trajs: Vec<PolyTraj>,
    poly_gap_threshold: u32,
    enable_query_out: bool,
    stats: TrqsStats,
}

const FILE_MAGIC: &[u8; 8] = b"TRQLDB\0\0";
/// Versions 1–3 were the older `VZALGDB` format, which this magic rejects.
const FILE_VERSION: u32 = 4;

/// `magic (8) | version (4, LE) | stats_len (4, LE)`, followed by the stats
/// block and then the database payload. The header is fixed size so
/// [`Trqs::read_stats`] can pull the stats out with two small reads instead of
/// loading the whole file.
const HEADER_LEN: usize = FILE_MAGIC.len() + 4 + 4;

/// Stats are encoded with fixed-width integers, so the block is the same length
/// for every database regardless of the values it holds.
fn stats_config() -> impl config::Config {
    config::standard().with_fixed_int_encoding()
}

#[derive(Encode, Decode)]
struct FileEnvelope {
    version: u32,
    db: Trqs,
}

impl TrqsBuilder {
    /// Creates a new builder for the given schema.
    pub fn new(schema: &Schema) -> Self {
        Self {
            schema: schema.clone(),
            prop_idx_to_prop: schema
                .col_iter()
                .map(|(name, type_)| Prop::empty(name, type_.clone()))
                .collect(),
            exact_trajs: Vec::new(),
            enable_query_out: true,
            frame_paths: HashMap::new(),
        }
    }

    /// Registers the image file path for a frame ID.
    ///
    /// Can be called any time before [`build_ext`](Self::build_ext). Frame IDs with
    /// no registered path are silently absent from the resulting [`FramePathIndex`].
    pub fn register_frame_path(&mut self, frame_id: u32, path: impl Into<String>) {
        self.frame_paths.insert(frame_id, path.into());
    }

    /// Adds an object, validating its properties against the schema.
    pub fn ingest(&mut self, obj: ObjWithBBox) -> anyhow::Result<()> {
        validate_props(&self.schema, &obj.props)?;

        for (idx, prop) in self.prop_idx_to_prop.iter_mut().enumerate() {
            let val = obj
                .props
                .get(idx)
                .ok_or_else(|| anyhow!("missing property `{}` in object", idx))?;
            prop.append(val.clone());
        }

        self.exact_trajs.push(obj.exact_traj);
        Ok(())
    }

    /// Finalizes the builder, returning a [`Trqs`] and a [`FramePathIndex`].
    ///
    /// The `FramePathIndex` holds the frame ID → image path mappings registered via
    /// [`register_frame_path`](Self::register_frame_path). Save it alongside the
    /// database with [`FramePathIndex::save`].
    pub fn build_ext(self, gap_threshold: u32, max_error_px: f64) -> (Trqs, FramePathIndex) {
        self.build_ext_with_progress(gap_threshold, max_error_px, |_, _| {})
    }

    /// Like [`build_ext`](Self::build_ext), but reports fitting progress.
    ///
    /// `on_fit` is called after each trajectory is fitted, with
    /// `(completed, total)`. Polynomial fitting dominates the cost of a build,
    /// so this is the phase worth showing a progress bar for.
    pub fn build_ext_with_progress(
        mut self,
        gap_threshold: u32,
        max_error_px: f64,
        mut on_fit: impl FnMut(usize, usize),
    ) -> (Trqs, FramePathIndex) {
        let frame_paths = FramePathIndex(self.frame_paths);

        for prop in &mut self.prop_idx_to_prop {
            prop.build_index();
        }

        let total = self.exact_trajs.len();
        let poly_trajs: Vec<PolyTraj> = self
            .exact_trajs
            .iter()
            .enumerate()
            .map(|(i, traj)| {
                let poly =
                    PolyTraj::from_traj_with_bboxes_independent(traj, gap_threshold, max_error_px);
                on_fit(i + 1, total);
                poly
            })
            .collect();

        let stats = TrqsStats::from_trajs(&poly_trajs);

        let db = Trqs {
            schema: self.schema,
            prop_idx_to_prop: self.prop_idx_to_prop,
            poly_trajs,
            poly_gap_threshold: gap_threshold,
            enable_query_out: self.enable_query_out,
            stats,
        };

        (db, frame_paths)
    }

    /// Finalizes the builder and returns a [`Trqs`], discarding any registered frame paths.
    ///
    /// Use [`build_ext`](Self::build_ext) instead if you need the [`FramePathIndex`].
    pub fn build(self, gap_threshold: u32, max_error_px: f64) -> Trqs {
        self.build_ext(gap_threshold, max_error_px).0
    }
}

/// Validates the magic bytes and version, returning the stats block length and
/// the bytes following the header.
fn parse_header(bytes: &[u8]) -> anyhow::Result<(usize, &[u8])> {
    if bytes.len() < HEADER_LEN {
        anyhow::bail!("file too short to be a trql database");
    }
    let (magic, rest) = bytes.split_at(FILE_MAGIC.len());
    if magic != FILE_MAGIC {
        anyhow::bail!("invalid trql file magic");
    }

    let version = u32::from_le_bytes(rest[..4].try_into().unwrap());
    if version != FILE_VERSION {
        anyhow::bail!(
            "unsupported trql file version {}, expected {}",
            version,
            FILE_VERSION
        );
    }
    let stats_len = u32::from_le_bytes(rest[4..8].try_into().unwrap()) as usize;
    Ok((stats_len, &rest[8..]))
}

impl Trqs {
    /// Loads a database from a `.trql` binary file, checking magic bytes and version.
    pub fn from_bin(path: &PathBuf) -> anyhow::Result<Self> {
        let bytes = fs::read(path)?;
        Self::from_bytes(&bytes)
    }

    /// Parses a database from raw `.trql` bytes (header + stats block + bincode payload).
    pub fn from_bytes(bytes: &[u8]) -> anyhow::Result<Self> {
        let (stats_len, rest) = parse_header(bytes)?;
        if rest.len() < stats_len {
            anyhow::bail!("trql database truncated inside the stats block");
        }
        let payload = &rest[stats_len..];

        let (env, _): (FileEnvelope, usize) =
            bincode::decode_from_slice(payload, config::standard())?;
        if env.version != FILE_VERSION {
            anyhow::bail!(
                "unsupported trql file version {}, expected {}",
                env.version,
                FILE_VERSION
            );
        }
        Ok(env.db)
    }

    /// Reads just the [`TrqsStats`] of a `.trql` file, without decoding the
    /// database itself.
    ///
    /// Only the fixed-size header and the stats block that follows it are read
    /// off disk, so the cost is independent of the database's size.
    pub fn read_stats(path: &Path) -> anyhow::Result<TrqsStats> {
        let mut file = File::open(path)?;

        let mut header = [0u8; HEADER_LEN];
        file.read_exact(&mut header)
            .map_err(|_| anyhow!("file too short to be a trql database"))?;
        let (stats_len, _) = parse_header(&header)?;

        let mut stats_bytes = vec![0u8; stats_len];
        file.read_exact(&mut stats_bytes)
            .map_err(|_| anyhow!("trql database truncated inside the stats block"))?;
        let (stats, _): (TrqsStats, usize) =
            bincode::decode_from_slice(&stats_bytes, stats_config())?;
        Ok(stats)
    }

    /// Persists the database to a `.trql` binary file.
    pub fn save(&self, path: &PathBuf) -> anyhow::Result<()> {
        let stats = bincode::encode_to_vec(&self.stats, stats_config())?;
        let payload = bincode::encode_to_vec(
            FileEnvelope {
                version: FILE_VERSION,
                db: self.clone(),
            },
            config::standard(),
        )?;

        let mut file = File::create(path)?;
        file.write_all(FILE_MAGIC)?;
        file.write_all(&FILE_VERSION.to_le_bytes())?;
        file.write_all(&(stats.len() as u32).to_le_bytes())?;
        file.write_all(&stats)?;
        file.write_all(&payload)?;
        file.flush()?;
        Ok(())
    }

    /// Returns the trajectory statistics computed when the database was built.
    pub fn stats(&self) -> &TrqsStats {
        &self.stats
    }

    /// Returns the total number of objects.
    pub fn num_objs(&self) -> u32 {
        self.stats.num_objs as u32
    }

    /// Returns the type descriptor for the named property column.
    pub fn prop_type(&self, col: &str) -> Option<&TypeDesc> {
        Some(self.prop_idx_to_prop[self.schema.get_col_idx(col)?].type_())
    }

    /// Returns the polynomial trajectories for all objects.
    pub fn poly_trajs(&self) -> &[PolyTraj] {
        &self.poly_trajs
    }

    /// Returns the gap threshold used during polynomial fitting.
    pub fn poly_gap_threshold(&self) -> u32 {
        self.poly_gap_threshold
    }

    // /// Returns the frame → object IDs mapping.
    // pub fn frame_to_objs(&self) -> &BTreeMap<u32, Vec<u32>> {
    //     &self.frame_to_objs
    // }

    /// Returns the property schema.
    pub fn schema(&self) -> &Schema {
        &self.schema
    }

    /// Performs an equality index scan on the named property, returning matching object IDs.
    pub fn index_scan(&self, prop_name: &str, prop_val: &Value) -> anyhow::Result<&[u32]> {
        let prop_idx = self
            .prop_idx_to_prop
            .iter()
            .position(|p| p.name() == prop_name)
            .ok_or_else(|| anyhow!("Property {prop_name} not found"))?;
        let pred = match prop_val {
            Value::UInt(v) => AttrPred::UIntPred(UIntPred::Eq(*v)),
            Value::Int(v) => AttrPred::IntPred(IntPred::Eq(*v)),
            Value::Bool(v) => AttrPred::BoolPred(*v),
            Value::Enum(v) => AttrPred::EnumPred(*v),
            Value::String(s) => AttrPred::StringPred(s.clone()),
        };
        self.prop_idx_to_prop[prop_idx].index_scan(pred)
    }

    /// A new database tiling this one `times` times end-to-end, to grow it for
    /// benchmarking.
    ///
    /// Replica `i` has its trajectories shifted forward by `i * (max_frame + 1)`
    /// frames, so replicas never overlap in time; each object's property values
    /// are duplicated per replica. This mirrors the evaluator's DuckDB
    /// replication but operates on the already-fitted polynomials, so nothing is
    /// re-fitted. `times <= 1` returns an unchanged clone.
    pub fn repeat(&self, times: u32) -> Trqs {
        if times <= 1 {
            return self.clone();
        }

        // Stride = one past the last covered frame, matching the evaluator's
        // `COALESCE(MAX(fid), -1) + 1`.
        let stride = self
            .poly_trajs
            .iter()
            .filter_map(PolyTraj::end_frame)
            .max()
            .map_or(0, |m| m + 1);

        let mut prop_idx_to_prop: Vec<Prop> = self
            .schema
            .col_iter()
            .map(|(name, ty)| Prop::empty(name, ty.clone()))
            .collect();
        let mut poly_trajs = Vec::with_capacity(self.poly_trajs.len() * times as usize);

        for i in 0..times {
            let offset = i * stride;
            for (oid, traj) in self.poly_trajs.iter().enumerate() {
                for (col, prop) in prop_idx_to_prop.iter_mut().enumerate() {
                    prop.append(self.prop_idx_to_prop[col].get(oid as u32).clone());
                }
                poly_trajs.push(traj.shifted(offset));
            }
        }

        for prop in &mut prop_idx_to_prop {
            prop.build_index();
        }

        let stats = TrqsStats::from_trajs(&poly_trajs);
        Trqs {
            schema: self.schema.clone(),
            prop_idx_to_prop,
            poly_trajs,
            poly_gap_threshold: self.poly_gap_threshold,
            enable_query_out: self.enable_query_out,
            stats,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use trql_types::obj::BoundingBox;

    fn test_schema() -> Schema {
        Schema::new(vec![
            ("speed".into(), TypeDesc::UInt),
            (
                "color".into(),
                TypeDesc::Enum(vec!["red".into(), "blue".into()]),
            ),
        ])
    }

    fn make_obj(speed: u32, color: usize, frames: Vec<u32>) -> ObjWithBBox {
        let bboxes = frames
            .iter()
            .map(|&f| BoundingBox {
                x1: f * 10,
                y1: 0,
                x2: f * 10 + 10,
                y2: 10,
            })
            .collect();
        ObjWithBBox {
            props: vec![Value::UInt(speed), Value::Enum(color)],
            exact_traj: ExactTraj { frames, bboxes },
        }
    }

    fn build_test_db() -> Trqs {
        let schema = test_schema();
        let mut builder = TrqsBuilder::new(&schema);
        builder
            .ingest(make_obj(50, 0, vec![0, 1, 2, 3, 4]))
            .unwrap(); // obj 0: red, speed 50
        builder
            .ingest(make_obj(80, 1, vec![2, 3, 4, 5, 6]))
            .unwrap(); // obj 1: blue, speed 80
        builder.ingest(make_obj(50, 0, vec![10, 11, 12])).unwrap(); // obj 2: red, speed 50
        builder.build(5, 0.0)
    }

    // --- TrqsBuilder ---

    #[test]
    fn builder_creates_db_with_correct_count() {
        let db = build_test_db();
        assert_eq!(db.num_objs(), 3);
    }

    #[test]
    fn repeat_tiles_objects_and_shifts_frames() {
        let db = build_test_db(); // 3 objects, last covered frame = 12
        let repeated = db.repeat(3);

        // Object count and trajectory count triple.
        assert_eq!(repeated.num_objs(), 9);
        assert_eq!(repeated.poly_trajs().len(), 9);

        // Replica i is shifted by i * (max_frame + 1) = i * 13. Object 0 covers
        // frames 0..=4 in the original; its copies land at 13.. and 26...
        let stride = 13;
        let seg0 = repeated.poly_trajs()[0].segments()[0].start;
        let seg0_r1 = repeated.poly_trajs()[3].segments()[0].start;
        let seg0_r2 = repeated.poly_trajs()[6].segments()[0].start;
        assert_eq!(seg0_r1, seg0 + stride);
        assert_eq!(seg0_r2, seg0 + 2 * stride);

        // Property values are duplicated per replica (object 0 is red/speed 50).
        assert_eq!(
            repeated
                .index_scan("speed", &Value::UInt(50))
                .unwrap()
                .len(),
            6 // 2 speed-50 objects × 3 replicas
        );
    }

    #[test]
    fn repeat_once_is_unchanged() {
        let db = build_test_db();
        assert_eq!(db.repeat(1).num_objs(), db.num_objs());
    }

    #[test]
    fn build_reports_fitting_progress_once_per_trajectory() {
        let schema = test_schema();
        let mut builder = TrqsBuilder::new(&schema);
        builder.ingest(make_obj(50, 0, vec![0, 1, 2])).unwrap();
        builder.ingest(make_obj(80, 1, vec![2, 3, 4])).unwrap();
        builder.ingest(make_obj(50, 0, vec![5, 6, 7])).unwrap();

        let mut seen = Vec::new();
        let db = builder
            .build_ext_with_progress(5, 0.0, |done, total| seen.push((done, total)))
            .0;

        // One callback per trajectory, counting up to the total.
        assert_eq!(seen, vec![(1, 3), (2, 3), (3, 3)]);
        assert_eq!(db.num_objs(), 3);
    }

    #[test]
    fn builder_rejects_wrong_arity() {
        let schema = test_schema();
        let mut builder = TrqsBuilder::new(&schema);
        let obj = ObjWithBBox {
            props: vec![Value::UInt(10)], // missing second prop
            exact_traj: ExactTraj {
                frames: vec![0],
                bboxes: vec![BoundingBox::empty()],
            },
        };
        assert!(builder.ingest(obj).is_err());
    }

    #[test]
    fn builder_rejects_wrong_type() {
        let schema = test_schema();
        let mut builder = TrqsBuilder::new(&schema);
        let obj = ObjWithBBox {
            props: vec![Value::Bool(true), Value::Enum(0)], // first should be UInt
            exact_traj: ExactTraj {
                frames: vec![0],
                bboxes: vec![BoundingBox::empty()],
            },
        };
        assert!(builder.ingest(obj).is_err());
    }

    // --- Trqs accessors ---

    #[test]
    fn db_schema() {
        let db = build_test_db();
        assert_eq!(db.schema().num_cols(), 2);
        assert_eq!(db.schema().get_type("speed"), Some(&TypeDesc::UInt));
    }

    // #[test]
    // fn db_frame_range() {
    //     let db = build_test_db();
    //     let (min, max) = db.frame_range().unwrap();
    //     assert_eq!(min, 0);
    //     assert_eq!(max, 12);
    // }

    #[test]
    fn db_poly_trajs_count() {
        let db = build_test_db();
        assert_eq!(db.poly_trajs().len(), 3);
    }

    #[test]
    fn db_prop_type() {
        let db = build_test_db();
        assert_eq!(db.prop_type("speed"), Some(&TypeDesc::UInt));
        assert_eq!(db.prop_type("nonexistent"), None);
    }

    // --- index_scan ---

    #[test]
    fn index_scan_uint_eq() {
        let db = build_test_db();
        let ids = db.index_scan("speed", &Value::UInt(50)).unwrap();
        let mut sorted: Vec<u32> = ids.to_vec();
        sorted.sort();
        assert_eq!(sorted, vec![0, 2]);
    }

    #[test]
    fn index_scan_enum_eq() {
        let db = build_test_db();
        let ids = db.index_scan("color", &Value::Enum(1)).unwrap(); // blue
        assert_eq!(ids, &[1]);
    }

    #[test]
    fn index_scan_string_eq() {
        let schema = Schema::new(vec![("name".into(), TypeDesc::String)]);
        let mut builder = TrqsBuilder::new(&schema);
        for (name, frames) in [
            ("alice", vec![0, 1, 2]),
            ("bob", vec![1, 2, 3]),
            ("alice", vec![5, 6]),
        ] {
            let bboxes = frames
                .iter()
                .map(|&f| BoundingBox {
                    x1: f * 10,
                    y1: 0,
                    x2: f * 10 + 10,
                    y2: 10,
                })
                .collect();
            builder
                .ingest(ObjWithBBox {
                    props: vec![Value::String(name.into())],
                    exact_traj: ExactTraj { frames, bboxes },
                })
                .unwrap();
        }
        let db = builder.build(5, 0.0);

        let ids = db
            .index_scan("name", &Value::String("alice".into()))
            .unwrap();
        let mut sorted: Vec<u32> = ids.to_vec();
        sorted.sort();
        assert_eq!(sorted, vec![0, 2]);

        let bob_ids = db.index_scan("name", &Value::String("bob".into())).unwrap();
        assert_eq!(bob_ids, &[1]);

        let none = db
            .index_scan("name", &Value::String("carol".into()))
            .unwrap();
        assert!(none.is_empty());
    }

    #[test]
    fn index_scan_missing_prop() {
        let db = build_test_db();
        assert!(db.index_scan("nonexistent", &Value::UInt(0)).is_err());
    }

    // --- frame_to_objs ---

    // #[test]
    // fn frame_to_objs_overlap() {
    //     let db = build_test_db();
    //     let ftoo = db.frame_to_objs();
    //     // Frame 3 should contain objects 0 and 1
    //     let objs_at_3 = &ftoo[&3];
    //     assert!(objs_at_3.contains(&0));
    //     assert!(objs_at_3.contains(&1));
    //     // Frame 11 should contain only object 2
    //     let objs_at_11 = &ftoo[&11];
    //     assert_eq!(objs_at_11, &[2]);
    // }

    // --- repeat_db ---

    // #[test]
    // fn repeat_db_doubles_objects() {
    //     let db = build_test_db();
    //     let repeated = repeat_db(&db, 2);
    //     assert_eq!(repeated.num_objs(), 6);
    //     assert_eq!(repeated.poly_trajs().len(), 6);
    // }

    // // --- save / from_bin round-trip ---

    // #[test]
    // fn save_and_load_roundtrip() {
    //     let db = build_test_db();
    //     let dir = std::env::temp_dir();
    //     let path = dir.join("test_trql_roundtrip.trql");

    //     db.save(&path).unwrap();
    //     let loaded = Trqs::from_bin(&path).unwrap();

    //     assert_eq!(loaded.num_objs(), db.num_objs());
    //     assert_eq!(loaded.frame_range(), db.frame_range());
    //     assert_eq!(loaded.poly_trajs().len(), db.poly_trajs().len());

    //     std::fs::remove_file(&path).ok();
    // }

    // --- stats header ---

    fn save_temp(db: &Trqs, name: &str) -> PathBuf {
        let path = std::env::temp_dir().join(name);
        db.save(&path).unwrap();
        path
    }

    #[test]
    fn read_stats_matches_full_load() {
        let db = build_test_db();
        let path = save_temp(&db, "test_trql_read_stats.trql");

        let stats = Trqs::read_stats(&path).unwrap();
        let full = Trqs::from_bin(&path).unwrap();
        fs::remove_file(&path).ok();

        assert_eq!(stats.num_objs, full.stats().num_objs);
        assert_eq!(stats.total_segments, full.stats().total_segments);
        assert_eq!(stats.total_canonical, full.stats().total_canonical);
        assert_eq!(stats.avg_traj_len, full.stats().avg_traj_len);
        assert_eq!(stats.avg_span, full.stats().avg_span);
        assert_eq!(stats.avg_segments, full.stats().avg_segments);
        assert_eq!(stats.poly_segments, full.stats().poly_segments);
        assert_eq!(stats.direct_segments, full.stats().direct_segments);
        assert_eq!(stats.mixed_segments, full.stats().mixed_segments);
    }

    /// The stats block sits at a fixed offset and has a fixed length, so it can
    /// be read without knowing anything about the database behind it.
    #[test]
    fn stats_block_is_fixed_size() {
        let small = save_temp(&build_test_db(), "test_trql_stats_small.trql");

        let schema = test_schema();
        let mut builder = TrqsBuilder::new(&schema);
        for i in 0..200u32 {
            builder
                .ingest(make_obj(
                    i,
                    (i % 2) as usize,
                    (0..50).map(|f| i * 50 + f).collect(),
                ))
                .unwrap();
        }
        let big = save_temp(&builder.build(5, 0.0), "test_trql_stats_big.trql");

        let small_bytes = fs::read(&small).unwrap();
        let big_bytes = fs::read(&big).unwrap();
        fs::remove_file(&small).ok();
        fs::remove_file(&big).ok();

        let stats_len =
            |bytes: &[u8]| u32::from_le_bytes(bytes[12..HEADER_LEN].try_into().unwrap());
        let small_len = stats_len(&small_bytes);
        let big_len = stats_len(&big_bytes);
        assert_eq!(small_len, big_len);
        // Reading the stats never touches more than the header + block.
        assert!(big_bytes.len() > HEADER_LEN + big_len as usize);
    }

    #[test]
    fn read_stats_rejects_bad_magic() {
        let path = std::env::temp_dir().join("test_trql_bad_magic.trql");
        fs::write(&path, b"NOTATRQL\0____________").unwrap();
        let err = Trqs::read_stats(&path).unwrap_err();
        fs::remove_file(&path).ok();
        assert!(err.to_string().contains("magic"), "{err}");
    }

    #[test]
    fn read_stats_rejects_short_file() {
        let path = std::env::temp_dir().join("test_trql_short.trql");
        fs::write(&path, FILE_MAGIC).unwrap();
        let err = Trqs::read_stats(&path).unwrap_err();
        fs::remove_file(&path).ok();
        assert!(err.to_string().contains("too short"), "{err}");
    }
}
