//! Builds a `.trql` from a sub-dataset's source `duckdb.db`.
//!
//! Shared by `trql-tui` and `trql-cli`; both go through
//! [`build_trql`], which pairs schema inference with the conversion itself.

use std::path::Path;

use anyhow::{Context as _, anyhow};
use duckdb::{Connection, types::OrderedMap};
use trql_types::{
    obj::{BoundingBox, ExactTraj},
    prop::{Schema, TypeDesc, Value},
};

use crate::{
    datasets::{ConvertProgress, DatasetMetadata, SchemaField},
    db::{ObjWithBBox, TrqsBuilder},
    duckdb_meta,
};

/// Outcome of a successful conversion.
pub struct ConvertSummary {
    pub num_objs: u32,
}

/// Inspects `duckdb_path` to infer its property schema, then converts it into a
/// `.trql` at `out_trql`, reporting [`ConvertProgress`] as it goes.
///
/// This is the whole ingestion path in one call: prefer it over driving
/// [`duckdb_meta::inspect_duckdb`] and [`convert_duckdb_to_trql_with_progress`]
/// separately, unless the caller already holds a
/// [`DuckDbMeta`](crate::datasets::DuckDbMeta) it wants to reuse. `metadata` is
/// the sub-dataset's parsed `metadata.json`, used as a fallback when the source
/// has no property columns.
pub(crate) fn build_trql_with_progress(
    duckdb_path: &Path,
    out_trql: &Path,
    metadata: Option<&DatasetMetadata>,
    max_gap: u32,
    max_error_px: f64,
    on_progress: &mut dyn FnMut(ConvertProgress),
) -> anyhow::Result<ConvertSummary> {
    on_progress(ConvertProgress::Inspecting);
    let meta = duckdb_meta::inspect_duckdb(duckdb_path, metadata)?;
    if meta.schema_fields.is_empty() {
        anyhow::bail!("no convertible object columns");
    }
    convert_duckdb_to_trql_with_progress(
        duckdb_path,
        out_trql,
        &meta.schema_fields,
        max_gap,
        max_error_px,
        on_progress,
    )
}

/// Converts `duckdb_path` into a `.trql` at `out_trql` using an already-known
/// schema, reporting [`ConvertProgress`] as it goes. Creates `out_trql`'s parent
/// directory if needed, and overwrites any existing file at that path.
pub(crate) fn convert_duckdb_to_trql_with_progress(
    duckdb_path: &Path,
    out_trql: &Path,
    fields: &[SchemaField],
    max_gap: u32,
    max_error_px: f64,
    on_progress: &mut dyn FnMut(ConvertProgress),
) -> anyhow::Result<ConvertSummary> {
    if fields.is_empty() {
        anyhow::bail!("at least one schema field is required");
    }
    let conn = duckdb_meta::open_read_only(duckdb_path)?;
    verify_columns(&conn, fields)?;
    let schema = Schema::new(
        fields
            .iter()
            .map(|f| (f.name.clone(), f.type_desc.clone()))
            .collect(),
    );

    on_progress(ConvertProgress::Reading);
    let objects = read_objects(&conn, fields)?;
    let mut builder = TrqsBuilder::new(&schema);
    for (props, traj) in objects {
        builder.ingest(ObjWithBBox {
            props,
            exact_traj: traj,
        })?;
    }
    if let Some(parent) = out_trql.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let db = builder
        .build_ext_with_progress(max_gap, max_error_px, |done, total| {
            on_progress(ConvertProgress::Fitting { done, total })
        })
        .0;

    on_progress(ConvertProgress::Saving);
    db.save(&out_trql.to_path_buf())?;
    Ok(ConvertSummary {
        num_objs: db.num_objs(),
    })
}

fn verify_columns(conn: &Connection, fields: &[SchemaField]) -> anyhow::Result<()> {
    for table in ["objects", "bbox"] {
        let exists: bool = conn.query_row(
            "SELECT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_schema = 'main' AND table_name = ?
             )",
            [table],
            |r| r.get(0),
        )?;
        if !exists {
            anyhow::bail!("required table `{table}` not found");
        }
    }

    for field in fields {
        let exists: bool = conn.query_row(
            "SELECT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_schema = 'main' AND table_name = 'objects' AND column_name = ?
             )",
            [&field.column],
            |r| r.get(0),
        )?;
        if !exists {
            anyhow::bail!("objects column `{}` not found", field.column);
        }
    }
    Ok(())
}

fn read_objects(
    conn: &Connection,
    fields: &[SchemaField],
) -> anyhow::Result<Vec<(Vec<Value>, ExactTraj)>> {
    let prop_cols = fields
        .iter()
        .map(|f| format!("o.\"{}\"", f.column))
        .collect::<Vec<_>>()
        .join(", ");
    let group_cols = if prop_cols.is_empty() {
        "b.oid".to_owned()
    } else {
        format!("b.oid, {prop_cols}")
    };
    let select_props = if prop_cols.is_empty() {
        String::new()
    } else {
        format!(", {prop_cols}")
    };
    let query = format!(
        "SELECT b.oid{select_props},
         array_agg({{'fid': b.fid, 'x1': b.x1, 'y1': b.y1, 'x2': b.x2, 'y2': b.y2}}
                   ORDER BY b.fid) AS frames
         FROM bbox b
         JOIN objects o ON b.oid = o.oid
         GROUP BY {group_cols}
         ORDER BY b.oid"
    );

    let mut stmt = conn.prepare(&query)?;
    let mut rows = stmt.query([])?;
    let mut objects = Vec::new();
    let n_props = fields.len();
    while let Some(row) = rows.next()? {
        let oid: i64 = row.get(0)?;
        let mut props = Vec::with_capacity(n_props);
        for (i, field) in fields.iter().enumerate() {
            props.push(read_prop(row, 1 + i, field)?);
        }
        let frames_val: duckdb::types::Value = row.get(1 + n_props)?;
        let frame_list = match frames_val {
            duckdb::types::Value::List(items) => items,
            other => anyhow::bail!("oid={oid}: expected List for frames, got {other:?}"),
        };
        let mut frames = Vec::with_capacity(frame_list.len());
        let mut bboxes = Vec::with_capacity(frame_list.len());
        let mut last_fid = None;
        for item in frame_list {
            let fmap = match item {
                duckdb::types::Value::Struct(m) => m,
                other => anyhow::bail!("oid={oid}: expected Struct in frames, got {other:?}"),
            };
            let fid = get_u32(&fmap, "fid", oid)?;
            if last_fid == Some(fid) {
                continue;
            }
            last_fid = Some(fid);
            frames.push(fid);
            bboxes.push(BoundingBox {
                x1: get_f64(&fmap, "x1", oid)? as u32,
                y1: get_f64(&fmap, "y1", oid)? as u32,
                x2: get_f64(&fmap, "x2", oid)? as u32,
                y2: get_f64(&fmap, "y2", oid)? as u32,
            });
        }
        objects.push((props, ExactTraj { frames, bboxes }));
    }
    Ok(objects)
}

fn read_prop(row: &duckdb::Row<'_>, idx: usize, field: &SchemaField) -> anyhow::Result<Value> {
    Ok(match &field.type_desc {
        TypeDesc::String => Value::String(row.get::<_, String>(idx)?),
        TypeDesc::UInt => {
            let v: i64 = row.get(idx)?;
            Value::UInt(v.try_into().with_context(|| {
                format!(
                    "negative value cannot be converted to uint for `{}`",
                    field.column
                )
            })?)
        }
        TypeDesc::Int => {
            let v: i64 = row.get(idx)?;
            Value::Int(
                v.try_into()
                    .with_context(|| format!("value out of i32 range for `{}`", field.column))?,
            )
        }
        TypeDesc::Bool => Value::Bool(row.get::<_, bool>(idx)?),
        TypeDesc::Enum(variants) => {
            let label: String = row.get(idx)?;
            let variant_idx = variants
                .iter()
                .position(|v| v == &label)
                .ok_or_else(|| anyhow!("value {:?} not in enum variants {:?}", label, variants))?;
            Value::Enum(variant_idx)
        }
    })
}

fn get_u32(
    fmap: &OrderedMap<String, duckdb::types::Value>,
    key: &str,
    oid: i64,
) -> anyhow::Result<u32> {
    match fmap.get(&key.to_owned()) {
        Some(duckdb::types::Value::Int(v)) => Ok(*v as u32),
        Some(duckdb::types::Value::BigInt(v)) => Ok(*v as u32),
        Some(duckdb::types::Value::UInt(v)) => Ok(*v),
        Some(duckdb::types::Value::UBigInt(v)) => Ok((*v).try_into()?),
        other => Err(anyhow!("oid={oid}: unexpected type for {key}: {other:?}")),
    }
}

fn get_f64(
    fmap: &OrderedMap<String, duckdb::types::Value>,
    key: &str,
    oid: i64,
) -> anyhow::Result<f64> {
    match fmap.get(&key.to_owned()) {
        Some(duckdb::types::Value::Double(v)) => Ok(*v),
        Some(duckdb::types::Value::Float(v)) => Ok(*v as f64),
        Some(duckdb::types::Value::Int(v)) => Ok(*v as f64),
        Some(duckdb::types::Value::BigInt(v)) => Ok(*v as f64),
        // Unsigned columns: a dataset whose coordinates are already stored as
        // non-negative scaled integers (i24_motion) uses UINTEGER, so that the
        // `as u32` below is exact rather than a truncating, saturating cast.
        Some(duckdb::types::Value::UInt(v)) => Ok(*v as f64),
        Some(duckdb::types::Value::UBigInt(v)) => Ok(*v as f64),
        other => Err(anyhow!("oid={oid}: unexpected type for {key}: {other:?}")),
    }
}
