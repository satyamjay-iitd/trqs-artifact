//! Reads the shape of a sub-dataset's source `duckdb.db`: row counts, column
//! types, and the TrQL schema inferred from them.
//!
//! The result types live in [`crate::datasets`] alongside the dataset tree they
//! describe; this module only fills them in from a DuckDB connection.

use std::path::Path;

use anyhow::Context as _;
use duckdb::{AccessMode, Config, Connection};
use trql_types::prop::TypeDesc;

use crate::datasets::{DatasetMetadata, DuckDbMeta, SchemaField, TableColumn};

/// Opens `path` read-only, so inspection can never mutate a source database.
pub fn open_read_only(path: &Path) -> anyhow::Result<Connection> {
    let cfg = Config::default().access_mode(AccessMode::ReadOnly)?;
    Connection::open_with_flags(path, cfg)
        .with_context(|| format!("failed to open DuckDB read-only: {}", path.display()))
}

/// Opens `path` for writing.
///
/// Separate from [`open_read_only`] and deliberately rarer: almost everything
/// here inspects a source database and must not be able to change it. Writing is
/// for the few steps that own their output — a predicted database being relabelled
/// after detection, not a ground-truth one.
pub fn open_read_write(path: &Path) -> anyhow::Result<Connection> {
    Connection::open(path)
        .with_context(|| format!("failed to open DuckDB for writing: {}", path.display()))
}

/// Inspects a sub-dataset's `duckdb.db`, inferring its property schema.
///
/// `metadata` is the sub-dataset's parsed `metadata.json`, used only as a
/// fallback source of object classes when the `objects` table carries no
/// property columns of its own.
pub fn inspect_duckdb(
    path: &Path,
    metadata: Option<&DatasetMetadata>,
) -> anyhow::Result<DuckDbMeta> {
    let conn = open_read_only(path)?;
    verify_required_tables(&conn)?;
    let object_columns = describe_table(&conn, "objects")?;
    let bbox_columns = describe_table(&conn, "bbox")?;
    let object_count = count_rows(&conn, "objects")?;
    let bbox_count = count_rows(&conn, "bbox")?;
    let (frame_min, frame_max) = frame_range(&conn)?;
    let schema_fields = infer_schema_fields(&conn, &object_columns, metadata)?;

    Ok(DuckDbMeta {
        object_count,
        bbox_count,
        frame_min,
        frame_max,
        object_columns,
        bbox_columns,
        schema_fields,
    })
}

fn verify_required_tables(conn: &Connection) -> anyhow::Result<()> {
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
    Ok(())
}

fn describe_table(conn: &Connection, table: &str) -> anyhow::Result<Vec<TableColumn>> {
    let sql = "SELECT column_name, data_type, is_nullable
               FROM information_schema.columns
               WHERE table_schema = 'main' AND table_name = ?
               ORDER BY ordinal_position";
    let mut stmt = conn.prepare(sql)?;
    let mut rows = stmt.query([table])?;
    let mut cols = Vec::new();
    while let Some(row) = rows.next()? {
        let nullable: String = row.get(2)?;
        cols.push(TableColumn {
            name: row.get(0)?,
            ty: row.get(1)?,
            nullable: nullable.eq_ignore_ascii_case("YES"),
        });
    }
    Ok(cols)
}

fn count_rows(conn: &Connection, table: &str) -> anyhow::Result<u64> {
    let sql = format!("SELECT COUNT(*) FROM \"{table}\"");
    let count: i64 = conn.query_row(&sql, [], |r| r.get(0))?;
    Ok(count.max(0) as u64)
}

fn frame_range(conn: &Connection) -> anyhow::Result<(Option<u32>, Option<u32>)> {
    let (min_fid, max_fid): (Option<i64>, Option<i64>) =
        conn.query_row("SELECT MIN(fid), MAX(fid) FROM bbox", [], |r| {
            Ok((r.get(0)?, r.get(1)?))
        })?;
    Ok((min_fid.map(|v| v as u32), max_fid.map(|v| v as u32)))
}

fn infer_schema_fields(
    conn: &Connection,
    object_columns: &[TableColumn],
    metadata: Option<&DatasetMetadata>,
) -> anyhow::Result<Vec<SchemaField>> {
    let mut fields = Vec::new();
    for col in object_columns {
        if col.name == "oid" {
            continue;
        }
        let upper = col.ty.to_ascii_uppercase();
        let type_desc = if upper.contains("BOOL") {
            TypeDesc::Bool
        } else if upper.contains("INT") {
            if upper.contains("UINTEGER")
                || upper.contains("UBIGINT")
                || upper.contains("USMALLINT")
            {
                TypeDesc::UInt
            } else {
                TypeDesc::Int
            }
        } else if upper.contains("VARCHAR") || upper.contains("TEXT") || upper.contains("STRING") {
            let variants = distinct_strings(conn, &col.name)?;
            if !variants.is_empty() && variants.len() <= 64 {
                TypeDesc::Enum(variants)
            } else {
                TypeDesc::String
            }
        } else {
            TypeDesc::String
        };
        fields.push(SchemaField {
            name: sanitize_name(&col.name),
            column: col.name.clone(),
            type_desc,
        });
    }

    if fields.is_empty() {
        if let Some(meta) = metadata {
            if !meta.obj_classes.is_empty() {
                fields.push(SchemaField {
                    name: "class".to_owned(),
                    column: "class".to_owned(),
                    type_desc: TypeDesc::Enum(meta.obj_classes.clone()),
                });
            }
        }
    }

    Ok(fields)
}

fn distinct_strings(conn: &Connection, col: &str) -> anyhow::Result<Vec<String>> {
    let sql = format!(
        "SELECT DISTINCT \"{col}\" FROM objects WHERE \"{col}\" IS NOT NULL ORDER BY \"{col}\""
    );
    let mut stmt = conn.prepare(&sql)?;
    let mut rows = stmt.query([])?;
    let mut values = Vec::new();
    while let Some(row) = rows.next()? {
        values.push(row.get(0)?);
    }
    Ok(values)
}

fn sanitize_name(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    for ch in name.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        "prop".to_owned()
    } else {
        out
    }
}
