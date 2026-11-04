//! Compiles a TrQL query to SQL and prints it.
//!
//! ```text
//! cargo run -q -p trql-sql --example compile -- "<query>"
//! cargo run -q -p trql-sql --example compile -- "<query>" | duckdb path/to/duckdb.db
//! ```
//!
//! `class` is typed as a plain string here rather than an enum, so no database is needed
//! to resolve class names — the compiler emits whatever the query wrote.

use trql_types::prop::{Schema, TypeDesc};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let query = std::env::args()
        .nth(1)
        .expect("usage: compile <query> [--tuples]");
    let output = if std::env::args().any(|a| a == "--tuples") {
        trql_sql::Output::Tuples
    } else {
        trql_sql::Output::Spans
    };
    let schema = Schema::new(vec![("class".into(), TypeDesc::String)]);

    let (vars, expr) = trql_dsl::parse_query(&query, &schema)?;
    println!(
        "{};",
        trql_sql::compile_with(&vars, &expr, &schema, output)?
    );
    Ok(())
}
