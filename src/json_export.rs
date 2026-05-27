//! Pure-Rust JSON export of `Cell`. Shared between the renderer's inspector
//! (file dialog + clipboard wrappers in `renderer::inspector::export`) and
//! the headless dump path (`bin/headless/dump`). No Bevy / egui deps.
//!
//! `serialize_cell` produces a human-readable JSON document where arrays of
//! primitives collapse onto a single line. `serde_json::to_string_pretty`
//! blows up to ~10 k lines for one Cell (45×84 brain weight matrices land
//! one float per line); this formatter keeps the same data in ~500–1000
//! readable lines without losing precision.

use crate::Cell;
use std::time::{SystemTime, UNIX_EPOCH};

pub fn serialize_cell(cell: &Cell) -> Result<String, serde_json::Error> {
    let value = serde_json::to_value(cell)?;
    Ok(format_human(&value))
}

pub fn format_human(value: &serde_json::Value) -> String {
    // A full Cell dump is tens of KB (45-row weight matrices, one collapsed
    // line each) — pre-size so the recursive writer skips ~a dozen regrows.
    let mut out = String::with_capacity(64 * 1024);
    write_value(value, 0, &mut out);
    out.push('\n');
    out
}

fn write_value(v: &serde_json::Value, indent: usize, out: &mut String) {
    use serde_json::Value;
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => out.push_str(&n.to_string()),
        Value::String(s) => {
            out.push_str(&serde_json::to_string(s).unwrap_or_else(|_| String::from("\"\"")))
        }
        Value::Array(arr) => write_array(arr, indent, out),
        Value::Object(map) => write_object(map, indent, out),
    }
}

fn write_array(arr: &[serde_json::Value], indent: usize, out: &mut String) {
    if arr.is_empty() {
        out.push_str("[]");
        return;
    }
    if arr.iter().all(is_primitive) {
        out.push('[');
        for (i, item) in arr.iter().enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            write_value(item, indent + 1, out);
        }
        out.push(']');
        return;
    }
    out.push_str("[\n");
    let pad_inner = "  ".repeat(indent + 1);
    let pad_outer = "  ".repeat(indent);
    for (i, item) in arr.iter().enumerate() {
        out.push_str(&pad_inner);
        write_value(item, indent + 1, out);
        if i + 1 < arr.len() {
            out.push(',');
        }
        out.push('\n');
    }
    out.push_str(&pad_outer);
    out.push(']');
}

fn write_object(map: &serde_json::Map<String, serde_json::Value>, indent: usize, out: &mut String) {
    if map.is_empty() {
        out.push_str("{}");
        return;
    }
    out.push_str("{\n");
    let pad_inner = "  ".repeat(indent + 1);
    let pad_outer = "  ".repeat(indent);
    let len = map.len();
    for (i, (k, v)) in map.iter().enumerate() {
        out.push_str(&pad_inner);
        let key = serde_json::to_string(k).unwrap_or_else(|_| format!("\"{}\"", k));
        out.push_str(&key);
        out.push_str(": ");
        write_value(v, indent + 1, out);
        if i + 1 < len {
            out.push(',');
        }
        out.push('\n');
    }
    out.push_str(&pad_outer);
    out.push('}');
}

fn is_primitive(v: &serde_json::Value) -> bool {
    use serde_json::Value;
    matches!(
        v,
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_)
    )
}

/// Filename embedding `cell_id`, `age`, `lineage_id`, and a UTC timestamp so a
/// researcher can dump many cells into one folder without manual renaming.
pub fn default_filename(cell: &Cell) -> String {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!(
        "cell_{}_age{}_lin{}_{}.json",
        cell.cell_id, cell.age, cell.lineage_id, ts
    )
}

/// Stable filename without timestamp — useful when a dump pass writes many
/// cells into a per-generation directory where collisions are impossible by
/// construction (cell_id is unique within a run).
pub fn stable_filename(cell: &Cell) -> String {
    format!(
        "cell_{}_age{}_lin{}.json",
        cell.cell_id, cell.age, cell.lineage_id
    )
}
