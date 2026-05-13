//! JSON export pipeline. The "Save…" path spawns an
//! `rfd::AsyncFileDialog` on the IO task pool so the Bevy main thread
//! never blocks on the native file picker. "Copy" uses egui's
//! `commands().copy_text` and writes inline. Default filename embeds
//! `cell_id` + a UTC timestamp so a researcher can dump many cells into
//! one folder without manual renaming.

use bevy::prelude::*;
use bevy::tasks::{block_on, futures_lite::future, IoTaskPool};
use bevy_egui::egui;
use bioscape::Cell;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use super::{PendingSave, SaveStatus};

pub(super) fn serialize_cell(cell: &Cell) -> Result<String, serde_json::Error> {
    let value = serde_json::to_value(cell)?;
    Ok(format_human(&value))
}

/// Pretty-print a JSON value with one rule: arrays whose every element is
/// a primitive (number, bool, null, string) collapse onto a single line,
/// e.g. `"last_hidden": [0.0, 0.12, -0.34, ...]`. Objects and arrays-of-
/// objects stay multi-line so the structure remains scannable.
///
/// `to_string_pretty` blows up to ~10 k lines for a single cell because
/// the brain weight matrices (45×84 = 3780 floats each, plus traces) get
/// one float per line. This formatter shrinks the same dump to ~500-1000
/// readable lines without losing precision.
fn format_human(value: &serde_json::Value) -> String {
    let mut out = String::new();
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

pub(super) fn default_filename(cell: &Cell) -> String {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!(
        "cell_{}_age{}_lin{}_{}.json",
        cell.cell_id, cell.age, cell.lineage_id, ts
    )
}

/// Schedule a save dialog. Stores the JSON payload in `PendingSave` so
/// the poll system can write it once the dialog resolves; cancelling
/// the dialog leaves the payload in place but produces a `Cancelled`
/// status the dialog can surface to the user.
pub(super) fn spawn_save(pending: &mut PendingSave, cell: &Cell) {
    let Ok(json) = serialize_cell(cell) else {
        pending.last_status = Some(SaveStatus::Failed("serialization failed".into()));
        return;
    };
    let filename = default_filename(cell);
    let task = IoTaskPool::get().spawn(async move {
        rfd::AsyncFileDialog::new()
            .set_file_name(&filename)
            .add_filter("JSON", &["json"])
            .save_file()
            .await
            .map(|handle| handle.path().to_path_buf())
    });
    pending.task = Some(task);
    pending.payload = Some(json);
    pending.last_status = None;
}

/// Copy the cell's JSON to the clipboard via egui's output channel. The
/// system clipboard write happens during egui's end-of-frame flush.
pub(super) fn copy_to_clipboard(ctx: &egui::Context, cell: &Cell) -> SaveStatus {
    match serialize_cell(cell) {
        Ok(json) => {
            ctx.copy_text(json);
            SaveStatus::Ok(PathBuf::from("(clipboard)"))
        }
        Err(e) => SaveStatus::Failed(e.to_string()),
    }
}

/// Per-frame system: peek at the running dialog task; if it resolved,
/// either write the chosen path or record a Cancelled / Failed status.
/// Errors only abort the current save — the inspector window stays
/// usable for retry.
pub(super) fn poll_pending_save(mut pending: ResMut<PendingSave>) {
    let Some(task) = pending.task.as_mut() else {
        return;
    };
    let Some(result) = block_on(future::poll_once(task)) else {
        return;
    };
    pending.task = None;
    let payload = pending.payload.take();
    match (result, payload) {
        (Some(path), Some(json)) => match std::fs::write(&path, json) {
            Ok(()) => pending.last_status = Some(SaveStatus::Ok(path)),
            Err(e) => pending.last_status = Some(SaveStatus::Failed(e.to_string())),
        },
        (Some(_), None) => {
            pending.last_status = Some(SaveStatus::Failed("payload missing".into()));
        }
        (None, _) => {
            pending.last_status = Some(SaveStatus::Cancelled);
        }
    }
}
