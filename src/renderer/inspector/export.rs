//! JSON export pipeline. The "Save…" path spawns an
//! `rfd::AsyncFileDialog` on the IO task pool so the Bevy main thread
//! never blocks on the native file picker. "Copy" uses egui's
//! `commands().copy_text` and writes inline. Pure serialization +
//! filename logic lives in `bioscape::json_export` so the headless
//! binary can reuse it.

use bevy::prelude::*;
use bevy::tasks::{block_on, futures_lite::future, IoTaskPool};
use bevy_egui::egui;
use bioscape::json_export;
use bioscape::Cell;
use std::path::PathBuf;

use super::{PendingSave, SaveStatus};

/// Schedule a save dialog. Stores the JSON payload in `PendingSave` so
/// the poll system can write it once the dialog resolves; cancelling
/// the dialog leaves the payload in place but produces a `Cancelled`
/// status the dialog can surface to the user.
pub(super) fn spawn_save(pending: &mut PendingSave, cell: &Cell) {
    let Ok(json) = json_export::serialize_cell(cell) else {
        pending.last_status = Some(SaveStatus::Failed("serialization failed".into()));
        return;
    };
    let filename = json_export::default_filename(cell);
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
    match json_export::serialize_cell(cell) {
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
