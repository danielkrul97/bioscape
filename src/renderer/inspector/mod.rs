//! Sprint 187: Cell inspector — pick a cell with LMB, see a subtle gizmo
//! ring around it, and open an egui dialog with live brain activations,
//! metadata, and a JSON export (Save… / Copy). The dialog freezes a
//! snapshot when the selected cell dies so the user can still read +
//! download the last known state.
//!
//! Module layout:
//! - `state` — `SelectedCell` / `HoverCell` / `PendingSave` resources
//!   (defined inline below for proximity to the plugin)
//! - `picking` — LMB ray-vs-cell hit test, hover preview
//! - `outline` — gizmo ring around the selected cell
//! - `dialog` — egui inspector window orchestrator
//! - `brain_viz` — activation bar widget for the brain panel
//! - `export` — JSON serialization + rfd save dialog

mod brain_viz;
mod dialog;
mod export;
mod history;
mod outline;
mod picking;

use bevy::prelude::*;
use bevy::tasks::Task;
use bevy_egui::{
    input::{egui_wants_any_keyboard_input, egui_wants_any_pointer_input},
    EguiGlobalSettings, EguiPlugin, EguiPrimaryContextPass, PrimaryEguiContext,
};
use bioscape::{Cell, BRAIN_HIDDEN, BRAIN_OUTPUTS};
use std::collections::VecDeque;
use std::path::PathBuf;

/// Identity of the cell the user clicked on, plus a frozen snapshot used
/// after the cell dies so the inspector can still render its last known
/// state. Updated live each tick while `deceased == false`.
#[derive(Resource, Default)]
pub(super) struct SelectedCell {
    pub(super) cell_id: Option<u64>,
    pub(super) snapshot: Option<Cell>,
    pub(super) deceased: bool,
}

impl SelectedCell {
    pub(super) fn clear(&mut self) {
        self.cell_id = None;
        self.snapshot = None;
        self.deceased = false;
    }

    pub(super) fn is_active(&self) -> bool {
        self.cell_id.is_some()
    }
}

/// Cell currently under the mouse cursor (set every frame). Drives the
/// faint hover ring so users can preview which cell a click would select.
#[derive(Resource, Default)]
pub(super) struct HoverCell {
    pub(super) cell_id: Option<u64>,
}

/// Background rfd::AsyncFileDialog task. Polled each frame; when it
/// resolves, the cached JSON payload is written to the chosen path.
#[derive(Resource, Default)]
pub(super) struct PendingSave {
    pub(super) task: Option<Task<Option<PathBuf>>>,
    pub(super) payload: Option<String>,
    pub(super) last_status: Option<SaveStatus>,
}

#[derive(Debug, Clone)]
pub(super) enum SaveStatus {
    Ok(PathBuf),
    Cancelled,
    Failed(String),
}

/// Rolling buffer of last-N brain activations for the currently selected
/// cell. Recording only runs while a cell is selected and alive — quiet
/// otherwise. The buffer is `VecDeque` so trimming the oldest sample is
/// O(1). Resets whenever the user picks a different cell so we never mix
/// activations across identities.
pub(super) const HISTORY_LEN: usize = 360;

#[derive(Resource, Default)]
pub(super) struct ActivationHistory {
    pub(super) cell_id: Option<u64>,
    pub(super) hidden: VecDeque<[f32; BRAIN_HIDDEN]>,
    pub(super) outputs: VecDeque<[f32; BRAIN_OUTPUTS]>,
}

impl ActivationHistory {
    pub(super) fn reset_for(&mut self, cell_id: Option<u64>) {
        self.cell_id = cell_id;
        self.hidden.clear();
        self.outputs.clear();
    }
}

/// Disable bevy_egui's `auto_create_primary_context` before any camera is
/// spawned. Our app has two `Camera` entities (the renderer's `Camera3d`
/// + an unidentified one with `Camera` only and no render graph) — the
/// auto-attach picks one non-deterministically, which collided with our
/// explicit attach below and caused `Multiple entities fit the query`
/// panics. With auto-attach off, only the `Camera3d` gets the context.
///
/// Also enable `enable_absorb_bevy_input_system` so clicks on the
/// inspector window don't leak into camera orbit, god-mode RMB, or any
/// other input-reading system in the renderer. egui clears the relevant
/// `ButtonInput`/event resources whenever its widgets want pointer or
/// keyboard input — no per-system `run_if` plumbing needed.
fn disable_egui_auto_context(mut settings: ResMut<EguiGlobalSettings>) {
    settings.auto_create_primary_context = false;
    settings.enable_absorb_bevy_input_system = true;
}

/// Explicitly attach `PrimaryEguiContext` to the renderer's `Camera3d`.
/// Runs in `PostStartup` so the renderer's `setup` chain has already
/// spawned the camera. `Camera3d` filter excludes the mystery
/// `Camera`-only entity from any plugin we don't control.
fn attach_primary_egui_context(
    mut commands: Commands,
    cameras: Query<Entity, With<Camera3d>>,
) {
    for entity in &cameras {
        commands.entity(entity).insert(PrimaryEguiContext);
        info!("inspector: attached PrimaryEguiContext to {:?}", entity);
    }
}

pub(super) struct InspectorPlugin;

impl Plugin for InspectorPlugin {
    fn build(&self, app: &mut App) {
        // Try single-pass mode first — multipass with our existing
        // Camera3d (Hdr/Bloom/DistanceFog stack) appears to skip the
        // primary-context attach in this project. Single-pass renders
        // unconditionally through the loop system.
        #[allow(deprecated)]
        app.add_plugins(EguiPlugin {
            enable_multipass_for_primary_context: false,
            ..Default::default()
        })
            .init_resource::<SelectedCell>()
            .init_resource::<HoverCell>()
            .init_resource::<PendingSave>()
            .init_resource::<ActivationHistory>()
            // PreStartup: disable auto-attach BEFORE any camera spawns
            // so bevy_egui doesn't latch onto the wrong Camera entity.
            // PostStartup: explicit attach after the renderer's `setup`
            // has run.
            .add_systems(PreStartup, disable_egui_auto_context)
            .add_systems(PostStartup, attach_primary_egui_context)
            // Picking & hover read mouse state; gate them off when egui owns
            // the cursor so the dialog itself is clickable. Picking runs
            // before camera orbit input so a click never starts an orbit
            // and a selection at the same time.
            .add_systems(
                Update,
                (
                    picking::hover_cell.run_if(not(egui_wants_any_pointer_input)),
                    picking::pick_cell.run_if(not(egui_wants_any_pointer_input)),
                    picking::clear_on_escape.run_if(not(egui_wants_any_keyboard_input)),
                    picking::sync_selection_snapshot,
                    history::record_history,
                    outline::draw_outline,
                    outline::draw_hover_outline,
                    export::poll_pending_save,
                )
                    .chain(),
            )
            .add_systems(EguiPrimaryContextPass, dialog::draw_inspector_window);
    }
}
