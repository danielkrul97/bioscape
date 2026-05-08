use bevy::diagnostic::{
    Diagnostic, Diagnostics, DiagnosticsStore, FrameTimeDiagnosticsPlugin, RegisterDiagnostic,
};
use bevy::prelude::*;
use std::time::Instant;

use super::components::{EpochEnded, GenerationEnded};
use super::config::*;
use super::resources::{Clock, TickCounter};

pub(super) fn tick_start(mut counter: ResMut<TickCounter>) {
    counter.tick_start = Some(Instant::now());
}

pub(super) fn tick_end(mut counter: ResMut<TickCounter>) {
    if let Some(t0) = counter.tick_start.take() {
        counter.sim_ms_this_frame += t0.elapsed().as_secs_f64() * 1000.0;
    }
    counter.ticks_this_frame += 1;
}

pub(super) fn advance_clock(
    mut clock: ResMut<Clock>,
    mut generation_ended: MessageWriter<GenerationEnded>,
    mut epoch_ended: MessageWriter<EpochEnded>,
) {
    let transitions = clock.0.advance();
    if let Some(generation) = transitions.generation_ended {
        generation_ended.write(GenerationEnded { generation });
    }
    if let Some(epoch) = transitions.epoch_ended {
        epoch_ended.write(EpochEnded { epoch });
    }
}

pub(super) fn report_frame_diagnostics(
    mut counter: ResMut<TickCounter>,
    diag_store: Res<DiagnosticsStore>,
    mut diag: Diagnostics,
) {
    let ticks = counter.ticks_this_frame;
    let sim_ms = counter.sim_ms_this_frame;
    counter.ticks_this_frame = 0;
    counter.sim_ms_this_frame = 0.0;
    diag.add_measurement(&DIAG_TICKS_PER_FRAME, || ticks as f64);
    let frame_ms = diag_store
        .get(&FrameTimeDiagnosticsPlugin::FRAME_TIME)
        .and_then(|d| d.value())
        .unwrap_or(0.0);
    if frame_ms > 0.0 {
        diag.add_measurement(&DIAG_RENDER_OVERHEAD, || (frame_ms - sim_ms).max(0.0));
    }
}

pub(super) fn register_diagnostics(app: &mut App) {
    app.register_diagnostic(Diagnostic::new(DIAG_CELL_COUNT).with_suffix(" cells"))
        .register_diagnostic(Diagnostic::new(DIAG_FOOD_COUNT).with_suffix(" food"))
        .register_diagnostic(Diagnostic::new(DIAG_BRAIN_ACT).with_suffix(" ms"))
        .register_diagnostic(Diagnostic::new(DIAG_BRAIN_GPU_RT).with_suffix(" ms"))
        .register_diagnostic(Diagnostic::new(DIAG_BROWNIAN).with_suffix(" ms"))
        .register_diagnostic(Diagnostic::new(DIAG_BROWNIAN_GPU_RT).with_suffix(" ms"))
        .register_diagnostic(Diagnostic::new(DIAG_COLLISIONS).with_suffix(" ms"))
        .register_diagnostic(Diagnostic::new(DIAG_PREDATION).with_suffix(" ms"))
        .register_diagnostic(Diagnostic::new(DIAG_EAT_FOOD).with_suffix(" ms"))
        .register_diagnostic(Diagnostic::new(DIAG_SMELL).with_suffix(" ms"))
        .register_diagnostic(Diagnostic::new(DIAG_PHEROMONE).with_suffix(" ms"))
        .register_diagnostic(Diagnostic::new(DIAG_GRID_REBUILD).with_suffix(" ms"))
        .register_diagnostic(Diagnostic::new(DIAG_SYNC_TRANSFORMS).with_suffix(" ms"))
        .register_diagnostic(Diagnostic::new(DIAG_TICKS_PER_FRAME).with_suffix(" ticks"))
        .register_diagnostic(Diagnostic::new(DIAG_RENDER_OVERHEAD).with_suffix(" ms"));
}
