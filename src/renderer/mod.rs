use bevy::diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin};
use bevy::prelude::*;
use bioscape::{
    SimClock, FIXED_TIMESTEP_HZ, GENERATIONS_PER_EPOCH, TICKS_PER_GENERATION,
};
use std::path::PathBuf;

#[cfg(feature = "audio")]
mod audio;
mod camera;
mod components;
mod config;
mod diagnostics;
mod gizmos;
mod godmode;
mod input;
mod inspector;
mod material;
mod resources;
mod resources_gpu;
mod screencast;
mod setup;
mod sim_config;
mod stats;
mod systems;
mod world_map;

use camera::*;
use components::*;
use gizmos::*;
use godmode::*;
use input::*;
use material::*;
use resources::*;
use screencast::*;
use setup::*;
use stats::*;
use systems::*;

use diagnostics::{advance_clock, report_frame_diagnostics, tick_end, tick_start};

pub fn run() {
    // Sprint 87: bevy_diagnostic plugins jen na CLI flag `--diag`. Default
    // run je tichý (žádný LogDiagnostics spam, žádný frame-time tracking
    // overhead). `add_measurement` v existujících systémech zůstává unconditional
    // — pokud diag není registrovaný, je no-op.
    let want_diag = std::env::args().any(|a| a == "--diag");
    // Sprint 97 follow-up: in-process screencast (Bevy `Screenshot` API).
    // CLI: `--screencast=<dir>[,fps,duration_secs]` — fps default 1, duration
    // default 300s (= 5 min). PNG sequence; assemble s ffmpeg ven.
    // Důvod: ffmpeg x11grab + xfce4-screenshooter vrátily black frame přes
    // NVIDIA proprietary driver compositor — Bevy in-process capture čte
    // přímo z own swap-chain a nezávisí na external grabber.
    let screencast_cfg = std::env::args()
        .find_map(|a| a.strip_prefix("--screencast=").map(String::from))
        .map(|spec| {
            let parts: Vec<&str> = spec.split(',').collect();
            ScreencastConfig {
                dir: PathBuf::from(parts[0]),
                interval_secs: parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(1.0_f32),
                duration_secs: parts.get(2).and_then(|s| s.parse().ok()).unwrap_or(300.0_f32),
                started_at: None,
                last_capture: 0.0,
                frame_idx: 0,
            }
        });

    let mut app = App::new();
    app.add_plugins(DefaultPlugins.set(WindowPlugin {
        primary_window: Some(Window {
            title: "Bioscape".into(),
            ..default()
        }),
        ..default()
    }));
    // Sprint 91: registrace ExtendedMaterial<StandardMaterial, BioMaterialExt>
    // pipeline. Bez toho by Bevy nevěděl, jak rendrovat naše custom material
    // (asset by se loadnul ale shader by se nezkompiloval).
    app.add_plugins(MaterialPlugin::<BioMaterial>::default());

    // Sprint 187: Cell inspector — picking + gizmo outline + egui dialog
    // + JSON export. Pulls in `EguiPlugin` automatically.
    app.add_plugins(inspector::InspectorPlugin);

    #[cfg(feature = "audio")]
    app.add_plugins(audio::BioscapeAudioPlugin);

    // FrameTimeDiagnosticsPlugin runs unconditionally so the stats overlay
    // can show FPS — its overhead is just a few timestamps per frame. The
    // verbose log spam and custom per-system diagnostics stay behind `--diag`.
    app.add_plugins(FrameTimeDiagnosticsPlugin::default());
    if want_diag {
        app.add_plugins(LogDiagnosticsPlugin::default());
        diagnostics::register_diagnostics(&mut app);
    }

    // Sprint 184: EventCalendar generation moved into `setup` so it can
    // pick up `SimConfig::seed` (loaded from `bioscape.json`). Pre-S184
    // this used the literal `WORLD_MAP_SEED`, which made the shock
    // schedule diverge from the headless run for any non-default seed.

    app.init_resource::<gizmos::ShowVibration>()
        .init_resource::<ShowWhiskers>()
        .init_resource::<TickCounter>()
        // Sprint 36: clear color matchnut s HIGH richness color z `world_map_image`.
        // Sprint 88: white → ocean blue. Match s DistanceFog color tak aby
        // fog-fadeout splynul s pozadím (no harsh edges).
        // Sprint 88.1: bumped up — pre-fix bylo příliš tmavé, scéna prakticky
        // black. Medium ocean blue drží underwater feel ale zachová viditelnost.
        .insert_resource(ClearColor(Color::srgb(0.08, 0.18, 0.30)))
        .init_resource::<AdhesionMaterials>()
        .init_resource::<OrbitCamera>()
        .insert_resource(Time::<Fixed>::from_hz(FIXED_TIMESTEP_HZ as f64))
        .insert_resource(Clock(SimClock::new(
            TICKS_PER_GENERATION,
            GENERATIONS_PER_EPOCH,
        )))
        .init_resource::<CellEntityLookups>()
        .init_resource::<CellEntityPool>()
        .init_resource::<FoodDensityFactor>()
        .init_resource::<NextCellId>()
        .init_resource::<ContactProgress>()
        .init_resource::<CoopFoodResource>()
        .init_resource::<GodMode>()
        .init_resource::<MazeWorld>()
        .add_message::<GenerationEnded>()
        .add_message::<EpochEnded>()
        .add_systems(Startup, (setup_time_cap, setup, setup_stats_overlay).chain())
        // Sprint 178: minimal FixedUpdate chain — shared `sim_tick` is now
        // canonical. Legacy renderer tick systems (~25 functions across
        // `brains.rs`, `brains_gpu.rs`, `collisions.rs`, `fields.rs`,
        // `food.rs`, `lifecycle.rs`, `physics.rs`) jsou unscheduled here;
        // jejich code zatím zůstává v tree pro reference, ale Bevy je
        // nikdy nevolá. Dead-code warnings se objeví v cargo build —
        // S181 cleanup je smaže.
        //
        // Schedule:
        //   tick_start  → frame-timing diagnostic open
        //   advance_clock  → renderer's SimClock resource (used by some UI)
        //   sim_tick  → world.tick(&mut rng), all S133-S172 plasticity
        //   sync_simworld_to_cellentity  → world.cells → CellEntity copy
        //   tick_death_fade  → visual fade-out animation pro Dying entities
        //   tick_end  → frame-timing diagnostic close
        .add_systems(
            FixedUpdate,
            (
                tick_start,
                advance_clock,
                sim_tick,
                sync_simworld_to_cellentity,
                tick_death_fade,
                tick_end,
            )
                .chain(),
        )
        .add_systems(
            Update,
            (
                speed_input,
                god_mode_button_hover,
                camera_orbit_input,
                camera_zoom_input,
                camera_pan_input,
                update_orbit_camera_transform,
                sync_transforms,
                sync_spikes,
                draw_bond_gizmos,
                draw_cell_state_gizmos,
                draw_vibration_gizmos,
                draw_hazard_pulse_gizmos,
                toggle_vibration_overlay,
                toggle_maze_world,
                log_clock_events,
                toggle_stats_overlay,
                toggle_world_map_overlay,
                update_stats_overlay,
                report_frame_diagnostics,
                screencast_capture,
            ),
        )
        .add_systems(Update, (toggle_whiskers, draw_whiskers, sync_symbionts))
        // God-mode pipeline runs before camera input so RMB orbit suppression
        // takes effect on the same tick as the press. Order inside the chain:
        // handle button hits first, then run the RMB state machine, then close
        // on outside-clicks. Separate `add_systems` call avoids nested-tuple
        // trait resolution issues with `.chain()`.
        .add_systems(
            Update,
            (
                god_mode_handle_action,
                god_mode_input,
                close_menu_on_outside_click,
            )
                .chain()
                .before(camera_orbit_input),
        )
        .add_systems(Update, restart_simulation);
    if let Some(cfg) = screencast_cfg {
        let _ = std::fs::create_dir_all(&cfg.dir);
        eprintln!(
            "screencast: dir={:?} interval={}s duration={}s",
            cfg.dir, cfg.interval_secs, cfg.duration_secs
        );
        app.insert_resource(cfg);
    }
    app.run();
}


#[cfg(test)]
mod tests;
