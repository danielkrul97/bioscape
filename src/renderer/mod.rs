use bevy::diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin};
use bevy::prelude::*;
use bioscape::{
    EventCalendar, ShockScheduleConfig, SimClock, FIXED_TIMESTEP_HZ, GENERATIONS_PER_EPOCH,
    TICKS_PER_GENERATION, WORLD_MAP_SEED,
};
use std::path::PathBuf;

mod camera;
mod components;
mod config;
mod diagnostics;
mod gizmos;
mod input;
mod material;
mod resources;
#[cfg(feature = "gpu")]
mod resources_gpu;
mod screencast;
mod setup;
mod stats;
mod systems;
mod world_map;

use camera::*;
use components::*;
use gizmos::*;
use input::*;
use material::*;
use resources::*;
#[cfg(feature = "gpu")]
use resources_gpu::*;
use screencast::*;
use setup::*;
use stats::*;
use systems::*;
use world_map::*;

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

    // FrameTimeDiagnosticsPlugin runs unconditionally so the stats overlay
    // can show FPS — its overhead is just a few timestamps per frame. The
    // verbose log spam and custom per-system diagnostics stay behind `--diag`.
    app.add_plugins(FrameTimeDiagnosticsPlugin::default());
    if want_diag {
        app.add_plugins(LogDiagnosticsPlugin::default());
        diagnostics::register_diagnostics(&mut app);
    }

    // Sprint 109: shock kalendář z env var `BIOSCAPE_SHOCKS_MEAN_GENS`. Když
    // unset / 0 / parse fail, kalendář je prázdný (no-op, default). MAX_GENS
    // pro rendererský běh je velký — interaktivní session typicky < 10k gen,
    // 1M je hard cap aby `EventCalendar::generate` netočilo donekonečna.
    let shocks_mean_gens: u32 = std::env::var("BIOSCAPE_SHOCKS_MEAN_GENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let shock_cfg = if shocks_mean_gens > 0 {
        ShockScheduleConfig {
            mean_gens_between: shocks_mean_gens,
            ..Default::default()
        }
    } else {
        ShockScheduleConfig::default()
    };
    let event_calendar = EventCalendar::generate(WORLD_MAP_SEED, &shock_cfg, 1_000_000);
    if shocks_mean_gens > 0 {
        eprintln!(
            "shocks: mean_gens_between={} scheduled={} (sim integration arrives in S110+)",
            shocks_mean_gens,
            event_calendar.events.len()
        );
    }

    app.init_resource::<TickCounter>()
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
        .insert_resource(EventCalendarResource(event_calendar))
        .init_resource::<CellGrid>()
        .init_resource::<FoodGrid>()
        .init_resource::<CellEntityLookups>()
        .init_resource::<FoodDensityFactor>()
        .init_resource::<NextCellId>()
        .init_resource::<ContactProgress>()
        .init_resource::<CoopFoodResource>()
        .add_message::<GenerationEnded>()
        .add_message::<EpochEnded>()
        .add_systems(Startup, (setup_time_cap, setup, setup_stats_overlay, rebuild_cell_grid).chain())
        .add_systems(
            FixedUpdate,
            (
                (
                    tick_start,
                    advance_clock,
                    update_food_density_cycle,
                    rebuild_food_grid,
                    rebuild_cell_entity_lookups,
                    update_smell_field,
                    update_pheromone_field,
                    pool_bonded_hidden_cells,
                    #[cfg(feature = "gpu")]
                    cells_brain_act_gpu_full
                        .run_if(resource_exists::<GpuFullPipeline>),
                    cells_brain_act,
                    emit_pheromones,
                    apply_cell_morph,
                    apply_brownian_motion,
                )
                    .chain(),
                (
                    step_cells,
                    apply_food_gravity,
                    apply_environmental_hazards,
                    rebuild_cell_grid,
                    resolve_cell_collisions,
                    cell_predates_on_neighbor,
                    cell_eats_food,
                    spawn_food,
                    spawn_coop_food,
                    update_coop_food,
                    cell_reproduces_on_threshold,
                    cell_dies_on_zero_energy,
                    tick_death_fade,
                    tick_end,
                )
                    .chain(),
            )
                .chain(),
        )
        .add_systems(
            Update,
            (
                speed_input,
                camera_orbit_input,
                camera_zoom_input,
                camera_pan_input,
                update_orbit_camera_transform,
                sync_transforms,
                draw_bond_gizmos,
                draw_cell_state_gizmos,
                log_clock_events,
                toggle_stats_overlay,
                toggle_world_map_overlay,
                update_stats_overlay,
                report_frame_diagnostics,
                screencast_capture,
            ),
        );
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
