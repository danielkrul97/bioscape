use bevy::ecs::system::SystemParam;
use bevy::prelude::*;
use bioscape::{
    reject_food_for_richness, EventCalendar, Food, MazeDifficulty, ObstacleField,
    ShockScheduleConfig, SimClock, SmellField, GENERATIONS_PER_EPOCH, MAX_SPAWN_ATTEMPTS,
    N_PHEROMONE_CHANNELS, PHEROMONE_GRID_RES, PHEROMONE_GRID_RES_Z, SMELL_GRID_RES,
    SMELL_GRID_RES_Z, SPIKE_SLOTS, TICKS_PER_GENERATION, VIBRATION_GRID_RES, VIBRATION_GRID_RES_Z,
    WORLD_HALF, WORLD_MAP_SEED,
};
use rand::rngs::StdRng;
use rand::SeedableRng;

use super::components::{
    CellEntity, FoodEntity, MazeWallEntity, SpikeEntity, StatsRoot, WorldMapOverlay,
};
use super::gizmos::ShowVibration;
use super::godmode::{GodMenuRoot, GodMode, GodModeState};
use super::material::{adhesion_material, cell_rotation, cell_scale, BioMaterial};
use super::resources::{
    AdhesionMaterials, CellEntityLookups, CellEntityPool, CellMesh, CellSlotMap, Clock,
    ContactProgress, CoopFoodResource, EventCalendarResource, FoodDensityFactor, FoodMaterial,
    FoodMesh, MazeWorld, NextCellId, PheromoneResource, SimRng, SimWorld, SmellResource,
    SpikeMaterial, SpikeMesh, TickCounter, VibrationResource, WorldExtent, WorldMapResource,
};
use super::resources_gpu::GpuFullPipeline;
use super::sim_config::SimConfig;
use super::world_map::food_target;

pub(super) fn speed_input(keys: Res<ButtonInput<KeyCode>>, mut time: ResMut<Time<Virtual>>) {
    if keys.just_pressed(KeyCode::Space) {
        if time.is_paused() {
            time.unpause();
            info!("sim: unpaused");
        } else {
            time.pause();
            info!("sim: paused");
        }
    }

    let preset = if keys.just_pressed(KeyCode::Digit1) {
        Some(1.0)
    } else if keys.just_pressed(KeyCode::Digit2) {
        Some(10.0)
    } else if keys.just_pressed(KeyCode::Digit3) {
        Some(100.0)
    } else if keys.just_pressed(KeyCode::Digit4) {
        Some(1000.0)
    } else {
        None
    };

    let delta = if keys.just_pressed(KeyCode::ArrowUp) {
        1.0
    } else if keys.just_pressed(KeyCode::ArrowDown) {
        -1.0
    } else {
        0.0
    };

    // Asymetrický step: nad 1× ±1 (1, 2, 3, …, 1000), pod 1× půlení/zdvojení
    // (1, 0.5, 0.25, …, 0.0625). Floor 1/16 dává ~4 ticks/s (z 60 Hz fixed
    // timestepu) — užitečné pro pozorování single-tick eventů (Hebbian update,
    // bond resolution, predation hit) bez fully-paused stop.
    let new_speed = match (preset, delta) {
        (Some(p), _) => Some(p),
        (None, d) if d != 0.0 => {
            let s = time.relative_speed();
            let next = if d > 0.0 {
                if s >= 1.0 {
                    s + 1.0
                } else {
                    s * 2.0
                }
            } else if s > 1.0 {
                s - 1.0
            } else {
                s * 0.5
            };
            Some(next.clamp(0.0625, 1000.0))
        }
        _ => None,
    };

    if let Some(speed) = new_speed {
        time.set_relative_speed(speed);
        if time.is_paused() {
            time.unpause();
        }
        info!("sim: {}× speed", speed);
    }
}

pub(super) fn toggle_stats_overlay(
    keys: Res<ButtonInput<KeyCode>>,
    mut nodes: Query<&mut Node, With<StatsRoot>>,
) {
    if !keys.just_pressed(KeyCode::KeyH) {
        return;
    }
    let Ok(mut node) = nodes.single_mut() else {
        return;
    };
    node.display = match node.display {
        Display::None => Display::Flex,
        _ => Display::None,
    };
}

pub(super) fn toggle_vibration_overlay(
    keys: Res<ButtonInput<KeyCode>>,
    mut show: ResMut<ShowVibration>,
) {
    if !keys.just_pressed(KeyCode::KeyV) {
        return;
    }
    show.0 = !show.0;
    info!("vibration overlay: {}", if show.0 { "on" } else { "off" });
}

pub(super) fn toggle_world_map_overlay(
    keys: Res<ButtonInput<KeyCode>>,
    mut overlays: Query<&mut Visibility, With<WorldMapOverlay>>,
) {
    if !keys.just_pressed(KeyCode::KeyM) {
        return;
    }
    for mut vis in &mut overlays {
        *vis = match *vis {
            Visibility::Hidden => Visibility::Visible,
            _ => Visibility::Hidden,
        };
    }
}

/// `KeyL` toggle for the maze world. Off → On allocates an `ObstacleField`
/// (medium difficulty by default), precomputes the per-grid Neumann masks,
/// and spawns one wall-voxel mesh per occupied voxel. On → Off deallocates
/// everything and despawns every `MazeWallEntity`. The seed mirrors the
/// renderer's `WORLD_MAP_SEED` so each session generates the same maze
/// across toggles within that session. Note: when the GPU full pipeline is
/// active (default), wall collision / LOS / mask diffusion are CPU-only
/// code paths that the GPU systems bypass — the visual walls render but
/// cells will pass through them. Restart with `BIOSCAPE_GPU_FULL=0` to use
/// maze physics. Wave 2 brings GPU shader parity.
pub(super) fn toggle_maze_world(
    keys: Res<ButtonInput<KeyCode>>,
    mut maze: ResMut<MazeWorld>,
    walls: Query<Entity, With<MazeWallEntity>>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    gpu_full: Option<ResMut<super::resources_gpu::GpuFullPipeline>>,
) {
    if !keys.just_pressed(KeyCode::KeyL) {
        return;
    }
    if maze.is_active() {
        info!("maze: off");
        maze.field = None;
        maze.smell_mask = None;
        maze.pheromone_masks = std::array::from_fn(|_| None);
        maze.vibration_mask = None;
        for entity in &walls {
            commands.entity(entity).despawn();
        }
        // Wave 4: clear GPU masks so the diffusion path falls back to
        // homogeneous behavior. Step shader leaves the mask buffer alone
        // (maze_active flag flips off via params next dispatch).
        if let Some(mut gpu) = gpu_full {
            gpu.smell.clear_obstacle_mask();
            gpu.pheromone.clear_obstacle_mask();
            gpu.vibration.clear_obstacle_mask();
            // FoodSpawn obstacle mask is read only when `obstacle_active`
            // param is set; the next spawn_food dispatch will pass 0 and
            // skip the mask binding regardless of buffer content.
            let _ = &gpu.food_spawn;
        }
        return;
    }
    let field = ObstacleField::new_maze(WORLD_HALF, WORLD_MAP_SEED, MazeDifficulty::Medium);
    info!(
        "maze: on (medium, {}×{} voxels, goal at [{:.0}, {:.0}])",
        field.resolution[0], field.resolution[1], field.goal_position[0], field.goal_position[1]
    );
    let smell_mask = Some(field.mask_for_grid([SMELL_GRID_RES, SMELL_GRID_RES, SMELL_GRID_RES_Z]));
    let pheromone_masks: [Option<Vec<bool>>; N_PHEROMONE_CHANNELS] = std::array::from_fn(|_| {
        Some(field.mask_for_grid([PHEROMONE_GRID_RES, PHEROMONE_GRID_RES, PHEROMONE_GRID_RES_Z]))
    });
    let vibration_mask =
        Some(field.mask_for_grid([VIBRATION_GRID_RES, VIBRATION_GRID_RES, VIBRATION_GRID_RES_Z]));
    let cs = field.voxel_size();
    let half_z = WORLD_HALF[2];
    let wall_mesh = meshes.add(Cuboid::new(cs[0], cs[1], 2.0 * half_z).mesh());
    let wall_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.20, 0.18, 0.16),
        perceptual_roughness: 0.85,
        metallic: 0.05,
        ..default()
    });
    let res_x = field.resolution[0];
    let res_y = field.resolution[1];
    for vy in 0..res_y {
        for vx in 0..res_x {
            if !field.occupied()[vy * res_x + vx] {
                continue;
            }
            let cx = -WORLD_HALF[0] + (vx as f32 + 0.5) * cs[0];
            let cy = -WORLD_HALF[1] + (vy as f32 + 0.5) * cs[1];
            commands.spawn((
                MazeWallEntity,
                Mesh3d(wall_mesh.clone()),
                MeshMaterial3d(wall_material.clone()),
                Transform::from_xyz(cx, cy, 0.0),
            ));
        }
    }
    // Wave 4: upload mask to GPU step shader + per-grid masks to FieldGpu
    // so in-shader collision and masked diffusion both see walls.
    if let Some(mut gpu) = gpu_full {
        let packed = field.packed_for_gpu();
        let smell_mask_for_gpu =
            field.mask_for_grid([SMELL_GRID_RES, SMELL_GRID_RES, SMELL_GRID_RES_Z]);
        let phero_mask_for_gpu =
            field.mask_for_grid([PHEROMONE_GRID_RES, PHEROMONE_GRID_RES, PHEROMONE_GRID_RES_Z]);
        let vib_mask_for_gpu =
            field.mask_for_grid([VIBRATION_GRID_RES, VIBRATION_GRID_RES, VIBRATION_GRID_RES_Z]);
        gpu.step.upload_maze(&packed);
        gpu.sensor.upload_maze(&packed);
        gpu.smell.upload_obstacle_mask(&smell_mask_for_gpu);
        gpu.pheromone.upload_obstacle_mask(&phero_mask_for_gpu);
        gpu.vibration.upload_obstacle_mask(&vib_mask_for_gpu);
        // Wave J port: food spawn shader rejects candidates inside walls
        // using the same packed mask as the step / sensor shaders.
        gpu.food_spawn.upload_obstacle(&packed);
    }
    maze.field = Some(field);
    maze.smell_mask = smell_mask;
    maze.pheromone_masks = pheromone_masks;
    maze.vibration_mask = vibration_mask;
}

#[derive(SystemParam)]
pub(super) struct RestartAssets<'w> {
    pub(super) cell_mesh: Res<'w, CellMesh>,
    pub(super) spike_mesh: Res<'w, SpikeMesh>,
    pub(super) spike_material: Res<'w, SpikeMaterial>,
    pub(super) food_mesh: Res<'w, FoodMesh>,
    pub(super) food_material: Res<'w, FoodMaterial>,
}

#[derive(SystemParam)]
pub(super) struct RestartRegistries<'w> {
    pub(super) adhesion_materials: ResMut<'w, AdhesionMaterials>,
    pub(super) bio_materials: ResMut<'w, Assets<BioMaterial>>,
    pub(super) slot_map: ResMut<'w, CellSlotMap>,
    pub(super) pool: ResMut<'w, CellEntityPool>,
    pub(super) next_id: ResMut<'w, NextCellId>,
}

#[derive(SystemParam)]
pub(super) struct RestartSimResources<'w> {
    pub(super) clock: ResMut<'w, Clock>,
    pub(super) tick_counter: ResMut<'w, TickCounter>,
    pub(super) food_density: ResMut<'w, FoodDensityFactor>,
    pub(super) contact_progress: ResMut<'w, ContactProgress>,
    pub(super) coop_food: ResMut<'w, CoopFoodResource>,
    pub(super) lookups: ResMut<'w, CellEntityLookups>,
    pub(super) god: ResMut<'w, GodMode>,
    pub(super) sim_world: ResMut<'w, SimWorld>,
    pub(super) sim_rng: ResMut<'w, SimRng>,
    pub(super) event_calendar: ResMut<'w, EventCalendarResource>,
    pub(super) sim_config: Res<'w, SimConfig>,
}

#[derive(SystemParam)]
pub(super) struct RestartEntities<'w, 's> {
    pub(super) cells: Query<'w, 's, Entity, With<CellEntity>>,
    pub(super) foods: Query<'w, 's, Entity, With<FoodEntity>>,
    pub(super) spikes: Query<'w, 's, Entity, With<SpikeEntity>>,
    pub(super) menus: Query<'w, 's, Entity, With<GodMenuRoot>>,
}

/// `KeyR` restarts the simulation in place: despawn all entities, rebuild
/// the shared `SimWorld` from `SimConfig` (matching the original startup),
/// and respawn visual entities sized to the freshly-initialised world.
/// Camera, maze toggle and time speed are preserved.
///
/// Sprint 184: pre-S184 restart only reset the renderer-side scaffolding
/// and spawned fresh `CellEntity` items from a thread-local `rand::rng()`;
/// `SimWorld` was left untouched, so the next `sync_simworld_to_cellentity`
/// immediately overwrote the new entities with the old, in-flight sim
/// state. The fix rebuilds the shared world (the only canonical sim
/// state) and lets the legacy GPU pipeline tag along on the upload it
/// still mirrors.
pub(super) fn restart_simulation(
    keys: Res<ButtonInput<KeyCode>>,
    mut commands: Commands,
    entities: RestartEntities,
    extent: Res<WorldExtent>,
    world_map_res: Res<WorldMapResource>,
    assets: RestartAssets,
    mut registries: RestartRegistries,
    mut sim_res: RestartSimResources,
    gpu_full: ResMut<GpuFullPipeline>,
) {
    if !keys.just_pressed(KeyCode::KeyR) {
        return;
    }

    for entity in &entities.cells {
        commands.entity(entity).despawn();
    }
    for entity in &entities.foods {
        commands.entity(entity).despawn();
    }
    for entity in &entities.spikes {
        commands.entity(entity).despawn();
    }
    for entity in &entities.menus {
        commands.entity(entity).despawn();
    }

    sim_res.clock.0 = SimClock::new(TICKS_PER_GENERATION, GENERATIONS_PER_EPOCH);
    *sim_res.tick_counter = TickCounter::default();
    sim_res.food_density.0 = 1.0;
    sim_res.contact_progress.0.clear();
    sim_res.coop_food.0.clear();
    sim_res.lookups.id_to_entity.clear();
    sim_res.lookups.id_to_position.clear();
    sim_res.lookups.entity_to_idx.clear();
    sim_res.lookups.positions_by_idx.clear();
    sim_res.god.state = GodModeState::Idle;
    registries.slot_map.slot_to_entity.clear();
    registries.slot_map.entity_to_slot.clear();
    // R-press despawns all CellEntity (including hidden+pooled ones), so any
    // Entity refs left in `free_cells` are now stale (generation bumped).
    // The next `sync_simworld_to_cellentity` Grow phase would `pop()` a stale
    // ref and panic on `insert<CellEntity>`. Clear the pool alongside the
    // slot_map so the post-restart world starts with a fresh entity pool.
    registries.pool.free_cells.clear();

    let half = extent.as_array();
    commands.insert_resource(SmellResource(SmellField::new(
        [SMELL_GRID_RES, SMELL_GRID_RES, SMELL_GRID_RES_Z],
        half,
    )));
    commands.insert_resource(PheromoneResource {
        fields: std::array::from_fn(|_| {
            SmellField::new(
                [PHEROMONE_GRID_RES, PHEROMONE_GRID_RES, PHEROMONE_GRID_RES_Z],
                half,
            )
        }),
    });
    commands.insert_resource(VibrationResource(SmellField::new(
        [VIBRATION_GRID_RES, VIBRATION_GRID_RES, VIBRATION_GRID_RES_Z],
        half,
    )));

    let config: &SimConfig = &sim_res.sim_config;
    let mut new_sim_rng = StdRng::seed_from_u64(config.seed);
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
    let new_calendar = EventCalendar::generate(config.seed, &shock_cfg, 1_000_000);
    let mut new_world = bioscape::sim::World::new_with_maze(
        &mut new_sim_rng,
        config.resolved_map_seed(),
        config.resolved_mating_radius(),
        config.resolved_initial_cells(),
        config.resolved_max_population(),
        new_calendar.clone(),
        config.resolved_maze(),
    );
    let izh_frac = config.initial_izhikevich_frac.clamp(0.0, 1.0);
    if izh_frac > 0.0 {
        let target = (izh_frac * new_world.cells.len() as f32).round() as usize;
        for cell in new_world.cells.iter_mut().take(target) {
            cell.genome.neuron_model = bioscape::NeuronModel::Izhikevich;
        }
    }
    if let Err(e) = new_world.init_gpu_full() {
        panic!("sim: restart init_gpu_full failed ({e})");
    }

    registries.next_id.0 = new_world.cells.len() as u64;

    for cell in &new_world.cells {
        let mat = adhesion_material(
            &mut registries.adhesion_materials,
            &mut registries.bio_materials,
            cell.genome.adhesion_type,
        );
        let entity = commands
            .spawn((
                CellEntity(*cell),
                Mesh3d(assets.cell_mesh.0.clone()),
                MeshMaterial3d(mat),
                Transform::from_xyz(cell.position[0], cell.position[1], cell.position[2])
                    .with_rotation(cell_rotation(cell.heading, cell.pitch))
                    .with_scale(cell_scale(&cell.phenotype)),
            ))
            .id();
        for slot in 0..SPIKE_SLOTS as u8 {
            commands.spawn((
                SpikeEntity {
                    owner: entity,
                    slot,
                },
                Mesh3d(assets.spike_mesh.0.clone()),
                MeshMaterial3d(assets.spike_material.0.clone()),
                Transform::default(),
                Visibility::Hidden,
            ));
        }
        registries.slot_map.allocate(entity);
    }

    gpu_full
        .cells
        .upload_brains(new_world.cells.iter().map(|c| &c.genome.brain));
    gpu_full
        .cells
        .upload_xoshiro_seeds(new_world.cells.iter().map(|c| c.cell_id));
    let turn_rates: Vec<f32> = new_world.cells.iter().map(|c| c.genome.turn_rate).collect();
    gpu_full.cells.upload_turn_rates(&turn_rates);
    gpu_full.cells.zero_persistent_state();

    let new_cell_count = new_world.cells.len();
    let new_food_count = new_world.foods.len();
    sim_res.sim_world.0 = new_world;
    sim_res.sim_rng.0 = new_sim_rng;
    sim_res.event_calendar.0 = new_calendar;

    let mut food_rng = rand::rng();
    let initial_food = food_target(&extent, 1.0);
    for _ in 0..initial_food {
        let mut food = Food::random(&mut food_rng, half);
        for _ in 0..MAX_SPAWN_ATTEMPTS {
            let richness = world_map_res
                .0
                .sample([food.position[0], food.position[1], 0.0]);
            if !reject_food_for_richness(&mut food_rng, richness) {
                break;
            }
            food = Food::random(&mut food_rng, half);
        }
        commands.spawn((
            FoodEntity(food),
            Mesh3d(assets.food_mesh.0.clone()),
            MeshMaterial3d(assets.food_material.0.clone()),
            Transform::from_xyz(food.position[0], food.position[1], food.position[2]),
            Visibility::Hidden,
        ));
    }

    info!(
        "sim: restarted (seed={}, sim cells={}, sim food={}, visual food={})",
        config.seed, new_cell_count, new_food_count, initial_food
    );
}
