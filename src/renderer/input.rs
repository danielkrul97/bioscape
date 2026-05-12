use bevy::ecs::system::SystemParam;
use bevy::prelude::*;
use bioscape::{
    reject_food_for_richness, Cell, Food, GENERATIONS_PER_EPOCH, INITIAL_CELLS,
    MAX_SPAWN_ATTEMPTS, MazeDifficulty, N_PHEROMONE_CHANNELS, ObstacleField, PHEROMONE_GRID_RES,
    PHEROMONE_GRID_RES_Z, SimClock, SmellField, SMELL_GRID_RES, SMELL_GRID_RES_Z, SPIKE_SLOTS,
    TICKS_PER_GENERATION, VIBRATION_GRID_RES, VIBRATION_GRID_RES_Z, WORLD_HALF, WORLD_MAP_SEED,
};

use super::components::{CellEntity, FoodEntity, MazeWallEntity, SpikeEntity, StatsRoot, WorldMapOverlay};
use super::gizmos::ShowVibration;
use super::godmode::{GodMenuRoot, GodMode, GodModeState};
use super::material::{adhesion_material, cell_rotation, cell_scale, BioMaterial};
use super::resources::{
    AdhesionMaterials, CellEntityLookups, CellMesh, CellSlotMap, Clock, ContactProgress,
    CoopFoodResource, FoodDensityFactor, FoodMaterial, FoodMesh, MazeWorld, NextCellId,
    PheromoneResource, SmellResource, SpikeMaterial, SpikeMesh, TickCounter, VibrationResource,
    WorldExtent, WorldMapResource,
};
use super::resources_gpu::GpuFullPipeline;
use super::world_map::food_target;

pub(super) fn speed_input(
    keys: Res<ButtonInput<KeyCode>>,
    mut time: ResMut<Time<Virtual>>,
) {
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
                if s >= 1.0 { s + 1.0 } else { s * 2.0 }
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
    let smell_mask =
        Some(field.mask_for_grid([SMELL_GRID_RES, SMELL_GRID_RES, SMELL_GRID_RES_Z]));
    let pheromone_masks: [Option<Vec<bool>>; N_PHEROMONE_CHANNELS] = std::array::from_fn(|_| {
        Some(field.mask_for_grid([
            PHEROMONE_GRID_RES,
            PHEROMONE_GRID_RES,
            PHEROMONE_GRID_RES_Z,
        ]))
    });
    let vibration_mask = Some(field.mask_for_grid([
        VIBRATION_GRID_RES,
        VIBRATION_GRID_RES,
        VIBRATION_GRID_RES_Z,
    ]));
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
        let phero_mask_for_gpu = field.mask_for_grid([
            PHEROMONE_GRID_RES,
            PHEROMONE_GRID_RES,
            PHEROMONE_GRID_RES_Z,
        ]);
        let vib_mask_for_gpu = field.mask_for_grid([
            VIBRATION_GRID_RES,
            VIBRATION_GRID_RES,
            VIBRATION_GRID_RES_Z,
        ]);
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
}

#[derive(SystemParam)]
pub(super) struct RestartEntities<'w, 's> {
    pub(super) cells: Query<'w, 's, Entity, With<CellEntity>>,
    pub(super) foods: Query<'w, 's, Entity, With<FoodEntity>>,
    pub(super) spikes: Query<'w, 's, Entity, With<SpikeEntity>>,
    pub(super) menus: Query<'w, 's, Entity, With<GodMenuRoot>>,
}

/// `KeyR` restarts the simulation in place: despawn all cells / food / spikes,
/// reset sim state, re-seed INITIAL_CELLS fresh cells + initial food. Camera,
/// maze toggle, time speed, world map and event calendar are preserved.
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
    registries.next_id.0 = INITIAL_CELLS as u64;
    registries.slot_map.slot_to_entity.clear();
    registries.slot_map.entity_to_slot.clear();

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

    let mut rng = rand::rng();
    let mut initial_cells: Vec<Cell> = Vec::with_capacity(INITIAL_CELLS);
    for i in 0..INITIAL_CELLS {
        let cell = Cell::random(&mut rng, half, i as u64, 0, i as u64);
        let mat = adhesion_material(
            &mut registries.adhesion_materials,
            &mut registries.bio_materials,
            cell.genome.adhesion_type,
        );
        let entity = commands
            .spawn((
                CellEntity(cell),
                Mesh3d(assets.cell_mesh.0.clone()),
                MeshMaterial3d(mat),
                Transform::from_xyz(cell.position[0], cell.position[1], cell.position[2])
                    .with_rotation(cell_rotation(cell.heading, cell.pitch))
                    .with_scale(cell_scale(&cell.phenotype)),
            ))
            .id();
        for slot in 0..SPIKE_SLOTS as u8 {
            commands.spawn((
                SpikeEntity { owner: entity, slot },
                Mesh3d(assets.spike_mesh.0.clone()),
                MeshMaterial3d(assets.spike_material.0.clone()),
                Transform::default(),
                Visibility::Hidden,
            ));
        }
        registries.slot_map.allocate(entity);
        initial_cells.push(cell);
    }

    gpu_full
        .cells
        .upload_brains(initial_cells.iter().map(|c| &c.genome.brain));
    gpu_full
        .cells
        .upload_xoshiro_seeds(initial_cells.iter().map(|c| c.cell_id));
    let turn_rates: Vec<f32> = initial_cells.iter().map(|c| c.genome.turn_rate).collect();
    gpu_full.cells.upload_turn_rates(&turn_rates);
    gpu_full.cells.zero_persistent_state();

    let initial_food = food_target(&extent, 1.0);
    for _ in 0..initial_food {
        let mut food = Food::random(&mut rng, half);
        for _ in 0..MAX_SPAWN_ATTEMPTS {
            let richness = world_map_res.0.sample([food.position[0], food.position[1], 0.0]);
            if !reject_food_for_richness(&mut rng, richness) {
                break;
            }
            food = Food::random(&mut rng, half);
        }
        commands.spawn((
            FoodEntity(food),
            Mesh3d(assets.food_mesh.0.clone()),
            MeshMaterial3d(assets.food_material.0.clone()),
            Transform::from_xyz(food.position[0], food.position[1], food.position[2]),
            Visibility::Hidden,
        ));
    }

    info!("sim: restarted ({} cells, {} food)", INITIAL_CELLS, initial_food);
}
