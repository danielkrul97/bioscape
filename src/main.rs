use bevy::asset::RenderAssetUsages;
use bevy::diagnostic::{
    Diagnostic, DiagnosticPath, Diagnostics, DiagnosticsStore, FrameTimeDiagnosticsPlugin,
    LogDiagnosticsPlugin, RegisterDiagnostic,
};
use bevy::image::Image;
use bevy::input::mouse::{MouseMotion, MouseScrollUnit, MouseWheel};
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use std::time::Instant;

const DIAG_CELL_COUNT: DiagnosticPath = DiagnosticPath::const_new("sim/cell_count");
const DIAG_FOOD_COUNT: DiagnosticPath = DiagnosticPath::const_new("sim/food_count");
const DIAG_BRAIN_ACT: DiagnosticPath = DiagnosticPath::const_new("sim/brain_act_ms");
const DIAG_BRAIN_GPU_RT: DiagnosticPath = DiagnosticPath::const_new("sim/brain_gpu_rt_ms");
const DIAG_BROWNIAN: DiagnosticPath = DiagnosticPath::const_new("sim/brownian_ms");
const DIAG_BROWNIAN_GPU_RT: DiagnosticPath = DiagnosticPath::const_new("sim/brownian_gpu_rt_ms");
const DIAG_COLLISIONS: DiagnosticPath = DiagnosticPath::const_new("sim/collisions_ms");
const DIAG_PREDATION: DiagnosticPath = DiagnosticPath::const_new("sim/predation_ms");
const DIAG_EAT_FOOD: DiagnosticPath = DiagnosticPath::const_new("sim/eat_food_ms");
const DIAG_SMELL: DiagnosticPath = DiagnosticPath::const_new("sim/smell_field_ms");
const DIAG_PHEROMONE: DiagnosticPath = DiagnosticPath::const_new("sim/pheromone_field_ms");
const DIAG_GRID_REBUILD: DiagnosticPath = DiagnosticPath::const_new("sim/grid_rebuild_ms");
const DIAG_SYNC_TRANSFORMS: DiagnosticPath = DiagnosticPath::const_new("sim/sync_transforms_ms");
const DIAG_TICKS_PER_FRAME: DiagnosticPath = DiagnosticPath::const_new("sim/ticks_per_frame");
const DIAG_RENDER_OVERHEAD: DiagnosticPath = DiagnosticPath::const_new("sim/render_overhead_ms");
use bioscape::{
    reject_food_for_richness, Cell, Food, Phenotype, SimClock, SmellField, SpatialGrid, WorldMap,
    ATTACK_THRESHOLD, CARRION_FOOD_COUNT, CELL_RADIUS, CYCLE_AMPLITUDE, CYCLE_GEN_PERIOD,
    DILUTION_K, EAT_RADIUS, FIXED_TIMESTEP_HZ, FOOD_SPAWN_RATE, FOOD_VALUE, GENERATIONS_PER_EPOCH,
    HAZARD_AMP, HAZARD_DRAIN_PER_SEC, HAZARD_FLOOR, HERD_RADIUS, INITIAL_CELLS, LEARNING_RATE,
    MATING_COOLDOWN_TICKS, MATING_PHEROMONE_THRESHOLD, MATING_RADIUS, MAX_BODY_LENGTH,
    MAX_POPULATION, MAX_SPAWN_ATTEMPTS,
    PHEROMONE_BASELINE_EMIT, PHEROMONE_BRAIN_MOD, PHEROMONE_COST_PER_RATE, PHEROMONE_DECAY,
    PHEROMONE_DIFFUSION, PHEROMONE_GRID_RES, PHEROMONE_GRID_RES_Z, PHEROMONE_SAMPLE_EPSILON,
    PHYSICS_CONFIG, PREDATION_DRAIN_PER_TICK, PREDATION_GAIN_PER_TICK, REPRODUCE_THRESHOLD,
    SIZE_RATIO_THRESHOLD, SMELL_DECAY, SMELL_DIFFUSION, SMELL_GRID_RES, SMELL_GRID_RES_Z,
    SMELL_PER_FOOD, SMELL_SAMPLE_EPSILON, THERMAL_NOISE, TICKS_PER_GENERATION,
    WORLD_MAP_BASE_RES, WORLD_MAP_BASE_RES_Z, WORLD_MAP_FOOD_AMP, WORLD_MAP_FOOD_FLOOR,
    WORLD_MAP_RES, WORLD_MAP_RES_Z, WORLD_MAP_SEED, WORLD_UNITS_PER_FOOD, BRAIN_INPUTS,
};
#[cfg(feature = "gpu")]
use bioscape::gpu::{BrainGpu, BrownianGpu, CellsGpu, FieldGpu, GpuContext, HebbianGpu};
use rand::Rng;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use std::time::Duration;

#[derive(Resource, Default)]
struct TickCounter {
    ticks_this_frame: u32,
    sim_ms_this_frame: f64,
    tick_start: Option<Instant>,
}

// Renderer-only knobs. Sim parameters live in `bioscape` (lib.rs).
// Sprint 53: zmenšeno z 2.5 (Sprint 53 volumetric expansion 10× food count
// dělalo 2.5 mesh visuálně dominantní).
const FOOD_RADIUS: f32 = 1.0;
const DEATH_FADE_TICKS: u32 = 30;
const GRID_CELL_SIZE: f32 = 100.0;
const CAMERA_ZOOM_STEP: f32 = 0.1;
// Sprint 53: WORLD_HALF[2] expanded z=2 → z=20. Volumetric environment.
const SIMULATION_HALF: [f32; 3] = [960.0, 540.0, 20.0];
// Sprint 36: orbit Camera3d s ORTHOGRAPHIC projection. Distance je fixní;
// "zoom" modifikuje ortho scale (= world units per pixel), takže větší zoom
// out neudělá black void kolem scény (na rozdíl od perspective). Cells stále
// vypadají jako 3D body díky lighting + tilted angle, jen bez perspective
// foreshortening.
/// Fixní vzdálenost camera od target. Pro ortho neovlivňuje velikost cells,
/// jen znear/zfar clipping plane positioning. 3000 dává dostatek depth bufferu.
const CAMERA_OFFSET_DISTANCE: f32 = 3000.0;
const CAMERA_PITCH_INITIAL: f32 = 0.95; // ~55° from xy plane
/// Ortho scale (Bevy `OrthographicProjection.scale`): 1 world unit = 1 / scale
/// pixelů. Initial 1.2 dává mírný margin kolem world bounds (1920×1080 přesně
/// padne při scale=1.0, +20 % je rezerva pro tilted view).
const CAMERA_SCALE_INITIAL: f32 = 1.2;
const CAMERA_SCALE_MIN: f32 = 0.2; // hluboký zoom in (~6× větší cells)
const CAMERA_SCALE_MAX: f32 = 2.0; // limit zoom out — vždy dohlédne ke kraji world
/// Pitch clamp tight near ±π/2 — `looking_at` s up vektorem +Z degeneruje při
/// pohledu kolmo dolů. 0.05 rad ≈ 2.9° margin.
const CAMERA_PITCH_MIN: f32 = 0.05;
const CAMERA_PITCH_MAX: f32 = std::f32::consts::FRAC_PI_2 - 0.05;
/// Mouse drag → orbit angle delta. Tuned pro 1080p screen — full screen drag
/// = ~π rotace.
const ORBIT_SENSITIVITY: f32 = 0.005;

#[derive(Component)]
struct CellEntity(Cell);

#[derive(Component)]
struct FoodEntity(Food);

#[derive(Component)]
struct Dying {
    ticks_left: u32,
}

#[derive(Component)]
struct StatsRoot;

#[derive(Component)]
struct StatsText;

#[derive(Resource, Debug, Clone, Copy)]
struct WorldExtent {
    half_x: f32,
    half_y: f32,
    half_z: f32,
}

impl WorldExtent {
    fn as_array(self) -> [f32; 3] {
        [self.half_x, self.half_y, self.half_z]
    }
}

#[derive(Resource, Debug)]
struct Clock(SimClock);

#[derive(Resource, Debug, Clone, Copy)]
struct FoodDensityFactor(f32);

impl Default for FoodDensityFactor {
    fn default() -> Self {
        Self(1.0)
    }
}

#[derive(Resource)]
struct CellGrid(SpatialGrid<Entity, f32>);

impl Default for CellGrid {
    fn default() -> Self {
        Self(SpatialGrid::new(GRID_CELL_SIZE))
    }
}

#[derive(Resource)]
struct FoodGrid(SpatialGrid<Entity, ()>);

impl Default for FoodGrid {
    fn default() -> Self {
        Self(SpatialGrid::new(GRID_CELL_SIZE))
    }
}

/// Sprint 52: GPU compute state pro renderer. Drží persistent CellsGpu +
/// BrainGpu/HebbianGpu/BrownianGpu na shared GpuContext. Insert se v `setup`
/// pokud GpuContext::new uspěje; pokud selže, Resource zůstává `None` a
/// systems gracefully fallbacknou na CPU.
#[cfg(feature = "gpu")]
#[derive(Resource)]
struct GpuBrainState {
    cells: CellsGpu,
    brain: BrainGpu,
    hebbian: HebbianGpu,
    brownian: BrownianGpu,
}

/// Sprint 59: separate Resource pro GPU smell + pheromone fields. Oddělen od
/// `GpuBrainState` aby `update_smell_field` (ResMut<GpuFieldState>) nesoutěžil
/// s ostatními systemy o brain access. Insert v setup pokud GpuContext init
/// uspěje; jinak CPU SmellResource path drží.
#[cfg(feature = "gpu")]
#[derive(Resource)]
struct GpuFieldState {
    smell: FieldGpu,
    pheromone: FieldGpu,
}

/// Sprint 52: maps Bevy `Entity` ↔ slot index v `CellsGpu` SoA bufferech.
/// Sloty jsou dense (0..n, žádné holes) přes swap_remove pattern při death.
#[derive(Resource, Default)]
struct CellSlotMap {
    slot_to_entity: Vec<Entity>,
    entity_to_slot: FxHashMap<Entity, usize>,
}

impl CellSlotMap {
    fn allocate(&mut self, entity: Entity) -> usize {
        let slot = self.slot_to_entity.len();
        self.slot_to_entity.push(entity);
        self.entity_to_slot.insert(entity, slot);
        slot
    }

    /// Release slot pro entity. Vrací `Some((freed_slot, moved_entity))`
    /// pokud entity byla zaregistrovaná. `moved_entity` je Some pokud
    /// freed_slot byl zaplněn cell ze zadního slotu (swap_remove pattern).
    fn release(&mut self, entity: Entity) -> Option<(usize, Option<Entity>)> {
        let slot = self.entity_to_slot.remove(&entity)?;
        let last = self.slot_to_entity.len() - 1;
        let moved = if slot != last {
            let moved_entity = self.slot_to_entity[last];
            self.slot_to_entity[slot] = moved_entity;
            self.entity_to_slot.insert(moved_entity, slot);
            Some(moved_entity)
        } else {
            None
        };
        self.slot_to_entity.pop();
        Some((slot, moved))
    }

    fn slot_of(&self, entity: Entity) -> Option<usize> {
        self.entity_to_slot.get(&entity).copied()
    }

    fn len(&self) -> usize {
        self.slot_to_entity.len()
    }
}

#[derive(Resource)]
struct CellMesh(Handle<Mesh>);

#[derive(Resource)]
struct FoodMesh(Handle<Mesh>);

#[derive(Resource)]
struct FoodMaterial(Handle<StandardMaterial>);

/// Sprint 36: per-lineage material cache. Lineage hue → handle do
/// `Assets<StandardMaterial>`. Bevy automaticky deduplikuje stejné materialy
/// na renderer instances draw call.
#[derive(Resource, Default)]
struct LineageMaterials(FxHashMap<u64, Handle<StandardMaterial>>);

/// Sprint 36 orbit camera state. Camera obíhá kolem `target` ve sférických
/// souřadnicích (yaw + pitch). Distance camera→target je fixní
/// `CAMERA_OFFSET_DISTANCE`; "zoom" modifikuje `scale` (orthographic projection
/// scale). Yaw = rotace kolem world Z, pitch = elevace nad xy plochou
/// (0 = horizon, π/2 = top-down).
#[derive(Resource, Debug, Clone, Copy)]
struct OrbitCamera {
    target: Vec3,
    yaw: f32,
    pitch: f32,
    /// Orthographic scale (world units per pixel). Menší = zoom in.
    scale: f32,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self {
            target: Vec3::ZERO,
            yaw: 0.0,
            pitch: CAMERA_PITCH_INITIAL,
            scale: CAMERA_SCALE_INITIAL,
        }
    }
}

impl OrbitCamera {
    fn transform(&self) -> Transform {
        let cos_p = self.pitch.cos();
        let offset = Vec3::new(
            -self.yaw.sin() * cos_p,
            -self.yaw.cos() * cos_p,
            self.pitch.sin(),
        ) * CAMERA_OFFSET_DISTANCE;
        let pos = self.target + offset;
        Transform::from_translation(pos).looking_at(self.target, Vec3::Z)
    }
}

#[derive(Resource)]
struct SmellResource(SmellField);

#[derive(Resource)]
struct PheromoneResource(SmellField);

#[derive(Resource)]
struct WorldMapResource(WorldMap);

#[derive(Component)]
struct WorldMapOverlay;

#[derive(Message, Debug, Clone, Copy)]
struct GenerationEnded {
    generation: u64,
}

#[derive(Message, Debug, Clone, Copy)]
struct EpochEnded {
    epoch: u64,
}

fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins.set(WindowPlugin {
                primary_window: Some(Window {
                    title: "Bioscape".into(),
                    ..default()
                }),
                ..default()
            }),
            FrameTimeDiagnosticsPlugin::default(),
            LogDiagnosticsPlugin::default(),
        ))
        .register_diagnostic(Diagnostic::new(DIAG_CELL_COUNT).with_suffix(" cells"))
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
        .register_diagnostic(Diagnostic::new(DIAG_RENDER_OVERHEAD).with_suffix(" ms"))
        .init_resource::<TickCounter>()
        // Sprint 36: clear color matchnut s HIGH richness color z `world_map_image`
        // (rich zones jsou bílé, poor zelené). Margins jsou bílé.
        .insert_resource(ClearColor(Color::WHITE))
        .init_resource::<LineageMaterials>()
        .init_resource::<OrbitCamera>()
        .insert_resource(Time::<Fixed>::from_hz(FIXED_TIMESTEP_HZ as f64))
        .insert_resource(Clock(SimClock::new(
            TICKS_PER_GENERATION,
            GENERATIONS_PER_EPOCH,
        )))
        .init_resource::<CellGrid>()
        .init_resource::<FoodGrid>()
        .init_resource::<FoodDensityFactor>()
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
                    update_smell_field,
                    update_pheromone_field,
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
                log_clock_events,
                toggle_stats_overlay,
                toggle_world_map_overlay,
                update_stats_overlay,
                report_frame_diagnostics,
            ),
        )
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
    mut lineage_materials: ResMut<LineageMaterials>,
    mut window: Single<&mut Window>,
) {
    window.set_maximized(true);
    let half = SIMULATION_HALF;
    let extent = WorldExtent {
        half_x: half[0],
        half_y: half[1],
        half_z: half[2],
    };
    commands.insert_resource(extent);

    // Sprint 36: Camera3d s orthographic projection — "scale" zoom feel bez
    // perspective void okolo scény. `IsDefaultUiCamera` marker říká
    // bevy_ui_render ať použije tuto kameru pro UI.
    // Near/far explicitně dimenzované na CAMERA_OFFSET_DISTANCE — default_3d()
    // má far ~1000, ale camera je 3000 od target, takže by scéna padla za far
    // plane a vše by bylo culled.
    let initial_orbit = OrbitCamera::default();
    commands.spawn((
        Camera3d::default(),
        IsDefaultUiCamera,
        Projection::Orthographic(OrthographicProjection {
            scale: initial_orbit.scale,
            near: 0.1,
            far: CAMERA_OFFSET_DISTANCE * 3.0,
            ..OrthographicProjection::default_3d()
        }),
        initial_orbit.transform(),
    ));

    // Ambient + DirectionalLight pro 3D scénu. Bevy 0.18 typické hodnoty:
    // ambient ~1000-2000, directional ~10000+ pro outdoor scénu.
    commands.spawn(AmbientLight {
        color: Color::WHITE,
        brightness: 1500.0,
        ..default()
    });
    commands.spawn((
        DirectionalLight {
            illuminance: 12000.0,
            shadows_enabled: false,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(
            EulerRot::XYZ,
            -std::f32::consts::FRAC_PI_4,
            std::f32::consts::FRAC_PI_6,
            0.0,
        )),
    ));

    // Sprint 53: WorldMap + SmellField/Pheromone plně 3D volumetric.
    let world_map = WorldMap::new(
        [WORLD_MAP_RES, WORLD_MAP_RES, WORLD_MAP_RES_Z],
        [WORLD_MAP_BASE_RES, WORLD_MAP_BASE_RES, WORLD_MAP_BASE_RES_Z],
        half,
        WORLD_MAP_SEED,
    );

    // Sprint 36: WorldMap overlay jako ground plane na z=-half_z-5 (pod cells).
    // Texture je grayscale richness; v 3D pohledu funguje jako "podlaha" světa.
    let overlay_image_handle = images.add(world_map_image(&world_map));
    let overlay_material = materials.add(StandardMaterial {
        base_color_texture: Some(overlay_image_handle),
        unlit: true,
        ..default()
    });
    let overlay_mesh =
        meshes.add(Plane3d::default().mesh().size(2.0 * half[0], 2.0 * half[1]));
    commands.spawn((
        Mesh3d(overlay_mesh),
        MeshMaterial3d(overlay_material),
        Transform::from_xyz(0.0, 0.0, -half[2] - 5.0)
            // Plane3d defaultně leží v xz; rotujem do xy aby normála ukazovala +z.
            .with_rotation(Quat::from_rotation_x(std::f32::consts::FRAC_PI_2)),
        WorldMapOverlay,
    ));

    // Sprint 36: cell mesh = unit-radius sphere, scale aplikuje ellipsoid
    // (length × width × height) per cell. Spike rendering vynechán (visual
    // loss; predace mechanika beze změny).
    let cell_mesh_handle = meshes.add(Sphere::new(CELL_RADIUS).mesh().ico(2).unwrap());
    // Sprint 53: jídlo decentnější — menší radius (10× větší food count po
    // 3D volume scaling jinak vytváří plný display) + ground-matching tint
    // (low-saturation green) místo skoro-černé proti bílému ClearColoru.
    let food_mesh_handle = meshes.add(Sphere::new(FOOD_RADIUS).mesh().ico(1).unwrap());
    let food_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.20, 0.30, 0.18),
        ..default()
    });

    let mut rng = rand::rng();
    let mut initial_cells: Vec<Cell> = Vec::with_capacity(INITIAL_CELLS);
    let mut slot_map = CellSlotMap::default();
    for i in 0..INITIAL_CELLS {
        let cell = Cell::random(&mut rng, half, i as u64, 0);
        let mat = lineage_material(&mut lineage_materials, &mut materials, cell.lineage_id);
        let entity = commands
            .spawn((
                CellEntity(cell),
                Mesh3d(cell_mesh_handle.clone()),
                MeshMaterial3d(mat),
                Transform::from_xyz(cell.position[0], cell.position[1], cell.position[2])
                    .with_rotation(cell_rotation(cell.heading, cell.pitch))
                    .with_scale(cell_scale(&cell.phenotype)),
            ))
            .id();
        slot_map.allocate(entity);
        initial_cells.push(cell);
    }

    // Sprint 52: GPU compute init. Při failu fallback na CPU (Resource None).
    #[cfg(feature = "gpu")]
    {
        // Capacity = MAX_POPULATION + slack pro birth ticks (Dying entities
        // už slot drží do despawn — ale po Sprint 52 pattern se uvolňují
        // ihned na cell_dies; slack pokrývá race window).
        let cap = MAX_POPULATION + 64;
        // Sprint 59: FieldGpu sources capacity. Per-tick deposit count =
        // foods (smell) + cells (pheromone). Upper bound přes density cycle peak.
        let initial_food_target = food_target(&extent, 1.0 + CYCLE_AMPLITUDE);
        let field_sources_cap = (initial_food_target + cap) * 2;
        let world_half = extent.as_array();
        let init = || -> Result<(GpuBrainState, GpuFieldState), String> {
            let ctx = GpuContext::new()?;
            let cells = CellsGpu::with_context(&ctx, cap);
            cells.upload_brains(initial_cells.iter().map(|c| &c.genome.brain));
            cells.upload_xoshiro_seeds(initial_cells.iter().enumerate().map(|(slot, c)| {
                c.lineage_id ^ (slot as u64).wrapping_mul(0x9E3779B97F4A7C15)
            }));
            let brain = BrainGpu::with_context(&ctx, cap)?;
            let hebbian = HebbianGpu::with_context(&ctx, cap)?;
            let brownian = BrownianGpu::with_context(&ctx, cap)?;
            let smell = FieldGpu::with_context(
                &ctx,
                [SMELL_GRID_RES, SMELL_GRID_RES, SMELL_GRID_RES_Z],
                world_half,
                field_sources_cap,
            )?;
            let pheromone = FieldGpu::with_context(
                &ctx,
                [PHEROMONE_GRID_RES, PHEROMONE_GRID_RES, PHEROMONE_GRID_RES_Z],
                world_half,
                field_sources_cap,
            )?;
            Ok((
                GpuBrainState { cells, brain, hebbian, brownian },
                GpuFieldState { smell, pheromone },
            ))
        };
        match init() {
            Ok((brain_state, field_state)) => {
                info!(
                    "renderer-gpu: persistent brain weights + Hebbian + Brownian + Field (cap {} cells, {} field sources)",
                    cap, field_sources_cap
                );
                commands.insert_resource(brain_state);
                commands.insert_resource(field_state);
            }
            Err(e) => {
                warn!("renderer-gpu: init failed ({}); falling back to CPU compute", e);
            }
        }
    }
    commands.insert_resource(slot_map);
    let _ = initial_cells;
    let initial_food = food_target(&extent, 1.0);
    for _ in 0..initial_food {
        let mut food = Food::random(&mut rng, half);
        for _ in 0..MAX_SPAWN_ATTEMPTS {
            let richness = world_map.sample([food.position[0], food.position[1], 0.0]);
            if !reject_food_for_richness(&mut rng, richness) {
                break;
            }
            food = Food::random(&mut rng, half);
        }
        commands.spawn((
            FoodEntity(food),
            Mesh3d(food_mesh_handle.clone()),
            MeshMaterial3d(food_material.clone()),
            Transform::from_xyz(food.position[0], food.position[1], food.position[2]),
        ));
    }

    commands.insert_resource(CellMesh(cell_mesh_handle));
    commands.insert_resource(FoodMesh(food_mesh_handle));
    commands.insert_resource(FoodMaterial(food_material));
    commands.insert_resource(SmellResource(SmellField::new(
        [SMELL_GRID_RES, SMELL_GRID_RES, SMELL_GRID_RES_Z],
        half,
    )));
    commands.insert_resource(PheromoneResource(SmellField::new(
        [PHEROMONE_GRID_RES, PHEROMONE_GRID_RES, PHEROMONE_GRID_RES_Z],
        half,
    )));
    commands.insert_resource(WorldMapResource(world_map));
}

/// Sprint 36: vrátí (případně vytvoří) StandardMaterial handle pro daný
/// lineage_id. Hue mapuje deterministicky přes `lineage_hue`. Cache zaručuje,
/// že cells se stejným lineage sdílejí jeden material — Bevy je instance
/// podle materialu pro draw call binning, takže shared material = 1 batch.
fn lineage_material(
    cache: &mut LineageMaterials,
    materials: &mut Assets<StandardMaterial>,
    lineage_id: u64,
) -> Handle<StandardMaterial> {
    if let Some(h) = cache.0.get(&lineage_id) {
        return h.clone();
    }
    let hue = lineage_hue(lineage_id);
    let color = Color::hsl(hue, 0.75, 0.55);
    let handle = materials.add(StandardMaterial {
        base_color: color,
        perceptual_roughness: 0.6,
        ..default()
    });
    cache.0.insert(lineage_id, handle.clone());
    handle
}

/// Sprint 36: Quat z yaw + pitch pro orientaci ellipsoidu. Body's local +X
/// musí mířit ve forward direction = (cos(y)cos(p), sin(y)cos(p), sin(p)).
/// Quat::from_rotation_z(yaw) * Quat::from_rotation_y(pitch) splňuje
/// (1,0,0) → forward (viz `bioscape::forward_vector`).
fn cell_rotation(yaw: f32, pitch: f32) -> Quat {
    Quat::from_rotation_z(yaw) * Quat::from_rotation_y(-pitch)
}

fn world_map_image(map: &WorldMap) -> Image {
    // Sprint 53: WorldMap je 3D. Ground plane overlay vykreslí xy-slice na
    // z = floor(nz/2) (canonical surface layer); food spawn taktéž samples
    // z=0 svět ⇒ middle z-slice.
    let nx = map.resolution[0];
    let ny = map.resolution[1];
    let nz = map.resolution[2];
    let z_slice = nz / 2;
    let mut data = Vec::with_capacity(nx * ny * 4);
    let low = [0.10_f32, 0.42, 0.12];
    let high = [1.00_f32, 1.00, 1.00];
    let field = map.field();
    let plane = nx * ny;
    for j in 0..ny {
        for i in 0..nx {
            let v = field[z_slice * plane + j * nx + i];
            let t = v.clamp(0.0, 1.0);
            let r = ((low[0] + t * (high[0] - low[0])) * 255.0).clamp(0.0, 255.0) as u8;
            let g = ((low[1] + t * (high[1] - low[1])) * 255.0).clamp(0.0, 255.0) as u8;
            let b = ((low[2] + t * (high[2] - low[2])) * 255.0).clamp(0.0, 255.0) as u8;
            data.push(r);
            data.push(g);
            data.push(b);
            data.push(255);
        }
    }
    Image::new(
        Extent3d {
            width: nx as u32,
            height: ny as u32,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        data,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::RENDER_WORLD,
    )
}

fn food_multiplier(noise: f32) -> f32 {
    WORLD_MAP_FOOD_FLOOR + WORLD_MAP_FOOD_AMP * noise
}

fn hazard_drain(noise: f32) -> f32 {
    HAZARD_DRAIN_PER_SEC * (HAZARD_FLOOR + HAZARD_AMP * noise)
}

fn food_target(extent: &WorldExtent, factor: f32) -> usize {
    // Sprint 53: scale s 3D objemem (mirror headless logiky).
    let area = (2.0 * extent.half_x) * (2.0 * extent.half_y);
    let z_extent = 2.0 * extent.half_z;
    let z_factor = (z_extent / 4.0).max(1.0);
    ((area / WORLD_UNITS_PER_FOOD) * factor.max(0.0) * z_factor) as usize
}

/// Stable u64 → hue mapping for lineage visualization. Knuth-style integer
/// hash mixing — short, no allocation, decent distribution across [0, 360).
fn lineage_hue(id: u64) -> f32 {
    let h = id.wrapping_mul(2654435761).wrapping_add(id >> 16);
    ((h % 360) as f32).rem_euclid(360.0)
}

fn update_food_density_cycle(
    mut events: MessageReader<GenerationEnded>,
    clock: Res<Clock>,
    mut factor: ResMut<FoodDensityFactor>,
) {
    if events.read().next().is_none() {
        return;
    }
    let phase = (clock.0.generation as f32 / CYCLE_GEN_PERIOD as f32) * std::f32::consts::TAU;
    factor.0 = 1.0 + CYCLE_AMPLITUDE * phase.sin();
}

/// Cap virtual-time delta na 50 ms — limit catch-up FixedUpdate ticků (~4 při
/// 60 Hz) po lag spike. Default 250 ms (Bevy's `DEFAULT_MAX_DELTA`) by povolil
/// 15+ ticků a exponenciálně by dohánělo zpoždění (death spiral). Sim po lagu
/// poběží pomaleji než real time, ale zotaví se.
///
/// Musí běžet jako Startup systém přes ResMut, ne přes `insert_resource` v
/// `App` builderu — `DefaultPlugins.build()` přepíše Time<Virtual> až po
/// našem `insert_resource`, takže ten by se ztratil.
fn setup_time_cap(mut virtual_time: ResMut<Time<Virtual>>) {
    virtual_time.set_max_delta(Duration::from_millis(50));
    info!(
        "Time<Virtual>::max_delta capped to {:?}",
        virtual_time.max_delta()
    );
}

fn tick_start(mut counter: ResMut<TickCounter>) {
    counter.tick_start = Some(Instant::now());
}

fn tick_end(mut counter: ResMut<TickCounter>) {
    if let Some(t0) = counter.tick_start.take() {
        counter.sim_ms_this_frame += t0.elapsed().as_secs_f64() * 1000.0;
    }
    counter.ticks_this_frame += 1;
}

fn report_frame_diagnostics(
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

fn advance_clock(
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

fn step_cells(
    time: Res<Time>,
    extent: Res<WorldExtent>,
    mut cells: Query<&mut CellEntity>,
) {
    let dt = time.delta_secs();
    let half = extent.as_array();
    for mut cell in &mut cells {
        cell.0.step(dt, half, &PHYSICS_CONFIG);
    }
}

/// Sprint 42: Brownův pohyb — gaussian perturbation na velocity.
/// Sprint 52: pokud `GpuBrainState` Resource available, dispatch GPU brownian
/// (xoshiro128++ per-cell). CPU fallback při absenci GPU.
fn apply_brownian_motion(
    time: Res<Time>,
    extent: Res<WorldExtent>,
    slot_map: Res<CellSlotMap>,
    #[cfg(feature = "gpu")] gpu_state: Option<Res<GpuBrainState>>,
    mut cells: Query<(Entity, &mut CellEntity), Without<Dying>>,
    mut diag: Diagnostics,
) {
    let t_total = Instant::now();
    let dt = time.delta_secs();
    let half_z = extent.as_array()[2];

    #[cfg(feature = "gpu")]
    if let Some(gpu) = gpu_state {
        let n = slot_map.len();
        if n == 0 {
            diag.add_measurement(&DIAG_BROWNIAN, || t_total.elapsed().as_secs_f64() * 1000.0);
            return;
        }
        let mut velocities_by_slot: Vec<[f32; 3]> = vec![[0.0; 3]; n];
        for (entity, cell) in cells.iter() {
            if let Some(slot) = slot_map.slot_of(entity) {
                velocities_by_slot[slot] = cell.0.velocity;
            }
        }
        let t_gpu = Instant::now();
        gpu.cells.upload_velocities(&velocities_by_slot);
        gpu.brownian
            .compute_persistent(&gpu.cells, n, THERMAL_NOISE, dt, half_z > 0.0);
        let new_vels = gpu.cells.download_velocities(n);
        diag.add_measurement(&DIAG_BROWNIAN_GPU_RT, || t_gpu.elapsed().as_secs_f64() * 1000.0);
        for (entity, mut cell) in &mut cells {
            if let Some(slot) = slot_map.slot_of(entity) {
                cell.0.velocity = new_vels[slot];
            }
        }
        diag.add_measurement(&DIAG_BROWNIAN, || t_total.elapsed().as_secs_f64() * 1000.0);
        return;
    }

    let _ = slot_map;
    let mut rng = rand::rng();
    for (_, mut cell) in &mut cells {
        cell.0.apply_brownian(&mut rng, dt, half_z);
    }
    diag.add_measurement(&DIAG_BROWNIAN, || t_total.elapsed().as_secs_f64() * 1000.0);
}

/// Sprint 38: gravity drift na food. Aktualizuje Food.position[2] + sync
/// Transform.translation.z aby viditelně klesalo k dnu.
fn apply_food_gravity(
    time: Res<Time>,
    extent: Res<WorldExtent>,
    mut foods: Query<(Entity, &mut FoodEntity, &mut Transform)>,
    mut commands: Commands,
) {
    let dt = time.delta_secs();
    let half_z = extent.as_array()[2];
    for (entity, mut food, mut transform) in &mut foods {
        food.0.apply_gravity(dt, half_z);
        transform.translation.z = food.0.position[2];
        // Sprint 42: increment age + despawn expired (value_factor ≤ 0).
        if !food.0.age_step() {
            commands.entity(entity).despawn();
        }
    }
}

fn apply_environmental_hazards(
    time: Res<Time>,
    world_map: Res<WorldMapResource>,
    mut cells: Query<&mut CellEntity, Without<Dying>>,
) {
    let dt = time.delta_secs();
    for mut cell in &mut cells {
        let noise = world_map
            .0
            .sample([cell.0.position[0], cell.0.position[1], cell.0.position[2]]);
        let drain = hazard_drain(noise) * dt;
        cell.0.energy -= drain;
        cell.0.damage_accum += drain;
    }
}

fn update_smell_field(
    time: Res<Time>,
    foods: Query<&FoodEntity>,
    mut smell: ResMut<SmellResource>,
    #[cfg(feature = "gpu")] gpu_field: Option<ResMut<GpuFieldState>>,
    mut diag: Diagnostics,
) {
    let t = Instant::now();
    let dt = time.delta_secs();
    diag.add_measurement(&DIAG_FOOD_COUNT, || foods.iter().count() as f64);

    // Sprint 59: pokud GpuFieldState available, GPU deposit + diffuse, readback
    // do CPU SmellResource pro sensor gather (gradient_at v cells_brain_act).
    #[cfg(feature = "gpu")]
    if let Some(mut gpu) = gpu_field {
        for food in &foods {
            gpu.smell.add_source(
                [food.0.position[0], food.0.position[1], food.0.position[2]],
                SMELL_PER_FOOD * dt,
            );
        }
        gpu.smell.step(SMELL_DIFFUSION, SMELL_DECAY, dt);
        let grid = gpu.smell.download();
        smell.0.replace_grid_from(&grid);
        diag.add_measurement(&DIAG_SMELL, || t.elapsed().as_secs_f64() * 1000.0);
        return;
    }

    for food in &foods {
        smell
            .0
            .add_source([food.0.position[0], food.0.position[1], food.0.position[2]], SMELL_PER_FOOD * dt);
    }
    smell.0.step(SMELL_DIFFUSION, SMELL_DECAY, dt);
    diag.add_measurement(&DIAG_SMELL, || t.elapsed().as_secs_f64() * 1000.0);
}

fn update_pheromone_field(
    time: Res<Time>,
    mut pheromone: ResMut<PheromoneResource>,
    #[cfg(feature = "gpu")] gpu_field: Option<ResMut<GpuFieldState>>,
    mut diag: Diagnostics,
) {
    // Diffuse + decay BEFORE this tick's emissions (in emit_pheromones, which
    // runs after brain_act). Stejně jako headless — brainy detekují gradient
    // ze stavu pole na konci minulého ticku, žádný self-feedback.
    let t = Instant::now();
    let dt = time.delta_secs();

    // Sprint 59: GPU step + readback pokud field state available.
    #[cfg(feature = "gpu")]
    if let Some(mut gpu) = gpu_field {
        gpu.pheromone.step(PHEROMONE_DIFFUSION, PHEROMONE_DECAY, dt);
        let grid = gpu.pheromone.download();
        pheromone.0.replace_grid_from(&grid);
        diag.add_measurement(&DIAG_PHEROMONE, || t.elapsed().as_secs_f64() * 1000.0);
        return;
    }

    pheromone.0.step(PHEROMONE_DIFFUSION, PHEROMONE_DECAY, dt);
    diag.add_measurement(&DIAG_PHEROMONE, || t.elapsed().as_secs_f64() * 1000.0);
}

fn emit_pheromones(
    time: Res<Time>,
    mut pheromone: ResMut<PheromoneResource>,
    #[cfg(feature = "gpu")] gpu_field: Option<ResMut<GpuFieldState>>,
    mut cells: Query<&mut CellEntity, Without<Dying>>,
) {
    let dt = time.delta_secs();

    // Sprint 59: deposit do GPU pending_sources (flushed v dalším ticku
    // update_pheromone_field step). CPU pheromone.grid není updated — sensor
    // gather už proběhl s pre-emission stavem.
    #[cfg(feature = "gpu")]
    if let Some(mut gpu) = gpu_field {
        for mut cell in &mut cells {
            let mod_strength = cell.0.last_outputs[2].max(0.0);
            let brain_emit = PHEROMONE_BRAIN_MOD * mod_strength;
            let rate = PHEROMONE_BASELINE_EMIT + brain_emit;
            gpu.pheromone.add_source(
                [cell.0.position[0], cell.0.position[1], cell.0.position[2]],
                rate * dt,
            );
            cell.0.energy -= PHEROMONE_COST_PER_RATE * brain_emit * dt;
        }
        return;
    }

    for mut cell in &mut cells {
        let mod_strength = cell.0.last_outputs[2].max(0.0);
        let brain_emit = PHEROMONE_BRAIN_MOD * mod_strength;
        let rate = PHEROMONE_BASELINE_EMIT + brain_emit;
        pheromone
            .0
            .add_source([cell.0.position[0], cell.0.position[1], cell.0.position[2]], rate * dt);
        cell.0.energy -= PHEROMONE_COST_PER_RATE * brain_emit * dt;
    }
}

fn cells_brain_act(
    time: Res<Time>,
    cell_grid: Res<CellGrid>,
    food_grid: Res<FoodGrid>,
    smell: Res<SmellResource>,
    pheromone: Res<PheromoneResource>,
    slot_map: Res<CellSlotMap>,
    #[cfg(feature = "gpu")] gpu_state: Option<Res<GpuBrainState>>,
    mut cells: Query<(Entity, &mut CellEntity), Without<Dying>>,
    mut diag: Diagnostics,
) {
    let _t_total = Instant::now();
    let dt = time.delta_secs();
    diag.add_measurement(&DIAG_CELL_COUNT, || cells.iter().count() as f64);

    // Sprint 52: helper closure pro per-cell sensor gather + populate_brain_inputs.
    // Reused jak v CPU tak GPU path. Takes &mut Cell + Entity, vrací inputs[36].
    let gather = |entity: Entity, cell: &mut Cell| -> [f32; BRAIN_INPUTS] {
        let pos = cell.position;
        let vision_r = cell.genome.vision_radius;
        let vr2 = vision_r * vision_r;
        let mut nearest_food: Option<[f32; 3]> = None;
        let mut best_food_d2 = f32::MAX;
        food_grid.0.for_each_in_radius_toroidal(pos, vision_r, SIMULATION_HALF, |_, fp, _| {
            let d = bioscape::min_image_delta(pos, fp, SIMULATION_HALF);
            let d2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            if d2 <= vr2 && d2 < best_food_d2 {
                best_food_d2 = d2;
                nearest_food = Some(d);
            }
        });
        let mut nearest_cell: Option<([f32; 3], f32)> = None;
        let mut best_cell_d2 = f32::MAX;
        let mut neighbors_in_vision: u32 = 0;
        cell_grid
            .0
            .for_each_in_radius_toroidal(pos, vision_r, SIMULATION_HALF, |other, other_pos, other_radius| {
                if other == entity {
                    return;
                }
                let d = bioscape::min_image_delta(pos, other_pos, SIMULATION_HALF);
                let d2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                if d2 <= vr2 {
                    neighbors_in_vision += 1;
                    if d2 < best_cell_d2 {
                        best_cell_d2 = d2;
                        nearest_cell = Some((d, other_radius));
                    }
                }
            });
        let pos_xyz = [pos[0], pos[1], pos[2]];
        let smell_grad = smell.0.gradient_at(pos_xyz, SMELL_SAMPLE_EPSILON);
        let pheromone_grad = pheromone.0.gradient_at(pos_xyz, PHEROMONE_SAMPLE_EPSILON);
        let sensors = bioscape::BrainSensors {
            nearest_food,
            nearest_cell,
            neighbors_in_vision,
            smell_grad,
            pheromone_grad,
        };
        cell.apply_shell_absorb(dt);
        bioscape::populate_brain_inputs(cell, &sensors, vision_r)
    };

    #[cfg(feature = "gpu")]
    if let Some(gpu) = gpu_state {
        let n = slot_map.len();
        if n == 0 {
            diag.add_measurement(&DIAG_BRAIN_ACT, || _t_total.elapsed().as_secs_f64() * 1000.0);
            return;
        }
        // Build inputs vec indexed by slot. Iterate alive query, look up slot,
        // place inputs at slot index. Slots jsou dense 0..n.
        let mut inputs_by_slot: Vec<[f32; BRAIN_INPUTS]> = vec![[0.0; BRAIN_INPUTS]; n];
        for (entity, mut cell) in &mut cells {
            let Some(slot) = slot_map.slot_of(entity) else {
                continue;
            };
            inputs_by_slot[slot] = gather(entity, &mut cell.0);
        }
        let t_gpu = Instant::now();
        gpu.cells.upload_inputs(&inputs_by_slot);
        gpu.brain.forward_persistent(&gpu.cells, n);
        let (hiddens, outputs) = gpu.cells.download_hidden_outputs(n);
        diag.add_measurement(&DIAG_BRAIN_GPU_RT, || t_gpu.elapsed().as_secs_f64() * 1000.0);
        for (entity, mut cell) in &mut cells {
            let Some(slot) = slot_map.slot_of(entity) else {
                continue;
            };
            cell.0.last_inputs = inputs_by_slot[slot];
            cell.0.last_hidden = hiddens[slot];
            cell.0.last_outputs = outputs[slot];
            cell.0.apply_brain_motor(&outputs[slot], dt);
        }
        diag.add_measurement(&DIAG_BRAIN_ACT, || _t_total.elapsed().as_secs_f64() * 1000.0);
        return;
    }

    // CPU fallback (no GPU available or feature disabled).
    let _ = slot_map;
    for (entity, mut cell) in &mut cells {
        let inputs = gather(entity, &mut cell.0);
        let (hidden, outputs) = cell.0.genome.brain.forward_with_state(&inputs);
        cell.0.last_inputs = inputs;
        cell.0.last_hidden = hidden;
        cell.0.last_outputs = outputs;
        cell.0.apply_brain_motor(&outputs, dt);
    }
    diag.add_measurement(&DIAG_BRAIN_ACT, || _t_total.elapsed().as_secs_f64() * 1000.0);
}

fn apply_cell_morph(time: Res<Time>, mut cells: Query<&mut CellEntity, Without<Dying>>) {
    let dt = time.delta_secs();
    for mut cell in &mut cells {
        cell.0.apply_morph(dt);
    }
}

fn spawn_food(
    foods: Query<(), With<FoodEntity>>,
    extent: Res<WorldExtent>,
    factor: Res<FoodDensityFactor>,
    cell_grid: Res<CellGrid>,
    world_map: Res<WorldMapResource>,
    food_mesh: Res<FoodMesh>,
    food_material: Res<FoodMaterial>,
    mut commands: Commands,
) {
    let target = food_target(&extent, factor.0);
    let count = foods.iter().count();
    if count >= target {
        return;
    }
    let to_spawn = (target - count).min(FOOD_SPAWN_RATE);
    let mut rng = rand::rng();
    let half = extent.as_array();
    // Sprint 41: bump broad-phase budget na MAX_BODY_LENGTH — worst-case max_axis
    // ellipsoid může extending podél long axis až o tuto velikost. BROAD_PHASE_SIZE_BUDGET
    // = 3.0 (effective_radius default) by missnul cells s max_axis blízko 4.0.
    let broad_r = EAT_RADIUS * MAX_BODY_LENGTH;

    'spawn: for _ in 0..to_spawn {
        for _ in 0..MAX_SPAWN_ATTEMPTS {
            let candidate = Food::random(&mut rng, half);
            // Sprint 31: rejection sampling proti uniform — bias k rich zonám.
            // Spotřebovává retry budget jako cell-exclusion check níž.
            let richness = world_map
                .0
                .sample([candidate.position[0], candidate.position[1], 0.0]);
            if reject_food_for_richness(&mut rng, richness) {
                continue;
            }
            let mut blocked = false;
            cell_grid.0.for_each_in_radius_toroidal(
                candidate.position,
                broad_r,
                SIMULATION_HALF,
                |_, cell_pos, radius| {
                    if blocked {
                        return;
                    }
                    let exclusion = EAT_RADIUS * radius;
                    let d = bioscape::min_image_delta(candidate.position, cell_pos, SIMULATION_HALF);
                    if d[0] * d[0] + d[1] * d[1] + d[2] * d[2] < exclusion * exclusion {
                        blocked = true;
                    }
                },
            );
            if !blocked {
                commands.spawn((
                    FoodEntity(candidate),
                    Mesh3d(food_mesh.0.clone()),
                    MeshMaterial3d(food_material.0.clone()),
                    Transform::from_xyz(
                        candidate.position[0],
                        candidate.position[1],
                        candidate.position[2],
                    ),
                ));
                continue 'spawn;
            }
        }
        // All MAX_SPAWN_ATTEMPTS rolls fell inside someone's eat radius — skip
        // this slot. Population is dense enough that the food world is at
        // (ecological) saturation; we'll catch up later when cells move.
    }
}

fn cell_eats_food(
    mut cells: Query<(Entity, &mut CellEntity), Without<Dying>>,
    food_grid: Res<FoodGrid>,
    world_map: Res<WorldMapResource>,
    slot_map: Res<CellSlotMap>,
    #[cfg(feature = "gpu")] gpu_state: Option<Res<GpuBrainState>>,
    mut commands: Commands,
    mut diag: Diagnostics,
) {
    let t_total = Instant::now();

    // Sprint 52: pokud GPU available, sbíráme rewards Vec[N] a dispatchneme
    // GPU Hebbian na konci místo per-cell CPU brain.hebbian_update.
    #[cfg(feature = "gpu")]
    let use_gpu_hebbian = gpu_state.is_some();
    #[cfg(not(feature = "gpu"))]
    let use_gpu_hebbian = false;

    let mut rewards: Vec<f32> = if use_gpu_hebbian {
        vec![0.0; slot_map.len()]
    } else {
        Vec::new()
    };

    // Sprint 58: 3-pass refactor (mirror Sprint 57 headless eat_food).
    // Pass 1 (par): per-cell candidate selection. Snapshot s pose data pro
    // toroidal-aware eat_test_pose (lib helper bez `&Cell`).
    let snapshot: Vec<(Entity, [f32; 3], f32, [f32; 3], f32, f32)> = cells
        .iter()
        .map(|(e, c)| {
            (
                e,
                c.0.position,
                c.0.phenotype.max_axis(),
                [
                    c.0.phenotype.body_length,
                    c.0.phenotype.body_width,
                    c.0.phenotype.body_height,
                ],
                c.0.heading,
                c.0.pitch,
            )
        })
        .collect();
    let food_grid_ref = &food_grid.0;
    let candidates: Vec<Option<(Entity, f32)>> = snapshot
        .par_iter()
        .map(|(_entity, pos, max_axis, dims, heading, pitch)| {
            let eat_r = EAT_RADIUS * *max_axis;
            let mut ate: Option<(Entity, f32)> = None;
            food_grid_ref.for_each_in_radius_toroidal(
                *pos,
                eat_r,
                SIMULATION_HALF,
                |food_e, food_pos, _| {
                    if ate.is_some() {
                        return;
                    }
                    // Sprint 54: ghost food s min-imaged position pro toroidal eat_test.
                    let md = bioscape::min_image_delta(*pos, food_pos, SIMULATION_HALF);
                    let ghost_pos = [pos[0] + md[0], pos[1] + md[1], food_pos[2]];
                    if bioscape::eat_test_pose(*pos, *heading, *pitch, *dims, ghost_pos, EAT_RADIUS) {
                        let value = FOOD_VALUE
                            * food_multiplier(
                                world_map.0.sample([food_pos[0], food_pos[1], 0.0]),
                            )
                            * Food {
                                position: food_pos,
                                age_ticks: 0,
                            }
                            .value_factor();
                        ate = Some((food_e, value));
                    }
                },
            );
            ate
        })
        .collect();

    // Pass 2 (sequential): resolve race + apply energy + Hebbian. First-cell-wins
    // per food entity (matches pre-Sprint-58 ordering).
    let mut eaten: FxHashSet<Entity> = FxHashSet::default();
    for ((entity, _, _, _, _, _), opt) in snapshot.iter().zip(candidates.iter()) {
        if let Some((food_e, value)) = opt {
            if eaten.contains(food_e) {
                continue;
            }
            eaten.insert(*food_e);
            if let Ok((_, mut cell)) = cells.get_mut(*entity) {
                cell.0.energy += *value;
                if use_gpu_hebbian {
                    if let Some(slot) = slot_map.slot_of(*entity) {
                        if slot < rewards.len() {
                            rewards[slot] = 1.0;
                        }
                    }
                } else {
                    let last_inputs = cell.0.last_inputs;
                    let last_hidden = cell.0.last_hidden;
                    let last_outputs = cell.0.last_outputs;
                    cell.0.genome.brain.hebbian_update(
                        &last_inputs,
                        &last_hidden,
                        &last_outputs,
                        1.0,
                        LEARNING_RATE,
                    );
                }
            }
        }
    }

    // Pass 3: main-thread Commands flush (despawn nelze v par_iter).
    for food_e in &eaten {
        commands.entity(*food_e).despawn();
    }

    #[cfg(feature = "gpu")]
    if let Some(gpu) = gpu_state {
        let n = slot_map.len();
        if n > 0 && rewards.iter().any(|&r| r > 0.0) {
            gpu.cells.upload_rewards(&rewards);
            gpu.hebbian.compute_persistent(&gpu.cells, n, LEARNING_RATE);
        }
    }
    let _ = rewards;
    diag.add_measurement(&DIAG_EAT_FOOD, || t_total.elapsed().as_secs_f64() * 1000.0);
}

fn sync_transforms(
    mut cells: Query<(&CellEntity, &mut Transform), Without<Dying>>,
    mut diag: Diagnostics,
) {
    let t = Instant::now();
    for (cell, mut transform) in &mut cells {
        transform.translation.x = cell.0.position[0];
        transform.translation.y = cell.0.position[1];
        transform.translation.z = cell.0.position[2];
        transform.rotation = cell_rotation(cell.0.heading, cell.0.pitch);
        let target_scale = cell_scale(&cell.0.phenotype);
        if (transform.scale.x - target_scale.x).abs() > 1e-3
            || (transform.scale.y - target_scale.y).abs() > 1e-3
            || (transform.scale.z - target_scale.z).abs() > 1e-3
        {
            transform.scale = target_scale;
        }
    }
    diag.add_measurement(&DIAG_SYNC_TRANSFORMS, || t.elapsed().as_secs_f64() * 1000.0);
}

fn speed_input(keys: Res<ButtonInput<KeyCode>>, mut time: ResMut<Time<Virtual>>) {
    if keys.just_pressed(KeyCode::Space) {
        if time.is_paused() {
            time.unpause();
            info!("sim: unpaused");
        } else {
            time.pause();
            info!("sim: paused");
        }
    }

    let new_speed = if keys.just_pressed(KeyCode::Digit1) {
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

    if let Some(speed) = new_speed {
        time.set_relative_speed(speed);
        if time.is_paused() {
            time.unpause();
        }
        info!("sim: {}× speed", speed);
    }
}

/// Sprint 36: mouse drag rotuje kamerou kolem `target` (orbit) NEBO pannuje
/// `target` — left = orbit, middle = pan. Horizontální delta orbit → yaw,
/// vertical → pitch. Pan v "cursor pulls world" módu (drag right ⇒ target left).
fn camera_orbit_input(
    buttons: Res<ButtonInput<MouseButton>>,
    mut motion: MessageReader<MouseMotion>,
    mut orbit: ResMut<OrbitCamera>,
) {
    let orbit_active = buttons.pressed(MouseButton::Left);
    let pan_active = buttons.pressed(MouseButton::Middle);
    if !orbit_active && !pan_active {
        // Drop accumulated motion when not actively dragging — jinak by
        // se delta nasčítaly a po stisku tlačítka by kamera skočila.
        motion.clear();
        return;
    }
    let mut delta = Vec2::ZERO;
    for ev in motion.read() {
        delta += ev.delta;
    }
    if delta == Vec2::ZERO {
        return;
    }
    if orbit_active {
        orbit.yaw = (orbit.yaw + delta.x * ORBIT_SENSITIVITY).rem_euclid(std::f32::consts::TAU);
        orbit.pitch =
            (orbit.pitch + delta.y * ORBIT_SENSITIVITY).clamp(CAMERA_PITCH_MIN, CAMERA_PITCH_MAX);
    } else if pan_active {
        // Pan target proti směru drag (cursor pulls world). Pan rovina v xy
        // podle yaw — right vector + forward vector v xy projekci.
        // Vertical screen drag (y) ≡ "do scény" → forward; Y screen jde dolů,
        // takže invertovat (y- = forward+).
        let cos_y = orbit.yaw.cos();
        let sin_y = orbit.yaw.sin();
        let forward_xy = Vec2::new(sin_y, cos_y);
        let right_xy = Vec2::new(cos_y, -sin_y);
        // Pan rychlost ∝ scale (víc zoomout = rychlejší pan, drag-distance
        // odpovídá viditelnému světu).
        let speed = orbit.scale;
        let world_xy = -right_xy * delta.x * speed + forward_xy * delta.y * speed;
        orbit.target.x += world_xy.x;
        orbit.target.y += world_xy.y;
    }
}

/// Sprint 36: mouse wheel zoom — adjustuje orthographic scale. Scroll up =
/// zoom in (menší scale = víc pixelů per world unit). Clamp brání zoom out
/// pryč ze scény (nebyly by vidět hranice světa, jen black void).
fn camera_zoom_input(mut wheel: MessageReader<MouseWheel>, mut orbit: ResMut<OrbitCamera>) {
    let mut scroll = 0.0_f32;
    for ev in wheel.read() {
        scroll += match ev.unit {
            MouseScrollUnit::Line => ev.y,
            MouseScrollUnit::Pixel => ev.y / 50.0,
        };
    }
    if scroll == 0.0 {
        return;
    }
    let factor = (-scroll * CAMERA_ZOOM_STEP).exp();
    orbit.scale = (orbit.scale * factor).clamp(CAMERA_SCALE_MIN, CAMERA_SCALE_MAX);
}

/// Sprint 36: WASD/šipky pannují `OrbitCamera.target` v xy-plochy ve frame
/// kamery (W = posun "do scény", A = doleva). Pan rychlost ∝ distance
/// (víc zoomout = rychlejší pan), takže feel je konzistentní napříč zoom.
fn camera_pan_input(
    keys: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    mut orbit: ResMut<OrbitCamera>,
) {
    let mut delta = Vec2::ZERO;
    if keys.pressed(KeyCode::ArrowLeft) || keys.pressed(KeyCode::KeyA) {
        delta.x -= 1.0;
    }
    if keys.pressed(KeyCode::ArrowRight) || keys.pressed(KeyCode::KeyD) {
        delta.x += 1.0;
    }
    if keys.pressed(KeyCode::ArrowDown) || keys.pressed(KeyCode::KeyS) {
        delta.y -= 1.0;
    }
    if keys.pressed(KeyCode::ArrowUp) || keys.pressed(KeyCode::KeyW) {
        delta.y += 1.0;
    }
    if delta == Vec2::ZERO {
        return;
    }
    // Pan rychlost ∝ scale — při zoom in pan jemně, při zoom out rychle.
    // 800 world units/s při scale=1.0 = ~3 sec přejet šíři screenu.
    let speed = orbit.scale * 800.0 * time.delta_secs();
    // Pan v rovině xy podle yaw orientace kamery: forward (do scény) je směr,
    // kterým camera kouká po xy projekci. Right = ⊥ k forwardu v xy.
    let cos_y = orbit.yaw.cos();
    let sin_y = orbit.yaw.sin();
    let forward_xy = Vec2::new(sin_y, cos_y);
    let right_xy = Vec2::new(cos_y, -sin_y);
    let world_xy = forward_xy * delta.y + right_xy * delta.x;
    orbit.target.x += world_xy.x * speed;
    orbit.target.y += world_xy.y * speed;
}

/// Sprint 36: aplikuje OrbitCamera state na Camera3d Transform a Projection
/// scale. Běží každý frame po input systemech, takže input změny se okamžitě
/// projeví ve view matici.
fn update_orbit_camera_transform(
    orbit: Res<OrbitCamera>,
    camera: Single<(&mut Transform, &mut Projection), With<Camera3d>>,
) {
    let (mut transform, mut projection) = camera.into_inner();
    *transform = orbit.transform();
    if let Projection::Orthographic(ortho) = &mut *projection {
        ortho.scale = orbit.scale;
    }
}

fn log_clock_events(
    mut generation_ended: MessageReader<GenerationEnded>,
    mut epoch_ended: MessageReader<EpochEnded>,
) {
    for ev in generation_ended.read() {
        info!("generation {} ended", ev.generation);
    }
    for ev in epoch_ended.read() {
        info!("epoch {} ended", ev.epoch);
    }
}

fn setup_stats_overlay(mut commands: Commands) {
    commands.spawn((
        StatsRoot,
        Node {
            position_type: PositionType::Absolute,
            right: Val::Px(8.0),
            bottom: Val::Px(8.0),
            padding: UiRect::all(Val::Px(8.0)),
            flex_direction: FlexDirection::Column,
            border_radius: BorderRadius::all(Val::Px(4.0)),
            ..default()
        },
        BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.6)),
        children![(
            StatsText,
            Text::new(""),
            TextColor(Color::srgb(0.9, 0.9, 0.9)),
            TextFont {
                font_size: 13.0,
                ..default()
            },
        )],
    ));
}

fn update_stats_overlay(
    clock: Res<Clock>,
    time: Res<Time<Virtual>>,
    density: Res<FoodDensityFactor>,
    diagnostics: Res<DiagnosticsStore>,
    cells: Query<&CellEntity, Without<Dying>>,
    foods: Query<(), With<FoodEntity>>,
    text: Single<&mut Text, With<StatsText>>,
) {
    let fps = diagnostics
        .get(&FrameTimeDiagnosticsPlugin::FPS)
        .and_then(|d| d.smoothed())
        .unwrap_or(0.0);
    let speed = if time.is_paused() {
        "paused".to_string()
    } else {
        format!("{}×", time.relative_speed())
    };

    let mut count = 0usize;
    let mut spd_sum = 0.0_f64;
    let mut spd_sumsq = 0.0_f64;
    let mut vis_sum = 0.0_f64;
    let mut vis_sumsq = 0.0_f64;
    let mut trn_sum = 0.0_f64;
    let mut len_sum = 0.0_f64;
    let mut wid_sum = 0.0_f64;
    let mut asp_sum = 0.0_f64;
    let mut asp_sumsq = 0.0_f64;
    let mut spk_sum = 0.0_f64;
    let mut spk_max = 0.0_f64;
    let mut e_sum = 0.0_f64;
    let mut lineages: FxHashSet<u64> = FxHashSet::default();
    let mut oldest_age: u64 = 0;
    let current_gen = clock.0.generation;
    for c in &cells {
        count += 1;
        let s = c.0.genome.max_speed as f64;
        let v = c.0.genome.vision_radius as f64;
        let t = c.0.genome.turn_rate as f64;
        let l = c.0.phenotype.body_length as f64;
        let w = c.0.phenotype.body_width as f64;
        let aspect = if w > 1e-6 { l / w } else { 0.0 };
        let spk = c.0.phenotype.spike_length as f64;
        let e = c.0.energy as f64;
        spd_sum += s;
        spd_sumsq += s * s;
        vis_sum += v;
        vis_sumsq += v * v;
        trn_sum += t;
        len_sum += l;
        wid_sum += w;
        asp_sum += aspect;
        asp_sumsq += aspect * aspect;
        spk_sum += spk;
        if spk > spk_max {
            spk_max = spk;
        }
        e_sum += e;
        lineages.insert(c.0.lineage_id);
        let age = current_gen.saturating_sub(c.0.lineage_birth_gen);
        if age > oldest_age {
            oldest_age = age;
        }
    }
    let food_count = foods.iter().count();
    let lineage_count = lineages.len();

    let (spd_avg, spd_dev, vis_avg, vis_dev, trn_avg, len_avg, wid_avg, asp_avg, asp_dev, spk_avg, e_avg) =
        if count == 0 {
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        } else {
            let n = count as f64;
            let spd_m = spd_sum / n;
            let vis_m = vis_sum / n;
            let asp_m = asp_sum / n;
            (
                spd_m,
                ((spd_sumsq / n) - spd_m * spd_m).max(0.0).sqrt(),
                vis_m,
                ((vis_sumsq / n) - vis_m * vis_m).max(0.0).sqrt(),
                trn_sum / n,
                len_sum / n,
                wid_sum / n,
                asp_m,
                ((asp_sumsq / n) - asp_m * asp_m).max(0.0).sqrt(),
                spk_sum / n,
                e_sum / n,
            )
        };

    let mut text = text.into_inner();
    text.0 = format!(
        "tick     {}\ngen      {}\nepoch    {}\nspeed    {}\ncells    {}\nfood     {}\ndensity  {:.2}\nfps      {:.0}\nspd_avg  {:.1}\nspd_dev  {:.2}\nvis_avg  {:.1}\nvis_dev  {:.2}\ntrn_avg  {:.2}\nlen_avg  {:.2}\nwid_avg  {:.2}\nasp_avg  {:.2}\nasp_dev  {:.2}\nspk_avg  {:.2}\nspk_max  {:.2}\ne_avg    {:.1}\nlineages {}\noldest   {}",
        clock.0.tick,
        clock.0.generation,
        clock.0.epoch,
        speed,
        count,
        food_count,
        density.0,
        fps,
        spd_avg,
        spd_dev,
        vis_avg,
        vis_dev,
        trn_avg,
        len_avg,
        wid_avg,
        asp_avg,
        asp_dev,
        spk_avg,
        spk_max,
        e_avg,
        lineage_count,
        oldest_age,
    );
}

fn cell_reproduces_on_threshold(
    mut cells: Query<(Entity, &mut CellEntity), Without<Dying>>,
    cell_mesh: Res<CellMesh>,
    mut lineage_materials: ResMut<LineageMaterials>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut slot_map: ResMut<CellSlotMap>,
    #[cfg(feature = "gpu")] gpu_state: Option<Res<GpuBrainState>>,
    mut commands: Commands,
) {
    let current_pop = cells.iter().count();
    if current_pop >= MAX_POPULATION {
        return;
    }
    let budget = MAX_POPULATION - current_pop;

    let fertile: Vec<(Entity, [f32; 3])> = cells
        .iter()
        .filter(|(_, c)| {
            c.0.energy >= REPRODUCE_THRESHOLD
                && c.0.last_outputs[2] > MATING_PHEROMONE_THRESHOLD
                && c.0.reproduce_cooldown_ticks == 0
        })
        .map(|(e, c)| (e, c.0.position))
        .collect();
    let mating_r2 = MATING_RADIUS * MATING_RADIUS;
    let matings = bioscape::pair_fertile(&fertile, mating_r2, budget, SIMULATION_HALF);

    // Sprint 52: před crossover sync parent brains z GPU (post-Hebbian je
    // canonical). Pokud GPU available; jinak no-op (CPU brain je canonical).
    #[cfg(feature = "gpu")]
    if let Some(gpu) = gpu_state.as_ref() {
        for &(a, b) in &matings {
            if let (Some(slot_a), Some(slot_b)) = (slot_map.slot_of(a), slot_map.slot_of(b)) {
                let brain_a = gpu.cells.download_brain_at(slot_a);
                let brain_b = gpu.cells.download_brain_at(slot_b);
                if let Ok([(_, mut ca), (_, mut cb)]) = cells.get_many_mut([a, b]) {
                    ca.0.genome.brain = brain_a;
                    cb.0.genome.brain = brain_b;
                }
            }
        }
    }

    let mut rng = rand::rng();
    let mut to_spawn: Vec<Cell> = Vec::new();
    for (a, b) in matings {
        let Ok([(_, mut cell_a), (_, mut cell_b)]) = cells.get_many_mut([a, b]) else {
            continue;
        };
        cell_a.0.energy *= 0.5;
        cell_b.0.energy *= 0.5;
        cell_a.0.reproduce_cooldown_ticks = MATING_COOLDOWN_TICKS;
        cell_b.0.reproduce_cooldown_ticks = MATING_COOLDOWN_TICKS;
        to_spawn.push(bioscape::make_mating_child(&cell_a.0, &cell_b.0, &mut rng));
    }

    let mesh = cell_mesh.0.clone();
    for cell in to_spawn {
        let mat = lineage_material(&mut lineage_materials, &mut materials, cell.lineage_id);
        let entity = commands
            .spawn((
                CellEntity(cell),
                Mesh3d(mesh.clone()),
                MeshMaterial3d(mat),
                Transform::from_xyz(cell.position[0], cell.position[1], cell.position[2])
                    .with_rotation(cell_rotation(cell.heading, cell.pitch))
                    .with_scale(cell_scale(&cell.phenotype)),
            ))
            .id();
        let slot = slot_map.allocate(entity);
        // Sprint 52: upload child brain + xoshiro seed na nový slot.
        #[cfg(feature = "gpu")]
        if let Some(gpu) = gpu_state.as_ref() {
            gpu.cells.upload_brain_at(slot, &cell.genome.brain);
            gpu.cells.upload_xoshiro_seed_at(
                slot,
                cell.lineage_id ^ (slot as u64).wrapping_mul(0x9E3779B97F4A7C15),
            );
        }
        let _ = slot;
    }
}

fn cell_dies_on_zero_energy(
    cells: Query<(Entity, &CellEntity), Without<Dying>>,
    extent: Res<WorldExtent>,
    food_mesh: Res<FoodMesh>,
    food_material: Res<FoodMaterial>,
    mut slot_map: ResMut<CellSlotMap>,
    #[cfg(feature = "gpu")] gpu_state: Option<Res<GpuBrainState>>,
    mut commands: Commands,
) {
    let mut rng = rand::rng();
    let half = extent.as_array();
    for (entity, cell) in &cells {
        if cell.0.energy <= 0.0 {
            commands.entity(entity).insert(Dying {
                ticks_left: DEATH_FADE_TICKS,
            });
            for _ in 0..CARRION_FOOD_COUNT {
                let pos = [
                    (cell.0.position[0] + rng.random_range(-CELL_RADIUS..CELL_RADIUS))
                        .clamp(-half[0], half[0]),
                    (cell.0.position[1] + rng.random_range(-CELL_RADIUS..CELL_RADIUS))
                        .clamp(-half[1], half[1]),
                    cell.0.position[2].clamp(-half[2], half[2]),
                ];
                commands.spawn((
                    FoodEntity(Food { position: pos, age_ticks: 0 }),
                    Mesh3d(food_mesh.0.clone()),
                    MeshMaterial3d(food_material.0.clone()),
                    Transform::from_xyz(pos[0], pos[1], pos[2]),
                ));
            }
            // Sprint 52: release slot ihned na Dying. Entity ještě existuje
            // pro fade animaci (Without<Dying> ji vyloučí ze sim systems).
            // GPU swap_to drží sloty dense.
            if let Some((freed_slot, moved)) = slot_map.release(entity) {
                #[cfg(feature = "gpu")]
                if let Some(gpu) = gpu_state.as_ref() {
                    if let Some(_moved_entity) = moved {
                        // moved cell je ve slot_map.slot_of(moved_entity) = freed_slot
                        // teď. Source je old_slot = current cell count (po release).
                        gpu.cells.swap_to(freed_slot, slot_map.len());
                    }
                }
                let _ = freed_slot;
                let _ = moved;
            }
        }
    }
}

fn rebuild_cell_grid(
    mut grid: ResMut<CellGrid>,
    cells: Query<(Entity, &CellEntity), Without<Dying>>,
    mut diag: Diagnostics,
) {
    let t = Instant::now();
    grid.0.rebuild(
        cells
            .iter()
            .map(|(e, c)| (e, c.0.position, c.0.phenotype.effective_radius())),
    );
    diag.add_measurement(&DIAG_GRID_REBUILD, || t.elapsed().as_secs_f64() * 1000.0);
}

fn rebuild_food_grid(
    mut grid: ResMut<FoodGrid>,
    foods: Query<(Entity, &FoodEntity)>,
) {
    grid.0.rebuild(foods.iter().map(|(e, f)| (e, f.0.position, ())));
}

// Generous broad-phase upper bound on "other" effective_radius — captures
// candidates even when neighbors are oversized. Narrow-phase uses pair sum.
const BROAD_PHASE_SIZE_BUDGET: f32 = 3.0;

fn cell_predates_on_neighbor(
    grid: Res<CellGrid>,
    mut cells: Query<(Entity, &mut CellEntity), Without<Dying>>,
    mut diag: Diagnostics,
) {
    let t_total = Instant::now();
    // Sprint 58: HashMap → FxHashMap (5-10× rychlejší hash than SipHash).
    // Predate je hot path s 3 entity-keyed mapami × n insertů × n lookupů.
    let mut energy_changes: FxHashMap<Entity, f32> = FxHashMap::default();
    // Sprint 30: nedobrovolný drain do brain damage signálu (input[14]).
    // Voluntární cost (movement, morph, attack) sem nepatří, jen predation.
    let mut damage_changes: FxHashMap<Entity, f32> = FxHashMap::default();

    // Sprint 29 selfish-herd: pre-compute herd count per cell (počet sousedů
    // ve `HERD_RADIUS`). V predaci níže se gain násobí 1/(1 + K × herd_count_prey)
    // — kořist obklopena hejnem dává predátorovi menší odměnu.
    // Sprint 58: snapshot + rayon par compute. Snapshot drží jen entity+pos,
    // herd query je read-only přes grid Res. Indexed Vec<u32> result.
    let herd_r2 = HERD_RADIUS * HERD_RADIUS;
    let snapshot: Vec<(Entity, [f32; 3])> = cells
        .iter()
        .map(|(e, c)| (e, c.0.position))
        .collect();
    let grid_ref = &grid.0;
    let herd_counts_vec: Vec<u32> = snapshot
        .par_iter()
        .map(|(entity, pos)| {
            let mut count: u32 = 0;
            grid_ref.for_each_in_radius_toroidal(
                *pos,
                HERD_RADIUS,
                SIMULATION_HALF,
                |other, other_pos, _| {
                    if other == *entity {
                        return;
                    }
                    let d = bioscape::min_image_delta(*pos, other_pos, SIMULATION_HALF);
                    if d[0] * d[0] + d[1] * d[1] + d[2] * d[2] < herd_r2 {
                        count += 1;
                    }
                },
            );
            count
        })
        .collect();
    let herd_counts: FxHashMap<Entity, u32> = snapshot
        .iter()
        .zip(herd_counts_vec.iter())
        .map(|((e, _), c)| (*e, *c))
        .collect();

    for (entity_a, cell_a) in &cells {
        // Sprint 27: attack je opt-in přes brain output[6]. Bez aktivního
        // signálu kontakty s menšími cells jen kolize (řešené v
        // resolve_cell_collisions), ne predace.
        if cell_a.0.last_outputs[6].max(0.0) <= ATTACK_THRESHOLD {
            continue;
        }
        let pos_a = cell_a.0.position;
        let radius_a = cell_a.0.phenotype.effective_radius();
        let broad_r = CELL_RADIUS * (radius_a + BROAD_PHASE_SIZE_BUDGET);

        grid.0
            .for_each_in_radius_toroidal(pos_a, broad_r, SIMULATION_HALF, |entity_b, pos_b, radius_b| {
                if entity_b == entity_a {
                    return;
                }
                if radius_a < SIZE_RATIO_THRESHOLD * radius_b {
                    return;
                }
                let pair_r = CELL_RADIUS * (radius_a + radius_b);
                let pair_r2 = pair_r * pair_r;
                // Sprint 54: min-image delta a→b. Spike bonus volá `spike_bonus_against`
                // s pos_b — pro toroidal upravíme target pos do min-image frame.
                let d = bioscape::min_image_delta(pos_a, pos_b, SIMULATION_HALF);
                let d2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                if d2 >= pair_r2 {
                    return;
                }
                let ghost_b = [pos_a[0] + d[0], pos_a[1] + d[1], pos_a[2] + d[2]];
                let bonus = cell_a.0.spike_bonus_against(ghost_b);
                let gain_raw = PREDATION_GAIN_PER_TICK + bonus;
                let prey_neighbors = *herd_counts.get(&entity_b).unwrap_or(&0);
                let dilution = 1.0 / (1.0 + DILUTION_K * prey_neighbors as f32);
                let gain = gain_raw * dilution;
                *energy_changes.entry(entity_a).or_insert(0.0) += gain;
                *energy_changes.entry(entity_b).or_insert(0.0) -= PREDATION_DRAIN_PER_TICK;
                *damage_changes.entry(entity_b).or_insert(0.0) += PREDATION_DRAIN_PER_TICK;
            });
    }

    for (entity, delta) in energy_changes {
        if let Ok((_, mut cell)) = cells.get_mut(entity) {
            cell.0.energy += delta;
        }
    }
    for (entity, delta) in damage_changes {
        if let Ok((_, mut cell)) = cells.get_mut(entity) {
            cell.0.damage_accum += delta;
        }
    }
    diag.add_measurement(&DIAG_PREDATION, || t_total.elapsed().as_secs_f64() * 1000.0);
}

fn resolve_cell_collisions(
    grid: Res<CellGrid>,
    mut cells: Query<(Entity, &mut CellEntity), Without<Dying>>,
    mut diag: Diagnostics,
) {
    let t_total = Instant::now();
    // Sprint 58: snapshot + rayon par compute deltas. Pass 1 sběr (entity, pos,
    // radius) → par filter_map vrací jen non-zero deltas. Pass 2 sekvenčně
    // applikuje přes Query::get_mut (ECS write requires single-threaded access).
    let snapshot: Vec<(Entity, [f32; 3], f32)> = cells
        .iter()
        .map(|(e, c)| (e, c.0.position, c.0.phenotype.effective_radius()))
        .collect();
    let grid_ref = &grid.0;
    let deltas: Vec<(Entity, [f32; 2])> = snapshot
        .par_iter()
        .filter_map(|(entity_a, pos_a, radius_a)| {
            let broad_r = CELL_RADIUS * (*radius_a + BROAD_PHASE_SIZE_BUDGET);
            let mut delta = [0.0_f32, 0.0_f32];
            grid_ref.for_each_in_radius_toroidal(
                *pos_a,
                broad_r,
                SIMULATION_HALF,
                |entity_b, pos_b, radius_b| {
                    if entity_b == *entity_a {
                        return;
                    }
                    let pair_r = CELL_RADIUS * (*radius_a + radius_b);
                    let pair_r2 = pair_r * pair_r;
                    // Sprint 54: min-image delta b→a (push direction).
                    let d_vec = bioscape::min_image_delta(pos_b, *pos_a, SIMULATION_HALF);
                    let d2 = d_vec[0] * d_vec[0] + d_vec[1] * d_vec[1] + d_vec[2] * d_vec[2];
                    if d2 < pair_r2 && d2 > 0.0 {
                        let d = d2.sqrt();
                        let overlap = pair_r - d;
                        delta[0] += (d_vec[0] / d) * overlap * 0.5;
                        delta[1] += (d_vec[1] / d) * overlap * 0.5;
                    }
                },
            );
            if delta != [0.0, 0.0] {
                Some((*entity_a, delta))
            } else {
                None
            }
        })
        .collect();

    for (entity, delta) in deltas {
        if let Ok((_, mut cell)) = cells.get_mut(entity) {
            cell.0.position[0] += delta[0];
            cell.0.position[1] += delta[1];
        }
    }
    diag.add_measurement(&DIAG_COLLISIONS, || t_total.elapsed().as_secs_f64() * 1000.0);
}

fn tick_death_fade(
    mut dying: Query<(
        Entity,
        &mut Dying,
        &CellEntity,
        &mut Transform,
    )>,
    mut commands: Commands,
) {
    for (entity, mut d, cell, mut transform) in &mut dying {
        if d.ticks_left == 0 {
            commands.entity(entity).despawn();
            continue;
        }
        d.ticks_left -= 1;
        let progress = d.ticks_left as f32 / DEATH_FADE_TICKS as f32;
        // Sprint 36: fade jen přes scale shrinkout. Alpha fade by chtělo
        // Material handle adjustment per cell (StandardMaterial alpha_mode +
        // base_color.alpha). Sprint 38+ může to vyřešit; teď postačí scaling.
        transform.scale = cell_scale(&cell.0.phenotype) * progress;
    }
}

/// Sprint 36: 3-axis ellipsoid scale (length × width × height). Bevy non-uniform
/// scale aplikuje na unit-radius sphere, vytváří ellipsoid s poloosami
/// (L, W, H) podél x, y, z. Po `cell_rotation(yaw, pitch)` je local +X
/// alignovaný s forward vektorem buňky.
fn cell_scale(phenotype: &Phenotype) -> Vec3 {
    Vec3::new(
        phenotype.body_length,
        phenotype.body_width,
        phenotype.body_height,
    )
}


fn toggle_stats_overlay(
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

fn toggle_world_map_overlay(
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
