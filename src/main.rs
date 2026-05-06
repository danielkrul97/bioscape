use bevy::asset::RenderAssetUsages;
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::diagnostic::{
    Diagnostic, DiagnosticPath, Diagnostics, DiagnosticsStore, FrameTimeDiagnosticsPlugin,
    LogDiagnosticsPlugin, RegisterDiagnostic,
};
use bevy::image::Image;
use bevy::input::mouse::{MouseMotion, MouseScrollUnit, MouseWheel};
use bevy::pbr::{DistanceFog, ExtendedMaterial, FogFalloff, MaterialExtension};
use bevy::post_process::bloom::Bloom;
use bevy::prelude::*;
use bevy::render::render_resource::{AsBindGroup, Extent3d, TextureDimension, TextureFormat};
use bevy::shader::ShaderRef;
use bevy::render::view::Hdr;
use bevy::render::view::screenshot::{save_to_disk, Screenshot};
use std::time::Instant;
use std::path::PathBuf;

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
    adhesion_velocity_delta, bond_velocity_delta, nearest_attackable_cell,
    reject_food_for_richness, Bond, Cell, Food, Hunter, Phenotype, SimClock, SmellField,
    SpatialGrid, WorldMap, ADHESION_RANGE_FACTOR, ATTACK_THRESHOLD, BOND_BREAK_THRESHOLD,
    BOND_FORMATION_COST, BOND_FORM_THRESHOLD, BOND_FORM_TICKS, BOND_MAINTENANCE_PER_SEC,
    BOND_REST_LENGTH_SLACK, BRAIN_INPUTS, CARRION_FOOD_COUNT, CELL_RADIUS,
    CONTACT_DECAY_TICKS, CYCLE_AMPLITUDE, CYCLE_GEN_PERIOD, DILUTION_K, EAT_RADIUS,
    FIXED_TIMESTEP_HZ, FOOD_SPAWN_RATE, GENERATIONS_PER_EPOCH, HAZARD_AMP,
    HAZARD_DRAIN_PER_SEC, HAZARD_FLOOR, HERD_RADIUS,
    HUNTER_TARGET_COUNT, INITIAL_CELLS, LEARNING_RATE,
    MATING_COOLDOWN_TICKS, MATING_PHEROMONE_THRESHOLD, MATING_RADIUS, MAX_BODY_LENGTH,
    MAX_BONDS_PER_CELL, MAX_POPULATION, MAX_SPAWN_ATTEMPTS, PHEROMONE_BASELINE_EMIT,
    PHEROMONE_BRAIN_MOD, PHEROMONE_COST_PER_RATE, PHEROMONE_DECAY, PHEROMONE_DIFFUSION,
    PHEROMONE_GRID_RES, PHEROMONE_GRID_RES_Z, PHEROMONE_SAMPLE_EPSILON, PHYSICS_CONFIG,
    PREDATION_DRAIN_PER_TICK, PREDATION_GAIN_PER_TICK, REPRODUCE_THRESHOLD,
    SIZE_RATIO_THRESHOLD, SMELL_DECAY, SMELL_DIFFUSION, SMELL_GRID_RES, SMELL_GRID_RES_Z,
    SMELL_PER_FOOD, SMELL_SAMPLE_EPSILON, THERMAL_NOISE, TICKS_PER_GENERATION, WORLD_HALF,
    WORLD_MAP_BASE_RES, WORLD_MAP_BASE_RES_Z, WORLD_MAP_FOOD_AMP, WORLD_MAP_FOOD_FLOOR,
    WORLD_MAP_RES, WORLD_MAP_RES_Z, WORLD_MAP_SEED, WORLD_UNITS_PER_FOOD,
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

/// Sprint 91: shader asset path pro `BioMaterialExt`. Loaded přes AssetServer
/// při startu, hot-reload v dev mode.
const BIO_SHADER_PATH: &str = "shaders/bio_material.wgsl";

/// Sprint 91: empty marker extension nad `StandardMaterial` — žádné custom
/// uniformy. Shader detekuje hunter vs cell přes `emissive.r > 2.0` (hunter
/// má LinearRgba(3.5, 0, 0)). Tím se vyhne Bevy 0.18 ExtendedMaterial uniform
/// binding layout issue (binding 100 neproside validation v `pbr_opaque_mesh_pipeline`).
///
/// Pattern_kind se přepíná in-shader podle base material color → 1 shader
/// handles obě cell + hunter. Hunter material má pure-red HDR emissive,
/// cell materials max ~1.0 emissive intensity.
#[derive(Asset, AsBindGroup, Reflect, Debug, Clone, Default)]
pub struct BioMaterialExt {}

impl MaterialExtension for BioMaterialExt {
    fn fragment_shader() -> ShaderRef {
        BIO_SHADER_PATH.into()
    }
    fn deferred_fragment_shader() -> ShaderRef {
        BIO_SHADER_PATH.into()
    }
}

/// Sprint 91: alias pro extended material handle type. `MaterialPlugin`
/// musí být registrován pro tento typ aby Bevy renderoval s naším shaderem.
type BioMaterial = ExtendedMaterial<StandardMaterial, BioMaterialExt>;

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
struct FoodGrid(SpatialGrid<Entity, bioscape::FoodKind>);

impl Default for FoodGrid {
    fn default() -> Self {
        Self(SpatialGrid::new(GRID_CELL_SIZE))
    }
}

/// Sprint 66: monotonic counter pro Cell.cell_id přidělování. Initial pop
/// uses ids 0..INITIAL_CELLS, takže start = INITIAL_CELLS. Children z
/// reproduce čerpají odsud.
#[derive(Resource)]
struct NextCellId(u64);

impl Default for NextCellId {
    fn default() -> Self {
        Self(INITIAL_CELLS as u64)
    }
}

/// Sprint 89: monotonic counter pro hunter_id + lineage_id při reproduce
/// nebo floor respawn. Init seed uses ids 0..HUNTER_TARGET_COUNT.
#[derive(Resource)]
struct NextHunterId(u64);

impl Default for NextHunterId {
    fn default() -> Self {
        Self(HUNTER_TARGET_COUNT as u64)
    }
}

/// Sprint 66: per-pair contact tick tracker. Klíč je `(min_id, max_id)`
/// stable Cell.cell_id páru. Resource žije celý běh — generation reset
/// nemažeme (kontakt může běžet napříč generační hranicí).
#[derive(Resource, Default)]
struct ContactProgress(FxHashMap<(u64, u64), u32>);

/// Sprint 99: hunter-hunter contact tracker (mirror cells). Survives
/// across ticks — bond formation gates na BOND_FORM_TICKS consecutive.
#[derive(Resource, Default)]
struct HunterContactProgress(FxHashMap<(u64, u64), u32>);

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

/// Sprint 71: shared mesh + material pro Hunter entities. Single resource —
/// všichni hunters vypadají stejně, žádný cache potřeba. Resources drží
/// handles při životě (Assets refcount); fields se přímo nečtou.
#[derive(Resource)]
#[allow(dead_code)]
struct HunterMesh(Handle<Mesh>);

#[derive(Resource)]
#[allow(dead_code)]
struct HunterMaterial(Handle<BioMaterial>);

/// Sprint 71: ECS component wrapping `Hunter` data pro renderer hot loop.
#[derive(Component)]
struct HunterEntity(Hunter);

/// Sprint 36: per-lineage material cache. Lineage hue → handle do
/// `Assets<StandardMaterial>`. Bevy automaticky deduplikuje stejné materialy
/// na renderer instances draw call.
///
/// Sprint 69: keyovaný podle `adhesion_type` (0..ADHESION_TYPE_COUNT) místo
/// `lineage_id`. 8 distinct hues = vidíš "tribes" na první pohled, jakmile
/// začne Steinberg sorting (same-type cells gravitují k sobě). Pre-Sprint 69
/// se barvilo podle lineage_hue (random hue per linii) — ten signal byl
/// užitečný pro fyzickou separaci, ale přebíjel adhesion clustering. Lineage
/// info zůstává v HUD + CSV `lineages` count.
#[derive(Resource, Default)]
struct AdhesionMaterials([Option<Handle<BioMaterial>>; 8]);

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

    if want_diag {
        app.add_plugins((
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
        .register_diagnostic(Diagnostic::new(DIAG_RENDER_OVERHEAD).with_suffix(" ms"));
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
        .init_resource::<CellGrid>()
        .init_resource::<FoodGrid>()
        .init_resource::<FoodDensityFactor>()
        .init_resource::<NextCellId>()
        .init_resource::<NextHunterId>()
        .init_resource::<ContactProgress>()
        .init_resource::<HunterContactProgress>()
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
                    pool_bonded_hidden_cells,
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
                    resolve_hunter_collisions,
                    pool_bonded_hunter_hidden_system,
                    cell_predates_on_neighbor,
                    step_hunters,
                    hunters_lifecycle,
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
                sync_hunter_transforms,
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

#[derive(Resource, Clone)]
struct ScreencastConfig {
    dir: PathBuf,
    interval_secs: f32,
    duration_secs: f32,
    started_at: Option<f32>,
    last_capture: f32,
    frame_idx: u32,
}

fn screencast_capture(
    mut commands: Commands,
    time: Res<Time<Real>>,
    cfg: Option<ResMut<ScreencastConfig>>,
    mut exit: MessageWriter<AppExit>,
) {
    // Sprint 97 follow-up: `Time<Real>` (wall clock), ne `Time<Virtual>` —
    // virtual má 50ms max_delta cap, takže pod heavy sim load by virtual
    // čas běžel 20× pomaleji než wall a 5min screencast by trval >1h.
    let Some(mut cfg) = cfg else { return; };
    let elapsed = time.elapsed_secs();
    if cfg.started_at.is_none() {
        cfg.started_at = Some(elapsed);
        cfg.last_capture = elapsed - cfg.interval_secs;
    }
    let started = cfg.started_at.unwrap();
    let dt_since_start = elapsed - started;
    if dt_since_start >= cfg.duration_secs {
        eprintln!("screencast: done, captured {} frames", cfg.frame_idx);
        exit.write(AppExit::Success);
        return;
    }
    if elapsed - cfg.last_capture < cfg.interval_secs {
        return;
    }
    let path = cfg.dir.join(format!("cap_{:05}.png", cfg.frame_idx));
    commands
        .spawn(Screenshot::primary_window())
        .observe(save_to_disk(path));
    cfg.frame_idx += 1;
    cfg.last_capture = elapsed;
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut bio_materials: ResMut<Assets<BioMaterial>>,
    mut images: ResMut<Assets<Image>>,
    mut adhesion_materials: ResMut<AdhesionMaterials>,
    mut window: Single<&mut Window>,
) {
    window.set_maximized(true);
    let half = WORLD_HALF;
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
    //
    // Sprint 88: HDR + Bloom + Tonemapping + DistanceFog atmospheric pass.
    // HDR backbuffer dovolí emissive > 1.0 (cells/hunter glow), Bloom rozšíří
    // bright pixels na soft halos, Tonemapping namapuje HDR rozsah na sRGB.
    // DistanceFog přidá deep-ocean blue tint na vzdálené objekty (ortho má
    // limited depth differentiation, ale fade k floor overlay je signifikantní).
    let initial_orbit = OrbitCamera::default();
    commands.spawn((
        Camera3d::default(),
        Hdr,
        // Sprint 88.2: TonyMcMapface → AcesFitted (increased saturation), pak
        // S88.3: AcesFitted → Reinhard. ACES desaturuje brights („brights
        // desaturate across the spectrum"); Reinhard je jediný tonemapper, kde
        // „bright primaries and secondaries don't desaturate at all". Tradeoff:
        // lots of hue shifting v brights (nepatrné posuny barev), ale pro
        // 8 distinct adhesion hues je full saturation > hue purity.
        Tonemapping::Reinhard,
        Bloom::NATURAL,
        DistanceFog {
            color: Color::srgb(0.08, 0.18, 0.30),
            falloff: FogFalloff::ExponentialSquared { density: 0.0002 },
            ..default()
        },
        IsDefaultUiCamera,
        Projection::Orthographic(OrthographicProjection {
            scale: initial_orbit.scale,
            near: 0.1,
            far: CAMERA_OFFSET_DISTANCE * 3.0,
            ..OrthographicProjection::default_3d()
        }),
        initial_orbit.transform(),
    ));

    // Ambient + DirectionalLight pro 3D scénu. Sprint 88: tinted bluish ambient
    // (underwater feel) + DirectionalLight jako "sluneční" key light pronikající
    // od povrchu šikmo. Sprint 88.1: bumped up brightness — pre-fix illuminance
    // 6000 + ambient 600 produkovaly blackout scene. HDR + bloom kombinaci nutno
    // krmit dostatkem světla aby base scene byla viditelná, ne jen emissive
    // bloom highlights.
    // Sprint 88.2: ambient méně blue (0.6 → 0.85) — silně modré ambient
    // multiplikuje s cell hue a desaturuje warm colors (red/orange/yellow
    // adhesion types). Subtler tint zachová cell color identity.
    commands.spawn(AmbientLight {
        color: Color::srgb(0.85, 0.92, 1.0),
        brightness: 1500.0,
        ..default()
    });
    commands.spawn((
        DirectionalLight {
            color: Color::srgb(0.95, 0.97, 1.0),
            illuminance: 10000.0,
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
        // Sprint 66: cell_id == lineage_id pro initial pop (1:1 mapping). Po
        // mating se cell_id čerpá z `NextCellId` resource counteru.
        let cell = Cell::random(&mut rng, half, i as u64, 0, i as u64);
        let mat = adhesion_material(
            &mut adhesion_materials,
            &mut bio_materials,
            cell.genome.adhesion_type,
        );
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

    // Sprint 71: macropredator setup. Hunter mesh = větší sphere (4× CELL_RADIUS),
    // tmavě červený material — visually distinct od cells. HUNTER_TARGET_COUNT
    // hunters spawnou na náhodné pozice; constant pop, žádný respawn.
    // Sprint 88: bumped emissive na red glow s HDR > 1.0 hodnoty — Bloom catches
    // hunter jako menacing red beacon viditelný z dálky.
    // Sprint 88.4: pure-red emissive (zero green/blue). Reinhard tonemapper
    // má dokumentované „lots of hue shifting" v brights — předchozí
    // LinearRgba(2.5, 0.2, 0.1) se posouvalo směrem k oranžové. Pure red
    // (3.5, 0.0, 0.0) zůstává nezpochybnitelně červené i pod tonemap +
    // bloom redistribution. Brighter base 0.4 → 0.85 aby hunter byl viditelně
    // červený i bez bloom kontribuce (např. v post-process toggle off).
    let hunter_mesh_handle = meshes.add(Sphere::new(CELL_RADIUS * 4.0).mesh().ico(2).unwrap());
    // Sprint 91: hunter ExtendedMaterial s chitinous-scales pattern (kind=1).
    // Scale 14 = denser scales than cells; intensity 1.0.
    let hunter_material = bio_materials.add(BioMaterial {
        base: StandardMaterial {
            base_color: Color::srgb(0.85, 0.05, 0.05),
            perceptual_roughness: 0.4,
            // Sprint 91: emissive.r >= 3.5 → shader detekuje jako HUNTER pattern.
            emissive: LinearRgba::new(3.5, 0.0, 0.0, 1.0),
            ..default()
        },
        extension: BioMaterialExt {},
    });
    // Sprint 89: každý hunter dostává random genome + lineage. Initial
    // population spawnuje se tady; Sprint 89+ lifecycle (death/reproduce)
    // mění populaci dynamicky v `step_hunters`.
    let mut hunter_rng = rand::rng();
    for i in 0..HUNTER_TARGET_COUNT {
        let h = Hunter::random(&mut hunter_rng, half, i as u64, i as u64, 0);
        commands.spawn((
            HunterEntity(h),
            Mesh3d(hunter_mesh_handle.clone()),
            MeshMaterial3d(hunter_material.clone()),
            Transform::from_xyz(h.position[0], h.position[1], h.position[2]),
        ));
    }
    commands.insert_resource(HunterMesh(hunter_mesh_handle));
    commands.insert_resource(HunterMaterial(hunter_material));
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
/// Sprint 69: 8 distinctních hues per `adhesion_type`, evenly spaced kolem
/// kruhu. Lazy-cache — handle vznikne při první cell s daným typem; pak
/// re-use. Same hue se zrcadlí do bond gizmo lines, takže shluk = barva
/// těla + barva bond lines = jednolitý vizuální chunk.
fn adhesion_material(
    cache: &mut AdhesionMaterials,
    bio_materials: &mut Assets<BioMaterial>,
    adhesion_type: u8,
) -> Handle<BioMaterial> {
    let idx = (adhesion_type as usize) % 8;
    if let Some(h) = &cache.0[idx] {
        return h.clone();
    }
    let hue = idx as f32 * (360.0 / 8.0);
    // Sprint 85: saturation 0.85 → 1.0 — sytější body color.
    // Sprint 88: emissive ∝ hue color. Pod HDR + Bloom cells „bioluminescent".
    // Sprint 91: ExtendedMaterial s pattern_kind=0 (jelly membrane). Voronoi
    // procedural shader moduluje base_color + emissive na povrchu mesh.
    let color = Color::hsl(hue, 1.0, 0.50);
    let emissive_color = Color::hsl(hue, 1.0, 0.50);
    let emissive_linear = emissive_color.to_linear();
    let handle = bio_materials.add(BioMaterial {
        base: StandardMaterial {
            base_color: color,
            // Sprint 91: emissive max ~1.0 → shader detekuje jako CELL pattern.
            emissive: LinearRgba::new(
                emissive_linear.red,
                emissive_linear.green,
                emissive_linear.blue,
                1.0,
            ),
            perceptual_roughness: 0.5,
            ..default()
        },
        extension: BioMaterialExt {},
    });
    cache.0[idx] = Some(handle.clone());
    handle
}

/// Sprint 69: hue pro adhesion gizmo lines. Match s `adhesion_material`
/// (= rovnoměrné rozdělení 360°/8 = 45° per type).
fn adhesion_hue(adhesion_type: u8) -> f32 {
    (adhesion_type as usize % 8) as f32 * (360.0 / 8.0)
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
    let low = [0.55_f32, 0.55, 0.55];
    let high = [0.92_f32, 0.92, 0.92];
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
    clock: Res<Clock>,
    mut cells: Query<&mut CellEntity>,
) {
    let dt = time.delta_secs();
    let half = extent.as_array();
    let tick = clock.0.tick;
    let gen = clock.0.generation;
    for mut cell in &mut cells {
        cell.0.step(dt, half, tick, gen, &PHYSICS_CONFIG);
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

/// Sprint 94: pre-brain pass. Compute `pooled_hidden` per cell = mean
/// `last_hidden` over self + bonded partners (1-hop). Cluster cells získají
/// shared recurrent state. Solo cells: pooled == self. Runs before
/// `cells_brain_act` v `FixedUpdate` chain.
fn pool_bonded_hidden_cells(
    mut cells: Query<&mut CellEntity, Without<Dying>>,
) {
    let snapshot: Vec<(u64, [f32; bioscape::BRAIN_HIDDEN])> = cells
        .iter()
        .map(|c| (c.0.cell_id, c.0.last_hidden))
        .collect();
    if snapshot.is_empty() {
        return;
    }
    let id_to_hidden: rustc_hash::FxHashMap<u64, [f32; bioscape::BRAIN_HIDDEN]> =
        snapshot.into_iter().collect();
    for mut cell in &mut cells {
        let pooled = bioscape::pool_bonded_hidden(&cell.0, |partner_id| {
            if partner_id == cell.0.cell_id {
                return None;
            }
            id_to_hidden.get(&partner_id).copied()
        });
        cell.0.pooled_hidden = pooled;
    }
}

fn cells_brain_act(
    time: Res<Time>,
    cell_grid: Res<CellGrid>,
    food_grid: Res<FoodGrid>,
    smell: Res<SmellResource>,
    pheromone: Res<PheromoneResource>,
    slot_map: Res<CellSlotMap>,
    clock: Res<Clock>,
    #[cfg(feature = "gpu")] gpu_state: Option<Res<GpuBrainState>>,
    mut cells: Query<(Entity, &mut CellEntity), Without<Dying>>,
    mut diag: Diagnostics,
) {
    let _t_total = Instant::now();
    let dt = time.delta_secs();
    diag.add_measurement(&DIAG_CELL_COUNT, || cells.iter().count() as f64);

    // Sprint 87: clock pro thermal_local sensor. Capture u64 mimo closure aby
    // kopie šly do Bevy threadu bez dalšího `Clock` borrow.
    let tick = clock.0.tick;
    let gen = clock.0.generation;

    // Sprint 52: helper closure pro per-cell sensor gather + populate_brain_inputs.
    // Reused jak v CPU tak GPU path. Takes &mut Cell + Entity, vrací inputs[36].
    let gather = |entity: Entity, cell: &mut Cell| -> [f32; BRAIN_INPUTS] {
        let pos = cell.position;
        let vision_r = cell.genome.vision_radius;
        let vr2 = vision_r * vision_r;
        // Sprint 83: precomputed cone parametry. `skip_cone` short-circuit pro
        // full-sphere FOV (cos(π) ≈ −1, jakýkoliv kandidát uvnitř radia by
        // procházel) — vyhne se per-callback sqrt.
        let fov = cell.genome.vision_fov;
        let skip_cone = fov >= bioscape::MAX_VISION_FOV;
        let cos_fov = fov.cos();
        let fwd = bioscape::forward_vector(cell.heading, cell.pitch);
        let mut nearest_food: Option<[f32; 3]> = None;
        let mut best_food_d2 = f32::MAX;
        food_grid.0.for_each_in_radius_toroidal(pos, vision_r, WORLD_HALF, |_, fp, _| {
            let d = bioscape::min_image_delta(pos, fp, WORLD_HALF);
            let d2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            if d2 > vr2 || d2 >= best_food_d2 {
                return;
            }
            if !skip_cone && !bioscape::fov_cone_accept(d, d2, fwd, cos_fov) {
                return;
            }
            best_food_d2 = d2;
            nearest_food = Some(d);
        });
        let mut nearest_cell: Option<([f32; 3], f32)> = None;
        let mut best_cell_d2 = f32::MAX;
        let mut neighbors_in_vision: u32 = 0;
        cell_grid
            .0
            .for_each_in_radius_toroidal(pos, vision_r, WORLD_HALF, |other, other_pos, other_radius| {
                if other == entity {
                    return;
                }
                let d = bioscape::min_image_delta(pos, other_pos, WORLD_HALF);
                let d2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                if d2 > vr2 {
                    return;
                }
                if !skip_cone && !bioscape::fov_cone_accept(d, d2, fwd, cos_fov) {
                    return;
                }
                neighbors_in_vision += 1;
                if d2 < best_cell_d2 {
                    best_cell_d2 = d2;
                    nearest_cell = Some((d, other_radius));
                }
            });
        let pos_xyz = [pos[0], pos[1], pos[2]];
        let smell_grad = smell.0.gradient_at(pos_xyz, SMELL_SAMPLE_EPSILON);
        let pheromone_grad = pheromone.0.gradient_at(pos_xyz, PHEROMONE_SAMPLE_EPSILON);
        let temperature_local = bioscape::temperature_at_z(pos[2], WORLD_HALF, tick, gen);
        let sensors = bioscape::BrainSensors {
            nearest_food,
            nearest_cell,
            neighbors_in_vision,
            smell_grad,
            pheromone_grad,
            temperature_local,
        };
        cell.apply_shell_absorb(dt);
        bioscape::populate_brain_inputs(cell, &sensors, vision_r)
    };

    // Sprint 97: dvojfázový pipeline pro cluster sensor pooling.
    // Phase 1: gather + apply per-cell sensor gains, ulož do id-keyed mapy.
    // Phase 2: pool max-magnitude přes bond network + brain forward.
    let mut id_to_inputs: rustc_hash::FxHashMap<u64, [f32; BRAIN_INPUTS]> =
        rustc_hash::FxHashMap::default();
    for (entity, mut cell) in &mut cells {
        let mut inputs = gather(entity, &mut cell.0);
        bioscape::apply_sensor_gains(&mut inputs, &cell.0.genome.sensor_gains);
        id_to_inputs.insert(cell.0.cell_id, inputs);
    }

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
        for (entity, cell) in cells.iter() {
            let Some(slot) = slot_map.slot_of(entity) else {
                continue;
            };
            let own = id_to_inputs
                .get(&cell.0.cell_id)
                .copied()
                .unwrap_or([0.0; BRAIN_INPUTS]);
            inputs_by_slot[slot] = bioscape::pool_bonded_sensors(&cell.0, &own, |partner_id| {
                if partner_id == cell.0.cell_id {
                    return None;
                }
                id_to_inputs.get(&partner_id).copied()
            });
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
    for (_entity, mut cell) in &mut cells {
        let own = id_to_inputs
            .get(&cell.0.cell_id)
            .copied()
            .unwrap_or([0.0; BRAIN_INPUTS]);
        let inputs = bioscape::pool_bonded_sensors(&cell.0, &own, |partner_id| {
            if partner_id == cell.0.cell_id {
                return None;
            }
            id_to_inputs.get(&partner_id).copied()
        });
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
    cells: Query<&CellEntity, Without<Dying>>,
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
                WORLD_HALF,
                |entity, cell_pos, _radius| {
                    if blocked {
                        return;
                    }
                    // Match headless: exclusion uses ellipsoid's max_axis, not
                    // effective_radius. Elongated cells extend past their sphere
                    // approximation along the long axis, so a sphere-radius
                    // exclusion would let food spawn inside the ellipsoid.
                    let max_axis = cells
                        .get(entity)
                        .map(|c| c.0.phenotype.max_axis())
                        .unwrap_or(MAX_BODY_LENGTH);
                    let exclusion = EAT_RADIUS * max_axis;
                    let d = bioscape::min_image_delta(candidate.position, cell_pos, WORLD_HALF);
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
    // Sprint 92: snapshot s carnivore_score per cell pro food efficiency lookup.
    let cell_carnivore: std::collections::HashMap<Entity, f32> = cells
        .iter()
        .map(|(e, c)| (e, c.0.genome.carnivore_score))
        .collect();
    let candidates: Vec<Option<(Entity, f32)>> = snapshot
        .par_iter()
        .map(|(entity, pos, max_axis, dims, heading, pitch)| {
            let eat_r = EAT_RADIUS * *max_axis;
            let carnivore_score = cell_carnivore.get(entity).copied().unwrap_or(0.0);
            let mut ate: Option<(Entity, f32)> = None;
            food_grid_ref.for_each_in_radius_toroidal(
                *pos,
                eat_r,
                WORLD_HALF,
                |food_e, food_pos, food_kind| {
                    if ate.is_some() {
                        return;
                    }
                    // Sprint 54: ghost food s min-imaged position pro toroidal eat_test.
                    let md = bioscape::min_image_delta(*pos, food_pos, WORLD_HALF);
                    let ghost_pos = [pos[0] + md[0], pos[1] + md[1], food_pos[2]];
                    if bioscape::eat_test_pose(*pos, *heading, *pitch, *dims, ghost_pos, EAT_RADIUS) {
                        // Sprint 92: food value = base_value(kind) × multiplier × value_factor
                        // × eat_efficiency(kind, carnivore_score). Hunter carrion má
                        // vyšší base ale vyžaduje carnivore digestion; plant herbivore.
                        let efficiency = bioscape::eat_efficiency(food_kind, carnivore_score);
                        let value = bioscape::food_base_value(food_kind)
                            * food_multiplier(
                                world_map.0.sample([food_pos[0], food_pos[1], 0.0]),
                            )
                            * Food {
                                position: food_pos,
                                age_ticks: 0,
                                kind: food_kind,
                            }
                            .value_factor()
                            * efficiency;
                        ate = Some((food_e, value));
                    }
                },
            );
            ate
        })
        .collect();

    // Sprint 78: cell_id → entity map pro food share lookup. Cells layout
    // se v eat_food fázi nemění, takže build-once je bezpečný.
    let id_to_entity: FxHashMap<u64, Entity> = cells
        .iter()
        .map(|(e, c)| (c.0.cell_id, e))
        .collect();

    // Pass 2 (sequential): resolve race + apply energy + Hebbian. First-cell-wins
    // per food entity (matches pre-Sprint-58 ordering).
    // Sprint 78: cluster food share. Sebráno do share_deltas Vec během iterace,
    // aplikováno post-loop kvůli simultaneous mutable borrow.
    let mut eaten: FxHashSet<Entity> = FxHashSet::default();
    let mut share_deltas: Vec<(Entity, f32)> = Vec::new();
    for ((entity, _, _, _, _, _), opt) in snapshot.iter().zip(candidates.iter()) {
        if let Some((food_e, value)) = opt {
            if eaten.contains(food_e) {
                continue;
            }
            eaten.insert(*food_e);
            let (bonds_copy, donor_state) = if let Ok((_, mut cell)) = cells.get_mut(*entity) {
                cell.0.energy += *value;
                let copy = cell.0.bonds;
                let state = cell.0.cell_state;
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
                (copy, state)
            } else {
                continue;
            };
            // Sprint 78: share s bonded partnery (free reward, no
            // conservation — modeluje tissue cooperation).
            // Sprint 80: donor's cell_state moduluje fraction. State≈0
            // (selfish) → ~0% share; state≈1 (altruist) → plný 30% share.
            // Sprint 87: cluster-size bonus — cells hluboko v tkáni sdílí
            // víc, posiluje selekci proti tissue-regime collapse.
            let n_bonds = bonds_copy.iter().filter(|b| b.is_some()).count() as f32;
            let cluster_mult = 1.0 + (n_bonds - 1.0).max(0.0)
                * bioscape::BOND_FOOD_SHARE_CLUSTER_BONUS;
            let share_value =
                *value * bioscape::BOND_FOOD_SHARE_FRAC * donor_state * cluster_mult;
            if share_value > 0.0 {
                for bond_opt in bonds_copy.iter() {
                    if let Some(bond) = bond_opt {
                        if let Some(&partner_e) =
                            id_to_entity.get(&bond.other_cell_id)
                        {
                            if partner_e != *entity {
                                share_deltas.push((partner_e, share_value));
                            }
                        }
                    }
                }
            }
        }
    }

    // Sprint 78: aplikuj food share delty (po Pass 2 main loop).
    for (e, delta) in share_deltas {
        if let Ok((_, mut cell)) = cells.get_mut(e) {
            cell.0.energy += delta;
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

/// Sprint 71: macropredator step + attack. Per Hunter: nearest_attackable_cell
/// (vision range, n_bonds < threshold), pohni se k němu, attack pokud v dosahu.
/// Two-pass kvůli borrow checkeru: pass 1 sbírá (entity, damage) během
/// `&mut HunterEntity` iterace, pass 2 mutuje cells po uvolnění hunter borrow.
///
/// Sprint 89: heritable predator parameters. Vision/attack radius/damage z
/// `genome` místo const. Per-tick energy costs (vision/motion/body/attack
/// upkeep) volá Hunter::apply_energy_costs. Energy gain z attack proporčně
/// damage × HUNTER_ENERGY_PER_DAMAGE — predator se musí krmit aby přežil.
/// Lifecycle (death + reproduce + floor respawn) v separate systemu
/// `hunters_lifecycle` runs po `step_hunters`.
///
/// Sprint 90: brain-driven motion. Sensor gather → brain forward →
/// apply_brain_motor → step (kinematic). Brain learns chase tactics over
/// generations; INNATE_THRUST_BIAS dává initial forward motion.
fn step_hunters(
    mut hunters: Query<&mut HunterEntity>,
    mut cells: Query<(Entity, &mut CellEntity), Without<Dying>>,
    smell: Res<SmellResource>,
    fixed_time: Res<Time<Fixed>>,
) {
    let dt = fixed_time.delta_secs();
    let cell_snapshot: Vec<(Entity, Cell)> = cells.iter().map(|(e, c)| (e, c.0)).collect();
    let cells_only: Vec<Cell> = cell_snapshot.iter().map(|(_, c)| *c).collect();
    // Sprint 100: snapshot hunters pro pack sensing.
    let hunters_snapshot: Vec<Hunter> = hunters.iter().map(|h| h.0).collect();
    let mut attacks: Vec<(Entity, f32)> = Vec::new();
    for mut h in &mut hunters {
        // Sprint 90: sensor gather + brain forward + hybrid motor (seek+brain).
        let sensors = bioscape::gather_hunter_sensors(
            &h.0,
            &cells_only,
            &hunters_snapshot,
            &smell.0,
            WORLD_HALF,
        );
        let target_idx_pre = nearest_attackable_cell(&h.0, &cells_only, WORLD_HALF);
        let seek_target = target_idx_pre.map(|i| cells_only[i].position);
        let inputs = bioscape::populate_hunter_brain_inputs(&mut h.0, &sensors);
        let (hidden, outputs) = h.0.genome.brain.forward_with_state(&inputs);
        h.0.last_inputs = inputs;
        h.0.last_hidden = hidden;
        h.0.last_outputs = outputs;
        h.0.apply_brain_motor(&outputs, seek_target, dt, WORLD_HALF);
        h.0.step(dt, WORLD_HALF);
        // Attack check (post-step pozice).
        let target_idx = nearest_attackable_cell(&h.0, &cells_only, WORLD_HALF);
        let attack_r = h.0.genome.attack_radius;
        let attack_r2 = attack_r * attack_r;
        let damage = h.0.genome.damage_per_tick;
        let mut gain = 0.0_f32;
        if let Some(i) = target_idx {
            let d = bioscape::min_image_delta(
                h.0.position,
                cells_only[i].position,
                WORLD_HALF,
            );
            let d2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            if d2 < attack_r2 {
                // Sprint 92: edge-vulnerability — damage scales s exposure
                // (= 1 - n_bonds × EXPOSURE_PER_BOND). Solo cells take full
                // damage, deeply interior cells take 0.
                let exposure = bioscape::cell_exposure(cells_only[i].n_bonds());
                let damage_dealt = damage * exposure * dt;
                attacks.push((cell_snapshot[i].0, damage_dealt));
                gain = damage_dealt * bioscape::HUNTER_ENERGY_PER_DAMAGE;
            }
        }
        h.0.apply_energy_costs(dt);
        h.0.energy += gain;
    }
    for (entity, damage) in attacks {
        if let Ok((_, mut cell)) = cells.get_mut(entity) {
            cell.0.energy -= damage;
            cell.0.damage_accum += damage;
        }
    }
}

/// Sprint 89: hunter lifecycle — death (energy ≤ 0 → drop carrion + despawn),
/// reproduce (energy ≥ THRESHOLD + cooldown 0 → split + clone+mutate child),
/// floor respawn (n_hunters == 0 → 1 fresh genome). MAX_POP cap brání runaway.
/// Asexual v1 — Sprint 91+ může přidat sexual pairing.
fn hunters_lifecycle(
    hunters: Query<(Entity, &HunterEntity)>,
    extent: Res<WorldExtent>,
    hunter_mesh: Res<HunterMesh>,
    hunter_material: Res<HunterMaterial>,
    food_mesh: Res<FoodMesh>,
    food_material: Res<FoodMaterial>,
    clock: Res<Clock>,
    mut next_hunter_id: ResMut<NextHunterId>,
    mut commands: Commands,
) {
    let mut rng = rand::rng();
    let half = extent.as_array();
    let current_gen = clock.0.generation;
    let alive: Vec<(Entity, Hunter)> = hunters.iter().map(|(e, h)| (e, h.0)).collect();

    // Floor respawn: pokud all extinct, spawn 1 fresh genome (předchází total
    // predator collapse blokující arms race).
    if alive.is_empty() {
        let id = next_hunter_id.0;
        next_hunter_id.0 += 1;
        let h = Hunter::random(&mut rng, half, id, id, current_gen);
        commands.spawn((
            HunterEntity(h),
            Mesh3d(hunter_mesh.0.clone()),
            MeshMaterial3d(hunter_material.0.clone()),
            Transform::from_xyz(h.position[0], h.position[1], h.position[2]),
        ));
        return;
    }

    // Death pass.
    for (entity, h) in &alive {
        if h.energy <= 0.0 {
            commands.entity(*entity).despawn();
            for _ in 0..bioscape::HUNTER_CARRION_DROP {
                let pos = [
                    (h.position[0] + rng.random_range(-CELL_RADIUS..CELL_RADIUS))
                        .clamp(-half[0], half[0]),
                    (h.position[1] + rng.random_range(-CELL_RADIUS..CELL_RADIUS))
                        .clamp(-half[1], half[1]),
                    h.position[2].clamp(-half[2], half[2]),
                ];
                commands.spawn((
                    FoodEntity(Food {
                        position: pos,
                        age_ticks: 0,
                        kind: bioscape::FoodKind::HunterCarrion,
                    }),
                    Mesh3d(food_mesh.0.clone()),
                    MeshMaterial3d(food_material.0.clone()),
                    Transform::from_xyz(pos[0], pos[1], pos[2]),
                ));
            }
        }
    }

    // Sprint 98: sexual reproduction. Mirror headless logic — pair fertile
    // entities přes prostorovou blízkost, každý pár → 1 mating child, oba
    // rodiče zaplatí (halve + cooldown). Floor respawn nahoře pokrývá total
    // extinction.
    let alive_count = alive.iter().filter(|(_, h)| h.energy > 0.0).count();
    let budget = bioscape::HUNTER_MAX_POP.saturating_sub(alive_count);
    if budget == 0 {
        return;
    }
    let fertile: Vec<(Entity, [f32; 3])> = alive
        .iter()
        .filter(|(_, h)| {
            h.energy >= bioscape::HUNTER_REPRODUCE_THRESHOLD
                && h.reproduce_cooldown_ticks == 0
        })
        .map(|(e, h)| (*e, h.position))
        .collect();
    if fertile.len() < 2 {
        return;
    }
    let mating_r2 = bioscape::HUNTER_MATING_RADIUS * bioscape::HUNTER_MATING_RADIUS;
    let matings = bioscape::pair_fertile(&fertile, mating_r2, budget, WORLD_HALF);
    let lookup: FxHashMap<Entity, Hunter> =
        alive.iter().map(|(e, h)| (*e, *h)).collect();
    for &(ea, eb) in &matings {
        let parent_a = match lookup.get(&ea) {
            Some(p) => *p,
            None => continue,
        };
        let parent_b = match lookup.get(&eb) {
            Some(p) => *p,
            None => continue,
        };
        // Halve both parents PŘED make_*_mating_child (energy semantics z cell
        // mating: child.energy = a + b součet už halved values, takže celkem
        // konzervovaná energy a + b post-mating).
        let mut a_halved = parent_a;
        let mut b_halved = parent_b;
        a_halved.energy *= 0.5;
        b_halved.energy *= 0.5;
        let id = next_hunter_id.0;
        next_hunter_id.0 += 1;
        let child = bioscape::make_hunter_mating_child(
            &a_halved,
            &b_halved,
            &mut rng,
            half,
            id,
            current_gen,
        );
        commands.spawn((
            HunterEntity(child),
            Mesh3d(hunter_mesh.0.clone()),
            MeshMaterial3d(hunter_material.0.clone()),
            Transform::from_xyz(child.position[0], child.position[1], child.position[2]),
        ));
        // Update parent ECS components: halved energy + cooldown.
        a_halved.reproduce_cooldown_ticks = bioscape::HUNTER_REPRODUCE_COOLDOWN_TICKS;
        b_halved.reproduce_cooldown_ticks = bioscape::HUNTER_REPRODUCE_COOLDOWN_TICKS;
        commands.entity(ea).insert(HunterEntity(a_halved));
        commands.entity(eb).insert(HunterEntity(b_halved));
    }
}

/// Sprint 71: sync Hunter.position → Transform pro renderer. Hunter mesh
/// se nerotuje (sphere — žádná orientace), takže Transform se updatuje jen
/// translation. Mesh scale je fixed v setup, neměnné per-tick.
fn sync_hunter_transforms(mut hunters: Query<(&HunterEntity, &mut Transform)>) {
    for (h, mut transform) in &mut hunters {
        transform.translation = Vec3::new(h.0.position[0], h.0.position[1], h.0.position[2]);
    }
}

/// Sprint 69: render persistent spring bonds jako gizmo lines. Hue podle
/// `adhesion_type` (bondy se tvoří jen mezi same-type páry, takže obě cells
/// sdílí hue). Toroidal wrap-aware: skip line, pokud raw distance > poloviny
/// world (znamená že bond jde "přes okraj", straight line by visuálně lhala).
fn draw_bond_gizmos(
    cells: Query<&CellEntity, Without<Dying>>,
    mut gizmos: Gizmos,
) {
    let mut id_to_pos: FxHashMap<u64, Vec3> = FxHashMap::default();
    for cell in &cells {
        id_to_pos.insert(
            cell.0.cell_id,
            Vec3::new(cell.0.position[0], cell.0.position[1], cell.0.position[2]),
        );
    }
    let half_x = WORLD_HALF[0];
    let half_y = WORLD_HALF[1];
    for cell in &cells {
        let start = Vec3::new(cell.0.position[0], cell.0.position[1], cell.0.position[2]);
        let hue = adhesion_hue(cell.0.genome.adhesion_type);
        // Sprint 85: saturation 0.85 → 1.0, match s body color v adhesion_material.
        // Sprint 88: linear color × 3.0 multiplier — Bevy gizmos render do HDR
        // backbufferu, super-bright hodnoty Bloom catches → bondy svítí jako
        // skutečné spring laser-lines.
        let base = Color::hsl(hue, 1.0, 0.6).to_linear();
        let color = Color::linear_rgba(base.red * 3.0, base.green * 3.0, base.blue * 3.0, 1.0);
        for bond in cell.0.bonds.iter().flatten() {
            let Some(end) = id_to_pos.get(&bond.other_cell_id) else {
                continue;
            };
            // Each bond rendered jen jednou — kresli pouze pokud cell_id <
            // partner_id (canonical owner pravidlo).
            if cell.0.cell_id >= bond.other_cell_id {
                continue;
            }
            let dx = (start.x - end.x).abs();
            let dy = (start.y - end.y).abs();
            if dx > half_x || dy > half_y {
                continue;
            }
            gizmos.line(start, *end, color);
        }
    }
}

/// Sprint 80: vertical marker per cell colored by `cell_state`. Modrá =
/// selfish (state≈0), červená = altruist (state≈1). Per-cell StandardMaterial
/// rebind by byl drahý (každý tick allocate handle), gizmo line je free.
fn draw_cell_state_gizmos(
    cells: Query<&CellEntity, Without<Dying>>,
    mut gizmos: Gizmos,
) {
    for cell in &cells {
        let s = cell.0.cell_state.clamp(0.0, 1.0);
        let pos = Vec3::new(cell.0.position[0], cell.0.position[1], cell.0.position[2]);
        // Marker výška: 1.5× max body axis nad cell, viditelné nezávisle
        // na velikosti těla.
        let h = cell.0.phenotype.max_axis() * 1.5 + 1.0;
        let top = pos + Vec3::new(0.0, 0.0, h);
        // Lerp blue → red v sRGB. Mezistav kolem 0.5 = magenta = vidíme,
        // jak cells přecházejí přes attractor boundary.
        let color = Color::srgb(s, 0.05, 1.0 - s);
        gizmos.line(pos, top, color);
    }
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
    // Sprint 73 chore: right button orbit alias. Blender/CAD-style users
    // očekávají rotaci na pravém tlačítku; left button zůstává jako
    // primary (Bevy default), right je pohodlná alternativa.
    let orbit_active =
        buttons.pressed(MouseButton::Left) || buttons.pressed(MouseButton::Right);
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
    mut adhesion_materials: ResMut<AdhesionMaterials>,
    mut bio_materials: ResMut<Assets<BioMaterial>>,
    mut slot_map: ResMut<CellSlotMap>,
    mut next_cell_id: ResMut<NextCellId>,
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
    let matings = bioscape::pair_fertile(&fertile, mating_r2, budget, WORLD_HALF);

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
        // Sprint 66: child gets stable cell_id from monotonic counter.
        let child_id = next_cell_id.0;
        next_cell_id.0 += 1;
        to_spawn.push(bioscape::make_mating_child(
            &cell_a.0, &cell_b.0, &mut rng, child_id,
        ));
    }

    let mesh = cell_mesh.0.clone();
    for cell in to_spawn {
        let mat = adhesion_material(
            &mut adhesion_materials,
            &mut bio_materials,
            cell.genome.adhesion_type,
        );
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
                    FoodEntity(Food {
                        position: pos,
                        age_ticks: 0,
                        kind: bioscape::FoodKind::Carrion,
                    }),
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

/// Sprint 100: pool last_hidden napříč hunter packem (mirror headless
/// `pool_bonded_hunter_hidden`). Snapshot all → call lib helper → write
/// back via commands.entity().insert. ECS-flavored wrapper.
fn pool_bonded_hunter_hidden_system(
    hunters: Query<(Entity, &HunterEntity)>,
    mut commands: Commands,
) {
    let mut state: Vec<(Entity, Hunter)> = hunters.iter().map(|(e, h)| (e, h.0)).collect();
    if state.is_empty() {
        return;
    }
    let mut hunters_only: Vec<Hunter> = state.iter().map(|(_, h)| *h).collect();
    bioscape::pool_bonded_hunter_hidden(&mut hunters_only);
    for ((entity, _), updated) in state.iter_mut().zip(hunters_only.iter()) {
        commands.entity(*entity).insert(HunterEntity(*updated));
    }
}

/// Sprint 99: hunter-hunter collision + adhesion + bond physics. Mirror
/// headless `resolve_hunter_collisions` — O(N²) sequential pro N ≤ 50.
/// Snapshot all hunters → compute deltas + bond formation/pruning →
/// write back via `commands.entity().insert()`.
fn resolve_hunter_collisions(
    hunters: Query<(Entity, &HunterEntity)>,
    mut contact: ResMut<HunterContactProgress>,
    mut commands: Commands,
) {
    let alive: Vec<(Entity, Hunter)> = hunters.iter().map(|(e, h)| (e, h.0)).collect();
    let n = alive.len();
    if n < 2 {
        return;
    }
    let hunter_radius = |h: &Hunter| h.genome.body_size * CELL_RADIUS;
    let id_to_pos: FxHashMap<u64, usize> = alive
        .iter()
        .enumerate()
        .map(|(i, (_, h))| (h.hunter_id, i))
        .collect();

    let mut pos_deltas: Vec<[f32; 3]> = vec![[0.0; 3]; n];
    let mut vel_deltas: Vec<[f32; 3]> = vec![[0.0; 3]; n];
    let mut in_contact_pairs: FxHashSet<(u64, u64)> = FxHashSet::default();

    for i in 0..n {
        let (_, hunter_i) = &alive[i];
        let pos_i = hunter_i.position;
        let vel_i = hunter_i.velocity;
        let radius_i = hunter_radius(hunter_i);
        let type_i = hunter_i.genome.adhesion_type;
        let id_i = hunter_i.hunter_id;
        for j in 0..n {
            if i == j {
                continue;
            }
            let (_, hunter_j) = &alive[j];
            let pos_j = hunter_j.position;
            let radius_j = hunter_radius(hunter_j);
            let pair_r = radius_i + radius_j;
            let d_vec = bioscape::min_image_delta(pos_j, pos_i, WORLD_HALF);
            let d2 = d_vec[0] * d_vec[0] + d_vec[1] * d_vec[1] + d_vec[2] * d_vec[2];
            let d = d2.sqrt();
            let in_contact = d2 < pair_r * pair_r && d2 > 0.0;
            if in_contact {
                let overlap = pair_r - d;
                let nx = d_vec[0] / d;
                let ny = d_vec[1] / d;
                let nz = d_vec[2] / d;
                pos_deltas[i][0] -= nx * overlap * 0.5;
                pos_deltas[i][1] -= ny * overlap * 0.5;
                pos_deltas[i][2] -= nz * overlap * 0.5;
                let id_j = hunter_j.hunter_id;
                let pair = if id_i < id_j { (id_i, id_j) } else { (id_j, id_i) };
                in_contact_pairs.insert(pair);
            } else if d > 0.0 {
                let type_j = hunter_j.genome.adhesion_type;
                let same_type = type_i == type_j;
                let dv = bioscape::adhesion_velocity_delta(d_vec, d, pair_r, same_type);
                vel_deltas[i][0] += dv[0];
                vel_deltas[i][1] += dv[1];
                vel_deltas[i][2] += dv[2];
            }
        }
        // Apply bond spring forces.
        for bond_opt in hunter_i.bonds.iter() {
            if let Some(bond) = bond_opt {
                if let Some(&j_idx) = id_to_pos.get(&bond.other_cell_id) {
                    let (_, hunter_j) = &alive[j_idx];
                    let pos_j = hunter_j.position;
                    let vel_j = hunter_j.velocity;
                    let d_vec = bioscape::min_image_delta(pos_j, pos_i, WORLD_HALF);
                    let dist = (d_vec[0] * d_vec[0]
                        + d_vec[1] * d_vec[1]
                        + d_vec[2] * d_vec[2])
                        .sqrt();
                    let (dv, _broken) =
                        bioscape::bond_velocity_delta(bond, d_vec, dist, vel_i, vel_j);
                    vel_deltas[i][0] += dv[0];
                    vel_deltas[i][1] += dv[1];
                    vel_deltas[i][2] += dv[2];
                }
            }
        }
    }

    // Contact tracker update (mirror headless).
    let mut new_progress: FxHashMap<(u64, u64), u32> = FxHashMap::default();
    for &pair in &in_contact_pairs {
        let prev = contact.0.get(&pair).copied().unwrap_or(0);
        new_progress.insert(pair, prev.saturating_add(1));
    }
    for (&pair, &val) in contact.0.iter() {
        if !in_contact_pairs.contains(&pair) && val > 1 {
            new_progress.insert(pair, val - 1);
        }
    }
    contact.0 = new_progress;

    // Build mutable snapshot pro deltas + bond updates.
    let mut new_state: Vec<(Entity, Hunter)> = alive.clone();
    for ((entity_pair, pd), vd) in new_state
        .iter_mut()
        .zip(pos_deltas.iter())
        .zip(vel_deltas.iter())
    {
        let h = &mut entity_pair.1;
        h.position[0] += pd[0];
        h.position[1] += pd[1];
        h.position[2] += pd[2];
        h.velocity[0] += vd[0];
        h.velocity[1] += vd[1];
        h.velocity[2] += vd[2];
    }

    // Bond formation.
    let candidates: Vec<(u64, u64)> = contact
        .0
        .iter()
        .filter(|(_, &t)| t >= bioscape::BOND_FORM_TICKS)
        .map(|(&pair, _)| pair)
        .collect();
    for (id_a, id_b) in candidates {
        let (Some(&a_idx), Some(&b_idx)) = (id_to_pos.get(&id_a), id_to_pos.get(&id_b)) else {
            continue;
        };
        if new_state[a_idx].1.genome.adhesion_type != new_state[b_idx].1.genome.adhesion_type {
            continue;
        }
        // Sprint 100: brain output[9] gate.
        if new_state[a_idx].1.last_outputs[9] < bioscape::BOND_FORM_THRESHOLD
            || new_state[b_idx].1.last_outputs[9] < bioscape::BOND_FORM_THRESHOLD
        {
            continue;
        }
        let already = new_state[a_idx]
            .1
            .bonds
            .iter()
            .any(|b| b.as_ref().map_or(false, |bb| bb.other_cell_id == id_b));
        if already {
            continue;
        }
        let slot_a = new_state[a_idx].1.bonds.iter().position(|b| b.is_none());
        let slot_b = new_state[b_idx].1.bonds.iter().position(|b| b.is_none());
        if let (Some(sa), Some(sb)) = (slot_a, slot_b) {
            let pos_a = new_state[a_idx].1.position;
            let pos_b = new_state[b_idx].1.position;
            let d_vec = bioscape::min_image_delta(pos_b, pos_a, WORLD_HALF);
            let dist =
                (d_vec[0] * d_vec[0] + d_vec[1] * d_vec[1] + d_vec[2] * d_vec[2]).sqrt();
            let rest = dist * bioscape::BOND_REST_LENGTH_SLACK;
            new_state[a_idx].1.bonds[sa] = Some(Bond {
                other_cell_id: id_b,
                rest_length: rest,
                stiffness: bioscape::BOND_STIFFNESS,
                damping: bioscape::BOND_DAMPING,
                age_ticks: 0,
            });
            new_state[b_idx].1.bonds[sb] = Some(Bond {
                other_cell_id: id_a,
                rest_length: rest,
                stiffness: bioscape::BOND_STIFFNESS,
                damping: bioscape::BOND_DAMPING,
                age_ticks: 0,
            });
        }
    }

    // Pruning + age increment.
    for (_, hunter) in new_state.iter_mut() {
        for bond_opt in hunter.bonds.iter_mut() {
            if let Some(bond) = bond_opt {
                if !id_to_pos.contains_key(&bond.other_cell_id) {
                    *bond_opt = None;
                } else {
                    bond.age_ticks = bond.age_ticks.saturating_add(1);
                }
            }
        }
    }

    // Writeback ECS state.
    for (entity, h) in new_state {
        commands.entity(entity).insert(HunterEntity(h));
    }
}

fn rebuild_food_grid(
    mut grid: ResMut<FoodGrid>,
    foods: Query<(Entity, &FoodEntity)>,
) {
    grid.0.rebuild(foods.iter().map(|(e, f)| (e, f.0.position, f.0.kind)));
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
    // Sprint 69: per-cell bond count pro bond_defense_factor lookup v
    // grid callback. Sebrané jednorázově ze stejného iter() — kompatibilní
    // s rayon herd_counts pre-pass.
    let bond_counts: FxHashMap<Entity, u32> = cells
        .iter()
        .map(|(e, c)| (e, c.0.n_bonds()))
        .collect();
    let grid_ref = &grid.0;
    let herd_counts_vec: Vec<u32> = snapshot
        .par_iter()
        .map(|(entity, pos)| {
            let mut count: u32 = 0;
            grid_ref.for_each_in_radius_toroidal(
                *pos,
                HERD_RADIUS,
                WORLD_HALF,
                |other, other_pos, _| {
                    if other == *entity {
                        return;
                    }
                    let d = bioscape::min_image_delta(*pos, other_pos, WORLD_HALF);
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
            .for_each_in_radius_toroidal(pos_a, broad_r, WORLD_HALF, |entity_b, pos_b, radius_b| {
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
                let d = bioscape::min_image_delta(pos_a, pos_b, WORLD_HALF);
                let d2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                if d2 >= pair_r2 {
                    return;
                }
                let ghost_b = [pos_a[0] + d[0], pos_a[1] + d[1], pos_a[2] + d[2]];
                let bonus = cell_a.0.spike_bonus_against(ghost_b);
                let gain_raw = PREDATION_GAIN_PER_TICK + bonus;
                let prey_neighbors = *herd_counts.get(&entity_b).unwrap_or(&0);
                let dilution = 1.0 / (1.0 + DILUTION_K * prey_neighbors as f32);
                // Sprint 69: bonded prey takes less damage + yields less energy.
                // bond_count_b je 0 pokud entity_b není v map (mrtvá / not yet
                // v snapshot) — graceful fallback na "no defense".
                let bond_count_b = *bond_counts.get(&entity_b).unwrap_or(&0);
                let defense = bioscape::bond_defense_factor(bond_count_b);
                let gain = gain_raw * dilution * defense;
                let drain = PREDATION_DRAIN_PER_TICK * defense;
                *energy_changes.entry(entity_a).or_insert(0.0) += gain;
                *energy_changes.entry(entity_b).or_insert(0.0) -= drain;
                *damage_changes.entry(entity_b).or_insert(0.0) += drain;
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
    mut contact_progress: ResMut<ContactProgress>,
    mut diag: Diagnostics,
) {
    let t_total = Instant::now();
    // Sprint 58: snapshot + rayon par compute deltas.
    // Sprint 65: 3D position delta + inelastic velocity damping.
    // Sprint 66: differential adhesion + persistent spring bonds + contact
    // tick tracker pro hybrid bond formation. Snapshot rozšířen o cell_id,
    // adhesion_type, bonds.
    let snapshot: Vec<SnapEntry> = cells
        .iter()
        .map(|(e, c)| SnapEntry {
            entity: e,
            cell_id: c.0.cell_id,
            position: c.0.position,
            velocity: c.0.velocity,
            radius: c.0.phenotype.effective_radius(),
            adhesion_type: c.0.genome.adhesion_type,
            bonds: c.0.bonds,
        })
        .collect();
    // O(1) lookups pro Phase 1 hot loop.
    let entity_to_idx: FxHashMap<Entity, usize> = snapshot
        .iter()
        .enumerate()
        .map(|(i, s)| (s.entity, i))
        .collect();
    let id_to_idx: FxHashMap<u64, usize> = snapshot
        .iter()
        .enumerate()
        .map(|(i, s)| (s.cell_id, i))
        .collect();
    let grid_ref = &grid.0;
    // Phase 1 (parallel): per-cell delta + vel_delta + collected contacts.
    let results: Vec<(Entity, [f32; 3], [f32; 3], Vec<u64>)> = snapshot
        .par_iter()
        .map(|s_a| {
            let entity_a = s_a.entity;
            let pos_a = s_a.position;
            let vel_a = s_a.velocity;
            let radius_a = s_a.radius;
            let cell_id_a = s_a.cell_id;
            let collision_r = CELL_RADIUS * (radius_a + BROAD_PHASE_SIZE_BUDGET);
            let adhesion_r = collision_r * ADHESION_RANGE_FACTOR;
            let broad_r = collision_r.max(adhesion_r);
            let mut delta = [0.0_f32, 0.0_f32, 0.0_f32];
            let mut vel_delta = [0.0_f32, 0.0_f32, 0.0_f32];
            let mut local_contacts: Vec<u64> = Vec::new();
            grid_ref.for_each_in_radius_toroidal(
                pos_a,
                broad_r,
                WORLD_HALF,
                |entity_b, pos_b, radius_b| {
                    if entity_b == entity_a {
                        return;
                    }
                    let Some(&j_idx) = entity_to_idx.get(&entity_b) else {
                        return;
                    };
                    let pair_r = CELL_RADIUS * (radius_a + radius_b);
                    let pair_r2 = pair_r * pair_r;
                    let d_vec = bioscape::min_image_delta(pos_b, pos_a, WORLD_HALF);
                    let d2 = d_vec[0] * d_vec[0] + d_vec[1] * d_vec[1] + d_vec[2] * d_vec[2];
                    let d = d2.sqrt();
                    let in_contact = d2 < pair_r2 && d2 > 0.0;
                    if in_contact {
                        let overlap = pair_r - d;
                        let nx = d_vec[0] / d;
                        let ny = d_vec[1] / d;
                        let nz = d_vec[2] / d;
                        delta[0] += nx * overlap * 0.5;
                        delta[1] += ny * overlap * 0.5;
                        delta[2] += nz * overlap * 0.5;
                        let vel_b = snapshot[j_idx].velocity;
                        let v_rel = [
                            vel_a[0] - vel_b[0],
                            vel_a[1] - vel_b[1],
                            vel_a[2] - vel_b[2],
                        ];
                        let v_rel_n = v_rel[0] * nx + v_rel[1] * ny + v_rel[2] * nz;
                        if v_rel_n < 0.0 {
                            let damp =
                                -v_rel_n * 0.5 * (1.0 - bioscape::COLLISION_RESTITUTION);
                            vel_delta[0] += damp * nx;
                            vel_delta[1] += damp * ny;
                            vel_delta[2] += damp * nz;
                        }
                        let cell_id_b = snapshot[j_idx].cell_id;
                        if cell_id_a < cell_id_b {
                            local_contacts.push(cell_id_b);
                        }
                    } else if d > 0.0 {
                        let same_type =
                            s_a.adhesion_type == snapshot[j_idx].adhesion_type;
                        let dv = adhesion_velocity_delta(d_vec, d, pair_r, same_type);
                        vel_delta[0] += dv[0];
                        vel_delta[1] += dv[1];
                        vel_delta[2] += dv[2];
                    }
                },
            );
            // Sprint 66: spring bond force pro každý živý bond.
            for bond_opt in s_a.bonds.iter() {
                if let Some(bond) = bond_opt {
                    if let Some(&j_idx) = id_to_idx.get(&bond.other_cell_id) {
                        let pos_j = snapshot[j_idx].position;
                        let vel_j = snapshot[j_idx].velocity;
                        let d_vec =
                            bioscape::min_image_delta(pos_j, pos_a, WORLD_HALF);
                        let dist =
                            (d_vec[0] * d_vec[0] + d_vec[1] * d_vec[1] + d_vec[2] * d_vec[2])
                                .sqrt();
                        let (dv, _broken) =
                            bond_velocity_delta(bond, d_vec, dist, vel_a, vel_j);
                        vel_delta[0] += dv[0];
                        vel_delta[1] += dv[1];
                        vel_delta[2] += dv[2];
                    }
                }
            }
            (entity_a, delta, vel_delta, local_contacts)
        })
        .collect();

    // Phase 2 (sequential): apply deltas + bond age/prune + contact tracker
    // + bond formation.
    let dt = 1.0 / FIXED_TIMESTEP_HZ;
    let mut seen_pairs: FxHashSet<(u64, u64)> = FxHashSet::default();
    for (entity, delta, vel_delta, contacts) in &results {
        let Ok((_, mut cell)) = cells.get_mut(*entity) else {
            continue;
        };
        cell.0.position[0] += delta[0];
        cell.0.position[1] += delta[1];
        cell.0.position[2] += delta[2];
        cell.0.velocity[0] += vel_delta[0];
        cell.0.velocity[1] += vel_delta[1];
        cell.0.velocity[2] += vel_delta[2];
        let cell_id_a = cell.0.cell_id;
        for &other_id in contacts {
            let key = (cell_id_a, other_id);
            seen_pairs.insert(key);
            let entry = contact_progress.0.entry(key).or_insert(0);
            *entry = entry.saturating_add(1);
        }
    }
    // Bond pruning + maintenance — re-snapshot positions po Phase 2.
    let positions: FxHashMap<u64, [f32; 3]> = cells
        .iter()
        .map(|(_, c)| (c.0.cell_id, c.0.position))
        .collect();
    for (_, mut cell) in cells.iter_mut() {
        let outputs_9 = cell.0.last_outputs[9];
        let explicit_break = outputs_9 < BOND_BREAK_THRESHOLD;
        let pos_i = cell.0.position;
        let mut bond_count = 0_usize;
        for slot in 0..MAX_BONDS_PER_CELL {
            let Some(bond) = cell.0.bonds[slot] else { continue };
            if explicit_break {
                cell.0.bonds[slot] = None;
                continue;
            }
            let Some(&pos_j) = positions.get(&bond.other_cell_id) else {
                cell.0.bonds[slot] = None;
                continue;
            };
            let d_vec = bioscape::min_image_delta(pos_j, pos_i, WORLD_HALF);
            let d = (d_vec[0] * d_vec[0] + d_vec[1] * d_vec[1] + d_vec[2] * d_vec[2]).sqrt();
            if d > bond.rest_length * bioscape::BOND_BREAK_FACTOR || d <= f32::EPSILON {
                cell.0.bonds[slot] = None;
                continue;
            }
            if let Some(b) = cell.0.bonds[slot].as_mut() {
                b.age_ticks = b.age_ticks.saturating_add(1);
            }
            bond_count += 1;
        }
        if bond_count > 0 {
            cell.0.energy -= bond_count as f32 * BOND_MAINTENANCE_PER_SEC * dt;
        }
    }
    // Contact tracker decay.
    contact_progress.0.retain(|key, ticks| {
        if seen_pairs.contains(key) {
            true
        } else if *ticks > CONTACT_DECAY_TICKS {
            *ticks -= CONTACT_DECAY_TICKS;
            true
        } else {
            false
        }
    });
    // Bond formation — kandidáti, kteří dosáhli BOND_FORM_TICKS thresholdu.
    let candidates: Vec<(u64, u64)> = contact_progress
        .0
        .iter()
        .filter_map(|(&pair, &ticks)| if ticks >= BOND_FORM_TICKS { Some(pair) } else { None })
        .collect();
    for (id_a, id_b) in candidates {
        let Some(&i_a) = id_to_idx.get(&id_a) else { continue };
        let Some(&i_b) = id_to_idx.get(&id_b) else { continue };
        let sa = &snapshot[i_a];
        let sb = &snapshot[i_b];
        if sa.adhesion_type != sb.adhesion_type {
            continue;
        }
        let Ok([(_, mut ca), (_, mut cb)]) = cells.get_many_mut([sa.entity, sb.entity]) else {
            continue;
        };
        if ca.0.last_outputs[9] <= BOND_FORM_THRESHOLD
            || cb.0.last_outputs[9] <= BOND_FORM_THRESHOLD
        {
            continue;
        }
        let already = ca
            .0
            .bonds
            .iter()
            .any(|b| b.map(|bb| bb.other_cell_id == id_b).unwrap_or(false));
        if already {
            continue;
        }
        let slot_a = ca.0.bonds.iter().position(|b| b.is_none());
        let slot_b = cb.0.bonds.iter().position(|b| b.is_none());
        let (Some(sa_slot), Some(sb_slot)) = (slot_a, slot_b) else { continue };
        let pos_a = ca.0.position;
        let pos_b = cb.0.position;
        let d_vec = bioscape::min_image_delta(pos_b, pos_a, WORLD_HALF);
        let dist =
            (d_vec[0] * d_vec[0] + d_vec[1] * d_vec[1] + d_vec[2] * d_vec[2]).sqrt();
        let rest = dist * BOND_REST_LENGTH_SLACK;
        // Sprint 68: per-bond stiffness/damping = mean obou cells' genes.
        let stiffness =
            (ca.0.genome.bond_stiffness + cb.0.genome.bond_stiffness) * 0.5;
        let damping = (ca.0.genome.bond_damping + cb.0.genome.bond_damping) * 0.5;
        ca.0.bonds[sa_slot] = Some(Bond {
            other_cell_id: id_b,
            rest_length: rest,
            stiffness,
            damping,
            age_ticks: 0,
        });
        cb.0.bonds[sb_slot] = Some(Bond {
            other_cell_id: id_a,
            rest_length: rest,
            stiffness,
            damping,
            age_ticks: 0,
        });
        ca.0.energy -= BOND_FORMATION_COST;
        cb.0.energy -= BOND_FORMATION_COST;
        contact_progress.0.remove(&(id_a, id_b));
    }
    diag.add_measurement(&DIAG_COLLISIONS, || t_total.elapsed().as_secs_f64() * 1000.0);
}

/// Sprint 66: snapshot row pro renderer collision/adhesion/bond pass.
struct SnapEntry {
    entity: Entity,
    cell_id: u64,
    position: [f32; 3],
    velocity: [f32; 3],
    radius: f32,
    adhesion_type: u8,
    bonds: [Option<Bond>; MAX_BONDS_PER_CELL],
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
