use bevy::asset::RenderAssetUsages;
use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::image::Image;
use bevy::input::mouse::{MouseScrollUnit, MouseWheel};
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use bioscape::{
    reject_food_for_richness,
    Cell, Food, Genome, Phenotype, SimClock, SmellField, WorldMap, ATTACK_THRESHOLD, BRAIN_HIDDEN,
    BRAIN_INPUTS, BRAIN_INPUTS_SENSORY, BRAIN_OUTPUTS, BRAIN_RECURRENT, CARRION_FOOD_COUNT,
    CELL_RADIUS, CYCLE_AMPLITUDE, CYCLE_GEN_PERIOD, DAMAGE_NORMALIZATION_GAIN,
    DENSITY_NORM_COUNT, DILUTION_K,
    DRAG_COEFFICIENT, EAT_RADIUS, FIXED_TIMESTEP_HZ, FOOD_SPAWN_RATE, FOOD_VALUE,
    GENERATIONS_PER_EPOCH, HAZARD_AMP, HAZARD_DRAIN_PER_SEC, HAZARD_FLOOR, HERD_RADIUS,
    INITIAL_CELLS,
    LEARNING_RATE, MATING_PHEROMONE_THRESHOLD, MATING_RADIUS, MAX_POPULATION, MAX_SPAWN_ATTEMPTS,
    MUTATION_CONFIG, PHEROMONE_BASELINE_EMIT, PHEROMONE_BRAIN_MOD,
    PHEROMONE_COST_PER_RATE, PHEROMONE_DECAY, PHEROMONE_DIFFUSION, PHEROMONE_GRID_RES,
    PHEROMONE_NORMALIZATION_GAIN, PHEROMONE_SAMPLE_EPSILON, PHYSICS_CONFIG, PREDATION_DRAIN_PER_TICK,
    PREDATION_GAIN_PER_TICK, REPRODUCE_THRESHOLD, SIZE_RATIO_THRESHOLD, SMELL_DECAY,
    SMELL_DIFFUSION, SMELL_GRID_RES, SMELL_NORMALIZATION_GAIN, SMELL_PER_FOOD, SMELL_SAMPLE_EPSILON,
    TICKS_PER_GENERATION, WORLD_MAP_BASE_RES, WORLD_MAP_FOOD_AMP, WORLD_MAP_FOOD_FLOOR,
    WORLD_MAP_RES, WORLD_MAP_SEED, WORLD_UNITS_PER_FOOD,
};
use rand::Rng;
use std::collections::{HashMap, HashSet};

// Renderer-only knobs. Sim parameters live in `bioscape` (lib.rs).
const FOOD_RADIUS: f32 = 2.5;
const DEATH_FADE_TICKS: u32 = 30;
const GRID_CELL_SIZE: f32 = 100.0;
const CAMERA_ZOOM_STEP: f32 = 0.1;
const CAMERA_ZOOM_MIN: f32 = 0.1;
const CAMERA_ZOOM_MAX: f32 = 10.0;
// Sprint 35: z-osa aktivovaná, mírný 3D layer.
const SIMULATION_HALF: [f32; 3] = [960.0, 540.0, 2.0];
// Sprint 36: aerial Camera3d view. Camera nahoru po +Z na výšce sufficient,
// aby celá xy plochá scéna se vešla do FoV při fov=π/3 (=60°).
// half_y / tan(fov/2) = 540 / tan(30°) ≈ 935. S marginem volíme 1500.
const CAMERA_HEIGHT_INITIAL: f32 = 1500.0;

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

type Bucket<P> = Vec<(Entity, [f32; 3], P)>;

struct SpatialGrid<P: Copy> {
    cell_size: f32,
    buckets: HashMap<(i32, i32, i32), Bucket<P>>,
}

impl<P: Copy> SpatialGrid<P> {
    fn new(cell_size: f32) -> Self {
        Self {
            cell_size,
            buckets: HashMap::new(),
        }
    }

    fn key_of(&self, pos: [f32; 3]) -> (i32, i32, i32) {
        (
            (pos[0] / self.cell_size).floor() as i32,
            (pos[1] / self.cell_size).floor() as i32,
            (pos[2] / self.cell_size).floor() as i32,
        )
    }

    fn rebuild<I: IntoIterator<Item = (Entity, [f32; 3], P)>>(&mut self, items: I) {
        // Preserve bucket Vec capacities across ticks — population and food
        // count are roughly stable, so reusing allocations beats clearing the
        // whole HashMap.
        for bucket in self.buckets.values_mut() {
            bucket.clear();
        }
        for (e, pos, payload) in items {
            let key = self.key_of(pos);
            self.buckets.entry(key).or_default().push((e, pos, payload));
        }
    }

    /// Sprint 32: 3D bucketing. Pro Sprint 32 (z=0 locked) je dz iterace
    /// degenerate single-bucket loop, takže overhead minimální. Po Sprintu 33
    /// (z motion) se naplní celá 3D mřížka.
    fn for_each_in_radius<F: FnMut(Entity, [f32; 3], P)>(
        &self,
        pos: [f32; 3],
        radius: f32,
        mut f: F,
    ) {
        let r_cells = (radius / self.cell_size).ceil() as i32;
        let (cx, cy, cz) = self.key_of(pos);
        for dx in -r_cells..=r_cells {
            for dy in -r_cells..=r_cells {
                for dz in -r_cells..=r_cells {
                    if let Some(bucket) = self.buckets.get(&(cx + dx, cy + dy, cz + dz)) {
                        for &(e, p, payload) in bucket {
                            f(e, p, payload);
                        }
                    }
                }
            }
        }
    }
}

#[derive(Resource)]
struct CellGrid(SpatialGrid<f32>);

impl Default for CellGrid {
    fn default() -> Self {
        Self(SpatialGrid::new(GRID_CELL_SIZE))
    }
}

#[derive(Resource)]
struct FoodGrid(SpatialGrid<()>);

impl Default for FoodGrid {
    fn default() -> Self {
        Self(SpatialGrid::new(GRID_CELL_SIZE))
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
struct LineageMaterials(HashMap<u64, Handle<StandardMaterial>>);

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
        ))
        .insert_resource(ClearColor(Color::srgb(0.05, 0.05, 0.08)))
        .init_resource::<LineageMaterials>()
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
        .add_systems(Startup, (setup, setup_stats_overlay, rebuild_cell_grid).chain())
        .add_systems(
            FixedUpdate,
            (
                advance_clock,
                update_food_density_cycle,
                rebuild_food_grid,
                update_smell_field,
                update_pheromone_field,
                cells_brain_act,
                emit_pheromones,
                apply_cell_morph,
                step_cells,
                apply_environmental_hazards,
                rebuild_cell_grid,
                resolve_cell_collisions,
                cell_predates_on_neighbor,
                cell_eats_food,
                spawn_food,
                cell_reproduces_on_threshold,
                cell_dies_on_zero_energy,
                tick_death_fade,
            )
                .chain(),
        )
        .add_systems(
            Update,
            (
                speed_input,
                camera_zoom,
                camera_pan,
                sync_transforms,
                log_clock_events,
                toggle_stats_overlay,
                toggle_world_map_overlay,
                update_stats_overlay,
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

    // Sprint 36: Camera3d nahoru po +Z, looking at origin. Bevy 0.18 Camera3d
    // má default perspective projection. `IsDefaultUiCamera` říká bevy_ui_render
    // ať použije tuto kameru pro UI overlay — bez toho by Bevy spawnul vlastní
    // UI kameru bez render graph konfigurace (warning "Entity X has Camera but
    // no render graph").
    commands.spawn((
        Camera3d::default(),
        IsDefaultUiCamera,
        Transform::from_xyz(0.0, 0.0, CAMERA_HEIGHT_INITIAL).looking_at(Vec3::ZERO, Vec3::Y),
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

    // Sprint 32: WorldMap + SmellField stále 2D — projekce xy z 3D extentu.
    let half_xy = [half[0], half[1]];
    let world_map = WorldMap::new(WORLD_MAP_RES, WORLD_MAP_BASE_RES, half_xy, WORLD_MAP_SEED);

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
    let food_mesh_handle = meshes.add(Sphere::new(FOOD_RADIUS).mesh().ico(1).unwrap());
    let food_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.95, 0.95, 0.85),
        emissive: LinearRgba::new(0.4, 0.4, 0.3, 1.0),
        ..default()
    });

    let mut rng = rand::rng();
    for i in 0..INITIAL_CELLS {
        let cell = Cell::random(&mut rng, half, i as u64, 0);
        let mat = lineage_material(&mut lineage_materials, &mut materials, cell.lineage_id);
        commands.spawn((
            CellEntity(cell),
            Mesh3d(cell_mesh_handle.clone()),
            MeshMaterial3d(mat),
            Transform::from_xyz(cell.position[0], cell.position[1], cell.position[2])
                .with_rotation(cell_rotation(cell.heading, cell.pitch))
                .with_scale(cell_scale(&cell.phenotype)),
        ));
    }
    let initial_food = food_target(&extent, 1.0);
    for _ in 0..initial_food {
        let mut food = Food::random(&mut rng, half);
        for _ in 0..MAX_SPAWN_ATTEMPTS {
            let richness = world_map.sample([food.position[0], food.position[1]]);
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
    commands.insert_resource(SmellResource(SmellField::new(SMELL_GRID_RES, half_xy)));
    commands.insert_resource(PheromoneResource(SmellField::new(
        PHEROMONE_GRID_RES,
        half_xy,
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
    let n = map.resolution;
    let mut data = Vec::with_capacity(n * n * 4);
    for &v in map.field() {
        let g = (v.clamp(0.0, 1.0) * 255.0) as u8;
        // Bohaté oblasti = teplá zelená, chudé = tmavé. Alpha jednotná, finální
        // alpha řízená Sprite.color.
        data.push((g / 4).max(20));
        data.push(g);
        data.push((g / 3).max(20));
        data.push(255);
    }
    Image::new(
        Extent3d {
            width: n as u32,
            height: n as u32,
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
    let area = (2.0 * extent.half_x) * (2.0 * extent.half_y);
    ((area / WORLD_UNITS_PER_FOOD) * factor.max(0.0)) as usize
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

fn apply_environmental_hazards(
    time: Res<Time>,
    world_map: Res<WorldMapResource>,
    mut cells: Query<&mut CellEntity, Without<Dying>>,
) {
    let dt = time.delta_secs();
    for mut cell in &mut cells {
        let noise = world_map
            .0
            .sample([cell.0.position[0], cell.0.position[1]]);
        let drain = hazard_drain(noise) * dt;
        cell.0.energy -= drain;
        cell.0.damage_accum += drain;
    }
}

fn update_smell_field(
    time: Res<Time>,
    foods: Query<&FoodEntity>,
    mut smell: ResMut<SmellResource>,
) {
    let dt = time.delta_secs();
    for food in &foods {
        smell
            .0
            .add_source([food.0.position[0], food.0.position[1]], SMELL_PER_FOOD * dt);
    }
    smell.0.step(SMELL_DIFFUSION, SMELL_DECAY, dt);
}

fn update_pheromone_field(time: Res<Time>, mut pheromone: ResMut<PheromoneResource>) {
    // Diffuse + decay BEFORE this tick's emissions (in emit_pheromones, which
    // runs after brain_act). Stejně jako headless — brainy detekují gradient
    // ze stavu pole na konci minulého ticku, žádný self-feedback.
    let dt = time.delta_secs();
    pheromone.0.step(PHEROMONE_DIFFUSION, PHEROMONE_DECAY, dt);
}

fn emit_pheromones(
    time: Res<Time>,
    mut pheromone: ResMut<PheromoneResource>,
    mut cells: Query<&mut CellEntity, Without<Dying>>,
) {
    let dt = time.delta_secs();
    for mut cell in &mut cells {
        let mod_strength = cell.0.last_outputs[2].max(0.0);
        let brain_emit = PHEROMONE_BRAIN_MOD * mod_strength;
        let rate = PHEROMONE_BASELINE_EMIT + brain_emit;
        pheromone
            .0
            .add_source([cell.0.position[0], cell.0.position[1]], rate * dt);
        cell.0.energy -= PHEROMONE_COST_PER_RATE * brain_emit * dt;
    }
}

fn cells_brain_act(
    time: Res<Time>,
    cell_grid: Res<CellGrid>,
    food_grid: Res<FoodGrid>,
    smell: Res<SmellResource>,
    pheromone: Res<PheromoneResource>,
    mut cells: Query<(Entity, &mut CellEntity), Without<Dying>>,
) {
    let dt = time.delta_secs();

    for (entity, mut cell) in &mut cells {
        let pos = cell.0.position;
        let vision_r = cell.0.genome.vision_radius;
        let vr2 = vision_r * vision_r;

        let mut nearest_food: Option<[f32; 3]> = None;
        let mut best_food_d2 = f32::MAX;
        food_grid.0.for_each_in_radius(pos, vision_r, |_, fp, _| {
            let dx = fp[0] - pos[0];
            let dy = fp[1] - pos[1];
            let dz = fp[2] - pos[2];
            let d2 = dx * dx + dy * dy + dz * dz;
            if d2 <= vr2 && d2 < best_food_d2 {
                best_food_d2 = d2;
                nearest_food = Some(fp);
            }
        });

        let mut nearest_cell: Option<([f32; 3], f32)> = None;
        let mut best_cell_d2 = f32::MAX;
        let mut neighbors_in_vision: u32 = 0;
        cell_grid
            .0
            .for_each_in_radius(pos, vision_r, |other, other_pos, other_radius| {
                if other == entity {
                    return;
                }
                let dx = other_pos[0] - pos[0];
                let dy = other_pos[1] - pos[1];
                let dz = other_pos[2] - pos[2];
                let d2 = dx * dx + dy * dy + dz * dz;
                if d2 <= vr2 {
                    neighbors_in_vision += 1;
                    if d2 < best_cell_d2 {
                        best_cell_d2 = d2;
                        nearest_cell = Some((other_pos, other_radius));
                    }
                }
            });

        let max_speed = cell.0.genome.max_speed;
        let my_radius = cell.0.phenotype.effective_radius().max(0.01);
        // Sprint 32: 2D hypot pro CSV identity (vz=0). Sprint 33+ přejde na 3D.
        let speed_norm =
            (cell.0.velocity[0].hypot(cell.0.velocity[1]) / max_speed).clamp(0.0, 1.0);
        let energy_norm = (cell.0.energy / REPRODUCE_THRESHOLD).clamp(0.0, 1.5);

        let mut inputs = [0.0_f32; BRAIN_INPUTS];
        if let Some(target) = nearest_food {
            inputs[0] = (target[0] - pos[0]) / vision_r;
            inputs[1] = (target[1] - pos[1]) / vision_r;
            inputs[15] = (target[2] - pos[2]) / vision_r;
        }
        if let Some((target, other_radius)) = nearest_cell {
            inputs[2] = (target[0] - pos[0]) / vision_r;
            inputs[3] = (target[1] - pos[1]) / vision_r;
            inputs[6] = (other_radius - my_radius) / my_radius;
            inputs[16] = (target[2] - pos[2]) / vision_r;
        }
        inputs[4] = energy_norm;
        inputs[5] = speed_norm;
        // Sprint 32: SmellField + Pheromone stále 2D — projekce xy.
        let pos_xy = [pos[0], pos[1]];
        let grad = smell.0.gradient_at(pos_xy, SMELL_SAMPLE_EPSILON);
        inputs[7] = (grad[0] * SMELL_NORMALIZATION_GAIN).tanh();
        inputs[8] = (grad[1] * SMELL_NORMALIZATION_GAIN).tanh();
        // Sprint 33: heading_x, _y nově xy projekce 3D forward; heading_z = sin(pitch).
        let fwd = bioscape::forward_vector(cell.0.heading, cell.0.pitch);
        inputs[9] = fwd[0];
        inputs[10] = fwd[1];
        inputs[18] = fwd[2];
        let pgrad = pheromone.0.gradient_at(pos_xy, PHEROMONE_SAMPLE_EPSILON);
        inputs[11] = (pgrad[0] * PHEROMONE_NORMALIZATION_GAIN).tanh();
        inputs[12] = (pgrad[1] * PHEROMONE_NORMALIZATION_GAIN).tanh();
        // Sprint 29 quorum sensing: počet viditelných sousedů normovaný přes
        // DENSITY_NORM_COUNT, saturován tanhem do [0, 1).
        inputs[13] = (neighbors_in_vision as f32 / DENSITY_NORM_COUNT).tanh();
        // Sprint 30: damage signál z minulého ticku (predace + hazard).
        // Voluntární cost se nezapisuje. Reset na 0 hned po čtení = 1-tick
        // delay bez self-feedback.
        inputs[14] = (cell.0.damage_accum * DAMAGE_NORMALIZATION_GAIN).tanh();
        cell.0.damage_accum = 0.0;
        // Sprint 28: Elman recurrent feedback — předchozí tick `last_hidden`
        // jako input. Při t=0 jsou všechno zero (Cell::random / reproduce
        // init), takže první tick je identický s feed-forward; paměť nabíhá
        // od ticku 1.
        inputs[BRAIN_INPUTS_SENSORY..BRAIN_INPUTS_SENSORY + BRAIN_RECURRENT]
            .copy_from_slice(&cell.0.last_hidden[..BRAIN_RECURRENT]);

        let (hidden, outputs) = cell.0.genome.brain.forward_with_state(&inputs);
        cell.0.last_inputs = inputs;
        cell.0.last_hidden = hidden;
        cell.0.last_outputs = outputs;
        let turn_signal = outputs[0];
        let thrust_norm = (outputs[1] + 1.0) * 0.5;
        // Sprint 35: pitch control aktivován.
        let pitch_signal = outputs[7];

        // Sprint 26: torque/thrust scaling teď používá effective_radius místo
        // body_size. Pro length=width=s redukuje na původní semantiku.
        let body_proxy = my_radius;
        let turn_rate = cell.0.genome.turn_rate;
        let ang_acc = turn_signal * turn_rate / body_proxy;
        cell.0.angular_velocity += ang_acc * dt;
        let pitch_acc = pitch_signal * turn_rate / body_proxy;
        cell.0.pitch_velocity += pitch_acc * dt;

        let a_max = DRAG_COEFFICIENT * max_speed * max_speed / body_proxy;
        let a = thrust_norm * a_max;
        // Sprint 33: pitch=0 → fwd[2]=0; 3D-ready math přes forward_vector helper.
        let fwd = bioscape::forward_vector(cell.0.heading, cell.0.pitch);
        cell.0.velocity[0] += a * fwd[0] * dt;
        cell.0.velocity[1] += a * fwd[1] * dt;
        cell.0.velocity[2] += a * fwd[2] * dt;
    }
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
    let broad_r = EAT_RADIUS * BROAD_PHASE_SIZE_BUDGET;

    'spawn: for _ in 0..to_spawn {
        for _ in 0..MAX_SPAWN_ATTEMPTS {
            let candidate = Food::random(&mut rng, half);
            // Sprint 31: rejection sampling proti uniform — bias k rich zonám.
            // Spotřebovává retry budget jako cell-exclusion check níž.
            let richness = world_map
                .0
                .sample([candidate.position[0], candidate.position[1]]);
            if reject_food_for_richness(&mut rng, richness) {
                continue;
            }
            let mut blocked = false;
            cell_grid
                .0
                .for_each_in_radius(candidate.position, broad_r, |_, cell_pos, radius| {
                    if blocked {
                        return;
                    }
                    let exclusion = EAT_RADIUS * radius;
                    let dx = candidate.position[0] - cell_pos[0];
                    let dy = candidate.position[1] - cell_pos[1];
                    let dz = candidate.position[2] - cell_pos[2];
                    if dx * dx + dy * dy + dz * dz < exclusion * exclusion {
                        blocked = true;
                    }
                });
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
    mut cells: Query<&mut CellEntity, Without<Dying>>,
    food_grid: Res<FoodGrid>,
    world_map: Res<WorldMapResource>,
    mut commands: Commands,
) {
    let mut eaten: HashSet<Entity> = HashSet::new();

    for mut cell in &mut cells {
        let pos = cell.0.position;
        let eat_r = EAT_RADIUS * cell.0.phenotype.effective_radius();
        let r2 = eat_r * eat_r;
        let mut to_eat: Option<(Entity, [f32; 3])> = None;
        food_grid.0.for_each_in_radius(pos, eat_r, |food_e, food_pos, _| {
            if to_eat.is_some() || eaten.contains(&food_e) {
                return;
            }
            let dx = pos[0] - food_pos[0];
            let dy = pos[1] - food_pos[1];
            let dz = pos[2] - food_pos[2];
            if dx * dx + dy * dy + dz * dz <= r2 {
                to_eat = Some((food_e, food_pos));
            }
        });
        if let Some((food_e, food_pos)) = to_eat {
            cell.0.energy +=
                FOOD_VALUE * food_multiplier(world_map.0.sample([food_pos[0], food_pos[1]]));
            eaten.insert(food_e);
            commands.entity(food_e).despawn();
            // Reward-modulated Hebbian update — reinforce the recent decision
            // pathway (last forward pass) on positive outcome.
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

fn sync_transforms(mut cells: Query<(&CellEntity, &mut Transform), Without<Dying>>) {
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

/// Sprint 36: Camera3d pan v xy-plochy přes WASD/arrow klávesy. Mouse motion
/// resp. middle-drag se v 3D pohledu typicky hodí na orbit (rotace), takže
/// pan necháme klávesnicovou.
fn camera_pan(
    keys: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    camera: Single<&mut Transform, With<Camera3d>>,
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
    let mut transform = camera.into_inner();
    // Pan rychlost ∝ camera Z výšce (víc zoomout = rychlejší pan).
    let speed = transform.translation.z.abs() * 0.5;
    transform.translation.x += delta.x * speed * time.delta_secs();
    transform.translation.y += delta.y * speed * time.delta_secs();
}

/// Sprint 36: zoom přes mouse wheel — adjustuje camera Z distance. Bližší
/// kamera = větší cells na obrazovce. Clamp brání tomu aby kamera prošla
/// scénou nebo se ztratila daleko.
fn camera_zoom(
    mut wheel: MessageReader<MouseWheel>,
    camera: Single<&mut Transform, With<Camera3d>>,
) {
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
    let mut transform = camera.into_inner();
    let factor = (-scroll * CAMERA_ZOOM_STEP).exp();
    let new_z = (transform.translation.z * factor)
        .clamp(CAMERA_HEIGHT_INITIAL * CAMERA_ZOOM_MIN, CAMERA_HEIGHT_INITIAL * CAMERA_ZOOM_MAX);
    transform.translation.z = new_z;
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
    let mut lineages: HashSet<u64> = HashSet::new();
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
    mut commands: Commands,
) {
    let current_pop = cells.iter().count();
    if current_pop >= MAX_POPULATION {
        return;
    }
    let budget = MAX_POPULATION - current_pop;

    // Snapshot fertile cells (immutable iter on a mut query is fine).
    // Sprint 25: cells musí AKTIVNĚ emitovat pheromone (output[2] >
    // threshold) aby byly fertile.
    let fertile: Vec<(Entity, [f32; 3])> = cells
        .iter()
        .filter(|(_, c)| {
            c.0.energy >= REPRODUCE_THRESHOLD
                && c.0.last_outputs[2] > MATING_PHEROMONE_THRESHOLD
        })
        .map(|(e, c)| (e, c.0.position))
        .collect();

    // Greedy O(N²) pairing on fertile pool only — typically a few dozen cells,
    // so the cost is negligible compared to grid-scale work elsewhere.
    let mut paired: HashSet<Entity> = HashSet::new();
    let mut matings: Vec<(Entity, Entity)> = Vec::new();
    let mating_r2 = MATING_RADIUS * MATING_RADIUS;
    for i in 0..fertile.len() {
        if matings.len() >= budget {
            break;
        }
        let (a, pos_a) = fertile[i];
        if paired.contains(&a) {
            continue;
        }
        let mut best: Option<(Entity, f32)> = None;
        for (j, &(b, pos_b)) in fertile.iter().enumerate() {
            if i == j {
                continue;
            }
            if paired.contains(&b) {
                continue;
            }
            let dx = pos_a[0] - pos_b[0];
            let dy = pos_a[1] - pos_b[1];
            let dz = pos_a[2] - pos_b[2];
            let d2 = dx * dx + dy * dy + dz * dz;
            if d2 <= mating_r2 && best.is_none_or(|(_, bd2)| d2 < bd2) {
                best = Some((b, d2));
            }
        }
        if let Some((b, _)) = best {
            paired.insert(a);
            paired.insert(b);
            matings.push((a, b));
        }
    }

    let mut rng = rand::rng();
    let mut to_spawn: Vec<Cell> = Vec::new();
    for (a, b) in matings {
        let Ok([(_, mut cell_a), (_, mut cell_b)]) = cells.get_many_mut([a, b]) else {
            continue;
        };
        let energy_a = cell_a.0.energy * 0.5;
        let energy_b = cell_b.0.energy * 0.5;
        cell_a.0.energy *= 0.5;
        cell_b.0.energy *= 0.5;

        let child_genome = Genome::crossover(&cell_a.0.genome, &cell_b.0.genome, &mut rng)
            .mutate(&mut rng, &MUTATION_CONFIG);

        let direction = rng.random_range(0.0..core::f32::consts::TAU);
        let mid_pos = [
            (cell_a.0.position[0] + cell_b.0.position[0]) * 0.5,
            (cell_a.0.position[1] + cell_b.0.position[1]) * 0.5,
            (cell_a.0.position[2] + cell_b.0.position[2]) * 0.5,
        ];
        let child_phenotype = Phenotype::from_genome(&child_genome);
        to_spawn.push(Cell {
            position: mid_pos,
            velocity: [
                direction.cos() * child_genome.max_speed,
                direction.sin() * child_genome.max_speed,
                0.0,
            ],
            angular_velocity: 0.0,
            pitch_velocity: 0.0,
            energy: energy_a + energy_b,
            heading: direction,
            pitch: 0.0,
            lineage_id: cell_a.0.lineage_id,
            lineage_birth_gen: cell_a.0.lineage_birth_gen,
            last_inputs: [0.0; BRAIN_INPUTS],
            last_hidden: [0.0; BRAIN_HIDDEN],
            last_outputs: [0.0; BRAIN_OUTPUTS],
            damage_accum: 0.0,
            phenotype: child_phenotype,
            genome: child_genome,
        });
    }

    let mesh = cell_mesh.0.clone();
    for cell in to_spawn {
        let mat = lineage_material(&mut lineage_materials, &mut materials, cell.lineage_id);
        commands.spawn((
            CellEntity(cell),
            Mesh3d(mesh.clone()),
            MeshMaterial3d(mat),
            Transform::from_xyz(cell.position[0], cell.position[1], cell.position[2])
                .with_rotation(cell_rotation(cell.heading, cell.pitch))
                .with_scale(cell_scale(&cell.phenotype)),
        ));
    }
}

fn cell_dies_on_zero_energy(
    cells: Query<(Entity, &CellEntity), Without<Dying>>,
    extent: Res<WorldExtent>,
    food_mesh: Res<FoodMesh>,
    food_material: Res<FoodMaterial>,
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
                    FoodEntity(Food { position: pos }),
                    Mesh3d(food_mesh.0.clone()),
                    MeshMaterial3d(food_material.0.clone()),
                    Transform::from_xyz(pos[0], pos[1], pos[2]),
                ));
            }
        }
    }
}

fn rebuild_cell_grid(
    mut grid: ResMut<CellGrid>,
    cells: Query<(Entity, &CellEntity), Without<Dying>>,
) {
    grid.0.rebuild(
        cells
            .iter()
            .map(|(e, c)| (e, c.0.position, c.0.phenotype.effective_radius())),
    );
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
) {
    let mut energy_changes: HashMap<Entity, f32> = HashMap::new();
    // Sprint 30: nedobrovolný drain do brain damage signálu (input[14]).
    // Voluntární cost (movement, morph, attack) sem nepatří, jen predation.
    let mut damage_changes: HashMap<Entity, f32> = HashMap::new();

    // Sprint 29 selfish-herd: pre-compute herd count per cell (počet sousedů
    // ve `HERD_RADIUS`). V predaci níže se gain násobí 1/(1 + K × herd_count_prey)
    // — kořist obklopena hejnem dává predátorovi menší odměnu. `for_each_in_radius`
    // je broad-phase (vrací bucket-grid kandidáty), narrow-phase distance check
    // musí být explicitně, jinak se počítají i cells mimo HERD_RADIUS a dilution
    // je silnější než v headless harness.
    let herd_r2 = HERD_RADIUS * HERD_RADIUS;
    let mut herd_counts: HashMap<Entity, u32> = HashMap::new();
    for (entity, cell) in &cells {
        let pos = cell.0.position;
        let mut count: u32 = 0;
        grid.0
            .for_each_in_radius(pos, HERD_RADIUS, |other, other_pos, _| {
                if other == entity {
                    return;
                }
                let dx = other_pos[0] - pos[0];
                let dy = other_pos[1] - pos[1];
                if dx * dx + dy * dy < herd_r2 {
                    count += 1;
                }
            });
        herd_counts.insert(entity, count);
    }

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
            .for_each_in_radius(pos_a, broad_r, |entity_b, pos_b, radius_b| {
                if entity_b == entity_a {
                    return;
                }
                if radius_a < SIZE_RATIO_THRESHOLD * radius_b {
                    return;
                }
                let pair_r = CELL_RADIUS * (radius_a + radius_b);
                let pair_r2 = pair_r * pair_r;
                let dx = pos_a[0] - pos_b[0];
                let dy = pos_a[1] - pos_b[1];
                let d2 = dx * dx + dy * dy;
                if d2 >= pair_r2 {
                    return;
                }
                let bonus = cell_a.0.spike_bonus_against(pos_b);
                let gain_raw = PREDATION_GAIN_PER_TICK + bonus;
                // Sprint 29 dilution: gain × 1/(1 + K × n_neighbors_prey).
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
}

fn resolve_cell_collisions(
    grid: Res<CellGrid>,
    mut cells: Query<(Entity, &mut CellEntity), Without<Dying>>,
) {
    let mut deltas: Vec<(Entity, [f32; 2])> = Vec::new();

    for (entity_a, cell_a) in &cells {
        let pos_a = cell_a.0.position;
        let radius_a = cell_a.0.phenotype.effective_radius();
        let broad_r = CELL_RADIUS * (radius_a + BROAD_PHASE_SIZE_BUDGET);
        let mut delta = [0.0_f32, 0.0_f32];
        grid.0
            .for_each_in_radius(pos_a, broad_r, |entity_b, pos_b, radius_b| {
                if entity_b == entity_a {
                    return;
                }
                let pair_r = CELL_RADIUS * (radius_a + radius_b);
                let pair_r2 = pair_r * pair_r;
                let dx = pos_a[0] - pos_b[0];
                let dy = pos_a[1] - pos_b[1];
                let d2 = dx * dx + dy * dy;
                if d2 < pair_r2 && d2 > 0.0 {
                    let d = d2.sqrt();
                    let overlap = pair_r - d;
                    delta[0] += (dx / d) * overlap * 0.5;
                    delta[1] += (dy / d) * overlap * 0.5;
                }
            });
        if delta != [0.0, 0.0] {
            deltas.push((entity_a, delta));
        }
    }

    for (entity, delta) in deltas {
        if let Ok((_, mut cell)) = cells.get_mut(entity) {
            cell.0.position[0] += delta[0];
            cell.0.position[1] += delta[1];
        }
    }
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
