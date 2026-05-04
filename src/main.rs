mod cell_material;

use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::input::mouse::{MouseScrollUnit, MouseWheel};
use bevy::mesh::MeshTag;
use bevy::prelude::*;
use bevy::asset::RenderAssetUsages;
use bevy::image::Image;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use bioscape::{
    Cell, Food, Genome, SimClock, SmellField, WorldMap, BRAIN_HIDDEN, BRAIN_INPUTS, BRAIN_OUTPUTS,
    CARRION_FOOD_COUNT, CELL_RADIUS, CYCLE_AMPLITUDE, CYCLE_GEN_PERIOD, DRAG_COEFFICIENT,
    EAT_RADIUS, FIXED_TIMESTEP_HZ, FOOD_SPAWN_RATE, FOOD_VALUE, GENERATIONS_PER_EPOCH,
    INITIAL_CELLS, LEARNING_RATE, MATING_RADIUS, MAX_POPULATION, MAX_SPAWN_ATTEMPTS,
    MUTATION_CONFIG, PHYSICS_CONFIG, PREDATION_DRAIN_PER_TICK, PREDATION_GAIN_PER_TICK,
    REPRODUCE_THRESHOLD, SIZE_RATIO_THRESHOLD, SMELL_DECAY, SMELL_DIFFUSION, SMELL_GRID_RES,
    SMELL_NORMALIZATION_GAIN, SMELL_PER_FOOD, SMELL_SAMPLE_EPSILON, TICKS_PER_GENERATION,
    WORLD_MAP_BASE_RES, WORLD_MAP_FOOD_AMP, WORLD_MAP_FOOD_FLOOR, WORLD_MAP_RES, WORLD_MAP_SEED,
    WORLD_UNITS_PER_FOOD,
};
use cell_material::{pack_cell_tag, CellMaterial, CellMaterialPlugin};
use rand::Rng;
use std::collections::{HashMap, HashSet};

// Renderer-only knobs. Sim parameters live in `bioscape` (lib.rs).
const FOOD_RADIUS: f32 = 2.5;
const DEATH_FADE_TICKS: u32 = 30;
const GRID_CELL_SIZE: f32 = 100.0;
const CAMERA_ZOOM_STEP: f32 = 0.1;
const CAMERA_ZOOM_MIN: f32 = 0.1;
const CAMERA_ZOOM_MAX: f32 = 10.0;
// Fixní simulační svět (matches headless WORLD_HALF). Window je jen viewport —
// WorldMap, cell bounds a vše sim mechanika pracují v těchto rozměrech, nezávisle
// na velikosti okna. Bez tohoto by overlay neodpovídal pozicím buněk po
// maximalizaci a seed by dával různé mapy na různých monitorech.
const SIMULATION_HALF: [f32; 2] = [960.0, 540.0];
const WORLD_MAP_OVERLAY_ALPHA: f32 = 0.3;
const WORLD_MAP_OVERLAY_Z: f32 = -10.0;

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
}

impl WorldExtent {
    fn as_array(self) -> [f32; 2] {
        [self.half_x, self.half_y]
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

type Bucket<P> = Vec<(Entity, [f32; 2], P)>;

struct SpatialGrid<P: Copy> {
    cell_size: f32,
    buckets: HashMap<(i32, i32), Bucket<P>>,
}

impl<P: Copy> SpatialGrid<P> {
    fn new(cell_size: f32) -> Self {
        Self {
            cell_size,
            buckets: HashMap::new(),
        }
    }

    fn key_of(&self, pos: [f32; 2]) -> (i32, i32) {
        (
            (pos[0] / self.cell_size).floor() as i32,
            (pos[1] / self.cell_size).floor() as i32,
        )
    }

    fn rebuild<I: IntoIterator<Item = (Entity, [f32; 2], P)>>(&mut self, items: I) {
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

    fn for_each_in_radius<F: FnMut(Entity, [f32; 2], P)>(
        &self,
        pos: [f32; 2],
        radius: f32,
        mut f: F,
    ) {
        let r_cells = (radius / self.cell_size).ceil() as i32;
        let (cx, cy) = self.key_of(pos);
        for dx in -r_cells..=r_cells {
            for dy in -r_cells..=r_cells {
                if let Some(bucket) = self.buckets.get(&(cx + dx, cy + dy)) {
                    for &(e, p, payload) in bucket {
                        f(e, p, payload);
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
struct FoodMaterial(Handle<ColorMaterial>);

#[derive(Resource)]
struct CellMaterialHandle(Handle<CellMaterial>);

#[derive(Resource)]
struct SmellResource(SmellField);

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
            CellMaterialPlugin,
        ))
        .insert_resource(ClearColor(Color::srgb(0.05, 0.05, 0.08)))
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
                cells_brain_act,
                step_cells,
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
    mut color_materials: ResMut<Assets<ColorMaterial>>,
    mut cell_materials: ResMut<Assets<CellMaterial>>,
    mut images: ResMut<Assets<Image>>,
    mut window: Single<&mut Window>,
) {
    window.set_maximized(true);
    let half = SIMULATION_HALF;
    let extent = WorldExtent {
        half_x: half[0],
        half_y: half[1],
    };
    commands.insert_resource(extent);

    commands.spawn(Camera2d);

    let world_map = WorldMap::new(WORLD_MAP_RES, WORLD_MAP_BASE_RES, half, WORLD_MAP_SEED);
    let overlay_image = images.add(world_map_image(&world_map));
    commands.spawn((
        Sprite {
            image: overlay_image,
            custom_size: Some(Vec2::new(2.0 * half[0], 2.0 * half[1])),
            color: Color::srgba(1.0, 1.0, 1.0, WORLD_MAP_OVERLAY_ALPHA),
            ..default()
        },
        Transform::from_xyz(0.0, 0.0, WORLD_MAP_OVERLAY_Z),
        WorldMapOverlay,
    ));

    let cell_mesh = meshes.add(teardrop_mesh(CELL_RADIUS));
    let food_mesh = meshes.add(Circle::new(FOOD_RADIUS));
    let food_material = color_materials.add(Color::srgb(0.95, 0.95, 0.85));
    let cell_material = cell_materials.add(CellMaterial::default());

    let mut rng = rand::rng();
    for i in 0..INITIAL_CELLS {
        let cell = Cell::random(&mut rng, half, i as u64, 0);
        commands.spawn((
            CellEntity(cell),
            Mesh2d(cell_mesh.clone()),
            MeshMaterial2d(cell_material.clone()),
            MeshTag(pack_cell_tag(lineage_hue(cell.lineage_id), 1.0)),
            Transform::from_xyz(cell.position[0], cell.position[1], 0.0)
                .with_rotation(Quat::from_rotation_z(cell.heading))
                .with_scale(Vec3::splat(cell.genome.body_size)),
        ));
    }
    let initial_food = food_target(&extent, 1.0);
    for _ in 0..initial_food {
        let food = Food::random(&mut rng, half);
        commands.spawn((
            FoodEntity(food),
            Mesh2d(food_mesh.clone()),
            MeshMaterial2d(food_material.clone()),
            Transform::from_xyz(food.position[0], food.position[1], -1.0),
        ));
    }

    commands.insert_resource(CellMesh(cell_mesh));
    commands.insert_resource(FoodMesh(food_mesh));
    commands.insert_resource(FoodMaterial(food_material));
    commands.insert_resource(CellMaterialHandle(cell_material));
    commands.insert_resource(SmellResource(SmellField::new(SMELL_GRID_RES, half)));
    commands.insert_resource(WorldMapResource(world_map));
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

fn teardrop_mesh(radius: f32) -> Mesh {
    use bevy::asset::RenderAssetUsages;
    use bevy::mesh::{Indices, PrimitiveTopology};

    // Parametric teardrop, tip in +x: x = r·cos t, y = r·sin t · sin(t/2).
    // Pointy at t=0, round bulb at t=π. Convex, so triangle fan from origin works.
    let segments = 24;
    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(segments + 1);
    let mut normals: Vec<[f32; 3]> = Vec::with_capacity(segments + 1);
    let mut uvs: Vec<[f32; 2]> = Vec::with_capacity(segments + 1);

    positions.push([0.0, 0.0, 0.0]);
    normals.push([0.0, 0.0, 1.0]);
    uvs.push([0.5, 0.5]);

    for i in 0..segments {
        let t = (i as f32 / segments as f32) * std::f32::consts::TAU;
        let x = radius * t.cos();
        let y = radius * t.sin() * (t * 0.5).sin();
        positions.push([x, y, 0.0]);
        normals.push([0.0, 0.0, 1.0]);
        uvs.push([x / (2.0 * radius) + 0.5, y / (2.0 * radius) + 0.5]);
    }

    let mut indices: Vec<u32> = Vec::with_capacity(segments * 3);
    for i in 0..segments {
        let next = (i + 1) % segments;
        indices.push(0);
        indices.push((i + 1) as u32);
        indices.push((next + 1) as u32);
    }

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_indices(Indices::U32(indices));
    mesh
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

fn update_smell_field(
    time: Res<Time>,
    foods: Query<&FoodEntity>,
    mut smell: ResMut<SmellResource>,
) {
    let dt = time.delta_secs();
    for food in &foods {
        smell.0.add_source(food.0.position, SMELL_PER_FOOD * dt);
    }
    smell.0.step(SMELL_DIFFUSION, SMELL_DECAY, dt);
}

fn cells_brain_act(
    time: Res<Time>,
    cell_grid: Res<CellGrid>,
    food_grid: Res<FoodGrid>,
    smell: Res<SmellResource>,
    mut cells: Query<(Entity, &mut CellEntity), Without<Dying>>,
) {
    let dt = time.delta_secs();

    for (entity, mut cell) in &mut cells {
        let pos = cell.0.position;
        let vision_r = cell.0.genome.vision_radius;
        let vr2 = vision_r * vision_r;

        let mut nearest_food: Option<[f32; 2]> = None;
        let mut best_food_d2 = f32::MAX;
        food_grid.0.for_each_in_radius(pos, vision_r, |_, fp, _| {
            let dx = fp[0] - pos[0];
            let dy = fp[1] - pos[1];
            let d2 = dx * dx + dy * dy;
            if d2 <= vr2 && d2 < best_food_d2 {
                best_food_d2 = d2;
                nearest_food = Some(fp);
            }
        });

        let mut nearest_cell: Option<([f32; 2], f32)> = None;
        let mut best_cell_d2 = f32::MAX;
        cell_grid
            .0
            .for_each_in_radius(pos, vision_r, |other, other_pos, other_size| {
                if other == entity {
                    return;
                }
                let dx = other_pos[0] - pos[0];
                let dy = other_pos[1] - pos[1];
                let d2 = dx * dx + dy * dy;
                if d2 <= vr2 && d2 < best_cell_d2 {
                    best_cell_d2 = d2;
                    nearest_cell = Some((other_pos, other_size));
                }
            });

        let max_speed = cell.0.genome.max_speed;
        let my_size = cell.0.genome.body_size.max(0.01);
        let speed_norm =
            (cell.0.velocity[0].hypot(cell.0.velocity[1]) / max_speed).clamp(0.0, 1.0);
        let energy_norm = (cell.0.energy / REPRODUCE_THRESHOLD).clamp(0.0, 1.5);

        let mut inputs = [0.0_f32; BRAIN_INPUTS];
        if let Some(target) = nearest_food {
            inputs[0] = (target[0] - pos[0]) / vision_r;
            inputs[1] = (target[1] - pos[1]) / vision_r;
        }
        if let Some((target, other_size)) = nearest_cell {
            inputs[2] = (target[0] - pos[0]) / vision_r;
            inputs[3] = (target[1] - pos[1]) / vision_r;
            inputs[6] = (other_size - my_size) / my_size;
        }
        inputs[4] = energy_norm;
        inputs[5] = speed_norm;
        let grad = smell.0.gradient_at(pos, SMELL_SAMPLE_EPSILON);
        inputs[7] = (grad[0] * SMELL_NORMALIZATION_GAIN).tanh();
        inputs[8] = (grad[1] * SMELL_NORMALIZATION_GAIN).tanh();

        let (hidden, outputs) = cell.0.genome.brain.forward_with_state(&inputs);
        cell.0.last_inputs = inputs;
        cell.0.last_hidden = hidden;
        cell.0.last_outputs = outputs;
        let turn_signal = outputs[0];
        let thrust_norm = (outputs[1] + 1.0) * 0.5;

        // Force-based dynamics: brain outputs torque + thrust, integrated into
        // angular_velocity / velocity. Heading and position update in step
        // (drag applies there too).
        let body_size = cell.0.genome.body_size.max(0.01);
        let turn_rate = cell.0.genome.turn_rate;
        let ang_acc = turn_signal * turn_rate / body_size;
        cell.0.angular_velocity += ang_acc * dt;

        // a_max = DRAG · max_speed² / body_size → terminal v at full thrust
        // = max_speed / sqrt(body_size). Big cells are sluggish AND have a
        // lower top speed.
        let a_max = DRAG_COEFFICIENT * max_speed * max_speed / body_size;
        let a = thrust_norm * a_max;
        let heading = cell.0.heading;
        cell.0.velocity[0] += a * heading.cos() * dt;
        cell.0.velocity[1] += a * heading.sin() * dt;
    }
}

fn spawn_food(
    foods: Query<(), With<FoodEntity>>,
    extent: Res<WorldExtent>,
    factor: Res<FoodDensityFactor>,
    cell_grid: Res<CellGrid>,
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
            let mut blocked = false;
            cell_grid
                .0
                .for_each_in_radius(candidate.position, broad_r, |_, cell_pos, size| {
                    if blocked {
                        return;
                    }
                    let exclusion = EAT_RADIUS * size;
                    let dx = candidate.position[0] - cell_pos[0];
                    let dy = candidate.position[1] - cell_pos[1];
                    if dx * dx + dy * dy < exclusion * exclusion {
                        blocked = true;
                    }
                });
            if !blocked {
                commands.spawn((
                    FoodEntity(candidate),
                    Mesh2d(food_mesh.0.clone()),
                    MeshMaterial2d(food_material.0.clone()),
                    Transform::from_xyz(candidate.position[0], candidate.position[1], -1.0),
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
        let eat_r = EAT_RADIUS * cell.0.genome.body_size;
        let r2 = eat_r * eat_r;
        let mut to_eat: Option<(Entity, [f32; 2])> = None;
        food_grid.0.for_each_in_radius(pos, eat_r, |food_e, food_pos, _| {
            if to_eat.is_some() || eaten.contains(&food_e) {
                return;
            }
            let dx = pos[0] - food_pos[0];
            let dy = pos[1] - food_pos[1];
            if dx * dx + dy * dy <= r2 {
                to_eat = Some((food_e, food_pos));
            }
        });
        if let Some((food_e, food_pos)) = to_eat {
            cell.0.energy += FOOD_VALUE * food_multiplier(world_map.0.sample(food_pos));
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

fn sync_transforms(mut cells: Query<(&CellEntity, &mut Transform)>) {
    for (cell, mut transform) in &mut cells {
        transform.translation.x = cell.0.position[0];
        transform.translation.y = cell.0.position[1];
        transform.rotation = Quat::from_rotation_z(cell.0.heading);
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

fn camera_zoom(
    mut wheel: MessageReader<MouseWheel>,
    window: Single<&Window>,
    camera: Single<(&mut Transform, &mut Projection), With<Camera2d>>,
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
    let (mut transform, mut projection) = camera.into_inner();
    let Projection::Orthographic(ortho) = &mut *projection else {
        return;
    };

    let old_scale = ortho.scale;
    let new_scale = (old_scale * (-scroll * CAMERA_ZOOM_STEP).exp())
        .clamp(CAMERA_ZOOM_MIN, CAMERA_ZOOM_MAX);
    if new_scale == old_scale {
        return;
    }
    ortho.scale = new_scale;

    // Cursor-anchored zoom: shift camera so the world point under the cursor
    // stays put. World under cursor = C + offset · scale, where offset is the
    // cursor's distance from viewport center (Y flipped). To keep it fixed
    // when scale goes s → s', move camera by offset · s · (1 - s'/s).
    let Some(cursor) = window.cursor_position() else {
        return;
    };
    let viewport = Vec2::new(window.resolution.width(), window.resolution.height());
    let offset = (cursor - viewport * 0.5) * Vec2::new(1.0, -1.0);
    let factor = new_scale / old_scale;
    let delta = offset * old_scale * (1.0 - factor);
    transform.translation.x += delta.x;
    transform.translation.y += delta.y;
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

    // Single pass over the cells query — sum / sum_sq feed both mean and
    // population variance without keeping per-cell Vecs around. f64 accum so
    // sumsq - mean² doesn't lose precision once the population grows.
    let mut count = 0usize;
    let mut spd_sum = 0.0_f64;
    let mut spd_sumsq = 0.0_f64;
    let mut vis_sum = 0.0_f64;
    let mut vis_sumsq = 0.0_f64;
    let mut trn_sum = 0.0_f64;
    let mut size_sum = 0.0_f64;
    let mut size_sumsq = 0.0_f64;
    let mut e_sum = 0.0_f64;
    let mut lineages: HashSet<u64> = HashSet::new();
    let mut oldest_age: u64 = 0;
    let current_gen = clock.0.generation;
    for c in &cells {
        count += 1;
        let s = c.0.genome.max_speed as f64;
        let v = c.0.genome.vision_radius as f64;
        let t = c.0.genome.turn_rate as f64;
        let bs = c.0.genome.body_size as f64;
        let e = c.0.energy as f64;
        spd_sum += s;
        spd_sumsq += s * s;
        vis_sum += v;
        vis_sumsq += v * v;
        trn_sum += t;
        size_sum += bs;
        size_sumsq += bs * bs;
        e_sum += e;
        lineages.insert(c.0.lineage_id);
        let age = current_gen.saturating_sub(c.0.lineage_birth_gen);
        if age > oldest_age {
            oldest_age = age;
        }
    }
    let food_count = foods.iter().count();
    let lineage_count = lineages.len();

    let (spd_avg, spd_dev, vis_avg, vis_dev, trn_avg, size_avg, size_dev, e_avg) = if count == 0 {
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    } else {
        let n = count as f64;
        let spd_m = spd_sum / n;
        let vis_m = vis_sum / n;
        let size_m = size_sum / n;
        (
            spd_m,
            ((spd_sumsq / n) - spd_m * spd_m).max(0.0).sqrt(),
            vis_m,
            ((vis_sumsq / n) - vis_m * vis_m).max(0.0).sqrt(),
            trn_sum / n,
            size_m,
            ((size_sumsq / n) - size_m * size_m).max(0.0).sqrt(),
            e_sum / n,
        )
    };

    let mut text = text.into_inner();
    text.0 = format!(
        "tick     {}\ngen      {}\nepoch    {}\nspeed    {}\ncells    {}\nfood     {}\ndensity  {:.2}\nfps      {:.0}\nspd_avg  {:.1}\nspd_dev  {:.2}\nvis_avg  {:.1}\nvis_dev  {:.2}\ntrn_avg  {:.2}\nsize_avg {:.2}\nsize_dev {:.3}\ne_avg    {:.1}\nlineages {}\noldest   {}",
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
        size_avg,
        size_dev,
        e_avg,
        lineage_count,
        oldest_age,
    );
}

fn cell_reproduces_on_threshold(
    mut cells: Query<(Entity, &mut CellEntity), Without<Dying>>,
    cell_mesh: Res<CellMesh>,
    cell_material: Res<CellMaterialHandle>,
    mut commands: Commands,
) {
    let current_pop = cells.iter().count();
    if current_pop >= MAX_POPULATION {
        return;
    }
    let budget = MAX_POPULATION - current_pop;

    // Snapshot fertile cells (immutable iter on a mut query is fine).
    let fertile: Vec<(Entity, [f32; 2])> = cells
        .iter()
        .filter(|(_, c)| c.0.energy >= REPRODUCE_THRESHOLD)
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
            let d2 = dx * dx + dy * dy;
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
        ];
        to_spawn.push(Cell {
            position: mid_pos,
            velocity: [
                direction.cos() * child_genome.max_speed,
                direction.sin() * child_genome.max_speed,
            ],
            angular_velocity: 0.0,
            energy: energy_a + energy_b,
            heading: direction,
            lineage_id: cell_a.0.lineage_id,
            lineage_birth_gen: cell_a.0.lineage_birth_gen,
            last_inputs: [0.0; BRAIN_INPUTS],
            last_hidden: [0.0; BRAIN_HIDDEN],
            last_outputs: [0.0; BRAIN_OUTPUTS],
            genome: child_genome,
        });
    }

    let mesh = cell_mesh.0.clone();
    let material = cell_material.0.clone();
    for cell in to_spawn {
        commands.spawn((
            CellEntity(cell),
            Mesh2d(mesh.clone()),
            MeshMaterial2d(material.clone()),
            MeshTag(pack_cell_tag(lineage_hue(cell.lineage_id), 1.0)),
            Transform::from_xyz(cell.position[0], cell.position[1], 0.0)
                .with_rotation(Quat::from_rotation_z(cell.heading))
                .with_scale(Vec3::splat(cell.genome.body_size)),
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
                ];
                commands.spawn((
                    FoodEntity(Food { position: pos }),
                    Mesh2d(food_mesh.0.clone()),
                    MeshMaterial2d(food_material.0.clone()),
                    Transform::from_xyz(pos[0], pos[1], -1.0),
                ));
            }
        }
    }
}

fn rebuild_cell_grid(
    mut grid: ResMut<CellGrid>,
    cells: Query<(Entity, &CellEntity), Without<Dying>>,
) {
    grid.0
        .rebuild(cells.iter().map(|(e, c)| (e, c.0.position, c.0.genome.body_size)));
}

fn rebuild_food_grid(
    mut grid: ResMut<FoodGrid>,
    foods: Query<(Entity, &FoodEntity)>,
) {
    grid.0.rebuild(foods.iter().map(|(e, f)| (e, f.0.position, ())));
}

// Generous broad-phase upper bound on "other" body_size — captures candidates
// even when neighbors are oversized. Narrow-phase uses the actual pair sum.
const BROAD_PHASE_SIZE_BUDGET: f32 = 3.0;

fn cell_predates_on_neighbor(
    grid: Res<CellGrid>,
    mut cells: Query<(Entity, &mut CellEntity), Without<Dying>>,
) {
    let mut energy_changes: HashMap<Entity, f32> = HashMap::new();

    for (entity_a, cell_a) in &cells {
        let pos_a = cell_a.0.position;
        let size_a = cell_a.0.genome.body_size;
        let broad_r = CELL_RADIUS * (size_a + BROAD_PHASE_SIZE_BUDGET);

        grid.0
            .for_each_in_radius(pos_a, broad_r, |entity_b, pos_b, size_b| {
                if entity_b == entity_a {
                    return;
                }
                if size_a < SIZE_RATIO_THRESHOLD * size_b {
                    return;
                }
                let pair_r = CELL_RADIUS * (size_a + size_b);
                let pair_r2 = pair_r * pair_r;
                let dx = pos_a[0] - pos_b[0];
                let dy = pos_a[1] - pos_b[1];
                let d2 = dx * dx + dy * dy;
                if d2 >= pair_r2 {
                    return;
                }
                *energy_changes.entry(entity_a).or_insert(0.0) += PREDATION_GAIN_PER_TICK;
                *energy_changes.entry(entity_b).or_insert(0.0) -= PREDATION_DRAIN_PER_TICK;
            });
    }

    for (entity, delta) in energy_changes {
        if let Ok((_, mut cell)) = cells.get_mut(entity) {
            cell.0.energy += delta;
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
        let size_a = cell_a.0.genome.body_size;
        let broad_r = CELL_RADIUS * (size_a + BROAD_PHASE_SIZE_BUDGET);
        let mut delta = [0.0_f32, 0.0_f32];
        grid.0
            .for_each_in_radius(pos_a, broad_r, |entity_b, pos_b, size_b| {
                if entity_b == entity_a {
                    return;
                }
                let pair_r = CELL_RADIUS * (size_a + size_b);
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
        &mut MeshTag,
    )>,
    mut commands: Commands,
) {
    for (entity, mut d, cell, mut transform, mut tag) in &mut dying {
        if d.ticks_left == 0 {
            commands.entity(entity).despawn();
            continue;
        }
        d.ticks_left -= 1;
        let progress = d.ticks_left as f32 / DEATH_FADE_TICKS as f32;
        transform.scale = Vec3::splat(cell.0.genome.body_size * progress);
        tag.0 = pack_cell_tag(lineage_hue(cell.0.lineage_id), progress);
    }
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
