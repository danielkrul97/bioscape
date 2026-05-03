use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::prelude::*;
use bevy::sprite_render::AlphaMode2d;
use bevy::window::WindowResized;
use bioscape::{Cell, Food, MutationConfig, SimClock, BRAIN_INPUTS};
use rand::Rng;
use std::collections::HashMap;

const CELL_RADIUS: f32 = 5.0;
const FOOD_RADIUS: f32 = 2.5;
const INITIAL_CELLS: usize = 200;
const FIXED_TIMESTEP_HZ: f64 = 60.0;
const TICKS_PER_GENERATION: u64 = 600;
const GENERATIONS_PER_EPOCH: u64 = 100;
const ENERGY_COST_PER_DISTANCE: f32 = 0.1;
const VISION_COST_PER_RADIUS: f32 = 0.05;
const FOOD_VALUE: f32 = 20.0;
const FOOD_COUNT_TARGET: usize = 300;
const FOOD_SPAWN_RATE: usize = 5;
const EAT_RADIUS: f32 = 8.0;
const REPRODUCE_THRESHOLD: f32 = 200.0;
const MAX_POPULATION: usize = 1000;
const DEATH_FADE_TICKS: u32 = 30;
const GRID_CELL_SIZE: f32 = 100.0;
const CARRION_FOOD_COUNT: usize = 2;
const MUTATION_CONFIG: MutationConfig = MutationConfig {
    sigma_speed: 3.0,
    sigma_hue: 5.0,
    sigma_vision: 3.0,
    sigma_turn_rate: 0.3,
    sigma_brain: 0.2,
};

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

type GridBuckets = HashMap<(i32, i32), Vec<(Entity, [f32; 2])>>;

#[derive(Resource)]
struct CellGrid {
    cell_size: f32,
    buckets: GridBuckets,
}

impl Default for CellGrid {
    fn default() -> Self {
        Self {
            cell_size: GRID_CELL_SIZE,
            buckets: HashMap::new(),
        }
    }
}

impl CellGrid {
    fn key_of(&self, pos: [f32; 2]) -> (i32, i32) {
        (
            (pos[0] / self.cell_size).floor() as i32,
            (pos[1] / self.cell_size).floor() as i32,
        )
    }

    fn rebuild<I: IntoIterator<Item = (Entity, [f32; 2])>>(&mut self, items: I) {
        self.buckets.clear();
        for (e, pos) in items {
            let key = self.key_of(pos);
            self.buckets.entry(key).or_default().push((e, pos));
        }
    }

    fn neighbors_within(&self, pos: [f32; 2], radius: f32) -> Vec<(Entity, [f32; 2])> {
        let r_cells = (radius / self.cell_size).ceil() as i32;
        let (cx, cy) = self.key_of(pos);
        let mut out = Vec::new();
        for dx in -r_cells..=r_cells {
            for dy in -r_cells..=r_cells {
                if let Some(bucket) = self.buckets.get(&(cx + dx, cy + dy)) {
                    out.extend(bucket.iter().copied());
                }
            }
        }
        out
    }
}

#[derive(Resource)]
struct CellMesh(Handle<Mesh>);

#[derive(Resource)]
struct FoodMesh(Handle<Mesh>);

#[derive(Resource)]
struct FoodMaterial(Handle<ColorMaterial>);

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
        .insert_resource(Time::<Fixed>::from_hz(FIXED_TIMESTEP_HZ))
        .insert_resource(Clock(SimClock::new(
            TICKS_PER_GENERATION,
            GENERATIONS_PER_EPOCH,
        )))
        .init_resource::<CellGrid>()
        .add_message::<GenerationEnded>()
        .add_message::<EpochEnded>()
        .add_systems(Startup, (setup, setup_stats_overlay, rebuild_cell_grid).chain())
        .add_systems(
            FixedUpdate,
            (
                advance_clock,
                cells_brain_act,
                step_cells,
                rebuild_cell_grid,
                resolve_cell_collisions,
                spawn_food,
                cell_eats_food,
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
                sync_transforms,
                log_clock_events,
                toggle_stats_overlay,
                track_window_resize,
                update_stats_overlay,
            ),
        )
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    mut window: Single<&mut Window>,
) {
    window.set_maximized(true);
    let extent = WorldExtent {
        half_x: window.resolution.width() / 2.0,
        half_y: window.resolution.height() / 2.0,
    };
    let half = extent.as_array();
    commands.insert_resource(extent);

    commands.spawn(Camera2d);

    let cell_mesh = meshes.add(Circle::new(CELL_RADIUS));
    let food_mesh = meshes.add(Circle::new(FOOD_RADIUS));
    let food_material = materials.add(Color::srgb(0.95, 0.95, 0.85));

    let mut rng = rand::rng();
    for _ in 0..INITIAL_CELLS {
        let cell = Cell::random(&mut rng, half);
        let material = make_cell_material(&mut materials, cell.genome.color_hue);
        commands.spawn((
            CellEntity(cell),
            Mesh2d(cell_mesh.clone()),
            MeshMaterial2d(material),
            Transform::from_xyz(cell.position[0], cell.position[1], 0.0),
        ));
    }
    for _ in 0..FOOD_COUNT_TARGET {
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
}

fn track_window_resize(
    mut events: MessageReader<WindowResized>,
    mut extent: ResMut<WorldExtent>,
) {
    for ev in events.read() {
        extent.half_x = ev.width / 2.0;
        extent.half_y = ev.height / 2.0;
    }
}

fn hue_to_color(hue_deg: f32) -> Color {
    Color::hsl(hue_deg, 0.75, 0.55)
}

fn make_cell_material(
    materials: &mut Assets<ColorMaterial>,
    hue_deg: f32,
) -> Handle<ColorMaterial> {
    // Explicit Blend so death fade-out alpha actually renders;
    // ColorMaterial::From<Color> picks Opaque when color.alpha() == 1.0.
    materials.add(ColorMaterial {
        color: hue_to_color(hue_deg),
        alpha_mode: AlphaMode2d::Blend,
        ..default()
    })
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
        cell.0
            .step(dt, half, ENERGY_COST_PER_DISTANCE, VISION_COST_PER_RADIUS);
    }
}

fn cells_brain_act(
    time: Res<Time>,
    grid: Res<CellGrid>,
    mut cells: Query<(Entity, &mut CellEntity), Without<Dying>>,
    foods: Query<&FoodEntity>,
) {
    let dt = time.delta_secs();
    let food_positions: Vec<[f32; 2]> = foods.iter().map(|f| f.0.position).collect();

    for (entity, mut cell) in &mut cells {
        let pos = cell.0.position;
        let vision_r = cell.0.genome.vision_radius;
        let vr2 = vision_r * vision_r;

        let mut nearest_food: Option<[f32; 2]> = None;
        let mut best_food_d2 = f32::MAX;
        for fp in &food_positions {
            let dx = fp[0] - pos[0];
            let dy = fp[1] - pos[1];
            let d2 = dx * dx + dy * dy;
            if d2 <= vr2 && d2 < best_food_d2 {
                best_food_d2 = d2;
                nearest_food = Some(*fp);
            }
        }

        let mut nearest_cell: Option<[f32; 2]> = None;
        let mut best_cell_d2 = f32::MAX;
        for (other, other_pos) in grid.neighbors_within(pos, vision_r) {
            if other == entity {
                continue;
            }
            let dx = other_pos[0] - pos[0];
            let dy = other_pos[1] - pos[1];
            let d2 = dx * dx + dy * dy;
            if d2 <= vr2 && d2 < best_cell_d2 {
                best_cell_d2 = d2;
                nearest_cell = Some(other_pos);
            }
        }

        let max_speed = cell.0.genome.max_speed;
        let speed_norm =
            (cell.0.velocity[0].hypot(cell.0.velocity[1]) / max_speed).clamp(0.0, 1.0);
        let energy_norm = (cell.0.energy / REPRODUCE_THRESHOLD).clamp(0.0, 1.5);

        let mut inputs = [0.0_f32; BRAIN_INPUTS];
        if let Some(target) = nearest_food {
            inputs[0] = (target[0] - pos[0]) / vision_r;
            inputs[1] = (target[1] - pos[1]) / vision_r;
        }
        if let Some(target) = nearest_cell {
            inputs[2] = (target[0] - pos[0]) / vision_r;
            inputs[3] = (target[1] - pos[1]) / vision_r;
        }
        inputs[4] = energy_norm;
        inputs[5] = speed_norm;

        let outputs = cell.0.genome.brain.forward(&inputs);
        let turn_signal = outputs[0];
        let thrust_norm = (outputs[1] + 1.0) * 0.5;

        let new_angle = cell.0.heading + turn_signal * cell.0.genome.turn_rate * dt;
        let target_speed = thrust_norm * max_speed;
        cell.0.heading = new_angle;
        cell.0.velocity[0] = new_angle.cos() * target_speed;
        cell.0.velocity[1] = new_angle.sin() * target_speed;
    }
}

fn spawn_food(
    foods: Query<(), With<FoodEntity>>,
    extent: Res<WorldExtent>,
    food_mesh: Res<FoodMesh>,
    food_material: Res<FoodMaterial>,
    mut commands: Commands,
) {
    let count = foods.iter().count();
    if count >= FOOD_COUNT_TARGET {
        return;
    }
    let to_spawn = (FOOD_COUNT_TARGET - count).min(FOOD_SPAWN_RATE);
    let mut rng = rand::rng();
    let half = extent.as_array();
    for _ in 0..to_spawn {
        let food = Food::random(&mut rng, half);
        commands.spawn((
            FoodEntity(food),
            Mesh2d(food_mesh.0.clone()),
            MeshMaterial2d(food_material.0.clone()),
            Transform::from_xyz(food.position[0], food.position[1], -1.0),
        ));
    }
}

fn cell_eats_food(
    mut cells: Query<&mut CellEntity, Without<Dying>>,
    foods: Query<(Entity, &FoodEntity)>,
    mut commands: Commands,
) {
    let mut food_pool: Vec<(Entity, Food, bool)> =
        foods.iter().map(|(e, f)| (e, f.0, false)).collect();

    for mut cell in &mut cells {
        for entry in food_pool.iter_mut() {
            if entry.2 {
                continue;
            }
            if cell.0.try_eat(&entry.1, EAT_RADIUS, FOOD_VALUE) {
                entry.2 = true;
                commands.entity(entry.0).despawn();
                break;
            }
        }
    }
}

fn sync_transforms(mut cells: Query<(&CellEntity, &mut Transform)>) {
    for (cell, mut transform) in &mut cells {
        transform.translation.x = cell.0.position[0];
        transform.translation.y = cell.0.position[1];
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
    let speeds: Vec<f32> = cells.iter().map(|c| c.0.genome.max_speed).collect();
    let visions: Vec<f32> = cells.iter().map(|c| c.0.genome.vision_radius).collect();
    let turns: Vec<f32> = cells.iter().map(|c| c.0.genome.turn_rate).collect();
    let energies: Vec<f32> = cells.iter().map(|c| c.0.energy).collect();
    let cell_count = speeds.len();
    let food_count = foods.iter().count();
    let (spd_avg, spd_dev) = mean_stddev(&speeds);
    let (vis_avg, vis_dev) = mean_stddev(&visions);
    let (trn_avg, _) = mean_stddev(&turns);
    let (e_avg, _) = mean_stddev(&energies);

    let mut text = text.into_inner();
    text.0 = format!(
        "tick    {}\ngen     {}\nepoch   {}\nspeed   {}\ncells   {}\nfood    {}\nfps     {:.0}\nspd_avg {:.1}\nspd_dev {:.2}\nvis_avg {:.1}\nvis_dev {:.2}\ntrn_avg {:.2}\ne_avg   {:.1}",
        clock.0.tick,
        clock.0.generation,
        clock.0.epoch,
        speed,
        cell_count,
        food_count,
        fps,
        spd_avg,
        spd_dev,
        vis_avg,
        vis_dev,
        trn_avg,
        e_avg,
    );
}

fn mean_stddev(values: &[f32]) -> (f32, f32) {
    if values.is_empty() {
        return (0.0, 0.0);
    }
    let n = values.len() as f32;
    let mean = values.iter().sum::<f32>() / n;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
    (mean, variance.sqrt())
}

fn cell_reproduces_on_threshold(
    mut cells: Query<&mut CellEntity, Without<Dying>>,
    cell_mesh: Res<CellMesh>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    mut commands: Commands,
) {
    let current_pop = cells.iter().count();
    if current_pop >= MAX_POPULATION {
        return;
    }
    let mut budget = MAX_POPULATION - current_pop;

    let mut rng = rand::rng();
    let mut to_spawn: Vec<Cell> = Vec::new();

    for mut cell in &mut cells {
        if budget == 0 {
            break;
        }
        if cell.0.energy < REPRODUCE_THRESHOLD {
            continue;
        }
        cell.0.energy *= 0.5;
        let child_genome = cell.0.genome.mutate(&mut rng, &MUTATION_CONFIG);
        let direction = rng.random_range(0.0..core::f32::consts::TAU);
        to_spawn.push(Cell {
            position: cell.0.position,
            velocity: [
                direction.cos() * child_genome.max_speed,
                direction.sin() * child_genome.max_speed,
            ],
            energy: cell.0.energy,
            heading: direction,
            genome: child_genome,
        });
        budget -= 1;
    }

    let mesh = cell_mesh.0.clone();
    for cell in to_spawn {
        let material = make_cell_material(&mut materials, cell.genome.color_hue);
        commands.spawn((
            CellEntity(cell),
            Mesh2d(mesh.clone()),
            MeshMaterial2d(material),
            Transform::from_xyz(cell.position[0], cell.position[1], 0.0),
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
    grid.rebuild(cells.iter().map(|(e, c)| (e, c.0.position)));
}

fn resolve_cell_collisions(
    grid: Res<CellGrid>,
    mut cells: Query<(Entity, &mut CellEntity), Without<Dying>>,
) {
    let positions: HashMap<Entity, [f32; 2]> =
        cells.iter().map(|(e, c)| (e, c.0.position)).collect();
    let collision_r = 2.0 * CELL_RADIUS;
    let collision_r2 = collision_r * collision_r;
    let mut adjustments: HashMap<Entity, [f32; 2]> = HashMap::new();

    for (&entity_a, &pos_a) in &positions {
        let mut delta = [0.0_f32, 0.0_f32];
        for (entity_b, _grid_pos) in grid.neighbors_within(pos_a, collision_r) {
            if entity_b == entity_a {
                continue;
            }
            let Some(&pos_b) = positions.get(&entity_b) else {
                continue;
            };
            let dx = pos_a[0] - pos_b[0];
            let dy = pos_a[1] - pos_b[1];
            let d2 = dx * dx + dy * dy;
            if d2 < collision_r2 && d2 > 0.0 {
                let d = d2.sqrt();
                let overlap = collision_r - d;
                delta[0] += (dx / d) * overlap * 0.5;
                delta[1] += (dy / d) * overlap * 0.5;
            }
        }
        if delta != [0.0, 0.0] {
            adjustments.insert(entity_a, delta);
        }
    }

    for (entity, delta) in adjustments {
        if let Ok((_, mut cell)) = cells.get_mut(entity) {
            cell.0.position[0] += delta[0];
            cell.0.position[1] += delta[1];
        }
    }
}

fn tick_death_fade(
    mut dying: Query<(Entity, &mut Dying, &mut Transform, &MeshMaterial2d<ColorMaterial>)>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    mut commands: Commands,
) {
    for (entity, mut d, mut transform, mat_handle) in &mut dying {
        if d.ticks_left == 0 {
            commands.entity(entity).despawn();
            continue;
        }
        d.ticks_left -= 1;
        let progress = d.ticks_left as f32 / DEATH_FADE_TICKS as f32;
        transform.scale = Vec3::splat(progress);
        if let Some(mat) = materials.get_mut(&mat_handle.0) {
            mat.color = mat.color.with_alpha(progress);
        }
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
