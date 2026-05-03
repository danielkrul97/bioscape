use bevy::prelude::*;
use bioscape::{Cell, SimClock};

const WORLD_HALF_EXTENT: f32 = 400.0;
const CELL_RADIUS: f32 = 5.0;
const INITIAL_CELLS: usize = 200;
const FIXED_TIMESTEP_HZ: f64 = 60.0;
const TICKS_PER_GENERATION: u64 = 600;
const GENERATIONS_PER_EPOCH: u64 = 100;

#[derive(Component)]
struct CellEntity(Cell);

#[derive(Resource, Debug)]
struct Clock(SimClock);

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
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Bioscape".into(),
                resolution: (1024u32, 768u32).into(),
                ..default()
            }),
            ..default()
        }))
        .insert_resource(ClearColor(Color::srgb(0.05, 0.05, 0.08)))
        .insert_resource(Time::<Fixed>::from_hz(FIXED_TIMESTEP_HZ))
        .insert_resource(Clock(SimClock::new(
            TICKS_PER_GENERATION,
            GENERATIONS_PER_EPOCH,
        )))
        .add_message::<GenerationEnded>()
        .add_message::<EpochEnded>()
        .add_systems(Startup, setup)
        .add_systems(FixedUpdate, (advance_clock, step_cells).chain())
        .add_systems(Update, (speed_input, sync_transforms, log_clock_events))
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    commands.spawn(Camera2d);

    let cell_mesh = meshes.add(Circle::new(CELL_RADIUS));
    let cell_material = materials.add(Color::srgb(0.35, 0.9, 0.55));

    let mut rng = rand::rng();
    for _ in 0..INITIAL_CELLS {
        let cell = Cell::random(&mut rng, WORLD_HALF_EXTENT);
        commands.spawn((
            CellEntity(cell),
            Mesh2d(cell_mesh.clone()),
            MeshMaterial2d(cell_material.clone()),
            Transform::from_xyz(cell.position[0], cell.position[1], 0.0),
        ));
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

fn step_cells(time: Res<Time>, mut cells: Query<&mut CellEntity>) {
    let dt = time.delta_secs();
    for mut cell in &mut cells {
        cell.0.step(dt, WORLD_HALF_EXTENT);
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
