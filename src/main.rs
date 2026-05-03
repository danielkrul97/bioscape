use bevy::prelude::*;
use bioscape::Cell;

const WORLD_HALF_EXTENT: f32 = 400.0;
const CELL_RADIUS: f32 = 5.0;
const INITIAL_CELLS: usize = 200;

#[derive(Component)]
struct CellEntity(Cell);

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
        .add_systems(Startup, setup)
        .add_systems(Update, (step_cells, sync_transforms).chain())
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
