use bevy::prelude::*;
use bioscape::{
    MazeDifficulty, ObstacleField, N_PHEROMONE_CHANNELS, PHEROMONE_GRID_RES, PHEROMONE_GRID_RES_Z,
    SMELL_GRID_RES, SMELL_GRID_RES_Z, VIBRATION_GRID_RES, VIBRATION_GRID_RES_Z, WORLD_HALF,
    WORLD_MAP_SEED,
};

use super::components::{MazeWallEntity, StatsRoot, WorldMapOverlay};
use super::gizmos::ShowVibration;
use super::resources::MazeWorld;

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
    #[cfg(feature = "gpu")] gpu_full: Option<ResMut<super::resources_gpu::GpuFullPipeline>>,
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
        #[cfg(feature = "gpu")]
        if let Some(mut gpu) = gpu_full {
            gpu.smell.clear_obstacle_mask();
            gpu.pheromone.clear_obstacle_mask();
            gpu.vibration.clear_obstacle_mask();
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
    #[cfg(feature = "gpu")]
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
    }
    maze.field = Some(field);
    maze.smell_mask = smell_mask;
    maze.pheromone_masks = pheromone_masks;
    maze.vibration_mask = vibration_mask;
}
