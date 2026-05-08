use bevy::prelude::*;

use super::components::{StatsRoot, WorldMapOverlay};

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
