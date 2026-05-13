use super::*;
use super::config::{
    CAMERA_OFFSET_DISTANCE, CAMERA_PITCH_INITIAL, CAMERA_PITCH_MAX, CAMERA_PITCH_MIN,
    CAMERA_SCALE_INITIAL,
};
use super::world_map::{food_multiplier, hazard_drain};
use super::resources::WorldExtent;
use super::world_map::food_target;
use bioscape::{
    HAZARD_AMP, HAZARD_DRAIN_PER_SEC, HAZARD_FLOOR, WORLD_MAP_FOOD_AMP, WORLD_MAP_FOOD_FLOOR,
};
use std::time::Duration;

const FLT_EPS: f32 = 1e-5;

fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
    (a - b).abs() <= eps
}

#[test]
fn adhesion_hue_covers_eight_bins() {
    let mut seen = std::collections::BTreeSet::new();
    for ty in 0u8..8 {
        let h = adhesion_hue(ty);
        assert!(h >= 0.0 && h < 360.0);
        seen.insert((h * 1000.0) as i32);
    }
    assert_eq!(seen.len(), 8);
}

#[test]
fn adhesion_hue_is_modular_on_8() {
    for ty in 0u8..8 {
        assert!(approx_eq(adhesion_hue(ty), adhesion_hue(ty + 8), FLT_EPS));
        assert!(approx_eq(adhesion_hue(ty), adhesion_hue(ty + 16), FLT_EPS));
    }
}

#[test]
fn adhesion_hue_step_is_45_deg() {
    for ty in 0u8..7 {
        let delta = adhesion_hue(ty + 1) - adhesion_hue(ty);
        assert!(approx_eq(delta, 45.0, 1e-3));
    }
}

#[test]
fn cell_rotation_identity_at_origin() {
    let q = cell_rotation(0.0, 0.0);
    let v = q * Vec3::X;
    assert!(approx_eq(v.x, 1.0, FLT_EPS));
    assert!(approx_eq(v.y, 0.0, FLT_EPS));
    assert!(approx_eq(v.z, 0.0, FLT_EPS));
}

#[test]
fn cell_rotation_yaw_only_rotates_in_xy_plane() {
    let q = cell_rotation(std::f32::consts::FRAC_PI_2, 0.0);
    let v = q * Vec3::X;
    assert!(approx_eq(v.x, 0.0, 1e-5));
    assert!(approx_eq(v.y, 1.0, 1e-5));
    assert!(approx_eq(v.z, 0.0, 1e-5));
}

#[test]
fn cell_rotation_pitch_lifts_z() {
    let q = cell_rotation(0.0, std::f32::consts::FRAC_PI_2);
    let v = q * Vec3::X;
    assert!(approx_eq(v.x, 0.0, 1e-5));
    assert!(approx_eq(v.y, 0.0, 1e-5));
    assert!(approx_eq(v.z, 1.0, 1e-5));
}

#[test]
fn cell_rotation_matches_forward_vector() {
    let cases = [
        (0.0_f32, 0.0_f32),
        (0.5, 0.3),
        (-1.2, 0.8),
        (3.1, -0.4),
    ];
    for (yaw, pitch) in cases {
        let q = cell_rotation(yaw, pitch);
        let v = q * Vec3::X;
        let fwd = bioscape::forward_vector(yaw, pitch);
        assert!(approx_eq(v.x, fwd[0], 1e-4), "yaw={yaw} pitch={pitch}");
        assert!(approx_eq(v.y, fwd[1], 1e-4));
        assert!(approx_eq(v.z, fwd[2], 1e-4));
    }
}

#[test]
fn food_multiplier_at_noise_zero_is_floor() {
    assert!(approx_eq(food_multiplier(0.0), WORLD_MAP_FOOD_FLOOR, FLT_EPS));
}

#[test]
fn food_multiplier_at_noise_one_is_floor_plus_amp() {
    assert!(approx_eq(
        food_multiplier(1.0),
        WORLD_MAP_FOOD_FLOOR + WORLD_MAP_FOOD_AMP,
        FLT_EPS,
    ));
}

#[test]
fn food_multiplier_is_linear() {
    let lo = food_multiplier(0.25);
    let hi = food_multiplier(0.75);
    let mid = food_multiplier(0.5);
    assert!(approx_eq((lo + hi) * 0.5, mid, FLT_EPS));
}

#[test]
fn hazard_drain_zero_at_floor_zero_noise() {
    assert!(approx_eq(hazard_drain(0.0), HAZARD_DRAIN_PER_SEC * HAZARD_FLOOR, FLT_EPS));
}

#[test]
fn hazard_drain_max_at_noise_one() {
    let expected = HAZARD_DRAIN_PER_SEC * (HAZARD_FLOOR + HAZARD_AMP);
    assert!(approx_eq(hazard_drain(1.0), expected, FLT_EPS));
}

#[test]
fn hazard_drain_is_monotonic_in_noise() {
    let samples = [-0.5_f32, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5];
    let mut prev = f32::NEG_INFINITY;
    for s in samples {
        let v = hazard_drain(s);
        assert!(v >= prev);
        prev = v;
    }
}

#[test]
fn food_target_scales_with_factor() {
    let extent = WorldExtent {
        half_x: 100.0,
        half_y: 100.0,
        half_z: 50.0,
    };
    let n0 = food_target(&extent, 0.0);
    let n1 = food_target(&extent, 1.0);
    let n2 = food_target(&extent, 2.0);
    assert_eq!(n0, 0);
    assert!(n2 >= n1 * 2 - 1 && n2 <= n1 * 2 + 1);
}

#[test]
fn food_target_clamps_negative_factor_to_zero() {
    let extent = WorldExtent {
        half_x: 100.0,
        half_y: 100.0,
        half_z: 50.0,
    };
    assert_eq!(food_target(&extent, -1.0), 0);
    assert_eq!(food_target(&extent, -100.0), 0);
}

#[test]
fn food_target_z_factor_clamps_at_one() {
    let small_z = WorldExtent {
        half_x: 100.0,
        half_y: 100.0,
        half_z: 0.5,
    };
    let larger_z = WorldExtent {
        half_x: 100.0,
        half_y: 100.0,
        half_z: 2.0,
    };
    assert_eq!(food_target(&small_z, 1.0), food_target(&larger_z, 1.0));
}

#[test]
fn food_target_scales_with_area() {
    let small = WorldExtent {
        half_x: 100.0,
        half_y: 100.0,
        half_z: 50.0,
    };
    let big = WorldExtent {
        half_x: 200.0,
        half_y: 200.0,
        half_z: 50.0,
    };
    let n_small = food_target(&small, 1.0);
    let n_big = food_target(&big, 1.0);
    let diff = (n_big as i64 - (n_small as i64) * 4).abs();
    assert!(diff <= 5, "n_big={n_big} ≠ 4×n_small={} (±5 tolerance kvůli `as usize` truncation)", n_small * 4);
}

#[test]
fn world_extent_as_array_round_trips() {
    let e = WorldExtent {
        half_x: 1.0,
        half_y: 2.0,
        half_z: 3.0,
    };
    assert_eq!(e.as_array(), [1.0_f32, 2.0, 3.0]);
}

#[test]
fn food_density_factor_default_is_one() {
    let f: FoodDensityFactor = FoodDensityFactor::default();
    assert!(approx_eq(f.0, 1.0, FLT_EPS));
}

#[test]
fn orbit_camera_default_uses_initial_constants() {
    let cam = OrbitCamera::default();
    assert!(approx_eq(cam.pitch, CAMERA_PITCH_INITIAL, FLT_EPS));
    assert!(approx_eq(cam.scale, CAMERA_SCALE_INITIAL, FLT_EPS));
    assert_eq!(cam.target, Vec3::ZERO);
    assert!(approx_eq(cam.yaw, 0.0, FLT_EPS));
}

#[test]
fn orbit_camera_transform_keeps_distance_constant() {
    let mut cam = OrbitCamera::default();
    cam.target = Vec3::new(10.0, 20.0, 0.0);
    let yaws = [-2.0_f32, -0.5, 0.0, 0.7, 1.5];
    let pitches = [
        CAMERA_PITCH_MIN,
        0.5,
        CAMERA_PITCH_INITIAL,
        CAMERA_PITCH_MAX,
    ];
    for &yaw in &yaws {
        for &pitch in &pitches {
            cam.yaw = yaw;
            cam.pitch = pitch;
            let t = cam.transform();
            let d = (t.translation - cam.target).length();
            assert!(
                (d - CAMERA_OFFSET_DISTANCE).abs() < 0.5,
                "yaw={yaw} pitch={pitch} d={d}"
            );
        }
    }
}

#[test]
fn cell_slot_map_allocate_returns_dense_slots() {
    let mut map = CellSlotMap::default();
    let e0 = Entity::from_raw_u32(1).unwrap();
    let e1 = Entity::from_raw_u32(2).unwrap();
    let e2 = Entity::from_raw_u32(3).unwrap();
    assert_eq!(map.allocate(e0), 0);
    assert_eq!(map.allocate(e1), 1);
    assert_eq!(map.allocate(e2), 2);
    assert_eq!(map.len(), 3);
    assert_eq!(map.slot_of(e1), Some(1));
}

#[test]
fn cell_slot_map_release_last_returns_no_move() {
    let mut map = CellSlotMap::default();
    let e0 = Entity::from_raw_u32(1).unwrap();
    let e1 = Entity::from_raw_u32(2).unwrap();
    map.allocate(e0);
    map.allocate(e1);
    let result = map.release(e1);
    assert_eq!(result, Some((1, None)));
    assert_eq!(map.len(), 1);
    assert_eq!(map.slot_of(e1), None);
    assert_eq!(map.slot_of(e0), Some(0));
}

#[test]
fn cell_slot_map_release_middle_swaps_last_into_freed_slot() {
    let mut map = CellSlotMap::default();
    let e0 = Entity::from_raw_u32(1).unwrap();
    let e1 = Entity::from_raw_u32(2).unwrap();
    let e2 = Entity::from_raw_u32(3).unwrap();
    map.allocate(e0);
    map.allocate(e1);
    map.allocate(e2);
    let result = map.release(e0);
    assert_eq!(result, Some((0, Some(e2))));
    assert_eq!(map.len(), 2);
    assert_eq!(map.slot_of(e0), None);
    assert_eq!(map.slot_of(e2), Some(0));
    assert_eq!(map.slot_of(e1), Some(1));
}

#[test]
fn cell_slot_map_release_unknown_entity_is_none() {
    let mut map = CellSlotMap::default();
    let e0 = Entity::from_raw_u32(1).unwrap();
    let unknown = Entity::from_raw_u32(99).unwrap();
    map.allocate(e0);
    assert_eq!(map.release(unknown), None);
    assert_eq!(map.len(), 1);
}

#[test]
fn adhesion_materials_default_is_eight_empty_slots() {
    let cache: AdhesionMaterials = AdhesionMaterials::default();
    assert_eq!(cache.0.len(), 8);
    assert!(cache.0.iter().all(|h| h.is_none()));
}

#[test]
fn tick_counter_default_is_zero() {
    let counter: TickCounter = TickCounter::default();
    assert_eq!(counter.ticks_this_frame, 0);
    assert_eq!(counter.sim_ms_this_frame, 0.0);
    assert!(counter.tick_start.is_none());
}

#[test]
fn setup_time_cap_caps_max_delta_at_50ms() {
    use bevy::app::App;
    use bevy::prelude::*;
    use bevy::time::TimePlugin;

    let mut app = App::new();
    app.add_plugins(TimePlugin);
    app.add_systems(Startup, setup_time_cap);
    app.update();

    let virt = app.world().resource::<Time<Virtual>>();
    assert_eq!(virt.max_delta(), Duration::from_millis(50));
}

#[test]
fn tick_start_records_instant_and_tick_end_increments_counter() {
    use bevy::app::App;
    use bevy::prelude::*;

    let mut app = App::new();
    app.init_resource::<TickCounter>();
    app.add_systems(Startup, (tick_start, tick_end).chain());
    app.update();

    let counter = app.world().resource::<TickCounter>();
    assert_eq!(counter.ticks_this_frame, 1);
    assert!(counter.tick_start.is_none());
    assert!(counter.sim_ms_this_frame >= 0.0);
}

#[test]
fn advance_clock_writes_generation_ended_message() {
    use bevy::app::App;
    use bevy::prelude::*;

    let mut app = App::new();
    let mut clock = SimClock::new(2, 1000);
    clock.tick = 1;
    app.insert_resource(Clock(clock));
    app.add_message::<GenerationEnded>();
    app.add_message::<EpochEnded>();
    app.add_systems(Update, advance_clock);
    app.update();

    let resource_clock = app.world().resource::<Clock>();
    assert_eq!(resource_clock.0.tick, 2);
    assert_eq!(resource_clock.0.generation, 1);
    let msgs = app.world().resource::<Messages<GenerationEnded>>();
    assert_eq!(msgs.len(), 1);
}
