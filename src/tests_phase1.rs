//! Fáze 1 — pure-fn moduly: events, reproduction, food, chemistry edge cases,
//! sensors, physics_utils. Cíl: doplnit větve, které dnes nejsou v src/tests.rs.

#![allow(unused_imports)]

use crate::test_helpers::*;
use crate::*;
use core::f32::consts::{FRAC_PI_2, FRAC_PI_4, PI, TAU};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

// ─── events.rs ──────────────────────────────────────────────────────────────

fn shock_cfg_with_mean(mean: u32) -> ShockScheduleConfig {
    ShockScheduleConfig {
        mean_gens_between: mean,
        type_weights: [1.0, 1.0, 1.0],
        intensity_min: 0.5,
        intensity_max: 0.9,
        duration_min_gens: 5,
        duration_max_gens: 10,
        ramp_gens: 2,
        spatial_global_prob: 0.5,
        spatial_radius_min_frac: 0.2,
        spatial_radius_max_frac: 0.6,
    }
}

#[test]
fn event_calendar_zero_max_gens_is_empty() {
    let cfg = shock_cfg_with_mean(20);
    let cal = EventCalendar::generate(42, &cfg, 0);
    assert!(cal.events.is_empty());
    assert_eq!(cal.seed, 42);
}

#[test]
fn event_calendar_active_iterator_filters_inactive_events() {
    let cal = EventCalendar {
        events: vec![
            ShockEvent {
                kind: ShockKind::HazardPulse,
                start_gen: 10,
                duration_gen: 5,
                ramp_gens: 1,
                intensity: 1.0,
                center_xy: None,
                radius: None,
            },
            ShockEvent {
                kind: ShockKind::ClimateShift,
                start_gen: 50,
                duration_gen: 3,
                ramp_gens: 1,
                intensity: 1.0,
                center_xy: None,
                radius: None,
            },
        ],
        seed: 0,
    };
    let active_at_12: Vec<_> = cal.active(12, 0).collect();
    assert_eq!(active_at_12.len(), 1);
    assert_eq!(active_at_12[0].kind, ShockKind::HazardPulse);
    let active_at_15: Vec<_> = cal.active(15, 0).collect();
    assert!(active_at_15.is_empty());
    let active_at_51: Vec<_> = cal.active(51, 0).collect();
    assert_eq!(active_at_51.len(), 1);
    assert_eq!(active_at_51[0].kind, ShockKind::ClimateShift);
}

#[test]
fn event_calendar_inverted_intensity_range_clamps() {
    let mut cfg = shock_cfg_with_mean(20);
    cfg.intensity_min = 0.9;
    cfg.intensity_max = 0.3;
    let cal = EventCalendar::generate(7, &cfg, 500);
    for e in &cal.events {
        assert!(e.intensity >= 0.3 - 1e-6 && e.intensity <= 0.9 + 1e-6);
    }
}

#[test]
fn event_calendar_zero_type_weights_falls_back_to_hazard_pulse() {
    let mut cfg = shock_cfg_with_mean(20);
    cfg.type_weights = [0.0; SHOCK_KIND_COUNT];
    let cal = EventCalendar::generate(13, &cfg, 500);
    assert!(!cal.events.is_empty());
    for e in &cal.events {
        assert_eq!(e.kind, ShockKind::HazardPulse);
    }
}

#[test]
fn event_calendar_only_climate_weight_yields_only_climate_shifts() {
    let mut cfg = shock_cfg_with_mean(20);
    cfg.type_weights = [0.0, 1.0, 0.0];
    let cal = EventCalendar::generate(17, &cfg, 500);
    assert!(!cal.events.is_empty());
    for e in &cal.events {
        assert_eq!(e.kind, ShockKind::ClimateShift);
    }
}

#[test]
fn shock_ramp_factor_zero_outside_window() {
    let evt = ShockEvent {
        kind: ShockKind::HazardPulse,
        start_gen: 100,
        duration_gen: 5,
        ramp_gens: 1,
        intensity: 1.0,
        center_xy: None,
        radius: None,
    };
    assert_eq!(shock_ramp_factor(&evt, 50), 0.0);
    assert_eq!(shock_ramp_factor(&evt, 100 + 5), 0.0);
    assert_eq!(shock_ramp_factor(&evt, 200), 0.0);
}

#[test]
fn shock_ramp_factor_zero_duration_returns_zero() {
    let evt = ShockEvent {
        kind: ShockKind::HazardPulse,
        start_gen: 0,
        duration_gen: 0,
        ramp_gens: 1,
        intensity: 1.0,
        center_xy: None,
        radius: None,
    };
    assert_eq!(shock_ramp_factor(&evt, 0), 0.0);
}

#[test]
fn shock_ramp_factor_zero_ramp_uses_triangle_branch() {
    let evt = ShockEvent {
        kind: ShockKind::HazardPulse,
        start_gen: 0,
        duration_gen: 4,
        ramp_gens: 0,
        intensity: 1.0,
        center_xy: None,
        radius: None,
    };
    let f0 = shock_ramp_factor(&evt, 0);
    let f1 = shock_ramp_factor(&evt, 1);
    let f2 = shock_ramp_factor(&evt, 2);
    let f3 = shock_ramp_factor(&evt, 3);
    assert!(f0 > 0.0 && f0 <= 1.0);
    assert!(f1 > f0);
    assert!(f2 >= f1 - 1e-6 || f1 >= f2 - 1e-6);
    assert!(f3 < f1.max(f2));
}

#[test]
fn hazard_shock_multiplier_skips_non_hazard_events() {
    let evt = ShockEvent {
        kind: ShockKind::ClimateShift,
        start_gen: 0,
        duration_gen: 10,
        ramp_gens: 1,
        intensity: 1.0,
        center_xy: None,
        radius: None,
    };
    let m = hazard_shock_multiplier([0.0, 0.0, 0.0], &[evt], 5, 0, WORLD_HALF);
    assert_eq!(m, 1.0);
}

#[test]
fn hazard_shock_multiplier_outside_radius_no_effect() {
    let evt = ShockEvent {
        kind: ShockKind::HazardPulse,
        start_gen: 0,
        duration_gen: 10,
        ramp_gens: 1,
        intensity: 1.0,
        center_xy: Some([0.0, 0.0]),
        radius: Some(50.0),
    };
    let m_inside = hazard_shock_multiplier([0.0, 0.0, 0.0], &[evt], 5, 0, WORLD_HALF);
    let m_outside = hazard_shock_multiplier([200.0, 0.0, 0.0], &[evt], 5, 0, WORLD_HALF);
    assert!(m_inside > 1.0);
    assert_eq!(m_outside, 1.0);
}

#[test]
fn hazard_shock_multiplier_compounds_multiple_pulses() {
    let mk = |start: u64| ShockEvent {
        kind: ShockKind::HazardPulse,
        start_gen: start,
        duration_gen: 10,
        ramp_gens: 1,
        intensity: 1.0,
        center_xy: None,
        radius: None,
    };
    let single = hazard_shock_multiplier([0.0; 3], &[mk(0)], 5, 0, WORLD_HALF);
    let pair = hazard_shock_multiplier([0.0; 3], &[mk(0), mk(0)], 5, 0, WORLD_HALF);
    assert!(pair > single);
    let expected_ratio = single;
    assert!((pair / single - expected_ratio).abs() < 1e-3);
}

#[test]
fn hazard_shock_multiplier_zero_radius_treated_as_global() {
    let evt = ShockEvent {
        kind: ShockKind::HazardPulse,
        start_gen: 0,
        duration_gen: 10,
        ramp_gens: 1,
        intensity: 1.0,
        center_xy: Some([0.0, 0.0]),
        radius: Some(0.0),
    };
    let m_far = hazard_shock_multiplier([500.0, 500.0, 0.0], &[evt], 5, 0, WORLD_HALF);
    assert!(m_far > 1.0);
}

#[test]
fn climate_shock_offset_skips_non_climate_events() {
    let evt = ShockEvent {
        kind: ShockKind::FoodCrash,
        start_gen: 0,
        duration_gen: 10,
        ramp_gens: 1,
        intensity: 1.0,
        center_xy: None,
        radius: None,
    };
    let off = climate_shock_offset(&[evt], 5, [0.0, 0.0], WORLD_HALF);
    assert_eq!(off, 0.0);
}

#[test]
fn climate_shock_offset_spatial_mask_falloff() {
    let evt = ShockEvent {
        kind: ShockKind::ClimateShift,
        start_gen: 0,
        duration_gen: 10,
        ramp_gens: 1,
        intensity: 1.0,
        center_xy: Some([0.0, 0.0]),
        radius: Some(100.0),
    };
    let center = climate_shock_offset(&[evt], 5, [0.0, 0.0], WORLD_HALF);
    let edge = climate_shock_offset(&[evt], 5, [50.0, 0.0], WORLD_HALF);
    let outside = climate_shock_offset(&[evt], 5, [150.0, 0.0], WORLD_HALF);
    assert!(center > edge);
    assert!(edge > 0.0);
    assert_eq!(outside, 0.0);
}

#[test]
fn climate_shock_offset_additive_compound() {
    let mk = || ShockEvent {
        kind: ShockKind::ClimateShift,
        start_gen: 0,
        duration_gen: 10,
        ramp_gens: 1,
        intensity: 1.0,
        center_xy: None,
        radius: None,
    };
    let single = climate_shock_offset(&[mk()], 5, [0.0, 0.0], WORLD_HALF);
    let double = climate_shock_offset(&[mk(), mk()], 5, [0.0, 0.0], WORLD_HALF);
    assert!((double - 2.0 * single).abs() < 1e-3);
}

#[test]
fn food_density_shock_multiplier_skips_non_foodcrash() {
    let evt = ShockEvent {
        kind: ShockKind::HazardPulse,
        start_gen: 0,
        duration_gen: 10,
        ramp_gens: 1,
        intensity: 1.0,
        center_xy: None,
        radius: None,
    };
    let m = food_density_shock_multiplier(&[evt], 5);
    assert_eq!(m, 1.0);
}

#[test]
fn food_density_shock_multiplier_clamped_to_min_factor() {
    let mk = || ShockEvent {
        kind: ShockKind::FoodCrash,
        start_gen: 0,
        duration_gen: 10,
        ramp_gens: 1,
        intensity: 1.0,
        center_xy: None,
        radius: None,
    };
    let many = vec![mk(); 20];
    let m = food_density_shock_multiplier(&many, 5);
    assert!(m >= FOOD_CRASH_MIN_FACTOR - 1e-6);
}

#[test]
fn temperature_at_z_with_shocks_adds_climate_offset() {
    let evt = ShockEvent {
        kind: ShockKind::ClimateShift,
        start_gen: 0,
        duration_gen: 10,
        ramp_gens: 1,
        intensity: 1.0,
        center_xy: None,
        radius: None,
    };
    let base = temperature_at_z(0.0, WORLD_HALF, 0, 0);
    let shocked = temperature_at_z_with_shocks(0.0, WORLD_HALF, 0, 5, &[evt], [0.0, 0.0]);
    assert!(shocked > base);
}

// ─── reproduction.rs ────────────────────────────────────────────────────────

#[test]
fn reject_food_for_richness_clamps_high_richness() {
    let mut rng = StdRng::seed_from_u64(1);
    for _ in 0..200 {
        assert!(!reject_food_for_richness(&mut rng, 5.0));
    }
}

#[test]
fn reject_food_for_richness_clamps_negative_richness() {
    let mut rng = StdRng::seed_from_u64(2);
    let mut rejects = 0;
    for _ in 0..2000 {
        if reject_food_for_richness(&mut rng, -1.0) {
            rejects += 1;
        }
    }
    let frac = rejects as f32 / 2000.0;
    assert!(
        (frac - FOOD_REJECTION_STRENGTH).abs() < 0.05,
        "rejection rate {} should approach STRENGTH {}",
        frac,
        FOOD_REJECTION_STRENGTH
    );
}

#[test]
fn pair_fertile_empty_input_returns_empty() {
    let fertile: Vec<(usize, [f32; 3])> = Vec::new();
    let pairs = pair_fertile(&fertile, 100.0, 10, WORLD_HALF);
    assert!(pairs.is_empty());
}

#[test]
fn pair_fertile_single_cell_returns_empty() {
    let fertile = vec![(0usize, [0.0, 0.0, 0.0])];
    let pairs = pair_fertile(&fertile, 100.0, 10, WORLD_HALF);
    assert!(pairs.is_empty());
}

#[test]
fn pair_fertile_zero_budget_returns_empty() {
    let fertile = vec![
        (0usize, [0.0, 0.0, 0.0]),
        (1usize, [10.0, 0.0, 0.0]),
    ];
    let pairs = pair_fertile(&fertile, 1000.0, 0, WORLD_HALF);
    assert!(pairs.is_empty());
}

#[test]
fn pair_fertile_pairs_each_cell_at_most_once() {
    let fertile = vec![
        (0usize, [0.0, 0.0, 0.0]),
        (1usize, [5.0, 0.0, 0.0]),
        (2usize, [10.0, 0.0, 0.0]),
        (3usize, [200.0, 0.0, 0.0]),
        (4usize, [205.0, 0.0, 0.0]),
    ];
    let pairs = pair_fertile(&fertile, 100.0 * 100.0, 10, WORLD_HALF);
    let mut seen = std::collections::HashSet::new();
    for &(a, b) in &pairs {
        assert!(seen.insert(a), "cell {} paired twice", a);
        assert!(seen.insert(b), "cell {} paired twice", b);
    }
    assert!(!pairs.is_empty());
}

#[test]
fn pair_fertile_picks_nearest_partner() {
    let fertile = vec![
        (0usize, [0.0, 0.0, 0.0]),
        (1usize, [50.0, 0.0, 0.0]),
        (2usize, [10.0, 0.0, 0.0]),
    ];
    let pairs = pair_fertile(&fertile, 100.0 * 100.0, 10, WORLD_HALF);
    assert_eq!(pairs.len(), 1);
    let (a, b) = pairs[0];
    assert!((a == 0 && b == 2) || (a == 2 && b == 0));
}

#[test]
fn pair_fertile_respects_budget_cap() {
    let fertile: Vec<(usize, [f32; 3])> = (0..20)
        .map(|i| (i, [(i as f32) * 5.0, 0.0, 0.0]))
        .collect();
    let pairs = pair_fertile(&fertile, 100.0 * 100.0, 3, WORLD_HALF);
    assert!(pairs.len() <= 3);
}

#[test]
fn pair_fertile_uses_toroidal_distance() {
    let half_x = WORLD_HALF[0];
    let fertile = vec![
        (0usize, [half_x - 5.0, 0.0, 0.0]),
        (1usize, [-half_x + 5.0, 0.0, 0.0]),
    ];
    let pairs = pair_fertile(&fertile, 50.0 * 50.0, 10, WORLD_HALF);
    assert_eq!(pairs.len(), 1);
}

#[test]
fn pick_cluster_parent_match_a_beats_match_b() {
    let mut a = base_cell();
    let mut b = base_cell();
    a.bonds[0] = Some(Bond {
        other_cell_id: 1,
        rest_length: 10.0,
        stiffness: BOND_STIFFNESS,
        damping: BOND_DAMPING,
        age_ticks: 0,
    });
    b.bonds[0] = Some(Bond {
        other_cell_id: 2,
        rest_length: 10.0,
        stiffness: BOND_STIFFNESS,
        damping: BOND_DAMPING,
        age_ticks: 0,
    });
    a.genome.adhesion_type = 1;
    b.genome.adhesion_type = 1;
    let picked = pick_cluster_parent(&a, &b, 1).unwrap();
    assert_eq!(picked.genome.adhesion_type, 1);
    assert!(std::ptr::eq(picked, &a));
}

#[test]
fn pick_cluster_parent_b_match_when_a_unbonded() {
    let a = base_cell();
    let mut b = base_cell();
    b.bonds[0] = Some(Bond {
        other_cell_id: 99,
        rest_length: 10.0,
        stiffness: BOND_STIFFNESS,
        damping: BOND_DAMPING,
        age_ticks: 0,
    });
    b.genome.adhesion_type = 3;
    let picked = pick_cluster_parent(&a, &b, 3).unwrap();
    assert!(std::ptr::eq(picked, &b));
}

#[test]
fn make_mating_child_inherits_lineage_from_a() {
    let mut a = base_cell();
    let mut b = base_cell();
    a.lineage_id = 7;
    a.lineage_birth_gen = 100;
    b.lineage_id = 99;
    b.lineage_birth_gen = 200;
    let mut rng = StdRng::seed_from_u64(0);
    let child = make_mating_child(&a, &b, &mut rng, 42);
    assert_eq!(child.lineage_id, 7);
    assert_eq!(child.lineage_birth_gen, 100);
    assert_eq!(child.cell_id, 42);
    assert_eq!(child.age, 0);
    assert_eq!(child.reproduce_cooldown_ticks, 0);
}

#[test]
fn make_mating_child_energy_is_sum_of_parents() {
    let mut a = base_cell();
    let mut b = base_cell();
    a.energy = 60.0;
    b.energy = 80.0;
    let mut rng = StdRng::seed_from_u64(1);
    let child = make_mating_child(&a, &b, &mut rng, 0);
    assert!((child.energy - 140.0).abs() < 1e-3);
}

#[test]
fn make_mating_child_cell_state_clamped_to_unit_range() {
    let mut a = base_cell();
    let mut b = base_cell();
    a.cell_state = 1.0;
    b.cell_state = 1.0;
    let mut rng = StdRng::seed_from_u64(2);
    for _ in 0..50 {
        let child = make_mating_child(&a, &b, &mut rng, 0);
        assert!(child.cell_state >= 0.0 && child.cell_state <= 1.0);
    }
}

#[test]
fn make_mating_child_starts_without_bonds() {
    let mut a = base_cell();
    let mut b = base_cell();
    a.bonds[0] = Some(Bond {
        other_cell_id: 99,
        rest_length: 10.0,
        stiffness: BOND_STIFFNESS,
        damping: BOND_DAMPING,
        age_ticks: 100,
    });
    b.bonds[0] = Some(Bond {
        other_cell_id: 88,
        rest_length: 10.0,
        stiffness: BOND_STIFFNESS,
        damping: BOND_DAMPING,
        age_ticks: 100,
    });
    let mut rng = StdRng::seed_from_u64(3);
    let child = make_mating_child(&a, &b, &mut rng, 0);
    assert_eq!(child.n_bonds(), 0);
}

#[test]
fn adhesion_velocity_delta_zero_at_exact_contact() {
    let pair_r = 10.0;
    let v = adhesion_velocity_delta([pair_r, 0.0, 0.0], pair_r, pair_r, true);
    assert_eq!(v, [0.0; 3]);
}

#[test]
fn adhesion_velocity_delta_zero_for_zero_dist() {
    let v = adhesion_velocity_delta([0.0; 3], 0.0, 5.0, true);
    assert_eq!(v, [0.0; 3]);
}

#[test]
fn adhesion_velocity_delta_falloff_strongest_near_contact() {
    let pair_r = 10.0;
    let near = adhesion_velocity_delta([pair_r + 0.5, 0.0, 0.0], pair_r + 0.5, pair_r, true);
    let far_dist = pair_r * (ADHESION_RANGE_FACTOR - 0.1);
    let far = adhesion_velocity_delta([far_dist, 0.0, 0.0], far_dist, pair_r, true);
    assert!(near[0].abs() > far[0].abs());
}

#[test]
fn bond_velocity_delta_breaks_at_zero_dist() {
    let bond = Bond {
        other_cell_id: 1,
        rest_length: 10.0,
        stiffness: BOND_STIFFNESS,
        damping: BOND_DAMPING,
        age_ticks: 0,
    };
    let (delta, broken) = bond_velocity_delta(&bond, [0.0; 3], 0.0, [0.0; 3], [0.0; 3]);
    assert!(broken);
    assert_eq!(delta, [0.0; 3]);
}

#[test]
fn bond_velocity_delta_zero_force_at_rest_with_zero_velocities() {
    let bond = Bond {
        other_cell_id: 1,
        rest_length: 10.0,
        stiffness: BOND_STIFFNESS,
        damping: BOND_DAMPING,
        age_ticks: 0,
    };
    let (delta, broken) = bond_velocity_delta(
        &bond,
        [10.0, 0.0, 0.0],
        10.0,
        [0.0; 3],
        [0.0; 3],
    );
    assert!(!broken);
    assert!(delta[0].abs() < 1e-6);
}

// ─── food.rs ────────────────────────────────────────────────────────────────

#[test]
fn food_kind_default_is_plant() {
    let kind = FoodKind::default();
    assert_eq!(kind, FoodKind::Plant);
}

#[test]
fn eat_efficiency_carrion_invariant_to_score() {
    let lo = eat_efficiency(FoodKind::Carrion, 0.0);
    let mid = eat_efficiency(FoodKind::Carrion, 0.5);
    let hi = eat_efficiency(FoodKind::Carrion, 1.0);
    assert_eq!(lo, 0.5);
    assert_eq!(mid, 0.5);
    assert_eq!(hi, 0.5);
}

#[test]
fn eat_efficiency_clamps_negative_score() {
    let v = eat_efficiency(FoodKind::Plant, -1.0);
    assert_eq!(v, 1.0);
}

#[test]
fn eat_efficiency_clamps_score_above_one() {
    let v = eat_efficiency(FoodKind::Plant, 5.0);
    assert_eq!(v, 0.0);
}

#[test]
fn food_random_z_zero_for_flat_world() {
    let mut rng = StdRng::seed_from_u64(10);
    let flat = [100.0, 100.0, 0.0];
    for _ in 0..20 {
        let f = Food::random(&mut rng, flat);
        assert_eq!(f.position[2], 0.0);
        assert_eq!(f.kind, FoodKind::Plant);
        assert_eq!(f.age_ticks, 0);
    }
}

#[test]
fn food_random_z_in_range_for_3d_world() {
    let mut rng = StdRng::seed_from_u64(11);
    for _ in 0..100 {
        let f = Food::random(&mut rng, WORLD_HALF);
        assert!(f.position[2] >= -WORLD_HALF[2] && f.position[2] < WORLD_HALF[2]);
        assert!(f.position[0] >= -WORLD_HALF[0] && f.position[0] < WORLD_HALF[0]);
    }
}

#[test]
fn food_apply_gravity_noop_in_flat_world() {
    let mut food = Food {
        position: [0.0, 0.0, 0.0],
        age_ticks: 0,
        kind: FoodKind::Plant,
    };
    food.apply_gravity(1.0, 0.0);
    assert_eq!(food.position[2], 0.0);
}

#[test]
fn food_apply_gravity_clamped_to_floor() {
    let mut food = Food {
        position: [0.0, 0.0, -90.0],
        age_ticks: 0,
        kind: FoodKind::Carrion,
    };
    food.apply_gravity(100.0, 100.0);
    assert!((food.position[2] - (-100.0)).abs() < 1e-3);
}

#[test]
fn food_apply_gravity_subtracts_sink_rate() {
    let mut food = Food {
        position: [0.0, 0.0, 50.0],
        age_ticks: 0,
        kind: FoodKind::Plant,
    };
    let dt = 0.5;
    food.apply_gravity(dt, 100.0);
    assert!((food.position[2] - (50.0 - FOOD_SINK_RATE * dt)).abs() < 1e-3);
}

#[test]
fn food_value_factor_full_at_age_zero() {
    let food = Food {
        position: [0.0; 3],
        age_ticks: 0,
        kind: FoodKind::Plant,
    };
    assert!((food.value_factor() - 1.0).abs() < 1e-6);
}

#[test]
fn food_value_factor_clamped_to_zero_at_high_age() {
    let food = Food {
        position: [0.0; 3],
        age_ticks: u32::MAX,
        kind: FoodKind::Plant,
    };
    assert!(food.value_factor() >= 0.0);
}

#[test]
fn food_age_step_increments_age() {
    let mut food = Food {
        position: [0.0; 3],
        age_ticks: 5,
        kind: FoodKind::Plant,
    };
    let alive = food.age_step();
    assert!(alive);
    assert_eq!(food.age_ticks, 6);
}

#[test]
fn food_age_step_returns_false_when_value_zero() {
    let mut food = Food {
        position: [0.0; 3],
        age_ticks: 0,
        kind: FoodKind::Plant,
    };
    let needed_secs = 1.0 / CARRION_DECAY_PER_SEC + 10.0;
    food.age_ticks = (needed_secs * FIXED_TIMESTEP_HZ) as u32;
    let alive = food.age_step();
    assert!(!alive);
}

#[test]
fn coop_food_register_arrival_returns_true_first_time() {
    let mut coop = CoopFood::new([0.0; 3], 0);
    let added = register_coop_arrival(&mut coop, 1);
    assert!(added);
    assert_eq!(coop.arrivals.len(), 1);
}

#[test]
fn coop_food_register_arrival_dedupes() {
    let mut coop = CoopFood::new([0.0; 3], 0);
    register_coop_arrival(&mut coop, 5);
    let again = register_coop_arrival(&mut coop, 5);
    assert!(!again);
    assert_eq!(coop.arrivals.len(), 1);
}

#[test]
fn coop_food_is_expired_at_window_boundary() {
    let coop = CoopFood::new([0.0; 3], 0);
    assert!(!coop.is_expired(COOP_FOOD_TIME_WINDOW_TICKS as u64 - 1));
    assert!(coop.is_expired(COOP_FOOD_TIME_WINDOW_TICKS as u64));
}

#[test]
fn coop_food_try_trigger_skips_when_already_triggered() {
    let mut coop = CoopFood::new([0.0; 3], 0);
    let mut cells = vec![
        {
            let mut c = base_cell();
            c.cell_id = 1;
            c.energy = 100.0;
            c
        },
        {
            let mut c = base_cell();
            c.cell_id = 2;
            c.energy = 100.0;
            c
        },
        {
            let mut c = base_cell();
            c.cell_id = 3;
            c.energy = 100.0;
            c
        },
    ];
    for i in 1..=3u64 {
        register_coop_arrival(&mut coop, i);
    }
    let first = try_trigger_coop(&mut coop, &mut cells);
    assert!(first);
    let energies_after_first: Vec<f32> = cells.iter().map(|c| c.energy).collect();
    let second = try_trigger_coop(&mut coop, &mut cells);
    assert!(!second);
    let energies_after_second: Vec<f32> = cells.iter().map(|c| c.energy).collect();
    assert_eq!(energies_after_first, energies_after_second);
}

#[test]
fn coop_food_register_arrivals_for_all_skips_triggered() {
    let mut coops = vec![{
        let mut c = CoopFood::new([0.0; 3], 0);
        c.triggered = true;
        c
    }];
    let mut cell = base_cell();
    cell.position = [0.0; 3];
    cell.cell_id = 7;
    register_coop_arrivals_for_all(&mut coops, &[cell], WORLD_HALF);
    assert!(coops[0].arrivals.is_empty());
}

#[test]
fn coop_food_register_arrivals_for_all_picks_in_radius_only() {
    let mut coops = vec![CoopFood::new([0.0, 0.0, 0.0], 0)];
    let mut a = base_cell();
    a.position = [0.0, 0.0, 0.0];
    a.cell_id = 1;
    let mut b = base_cell();
    b.position = [
        COOP_FOOD_ARRIVAL_RADIUS + 50.0,
        0.0,
        0.0,
    ];
    b.cell_id = 2;
    register_coop_arrivals_for_all(&mut coops, &[a, b], WORLD_HALF);
    assert_eq!(coops[0].arrivals, vec![1]);
}

#[test]
fn random_coop_position_within_world_bounds() {
    let mut rng = StdRng::seed_from_u64(20);
    for _ in 0..50 {
        let p = random_coop_position(&mut rng, WORLD_HALF);
        assert!(p[0] >= -WORLD_HALF[0] && p[0] < WORLD_HALF[0]);
        assert!(p[1] >= -WORLD_HALF[1] && p[1] < WORLD_HALF[1]);
        assert!(p[2] >= -WORLD_HALF[2] && p[2] < WORLD_HALF[2]);
    }
}

#[test]
fn random_coop_position_z_zero_for_flat_world() {
    let mut rng = StdRng::seed_from_u64(21);
    let flat = [100.0, 100.0, 0.0];
    for _ in 0..20 {
        let p = random_coop_position(&mut rng, flat);
        assert_eq!(p[2], 0.0);
    }
}

// ─── chemistry.rs ───────────────────────────────────────────────────────────

#[test]
fn smell_field_new_zero_initialized() {
    let f = SmellField::new([4, 4, 2], [10.0, 10.0, 5.0]);
    for &v in f.grid_ref() {
        assert_eq!(v, 0.0);
    }
}

#[test]
fn smell_field_add_source_ignores_out_of_bounds_z() {
    let mut f = SmellField::new([4, 4, 2], [10.0, 10.0, 5.0]);
    f.add_source([0.0, 0.0, 100.0], 5.0);
    for &v in f.grid_ref() {
        assert_eq!(v, 0.0);
    }
}

#[test]
fn smell_field_add_source_xy_wraps() {
    let mut f = SmellField::new([4, 4, 2], [10.0, 10.0, 5.0]);
    f.add_source([200.0, 0.0, 0.0], 1.0);
    let total: f32 = f.grid_ref().iter().sum();
    assert!((total - 1.0).abs() < 1e-6);
}

#[test]
fn smell_field_step_decay_drives_to_zero_with_full_decay() {
    let mut f = SmellField::new([4, 4, 2], [10.0, 10.0, 5.0]);
    f.add_source([0.0, 0.0, 0.0], 100.0);
    f.step(0.0, 1.0, 1.0);
    let total: f32 = f.grid_ref().iter().sum();
    assert!(total.abs() < 1e-6);
}

#[test]
fn smell_field_step_diffusion_spreads_concentration() {
    let mut f = SmellField::new([8, 8, 4], [40.0, 40.0, 20.0]);
    f.add_source([0.0, 0.0, 0.0], 10.0);
    let center_before = f.sample([0.0, 0.0, 0.0]);
    for _ in 0..5 {
        f.step(0.15, 0.0, 1.0);
    }
    let center_after = f.sample([0.0, 0.0, 0.0]);
    assert!(center_after < center_before);
}

#[test]
fn smell_field_sample_out_of_z_range_returns_zero() {
    let mut f = SmellField::new([4, 4, 2], [10.0, 10.0, 5.0]);
    f.add_source([0.0, 0.0, 0.0], 5.0);
    let v = f.sample([0.0, 0.0, 1000.0]);
    assert_eq!(v, 0.0);
}

#[test]
fn smell_field_gradient_at_uniform_field_is_zero() {
    let mut f = SmellField::new([4, 4, 2], [10.0, 10.0, 5.0]);
    let n = 4 * 4 * 2;
    let uniform = vec![3.0; n];
    f.replace_grid_from(&uniform);
    let g = f.gradient_at([0.0, 0.0, 0.0], 1.0);
    assert!(g[0].abs() < 1e-6 && g[1].abs() < 1e-6);
}

#[test]
fn smell_field_replace_grid_overrides_state() {
    let mut f = SmellField::new([2, 2, 2], [10.0, 10.0, 5.0]);
    let new_grid = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    f.replace_grid_from(&new_grid);
    assert_eq!(f.grid_ref(), new_grid.as_slice());
}

#[test]
fn smell_field_step_decay_clamped_at_zero() {
    let mut f = SmellField::new([4, 4, 2], [10.0, 10.0, 5.0]);
    f.add_source([0.0, 0.0, 0.0], 1.0);
    f.step(0.0, 100.0, 1.0);
    let total: f32 = f.grid_ref().iter().sum();
    assert!(total >= 0.0);
}

// ─── sensors.rs (FOV / fov_cone_accept edge cases) ──────────────────────────

#[test]
fn fov_cone_accept_self_overlap_always_true() {
    let fwd = forward_vector(0.0, 0.0);
    let cos_thresh = 0.9;
    assert!(fov_cone_accept([0.0, 0.0, 0.0], 0.0, fwd, cos_thresh));
    assert!(fov_cone_accept([1e-10, 0.0, 0.0], 1e-20, fwd, cos_thresh));
}

#[test]
fn fov_cone_accept_full_sphere_threshold_accepts_back() {
    let fwd = forward_vector(0.0, 0.0);
    let back_delta = [-10.0, 0.0, 0.0];
    let d2 = 100.0;
    let accepted = fov_cone_accept(back_delta, d2, fwd, -1.0);
    assert!(accepted);
}

#[test]
fn fov_cone_accept_back_hemisphere_rejected_with_zero_threshold() {
    let fwd = forward_vector(0.0, 0.0);
    let back = [-10.0, 0.0, 0.0];
    assert!(!fov_cone_accept(back, 100.0, fwd, 0.0));
}

#[test]
fn fov_cone_accept_lateral_at_zero_threshold_borderline() {
    let fwd = forward_vector(0.0, 0.0);
    let lateral = [0.0, 10.0, 0.0];
    let d2 = 100.0;
    let _ = fov_cone_accept(lateral, d2, fwd, 0.0);
}

#[test]
fn fov_cone_accept_pitched_forward_in_3d() {
    let fwd = forward_vector(0.0, FRAC_PI_4);
    let target_up_forward = [10.0, 0.0, 10.0];
    let d2 = 200.0;
    let cos_thresh = (FRAC_PI_4 + 0.01).cos();
    assert!(fov_cone_accept(target_up_forward, d2, fwd, cos_thresh));
    let target_down = [10.0, 0.0, -10.0];
    let cos_thresh_narrow = (FRAC_PI_4 - 0.1).cos();
    assert!(!fov_cone_accept(target_down, d2, fwd, cos_thresh_narrow));
}

#[test]
fn populate_brain_inputs_resets_damage_accum() {
    let mut cell = base_cell();
    cell.damage_accum = 5.0;
    let sensors = BrainSensors {
        nearest_food: None,
        nearest_cell: None,
        neighbors_in_vision: 0,
        smell_grad: [0.0; 3],
        pheromone_grads: [[0.0; 3]; N_PHEROMONE_CHANNELS],
        temperature_local: THERMAL_REF_TEMP,
    };
    let _ = populate_brain_inputs(&mut cell, &sensors, 50.0);
    assert_eq!(cell.damage_accum, 0.0);
}

#[test]
fn populate_brain_inputs_writes_food_and_cell_deltas() {
    let mut cell = base_cell();
    let sensors = BrainSensors {
        nearest_food: Some([10.0, 20.0, 30.0]),
        nearest_cell: Some(([5.0, -5.0, 15.0], 2.0)),
        neighbors_in_vision: 4,
        smell_grad: [0.0; 3],
        pheromone_grads: [[0.0; 3]; N_PHEROMONE_CHANNELS],
        temperature_local: THERMAL_REF_TEMP,
    };
    let inputs = populate_brain_inputs(&mut cell, &sensors, 100.0);
    assert!((inputs[0] - 0.1).abs() < 1e-4);
    assert!((inputs[1] - 0.2).abs() < 1e-4);
    assert!((inputs[15] - 0.3).abs() < 1e-4);
    assert!((inputs[2] - 0.05).abs() < 1e-4);
    assert!((inputs[16] - 0.15).abs() < 1e-4);
}

#[test]
fn populate_brain_inputs_writes_multichannel_pheromone_gradients() {
    let mut cell = base_cell();
    let mut grads = [[0.0; 3]; N_PHEROMONE_CHANNELS];
    grads[1] = [10.0, 0.0, 0.0];
    grads[2] = [0.0, 10.0, 0.0];
    let sensors = BrainSensors {
        nearest_food: None,
        nearest_cell: None,
        neighbors_in_vision: 0,
        smell_grad: [0.0; 3],
        pheromone_grads: grads,
        temperature_local: THERMAL_REF_TEMP,
    };
    let inputs = populate_brain_inputs(&mut cell, &sensors, 50.0);
    assert!(inputs[21].abs() > 0.0);
    assert!(inputs[25].abs() > 0.0);
    assert_eq!(inputs[24], 0.0);
}

#[test]
fn pool_bonded_hidden_returns_self_for_no_partners() {
    let cell = base_cell();
    let pooled = pool_bonded_hidden(&cell, |_| None);
    assert_eq!(pooled, cell.last_hidden);
}

#[test]
fn pool_bonded_sensors_partner_with_smaller_signal_does_not_replace() {
    let mut cell = base_cell();
    cell.bonds[0] = Some(Bond {
        other_cell_id: 99,
        rest_length: 10.0,
        stiffness: BOND_STIFFNESS,
        damping: BOND_DAMPING,
        age_ticks: 0,
    });
    let mut own = [0.0_f32; BRAIN_INPUTS];
    own[0] = 0.8;
    let mut partner = [0.0_f32; BRAIN_INPUTS];
    partner[0] = 0.3;
    let pooled = pool_bonded_sensors(&cell, &own, |id| if id == 99 { Some(partner) } else { None });
    assert_eq!(pooled[0], 0.8);
}

// ─── physics_utils.rs ───────────────────────────────────────────────────────

#[test]
fn forward_vector_is_unit_length() {
    let cases = [
        (0.0, 0.0),
        (FRAC_PI_2, 0.0),
        (PI, FRAC_PI_4),
        (-FRAC_PI_2, -FRAC_PI_4),
    ];
    for (yaw, pitch) in cases {
        let v = forward_vector(yaw, pitch);
        let mag = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        assert!((mag - 1.0).abs() < 1e-5, "yaw={} pitch={} mag={}", yaw, pitch, mag);
    }
}

#[test]
fn spike_direction_zero_offsets_matches_forward_vector() {
    let yaw = 0.7;
    let pitch = 0.2;
    let spike = Spike::ZERO;
    let v_spike = spike_direction(yaw, pitch, &spike);
    let v_fwd = forward_vector(yaw, pitch);
    for i in 0..3 {
        assert!((v_spike[i] - v_fwd[i]).abs() < 1e-6);
    }
}

#[test]
fn spike_direction_applies_azimuth_offset() {
    let spike = Spike {
        length: 1.0,
        azimuth_offset: FRAC_PI_2,
        elevation_offset: 0.0,
        complexity: 0.0,
    };
    let v = spike_direction(0.0, 0.0, &spike);
    assert!(v[0].abs() < 1e-5);
    assert!((v[1] - 1.0).abs() < 1e-5);
}

#[test]
fn spike_complexity_attack_factor_clamped() {
    let neg = spike_complexity_attack_factor(-1.0);
    let high = spike_complexity_attack_factor(2.0);
    assert!((neg - 1.0).abs() < 1e-6);
    assert!((high - (1.0 + COMPLEXITY_ATTACK_GAIN)).abs() < 1e-6);
}

#[test]
fn spike_complexity_cost_factor_quadratic() {
    let half = spike_complexity_cost_factor(0.5);
    let full = spike_complexity_cost_factor(1.0);
    let expected_half = 1.0 + COMPLEXITY_COST_GAIN * 0.25;
    assert!((half - expected_half).abs() < 1e-6);
    assert!((full - (1.0 + COMPLEXITY_COST_GAIN)).abs() < 1e-6);
}

#[test]
fn spike_complexity_grab_factor_endpoints() {
    let zero = spike_complexity_grab_factor(0.0);
    let one = spike_complexity_grab_factor(1.0);
    assert!((zero - 1.0).abs() < 1e-6);
    assert!((one - (1.0 + COMPLEXITY_GRAB_GAIN)).abs() < 1e-6);
}

#[test]
fn vision_fov_factor_clamped_to_max() {
    let v = vision_fov_factor(MAX_VISION_FOV + 1.0);
    let expected = (1.0 - MAX_VISION_FOV.cos()) * 0.5;
    assert!((v - expected).abs() < 1e-6);
}

#[test]
fn vision_fov_factor_negative_input_clamped_to_zero() {
    let v = vision_fov_factor(-1.0);
    assert!(v.abs() < 1e-6);
}

#[test]
fn temperature_at_z_flat_world_returns_ref_temp() {
    let flat = [100.0, 100.0, 0.0];
    let t = temperature_at_z(50.0, flat, 0, 0);
    assert!((t - THERMAL_REF_TEMP).abs() < 1e-6);
}

#[test]
fn temperature_at_z_clamps_above_top() {
    let t = temperature_at_z(WORLD_HALF[2] * 2.0, WORLD_HALF, 0, 0);
    assert!(t <= THERMAL_TOP + 1e-3);
}

#[test]
fn temperature_at_z_clamps_below_bottom() {
    let t = temperature_at_z(-WORLD_HALF[2] * 2.0, WORLD_HALF, 0, 0);
    assert!(t >= THERMAL_BOTTOM - 1e-3);
}

#[test]
fn metabolism_factor_at_ref_is_unity() {
    assert!((metabolism_factor(THERMAL_REF_TEMP) - 1.0).abs() < 1e-6);
}

#[test]
fn metabolism_factor_extreme_hot_above_q10_squared() {
    let t = THERMAL_REF_TEMP + 20.0;
    let m = metabolism_factor(t);
    assert!((m - THERMAL_Q10 * THERMAL_Q10).abs() < 1e-3);
}

#[test]
fn metabolism_factor_extreme_cold_below_unit_inverse() {
    let t = THERMAL_REF_TEMP - 30.0;
    let m = metabolism_factor(t);
    assert!(m > 0.0);
    assert!(m < 1.0 / (THERMAL_Q10 * THERMAL_Q10) + 1e-3);
}

#[test]
fn body_basis_pitch_up_yaw_zero_yields_correct_axes() {
    let (fwd, right, up) = body_basis(0.0, FRAC_PI_4);
    let mag_fwd = (fwd[0] * fwd[0] + fwd[1] * fwd[1] + fwd[2] * fwd[2]).sqrt();
    let mag_right = (right[0] * right[0] + right[1] * right[1] + right[2] * right[2]).sqrt();
    let mag_up = (up[0] * up[0] + up[1] * up[1] + up[2] * up[2]).sqrt();
    assert!((mag_fwd - 1.0).abs() < 1e-5);
    assert!((mag_right - 1.0).abs() < 1e-5);
    assert!((mag_up - 1.0).abs() < 1e-5);
    let dot_fr = fwd[0] * right[0] + fwd[1] * right[1] + fwd[2] * right[2];
    let dot_fu = fwd[0] * up[0] + fwd[1] * up[1] + fwd[2] * up[2];
    let dot_ru = right[0] * up[0] + right[1] * up[1] + right[2] * up[2];
    assert!(dot_fr.abs() < 1e-5);
    assert!(dot_fu.abs() < 1e-5);
    assert!(dot_ru.abs() < 1e-5);
}

#[test]
fn eat_test_pose_extreme_aspect_ratio_picks_long_axis() {
    let cell_pos = [0.0, 0.0, 0.0];
    let body_dims = [10.0, 0.5, 0.5];
    let along = eat_test_pose(cell_pos, 0.0, 0.0, body_dims, [9.0, 0.0, 0.0], 1.0);
    let across = eat_test_pose(cell_pos, 0.0, 0.0, body_dims, [0.0, 9.0, 0.0], 1.0);
    assert!(along);
    assert!(!across);
}

#[test]
fn eat_test_pose_eat_factor_scales_zone() {
    let cell_pos = [0.0, 0.0, 0.0];
    let body_dims = [1.0, 1.0, 1.0];
    let target = [1.5, 0.0, 0.0];
    assert!(!eat_test_pose(cell_pos, 0.0, 0.0, body_dims, target, 1.0));
    assert!(eat_test_pose(cell_pos, 0.0, 0.0, body_dims, target, 2.0));
}

#[test]
fn eat_test_pose_zero_eat_factor_falls_back_to_epsilon() {
    let cell_pos = [0.0, 0.0, 0.0];
    let body_dims = [1.0, 1.0, 1.0];
    let target = [0.0, 0.0, 0.0];
    assert!(eat_test_pose(cell_pos, 0.0, 0.0, body_dims, target, 0.0));
}

// ─── physics_utils.rs — anisotropic drag extreme aspect ratio ───────────────

#[test]
fn anisotropic_drag_extreme_aspect_ratio_pulls_sideways_harder() {
    let physics = no_drag_physics(0.0, 0.0);
    let physics = PhysicsConfig {
        drag: 0.01,
        ..physics
    };
    let make_cell = |vel: [f32; 3], length: f32, width: f32| {
        let mut c = base_cell();
        c.phenotype = Phenotype {
            body_length: length,
            body_width: width,
            body_height: 1.0,
            spikes: [Spike::ZERO; SPIKE_SLOTS],
            spike_count: 1,
            shell_thickness: 0.0,
        };
        c.velocity = vel;
        c.heading = 0.0;
        c
    };
    let mut needle_forward = make_cell([10.0, 0.0, 0.0], 8.0, 0.3);
    let mut needle_sideways = make_cell([0.0, 10.0, 0.0], 8.0, 0.3);
    needle_forward.step(1.0, [1000.0, 1000.0, 0.0], 0, 0, &physics);
    needle_sideways.step(1.0, [1000.0, 1000.0, 0.0], 0, 0, &physics);
    let v_forward = needle_forward.velocity[0].hypot(needle_forward.velocity[1]);
    let v_sideways = needle_sideways.velocity[0].hypot(needle_sideways.velocity[1]);
    assert!(
        v_forward > v_sideways + 1.0,
        "extreme aspect: forward {} should be much faster than sideways {}",
        v_forward,
        v_sideways
    );
}

#[test]
fn anisotropic_drag_pancake_horizontal_motion_low_drag() {
    let physics = PhysicsConfig {
        drag: 0.01,
        ..no_drag_physics(0.0, 0.0)
    };
    let make_cell = || {
        let mut c = base_cell();
        c.phenotype = Phenotype {
            body_length: 4.0,
            body_width: 4.0,
            body_height: 0.3,
            spikes: [Spike::ZERO; SPIKE_SLOTS],
            spike_count: 1,
            shell_thickness: 0.0,
        };
        c.velocity = [10.0, 0.0, 0.0];
        c.heading = 0.0;
        c
    };
    let mut cell = make_cell();
    let v0 = cell.velocity[0];
    cell.step(1.0, [1000.0, 1000.0, 0.0], 0, 0, &physics);
    let v1 = cell.velocity[0];
    assert!(v1 < v0);
    assert!(v1 > 0.0);
}
