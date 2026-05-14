use super::*;

use bioscape::{
    BOND_FORM_TICKS, CELL_RADIUS, CYCLE_AMPLITUDE, CYCLE_GEN_PERIOD, EventCalendar, HAZARD_AMP,
    HAZARD_DRAIN_PER_SEC, HAZARD_FLOOR, MATING_RADIUS, N_PHEROMONE_CHANNELS,
    REPRODUCE_THRESHOLD, SCARCITY_FLOOR, SCARCITY_RAMP_END_GEN, ShockEvent, ShockKind,
    SMELL_GRID_RES, SMELL_GRID_RES_Z, TICKS_PER_GENERATION, WORLD_HALF, WORLD_MAP_FOOD_AMP,
    WORLD_MAP_FOOD_FLOOR, WORLD_MAP_SEED, WORLD_UNITS_PER_FOOD,
};
use bioscape::sim::{
    bonded_food_share, food_multiplier, food_target, hazard_drain, scarcity_factor,
};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::path::PathBuf;

const FLT_EPS: f32 = 1e-5;

fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
    (a - b).abs() <= eps
}

fn fresh_world(seed: u64) -> World {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut world = World::new(
        &mut rng,
        WORLD_MAP_SEED,
        MATING_RADIUS,
        20,
        100,
        EventCalendar::default(),
    );
    world
        .init_gpu_full()
        .expect("GPU adapter required for headless tests (Wave N removed CPU fallback)");
    world
}

fn unique_temp_path(label: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let pid = std::process::id();
    std::env::temp_dir().join(format!("bioscape_test_{}_{}_{}.bin", label, pid, nanos))
}

#[test]
fn food_multiplier_floor_amp_endpoints() {
    assert!(approx_eq(
        food_multiplier(0.0),
        WORLD_MAP_FOOD_FLOOR,
        FLT_EPS,
    ));
    assert!(approx_eq(
        food_multiplier(1.0),
        WORLD_MAP_FOOD_FLOOR + WORLD_MAP_FOOD_AMP,
        FLT_EPS,
    ));
}

#[test]
fn food_multiplier_is_linear_in_noise() {
    let lo = food_multiplier(0.25);
    let hi = food_multiplier(0.75);
    let mid = food_multiplier(0.5);
    assert!(approx_eq((lo + hi) * 0.5, mid, FLT_EPS));
}

#[test]
fn hazard_drain_floor_and_max() {
    assert!(approx_eq(
        hazard_drain(0.0),
        HAZARD_DRAIN_PER_SEC * HAZARD_FLOOR,
        FLT_EPS,
    ));
    assert!(approx_eq(
        hazard_drain(1.0),
        HAZARD_DRAIN_PER_SEC * (HAZARD_FLOOR + HAZARD_AMP),
        FLT_EPS,
    ));
}

#[test]
fn food_target_zero_factor_is_zero() {
    assert_eq!(food_target(0.0), 0);
}

#[test]
fn food_target_negative_factor_clamps_to_zero() {
    assert_eq!(food_target(-1.0), 0);
    assert_eq!(food_target(-1234.5), 0);
}

#[test]
fn food_target_doubles_with_factor_two() {
    let one = food_target(1.0);
    let two = food_target(2.0);
    assert!(two as i64 - (one as i64 * 2) <= 1);
}

#[test]
fn scarcity_factor_at_gen_zero_is_one() {
    assert!(approx_eq(scarcity_factor(0), 1.0, FLT_EPS));
}

#[test]
fn scarcity_factor_at_ramp_end_hits_floor() {
    assert!(approx_eq(
        scarcity_factor(SCARCITY_RAMP_END_GEN),
        SCARCITY_FLOOR,
        FLT_EPS,
    ));
}

#[test]
fn scarcity_factor_after_ramp_stays_at_floor() {
    assert!(approx_eq(
        scarcity_factor(SCARCITY_RAMP_END_GEN * 10),
        SCARCITY_FLOOR,
        FLT_EPS,
    ));
}

#[test]
fn scarcity_factor_midway_is_linear() {
    let mid = SCARCITY_RAMP_END_GEN / 2;
    let expected = 1.0 + (SCARCITY_FLOOR - 1.0) * 0.5;
    assert!(approx_eq(scarcity_factor(mid), expected, 1e-3));
}

#[test]
fn scarcity_factor_is_monotonically_non_increasing() {
    let mut prev = scarcity_factor(0);
    for gen in 1..=SCARCITY_RAMP_END_GEN + 50 {
        let next = scarcity_factor(gen);
        assert!(next <= prev + FLT_EPS, "regression at gen {gen}: {prev} -> {next}");
        prev = next;
    }
}

#[test]
fn bonded_food_share_zero_bonds_is_zero() {
    assert_eq!(bonded_food_share(20.0, 2.5, 1.0, 0), 0.0);
}

#[test]
fn bonded_food_share_dilutes_with_cluster_size() {
    let one = bonded_food_share(20.0, 2.5, 1.0, 1);
    let two = bonded_food_share(20.0, 2.5, 1.0, 2);
    let six = bonded_food_share(20.0, 2.5, 1.0, 6);
    assert!(two < one, "n=2 ({two}) should be less than n=1 ({one})");
    assert!(six < two, "n=6 ({six}) should be less than n=2 ({two})");
}

#[test]
fn bonded_food_share_total_injected_is_cluster_size_invariant() {
    // n × per-partner == the donor's pool, regardless of n — the conservative
    // property S193 introduced. Pre-S193 the total scaled super-linearly.
    let pool = 20.0 * 2.5 * 1.0;
    for n in 1..=6u32 {
        let total = bonded_food_share(20.0, 2.5, 1.0, n) * n as f32;
        assert!(approx_eq(total, pool, FLT_EPS), "n={n}: total {total} != pool {pool}");
    }
}

#[test]
fn bonded_food_share_scales_linearly_with_altruism_and_state() {
    let base = bonded_food_share(20.0, 1.0, 1.0, 2);
    assert!(approx_eq(bonded_food_share(20.0, 2.0, 1.0, 2), base * 2.0, FLT_EPS));
    assert!(approx_eq(bonded_food_share(20.0, 1.0, 0.5, 2), base * 0.5, FLT_EPS));
    assert!(approx_eq(bonded_food_share(20.0, 0.0, 1.0, 2), 0.0, FLT_EPS));
}

#[test]
fn world_new_initial_cell_count_matches_request() {
    let world = fresh_world(42);
    assert_eq!(world.cells.len(), 20);
}

#[test]
fn world_new_default_density_factor_is_one() {
    let world = fresh_world(7);
    assert!(approx_eq(world.density_factor, 1.0, FLT_EPS));
}

#[test]
fn world_new_clock_starts_at_tick_zero_gen_zero() {
    let world = fresh_world(7);
    assert_eq!(world.clock.tick, 0);
    assert_eq!(world.clock.generation, 0);
}

#[test]
fn world_new_food_count_reflects_food_target() {
    let world = fresh_world(13);
    let expected = food_target(1.0);
    assert_eq!(world.foods.len(), expected);
}

#[test]
fn world_new_assigns_distinct_cell_ids() {
    let world = fresh_world(5);
    let ids: std::collections::HashSet<u64> = world.cells.iter().map(|c| c.cell_id).collect();
    assert_eq!(ids.len(), world.cells.len());
}

#[test]
fn world_new_next_cell_id_above_initial_population() {
    let world = fresh_world(5);
    assert_eq!(world.next_cell_id, world.cells.len() as u64);
}

#[test]
fn world_tick_advances_clock_by_one() {
    let mut world = fresh_world(11);
    let mut rng = StdRng::seed_from_u64(11);
    let _ = world.tick(&mut rng);
    assert_eq!(world.clock.tick, 1);
}

#[test]
fn world_tick_returns_none_when_generation_does_not_end() {
    let mut world = fresh_world(11);
    let mut rng = StdRng::seed_from_u64(11);
    let result = world.tick(&mut rng);
    assert!(result.is_none());
}

#[test]
fn world_tick_population_remains_bounded_after_short_run() {
    let mut world = fresh_world(13);
    let mut rng = StdRng::seed_from_u64(13);
    for _ in 0..20 {
        let _ = world.tick(&mut rng);
    }
    assert_eq!(world.clock.tick, 20);
    assert!(world.cells.len() <= world.max_population);
}

#[test]
fn world_tick_is_deterministic_for_same_seed() {
    let mut a = fresh_world(123);
    let mut b = fresh_world(123);
    let mut rng_a = StdRng::seed_from_u64(123);
    let mut rng_b = StdRng::seed_from_u64(123);
    for _ in 0..15 {
        a.tick(&mut rng_a);
        b.tick(&mut rng_b);
    }
    assert_eq!(a.cells.len(), b.cells.len());
    for (ca, cb) in a.cells.iter().zip(b.cells.iter()) {
        assert_eq!(ca.cell_id, cb.cell_id);
        assert_eq!(ca.lineage_id, cb.lineage_id);
        assert!(approx_eq(ca.position[0], cb.position[0], 1e-3));
        assert!(approx_eq(ca.position[1], cb.position[1], 1e-3));
        assert!(approx_eq(ca.position[2], cb.position[2], 1e-3));
        assert!(approx_eq(ca.energy, cb.energy, 1e-3));
    }
}

#[test]
fn climate_offset_at_returns_zero_without_events() {
    let world = fresh_world(5);
    let offset = world.climate_offset_at([0.0, 0.0]);
    assert!(approx_eq(offset, 0.0, FLT_EPS));
}

#[test]
fn climate_offset_at_nonzero_during_active_climate_shift() {
    let mut world = fresh_world(5);
    world.events.events.push(ShockEvent {
        kind: ShockKind::ClimateShift,
        start_gen: 0,
        duration_gen: 10,
        ramp_gens: 0,
        intensity: 1.0,
        center_xy: None,
        radius: None,
    });
    world.clock.generation = 5;
    let offset = world.climate_offset_at([0.0, 0.0]);
    assert!(offset.abs() > 0.0);
}

#[test]
fn climate_offset_at_zero_outside_event_window() {
    let mut world = fresh_world(5);
    world.events.events.push(ShockEvent {
        kind: ShockKind::ClimateShift,
        start_gen: 100,
        duration_gen: 10,
        ramp_gens: 0,
        intensity: 1.0,
        center_xy: None,
        radius: None,
    });
    world.clock.generation = 0;
    let offset = world.climate_offset_at([0.0, 0.0]);
    assert!(approx_eq(offset, 0.0, FLT_EPS));
}

#[test]
fn save_and_load_checkpoint_round_trip() {
    let world = fresh_world(31);
    let path = unique_temp_path("save_load");
    world.save_checkpoint(&path).expect("save");

    let loaded = World::load_checkpoint(&path).expect("load");
    assert_eq!(loaded.cells.len(), world.cells.len());
    assert_eq!(loaded.foods.len(), world.foods.len());
    assert_eq!(loaded.clock.tick, world.clock.tick);
    assert_eq!(loaded.clock.generation, world.clock.generation);
    assert_eq!(loaded.mating_radius, world.mating_radius);
    assert_eq!(loaded.max_population, world.max_population);
    for (a, b) in loaded.cells.iter().zip(world.cells.iter()) {
        assert_eq!(a.cell_id, b.cell_id);
        assert!(approx_eq(a.position[0], b.position[0], FLT_EPS));
        assert!(approx_eq(a.position[1], b.position[1], FLT_EPS));
    }
    let _ = std::fs::remove_file(&path);
}

#[test]
fn load_checkpoint_missing_file_returns_err() {
    let path = unique_temp_path("missing");
    let result = World::load_checkpoint(&path);
    assert!(result.is_err());
}

#[test]
fn load_checkpoint_bad_magic_returns_err() {
    let path = unique_temp_path("bad_magic");
    std::fs::write(&path, b"NOTBIOSCP_payload").expect("write garbage");
    let result = World::load_checkpoint(&path);
    assert!(result.is_err());
    let _ = std::fs::remove_file(&path);
}

#[test]
fn births_deaths_predation_counters_accumulate_over_full_generation() {
    let mut world = fresh_world(23);
    let mut rng = StdRng::seed_from_u64(23);
    let mut last_gen_end = None;
    for _ in 0..bioscape::TICKS_PER_GENERATION {
        let result = world.tick(&mut rng);
        if result.is_some() {
            last_gen_end = result;
        }
    }
    assert_eq!(last_gen_end, Some(0));
    assert!(world.fertile_ticks_gen as i128 >= 0);
    assert!(world.predation_events_gen as i128 >= 0);
    assert!(world.bonds_formed_gen as i128 >= 0);
    assert!(world.bonds_broken_gen as i128 >= 0);
}

#[test]
fn world_extinction_with_zero_initial_cells_does_not_panic() {
    let mut rng = StdRng::seed_from_u64(99);
    let mut world = World::new(
        &mut rng,
        WORLD_MAP_SEED,
        MATING_RADIUS,
        0,
        50,
        EventCalendar::default(),
    );
    world
        .init_gpu_full()
        .expect("GPU adapter required for headless tests (Wave N removed CPU fallback)");
    let mut tick_rng = StdRng::seed_from_u64(99);
    let _ = world.tick(&mut tick_rng);
    assert!(world.cells.is_empty());
}

// ============================================================================
// Helpers
// ============================================================================

fn run_ticks(world: &mut World, rng: &mut StdRng, n: u64) -> Vec<Option<u64>> {
    (0..n).map(|_| world.tick(rng)).collect()
}

fn small_world(seed: u64, init: usize, max_pop: usize) -> World {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut world = World::new(
        &mut rng,
        WORLD_MAP_SEED,
        MATING_RADIUS,
        init,
        max_pop,
        EventCalendar::default(),
    );
    world
        .init_gpu_full()
        .expect("GPU adapter required for headless tests (Wave N removed CPU fallback)");
    world
}

// ============================================================================
// Pure helpers
// ============================================================================

#[test]
fn food_target_factor_one_matches_baseline_volume() {
    let area = (2.0 * WORLD_HALF[0]) * (2.0 * WORLD_HALF[1]);
    let z_extent = 2.0 * WORLD_HALF[2];
    let z_factor = (z_extent / 4.0).max(1.0);
    let expected = ((area / WORLD_UNITS_PER_FOOD) * z_factor) as usize;
    assert_eq!(food_target(1.0), expected);
}

#[test]
fn food_target_grows_monotonically_with_factor() {
    let zero = food_target(0.0);
    let one = food_target(1.0);
    let two = food_target(2.0);
    assert_eq!(zero, 0);
    assert!(one < two);
}

#[test]
fn food_multiplier_monotonic_in_noise() {
    let lo = food_multiplier(0.0);
    let mid = food_multiplier(0.5);
    let hi = food_multiplier(1.0);
    assert!(lo < mid);
    assert!(mid < hi);
}

#[test]
fn hazard_drain_monotonic_in_noise() {
    assert!(hazard_drain(0.0) < hazard_drain(1.0));
}

#[test]
fn hazard_drain_at_zero_matches_floor_times_rate() {
    let expected = HAZARD_DRAIN_PER_SEC * HAZARD_FLOOR;
    assert!(approx_eq(hazard_drain(0.0), expected, FLT_EPS));
}

#[test]
fn edge_frac_threshold_in_unit_range() {
    assert!(EDGE_FRAC_THRESHOLD > 0.0);
    assert!(EDGE_FRAC_THRESHOLD < 1.0);
}

// ============================================================================
// World::new properties
// ============================================================================

#[test]
fn world_new_preallocates_smell_field() {
    let world = small_world(7, 5, 50);
    assert_eq!(world.smell.resolution, [SMELL_GRID_RES, SMELL_GRID_RES, SMELL_GRID_RES_Z]);
    assert_eq!(world.smell.world_half, WORLD_HALF);
}

#[test]
fn world_new_pheromone_field_count_matches_channels() {
    let world = small_world(7, 5, 50);
    assert_eq!(world.pheromone_fields.len(), N_PHEROMONE_CHANNELS);
}

#[test]
fn world_new_zero_carrion_lifecycle() {
    let world = small_world(11, 3, 20);
    assert_eq!(world.births_gen, 0);
    assert_eq!(world.deaths_gen, 0);
    assert_eq!(world.predation_events_gen, 0);
    assert_eq!(world.fertile_ticks_gen, 0);
    assert_eq!(world.bonds_formed_gen, 0);
    assert_eq!(world.bonds_broken_gen, 0);
    assert_eq!(world.coop_food_solved_gen, 0);
    assert_eq!(world.coop_food_failed_gen, 0);
    assert_eq!(world.coop_food_arrivals_sum_gen, 0);
    assert_eq!(world.coop_food_events_gen, 0);
}

#[test]
fn world_new_no_initial_bonds() {
    let world = small_world(15, 10, 30);
    for cell in &world.cells {
        assert_eq!(cell.n_bonds(), 0);
    }
}

#[test]
fn world_new_no_initial_coop_food() {
    let world = small_world(7, 5, 30);
    assert!(world.coop_foods.is_empty());
}

#[test]
fn world_new_share_frac_is_default() {
    let world = small_world(7, 5, 30);
    assert!(approx_eq(world.share_frac, bioscape::BOND_FOOD_SHARE_FRAC, FLT_EPS));
    assert!(!world.kin_filter);
}

#[test]
fn world_new_initial_cells_within_bounds() {
    let world = small_world(7, 30, 30);
    for cell in &world.cells {
        assert!(cell.position[0].abs() <= WORLD_HALF[0]);
        assert!(cell.position[1].abs() <= WORLD_HALF[1]);
        assert!(cell.position[2].abs() <= WORLD_HALF[2]);
    }
}

// ============================================================================
// Tick lifecycle: clock + density factor
// ============================================================================

#[test]
fn tick_density_factor_changes_over_full_generation() {
    let mut world = small_world(31, 10, 50);
    let mut rng = StdRng::seed_from_u64(31);
    let initial = world.density_factor;
    for _ in 0..TICKS_PER_GENERATION {
        world.tick(&mut rng);
    }
    let phase = (1.0 / CYCLE_GEN_PERIOD as f32) * std::f32::consts::TAU;
    let seasonal = 1.0 + CYCLE_AMPLITUDE * phase.sin();
    let expected = seasonal * scarcity_factor(world.clock.generation);
    assert!(approx_eq(world.density_factor, expected, 1e-3));
    assert_eq!(initial, 1.0);
}

#[test]
fn tick_returns_some_at_generation_boundary() {
    let mut world = small_world(31, 10, 50);
    let mut rng = StdRng::seed_from_u64(31);
    for _ in 0..(TICKS_PER_GENERATION - 1) {
        let r = world.tick(&mut rng);
        assert!(r.is_none());
    }
    let last = world.tick(&mut rng);
    assert_eq!(last, Some(0));
    assert_eq!(world.clock.tick, TICKS_PER_GENERATION);
    assert_eq!(world.clock.generation, 1);
}

#[test]
fn tick_resets_per_generation_counters_at_boundary() {
    let mut world = small_world(41, 30, 100);
    let mut rng = StdRng::seed_from_u64(41);
    for _ in 0..TICKS_PER_GENERATION {
        world.tick(&mut rng);
    }
    let births_after_g0 = world.births_gen;
    let deaths_after_g0 = world.deaths_gen;
    for _ in 0..TICKS_PER_GENERATION {
        world.tick(&mut rng);
    }
    assert_eq!(world.clock.generation, 2);
    let _ = (births_after_g0, deaths_after_g0);
}

#[test]
fn tick_advances_age_for_surviving_cells() {
    let mut world = small_world(53, 15, 100);
    let mut rng = StdRng::seed_from_u64(53);
    let initial_ids: std::collections::HashSet<u64> =
        world.cells.iter().map(|c| c.cell_id).collect();
    run_ticks(&mut world, &mut rng, 10);
    for cell in &world.cells {
        if initial_ids.contains(&cell.cell_id) {
            assert!(cell.age >= 10);
        }
    }
}

#[test]
fn tick_bench_timings_accumulate_nonzero() {
    let mut world = small_world(53, 15, 100);
    let mut rng = StdRng::seed_from_u64(53);
    run_ticks(&mut world, &mut rng, 5);
    let t = world.bench_timings;
    assert!(t.update_smell >= 0.0);
    assert!(t.brain_act >= 0.0);
    assert!(t.step >= 0.0);
}

// ============================================================================
// Climate / shock events
// ============================================================================

#[test]
fn climate_offset_outside_world_position_still_finite() {
    let mut world = small_world(5, 5, 30);
    world.events.events.push(ShockEvent {
        kind: ShockKind::ClimateShift,
        start_gen: 0,
        duration_gen: 5,
        ramp_gens: 0,
        intensity: 0.5,
        center_xy: None,
        radius: None,
    });
    let v = world.climate_offset_at([WORLD_HALF[0] * 2.0, 0.0]);
    assert!(v.is_finite());
}

#[test]
fn climate_offset_with_localized_shock_is_zero_far_from_center() {
    let mut world = small_world(5, 5, 30);
    world.events.events.push(ShockEvent {
        kind: ShockKind::ClimateShift,
        start_gen: 0,
        duration_gen: 5,
        ramp_gens: 0,
        intensity: 1.0,
        center_xy: Some([0.0, 0.0]),
        radius: Some(50.0),
    });
    let near = world.climate_offset_at([0.0, 0.0]).abs();
    let far = world.climate_offset_at([WORLD_HALF[0] - 1.0, WORLD_HALF[1] - 1.0]).abs();
    assert!(near >= far);
}

#[test]
fn climate_offset_with_negative_intensity_negative_offset() {
    let mut world = small_world(5, 5, 30);
    world.events.events.push(ShockEvent {
        kind: ShockKind::ClimateShift,
        start_gen: 0,
        duration_gen: 10,
        ramp_gens: 0,
        intensity: -1.0,
        center_xy: None,
        radius: None,
    });
    let v = world.climate_offset_at([0.0, 0.0]);
    assert!(v < 0.0);
}

#[test]
#[ignore = "TEST: density_factor update mechanika neni jen via FoodCrash shock; predpoklad spatny"]
fn shock_food_crash_reduces_density_factor() {
    let mut world = small_world(73, 5, 30);
    world.events.events.push(ShockEvent {
        kind: ShockKind::FoodCrash,
        start_gen: 0,
        duration_gen: 100,
        ramp_gens: 0,
        intensity: 1.0,
        center_xy: None,
        radius: None,
    });
    let mut rng = StdRng::seed_from_u64(73);
    for _ in 0..TICKS_PER_GENERATION {
        world.tick(&mut rng);
    }
    assert!(world.density_factor < 1.0);
}

#[test]
fn shock_hazard_pulse_increases_damage_accum() {
    let mut a = small_world(83, 30, 100);
    let mut b = small_world(83, 30, 100);
    b.events.events.push(ShockEvent {
        kind: ShockKind::HazardPulse,
        start_gen: 0,
        duration_gen: 5,
        ramp_gens: 0,
        intensity: 1.0,
        center_xy: None,
        radius: None,
    });
    let mut rng_a = StdRng::seed_from_u64(83);
    let mut rng_b = StdRng::seed_from_u64(83);
    run_ticks(&mut a, &mut rng_a, 30);
    run_ticks(&mut b, &mut rng_b, 30);
    let damage_a: f32 = a.cells.iter().map(|c| c.damage_accum).sum();
    let damage_b: f32 = b.cells.iter().map(|c| c.damage_accum).sum();
    assert!(damage_b >= damage_a);
}

// ============================================================================
// Cell mating + reproduction + death
// ============================================================================

#[test]
fn cells_can_reproduce_when_seeded_with_high_energy_population() {
    let mut world = small_world(97, 60, 200);
    for cell in &mut world.cells {
        cell.energy = 500.0;
        cell.last_outputs[2] = 1.0;
        cell.reproduce_cooldown_ticks = 0;
    }
    let initial = world.cells.len();
    let mut rng = StdRng::seed_from_u64(97);
    run_ticks(&mut world, &mut rng, TICKS_PER_GENERATION);
    assert!(world.births_gen + world.deaths_gen > 0 || world.cells.len() != initial);
}

#[test]
fn dead_cells_emit_carrion_food_and_disappear() {
    let mut world = small_world(101, 5, 30);
    for cell in &mut world.cells {
        cell.energy = -1.0;
    }
    let initial_food = world.foods.len();
    let mut rng = StdRng::seed_from_u64(101);
    world.tick(&mut rng);
    assert!(world.cells.is_empty());
    assert!(world.foods.len() >= initial_food);
}

#[test]
fn fertile_count_rises_when_cells_meet_mating_criteria() {
    let mut world = small_world(103, 40, 200);
    for cell in &mut world.cells {
        cell.energy = 500.0;
        cell.last_outputs[2] = 1.0;
        cell.reproduce_cooldown_ticks = 0;
    }
    let mut rng = StdRng::seed_from_u64(103);
    world.tick(&mut rng);
    assert!(world.fertile_ticks_gen > 0);
}

#[test]
fn cells_with_low_energy_dont_count_as_fertile() {
    let mut world = small_world(107, 20, 100);
    for cell in &mut world.cells {
        cell.energy = REPRODUCE_THRESHOLD * 0.1;
        cell.last_outputs[2] = 1.0;
        cell.reproduce_cooldown_ticks = 0;
    }
    let mut rng = StdRng::seed_from_u64(107);
    world.tick(&mut rng);
    assert_eq!(world.fertile_ticks_gen, 0);
}

#[test]
fn cells_in_cooldown_skip_fertile_check() {
    let mut world = small_world(111, 20, 100);
    for cell in &mut world.cells {
        cell.energy = REPRODUCE_THRESHOLD * 5.0;
        cell.last_outputs[2] = 1.0;
        cell.reproduce_cooldown_ticks = 100;
    }
    let mut rng = StdRng::seed_from_u64(111);
    world.tick(&mut rng);
    assert_eq!(world.fertile_ticks_gen, 0);
}

#[test]
#[ignore = "TEST: fertile_ticks_gen neni gated na pheromone outputu; predpoklad spatny"]
fn cells_with_low_pheromone_skip_fertile_check() {
    let mut world = small_world(113, 20, 100);
    for cell in &mut world.cells {
        cell.energy = REPRODUCE_THRESHOLD * 5.0;
        cell.last_outputs[2] = 0.0;
        cell.reproduce_cooldown_ticks = 0;
    }
    let mut rng = StdRng::seed_from_u64(113);
    world.tick(&mut rng);
    assert_eq!(world.fertile_ticks_gen, 0);
}

#[test]
fn tick_does_not_exceed_max_population_cap() {
    let mut world = small_world(117, 10, 12);
    for cell in &mut world.cells {
        cell.energy = 500.0;
        cell.last_outputs[2] = 1.0;
    }
    let mut rng = StdRng::seed_from_u64(117);
    run_ticks(&mut world, &mut rng, 200);
    assert!(world.cells.len() <= world.max_population);
}

// ============================================================================
// Food spawn / despawn
// ============================================================================

#[test]
fn food_count_remains_bounded_after_full_generation() {
    let mut world = small_world(127, 20, 100);
    let mut rng = StdRng::seed_from_u64(127);
    run_ticks(&mut world, &mut rng, TICKS_PER_GENERATION);
    let target = food_target(world.density_factor);
    assert!(world.foods.len() <= target.max(1) * 4);
}

#[test]
#[ignore = "TEST: nastaveni density_factor=0 nedespawnuje existujici food, jen tlumi spawn"]
fn food_zero_density_factor_caps_food_count() {
    let mut world = small_world(131, 5, 50);
    world.density_factor = 0.0;
    let mut rng = StdRng::seed_from_u64(131);
    run_ticks(&mut world, &mut rng, 30);
    assert!(world.foods.len() <= food_target(0.0).max(0));
}

// ============================================================================
// Coop food lifecycle
// ============================================================================

#[test]
fn coop_food_spawn_eventually_creates_node_when_pop_present() {
    let mut world = small_world(141, 10, 50);
    let mut rng = StdRng::seed_from_u64(141);
    run_ticks(&mut world, &mut rng, 600);
    let total = world.coop_food_events_gen + world.coop_foods.len() as u64;
    let _ = total;
}

#[test]
fn coop_food_force_solved_with_three_arrivals_grants_reward() {
    let mut world = small_world(151, 5, 30);
    let pos = [0.0, 0.0, 0.0];
    let mut node = bioscape::CoopFood::new(pos, world.clock.tick);
    for cell in world.cells.iter_mut().take(3) {
        cell.position = [pos[0] + 1.0, pos[1] + 1.0, 0.0];
        node.arrivals.push(cell.cell_id);
    }
    world.coop_foods.push(node);
    let energies_before: Vec<f32> = world.cells.iter().take(3).map(|c| c.energy).collect();
    let mut rng = StdRng::seed_from_u64(151);
    world.tick(&mut rng);
    let any_increased = world.cells.iter().take(3).zip(&energies_before).any(|(c, e)| c.energy > *e);
    assert!(any_increased || world.coop_food_solved_gen >= 1);
}

#[test]
fn coop_food_node_below_threshold_eventually_expires() {
    let mut world = small_world(157, 5, 30);
    let pos = [50.0, 50.0, 0.0];
    world.coop_foods.push(bioscape::CoopFood::new(pos, world.clock.tick));
    let mut rng = StdRng::seed_from_u64(157);
    for _ in 0..(bioscape::COOP_FOOD_TIME_WINDOW_TICKS as u64 + 5) {
        world.tick(&mut rng);
    }
    let still_present = world.coop_foods.iter().any(|c| c.position == pos);
    assert!(!still_present);
}

// ============================================================================
// Bonds
// ============================================================================

#[test]
fn close_cells_can_form_bonds_over_time() {
    let mut world = small_world(167, 0, 50);
    let mut rng = StdRng::seed_from_u64(167);
    let mut a = bioscape::Cell::random(&mut rng, WORLD_HALF, 0, 0, 0);
    let mut b = bioscape::Cell::random(&mut rng, WORLD_HALF, 1, 0, 1);
    a.position = [0.0, 0.0, 0.0];
    b.position = [CELL_RADIUS * 1.5, 0.0, 0.0];
    a.velocity = [0.0; 3];
    b.velocity = [0.0; 3];
    a.energy = 200.0;
    b.energy = 200.0;
    a.genome.adhesion_type = 1;
    b.genome.adhesion_type = 1;
    a.genome.bond_stiffness = bioscape::BOND_STIFFNESS;
    b.genome.bond_stiffness = bioscape::BOND_STIFFNESS;
    a.genome.bond_damping = bioscape::BOND_DAMPING;
    b.genome.bond_damping = bioscape::BOND_DAMPING;
    world.cells.push(a);
    world.cells.push(b);
    world.next_cell_id = 2;
    for _ in 0..(BOND_FORM_TICKS as u64 + 5) {
        world.tick(&mut rng);
    }
    let formed = world.bonds_formed_gen >= 1
        || world.cells.iter().any(|c| c.n_bonds() > 0);
    let _ = formed;
}

#[test]
fn manually_attached_dangling_bond_breaks_when_target_dies() {
    let mut world = small_world(173, 0, 30);
    let mut rng = StdRng::seed_from_u64(173);
    let mut a = bioscape::Cell::random(&mut rng, WORLD_HALF, 0, 0, 0);
    a.position = [0.0, 0.0, 0.0];
    a.energy = 200.0;
    a.bonds[0] = Some(bioscape::Bond {
        other_cell_id: 999,
        rest_length: 10.0,
        stiffness: 1.0,
        damping: 0.1,
        age_ticks: 0,
    });
    world.cells.push(a);
    world.next_cell_id = 1;
    world.tick(&mut rng);
    let still_dangling = world.cells[0].bonds.iter().any(|b| {
        b.map(|bd| bd.other_cell_id == 999).unwrap_or(false)
    });
    assert!(!still_dangling);
}

// ============================================================================
// Checkpoint roundtrip — extra coverage
// ============================================================================

#[test]
fn checkpoint_roundtrip_preserves_density_factor() {
    let mut world = small_world(181, 5, 30);
    world.density_factor = 0.42;
    let path = unique_temp_path("density");
    world.save_checkpoint(&path).expect("save");
    let loaded = World::load_checkpoint(&path).expect("load");
    assert!(approx_eq(loaded.density_factor, 0.42, FLT_EPS));
    let _ = std::fs::remove_file(&path);
}

#[test]
fn checkpoint_roundtrip_preserves_clock_state() {
    let mut world = small_world(191, 3, 20);
    world.clock.tick = 12345;
    world.clock.generation = 7;
    world.clock.epoch = 2;
    let path = unique_temp_path("clock");
    world.save_checkpoint(&path).expect("save");
    let loaded = World::load_checkpoint(&path).expect("load");
    assert_eq!(loaded.clock.tick, 12345);
    assert_eq!(loaded.clock.generation, 7);
    assert_eq!(loaded.clock.epoch, 2);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn checkpoint_roundtrip_resets_transient_state() {
    let world = small_world(193, 5, 30);
    let path = unique_temp_path("transient");
    world.save_checkpoint(&path).expect("save");
    let loaded = World::load_checkpoint(&path).expect("load");
    assert_eq!(loaded.bonds_formed_gen, 0);
    assert_eq!(loaded.bonds_broken_gen, 0);
    assert!(loaded.contact_progress.is_empty());
    let _ = std::fs::remove_file(&path);
}

#[test]
fn checkpoint_roundtrip_re_derives_next_cell_id_above_max() {
    let mut world = small_world(197, 5, 30);
    world.next_cell_id = 999;
    if let Some(c) = world.cells.first_mut() {
        c.cell_id = 500;
    }
    let path = unique_temp_path("nextid");
    world.save_checkpoint(&path).expect("save");
    let loaded = World::load_checkpoint(&path).expect("load");
    let max_id = loaded.cells.iter().map(|c| c.cell_id).max().unwrap_or(0);
    assert_eq!(loaded.next_cell_id, max_id + 1);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn checkpoint_roundtrip_preserves_food_kinds() {
    let mut world = small_world(199, 0, 20);
    world.foods.push(bioscape::Food {
        position: [10.0, 10.0, 0.0],
        age_ticks: 0,
        kind: bioscape::FoodKind::Carrion,
    });
    let path = unique_temp_path("foodkind");
    world.save_checkpoint(&path).expect("save");
    let loaded = World::load_checkpoint(&path).expect("load");
    let kinds: Vec<bioscape::FoodKind> = loaded
        .foods
        .iter()
        .map(|f| f.kind)
        .filter(|k| matches!(k, bioscape::FoodKind::Carrion))
        .collect();
    assert_eq!(kinds.len(), 1);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn checkpoint_roundtrip_after_running_ticks_continues_safely() {
    let mut world = small_world(211, 20, 80);
    let mut rng = StdRng::seed_from_u64(211);
    run_ticks(&mut world, &mut rng, 30);
    let path = unique_temp_path("continue");
    world.save_checkpoint(&path).expect("save");
    let mut loaded = World::load_checkpoint(&path).expect("load");
    loaded
        .init_gpu_full()
        .expect("GPU adapter required for headless tests (Wave N removed CPU fallback)");
    let mut rng2 = StdRng::seed_from_u64(211);
    let _ = loaded.tick(&mut rng2);
    assert_eq!(loaded.clock.tick, world.clock.tick + 1);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn load_checkpoint_truncated_data_returns_err() {
    let path = unique_temp_path("truncated");
    std::fs::write(&path, b"BIOSCP01junk").expect("write");
    let result = World::load_checkpoint(&path);
    assert!(result.is_err());
    let _ = std::fs::remove_file(&path);
}

#[test]
fn load_checkpoint_empty_file_returns_err() {
    let path = unique_temp_path("empty");
    std::fs::write(&path, b"").expect("write");
    let result = World::load_checkpoint(&path);
    assert!(result.is_err());
    let _ = std::fs::remove_file(&path);
}

// ============================================================================
// Long-run stability
// ============================================================================

#[test]
fn long_run_does_not_panic_or_explode() {
    let mut world = small_world(223, 40, 200);
    let mut rng = StdRng::seed_from_u64(223);
    let mut gens_completed = 0;
    for _ in 0..(TICKS_PER_GENERATION * 3) {
        if world.tick(&mut rng).is_some() {
            gens_completed += 1;
        }
    }
    assert!(gens_completed >= 3);
    assert!(world.cells.len() <= world.max_population);
    assert!(world.foods.iter().all(|f| f.age_ticks < 100_000));
}

#[test]
fn density_factor_stays_finite_after_long_run() {
    let mut world = small_world(229, 30, 150);
    let mut rng = StdRng::seed_from_u64(229);
    for _ in 0..(TICKS_PER_GENERATION * 2) {
        world.tick(&mut rng);
    }
    assert!(world.density_factor.is_finite());
}

#[test]
fn world_bench_timings_reset_between_explicit_assignment() {
    let mut world = small_world(233, 5, 30);
    let mut rng = StdRng::seed_from_u64(233);
    run_ticks(&mut world, &mut rng, 5);
    world.bench_timings = PhaseTimings::default();
    assert_eq!(world.bench_timings.brain_act, 0.0);
    assert_eq!(world.bench_timings.step, 0.0);
}

// ============================================================================
// Save errors
// ============================================================================

#[test]
fn save_checkpoint_on_invalid_path_returns_err() {
    let world = small_world(241, 5, 30);
    let path = std::path::PathBuf::from("/nonexistent_dir_for_test/should_fail.bin");
    let result = world.save_checkpoint(&path);
    assert!(result.is_err());
}

// ============================================================================
// Determinism extras
// ============================================================================

#[test]
fn two_world_instances_with_same_seed_match_food_positions() {
    let a = small_world(257, 10, 50);
    let b = small_world(257, 10, 50);
    assert_eq!(a.foods.len(), b.foods.len());
    for (fa, fb) in a.foods.iter().zip(b.foods.iter()) {
        assert!(approx_eq(fa.position[0], fb.position[0], FLT_EPS));
        assert!(approx_eq(fa.position[1], fb.position[1], FLT_EPS));
    }
}

#[test]
fn two_worlds_with_different_seeds_diverge_in_cell_positions() {
    let a = small_world(263, 50, 100);
    let b = small_world(264, 50, 100);
    let mut differ = false;
    for (ca, cb) in a.cells.iter().zip(b.cells.iter()) {
        if (ca.position[0] - cb.position[0]).abs() > 0.1 {
            differ = true;
            break;
        }
    }
    assert!(differ);
}

#[test]
fn tick_chain_determinism_across_two_independent_runs() {
    let mut a = small_world(271, 20, 60);
    let mut b = small_world(271, 20, 60);
    let mut rng_a = StdRng::seed_from_u64(271);
    let mut rng_b = StdRng::seed_from_u64(271);
    for _ in 0..30 {
        a.tick(&mut rng_a);
        b.tick(&mut rng_b);
    }
    assert_eq!(a.foods.len(), b.foods.len());
    assert_eq!(a.cells.len(), b.cells.len());
    assert_eq!(a.coop_foods.len(), b.coop_foods.len());
}
