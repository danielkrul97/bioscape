//! Fáze 2 — domain logic: predator.rs, cell.rs další větve, genetics/genome.rs
//! kompletní crossover.

#![allow(unused_imports)]

use crate::test_helpers::*;
use crate::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

// ─── Local helpers (unique to phase2 — neduplikuje tests.rs:make_test_hunter
// které je private uvnitř toho souboru) ───

fn make_hunter_at(pos: [f32; 3], vel: [f32; 3]) -> Hunter {
    let genome = HunterGenome {
        vision_radius: HUNTER_VISION_RADIUS,
        vision_fov: HUNTER_VISION_FOV,
        max_speed: HUNTER_MAX_SPEED,
        acceleration: HUNTER_ACC,
        attack_radius: HUNTER_ATTACK_RADIUS,
        damage_per_tick: HUNTER_DAMAGE_PER_TICK,
        body_size: 1.0,
        color_hue: 0.0,
        adhesion_type: 0,
        brain: dummy_brain(),
    };
    Hunter {
        position: pos,
        velocity: vel,
        hunter_id: 0,
        genome,
        energy: HUNTER_INITIAL_ENERGY,
        age: 0,
        reproduce_cooldown_ticks: 0,
        lineage_id: 0,
        lineage_birth_gen: 0,
        heading: 0.0,
        pitch: 0.0,
        angular_velocity: 0.0,
        pitch_velocity: 0.0,
        last_inputs: [0.0; BRAIN_INPUTS],
        last_hidden: [0.0; BRAIN_HIDDEN],
        last_outputs: [0.0; BRAIN_OUTPUTS],
        bonds: [None; MAX_BONDS_PER_CELL],
        pooled_hidden: [0.0; BRAIN_HIDDEN],
    }
}

fn cell_grid_from(cells: &[Cell]) -> SpatialGrid<usize, ()> {
    let mut g: SpatialGrid<usize, ()> = SpatialGrid::new(GRID_CELL_SIZE, WORLD_HALF);
    g.rebuild(cells.iter().enumerate().map(|(i, c)| (i, c.position, ())));
    g
}

// ─── predator.rs: Hunter parameter ranges + helpers ───────────────────────────

#[test]
fn sensor_slot_category_proprio_returns_none() {
    for slot in [4usize, 5, 9, 10, 18] {
        assert_eq!(
            sensor_slot_category(slot),
            None,
            "proprio slot {} must not have category",
            slot
        );
    }
}

#[test]
fn sensor_slot_category_unknown_slot_returns_none() {
    for slot in [21usize, 30, 50, 52, BRAIN_INPUTS - 1] {
        assert_eq!(sensor_slot_category(slot), None);
    }
}

#[test]
fn hunter_genome_random_within_initial_band() {
    // `random` draws z užšího middle bandu než [MIN, MAX] clamp range.
    // Initial population diversity check.
    let mut rng = StdRng::seed_from_u64(0x9001);
    for _ in 0..200 {
        let g = HunterGenome::random(&mut rng);
        assert!(g.vision_radius >= 100.0 && g.vision_radius < 300.0);
        assert!(g.max_speed >= 200.0 && g.max_speed < 400.0);
        assert!(g.acceleration >= 60.0 && g.acceleration < 120.0);
        assert!(g.attack_radius >= 12.0 && g.attack_radius < 28.0);
        assert!(g.damage_per_tick >= 4.0 && g.damage_per_tick < 12.0);
        assert!(g.body_size >= 0.8 && g.body_size < 1.6);
        assert!(g.adhesion_type < ADHESION_TYPE_COUNT);
    }
}

#[test]
fn hunter_mutate_zero_sigma_preserves_scalars() {
    let mut rng = StdRng::seed_from_u64(0x9002);
    let g = HunterGenome::random(&mut rng);
    let cfg = HunterMutationConfig {
        sigma_vision_radius: 0.0,
        sigma_vision_fov: 0.0,
        sigma_max_speed: 0.0,
        sigma_acceleration: 0.0,
        sigma_attack_radius: 0.0,
        sigma_damage: 0.0,
        sigma_body_size: 0.0,
        sigma_color_hue: 0.0,
        sigma_brain: 0.0,
        adhesion_flip_rate: 0.0,
    };
    let m = g.mutate(&mut rng, &cfg);
    assert_eq!(m.vision_radius, g.vision_radius);
    assert_eq!(m.vision_fov, g.vision_fov);
    assert_eq!(m.max_speed, g.max_speed);
    assert_eq!(m.acceleration, g.acceleration);
    assert_eq!(m.attack_radius, g.attack_radius);
    assert_eq!(m.damage_per_tick, g.damage_per_tick);
    assert_eq!(m.body_size, g.body_size);
    assert_eq!(m.color_hue, g.color_hue);
    assert_eq!(m.adhesion_type, g.adhesion_type);
}

#[test]
fn hunter_mutate_adhesion_flip_full_rate_changes_type() {
    let mut rng = StdRng::seed_from_u64(0x9003);
    let mut g = HunterGenome::random(&mut rng);
    g.adhesion_type = 3;
    let cfg = HunterMutationConfig {
        sigma_vision_radius: 0.0,
        sigma_vision_fov: 0.0,
        sigma_max_speed: 0.0,
        sigma_acceleration: 0.0,
        sigma_attack_radius: 0.0,
        sigma_damage: 0.0,
        sigma_body_size: 0.0,
        sigma_color_hue: 0.0,
        sigma_brain: 0.0,
        adhesion_flip_rate: 1.0,
    };
    for _ in 0..50 {
        let m = g.mutate(&mut rng, &cfg);
        assert_ne!(m.adhesion_type, 3);
        assert!(m.adhesion_type < ADHESION_TYPE_COUNT);
    }
}

#[test]
fn hunter_crossover_brain_and_all_genes_from_parents() {
    let mut rng = StdRng::seed_from_u64(0x9004);
    let a = HunterGenome {
        vision_radius: MIN_HUNTER_VISION_RADIUS,
        vision_fov: MIN_HUNTER_VISION_FOV,
        max_speed: MIN_HUNTER_MAX_SPEED,
        acceleration: MIN_HUNTER_ACC,
        attack_radius: MIN_HUNTER_ATTACK_RADIUS,
        damage_per_tick: MIN_HUNTER_DAMAGE,
        body_size: MIN_HUNTER_BODY_SIZE,
        color_hue: 5.0,
        adhesion_type: 0,
        brain: dummy_brain(),
    };
    let b = HunterGenome {
        vision_radius: MAX_HUNTER_VISION_RADIUS,
        vision_fov: MAX_HUNTER_VISION_FOV,
        max_speed: MAX_HUNTER_MAX_SPEED,
        acceleration: MAX_HUNTER_ACC,
        attack_radius: MAX_HUNTER_ATTACK_RADIUS,
        damage_per_tick: MAX_HUNTER_DAMAGE,
        body_size: MAX_HUNTER_BODY_SIZE,
        color_hue: 250.0,
        adhesion_type: 7,
        brain: dummy_brain(),
    };
    for _ in 0..100 {
        let c = HunterGenome::crossover(&a, &b, &mut rng);
        assert!(c.vision_fov == a.vision_fov || c.vision_fov == b.vision_fov);
        assert!(c.acceleration == a.acceleration || c.acceleration == b.acceleration);
        assert!(c.attack_radius == a.attack_radius || c.attack_radius == b.attack_radius);
        assert!(c.body_size == a.body_size || c.body_size == b.body_size);
        assert!(c.color_hue == a.color_hue || c.color_hue == b.color_hue);
    }
}

#[test]
fn hunter_apply_energy_costs_zero_dt_is_noop() {
    let mut h = make_hunter_at([0.0; 3], [10.0, 0.0, 0.0]);
    let initial = h.energy;
    h.apply_energy_costs(0.0);
    assert_eq!(h.energy, initial);
}

#[test]
fn hunter_apply_energy_costs_scales_linearly_with_dt() {
    let mut a = make_hunter_at([0.0; 3], [50.0, 0.0, 0.0]);
    let mut b = make_hunter_at([0.0; 3], [50.0, 0.0, 0.0]);
    a.apply_energy_costs(0.5);
    b.apply_energy_costs(1.0);
    let drain_a = HUNTER_INITIAL_ENERGY - a.energy;
    let drain_b = HUNTER_INITIAL_ENERGY - b.energy;
    assert!(
        (drain_b - 2.0 * drain_a).abs() < 1e-3,
        "linear: drain_b={} should be 2× drain_a={}",
        drain_b,
        drain_a
    );
}

#[test]
fn hunter_apply_energy_costs_body_size_cubic() {
    let mut small = make_hunter_at([0.0; 3], [0.0; 3]);
    small.genome.body_size = 1.0;
    let mut big = make_hunter_at([0.0; 3], [0.0; 3]);
    big.genome.body_size = 2.0;
    small.apply_energy_costs(1.0);
    big.apply_energy_costs(1.0);
    let drain_small = HUNTER_INITIAL_ENERGY - small.energy;
    let drain_big = HUNTER_INITIAL_ENERGY - big.energy;
    // Body part of drain scales s s³, takže big drain > small drain o body component.
    assert!(drain_big > drain_small);
}

#[test]
fn hunter_step_increments_age_and_decrements_cooldown() {
    let half = [960.0, 540.0, 50.0];
    let mut h = make_hunter_at([0.0; 3], [0.0; 3]);
    h.reproduce_cooldown_ticks = 5;
    h.step(1.0 / 60.0, half);
    assert_eq!(h.age, 1);
    assert_eq!(h.reproduce_cooldown_ticks, 4);
}

#[test]
fn hunter_step_cooldown_does_not_underflow() {
    let half = [960.0, 540.0, 50.0];
    let mut h = make_hunter_at([0.0; 3], [0.0; 3]);
    h.reproduce_cooldown_ticks = 0;
    for _ in 0..10 {
        h.step(1.0 / 60.0, half);
    }
    assert_eq!(h.reproduce_cooldown_ticks, 0);
    assert_eq!(h.age, 10);
}

#[test]
fn hunter_step_xy_wraps_toroidal() {
    let half = [100.0, 100.0, 50.0];
    let mut h = make_hunter_at([99.0, 0.0, 0.0], [120.0, 0.0, 0.0]);
    h.step(1.0, half);
    assert!(h.position[0] > -100.0 && h.position[0] < 100.0);
}

#[test]
fn hunter_step_z_bounces_off_floor() {
    let half = [100.0, 100.0, 10.0];
    let mut h = make_hunter_at([0.0, 0.0, 9.5], [0.0, 0.0, 50.0]);
    h.step(1.0, half);
    assert!(h.velocity[2] < 0.0, "z velocity should flip after bounce");
    assert!(h.position[2] <= 10.0);
}

#[test]
fn hunter_apply_brain_motor_clamps_thrust() {
    let half = [1000.0, 1000.0, 50.0];
    let mut h = make_hunter_at([0.0; 3], [0.0; 3]);
    let mut outputs = [0.0_f32; BRAIN_OUTPUTS];
    outputs[1] = 5.0; // way above 1.0 — should clamp
    h.apply_brain_motor(&outputs, None, 1.0, half);
    let speed_sq =
        h.velocity[0].powi(2) + h.velocity[1].powi(2) + h.velocity[2].powi(2);
    let max_sq = h.genome.max_speed.powi(2);
    assert!(speed_sq <= max_sq * 1.0001);
}

#[test]
fn hunter_apply_brain_motor_negative_thrust_reverses() {
    let half = [1000.0, 1000.0, 50.0];
    let mut h = make_hunter_at([0.0; 3], [0.0; 3]);
    let mut outputs = [0.0_f32; BRAIN_OUTPUTS];
    outputs[1] = -1.0;
    h.apply_brain_motor(&outputs, None, 1.0, half);
    // forward = (1, 0, 0), thrust=-1 → -x.
    assert!(h.velocity[0] < 0.0);
}

#[test]
fn hunter_apply_brain_motor_seek_target_at_origin_no_change_for_idle() {
    let half = [1000.0, 1000.0, 50.0];
    let mut h = make_hunter_at([0.0, 0.0, 0.0], [0.0; 3]);
    let outputs = [0.0_f32; BRAIN_OUTPUTS];
    // Target at same position → degenerate distance, no clear direction.
    h.apply_brain_motor(&outputs, Some([0.0, 0.0, 0.0]), 1.0 / 60.0, half);
    // No NaN / no inf.
    assert!(h.velocity[0].is_finite());
    assert!(h.angular_velocity.is_finite());
}

// ─── nearest_attackable_cell coverage ─────────────────────────────────────────

#[test]
fn nearest_attackable_picks_closer_of_two() {
    let mut rng = StdRng::seed_from_u64(0xA001);
    let half = [960.0, 540.0, 50.0];
    let mut far = Cell::random(&mut rng, half, 0, 0, 0);
    far.position = [150.0, 0.0, 0.0];
    let mut near = Cell::random(&mut rng, half, 1, 0, 1);
    near.position = [40.0, 0.0, 0.0];
    let cells = vec![far, near];
    let h = make_hunter_at([0.0, 0.0, 0.0], [0.0; 3]);
    let grid = cell_grid_from(&cells);
    assert_eq!(nearest_attackable_cell(&h, &cells, &grid, half), Some(1));
}

#[test]
fn nearest_attackable_returns_none_when_no_cells_in_vision() {
    let mut rng = StdRng::seed_from_u64(0xA002);
    let half = [960.0, 540.0, 50.0];
    let mut far = Cell::random(&mut rng, half, 0, 0, 0);
    far.position = [500.0, 0.0, 0.0]; // > HUNTER_VISION_RADIUS=200
    let cells = vec![far];
    let h = make_hunter_at([0.0, 0.0, 0.0], [0.0; 3]);
    let grid = cell_grid_from(&cells);
    assert!(nearest_attackable_cell(&h, &cells, &grid, half).is_none());
}

#[test]
fn nearest_attackable_table_driven_by_n_bonds() {
    // Cell s n_bonds < THRESHOLD → attackable; >= THRESHOLD → invisible.
    let mut rng = StdRng::seed_from_u64(0xA003);
    let half = [960.0, 540.0, 50.0];
    for n_bonds in 0..=(HUNTER_BOND_IMMUNITY_THRESHOLD as usize + 1) {
        let mut c = Cell::random(&mut rng, half, 0, 0, 0);
        c.position = [40.0, 0.0, 0.0];
        for slot in 0..n_bonds {
            c.bonds[slot] = Some(Bond {
                other_cell_id: 1000 + slot as u64,
                rest_length: 5.0,
                stiffness: BOND_STIFFNESS,
                damping: BOND_DAMPING,
                age_ticks: 0,
            });
        }
        let cells = vec![c];
        let h = make_hunter_at([0.0, 0.0, 0.0], [0.0; 3]);
        let grid = cell_grid_from(&cells);
        let pick = nearest_attackable_cell(&h, &cells, &grid, half);
        if (n_bonds as u32) < HUNTER_BOND_IMMUNITY_THRESHOLD {
            assert_eq!(pick, Some(0), "n_bonds={} should be attackable", n_bonds);
        } else {
            assert!(pick.is_none(), "n_bonds={} should be immune", n_bonds);
        }
    }
}

#[test]
fn nearest_attackable_skips_immune_picks_attackable() {
    // Mix: jeden immune cluster cell blízko, jeden solo dál — solo se vybere
    // přes immune (immune neviditelná).
    let mut rng = StdRng::seed_from_u64(0xA004);
    let half = [960.0, 540.0, 50.0];
    let mut immune = Cell::random(&mut rng, half, 0, 0, 0);
    immune.position = [10.0, 0.0, 0.0];
    for slot in 0..(HUNTER_BOND_IMMUNITY_THRESHOLD as usize) {
        immune.bonds[slot] = Some(Bond {
            other_cell_id: 100 + slot as u64,
            rest_length: 5.0,
            stiffness: BOND_STIFFNESS,
            damping: BOND_DAMPING,
            age_ticks: 0,
        });
    }
    let mut solo = Cell::random(&mut rng, half, 1, 0, 1);
    solo.position = [60.0, 0.0, 0.0];
    let cells = vec![immune, solo];
    let h = make_hunter_at([0.0, 0.0, 0.0], [0.0; 3]);
    let grid = cell_grid_from(&cells);
    assert_eq!(nearest_attackable_cell(&h, &cells, &grid, half), Some(1));
}

#[test]
fn nearest_attackable_works_across_toroidal_boundary() {
    let mut rng = StdRng::seed_from_u64(0xA005);
    let half = [100.0, 100.0, 50.0];
    let mut c = Cell::random(&mut rng, half, 0, 0, 0);
    c.position = [-95.0, 0.0, 0.0];
    let cells = vec![c];
    let h = make_hunter_at([95.0, 0.0, 0.0], [0.0; 3]);
    let grid = cell_grid_from(&cells);
    let pick = nearest_attackable_cell(&h, &cells, &grid, half);
    assert_eq!(pick, Some(0), "min-image distance should be ~10 across wrap");
}

// ─── Hunter sensors / brain inputs ─────────────────────────────────────────────

#[test]
fn gather_hunter_sensors_finds_nearest_prey() {
    let mut rng = StdRng::seed_from_u64(0xB001);
    let half = [960.0, 540.0, 50.0];
    let mut near = Cell::random(&mut rng, half, 0, 0, 0);
    near.position = [30.0, 0.0, 0.0];
    let mut far = Cell::random(&mut rng, half, 1, 0, 1);
    far.position = [120.0, 0.0, 0.0];
    let cells = vec![near, far];
    let h = make_hunter_at([0.0, 0.0, 0.0], [0.0; 3]);
    let grid = cell_grid_from(&cells);
    let smell = SmellField::new([8, 8, 4], half);
    let sensors = gather_hunter_sensors(&h, &cells, &grid, &[], &smell, half);
    assert!(sensors.nearest_prey.is_some());
    let d = sensors.nearest_prey.unwrap();
    assert!((d[0] - 30.0).abs() < 1e-3);
    assert_eq!(sensors.neighbors_in_vision, 2);
}

#[test]
fn gather_hunter_sensors_pack_member_same_type() {
    let half = [960.0, 540.0, 50.0];
    let h = make_hunter_at([0.0, 0.0, 0.0], [0.0; 3]);
    let mut other = make_hunter_at([50.0, 0.0, 0.0], [0.0; 3]);
    other.hunter_id = 99;
    other.genome.adhesion_type = h.genome.adhesion_type; // same
    let snap = vec![HunterSnapshotMin::from_hunter(&other)];
    let cells: Vec<Cell> = vec![];
    let grid = cell_grid_from(&cells);
    let smell = SmellField::new([8, 8, 4], half);
    let sensors = gather_hunter_sensors(&h, &cells, &grid, &snap, &smell, half);
    assert!(sensors.nearest_pack_member.is_some());
    assert_eq!(sensors.same_type_in_vision, 1);
}

#[test]
fn gather_hunter_sensors_ignores_cross_type_packmate() {
    let half = [960.0, 540.0, 50.0];
    let h = make_hunter_at([0.0, 0.0, 0.0], [0.0; 3]);
    let mut other = make_hunter_at([50.0, 0.0, 0.0], [0.0; 3]);
    other.hunter_id = 99;
    other.genome.adhesion_type = (h.genome.adhesion_type + 1) % ADHESION_TYPE_COUNT;
    let snap = vec![HunterSnapshotMin::from_hunter(&other)];
    let cells: Vec<Cell> = vec![];
    let grid = cell_grid_from(&cells);
    let smell = SmellField::new([8, 8, 4], half);
    let sensors = gather_hunter_sensors(&h, &cells, &grid, &snap, &smell, half);
    assert!(sensors.nearest_pack_member.is_none());
    assert_eq!(sensors.same_type_in_vision, 0);
}

#[test]
fn gather_hunter_sensors_skips_self() {
    let half = [960.0, 540.0, 50.0];
    let h = make_hunter_at([0.0, 0.0, 0.0], [0.0; 3]);
    let snap = vec![HunterSnapshotMin::from_hunter(&h)]; // self
    let cells: Vec<Cell> = vec![];
    let grid = cell_grid_from(&cells);
    let smell = SmellField::new([8, 8, 4], half);
    let sensors = gather_hunter_sensors(&h, &cells, &grid, &snap, &smell, half);
    assert!(sensors.nearest_pack_member.is_none());
    assert_eq!(sensors.same_type_in_vision, 0);
}

#[test]
fn populate_hunter_brain_inputs_pack_delta_in_slots_2_3_16() {
    let mut h = make_hunter_at([0.0; 3], [0.0; 3]);
    let sensors = HunterBrainSensors {
        nearest_prey: None,
        nearest_prey_size: 0.0,
        neighbors_in_vision: 0,
        smell_grad: [0.0; 3],
        nearest_pack_member: Some([100.0, 50.0, 10.0]),
        same_type_in_vision: 2,
    };
    let inputs = populate_hunter_brain_inputs(&mut h, &sensors);
    assert!((inputs[2] - 0.5).abs() < 1e-4);
    assert!((inputs[3] - 0.25).abs() < 1e-4);
    assert!((inputs[16] - 0.05).abs() < 1e-4);
}

#[test]
fn populate_hunter_brain_inputs_writes_energy_and_speed() {
    let mut h = make_hunter_at([0.0; 3], [60.0, 0.0, 0.0]);
    h.energy = HUNTER_REPRODUCE_THRESHOLD * 0.5;
    let sensors = HunterBrainSensors {
        nearest_prey: None,
        nearest_prey_size: 0.0,
        neighbors_in_vision: 0,
        smell_grad: [0.0; 3],
        nearest_pack_member: None,
        same_type_in_vision: 0,
    };
    let inputs = populate_hunter_brain_inputs(&mut h, &sensors);
    assert!((inputs[4] - 0.5).abs() < 1e-3);
    assert!(inputs[5] > 0.0);
    assert!((inputs[9] - 1.0).abs() < 1e-4); // forward.x = cos(0) = 1
    assert!((inputs[10] - 0.0).abs() < 1e-4);
    assert!((inputs[18] - 0.0).abs() < 1e-4);
}

#[test]
fn populate_hunter_brain_inputs_pack_size_and_density() {
    let mut h = make_hunter_at([0.0; 3], [0.0; 3]);
    h.bonds[0] = Some(Bond {
        other_cell_id: 1,
        rest_length: 10.0,
        stiffness: BOND_STIFFNESS,
        damping: BOND_DAMPING,
        age_ticks: 0,
    });
    h.bonds[1] = Some(Bond {
        other_cell_id: 2,
        rest_length: 10.0,
        stiffness: BOND_STIFFNESS,
        damping: BOND_DAMPING,
        age_ticks: 0,
    });
    let sensors = HunterBrainSensors {
        nearest_prey: None,
        nearest_prey_size: 0.0,
        neighbors_in_vision: 0,
        smell_grad: [0.0; 3],
        nearest_pack_member: None,
        same_type_in_vision: 4,
    };
    let inputs = populate_hunter_brain_inputs(&mut h, &sensors);
    let expected_pack_size = 2.0_f32 / MAX_BONDS_PER_CELL as f32;
    assert!((inputs[11] - expected_pack_size).abs() < 1e-4);
    assert!(inputs[12] > 0.0);
}

#[test]
fn populate_hunter_brain_inputs_pooled_hidden_in_recurrent_slots() {
    let mut h = make_hunter_at([0.0; 3], [0.0; 3]);
    for k in 0..BRAIN_HIDDEN {
        h.pooled_hidden[k] = (k as f32) * 0.01;
    }
    let sensors = HunterBrainSensors {
        nearest_prey: None,
        nearest_prey_size: 0.0,
        neighbors_in_vision: 0,
        smell_grad: [0.0; 3],
        nearest_pack_member: None,
        same_type_in_vision: 0,
    };
    let inputs = populate_hunter_brain_inputs(&mut h, &sensors);
    for k in 0..BRAIN_RECURRENT {
        assert!((inputs[BRAIN_INPUTS_SENSORY + k] - h.pooled_hidden[k]).abs() < 1e-6);
    }
}

// ─── pool_bonded_hunter_hidden ─────────────────────────────────────────────────

#[test]
fn pool_bonded_hunter_hidden_solo_returns_self() {
    let mut h = make_hunter_at([0.0; 3], [0.0; 3]);
    h.hunter_id = 1;
    for k in 0..BRAIN_HIDDEN {
        h.last_hidden[k] = 0.7;
    }
    let mut hunters = vec![h];
    pool_bonded_hunter_hidden(&mut hunters);
    for k in 0..BRAIN_HIDDEN {
        assert!((hunters[0].pooled_hidden[k] - 0.7).abs() < 1e-6);
    }
}

#[test]
fn pool_bonded_hunter_hidden_pair_averages() {
    let mut a = make_hunter_at([0.0; 3], [0.0; 3]);
    a.hunter_id = 1;
    let mut b = make_hunter_at([10.0, 0.0, 0.0], [0.0; 3]);
    b.hunter_id = 2;
    for k in 0..BRAIN_HIDDEN {
        a.last_hidden[k] = 1.0;
        b.last_hidden[k] = 3.0;
    }
    a.bonds[0] = Some(Bond {
        other_cell_id: 2,
        rest_length: 5.0,
        stiffness: BOND_STIFFNESS,
        damping: BOND_DAMPING,
        age_ticks: 0,
    });
    b.bonds[0] = Some(Bond {
        other_cell_id: 1,
        rest_length: 5.0,
        stiffness: BOND_STIFFNESS,
        damping: BOND_DAMPING,
        age_ticks: 0,
    });
    let mut hunters = vec![a, b];
    pool_bonded_hunter_hidden(&mut hunters);
    for k in 0..BRAIN_HIDDEN {
        assert!((hunters[0].pooled_hidden[k] - 2.0).abs() < 1e-5);
        assert!((hunters[1].pooled_hidden[k] - 2.0).abs() < 1e-5);
    }
}

#[test]
fn pool_bonded_hunter_hidden_dangling_bond_skipped() {
    let mut h = make_hunter_at([0.0; 3], [0.0; 3]);
    h.hunter_id = 1;
    for k in 0..BRAIN_HIDDEN {
        h.last_hidden[k] = 0.4;
    }
    h.bonds[0] = Some(Bond {
        other_cell_id: 999, // doesn't exist
        rest_length: 5.0,
        stiffness: BOND_STIFFNESS,
        damping: BOND_DAMPING,
        age_ticks: 0,
    });
    let mut hunters = vec![h];
    pool_bonded_hunter_hidden(&mut hunters);
    for k in 0..BRAIN_HIDDEN {
        assert!((hunters[0].pooled_hidden[k] - 0.4).abs() < 1e-6);
    }
}

#[test]
fn pool_bonded_hunter_hidden_empty_slice_no_panic() {
    let mut hunters: Vec<Hunter> = vec![];
    pool_bonded_hunter_hidden(&mut hunters);
}

// ─── make_hunter_child / make_hunter_mating_child ─────────────────────────────

#[test]
fn make_hunter_child_resets_age_and_brain_state() {
    let mut rng = StdRng::seed_from_u64(0xC001);
    let half = [960.0, 540.0, 50.0];
    let mut parent = make_hunter_at([10.0, 20.0, 0.0], [50.0, 0.0, 0.0]);
    parent.age = 12345;
    parent.last_hidden = [0.7; BRAIN_HIDDEN];
    parent.last_outputs = [0.5; BRAIN_OUTPUTS];
    let child = make_hunter_child(&parent, &mut rng, half, 200, 9);
    assert_eq!(child.age, 0);
    assert!(child.last_hidden.iter().all(|&x| x == 0.0));
    assert!(child.last_outputs.iter().all(|&x| x == 0.0));
    assert_eq!(child.lineage_birth_gen, 9);
}

#[test]
fn make_hunter_mating_child_lineage_from_parent_a() {
    let mut rng = StdRng::seed_from_u64(0xC002);
    let half = [960.0, 540.0, 50.0];
    let mut a = make_hunter_at([0.0; 3], [0.0; 3]);
    let mut b = make_hunter_at([10.0; 3], [0.0; 3]);
    a.lineage_id = 100;
    b.lineage_id = 200;
    let child = make_hunter_mating_child(&a, &b, &mut rng, half, 1, 5);
    assert_eq!(child.lineage_id, 100);
}

#[test]
fn make_hunter_mating_child_position_is_midpoint() {
    let mut rng = StdRng::seed_from_u64(0xC003);
    let half = [960.0, 540.0, 50.0];
    let a = make_hunter_at([100.0, 200.0, 10.0], [0.0; 3]);
    let b = make_hunter_at([200.0, 300.0, -10.0], [0.0; 3]);
    let child = make_hunter_mating_child(&a, &b, &mut rng, half, 1, 5);
    assert!((child.position[0] - 150.0).abs() < 1e-3);
    assert!((child.position[1] - 250.0).abs() < 1e-3);
    assert!((child.position[2] - 0.0).abs() < 1e-3);
}

#[test]
fn make_hunter_child_position_inherits_parent() {
    let mut rng = StdRng::seed_from_u64(0xC004);
    let half = [960.0, 540.0, 50.0];
    let parent = make_hunter_at([42.0, -33.0, 7.0], [0.0; 3]);
    let child = make_hunter_child(&parent, &mut rng, half, 10, 0);
    assert_eq!(child.position, parent.position);
}

// ─── Hunter constants sanity ──────────────────────────────────────────────────

#[test]
fn hunter_attack_radius_smaller_than_vision() {
    assert!(HUNTER_ATTACK_RADIUS < HUNTER_VISION_RADIUS);
    assert!(MAX_HUNTER_ATTACK_RADIUS < MAX_HUNTER_VISION_RADIUS);
}

#[test]
fn hunter_immunity_threshold_le_max_bonds() {
    assert!(HUNTER_BOND_IMMUNITY_THRESHOLD as usize <= MAX_BONDS_PER_CELL);
}

#[test]
fn hunter_genome_random_brain_has_default_hidden() {
    let mut rng = StdRng::seed_from_u64(0xD001);
    let g = HunterGenome::random(&mut rng);
    assert_eq!(g.brain.hidden_n, BRAIN_HIDDEN_DEFAULT as u32);
}

// ─── cell.rs additional coverage ──────────────────────────────────────────────

#[test]
fn n_bonds_zero_for_empty_cell() {
    let c = base_cell();
    assert_eq!(c.n_bonds(), 0);
}

#[test]
fn n_bonds_counts_all_full_slots() {
    let mut c = base_cell();
    for slot in 0..MAX_BONDS_PER_CELL {
        c.bonds[slot] = Some(Bond {
            other_cell_id: slot as u64 + 1,
            rest_length: 10.0,
            stiffness: BOND_STIFFNESS,
            damping: BOND_DAMPING,
            age_ticks: 0,
        });
    }
    assert_eq!(c.n_bonds() as usize, MAX_BONDS_PER_CELL);
}

#[test]
fn cell_state_feedback_pushes_above_half_toward_one() {
    let mut c = base_cell();
    c.cell_state = 0.7;
    let cfg = no_drag_physics(0.0, 0.0);
    let half = [1000.0, 1000.0, 0.0];
    for _ in 0..600 {
        c.step(1.0 / 60.0, half, 0, 0, &cfg);
    }
    assert!(c.cell_state > 0.7, "feedback should push >0.5 toward 1.0");
}

#[test]
fn cell_state_feedback_pushes_below_half_toward_zero() {
    let mut c = base_cell();
    c.cell_state = 0.3;
    let cfg = no_drag_physics(0.0, 0.0);
    let half = [1000.0, 1000.0, 0.0];
    for _ in 0..600 {
        c.step(1.0 / 60.0, half, 0, 0, &cfg);
    }
    assert!(c.cell_state < 0.3);
}

#[test]
fn cell_state_clamped_to_unit_range() {
    let mut c = base_cell();
    c.cell_state = 0.99;
    for slot in 0..MAX_BONDS_PER_CELL {
        c.bonds[slot] = Some(Bond {
            other_cell_id: slot as u64 + 1,
            rest_length: 10.0,
            stiffness: BOND_STIFFNESS,
            damping: BOND_DAMPING,
            age_ticks: 0,
        });
    }
    let cfg = no_drag_physics(0.0, 0.0);
    let half = [1000.0, 1000.0, 0.0];
    for _ in 0..1000 {
        c.step(1.0 / 60.0, half, 0, 0, &cfg);
    }
    assert!(c.cell_state <= 1.0 && c.cell_state >= 0.0);
}

#[test]
fn cell_state_bond_bias_drives_altruist() {
    // Cell s bondy se posune k 1.0 rychleji než solo cell at same s0.
    let cfg = no_drag_physics(0.0, 0.0);
    let half = [1000.0, 1000.0, 0.0];
    let mut solo = base_cell();
    solo.cell_state = 0.55;
    let mut bonded = base_cell();
    bonded.cell_state = 0.55;
    for slot in 0..3 {
        bonded.bonds[slot] = Some(Bond {
            other_cell_id: slot as u64 + 1,
            rest_length: 10.0,
            stiffness: BOND_STIFFNESS,
            damping: BOND_DAMPING,
            age_ticks: 0,
        });
    }
    for _ in 0..120 {
        solo.step(1.0 / 60.0, half, 0, 0, &cfg);
        bonded.step(1.0 / 60.0, half, 0, 0, &cfg);
    }
    assert!(bonded.cell_state > solo.cell_state);
}

#[test]
fn step_drains_energy_proportional_to_vision_radius() {
    let mut a = base_cell();
    let mut b = base_cell();
    b.genome.vision_radius = a.genome.vision_radius * 2.0;
    let cfg = no_drag_physics(0.0, 0.1);
    let half = [1000.0, 1000.0, 0.0];
    a.step(1.0, half, 0, 0, &cfg);
    b.step(1.0, half, 0, 0, &cfg);
    let drain_a = 100.0 - a.energy;
    let drain_b = 100.0 - b.energy;
    assert!(drain_b > drain_a * 1.5, "vision drain ∝ radius");
}

#[test]
fn step_thermal_penalty_drains_when_temp_offset_from_optimum() {
    let mut hot_optimum = base_cell();
    hot_optimum.genome.thermal_optimum = THERMAL_TOP;
    let mut cold_optimum = base_cell();
    cold_optimum.genome.thermal_optimum = THERMAL_BOTTOM;
    let cfg = PhysicsConfig {
        drag: 0.0,
        angular_drag: 0.0,
        energy_cost_per_v_sq: 0.0,
        angular_energy_cost: 0.0,
        vision_cost_per_radius: 0.0,
        body_cost_factor: 0.0,
        thermal_optimum_penalty: 1.0,
    };
    let half = [1000.0, 1000.0, 50.0];
    // Both at top — warm. cold_optimum cell penalized; hot_optimum not.
    hot_optimum.position = [0.0, 0.0, 50.0];
    cold_optimum.position = [0.0, 0.0, 50.0];
    hot_optimum.step(1.0, half, 0, 0, &cfg);
    cold_optimum.step(1.0, half, 0, 0, &cfg);
    assert!(
        cold_optimum.energy < hot_optimum.energy,
        "cell with mismatched thermal optimum should drain more"
    );
}

#[test]
fn step_sensor_gain_drain_proportional_to_sum() {
    let mut a = base_cell();
    a.genome.sensor_gains = [0.0; N_SENSOR_CATEGORIES];
    let mut b = base_cell();
    b.genome.sensor_gains = [1.0; N_SENSOR_CATEGORIES];
    let cfg = no_drag_physics(0.0, 0.0);
    let half = [1000.0, 1000.0, 0.0];
    a.step(1.0, half, 0, 0, &cfg);
    b.step(1.0, half, 0, 0, &cfg);
    let drain_a = 100.0 - a.energy;
    let drain_b = 100.0 - b.energy;
    assert!(drain_b > drain_a, "non-zero gains should drain more");
}

#[test]
fn apply_morph_zero_outputs_no_phenotype_change() {
    let mut c = base_cell();
    let pre_l = c.phenotype.body_length;
    let pre_w = c.phenotype.body_width;
    let pre_h = c.phenotype.body_height;
    c.apply_morph(1.0);
    assert_eq!(c.phenotype.body_length, pre_l);
    assert_eq!(c.phenotype.body_width, pre_w);
    assert_eq!(c.phenotype.body_height, pre_h);
}

#[test]
fn apply_morph_strong_signal_costs_energy() {
    let mut c = base_cell();
    c.last_outputs[3] = 1.0;
    c.last_outputs[4] = 1.0;
    c.last_outputs[8] = 1.0;
    let pre = c.energy;
    c.apply_morph(1.0);
    assert!(c.energy < pre);
}

#[test]
fn apply_shell_absorb_clamps_floor_at_zero() {
    let mut c = base_cell();
    c.genome.shell_thickness = 0.5;
    c.phenotype.shell_thickness = 0.5;
    c.damage_accum = 0.05;
    c.apply_shell_absorb(1.0);
    assert_eq!(c.damage_accum, 0.0);
}

#[test]
fn apply_shell_absorb_partial_when_high_damage() {
    let mut c = base_cell();
    c.genome.shell_thickness = 0.5;
    c.phenotype.shell_thickness = 0.5;
    c.damage_accum = 5.0;
    c.apply_shell_absorb(1.0);
    let expected = 5.0 - 0.5 * SHELL_ABSORB_PER_TICK;
    assert!((c.damage_accum - expected).abs() < 1e-4);
}

#[test]
fn try_eat_returns_false_when_no_overlap() {
    let mut c = base_cell();
    let f = Food {
        position: [1000.0, 1000.0, 0.0],
        age_ticks: 0,
        kind: FoodKind::default(),
    };
    let pre = c.energy;
    assert!(!c.try_eat(&f, 8.0, 20.0));
    assert_eq!(c.energy, pre);
}

#[test]
fn try_eat_with_spike_grab_extends_reach() {
    let mut c = base_cell();
    c.genome.spike_count = 1;
    c.genome.spikes[0].length = 1.5;
    c.phenotype = Phenotype::from_genome(&c.genome);
    // Place food at tip of spike (forward) but outside ellipsoid.
    let eff_r = c.phenotype.effective_radius();
    let tip_dist = eff_r + 1.5;
    let f = Food {
        position: [tip_dist, 0.0, 0.0],
        age_ticks: 0,
        kind: FoodKind::default(),
    };
    assert!(c.eat_test_with_spikes(&f, 8.0));
}

#[test]
fn spike_bonus_scales_with_spike_length() {
    let mut a = base_cell();
    a.genome.spike_count = 1;
    a.genome.spikes[0].length = 0.5;
    a.phenotype = Phenotype::from_genome(&a.genome);
    let mut b = base_cell();
    b.genome.spike_count = 1;
    b.genome.spikes[0].length = 1.5;
    b.phenotype = Phenotype::from_genome(&b.genome);
    let target = [10.0, 0.0, 0.0];
    let bonus_a = a.spike_bonus_against(target);
    let bonus_b = b.spike_bonus_against(target);
    assert!(bonus_b > bonus_a * 2.0);
}

#[test]
fn brownian_zero_dt_is_noop() {
    let mut rng = StdRng::seed_from_u64(0xE001);
    let mut c = base_cell();
    let v0 = c.velocity;
    c.apply_brownian(&mut rng, 0.0_f32, 50.0);
    assert_eq!(c.velocity, v0);
}

#[test]
fn random_cell_initial_energy_matches_constant() {
    let mut rng = StdRng::seed_from_u64(0xE002);
    let half = [1000.0, 1000.0, 50.0];
    let c = Cell::random(&mut rng, half, 0, 0, 0);
    assert!((c.energy - INITIAL_ENERGY).abs() < 1e-4);
}

#[test]
fn random_cell_position_within_bounds() {
    let mut rng = StdRng::seed_from_u64(0xE003);
    let half = [500.0, 400.0, 30.0];
    for _ in 0..50 {
        let c = Cell::random(&mut rng, half, 0, 0, 0);
        assert!(c.position[0].abs() <= half[0]);
        assert!(c.position[1].abs() <= half[1]);
        assert!(c.position[2].abs() <= half[2]);
    }
}

#[test]
fn random_cell_state_kicked_around_half() {
    let mut rng = StdRng::seed_from_u64(0xE004);
    let half = [1000.0, 1000.0, 0.0];
    for _ in 0..50 {
        let c = Cell::random(&mut rng, half, 0, 0, 0);
        assert!(c.cell_state >= 0.5 - CELL_STATE_INIT_KICK);
        assert!(c.cell_state <= 0.5 + CELL_STATE_INIT_KICK);
    }
}

// ─── genetics/genome.rs: per-gene mutation clamp / crossover ──────────────────

fn aggressive_cfg() -> MutationConfig {
    MutationConfig {
        sigma_speed: 1000.0,
        sigma_hue: 1000.0,
        sigma_vision: 1000.0,
        sigma_turn_rate: 1000.0,
        sigma_body_length: 100.0,
        sigma_body_width: 100.0,
        sigma_body_height: 100.0,
        sigma_spike_length: 100.0,
        sigma_shell: 100.0,
        sigma_brain: 0.0,
        adhesion_flip_rate: 0.0,
        sigma_bond_stiffness: 1000.0,
        sigma_bond_damping: 1000.0,
        add_neuron_rate: 0.0,
        split_link_rate: 0.0,
        remove_neuron_rate: 0.0,
        sigma_vision_fov: 100.0,
        sigma_thermal_optimum: 1000.0,
        sigma_carnivore_score: 100.0,
        sigma_sensor_gain: 100.0,
        spike_count_mutation_rate: 0.0,
        sigma_spike_orientation: 100.0,
        sigma_spike_complexity: 100.0,
        sigma_spike_length_secondary: 100.0,
    }
}

#[test]
fn mutate_clamps_bond_stiffness() {
    let mut rng = StdRng::seed_from_u64(0xF001);
    let g = dummy_genome();
    let cfg = aggressive_cfg();
    for _ in 0..200 {
        let m = g.mutate(&mut rng, &cfg);
        assert!((MIN_BOND_STIFFNESS..=MAX_BOND_STIFFNESS).contains(&m.bond_stiffness));
    }
}

#[test]
fn mutate_clamps_bond_damping() {
    let mut rng = StdRng::seed_from_u64(0xF002);
    let g = dummy_genome();
    let cfg = aggressive_cfg();
    for _ in 0..200 {
        let m = g.mutate(&mut rng, &cfg);
        assert!((MIN_BOND_DAMPING..=MAX_BOND_DAMPING).contains(&m.bond_damping));
    }
}

#[test]
fn mutate_clamps_carnivore_score_to_unit() {
    let mut rng = StdRng::seed_from_u64(0xF003);
    let g = dummy_genome();
    let cfg = aggressive_cfg();
    for _ in 0..200 {
        let m = g.mutate(&mut rng, &cfg);
        assert!((0.0..=1.0).contains(&m.carnivore_score));
    }
}

#[test]
fn mutate_clamps_sensor_gains() {
    let mut rng = StdRng::seed_from_u64(0xF004);
    let g = dummy_genome();
    let cfg = aggressive_cfg();
    for _ in 0..100 {
        let m = g.mutate(&mut rng, &cfg);
        for &gain in &m.sensor_gains {
            assert!((MIN_SENSOR_GAIN..=MAX_SENSOR_GAIN).contains(&gain));
        }
    }
}

#[test]
fn mutate_clamps_shell_thickness() {
    let mut rng = StdRng::seed_from_u64(0xF005);
    let g = dummy_genome();
    let cfg = aggressive_cfg();
    for _ in 0..200 {
        let m = g.mutate(&mut rng, &cfg);
        assert!((MIN_SHELL_THICKNESS..=MAX_SHELL_THICKNESS).contains(&m.shell_thickness));
    }
}

#[test]
fn mutate_clamps_body_height() {
    let mut rng = StdRng::seed_from_u64(0xF006);
    let g = dummy_genome();
    let cfg = aggressive_cfg();
    for _ in 0..200 {
        let m = g.mutate(&mut rng, &cfg);
        assert!((MIN_BODY_HEIGHT..=MAX_BODY_HEIGHT).contains(&m.body_height));
    }
}

#[test]
fn mutate_zero_carnivore_sigma_skips_rng_draw() {
    let mut rng_zero = StdRng::seed_from_u64(0xCAFE_F00D);
    let mut rng_active = StdRng::seed_from_u64(0xCAFE_F00D);
    let cfg_zero = MutationConfig {
        sigma_carnivore_score: 0.0,
        ..MUTATION_CONFIG
    };
    let cfg_active = MutationConfig {
        sigma_carnivore_score: 0.02,
        ..MUTATION_CONFIG
    };
    let g = dummy_genome();
    let _ = g.mutate(&mut rng_zero, &cfg_zero);
    let _ = g.mutate(&mut rng_active, &cfg_active);
    let _: u32 = rng_zero.random();
    let _: u32 = rng_zero.random();
    let next_zero: u32 = rng_zero.random();
    let next_active: u32 = rng_active.random();
    assert_eq!(next_zero, next_active);
}

#[test]
fn mutate_zero_thermal_sigma_skips_rng_draw() {
    let mut rng_zero = StdRng::seed_from_u64(0xBEEF_0001);
    let mut rng_active = StdRng::seed_from_u64(0xBEEF_0001);
    let cfg_zero = MutationConfig {
        sigma_thermal_optimum: 0.0,
        ..MUTATION_CONFIG
    };
    let cfg_active = MutationConfig {
        sigma_thermal_optimum: 0.5,
        ..MUTATION_CONFIG
    };
    let g = dummy_genome();
    let _ = g.mutate(&mut rng_zero, &cfg_zero);
    let _ = g.mutate(&mut rng_active, &cfg_active);
    let _: u32 = rng_zero.random();
    let _: u32 = rng_zero.random();
    let next_zero: u32 = rng_zero.random();
    let next_active: u32 = rng_active.random();
    assert_eq!(next_zero, next_active);
}

#[test]
fn mutate_spike_count_zero_rate_keeps_value() {
    let mut rng = StdRng::seed_from_u64(0xFA00);
    let mut g = dummy_genome();
    g.spike_count = 3;
    let cfg = MutationConfig {
        spike_count_mutation_rate: 0.0,
        ..zero_cfg()
    };
    for _ in 0..50 {
        let m = g.mutate(&mut rng, &cfg);
        assert_eq!(m.spike_count, 3);
    }
}

#[test]
fn mutate_spike_count_full_rate_clamps_to_slot_range() {
    let mut rng = StdRng::seed_from_u64(0xFA01);
    let mut g = dummy_genome();
    g.spike_count = 0;
    let cfg = MutationConfig {
        spike_count_mutation_rate: 1.0,
        ..zero_cfg()
    };
    for _ in 0..200 {
        let m = g.mutate(&mut rng, &cfg);
        assert!(m.spike_count <= SPIKE_SLOTS as u8);
    }
}

#[test]
fn mutate_spike_orientation_clamps_per_slot() {
    let mut rng = StdRng::seed_from_u64(0xFA02);
    let g = dummy_genome();
    let cfg = MutationConfig {
        sigma_spike_orientation: 100.0,
        ..zero_cfg()
    };
    for _ in 0..100 {
        let m = g.mutate(&mut rng, &cfg);
        for s in m.spikes.iter() {
            assert!((MIN_SPIKE_AZIMUTH..=MAX_SPIKE_AZIMUTH).contains(&s.azimuth_offset));
            assert!((MIN_SPIKE_ELEVATION..=MAX_SPIKE_ELEVATION).contains(&s.elevation_offset));
        }
    }
}

#[test]
fn mutate_spike_complexity_clamps_to_unit_range() {
    let mut rng = StdRng::seed_from_u64(0xFA03);
    let g = dummy_genome();
    let cfg = MutationConfig {
        sigma_spike_complexity: 100.0,
        ..zero_cfg()
    };
    for _ in 0..100 {
        let m = g.mutate(&mut rng, &cfg);
        for s in m.spikes.iter() {
            assert!((MIN_SPIKE_COMPLEXITY..=MAX_SPIKE_COMPLEXITY).contains(&s.complexity));
        }
    }
}

#[test]
fn mutate_spike_length_secondary_clamps() {
    let mut rng = StdRng::seed_from_u64(0xFA04);
    let g = dummy_genome();
    let cfg = MutationConfig {
        sigma_spike_length_secondary: 100.0,
        ..zero_cfg()
    };
    for _ in 0..100 {
        let m = g.mutate(&mut rng, &cfg);
        for i in 1..SPIKE_SLOTS {
            assert!((MIN_SPIKE_LENGTH..=MAX_SPIKE_LENGTH).contains(&m.spikes[i].length));
        }
    }
}

#[test]
fn crossover_carnivore_score_picks_either_parent() {
    let mut rng = StdRng::seed_from_u64(0xCC01);
    let mut a = dummy_genome();
    let mut b = dummy_genome();
    a.carnivore_score = 0.1;
    b.carnivore_score = 0.9;
    let mut saw_a = false;
    let mut saw_b = false;
    for _ in 0..200 {
        let c = Genome::crossover(&a, &b, &mut rng);
        if (c.carnivore_score - 0.1).abs() < 1e-6 {
            saw_a = true;
        }
        if (c.carnivore_score - 0.9).abs() < 1e-6 {
            saw_b = true;
        }
    }
    assert!(saw_a && saw_b, "both parent values should appear");
}

#[test]
fn crossover_sensor_gains_per_category() {
    let mut rng = StdRng::seed_from_u64(0xCC02);
    let mut a = dummy_genome();
    let mut b = dummy_genome();
    a.sensor_gains = [0.0, 0.0, 0.0];
    b.sensor_gains = [2.0, 2.0, 2.0];
    for _ in 0..200 {
        let c = Genome::crossover(&a, &b, &mut rng);
        for k in 0..N_SENSOR_CATEGORIES {
            assert!(c.sensor_gains[k] == 0.0 || c.sensor_gains[k] == 2.0);
        }
    }
}

#[test]
fn crossover_bond_stiffness_picks_either_parent() {
    let mut rng = StdRng::seed_from_u64(0xCC03);
    let mut a = dummy_genome();
    let mut b = dummy_genome();
    a.bond_stiffness = MIN_BOND_STIFFNESS;
    b.bond_stiffness = MAX_BOND_STIFFNESS;
    for _ in 0..100 {
        let c = Genome::crossover(&a, &b, &mut rng);
        assert!(c.bond_stiffness == a.bond_stiffness || c.bond_stiffness == b.bond_stiffness);
    }
}

#[test]
fn crossover_bond_damping_picks_either_parent() {
    let mut rng = StdRng::seed_from_u64(0xCC04);
    let mut a = dummy_genome();
    let mut b = dummy_genome();
    a.bond_damping = MIN_BOND_DAMPING;
    b.bond_damping = MAX_BOND_DAMPING;
    for _ in 0..100 {
        let c = Genome::crossover(&a, &b, &mut rng);
        assert!(c.bond_damping == a.bond_damping || c.bond_damping == b.bond_damping);
    }
}

#[test]
fn crossover_adhesion_type_picks_either_parent() {
    let mut rng = StdRng::seed_from_u64(0xCC05);
    let mut a = dummy_genome();
    let mut b = dummy_genome();
    a.adhesion_type = 0;
    b.adhesion_type = 7;
    for _ in 0..100 {
        let c = Genome::crossover(&a, &b, &mut rng);
        assert!(c.adhesion_type == 0 || c.adhesion_type == 7);
    }
}

#[test]
fn crossover_body_height_picks_either_parent() {
    let mut rng = StdRng::seed_from_u64(0xCC06);
    let mut a = dummy_genome();
    let mut b = dummy_genome();
    a.body_height = MIN_BODY_HEIGHT;
    b.body_height = MAX_BODY_HEIGHT;
    for _ in 0..100 {
        let c = Genome::crossover(&a, &b, &mut rng);
        assert!(c.body_height == a.body_height || c.body_height == b.body_height);
    }
}

#[test]
fn crossover_spike_count_keeps_value_when_parents_equal() {
    let mut rng = StdRng::seed_from_u64(0xCC07);
    let mut a = dummy_genome();
    let mut b = dummy_genome();
    a.spike_count = 2;
    b.spike_count = 2;
    for _ in 0..50 {
        let c = Genome::crossover(&a, &b, &mut rng);
        assert_eq!(c.spike_count, 2);
    }
}

#[test]
fn crossover_spike_count_picks_either_parent_when_different() {
    let mut rng = StdRng::seed_from_u64(0xCC08);
    let mut a = dummy_genome();
    let mut b = dummy_genome();
    a.spike_count = 1;
    b.spike_count = 4;
    for _ in 0..100 {
        let c = Genome::crossover(&a, &b, &mut rng);
        assert!(c.spike_count == 1 || c.spike_count == 4);
    }
}

// ─── Brain structural mutation edge cases ─────────────────────────────────────

#[test]
fn brain_add_neuron_fails_at_storage_cap() {
    let mut rng = StdRng::seed_from_u64(0xB001);
    let mut brain = dummy_brain();
    brain.hidden_n = BRAIN_HIDDEN as u32;
    let added = brain.add_neuron(&mut rng, 0.1);
    assert!(!added);
    assert_eq!(brain.hidden_n, BRAIN_HIDDEN as u32);
}

#[test]
fn brain_remove_neuron_fails_at_min_floor() {
    let mut rng = StdRng::seed_from_u64(0xB002);
    let mut brain = dummy_brain();
    brain.hidden_n = BRAIN_HIDDEN_MIN as u32;
    let removed = brain.remove_neuron(&mut rng);
    assert!(!removed);
    assert_eq!(brain.hidden_n, BRAIN_HIDDEN_MIN as u32);
}

#[test]
fn brain_remove_neuron_decrements_hidden_n() {
    let mut rng = StdRng::seed_from_u64(0xB003);
    let mut brain = Brain::random(&mut rng);
    let pre = brain.hidden_n;
    let removed = brain.remove_neuron(&mut rng);
    assert!(removed);
    assert_eq!(brain.hidden_n, pre - 1);
}

#[test]
fn brain_split_link_returns_false_with_no_active_links() {
    let mut rng = StdRng::seed_from_u64(0xB004);
    let mut brain = dummy_brain();
    let split = brain.split_link(&mut rng, 0.05);
    assert!(!split);
}

#[test]
fn brain_split_link_grows_hidden_n_when_link_exists() {
    let mut rng = StdRng::seed_from_u64(0xB005);
    let mut brain = dummy_brain();
    brain.w1[0][0] = 1.0; // active link
    let pre = brain.hidden_n;
    let split = brain.split_link(&mut rng, 0.05);
    assert!(split);
    assert_eq!(brain.hidden_n, pre + 1);
    // direct path zeroed
    assert_eq!(brain.w1[0][0], 0.0);
}

#[test]
fn brain_split_link_fails_at_storage_cap() {
    let mut rng = StdRng::seed_from_u64(0xB006);
    let mut brain = dummy_brain();
    brain.hidden_n = BRAIN_HIDDEN as u32;
    brain.w1[0][0] = 1.0;
    let split = brain.split_link(&mut rng, 0.05);
    assert!(!split);
}

#[test]
fn brain_add_neuron_increments_hidden_n_under_cap() {
    let mut rng = StdRng::seed_from_u64(0xB007);
    let mut brain = dummy_brain();
    let pre = brain.hidden_n;
    let added = brain.add_neuron(&mut rng, 0.1);
    assert!(added);
    assert_eq!(brain.hidden_n, pre + 1);
}
