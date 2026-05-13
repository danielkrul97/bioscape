//! Fáze 2 — domain logic: cell.rs další větve, genetics/genome.rs
//! kompletní crossover.

#![allow(unused_imports)]

use crate::test_helpers::*;
use crate::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

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
    // Bond inbox (27, 28), pheromone ch1/ch2 reserved (21..27), and recurrent
    // slots (33..78) currently fall through to `None` — they are not gained.
    // Slot 30 used to be in this list before V7; now it is mechano.
    for slot in [21usize, 27, 28, 50, 52, BRAIN_INPUTS - 1] {
        assert_eq!(sensor_slot_category(slot), None);
    }
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
    let mut c = base_cell();
    let v0 = c.velocity;
    c.apply_brownian(0.0_f32, 50.0);
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
        sigma_learning_rate: 100.0,
        sigma_trace_decay: 100.0,
        model_flip_rate: 1.0,
        sigma_stdp_a: 100.0,
        sigma_stdp_tau: 100.0,
        sigma_reproduce_at_energy: 100.0,
        sigma_birth_energy: 100.0,
        sigma_altruism_share_frac: 100.0,
        sigma_cluster_share_bonus: 100.0,
        sigma_attack_gate: 100.0,
        sigma_predation_size_ratio: 100.0,
        sigma_defense_contribution: 100.0,
        sigma_reward_weights: [100.0; N_REWARD_KINDS],
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
    a.sensor_gains = [0.0; N_SENSOR_CATEGORIES];
    b.sensor_gains = [2.0; N_SENSOR_CATEGORIES];
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
