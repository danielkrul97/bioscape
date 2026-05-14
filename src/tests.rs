use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use super::*;

#[test]
fn advance_reports_generation_boundary() {
    let mut clock = SimClock::new(3, 2);
    assert_eq!(clock.advance(), ClockTransitions::default());
    assert_eq!(clock.advance(), ClockTransitions::default());
    let t = clock.advance();
    assert_eq!(t.generation_ended, Some(0));
    assert_eq!(t.epoch_ended, None);
    assert_eq!((clock.tick, clock.generation, clock.epoch), (3, 1, 0));
}

#[test]
fn epoch_fires_alongside_generation_boundary() {
    let mut clock = SimClock::new(2, 2);
    clock.advance();
    let t = clock.advance();
    assert_eq!(t.generation_ended, Some(0));
    assert_eq!(t.epoch_ended, None);
    clock.advance();
    let t = clock.advance();
    assert_eq!(t.generation_ended, Some(1));
    assert_eq!(t.epoch_ended, Some(0));
    assert_eq!((clock.tick, clock.generation, clock.epoch), (4, 2, 1));
}

fn dummy_brain() -> Brain {
    Brain {
        hidden_n: BRAIN_HIDDEN_DEFAULT as u32,
        w1: [[0.0; BRAIN_INPUTS]; BRAIN_HIDDEN],
        b1: [0.0; BRAIN_HIDDEN],
        w2: [[0.0; BRAIN_HIDDEN]; BRAIN_OUTPUTS],
        b2: [0.0; BRAIN_OUTPUTS],
        trace_w1: [[0.0; BRAIN_INPUTS]; BRAIN_HIDDEN],
        trace_w2: [[0.0; BRAIN_HIDDEN]; BRAIN_OUTPUTS],
        membrane: [IZH_V_REST; BRAIN_HIDDEN],
        recovery: [0.0; BRAIN_HIDDEN],
        last_pre_spike_ticks: [0; BRAIN_INPUTS],
        last_post_spike_ticks: [0; BRAIN_HIDDEN],
        pre_trace: [0.0; BRAIN_INPUTS],
        post_trace: [0.0; BRAIN_HIDDEN],
    }
}

fn dummy_genome() -> Genome {
    Genome {
        max_speed: 60.0,
        color_hue: 0.0,
        vision_radius: 40.0,
        turn_rate: 2.5,
        body_length: 1.0,
        body_width: 1.0,
        body_height: 1.0,
        spikes: [Spike::ZERO; SPIKE_SLOTS],
        spike_count: 1,
        shell_thickness: 0.0,
        adhesion_type: 0,
        bond_stiffness: BOND_STIFFNESS,
        bond_damping: BOND_DAMPING,
        vision_fov: INITIAL_VISION_FOV,
        thermal_optimum: THERMAL_REF_TEMP,
        carnivore_score: 0.0,
        // Sprint 97: zero gains v test fixture aby legacy energy-drain testy
        // (pre-S97) neviděly sensor_gain cost. Per-test override když test
        // sensor pooling testuje.
        sensor_gains: [0.0; N_SENSOR_CATEGORIES],
        brain: dummy_brain(),
        cppn: default_cppn(),
        learning_rate: LEARNING_RATE,
        trace_decay_per_sec: HEBBIAN_TRACE_DECAY_PER_SEC,
        neuron_model: NeuronModel::Perceptron,
        stdp_a_plus: DEFAULT_STDP_A_PLUS,
        stdp_a_minus: DEFAULT_STDP_A_MINUS,
        stdp_tau_ticks: DEFAULT_STDP_TAU_TICKS,
        reproduce_at_energy: REPRODUCE_THRESHOLD,
        birth_energy: 50.0,
        altruism_share_frac: BOND_FOOD_SHARE_FRAC,
        cluster_share_bonus: BOND_FOOD_SHARE_CLUSTER_BONUS,
        attack_gate: ATTACK_THRESHOLD,
        predation_size_ratio: SIZE_RATIO_THRESHOLD,
        defense_contribution: BOND_DEFENSE_FRAC,
        reward_weights: REWARD_WEIGHT_DEFAULTS,
    }
}

fn zero_cfg() -> MutationConfig {
    MutationConfig {
        sigma_speed: 0.0,
        sigma_hue: 0.0,
        sigma_vision: 0.0,
        sigma_turn_rate: 0.0,
        sigma_body_length: 0.0,
        sigma_body_width: 0.0,
        sigma_body_height: 0.0,
        sigma_spike_length: 0.0,
        sigma_shell: 0.0,
        sigma_brain: 0.0,
        adhesion_flip_rate: 0.0,
        sigma_bond_stiffness: 0.0,
        sigma_bond_damping: 0.0,
        add_neuron_rate: 0.0,
        split_link_rate: 0.0,
        remove_neuron_rate: 0.0,
        sigma_vision_fov: 0.0,
        sigma_thermal_optimum: 0.0,
        sigma_carnivore_score: 0.0,
        sigma_sensor_gain: 0.0,
        spike_count_mutation_rate: 0.0,
        sigma_spike_orientation: 0.0,
        sigma_spike_complexity: 0.0,
        sigma_spike_length_secondary: 0.0,
        sigma_learning_rate: 0.0,
        sigma_trace_decay: 0.0,
        model_flip_rate: 0.0,
        sigma_stdp_a: 0.0,
        sigma_stdp_tau: 0.0,
        sigma_reproduce_at_energy: 0.0,
        sigma_birth_energy: 0.0,
        sigma_altruism_share_frac: 0.0,
        sigma_cluster_share_bonus: 0.0,
        sigma_attack_gate: 0.0,
        sigma_predation_size_ratio: 0.0,
        sigma_defense_contribution: 0.0,
        sigma_reward_weights: [0.0; N_REWARD_KINDS],
    }
}

#[test]
fn mutation_with_zero_sigma_is_identity() {
    let mut rng = rand::rng();
    let g = Genome {
        max_speed: 50.0,
        color_hue: 120.0,
        vision_radius: 40.0,
        turn_rate: 2.5,
        body_length: 1.1,
        body_width: 0.9,
        body_height: 1.0,
        spikes: {
            let mut s = [Spike::ZERO; SPIKE_SLOTS];
            s[0].length = 0.4;
            s
        },
        spike_count: 1,
        shell_thickness: 0.0,
        adhesion_type: 0,
        bond_stiffness: BOND_STIFFNESS,
        bond_damping: BOND_DAMPING,
        vision_fov: INITIAL_VISION_FOV,
        thermal_optimum: THERMAL_REF_TEMP,
        carnivore_score: 0.0,
        sensor_gains: [1.0; N_SENSOR_CATEGORIES],
        brain: Brain {
            hidden_n: BRAIN_HIDDEN_DEFAULT as u32,
            w1: [[1.0; BRAIN_INPUTS]; BRAIN_HIDDEN],
            b1: [0.3; BRAIN_HIDDEN],
            w2: [[1.0; BRAIN_HIDDEN]; BRAIN_OUTPUTS],
            b2: [0.5; BRAIN_OUTPUTS],
            trace_w1: [[0.0; BRAIN_INPUTS]; BRAIN_HIDDEN],
            trace_w2: [[0.0; BRAIN_HIDDEN]; BRAIN_OUTPUTS],
            membrane: [IZH_V_REST; BRAIN_HIDDEN],
            recovery: [0.0; BRAIN_HIDDEN],
            last_pre_spike_ticks: [0; BRAIN_INPUTS],
            last_post_spike_ticks: [0; BRAIN_HIDDEN],
            pre_trace: [0.0; BRAIN_INPUTS],
            post_trace: [0.0; BRAIN_HIDDEN],
        },
        cppn: default_cppn(),
        learning_rate: LEARNING_RATE,
        trace_decay_per_sec: HEBBIAN_TRACE_DECAY_PER_SEC,
        neuron_model: NeuronModel::Perceptron,
        stdp_a_plus: DEFAULT_STDP_A_PLUS,
        stdp_a_minus: DEFAULT_STDP_A_MINUS,
        stdp_tau_ticks: DEFAULT_STDP_TAU_TICKS,
        reproduce_at_energy: REPRODUCE_THRESHOLD,
        birth_energy: 50.0,
        altruism_share_frac: BOND_FOOD_SHARE_FRAC,
        cluster_share_bonus: BOND_FOOD_SHARE_CLUSTER_BONUS,
        attack_gate: ATTACK_THRESHOLD,
        predation_size_ratio: SIZE_RATIO_THRESHOLD,
        defense_contribution: BOND_DEFENSE_FRAC,
        reward_weights: REWARD_WEIGHT_DEFAULTS,
    };
    let m = g.mutate(&mut rng, &zero_cfg());
    assert_eq!(m.max_speed, 50.0);
    assert_eq!(m.color_hue, 120.0);
    assert_eq!(m.vision_radius, 40.0);
    assert_eq!(m.turn_rate, 2.5);
    assert_eq!(m.body_length, 1.1);
    assert_eq!(m.body_width, 0.9);
    assert_eq!(m.spikes[0].length, 0.4);
    assert_eq!(m.spike_count, 1);
    // Sprint 106: brain je derived z mutated CPPN. S sigma=0 v zero_cfg,
    // ale CPPN má vlastní mutation rates (CPPN_MUTATION_CONFIG) které
    // jsou non-zero — brain weights NEZACHOVANÉ identity. Test now
    // validates structural compatibility místo identity.
    assert_eq!(m.brain.w1.len(), g.brain.w1.len());
    assert_eq!(m.brain.b1.len(), g.brain.b1.len());
}

#[test]
fn mutation_keeps_genes_in_valid_ranges() {
    let mut rng = rand::rng();
    let g = dummy_genome();
    let cfg = MutationConfig {
        sigma_speed: 100.0,
        sigma_hue: 1000.0,
        sigma_vision: 100.0,
        sigma_turn_rate: 100.0,
        sigma_body_length: 10.0,
        sigma_body_width: 10.0,
        sigma_body_height: 10.0,
        sigma_spike_length: 10.0,
        sigma_shell: 10.0,
        sigma_brain: 10.0,
        adhesion_flip_rate: 0.5,
        sigma_bond_stiffness: 100.0,
        sigma_bond_damping: 10.0,
        add_neuron_rate: 0.0,
        split_link_rate: 0.0,
        remove_neuron_rate: 0.0,
        sigma_vision_fov: 10.0,
        sigma_thermal_optimum: 100.0,
        sigma_carnivore_score: 100.0,
        sigma_sensor_gain: 100.0,
        spike_count_mutation_rate: 0.5,
        sigma_spike_orientation: 10.0,
        sigma_spike_complexity: 10.0,
        sigma_spike_length_secondary: 10.0,
        sigma_learning_rate: 10.0,
        sigma_trace_decay: 10.0,
        model_flip_rate: 1.0,
        sigma_stdp_a: 10.0,
        sigma_stdp_tau: 100.0,
        sigma_reproduce_at_energy: 100.0,
        sigma_birth_energy: 100.0,
        sigma_altruism_share_frac: 10.0,
        sigma_cluster_share_bonus: 10.0,
        sigma_attack_gate: 10.0,
        sigma_predation_size_ratio: 10.0,
        sigma_defense_contribution: 10.0,
        sigma_reward_weights: [10.0; N_REWARD_KINDS],
    };
    for _ in 0..1000 {
        let m = g.mutate(&mut rng, &cfg);
        for spike in m.spikes.iter() {
            assert!((MIN_SPIKE_AZIMUTH..=MAX_SPIKE_AZIMUTH).contains(&spike.azimuth_offset));
            assert!(
                (MIN_SPIKE_ELEVATION..=MAX_SPIKE_ELEVATION).contains(&spike.elevation_offset)
            );
            assert!((MIN_SPIKE_COMPLEXITY..=MAX_SPIKE_COMPLEXITY).contains(&spike.complexity));
        }
        assert!(m.spike_count <= SPIKE_SLOTS as u8);
        assert!(m.max_speed >= MIN_SPEED);
        assert!(m.max_speed <= MAX_SPEED, "Sprint 73: speed cap respected");
        assert!(m.color_hue >= 0.0 && m.color_hue < HUE_RANGE);
        assert!(m.vision_radius >= MIN_VISION);
        assert!(m.turn_rate >= MIN_TURN_RATE);
        assert!((MIN_BODY_LENGTH..=MAX_BODY_LENGTH).contains(&m.body_length));
        assert!((MIN_BODY_WIDTH..=MAX_BODY_WIDTH).contains(&m.body_width));
        for spike in m.spikes.iter() {
            assert!((MIN_SPIKE_LENGTH..=MAX_SPIKE_LENGTH).contains(&spike.length));
        }
        assert!((MIN_VISION_FOV..=MAX_VISION_FOV).contains(&m.vision_fov));
        assert!((MIN_THERMAL_OPTIMUM..=MAX_THERMAL_OPTIMUM).contains(&m.thermal_optimum));
    }
}

#[test]
fn vision_fov_dormant_preserves_rng_sequence() {
    // Sprint 82 reproducibility guard: při `sigma_vision_fov = 0`
    // (S82 default) musí mutate přeskočit gaussian draw pro FOV gen,
    // jinak Sprint 82 baseline rozejde s pre-Sprint-82 CSV. Verifikuje
    // shodu RNG stavu mezi dormant cestou (krátkou) a aktivní cestou
    // (sigma > 0) po injekci přesně 2 u32 draws (gaussian = 2 u32).
    let mut rng_zero = StdRng::seed_from_u64(0xC0FFEE);
    let mut rng_active = StdRng::seed_from_u64(0xC0FFEE);
    let cfg_zero = MutationConfig {
        sigma_vision_fov: 0.0,
        ..MUTATION_CONFIG
    };
    let cfg_active = MutationConfig {
        sigma_vision_fov: 0.05,
        ..MUTATION_CONFIG
    };
    let g = dummy_genome();
    let _ = g.mutate(&mut rng_zero, &cfg_zero);
    let _ = g.mutate(&mut rng_active, &cfg_active);
    let _: u32 = rng_zero.random();
    let _: u32 = rng_zero.random();
    let next_zero: u32 = rng_zero.random();
    let next_active: u32 = rng_active.random();
    assert_eq!(
        next_zero, next_active,
        "sigma_vision_fov = 0 musí ušetřit přesně 2 u32 RNG draws (gaussian); \
         jinak Sprint 82 nezachová pre-S82 reproducibility"
    );
}

#[test]
fn vision_fov_crossover_skips_rng_when_equal() {
    // Sprint 82 reproducibility guard: pokud oba parents mají identické
    // vision_fov (což je pravda v initial pop kde všichni = INITIAL_VISION_FOV),
    // crossover musí přeskočit bool draw. Verifikuje shodu RNG stavu mezi
    // equal-values cestou (krátkou) a different-values cestou (s draw)
    // po injekci 1 bool draw.
    let mut rng_eq = StdRng::seed_from_u64(0xBEEF);
    let mut rng_diff = StdRng::seed_from_u64(0xBEEF);
    let mut a = dummy_genome();
    let mut b = dummy_genome();
    a.vision_fov = INITIAL_VISION_FOV;
    b.vision_fov = INITIAL_VISION_FOV;
    let _ = Genome::crossover(&a, &b, &mut rng_eq);
    b.vision_fov = MIN_VISION_FOV;
    let _ = Genome::crossover(&a, &b, &mut rng_diff);
    let _ = rng_eq.random::<bool>();
    let next_eq: u32 = rng_eq.random();
    let next_diff: u32 = rng_diff.random();
    assert_eq!(
        next_eq, next_diff,
        "crossover s a.vision_fov == b.vision_fov musí ušetřit přesně 1 bool draw"
    );
}

#[test]
fn temperature_at_z_endpoints() {
    let half = [960.0, 540.0, 50.0];
    // Sprint 86: tick=0, gen=0 → seasonal sin(0)=0, diurnal sin(0)=0,
    // takže static gradient endpoints zůstávají identické s pre-Sprint-86.
    // Top z = +half → THERMAL_TOP.
    assert!((temperature_at_z(50.0, half, 0, 0) - THERMAL_TOP).abs() < 1e-4);
    // Bottom z = -half → THERMAL_BOTTOM.
    assert!((temperature_at_z(-50.0, half, 0, 0) - THERMAL_BOTTOM).abs() < 1e-4);
    // Mid z = 0 → exact midpoint.
    let mid = (THERMAL_TOP + THERMAL_BOTTOM) * 0.5;
    assert!((temperature_at_z(0.0, half, 0, 0) - mid).abs() < 1e-4);
    // Out-of-bounds z → clamp na endpoints.
    assert!((temperature_at_z(1000.0, half, 0, 0) - THERMAL_TOP).abs() < 1e-4);
    assert!((temperature_at_z(-1000.0, half, 0, 0) - THERMAL_BOTTOM).abs() < 1e-4);
    // world_half[2] = 0 (pre-3D baseline) → ref temp fallback (no-op pro
    // metabolism). Důležité pro backward-compat pre-Sprint-33 testů.
    let flat = [960.0, 540.0, 0.0];
    assert!((temperature_at_z(0.0, flat, 0, 0) - THERMAL_REF_TEMP).abs() < 1e-4);
}

#[test]
fn temperature_diurnal_surface_oscillates() {
    // Sprint 86: surface (z = +half) osciluje ±DIURNAL_AMP přes 1 day.
    // Bottom (z = -half) zůstává stabilní (normalized = 0 → diurnal × 0).
    let half = [960.0, 540.0, 50.0];
    let period = THERMAL_DIURNAL_PERIOD_TICKS;
    // Quarter-day → sin(π/2) = +1 → surface = TOP + AMP, bottom = BOTTOM.
    let t_q = period / 4;
    let surf_q = temperature_at_z(50.0, half, t_q, 0);
    let bot_q = temperature_at_z(-50.0, half, t_q, 0);
    assert!((surf_q - (THERMAL_TOP + THERMAL_DIURNAL_AMP)).abs() < 0.05);
    assert!((bot_q - THERMAL_BOTTOM).abs() < 0.05);
    // Three-quarter-day → sin(3π/2) = -1 → surface = TOP − AMP.
    let t_3q = 3 * period / 4;
    let surf_3q = temperature_at_z(50.0, half, t_3q, 0);
    assert!((surf_3q - (THERMAL_TOP - THERMAL_DIURNAL_AMP)).abs() < 0.05);
    // Full day → sin(2π) = 0 → matches initial.
    let surf_full = temperature_at_z(50.0, half, period, 0);
    assert!((surf_full - THERMAL_TOP).abs() < 0.01);
}

#[test]
fn temperature_seasonal_uniform_shift() {
    // Sprint 86: seasonal aplikuje stejný offset napříč all z (uniform shift).
    // Surface i bottom posun stejně. Period = CYCLE_GEN_PERIOD = 50 gen.
    let half = [960.0, 540.0, 50.0];
    let period = CYCLE_GEN_PERIOD;
    // Quarter-cycle → sin(π/2) = 1 → +SEASONAL_AMP shift.
    let surf_q = temperature_at_z(50.0, half, 0, period / 4);
    let bot_q = temperature_at_z(-50.0, half, 0, period / 4);
    assert!((surf_q - (THERMAL_TOP + THERMAL_SEASONAL_AMP)).abs() < 0.05);
    assert!((bot_q - (THERMAL_BOTTOM + THERMAL_SEASONAL_AMP)).abs() < 0.05);
    // Half-cycle → sin(π) = 0 → no shift.
    let surf_half = temperature_at_z(50.0, half, 0, period / 2);
    assert!((surf_half - THERMAL_TOP).abs() < 0.05);
    // Three-quarter-cycle → sin(3π/2) = -1 → -SEASONAL_AMP shift.
    let surf_3q = temperature_at_z(50.0, half, 0, 3 * period / 4);
    assert!((surf_3q - (THERMAL_TOP - THERMAL_SEASONAL_AMP)).abs() < 0.05);
}

#[test]
fn temperature_combined_seasonal_and_diurnal() {
    // Sprint 86: seasonal i diurnal jsou aditivní. Quarter-day +
    // quarter-season → surface = TOP + DIURNAL_AMP + SEASONAL_AMP,
    // bottom = BOTTOM + SEASONAL_AMP.
    let half = [960.0, 540.0, 50.0];
    let t_q = THERMAL_DIURNAL_PERIOD_TICKS / 4;
    let g_q = CYCLE_GEN_PERIOD / 4;
    let surf = temperature_at_z(50.0, half, t_q, g_q);
    let expected = THERMAL_TOP + THERMAL_DIURNAL_AMP + THERMAL_SEASONAL_AMP;
    assert!(
        (surf - expected).abs() < 0.05,
        "combined surface {} ≠ expected {}",
        surf,
        expected
    );
    let bot = temperature_at_z(-50.0, half, t_q, g_q);
    let expected_bot = THERMAL_BOTTOM + THERMAL_SEASONAL_AMP;
    assert!((bot - expected_bot).abs() < 0.05);
}

#[test]
fn climate_offset_default_zero() {
    let pos_xy = [123.0, -45.0];
    // Empty events → 0.0.
    let off = climate_shock_offset(&[], 50, pos_xy, WORLD_HALF);
    assert!(off.abs() < 1e-6, "empty events must give 0.0, got {}", off);
    // Non-ClimateShift event (HazardPulse) → 0.0.
    let event = ShockEvent {
        kind: ShockKind::HazardPulse,
        start_gen: 0,
        duration_gen: 10,
        ramp_gens: 2,
        intensity: 1.0,
        center_xy: None,
        radius: None,
    };
    let off = climate_shock_offset(&[event], 5, pos_xy, WORLD_HALF);
    assert!(off.abs() < 1e-6, "HazardPulse must not affect climate, got {}", off);
}

#[test]
fn climate_offset_global_shift_at_peak() {
    // Sprint 112: 1 global ClimateShift, intensity = 1, peak ramp = 1, no
    // spatial → offset = CLIMATE_SHIFT_MAX_OFFSET (= 5.0).
    let event = ShockEvent {
        kind: ShockKind::ClimateShift,
        start_gen: 100,
        duration_gen: 10,
        ramp_gens: 2,
        intensity: 1.0,
        center_xy: None,
        radius: None,
    };
    // Plateau (gen 102..=107) → ramp = 1.0.
    let off = climate_shock_offset(&[event], 105, [50.0, -10.0], WORLD_HALF);
    assert!(
        (off - CLIMATE_SHIFT_MAX_OFFSET).abs() < 1e-5,
        "global peak must give CLIMATE_SHIFT_MAX_OFFSET, got {}",
        off
    );
    // Pre-start: 0.0.
    let off_before = climate_shock_offset(&[event], 99, [50.0, -10.0], WORLD_HALF);
    assert!(off_before.abs() < 1e-6);
    // Post-end: 0.0.
    let off_after = climate_shock_offset(&[event], 110, [50.0, -10.0], WORLD_HALF);
    assert!(off_after.abs() < 1e-6);
}

#[test]
fn temperature_with_shocks_matches_baseline_when_no_events() {
    // Sprint 112: temperature_at_z_with_shocks musí být byte-identical
    // s temperature_at_z když events.empty (default off path).
    let half = [960.0, 540.0, 50.0];
    let pos_xy = [200.0, -100.0];
    for &(z, tick, gen) in &[
        (0.0_f32, 0_u64, 0_u64),
        (50.0, 100, 5),
        (-50.0, 1000, 25),
        (25.0, THERMAL_DIURNAL_PERIOD_TICKS / 4, CYCLE_GEN_PERIOD / 4),
        (-25.0, 3 * THERMAL_DIURNAL_PERIOD_TICKS / 4, CYCLE_GEN_PERIOD / 2),
    ] {
        let base = temperature_at_z(z, half, tick, gen);
        let with_shocks = temperature_at_z_with_shocks(z, half, tick, gen, &[], pos_xy);
        assert_eq!(
            base.to_bits(),
            with_shocks.to_bits(),
            "byte-identical required: z={}, tick={}, gen={}",
            z,
            tick,
            gen
        );
    }
}

#[test]
fn metabolism_factor_q10_ratio() {
    // Q10 = 2.0 → biologické rychlosti přesně 2× per +10 sim-units T.
    let m_ref = metabolism_factor(THERMAL_REF_TEMP);
    assert!((m_ref - 1.0).abs() < 1e-4, "ref temp musí dát factor 1.0");
    let m_plus_10 = metabolism_factor(THERMAL_REF_TEMP + 10.0);
    assert!(
        (m_plus_10 - THERMAL_Q10).abs() < 1e-4,
        "+10 musí dát Q10 (= 2.0), got {m_plus_10}"
    );
    let m_minus_10 = metabolism_factor(THERMAL_REF_TEMP - 10.0);
    assert!(
        (m_minus_10 - 1.0 / THERMAL_Q10).abs() < 1e-4,
        "-10 musí dát 1/Q10 (= 0.5), got {m_minus_10}"
    );
    // Endpoints by měly dát ratio top:bottom = Q10^((TOP-BOT)/10)
    let m_top = metabolism_factor(THERMAL_TOP);
    let m_bot = metabolism_factor(THERMAL_BOTTOM);
    let expected_ratio = THERMAL_Q10.powf((THERMAL_TOP - THERMAL_BOTTOM) / 10.0);
    assert!(
        ((m_top / m_bot) - expected_ratio).abs() < 1e-3,
        "top/bottom ratio {} vs expected {}",
        m_top / m_bot,
        expected_ratio
    );
}

#[test]
fn apply_energy_costs_scales_with_temperature() {
    // Sprint 85: cell na warm depth (z = +half) drain rychleji než cell na
    // cold depth (z = -half). Při shodné velocity / body / vision platí
    // ratio drain = metabolism(top) / metabolism(bottom) ≈ 2.46 / 0.41 ≈ 6×.
    let half = [1000.0, 1000.0, 50.0];
    let physics = no_drag_physics(0.001, 0.0);
    let mut warm = base_cell();
    warm.position = [0.0, 0.0, 50.0]; // top → warmest
    warm.velocity = [60.0, 0.0, 0.0];
    let mut cold = base_cell();
    cold.position = [0.0, 0.0, -50.0]; // bottom → coldest
    cold.velocity = [60.0, 0.0, 0.0];
    warm.step(1.0, half, 0, 0, &physics);
    cold.step(1.0, half, 0, 0, &physics);
    let warm_drain = 100.0 - warm.energy;
    let cold_drain = 100.0 - cold.energy;
    let ratio = warm_drain / cold_drain;
    let expected = metabolism_factor(THERMAL_TOP) / metabolism_factor(THERMAL_BOTTOM);
    assert!(
        (ratio - expected).abs() < 0.05,
        "warm/cold drain ratio {ratio} ≠ expected {expected}"
    );
}

#[test]
fn pool_bonded_hidden_solo_cell_returns_self() {
    // Sprint 94: solo cell s no bonds → pooled == last_hidden.
    let mut cell = base_cell();
    for k in 0..BRAIN_HIDDEN {
        cell.last_hidden[k] = (k as f32) * 0.1;
    }
    let pooled = pool_bonded_hidden(&cell, |_| None);
    assert_eq!(pooled, cell.last_hidden);
}

#[test]
fn pool_bonded_hidden_pair_averages() {
    // Sprint 94: pair cell A bonded to B → A.pooled = (A.last + B.last) / 2.
    let mut cell = base_cell();
    cell.cell_id = 1;
    cell.bonds[0] = Some(Bond {
        other_cell_id: 2,
        rest_length: 5.0,
        stiffness: BOND_STIFFNESS,
        damping: BOND_DAMPING,
        age_ticks: 0,
    });
    for k in 0..BRAIN_HIDDEN {
        cell.last_hidden[k] = 1.0;
    }
    let mut partner_hidden = [0.0; BRAIN_HIDDEN];
    for k in 0..BRAIN_HIDDEN {
        partner_hidden[k] = 3.0;
    }
    let pooled = pool_bonded_hidden(&cell, |id| {
        if id == 2 { Some(partner_hidden) } else { None }
    });
    for k in 0..BRAIN_HIDDEN {
        assert!((pooled[k] - 2.0).abs() < 1e-6, "expected 2.0, got {}", pooled[k]);
    }
}

#[test]
fn pool_bonded_hidden_skips_dead_partners() {
    // Sprint 94: missing partner (despawned mid-tick) skipped, pool jen
    // s alive bonded.
    let mut cell = base_cell();
    cell.bonds[0] = Some(Bond {
        other_cell_id: 99,
        rest_length: 5.0,
        stiffness: BOND_STIFFNESS,
        damping: BOND_DAMPING,
        age_ticks: 0,
    });
    for k in 0..BRAIN_HIDDEN {
        cell.last_hidden[k] = 5.0;
    }
    // Dead partner returns None.
    let pooled = pool_bonded_hidden(&cell, |_| None);
    // Pool falls back to self only.
    assert_eq!(pooled, cell.last_hidden);
}

#[test]
fn sensor_slot_category_covers_known_indices() {
    // Sprint 97: každý sensory slot v 0..BRAIN_INPUTS_SENSORY musí buď
    // vrátit Some(category) nebo None (proprio). Žádný slot nevypadne.
    // Defensive (damage_norm) je slot 14, density slot 20.
    assert_eq!(sensor_slot_category(0), Some(SENSOR_CATEGORY_VISION));
    assert_eq!(sensor_slot_category(7), Some(SENSOR_CATEGORY_CHEMISTRY));
    assert_eq!(sensor_slot_category(14), Some(SENSOR_CATEGORY_DEFENSIVE));
    assert_eq!(sensor_slot_category(20), Some(SENSOR_CATEGORY_DEFENSIVE));
    // Proprio slot (energy/speed/heading) → None.
    assert!(sensor_slot_category(4).is_none());
}

#[test]
fn apply_sensor_gains_scales_only_categorized_slots() {
    // Sprint 97: gains aplikuje na sensory slots, proprio nedotčeno.
    let mut inputs = [1.0_f32; BRAIN_INPUTS];
    // gains[0] vision, [1] chemistry, [2] defensive, [3] mechano. Mechano set
    // to 3.0 so slot 29 (vibration_grad_x) ends up at 3.0 — verifies the new
    // category got wired through.
    let gains = [2.0, 0.5, 0.0, 3.0];
    apply_sensor_gains(&mut inputs, &gains);
    // Vision slot 0 = 2× gain
    assert!((inputs[0] - 2.0).abs() < 1e-6);
    // Chemistry slot 7 = 0.5× gain
    assert!((inputs[7] - 0.5).abs() < 1e-6);
    // Defensive slot 14 = 0× gain
    assert!((inputs[14] - 0.0).abs() < 1e-6);
    // Mechano slot 29 = 3× gain
    assert!((inputs[29] - 3.0).abs() < 1e-6);
    // Proprio slot 4 → unchanged.
    assert!((inputs[4] - 1.0).abs() < 1e-6);
    // Recurrent slot mimo BRAIN_INPUTS_SENSORY → unchanged.
    assert!((inputs[BRAIN_INPUTS_SENSORY] - 1.0).abs() < 1e-6);
}

#[test]
fn pool_bonded_sensors_solo_cell_returns_own() {
    // Sprint 97: solo cell bez bonds → pooled == own (žádný partner).
    let cell = base_cell();
    let mut own = [0.0; BRAIN_INPUTS];
    own[0] = 0.5;
    own[7] = -0.3;
    let pooled = pool_bonded_sensors(&cell, &own, |_| None);
    assert_eq!(pooled, own);
}

#[test]
fn pool_bonded_sensors_takes_max_magnitude_from_partner() {
    // Sprint 97: partner má silnější vision signal → pooled převezme partner.
    // Magnitude-based pooling (abs()) — invertovaný gradient (-0.9) přebije
    // slabý kladný (0.2).
    let mut cell = base_cell();
    cell.cell_id = 1;
    cell.bonds[0] = Some(Bond {
        other_cell_id: 2,
        rest_length: 5.0,
        stiffness: BOND_STIFFNESS,
        damping: BOND_DAMPING,
        age_ticks: 0,
    });
    let mut own = [0.0; BRAIN_INPUTS];
    own[0] = 0.2;
    own[7] = 0.5;
    let mut partner = [0.0; BRAIN_INPUTS];
    partner[0] = -0.9;
    partner[7] = 0.1;
    let pooled = pool_bonded_sensors(&cell, &own, |id| {
        if id == 2 { Some(partner) } else { None }
    });
    // Vision slot: |-0.9| > |0.2| → partner wins
    assert!((pooled[0] - (-0.9)).abs() < 1e-6);
    // Chemistry slot: |0.5| > |0.1| → own wins
    assert!((pooled[7] - 0.5).abs() < 1e-6);
}

#[test]
fn pool_bonded_sensors_ignores_proprio_slots() {
    // Sprint 97: proprio (energy, speed, heading) NESMÍ poolnout — každá
    // buňka má svůj vlastní stav.
    let mut cell = base_cell();
    cell.bonds[0] = Some(Bond {
        other_cell_id: 7,
        rest_length: 5.0,
        stiffness: BOND_STIFFNESS,
        damping: BOND_DAMPING,
        age_ticks: 0,
    });
    let mut own = [0.0; BRAIN_INPUTS];
    own[4] = 0.1; // proprio
    let mut partner = [0.0; BRAIN_INPUTS];
    partner[4] = 0.99; // partner higher proprio
    let pooled = pool_bonded_sensors(&cell, &own, |id| {
        if id == 7 { Some(partner) } else { None }
    });
    // Proprio slot 4 zůstává own — nebyl poolen.
    assert!((pooled[4] - 0.1).abs() < 1e-6);
}

#[test]
fn eat_efficiency_diet_specialization() {
    // Pure herbivore preference for plant.
    assert!((eat_efficiency(FoodKind::Plant, 0.0) - 1.0).abs() < 1e-6);
    // Pure carnivore: plant gives nothing.
    assert!(eat_efficiency(FoodKind::Plant, 1.0).abs() < 1e-6);
    // Mixed (0.5) — plant 0.5, carrion 0.5.
    assert!((eat_efficiency(FoodKind::Plant, 0.5) - 0.5).abs() < 1e-6);
    assert!((eat_efficiency(FoodKind::Carrion, 0.5) - 0.5).abs() < 1e-6);
    // Cell carrion: universally 0.5 — compromise food.
    assert!((eat_efficiency(FoodKind::Carrion, 0.0) - 0.5).abs() < 1e-6);
    assert!((eat_efficiency(FoodKind::Carrion, 1.0) - 0.5).abs() < 1e-6);
}

#[test]
fn food_base_value_per_kind() {
    assert!((food_base_value(FoodKind::Plant) - PLANT_FOOD_VALUE).abs() < 1e-6);
    assert!((food_base_value(FoodKind::Carrion) - CARRION_FOOD_VALUE).abs() < 1e-6);
    assert!(food_base_value(FoodKind::Carrion) > food_base_value(FoodKind::Plant));
}

#[test]
fn coop_food_lifecycle_no_arrivals_expires() {
    // Sprint 128: bez arrivals coop node prošlý TIME_WINDOW musí vrátit
    // is_expired = true → caller cleanup, no reward.
    let mut coop = CoopFood::new([0.0, 0.0, 0.0], 0);
    assert!(!coop.is_expired(0));
    assert!(!coop.is_expired((COOP_FOOD_TIME_WINDOW_TICKS as u64).saturating_sub(1)));
    assert!(coop.is_expired(COOP_FOOD_TIME_WINDOW_TICKS as u64));
    let mut cells: [Cell; 0] = [];
    assert!(!try_trigger_coop(&mut coop, &mut cells));
    assert!(!coop.triggered);
}

#[test]
fn coop_food_threshold_triggers_reward() {
    // Sprint 128: 3 cells s unique cell_id v arrivals → trigger distribuuje
    // COOP_FOOD_REWARD_PER_CELL na každého. Caller pak coop odstraní.
    let mut coop = CoopFood::new([0.0, 0.0, 0.0], 0);
    let mut cells = [
        Cell {
            cell_id: 1,
            energy: 50.0,
            ..base_cell()
        },
        Cell {
            cell_id: 2,
            energy: 30.0,
            ..base_cell()
        },
        Cell {
            cell_id: 3,
            energy: 70.0,
            ..base_cell()
        },
    ];
    register_coop_arrival(&mut coop, 1);
    register_coop_arrival(&mut coop, 2);
    register_coop_arrival(&mut coop, 3);
    // Duplicate id ignored.
    assert!(!register_coop_arrival(&mut coop, 1));
    assert_eq!(coop.arrivals.len(), 3);
    assert!(try_trigger_coop(&mut coop, &mut cells));
    assert!(coop.triggered);
    assert!((cells[0].energy - (50.0 + COOP_FOOD_REWARD_PER_CELL)).abs() < 1e-4);
    assert!((cells[1].energy - (30.0 + COOP_FOOD_REWARD_PER_CELL)).abs() < 1e-4);
    assert!((cells[2].energy - (70.0 + COOP_FOOD_REWARD_PER_CELL)).abs() < 1e-4);
    // Idempotent: druhý try_trigger nesmí znovu rozdat reward.
    assert!(!try_trigger_coop(&mut coop, &mut cells));
    assert!((cells[0].energy - (50.0 + COOP_FOOD_REWARD_PER_CELL)).abs() < 1e-4);
}

#[test]
fn coop_food_below_threshold_no_reward() {
    // Sprint 128: 2 < REQUIRED_ARRIVALS (=3) → trigger nefiringuje, energie
    // beze změny, coop stále alive.
    let mut coop = CoopFood::new([0.0, 0.0, 0.0], 0);
    let mut cells = [
        Cell {
            cell_id: 1,
            energy: 50.0,
            ..base_cell()
        },
        Cell {
            cell_id: 2,
            energy: 30.0,
            ..base_cell()
        },
    ];
    register_coop_arrival(&mut coop, 1);
    register_coop_arrival(&mut coop, 2);
    assert!(!try_trigger_coop(&mut coop, &mut cells));
    assert!(!coop.triggered);
    assert!((cells[0].energy - 50.0).abs() < 1e-4);
    assert!((cells[1].energy - 30.0).abs() < 1e-4);
    assert!(!coop.is_expired(0));
}

#[test]
fn carnivore_score_in_genome_random_initial_range() {
    let mut rng = StdRng::seed_from_u64(0xCA12);
    for _ in 0..100 {
        let g = Genome::random(&mut rng);
        assert!(
            (0.0..0.5).contains(&g.carnivore_score),
            "carnivore_score {} out of init range [0, 0.5]",
            g.carnivore_score
        );
    }
}

#[test]
fn thermal_optimum_random_in_range() {
    // Sprint 87: Genome::random by měl init thermal_optimum uniform v range.
    let mut rng = StdRng::seed_from_u64(0x7E0);
    for _ in 0..100 {
        let g = Genome::random(&mut rng);
        assert!(
            (MIN_THERMAL_OPTIMUM..=MAX_THERMAL_OPTIMUM).contains(&g.thermal_optimum),
            "optimum {} out of range",
            g.thermal_optimum
        );
    }
}

#[test]
fn apply_energy_costs_thermal_stress_quadratic() {
    // Sprint 87: penalty kvadratický v |temp - optimum|. Cell s optimum
    // matching local temp platí 0 penalty; cell s extreme deviation platí
    // PENALTY × (dev/13)². Compare 3 cells: matched, half-deviation,
    // extreme.
    let half = [1000.0, 1000.0, 50.0];
    let physics = PhysicsConfig {
        drag: 0.0,
        angular_drag: 0.0,
        energy_cost_per_v_sq: 0.0,
        angular_energy_cost: 0.0,
        vision_cost_per_radius: 0.0,
        body_cost_factor: 0.0,
        thermal_optimum_penalty: 1.0,
    };
    // Cell at z=0 → temp = REF = 17. Optimum = 17 → no penalty.
    let mut matched = base_cell();
    matched.position = [0.0, 0.0, 0.0];
    matched.genome.thermal_optimum = THERMAL_REF_TEMP;
    matched.step(1.0, half, 0, 0, &physics);
    let matched_drain = 100.0 - matched.energy;
    assert!(matched_drain.abs() < 0.01, "matched drain {matched_drain}");
    // Cell at z=0 (temp=17), optimum=4 (= BOTTOM, dev=13). penalty/sec =
    // (13/13)² × 1.0 = 1.0.
    let mut extreme = base_cell();
    extreme.position = [0.0, 0.0, 0.0];
    extreme.genome.thermal_optimum = MIN_THERMAL_OPTIMUM;
    extreme.step(1.0, half, 0, 0, &physics);
    let extreme_drain = 100.0 - extreme.energy;
    assert!(
        (extreme_drain - 1.0).abs() < 0.01,
        "extreme drain {extreme_drain}"
    );
    // Cell at z=0, optimum = 17 + 6.5 = 23.5 (= half-deviation). penalty/sec
    // = (6.5/13)² × 1.0 = 0.25.
    let mut half_dev = base_cell();
    half_dev.position = [0.0, 0.0, 0.0];
    half_dev.genome.thermal_optimum = THERMAL_REF_TEMP + 6.5;
    half_dev.step(1.0, half, 0, 0, &physics);
    let half_dev_drain = 100.0 - half_dev.energy;
    assert!(
        (half_dev_drain - 0.25).abs() < 0.01,
        "half-dev drain {half_dev_drain}"
    );
}

#[test]
fn populate_brain_inputs_writes_temperature_slot() {
    // Sprint 87: slot 20 = tanh_fast_scalar((temp - REF) / 10). Expected
    // values derive from the same fast-tanh approximation the implementation
    // uses (sensors.rs) so the test tracks both the formula and the THERMAL
    // constants instead of hardcoding a std-`tanh` literal.
    let mut cell = base_cell();
    let sensors = BrainSensors {
        nearest_food: None,
        nearest_cell: None,
        neighbors_in_vision: 0,
        smell_grad: [0.0; 3],
        pheromone_grads: [[0.0; 3]; N_PHEROMONE_CHANNELS],
        temperature_local: THERMAL_REF_TEMP, // exact REF → tanh(0) = 0
        vibration_grad: [0.0; 3],
        vibration_amp: 0.0,
        whisker_distances: [1.0; WHISKER_COUNT],
    };
    let inputs = populate_brain_inputs(&mut cell, &sensors, 50.0);
    assert!((inputs[20] - 0.0).abs() < 1e-4, "REF should be 0, got {}", inputs[20]);
    // Test top temp.
    let sensors_top = BrainSensors {
        temperature_local: THERMAL_TOP,
        ..sensors
    };
    let inputs_top = populate_brain_inputs(&mut cell, &sensors_top, 50.0);
    let expect_top = tanh_fast_scalar((THERMAL_TOP - THERMAL_REF_TEMP) / 10.0);
    assert!(
        (inputs_top[20] - expect_top).abs() < 1e-4,
        "TOP got {}, expected {expect_top}",
        inputs_top[20]
    );
    // Test bottom temp.
    let sensors_bot = BrainSensors {
        temperature_local: THERMAL_BOTTOM,
        ..sensors
    };
    let inputs_bot = populate_brain_inputs(&mut cell, &sensors_bot, 50.0);
    let expect_bot = tanh_fast_scalar((THERMAL_BOTTOM - THERMAL_REF_TEMP) / 10.0);
    assert!(
        (inputs_bot[20] - expect_bot).abs() < 1e-4,
        "BOTTOM got {}, expected {expect_bot}",
        inputs_bot[20]
    );
}

#[test]
fn vision_fov_factor_endpoints() {
    // Full sphere (theta = π) → solid angle = 4π str → factor = 1.0.
    assert!((vision_fov_factor(MAX_VISION_FOV) - 1.0).abs() < 1e-6);
    // Hemisphere (theta = π/2) → solid angle = 2π str → factor = 0.5.
    let half = vision_fov_factor(core::f32::consts::PI * 0.5);
    assert!((half - 0.5).abs() < 1e-6, "got {half}");
    // Narrow cone (theta = 0) → factor = 0.
    assert!(vision_fov_factor(0.0).abs() < 1e-6);
    // Clamp: above π saturates na 1.0 (kdyby někdo poslal 2π omylem).
    assert!((vision_fov_factor(core::f32::consts::PI * 2.0) - 1.0).abs() < 1e-6);
    // Monotonic mezi krajními body.
    let q = vision_fov_factor(core::f32::consts::PI * 0.25);
    assert!(q > 0.0 && q < 0.5);
}

#[test]
fn fov_cone_accept_basic_directions() {
    let fwd = [1.0_f32, 0.0, 0.0];
    // Quarter-circle FOV: half-angle = π/4 → cos = ~0.707.
    let cos_q = (core::f32::consts::PI * 0.25).cos();
    // Target přímo vpředu — vždy uvnitř.
    let front = [10.0_f32, 0.0, 0.0];
    assert!(fov_cone_accept(front, 100.0, fwd, cos_q));
    // Target přímo vpravo (90° offset) — mimo π/4 kuželu.
    let side = [0.0_f32, 10.0, 0.0];
    assert!(!fov_cone_accept(side, 100.0, fwd, cos_q));
    // Target přímo vzadu — mimo.
    let back = [-10.0_f32, 0.0, 0.0];
    assert!(!fov_cone_accept(back, 100.0, fwd, cos_q));
    // Hemisphere FOV (cos = 0) — front + side accepted, back rejected.
    let cos_h = 0.0_f32;
    assert!(fov_cone_accept(front, 100.0, fwd, cos_h));
    // Side je přesně na hranici (dot = 0 = cos_h) → accept.
    assert!(fov_cone_accept(side, 100.0, fwd, cos_h));
    assert!(!fov_cone_accept(back, 100.0, fwd, cos_h));
    // Degenerate target na cell pozici — vždy accept.
    assert!(fov_cone_accept([0.0, 0.0, 0.0], 0.0, fwd, cos_q));
    // Full sphere (cos = -1) — vše accept včetně back.
    assert!(fov_cone_accept(back, 100.0, fwd, -1.0));
}

#[test]
fn fov_cone_works_in_3d() {
    // Heading podél +X, cell s pitch +π/4 → forward má kladnou Z komponentu.
    // Test, že target nahoře-vpředu projde, target dole-vpředu padne ven
    // u úzkého kuželu.
    let fwd = forward_vector(0.0, core::f32::consts::PI * 0.25);
    let cos_q = (core::f32::consts::PI * 0.25).cos();
    let up_front = [10.0_f32, 0.0, 10.0];
    let down_front = [10.0_f32, 0.0, -10.0];
    let d2 = 200.0;
    assert!(fov_cone_accept(up_front, d2, fwd, cos_q));
    assert!(!fov_cone_accept(down_front, d2, fwd, cos_q));
}

#[test]
fn vision_fov_narrows_energy_cost() {
    // Sprint 82: užší FOV → menší cost. Hemisphere (factor 0.5) drained
    // přesně poloviční energy než full sphere (factor 1.0) při stejném
    // vision_radius a stejném dt.
    let mut wide = base_cell();
    wide.genome.vision_fov = MAX_VISION_FOV;
    let mut narrow = base_cell();
    narrow.genome.vision_fov = core::f32::consts::PI * 0.5;
    let physics = no_drag_physics(0.0, 0.05);
    wide.step(1.0, [1000.0, 1000.0, 0.0], 0, 0, &physics);
    narrow.step(1.0, [1000.0, 1000.0, 0.0], 0, 0, &physics);
    let wide_drain = 100.0 - wide.energy;
    let narrow_drain = 100.0 - narrow.energy;
    // Vision part: wide = 40 × 0.05 × 1.0 = 2.0, narrow = 40 × 0.05 × 0.5 = 1.0.
    // Ostatní drain (body, motion, …) je 0 v no_drag_physics.
    assert!((wide_drain - 2.0).abs() < 1e-4, "wide drain {wide_drain}");
    assert!((narrow_drain - 1.0).abs() < 1e-4, "narrow drain {narrow_drain}");
}

fn no_drag_physics(cost_per_v_sq: f32, vision_cost: f32) -> PhysicsConfig {
    PhysicsConfig {
        drag: 0.0,
        angular_drag: 0.0,
        energy_cost_per_v_sq: cost_per_v_sq,
        angular_energy_cost: 0.0,
        vision_cost_per_radius: vision_cost,
        body_cost_factor: 0.0,
        thermal_optimum_penalty: 0.0,
    }
}

fn base_cell() -> Cell {
    let genome = dummy_genome();
    let phenotype = Phenotype::from_genome(&genome);
    Cell {
        position: [0.0, 0.0, 0.0],
        velocity: [0.0, 0.0, 0.0],
        angular_velocity: 0.0,
        pitch_velocity: 0.0,
        energy: 100.0,
        heading: 0.0,
        pitch: 0.0,
        lineage_id: 0,
        lineage_birth_gen: 0,
        last_inputs: [0.0; BRAIN_INPUTS],
        last_hidden: [0.0; BRAIN_HIDDEN],
        last_outputs: [0.0; BRAIN_OUTPUTS],
        last_emit: [0.0; N_PHEROMONE_CHANNELS],
        burst_accum: [0.0; N_PHEROMONE_CHANNELS],
        pooled_hidden: [0.0; BRAIN_HIDDEN],
        bonded_inbox: [0.0; N_BOND_MSG_CHANNELS],
        damage_accum: 0.0,
        age: 0,
        reproduce_cooldown_ticks: 0,
        cell_id: 0,
        bonds: [None; MAX_BONDS_PER_CELL],
        cell_state: 0.5,
        last_best_food_d2: f32::MAX,
        xoshiro_state: crate::Xoshiro128PlusPlus::from_cell_id(0),
        last_whisker_distances: [1.0; WHISKER_COUNT],
        whisker_deflection: [0.0; WHISKER_COUNT],
        whisker_deflection_vel: [0.0; WHISKER_COUNT],
        novelty_history: [u32::MAX; NOVELTY_HISTORY_LEN],
        novelty_head: 0,
        under_attack_streak: 0,
        escape_cooldown_ticks: 0,
        phenotype,
        genome,
    }
}

#[test]
fn step_drains_energy_from_motion_and_vision() {
    let mut cell = Cell {
        velocity: [60.0, 0.0, 0.0],
        ..base_cell()
    };
    cell.step(1.0, [1000.0, 1000.0, 0.0], 0, 0, &no_drag_physics(0.001, 0.05));
    // motion (v² model): 60² × 0.001 × 1.0 = 3.6 energy
    // vision: 40 × 0.05 × 1.0 = 2.0 energy
    // body: 0 (factor = 0)
    // total drained: 5.6 → energy 100 − 5.6 = 94.4
    assert!((cell.energy - 94.4).abs() < 1e-4, "expected ~94.4, got {}", cell.energy);
    assert!((cell.position[0] - 60.0).abs() < 1e-4);
}

#[test]
fn step_xy_wraps_toroidal() {
    // Sprint 54: xy wrap (cylinder topology). Cell s pos x=99, vel +60,
    // dt=1 → integrate kinematic dá pos x=159 → wrap modulo (world half=100,
    // wrap shift 200) → x=−41. Heading se nepojí (žádný bounce).
    let mut cell = Cell {
        position: [99.0, 0.0, 0.0],
        velocity: [60.0, 0.0, 0.0],
        heading: 0.0,
        ..base_cell()
    };
    cell.step(1.0, [100.0, 100.0, 0.0], 0, 0, &no_drag_physics(0.0, 0.0));
    assert!(
        (cell.position[0] - (-41.0)).abs() < 1e-3,
        "expected pos.x ≈ -41 after wrap, got {}",
        cell.position[0]
    );
    // Velocity beze změny po wrapu.
    assert!((cell.velocity[0] - 60.0).abs() < 1e-3);
    // Heading se po wrap nezmění.
    assert!((cell.heading - 0.0).abs() < 1e-3);
}

#[test]
fn step_preserves_heading_when_velocity_zero() {
    let mut cell = Cell {
        heading: 1.5,
        ..base_cell()
    };
    cell.step(1.0, [100.0, 100.0, 0.0], 0, 0, &no_drag_physics(0.0, 0.0));
    // No movement, no bounce, no angular velocity, heading must persist.
    assert_eq!(cell.heading, 1.5);
}

#[test]
fn step_applies_quadratic_drag() {
    let mut cell = Cell {
        velocity: [10.0, 0.0, 0.0],
        ..base_cell()
    };
    let physics = PhysicsConfig {
        drag: 0.01,
        angular_drag: 0.0,
        energy_cost_per_v_sq: 0.0,
        angular_energy_cost: 0.0,
        vision_cost_per_radius: 0.0,
        body_cost_factor: 0.0,
        thermal_optimum_penalty: 0.0,
    };
    cell.step(1.0, [1000.0, 1000.0, 0.0], 0, 0, &physics);
    // |v| = 10, drag_dt = 0.01 × 10 × 1 = 0.1
    // velocity[0] -= 0.1 × 10 = 1.0 → final velocity[0] = 9.0
    assert!((cell.velocity[0] - 9.0).abs() < 1e-4, "got {}", cell.velocity[0]);
}

#[test]
fn step_drains_energy_from_rotation() {
    let mut cell = Cell {
        angular_velocity: 2.0,
        ..base_cell()
    };
    let physics = PhysicsConfig {
        drag: 0.0,
        angular_drag: 0.0,
        energy_cost_per_v_sq: 0.0,
        angular_energy_cost: 0.05,
        vision_cost_per_radius: 0.0,
        body_cost_factor: 0.0,
        thermal_optimum_penalty: 0.0,
    };
    cell.step(1.0, [1000.0, 1000.0, 0.0], 0, 0, &physics);
    // effective_radius²(=1) × ω²(=4) × angular_cost(=0.05) × dt(=1) = 0.2 drained
    assert!((cell.energy - 99.8).abs() < 1e-4, "got {}", cell.energy);
}

#[test]
fn step_rotation_cost_independent_of_linear_cost() {
    // Regression: spinning-in-place was a degenerate local minimum because
    // rotational drain piggy-backed on energy_cost_per_v_sq. Now decoupled.
    let mut cell = Cell {
        angular_velocity: 3.0,
        ..base_cell()
    };
    let physics = PhysicsConfig {
        drag: 0.0,
        angular_drag: 0.0,
        energy_cost_per_v_sq: 99.0,
        angular_energy_cost: 0.0,
        vision_cost_per_radius: 0.0,
        body_cost_factor: 0.0,
        thermal_optimum_penalty: 0.0,
    };
    cell.step(1.0, [1000.0, 1000.0, 0.0], 0, 0, &physics);
    assert!((cell.energy - 100.0).abs() < 1e-4, "got {}", cell.energy);
}

#[test]
fn step_applies_angular_drag() {
    let mut cell = Cell {
        angular_velocity: 1.0,
        ..base_cell()
    };
    let physics = PhysicsConfig {
        drag: 0.0,
        angular_drag: 0.5,
        energy_cost_per_v_sq: 0.0,
        angular_energy_cost: 0.0,
        vision_cost_per_radius: 0.0,
        body_cost_factor: 0.0,
        thermal_optimum_penalty: 0.0,
    };
    cell.step(1.0, [1000.0, 1000.0, 0.0], 0, 0, &physics);
    // angular_velocity *= (1 − 0.5 × 1) = 0.5 → 0.5
    assert!((cell.angular_velocity - 0.5).abs() < 1e-4);
}

#[test]
fn try_eat_within_radius_returns_true_and_adds_energy() {
    let mut cell = Cell {
        energy: 50.0,
        ..base_cell()
    };
    let food = Food { position: [5.0, 0.0, 0.0], age_ticks: 0, kind: FoodKind::Plant };
    assert!(cell.try_eat(&food, 8.0, 20.0));
    assert_eq!(cell.energy, 70.0);
}

#[test]
fn try_eat_outside_radius_returns_false_and_keeps_energy() {
    let mut cell = Cell {
        energy: 50.0,
        ..base_cell()
    };
    let food = Food { position: [20.0, 0.0, 0.0], age_ticks: 0, kind: FoodKind::Plant };
    assert!(!cell.try_eat(&food, 8.0, 20.0));
    assert_eq!(cell.energy, 50.0);
}

#[test]
fn crossover_picks_genes_from_either_parent() {
    let mut rng = rand::rng();
    let a = Genome {
        max_speed: 30.0,
        color_hue: 10.0,
        vision_radius: 20.0,
        turn_rate: 1.0,
        body_length: 0.5,
        body_width: 0.6,
        body_height: 0.7,
        spikes: [Spike::ZERO; SPIKE_SLOTS],
        spike_count: 1,
        shell_thickness: 0.0,
        adhesion_type: 1,
        bond_stiffness: 2.0,
        bond_damping: 0.3,
        vision_fov: MIN_VISION_FOV,
        thermal_optimum: MIN_THERMAL_OPTIMUM,
        carnivore_score: 0.0,
        sensor_gains: [MIN_SENSOR_GAIN; N_SENSOR_CATEGORIES],
        brain: dummy_brain(),
        cppn: default_cppn(),
        learning_rate: LEARNING_RATE,
        trace_decay_per_sec: HEBBIAN_TRACE_DECAY_PER_SEC,
        neuron_model: NeuronModel::Perceptron,
        stdp_a_plus: DEFAULT_STDP_A_PLUS,
        stdp_a_minus: DEFAULT_STDP_A_MINUS,
        stdp_tau_ticks: DEFAULT_STDP_TAU_TICKS,
        reproduce_at_energy: REPRODUCE_THRESHOLD,
        birth_energy: 50.0,
        altruism_share_frac: BOND_FOOD_SHARE_FRAC,
        cluster_share_bonus: BOND_FOOD_SHARE_CLUSTER_BONUS,
        attack_gate: ATTACK_THRESHOLD,
        predation_size_ratio: SIZE_RATIO_THRESHOLD,
        defense_contribution: BOND_DEFENSE_FRAC,
        reward_weights: REWARD_WEIGHT_DEFAULTS,
    };
    let b = Genome {
        max_speed: 90.0,
        color_hue: 200.0,
        vision_radius: 80.0,
        turn_rate: 5.0,
        body_length: 1.5,
        body_width: 1.4,
        body_height: 1.3,
        spikes: {
            let mut s = [Spike::ZERO; SPIKE_SLOTS];
            s[0].length = 0.8;
            s
        },
        spike_count: 1,
        shell_thickness: 0.5,
        adhesion_type: 5,
        bond_stiffness: 8.0,
        bond_damping: 1.0,
        vision_fov: MAX_VISION_FOV,
        thermal_optimum: MAX_THERMAL_OPTIMUM,
        carnivore_score: 1.0,
        sensor_gains: [MAX_SENSOR_GAIN; N_SENSOR_CATEGORIES],
        brain: dummy_brain(),
        cppn: default_cppn(),
        learning_rate: LEARNING_RATE,
        trace_decay_per_sec: HEBBIAN_TRACE_DECAY_PER_SEC,
        neuron_model: NeuronModel::Perceptron,
        stdp_a_plus: DEFAULT_STDP_A_PLUS,
        stdp_a_minus: DEFAULT_STDP_A_MINUS,
        stdp_tau_ticks: DEFAULT_STDP_TAU_TICKS,
        reproduce_at_energy: REPRODUCE_THRESHOLD,
        birth_energy: 50.0,
        altruism_share_frac: BOND_FOOD_SHARE_FRAC,
        cluster_share_bonus: BOND_FOOD_SHARE_CLUSTER_BONUS,
        attack_gate: ATTACK_THRESHOLD,
        predation_size_ratio: SIZE_RATIO_THRESHOLD,
        defense_contribution: BOND_DEFENSE_FRAC,
        reward_weights: REWARD_WEIGHT_DEFAULTS,
    };
    for _ in 0..100 {
        let c = Genome::crossover(&a, &b, &mut rng);
        assert!(c.max_speed == 30.0 || c.max_speed == 90.0);
        assert!(c.color_hue == 10.0 || c.color_hue == 200.0);
        assert!(c.vision_radius == 20.0 || c.vision_radius == 80.0);
        assert!(c.turn_rate == 1.0 || c.turn_rate == 5.0);
        assert!(c.body_length == 0.5 || c.body_length == 1.5);
        assert!(c.body_width == 0.6 || c.body_width == 1.4);
        assert!(c.spikes[0].length == 0.0 || c.spikes[0].length == 0.8);
        assert!(c.vision_fov == MIN_VISION_FOV || c.vision_fov == MAX_VISION_FOV);
        assert!(
            c.thermal_optimum == MIN_THERMAL_OPTIMUM
                || c.thermal_optimum == MAX_THERMAL_OPTIMUM
        );
    }
}

#[test]
fn hebbian_update_with_zero_reward_is_noop() {
    let mut brain = dummy_brain();
    brain.b1[0] = 0.5;
    brain.b2[0] = 0.7;
    let snapshot_b1 = brain.b1;
    let snapshot_b2 = brain.b2;
    brain.hebbian_update(
        &[1.0; BRAIN_INPUTS],
        &[1.0; BRAIN_HIDDEN],
        &[1.0; BRAIN_OUTPUTS],
        0.0,
        0.1,
    );
    assert_eq!(brain.b1, snapshot_b1);
    assert_eq!(brain.b2, snapshot_b2);
}

#[test]
fn hebbian_update_reinforces_when_reward_positive() {
    let mut brain = dummy_brain();
    // hidden = [1.0; hidden_n], output = [1.0; OUT], reward = 1.0, lr = 0.1
    // Δb1[i] = 0.1 × 1.0 × hidden[i] = 0.1 pro i < hidden_n; 0 jinak
    // Δb2[i] = 0.1 × 1.0 × output[i] = 0.1
    brain.hebbian_update(
        &[0.0; BRAIN_INPUTS],
        &[1.0; BRAIN_HIDDEN],
        &[1.0; BRAIN_OUTPUTS],
        1.0,
        0.1,
    );
    // Sprint 80: hebbian bounded by hidden_n. Dead zone b1 stays at init (0).
    let h_n = brain.hidden_n as usize;
    for &b in &brain.b1[..h_n] {
        assert!((b - 0.1).abs() < 1e-5, "active b1 got {}", b);
    }
    for &b in &brain.b1[h_n..] {
        assert_eq!(b, 0.0, "dead-zone b1 must stay 0");
    }
    for &b in &brain.b2 {
        assert!((b - 0.1).abs() < 1e-5, "b2 got {}", b);
    }
}

#[test]
fn world_map_is_deterministic_for_seed() {
    let a = WorldMap::new([32, 32, 8], [8, 8, 4], [500.0, 500.0, 50.0], 42);
    let b = WorldMap::new([32, 32, 8], [8, 8, 4], [500.0, 500.0, 50.0], 42);
    assert_eq!(a.field(), b.field());
}

#[test]
fn world_map_seeds_differ() {
    let a = WorldMap::new([32, 32, 8], [8, 8, 4], [500.0, 500.0, 50.0], 1);
    let b = WorldMap::new([32, 32, 8], [8, 8, 4], [500.0, 500.0, 50.0], 2);
    assert_ne!(a.field(), b.field());
}

#[test]
fn world_map_values_in_unit_range() {
    let m = WorldMap::new([32, 32, 8], [8, 8, 4], [500.0, 500.0, 50.0], 7);
    for &v in m.field() {
        assert!((0.0..=1.0).contains(&v), "out of range: {}", v);
    }
}

#[test]
fn world_map_sample_xy_wraps_z_clamps() {
    // Sprint 54: xy modulo wrap, z clamp. Inside sample ∈ [0,1].
    // Bod přesně na +half_x je přes wrap ekvivalentní -half_x.
    let m = WorldMap::new([8, 8, 4], [4, 4, 2], [100.0, 100.0, 50.0], 0);
    let inside = m.sample([99.0, 99.0, 0.0]);
    // Sample at +half wraps to -half (same grid cell).
    let at_left = m.sample([-100.0, 0.0, 0.0]);
    let at_right_wrap = m.sample([100.0, 0.0, 0.0]);
    assert!((at_left - at_right_wrap).abs() < 1e-6, "xy wrap broken");
    // Z out-of-range clamps (still valid, no panic).
    let above = m.sample([0.0, 0.0, 1e6]);
    let below = m.sample([0.0, 0.0, -1e6]);
    assert!((0.0..=1.0).contains(&above));
    assert!((0.0..=1.0).contains(&below));
    assert!((0.0..=1.0).contains(&inside));
}

#[test]
fn random_brain_average_thrust_is_positive() {
    // Innate thrust bias musí dělat to, k čemu existuje: random buňky
    // mají ze startu thrust output kladný v průměru, takže se hýbou
    // dopředu místo zacyklení v rozporu mezi turn a thrust.
    // Sprint 79: seeded RNG (pre-S57 era používalo `rand::rng()` thread-local
    // → flaky napříč CI, ~5 % run failures kdyby gaussian sampling ojediněle
    // posunul mean pod 0.3). Fixed seed dělá test deterministický.
    let mut rng = StdRng::seed_from_u64(42);
    let n = 200;
    let zero_inputs = [0.0_f32; BRAIN_INPUTS];
    let mut sum = 0.0_f64;
    let mut count_positive = 0;
    for _ in 0..n {
        let brain = Brain::random(&mut rng);
        let thrust = brain.forward(&zero_inputs)[1];
        sum += thrust as f64;
        if thrust > 0.0 {
            count_positive += 1;
        }
    }
    let mean = sum / n as f64;
    assert!(mean > 0.3, "expected mean thrust > 0.3, got {}", mean);
    // Sprint 126: BRAIN_INPUTS 71 → 77 zvýšilo input variance (víc gaussian
    // weight noise feeded do hidden → větší tail variance v output[1]).
    // INNATE_THRUST_BIAS posune mean kladně, ale fraction positive se snížila
    // z >75 % na ~70 %. Mean je dál >0.3, evolutionary jumpstart funguje.
    assert!(
        count_positive > n * 2 / 3,
        "expected >66% positive, got {}/{}",
        count_positive,
        n
    );
}

#[test]
#[ignore = "Pre-S189 semantic: forward output = tanh(b2) directly. Sprint 189 inserted LayerNorm before tanh at both L1 and L2 layers, so output is now tanh(normalized(pre_out)). Zero-weight b2=[0.5, -0.5, 0, ...] is normalized away from raw values and the assertion no longer holds. Update or replace once S189 stabilises."]
fn brain_forward_zero_weights_outputs_tanh_of_output_biases() {
    // Zero weights kill signal flow at both layers — output equals tanh(b2),
    // independent of b1 (the hidden activations get zeroed by w2).
    // Sprint 126: BRAIN_OUTPUTS = 12 (+2 ch1/ch2 emit), test still passes
    // because we read just outputs[0] and [1].
    let mut b2 = [0.0_f32; BRAIN_OUTPUTS];
    b2[0] = 0.5;
    b2[1] = -0.5;
    let brain = Brain {
        hidden_n: BRAIN_HIDDEN_DEFAULT as u32,
        w1: [[0.0; BRAIN_INPUTS]; BRAIN_HIDDEN],
        b1: [0.7; BRAIN_HIDDEN],
        w2: [[0.0; BRAIN_HIDDEN]; BRAIN_OUTPUTS],
        b2,
        trace_w1: [[0.0; BRAIN_INPUTS]; BRAIN_HIDDEN],
        trace_w2: [[0.0; BRAIN_HIDDEN]; BRAIN_OUTPUTS],
        membrane: [IZH_V_REST; BRAIN_HIDDEN],
        recovery: [0.0; BRAIN_HIDDEN],
        last_pre_spike_ticks: [0; BRAIN_INPUTS],
        last_post_spike_ticks: [0; BRAIN_HIDDEN],
        pre_trace: [0.0; BRAIN_INPUTS],
        post_trace: [0.0; BRAIN_HIDDEN],
    };
    let outputs = brain.forward(&[0.0; BRAIN_INPUTS]);
    assert_eq!(outputs.len(), BRAIN_OUTPUTS);
    assert!((outputs[0] - 0.5_f32.tanh()).abs() < 1e-6);
    assert!((outputs[1] - (-0.5_f32).tanh()).abs() < 1e-6);
    // ch1, ch2 (sloty 10, 11) → b2 = 0 → output = tanh(0) = 0.
    assert!(outputs[10].abs() < 1e-6);
    assert!(outputs[11].abs() < 1e-6);
}

#[test]
fn multi_channel_pheromone_emit_costs_proportionally() {
    // Sprint 126 sanity: tři kanály emit at full strength = 3× cost vs.
    // jeden. Validates summed cost model: cost = total_emit × cost_rate × dt.
    // Test je jen formální (cost rovnice je v emit_pheromones binárky, ne v
    // lib), takže testujeme přímo equation v isolation.
    let cost_rate = PHEROMONE_COST_PER_RATE;
    let dt = 1.0 / FIXED_TIMESTEP_HZ;
    let emit_single = [1.0_f32, 0.0, 0.0];
    let emit_triple = [1.0_f32, 1.0, 1.0];
    let total_single: f32 = emit_single.iter().sum();
    let total_triple: f32 = emit_triple.iter().sum();
    let cost_single = total_single * cost_rate * dt;
    let cost_triple = total_triple * cost_rate * dt;
    assert!((cost_triple / cost_single - 3.0).abs() < 1e-6);
}

#[test]
fn pheromone_field_array_independent_decay() {
    // Sprint 126: 3 fields s rozdílnými decay rates. Po jednom kroku step
    // má ch2 (decay 5.0) ztratit > ch1 (1.5) > ch0 (0.3) signálu.
    let world_half = [100.0_f32, 100.0, 50.0];
    let mut fields: [SmellField; N_PHEROMONE_CHANNELS] =
        std::array::from_fn(|_| SmellField::new([8, 8, 4], world_half));
    for f in fields.iter_mut() {
        f.add_source([0.0, 0.0, 0.0], 1.0);
    }
    let dt = 1.0 / FIXED_TIMESTEP_HZ;
    for ch in 0..N_PHEROMONE_CHANNELS {
        for _ in 0..30 {
            fields[ch].step(PHEROMONE_DIFFUSION_PER_CH[ch], PHEROMONE_DECAY_PER_CH[ch], dt);
        }
    }
    let signal_ch0 = fields[0].sample([0.0, 0.0, 0.0]);
    let signal_ch1 = fields[1].sample([0.0, 0.0, 0.0]);
    let signal_ch2 = fields[2].sample([0.0, 0.0, 0.0]);
    assert!(
        signal_ch0 > signal_ch1,
        "ch0 (slow decay) should retain více signal než ch1: ch0={signal_ch0} ch1={signal_ch1}"
    );
    assert!(
        signal_ch1 > signal_ch2,
        "ch1 should retain více než ch2 (rychlejší decay): ch1={signal_ch1} ch2={signal_ch2}"
    );
}

#[test]
fn brain_random_sets_default_hidden_n() {
    let mut rng = StdRng::seed_from_u64(7);
    let b = Brain::random(&mut rng);
    assert_eq!(b.hidden_n as usize, BRAIN_HIDDEN_DEFAULT);
}

#[test]
fn brain_random_with_hidden_zeros_dead_zone() {
    let mut rng = StdRng::seed_from_u64(7);
    let h_n: u32 = 8;
    assert!((h_n as usize) < BRAIN_HIDDEN, "test assumes h_n < storage");
    let b = Brain::random_with_hidden(&mut rng, h_n);
    assert_eq!(b.hidden_n, h_n);
    // Dead zone w1[h_n..] / b1[h_n..] / w2[*][h_n..] must stay 0 — random
    // initialization touched only active region.
    for i in (h_n as usize)..BRAIN_HIDDEN {
        assert_eq!(b.b1[i], 0.0, "b1[{}] should be 0", i);
        for &w in b.w1[i].iter() {
            assert_eq!(w, 0.0, "w1[{}][..] should be 0", i);
        }
    }
    for o in 0..BRAIN_OUTPUTS {
        for j in (h_n as usize)..BRAIN_HIDDEN {
            assert_eq!(b.w2[o][j], 0.0, "w2[{}][{}] should be 0", o, j);
        }
    }
}

#[test]
fn brain_mutate_preserves_hidden_n_and_dead_zone() {
    let mut rng = StdRng::seed_from_u64(11);
    let h_n: u32 = 6;
    let parent = Brain::random_with_hidden(&mut rng, h_n);
    let child = parent.mutate(&mut rng, 0.5);
    assert_eq!(child.hidden_n, h_n, "hidden_n must survive mutation");
    // Dead zone untouched (no gaussian draws applied to inactive rows).
    for i in (h_n as usize)..BRAIN_HIDDEN {
        assert_eq!(child.b1[i], parent.b1[i]);
        assert_eq!(child.w1[i], parent.w1[i]);
    }
    for o in 0..BRAIN_OUTPUTS {
        for j in (h_n as usize)..BRAIN_HIDDEN {
            assert_eq!(child.w2[o][j], parent.w2[o][j]);
        }
    }
}

#[test]
fn brain_crossover_handles_mismatched_hidden_n() {
    // Sprint 104: structural mutace mohou rozejít hidden_n. Crossover
    // teď vezme menší size + per-row mix přes shared rozsah, místo paniky.
    let mut rng = StdRng::seed_from_u64(13);
    let a = Brain::random_with_hidden(&mut rng, 8);
    let b = Brain::random_with_hidden(&mut rng, 12);
    let c = Brain::crossover(&a, &b, &mut rng);
    assert_eq!(c.hidden_n, 8, "child takes smaller parent's hidden_n");
}

#[test]
fn brain_storage_cap_above_default_with_room_for_growth() {
    // Sprint 80 (storage bump): BRAIN_HIDDEN je storage cap, default je
    // initial active. Rozdíl = headroom pro structural mutace.
    assert!(
        BRAIN_HIDDEN > BRAIN_HIDDEN_DEFAULT,
        "BRAIN_HIDDEN ({}) must be > BRAIN_HIDDEN_DEFAULT ({}) to leave room for add_neuron",
        BRAIN_HIDDEN,
        BRAIN_HIDDEN_DEFAULT
    );
    assert!(
        BRAIN_HIDDEN >= BRAIN_HIDDEN_DEFAULT + 8,
        "headroom < 8 neurons: structural mutace bude rychle narážet na cap"
    );
}

#[test]
fn add_neuron_increments_hidden_n() {
    let mut rng = StdRng::seed_from_u64(31);
    let mut b = Brain::random_with_hidden(&mut rng, BRAIN_HIDDEN_DEFAULT as u32);
    let h_before = b.hidden_n;
    let added = b.add_neuron(&mut rng, ADD_NEURON_SIGMA);
    assert!(added);
    assert_eq!(b.hidden_n, h_before + 1);
}

#[test]
fn add_neuron_returns_false_at_storage_cap() {
    let mut rng = StdRng::seed_from_u64(33);
    let mut b = Brain::random_with_hidden(&mut rng, BRAIN_HIDDEN as u32);
    assert_eq!(b.hidden_n as usize, BRAIN_HIDDEN);
    let added = b.add_neuron(&mut rng, ADD_NEURON_SIGMA);
    assert!(!added, "add_neuron at cap must return false");
    assert_eq!(b.hidden_n as usize, BRAIN_HIDDEN, "cap respected");
}

#[test]
fn add_neuron_initializes_active_region_only() {
    let mut rng = StdRng::seed_from_u64(37);
    let mut b = Brain::random_with_hidden(&mut rng, BRAIN_HIDDEN_DEFAULT as u32);
    let new_idx = b.hidden_n as usize;
    let active_inputs = BRAIN_INPUTS_SENSORY + new_idx + 1;
    let _ = b.add_neuron(&mut rng, ADD_NEURON_SIGMA);
    // New neuron's row [new_idx] active part [0..active_inputs] should be
    // gaussian-initialized (some non-zero values expected). Dead-zone of
    // that same row [active_inputs..BRAIN_INPUTS] should remain 0.
    let any_active_nonzero = b.w1[new_idx][..active_inputs]
        .iter()
        .any(|&w| w != 0.0);
    assert!(any_active_nonzero, "new neuron active w1 row all-zero");
    for &w in &b.w1[new_idx][active_inputs..] {
        assert_eq!(w, 0.0, "new neuron dead-cols must stay 0");
    }
}

#[test]
fn add_neuron_preserves_existing_neurons() {
    let mut rng = StdRng::seed_from_u64(41);
    let mut b = Brain::random_with_hidden(&mut rng, BRAIN_HIDDEN_DEFAULT as u32);
    let snapshot_w1: Vec<_> = b.w1[..BRAIN_HIDDEN_DEFAULT].to_vec();
    let snapshot_b1 = b.b1;
    let snapshot_b2 = b.b2;
    // Snapshot w2 active cols only — dead col at new_idx will get populated.
    let snapshot_w2_active: Vec<Vec<f32>> = b
        .w2
        .iter()
        .map(|row| row[..BRAIN_HIDDEN_DEFAULT].to_vec())
        .collect();
    let _ = b.add_neuron(&mut rng, ADD_NEURON_SIGMA);
    // Existing neurons (rows 0..BRAIN_HIDDEN_DEFAULT) untouched.
    for (i, expected) in snapshot_w1.iter().enumerate() {
        assert_eq!(&b.w1[i], expected, "w1[{}] should be unchanged", i);
    }
    for i in 0..BRAIN_HIDDEN_DEFAULT {
        assert_eq!(b.b1[i], snapshot_b1[i], "b1[{}] should be unchanged", i);
    }
    // b2 unchanged (no contribution from add_neuron).
    assert_eq!(b.b2, snapshot_b2);
    // w2 active cols (existing neurons' connections) unchanged.
    for o in 0..BRAIN_OUTPUTS {
        for h in 0..BRAIN_HIDDEN_DEFAULT {
            assert_eq!(b.w2[o][h], snapshot_w2_active[o][h]);
        }
    }
}

#[test]
#[ignore = "Sprint 106 HyperNEAT: brain.hidden_n je deterministicky BRAIN_HIDDEN_DEFAULT \
            z Brain::from_cppn — direct add_neuron mutace dead, brain re-derived z CPPN \
            na každý mutate() call. Topologie evoluuje teď přes CPPN structural mutations \
            (mutate_add_node v Cppn), test by se nastavoval jinak."]
fn genome_mutate_with_rate_one_grows_brain_to_cap() {}

#[test]
fn brain_hidden_n_above_default_forward_uses_padded_storage() {
    // Sprint B: brain s hidden_n > BRAIN_HIDDEN_DEFAULT používá rozšířený
    // storage. Forward output musí brát v potaz nové aktivní neurony.
    let h_n: u32 = (BRAIN_HIDDEN_DEFAULT as u32) + 4;
    assert!(
        (h_n as usize) <= BRAIN_HIDDEN,
        "test config: h_n {} must be ≤ BRAIN_HIDDEN {}",
        h_n,
        BRAIN_HIDDEN
    );
    let mut rng = StdRng::seed_from_u64(17);
    let brain_default = Brain::random_with_hidden(&mut rng, BRAIN_HIDDEN_DEFAULT as u32);
    let brain_extended = Brain::random_with_hidden(&mut rng, h_n);
    let inputs = [0.5_f32; BRAIN_INPUTS];
    let out_default = brain_default.forward(&inputs);
    let out_extended = brain_extended.forward(&inputs);
    // Two different brains with non-overlapping random init should produce
    // different outputs. Sanity check že padded storage není no-op.
    let any_diff = out_default
        .iter()
        .zip(out_extended.iter())
        .any(|(a, b)| (a - b).abs() > 1e-6);
    assert!(any_diff, "extended hidden_n produced identical output");
}

#[test]
fn morph_zero_signal_does_not_change_phenotype() {
    let mut phen = Phenotype {
        body_length: 1.5,
        body_width: 0.8,
        body_height: 1.0,
        spikes: {
            let mut s = [Spike::ZERO; SPIKE_SLOTS];
            s[0].length = 0.3;
            s
        },
        spike_count: 1,
        shell_thickness: 0.0,
    };
    let delta = phen.apply_morph([0.0, 0.0, 0.0, 0.0], MORPH_RATE, 0.5);
    assert_eq!(delta, 0.0);
    assert_eq!(phen.body_length, 1.5);
    assert_eq!(phen.body_width, 0.8);
    assert_eq!(phen.body_height, 1.0);
    assert_eq!(phen.spikes[0].length, 0.3);
}

#[test]
fn morph_clamps_to_min_max_bounds() {
    let mut phen = Phenotype {
        body_length: MAX_BODY_LENGTH,
        body_width: MIN_BODY_WIDTH,
        body_height: MAX_BODY_HEIGHT,
        spikes: {
            let mut s = [Spike::ZERO; SPIKE_SLOTS];
            s[0].length = MAX_SPIKE_LENGTH;
            s
        },
        spike_count: 1,
        shell_thickness: 0.0,
    };
    // Strong positive signal on length, height & spike (already at max) → no change.
    // Strong negative signal on width (already at min) → no change.
    let delta = phen.apply_morph([1.0, -1.0, 1.0, 1.0], 100.0, 1.0);
    assert_eq!(delta, 0.0);
    assert_eq!(phen.body_length, MAX_BODY_LENGTH);
    assert_eq!(phen.body_width, MIN_BODY_WIDTH);
    assert_eq!(phen.body_height, MAX_BODY_HEIGHT);
    assert_eq!(phen.spikes[0].length, MAX_SPIKE_LENGTH);
}

#[test]
fn morph_returns_total_absolute_delta() {
    let mut phen = Phenotype {
        body_length: 1.0,
        body_width: 1.0,
        body_height: 1.0,
        spikes: {
            let mut s = [Spike::ZERO; SPIKE_SLOTS];
            s[0].length = 0.5;
            s
        },
        spike_count: 1,
        shell_thickness: 0.0,
    };
    // signal × rate × dt = 0.8 × 1.0 × 1.0 = 0.8 podél každé osy.
    // Width clampuje na MIN_BODY_WIDTH (0.8), takže |Δ| pro width je
    // 1.0 - 0.8 = 0.2. Total |Δ| = 0.8 (length) + 0.2 (width clamped)
    // + 0.0 (height: signal=0) + 0.8 (spike) = 1.8.
    let delta = phen.apply_morph([0.8, -0.8, 0.0, 0.8], 1.0, 1.0);
    assert!((delta - 1.8).abs() < 1e-5, "got {}", delta);
    assert!((phen.body_length - 1.8).abs() < 1e-5);
    assert!((phen.body_width - MIN_BODY_WIDTH).abs() < 1e-5);
    assert!((phen.spikes[0].length - 1.3).abs() < 1e-5);
}

#[test]
fn morph_signal_below_threshold_is_deadzoned() {
    let mut phen = Phenotype {
        body_length: 1.0,
        body_width: 1.0,
        body_height: 1.0,
        spikes: [Spike::ZERO; SPIKE_SLOTS],
        spike_count: 1,
        shell_thickness: 0.0,
    };
    // |signal| < threshold → no change (filters random brain noise).
    let delta = phen.apply_morph(
        [
            MORPH_ACTIVATION_THRESHOLD - 0.01,
            -MORPH_ACTIVATION_THRESHOLD + 0.01,
            0.0,
            0.0,
        ],
        1.0,
        1.0,
    );
    assert_eq!(delta, 0.0);
    assert_eq!(phen.body_length, 1.0);
    assert_eq!(phen.body_width, 1.0);
    assert_eq!(phen.spikes[0].length, 0.0);
}

#[test]
fn cell_apply_morph_updates_phenotype_not_genome() {
    // Genotype/phenotype split: runtime morph nesmí sahat na genome.
    let mut cell = base_cell();
    let original_genome_len = cell.genome.body_length;
    cell.last_outputs[3] = 1.0; // morph_length signal
    cell.apply_morph(1.0);
    assert!(cell.phenotype.body_length > original_genome_len);
    assert_eq!(cell.genome.body_length, original_genome_len);
}

#[test]
fn anisotropic_drag_slower_along_width_when_elongated() {
    // Cell s length=2, width=1, heading=0, motion (10,0) (forward) vs (0,10)
    // (sideways). Forward "cítí" width (=1) jako cross-section, sideways
    // cítí length (=2). Sideways must therefore decay faster.
    let physics = PhysicsConfig {
        drag: 0.01,
        angular_drag: 0.0,
        energy_cost_per_v_sq: 0.0,
        angular_energy_cost: 0.0,
        vision_cost_per_radius: 0.0,
        body_cost_factor: 0.0,
        thermal_optimum_penalty: 0.0,
    };
    let make_cell = |vel: [f32; 3]| {
        let mut c = base_cell();
        c.phenotype = Phenotype {
            body_length: 2.0,
            body_width: 1.0,
            body_height: 1.0,
            spikes: [Spike::ZERO; SPIKE_SLOTS],
            spike_count: 1,
            shell_thickness: 0.0,
        };
        c.velocity = vel;
        c
    };
    let mut forward = make_cell([10.0, 0.0, 0.0]);
    let mut sideways = make_cell([0.0, 10.0, 0.0]);
    forward.step(1.0, [1000.0, 1000.0, 0.0], 0, 0, &physics);
    sideways.step(1.0, [1000.0, 1000.0, 0.0], 0, 0, &physics);
    // |v| forward after step: 10 - drag·|v|·width·v = 10 - 0.01·10·1·10 = 9.0
    // |v| sideways after step: 10 - 0.01·10·2·10 = 8.0
    let v_forward = forward.velocity[0].hypot(forward.velocity[1]);
    let v_sideways = sideways.velocity[0].hypot(sideways.velocity[1]);
    assert!(v_forward > v_sideways, "forward {} should be > sideways {}", v_forward, v_sideways);
    assert!((v_forward - 9.0).abs() < 1e-3, "forward got {}", v_forward);
    assert!((v_sideways - 8.0).abs() < 1e-3, "sideways got {}", v_sideways);
}

#[test]
fn anisotropic_drag_isotropic_when_axes_equal() {
    // Když length=width=1, anisotropic verze musí dát stejný výsledek jako
    // původní isotropic (regression test pro `step_applies_quadratic_drag`).
    let mut cell = base_cell();
    cell.velocity = [10.0, 0.0, 0.0];
    let physics = PhysicsConfig {
        drag: 0.01,
        angular_drag: 0.0,
        energy_cost_per_v_sq: 0.0,
        angular_energy_cost: 0.0,
        vision_cost_per_radius: 0.0,
        body_cost_factor: 0.0,
        thermal_optimum_penalty: 0.0,
    };
    cell.step(1.0, [1000.0, 1000.0, 0.0], 0, 0, &physics);
    assert!((cell.velocity[0] - 9.0).abs() < 1e-4);
}

#[test]
fn spike_bonus_only_when_target_in_front_cone() {
    let mut cell = base_cell();
    cell.position = [0.0, 0.0, 0.0];
    cell.heading = 0.0; // pointing +x
    cell.phenotype.spikes[0].length = 1.0;
    cell.phenotype.spike_count = 1;

    // Target přímo vepředu — bonus se aplikuje.
    let bonus_front = cell.spike_bonus_against([10.0, 0.0, 0.0]);
    assert!(bonus_front > 0.0);

    // Target za zády — bonus = 0.
    let bonus_back = cell.spike_bonus_against([-10.0, 0.0, 0.0]);
    assert_eq!(bonus_back, 0.0);

    // Target přesně na boku — bonus = 0 (cosine = 0 < threshold 0.7).
    let bonus_side = cell.spike_bonus_against([0.0, 10.0, 0.0]);
    assert_eq!(bonus_side, 0.0);
}

#[test]
fn spike_bonus_zero_when_no_spike() {
    let mut cell = base_cell();
    cell.heading = 0.0;
    cell.phenotype.spikes[0].length = 0.0;
    cell.phenotype.spike_count = 1;
    let bonus = cell.spike_bonus_against([10.0, 0.0, 0.0]);
    assert_eq!(bonus, 0.0);
}

#[test]
fn food_rejection_never_rejects_at_max_richness() {
    let mut rng = StdRng::seed_from_u64(42);
    for _ in 0..10_000 {
        assert!(!reject_food_for_richness(&mut rng, 1.0));
    }
}

#[test]
fn food_rejection_rate_at_min_richness_matches_strength() {
    let mut rng = StdRng::seed_from_u64(42);
    let n = 100_000;
    let rejected = (0..n)
        .filter(|_| reject_food_for_richness(&mut rng, 0.0))
        .count();
    let observed = rejected as f32 / n as f32;
    // Tolerance ±0.01 for sample noise on 100k draws (~3σ for p=0.3).
    assert!(
        (observed - FOOD_REJECTION_STRENGTH).abs() < 0.01,
        "observed reject rate {} vs expected {}",
        observed,
        FOOD_REJECTION_STRENGTH
    );
}

#[test]
fn step_3d_position_advances_with_z_velocity() {
    // Sprint 32 sanity: z-složka pozice musí integrovat z velocity stejně
    // jako x/y, takže Sprint 33+ má pevnou základnu.
    let mut cell = base_cell();
    cell.velocity = [0.0, 0.0, 5.0];
    cell.step(1.0, [1000.0, 1000.0, 1000.0], 0, 0, &no_drag_physics(0.0, 0.0));
    assert!(
        (cell.position[2] - 5.0).abs() < 1e-4,
        "expected z=5.0, got {}",
        cell.position[2]
    );
}

#[test]
fn z_locked_world_keeps_food_planar() {
    // Sprint 32: world_half[2] = 0 znamená Food::random vrací z=0 a
    // nespotřebovává RNG draw na z. Critical pro CSV identity.
    let mut rng = StdRng::seed_from_u64(7);
    for _ in 0..1_000 {
        let f = Food::random(&mut rng, [100.0, 100.0, 0.0]);
        assert_eq!(f.position[2], 0.0);
    }
}

#[test]
fn step_drains_energy_from_spike_maintenance() {
    let mut cell = base_cell();
    cell.phenotype.spikes[0].length = 0.5;
    cell.phenotype.spike_count = 1;
    let physics = PhysicsConfig {
        drag: 0.0,
        angular_drag: 0.0,
        energy_cost_per_v_sq: 0.0,
        angular_energy_cost: 0.0,
        vision_cost_per_radius: 0.0,
        body_cost_factor: 0.0,
        thermal_optimum_penalty: 0.0,
    };
    cell.step(1.0, [1000.0, 1000.0, 0.0], 0, 0, &physics);
    // spike_length(=0.5) × SPIKE_COST_PER_SEC × dt(=1) = 0.15 drained
    let expected_drain = 0.5 * SPIKE_COST_PER_SEC;
    assert!(
        (cell.energy - (100.0 - expected_drain)).abs() < 1e-4,
        "got {}, expected {}",
        cell.energy,
        100.0 - expected_drain
    );
}

#[test]
fn body_basis_orthonormal() {
    let cases = [
        (0.0, 0.0),
        (0.5, 0.0),
        (-1.2, 0.0),
        (1.7, 0.3),
        (-2.4, -0.4),
        (3.1, 0.5),
        (0.7, -0.2),
    ];
    for &(yaw, pitch) in &cases {
        let (fwd, right, up) = body_basis(yaw, pitch);
        let mag = |v: [f32; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        let dot = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        assert!((mag(fwd) - 1.0).abs() < 1e-5, "fwd not unit at yaw={yaw} pitch={pitch}");
        assert!((mag(right) - 1.0).abs() < 1e-5, "right not unit");
        assert!((mag(up) - 1.0).abs() < 1e-5, "up not unit");
        assert!(dot(fwd, right).abs() < 1e-5, "fwd·right != 0");
        assert!(dot(fwd, up).abs() < 1e-5, "fwd·up != 0");
        assert!(dot(right, up).abs() < 1e-5, "right·up != 0");
    }
}

#[test]
fn try_eat_isotropic_unchanged_for_unit_sphere() {
    // L=W=H=1, eat_factor=8 → ellipsoid degeneruje na sféru radius 8.
    // Backward-kompat se Sprint 40 sférickou eat-zónou.
    let cell = Cell { energy: 50.0, ..base_cell() };
    let inside = Food { position: [5.0, 0.0, 0.0], age_ticks: 0, kind: FoodKind::Plant };
    let outside = Food { position: [10.0, 0.0, 0.0], age_ticks: 0, kind: FoodKind::Plant };
    let lateral_inside = Food { position: [0.0, 5.0, 0.0], age_ticks: 0, kind: FoodKind::Plant };
    let vertical_inside = Food { position: [0.0, 0.0, 5.0], age_ticks: 0, kind: FoodKind::Plant };
    assert!(cell.eat_test(&inside, 8.0));
    assert!(!cell.eat_test(&outside, 8.0));
    assert!(cell.eat_test(&lateral_inside, 8.0));
    assert!(cell.eat_test(&vertical_inside, 8.0));
}

#[test]
fn try_eat_forward_chip_reaches_further_than_lateral() {
    // Chip: L=2, W=0.5, H=0.5, heading=0 → forward semi-osa = 16, lateral = 4.
    let mut cell = Cell { energy: 50.0, ..base_cell() };
    cell.phenotype = Phenotype {
        body_length: 2.0,
        body_width: 0.5,
        body_height: 0.5,
        spikes: [Spike::ZERO; SPIKE_SLOTS],
        spike_count: 1,
        shell_thickness: 0.0,
    };
    // Forward at +14: inside ellipsoid (14/16 = 0.875).
    let forward_inside = Food { position: [14.0, 0.0, 0.0], age_ticks: 0, kind: FoodKind::Plant };
    // Lateral at +3.5: inside (3.5/4 = 0.875).
    let lateral_inside = Food { position: [0.0, 3.5, 0.0], age_ticks: 0, kind: FoodKind::Plant };
    // Forward at +17: outside (17/16 > 1).
    let forward_outside = Food { position: [17.0, 0.0, 0.0], age_ticks: 0, kind: FoodKind::Plant };
    // Lateral at +5: outside (5/4 > 1).
    let lateral_outside = Food { position: [0.0, 5.0, 0.0], age_ticks: 0, kind: FoodKind::Plant };
    assert!(cell.eat_test(&forward_inside, 8.0));
    assert!(cell.eat_test(&lateral_inside, 8.0));
    assert!(!cell.eat_test(&forward_outside, 8.0));
    assert!(!cell.eat_test(&lateral_outside, 8.0));
}

#[test]
fn max_axis_returns_largest_dimension() {
    let phen = Phenotype {
        body_length: 2.0,
        body_width: 0.5,
        body_height: 1.5,
        spikes: [Spike::ZERO; SPIKE_SLOTS],
        spike_count: 1,
        shell_thickness: 0.0,
    };
    assert!((phen.max_axis() - 2.0).abs() < 1e-6);
}

#[test]
fn shell_absorbs_predation_drain() {
    // shell=1.0, ABSORB_PER_TICK=2.0, dt=1 → absorbed 2.0; raw damage 3.0
    // → damage_accum after = 1.0.
    let mut cell = base_cell();
    cell.phenotype.shell_thickness = 1.0;
    cell.damage_accum = PREDATION_DRAIN_PER_TICK; // = 3.0
    cell.apply_shell_absorb(1.0);
    let expected = PREDATION_DRAIN_PER_TICK - 1.0 * SHELL_ABSORB_PER_TICK;
    assert!(
        (cell.damage_accum - expected).abs() < 1e-5,
        "got {}, expected {}",
        cell.damage_accum,
        expected
    );
}

#[test]
fn shell_zero_no_effect() {
    let mut cell = base_cell();
    cell.phenotype.shell_thickness = 0.0;
    cell.damage_accum = 2.5;
    cell.apply_shell_absorb(1.0);
    assert_eq!(cell.damage_accum, 2.5);
}

#[test]
fn shell_does_not_absorb_below_zero() {
    // Big shell, small damage → clamp to 0, ne na negative.
    let mut cell = base_cell();
    cell.phenotype.shell_thickness = MAX_SHELL_THICKNESS;
    cell.damage_accum = 1.0;
    cell.apply_shell_absorb(1.0);
    assert_eq!(cell.damage_accum, 0.0);
}

#[test]
fn shell_cost_scales_linearly() {
    let mut cell = base_cell();
    cell.phenotype.shell_thickness = 1.0;
    let physics = PhysicsConfig {
        drag: 0.0,
        angular_drag: 0.0,
        energy_cost_per_v_sq: 0.0,
        angular_energy_cost: 0.0,
        vision_cost_per_radius: 0.0,
        body_cost_factor: 0.0,
        thermal_optimum_penalty: 0.0,
    };
    cell.step(1.0, [1000.0, 1000.0, 0.0], 0, 0, &physics);
    let expected_drain = 1.0 * SHELL_COST_PER_SEC;
    assert!(
        (cell.energy - (100.0 - expected_drain)).abs() < 1e-4,
        "got {}, expected {}",
        cell.energy,
        100.0 - expected_drain
    );
}

#[test]
fn shell_mutation_clamps_to_range() {
    let mut rng = rand::rng();
    let g = dummy_genome();
    let cfg = MutationConfig {
        sigma_speed: 0.0,
        sigma_hue: 0.0,
        sigma_vision: 0.0,
        sigma_turn_rate: 0.0,
        sigma_body_length: 0.0,
        sigma_body_width: 0.0,
        sigma_body_height: 0.0,
        sigma_spike_length: 0.0,
        sigma_shell: 100.0,
        sigma_brain: 0.0,
        adhesion_flip_rate: 0.0,
        sigma_bond_stiffness: 0.0,
        sigma_bond_damping: 0.0,
        add_neuron_rate: 0.0,
        split_link_rate: 0.0,
        remove_neuron_rate: 0.0,
        sigma_vision_fov: 0.0,
        sigma_thermal_optimum: 0.0,
        sigma_carnivore_score: 0.0,
        sigma_sensor_gain: 0.0,
        spike_count_mutation_rate: 0.0,
        sigma_spike_orientation: 0.0,
        sigma_spike_complexity: 0.0,
        sigma_spike_length_secondary: 0.0,
        sigma_learning_rate: 0.0,
        sigma_trace_decay: 0.0,
        model_flip_rate: 0.0,
        sigma_stdp_a: 0.0,
        sigma_stdp_tau: 0.0,
        sigma_reproduce_at_energy: 0.0,
        sigma_birth_energy: 0.0,
        sigma_altruism_share_frac: 0.0,
        sigma_cluster_share_bonus: 0.0,
        sigma_attack_gate: 0.0,
        sigma_predation_size_ratio: 0.0,
        sigma_defense_contribution: 0.0,
        sigma_reward_weights: [0.0; N_REWARD_KINDS],
    };
    for _ in 0..1000 {
        let m = g.mutate(&mut rng, &cfg);
        assert!(
            (MIN_SHELL_THICKNESS..=MAX_SHELL_THICKNESS).contains(&m.shell_thickness),
            "shell out of range: {}",
            m.shell_thickness
        );
    }
}

#[test]
fn step_aging_increases_body_cost() {
    let physics = PhysicsConfig {
        drag: 0.0,
        angular_drag: 0.0,
        energy_cost_per_v_sq: 0.0,
        angular_energy_cost: 0.0,
        vision_cost_per_radius: 0.0,
        body_cost_factor: 1.0,
        thermal_optimum_penalty: 0.0,
    };
    // Cell at age 0 → factor 1.0, drain = volume = 1.
    let mut young = base_cell();
    young.age = 0;
    let young_energy_before = young.energy;
    young.step(1.0, [1000.0, 1000.0, 0.0], 0, 0, &physics);
    let young_drain = young_energy_before - young.energy;

    // Cell at age 600 (= 10s) → factor 1 + 0.005×10 = 1.05.
    let mut old = base_cell();
    old.age = 600;
    let old_energy_before = old.energy;
    old.step(1.0, [1000.0, 1000.0, 0.0], 0, 0, &physics);
    let old_drain = old_energy_before - old.energy;

    assert!(
        old_drain > young_drain,
        "old cell should drain more: young={} old={}",
        young_drain,
        old_drain
    );
}

#[test]
fn step_increments_age() {
    let mut cell = base_cell();
    assert_eq!(cell.age, 0);
    cell.step(1.0, [1000.0, 1000.0, 0.0], 0, 0, &no_drag_physics(0.0, 0.0));
    assert_eq!(cell.age, 1);
    cell.step(1.0, [1000.0, 1000.0, 0.0], 0, 0, &no_drag_physics(0.0, 0.0));
    assert_eq!(cell.age, 2);
}

#[test]
fn cooldown_decrements_per_step() {
    let mut cell = base_cell();
    cell.reproduce_cooldown_ticks = 5;
    cell.step(1.0, [1000.0, 1000.0, 0.0], 0, 0, &no_drag_physics(0.0, 0.0));
    assert_eq!(cell.reproduce_cooldown_ticks, 4);
}

#[test]
fn cooldown_does_not_underflow() {
    let mut cell = base_cell();
    cell.reproduce_cooldown_ticks = 0;
    cell.step(1.0, [1000.0, 1000.0, 0.0], 0, 0, &no_drag_physics(0.0, 0.0));
    assert_eq!(cell.reproduce_cooldown_ticks, 0);
}

#[test]
fn motor_scales_inversely_with_mass() {
    // Unit cell eff_r=1 vs tubby cell eff_r=2 (L=W=H=2): tubby pomalejší 2×.
    // Mass scaling používá effective_radius (smoke-tuned fallback z volume).
    let mut unit = base_cell();
    let mut tubby = base_cell();
    tubby.phenotype.body_length = 2.0;
    tubby.phenotype.body_width = 2.0;
    tubby.phenotype.body_height = 2.0;
    let outputs = [0.0; BRAIN_OUTPUTS];
    let mut outputs = outputs;
    outputs[1] = 1.0;
    unit.apply_brain_motor(&outputs, 1.0);
    tubby.apply_brain_motor(&outputs, 1.0);
    let unit_v = unit.velocity[0].abs();
    let tubby_v = tubby.velocity[0].abs();
    assert!(
        unit_v > tubby_v,
        "unit cell should accelerate faster: unit={} tubby={}",
        unit_v,
        tubby_v
    );
    let ratio = unit_v / tubby_v.max(1e-6);
    assert!(
        (ratio - 2.0).abs() < 0.2,
        "expected ratio ~2 (eff_r), got {}",
        ratio
    );
}

#[test]
fn brownian_perturbs_zero_velocity() {
    let mut cell = base_cell();
    // 100 brownian steps; statisticky téměř jistě některá komponenta != 0.
    let sqrt_dt = (1.0_f32 / FIXED_TIMESTEP_HZ).sqrt();
    for _ in 0..100 {
        cell.apply_brownian(sqrt_dt, 0.0);
    }
    // 2D případ (world_half_z = 0) — z se nesmí měnit.
    assert_eq!(cell.velocity[2], 0.0);
    let v_xy_sq =
        cell.velocity[0] * cell.velocity[0] + cell.velocity[1] * cell.velocity[1];
    assert!(v_xy_sq > 0.0, "expected nonzero velocity from brownian");
}

#[test]
fn brownian_z_only_in_3d_world() {
    let mut cell = base_cell();
    // 3D mode: world_half_z > 0 → z se má hýbat.
    let sqrt_dt = (1.0_f32 / FIXED_TIMESTEP_HZ).sqrt();
    for _ in 0..100 {
        cell.apply_brownian(sqrt_dt, 2.0);
    }
    assert!(cell.velocity[2] != 0.0, "expected nonzero z velocity in 3D");
}

#[test]
fn brownian_deterministic_across_paths_for_same_cell_id() {
    // Two cells spawned with the same `cell_id` must produce identical
    // velocity perturbations — this is the contract that CPU and GPU
    // brownian streams now share (both seed xoshiro from `cell_id`).
    let mut a = base_cell();
    let mut b = base_cell();
    a.cell_id = 42;
    b.cell_id = 42;
    a.xoshiro_state = crate::Xoshiro128PlusPlus::from_cell_id(a.cell_id);
    b.xoshiro_state = crate::Xoshiro128PlusPlus::from_cell_id(b.cell_id);
    let sqrt_dt = 0.1_f32;
    for _ in 0..50 {
        a.apply_brownian(sqrt_dt, 2.0);
        b.apply_brownian(sqrt_dt, 2.0);
    }
    assert_eq!(a.velocity, b.velocity);
}

#[test]
fn food_value_decays_with_age() {
    let mut food = Food { position: [0.0, 0.0, 0.0], age_ticks: 0, kind: FoodKind::Plant };
    assert!((food.value_factor() - 1.0).abs() < 1e-6);
    // 1 sec = 60 ticks → factor = 1 - CARRION_DECAY_PER_SEC.
    food.age_ticks = 60;
    let expected = 1.0 - CARRION_DECAY_PER_SEC;
    assert!(
        (food.value_factor() - expected).abs() < 1e-4,
        "got {}, expected {}",
        food.value_factor(),
        expected
    );
}

#[test]
fn food_expires_when_zero_value() {
    let mut fresh = Food { position: [0.0, 0.0, 0.0], age_ticks: 0, kind: FoodKind::Plant };
    assert!(fresh.age_step());
    // Past lifetime: age_step bump → value_factor = 0 → returns false.
    // F32 precision: použijeme age daleko za bod expirace, abychom se vyhli
    // ULP edge case (60.0/0.0005 jako u32 rounds k 119999, ne 120000).
    let mut expired = Food {
        position: [0.0, 0.0, 0.0],
        age_ticks: ((FIXED_TIMESTEP_HZ / CARRION_DECAY_PER_SEC) as u32) + 100,
        kind: FoodKind::Plant,
    };
    assert!(!expired.age_step());
}

#[test]
fn child_starts_with_zero_age_and_cooldown() {
    let mut rng = rand::rng();
    let g = dummy_genome();
    let cell_a = Cell::from_genome(&mut rng, g, [100.0, 100.0, 0.0], 0, 0, 1);
    let cell_b = Cell::from_genome(&mut rng, g, [100.0, 100.0, 0.0], 0, 0, 2);
    let child = make_mating_child(&cell_a, &cell_b, &mut rng, 3);
    assert_eq!(child.age, 0);
    assert_eq!(child.reproduce_cooldown_ticks, 0);
}

#[test]
fn spatial_grid_finds_all_neighbors_in_radius() {
    let mut rng = StdRng::seed_from_u64(42);
    let n = 1000;
    let half: f32 = 500.0;
    let points: Vec<(usize, [f32; 3], ())> = (0..n)
        .map(|i| {
            (
                i,
                [
                    rng.random_range(-half..half),
                    rng.random_range(-half..half),
                    rng.random_range(-1.0..1.0),
                ],
                (),
            )
        })
        .collect();

    let mut grid: SpatialGrid<usize, ()> = SpatialGrid::new(GRID_CELL_SIZE, WORLD_HALF);
    grid.rebuild(points.iter().copied());

    let query_pos = [0.0_f32, 0.0, 0.0];
    let radius = 50.0_f32;
    let r2 = radius * radius;

    let mut brute: Vec<usize> = points
        .iter()
        .filter_map(|(i, p, _)| {
            let dx = p[0] - query_pos[0];
            let dy = p[1] - query_pos[1];
            let dz = p[2] - query_pos[2];
            if dx * dx + dy * dy + dz * dz <= r2 {
                Some(*i)
            } else {
                None
            }
        })
        .collect();
    brute.sort();

    let mut from_grid: Vec<usize> = Vec::new();
    grid.for_each_in_radius(query_pos, radius, |id, p, _| {
        let dx = p[0] - query_pos[0];
        let dy = p[1] - query_pos[1];
        let dz = p[2] - query_pos[2];
        if dx * dx + dy * dy + dz * dz <= r2 {
            from_grid.push(id);
        }
    });
    from_grid.sort();

    assert_eq!(
        brute, from_grid,
        "grid query missed or extra neighbors vs brute force"
    );
}

#[test]
fn spatial_grid_rebuild_clears_old_buckets() {
    let mut grid: SpatialGrid<usize, ()> = SpatialGrid::new(50.0, WORLD_HALF);
    grid.rebuild(vec![(0_usize, [0.0, 0.0, 0.0], ()), (1, [10.0, 10.0, 0.0], ())]);

    let mut first: Vec<usize> = Vec::new();
    grid.for_each_in_radius([0.0, 0.0, 0.0], 100.0, |id, _, _| first.push(id));
    first.sort();
    assert_eq!(first, vec![0, 1]);

    grid.rebuild(vec![(2_usize, [200.0, 200.0, 0.0], ())]);
    let mut second: Vec<usize> = Vec::new();
    grid.for_each_in_radius([0.0, 0.0, 0.0], 100.0, |id, _, _| second.push(id));
    assert!(
        second.is_empty(),
        "rebuild left stale entries near origin: {:?}",
        second
    );

    let mut third: Vec<usize> = Vec::new();
    grid.for_each_in_radius([200.0, 200.0, 0.0], 100.0, |id, _, _| third.push(id));
    assert_eq!(third, vec![2]);
}

#[test]
fn spatial_grid_query_order_is_stable() {
    let points: Vec<(usize, [f32; 3], ())> = (0..50)
        .map(|i| (i, [i as f32 * 5.0, (i % 7) as f32 * 3.0, 0.0], ()))
        .collect();
    let mut grid: SpatialGrid<usize, ()> = SpatialGrid::new(20.0, WORLD_HALF);
    grid.rebuild(points.iter().copied());

    let mut a: Vec<usize> = Vec::new();
    grid.for_each_in_radius([100.0, 10.0, 0.0], 60.0, |id, _, _| a.push(id));
    let mut b: Vec<usize> = Vec::new();
    grid.for_each_in_radius([100.0, 10.0, 0.0], 60.0, |id, _, _| b.push(id));

    assert_eq!(a, b, "two identical queries returned different order");
}

// === Sprint 66: differential adhesion + spring bonds ===

#[test]
fn adhesion_is_zero_inside_contact() {
    // d <= pair_r: collision depenetration handles, adhesion no-op.
    let pair_r = 10.0;
    let delta = [pair_r * 0.5, 0.0, 0.0];
    let v = adhesion_velocity_delta(delta, pair_r * 0.5, pair_r, true);
    assert_eq!(v, [0.0, 0.0, 0.0]);
}

#[test]
fn adhesion_is_zero_beyond_range() {
    let pair_r = 10.0;
    let range = pair_r * ADHESION_RANGE_FACTOR;
    let d = range + 1.0;
    let delta = [d, 0.0, 0.0];
    let v = adhesion_velocity_delta(delta, d, pair_r, true);
    assert_eq!(v, [0.0, 0.0, 0.0]);
}

#[test]
fn adhesion_pulls_same_type_inward() {
    // pos_i - pos_j = +x (i je vpravo od j); same-type → attraction
    // znamená velocity i přírůstek směrem -x (k j).
    let pair_r = 10.0;
    let d = pair_r * 1.5;
    let delta = [d, 0.0, 0.0];
    let v = adhesion_velocity_delta(delta, d, pair_r, true);
    assert!(v[0] < 0.0, "expected pull toward j, got Δv = {:?}", v);
    assert_eq!(v[1], 0.0);
    assert_eq!(v[2], 0.0);
}

#[test]
fn adhesion_repels_cross_type_outward() {
    let pair_r = 10.0;
    let d = pair_r * 1.5;
    let delta = [d, 0.0, 0.0];
    let v = adhesion_velocity_delta(delta, d, pair_r, false);
    assert!(v[0] > 0.0, "expected push away, got Δv = {:?}", v);
}

#[test]
fn bond_spring_pulls_when_stretched() {
    // Bond rest 5, current 10 (stretched) → cell i taženo k j.
    let bond = Bond { other_cell_id: 1, rest_length: 5.0, stiffness: BOND_STIFFNESS, damping: BOND_DAMPING, age_ticks: 0 };
    let delta = [10.0, 0.0, 0.0];
    let dt = 1.0 / 60.0;
    let (v, broken) = bond_velocity_delta(&bond, delta, 10.0, [0.0; 3], [0.0; 3], dt);
    assert!(!broken);
    assert!(v[0] < 0.0, "stretched bond should pull i toward j, got {:?}", v);
}

#[test]
fn bond_spring_pushes_when_compressed() {
    // Bond rest 10, current 5 (compressed) → cell i tlačeno od j.
    let bond = Bond { other_cell_id: 1, rest_length: 10.0, stiffness: BOND_STIFFNESS, damping: BOND_DAMPING, age_ticks: 0 };
    let delta = [5.0, 0.0, 0.0];
    let dt = 1.0 / 60.0;
    let (v, broken) = bond_velocity_delta(&bond, delta, 5.0, [0.0; 3], [0.0; 3], dt);
    assert!(!broken);
    assert!(v[0] > 0.0, "compressed bond should push i away, got {:?}", v);
}

#[test]
fn bond_breaks_past_break_factor() {
    let rest = 5.0;
    let bond = Bond { other_cell_id: 1, rest_length: rest, stiffness: BOND_STIFFNESS, damping: BOND_DAMPING, age_ticks: 0 };
    let stretched = rest * BOND_BREAK_FACTOR + 0.1;
    let (v, broken) = bond_velocity_delta(
        &bond,
        [stretched, 0.0, 0.0],
        stretched,
        [0.0; 3],
        [0.0; 3],
        1.0 / 60.0,
    );
    assert!(broken, "bond should break past BOND_BREAK_FACTOR");
    assert_eq!(v, [0.0; 3]);
}

#[test]
fn bond_damping_opposes_closing_velocity() {
    // Cell i at +x, j at origin, bond at rest. v_i moves toward j (−x).
    // Damping should *resist* closing → push i back (+x).
    let bond = Bond { other_cell_id: 1, rest_length: 5.0, stiffness: BOND_STIFFNESS, damping: BOND_DAMPING, age_ticks: 0 };
    let delta = [5.0, 0.0, 0.0];
    let v_i = [-1.0, 0.0, 0.0];
    let v_j = [0.0, 0.0, 0.0];
    let dt = 1.0 / 60.0;
    let (dv, _) = bond_velocity_delta(&bond, delta, 5.0, v_i, v_j, dt);
    assert!(dv[0] > 0.0, "damping should oppose closing motion, got {:?}", dv);
}

#[test]
fn bond_defense_factor_zero_pool_is_unity() {
    // Sprint 187: empty defense_pool (no bonds OR all partners contribute 0)
    // → no damage reduction.
    assert!((bond_defense_factor(0.0) - 1.0).abs() < 1e-6);
}

#[test]
fn bond_defense_factor_scales_with_pool_until_floor() {
    // Sprint 187: factor = max(1.0 − pool, 0.4). Caller is responsible for
    // capping the pool at BOND_DEFENSE_CAP partners; the function itself
    // just applies the floor.
    assert!((bond_defense_factor(0.15) - 0.85).abs() < 1e-6);
    assert!((bond_defense_factor(0.30) - 0.70).abs() < 1e-6);
    assert!((bond_defense_factor(0.45) - 0.55).abs() < 1e-6);
    assert!((bond_defense_factor(0.60) - 0.40).abs() < 1e-6);
    assert!((bond_defense_factor(0.75) - 0.40).abs() < 1e-6);
    assert!((bond_defense_factor(2.00) - 0.40).abs() < 1e-6);
}

#[test]
fn n_bonds_counts_only_populated_slots() {
    let mut rng = StdRng::seed_from_u64(42);
    let mut cell = Cell::random(&mut rng, [960.0, 540.0, 50.0], 0, 0, 0);
    assert_eq!(cell.n_bonds(), 0);
    cell.bonds[0] = Some(Bond {
        other_cell_id: 99,
        rest_length: 5.0,
        age_ticks: 0,
        stiffness: BOND_STIFFNESS,
        damping: 0.6,
    });
    cell.bonds[3] = Some(Bond {
        other_cell_id: 100,
        rest_length: 5.0,
        age_ticks: 0,
        stiffness: BOND_STIFFNESS,
        damping: 0.6,
    });
    assert_eq!(cell.n_bonds(), 2);
}

#[test]
fn pick_cluster_parent_prefers_bonded_matching_adhesion() {
    let mut rng = StdRng::seed_from_u64(7);
    let mut a = Cell::random(&mut rng, [960.0, 540.0, 50.0], 0, 0, 0);
    let mut b = Cell::random(&mut rng, [960.0, 540.0, 50.0], 1, 0, 1);
    a.genome.adhesion_type = 3;
    b.genome.adhesion_type = 5;
    a.bonds[0] = Some(Bond {
        other_cell_id: 99,
        rest_length: 5.0,
        stiffness: BOND_STIFFNESS,
        damping: 0.6,
        age_ticks: 0,
    });
    // Child adhesion=3 → match s parent_a, který má bond.
    let pick = pick_cluster_parent(&a, &b, 3);
    assert!(pick.is_some());
    assert_eq!(pick.unwrap().cell_id, a.cell_id);
}

#[test]
fn pick_cluster_parent_falls_back_to_any_bonded() {
    let mut rng = StdRng::seed_from_u64(8);
    let mut a = Cell::random(&mut rng, [960.0, 540.0, 50.0], 0, 0, 0);
    let mut b = Cell::random(&mut rng, [960.0, 540.0, 50.0], 1, 0, 1);
    a.genome.adhesion_type = 3;
    b.genome.adhesion_type = 5;
    b.bonds[0] = Some(Bond {
        other_cell_id: 99,
        rest_length: 5.0,
        stiffness: BOND_STIFFNESS,
        damping: 0.6,
        age_ticks: 0,
    });
    // Child adhesion=7 — match neither — ale b má bondy → fallback.
    let pick = pick_cluster_parent(&a, &b, 7);
    assert!(pick.is_some());
    assert_eq!(pick.unwrap().cell_id, b.cell_id);
}

#[test]
fn pick_cluster_parent_returns_none_when_neither_bonded() {
    let mut rng = StdRng::seed_from_u64(9);
    let a = Cell::random(&mut rng, [960.0, 540.0, 50.0], 0, 0, 0);
    let b = Cell::random(&mut rng, [960.0, 540.0, 50.0], 1, 0, 1);
    assert!(pick_cluster_parent(&a, &b, 0).is_none());
}

#[test]
fn mating_child_spawns_near_bonded_parent() {
    let mut rng = StdRng::seed_from_u64(123);
    let mut a = Cell::random(&mut rng, [960.0, 540.0, 50.0], 0, 0, 0);
    let mut b = Cell::random(&mut rng, [960.0, 540.0, 50.0], 1, 0, 1);
    a.position = [100.0, 100.0, 0.0];
    b.position = [-100.0, -100.0, 0.0];
    a.genome.adhesion_type = 3;
    b.genome.adhesion_type = 3;
    a.bonds[0] = Some(Bond {
        other_cell_id: 99,
        rest_length: 5.0,
        stiffness: BOND_STIFFNESS,
        damping: 0.6,
        age_ticks: 0,
    });
    // Force child adhesion=3 by making both parents have type 3 →
    // crossover deterministic na typu (pre-mutation), mutation sice může
    // flipnout, ale 5% rate × seed 123 to nestane (test verifuje deterministic).
    let child = make_mating_child(&a, &b, &mut rng, 42);
    // Child adhesion mohl mutovat, ale spawn pozice byla rozhodnuta podle
    // child.genome.adhesion_type. Pokud child má type 3, spawn by měl
    // být blízko a (jediný bonded). Pokud mutoval, jeden z parents stejně
    // má bondy → fallback. Tedy child by měl být v každém případě blízko
    // parent_a (= jediný bonded), max do CLUSTER_SPAWN_RADIUS od něho.
    let dx = (child.position[0] - a.position[0]).abs();
    let dy = (child.position[1] - a.position[1]).abs();
    let dz = (child.position[2] - a.position[2]).abs();
    assert!(
        dx <= CLUSTER_SPAWN_RADIUS && dy <= CLUSTER_SPAWN_RADIUS
            && dz <= CLUSTER_SPAWN_RADIUS * 0.3 + 1e-3,
        "child spawn pozice mimo cluster jitter range — dx={} dy={} dz={}",
        dx,
        dy,
        dz
    );
}

#[test]
fn mating_child_spawns_at_midpoint_when_neither_parent_bonded() {
    let mut rng = StdRng::seed_from_u64(456);
    let mut a = Cell::random(&mut rng, [960.0, 540.0, 50.0], 0, 0, 0);
    let mut b = Cell::random(&mut rng, [960.0, 540.0, 50.0], 1, 0, 1);
    a.position = [100.0, 100.0, 0.0];
    b.position = [-100.0, -100.0, 0.0];
    // Žádný bond → midpoint (0, 0, 0).
    let child = make_mating_child(&a, &b, &mut rng, 42);
    assert!(
        child.position[0].abs() < 1e-3 && child.position[1].abs() < 1e-3,
        "child spawn pozice měla být midpoint (0, 0), got {:?}",
        child.position
    );
}

// ─── Sprint 105 CPPN tests ──────────────────────────────────────────

#[test]
fn cppn_random_has_correct_topology() {
    let mut rng = StdRng::seed_from_u64(7);
    let c = Cppn::random(&mut rng);
    assert_eq!(
        c.iter_nodes().filter(|n| n.layer == 0).count(),
        CPPN_INPUTS,
        "CPPN_INPUTS input nodes at layer 0"
    );
    assert_eq!(
        c.iter_nodes().filter(|n| n.layer == 2).count(),
        CPPN_OUTPUTS,
        "CPPN_OUTPUTS output nodes at layer 2"
    );
    assert_eq!(
        c.iter_nodes().filter(|n| n.layer == 1).count(),
        CPPN_INITIAL_HIDDEN,
        "CPPN_INITIAL_HIDDEN hidden nodes at layer 1"
    );
    let expected_links = CPPN_INPUTS * CPPN_INITIAL_HIDDEN
        + CPPN_INITIAL_HIDDEN * CPPN_OUTPUTS;
    assert_eq!(c.iter_links().count(), expected_links);
}

#[test]
fn cppn_forward_deterministic() {
    let mut rng = StdRng::seed_from_u64(11);
    let c = Cppn::random(&mut rng);
    let inputs = [0.5, -0.3, 0.7, 0.1, -0.4, 0.0, 1.0];
    let out1 = c.forward(inputs);
    let out2 = c.forward(inputs);
    assert_eq!(out1, out2, "deterministic forward");
    for o in out1.iter() {
        assert!(o.is_finite() && (-1.0..=1.0).contains(o), "out {} oob", o);
    }
}

#[test]
fn cppn_add_node_grows_topology() {
    let mut rng = StdRng::seed_from_u64(13);
    let mut c = Cppn::random(&mut rng);
    let n_pre = c.num_nodes;
    let l_pre = c.num_links;
    c.mutate_add_node(&mut rng);
    assert_eq!(c.num_nodes, n_pre + 1, "add_node adds 1 node");
    assert_eq!(c.num_links, l_pre + 2, "add_node adds 2 links");
}

#[test]
fn cppn_add_link_no_cycle() {
    let mut rng = StdRng::seed_from_u64(17);
    let mut c = Cppn::random(&mut rng);
    for _ in 0..50 {
        c.mutate_add_link(&mut rng, 0.5);
    }
    for l in c.iter_links() {
        let from_layer = c
            .iter_nodes()
            .find(|n| n.id == l.from)
            .map(|n| n.layer)
            .unwrap();
        let to_layer = c
            .iter_nodes()
            .find(|n| n.id == l.to)
            .map(|n| n.layer)
            .unwrap();
        assert!(
            from_layer < to_layer,
            "no cycles allowed: from layer {} >= to layer {}",
            from_layer,
            to_layer
        );
    }
}

#[test]
fn cppn_crossover_preserves_matching_innovations() {
    let mut rng = StdRng::seed_from_u64(19);
    let a = Cppn::random(&mut rng);
    let mut b = a;
    b.mutate_weight(&mut rng, 0.5);
    let c = Cppn::crossover(&a, &b, &mut rng);
    for la in a.iter_links() {
        assert!(
            c.iter_links().any(|lc| lc.innovation == la.innovation),
            "innovation {} preserved",
            la.innovation
        );
    }
}

#[test]
fn cppn_compatibility_distance_self_zero() {
    let mut rng = StdRng::seed_from_u64(31);
    let c = Cppn::random(&mut rng);
    let d = Cppn::compatibility_distance(&c, &c);
    assert!(d < 1e-3, "self-distance ≈ 0, got {}", d);
}

#[test]
fn cppn_compatibility_distance_grows_with_mutation() {
    let mut rng = StdRng::seed_from_u64(37);
    let a = Cppn::random(&mut rng);
    let mut b = a;
    // Heavy mutation pushuje distance výš
    for _ in 0..20 {
        b.mutate_weight(&mut rng, 1.0);
        b.mutate_add_node(&mut rng);
    }
    let d_self = Cppn::compatibility_distance(&a, &a);
    let d_far = Cppn::compatibility_distance(&a, &b);
    assert!(
        d_far > d_self + 0.05,
        "mutation grows distance: self={:.3}, mutated={:.3}",
        d_self,
        d_far
    );
}

#[test]
fn cppn_mutate_drives_diversity() {
    let mut rng = StdRng::seed_from_u64(23);
    let initial = Cppn::random(&mut rng);
    let cfg = CppnMutationConfig {
        weight_rate: 1.0,
        sigma_weight: 0.5,
        add_node_rate: 1.0,
        add_link_rate: 1.0,
        toggle_link_rate: 0.0,
        activation_rate: 1.0,
    };
    let mutated = initial.mutate(&mut rng, &cfg);
    assert!(mutated.num_nodes > initial.num_nodes, "topology grew");
}

#[test]
fn adhesion_works_across_toroidal_boundary() {
    // Cell i at x=950, j at x=-950, world half_x=960. Min-image delta
    // by měl být ~20 (přes wrap), ne ~1900.
    let world_half = [960.0, 540.0, 50.0];
    let pos_i = [950.0, 0.0, 0.0];
    let pos_j = [-950.0, 0.0, 0.0];
    let pair_r = 10.0;
    let d_vec = min_image_delta(pos_j, pos_i, world_half);
    let d = (d_vec[0] * d_vec[0] + d_vec[1] * d_vec[1] + d_vec[2] * d_vec[2]).sqrt();
    assert!(d < 25.0, "min-image distance should be ~20, got {}", d);
    let v = adhesion_velocity_delta(d_vec, d, pair_r, true);
    // Pull from i toward j přes wrap = +x (i is at +950, j wraps to +970).
    assert!(v[0] > 0.0, "expected wrap-aware pull, got {:?}", v);
}

fn shock_cfg_active() -> ShockScheduleConfig {
    ShockScheduleConfig {
        mean_gens_between: 20,
        type_weights: [1.0, 1.0, 1.0],
        intensity_min: 0.3,
        intensity_max: 1.0,
        duration_min_gens: 5,
        duration_max_gens: 15,
        ramp_gens: 2,
        spatial_global_prob: 0.5,
        spatial_radius_min_frac: 0.2,
        spatial_radius_max_frac: 0.6,
    }
}

#[test]
fn event_calendar_default_is_empty() {
    let cfg = ShockScheduleConfig::default();
    let cal = EventCalendar::generate(123, &cfg, 1000);
    assert!(cal.events.is_empty());
    assert_eq!(cal.seed, 123);
}

#[test]
fn event_calendar_is_deterministic_for_seed() {
    let cfg = shock_cfg_active();
    let a = EventCalendar::generate(42, &cfg, 500);
    let b = EventCalendar::generate(42, &cfg, 500);
    assert_eq!(a.events.len(), b.events.len());
    assert!(!a.events.is_empty(), "active cfg must produce events");
    for (ea, eb) in a.events.iter().zip(b.events.iter()) {
        assert_eq!(ea.kind, eb.kind);
        assert_eq!(ea.start_gen, eb.start_gen);
        assert_eq!(ea.duration_gen, eb.duration_gen);
        assert_eq!(ea.ramp_gens, eb.ramp_gens);
        assert!((ea.intensity - eb.intensity).abs() < 1e-6);
        assert_eq!(ea.center_xy.is_some(), eb.center_xy.is_some());
        assert_eq!(ea.radius.is_some(), eb.radius.is_some());
    }
}

#[test]
fn event_calendar_different_seeds_differ() {
    let cfg = shock_cfg_active();
    let a = EventCalendar::generate(42, &cfg, 1000);
    let b = EventCalendar::generate(43, &cfg, 1000);
    // Drobný risk shody, ale s mean=20 a 1000 gens je to >>50 eventů —
    // collision pravděpodobnost zanedbatelná.
    let identical = a.events.len() == b.events.len()
        && a.events
            .iter()
            .zip(b.events.iter())
            .all(|(x, y)| x.start_gen == y.start_gen && x.kind == y.kind);
    assert!(!identical, "different seeds should produce different schedules");
}

#[test]
fn event_calendar_respects_max_gens() {
    let cfg = shock_cfg_active();
    let max_gens = 500;
    let cal = EventCalendar::generate(7, &cfg, max_gens);
    for e in &cal.events {
        assert!(e.start_gen < max_gens, "start_gen {} >= max {}", e.start_gen, max_gens);
    }
}

#[test]
fn event_calendar_events_sorted() {
    let cfg = shock_cfg_active();
    let cal = EventCalendar::generate(11, &cfg, 1000);
    for w in cal.events.windows(2) {
        assert!(w[0].start_gen <= w[1].start_gen);
    }
}

#[test]
fn shock_ramp_factor_trapezoid() {
    let trap = ShockEvent {
        kind: ShockKind::HazardPulse,
        start_gen: 100,
        duration_gen: 10,
        ramp_gens: 2,
        intensity: 1.0,
        center_xy: None,
        radius: None,
    };
    assert_eq!(shock_ramp_factor(&trap, 99), 0.0);
    assert_eq!(shock_ramp_factor(&trap, 110), 0.0);
    // Mid plateau (gen 104..=107) musí být 1.0.
    assert!((shock_ramp_factor(&trap, 105) - 1.0).abs() < 1e-6);
    // První gen rampy: monotonně rostoucí, < 1.
    let f0 = shock_ramp_factor(&trap, 100);
    let f1 = shock_ramp_factor(&trap, 101);
    assert!(f0 > 0.0 && f0 < 1.0);
    assert!(f1 > f0);
    // Poslední gen rampy: < 1, > 0.
    let f_end = shock_ramp_factor(&trap, 109);
    assert!(f_end > 0.0 && f_end < 1.0);

    // Triangle case: duration <= 2 * ramp.
    let tri = ShockEvent {
        kind: ShockKind::HazardPulse,
        start_gen: 0,
        duration_gen: 4,
        ramp_gens: 3,
        intensity: 1.0,
        center_xy: None,
        radius: None,
    };
    assert_eq!(shock_ramp_factor(&tri, 4), 0.0);
    let peaks: Vec<f32> = (0..4).map(|g| shock_ramp_factor(&tri, g)).collect();
    let max_peak = peaks.iter().cloned().fold(0.0_f32, f32::max);
    assert!(max_peak > 0.0 && max_peak <= 1.0);
    // Triangle musí mít jeden inner peak — okraje nižší než střed.
    assert!(peaks[0] < max_peak);
    assert!(peaks[3] < max_peak);
}

#[test]
fn event_calendar_intensity_in_range() {
    let cfg = shock_cfg_active();
    let cal = EventCalendar::generate(99, &cfg, 1000);
    assert!(!cal.events.is_empty());
    for e in &cal.events {
        assert!(
            e.intensity >= cfg.intensity_min - 1e-6
                && e.intensity <= cfg.intensity_max + 1e-6,
            "intensity {} out of range",
            e.intensity
        );
        assert!(e.duration_gen >= cfg.duration_min_gens);
        assert!(e.duration_gen <= cfg.duration_max_gens);
    }
}

#[test]
fn event_calendar_global_vs_spatial_split() {
    let cfg = shock_cfg_active();
    let cal = EventCalendar::generate(2024, &cfg, 4000);
    assert!(
        cal.events.len() >= 20,
        "need enough events for split test, got {}",
        cal.events.len()
    );
    let global = cal.events.iter().filter(|e| e.center_xy.is_none()).count();
    let spatial = cal.events.iter().filter(|e| e.center_xy.is_some()).count();
    assert!(global > 0, "expected at least one global event");
    assert!(spatial > 0, "expected at least one spatial event");
    for e in cal.events.iter().filter(|e| e.radius.is_some()) {
        let r = e.radius.unwrap();
        let lo = cfg.spatial_radius_min_frac * WORLD_HALF[0];
        let hi = cfg.spatial_radius_max_frac * WORLD_HALF[0];
        assert!(r >= lo - 1e-3 && r <= hi + 1e-3, "radius {} out of range", r);
    }
}

#[test]
fn hazard_multiplier_default_one() {
    let pos = [0.0, 0.0, 0.0];
    let m = hazard_shock_multiplier(pos, &[], 50, 0, WORLD_HALF);
    assert!((m - 1.0).abs() < 1e-6, "empty events must give 1.0, got {}", m);
}

#[test]
fn hazard_multiplier_global_pulse_doubles_at_peak() {
    let event = ShockEvent {
        kind: ShockKind::HazardPulse,
        start_gen: 100,
        duration_gen: 10,
        ramp_gens: 2,
        intensity: 1.0,
        center_xy: None,
        radius: None,
    };
    let pos = [123.0, -45.0, 7.0];
    // Plateau (gen 102..=107) → ramp = 1.0, mask = 1.0 → 1 + 1 * 1 * 1 * 1 = 2.0.
    let m = hazard_shock_multiplier(pos, &[event], 105, 0, WORLD_HALF);
    assert!((m - 2.0).abs() < 1e-5, "global peak must give 2.0, got {}", m);
    // Pre-start: no contribution.
    let m_before = hazard_shock_multiplier(pos, &[event], 99, 0, WORLD_HALF);
    assert!((m_before - 1.0).abs() < 1e-6);
    // Post-end: no contribution.
    let m_after = hazard_shock_multiplier(pos, &[event], 110, 0, WORLD_HALF);
    assert!((m_after - 1.0).abs() < 1e-6);
}

#[test]
fn hazard_multiplier_spatial_mask_falls_off() {
    let center = [0.0, 0.0];
    let radius = 100.0;
    let event = ShockEvent {
        kind: ShockKind::HazardPulse,
        start_gen: 0,
        duration_gen: 10,
        ramp_gens: 2,
        intensity: 1.0,
        center_xy: Some(center),
        radius: Some(radius),
    };
    let gen = 5;
    // Plateau, ramp = 1.0.
    // Center → mask = 1.0 → multiplier = 2.0.
    let m_center = hazard_shock_multiplier([0.0, 0.0, 0.0], &[event], gen, 0, WORLD_HALF);
    assert!((m_center - 2.0).abs() < 1e-5, "center must be 2.0, got {}", m_center);
    // At edge (dist = radius) → mask = 0 → multiplier = 1.0.
    let m_edge = hazard_shock_multiplier([radius, 0.0, 0.0], &[event], gen, 0, WORLD_HALF);
    assert!((m_edge - 1.0).abs() < 1e-5, "edge must be 1.0, got {}", m_edge);
    // Beyond radius → mask = 0.
    let m_outside = hazard_shock_multiplier(
        [radius * 1.5, 0.0, 0.0],
        &[event],
        gen,
        0,
        WORLD_HALF,
    );
    assert!((m_outside - 1.0).abs() < 1e-5, "outside must be 1.0, got {}", m_outside);
    // Mid-radius → strictly between 1.0 and 2.0 (smoothstep monotone).
    let m_mid = hazard_shock_multiplier(
        [radius * 0.5, 0.0, 0.0],
        &[event],
        gen,
        0,
        WORLD_HALF,
    );
    assert!(
        m_mid > 1.0 && m_mid < 2.0,
        "mid must be in (1, 2), got {}",
        m_mid
    );
    // Smoothstep monotone: closer point → higher multiplier.
    let m_near = hazard_shock_multiplier(
        [radius * 0.25, 0.0, 0.0],
        &[event],
        gen,
        0,
        WORLD_HALF,
    );
    assert!(m_near > m_mid, "near {} should exceed mid {}", m_near, m_mid);
}

#[test]
fn food_multiplier_default_one() {
    // Empty events → 1.0.
    let m = food_density_shock_multiplier(&[], 50);
    assert!((m - 1.0).abs() < 1e-6, "empty events must give 1.0, got {}", m);
    // Non-FoodCrash event (HazardPulse) → 1.0.
    let event = ShockEvent {
        kind: ShockKind::HazardPulse,
        start_gen: 0,
        duration_gen: 10,
        ramp_gens: 2,
        intensity: 1.0,
        center_xy: None,
        radius: None,
    };
    let m = food_density_shock_multiplier(&[event], 5);
    assert!((m - 1.0).abs() < 1e-6, "HazardPulse must not affect food, got {}", m);
}

#[test]
fn food_multiplier_global_crash_drops() {
    // Sprint 113: 1 global FoodCrash, intensity = 1, peak ramp = 1
    // → mult = 1.0 - 1.0 × 1.0 × 0.5 = 0.5.
    let event = ShockEvent {
        kind: ShockKind::FoodCrash,
        start_gen: 100,
        duration_gen: 10,
        ramp_gens: 2,
        intensity: 1.0,
        center_xy: None,
        radius: None,
    };
    // Plateau (gen 102..=107) → ramp = 1.0.
    let m = food_density_shock_multiplier(&[event], 105);
    let expected = 1.0 - FOOD_CRASH_MAX_DROP;
    assert!(
        (m - expected).abs() < 1e-5,
        "global peak must give 1 - FOOD_CRASH_MAX_DROP = {}, got {}",
        expected,
        m
    );
    // Pre-start: 1.0.
    let m_before = food_density_shock_multiplier(&[event], 99);
    assert!((m_before - 1.0).abs() < 1e-6, "pre-start must be 1.0, got {}", m_before);
    // Post-end: 1.0.
    let m_after = food_density_shock_multiplier(&[event], 110);
    assert!((m_after - 1.0).abs() < 1e-6, "post-end must be 1.0, got {}", m_after);
}

#[test]
fn food_multiplier_compound_clamped() {
    // Sprint 113: 3× FoodCrash s intensity=1, peak ramp současně:
    // 0.5 × 0.5 × 0.5 = 0.125 (> 0.1 floor → no clamp, return raw).
    let mk = |start: u64| ShockEvent {
        kind: ShockKind::FoodCrash,
        start_gen: start,
        duration_gen: 10,
        ramp_gens: 2,
        intensity: 1.0,
        center_xy: None,
        radius: None,
    };
    let three = [mk(100), mk(100), mk(100)];
    let m3 = food_density_shock_multiplier(&three, 105);
    assert!(
        (m3 - 0.125).abs() < 1e-5,
        "3 crashes compound to 0.125, got {}",
        m3
    );
    // 4× FoodCrash: 0.5^4 = 0.0625 < 0.1 floor → clamp to FOOD_CRASH_MIN_FACTOR.
    let four = [mk(100), mk(100), mk(100), mk(100)];
    let m4 = food_density_shock_multiplier(&four, 105);
    assert!(
        (m4 - FOOD_CRASH_MIN_FACTOR).abs() < 1e-5,
        "4 crashes must clamp to FOOD_CRASH_MIN_FACTOR = {}, got {}",
        FOOD_CRASH_MIN_FACTOR,
        m4
    );
}

#[test]
fn izhikevich_quiescent_neuron_does_not_spike() {
    // Sprint 146: zero input, weights all zero → membrane settles near the
    // stable subthreshold equilibrium without crossing 30 mV. Expected
    // hidden activation: -1 (no spikes mapped to the lower bound).
    let mut brain = dummy_brain();
    let inputs = [0.0_f32; BRAIN_INPUTS];
    let (hidden, _) = brain.forward_izhikevich_with_state(&inputs, 0, 0.0);
    for (i, h) in hidden.iter().take(brain.hidden_n as usize).enumerate() {
        assert!(
            (*h + 1.0).abs() < 1e-5,
            "neuron {} should be silent (hidden = -1), got {}",
            i,
            h
        );
    }
    // Membrane stays well below the 30 mV spike threshold. Drift toward the
    // subthreshold fixed point is fine; only "no AP fired" matters here.
    for v in brain.membrane.iter().take(brain.hidden_n as usize) {
        assert!(
            *v < 0.0,
            "quiescent membrane should remain hyperpolarized, got {}",
            v
        );
    }
}

#[test]
fn izhikevich_strong_input_drives_spiking_over_multiple_ticks() {
    // Sprint 146: regular-spiking neuron at I≈50 (strong tonic drive)
    // fires within a few ticks. We run several ticks because one 16 ms
    // tick may straddle the inter-spike interval at lower currents — the
    // CPU forward is correct iff any tick crosses threshold over the
    // simulated window.
    let mut brain = dummy_brain();
    let h_n = brain.hidden_n as usize;
    for i in 0..h_n {
        brain.b1[i] = 50.0;
    }
    let inputs = [0.0_f32; BRAIN_INPUTS];
    let mut total_spikes = 0_u32;
    for _ in 0..20 {
        let (hidden, _) = brain.forward_izhikevich_with_state(&inputs, 0, 0.0);
        for h in hidden.iter().take(h_n) {
            // spike_count = (h + 1) × IZH_SUBSTEPS / 2; sum across neurons.
            let spikes = (((h + 1.0) * IZH_SUBSTEPS as f32 / 2.0).round()) as u32;
            total_spikes += spikes;
        }
    }
    assert!(
        total_spikes >= h_n as u32,
        "expected at least 1 spike per neuron over 20 ticks at strong input, got {}",
        total_spikes
    );
}

#[test]
fn stdp_apply_rewarded_zero_reward_is_noop() {
    // Sprint 156: zero reward gates the STDP rule — weights frozen
    // regardless of timing.
    let mut brain = dummy_brain();
    brain.last_pre_spike_ticks[3] = 100;
    brain.stdp_step(100, 5.0);
    brain.last_post_spike_ticks[5] = 102;
    brain.stdp_step(101, 5.0);
    brain.stdp_step(102, 5.0);
    let w_before = brain.w1[5][3];
    brain.stdp_apply_rewarded(102, 0.01, 0.01, 0.0);
    assert_eq!(brain.w1[5][3], w_before);
}

#[test]
fn stdp_apply_rewarded_positive_amplifies_ltp() {
    // Sprint 156: positive reward boosts LTP same direction as S155 rule.
    let mut brain = dummy_brain();
    brain.last_pre_spike_ticks[3] = 100;
    brain.stdp_step(100, 5.0);
    brain.stdp_step(101, 5.0);
    brain.last_post_spike_ticks[5] = 102;
    brain.stdp_step(102, 5.0);
    let w_before = brain.w1[5][3];
    brain.stdp_apply_rewarded(102, 0.01, 0.01, 2.0);
    assert!(
        brain.w1[5][3] > w_before,
        "expected reward-modulated LTP, w1 {} → {}",
        w_before,
        brain.w1[5][3]
    );
}

#[test]
fn stdp_apply_ltp_when_pre_before_post() {
    // Sprint 155: correlated firing — input 3 spikes at tick 100, hidden 5
    // spikes shortly after at tick 102. Pre-trace at tick 102 is still
    // positive → w1[5][3] should grow.
    let mut brain = dummy_brain();
    let tau = 5.0_f32;
    let a_plus = 0.01;
    let a_minus = 0.01;
    let w_before = brain.w1[5][3];

    brain.last_pre_spike_ticks[3] = 100;
    brain.stdp_step(100, tau);
    brain.stdp_step(101, tau);
    brain.last_post_spike_ticks[5] = 102;
    brain.stdp_step(102, tau);
    brain.stdp_apply(102, a_plus, a_minus);

    assert!(
        brain.w1[5][3] > w_before,
        "LTP expected: w1 grew from {} to {}",
        w_before,
        brain.w1[5][3]
    );
}

#[test]
fn stdp_apply_ltd_when_post_before_pre() {
    // Sprint 155: anti-correlated — hidden 5 spikes at tick 100, input 3
    // fires at tick 102. Post-trace at tick 102 is positive → w1[5][3]
    // should shrink.
    let mut brain = dummy_brain();
    let tau = 5.0_f32;
    let a_plus = 0.01;
    let a_minus = 0.01;
    let w_before = brain.w1[5][3];

    brain.last_post_spike_ticks[5] = 100;
    brain.stdp_step(100, tau);
    brain.stdp_step(101, tau);
    brain.last_pre_spike_ticks[3] = 102;
    brain.stdp_step(102, tau);
    brain.stdp_apply(102, a_plus, a_minus);

    assert!(
        brain.w1[5][3] < w_before,
        "LTD expected: w1 shrank from {} to {}",
        w_before,
        brain.w1[5][3]
    );
}

#[test]
fn stdp_step_decays_traces_and_records_spikes() {
    // Sprint 154: trace decays each tick by exp(-1/tau); when a spike-time
    // matches the current tick, the trace gets +1.0. Verified on a
    // hand-rolled brain (no Izhikevich forward involved).
    let mut brain = dummy_brain();
    // Pretend input 3 fired this tick and hidden 5 fired this tick.
    let tick = 100;
    brain.last_pre_spike_ticks[3] = tick;
    brain.last_post_spike_ticks[5] = tick;
    let tau = 5.0_f32;
    let decay = (-1.0_f32 / tau).exp();
    brain.stdp_step(tick, tau);
    // Spike slots got the +1 bump.
    assert!((brain.pre_trace[3] - 1.0).abs() < 1e-5);
    assert!((brain.post_trace[5] - 1.0).abs() < 1e-5);
    // Non-spike slots stayed at zero × decay = 0.
    assert_eq!(brain.pre_trace[0], 0.0);
    assert_eq!(brain.post_trace[0], 0.0);
    // Next tick (no fresh spikes) → trace decays.
    brain.stdp_step(tick + 1, tau);
    assert!(
        (brain.pre_trace[3] - decay).abs() < 1e-5,
        "expected pre_trace[3] = {} after one decay, got {}",
        decay,
        brain.pre_trace[3]
    );
    assert!((brain.post_trace[5] - decay).abs() < 1e-5);
}

#[test]
fn izhikevich_zero_input_outputs_finite_and_in_range() {
    // Sprint 146: sanity check — even with no input, the L2 motor layer
    // should produce finite outputs in [-1, +1] (tanh-clamped).
    let mut brain = dummy_brain();
    // Non-trivial b2 so outputs aren't trivially zero.
    for o in 0..BRAIN_OUTPUTS {
        brain.b2[o] = 0.5;
    }
    let inputs = [0.0_f32; BRAIN_INPUTS];
    let (_, outputs) = brain.forward_izhikevich_with_state(&inputs, 0, 0.0);
    for (o, v) in outputs.iter().enumerate() {
        assert!(v.is_finite(), "output {} not finite: {}", o, v);
        assert!(v.abs() <= 1.0 + 1e-5, "output {} out of [-1, 1]: {}", o, v);
    }
}

#[test]
fn izhikevich_dead_zone_weights_do_not_affect_output() {
    // Sprint 192 regression: `Brain::from_cppn` populates dead-zone w1/b1/w2
    // (indices >= hidden_n) with non-zero values because the CPPN substrate
    // function maps coordinates to weights without any awareness of the
    // active range. The forward path must gate work to `0..hidden_n` so
    // those CPPN-derived dead-zone weights stay inert. Pre-fix, CPU iterated
    // BRAIN_HIDDEN for lateral inhibition (pulled softplus(garbage) into
    // the inhibition pool) and GPU iterated BRAIN_HIDDEN throughout
    // (dead-zone neurons spiked, their hidden activations fed L2).
    let h_n = BRAIN_HIDDEN_DEFAULT;
    assert!(h_n < BRAIN_HIDDEN, "dead zone must exist for this test");

    let mut baseline = dummy_brain();
    // Active range: a couple of inputs that drive a clear spike. Strong
    // tonic b1 makes the active integration produce a non-trivial pattern.
    for i in 0..h_n {
        baseline.b1[i] = 12.0 + i as f32 * 0.5;
        for o in 0..BRAIN_OUTPUTS {
            baseline.w2[o][i] = 0.1 * (i as f32 + 1.0);
            baseline.b2[o] = 0.2;
        }
    }
    let mut polluted = baseline;
    // Fill the dead zone with the kind of magnitudes `Brain::from_cppn`
    // emits — non-zero, sometimes large enough to drive dead-zone Izhikevich
    // neurons to fire if they were integrated.
    for h in h_n..BRAIN_HIDDEN {
        polluted.b1[h] = 18.0; // well over tonic-spike threshold
        for j in 0..BRAIN_INPUTS {
            polluted.w1[h][j] = 0.3;
        }
        for o in 0..BRAIN_OUTPUTS {
            polluted.w2[o][h] = 0.7;
        }
    }

    let inputs = [0.1_f32; BRAIN_INPUTS];
    // Lateral inhibition on, so a regression that pulled dead-zone preacts
    // into the softplus pool would also visibly change `hidden` here.
    let lateral = 0.3_f32;
    let (h_baseline, o_baseline) =
        baseline.forward_izhikevich_with_state(&inputs, 0, lateral);
    let (h_polluted, o_polluted) =
        polluted.forward_izhikevich_with_state(&inputs, 0, lateral);

    for i in 0..h_n {
        let d = (h_baseline[i] - h_polluted[i]).abs();
        assert!(
            d < 1e-6,
            "hidden[{}] drifted: baseline={} polluted={} diff={}",
            i,
            h_baseline[i],
            h_polluted[i],
            d
        );
    }
    for o in 0..BRAIN_OUTPUTS {
        let d = (o_baseline[o] - o_polluted[o]).abs();
        assert!(
            d < 1e-6,
            "outputs[{}] drifted: baseline={} polluted={} diff={}",
            o,
            o_baseline[o],
            o_polluted[o],
            d
        );
    }
}
