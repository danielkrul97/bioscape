//! Sdílené fixtures pro unit testy. Cílem je, aby paralelní test soubory
//! (`tests_*.rs`) sdílely stejné default Cell/Genome/Brain hodnoty bez
//! duplikace. Existující `src/tests.rs` zatím definuje vlastní lokální
//! kopie pro zpětnou kompatibilitu — nové testy by měly importovat odsud.

#![allow(dead_code)]

use crate::*;

pub fn dummy_brain() -> Brain {
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

pub fn dummy_genome() -> Genome {
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

pub fn zero_cfg() -> MutationConfig {
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

pub fn no_drag_physics(cost_per_v_sq: f32, vision_cost: f32) -> PhysicsConfig {
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

pub fn base_cell() -> Cell {
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
        xoshiro_state: Xoshiro128PlusPlus::from_cell_id(0),
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

/// Per-cell whisker spring-damper state buffer (12 f32/cell) — the test-side
/// mirror of `CellsGpu::whisker_state_buf`. Sensor-gather tests that run with
/// `maze_active = 0` still need binding 18 populated for the bind group.
pub fn whisker_state_buf(device: &wgpu::Device, n: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test-whisker-state"),
        size: (n * 12 * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}
