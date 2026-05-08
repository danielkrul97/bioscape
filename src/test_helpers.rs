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
        damage_accum: 0.0,
        age: 0,
        reproduce_cooldown_ticks: 0,
        cell_id: 0,
        bonds: [None; MAX_BONDS_PER_CELL],
        cell_state: 0.5,
        last_best_food_d2: f32::MAX,
        phenotype,
        genome,
    }
}
