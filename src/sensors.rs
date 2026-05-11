use crate::*;

/// Sprint 40: senzorický kontext brainu. Volá se z hot loop binárek po
/// průzkumu okolí (nejbližší food/cell, density, gradient field). Konkrétní
/// gathering algoritmus závisí na binárce (main grid lookup vs headless
/// O(N²) sweep), ale výstup struct je společný — pak `populate_brain_inputs`
/// vyplní sjednocený `[f32; BRAIN_INPUTS]` array.
#[derive(Debug, Clone, Copy)]
pub struct BrainSensors {
    /// Sprint 54: min-imaged signed delta (target − cell), ne absolutní target
    /// pozice. Pro toroidal world je delta správný relativní vektor i přes
    /// world wrap (cell na x=−950 vidí target na x=+950 jako Δx=20, ne 1900).
    /// Pre-Sprint-54 byl absolute target_pos; populate_brain_inputs odečetl
    /// cell.position. Po Sprintu 54 je odečtení už hotové (s wrap).
    pub nearest_food: Option<[f32; 3]>,
    pub nearest_cell: Option<([f32; 3], f32)>,
    pub neighbors_in_vision: u32,
    pub smell_grad: [f32; 3],
    /// Sprint 126: per-channel pheromone gradients. ch0 (= existing slow
    /// channel) backward-compat sloty 11/12/19 v populate_brain_inputs;
    /// ch1, ch2 nové sloty 21-23 / 24-26.
    pub pheromone_grads: [[f32; 3]; N_PHEROMONE_CHANNELS],
    /// Sprint 87: aktuální teplota na cell pozici (sim units, ne normalized).
    /// Caller spočítá `temperature_at_z(pos[2], world_half, tick, gen)`.
    /// `populate_brain_inputs` normalizuje přes `tanh((T − REF) / 10)` →
    /// brain input [-1, 1] (Q10-aware škálování).
    pub temperature_local: f32,
    /// Mechanosensory gradient at cell position. Source = motion-driven
    /// `VibrationField` (separate from chemical fields). Slots [29..32] in
    /// `populate_brain_inputs`, tanh-normalized with
    /// `VIBRATION_NORMALIZATION_GAIN`.
    pub vibration_grad: [f32; 3],
    /// Local amplitude of the vibration field (= scalar sample at cell pos).
    /// Slot [32]. Combined with `vibration_grad` the brain can locate +
    /// gauge nearby movers.
    pub vibration_amp: f32,
}

/// Sprint 40: jediný source of truth pro brain inputs layout. Pre-refactor byl
/// duplikovaný v `main::cells_brain_act` a `headless::brain_act` — drift mezi
/// nimi by tichost porušil binární CSV identity. `damage_accum` se zde čte +
/// resetuje (1-tick delay konzistentně se Sprint 30 semantikou).
pub fn populate_brain_inputs(
    cell: &mut Cell,
    sensors: &BrainSensors,
    vision_r: f32,
) -> [f32; BRAIN_INPUTS] {
    let pos = cell.position;
    let my_radius = cell.phenotype.effective_radius().max(0.01);
    let max_speed = cell.genome.max_speed;
    // Sprint 32 note: hypot(vx, vy) místo (vx²+vy²+vz²).sqrt() pro ULP
    // identity s pre-Sprint-32 trajektorií. Sprint 33+ vz != 0; hypot
    // ignoruje vz, ale rozdíl je sub-ULP. Sprint 41+ může přejít na 3D mag.
    let speed_norm = (cell.velocity[0].hypot(cell.velocity[1]) / max_speed).clamp(0.0, 1.0);
    let energy_norm = (cell.energy / REPRODUCE_THRESHOLD).clamp(0.0, 1.5);

    let mut inputs = [0.0_f32; BRAIN_INPUTS];
    if let Some(delta) = sensors.nearest_food {
        // Sprint 54: BrainSensors už nese min-imaged delta (toroidal-aware).
        inputs[0] = delta[0] / vision_r;
        inputs[1] = delta[1] / vision_r;
        inputs[15] = delta[2] / vision_r;
    }
    if let Some((delta, other_radius)) = sensors.nearest_cell {
        inputs[2] = delta[0] / vision_r;
        inputs[3] = delta[1] / vision_r;
        inputs[6] = (other_radius - my_radius) / my_radius;
        inputs[16] = delta[2] / vision_r;
    }
    let _ = pos;
    inputs[4] = energy_norm;
    inputs[5] = speed_norm;
    inputs[7] = (sensors.smell_grad[0] * SMELL_NORMALIZATION_GAIN).tanh();
    inputs[8] = (sensors.smell_grad[1] * SMELL_NORMALIZATION_GAIN).tanh();
    inputs[17] = (sensors.smell_grad[2] * SMELL_NORMALIZATION_GAIN).tanh();
    let fwd = forward_vector(cell.heading, cell.pitch);
    inputs[9] = fwd[0];
    inputs[10] = fwd[1];
    inputs[18] = fwd[2];
    inputs[11] = (sensors.pheromone_grads[0][0] * PHEROMONE_NORMALIZATION_GAIN).tanh();
    inputs[12] = (sensors.pheromone_grads[0][1] * PHEROMONE_NORMALIZATION_GAIN).tanh();
    inputs[19] = (sensors.pheromone_grads[0][2] * PHEROMONE_NORMALIZATION_GAIN).tanh();
    inputs[13] = (sensors.neighbors_in_vision as f32 / DENSITY_NORM_COUNT).tanh();
    inputs[14] = (cell.damage_accum * DAMAGE_NORMALIZATION_GAIN).tanh();
    cell.damage_accum = 0.0;
    // Sprint 87: thermal awareness, slot 20. Q10-aware tanh normalizace —
    // (T - REF) / 10 dává tanh(±1.3) ≈ ±0.86 na endpoints [BOTTOM, TOP],
    // tanh(0) = 0 na ref. Diurnal/seasonal posuny mohou krátkodobě saturovat
    // k ±1, což je akceptovatelná oversaturace pro brain signal.
    inputs[20] = ((sensors.temperature_local - THERMAL_REF_TEMP) / 10.0).tanh();
    // Sprint 126: ch1, ch2 pheromone gradients. ch0 zachované na sloty 11/12/19.
    inputs[21] = (sensors.pheromone_grads[1][0] * PHEROMONE_NORMALIZATION_GAIN).tanh();
    inputs[22] = (sensors.pheromone_grads[1][1] * PHEROMONE_NORMALIZATION_GAIN).tanh();
    inputs[23] = (sensors.pheromone_grads[1][2] * PHEROMONE_NORMALIZATION_GAIN).tanh();
    inputs[24] = (sensors.pheromone_grads[2][0] * PHEROMONE_NORMALIZATION_GAIN).tanh();
    inputs[25] = (sensors.pheromone_grads[2][1] * PHEROMONE_NORMALIZATION_GAIN).tanh();
    inputs[26] = (sensors.pheromone_grads[2][2] * PHEROMONE_NORMALIZATION_GAIN).tanh();
    // Bond-mediated communication inbox (mean of partners' last_outputs[12..14]).
    // Computed pre-brain_act by `pool_bond_messages`. Solo cells: 0.
    for k in 0..N_BOND_MSG_CHANNELS {
        inputs[27 + k] = cell.bonded_inbox[k];
    }
    // Mechanosensory inputs — vibration gradient + amplitude. Slots
    // [29..32] = grad_{x,y,z}, [32] = amp. tanh-normalized via
    // `VIBRATION_NORMALIZATION_GAIN`; saturates on a single loud neighbor.
    let vib_base = 27 + N_BOND_MSG_CHANNELS;
    inputs[vib_base] = (sensors.vibration_grad[0] * VIBRATION_NORMALIZATION_GAIN).tanh();
    inputs[vib_base + 1] = (sensors.vibration_grad[1] * VIBRATION_NORMALIZATION_GAIN).tanh();
    inputs[vib_base + 2] = (sensors.vibration_grad[2] * VIBRATION_NORMALIZATION_GAIN).tanh();
    inputs[vib_base + 3] = (sensors.vibration_amp * VIBRATION_NORMALIZATION_GAIN).tanh();
    // Sprint 94: cluster-shared brain. Recurrent slots (21..52) čtou
    // `pooled_hidden` (mean self + bonded neighbors z předchozího ticku)
    // místo `last_hidden`. Solo cells: pool == self → behavior identical
    // s pre-Sprint-94. Cluster cells: shared memory → effective větší
    // context window, drives proto-distributed cognition.
    inputs[BRAIN_INPUTS_SENSORY..BRAIN_INPUTS_SENSORY + BRAIN_RECURRENT]
        .copy_from_slice(&cell.pooled_hidden[..BRAIN_RECURRENT]);
    inputs
}

/// LOS check for the sensor gather closures. Returns true if the segment
/// `origin → target` is unobstructed by maze walls (or if no obstacle field
/// is active). Cells that fail the test must be excluded from
/// `nearest_food` / `nearest_cell` / `neighbors_in_vision` candidates so the
/// brain cannot see "through" walls.
#[inline]
pub fn los_clear(field: Option<&ObstacleField>, origin: [f32; 3], target: [f32; 3]) -> bool {
    match field {
        Some(f) => !f.raycast_blocked(origin, target),
        None => true,
    }
}

/// Per-tick drain coefficient for sensor specialization:
/// `drain = sum(sensor_gains) × this × dt`. Default neutral sum is 3 × 1.0,
/// so the baseline drain is ~0.9/s — comparable with body maintenance.
/// Cells that turn off duplicate sensors in a cluster save proportionally,
/// making specialization net-positive.
pub const SENSOR_GAIN_COST: f32 = 0.3;
/// Per-category gain range. 0 = sensor effectively off, 1 = neutral, 2 =
/// boosted (better detection, higher cost).
pub const MIN_SENSOR_GAIN: f32 = 0.0;
pub const MAX_SENSOR_GAIN: f32 = 2.0;
/// Four sensor categories indexed into `Genome.sensor_gains`:
/// 0 = Vision (food delta, cell delta, rel_size, density),
/// 1 = Chemistry (smell + pheromone gradients),
/// 2 = Defensive (damage signal, thermal_local),
/// 3 = Mechano (vibration gradient + amplitude — motion-driven field). Proprio
/// sensors (energy, speed, heading) are always-on and not gained — a cell
/// still needs its own state even in a deep-specialist mode.
pub const SENSOR_CATEGORY_VISION: usize = 0;
pub const SENSOR_CATEGORY_CHEMISTRY: usize = 1;
pub const SENSOR_CATEGORY_DEFENSIVE: usize = 2;
pub const SENSOR_CATEGORY_MECHANO: usize = 3;
pub const N_SENSOR_CATEGORIES: usize = 4;

/// Map a brain input slot index to its sensor category, or `None` for
/// proprio slots (not gained, not pooled). Used by `apply_sensor_gains`
/// (per-cell multiply) and `pool_bonded_sensors` (max-pool environmental
/// slots over the bond network).
#[inline]
pub fn sensor_slot_category(slot: usize) -> Option<usize> {
    match slot {
        // Food delta (slot 0,1,15) + cell delta (2,3,16) + rel_size (6) +
        // density (13) → Vision
        0 | 1 | 2 | 3 | 6 | 13 | 15 | 16 => Some(SENSOR_CATEGORY_VISION),
        // Smell (7,8,17) + pheromone (11,12,19) → Chemistry
        7 | 8 | 11 | 12 | 17 | 19 => Some(SENSOR_CATEGORY_CHEMISTRY),
        // Damage (14) + thermal (20) → Defensive
        14 | 20 => Some(SENSOR_CATEGORY_DEFENSIVE),
        // Vibration gradient + amplitude (29..32) → Mechano. Pooling over
        // bonded partners is automatic via `pool_bonded_sensors` once these
        // slots are categorized — that is how cells in a cluster share the
        // mechanosensory channel.
        29 | 30 | 31 | 32 => Some(SENSOR_CATEGORY_MECHANO),
        // Energy (4), speed (5), heading (9,10,18) → proprio, no gain
        _ => None,
    }
}

/// Sprint 97: in-place multiplication brain inputs × per-category sensor_gains.
/// Applies gains to environmental + defensive slots, leaves proprio slots
/// (energy, speed, heading) untouched. Solo cells s gain < 1.0 lose info,
/// cluster cells s pooling can compensate via partner signals.
#[inline]
pub fn apply_sensor_gains(inputs: &mut [f32; BRAIN_INPUTS], gains: &[f32; N_SENSOR_CATEGORIES]) {
    for slot in 0..BRAIN_INPUTS_SENSORY {
        if let Some(cat) = sensor_slot_category(slot) {
            inputs[slot] *= gains[cat];
        }
    }
}

/// Sprint 97: pool environmental + defensive sensor slots přes bond network.
/// Max-pooling — for each slot, take maximum over self + bonded partners
/// (post-gain-multiplied values). Allows specialization: cell A s vision gain=0
/// dostane visibility z partnera B s vision gain=2.0 přes max pool.
///
/// Proprio slots (energy, speed, heading) NEpoolováno — vlastní stav cells
/// musí být individuální. Recurrent slots (21..52) mají vlastní mechanismus
/// `pool_bonded_hidden` (S94 mean pooling).
///
/// Solo cell (no bonds): output == self_inputs. Pair / cluster: max nad
/// self + 1-hop partners.
pub fn pool_bonded_sensors<F>(
    cell: &Cell,
    own_inputs: &[f32; BRAIN_INPUTS],
    lookup_partner_inputs: F,
) -> [f32; BRAIN_INPUTS]
where
    F: Fn(u64) -> Option<[f32; BRAIN_INPUTS]>,
{
    let mut pooled = *own_inputs;
    for bond_slot in cell.bonds.iter().flatten() {
        if let Some(partner_inputs) = lookup_partner_inputs(bond_slot.other_cell_id) {
            for slot in 0..BRAIN_INPUTS_SENSORY {
                if sensor_slot_category(slot).is_some() {
                    let p = partner_inputs[slot];
                    if p.abs() > pooled[slot].abs() {
                        // Use max-magnitude (preserves sign for directional
                        // signals like food_dx; partner with stronger signal
                        // wins). Solo equivalent to self.
                        pooled[slot] = p;
                    }
                }
            }
        }
    }
    pooled
}

/// Sprint 94: compute pooled `last_hidden` for a single cell — mean over
/// self + bonded neighbors. Output should be assigned to `cell.pooled_hidden`
/// pre brain_act phase. Bond lookup: `bond.other_cell_id → idx` via caller-
/// supplied lookup (HashMap or array). Missing neighbors (despawned mid-tick)
/// jsou skipnuty.
///
/// Solo cell (n_bonds=0): output == self.last_hidden (no change).
/// Pair (1 bond): output = (self + partner) / 2.
/// Triad / cluster: arithmetic mean over alive bonded subgraph (1-hop only,
/// no transitive — keeps O(n_bonds) per cell, no graph traversal cost).
pub fn pool_bonded_hidden<F>(
    cell: &Cell,
    lookup_partner_hidden: F,
) -> [f32; BRAIN_HIDDEN]
where
    F: Fn(u64) -> Option<[f32; BRAIN_HIDDEN]>,
{
    let mut acc = cell.last_hidden;
    let mut count = 1.0_f32;
    for slot in cell.bonds.iter().flatten() {
        if let Some(partner_hidden) = lookup_partner_hidden(slot.other_cell_id) {
            for k in 0..BRAIN_HIDDEN {
                acc[k] += partner_hidden[k];
            }
            count += 1.0;
        }
    }
    if count > 1.0 {
        let inv = 1.0 / count;
        for k in 0..BRAIN_HIDDEN {
            acc[k] *= inv;
        }
    }
    acc
}

/// Bond-mediated message inbox. Mean over bonded peers' last_outputs[12..14]
/// (= partner brain message channels). Solo cell vrátí [0; N]. Used jako
/// pre-brain_act pass: caller assigns výsledek do `cell.bonded_inbox`,
/// `populate_brain_inputs` ho čte do slots [27..29].
pub fn pool_bond_messages<F>(
    cell: &Cell,
    lookup_partner_outputs: F,
) -> [f32; N_BOND_MSG_CHANNELS]
where
    F: Fn(u64) -> Option<[f32; BRAIN_OUTPUTS]>,
{
    let mut acc = [0.0_f32; N_BOND_MSG_CHANNELS];
    let mut count = 0.0_f32;
    for slot in cell.bonds.iter().flatten() {
        if let Some(partner_outputs) = lookup_partner_outputs(slot.other_cell_id) {
            for k in 0..N_BOND_MSG_CHANNELS {
                acc[k] += partner_outputs[BRAIN_OUTPUTS - N_BOND_MSG_CHANNELS + k];
            }
            count += 1.0;
        }
    }
    if count > 0.0 {
        let inv = 1.0 / count;
        for k in 0..N_BOND_MSG_CHANNELS {
            acc[k] *= inv;
        }
    }
    acc
}

/// Per-cell vibration emission rate — amplitude added to the field at
/// `cell.position` each tick (caller multiplies by `dt`). Composed of two
/// terms: linear speed normalized by the cell's own `max_speed`, and angular
/// speed normalized by `turn_rate`. Both terms are clamped so a cell never
/// emits unbounded amounts even if a numerical excursion temporarily exceeds
/// its quoted maxima. Pure read — no side effects.
///
/// Single source of truth: headless `World::update_vibration`,
/// `csv::write_stats`, and the renderer's `update_vibration_field` all call
/// this so the emission formula stays in one place.
#[inline]
pub fn vibration_emit_for_cell(cell: &Cell) -> f32 {
    let max_speed = cell.genome.max_speed.max(1e-3);
    let turn_rate = cell.genome.turn_rate.max(1e-3);
    let vx = cell.velocity[0];
    let vy = cell.velocity[1];
    let vz = cell.velocity[2];
    let speed = (vx * vx + vy * vy + vz * vz).sqrt();
    let speed_norm = (speed / max_speed).clamp(0.0, 1.0);
    let rot_norm = ((cell.angular_velocity.abs() + cell.pitch_velocity.abs()) / turn_rate)
        .clamp(0.0, 2.0);
    VIBRATION_K_LINEAR * speed_norm + VIBRATION_K_ANGULAR * rot_norm
}
