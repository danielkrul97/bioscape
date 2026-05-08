use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::*;

/// Sprint 66: persistent spring bond mezi dvěma buňkami. Stateful — drží se
/// mezi ticky, dokud se neutrhne (overstretch) nebo cíl nezemře. Bond je
/// **directed** v Cell.bonds slotu (cell_i drží bond → other_cell_id), ale
/// druhá strana má symmetrický slot (Newton 3rd law se realizuje tím, že
/// každý cell aplikuje vlastní spring force).
///
/// Sprint 68: per-bond stiffness + damping (mean obou cells' genome při
/// formaci) — bond je fyzicky jedna pružina, takže k a c jsou symmetric
/// per-pair. Mutace brzdy/tuhosti nezmění existující bondy, ovlivní jen
/// budoucí formace; tato semantika dělá bondy "stabilními kontrakty"
/// místo per-tick re-evaluation.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Bond {
    /// Stable identifier druhé buňky (Cell.cell_id). Rezolvuje se na index
    /// per-tick přes `FxHashMap<u64, usize>`. Pokud cíl zemřel, bond je
    /// dangling — pruning pass ho zahodí.
    pub other_cell_id: u64,
    /// Klidová délka spring (světové jednotky). Set při formaci bondu na
    /// `current_distance × BOND_REST_LENGTH_SLACK`.
    pub rest_length: f32,
    /// Sprint 68: per-bond spring constant. Set při formaci jako mean obou
    /// `genome.bond_stiffness`.
    pub stiffness: f32,
    /// Sprint 68: per-bond damping. Set při formaci jako mean obou
    /// `genome.bond_damping`.
    pub damping: f32,
    /// Tiků od formace. Diagnostic/logging — neovlivňuje fyziku.
    pub age_ticks: u32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Cell {
    pub position: [f32; 3],
    pub velocity: [f32; 3],
    /// Yaw rate — rotace okolo vertikální osy. Brain output[0].
    pub angular_velocity: f32,
    /// Sprint 33 pitch rate — rotace okolo horizontální osy kolmé k forwardu
    /// v xy. Brain output[7]. Drag stejný jako u yawu.
    pub pitch_velocity: f32,
    pub energy: f32,
    // Persists even when velocity hits zero — atan2(0, 0) would otherwise
    // collapse to 0 and bias evolution toward east-facing motion.
    pub heading: f32,
    /// Sprint 33: pitch úhel, klampovaný do [-π/2, π/2]. Spolu s heading (yaw)
    /// definuje forward unit vector: (cos(y)·cos(p), sin(y)·cos(p), sin(p)).
    /// Roll vynechán — cells axially symetrické.
    pub pitch: f32,
    // Lineage tracking — inherited from parent at reproduction (no mutation).
    // birth_gen records the generation when the lineage was created (initial
    // population: 0; new lineages from speciation events would bump it).
    pub lineage_id: u64,
    pub lineage_birth_gen: u64,
    // Recent activations from the last brain forward pass — Hebbian updates
    // read these to credit-assign on reward events (myopic, 1-tick window).
    #[serde(with = "serde_arr_inputs")]
    pub last_inputs: [f32; BRAIN_INPUTS],
    #[serde(with = "serde_arr_hidden")]
    pub last_hidden: [f32; BRAIN_HIDDEN],
    pub last_outputs: [f32; BRAIN_OUTPUTS],
    /// Sprint 126: per-channel emission z minulého ticku. Updated v
    /// `emit_pheromones` po výpočtu nového emit value (dříve čteno pro burst).
    #[serde(default)]
    pub last_emit: [f32; N_PHEROMONE_CHANNELS],
    /// Sprint 126: per-channel akumulátor squared tick-to-tick deltas.
    /// `burst_accum[ch] += (current - last_emit[ch])²` per emit_pheromones.
    /// Resetuje se v end-of-gen write_stats. Vyšší hodnota = víc bursty
    /// (continuous emit má small frame-to-frame deltas → low burst score).
    #[serde(default)]
    pub burst_accum: [f32; N_PHEROMONE_CHANNELS],
    /// Sprint 94: cluster-shared brain. Pre-tick mean `last_hidden` přes
    /// bond network (self + bonded partners). Brain recurrent input slots
    /// 21..52 čtou `pooled_hidden` místo `last_hidden` — cluster cells
    /// získají přístup ke kolektivní paměti (proto-distributed cognition).
    /// Solo cells: pooled_hidden == last_hidden (no neighbors). Bonded cells:
    /// average over cluster → bigger effective context window.
    /// `serde(default)` returns zeros pro backward-compat (= same as fresh
    /// cell s žádnou prior activity, behavior matches pre-Sprint-94).
    #[serde(default = "default_pooled_hidden", with = "serde_arr_hidden")]
    pub pooled_hidden: [f32; BRAIN_HIDDEN],
    /// Sprint 30: nedobrovolný energy drain akumulovaný v aktuálním ticku
    /// (predation + hazard). Brain ho čte v dalším ticku jako input[14]
    /// (damage signal), pak resetuje na 0. Voluntární cost se NEZAPISUJE
    /// — cell sama drives ty náklady přes outputs, není to externí útok.
    pub damage_accum: f32,
    /// Sprint 42: ticks od spawnu / mating-childu. Drives aging body cost ramp.
    pub age: u64,
    /// Sprint 42: refractory period po mating, decremented per tick. Mating
    /// gating čte `== 0`.
    pub reproduce_cooldown_ticks: u32,
    /// Sprint 66: stable identifier pro bond resolution. Monotonic, přidělen
    /// World-level counterem při spawnu / mating childu. Lineage_id je sdílený
    /// per linie (ne unique per cell), takže potřebujeme samostatný cell_id.
    pub cell_id: u64,
    /// Sprint 66: persistent spring bonds. Fixní array `Option<Bond>` aby Cell
    /// zůstal `Copy` a žádný heap alloc per cell. Empty slots = `None`.
    pub bonds: [Option<Bond>; MAX_BONDS_PER_CELL],
    /// Sprint 80: bistabilní cell-state [0,1]. Není v genomu — dědí se z
    /// parenta s šumem, takže slouží jako fenotypová paměť napříč generacemi.
    /// Pozitivní feedback okolo 0.5 + bias od `n_bonds()` produkuje dva
    /// stabilní attractory: ~0 (selfish) a ~1 (altruist). Reguluje food share
    /// frakci uvnitř bonded clusteru.
    pub cell_state: f32,
    /// Sensor-derived squared distance k nejbližšímu food. Set v `brain_act`
    /// (CPU sensor stage) z food_grid scanu, čte ho `eat_food` pro early-skip
    /// kandidát gather: pokud `d² > (EAT_RADIUS × max_axis + slack)²`, žádný
    /// food není dostupný k snědení tento tick i přes maximální per-tick pohyb,
    /// takže food_grid query lze úplně přeskočit.
    ///
    /// `f32::MAX` = no food in vision (sensor returned `None`).
    /// `0.0` (set in `--gpu-full` cestě) = disable skip (sensor běžel na GPU,
    /// CPU nemá hodnotu — bezpečně eat_food vždy spustí query).
    /// Backward-compat checkpoint: `serde(default = "f32_max_default")`.
    #[serde(default = "f32_max_default")]
    pub last_best_food_d2: f32,
    pub phenotype: Phenotype,
    pub genome: Genome,
}

fn f32_max_default() -> f32 {
    f32::MAX
}

impl Cell {
    pub fn random(
        rng: &mut impl Rng,
        world_half: [f32; 3],
        lineage_id: u64,
        lineage_birth_gen: u64,
        cell_id: u64,
    ) -> Self {
        let genome = Genome::random(rng);
        Self::from_genome(rng, genome, world_half, lineage_id, lineage_birth_gen, cell_id)
    }

    /// Sprint 69: count of populated bond slots. Used in predation defense
    /// (`bond_defense_factor`) — víc bondů = větší ochrana proti útoku.
    #[inline]
    pub fn n_bonds(&self) -> u32 {
        self.bonds.iter().filter(|b| b.is_some()).count() as u32
    }

    pub fn from_genome(
        rng: &mut impl Rng,
        genome: Genome,
        world_half: [f32; 3],
        lineage_id: u64,
        lineage_birth_gen: u64,
        cell_id: u64,
    ) -> Self {
        let direction = rng.random_range(0.0..TAU);
        let phenotype = Phenotype::from_genome(&genome);
        // Sprint 32 substrate: z-osa drží determinismus pre-3D semantiky.
        // Když world_half[2] == 0 (Sprint 32 default), z=0 a žádný RNG draw —
        // zachová bit-identical CSV trajektorii. Sprint 33+ unlockne motion +
        // initial draw přes celou krabici.
        let pos_z = if world_half[2] > 0.0 {
            rng.random_range(-world_half[2]..world_half[2])
        } else {
            0.0
        };
        Self {
            position: [
                rng.random_range(-world_half[0]..world_half[0]),
                rng.random_range(-world_half[1]..world_half[1]),
                pos_z,
            ],
            velocity: [
                direction.cos() * genome.max_speed,
                direction.sin() * genome.max_speed,
                0.0,
            ],
            angular_velocity: 0.0,
            pitch_velocity: 0.0,
            energy: INITIAL_ENERGY,
            heading: direction,
            // Sprint 33: cells startují horizontálně (pitch=0). Pohyb v z se
            // vyvine přes brain output[7] = turn_pitch. Random initial pitch
            // by zkonzumoval RNG draw a porušil pre-Sprint-33 reproducibility,
            // pokud bychom chtěli A/B; horizontální start je clean baseline.
            pitch: 0.0,
            lineage_id,
            lineage_birth_gen,
            last_inputs: [0.0; BRAIN_INPUTS],
            last_hidden: [0.0; BRAIN_HIDDEN],
            last_outputs: [0.0; BRAIN_OUTPUTS],
            last_emit: [0.0; N_PHEROMONE_CHANNELS],
            burst_accum: [0.0; N_PHEROMONE_CHANNELS],
            pooled_hidden: [0.0; BRAIN_HIDDEN],
            damage_accum: 0.0,
            age: 0,
            reproduce_cooldown_ticks: 0,
            cell_id,
            bonds: [None; MAX_BONDS_PER_CELL],
            // Sprint 80: kick okolo 0.5 (nestabilní fixed point feedback),
            // aby cells nezamrzly přesně v neutrální zóně. Append na konci
            // RNG sekvence — pre-Sprint-80 draws (direction, pos_z, pos_x,
            // pos_y) zůstávají v identickém pořadí, jen za nimi je 1 nový.
            cell_state: 0.5 + rng.random_range(-CELL_STATE_INIT_KICK..CELL_STATE_INIT_KICK),
            last_best_food_d2: f32::MAX,
            phenotype,
            genome,
        }
    }

    /// Per-tick physics: kinematic update from velocity / angular_velocity,
    /// quadratic drag on both, energy drains. Brain-applied forces should be
    /// integrated into velocity / angular_velocity *before* step (in
    /// `cells_brain_act`); step is purely passive integration + dissipation.
    ///
    /// Sprint 26 anisotropic drag: rozložím velocity na heading-aligned (par)
    /// vs heading-perpendicular (perp) komponentu. Cross-section je v body
    /// frame: forward motion „cítí" width (frontální), sideways motion cítí
    /// length. Pro length=width=s drag přesně reprodukuje původní isotropní.
    pub fn step(
        &mut self,
        dt: f32,
        world_half: [f32; 3],
        tick: u64,
        generation: u64,
        physics: &PhysicsConfig,
    ) {
        self.step_with_climate(dt, world_half, tick, generation, physics, 0.0);
    }

    /// Sprint 112: step se shock-aware climate offsetem. Caller (headless / main)
    /// předem spočítá `climate_offset` přes `climate_shock_offset(...)` na cell
    /// pozici a předá sem; vnitřní `apply_energy_costs` ho přičte k baseline
    /// temperatuře. `climate_offset = 0.0` → byte-identical chování s `step`.
    pub fn step_with_climate(
        &mut self,
        dt: f32,
        world_half: [f32; 3],
        tick: u64,
        generation: u64,
        physics: &PhysicsConfig,
        climate_offset: f32,
    ) {
        // Sprint 42: aging + cooldown decrement na začátku ticku, aby
        // apply_energy_costs viděl current age v ramp formuli.
        self.age = self.age.saturating_add(1);
        if self.reproduce_cooldown_ticks > 0 {
            self.reproduce_cooldown_ticks -= 1;
        }
        self.integrate_kinematics(dt, world_half);
        self.apply_anisotropic_drag(dt, physics);
        self.apply_angular_drag(dt, physics);
        // Sprint 86: tick + generation propagace pro time-varying thermal.
        // Sprint 112: + climate_offset (default 0.0) z aktivních ClimateShift
        // shocků, předem spočítaný callerem.
        self.apply_energy_costs(dt, world_half, tick, generation, physics, climate_offset);
        self.apply_world_bounce(world_half);
        self.update_cell_state(dt);
    }

    /// Sprint 80: bistabilní cell-state dynamika. Pure deterministic (no RNG)
    /// — RNG by zde rozbil seed reproducibility step()u. Šum vstupuje jen
    /// při dědičnosti v `make_mating_child`.
    ///
    /// Update rule:
    ///   s' = s + K · (s − 0.5) · dt + bias · n_bonds · dt
    ///
    /// Pozitivní feedback `(s − 0.5)` táhne stav od nestabilního fixed
    /// pointu k 0 nebo 1 podle aktuální orientace. `n_bonds` bias konzistentně
    /// tlačí cells s víc bondy směrem k altruist attractoru — tissue cells
    /// commitnou k altruismu, solo cells driftují k selfish.
    fn update_cell_state(&mut self, dt: f32) {
        let n_bonds = self.n_bonds() as f32;
        let feedback = CELL_STATE_FEEDBACK_K * (self.cell_state - 0.5) * dt;
        let env_drive = CELL_STATE_BOND_BIAS * n_bonds * dt;
        self.cell_state = (self.cell_state + feedback + env_drive).clamp(0.0, 1.0);
    }

    /// Sprint 40: čistý integrate (position += v · dt, heading + pitch),
    /// gravity, pitch clamp. Žádné drag ani energie — to dělají další fáze.
    fn integrate_kinematics(&mut self, dt: f32, world_half: [f32; 3]) {
        self.position[0] += self.velocity[0] * dt;
        self.position[1] += self.velocity[1] * dt;
        self.position[2] += self.velocity[2] * dt;
        self.heading += self.angular_velocity * dt;
        // Sprint 38: gravity působí pouze pokud je z-volume aktivní. Aplikuje
        // se před drag aby drag mohl balancovat → terminal velocity.
        if world_half[2] > 0.0 {
            self.velocity[2] -= GRAVITY * dt;
        }
        // Sprint 35: pitch range ±π/12. Random brain pitch noise s tight
        // rangem = nepatrný drift v z, ale dovolí vědomé naklonění do z motion.
        self.pitch = (self.pitch + self.pitch_velocity * dt).clamp(
            -core::f32::consts::FRAC_PI_6 * 0.5,
            core::f32::consts::FRAC_PI_6 * 0.5,
        );
    }

    /// Sprint 40: 3D anisotropic drag. Forward = unit vector (yaw + pitch).
    /// Velocity → along-forward + perpendicular-forward; drag_par váhuje
    /// width × length cross-section, drag_perp váhuje length × width.
    /// Pro length=width=s perfectly izotropní (kompat s pre-Sprint-26).
    fn apply_anisotropic_drag(&mut self, dt: f32, physics: &PhysicsConfig) {
        let [fx, fy, fz] = forward_vector(self.heading, self.pitch);
        let v_par = self.velocity[0] * fx + self.velocity[1] * fy + self.velocity[2] * fz;
        let v_perp_x = self.velocity[0] - v_par * fx;
        let v_perp_y = self.velocity[1] - v_par * fy;
        let v_perp_z = self.velocity[2] - v_par * fz;
        let v_perp_mag =
            (v_perp_x * v_perp_x + v_perp_y * v_perp_y + v_perp_z * v_perp_z).sqrt();
        let length = self.phenotype.body_length;
        let width = self.phenotype.body_width;
        let drag_par_factor = physics.drag * v_par.abs() * width * dt;
        let drag_perp_factor = physics.drag * v_perp_mag * length * dt;
        let new_v_par = v_par - drag_par_factor * v_par;
        let new_v_perp_x = v_perp_x - drag_perp_factor * v_perp_x;
        let new_v_perp_y = v_perp_y - drag_perp_factor * v_perp_y;
        let new_v_perp_z = v_perp_z - drag_perp_factor * v_perp_z;
        self.velocity[0] = new_v_par * fx + new_v_perp_x;
        self.velocity[1] = new_v_par * fy + new_v_perp_y;
        self.velocity[2] = new_v_par * fz + new_v_perp_z;
    }

    /// Sprint 40: rotační drag pro yaw + pitch. Sdílený multiplier zachovává
    /// pre-refactor chování (oba decay stejnou rychlostí).
    fn apply_angular_drag(&mut self, dt: f32, physics: &PhysicsConfig) {
        let drag_factor = (1.0 - physics.angular_drag * dt).max(0.0);
        self.angular_velocity *= drag_factor;
        self.pitch_velocity *= drag_factor;
    }

    /// Sprint 40: per-tick energy costs. v², rotational (yaw only — Sprint 33
    /// note: pitch je „free" rotace, jinak by random brainy ztrácely 2× rotační
    /// drain), vision, body volume maintenance, spike, attack-mode upkeep.
    ///
    /// Sprint 85: všechny drains škálují `metabolism_factor(T)` kde T je
    /// teplota na cell pozici. Warm cells (z = +half) drain ~2.46× rychleji
    /// než REF, cold cells (z = -half) drain ~0.41×. Niche separation by
    /// depth — selekce favorizuje cells co najdou energy-optimal z-vrstvu.
    /// Při `world_half[2] = 0` (pre-3D baseline) vrací temperature_at_z
    /// REF_TEMP → metabolism = 1.0 → drain backward-compat.
    fn apply_energy_costs(
        &mut self,
        dt: f32,
        world_half: [f32; 3],
        tick: u64,
        generation: u64,
        physics: &PhysicsConfig,
        climate_offset: f32,
    ) {
        let temp =
            temperature_at_z(self.position[2], world_half, tick, generation) + climate_offset;
        let metabolism = metabolism_factor(temp);
        let dt_eff = dt * metabolism;
        // Sprint 33: v_mag_sq zahrnuje 3D (vz != 0 v Sprint 35+).
        let v_mag_sq =
            self.velocity[0].powi(2) + self.velocity[1].powi(2) + self.velocity[2].powi(2);
        self.energy -= v_mag_sq * physics.energy_cost_per_v_sq * dt_eff;
        let av = self.angular_velocity;
        let eff_r = self.phenotype.effective_radius();
        self.energy -= eff_r * eff_r * av * av * physics.angular_energy_cost * dt_eff;
        // Sprint 82: cost ∝ radius × fov_factor. Full sphere (fov = π) → factor
        // 1.0 → identický s pre-Sprint-82. Užší kužel platí míň, ale Sprint 83
        // aktivuje cone filter v sensor gather → trade-off info-loss vs energy.
        let fov_factor = vision_fov_factor(self.genome.vision_fov);
        self.energy -=
            self.genome.vision_radius * physics.vision_cost_per_radius * fov_factor * dt_eff;
        // Sprint 34: maintenance ∝ 3D volume = length×width×height.
        // Sprint 42: aging ramp — starší cells platí postupně víc per volume unit.
        let age_sec = self.age as f32 / FIXED_TIMESTEP_HZ;
        let aging_factor = 1.0 + AGE_DECAY_PER_SEC * age_sec;
        self.energy -=
            self.phenotype.volume() * physics.body_cost_factor * aging_factor * dt_eff;
        // Sprint 121: sum maintenance přes aktivní spiky. Sprint 123: per-spike
        // quadratic complexity multiplier. S spike_count=1, complexity=0
        // redukuje na pre-S121 single spike cost.
        let mut spike_cost = 0.0;
        for spike in self.phenotype.active_spikes() {
            spike_cost += spike.length * spike_complexity_cost_factor(spike.complexity);
        }
        self.energy -= spike_cost * SPIKE_COST_PER_SEC * dt_eff;
        // Sprint 41: shell maintenance — defensive armor stojí víc než spike,
        // protože pokrývá celý povrch.
        self.energy -= self.phenotype.shell_thickness * SHELL_COST_PER_SEC * dt_eff;
        // Sprint 27 attack maintenance: cost ∝ max(0, output[6]).
        let attack_strength = self.last_outputs[6].max(0.0);
        self.energy -= attack_strength * ATTACK_COST_PER_SEC * dt_eff;
        // Sprint 87: thermal stress penalty. Quadratic na deviation od optima,
        // independent metabolism (tepelný stres = extra cost, ne reduced rate).
        // `dt`, ne `dt_eff` — penalty není Q10-modulated.
        let dev = (temp - self.genome.thermal_optimum) / 13.0;
        self.energy -= dev * dev * physics.thermal_optimum_penalty * dt;
        // Sprint 97: sensor gain drain. Sum gains × cost × dt_eff (Q10-modulated
        // jako ostatní biological costs). Cluster cells which off-load sensor
        // duties to specialists (gain → 0 v category) save energy proportionally.
        let total_sensor_gain: f32 = self.genome.sensor_gains.iter().sum();
        self.energy -= total_sensor_gain * SENSOR_GAIN_COST * dt_eff;
    }

    /// Sprint 54: xy modulo wrap (toroidal cylinder topology), z bounce.
    /// Pre-Sprint-54 byly xy walls reflective → cells akumulovaly u krajů
    /// (Sprint 30+ edge_frac/corner_frac metriky), evoluce mohla najít wall
    /// exploit (těsné otáčení před stěnou), smell/pheromone gradients
    /// degenerovaly u Neumann boundary. Toroidal odstraní edge bias —
    /// cell na x=−950 sousedí s cell na x=+950. Z osa zůstává bounded
    /// (gravita + food sink + carrion drop vyžadují pevný strop/dno).
    fn apply_world_bounce(&mut self, world_half: [f32; 3]) {
        let wx = 2.0 * world_half[0];
        let wy = 2.0 * world_half[1];
        // xy wrap (toroidal): pozice → [-half, half).
        if self.position[0] >= world_half[0] || self.position[0] < -world_half[0] {
            // rem_euclid emulace pro f32: pos - floor((pos + half) / w) * w.
            let p = self.position[0] + world_half[0];
            self.position[0] = p - (p / wx).floor() * wx - world_half[0];
        }
        if self.position[1] >= world_half[1] || self.position[1] < -world_half[1] {
            let p = self.position[1] + world_half[1];
            self.position[1] = p - (p / wy).floor() * wy - world_half[1];
        }
        // z stále bounce.
        if world_half[2] > 0.0 && self.position[2].abs() > world_half[2] {
            self.velocity[2] = -self.velocity[2];
            self.position[2] = self.position[2].clamp(-world_half[2], world_half[2]);
        }
        // Heading recompute odstraněn — wrap nezmění směr pohybu.
    }

    /// Aplikuje runtime morph z brain outputs na phenotype + naúčtuje
    /// energii za realizované Δ. Volá se mezi `brain_act` a `step`.
    /// Sprint 34: morph teď řídí 4 dimenze (length, width, height, spike).
    /// Mapování indexů: [3]=length, [4]=width, [8]=height (Sprint 34, append),
    /// [5]=spike. Index [8] je za attack[6] a turn_pitch[7], které předtím
    /// existovaly — nový height index zachoval stávající indexy beze změny.
    pub fn apply_morph(&mut self, dt: f32) {
        let morph = [
            self.last_outputs[3],
            self.last_outputs[4],
            self.last_outputs[8],
            self.last_outputs[5],
        ];
        let total_delta = self.phenotype.apply_morph(morph, MORPH_RATE, dt);
        self.energy -= MORPH_COST_PER_DELTA * total_delta;
    }

    /// Sprint 40: aplikuje brain motor outputs na velocity / angular_velocity /
    /// pitch_velocity. Použito z hot loop v rendereru i headless po `populate_brain_inputs`
    /// + `Brain::forward_with_state`. Sjednocuje motor mapping logiku, která
    /// byla pre-refactor duplikovaná v `cells_brain_act` (main) a `brain_act`
    /// (headless). Bere `outputs[0] = turn`, `[1] = thrust`, `[7] = pitch`.
    pub fn apply_brain_motor(&mut self, outputs: &[f32; BRAIN_OUTPUTS], dt: f32) {
        // Sprint 42: F=ma — denominator je `mass = effective_radius` (smoke-tuned
        // fallback z `volume()`). Plný volume() byl příliš agresivní inerce penalty
        // pro untrained brainy v Sprint 42 smoke (extinct gen 40). `effective_radius`
        // (= aritmetický průměr 3 os) zachovává inerce-by-size škálování bez
        // kvadratického cost shocku. Cells s objemnějším tělem stále inertia, ale
        // menší magnitude.
        let mass = self.phenotype.effective_radius().max(0.01);
        let turn_rate = self.genome.turn_rate;
        let max_speed = self.genome.max_speed;
        let turn_signal = outputs[0];
        let thrust_norm = (outputs[1] + 1.0) * 0.5;
        let pitch_signal = outputs[7];
        let ang_acc = turn_signal * turn_rate / mass;
        self.angular_velocity += ang_acc * dt;
        let pitch_acc = pitch_signal * turn_rate / mass;
        self.pitch_velocity += pitch_acc * dt;
        let a_max = DRAG_COEFFICIENT * max_speed * max_speed / mass;
        let a = thrust_norm * a_max;
        let fwd = forward_vector(self.heading, self.pitch);
        self.velocity[0] += a * fwd[0] * dt;
        self.velocity[1] += a * fwd[1] * dt;
        self.velocity[2] += a * fwd[2] * dt;
    }

    /// Bonus predation gain pokud má attacker aspoň jeden spike, který směřuje
    /// na cíl (cosine > `SPIKE_DOT_THRESHOLD`). Sprint 121: iteruje všechny
    /// aktivní spiky — každý kontribuuje vlastní cone test. S spike_count=1
    /// a azimuth/elev=0 redukuje na pre-S121 single-spike behavior.
    pub fn spike_bonus_against(&self, target_pos: [f32; 3]) -> f32 {
        let dx = target_pos[0] - self.position[0];
        let dy = target_pos[1] - self.position[1];
        let dz = target_pos[2] - self.position[2];
        let dist_sq = dx * dx + dy * dy + dz * dz;
        if dist_sq < f32::EPSILON {
            return 0.0;
        }
        let dist = dist_sq.sqrt();
        let to_target = [dx / dist, dy / dist, dz / dist];
        let mut total = 0.0;
        for spike in self.phenotype.active_spikes() {
            if spike.length <= 0.0 {
                continue;
            }
            let dir = spike_direction(self.heading, self.pitch, spike);
            let cos_angle = dir[0] * to_target[0] + dir[1] * to_target[1] + dir[2] * to_target[2];
            if cos_angle < SPIKE_DOT_THRESHOLD {
                continue;
            }
            total += PREDATION_GAIN_PER_TICK
                * spike.length
                * spike_complexity_attack_factor(spike.complexity)
                * SPIKE_PREDATION_BONUS;
        }
        total
    }

    /// Sprint 41: pure ellipsoidní acceptance test bez mutace. Semi-axes ∝
    /// phenotype × `eat_factor`. Pro L=W=H=s shell sféra radius `s × eat_factor`
    /// — backward kompat s pre-Sprint-41 isotropic buňkou. Použito v binárkách
    /// pro broad+narrow phase split (broad uses `max_axis`, narrow tento test).
    pub fn eat_test(&self, food: &Food, eat_factor: f32) -> bool {
        eat_test_pose(
            self.position,
            self.heading,
            self.pitch,
            [
                self.phenotype.body_length,
                self.phenotype.body_width,
                self.phenotype.body_height,
            ],
            food.position,
            eat_factor,
        )
    }

    /// Sprint 122: rozšířený eat test — kombinuje ellipsoid (`eat_test`)
    /// a per-spike forward grab cones u tipu spike. Spike kuželek má vrchol
    /// `cell_pos + dir × (eff_radius + length)` (tip), osu = spike direction,
    /// half-angle `SPIKE_GRAB_HALF_ANGLE × (1 + complexity × COMPLEXITY_GRAB_GAIN)`,
    /// range `length × SPIKE_GRAB_REACH_BONUS` od tipu. Cell sní food pokud
    /// projde ELLIPSOID NEBO kterýkoli aktivní spike kuželek. Pre-S122
    /// (`spike_count = 0` nebo všechny `length = 0`) redukuje na `eat_test`.
    pub fn eat_test_with_spikes(&self, food: &Food, eat_factor: f32) -> bool {
        if self.eat_test(food, eat_factor) {
            return true;
        }
        let eff_r = self.phenotype.effective_radius();
        for spike in self.phenotype.active_spikes() {
            if spike.length <= 0.0 {
                continue;
            }
            let dir = spike_direction(self.heading, self.pitch, spike);
            let tip_dist = eff_r + spike.length;
            let tip = [
                self.position[0] + dir[0] * tip_dist,
                self.position[1] + dir[1] * tip_dist,
                self.position[2] + dir[2] * tip_dist,
            ];
            let dx = food.position[0] - tip[0];
            let dy = food.position[1] - tip[1];
            let dz = food.position[2] - tip[2];
            let dist_sq = dx * dx + dy * dy + dz * dz;
            let max_range = spike.length * SPIKE_GRAB_REACH_BONUS;
            if dist_sq > max_range * max_range {
                continue;
            }
            if dist_sq < f32::EPSILON {
                return true;
            }
            let dist = dist_sq.sqrt();
            let cos_food = (dx * dir[0] + dy * dir[1] + dz * dir[2]) / dist;
            let half_angle =
                SPIKE_GRAB_HALF_ANGLE * spike_complexity_grab_factor(spike.complexity);
            if cos_food >= half_angle.cos() {
                return true;
            }
        }
        false
    }

    /// Sprint 41: ellipsoidální acceptance + energy gain při hitu.
    /// Sprint 122: zahrnuje per-spike grab cones (viz `eat_test_with_spikes`).
    pub fn try_eat(&mut self, food: &Food, eat_factor: f32, food_value: f32) -> bool {
        if self.eat_test_with_spikes(food, eat_factor) {
            self.energy += food_value;
            true
        } else {
            false
        }
    }

    /// Sprint 41: tlumí `damage_accum` shellem před tím, než brain čte damage
    /// signal. Voláno z hot loop binárek **před** `populate_brain_inputs`
    /// (která damage čte + resetuje). `shell × ABSORB × dt` units, floor 0.
    pub fn apply_shell_absorb(&mut self, dt: f32) {
        let absorb = self.phenotype.shell_thickness * SHELL_ABSORB_PER_TICK * dt;
        self.damage_accum = (self.damage_accum - absorb).max(0.0);
    }

    /// Sprint 42: Brownův pohyb — gaussian noise na velocity. `√dt` scaling je
    /// correct stochastic integration (Wiener process), ne lineární dt. z-osa
    /// se rušený jen když je z-volume aktivní (`world_half_z > 0`).
    pub fn apply_brownian(&mut self, rng: &mut impl Rng, dt: f32, world_half_z: f32) {
        let scale = THERMAL_NOISE * dt.sqrt();
        self.velocity[0] += gaussian(rng) * scale;
        self.velocity[1] += gaussian(rng) * scale;
        if world_half_z > 0.0 {
            self.velocity[2] += gaussian(rng) * scale;
        }
    }
}
