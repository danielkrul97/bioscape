use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::*;
use super::cppn::Cppn;

impl Brain {
    /// Sprint 106: derive brain weights z CPPN substrate query. Pro každý
    /// (input slot, hidden slot) pár zavolá cppn.forward([from.coord, to.coord, 1.0]),
    /// extrahuje weight + link_exists. Stejně pro (hidden, output).
    /// Biases (b1, b2) odvozeny z CPPN query s "self-loop" coord (oba inputs
    /// stejné). hidden_n = BRAIN_HIDDEN_DEFAULT.
    pub fn from_cppn(cppn: &Cppn) -> Brain {
        let mut w1 = [[0.0_f32; BRAIN_INPUTS]; BRAIN_HIDDEN];
        let mut b1 = [0.0_f32; BRAIN_HIDDEN];
        let mut w2 = [[0.0_f32; BRAIN_HIDDEN]; BRAIN_OUTPUTS];
        let mut b2 = [0.0_f32; BRAIN_OUTPUTS];

        // w1: input → hidden
        for h in 0..BRAIN_HIDDEN {
            let to_c = substrate_hidden_coords(h);
            for i in 0..BRAIN_INPUTS {
                let from_c = substrate_input_coords(i);
                let inputs = [
                    from_c[0], from_c[1], from_c[2],
                    to_c[0], to_c[1], to_c[2],
                    1.0,
                ];
                let out = cppn.forward(inputs);
                if out[1] >= CPPN_LINK_EXISTS_THRESHOLD {
                    w1[h][i] = out[0];
                }
            }
            // b1[h]: query CPPN with from = hidden (self-loop sentinel)
            let inputs = [to_c[0], to_c[1], to_c[2], to_c[0], to_c[1], to_c[2], 0.0];
            let out = cppn.forward(inputs);
            b1[h] = out[0] * 0.5; // dampen bias
        }

        // w2: hidden → output
        for o in 0..BRAIN_OUTPUTS {
            let to_c = substrate_output_coords(o);
            for h in 0..BRAIN_HIDDEN {
                let from_c = substrate_hidden_coords(h);
                let inputs = [
                    from_c[0], from_c[1], from_c[2],
                    to_c[0], to_c[1], to_c[2],
                    1.0,
                ];
                let out = cppn.forward(inputs);
                if out[1] >= CPPN_LINK_EXISTS_THRESHOLD {
                    w2[o][h] = out[0];
                }
            }
            // b2[o]: self-loop sentinel
            let inputs = [to_c[0], to_c[1], to_c[2], to_c[0], to_c[1], to_c[2], 0.0];
            let out = cppn.forward(inputs);
            b2[o] = out[0] * 0.5;
        }

        Brain {
            hidden_n: BRAIN_HIDDEN_DEFAULT as u32,
            w1,
            b1,
            w2,
            b2,
        }
    }
}

// ─── End Sprint 106 ──────────────────────────────────────────────────────────

// ─── End Sprint 105 CPPN scaffolding ─────────────────────────────────────────

pub const PHYSICS_CONFIG: PhysicsConfig = PhysicsConfig {
    drag: DRAG_COEFFICIENT,
    angular_drag: ANGULAR_DRAG,
    energy_cost_per_v_sq: ENERGY_COST_PER_V_SQ,
    angular_energy_cost: ANGULAR_ENERGY_COST,
    vision_cost_per_radius: VISION_COST_PER_RADIUS,
    body_cost_factor: BODY_COST_FACTOR,
    thermal_optimum_penalty: THERMAL_OPTIMUM_PENALTY,
};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Brain {
    /// Sprint 80: active hidden neuron count (≤ BRAIN_HIDDEN storage). Default
    /// = BRAIN_HIDDEN_DEFAULT. forward/mutate/crossover/hebbian iterují jen
    /// [0..hidden_n]; dead zone (hidden_n..BRAIN_HIDDEN) drží 0 a do výpočtu
    /// nepřispívá. Strukturální mutace (add/remove neuron) přijdou v dalších
    /// sprintech a budou tuhle hodnotu měnit.
    #[serde(default = "default_hidden_n")]
    pub hidden_n: u32,
    #[serde(with = "serde_arrays_w1")]
    pub w1: [[f32; BRAIN_INPUTS]; BRAIN_HIDDEN],
    #[serde(with = "serde_arr_hidden")]
    pub b1: [f32; BRAIN_HIDDEN],
    #[serde(with = "serde_arrays_w2")]
    pub w2: [[f32; BRAIN_HIDDEN]; BRAIN_OUTPUTS],
    pub b2: [f32; BRAIN_OUTPUTS],
}

fn default_hidden_n() -> u32 {
    BRAIN_HIDDEN_DEFAULT as u32
}

// Sprint 48: serde 1 has native const-generic support pro `[T; N]` ale
// nested fixed arrays (`[[f32; 36]; 16]`) potřebují manual workaround.
// Encode jako flat Vec<f32> length × 36, decode reverse.
pub mod serde_arrays_w1 {
    use super::{BRAIN_HIDDEN, BRAIN_INPUTS};
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    pub fn serialize<S: Serializer>(
        w: &[[f32; BRAIN_INPUTS]; BRAIN_HIDDEN],
        s: S,
    ) -> Result<S::Ok, S::Error> {
        let flat: Vec<f32> = w.iter().flat_map(|row| row.iter().copied()).collect();
        flat.serialize(s)
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(
        d: D,
    ) -> Result<[[f32; BRAIN_INPUTS]; BRAIN_HIDDEN], D::Error> {
        let flat: Vec<f32> = Vec::deserialize(d)?;
        if flat.len() != BRAIN_HIDDEN * BRAIN_INPUTS {
            return Err(serde::de::Error::custom("w1 length mismatch"));
        }
        let mut out = [[0.0_f32; BRAIN_INPUTS]; BRAIN_HIDDEN];
        for (i, row) in out.iter_mut().enumerate() {
            row.copy_from_slice(&flat[i * BRAIN_INPUTS..(i + 1) * BRAIN_INPUTS]);
        }
        Ok(out)
    }
}

pub mod serde_arrays_w2 {
    use super::{BRAIN_HIDDEN, BRAIN_OUTPUTS};
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    pub fn serialize<S: Serializer>(
        w: &[[f32; BRAIN_HIDDEN]; BRAIN_OUTPUTS],
        s: S,
    ) -> Result<S::Ok, S::Error> {
        let flat: Vec<f32> = w.iter().flat_map(|row| row.iter().copied()).collect();
        flat.serialize(s)
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(
        d: D,
    ) -> Result<[[f32; BRAIN_HIDDEN]; BRAIN_OUTPUTS], D::Error> {
        let flat: Vec<f32> = Vec::deserialize(d)?;
        if flat.len() != BRAIN_OUTPUTS * BRAIN_HIDDEN {
            return Err(serde::de::Error::custom("w2 length mismatch"));
        }
        let mut out = [[0.0_f32; BRAIN_HIDDEN]; BRAIN_OUTPUTS];
        for (i, row) in out.iter_mut().enumerate() {
            row.copy_from_slice(&flat[i * BRAIN_HIDDEN..(i + 1) * BRAIN_HIDDEN]);
        }
        Ok(out)
    }
}

// Sprint 48: serde 1 native podpora `[T; N]` jen pro N ≤ 32. `BRAIN_INPUTS` = 36.
pub mod serde_arr_inputs {
    use super::BRAIN_INPUTS;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    pub fn serialize<S: Serializer>(a: &[f32; BRAIN_INPUTS], s: S) -> Result<S::Ok, S::Error> {
        a.as_slice().serialize(s)
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(
        d: D,
    ) -> Result<[f32; BRAIN_INPUTS], D::Error> {
        let v: Vec<f32> = Vec::deserialize(d)?;
        if v.len() != BRAIN_INPUTS {
            return Err(serde::de::Error::custom("inputs length mismatch"));
        }
        let mut a = [0.0_f32; BRAIN_INPUTS];
        a.copy_from_slice(&v);
        Ok(a)
    }
}

// Sprint 103: BRAIN_HIDDEN > 32 → serde's native `[T; N]` impl nepokrývá.
// Wrapper sloučí pole do Vec<f32> na serializaci, resp. roundtripuje zpět.
pub mod serde_arr_hidden {
    use super::BRAIN_HIDDEN;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    pub fn serialize<S: Serializer>(a: &[f32; BRAIN_HIDDEN], s: S) -> Result<S::Ok, S::Error> {
        a.as_slice().serialize(s)
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(
        d: D,
    ) -> Result<[f32; BRAIN_HIDDEN], D::Error> {
        let v: Vec<f32> = Vec::deserialize(d)?;
        if v.len() != BRAIN_HIDDEN {
            return Err(serde::de::Error::custom("hidden length mismatch"));
        }
        let mut a = [0.0_f32; BRAIN_HIDDEN];
        a.copy_from_slice(&v);
        Ok(a)
    }
}

impl Brain {
    pub fn random(rng: &mut impl Rng) -> Self {
        Self::random_with_hidden(rng, BRAIN_HIDDEN_DEFAULT as u32)
    }

    /// Sprint 80: variable-size random init. `hidden_n` aktivních neuronů,
    /// zbytek storage (BRAIN_HIDDEN..) zero-initialized a do forward passu
    /// nepřispívá. RNG draws happen only for active region — same seed s
    /// hidden_n=BRAIN_HIDDEN_DEFAULT reprodukuje pre-Sprint-80 sekvenci.
    pub fn random_with_hidden(rng: &mut impl Rng, hidden_n: u32) -> Self {
        debug_assert!(
            (hidden_n as usize) >= BRAIN_HIDDEN_MIN
                && (hidden_n as usize) <= BRAIN_HIDDEN,
            "hidden_n {} out of [{}, {}]",
            hidden_n,
            BRAIN_HIDDEN_MIN,
            BRAIN_HIDDEN
        );
        let h_n = hidden_n as usize;
        // Sprint 80 (storage bump): active input width je sensory + hidden_n,
        // ne celá BRAIN_INPUTS storage. Pro default h_n=16: 20+16 = 36, match
        // Sprint A pre-bump RNG sekvence byte-identical.
        let active_inputs = BRAIN_INPUTS_SENSORY + h_n;
        let mut w1 = [[0.0; BRAIN_INPUTS]; BRAIN_HIDDEN];
        let mut b1 = [0.0; BRAIN_HIDDEN];
        let mut w2 = [[0.0; BRAIN_HIDDEN]; BRAIN_OUTPUTS];
        let mut b2 = [0.0; BRAIN_OUTPUTS];
        for i in 0..h_n {
            for j in 0..active_inputs {
                w1[i][j] = gaussian(rng);
            }
            b1[i] = gaussian(rng);
        }
        for (row, bias) in w2.iter_mut().zip(b2.iter_mut()) {
            for j in 0..h_n {
                row[j] = gaussian(rng);
            }
            *bias = gaussian(rng);
        }
        // Innate thrust bias: bumps b2[1] (thrust output) above zero. Posune
        // distribuci `thrust_norm = (tanh(b2 + ...) + 1) / 2` od mean ~0.5
        // (random walk stuck) k mean ~0.7 (consistent forward motion). Hebbian
        // + selekce dál ladí; tohle jen řeší kallové cells co se nehýbou.
        b2[1] += INNATE_THRUST_BIAS;
        // Innate pheromone bias: Sprint 25 vyžaduje active emisi pro mating.
        // Bez biasu by se polovina random cells nemohla reprodukovat.
        b2[2] += INNATE_PHEROMONE_BIAS;
        // Innate attack bias: Sprint 27 — default 0 (opt-in). Mění se přes
        // konstantu, ne ad-hoc tady, ať se chování dá testovat A/B.
        b2[6] += INNATE_ATTACK_BIAS;
        // Sprint 66 bond signal bias — opt-in jako attack.
        b2[9] += INNATE_BOND_BIAS;
        // Sprint 126: ch1, ch2 emit biases. Slabší než ch0 (mating gating
        // tam potřebuje high baseline) — jen aby evoluce neměla cold-start
        // s output ≈ 0.
        b2[10] += INNATE_PHEROMONE_AUX_BIAS;
        b2[11] += INNATE_PHEROMONE_AUX_BIAS;
        Self { hidden_n, w1, b1, w2, b2 }
    }

    pub fn forward(&self, inputs: &[f32; BRAIN_INPUTS]) -> [f32; BRAIN_OUTPUTS] {
        self.forward_with_state(inputs).1
    }

    /// Same forward pass as `forward`, but also returns hidden activations
    /// (needed for Hebbian updates).
    ///
    /// Sprint 112: `wide::f32x8` přes 10 lanes (padded BRAIN_INPUTS 71→80) v
    /// L1, 7 lanes (padded BRAIN_HIDDEN 50→56) v L2. Dead zone (h_n..BRAIN_HIDDEN)
    /// padding zero → mul_add se zapíše jako akumulace přes celou pad šíři bez
    /// větvení. `target-cpu=native` (Sprint 111) zapne FMA, takže jeden chunk
    /// dot product je 1× vfmadd231ps.
    pub fn forward_with_state(
        &self,
        inputs: &[f32; BRAIN_INPUTS],
    ) -> ([f32; BRAIN_HIDDEN], [f32; BRAIN_OUTPUTS]) {
        use wide::f32x8;
        const L1_PAD: usize = ((BRAIN_INPUTS + 7) / 8) * 8;
        const L1_LANES: usize = L1_PAD / 8;
        const L2_PAD: usize = ((BRAIN_HIDDEN + 7) / 8) * 8;
        const L2_LANES: usize = L2_PAD / 8;
        let h_n = self.hidden_n as usize;

        let mut padded_inputs = [0.0_f32; L1_PAD];
        padded_inputs[..BRAIN_INPUTS].copy_from_slice(inputs);
        let mut input_lanes = [f32x8::ZERO; L1_LANES];
        for (lane, chunk) in input_lanes.iter_mut().zip(padded_inputs.chunks_exact(8)) {
            *lane = f32x8::new(chunk.try_into().unwrap());
        }

        let mut hidden = [0.0_f32; BRAIN_HIDDEN];
        let mut padded_w1_row = [0.0_f32; L1_PAD];
        for i in 0..h_n {
            padded_w1_row[..BRAIN_INPUTS].copy_from_slice(&self.w1[i]);
            let mut acc = f32x8::ZERO;
            for (lane_idx, chunk) in padded_w1_row.chunks_exact(8).enumerate() {
                let w = f32x8::new(chunk.try_into().unwrap());
                acc = w.mul_add(input_lanes[lane_idx], acc);
            }
            let sum = self.b1[i] + acc.reduce_add();
            hidden[i] = sum.tanh();
        }

        let mut padded_hidden = [0.0_f32; L2_PAD];
        padded_hidden[..BRAIN_HIDDEN].copy_from_slice(&hidden);
        let mut hidden_lanes = [f32x8::ZERO; L2_LANES];
        for (lane, chunk) in hidden_lanes.iter_mut().zip(padded_hidden.chunks_exact(8)) {
            *lane = f32x8::new(chunk.try_into().unwrap());
        }

        let mut out = [0.0_f32; BRAIN_OUTPUTS];
        let mut padded_w2_row = [0.0_f32; L2_PAD];
        for ((o, row), &bias) in out.iter_mut().zip(self.w2.iter()).zip(self.b2.iter()) {
            padded_w2_row[..BRAIN_HIDDEN].copy_from_slice(row);
            let mut acc = f32x8::ZERO;
            for (lane_idx, chunk) in padded_w2_row.chunks_exact(8).enumerate() {
                let w = f32x8::new(chunk.try_into().unwrap());
                acc = w.mul_add(hidden_lanes[lane_idx], acc);
            }
            let sum = bias + acc.reduce_add();
            *o = sum.tanh();
        }
        (hidden, out)
    }

    /// Reward-modulated Hebbian update. `Δw = lr · reward · pre · post`.
    /// Pre-/post-synaptic activations come from a stored prior forward
    /// pass — this is "myopic" credit assignment (1-tick window). Reward
    /// fires on biologically meaningful events (eating, predation kills).
    pub fn hebbian_update(
        &mut self,
        last_inputs: &[f32; BRAIN_INPUTS],
        last_hidden: &[f32; BRAIN_HIDDEN],
        last_outputs: &[f32; BRAIN_OUTPUTS],
        reward: f32,
        learning_rate: f32,
    ) {
        let lr = learning_rate * reward;
        let h_n = self.hidden_n as usize;
        for i in 0..h_n {
            let h = last_hidden[i];
            for (w, &x) in self.w1[i].iter_mut().zip(last_inputs.iter()) {
                *w += lr * h * x;
            }
            self.b1[i] += lr * h;
        }
        for (out_o, &o) in self.w2.iter_mut().zip(last_outputs.iter()) {
            for j in 0..h_n {
                out_o[j] += lr * o * last_hidden[j];
            }
        }
        for (b, &o) in self.b2.iter_mut().zip(last_outputs.iter()) {
            *b += lr * o;
        }
    }

    pub fn mutate(&self, rng: &mut impl Rng, sigma: f32) -> Self {
        let mut out = *self;
        let h_n = self.hidden_n as usize;
        let active_inputs = BRAIN_INPUTS_SENSORY + h_n;
        for i in 0..h_n {
            for j in 0..active_inputs {
                out.w1[i][j] += gaussian(rng) * sigma;
            }
            out.b1[i] += gaussian(rng) * sigma;
        }
        for (row, bias) in out.w2.iter_mut().zip(out.b2.iter_mut()) {
            for j in 0..h_n {
                row[j] += gaussian(rng) * sigma;
            }
            *bias += gaussian(rng) * sigma;
        }
        out
    }

    /// Sprint 80 Sprint C: structural mutation — přidá jeden hidden neuron.
    /// Vrací `true` pokud se neuron přidal, `false` pokud cap (`hidden_n ==
    /// BRAIN_HIDDEN`) nebo `BRAIN_HIDDEN_MIN` violation prevented.
    ///
    /// **Init logic (NEAT-style minimal disruption):**
    /// - `w1[new_idx][0..active_inputs]` = gaussian × sigma (small, drift)
    /// - `b1[new_idx]` = gaussian × sigma
    /// - `w2[*][new_idx]` = gaussian × sigma (output contribution starts small)
    /// - Existing neurons NETKNUTÉ (jejich w1[i][20+new_idx] zůstává 0 = no
    ///   incoming connection from new recurrent slot; selekce + future
    ///   weight mutace mohou prokopnout, pokud má smysl).
    ///
    /// `active_inputs = BRAIN_INPUTS_SENSORY + (hidden_n+1)` zahrnuje vlastní
    /// recurrent slot nového neuronu (= připojení k své vlastní paměti).
    pub fn add_neuron(&mut self, rng: &mut impl Rng, sigma: f32) -> bool {
        let new_idx = self.hidden_n as usize;
        if new_idx >= BRAIN_HIDDEN {
            return false;
        }
        let active_inputs = BRAIN_INPUTS_SENSORY + new_idx + 1;
        for j in 0..active_inputs {
            self.w1[new_idx][j] = gaussian(rng) * sigma;
        }
        self.b1[new_idx] = gaussian(rng) * sigma;
        for o in 0..BRAIN_OUTPUTS {
            self.w2[o][new_idx] = gaussian(rng) * sigma;
        }
        self.hidden_n += 1;
        true
    }

    /// Sprint 104: classic NEAT split_link. Vyber random (input, hidden) pár
    /// s |w|>threshold, deaktivuj přímou cestu (w → 0), insert nový hidden
    /// neuron k mezi nimi: w1[k][input] = 1.0, w2[output][k] = original_w
    /// (resp. pro hidden→hidden link: posuneme přes prostřední neuron).
    /// Vrací `true` pokud mutace proběhla.
    ///
    /// **Topology-preserving:** forward output je při split exactly stejný
    /// jako pre-split (pre-tanh: 1.0 × x = x, post tanh × original_w =
    /// original × tanh(x) — pro malé x ≈ original × x). Drobná nelinearity
    /// drift, ale zhruba zachovává funkci.
    pub fn split_link(&mut self, rng: &mut impl Rng, threshold: f32) -> bool {
        let new_idx = self.hidden_n as usize;
        if new_idx >= BRAIN_HIDDEN {
            return false;
        }
        // Find candidates: w1 entries (h, i) s |w|>threshold (active links).
        let h_n = self.hidden_n as usize;
        let active_inputs = BRAIN_INPUTS_SENSORY + h_n;
        let mut candidates: Vec<(usize, usize, f32)> = Vec::new();
        for h in 0..h_n {
            for i in 0..active_inputs {
                let w = self.w1[h][i];
                if w.abs() > threshold {
                    candidates.push((h, i, w));
                }
            }
        }
        if candidates.is_empty() {
            return false;
        }
        let pick = rng.random_range(0..candidates.len());
        let (h_target, i_src, w_orig) = candidates[pick];
        // Disable direct path
        self.w1[h_target][i_src] = 0.0;
        // Wire through new node k = new_idx
        // input i_src → k weight = 1.0
        // k → h_target via recurrent slot (h_target is fed from inputs[BRAIN_INPUTS_SENSORY + k])
        // But w1 connects inputs to hidden, not hidden to hidden.
        // Original link was input i_src → hidden h_target via w1.
        // New: input i_src → hidden k (= new_idx) via w1[k][i_src] = 1.0
        // Then hidden k must influence hidden h_target. Recurrent path:
        //   k's tanh output → next tick's inputs[BRAIN_INPUTS_SENSORY + k]
        //   → w1[h_target][BRAIN_INPUTS_SENSORY + k] propagates to h_target.
        // Set that recurrent weight to w_orig.
        self.w1[new_idx][i_src] = 1.0;
        self.b1[new_idx] = 0.0;
        // recurrent index for k:
        let rec_idx = BRAIN_INPUTS_SENSORY + new_idx;
        if rec_idx < BRAIN_INPUTS {
            self.w1[h_target][rec_idx] = w_orig;
        }
        // Output side: leave existing w2 (k contributes only via recurrent
        // path, drives same downstream signal next tick).
        self.hidden_n += 1;
        true
    }

    /// Sprint 104: structural mutation — odeber nejnižší prioritní hidden
    /// neuron. Decrement hidden_n a zero-out jeho weights (oba w1 row +
    /// w2 column + b1 entry). Vrací `true` pokud proběhlo (jinak při
    /// hidden_n ≤ BRAIN_HIDDEN_MIN).
    pub fn remove_neuron(&mut self, rng: &mut impl Rng) -> bool {
        let h_n = self.hidden_n as usize;
        if h_n <= BRAIN_HIDDEN_MIN {
            return false;
        }
        let pick = rng.random_range(0..h_n);
        // Zero out w1 row + b1 + w2 column
        for j in 0..BRAIN_INPUTS {
            self.w1[pick][j] = 0.0;
        }
        self.b1[pick] = 0.0;
        for o in 0..BRAIN_OUTPUTS {
            self.w2[o][pick] = 0.0;
        }
        // Compact: pokud pick je poslední, jen decrementuj. Jinak swap-remove
        // (last neuron → pick slot) k zachování dense [0..hidden_n] layout.
        let last = h_n - 1;
        if pick != last {
            self.w1[pick] = self.w1[last];
            self.b1[pick] = self.b1[last];
            for o in 0..BRAIN_OUTPUTS {
                self.w2[o][pick] = self.w2[o][last];
            }
            // Zero out the (now duplicate) last slot.
            for j in 0..BRAIN_INPUTS {
                self.w1[last][j] = 0.0;
            }
            self.b1[last] = 0.0;
            for o in 0..BRAIN_OUTPUTS {
                self.w2[o][last] = 0.0;
            }
            // NOTE: recurrent slot remap — ostatní neurony, které měly
            // vstup z [BRAIN_INPUTS_SENSORY + last] (= last's recurrent feed)
            // teď čtou prázdný slot (0). Ostatní s inputem z [SENSORY+pick]
            // teď čtou last's signal. Je to slight semantic drift; pro tichou
            // kompatibilitu by chtělo plné remapping, ale acceptujeme drobný
            // disruption — selekce kompenzuje.
        }
        self.hidden_n -= 1;
        true
    }

    /// Per-row uniform crossover. Each hidden neuron's `w1` row + `b1`
    /// scalar comes from one parent (50/50); same for output neurons. Per-row
    /// rather than per-weight preserves coordinated patterns within a single
    /// neuron's receptive field.
    ///
    /// Sprint 104: structural mutace mohou rozejít `hidden_n` rodičů. Pokud
    /// neshoda, vezmi menší size (= disjoint hidden slots z většího parenta
    /// nesharedily — drop). Child = min(a.hidden_n, b.hidden_n), per-row
    /// crossover přes shared rozsah, base = parent s menším hidden_n
    /// (zachovává jeho dead-zone weights v 0).
    pub fn crossover(a: &Brain, b: &Brain, rng: &mut impl Rng) -> Brain {
        let h_n = a.hidden_n.min(b.hidden_n) as usize;
        // Base = parent s menším hidden_n (jeho rows beyond h_n jsou zero
        // díky add_neuron / remove_neuron logice).
        let (base, other) = if a.hidden_n <= b.hidden_n {
            (a, b)
        } else {
            (b, a)
        };
        let mut out = *base;
        out.hidden_n = h_n as u32;
        for i in 0..h_n {
            if rng.random::<bool>() {
                out.w1[i] = other.w1[i];
                out.b1[i] = other.b1[i];
            }
        }
        for i in 0..BRAIN_OUTPUTS {
            if rng.random::<bool>() {
                out.w2[i] = other.w2[i];
                out.b2[i] = other.b2[i];
            }
        }
        out
    }
}
