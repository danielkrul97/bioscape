use rand::Rng;
use serde::{Deserialize, Serialize};
use wide::f32x8;

use super::activation::tanh_fast_simd;
use super::cppn::Cppn;
use crate::*;

/// Sprint 144: per-cell neuron compute model. `Perceptron` is the pre-S144
/// rate-coded tanh path (default); `Izhikevich` switches the hidden layer
/// to a spiking model (membrane potential + recovery variable + sub-timestep
/// integration). S144 just plumbs the enum through Genome; the actual
/// Izhikevich forward arrives in S146 (CPU) and S147 (GPU).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u32)]
pub enum NeuronModel {
    Perceptron = 0,
    Izhikevich = 1,
}

impl Default for NeuronModel {
    fn default() -> Self {
        NeuronModel::Perceptron
    }
}

impl NeuronModel {
    /// GPU side stores the model as `u32` per cell; this is the canonical
    /// encoding for that buffer.
    pub fn as_u32(self) -> u32 {
        self as u32
    }

    pub fn from_u32(v: u32) -> Self {
        match v {
            1 => NeuronModel::Izhikevich,
            _ => NeuronModel::Perceptron,
        }
    }
}

/// One Kahan compensated add of `addend` into 8-lane SIMD accumulator
/// `acc` with 8-lane compensation `comp`. Mirrors the scalar GPU-shader
/// loop step in `shaders/brain_forward.wgsl` so error budgets line up
/// across compute paths.
#[inline(always)]
fn kahan_step_simd(addend: f32x8, acc: &mut f32x8, comp: &mut f32x8) {
    let y = addend - *comp;
    let t = *acc + y;
    *comp = (t - *acc) - y;
    *acc = t;
}

/// Horizontally combine an 8-lane Kahan accumulator into a scalar using
/// the same compensated-summation pattern. Per-lane partial sums merge in
/// fixed order (lane 0 → lane 7) so different runs produce identical
/// last-bit results given identical lane contents.
#[inline(always)]
fn kahan_reduce_lanes(acc: f32x8) -> f32 {
    let arr = acc.to_array();
    let mut sum = 0.0_f32;
    let mut comp = 0.0_f32;
    for &x in &arr {
        let y = x - comp;
        let t = sum + y;
        comp = (t - sum) - y;
        sum = t;
    }
    sum
}

impl Brain {
    /// All-zero brain. Used as a placeholder when the caller knows the
    /// brain will be materialised by the very next operation (e.g.,
    /// `Genome::crossover` returns this so the chained `.mutate()` can fill
    /// in the real weights via `Brain::from_cppn` once — instead of twice).
    pub const fn zeros() -> Self {
        Brain {
            hidden_n: BRAIN_HIDDEN_DEFAULT as u32,
            w1: [[0.0; BRAIN_INPUTS]; BRAIN_HIDDEN],
            b1: [0.0; BRAIN_HIDDEN],
            w2: [[0.0; BRAIN_HIDDEN]; BRAIN_OUTPUTS],
            b2: [0.0; BRAIN_OUTPUTS],
            trace_w1: [[0.0; BRAIN_INPUTS]; BRAIN_HIDDEN],
            trace_w2: [[0.0; BRAIN_HIDDEN]; BRAIN_OUTPUTS],
        }
    }

    /// Derive brain weights from a CPPN substrate query. For each
    /// (input, hidden) and (hidden, output) pair the CPPN is queried with
    /// both substrate coordinates and a sentinel input (1.0); the output
    /// supplies a weight and a link-existence bit. Biases come from a
    /// self-loop query (from = to, sentinel = 0). The resulting brain has
    /// `hidden_n = BRAIN_HIDDEN_DEFAULT`.
    ///
    /// All 4197 substrate queries go through `Cppn::forward_batch_x8`.
    /// Trailing partial batches (BRAIN_HIDDEN=45 has a 5-tail in L2 weights;
    /// the bias passes have 5- and 4-tails) are padded with copies of the
    /// last valid input so the SIMD lanes always do useful work; the
    /// padded outputs are discarded.
    pub fn from_cppn(cppn: &Cppn) -> Brain {
        // Pre-compute substrate coordinates — otherwise input coords are
        // recomputed BRAIN_HIDDEN times, hidden coords BRAIN_OUTPUTS times.
        let input_coords: [_; BRAIN_INPUTS] =
            std::array::from_fn(substrate_input_coords);
        let hidden_coords: [_; BRAIN_HIDDEN] =
            std::array::from_fn(substrate_hidden_coords);
        let output_coords: [_; BRAIN_OUTPUTS] =
            std::array::from_fn(substrate_output_coords);

        let mut w1 = [[0.0_f32; BRAIN_INPUTS]; BRAIN_HIDDEN];
        let mut b1 = [0.0_f32; BRAIN_HIDDEN];
        let mut w2 = [[0.0_f32; BRAIN_HIDDEN]; BRAIN_OUTPUTS];
        let mut b2 = [0.0_f32; BRAIN_OUTPUTS];

        let mut inputs_buf = [[0.0_f32; CPPN_INPUTS]; 8];

        // L1: input → hidden weights. BRAIN_INPUTS may have a tail (e.g. 74 =
        // 9×8 + 2). Trailing partial batch padded by replicating last valid
        // input; padded outputs are discarded.
        for h in 0..BRAIN_HIDDEN {
            let to_c = hidden_coords[h];
            let mut i = 0usize;
            while i < BRAIN_INPUTS {
                let count = (BRAIN_INPUTS - i).min(8);
                for k in 0..count {
                    let from_c = input_coords[i + k];
                    inputs_buf[k] = [
                        from_c[0], from_c[1], from_c[2],
                        to_c[0], to_c[1], to_c[2],
                        1.0,
                    ];
                }
                let pad = inputs_buf[count - 1];
                for k in count..8 {
                    inputs_buf[k] = pad;
                }
                let out = cppn.forward_batch_x8(&inputs_buf);
                for k in 0..count {
                    if out[k][1] >= CPPN_LINK_EXISTS_THRESHOLD {
                        w1[h][i + k] = out[k][0];
                    }
                }
                i += 8;
            }
        }

        // L1 biases: BRAIN_HIDDEN = 45 = 5 × 8 + 5 tail. Batch + pad.
        let mut h = 0usize;
        while h < BRAIN_HIDDEN {
            let count = (BRAIN_HIDDEN - h).min(8);
            for k in 0..count {
                let to_c = hidden_coords[h + k];
                inputs_buf[k] = [
                    to_c[0], to_c[1], to_c[2], to_c[0], to_c[1], to_c[2], 0.0,
                ];
            }
            let pad = inputs_buf[count - 1];
            for k in count..8 {
                inputs_buf[k] = pad;
            }
            let out = cppn.forward_batch_x8(&inputs_buf);
            for k in 0..count {
                b1[h + k] = out[k][0] * 0.5;
            }
            h += 8;
        }

        // L2: hidden → output weights. BRAIN_HIDDEN = 45 has a 5-lane tail.
        for o in 0..BRAIN_OUTPUTS {
            let to_c = output_coords[o];
            let mut h = 0usize;
            while h < BRAIN_HIDDEN {
                let count = (BRAIN_HIDDEN - h).min(8);
                for k in 0..count {
                    let from_c = hidden_coords[h + k];
                    inputs_buf[k] = [
                        from_c[0], from_c[1], from_c[2],
                        to_c[0], to_c[1], to_c[2],
                        1.0,
                    ];
                }
                let pad = inputs_buf[count - 1];
                for k in count..8 {
                    inputs_buf[k] = pad;
                }
                let out = cppn.forward_batch_x8(&inputs_buf);
                for k in 0..count {
                    if out[k][1] >= CPPN_LINK_EXISTS_THRESHOLD {
                        w2[o][h + k] = out[k][0];
                    }
                }
                h += 8;
            }
        }

        // L2 biases: BRAIN_OUTPUTS = 12 = 1 × 8 + 4 tail. Batch + pad.
        let mut o = 0usize;
        while o < BRAIN_OUTPUTS {
            let count = (BRAIN_OUTPUTS - o).min(8);
            for k in 0..count {
                let to_c = output_coords[o + k];
                inputs_buf[k] = [
                    to_c[0], to_c[1], to_c[2], to_c[0], to_c[1], to_c[2], 0.0,
                ];
            }
            let pad = inputs_buf[count - 1];
            for k in count..8 {
                inputs_buf[k] = pad;
            }
            let out = cppn.forward_batch_x8(&inputs_buf);
            for k in 0..count {
                b2[o + k] = out[k][0] * 0.5;
            }
            o += 8;
        }

        Brain {
            hidden_n: BRAIN_HIDDEN_DEFAULT as u32,
            w1,
            b1,
            w2,
            b2,
            trace_w1: [[0.0; BRAIN_INPUTS]; BRAIN_HIDDEN],
            trace_w2: [[0.0; BRAIN_HIDDEN]; BRAIN_OUTPUTS],
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Brain {
    /// Active hidden neuron count (≤ BRAIN_HIDDEN storage). forward, mutate,
    /// crossover, and hebbian iterate only `[0..hidden_n]`; the dead zone
    /// `[hidden_n..BRAIN_HIDDEN]` stays at zero and contributes nothing.
    /// Structural mutations (add_neuron, remove_neuron, split_link) move it.
    #[serde(default = "default_hidden_n")]
    pub hidden_n: u32,
    #[serde(with = "serde_arrays_w1")]
    pub w1: [[f32; BRAIN_INPUTS]; BRAIN_HIDDEN],
    #[serde(with = "serde_arr_hidden")]
    pub b1: [f32; BRAIN_HIDDEN],
    #[serde(with = "serde_arrays_w2")]
    pub w2: [[f32; BRAIN_HIDDEN]; BRAIN_OUTPUTS],
    pub b2: [f32; BRAIN_OUTPUTS],
    /// Wave 3 eligibility trace shadow buffers — same shape as `w1`/`w2`.
    /// Each tick, `hebbian_step` decays and accumulates `pre × post` into
    /// these slots. When a reward event fires (eat / predation / novelty),
    /// `hebbian_apply_reward` does `w += lr · reward · trace` instead of
    /// the classic instantaneous `w += lr · reward · pre · post`. Effective
    /// reward window scales with `1 / decay_per_sec` — sparse-reward maze
    /// goal-reaching can credit motor outputs from many ticks earlier.
    /// Defaults to all-zeros for old checkpoints so they keep loading.
    #[serde(default = "default_trace_w1", with = "serde_arrays_w1")]
    pub trace_w1: [[f32; BRAIN_INPUTS]; BRAIN_HIDDEN],
    #[serde(default = "default_trace_w2", with = "serde_arrays_w2")]
    pub trace_w2: [[f32; BRAIN_HIDDEN]; BRAIN_OUTPUTS],
}

fn default_trace_w1() -> [[f32; BRAIN_INPUTS]; BRAIN_HIDDEN] {
    [[0.0; BRAIN_INPUTS]; BRAIN_HIDDEN]
}

fn default_trace_w2() -> [[f32; BRAIN_HIDDEN]; BRAIN_OUTPUTS] {
    [[0.0; BRAIN_HIDDEN]; BRAIN_OUTPUTS]
}

fn default_hidden_n() -> u32 {
    BRAIN_HIDDEN_DEFAULT as u32
}

// Serde 1 has native const-generic support for `[T; N]`, but nested fixed
// arrays (`[[f32; 36]; 16]`) need a manual workaround — encode as flat
// `Vec<f32>`, reconstruct on the way back.
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

// Serde's native `[T; N]` impl only covers N ≤ 32 — these wrappers handle
// `BRAIN_INPUTS` and `BRAIN_HIDDEN` which exceed that.
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

    /// Variable-size random init. `hidden_n` neurons are populated; the rest
    /// of the storage is zero and stays inert in forward passes. RNG draws
    /// happen only inside the active region: the same seed with
    /// `hidden_n = BRAIN_HIDDEN_DEFAULT` reproduces the pre-storage-bump
    /// sequence byte-identical.
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
        // Active input width = sensory inputs + recurrent slots for live
        // hiddens. With h_n = 16 default, 20 + 16 = 36 — matches the
        // pre-storage-bump RNG sequence.
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
        // Innate thrust bias: pushes b2[1] above zero so post-tanh thrust
        // lands around 0.7 instead of 0.5; without it ~half of fresh genomes
        // would get stuck doing a stationary random walk.
        b2[1] += INNATE_THRUST_BIAS;
        // Innate pheromone bias: mating gating requires nonzero emission;
        // without this, half of fresh genomes couldn't reproduce.
        b2[2] += INNATE_PHEROMONE_BIAS;
        // Innate attack bias: defaults to 0 (opt-in); routed through a
        // constant for clean A/B testing.
        b2[6] += INNATE_ATTACK_BIAS;
        // Bond signal bias — opt-in, like attack.
        b2[9] += INNATE_BOND_BIAS;
        // ch1, ch2 emit biases — gentler than ch0 (which gates mating and
        // wants a high baseline). Just enough to avoid a cold start near 0.
        b2[10] += INNATE_PHEROMONE_AUX_BIAS;
        b2[11] += INNATE_PHEROMONE_AUX_BIAS;
        Self {
            hidden_n,
            w1,
            b1,
            w2,
            b2,
            trace_w1: [[0.0; BRAIN_INPUTS]; BRAIN_HIDDEN],
            trace_w2: [[0.0; BRAIN_HIDDEN]; BRAIN_OUTPUTS],
        }
    }

    pub fn forward(&self, inputs: &[f32; BRAIN_INPUTS]) -> [f32; BRAIN_OUTPUTS] {
        self.forward_with_state(inputs).1
    }

    /// Forward pass that also returns hidden activations (needed for Hebbian
    /// updates).
    ///
    /// Matvec uses **Kahan compensated summation** in 8-lane SIMD parallel
    /// across the dot-product, then a sequential Kahan horizontal reduce.
    /// The compensation term bounds the per-dot-product error to O(ε)
    /// regardless of summation order — that is what shrinks the CPU/GPU
    /// brain-forward drift after the V8 RNG unification closed Brownian as
    /// a divergence source. Trade-off: no FMA in the inner product (Kahan
    /// requires separate mul + add to keep the compensation term valid),
    /// but the dependency chain stays SIMD-pipelined.
    ///
    /// tanh activation is still vectorized via `tanh_fast` over the active
    /// hidden range.
    // L1_TAIL/L2_TAIL branches are statically dead when BRAIN_INPUTS /
    // BRAIN_HIDDEN are multiples of 8, but kept defensive in case the
    // dimensions change. Clippy lints flag the dead branch as unreachable.
    #[allow(clippy::absurd_extreme_comparisons, clippy::out_of_bounds_indexing)]
    pub fn forward_with_state(
        &self,
        inputs: &[f32; BRAIN_INPUTS],
    ) -> ([f32; BRAIN_HIDDEN], [f32; BRAIN_OUTPUTS]) {
        const L1_FULL: usize = BRAIN_INPUTS / 8;
        const L1_TAIL: usize = BRAIN_INPUTS % 8;
        const L1_LANES: usize = L1_FULL + (L1_TAIL != 0) as usize;
        const L2_FULL: usize = BRAIN_HIDDEN / 8;
        const L2_TAIL: usize = BRAIN_HIDDEN % 8;
        const L2_LANES: usize = L2_FULL + (L2_TAIL != 0) as usize;

        let h_n = self.hidden_n as usize;

        // Pack inputs into f32x8 lanes; tail zero-padded.
        let mut input_lanes = [f32x8::ZERO; L1_LANES];
        for k in 0..L1_FULL {
            input_lanes[k] = f32x8::new(inputs[k * 8..(k + 1) * 8].try_into().unwrap());
        }
        if L1_TAIL > 0 {
            let mut tail = [0.0_f32; 8];
            tail[..L1_TAIL].copy_from_slice(&inputs[L1_FULL * 8..]);
            input_lanes[L1_FULL] = f32x8::new(tail);
        }

        // L1 matvec with Kahan compensated summation in each SIMD lane.
        let mut pre_hidden = [0.0_f32; BRAIN_HIDDEN];
        for i in 0..h_n {
            let row = &self.w1[i];
            let mut acc = f32x8::ZERO;
            let mut comp = f32x8::ZERO;
            for k in 0..L1_FULL {
                let w = f32x8::new(row[k * 8..(k + 1) * 8].try_into().unwrap());
                kahan_step_simd(w * input_lanes[k], &mut acc, &mut comp);
            }
            if L1_TAIL > 0 {
                let mut tail = [0.0_f32; 8];
                tail[..L1_TAIL].copy_from_slice(&row[L1_FULL * 8..]);
                kahan_step_simd(
                    f32x8::new(tail) * input_lanes[L1_FULL],
                    &mut acc,
                    &mut comp,
                );
            }
            pre_hidden[i] = self.b1[i] + kahan_reduce_lanes(acc);
        }

        // Vectorized tanh in chunks of 8 over active hiddens; scalar tail.
        let mut hidden = [0.0_f32; BRAIN_HIDDEN];
        let full_chunks = h_n / 8;
        for c in 0..full_chunks {
            let start = c * 8;
            let arr: [f32; 8] = pre_hidden[start..start + 8].try_into().unwrap();
            let activated = tanh_fast_simd(f32x8::new(arr)).to_array();
            hidden[start..start + 8].copy_from_slice(&activated);
        }
        for i in full_chunks * 8..h_n {
            hidden[i] = pre_hidden[i].tanh();
        }

        // Pack hidden into lanes for L2.
        let mut hidden_lanes = [f32x8::ZERO; L2_LANES];
        for k in 0..L2_FULL {
            hidden_lanes[k] = f32x8::new(hidden[k * 8..(k + 1) * 8].try_into().unwrap());
        }
        if L2_TAIL > 0 {
            let mut tail = [0.0_f32; 8];
            tail[..L2_TAIL].copy_from_slice(&hidden[L2_FULL * 8..]);
            hidden_lanes[L2_FULL] = f32x8::new(tail);
        }

        // L2 matvec with Kahan compensated summation; BRAIN_OUTPUTS is small
        // so we still do scalar tanh per output at the end.
        let mut out = [0.0_f32; BRAIN_OUTPUTS];
        for ((o, row), &bias) in out.iter_mut().zip(self.w2.iter()).zip(self.b2.iter()) {
            let mut acc = f32x8::ZERO;
            let mut comp = f32x8::ZERO;
            for k in 0..L2_FULL {
                let w = f32x8::new(row[k * 8..(k + 1) * 8].try_into().unwrap());
                kahan_step_simd(w * hidden_lanes[k], &mut acc, &mut comp);
            }
            if L2_TAIL > 0 {
                let mut tail = [0.0_f32; 8];
                tail[..L2_TAIL].copy_from_slice(&row[L2_FULL * 8..]);
                kahan_step_simd(
                    f32x8::new(tail) * hidden_lanes[L2_FULL],
                    &mut acc,
                    &mut comp,
                );
            }
            *o = (bias + kahan_reduce_lanes(acc)).tanh();
        }
        (hidden, out)
    }

    /// Reward-modulated Hebbian update: `Δw = lr · reward · pre · post`.
    /// Pre/post activations come from a stored prior forward pass, so credit
    /// assignment is myopic (1-tick window). Reward fires on biologically
    /// meaningful events (eating, predation kills).
    ///
    /// Updates iterate the full row width via `f32x8`. This is safe because
    /// dead-zone activations and inputs are zero, so `lr · 0 · x = 0` leaves
    /// those weights untouched.
    #[allow(clippy::absurd_extreme_comparisons, clippy::out_of_bounds_indexing)]
    pub fn hebbian_update(
        &mut self,
        last_inputs: &[f32; BRAIN_INPUTS],
        last_hidden: &[f32; BRAIN_HIDDEN],
        last_outputs: &[f32; BRAIN_OUTPUTS],
        reward: f32,
        learning_rate: f32,
    ) {
        const L1_FULL: usize = BRAIN_INPUTS / 8;
        const L1_TAIL: usize = BRAIN_INPUTS % 8;
        const L2_FULL: usize = BRAIN_HIDDEN / 8;
        const L2_TAIL: usize = BRAIN_HIDDEN % 8;

        let lr = learning_rate * reward;
        let h_n = self.hidden_n as usize;

        // L1: w1[i][j] += lr · last_hidden[i] · last_inputs[j].
        for i in 0..h_n {
            let scale = f32x8::splat(lr * last_hidden[i]);
            let row = &mut self.w1[i];
            for k in 0..L1_FULL {
                let w = f32x8::new(row[k * 8..(k + 1) * 8].try_into().unwrap());
                let x = f32x8::new(last_inputs[k * 8..(k + 1) * 8].try_into().unwrap());
                let updated = scale.mul_add(x, w).to_array();
                row[k * 8..(k + 1) * 8].copy_from_slice(&updated);
            }
            if L1_TAIL > 0 {
                let mut tail_w = [0.0_f32; 8];
                let mut tail_x = [0.0_f32; 8];
                tail_w[..L1_TAIL].copy_from_slice(&row[L1_FULL * 8..]);
                tail_x[..L1_TAIL].copy_from_slice(&last_inputs[L1_FULL * 8..]);
                let updated = scale
                    .mul_add(f32x8::new(tail_x), f32x8::new(tail_w))
                    .to_array();
                row[L1_FULL * 8..].copy_from_slice(&updated[..L1_TAIL]);
            }
            self.b1[i] += lr * last_hidden[i];
        }

        // L2: w2[o][j] += lr · last_outputs[o] · last_hidden[j].
        for (o, row) in self.w2.iter_mut().enumerate() {
            let scale = f32x8::splat(lr * last_outputs[o]);
            for k in 0..L2_FULL {
                let w = f32x8::new(row[k * 8..(k + 1) * 8].try_into().unwrap());
                let h = f32x8::new(last_hidden[k * 8..(k + 1) * 8].try_into().unwrap());
                let updated = scale.mul_add(h, w).to_array();
                row[k * 8..(k + 1) * 8].copy_from_slice(&updated);
            }
            if L2_TAIL > 0 {
                let mut tail_w = [0.0_f32; 8];
                let mut tail_h = [0.0_f32; 8];
                tail_w[..L2_TAIL].copy_from_slice(&row[L2_FULL * 8..]);
                tail_h[..L2_TAIL].copy_from_slice(&last_hidden[L2_FULL * 8..]);
                let updated = scale
                    .mul_add(f32x8::new(tail_h), f32x8::new(tail_w))
                    .to_array();
                row[L2_FULL * 8..].copy_from_slice(&updated[..L2_TAIL]);
            }
        }
        for (b, &o) in self.b2.iter_mut().zip(last_outputs.iter()) {
            *b += lr * o;
        }
    }

    /// Wave 3: per-tick eligibility-trace decay + accumulate. Runs every
    /// tick after the brain forward pass; no weight changes happen here —
    /// just trace bookkeeping. `decay = (1 − decay_per_sec · dt)`. Applied
    /// `trace = decay · trace + pre · post`. When a reward event later
    /// fires `hebbian_apply_reward`, the cell can credit motor outputs from
    /// many ticks earlier (effective window ~ 1 / decay_per_sec). Iterates
    /// only the active hidden range to leave dead-zone weights at 0.
    pub fn hebbian_step(
        &mut self,
        last_inputs: &[f32; BRAIN_INPUTS],
        last_hidden: &[f32; BRAIN_HIDDEN],
        last_outputs: &[f32; BRAIN_OUTPUTS],
        dt: f32,
        decay_per_sec: f32,
    ) {
        let decay = (1.0 - decay_per_sec * dt).max(0.0);
        let h_n = self.hidden_n as usize;
        let active_inputs = BRAIN_INPUTS_SENSORY + h_n;
        for i in 0..h_n {
            let post = last_hidden[i];
            for j in 0..active_inputs {
                self.trace_w1[i][j] = decay * self.trace_w1[i][j] + post * last_inputs[j];
            }
        }
        for (o, row) in self.trace_w2.iter_mut().enumerate() {
            let post = last_outputs[o];
            for j in 0..h_n {
                row[j] = decay * row[j] + post * last_hidden[j];
            }
        }
    }

    /// Wave 3: apply a reward event against the accumulated eligibility
    /// trace. `Δw[i,j] = lr · reward · trace[i,j]`. Fires on eat / predation
    /// / novelty events. Does not reset traces — they keep decaying tick by
    /// tick, so a long reward streak reinforces the same recent motor
    /// pattern multiple times. Bias terms ride on `last_hidden` /
    /// `last_outputs` since traces only cover weights, not biases.
    pub fn hebbian_apply_reward(
        &mut self,
        last_hidden: &[f32; BRAIN_HIDDEN],
        last_outputs: &[f32; BRAIN_OUTPUTS],
        reward: f32,
        learning_rate: f32,
    ) {
        let lr = learning_rate * reward;
        let h_n = self.hidden_n as usize;
        let active_inputs = BRAIN_INPUTS_SENSORY + h_n;
        for i in 0..h_n {
            for j in 0..active_inputs {
                self.w1[i][j] += lr * self.trace_w1[i][j];
            }
            self.b1[i] += lr * last_hidden[i];
        }
        for (o, row) in self.w2.iter_mut().enumerate() {
            for j in 0..h_n {
                row[j] += lr * self.trace_w2[o][j];
            }
            self.b2[o] += lr * last_outputs[o];
        }
    }

    /// Sprint 138: row-wise L2 norm cap (synaptic scaling). For each `w1[i]`
    /// and `w2[o]` row, if `||row||_2 > cap`, scale the row to `cap`. Bias
    /// vectors are not touched. Mirrors `synaptic_scale.wgsl` for parity.
    pub fn synaptic_scale(&mut self, cap: f32) {
        let h_n = self.hidden_n as usize;
        let active_inputs = BRAIN_INPUTS_SENSORY + h_n;
        let cap_sq = cap * cap;
        for i in 0..h_n {
            let mut sum_sq = 0.0_f32;
            for j in 0..active_inputs {
                sum_sq += self.w1[i][j] * self.w1[i][j];
            }
            if sum_sq > cap_sq {
                let scale = cap / sum_sq.sqrt();
                for j in 0..active_inputs {
                    self.w1[i][j] *= scale;
                }
            }
        }
        for o in 0..BRAIN_OUTPUTS {
            let row = &mut self.w2[o];
            let mut sum_sq = 0.0_f32;
            for j in 0..h_n {
                sum_sq += row[j] * row[j];
            }
            if sum_sq > cap_sq {
                let scale = cap / sum_sq.sqrt();
                for j in 0..h_n {
                    row[j] *= scale;
                }
            }
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

    /// Structural mutation: append one hidden neuron. Returns `false` if
    /// `hidden_n` is already at the cap.
    ///
    /// NEAT-style minimal-disruption init:
    /// - `w1[new_idx][0..active_inputs]` = small gaussian (drift)
    /// - `b1[new_idx]` = small gaussian
    /// - `w2[*][new_idx]` = small gaussian (output contribution starts small)
    /// - Existing neurons untouched: their `w1[i][BRAIN_INPUTS_SENSORY + new_idx]`
    ///   stays 0 (no incoming connection from the new recurrent slot until
    ///   weight mutation or selection wires it in).
    ///
    /// `active_inputs = BRAIN_INPUTS_SENSORY + (hidden_n + 1)` includes the
    /// new neuron's own recurrent slot — it can connect to its own memory.
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

    /// Classic NEAT split-link. Picks a random (input, hidden) pair with
    /// |w| > threshold, disables the direct path (w → 0), and inserts a new
    /// hidden neuron k between them so the same signal flows via a
    /// recurrent hop: `w1[k][input] = 1.0`,
    /// `w1[h_target][BRAIN_INPUTS_SENSORY + k] = original_w`. Returns
    /// `true` if the mutation was applied.
    ///
    /// **Topology-preserving (approximately):** post-split, the next-tick
    /// signal at h_target equals `original_w · tanh(input)`, vs. the
    /// pre-split `original_w · input`. For small inputs `tanh(x) ≈ x`, so
    /// behaviour drifts only mildly; selection re-tunes from there.
    pub fn split_link(&mut self, rng: &mut impl Rng, threshold: f32) -> bool {
        let new_idx = self.hidden_n as usize;
        if new_idx >= BRAIN_HIDDEN {
            return false;
        }
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
        self.w1[h_target][i_src] = 0.0;
        // Wire input i_src → new hidden k via w1[k][i_src] = 1.0.
        // k → h_target uses the recurrent path: k's tanh output reaches
        // h_target on the next tick through inputs[BRAIN_INPUTS_SENSORY + k].
        self.w1[new_idx][i_src] = 1.0;
        self.b1[new_idx] = 0.0;
        let rec_idx = BRAIN_INPUTS_SENSORY + new_idx;
        if rec_idx < BRAIN_INPUTS {
            self.w1[h_target][rec_idx] = w_orig;
        }
        self.hidden_n += 1;
        true
    }

    /// Structural mutation: remove a uniformly-random hidden neuron. The
    /// removed slot is zeroed and the last live neuron is swapped into its
    /// place to keep `[0..hidden_n]` dense. Returns `false` at the floor.
    ///
    /// Note: recurrent slot indices shift after the swap, which is a slight
    /// semantic disruption — selection is expected to compensate.
    pub fn remove_neuron(&mut self, rng: &mut impl Rng) -> bool {
        let h_n = self.hidden_n as usize;
        if h_n <= BRAIN_HIDDEN_MIN {
            return false;
        }
        let pick = rng.random_range(0..h_n);
        for j in 0..BRAIN_INPUTS {
            self.w1[pick][j] = 0.0;
        }
        self.b1[pick] = 0.0;
        for o in 0..BRAIN_OUTPUTS {
            self.w2[o][pick] = 0.0;
        }
        let last = h_n - 1;
        if pick != last {
            self.w1[pick] = self.w1[last];
            self.b1[pick] = self.b1[last];
            for o in 0..BRAIN_OUTPUTS {
                self.w2[o][pick] = self.w2[o][last];
            }
            for j in 0..BRAIN_INPUTS {
                self.w1[last][j] = 0.0;
            }
            self.b1[last] = 0.0;
            for o in 0..BRAIN_OUTPUTS {
                self.w2[o][last] = 0.0;
            }
        }
        self.hidden_n -= 1;
        true
    }

    /// Per-row uniform crossover. Each hidden neuron's `w1` row + `b1` scalar
    /// comes from one parent (50/50); same for output neurons. Per-row rather
    /// than per-weight preserves coordinated patterns inside a single
    /// neuron's receptive field.
    ///
    /// Structural mutations may diverge `hidden_n` between parents. Child
    /// inherits `min(a.hidden_n, b.hidden_n)`; the smaller-`hidden_n` parent
    /// is the base (its dead-zone weights are already zero), with per-row
    /// crossover over the shared range.
    pub fn crossover(a: &Brain, b: &Brain, rng: &mut impl Rng) -> Brain {
        let h_n = a.hidden_n.min(b.hidden_n) as usize;
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