use rand::Rng;
use serde::{Deserialize, Serialize};
use wide::f32x8;

use super::activation::{tanh_fast_scalar, tanh_fast_simd};
use crate::*;

// ─── Sprint 105: HyperNEAT CPPN scaffolding ─────────────────────────────────
//
// CPPN (Compositional Pattern-Producing Network) je malá heterogenní NN
// s diverse activation functions. V S106 bude generovat Brain weights na
// základě geometric coords substrate neuronů. V S105 je standalone — datová
// struktura, mutace, crossover, forward pass, tests.

/// Activation functions for CPPN nodes. HyperNEAT-classic library —
/// rozmanité tvary vedou k symetrickým / periodic patterns ve weight space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActivationFn {
    Linear,
    Sigmoid,
    Tanh,
    Gaussian,
    Sine,
    Abs,
    Step,
}

impl ActivationFn {
    pub fn apply(&self, x: f32) -> f32 {
        match self {
            ActivationFn::Linear => x,
            ActivationFn::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            // Padé approximation, matches the SIMD lane to keep
            // `Cppn::forward` and `forward_batch_x8` bit-identical.
            ActivationFn::Tanh => tanh_fast_scalar(x),
            ActivationFn::Gaussian => (-x * x).exp(),
            ActivationFn::Sine => x.sin(),
            ActivationFn::Abs => x.abs(),
            ActivationFn::Step => {
                if x >= 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }

    /// f32x8 lane-parallel version of `apply`. Same activation applied to
    /// 8 batched samples that all share the node (so dispatch is once per
    /// node, not per lane).
    #[inline]
    fn apply_simd(&self, x: f32x8) -> f32x8 {
        match self {
            ActivationFn::Linear => x,
            ActivationFn::Sigmoid => f32x8::splat(1.0) / (f32x8::splat(1.0) + (-x).exp()),
            ActivationFn::Tanh => tanh_fast_simd(x),
            ActivationFn::Gaussian => (-(x * x)).exp(),
            ActivationFn::Sine => x.sin(),
            ActivationFn::Abs => x.abs(),
            // wide 1.3 has no `cmp_ge`; scalar fallback for the rare Step
            // case avoids a flaky bit-hack that would mishandle negative
            // zero (`-0.0 >= 0.0` is true in IEEE-754).
            ActivationFn::Step => {
                let arr = x.to_array();
                let mut out = [0.0_f32; 8];
                for i in 0..8 {
                    out[i] = if arr[i] >= 0.0 { 1.0 } else { 0.0 };
                }
                f32x8::new(out)
            }
        }
    }

    pub fn random(rng: &mut impl Rng) -> Self {
        match rng.random_range(0..7) {
            0 => ActivationFn::Linear,
            1 => ActivationFn::Sigmoid,
            2 => ActivationFn::Tanh,
            3 => ActivationFn::Gaussian,
            4 => ActivationFn::Sine,
            5 => ActivationFn::Abs,
            _ => ActivationFn::Step,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CppnNode {
    pub id: u32,
    pub activation: ActivationFn,
    pub bias: f32,
    /// Layer index pro topological sort. Inputs = 0, outputs = max_layer,
    /// hidden ∈ (0, max_layer). Add_node split insertem dostane layer mezi
    /// from a to.
    pub layer: u32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CppnLink {
    /// Innovation id — monotonic per-Cppn. Per-Cppn lokální (ne globální
    /// jako v classic NEAT). Stačí pro speciation distance v rámci jedné
    /// linie; cross-population alignment je proxy via crossover.
    pub innovation: u32,
    pub from: u32,
    pub to: u32,
    pub weight: f32,
    pub enabled: bool,
}

/// Sprint 105: CPPN config. CPPN_INPUTS=6 stačí pro 3D substrate (x1,y1,z1,
/// x2,y2,z2 = coords obou neuronů co spojuje). Plus volitelný bias input
/// (1.0 const). CPPN_OUTPUTS=2: weight + link_existence (gate via threshold).
pub const CPPN_INPUTS: usize = 7; // 6 coords + 1 bias-const
pub const CPPN_OUTPUTS: usize = 2; // weight + link_exists
/// Initial CPPN nodes count při random init: CPPN_INPUTS + CPPN_OUTPUTS
/// + 1 hidden neuron na startup. Growable přes add_node mutace.
pub const CPPN_INITIAL_HIDDEN: usize = 1;
/// Maximum CPPN nodes celkem. Soft cap k zabránění memory blow-up. 64 dává
/// (CPPN_INPUTS + CPPN_OUTPUTS + ~55 hidden), což pokryje phenotype rozsah
/// většiny HyperNEAT studií.
pub const CPPN_MAX_NODES: usize = 64;
/// Maximum CPPN links — quadratic-ish growth s nodes, soft cap.
pub const CPPN_MAX_LINKS: usize = 256;

/// Sprint 106: fixed-size arrays místo Vec — preserves Copy trait pro Genome.
/// Packed layout: nodes[0..num_nodes], links[0..num_links] valid; zbytek None.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Cppn {
    #[serde(with = "serde_arr_cppn_nodes")]
    pub nodes: [Option<CppnNode>; CPPN_MAX_NODES],
    #[serde(with = "serde_arr_cppn_links")]
    pub links: [Option<CppnLink>; CPPN_MAX_LINKS],
    pub num_nodes: u8,
    pub num_links: u16,
    pub next_innovation: u32,
}

mod serde_arr_cppn_nodes {
    use super::{CppnNode, CPPN_MAX_NODES};
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    pub fn serialize<S: Serializer>(
        a: &[Option<CppnNode>; CPPN_MAX_NODES],
        s: S,
    ) -> Result<S::Ok, S::Error> {
        a.as_slice().serialize(s)
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(
        d: D,
    ) -> Result<[Option<CppnNode>; CPPN_MAX_NODES], D::Error> {
        let v: Vec<Option<CppnNode>> = Vec::deserialize(d)?;
        if v.len() != CPPN_MAX_NODES {
            return Err(serde::de::Error::custom("cppn nodes length mismatch"));
        }
        let mut a: [Option<CppnNode>; CPPN_MAX_NODES] = [None; CPPN_MAX_NODES];
        for (i, x) in v.into_iter().enumerate() {
            a[i] = x;
        }
        Ok(a)
    }
}

mod serde_arr_cppn_links {
    use super::{CppnLink, CPPN_MAX_LINKS};
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    pub fn serialize<S: Serializer>(
        a: &[Option<CppnLink>; CPPN_MAX_LINKS],
        s: S,
    ) -> Result<S::Ok, S::Error> {
        a.as_slice().serialize(s)
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(
        d: D,
    ) -> Result<[Option<CppnLink>; CPPN_MAX_LINKS], D::Error> {
        let v: Vec<Option<CppnLink>> = Vec::deserialize(d)?;
        if v.len() != CPPN_MAX_LINKS {
            return Err(serde::de::Error::custom("cppn links length mismatch"));
        }
        let mut a: [Option<CppnLink>; CPPN_MAX_LINKS] = [None; CPPN_MAX_LINKS];
        for (i, x) in v.into_iter().enumerate() {
            a[i] = x;
        }
        Ok(a)
    }
}

impl Cppn {
    /// Active nodes iterator (skip None slots).
    pub fn iter_nodes(&self) -> impl Iterator<Item = &CppnNode> {
        self.nodes
            .iter()
            .take(self.num_nodes as usize)
            .filter_map(|n| n.as_ref())
    }

    /// Active links iterator.
    pub fn iter_links(&self) -> impl Iterator<Item = &CppnLink> {
        self.links
            .iter()
            .take(self.num_links as usize)
            .filter_map(|l| l.as_ref())
    }

    fn push_node(&mut self, n: CppnNode) -> bool {
        if (self.num_nodes as usize) >= CPPN_MAX_NODES {
            return false;
        }
        self.nodes[self.num_nodes as usize] = Some(n);
        self.num_nodes += 1;
        true
    }

    fn push_link(&mut self, l: CppnLink) -> bool {
        if (self.num_links as usize) >= CPPN_MAX_LINKS {
            return false;
        }
        self.links[self.num_links as usize] = Some(l);
        self.num_links += 1;
        true
    }
}

impl Cppn {
    /// Random init: CPPN_INPUTS input nodes (Linear, layer 0) +
    /// CPPN_OUTPUTS output nodes (Tanh, layer 2) + 1 hidden (random fn,
    /// layer 1). Initial links: každý input → hidden + hidden → output
    /// s random gaussian weight.
    pub fn random(rng: &mut impl Rng) -> Self {
        let mut cppn = Cppn {
            nodes: [None; CPPN_MAX_NODES],
            links: [None; CPPN_MAX_LINKS],
            num_nodes: 0,
            num_links: 0,
            next_innovation: 0,
        };
        let mut next_id: u32 = 0;
        // Inputs (layer 0).
        let mut input_ids: [u32; CPPN_INPUTS] = [0; CPPN_INPUTS];
        for slot in input_ids.iter_mut() {
            cppn.push_node(CppnNode {
                id: next_id,
                activation: ActivationFn::Linear,
                bias: 0.0,
                layer: 0,
            });
            *slot = next_id;
            next_id += 1;
        }
        // Outputs (layer 2). Tanh dává weights ∈ [-1,1] a link_exists ∈ [-1,1].
        let mut output_ids: [u32; CPPN_OUTPUTS] = [0; CPPN_OUTPUTS];
        for slot in output_ids.iter_mut() {
            cppn.push_node(CppnNode {
                id: next_id,
                activation: ActivationFn::Tanh,
                bias: 0.0,
                layer: 2,
            });
            *slot = next_id;
            next_id += 1;
        }
        // Hidden seed neurons (layer 1).
        let mut hidden_ids: [u32; CPPN_INITIAL_HIDDEN] = [0; CPPN_INITIAL_HIDDEN];
        for slot in hidden_ids.iter_mut() {
            cppn.push_node(CppnNode {
                id: next_id,
                activation: ActivationFn::random(rng),
                bias: gaussian(rng) * 0.5,
                layer: 1,
            });
            *slot = next_id;
            next_id += 1;
        }
        // Initial bipartite links.
        let mut innovation: u32 = 0;
        for &i in &input_ids {
            for &h in &hidden_ids {
                cppn.push_link(CppnLink {
                    innovation,
                    from: i,
                    to: h,
                    weight: gaussian(rng),
                    enabled: true,
                });
                innovation += 1;
            }
        }
        for &h in &hidden_ids {
            for &o in &output_ids {
                cppn.push_link(CppnLink {
                    innovation,
                    from: h,
                    to: o,
                    weight: gaussian(rng),
                    enabled: true,
                });
                innovation += 1;
            }
        }
        cppn.next_innovation = innovation;
        cppn
    }

    /// Forward pass — feed-forward by layer. Inputs are mapped do prvních
    /// CPPN_INPUTS nodů. Outputs returned ze posledních CPPN_OUTPUTS.
    /// Layer-wise computation; cycles unsupported (add_link mutace
    /// preventuje cykly — viz `mutate_add_link`).
    ///
    /// Activations are stored in a flat `[f32; CPPN_MAX_NODES]` indexed by
    /// node id — `mutate_add_node` only ever assigns ids in `[0, num_nodes)`
    /// and `num_nodes ≤ CPPN_MAX_NODES`, so the index is always in range.
    /// Replaces a per-call `FxHashMap` heap allocation that dominated
    /// `Brain::from_cppn` (~4200 forwards per child).
    pub fn forward(&self, inputs: [f32; CPPN_INPUTS]) -> [f32; CPPN_OUTPUTS] {
        let mut activations = [0.0_f32; CPPN_MAX_NODES];
        for i in 0..CPPN_INPUTS {
            if let Some(n) = self.nodes[i] {
                activations[n.id as usize] = inputs[i];
            }
        }
        let max_layer = self.iter_nodes().map(|n| n.layer).max().unwrap_or(0);
        for layer in 1..=max_layer {
            for n in self.iter_nodes() {
                if n.layer != layer {
                    continue;
                }
                let mut sum = n.bias;
                for link in self.iter_links() {
                    if !link.enabled || link.to != n.id {
                        continue;
                    }
                    sum += link.weight * activations[link.from as usize];
                }
                activations[n.id as usize] = n.activation.apply(sum);
            }
        }
        let mut out = [0.0; CPPN_OUTPUTS];
        for o in 0..CPPN_OUTPUTS {
            if let Some(n) = self.nodes[CPPN_INPUTS + o] {
                out[o] = activations[n.id as usize];
            }
        }
        out
    }

    /// Batch forward — evaluates 8 input vectors through the same topology
    /// using `f32x8` lanes. Activation functions are applied per-node
    /// (one dispatch, all 8 lanes vectorised). `Brain::from_cppn` chunks its
    /// 4197 weight queries into groups of 8 for ~6× speedup vs scalar.
    pub fn forward_batch_x8(
        &self,
        inputs: &[[f32; CPPN_INPUTS]; 8],
    ) -> [[f32; CPPN_OUTPUTS]; 8] {
        let mut activations = [f32x8::ZERO; CPPN_MAX_NODES];
        for i in 0..CPPN_INPUTS {
            if let Some(n) = self.nodes[i] {
                activations[n.id as usize] = f32x8::new([
                    inputs[0][i], inputs[1][i], inputs[2][i], inputs[3][i],
                    inputs[4][i], inputs[5][i], inputs[6][i], inputs[7][i],
                ]);
            }
        }
        let max_layer = self.iter_nodes().map(|n| n.layer).max().unwrap_or(0);
        for layer in 1..=max_layer {
            for n in self.iter_nodes() {
                if n.layer != layer {
                    continue;
                }
                let mut sum = f32x8::splat(n.bias);
                for link in self.iter_links() {
                    if !link.enabled || link.to != n.id {
                        continue;
                    }
                    sum += f32x8::splat(link.weight) * activations[link.from as usize];
                }
                activations[n.id as usize] = n.activation.apply_simd(sum);
            }
        }
        let mut out = [[0.0_f32; CPPN_OUTPUTS]; 8];
        for o in 0..CPPN_OUTPUTS {
            if let Some(n) = self.nodes[CPPN_INPUTS + o] {
                let lanes = activations[n.id as usize].to_array();
                for b in 0..8 {
                    out[b][o] = lanes[b];
                }
            }
        }
        out
    }

    /// Mutate weight of random enabled link gaussian-style.
    pub fn mutate_weight(&mut self, rng: &mut impl Rng, sigma: f32) {
        let active: Vec<usize> = (0..self.num_links as usize)
            .filter(|&i| self.links[i].as_ref().map_or(false, |l| l.enabled))
            .collect();
        if active.is_empty() {
            return;
        }
        let pick = active[rng.random_range(0..active.len())];
        if let Some(l) = self.links[pick].as_mut() {
            l.weight += gaussian(rng) * sigma;
        }
    }

    /// Add_node: split random enabled link, insert nový hidden node.
    pub fn mutate_add_node(&mut self, rng: &mut impl Rng) {
        if (self.num_nodes as usize) >= CPPN_MAX_NODES
            || (self.num_links as usize) + 2 > CPPN_MAX_LINKS
        {
            return;
        }
        let active: Vec<usize> = (0..self.num_links as usize)
            .filter(|&i| self.links[i].as_ref().map_or(false, |l| l.enabled))
            .collect();
        if active.is_empty() {
            return;
        }
        let pick = active[rng.random_range(0..active.len())];
        let original = match self.links[pick] {
            Some(l) => l,
            None => return,
        };
        let from_layer = self
            .iter_nodes()
            .find(|n| n.id == original.from)
            .map(|n| n.layer)
            .unwrap_or(0);
        let to_layer = self
            .iter_nodes()
            .find(|n| n.id == original.to)
            .map(|n| n.layer)
            .unwrap_or(0);
        let new_layer = if from_layer + 1 < to_layer {
            from_layer + 1
        } else {
            let new_l = from_layer + 1;
            for slot in self.nodes.iter_mut() {
                if let Some(n) = slot.as_mut() {
                    if n.layer >= new_l {
                        n.layer += 1;
                    }
                }
            }
            new_l
        };
        let new_id = self.iter_nodes().map(|n| n.id).max().unwrap_or(0) + 1;
        self.push_node(CppnNode {
            id: new_id,
            activation: ActivationFn::random(rng),
            bias: 0.0,
            layer: new_layer,
        });
        if let Some(l) = self.links[pick].as_mut() {
            l.enabled = false;
        }
        let inv1 = self.next_innovation;
        let inv2 = self.next_innovation + 1;
        self.next_innovation += 2;
        self.push_link(CppnLink {
            innovation: inv1,
            from: original.from,
            to: new_id,
            weight: 1.0,
            enabled: true,
        });
        self.push_link(CppnLink {
            innovation: inv2,
            from: new_id,
            to: original.to,
            weight: original.weight,
            enabled: true,
        });
    }

    /// Add_link: pick random pair (from, to) with from.layer < to.layer.
    pub fn mutate_add_link(&mut self, rng: &mut impl Rng, sigma: f32) {
        if (self.num_links as usize) >= CPPN_MAX_LINKS {
            return;
        }
        if self.num_nodes < 2 {
            return;
        }
        let n = self.num_nodes as usize;
        for _ in 0..16 {
            let i_idx = rng.random_range(0..n);
            let j_idx = rng.random_range(0..n);
            if i_idx == j_idx {
                continue;
            }
            let from_node = match self.nodes[i_idx] {
                Some(x) => x,
                None => continue,
            };
            let to_node = match self.nodes[j_idx] {
                Some(x) => x,
                None => continue,
            };
            if from_node.layer >= to_node.layer {
                continue;
            }
            let exists = self
                .iter_links()
                .any(|l| l.from == from_node.id && l.to == to_node.id);
            if exists {
                continue;
            }
            let inv = self.next_innovation;
            self.next_innovation += 1;
            self.push_link(CppnLink {
                innovation: inv,
                from: from_node.id,
                to: to_node.id,
                weight: gaussian(rng) * sigma,
                enabled: true,
            });
            return;
        }
    }

    /// Toggle enable/disable bit of random link.
    pub fn mutate_toggle_link(&mut self, rng: &mut impl Rng) {
        if self.num_links == 0 {
            return;
        }
        let pick = rng.random_range(0..self.num_links as usize);
        if let Some(l) = self.links[pick].as_mut() {
            l.enabled = !l.enabled;
        }
    }

    /// Mutate activation function of random hidden node (skip inputs/outputs).
    pub fn mutate_activation(&mut self, rng: &mut impl Rng) {
        let hidden: Vec<usize> = (0..self.num_nodes as usize)
            .filter(|&i| {
                self.nodes[i]
                    .as_ref()
                    .map_or(false, |n| n.layer != 0 && n.layer != 2)
            })
            .collect();
        if hidden.is_empty() {
            return;
        }
        let pick = hidden[rng.random_range(0..hidden.len())];
        if let Some(n) = self.nodes[pick].as_mut() {
            n.activation = ActivationFn::random(rng);
        }
    }

    pub fn mutate(&self, rng: &mut impl Rng, cfg: &CppnMutationConfig) -> Self {
        let mut out = *self;
        if rng.random::<f32>() < cfg.weight_rate {
            out.mutate_weight(rng, cfg.sigma_weight);
        }
        if rng.random::<f32>() < cfg.add_node_rate {
            out.mutate_add_node(rng);
        }
        if rng.random::<f32>() < cfg.add_link_rate {
            out.mutate_add_link(rng, cfg.sigma_weight);
        }
        if rng.random::<f32>() < cfg.toggle_link_rate {
            out.mutate_toggle_link(rng);
        }
        if rng.random::<f32>() < cfg.activation_rate {
            out.mutate_activation(rng);
        }
        out
    }

    /// Sprint 107: NEAT compatibility distance metric. Speciation gate.
    ///   δ(a, b) = c1 × E/N + c2 × D/N + c3 × W̄
    /// kde:
    ///   E = excess gene count (innovations beyond max in other parent)
    ///   D = disjoint gene count (within range, not matching)
    ///   N = max(genome size) — normalizace
    ///   W̄ = average |weight diff| pro matching genes
    /// Constants follow classical NEAT defaults.
    pub fn compatibility_distance(a: &Cppn, b: &Cppn) -> f32 {
        const C_EXCESS: f32 = 1.0;
        const C_DISJOINT: f32 = 1.0;
        const C_WEIGHT: f32 = 0.4;
        let max_inv_a = a.iter_links().map(|l| l.innovation).max().unwrap_or(0);
        let max_inv_b = b.iter_links().map(|l| l.innovation).max().unwrap_or(0);
        let cutoff = max_inv_a.min(max_inv_b);
        let mut excess: u32 = 0;
        let mut disjoint: u32 = 0;
        let mut weight_diff_sum: f32 = 0.0;
        let mut matching: u32 = 0;
        let a_links: rustc_hash::FxHashMap<u32, &CppnLink> =
            a.iter_links().map(|l| (l.innovation, l)).collect();
        let b_links: rustc_hash::FxHashMap<u32, &CppnLink> =
            b.iter_links().map(|l| (l.innovation, l)).collect();
        for (inv, la) in a_links.iter() {
            if let Some(lb) = b_links.get(inv) {
                weight_diff_sum += (la.weight - lb.weight).abs();
                matching += 1;
            } else if *inv > cutoff {
                excess += 1;
            } else {
                disjoint += 1;
            }
        }
        for inv in b_links.keys() {
            if a_links.contains_key(inv) {
                continue;
            }
            if *inv > cutoff {
                excess += 1;
            } else {
                disjoint += 1;
            }
        }
        let n = (a.num_links.max(b.num_links) as f32).max(1.0);
        let w_avg = if matching > 0 {
            weight_diff_sum / matching as f32
        } else {
            0.0
        };
        C_EXCESS * (excess as f32) / n + C_DISJOINT * (disjoint as f32) / n + C_WEIGHT * w_avg
    }

    /// Crossover: align matching innovations + nodes by id. Random pick na
    /// matching, inherit from both na disjoint. Cap respektován (CPPN_MAX_*).
    pub fn crossover(a: &Cppn, b: &Cppn, rng: &mut impl Rng) -> Cppn {
        let mut nodes_map: rustc_hash::FxHashMap<u32, CppnNode> =
            rustc_hash::FxHashMap::default();
        for n in a.iter_nodes() {
            nodes_map.insert(n.id, *n);
        }
        for n in b.iter_nodes() {
            match nodes_map.get(&n.id) {
                Some(_) if rng.random::<bool>() => {
                    nodes_map.insert(n.id, *n);
                }
                None => {
                    nodes_map.insert(n.id, *n);
                }
                _ => {}
            }
        }
        let mut sorted_nodes: Vec<CppnNode> = nodes_map.into_values().collect();
        sorted_nodes.sort_by_key(|n| n.id);

        let mut links_map: rustc_hash::FxHashMap<u32, CppnLink> =
            rustc_hash::FxHashMap::default();
        for l in a.iter_links() {
            links_map.insert(l.innovation, *l);
        }
        for l in b.iter_links() {
            match links_map.get(&l.innovation) {
                Some(_) if rng.random::<bool>() => {
                    links_map.insert(l.innovation, *l);
                }
                None => {
                    links_map.insert(l.innovation, *l);
                }
                _ => {}
            }
        }
        let mut sorted_links: Vec<CppnLink> = links_map.into_values().collect();
        sorted_links.sort_by_key(|l| l.innovation);

        let next_innovation = sorted_links
            .iter()
            .map(|l| l.innovation + 1)
            .max()
            .unwrap_or(0);

        let mut out = Cppn {
            nodes: [None; CPPN_MAX_NODES],
            links: [None; CPPN_MAX_LINKS],
            num_nodes: 0,
            num_links: 0,
            next_innovation,
        };
        for n in sorted_nodes.into_iter().take(CPPN_MAX_NODES) {
            out.push_node(n);
        }
        for l in sorted_links.into_iter().take(CPPN_MAX_LINKS) {
            out.push_link(l);
        }
        out
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CppnMutationConfig {
    pub weight_rate: f32,
    pub sigma_weight: f32,
    pub add_node_rate: f32,
    pub add_link_rate: f32,
    pub toggle_link_rate: f32,
    pub activation_rate: f32,
}

pub const CPPN_MUTATION_CONFIG: CppnMutationConfig = CppnMutationConfig {
    weight_rate: 0.8,    // time per child má change weight
    sigma_weight: 0.5,
    add_node_rate: 0.03, // structural growth, NEAT default ~0.03
    add_link_rate: 0.05, // higher than node — more new connections than nodes
    toggle_link_rate: 0.01,
    activation_rate: 0.02,
};

// ─── Sprint 106: HyperNEAT substrate + Brain phenotype generation ───────────
//
// Substrate je geometrické rozložení sensor / hidden / output neuronů ve
// 3D prostoru. CPPN (S105) přijímá coords obou neuronů jako 6 vstupů +
// 1 bias a vrací [weight, link_exists]. Brain::from_cppn projde všechny
// possible (input, hidden) a (hidden, output) páry, populuje weights.
//
// Substrate je jednoduchý 1D: každá vrstva má z-coord (input z=-1, hidden
// z=0, output z=+1) a x-coord normalizován do [-1, 1] podle slot indexu.
// y-coord = 0 (1D substrate, scope-reduced).

/// Spočítá substrate coords pro brain input slot. Slot < BRAIN_INPUTS_SENSORY
/// jsou sensory inputs (mapovány do x-axis); slot ≥ BRAIN_INPUTS_SENSORY
/// jsou recurrent inputs (sdílí coord s hidden neuronem stejného indexu —
/// recurrent slot k mapuje na hidden neuron k coords).
pub fn substrate_input_coords(slot: usize) -> [f32; 3] {
    if slot < BRAIN_INPUTS_SENSORY {
        let x = -1.0 + 2.0 * (slot as f32) / (BRAIN_INPUTS_SENSORY as f32 - 1.0).max(1.0);
        [x, 0.0, -1.0]
    } else {
        let h_idx = slot - BRAIN_INPUTS_SENSORY;
        substrate_hidden_coords(h_idx)
    }
}

pub fn substrate_hidden_coords(slot: usize) -> [f32; 3] {
    let x = -1.0 + 2.0 * (slot as f32) / (BRAIN_HIDDEN as f32 - 1.0).max(1.0);
    [x, 0.0, 0.0]
}

pub fn substrate_output_coords(slot: usize) -> [f32; 3] {
    let x = -1.0 + 2.0 * (slot as f32) / (BRAIN_OUTPUTS as f32 - 1.0).max(1.0);
    [x, 0.0, 1.0]
}

/// CPPN_LINK_EXISTS_THRESHOLD: pokud CPPN output[1] < threshold, link je
/// "expressed off" — weight = 0 (no connection). 0.0 (Tanh midpoint) dává
/// ~50 % links by default. Posun threshold mění density derived networks.
pub const CPPN_LINK_EXISTS_THRESHOLD: f32 = 0.0;
