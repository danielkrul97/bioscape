# Sprinty 183–192: Brain stability stack

Předchozí desítka (173–182) konsolidovala shared simulation driver — renderer
a headless volají identický `bioscape::sim::World::tick()`, plasticity
pipeline (Hebbian + STDP + Izhikevich) běží v obou. Tahle desítka přidává
**introspekci** (S187: cell inspector) a hned přes ni objevuje a opravuje
**latent failures v plasticitě**: stale CPU↔GPU mirrors, weight runaway, tanh
saturation, neuronovou kolapsovou symetrii.

Pre-S187 stav: žádný způsob jak vidět brain state buňky za běhu —
diagnostika jen přes per-gen CSV (`stats.rs`), který agreguje populační
průměry. Skutečná dynamika jednoho mozku (current activations, weight
distribution, learning trajectory) je neviditelná. To skrylo, že brain
v aktuálním tuningu produkuje **degenerovaná řešení** — saturated outputs,
1 dominant neuron drives everything — která **přežijí jen díky bonded
clusters** (food-share parazitism), ne díky chytrosti.

**Cíl desítky:** dát si nástroje na introspekci brainu a pomocí nich
diagnostikovat + opravit weight pipeline tak, aby mozek produkoval **graded
computation**, ne bipolární spam. Po S187 každý další sprint v desítce
adresuje jednu vrstvu defenz proti collapse:

- **187** — Cell inspector (egui dialog, picking, brain heatmaps, JSON export)
- **188** — Stale-mirror fixy (GPU last_inputs + brain weights readback)
  + weight decay + per-tick synaptic scaling + velocity cap
- **189** — LayerNorm před tanh (rozbití recurrent saturation feedback loop)
- **190** — Oja's rule (implicit weight regularization + decorrelation pressure)
- **191** — Init jitter v `Brain::from_cppn` (symmetry breaking at birth)

Každá z 188–191 řeší jiný failure mode který předchozí vrstva odhalila.
Stack je kumulativní — žádný z nich sám o sobě nestačí.

## Sprint 187 — Cell inspector

**Cíl:** in-renderer dialog pro picking buňky + zobrazení brain state v reálném
čase + export plného snapshot jako JSON. Diagnostický nástroj nutný před
jakýmkoli ladením plasticity.

**Výstup:** nový modul `src/renderer/inspector/*` (8 souborů, 1583 LOC):
- `mod.rs` — `InspectorPlugin`, resources (`SelectedCell`, `HoverCell`,
  `PendingSave`, `ActivationHistory`).
- `picking.rs` — LMB ray-vs-bounding-sphere (custom, no `bevy_picking` dep).
  `phenotype.max_axis() × 1.15` slack. Closest-along-ray wins.
- `outline.rs` — subtle gizmo wireframe ring (XY + XZ plane) okolo vybrané
  buňky + faintější preview ring na hover.
- `dialog.rs` — egui window `760 × 720` fixed size: header s lineage/age/
  energy/[deceased badge], collapsible sekce Brain (live activations) +
  Brain — activation history + Brain — weights + Identity/Kinematics/
  Energy/Body/Genome/Bonds + footer Save/Copy.
- `brain_viz.rs` — activation bars (red-white-green gradient), weight
  heatmaps w1/w2 (signed colormap, normalized per `max|w|`), bias rows,
  multi-line history plot (top-K hidden by activity, all outputs).
- `history.rs` — `ActivationHistory` rolling buffer (`VecDeque`, 360
  ticků = 6 s @ 60 Hz). Recording aktivní jen pro selected cell.
- `export.rs` — `serde_json::to_value` + custom compact formatter (arrays
  of primitives inline, structs pretty). `rfd::AsyncFileDialog` v
  `IoTaskPool` pro Save…, `egui_ctx.copy_text()` pro Copy.

Dependencies: `bevy_egui = "0.39.1"` + `rfd = "0.17"`. EguiPlugin v
single-pass mode (`enable_multipass_for_primary_context: false`) — multipass
default kolidoval s druhou non-render Camera entitou (mystery entity 17v0,
nepatrná, ale způsobila `Multiple entities fit the query` panic).
`auto_create_primary_context: false` + explicit `PrimaryEguiContext` attach
v `PostStartup` na `Camera3d`. `enable_absorb_bevy_input_system: true` aby
kliknutí v dialogu netekly do camera orbit / god-mode RMB.

**Poznámky:** S187 byl primárně tool-building, ale **JSON export okamžitě
odhalil dvě latent bugs**:
1. `cell.last_inputs` v dumpu byl plný nul — CPU mirror nikdy nedostane
   GPU-side hodnoty. `populate_inputs.wgsl` zapisuje pouze do
   `last_inputs_buf`; CPU `cell.last_inputs` se nemění od spawnu.
2. `cell.genome.brain.b1` v dumpu byly všechny nuly — CPU mirror brain
   weights je frozen na CPPN init, Hebbian/STDP pipeline aktualizuje
   pouze GPU `brain_weights_buf`.

Inspector ukazoval lži. Fix → S188. Druhý nepříjemný side effect: cell
měla `|velocity| ≈ 305` při `max_speed = 84.6` (3.6× over) — to taky → S188.

## Sprint 188 — Stale mirror fixy + weight regularization + velocity cap

**Cíl:** opravit tři "stale CPU mirror" issues co S187 objevil
(`last_inputs`, brain weights, žádný runtime weight decay) + symptomatický
fix overspeed. Bez tohohle inspector vrací nesmysl a brain saturuje weights.

**Výstup:**

- **`last_inputs` readback**: `CellsGpu::last_inputs_rb` staging buffer
  (`src/gpu/cells.rs`). `download_full_batch_into` má nový parametr
  `inputs_out: &mut Vec<[f32; BRAIN_INPUTS]>` — single Wait barrier sdílený
  s ostatními 9 buffery. `world.rs` Phase 11 writeback: `cell.last_inputs
  = inputs[i]`. `GpuFullScratch::dl_inputs` field. Tests aktualizovány
  (4 call sites + assertions).

- **`brain_weights` readback**: `World::sync_cell_brain_from_gpu(idx)`
  jednorázový download pro selected cell (`download_brain_at`, ~18 KB,
  single Wait). Inspector volá z `pick_cell` (na klik) a `sync_selection_
  snapshot` (každý tick) — weight heatmap a JSON export jsou **live**.

- **Per-tick synaptic scaling**: `SCALING_PERIOD_TICKS: 600 → 1`
  (`params/reproduction.rs`). Pre-S188 scaling clipoval row L2 jen jednou
  za 10 sek, Hebbian growth mezi dispatches překračoval cap. Live snapshot
  (age=1353) ukazoval `||w1[7]||₂ ≈ 70` při cap=8.

- **Weight decay**: nová konstanta `WEIGHT_DECAY_PER_TICK = 0.001`. Fused
  do `synaptic_scale.wgsl`: `scale = (1 - decay) × min(1, cap × rsqrt(
  sum_sq))`. Aplikuje se **i pod cap** — to je celý smysl, pre-S188 design
  clipoval jen overshoot, takže 1 dominant neuron mohl monopolizovat output
  weights (representational collapse). Biases dostávají taky
  multiplicative decay (symmetric).

- **Velocity cap**: `VELOCITY_CAP_FACTOR = 1.5` (`params/physics.rs`).
  Cap aplikován v `resolve_collisions` po aplikaci `vel_deltas`:
  pokud `|v| > max_speed × 1.5`, scale velocity zpět. Root cause:
  `bond_velocity_delta` v `collision.wgsl` aplikuje `mag = -k × extension`
  jako **per-tick impuls bez `dt`** — efektivně 60× silnější spring
  konstanta než by "normální" fyzika čekala. Cell s extension=10 a
  stiffness=3.5 dostane `Δv = -35`/tick = `-2100` units/sec².

**Poznámky:** Velocity cap je symptomatický fix, ne root cause. Proper fix
by byl `mag *= dt` v collision shaderu + recalibrace
`BOND_STIFFNESS`/`MAX_BOND_STIFFNESS` defaults ~60×. Breaking change,
odloženo do vlastního sprintu.

Live dump po S188 (Perceptron, age=2245): w1 abs_max 19.94 → 2.65 (−87 %),
w2 abs_max 20.25 → 1.99 (−90 %), všechny řádky pod L2 cap.
**Activations stále saturated** ale weights jsou bounded — to byla S188 ambice.
Saturace v aktivacích vyžaduje LayerNorm → S189.

## Sprint 189 — LayerNorm před tanh

**Cíl:** přerušit **recurrent saturation feedback loop**. S188 zabezpečil
weights (`||w||₂ ≤ 8`), ale `tanh(w · x + b)` stále saturuje protože
`||w|| × ||x|| × cos(θ)` snadno přesáhne tanh linearní zónu (`|x| > 2.3`).
S recurrent inputem (`last_inputs[39..83] = předchozí last_hidden`) vznikne
positive feedback: saturated hidden → saturated recurrent input → saturated
preact → saturated hidden.

**Výstup:** LayerNorm aplikovaná na preact před tanh ve **všech 3 brain
forward paths**:

- **GPU `shaders/brain_forward.wgsl`** (Perceptron): two-pass — (1) compute
  preacts L1 + L2, (2) LayerNorm: `normed = (pre - mean) / sqrt(var + 1e-6)`
  → tanh. L1 normalized přes active range `[0, h_n)`, L2 přes fixed
  `BRAIN_OUTPUTS=14`.

- **GPU `shaders/brain_forward_izhikevich.wgsl`**: L1 unchanged (spike-
  based, žádné tanh), L2 stejně jako Perceptron. Workgroup-koordinovaná
  reduction — lane 0 spočte mean/inv_std do workgroup-shared scalars
  (`norm_mean`, `norm_inv_std`), ostatní threads čtou.

- **CPU `Brain::forward_with_state`** + **`forward_izhikevich_with_state`**:
  pomocná fce `layer_norm_in_place(&mut xs, n)` — pure scalar implementation
  pro CPU/GPU parity (existující parity testy musí projít).

No learnable γ/β — genome unchanged, žádný architektonický breaking change.
Po LayerNorm má preact mean=0, std=1, takže typické tanh hodnoty leží v
`[-tanh(1), tanh(1)] ≈ ±0.76`. Jen ~5 % tail values přesáhne
`|normed| > 2` a saturuje.

**Poznámky:** Live dump po S189 (Perceptron, age=3694) potvrdil success na
úrovni aktivací: `last_hidden` range `[-0.81, +0.77]` s intermediate
hodnotami (h=11: -0.42, h=12: +0.53), `last_outputs` range `[-0.80, +0.77]`
také distribuované. **Žádné ±1.0 saturace.**

ALE: weights pořád částečně collapsed. 22 z 25 active hidden neurons mělo
identické `w1` rows (delta ~0.0001 mezi nimi), w2 rows měly uniform
within-row std = 1.18. Saturace symptom je pryč, ale **decorrelation
pressure mezi neurony stále chybí** — LayerNorm jen škáluje, neumí naučit
neurony specializovat se na různé features. → S190.

## Sprint 190 — Oja's rule

**Cíl:** přidat **implicit weight regularization + decorrelation pressure**
do Hebbian update. Klasický Hebbian (`Δw = lr · post · pre`) nemá
mathematical equilibrium besides hard L2 cap — multiple neurons co
dostávají similar inputs konvergují identicky.

**Výstup:** Oja's rule (Oja 1982) zavedena přes additive correction:

```
Δw = lr · post · (pre − post · w)
   = lr · post · pre  −  lr · post² · w
   = (classic Hebbian)  +  (Oja correction)
```

Implementace:

- **GPU `shaders/hebbian_apply_reward.wgsl`** (trace-based production path):
  Oja correction `−lr · post² · w` fused do existing loop. Per (h, in) /
  (o, h) jeden read `brain_weights[w_idx]` slouží jako accumulator i
  regularizer. Jeden extra multiply-add, free na GPU.

- **GPU `shaders/hebbian.wgsl`** (legacy non-trace parity test path):
  `w += lr × post × pre − lr × post² × w` stejná substituce.

- **CPU `Brain::hebbian_update`** + **`Brain::hebbian_apply_reward`**:
  SIMD mirror (`f32x8` lanes) — `w * (1 - lr_post_sq) + lr_post * pre`.

Bias terms keep classic Hebb (no `w` to regularize na scalar).

Math properties:
- **Equilibrium**: `Δw = 0 ⟺ w = pre / post` — finite, žádný runaway, žádný
  cap potřeba pro stabilitu.
- **PCA-like decorrelation**: napříč hidden neurons, každý tíhne k jinému
  principal component vstupního prostoru (Oja's classical result). To je
  **přesně decorrelation pressure** co Hebbian sám neměl.

**Poznámky:** Live dump po S190 (Perceptron, age=2245) ukázal **Oja sám
nestačí**. 22 z 25 hidden neurons stále zkolabovalo na identical
activations (`hidden[0..22] = 0.354`, identical to 4 decimal places).
3 neurony se odlišily ven (saturated -1). Outputs: 9× `+0.6324` literally
identical, 5× `-0.8721` literally identical (byte-identical, ne podobné).

Důvod: Oja decorreluje **když neurons dostávají different update signals**.
Pokud `post[h]` je stejný pro všech 22 neurons (protože jejich `w1` rows
začaly nearly identical z CPPN substrate funkce), `Δw[h] = lr · post · pre
− lr · post² · w[h]` je identický pro všechny — stable fixed point of
symmetry. Mathematical decorrelation requires diversity at the starting
point, kterou CPPN substrate nedává. → S191.

## Sprint 191 — Init jitter v `Brain::from_cppn`

**Cíl:** rozbit initial-symmetry stable fixed point. CPPN substrate
funkce mapuje similar (hidden) coordinates na similar weights — 22+ z 25
hidden neurons má při spawn nearly identical `w1` rows, na což se Oja
v runtime nedostane.

**Výstup:** na konci `Brain::from_cppn`, po CPPN materializaci weights,
přidat deterministic gaussian perturbaci:

```rust
const INIT_JITTER_SIGMA: f32 = 0.05;
const INIT_JITTER_SEED: u64 = 0xb13c_a591_91d3_9a17;

let mut rng = StdRng::seed_from_u64(INIT_JITTER_SEED);
for h in 0..BRAIN_HIDDEN {
    for i in 0..BRAIN_INPUTS {
        w1[h][i] += gaussian(&mut rng) * INIT_JITTER_SIGMA;
    }
    b1[h] += gaussian(&mut rng) * INIT_JITTER_SIGMA;
}
for o in 0..BRAIN_OUTPUTS {
    for h in 0..BRAIN_HIDDEN {
        w2[o][h] += gaussian(&mut rng) * INIT_JITTER_SIGMA;
    }
    b2[o] += gaussian(&mut rng) * INIT_JITTER_SIGMA;
}
```

Fixed seed → `from_cppn` zůstává **pure function** of `Cppn` argument
(stejná CPPN → stejný Brain napříč runs / lineages / replays). RNG stream
postupuje per (h, i), takže different rows dostávají different noise
patterns — přesně to, co Oja's PCA-like decorrelation potřebuje jako
starting point.

**Poznámky:** Live dump po S191 (Perceptron, age=1364) — **plný success
celého stacku**:

- **Hidden activations**: range `[-0.89, +0.78]`, **25 distinct values**,
  smooth gradient přes nulu (h=10: -0.15, h=11: +0.32 — transition
  neurons s graded values).
- **Outputs**: range `[-0.83, +0.80]`, smooth gradient (o=7: -0.25
  zero-crossing).
- **w2 rows**: o=0 negative cluster, o=7 small positive, o=13 large
  positive — **každý output funkčně unique**.
- **Biases**: b1 gradient `+0.74 → -0.42` per h, b2 smooth decay
  `+0.13 → +0.001` per o. Žádné uniform clusters.

Stack defeated representational collapse. Brain produkuje **graded
computation**, ne bipolární spam.

## Decade retro

Co stack dokazuje: **single mechanism not enough**. Každá z pěti vrstev
(S187 inspector, S188 stale mirror + weight decay + L2 cap, S189 LayerNorm,
S190 Oja, S191 init jitter) je nutná. Vrstvy odpovídají různým failure
modes:

| Vrstva | Co řeší | Co odhalila |
|---|---|---|
| S187 inspector | žádný způsob diagnostiky | dva stale mirrors |
| S188 mirrors + decay + cap | runaway weight magnitudes | tanh saturation |
| S189 LayerNorm | saturated activations | within-row weight uniformity |
| S190 Oja | unbounded Hebbian growth | symmetry trap na similar substrate positions |
| S191 init jitter | identical initial weights | (stack complete) |

**Co stack NEŘEŠÍ:**

- **Bond physics**: bond impulses still applied without `dt` scaling. S188
  velocity cap maskuje symptom (`|v| ≤ 1.5 × max_speed`), ale proper fix
  (scale `mag × dt` + retune stiffness defaults) odložený. Vlastní sprint.
- **Izhikevich starvation**: S188 weight cap (L2=8) + S190 Oja sníží
  injection current pod IZH spike threshold (≈22) — Izhikevich neurons
  v aktuálním tuningu nikdy nespikují. Vyžaduje per-model cap nebo lower
  threshold. Vlastní sprint.
- **Environment selection pressure**: degenerated brain (pre-S187) přežil
  protože **environment nevybírá pro inteligenci** — bonded cluster food-
  share je viable strategy bez computation. Stack zaručuje, že brain
  *může* dělat graded computation, ale evolution musí mít důvod ji
  *používat*. Otevřená research question (scarcity, mazes, multi-step
  tasks). Vlastní decade.

**Co stack umožňuje:** od S192+ je brain regularization stable substrate
na kterém lze build vyšší mechanismy (sparsity priors, lateral inhibition,
modulační signály, working memory). Pre-S187 by každá nová architektura
nasedla na degenerated baseline; teď nasedá na functional one.

## Soubory změněné v decade

**Nové:**
- `src/renderer/inspector/*.rs` (8 souborů, ~1583 LOC) — S187
- `docs/sprints/183-192-brain-stability.md` — tento dokument

**Modifikované shadery:**
- `shaders/synaptic_scale.wgsl` — fused scale + decay (S188)
- `shaders/brain_forward.wgsl` — LayerNorm L1 + L2 (S189)
- `shaders/brain_forward_izhikevich.wgsl` — LayerNorm L2 (S189)
- `shaders/hebbian_apply_reward.wgsl` — Oja correction (S190)
- `shaders/hebbian.wgsl` — Oja correction (S190, legacy parity)

**Modifikovaný Rust:**
- `Cargo.toml` — `bevy_egui = "0.39.1"`, `rfd = "0.17"` (S187)
- `src/renderer/mod.rs` — register `InspectorPlugin` (S187)
- `src/gpu/cells.rs` — `last_inputs_rb` + extended `download_full_batch_into`
  (S188)
- `src/gpu/scratch.rs` — `dl_inputs` field (S188)
- `src/gpu/synaptic_scale.rs` — `decay` param (S188)
- `src/sim/world.rs` — `sync_cell_brain_from_gpu` + velocity cap +
  per-tick scaling dispatch (S188)
- `src/params/physics.rs` — `VELOCITY_CAP_FACTOR` (S188)
- `src/params/reproduction.rs` — `WEIGHT_DECAY_PER_TICK`,
  `SCALING_PERIOD_TICKS: 600 → 1` (S188)
- `src/neural/brain.rs` — `layer_norm_in_place` helper + LayerNorm v
  forward (S189), Oja correction v `hebbian_update` +
  `hebbian_apply_reward` (S190), init jitter v `from_cppn` (S191)
