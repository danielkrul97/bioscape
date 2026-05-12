# Sprinty 133–142: Neuroplasticita

Předchozí desítky (083–092 perception, 121–130 spikes, 128–137 perf) škálovaly
**vstupní šířku** mozku a optimalizovaly **kompute throughput**. Tahle desítka
otevírá *lifetime plasticitu* — brain weights se učí během života, ne jen
přes generace.

Pre-S133 stav: reward-modulated Hebbian s eligibility traces (Wave 7) je
zapojený, ale **reward signál sám je úzký** — zapisuje se jen ve dvou místech
(`world.rs:1972` novelty v maze, `world.rs:2678` eat food). Predace, escape,
bond formace, mating success, damage avoidance — žádný z těchto eventů
netriggeruje Hebbian update. `LEARNING_RATE` a `HEBBIAN_TRACE_DECAY_PER_SEC`
jsou globální konstanty (`params/reproduction.rs:51`, `params/maze.rs:58`) —
neuromodulace (per-cell rychlost učení) chybí. Žádná homeostatická plasticita
(synaptic scaling / BCM threshold drift). Brain je čistý rate-based perceptron;
SNN/STDP je doc-only TODO (docs/05).

**Cíl desítky:** rozšířit lifetime brain plasticity z foraging-only Hebbian
na full reward funnel + per-cell evolved neuromodulace + homeostatic
stabilizaci. Sprint 142 přidává passive spike-event stream jako bridge pro
budoucí SNN/STDP desítku 143–152 (mimo scope této desítky — viz outline na
konci).

Rozdělení do sprintů (každý zhruba 1 commit):

- **133–135** Reward funnel rozšíření (RewardEvent abstrakce → predation/
  escape → damage/bond/mating)
- **136–137** Per-cell evolved `learning_rate` + `trace_decay_per_sec` (CPU
  → GPU)
- **138–139** Homeostatic plasticita (synaptic scaling + BCM-lite excitability)
- **140–141** Observability + cross-seed validace
- **142** SNN bridge (passive spike-event stream) + decade retro

## Sprint 133 — RewardEvent abstrakce + integration audit

**Cíl:** zavést `RewardKind` enum + `RewardAccumulator` jako producer-consumer
mechanism. Refaktor 2 existujících reward sources (eat `world.rs:2678`,
novelty `world.rs:1972`) na push-event pattern; flush na konci ticku do
existujícího `rewards: Vec<f32>`. Žádná změna logiky — byte-identical baseline.

**Výstup:** new `src/neural/reward.rs` (`RewardKind`, `RewardEvent`,
`RewardAccumulator`, `FlushMode::{Replace, SumAndClamp}`, `REWARD_CLAMP_*`).
`World::apply_episodic_novelty` (GPU path) a `World::eat_food` (GPU path)
oba routují přes `acc.push(idx, kind, magnitude)` + `acc.flush_to(&mut rewards,
FlushMode::Replace)` těsně před GPU dispatch. CPU paths (par_iter_mut novelty,
`ate_cell_indices` non-GPU eat) zachované beze změny — accumulator je v S133
GPU-side abstrakce. 7 unit testů pro `RewardAccumulator`. Lib testy 434
passed (1 ignored). Seed=0 1-gen smoke byte-identical s committed baseline
(`run_seed0.csv`) až na `ticks_per_sec` (standard wallclock drift).

**Poznámky:** flush v S133 zachovává **last-write-wins semantiku** existujícího
kódu (jeden cell ≤ 1 reward event/tick v praxi, ale ne formální invariant).
S134+ migruje na sum semantiku s clampem `[REWARD_CLAMP_MIN, REWARD_CLAMP_MAX] =
[-2, +2]`. CPU hebbian paths zatím obejdou accumulator — refactor odložen
na S135 spolu s expanzí flush sites (damage / bond / mate).

## Sprint 134 — predation + escape rewards

**Cíl:** attacker dostane `Predation(+gain × factor)` při úspěšném drain;
obeť dostane `EscapedAttack(+0.3)` pokud `under_attack_streak ≥ 30 ticks`
a v aktuálním ticku `damage_accum == 0` (cooldown 60 ticks aby se neopakovalo).

**Výstup:** new fields `Cell.under_attack_streak: u16`, `Cell.escape_cooldown_ticks: u16`
(serde default 0, backward-compat s pre-S134 checkpointy). Nové params v
`params/reproduction.rs`: `PREDATION_REWARD_SCALE = 0.4`, `PREDATION_REWARD_MAX = 1.0`,
`ESCAPE_REWARD_MAGNITUDE = 0.3`, `ESCAPE_STREAK_THRESHOLD = 30`, `ESCAPE_COOLDOWN_TICKS = 60`.
`World::predate` po aplikaci `energy_delta` / `damage_delta` buduje per-tick
`RewardAccumulator` (`SumAndClamp` flush), dispatchne separátní GPU Hebbian
apply pass (LEARNING_RATE). Streak inkrementuje při damage > 0, reset na
damage-free ticku; escape reward fire jen pokud streak ≥ threshold a cooldown
== 0. Lib testy 434 passed (1 ignored). Seed=0 2-gen smoke: pop 312 (zachováno),
`weight_diversity_w1_norm` drift 61.10→61.56 (gen 1) — očekávaná divergence
od reward signal pro predátory.

**Poznámky:** Renderer (`src/main.rs`) má vlastní predate path; S134 mění
jen headless. Renderer mirror odložen do S141 retro pokud cross-seed validace
potvrdí pozitivní efekt. Acceptance: 3 seedy × 15 gen — predátorské lineages
konvergují k attack policy o ≥30 % rychleji než pre-S134 baseline (validace
S141, ne v rámci S134 acceptance).

## Sprint 135 — damage = negative reward + bond/mating rewards

**Cíl:** uzavřít zápornou smyčku + odměnit social events. Migrate flush
na **sum semantiku** + globální clamp.

**Výstup:** nové params: `DAMAGE_REWARD_GAIN = 0.1`, `BOND_FORMED_REWARD_MAGNITUDE
= 0.2`, `MATING_REWARD_MAGNITUDE = 0.5`. `World::predate` emit `Damage(-damage
× gain)` u damage_this_tick > 0 (joined predate accumulator). `World::apply_hazards`
nový dispatch — push `Damage` per cell where `drain > 0`. `World::resolve_collisions`
bond-formation site collectne `BondFormed` pro oba i_a/i_b a dispatchne post-loop.
`World::reproduce` přidá pre-extend dispatch `MateSignalAccepted` pro oba
parents v matings list. Migrace eat + novelty flush mode na `SumAndClamp`
(byte-equivalent v single-event-per-cell režimu). Lib testy 434 passed (1 ignored).
Seed=0 2-gen smoke: pop 200→196→361, bonds_formed 0→2 (gen 2), predation_events
12→78 (predator brain odměňován → silnější attack policy). Žádná extinkce,
populace v rozsahu.

**Poznámky:** Asymetrické magnitudy potvrzeny — attacker `+1.0` cap (S134
PREDATION_REWARD_MAX), victim `-0.6/tick` typický (damage_delta 6 × gain 0.1)
→ útok je net-negative pro victima i po globálním clampu `[-2, +2]`. Damage
reward dominuje hazard signal (hazard drain ~0.01/tick × gain → -0.001 reward,
near noise floor). CSV metrika `damage_avoidance_score` odložena do S140
observability sprintu. Renderer (`src/main.rs`) nezahrnut — headless je
canonical pro plasticity sprinty 133-141; renderer mirror v S141 retro.

## Sprint 136 — Genome: per-cell `learning_rate` + `trace_decay` (CPU)

**Cíl:** evolvable per-cell traits jako dopamin/serotonin analog. `learning_rate ∈ [0, 0.02]`
+ `trace_decay_per_sec ∈ [0.1, 5.0]` v Genome; mutate + crossover; CPU
hebbian call sites zaměnit globální konstanty za per-cell hodnoty.

**Výstup:** nové bounds v `params/reproduction.rs`: `MIN/MAX_LEARNING_RATE`,
`MIN/MAX_TRACE_DECAY_PER_SEC`. `Genome` rozšířen o `learning_rate: f32`,
`trace_decay_per_sec: f32` (serde default na globální konstanty pro
backward-compat). `MutationConfig` rozšířen o `sigma_learning_rate`,
`sigma_trace_decay` (default 0 v `MUTATION_CONFIG`). `Genome::random`,
`mutate_no_brain` (sigma > 0 short-circuit pattern jako S82 `vision_fov`),
`crossover` (same-value short-circuit). CPU hebbian call sites
(`world.rs:1996, 2784`, `renderer/systems/brains.rs:39, 102`) čtou per-cell
hodnoty z `cell.genome`. GPU sites zatím dál uniform `LEARNING_RATE` (S137
plumbe per-cell). Test fixtures (tests.rs × 4 sites, tests_phase2.rs × 1)
aktualizovány. Lib testy 434 passed (1 ignored). Seed=0 1-gen smoke
byte-identical s S135 baseline (CPU paths dead code v `--gpu-full`, GPU
path unchanged).

**Poznámky:** Sigmas zůstávají `0.0` v S136 default — drift se aktivuje
spolu s GPU plumbingem v S137 (`sigma_learning_rate = 0.001`,
`sigma_trace_decay = 0.05`). Run-to-run pop variance při ≥2 gen pozorována
už od S134 (multi-dispatch round-off / rayon noise), ne S136-specific.

## Sprint 137 — GPU per-cell rates

**Cíl:** GPU `hebbian_step.wgsl` + `hebbian_apply_reward.wgsl` čtou rates
z storage bufferů místo uniform scalaru. Sigmas zapnuty.

**Výstup:** _(po dokončení)_

**Poznámky:**
- Nové bindings `learning_rates: array<f32>` + `trace_decays: array<f32>`
  (16 B/cell, ~24 KB @ 1500 cells).
- Storage limit check: dnes hebbian shadery 6 bindings, +2 = 8 (pod limitem 12).
- Upload v `reproduce` phase z Genome.
- Zapnout `sigma_learning_rate = 0.001` + `sigma_trace_decay = 0.05`.
- Parity test `hebbian_apply_reward_gpu_matches_cpu` s per-cell rates (ε=1e-4).
- Cross-seed 15 gen ukazuje non-trivial drift `lr_avg` z init mean.

## Sprint 138 — synaptic scaling (L2 norm cap)

**Cíl:** zabránit weight explosion přes periodickou L2 renormalizaci row-wise.

**Výstup:** _(po dokončení)_

**Poznámky:**
- `SCALING_PERIOD_TICKS = 600` (~10 s @ 60 Hz). Pro každý cell, pro každý row
  `w1[i]` (i `w2[o]`): pokud `||row||₂ > W_NORM_CAP` (8.0) → row ×= cap/norm.
- CPU `Brain::synaptic_scale()` + GPU `synaptic_scale.wgsl` (per-cell 1
  workgroup, sum-of-squares reduce + scale).
- Trigger v tick loop (modulo SCALING_PERIOD_TICKS).
- Parity GPU vs CPU ε=1e-4; CSV `w_norm_avg` bounded přes 50-gen single-seed.

## Sprint 139 — intrinsic excitability (BCM-lite threshold drift)

**Cíl:** per-neuron bias drift — chronicky over-active neuron si zvedne práh,
under-active sníží (Turrigiano-style).

**Výstup:** _(po dokončení)_

**Poznámky:**
- `Brain` add `pub activity_avg: [f32; BRAIN_HIDDEN]` (EMA alpha 0.01, persisted
  přes generace přes serde default).
- Per neuron každý tick: `if activity_avg[i] > 0.7 → b1[i] -= 0.001`;
  `if < 0.3 → b1[i] += 0.001`. Symetrická drift k aktivnímu středu ±0.5.
- CPU + GPU (možno fold do `hebbian_step.wgsl` jako secondary pass nebo
  samostatný shader).
- Validace: activity_avg distribuce per pop je centered (ne všechny ±1).

## Sprint 140 — observability (CSV breakdown)

**Cíl:** přidat per-generation CSV columns rozkládající reward funnel,
neuromodulaci, weight statistics, excitability.

**Výstup:** _(po dokončení)_

**Poznámky:**
- Nové sloupce: `reward_eat_avg`, `reward_novelty_avg`, `reward_predation_avg`,
  `reward_escape_avg`, `reward_damage_avg`, `reward_bond_avg`, `reward_mate_avg`,
  `lr_avg`, `lr_std`, `decay_avg`, `decay_std`, `w_norm_avg`, `activity_imbalance`.
- `RewardAccumulator` drží running sums per kind per gen.
- `Brain::weight_l2_norm()`, `Brain::activity_imbalance()` helpery.
- Perf regression < 5 % vs S139.

## Sprint 141 — cross-seed validation + retro

**Cíl:** 3 seedy × 30 gen × {maze, open} sweep; validation report.

**Výstup:** _(po dokončení)_

**Poznámky:**
- Generuj `docs/sprints/133-142-validation.md` se srovnání pre-S133 baseline.
- Acceptance: žádná extinkce; predator `predation_avg > 0.1` od gen 20;
  damage_avoidance ≥ baseline + 15 %.
- Update CLAUDE.md / README progress.

## Sprint 142 — SNN bridge + decade retro

**Cíl:** připravit infrastructure pro SNN/STDP v 143–152 jako passive event
stream nad rate-based brainem (zero behaviorální dopad).

**Výstup:** _(po dokončení)_

**Poznámky:**
- V `Brain::forward` post-tanh: emit `spike_event` pokud `last_hidden[i] > 0.8`
  s 1-tick refractory per neuron.
- Akumuluj per-gen count do CSV (`spike_count_avg`).
- Žádná logika měnící váhy.
- Decade retrospektiva ve stejném souboru — co fungovalo, co cross-seed
  nepřeneslo, doporučení pro 143–152.

## Outline navazující desítky 143–152 (SNN / STDP)

**Mimo scope desítky 133–142.** Plný SNN refactor (forward pass, hidden state
1D → 2D `(v, u)`, weight semantika, GPU shader) je ortogonální projekt:

- **143** `enum NeuronModel { Perceptron, Izhikevich }` v Genome, dual-path
  forward (perceptron default, Izhikevich opt-in přes mutaci).
- **144–145** Izhikevich hidden state `(v: membrane potential, u: recovery)`
  per neuron + sub-timestep integrátor (dt/4).
- **146** STDP rule pre-post timing window (~20 ms ≈ 1.2 ticks @ 60 Hz).
- **147–148** GPU shader rewrite (per-neuron state buffer, sub-timestep
  iterace, storage limit budget).
- **149** rate→Poisson encoding pro sensory inputs.
- **150–152** parity testy, cross-seed, retro.
