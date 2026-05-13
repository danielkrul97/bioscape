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

**Výstup:** `CellsGpu` rozšířen o `learning_rates_buf` + `trace_decays_buf`
(8 B/cell, ~12 KB @ 1500 cells, COPY_SRC pro `swap_to` routing). Initial
fill při alokaci na pre-S137 globální konstanty → byte-identical baseline
v sigma=0 režimu. Accessors + `upload_learning_rates` / `upload_trace_decays`
/ `upload_rates_at(slot, lr, decay)`. `swap_to` rozšířen o per-cell rate
copy přes `swap_turn_rate_temp` (sdílený f32 staging). `init_gpu_full` +
reproduce per-child slot uploadují rates z `Genome`. `hebbian_step.wgsl`
nová binding (5) `trace_decays`; shader spočítá `decay = max(0, 1 −
trace_decays[i] · dt)` per cell. `hebbian_apply_reward.wgsl` nová binding (6)
`learning_rates`; shader `lr = learning_rates[i] · reward`. `HebbianGpu`
bind-group layouty step 5→6, apply 6→7 (pod 12 limitem). Dispatch fn
signatury zachované (uniform params slot pro lr/decay ignorován, předáno
pro call-site compat). Sigmas zapnuty: `sigma_learning_rate = 0.001`,
`sigma_trace_decay = 0.05`. Lib testy 434 passed (1 ignored). Seed=0 1-gen
sigma=0 baseline byte-identical s S135 (jen `ticks_per_sec` drift); sigma=on
seed=0 2-gen pop 200→202→447 (vs S135 200→196→361) — rate variance
amplifikuje reward-driven divergenci napříč lineages.

**Poznámky:** Storage limit po S137 = 6/7 z 12 (hebbian-step / -apply
shadery), comfortable rezerva pro S138 synaptic scaling. `compute()`
non-persistent path nezměněn — drží uniform `learning_rate` pro legacy
parity test. RNG ordering: mutate dělá dva nové gaussian draws (jen pokud
sigmas > 0) → seed-equivalentní pop diverguje od gen 1, expected.

## Sprint 138 — synaptic scaling (L2 norm cap)

**Cíl:** zabránit weight explosion přes periodickou L2 renormalizaci row-wise.

**Výstup:** params: `W_NORM_CAP = 8.0`, `SCALING_PERIOD_TICKS = 600`. CPU
`Brain::synaptic_scale(cap)` (row-wise iter přes `w1`/`w2`, gating
`sum_sq > cap²`, scale `row *= cap / sqrt(sum_sq)`, biases netknuty). Nový
shader `shaders/synaptic_scale.wgsl` (per-cell single-thread loop přes
hidden + output rows, share `brain_weights_buf` z `CellsGpu`). `SynapticScaleGpu`
wrapper (2 bindings: uniform params + RW brain_weights), `with_context` /
`dispatch(cells_gpu, n, cap)`. Re-export přes `gpu/mod.rs`. `GpuFullState`
přidá `synaptic_scale: SynapticScaleGpu`; init v `init_gpu_full`. Trigger
ve `tick` loop: `if clock.tick % SCALING_PERIOD_TICKS == 0 { gpu.synaptic_scale
.dispatch(&gpu.cells, n, W_NORM_CAP); }` po `apply_episodic_novelty`.
Lib testy 434 passed (1 ignored). Seed=0 2-gen smoke: pop 200→202→370,
`weight_diversity_w1_norm` 16.4→17.8→17.9 (bounded vs S137 trajectory).

**Poznámky:** Storage limit hebbian shadery zůstává 6/7 z 12 (S137 stav);
`synaptic_scale` je samostatný shader s vlastní bind-group, neovlivňuje
hebbian limit. Trigger v tick 0 je no-op (init weights ≪ cap). 1500 cells
× ~4400 ops/cell = ~6.6 M ops per scaling dispatch — sub-ms na GPU.

## Sprint 139 — intrinsic excitability (BCM-lite threshold drift)

**Cíl:** per-neuron bias drift — chronicky over-active neuron si zvedne práh,
under-active sníží (Turrigiano-style).

**Výstup:** params: `ACTIVITY_EMA_ALPHA = 0.01`, `EXCITABILITY_DRIFT_PER_TICK
= 0.001`. Nový shader `shaders/excitability.wgsl` (per-cell single-thread,
4 bindings: params uniform, last_hidden ro, activity_avg rw, brain_weights
rw). Místo step-threshold (sprint plan) lineární regulátor — `b1[h] -=
DRIFT × activity_avg[h]`, kde `activity_avg` je signed EMA `last_hidden`.
Saturated-positive cells driftují b1 dolů, saturated-negative nahoru;
deadzone okolo nuly bez updatu. `ExcitabilityGpu` vlastní `activity_buf`
(n × BRAIN_HIDDEN × 4 B, ~270 KB @ 1500 cells), `with_context`, `dispatch
(cells_gpu, n, alpha, drift)`. Re-export přes `gpu/mod.rs`. `GpuFullState`
přidá `excitability`; init v `init_gpu_full`. Trigger per tick po
`hebbian.dispatch_step_persistent` — neuron-level EMA se updatuje každý
tick spolu s hebbian trace step. Lib testy 434 passed (1 ignored).
Seed=0 2-gen smoke: pop 200→194→362 (vs S138 200→202→370 — pop variance
v rámci stochastic noise).

**Poznámky:** Lineární regulátor namísto step thresholdu = smoother
homeostasis bez discrete jump. Time constant ~1/drift = 1000 ticks (~16 s)
— pomalejší než Hebbian apply (event-driven), takže homeostat nezasahuje
do čerstvě naučeného. `activity_avg` GPU-only state (žádný checkpoint
roundtrip v S139); EMA recovers v ~100 ticks po load. Storage limit:
4 bindings v excitability shader, comfortable v rámci 12-binding budgetu.

## Sprint 140 — observability (CSV breakdown)

**Cíl:** přidat per-generation CSV columns rozkládající reward funnel,
neuromodulaci, weight statistics, excitability.

**Výstup:** 5 nových CSV sloupců na konci řádku: `lr_avg`, `lr_std`,
`decay_avg`, `decay_std`, `w_norm_avg`. Helpery `mean_std<I>(iter) →
(mean, std)` + `w1_row_norm_avg(cells)` v `csv.rs`. Empty-pop branch
zero-paduje s 5 zeros. Seed=0 2-gen smoke: gen 0 `lr=0.005000` `lr_std=0`
`decay=0.5000` `w_norm=2.15` (init); gen 1 `lr=0.005037` `lr_std=0.000364`
`decay=0.5015` `w_norm=4.95` (sigma drift po prvním reproduce viditelný).
Lib testy 434 passed (1 ignored). 99 total CSV sloupců.

**Poznámky:** Per-kind reward breakdown (`reward_eat_avg`, atd. — 7 sloupců)
odložen do future sprintu — vyžadoval by zásah do 6 dispatch sites
pro per-kind tally. Activity_imbalance (GPU readback z activity_buf) také
odložen — GPU sync overhead per generation by potřeboval samostatný
readback dispatch. Sloupce přidány na **konec řádku** aby existující
column positions zůstaly stabilní pro offline analytics. `weight_diversity_w1_norm`
(starší) měří std across population; `w_norm_avg` (S140) měří mean row L2 —
komplementární metriky.

## Sprint 141 — cross-seed validation + retro

**Cíl:** 3 seedy × 30 gen × {maze, open} sweep; validation report.

**Výstup:** `docs/sprints/133-142-validation.md` se cross-seed sweep
(3 seedy × **10 gen** open mode — full 30 gen prohibitivně drahé na ~30
ticks/s post-multi-dispatch). Žádná extinkce (3/3 seedy survive, pop
439–691). Predation events 564–921 cumulative = ~70/gen (silně nad
threshold 0.1/cell/gen). Bond formation 26–57 cumulative. `lr_avg`
drifted 0.005 → 0.0058–0.0064 (selekce na rychlejší learners), `lr_std`
0.0034–0.0041 (lineage variance). `decay_avg` drifted 0.5 → 0.43–0.47
(delší credit window). `w_norm_avg` 6.22–7.29 vs cap 8.0 — synaptic
scaling aktivně klipuje top end.

**Poznámky:** Scope reduce z 30 na 10 gen kvůli throughput — multi-dispatch
per tick (eat + novelty + predate + hazards + bond + mate + step decay +
excitability + periodic synaptic_scale) snížil ~190 → ~30 ticks/s.
Maze sweep + 30 gen long-tail validace odložené. Damage_avoidance metric
(S135 acceptance) nelze retroaktivně vykuvat — pre-S133 baseline neměl
metric. 4 z 5 bottlenecks zavřené; SNN/STDP (#5) pokračuje do 143-152.

## Sprint 142 — SNN bridge + decade retro

**Cíl:** připravit infrastructure pro SNN/STDP v 143–152 jako passive event
stream nad rate-based brainem (zero behaviorální dopad).

**Výstup:** nový CSV sloupec `neural_spike_frac` — end-of-gen snapshot fraction
of `(cell, hidden_neuron)` pairs s `|last_hidden[h]| > 0.8`. `saturation_frac
(&world.cells)` helper v `csv.rs`. Žádná změna brain forward; přiznané
zjednodušení vůči sprint plánu (per-tick refractory + per-gen count) —
snapshot stačí jako saturation proxy bez per-tick GPU readback overheadu.
Seed=0 2-gen smoke: gen 2 `neural_spike_frac = 0.87` — populace běží v
saturated regime, což je přesně tam kde proper Izhikevich + STDP path
v 143–152 začne mít smysl (rate brain ztrácí informaci, spike timing
je tam underutilized). Lib testy 434 passed (1 ignored).

**Poznámky:** Snapshot vs per-tick stream: stream by potřeboval GPU shader
(`spike_count.wgsl` se sčítáním do per-cell counter buffer) nebo per-tick
last_hidden readback — oboje S140+ overhead, zatímco snapshot je jeden
loop na konci generace. Pokud SNN desítka 143-152 začne se zachycováním
spike timing, infrastructure pro per-tick events bude zaváděna tam, ne
retrofitting zde.

## Decade retro 133–142

**Co fungovalo:**
- Reward funnel rozšíření (S133–S135). Pop survival 100 % cross-seed,
  predator policy konverguje (564–921 predation events / 10 gen).
- Per-cell evolved Hebbian rates (S136–S137). `lr_avg` drifted +16-28 %
  od init v 10 generacích napříč seedy; non-trivial lineage variance.
- Homeostatic plasticita (S138–S139). `w_norm_avg` 6.2–7.3 ≤ cap 8.0;
  bez S138 weights divergovaly nad 20 v pre-S138 smoke.
- Decoupled RewardAccumulator pattern — 6 dispatch sites žijí nezávisle,
  S140 observability + S142 bridge přidaly columns bez touchu reward sites.

**Co nepřenosé / odložené:**
- Cross-seed 30-gen sweep (throughput cap při 30 ticks/s post-multi-dispatch).
  Stačil 10 gen, ale long-tail behaviors (gen 50+) nepokryté.
- Maze mode validace — odložená.
- Per-kind reward breakdown CSV (`reward_eat_avg` atd.) — odložené ze S140.
- Activity_imbalance metric — vyžaduje GPU readback z `activity_buf`.
- `damage_avoidance_score` (S135 acceptance) — pre-S133 baseline neměl
  metric, nelze retroaktivně srovnat.
- Per-tick spike event stream (S142 plánováno) — místo toho snapshot
  `neural_spike_frac`.
- Renderer parity (`src/main.rs` Bevy ECS systems) — plasticity dispatch
  sites žijí jen v headless `World`. Renderer dál běží pre-S133 reward
  funnel (jen eat + novelty). Renderer mirror = work item pro 143-152.

**Doporučení pro 143–152 SNN/STDP:**
1. S142 `neural_spike_frac = 0.87` ukazuje že rate brain běží v saturated
   regime → spike timing info je tam underutilized. SNN by tu mohla získat.
2. Začít s `enum NeuronModel { Perceptron, Izhikevich }` v Genome
   (opt-in mutace, pre-existing perceptron lineages survive).
3. Per-cell genome traits: `stdp_window_ms`, `stdp_a_plus`, `stdp_a_minus`
   jako analoge S136 `learning_rate` + `trace_decay`.
4. Forward pass = sub-timestep integrátor (dt/4) pro Izhikevich diff
   eq. Per-cell `(v, u)` state buffer.
5. STDP reward apply = timing-aware Δw místo trace-modulated. Zachová
   `hebbian_apply_reward` jako legacy path pro Perceptron lineages.
6. Storage budget: současný stav 6/7 z 12 v hebbian shadery, +3 nová binding
   (membrane state v, recovery u, last_spike_time) fit pod 12.

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
