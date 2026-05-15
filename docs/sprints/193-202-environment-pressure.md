# Sprinty 193–202: Environment pressure

Předchozí desítka (183–192) konsolidovala **brain stability stack** — weight
decay, LayerNorm, Oja, init jitter. Po S191 brain produkuje *graded
computation* (live dump: 25 distinct hidden activations, smooth output
gradient), ale evolution stále nemá důvod tu výpočetní kapacitu *používat*:
bonded-cluster food-share zůstává viable strategy bez computation, protože
plant density `WORLD_UNITS_PER_FOOD = 6500` (40 % pre-Hunter baseline) je
pořád dostatečně bohatá pro nečinné herbivory.

Decade retro 183–192 explicitně otevřela problém:
> Stack zaručuje, že brain *může* dělat graded computation, ale evolution
> musí mít důvod ji *používat*. Otevřená research question (scarcity,
> mazes, multi-step tasks). Vlastní decade.

**Cíl desítky 193–202:** zavést do simulace **monotonicky rostoucí
environmentální tlak**, který odměňuje smart foraging / koordinaci /
pamatování si zdrojů a postupně eliminuje pasivní cluster-share parazitism
jako stable strategy. Tlak se buduje vrstevnatě:

- **193** — Food scarcity ramp
- **194** — Conservative + sub-linear bonded food share
- **195** — Realistické vousy (mechanický model vibrissae)
- 196+ — dál se rozhoduje podle behavioral CSV signálů; kandidáti:
  MAX_POPULATION cap relax, harder mazes, multi-step coop foods, energy
  budget caps, spatially clustered food, dynamic seasonal shifts.

## Sprint 193 — Food scarcity ramp

**Cíl:** zavést deterministický monotonní pokles globální plant food
density v průběhu prvních 100 generací — vytvořit gradient selekčního
tlaku, který odměňuje efektivnější foraging policy bez nutnosti měnit
genetickou architekturu nebo brain pipeline. Ramp musí mít floor > 0 tak,
aby populace dál mohla přežít efektivním chováním (ne game-over).

**Výstup:**

- **Nové konstanty** v `src/params/physics.rs`:
  - `SCARCITY_RAMP_END_GEN: u64 = 100` — generace, kdy ramp dosedá na floor.
  - `SCARCITY_FLOOR: f32 = 0.5` — multiplikátor v terminálním stavu
    (50 % baseline density = `WORLD_UNITS_PER_FOOD = 13000` effective).

- **Helper** `pub fn scarcity_factor(generation: u64) -> f32` v
  `src/sim/world.rs` vedle `food_target` / `food_multiplier`. Lineární
  interpolace `1.0 → SCARCITY_FLOOR` přes `[0, SCARCITY_RAMP_END_GEN]`,
  za koncem clamp na floor. Pure function generation → factor, žádný
  state.

- **Integrace do `density_factor`**: end-of-generation update v
  `World::tick` (kolem řádku 961) skládá tři multiplikátory dohromady:
  ```rust
  self.density_factor = seasonal * shock_mult * scarcity_factor(gen);
  ```
  Spawn-side `food_target(self.density_factor * self.food_factor_mult)`
  zůstává beze změny — ramp je transparentní pro spawn pipeline,
  checkpoint serialization, CSV diagnostiku (`density_factor` už je
  per-gen sloupec).

**Proč floor 0.5, ne nižší:** pre-S193 plant density 6500 units/food byla
empiricky vyladěná (post-Hunter rebalance) na rovnováhu herbivore +
predator. Half-density (13000 units/food) snižuje food encounter rate
~50 % per cell, ale nutritive value per find zůstává `FOOD_VALUE = 20`.
Při `INITIAL_CELLS = 200` a baseline survival rate to znamená cca
half-as-many feeding events per generation — silný tlak, ale ne
collapse-grade. Tvrdší floor (0.3×) odložen do 194+ podle empirie ze
S193 runu.

**Proč 100 gen, ne pomalejší:** smoke runy projektu se typicky drží na
≤ 30 gen (feedback memory), full validation runy 300–500 gen. 100 gen
ramp znamená:
- gen 0–30 (smoke window): factor klesá `1.0 → 0.7`. Mírný tlak,
  populace by neměla collapse, ale CSV `food_count` / `births_gen` /
  `deaths_gen` by měly začít reagovat.
- gen 100+: factor sedí na 0.5. Stable steady-state pro long-horizon
  validation.

**Testy** (`src/bin/headless/world_tests.rs`):
- `scarcity_factor_at_gen_zero_is_one` — start = 1.0.
- `scarcity_factor_at_ramp_end_hits_floor` — gen 100 = 0.5.
- `scarcity_factor_after_ramp_stays_at_floor` — gen 1000 = 0.5
  (no overshoot below floor).
- `scarcity_factor_midway_is_linear` — gen 50 = 0.75
  (lineární interpolace).
- `scarcity_factor_is_monotonically_non_increasing` — žádný point
  v `[0, 150]` neroste oproti předchozímu.

**Poznámky:** Renderer-side `src/renderer/world_map.rs::food_target` a
`update_food_density_cycle` zůstávají beze změny — od S178 jsou
unscheduled (renderer čte `SimWorld::density_factor` přímo přes
`sim_tick` → `sync_simworld_to_cellentity` pipeline), takže scarcity
factor se do nich propíše automaticky. Setup-time capacity buffer
sizing `food_target(1.0 + CYCLE_AMPLITUDE)` v `setup.rs:292` a
`headless/main.rs:346` zůstává safe upper bound (peak food je vždy
v gen 0, scarcity může count pouze snižovat).

Co se NEŘEŠÍ v tomhle sprintu (otevřené pro 194+):
- **Spatial heterogenity scarcity**: aktuální implementace je globální
  multiplikátor. Spatially patchy scarcity (úrodné enklávy + pouště) by
  vytvořila tlak na exploration / memory, ne jen efektivitu. Vyžaduje
  rozšíření `WorldMap::field` o second channel — vlastní sprint.
- **Cost-side pressure**: ramp škrtí supply, neukrojuje demand
  (`ENERGY_COST_PER_V_SQ`, `VISION_COST_PER_RADIUS`). Symetrický druhý
  pilíř — kandidát na S194 podle CSV signálů.
- **Floor calibration**: 0.5 je educated guess; pokud populace ve full
  300-gen runu zhroutí na 0 nebo naopak ignoruje tlak, retune k 0.4
  nebo 0.6.

## Sprint 194 — Conservative + sub-linear bonded food share

**Cíl:** prolomit bonded-cluster food-share jako dominantní attractor.
S193 5×100-gen sweep ukázal, že scarcity ramp **nezasáhl výzkumný cíl** —
naopak posílil clustering (`bond_active_frac` 0 → 0.93, body size ×2.85,
`carnivore_avg` −70 %, lineages 200 → ~6). Příčina je v mechanice food
share samotné: pre-S194 dostal **každý** bonded partner plný
`value × altruism × state × cluster_mult`, kde `cluster_mult =
1 + (n−1) × cluster_bonus` rostl super-lineárně s velikostí clusteru.
Total energie injektovaná z jednoho foodu tak škálovala **~kvadraticky**
v počtu členů — „přidej se k největšímu clusteru" byl runaway zisk, který
přebil jakoukoli selekci na foraging skill.

**Výstup:**

- **Helper** `pub fn bonded_food_share(value, altruism, state, n_bonds) ->
  f32` v `src/sim/world.rs` vedle `scarcity_factor` / `food_target`. Vrací
  `(value × altruism × state) / n_bonds` (a `0.0` pro `n_bonds == 0`).
  Pure function — testovatelná bez GPU/World.

- **`eat_food` přepojen** (`world.rs` Pass 2 share blok): donor rozdělí
  **fixní pool** `value × altruism × state` rovnoměrně mezi partnery →
  per-partner share je `1/n`. Total energie injektovaná do clusteru je
  teď **nezávislá na jeho velikosti** (pre-S194 rostla super-lineárně).
  Super-lineární `cluster_mult` zrušen, lokální proměnná
  `donor_cluster_bonus` odstraněna.

- **`cluster_share_bonus` per-genome trait → vestigiální.** Mechanika ho
  už nečte. Field zůstává v genomu (serializace, mutace, crossover) +
  CSV sloupec `cluster_bonus_avg` — kvůli genome-schema / CSV stabilitě
  (full removal je follow-up cleanup). Doc komentáře v `genome.rs` +
  `lib.rs` (`BOND_FOOD_SHARE_CLUSTER_BONUS`) označeny.

- **Testy** (`src/bin/headless/world_tests.rs`):
  - `bonded_food_share_zero_bonds_is_zero` — `n=0 → 0`.
  - `bonded_food_share_dilutes_with_cluster_size` — per-partner striktně
    klesá v `n` (sub-lineární).
  - `bonded_food_share_total_injected_is_cluster_size_invariant` —
    `n × per-partner == pool` pro `n ∈ [1,6]` (konzervativní property).
  - `bonded_food_share_scales_linearly_with_altruism_and_state` —
    lineární v `altruism` i `state`, nula při `altruism = 0`.

**Sémantika změny:** pre-S194 byl food share **super-lineární odměna za
velikost** (větší cluster = kvadraticky víc free energy per food). Post-S194
je **konzervativní + sub-lineární** — donor pořád vytváří energii z ničeho
(`pool` není odečten z eateru, „tissue cooperation" sémantika zůstává), ale
ten pool je *fixní* a *ředí se* počtem partnerů. Velký cluster teď znamená
**méně per hlava**, ne víc. `altruism_share_frac` zůstává plně pod selekcí —
mění se jen jeho geometrie vůči `n_bonds`.

**Poznámky:** Toto je *jediný* GPU-free CPU share blok (`eat_food` Pass 2
je sekvenční pro oba paths — headless `--gpu-full` i renderer); GPU dělá
jen candidate selection, ne share resolution. Jeden edit pokrývá obě
binárky.

Pre-existing test fail opraven mimo scope S194: `populate_brain_inputs_
writes_temperature_slot` (`tests.rs`) očekával přesný `tanh`, ale slot 20
v `sensors.rs` byl v commitu `b81c0e2` přepnut na `tanh_fast_scalar`.
Test sladěn s implementací (expected hodnoty derivované z `tanh_fast_scalar`
+ THERMAL konstant místo hardcoded literálu).

Validace (5×100-gen sweep S194 vs S193 baseline) — **pending**. Klíčová
metrika: spadl `bond_active_frac` z 0.93? Sekundární: vrátila se diverzita
(lineages, body size, carnivore_avg, vision)?

Co se NEŘEŠÍ v S194 (otevřené pro 195+):
- **MAX_POPULATION cap (1500)**: S193 analýza ukázala, že populace narazí
  na cap kolem gen 25 a scarcity tím nikdy „nebolí" na populační úrovni.
  Sub-lineární share oslabí cluster zisk, ale dokud cap drží populaci nad
  food-limited steady state, plný selekční gradient se neprojeví. Cap
  relax je nejsilnější zbývající páka — kandidát na S196+.
- **Non-conservation**: share pool je pořád vytvořen z ničeho (eater si
  nechá plný `value`). Plně konzervativní varianta (split eaten value
  mezi eatera a partnery) je větší behavioral změna — odložena.
- **`cluster_share_bonus` full removal**: vestigiální trait + CSV sloupec
  čeká na cleanup sprint (genome-schema breaking change).

## Sprint 195 — Realistické vousy (mechanický model vibrissae)

Vousy (6-směrový proximity raycast proti maze `ObstacleField`) dosud hlásily
mozku **okamžitou** raycast vzdálenost — zeď, která se objeví/zmizí, překlopí
signál během jednoho ticku. Reálné vibrissae jsou pružné násadce: mají
výchylku a rychlost, blízká zeď je ohne, ony překmitnou, rozkmitají se a
doznívají. S195 zavádí per-vous **tlumený harmonický oscilátor 2. řádu**
(semi-implicitní Euler, `dt = 1/60`) + deterministický transdukční šum.
Uživatel explicitně zvolil model 2. řádu (pružina-tlumič) místo levnějšího
asymetrického filtru 1. řádu.

**Cíl:** vousy se chovají jako fyzické vibrissae — výchylka, překmit,
doznívání ~0.2–0.5 s; zachovat CPU/GPU paritu (jádrová strukturální záruka
projektu) novým paritním testem, protože vousy dosud žádné parity coverage
neměly.

**Výstup:**

- **Nové konstanty** v `src/params/maze.rs`: `WHISKER_STIFFNESS = 360.0`,
  `WHISKER_DAMPING = 11.0`, `WHISKER_NOISE_AMPLITUDE = 0.03`. Zrcadlené jako
  hardcoded literály v `shaders/sensor_gather.wgsl`.

- **Stateless hash** `pub fn whisker_noise(cell_index, tick, whisker_k) -> f32`
  v `src/obstacles.rs` (PCG-style, jen `wrapping_*` ops) + byte-identické WGSL
  zrcadlo v `sensor_gather.wgsl`. Žádný RNG buffer, žádný reproduction reset —
  noise je čistá funkce indexu buňky, ticku a vousu.

- **Integrátor** `pub fn whisker_step(deflection, velocity, raw, noise) -> f32`
  v `src/obstacles.rs` — jeden semi-implicitní Euler krok pružiny-tlumiče.
  Sdílený mezi CPU `World::update_whiskers` a paritním testem; zrcadlí
  per-vous tělo v `sensor_gather.wgsl`.

- **Nová `Cell` pole** `whisker_deflection`, `whisker_deflection_vel`
  (`[f32; WHISKER_COUNT]`, `#[serde(default)]`). `last_whisker_distances`
  přeznačeno — drží teď *sensed* hodnotu (`clamp(1 − deflection) + noise`),
  ne raw raycast.

- **Perzistentní GPU buffer** `whisker_state_buf` v `src/gpu/cells.rs` (12
  f32/buňku, layout `[deflection×6, velocity×6]`) — alokace + zero-fill +
  accessor + reset v `reset_persistent_brain_state_at` (recyklovaný slot
  nedědí kmitající vousy předchozího nájemníka). Binding 18 (`read_write`)
  v `sensor_gather.wgsl`, plumbing v `src/gpu/sensor_gather.rs`
  (`SensorParamsGpu.tick` + padding na 80 B, bind group 0..19, dispatch
  signatury). Spring-damper běh složen přímo do `sensor_gather` entry pointu
  za raycast blok, gated na `maze_active` aby non-maze runy držely přesně
  neutrální 1.0.

- **CPU integrátor** v `World::update_whiskers` (`src/sim/world.rs`):
  `par_iter_mut().enumerate()` (index `i` seeduje noise, musí se shodovat
  s GPU dispatch indexem), raycast → `whisker_step` per vous → `sensed`
  do `last_whisker_distances`.

- **Renderer overlay** (`K`) ukazuje filtrovaný/kmitající vous zadarmo —
  čte `last_whisker_distances`, které teď drží `sensed`.

- **Paritní test** `whisker_spring_damper_gpu_matches_cpu` v
  `src/tests_phase3.rs` (vůbec první whisker parity coverage): fixní maze +
  8 buněk `heading = 0` (raycast pak bit-identický CPU/GPU, test izoluje
  spring-damper + noise aritmetiku), 120 ticků, assert `sensed` per tick a
  finální `deflection`/`velocity` v toleranci.

**Poznámky:** Layout mozkových vstupů beze změny — 6 slotů `[33..38]`,
mapování `*2−1`. Mění se jen *sémantika hodnoty* (filtrovaný + kmitající +
zašuměný signál místo okamžitého), což je evoluční perturbace: vyvinuté
mozky uvidí zpožděný/kmitající vstup a selekce přeladí během několika
generací. Defaultní konstanty jsou empirické odhady — ladit přes `K` overlay
+ headless smoke run. Staré checkpointy se načtou s vynulovaným mechanickým
stavem (`#[serde(default)]`) a dokonvergují za ~0.5 s. ±z vousy projdou
filtrem uniformně (`raw = 1` → `target = 0` → stav zůstává v klidu).
Non-maze runy jsou bit-identické s pre-S195 (GPU gate na `maze_active`,
CPU `update_whiskers` early-return — `last_whisker_distances` zůstává
`[1.0; 6]`).

Smoke validace: 25-gen headless `--maze medium` (seed 1) doběhl bez
panic/NaN. Extinction gen 21 vs. pre-S195 baseline gen 15 na stejném seedu
— maze mód je tvrdý na gen-0 random brains nezávisle na vousech; filtrovaný
signál collapse nezhoršil.

## Sprint 196 — Endosymbiosis plumbing

**Cíl:** první slice endosymbiotické větve — zavést data + vertikální dědičnost + predací řízený origin pathway, **bez** energy mechaniky. Cílem je ověřit, že symbiont jako passenger genome se v populaci šíří, dědí a transferuje deterministicky, než Sprint 197 přidá fotosyntézu a conditional upkeep. Tier-1 designové rozhodnutí: plný druhý `Genome` (max evolvability, oproti mini-parametric variantě), origin z predačního "failed digestion" eventu, vertical inheritance s `P_inherit < 1.0` (3 loss channels: host death + transmission failure + attacker-already-bears predation skip).

**Výstup:**

- **Nové konstanty** v `src/lib.rs`:
  - `SYMBIONT_INIT_FRACTION = 0.10` — gen-0 bearer pool
  - `SYMBIONT_INHERIT_P = 0.95` — vertical transmission
  - `SYMBIONT_CAPTURE_P = 0.005` — predace-derived origin rate

- **Data layer** (`src/cell.rs`): nový `Symbiont { genome: Genome, lineage_id: u64, age: u64 }` se `#[derive(Copy, Clone, Serialize, Deserialize)]` (Genome už Copy z S187), a nové pole `Cell.symbiont: Option<Symbiont>` se `#[serde(default)]` pro forward-compat starých checkpointů.

- **Sim layer** (`src/sim/world.rs`):
  - `World.next_symbiont_lineage_id: u64` (counter nezávislý na `next_cell_id` — symbiontova identita přežívá predation transfer napříč hosty)
  - Post-init pass v `new_with_maze` rolluje `SYMBIONT_INIT_FRACTION` per buňku a vytváří fresh random `Genome` přes `Genome::random(rng)`
  - Checkpoint loader re-derivuje `next_symbiont_lineage_id` z `max(c.symbiont.lineage_id) + 1`
  - V `predate()` po swarm/pack diagnostice: pro každý `(victim, attacker)` hit z `result.victim_attackers`, když victim bearer a attacker není, s `P_capture` attacker kopíruje (victim si symbionta nechává — Copy semantika). `predate` má teď `&mut Rng` parametr.

- **Reprodukce** (`src/reproduction.rs`): v `make_mating_child_no_brain` před `Cell { ... }` literálem vzniká `child_symbiont: Option<Symbiont>` match-em přes `(parent_a.symbiont, parent_b.symbiont)` — `(None, None)` skip, single-bearer uses that one, both-bearer uniform pick. RNG draws jsou gated tak, že reprodukce v symbiont-free populaci nekonzumuje žádné nové draws (pre-Sprint-196 RNG sequence preserved pro non-bearer kohorty). Symbiontův genom mutuje přes `MUTATION_CONFIG.mutate_no_brain` paralelně s host genomem.

- **Per-tick age** (`src/cell.rs`): obě `step_with_thermal*` zvyšují `s.age` paralelně s `self.age`.

- **CSV** (`src/bin/headless/csv.rs`): 3 nové sloupce na konci řádku — `sym_count` (u64), `sym_fraction` (f64), `sym_lineage_count` (u64). Empty-pop variant taky upraven (zero-padded).

- **JSON dump** (`src/json_export.rs`): no-op — `serde_json::to_value(cell)` přes derived `Serialize` automaticky pokryje nové pole.

**Poznámky:**

- **GPU buffer plumbing odložen** na Sprint 197. Sprint 196 schválně nemodifikuje žádný shader — symbiont je čistě CPU-side passive data a žádný GPU compute pass ho nečte ani neuploaduje. Cell-side fields jsou bytemuck-extracted do existujících GPU bufferů (positions, headings, atd.), takže layout change Cell struct se na GPU layer neprojevila.

- **Renderer marker odložen** na Sprint 197 vedle samotné energy mechaniky. CSV dává plnou observabilitu (`sym_fraction` per gen). Visual diff lze sledovat skrz JSON dump (`--dump-dir` → human-readable per-cell dump obsahuje plný `symbiont` field včetně lineage_id).

- **Determinismus**: pre-Sprint-196 RNG sequence je broken v rámci tohoto sprintu schvalně (`SYMBIONT_INIT_FRACTION > 0` přidává draw per init cell; inheritance přidává draws při bearer reprodukci). Validation pro Sprint 196 spočívá v cross-seed sweepu (`feedback_validation_sweep`) sledování `sym_fraction` trajektorie — bez energy mechaniky očekáváme drift (3 loss channels minus zero gain channels), což je správné chování pro plumbing-only.

- **Risk callout**: 3 loss channels combined s low `P_capture` mohou erodovat populaci symbiona před Sprint 197 nasadí fotosyntézu. To je akceptovatelné — bearer fraction se v sprintu 196 nemusí stabilizovat, jen plumbing musí fungovat. Pokud bearer_fraction → 0 do 50 generací, Sprint 197 je tlačený.

- **Symbiont struct cost**: per-cell paměť ~2× (plný druhý Genome + 16 B metadata). Akceptovatelný trade-off za max evolvability. GPU paměť beze změny (symbiont nezůstává v shader paths).

## Sprint 197 — Endosymbiosis: photosynthesis + upkeep

**Cíl:** druhý slice endosymbiotické větve — aktivovat energy mechaniku, která dělá z bearer-fraction niche-conditional dynamiku. Surface hosts (upper z-band) získávají z symbionta čistý zisk přes "fotosyntézu"; hluboký pás platí jen upkeep cost a po prolonged deficitu host shazuje. Cíl je vidět vertikální stratifikaci v `sym_z_avg` CSV sloupci a sustainable bearer_fraction (oproti pure-drift v S196).

**Výstup:**

- **Nové konstanty** (`src/lib.rs`):
  - `SYMBIONT_PHOTO_RATE = 0.6` (per-sec gain na vrcholu světa)
  - `SYMBIONT_PHOTO_Z_THRESHOLD = 0.5` (jen horní polovina světa svítí)
  - `SYMBIONT_UPKEEP_PER_SEC = 0.15` (konstatní drain per bearer)
  - `SYMBIONT_UPKEEP_DEFICIT_TICKS = 600` (~10 s @ 60 Hz před shedding)

- **Symbiont struct** (`src/cell.rs`): nové pole `deficit_streak: u32` s `#[serde(default)]` (forward-compat s S196 checkpointy). Reset na 0 při origin (init, inheritance, predation transfer).

- **World** (`src/sim/world.rs`):
  - Nové pole `World.sym_sheds_gen: u64` (per-gen counter — kolik symbiontů zaniklo deficit-shedding)
  - Nová metoda `pub fn apply_symbiont_energy(&mut self, dt: f32)`:
    - `z_norm = (cell.z + half_z) / world_z_total` v `[0, 1]`
    - `light = max(0, z_norm − threshold) / (1 − threshold)` — lineární attenuation nad prahem
    - `photo_gain = PHOTO_RATE × light × metabolism_factor(T) × dt` — temperature-scaled (cold cells get less, just like other energy costs)
    - `upkeep = UPKEEP_PER_SEC × dt` — fixní per-tick drain
    - `cell.energy += (photo_gain − upkeep)`
    - Negativní net → `deficit_streak += 1`; positivní → reset na 0
    - `deficit_streak > THRESHOLD` → `cell.symbiont = None`, `sym_sheds_gen += 1`
  - Volání z `tick()` hned po `step` (po host energy costs, před predate)
  - 2D mode (`WORLD_HALF[2] == 0`) skipuje celou metodu — non-3D smokes zůstávají byte-identical

- **CSV** (`src/bin/headless/csv.rs`): 3 nové sloupce na konci řádky:
  - `sym_z_avg` (f64, 4 dec) — mean normalized z bearerů (stratification proxy; > 0.5 = surface-leaning)
  - `sym_deficit_avg` (f64, 2 dec) — mean `deficit_streak` přes bearers (stress proxy; > 0 = pod tlakem)
  - `sym_sheds` (u64) — `world.sym_sheds_gen`
  - Empty-pop variant dostává `0.0000,0.00,0`.

- **Per-gen reset** (`src/renderer/systems/sim_tick.rs` + `src/bin/headless/main.rs`): `world.sym_sheds_gen = 0;` u ostatních gen-counter resetů.

- **Tests** (`src/tests.rs`): 5 nových testů. `gains_at_top` (positive net + streak=0), `drains_at_bottom_streak_increments` (negative net + streak=1), `sheds_after_deficit_threshold` (po `THRESHOLD+1` ticks symbiont=None + sym_sheds_gen=1), `skips_non_bearers` (no-op pro hosty bez symbionta), `streak_resets_on_positive_tick` (deficit → top → streak=0).

**Poznámky:**

- **CPU-only mechanika** schválně — žádný shader symbionta nečte. Volání z `tick()` po GPU dispatch sérii. Kdyby fotosyntéza měla per-tick perf dopad (1000 cells × ~10 ops = trivial), refactor na GPU shader je triviální v Sprint 198+.

- **Tuning otevřený**: `PHOTO_RATE = 0.6` a `UPKEEP_PER_SEC = 0.15` jsou educated guesses. Reálná validace = 5×100-gen cross-seed sweep ([[feedback_validation_sweep]]), sledovat `sym_z_avg` a `sym_fraction` trajektorie. Cíle: (a) `sym_fraction` stabilizuje (≠ kolaps na 0, ≠ fixace na 1), (b) `sym_z_avg > 0.5` (bearers preferentially in upper band), (c) `sym_sheds > 0` v deep-niche generacích (mechanika opravdu shazuje). Pokud `sym_fraction → 0`, snižit upkeep nebo zvýšit photo rate. Pokud `sym_fraction → 1`, opačně.

- **Brain integration odložena** na Sprint 198. Brain zatím "nevidí" zda má symbionta — host nemůže behaviorálně reagovat (např. zaplavat nahoru pro fotosyntézu). To bude další iterace.

- **Renderer marker stále odložen** na Sprint 198+. S197 dělá energy mechaniku CPU-side; CSV (`sym_z_avg`, `sym_deficit_avg`) plus JSON dump (`Symbiont.deficit_streak`) plně pokrývají observability potřeby. Visual layer = bonus pro intuici, lze přidat až mechanika konverguje.

- **Determinismus**: S197 nepřidává RNG draws v tick() — `apply_symbiont_energy` je deterministic given cells state. Cross-seed reproducibility within S197 zachována, pre-S197 sequence broken (host energy trajectory změněn všude, kde běží 3D bearers).

- **Risk callout — niche margin**: parametry `PHOTO_RATE - UPKEEP_PER_SEC` při `light=1` (top of world) dávají net `+0.45/sec`; při `light=0` (anywhere below threshold) dávají `−0.15/sec`. Bearer ve middle band (z_norm=0.75, light=0.5) má `+0.15/sec`. Jsou to malé částky vůči host's body cost (řád 1–10/sec) — symbiont je *modulator*, ne *life-support*. Pokud měl měl výrazně přebíjet host metabolism, evoluce bude ignorovat zbytek mechanik.
