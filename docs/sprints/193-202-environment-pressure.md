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
