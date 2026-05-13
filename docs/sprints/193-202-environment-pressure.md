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

- **193** — Food scarcity ramp (this sprint)
- 194+ — dál se rozhoduje až podle behavioral CSV signálů ze S193 runu;
  kandidáti: harder mazes, multi-step coop foods, energy budget caps,
  spatially clustered food, dynamic seasonal shifts.

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
