# Sprinty 011–020 — Ecology

Druhá desítka: vizuální differentiation buněk → predace + carrion ekologie → environmental cycles → cognition (hidden layer brain). Žijeme nad evoluční mašinerií ze Sprintu 010, přidáváme bohatší interakce a měřitelnost.

## Sprint 11 — shape-and-size

- **Cíl:** odlišit buňky vizuálně mimo barvu — **(A)** tvar buňky se otáčí podle `heading` (jasně vidět "kam míří"), **(B)** nový gen `body_size` škáluje vizuální velikost (vidět "kdo je kdo" / lineage drift).

  **A) Heading-aware mesh — kapka:**
  - `Circle::new(CELL_RADIUS)` → custom **teardrop** mesh přes parametrickou křivku `x = R·cos t, y = R·sin t · sin(t/2)`. Tip ostrý v +x (t=0), bulb kulatý v −x (t=π). Pomocná funkce `teardrop_mesh(radius)` v `main.rs` postaví mesh přes triangle fan z 24 outline points.
  - Rotace: `Transform::with_rotation(Quat::from_rotation_z(cell.heading))` při spawnu, `sync_transforms` updatuje rotaci spolu s pozicí každý frame. Tip míří kam kapka jede — vizuálně organické "plave/svírá".

  **B) `body_size` gen v `lib.rs`:**
  - `Genome.body_size: f32` (init `0.7..1.3`, `MIN_BODY_SIZE = 0.3`, mutuje s `sigma_body_size = 0.05`). `MutationConfig` doplněna o nový sigma.
  - Aplikace: `Transform::with_scale(Vec3::splat(genome.body_size))` při spawnu v `setup` i `cell_reproduces_on_threshold`.
  - **Bez metabolic cost** — zatím čistě vizuální drift, jako `color_hue`. Pokud později chceme selekci na velikost, přidá se cost (`× body_size²`) a benefit (např. `EAT_RADIUS × body_size`).

  **Integrace s death fade:**
  - `tick_death_fade` přidá do query `&CellEntity`, čte `body_size`. Scale = `body_size × progress`. Tělo se smršťuje od své skutečné velikosti k 0, ne od 1×.

- **Konstanty:** `MIN_BODY_SIZE = 0.3`, `sigma_body_size = 0.05`.
- **Výstup:**
  - `src/lib.rs`: `Genome.body_size` field (init 0.7..1.3, `MIN_BODY_SIZE = 0.3`), `MutationConfig.sigma_body_size`. `dummy_genome` a tests aktualizovány. 10/10 testů průchozí.
  - `src/main.rs`: `MUTATION_CONFIG.sigma_body_size = 0.05`. `cell_mesh` přepnuto z `Circle::new(CELL_RADIUS)` na custom **teardrop** mesh přes pomocnou `teardrop_mesh(radius)` (parametrická křivka `R·cos t, R·sin t · sin(t/2)`, triangle fan). `setup` i `cell_reproduces_on_threshold` spawne s `Transform::with_rotation(Quat::from_rotation_z(cell.heading)).with_scale(Vec3::splat(cell.genome.body_size))`. `sync_transforms` updatuje rotation z `heading` per frame. `tick_death_fade` přidá `&CellEntity` do query, `transform.scale = body_size × progress`.
- **Poznámky:**
  - Teardrop je asymetrický podél heading osy (tip ostrý, bulb kulatý). Mesh je centered v Transform.translation, ale geometrické těžiště kapky je posunuté o ~0.1 R směrem k bulbu (širší konec má víc plochy). Pro kolize/vision negligible. Bounding box: x ∈ [−R, R], y ∈ [−0.75 R, 0.75 R] — perfektně sedí na `2 × CELL_RADIUS` collision distance.
  - `body_size` jako pure-drift gen — bez horního capu. V principu se může rozdrift až moc, ale Gaussian drift je symetrický, nedojde k runaway. Pokud uvidíme cells obrovské nebo téměř neviditelné, přidáme `MAX_BODY_SIZE` clamp.
  - `sync_transforms` teď updatuje i rotaci. Quat::from_rotation_z je triviální op, výkonový dopad nulový.
  - Stats panel: `size_avg` zatím nepřidávám — pure-drift neukáže nic zajímavého. Až `body_size` dostane biologickou roli, přidá se.

## Sprint 12 — intraspecies-predation

- **Cíl:** `body_size` se stane biologicky aktivním. Větší buňky drainují energii menších při kontaktu — single species, ale **divergentní strategie predátor / kořist emergují z brain evoluce**. Plus přidat metabolic cost na velikost a brain input pro relativní velikost.

  **A) Body_size dostane biologickou roli:**
  - **Metabolic cost:** v `Cell::step` přidat `energy -= body_size² × BODY_COST_FACTOR × dt`. Větší buňky dražší per tick. `Cell::step` má teď 5 parametrů.
  - **Eating reach:** v `cell_eats_food` `eat_r = EAT_RADIUS × body_size`. Větší tělo = větší dosah na jídlo.
  - Visual scale už existuje (Sprint 11) — beze změny.

  **B) Intraspecies predation:**
  - Nový systém `cell_predates_on_neighbor` v `FixedUpdate` po `resolve_cell_collisions`.
  - Pro každý pair v `2 × CELL_RADIUS` rozsahu: pokud `attacker.body_size > SIZE_RATIO_THRESHOLD × victim.body_size`, **attacker drainuje energii**.
  - `PREDATION_DRAIN_PER_TICK = 3.0` (victim ztrácí), `PREDATION_GAIN_PER_TICK = 1.5` (attacker dostává — 50 % efficiency, zbytek "ztrátá v procesu").
  - Při sustained contact (10–30 ticků) victim energy klesne k 0, mrtvola se stane carrion (existující systém).

  **C) Brain input pro relativní velikost:**
  - `BRAIN_INPUTS: 6 → 7`. Vstup `[6]: (other.body_size - my.body_size) / my.body_size` pro nejbližší jinou buňku v dohledu. Pozitivní = ohrožení (bigger), negativní = příležitost (smaller). Pokud nikdo v dohledu, 0.
  - Brain teď může evolvovat asymetrické chování: utíkat před větším, honit menšího.

  **Constants:**
  - `BODY_COST_FACTOR = 0.5` — body_size² × 0.5/sec metabolic drain.
  - `SIZE_RATIO_THRESHOLD = 1.3` — attacker musí být 30 % větší.
  - `PREDATION_DRAIN_PER_TICK = 3.0`, `PREDATION_GAIN_PER_TICK = 1.5`.

  **FixedUpdate ordering:**
  ```
  advance_clock, rebuild_food_grid, cells_brain_act, step_cells,
  rebuild_cell_grid, resolve_cell_collisions, cell_predates_on_neighbor,
  cell_eats_food, spawn_food, cell_reproduces_on_threshold,
  cell_dies_on_zero_energy, tick_death_fade
  ```

- **Výstup:**
  - `src/lib.rs`: `BRAIN_INPUTS: 6 → 7`. `Cell::step` má 5. parametr `body_cost_factor` (drain `body_size² × factor × dt`). Testy aktualizované — 10/10 průchozí.
  - `src/main.rs`: konstanty `BODY_COST_FACTOR`, `SIZE_RATIO_THRESHOLD`, `PREDATION_DRAIN_PER_TICK`, `PREDATION_GAIN_PER_TICK`. `step_cells` předává `BODY_COST_FACTOR`. `cell_eats_food` má `eat_r = EAT_RADIUS × body_size`. `cells_brain_act` snapshotne `body_sizes: HashMap<Entity, f32>` a počítá vstup `[6] = (other_size − my_size) / my_size` pro nejbližší jinou buňku.
  - **Nový systém `cell_predates_on_neighbor`** v `FixedUpdate` po `resolve_cell_collisions`: pro každý pair v `2 × CELL_RADIUS` rozsahu, pokud `attacker.body_size > SIZE_RATIO_THRESHOLD × victim.body_size`, drainuje energii (3.0/tick z victim, 1.5/tick attackerovi). Použivá `CellGrid` pro broad-phase, narrow-phase přes `body_sizes` snapshot a aktuální positions.
- **Poznámky:**
  - Tradeoff body_size: **bigger = predátorská výhoda + větší dosah jídla**, ale **dražší metabolic + horší fission overhead** (po fission má child energy/2 ale stejný metabolic load). Empirické optimum bude záviset na hustotě populace, hustotě jídla, predation rate.
  - Predation se odehrává ve stejném rozsahu jako kolize. Pair je processed 1× per tick (asymetrický filtr `attacker > threshold × victim`).
  - Bez `EAT_RADIUS × body_size` clampu na max — pokud body_size 5×, eat_radius 40. Akceptujeme pro experimentaci, can clamp v budoucnu.
  - 50 % efficiency drainu znamená že predace **není zero-sum** — energie se ztrácí v procesu (jako tepelné ztráty v real biology). Pyramid scheme se neuzavře, energy musí stále téct přes food.
  - Brain teď má 7 inputs (8 vah na output × 2 outputs = 14 → 16 vah, +2 biases = 18 parametrů). Drobný růst, žádné perf concerns.
  - Carrion ze zabitých buněk vrátí část energie zpět do food poolu — predace nesnižuje celkové dostupné kalorie.

## Sprint 13 — balance-pass

- **Cíl:** po Sprintech 10–12 populace narážela na `MAX_POPULATION = 1000` cap kolem gen 14 — carrying capacity byla vyšší než cap. Snížit zdroje a zvýšit metabolic, aby přirozená rovnováha klesla pod cap (cíl ~400–700). Současně přidat `size_avg` / `size_dev` do stats panelu (Sprint 12 učinil `body_size` biologicky aktivní, ale neměříme to).
  - **Rebalance konstanty:**
    - `WORLD_UNITS_PER_FOOD: 2000 → 3000` (food density −33 %)
    - `ENERGY_COST_PER_DISTANCE: 0.05 → 0.07`
    - `BODY_COST_FACTOR: 0.5 → 0.8` (selekce na body_size silnější)
    - `VISION_COST_PER_RADIUS` zůstává 0.02 (vize je evolučně cenná, nechci ji brzdit)
  - **Stats panel:** dva nové řádky `size_avg` a `size_dev`.
- **Výstup:**
  - `src/main.rs`: `WORLD_UNITS_PER_FOOD: 2000 → 3000`, `ENERGY_COST_PER_DISTANCE: 0.05 → 0.07`, `BODY_COST_FACTOR: 0.5 → 0.8`. `update_stats_overlay` rozšířen o `body_size` v single-pass akumulaci (size_sum + size_sumsq), výstup obsahuje `size_avg` a `size_dev` jako 13./14. řádek (15 řádků celkem). Label sloupec roztažen na 8 znaků pro `size_avg`/`size_dev` čitelnost.
- **Poznámky:**
  - Empirické cílení rovnováhy. Pokud se populace stále drží na cap, dál snížit `WORLD_UNITS_PER_FOOD` níž nebo zvýšit costs. Pokud zhroutí na 0, reverse tune.
  - Sprint 12 měl body_size² × 0.5/sec — typický cell s `body_size = 1.0` platil 0.5/sec = 5/gen, `body_size = 1.5` platil 11.25/gen. S 0.8 faktorem (= 8/gen vs 18/gen) se rozdíl bigger vs smaller stává výraznější — selekce body_size dostane víc šťávy.
  - `size_dev` je dobrý ukazatel "stratifikuje se populace?" — pokud predátoři vs kořist diverguje, body_size distribuce se rozpůlí na velké a malé, `size_dev` stoupne. Pokud stagnuje na 0.1ish (~ initial range 0.6 width / sqrt(12)), divergence neprobíhá.

## Sprint 14 — brain-and-cycles

- **Cíl:** **(A)** rozšířit `Brain` o **hidden vrstvu** — single-layer perceptron je strop pro complex chování, nedokáže non-linear logiku ("blízko jídlo + nízká energie + bezpečno → útok"). **(C)** přidat **environmental cycles** — `WORLD_UNITS_PER_FOOD` periodicky kolísá podle generation count (zima/léto), populace musí adaptovat na fluktuaci.

  **A) Hidden layer brain:**
  - V `lib.rs` `Brain` rozšířen na 2-layer MLP: `w1: [[f32; BRAIN_INPUTS]; BRAIN_HIDDEN]`, `b1: [f32; BRAIN_HIDDEN]`, `w2: [[f32; BRAIN_HIDDEN]; BRAIN_OUTPUTS]`, `b2: [f32; BRAIN_OUTPUTS]`.
  - `BRAIN_HIDDEN: usize = 8`.
  - Forward: `tanh(W2 · tanh(W1 · x + b1) + b2)`.
  - Random a mutate iterují přes obě vrstvy stejně — jeden `sigma_brain` pro všechny váhy/biasy.
  - Param count: 7×8 + 8 + 8×2 + 2 = **82 parametrů** (z 16 single-layer). 5× větší, ale per-cell forward stále ~100 ops, výkonově triviální.
  - **Breaking change pro brain weights** — existující populace by neseděla. Začínáme z čista (random init).

  **C) Environmental cycles (food density oscillation):**
  - Resource `FoodDensityFactor(f32)` — multiplier nad baseline `WORLD_UNITS_PER_FOOD`. Default 1.0.
  - Systém `update_food_density_cycle` v `FixedUpdate` (po `advance_clock`, čte `GenerationEnded` event): `factor = 1.0 + CYCLE_AMPLITUDE × sin(2π × generation / CYCLE_GEN_PERIOD)`.
  - `food_target(extent, factor)` aplikuje factor — efektivně `area / WORLD_UNITS_PER_FOOD × factor` cílů. `setup` i `spawn_food` čtou factor.
  - Konstanty: `CYCLE_GEN_PERIOD: u64 = 50` (perioda 50 generací = ~8 minut sim-time při 1×, ~5 sek při 100×), `CYCLE_AMPLITUDE: f32 = 0.4` (0.6× až 1.4× baseline density).
  - Vznikne periodický stres → adaptace, brzdí konvergenci stagnaci. Brain s hidden layer by měl umět "naučit se" reagovat — víc jíst v hojnosti, šetřit v hladu.

  **Stats panel:** přidat řádek `density {:.2}` (current food density factor).

  **FixedUpdate ordering** (po Sprintu 14):
  ```
  advance_clock, update_food_density_cycle, rebuild_food_grid,
  cells_brain_act, step_cells, rebuild_cell_grid,
  resolve_cell_collisions, cell_predates_on_neighbor, cell_eats_food,
  spawn_food, cell_reproduces_on_threshold, cell_dies_on_zero_energy,
  tick_death_fade
  ```

- **Konstanty po změně:** `BRAIN_HIDDEN = 8`, `CYCLE_GEN_PERIOD = 50`, `CYCLE_AMPLITUDE = 0.4`.
- **Výstup:**
  - `src/lib.rs`: `Brain` přepsán na 2-layer MLP (`w1: [[f32; 7]; 8]`, `b1: [f32; 8]`, `w2: [[f32; 8]; 2]`, `b2: [f32; 2]`). `Brain::random/forward/mutate` aktualizováno — forward = `tanh(W2·tanh(W1·x+b1)+b2)`. `pub const BRAIN_HIDDEN: usize = 8`. Test `dummy_brain`, `mutation_with_zero_sigma_is_identity`, a `brain_forward_zero_weights_outputs_tanh_of_output_biases` aktualizovány. **10/10 testů průchozí**.
  - `src/main.rs`: konstanty `CYCLE_GEN_PERIOD = 50`, `CYCLE_AMPLITUDE = 0.4`. Resource `FoodDensityFactor(f32)` (default 1.0) přes `init_resource`. Helper `food_target(extent, factor)` násobí baseline density faktorem. `setup` volá `food_target(&extent, 1.0)`, `spawn_food` čte `Res<FoodDensityFactor>`.
  - **Nový systém `update_food_density_cycle`** v `FixedUpdate` po `advance_clock`, čte `GenerationEnded`: `factor = 1.0 + AMPLITUDE × sin(2π × gen / PERIOD)` — sinová oscillace mezi 0.6× a 1.4× baseline.
  - Stats panel: nový řádek `density {:.2}` (16 řádků celkem).
- **Poznámky:**
  - Hidden layer brain je nejmenší krok od single-layer co dokáže non-linear funkce. `BRAIN_HIDDEN = 8` je heuristika — 4 málo, 16 zbytečně. Tunit empiricky pokud výsledky nejsou znát.
  - Cycle parametry: 50 gen perioda znamená cycle dokončený ~14 minut wall-clock při 1× (50 × 10 s sim time). Pri 100× je cycle za ~8 sekund — vidíš celou oscillaci v krátké session. Pokud chceme delší trvání, posunout.
  - Random init brainů = většina cells úplně chaotic. Selekce filtruje funkční brainy. **Při startu může extinkce přijít rychleji** — populace je hloupější dokud selekce nezafunguje. Pokud uvidíme extinkci v gen 2-3, vrátit se k single-layer nebo doplnit init bias (např. small positive thrust).
  - `update_food_density_cycle` čte `GenerationEnded` event. Pokud event nepřijde, factor se neupdate. To je OK (jen jednou per gen). Ale pokud user změní `TICKS_PER_GENERATION` extrémně velkým, oscillace bude pomalá. Konstanta cycle je v generacích, ne tickách.
  - Test brain forward s hidden layerem: zero w1 + zero w2 + non-zero b2 → output = tanh(b2) (hidden vrstva = tanh(b1), výstup ignoruje hidden při zero w2).

  **Patche po prvním běhu:**
  - **Bug fix — collision/predation radius škáluje s `body_size`:** `resolve_cell_collisions` a `cell_predates_on_neighbor` původně používaly fixní `2 × CELL_RADIUS = 10`, ale vizuální velikost je `CELL_RADIUS × body_size`. Velké buňky (`body_size = 1.5`) se vizuálně překrývaly. Oprava: pair-specific `pair_r = CELL_RADIUS × (size_a + size_b)` v narrow phase, broad-phase grid query za `CELL_RADIUS × (size_a + BROAD_PHASE_SIZE_BUDGET)` (= 3.0 = generous upper bound na "other" body_size). Snapshot `body_sizes: HashMap<Entity, f32>` v `resolve_cell_collisions` (predace ho měla už dřív).
  - **Tuning — `CYCLE_AMPLITUDE: 0.4 → 0.3`:** při amplitudě 0.4 byla scarcity peak na 0.6× baseline → populace v gen ~50 šla do extinkce. 0.3 dává oscillaci 0.7×–1.3×, výrazně mírnější, ale stále znatelný stres pro selekci.

## Sprint 15 — headless-harness

- **Cíl:** **třikrát odložený dluh splacen** — headless binární cíl pro reprodukovatelné experimenty se seedovanou RNG. Konečně lze měřit, jestli předchozí sprinty (zejména Sprint 14 hidden brain) skutečně zlepšily evoluční trajektorii. CSV per-generation log je vstup do externího pipeline (Python/Jupyter) na statistiky a plotting.
  - **Nový binární cíl:** `src/bin/headless.rs` → binárka `headless`. `cargo run --release --bin headless -- [seed] [max_gens] [out_path]`. Auto-detekováno přes `src/bin/*` konvenci, žádný `[[bin]]` v `Cargo.toml`.
  - **Seedovaná RNG:** `rand::rngs::StdRng::seed_from_u64(seed)` (= ChaCha12 deterministic). Stejný seed = identický run. Žádný thread-local nebo OS entropie.
  - **`World` struct** — plain Rust `Vec<Cell>` + `Vec<Food>` + `SimClock` + `density_factor`. Žádné Bevy ECS, žádný spatial grid (Sprint 15 přijímá O(N²) — s N = 200–1000 a bez render overheadu pořád běží řádově rychleji než windowed).
  - **Tick** sekvenčně volá: `brain_act → step → resolve_collisions → predate → eat_food → spawn_food → reproduce → die_and_drop_carrion`. Logika 1:1 s windowed verzí včetně env cyklu na `GenerationEnded`.
  - **CSV format:** `gen, cells, spd_avg, spd_dev, vis_avg, vis_dev, size_avg, size_dev, food, density` — header + 1 řádek per generation. Výstup přes `BufWriter` aby nesral I/O.
  - **Konstanty mirror** windowed verze (kopírujeme z `main.rs` do `headless.rs`). `WORLD_HALF = [960.0, 540.0]` fixně (full HD-equivalent) — bez okenního systému je deterministická extent kritická pro reprodukovatelnost.

- **Výstup:**
  - `src/bin/headless.rs` — kompletní headless harness (~340 řádků). Args: `seed=0`, `max_gens=500`, `out_path=run_seed{N}.csv` defaulty.
  - **Reprodukovatelnost ověřená:** `headless 42 30 a.csv` a `headless 42 30 b.csv` produkují **byte-identical** CSV (`diff` čisté).
  - **Performance:** ~1500 ticks/sec při 200 cells (debug build), ~16k ticks/sec při 35 cells (po extinction prep). Release build na 100 generací ~ 41 s pro 200→583 cells, ~15× rychlejší než windowed (capped na max_delta).
  - **Testy:** `cargo test --lib` ✓ (10/10), `cargo clippy --no-deps` ✓ (čistý), `cargo build` (windowed binárka) ✓.
- **Poznámky:**
  - **První skutečná empirická data:** seed 42, 100 gen — `spd_avg` 60→105, `vis_avg` 51→40, `size_avg` 1.0→2.07. **Predace tlačí body_size nahoru, vize klesá** (méně cenná u velkých predátorů?), populace roste 200→583. **Toto by se dalo bez headless jen tušit.**
  - O(N²) je pro Sprint 15 OK, ale s N → 1000 (max population) se brzdí. Až bude potřeba dlouhé běhy s plnou populací, port spatial grid z `main.rs` do `lib.rs` (sdílená logika) a použít v headless.
  - Konstanty duplicitní mezi `main.rs` a `headless.rs` — pokud se rozjedou, sweep configs přes `--food-density 0.5` apod. by udělal pořádek. Zatím manual sync.
  - Bevy je pořád v `[dependencies]`, takže `cargo build --bin headless` ho kompiluje (i když nepoužívá). Feature-gate Bevy by zrychlil headless-only build, ale to je out of scope.
  - **Co teď s tím:** spustit 30 replikátů × 500 gen, plotnout trajectory bands. Sweep `BRAIN_HIDDEN ∈ {0, 4, 8, 16}` (vyžaduje dočasné přepsání lib const + rebuild) a porovnat. Sweep `CYCLE_AMPLITUDE ∈ {0, 0.2, 0.4, 0.6}` na zjištění optimální fluktuace pro adaptaci.

## Sprint 16 — physics (mass + inertia + drag)

> **Split:** plán níže byl původně "physics-and-fields" pokrývající body 1, 2, 3 user requestu. Po dohodě o split rozděleno na **Sprint 16 = body 1 + 2 (fyzika)** a **Sprint 17 = bod 3 (smell field)**. Force-based dynamics potřebují empirické naladění samostatně — mixovat je s novou senzorickou modalitou by zkomplikovalo debug. Bod 5 (radius scaling) byl už opraven patchem po Sprintu 14.

- **Cíl:** přejít z kinematic modelu (brain dictates velocity přímo) na **rigid-body physics** (brain dictates force) s drag-em. Otevírá:
  - **Velikost vs. agility tradeoff** — velká buňka má hmotnost a moment setrvačnosti, pomalý rozjezd a neohrabané otáčení. Real biological constraint, ne ohýbatelný brainem.
  - **Realistic max speed strop daný drag-em** — žádný free perpetual motion. Energie z `v²`, ne `|distance|`.
  - **Search behavior přes chemické gradienty** — buňka "čichá" jídlo i mimo direct vision. Vznikají taxis (chemo-tropie), search trajectories, paměťové stopy v poli.
  - Pokrývá body 1, 2, 3 z user requestu. **Bod 5 (radius scaling) byl už opraven patchem po Sprintu 14** — `resolve_cell_collisions` a `cell_predates_on_neighbor` používají `pair_r = CELL_RADIUS × (size_a + size_b)` v narrow phase + `BROAD_PHASE_SIZE_BUDGET` v grid query. Sprint 16 jen formálně uzná, že fix je v place.

  **A) Mass + setrvačnost (rigid-body):**
  - V `lib.rs` `Cell` rozšířen o `pub angular_velocity: f32` (radiánů/sec). Heading drží stav skrz ticky, ne re-derivovaný z velocity.
  - **Mass = `body_size²`** (= 2D plocha pro standard density). **Moment of inertia = `body_size²`** (zjednodušení; striktně pro 2D disk je MoI ∝ m·r² = body_size⁴, ale to penalizuje velikost moc agresivně — kvadratický scaling dá rozumnější tradeoff).
  - **Brain output reinterpretace:**
    - `turn_signal ∈ [-1, 1]` → torque, angular_acceleration = `turn_signal × turn_rate / body_size²`.
    - `thrust_signal ∈ [-1, 1]` → linear force, linear_acceleration = `thrust_signal × DRAG × max_speed² × heading_unit / body_size²`. Vzorec si volíme tak, že **`max_speed` gene = terminal velocity** při full thrust (force = drag).
  - Per tick:
    ```
    angular_acceleration = (turn_signal × turn_rate) / body_size²
    angular_velocity     += angular_acceleration × dt
    angular_velocity     *= (1.0 - ANGULAR_DRAG × dt)
    heading              += angular_velocity × dt

    thrust_force_mag     = thrust_signal × DRAG_COEFFICIENT × max_speed² × body_size²
    velocity             += (thrust_force_mag / body_size²) × heading_unit × dt
    velocity             -= DRAG_COEFFICIENT × |v| × v × dt        (kvadratický drag)
    position             += velocity × dt
    ```
  - **Selekční důsledky:** velká buňka se rozjíždí pomaleji a otáčí neohrabaněji (dělí se body_size²), přitom drag platí stejnou kvadratickou formu. Tradeoff: velká predátorská výhoda (mass = damage) vs ovladatelnost.

  **B) Drag — kvadratický fluid drag:**
  - `F_drag = -DRAG_COEFFICIENT × |v| × v_vector` (kvadratický scaling, standardní pro nízká až střední Reynolds number).
  - Aplikace již zahrnuta v rovnici výše (řádek `velocity -= DRAG × |v| × v × dt`).
  - **Steady-state v_terminal = `max_speed`** (matematicky odvozeno tím, jak `thrust_force_mag` měřítkujeme).
  - Angular drag analogicky linear: `angular_velocity *= (1 - ANGULAR_DRAG × dt)`. Bez něj by se cell roztočil donekonečna.
  - **Energy cost změna:** dosavadní `energy -= |distance| × ENERGY_COST_PER_DISTANCE` mizí. Místo toho `energy -= |v|² × ENERGY_COST_PER_V_SQ × dt` (power dissipated by drag, fyzikálně odpovídá kinetic-energy loss). `Cell::step` má 5. parametr `energy_cost_per_v_sq` místo `energy_cost_per_distance`.

  **C) Smell field — chemické gradienty:**
  - **Resource `SmellField { grid: Vec<f32>, scratch: Vec<f32>, resolution: usize }`** — 2D grid 128×128 přes celý world. Při full HD `WORLD_HALF = [960, 540]`: cell size = 1920/128 = 15 sim units (širší než cell radius, ale dobrá granularita pro gradient).
  - **Per-tick update v novém systému `update_smell_field` v `FixedUpdate` po `spawn_food` (čerstvé sources) a před `cells_brain_act` (brain potřebuje aktuální gradient):**
    1. **Diffuse:** Jacobi step `new[i,j] = old[i,j] + α × (sum_4_neighbors − 4 × old[i,j])`. `α = 0.15` (stable < 0.25). Použít double-buffer (`grid` + `scratch`), swap po update.
    2. **Decay:** `new[i,j] *= 1 − DECAY × dt`. `DECAY = 0.3 / sec` (≈ 5 sec half-life).
    3. **Sources:** pro každou food entitu: `grid[idx_of(food.position)] += SMELL_PER_FOOD × dt`. Carrion food (z `cell_dies_on_zero_energy`) emituje stejně — žádné rozlišení channel zatím.
  - **Brain inputs:** `BRAIN_INPUTS: 7 → 9`. Index `[7]: smell_grad_x`, `[8]: smell_grad_y` (normalized). Genome breaks compat — start from scratch.
  - **Gradient computation per cell:** central difference, 4-tap sample. Sample field na `pos ± [SAMPLE_EPSILON, 0]` a `pos ± [0, SAMPLE_EPSILON]`, gradient = `(f(pos+ε) − f(pos−ε)) / (2ε)`. Normalize přes `tanh(grad × NORMALIZATION_GAIN)` aby vstup brainu byl v [-1, 1].

  **D) Verifikace bodu 5 (radius scaling):**
  - `resolve_cell_collisions` (line ~836+) **už používá** `pair_r = CELL_RADIUS × (size_a + size_b)` — patch po Sprintu 14.
  - `cell_predates_on_neighbor` (line ~791+) **už používá** stejný pattern.
  - `BROAD_PHASE_SIZE_BUDGET = 3.0` jako safe upper bound na "other" body_size pro grid query.
  - **Žádná akce v Sprintu 16 — jen kontrola, že to v `headless.rs` taky odpovídá**.

- **Konstanty:**
  - Physics: `DRAG_COEFFICIENT ≈ 0.001` (empirické, naladit aby `v_terminal ≈ max_speed` gene), `ANGULAR_DRAG ≈ 2.0/sec` (rychlé tlumení rotace).
  - Energy: `ENERGY_COST_PER_V_SQ ≈ 1e-5` (empirické — cílit na podobnou bilanci jako dosavadní `0.07 × distance`, tj. ~40 energie/gen při speed 60).
  - Smell: `SMELL_GRID_RES = 128`, `SMELL_DIFFUSION = 0.15`, `SMELL_DECAY = 0.3/sec`, `SMELL_PER_FOOD = 1.0`, `SMELL_SAMPLE_EPSILON = 10.0`, `SMELL_NORMALIZATION_GAIN = 0.5`.

- **Lib.rs změny:**
  - `Cell` přidává `pub angular_velocity: f32`.
  - `Cell::step` signature: nahradit `energy_cost_per_distance` parametr za `energy_cost_per_v_sq`. Přidat `drag_coefficient` a `angular_drag` parametry.
  - **Cell::step neaplikuje thrust force / brain logic** — to zůstane v `cells_brain_act` (Bevy systém / headless tick), jen aplikuje drag a position update s nově updaternou velocity. Možná čistší: `Cell::step` aplikuje pouze drag + position + bounce. Brain force application v `cells_brain_act`.
  - `BRAIN_INPUTS: 7 → 9`. Update `dummy_brain` (b1 size, w1 inner array), `mutation_with_zero_sigma_is_identity` literal, `brain_forward_zero_weights_outputs_tanh_of_output_biases` literal.

- **Main.rs + headless mirror:**
  - Resource `SmellField`, system `update_smell_field` v `FixedUpdate` ordering.
  - `cells_brain_act` přepsán: čte gradient z field, aplikuje force-based dynamics.
  - `step_cells` updated signature.
  - **Mirror v `headless.rs`** — same constants, same systems, same `SmellField` (čistě Vec<f32>, žádný Bevy resource).

- **Výstup:**
  - `src/lib.rs`: nový `pub struct PhysicsConfig { drag, angular_drag, energy_cost_per_v_sq, vision_cost_per_radius, body_cost_factor }`. `Cell` rozšířen o `pub angular_velocity: f32` (init 0 v `from_genome`). `Cell::step(dt, world_half, &PhysicsConfig)` — odstraněna 5-parametrová verze, kvadratický linear drag, multiplicative angular drag, energy drain z `v² × cost × dt` (místo linear distance). 12/12 testů průchozí (přibyly `step_applies_quadratic_drag` a `step_applies_angular_drag`).
  - `src/main.rs` + `src/bin/headless.rs`: konstanty `DRAG_COEFFICIENT = 0.005`, `ANGULAR_DRAG = 1.0`, `ENERGY_COST_PER_V_SQ = 0.0008` (empiricky vyladěno — viz poznámky). Const `PHYSICS_CONFIG`. `cells_brain_act` přepsán: brain output → torque (`turn_signal × turn_rate / body_size`) a thrust force (`a_max = DRAG × max_speed² / body_size`). Velocity / angular_velocity jsou integrované, heading update se přesunul do `Cell::step` z `angular_velocity`. Position update v `Cell::step` z velocity. Reproduction Cell literály doplněny `angular_velocity: 0.0`.
  - **Klíčové důsledky:**
    - **v_terminal = max_speed / sqrt(body_size)** — bigger cells mají lower top speed.
    - **Acceleration time** ~3 sec to terminal (DRAG=0.005, max_speed=60).
    - **Angular ang_vel_terminal = turn_rate / (body_size × angular_drag)** — bigger cells turn slower.
    - **Energy drain z v²** místo distance — fluid drag-style metabolic burn.

- **Poznámky:**
  - **Empirické tunění bylo nezbytné.** První run s `ENERGY_COST_PER_V_SQ = 0.001` dal extinkci v gen ~50 (200 → 1 cell). 0.0006 dal opačný extrém (cap 1000 už v gen 60). Finální 0.0008 dává populační dip na ~50 cells během scarcity (gen 37.5), ale recovery a později dosažení cap. Není perfektně stabilní — Sprint 18 / další balance pass může toto vylepšit.
  - **Sprint 17 = smell field** zůstává nedodělaný. Plán napsaný (bod 3 z user requestu), implementace samostatně.

## Sprint 17 — smell field

- **Cíl:** přidat **chemické gradienty** jako novou senzorickou modalitu — bod 3 z user requestu, druhá polovina dříve sloučeného Sprintu 16. Jídlo a carrion vyzařují difuzní pole, buňky čtou gradient přes 2 nové brain inputs. Otevírá search behavior (chemo-tropie), paměť stop, taxis.
  - **`SmellField` v `lib.rs`** — `pub struct SmellField { resolution, world_half, grid: Vec<f32>, scratch: Vec<f32> }`. Metody: `new`, `add_source(pos, amount)`, `step(diffusion, decay_per_sec, dt)` (explicit-Jacobi diffuse + multiplicative decay, double-buffered s `mem::swap`), `sample(pos)`, `gradient_at(pos, epsilon)` (central differences, 4 samples).
  - **`BRAIN_INPUTS: 7 → 9`** — index `[7] = tanh(grad_x × NORMALIZATION_GAIN)`, `[8] = tanh(grad_y × NORMALIZATION_GAIN)`. tanh wrapping bounds inputs do [-1, 1] bez ohledu na magnitude gradientu. Brain weights shape měněna `[[f32; 7]; 8]` → `[[f32; 9]; 8]` automaticky přes `BRAIN_INPUTS` const.
  - **Resource `SmellResource(SmellField)` v `main.rs`** — initialized v `setup` z aktuálního `WorldExtent`. Resize okna není handled (smell field zůstává na initial extent — accept lossy behavior u edges).
  - **Systém `update_smell_field`** v `FixedUpdate` po `rebuild_food_grid`, před `cells_brain_act`. Iteruje `Query<&FoodEntity>`, pro každý food volá `add_source(pos, SMELL_PER_FOOD × dt)`. Pak `smell.step(...)`.
  - **Mirror v `headless.rs`** — `World` rozšířen o `smell: SmellField`. `update_smell` metoda volaná z `tick()` před `brain_act`.

- **Konstanty:** `SMELL_GRID_RES = 128`, `SMELL_DIFFUSION = 0.15` (stable < 0.25 in 2D), `SMELL_DECAY = 0.3 / sec` (~2.3 sec half-life), `SMELL_PER_FOOD = 1.0`, `SMELL_SAMPLE_EPSILON = 10.0`, `SMELL_NORMALIZATION_GAIN = 0.5`.

- **Výstup:**
  - `src/lib.rs`: `pub struct SmellField` + impl. `BRAIN_INPUTS: 7 → 9`. Tests beze změny (`dummy_brain` používá `BRAIN_INPUTS` const).
  - `src/main.rs`: smell konstanty, resource `SmellResource`, systém `update_smell_field` v chainu po `rebuild_food_grid`. `cells_brain_act` má `Res<SmellResource>` parameter, čte gradient na cell pos a fills inputs[7], inputs[8].
  - `src/bin/headless.rs`: stejně, `World.smell: SmellField`, `update_smell` metoda.
  - **Reprodukovatelnost:** seed 42 dává byte-identický CSV i s smell field.
  - **Performance:** 369 ticks/s (release, 200→1000 cells) — z 1442 ticks/s před smell. ~4× zpomalení, drtivá většina v Jacobi step (128² = 16k cells × 60 Hz = 1M ops/sec). Akceptovatelné.

- **Poznámky:**
  - **Pozorovaná evoluce (seed 42, 100 gen):** spd_avg 60→86 (rychlejší než bez smell, kde 79), vis_avg 51→35 (méně klesá než bez smell, kde 26 — smell kompenzuje sníženou vizi), size_avg 1.0→1.33. **Smell pomáhá udržet vyšší vision_radius v selekci** — možná protože vision + smell dohromady umožňují efektivnější food tracking, ale samostatně by bylo třeba jednoho z nich víc.
  - **Sprint 17 NEpřekompiluje stávající brainy** — BRAIN_INPUTS změna mění shape weights. Začíná z čistá.
  - **128² grid = ~64 KB pole**, scratch další 64 KB. Pro full HD okno (2M unit²) je grid cell ~15 sim units. Rozumně jemné pro gradient sample epsilon 10.
  - **Resize okna invaliduje smell field** — pokud user změní velikost okna mid-run, smell zůstává na původní extent. Cells za hranicí grid dostávají smell = 0. Acceptable (rare resize, automatic recovery via diffusion).
  - **Carrion food** přispívá ke smell (každá `FoodEntity` regardless of source). Mrtvoly tedy vyzařují stejně jako rostlinné jídlo. Pokud chceme rozlišit (separate channels for "plant" vs "meat"), Sprint 18+ by mohl přidat second field.
  - **Performance optimization potenciál:** Jacobi step je nejdražší. Snížit `SMELL_GRID_RES` na 64 (4× rychlejší) nebo stride update (každý 2. tick) jsou možnosti. Pro Sprint 17 přijatelné, optimalizace dle potřeby.

  **Patche po Sprintu 17:**
  - **Rotational kinetic energy cost:** `Cell::step` přidá drain `body_size² × ω² × cost_per_v_sq × dt`. Před tímto patchem bylo otáčení free — brain mohl spinovat bez metabolic penalty. Teď je rotace symetrická k translaci (oboje platí kinetic energy proxy). Velké buňky platí 4× víc za stejnou ω — umocněný physics-based agility tradeoff. Test `step_drains_energy_from_rotation` přidán (13/13 testů celkem).
  - **Food anti-overlap spawn:** `spawn_food` (main.rs i headless) rejektuje pozice uvnitř `EAT_RADIUS × body_size` jakékoliv živé buňky. Max `MAX_SPAWN_ATTEMPTS = 5` pokusů per slot; pokud všech 5 narazí, slot se přeskočí. Před tímto patchem se food náhodně materializoval i přímo na buňce → free energy delivery se silným bias k velkým buňkám (větší eat_radius = větší šance být obslužen). Patch odstraňuje bias. V hustých populacích food count může být lehce pod target (saturace), ale to je realistické (v real ecosystem se food těžko hledá místo). Constant `MAX_SPAWN_ATTEMPTS` přidán v main.rs i headless.

## Sprint 18 — lineage-and-learning

- **Cíl:** dvě nezávislé vrstvy najednou — **(A) lineage tracking** dává okno do populační dynamiky (kdo dominuje, jak rychle selekce konverguje), **(B) reward-modulated Hebbian** přidá **lifetime učení** nad genetickou evolucí (dvě časové škály — pomalá evoluce mezi gens, rychlé učení v rámci života).

  **A) Lineage tracking:**
  - `Cell` rozšířen o `lineage_id: u64` a `lineage_birth_gen: u64`. Inherit od rodiče (no mutation).
  - V `setup`: 200 unikátních `lineage_id` (= `i as u64`), `birth_gen = 0`.
  - V `cell_reproduces_on_threshold` / headless `reproduce`: child kopíruje parent's `lineage_id` a `lineage_birth_gen` beze změny.
  - **Vizualizace:** `pack_cell_tag` použije `hash(lineage_id) % 360` jako hue místo `color_hue` genu. Vidět "jezera" stejných barev jak lineages dominují.
  - **Stats:** `lineages` (distinct lineage_ids alive), `oldest` (max age = `current_gen − lineage_birth_gen` přes všechny živé buňky).
  - **color_hue gene** zůstává v Genome ale visual unused — drift pokračuje, ale neviditelně. Future cleanup může gen odebrat (breaking change pro brain weights).

  **B) Reward-modulated Hebbian:**
  - `Cell` rozšířen o `last_inputs: [f32; BRAIN_INPUTS]`, `last_hidden: [f32; BRAIN_HIDDEN]`, `last_outputs: [f32; BRAIN_OUTPUTS]` — recent activation memory pro 1-tick credit assignment.
  - `Brain::forward_with_state(inputs) -> ([hidden], [outputs])` — vrací i hidden activations (vedle existing `forward`).
  - V `cells_brain_act`: po forward pass se store last_inputs/hidden/outputs do `Cell`.
  - V `cell_eats_food`: na eat event trigger `Brain::hebbian_update` s reward = +1.0:
    ```
    Δw1[i][j] = LR × reward × last_inputs[j] × last_hidden[i]
    Δw2[i][j] = LR × reward × last_hidden[j] × last_outputs[i]
    Δb1[i]   = LR × reward × last_hidden[i]
    Δb2[i]   = LR × reward × last_outputs[i]
    ```
  - **Lamarckian inheritance:** child kopíruje parent's CURRENT (po-učení) brain včetně mutace. Learned skills se předávají mezi gens. Zjednodušení proti reálné biologii (kde se neurální skills nepředávají Darwinovsky), ale **fits existing reproduce mechanic** a otevírá rychlou evoluci behaviors (selection má víc material k zachycení).

- **Konstanty:** `LEARNING_RATE = 0.01` (jednotný pro všechny weights/biases).
- **Výstup:**
  - `src/lib.rs`: `Cell` rozšířen o `lineage_id`, `lineage_birth_gen`, `last_inputs`, `last_hidden`, `last_outputs`. `Cell::random` / `from_genome` mají dva nové parametry (lineage info). `Brain::forward_with_state` (vrací hidden + outputs), `Brain::hebbian_update(last_in, last_hid, last_out, reward, lr)` — reinforces last decision pathway. `forward` zachován jako wrapper. Test helper `base_cell()` pro struct-update syntax v testech. 15/15 testů (přibyly 2 Hebbian).
  - `src/main.rs`: `LEARNING_RATE = 0.01`. Helper `lineage_hue(id)` (Knuth integer hash). `setup` přiřadí 200 unikátních `lineage_id`. `cell_reproduces_on_threshold` zachová parent's lineage. `cells_brain_act` volá `forward_with_state`, ukládá `last_*` na cell. `cell_eats_food` triggeruje `hebbian_update(reward=1.0)` na eat. `pack_cell_tag` čte `lineage_hue(cell.lineage_id)` místo `color_hue`. Stats panel: 2 nové řádky `lineages` a `oldest`.
  - `src/bin/headless.rs`: stejný mirror. CSV header rozšířen o `lineages` a `oldest` sloupce. Reprodukovatelnost ověřená (seed 42, 20 gen → byte-identical).
  - **První pozorování (seed 42, 50 gen):** `lineages` 200 → 112 → 45 → 21 → 5 → **1** (plná konvergence v gen 49). Populace také kolabuje 200 → 1 — Hebbian + cycle scarcity peak akcelerovaly selekci silně. Učení dává lepším brainům okamžitou výhodu, dominantní lineage se rozpíná rychle, slabší vymře.
- **Poznámky:**
  - Lineage tracking je pure analytical tool — nemění biologii ani selekční tlak. Jen dává okno.
  - Hebbian dramaticky mění dynamiku. Empirické tunění `LEARNING_RATE` bude nutné — moc velký = brain unstable, moc malý = bez efektu.
  - **Lamarckian** je velký compromise. Darwinian model = child inherits "innate" (birth) brain, ne current. Sprint 19 možná split innate vs learned.
  - **Reward = +1 jen při eat.** Mohli bychom přidat reward při predaci (+0.5), penalizaci při ohrožení predátorem (−0.5), atd. Sprint 18 je jen baseline.
  - **Memory cost per cell:** +76 bytes (last activations). Pro 1000 cells = 76 KB. Negligible.
  - **Bez eligibility traces** (exponentially decaying memory) — credit assignment je 1-tick myopic. Pokud učení nezvedne výkon, přidat traces (Sprint 19/20).
  - **Visualization compromise:** `color_hue` gene se z evolučního pohledu stává neutrální drift — nikdo ho nevidí. Když budeme chtít gen vrátit, můžeme přidat toggle (`K` key) mezi lineage-color a hue-color módem.

## Sprint 19 — sexual-reproduction-and-perf

- **Cíl:** **(A)** přejít z asexuální fission na **pohlavní reprodukci** s mate-finding a crossover-em, **(B)** zoptimalizovat smell field (~25 % CPU času) přes snížení gridu. Sex zpomalí populační růst, ale otevírá genetickou pestrost přes recombinaci. Sprint 18 byla rychlá Lamarckian propagation učení; sex přidá Mendelovský mixing — dva mechanismy v napětí.

  **A) Pohlavní reprodukce s mate choice:**
  - **Mate selection** v `cell_reproduces_on_threshold`: snapshotni fertilní (`energy ≥ REPRODUCE_THRESHOLD`) buňky do `Vec<(Entity, pos, ...)>`. Pro každou nepárovanou fertilní cell najdi nejbližšího nepárovaného partnera v `MATING_RADIUS = 30.0`. Greedy pairing přes O(N²) na fertile pool (typicky ~tens cells, manageable).
  - **Crossover:**
    - `Genome::crossover(a, b, rng)` — per-gene uniform (50/50 každý scalar gene).
    - `Brain::crossover(a, b, rng)` — **per-row uniform**: každý hidden neuron (`w1[i]`+`b1[i]`) a každý output neuron (`w2[i]`+`b2[i]`) z jednoho rodiče. Preserves coordinated weight patterns lepe než per-weight uniform.
  - **Energy:** oba rodiče ztratí 50 % energie, child dostane sumu příspěvků (`a/2 + b/2`). Total energy zachována.
  - **Population growth:** **1 child per 2 rodiče** (proti asexual 1 child per 1 rodič). Růst populace ~poloviční, kompenzace za vyšší diverzitu.
  - **Lineage:** child kopíruje **parent A's lineage** (initiator). Sprint 18 tracking pokračuje, jen s mixed genomes.
  - **Per-tick:** `paired: HashSet<Entity>` zabraňuje double-mating. Pro mating: lookup obou rodičů přes `cells.get_many_mut([a, b])`, halve energy, crossover + mutate, spawn child.

  **B) Smell field perf optimization:**
  - `SMELL_GRID_RES: 128 → 64`. World grid cell size 1920/64 = **30 sim units** = 6× CELL_RADIUS. Coarser, ale smell diffuse/decay je local — gradient na nearby food zůstává funkční.
  - **4× méně grid cells** → 4× rychlejší Jacobi step. Očekávaný total speedup ~2–3× (smell field byl ~25 % CPU).
  - `SMELL_SAMPLE_EPSILON = 10.0` zůstává — menší než cell size 30, gradient computation samples z bucket cell value (no interpolation).
  - Pokud blocky artefakty na obrazovce vidět, alternativa: 128 + stride update (každý 2. tick s 2× dt).

- **Konstanty:** `MATING_RADIUS = 30.0`, `SMELL_GRID_RES: 128 → 64`.

- **Lib.rs API:**
  - `Genome::crossover(a: &Genome, b: &Genome, rng) -> Genome`
  - `Brain::crossover(a: &Brain, b: &Brain, rng) -> Brain`

- **Výstup:**
  - `src/lib.rs`: `Genome::crossover(a, b, rng)` per-gene uniform, `Brain::crossover(a, b, rng)` per-row uniform (každý hidden + output neuron's whole row z jednoho rodiče). Test `crossover_picks_genes_from_either_parent`. 16/16 testů.
  - `src/main.rs` + `src/bin/headless.rs`: `MATING_RADIUS = 30.0`. `SMELL_GRID_RES: 128 → 64`. `Genome` přidaný do imports.
  - **`cell_reproduces_on_threshold` přepsán na sexual:** snapshotuje fertile cells, greedy O(N²) pairing v `MATING_RADIUS`, paired set zabraňuje double-mating. Pro každý mating: `cells.get_many_mut([a, b])`, halve obě energie, crossover + mutate genome, child z midpoint pozice, energy = sum_of_halves. Lineage z parent A.
  - Headless `reproduce`: stejný logic přes index-pair (`split_at_mut` pattern pro dual mut borrow).
  - **Performance:** 9721 ticks/s (z 6321 s smell res 128). ~1.5× speedup pro typical run, na pozdějších fázích (méně cells) můžou být víc.
  - **Reprodukovatelnost ověřená** (seed 42, 20 gen → byte-identical).
  - **Pozorovaná dynamika (seed 42, 50 gen):** `cells` 200 → 3 (vs Sprint 18 200 → 1 v gen 49 — sex dává trochu víc resilience). `lineages` 200 → 3 — i s genomic mixing konvergence pokračuje. `spd_avg` se evolvuje pomaleji (60 → 64 vs Sprint 18 60 → 82) — recombinace promíchává adaptace, žádný "winner takes all".
- **Poznámky:**
  - **Sex zpomaluje evoluci** — pop growth halved. S balance pass možná boost food density nebo lower REPRODUCE_THRESHOLD aby se kompenzovala.
  - **Greedy pairing** není optimal. Future Sprint by mohl přidat preference gene (= "kompatibilita" — která se podobá pak je víc lákavá), reálná mate choice.
  - **Lineage z parent A** je arbitrární. Alternativa: hash(A, B) jako nová ID (speciation events). Pro Sprint 19 jednoduchost.
  - **Per-row Brain crossover:** kompromis mezi per-weight (víc disruptive) a per-layer (méně mixování). Empirické tunění může být potřeba.
  - **Sex + Hebbian interakce zajímavá:** Hebbian tlačí učení (Lamarckian-like), crossover pak mixuje naučené brainy. Lamarckian aspect Sprintu 18 je sexem rozmlžený.
  - **Smell res 64 risks:** gradient na hraně grid cellu je discontinuous (sample epsilon 10 < cell size 30 = gradient closure ignoruje sub-cell variation). Cells možná uvidí "jezera" konstantního smell. Empirické.

## Sprint 20 — stabilization

- **Cíl:** ekosystém po Sprintu 19 stabilně kolaboval (200 → 3 cells během 50 gen). Sprint 20 = empirický balance pass — najít konstanty, které udrží populaci ve stabilní rovnováze ~300-700 cells, ne extinct ani cap. Po stabilizaci budou cognitive sprinty (eligibility traces, smarter sensing, NEAT-style topology) měřitelné.

  **Hypotézy o příčinách kolapsu:**
  1. **Sex potřebuje proximity:** `MATING_RADIUS = 30` v 1920×1080 světě. Při 50 cells je avg distance ~200 — sex přestává fungovat při low density. Death spiral.
  2. **Sex zpomalí growth:** 1 child per 2 parents (vs 1 per 1 asexual). Při sniženém growth + stejné mortality → kolapse.
  3. **Hebbian aggressively konverguje:** dominantní lineage rychle zabere zdroje. Sprint 18 už dáno, ale interakce se sexem (mixování brainů přes recombinaci) možná destabilizuje.
  4. **Cycle stress (gen 37.5)** přijde když je populace ještě "naivní" — selekce nestihne najít balance.

  **Plán:**
  - Empirický experimenter na headless: spustit s různými settings, najít ty, co dávají stabilní populaci 300-700 přes 200 generací.
  - **Kandidáti tunings:**
    - `MATING_RADIUS: 30 → 100` (cells najdou partnera i v sparse populaci)
    - `REPRODUCE_THRESHOLD: 200 → 150` (faster pairing, kompenzuje sexual halving)
    - `CYCLE_AMPLITUDE: 0.3 → 0.15` (mild stress, nezvládne to ještě naivní populaci)
    - `LEARNING_RATE: 0.01 → 0.005` (méně dominantní Hebbian, víc evoluční time)
    - `WORLD_UNITS_PER_FOOD: 3000 → 2500` (mírně víc jídla)
  - Iterativní: měnit po jedné, sledovat efekt na trajektorii.

- **Výstup:**
  - **Angular energy fix:** `PhysicsConfig.angular_energy_cost` oddělen od linear `energy_cost_per_v_sq`. Spinning-in-place byl degenerate local minimum: rotační drain `body_size² × ω² × 0.0008` ≈ 0.05/gen, vision/body ~10/gen. Spinning byl prakticky zdarma → buňky se točily na místě a Hebbian to občas reinforced. Nový `ANGULAR_ENERGY_COST = 0.05` (60× vyšší) → spinning ~3/sec ≈ srovnatelné s pohybem.
  - **Test `step_drains_energy_from_rotation` updated** + nový regression `step_rotation_cost_independent_of_linear_cost` (linear cost 99.0, angular 0.0 — spinning cell nesmí ztratit energii). 17/17 testů.
  - **Konstanty po stabilizaci** (best zatím nalezené): `WORLD_UNITS_PER_FOOD = 2600`, `REPRODUCE_THRESHOLD = 150`, `MATING_RADIUS = 100`, `CYCLE_AMPLITUDE = 0.15`, `LEARNING_RATE = 0.005`, `ANGULAR_ENERGY_COST = 0.05`.
  - **Pozorovaná dynamika (seed 0, 200 gen):** populace 200 → bottleneck 104 (gen 19) → recovery → cap 1000 (gen 34). **Stabilní, ale homogenní** — `lineages` 200 → 16 v gen 38, dále drift dolů. Roztočení vymizelo.
- **Poznámky:**
  - **Stabilizace boom-bust nedořešena:** Empirické tunings food/threshold dávají buď cap, nebo near-extinction, nebo mid-stuck. Z podstaty vychází: ~5 % random brainů je funkčních pohybovačů, zbytek hladoví, pak několik klonů zaplaví svět.
  - **Stable ≠ healthy:** populace cap 1000 s 16 liniemi je ekvivalent monokultury. Diverzita nutná pro pokračování evoluce — adresováno v Sprint 21+ (prostorová heterogenita, niche differentiation, reprodukční izolace).
  - Stabilizace je iterativní — Sprint 20 najde rozumné defaults, ale finální tuning bude průběžný napříč budoucími sprinty.
  - **Asexuální fallback** (cell se rozdělí asexuálně pokud nikdo k pářit ve `MATING_RADIUS`) by byl elegantnější fix než větší radius — biologicky reálné. Pokud tunings nestačí, Sprint 21 fallback.
  - **Initial brain bias:** všechno random startuje. Mohl by pomoct mírný positive bias na thrust output (cells startují mírně jdoucí dopředu místo random walk). Hack ale efektivní.
  - **Empirické tunění bude potřeba**: `DRAG_COEFFICIENT`, `ENERGY_COST_PER_V_SQ`, `MAX_TORQUE` (přes `turn_rate` gene), `SMELL_DIFFUSION`/`DECAY`/`PER_FOOD`. První běh téměř určitě skončí extinction, balance přes několik iterací.
  - **Headless harness ze Sprintu 15 je teď neocenitelný** — bez něj by se 5 různých konstant ladilo přes manuální okno, hodiny ztracené. S headless: changes → recompile → 30 replikátů × 200 gen = pár minut → CSV → diagnose.
  - Smell sample epsilon 10 sim units = ~0.7 grid cells. Pro robust gradient možná zvýšit (15-20). Empirické.
  - Při změně `BRAIN_INPUTS` musí mirror v `headless.rs` taky updatovat brain input building. Drobné, ale dvě místa.
  - `SmellField` headless: stejná logika, prostě `Vec<f32>` v `World` struct. Bez Bevy resource, žádný overhead.
  - Po Sprintu 16: brain má 9 inputs (food dir, cell dir, energy, speed, relative size, smell gradient). To je bohaté informační prostředí — capacity hidden layer 8 je možná nedostačující. **Sprint 17 možná `BRAIN_HIDDEN: 8 → 12 nebo 16`**, pokud emergent chování nezačne přicházet.
