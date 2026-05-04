# Sprinty 41–50: Morfogeneze

Decade rozšiřuje morfologický prostor za hranice rotačně symetrického elipsoidu ze Sprintů 26–34. **Sprint 41** zavírá dvě díry zjištěné po Sprintu 40 stabilizaci: (1) eat-zóna je sférická, takže evolvovaný "chip" tvar nemá geometrickou výhodu při sběru jídla — selekce na chip je čistě hydrodynamická, ne lov-geometrická; (2) buňky postrádají defensive nice — predace je jednosměrná škála (spike vs no-spike), bez passive obrany. Sprinty 42–43 dokončí povrchové modifiery (keel, ventro/dorzální asymetrie). Pozdější sprinty (44+) směřují k topologické morfogenezi (segmentace, multi-cell tělo) podle `docs/04-bunky-a-morfogeneze.md`.

## Sprint 41 — eat-ellipsoid + shell

- **Cíl:** vyřešit dvě konkrétní díry v morfologickém modelu identifikované po Sprintu 40:
  1. **Orientovaná eat-zóna** (body 1). Současné `Cell::try_eat` testuje sféru s `eat_radius = EAT_RADIUS × effective_radius()` = `EAT_RADIUS × (L+W+H)/3`. Long+narrow chip (L=2, W=0.5, H=0.5) má identickou eat-sféru jako koule s objemem 0.5. Selekce na chip je čistě o forward dragu, ne o "větší tlamě". Sprint 41 přepne eat zónu na **ellipsoid v body frame**: forward semi-osa ∝ length, lateral ∝ width, vertical ∝ height. Hypotéza: vznikne reálný tradeoff "úzká tlama, rychlý" vs "omniční záběr, pomalý", `len/wid` ratio konverguje níž než pre-Sprint-41.
  2. **Shell jako passive damage absorber** (body 2 — první ze tří sub-směrů). Nový gen `shell_thickness ∈ [0, MAX_SHELL_THICKNESS]`, který tlumí incoming damage před zápisem do `damage_accum`. Cost lineární (defensive armor je drahá). Otevře defensive niku — silně shellovaná buňka neuhne predátorovi ale levně přežije occasional hit + hazard zone.

  Sub-směry body 2 (keel, ventro/dorzální asymetrie) odložené na Sprint 42–43, viz Poznámky.

  **Plán:**

  *Body 1 — eat-ellipsoid:*
  - `Cell::try_eat(food, eat_factor, food_value)`: API změna — scalar `eat_radius` parametr nahrazen `eat_factor` (= globální `EAT_RADIUS` const). Interní výpočet: `semi_axes = [eat_factor × body_length, eat_factor × body_width, eat_factor × body_height]`.
  - Helper `body_basis(yaw, pitch) → ([f32;3], [f32;3], [f32;3])` v lib.rs vrací orthonormální `(fwd, right, up)`. `fwd` reuzuje existující `forward_vector`; `right` je rotace o 90° v xy-plane (nezávislé na pitch — předpoklad axiální symetrie kolem fwd, žádný roll); `up` je `fwd × right`.
  - Test: `(d_par / L_eat)² + (d_right / W_eat)² + (d_up / H_eat)² ≤ 1`. Pro L=W=H reduce na sférický test s radius = eat_factor × s (backward kompat při isotropní buňce).
  - Broad-phase fix: nový helper `Phenotype::max_axis() → body_length.max(body_width).max(body_height)`. Callsiteové (`headless::brain_act` + `main::predate_food_system`) místo `effective_radius()` v eat-broadphase + cell-exclusion computaci použijí `max_axis()`. Důvod: bucket musí pokrýt nejširší možnou semi-osu, jinak ellipsoid extending podél long axis vypadne mimo broad-phase a missneme valid eat target.
  - Unit testy:
    - `try_eat_forward_chip_reaches_further_than_lateral`: chip (L=2, W=0.5, H=0.5), eat_factor=8 → forward eat na 16, lateral na 4. Food at +14 podél heading = eaten; food at +14 podél right = not eaten.
    - `try_eat_isotropic_unchanged_for_unit_sphere`: L=W=H=1, eat_factor=8 → acceptance shell radius 8 v každém směru, identické s pre-Sprint-41 sférou.
    - `body_basis_orthonormal`: 5 yaw × 3 pitch kombinací, `|fwd|=|right|=|up|=1`, `fwd·right=fwd·up=right·up=0` v ε.

  *Body 2 — shell:*
  - Genom: `shell_thickness: f32`, init range `[0.0, 0.2]` (mírný počáteční mean, žádný extreme spawn), mutation `sigma_shell = 0.03`, crossover stejně jako ostatní geny.
  - Phenotype: `shell_thickness: f32` (snapshot z genomu, runtime morph **neexistuje** — Sprint 41 šetří brain output indexy; Sprint 42 evaluuje runtime morph signál pro shell).
  - Mechanika: nová metoda `Cell::apply_shell_absorb(dt)` se volá v hot loopu binárek **před** `populate_brain_inputs`. Snižuje `damage_accum` o `shell_thickness × SHELL_ABSORB_PER_TICK × dt`, floor at 0. `populate_brain_inputs` čte už absorbnutý damage a resetuje. Symmetric pattern se Sprint 30 damage signal lifecycle.
  - Cost: v `apply_energy_costs` přidat `self.energy -= self.phenotype.shell_thickness × SHELL_COST_PER_SEC × dt`.
  - Unit testy:
    - `shell_absorbs_predation_drain`: shell=1.0, raw damage=3.0, ABSORB=2.0, dt=1 → po absorb damage_accum=1.0.
    - `shell_zero_no_effect`: shell=0.0, damage_accum identický s pre-Sprint-41 trajectorií.
    - `shell_does_not_absorb_below_zero`: shell=10, damage=1 → clamp na 0, ne na -19.
    - `shell_cost_scales_linearly`: shell=1.0, dt=1.0 → energy drain = SHELL_COST_PER_SEC v ε.
    - `shell_mutation_clamps_to_range`: 1000 mutation iterací, žádná hodnota mimo [MIN, MAX].

- **Konstanty:**
  - `MIN_SHELL_THICKNESS = 0.0`
  - `MAX_SHELL_THICKNESS = 1.5` — pod max body axis (4.0), dost na meaningful absorb. Linear cost při max = 1.5 × 0.4 = 0.6/s, srovnatelné se spike při max length.
  - `SHELL_ABSORB_PER_TICK = 2.0` — single predation hit = `PREDATION_DRAIN_PER_TICK = 3.0`, takže shell=1.5 plně absorbnu hit (3.0 ≤ 1.5 × 2.0), shell=1.0 absorbnu 2/3 hit, shell=0.5 půlí incoming damage.
  - `SHELL_COST_PER_SEC = 0.4` — vyšší než spike (0.3/s) protože shell pokrývá celý povrch, ne point structure.
  - `MUTATION_CONFIG.sigma_shell = 0.03`

- **Výstup:**
  - `lib.rs`: `MIN_SHELL_THICKNESS`, `MAX_SHELL_THICKNESS`, `SHELL_ABSORB_PER_TICK`, `SHELL_COST_PER_SEC` consts. `MutationConfig.sigma_shell` field + `MUTATION_CONFIG.sigma_shell = 0.03`. `Genome.shell_thickness` field + init/mutate/crossover. `Phenotype.shell_thickness` field + `from_genome`. `Phenotype::max_axis()` helper. `body_basis(yaw, pitch)` helper. `Cell::eat_test()` (pure narrow-phase test) + přepsaný `Cell::try_eat(food, eat_factor, value)` na ellipsoid acceptance. `Cell::apply_shell_absorb(dt)`. Shell maintenance cost v `apply_energy_costs`.
  - `src/bin/headless.rs`: `eat_food` přepsán na `cell.try_eat(food, EAT_RADIUS, value)` call (no inlined sphere check). `spawn_food` cell-exclusion zóna používá `max_axis`. `brain_act` volá `apply_shell_absorb(dt)` před `populate_brain_inputs`.
  - `src/main.rs`: `cell_eats_food` broad-phase `eat_r = EAT_RADIUS × max_axis`, narrow-phase `cell.eat_test()` v closure. `spawn_food` broad-phase budget bumped na `EAT_RADIUS × MAX_BODY_LENGTH` (z `EAT_RADIUS × BROAD_PHASE_SIZE_BUDGET = 24 → 32`). `cells_brain_act` volá `apply_shell_absorb(dt)` před `populate_brain_inputs`. `MAX_BODY_LENGTH` přidaný do importů z `bioscape::`.
  - **9 nových unit testů**: `body_basis_orthonormal`, `try_eat_isotropic_unchanged_for_unit_sphere`, `try_eat_forward_chip_reaches_further_than_lateral`, `max_axis_returns_largest_dimension`, `shell_absorbs_predation_drain`, `shell_zero_no_effect`, `shell_does_not_absorb_below_zero`, `shell_cost_scales_linearly`, `shell_mutation_clamps_to_range`. **45/45 testů pass** (36 původních beze změny po update fixtur).
  - **Smoke run (seed 0, 60 gen, headless):**
    - Pop trajektorie: 200 → 132 (gen 8) → 89 (gen 18, predation oscillation) → 100 (gen 38) → 207 (gen 48) → **1000 (gen 58, saturated MAX_POPULATION)**. Žádná extinkce.
    - Lineages: 200 → 71 → 20 → **9 (gen 28+ stable)** — typical Sprint 22+ konvergence.
    - **`len/wid` ratio FLIP**: gen 0 = 1.00, gen 18 = 0.61, gen 58 = 0.67. Pre-Sprint-41 baseline (Sprint 34) byla ~1.7 (chip pressure). Post-Sprint-41 ratio < 1 znamená cells preferují **lateral coverage** přes forward chip — přesně predikovaná inverze hypothesis. Selekce už neselektuje chip kvůli identickému eat-radiusu napříč osami; teď width-grow optimalizuje lateral eat capability.
    - `wid_avg` 0.98 → **1.50** (gen 59). `len_avg` zůstává blízko 1.0. Cells konvergují k "flat plate" tvaru (high width, baseline length+height).
    - Predation: 0 → 4869 events gen 59. Aktivní, ne degenerate stalemate.
    - Spike: 0.05 → 0.12 (gen 59). Mírný pokles oproti gen 8 peak (0.28).

- **Poznámky:**
  - **Proč `eat_factor` jako scalar místo passthrough Phenotype:** API zachovává callsite control. Cell vlastní phenotype + heading + pitch, ale globální násobitel zůstává externí konfigurace — nechává prostor pro per-cell hunger modifier nebo time-of-day modulaci v pozdějších sprintech bez další API změny.
  - **Proč shell *NE* runtime-morphable:** BRAIN_OUTPUTS=8 už má 4 morph signály (length, width, height, spike). Pátý morph = jeden další output index + brain matrix grow. Sprint 41 šetří scope. Sprint 42 evaluuje, jestli shell potřebuje runtime adjustment (defensive ramp-up po prvním damage hitu) nebo jestli inheritní hodnota stačí.
  - **Predace neighnoruje shell na straně útočníka.** Attacker vidí target přes vision inputs jako dřív, spike_bonus se neredukuje shellem cíle. Selekce sama najde, jestli spike + shell coevolve do "anti-shell spike" (vyšší spike_length) nebo non-spike bypass (speed, scavenging). Sprint 42+ může experimentovat s explicit spike-pierces-shell mechanics, pokud vznikne degenerate all-shell stalemate populace.
  - **CSV identity NEBUDE zachována.** Sprint 41 mění semantiku eat zóny → různá jídla po prvním ticku → různé energie → různá smrt/reprodukce. Žádný backward-compat seed run není možný. Acceptance kritérium: post-Sprint-41 seed=0 60gen produkuje stable population (≥ 100 cells gen 59), žádná extinkce.
  - **Smoke kritéria pro body 1 (validace hypotézy):**
    - Pre-Sprint-41 baseline (ze Sprintu 34 dynamiky): `len_avg ≈ 1.5`, `wid_avg ≈ 0.9`, ratio ≈ 1.7. Po Sprint 40 patches očekáváme podobnou trajectorii.
    - Post-Sprint-41 hypotéza: ratio přistane níž (1.2–1.5), protože extreme chip teď ztrácí lateral/vertical eat reach. Diversification: některé linie chip-pursue (high ratio), jiné omni-grazer (low ratio).
    - Pokud ratio neuhne (Δ < 0.1), eat-ellipsoid efekt je slabý a body 1 nestačí — Sprint 42 by měl zvýšit `eat_factor` nebo zavést přímo per-axis multiplier (nelinearitu).
  - **Smoke kritéria pro body 2 (validace shell niche):**
    - `shell_avg` napříč generacemi: pokud konverguje k 0 nebo MAX, niche je degenerate (cost too high / too low). Cíl: stabilní intermediate (~0.3–0.7) nebo bimodal distribution (predátoři low-shell, scavenger high-shell).
    - `predation_events / pop` ratio: pokud klesá výrazně, shell je príliš silný — predace ztratí ekonomický smysl. Sprint 42 by snížil `SHELL_ABSORB_PER_TICK`.
    - Korelace `shell_avg × spike_avg` per lineage: záporná = clean tradeoff (predátor vs defender), kladná = shell je free upgrade (bug v cost balanci).
  - **Sub-směry body 2 odložené:**
    - **Sprint 42 — keel pro směrovou stabilitu**: `keel_length` gen + phenotype, multiplikuje `drag_perp` (přidá lineární term k existujícímu `length` faktoru). Cost ∝ keel × per_sec. Otevře "ramming" niku — high-keel cell drží přímou linii i při bočním kontaktu, ale ztrácí agility při zatáčení.
    - **Sprint 43 — ventro/dorzální asymetrie**: split `body_height` na `body_height_top` + `body_height_bot`. Volume = L × W × (top + bot). Eat-ellipsoid → asymetrické vertical semi-osy podle znaménka d_up. Smysl má jen pokud existuje vertical heterogenita prostředí (Sprint 35 odložil 3D volumetric noise na "Sprint 38+"). Sprint 43 možná závisí na re-aktivovaném 3D environment field.
  - **Co Sprint 41 NEMĚNÍ:** brain I/O dimenze, MORPH_RATE / morph signal indexy, predation cone math, hazard zone mechanika, gravita (Sprint 38), 3D environment fields. Pure additive: 1 nový gen, 1 přepsaná funkce, 1 nový helper.

## Sprint 42 — life-history (aging + mass + Brownian + cooldown + carrion decay)

- **Cíl:** přidat 5 chybějících biofyzikálních realismů, které dnes simulace abstrahuje. Každá z nich je nezávislý mikro-mechanismus s individuální hypotézou; bundle do jednoho sprintu, protože per-bod scope je malý (~30–60 řádků) a sub-systémy nemají sdílený stav (lze izolovat A/B testem). Body 42 NEPŘIDÁVÁ keel ani ventro/dorzální asymetrii (původně plánované jako Sprint 42–43 sub-směry body 2 ze Sprintu 41); ty zůstávají v backlogu pro Sprint 43+.

  1. **Stárnutí (aging)**. Cells dnes umřou JEN když energy ≤ 0. Žádný věk, žádná degradace v čase. Sprint 42 přidá `Cell.age: u64` (ticks od spawnu) a ramp na body maintenance: `body_cost × volume × (1 + AGE_DECAY_PER_SEC × age_sec) × dt`. Hypotéza: vznikne **selekční tlak na životní strategii** — short-lived r-strateg (rychlá reprodukce než stárnutí udeří) vs long-lived K-strateg (drahé tělo s dlouhou životností + odolnost vůči stárnutí přes lower volume). Generační turnover by se měl zrychlit napříč všemi liniemi.

  2. **Mass / inerce**. Pre-Sprint-42 implicit `m = 1`. Reálně F = ma → velký objem víc inertia. Sprint 42 nahradí `body_proxy = effective_radius()` v `apply_brain_motor` za `mass = volume()` v denominátorech: linear thrust `a_max = drag × max_speed² / mass`, angular `ang_acc = turn_signal × turn_rate / mass`, pitch analogicky. Pro unit cell (L=W=H=1) se chování nemění (vol=1=eff_r=1). Pro chip (vol=0.5) cells akcelerují/zatáčejí 2× rychleji, pro tubby (vol=8) 8× pomaleji. Hypotéza: **další kontrastní tlak na flat-plate vs chunky body** — flat plate ze Sprintu 41 (high width, baseline length+height) je objemnější (vol > 1) než pre-Sprint-41 sférická (~1), takže selekce by ji mohla penalizovat za vyšší inerci. Sprint 41 + 42 se navzájem balancují.

  3. **Brownův pohyb**. Micro-scale fluid je dominantně thermal noise. Sprint 42 přidá `Cell::apply_brownian(rng, dt)` volaný v hot loopu před `step`: `velocity[i] += gaussian(rng) × THERMAL_NOISE × √dt` (Wiener-process scaling — sqrt(dt) je correct stochastic integration, ne lineární dt). Pro z-osu jen pokud `world_half[2] > 0`. RNG je propagován per-cell přes hot loop existujícího binárkového RNG. Hypotéza: **brain robustness test** — naučené řízení musí být robustní proti malým perturbacím; selekce odmění persistence (multi-tick averaging výstupů) místo single-tick reactive control.

  4. **Reproduction cooldown / refractory**. Mating je dnes cheap — energy split, instant child, žádná regenerační doba. Sprint 42 přidá `Cell.reproduce_cooldown_ticks: u32`. Po mating obou rodičů `cooldown = MATING_COOLDOWN_TICKS`. Per-tick decrement v `Cell::step`. `collect_fertile` kontroluje `cooldown == 0` navíc k existujícím podmínkám (energy + pheromone). Hypotéza: **omezení r-strategy nadprodukce** — cell po mating je X ticků „nesvéprávný"; stabilizuje populační oscilace, dává selekční prostor pro K-strategy + monogamii.

  5. **Carrion decomposition**. Carrion drop je dnes instant fresh food (=20 energy units). Reálně decay timer s lineárně klesající hodnotou. Sprint 42 přidá `Food.age_ticks: u32` (init 0 pro fresh food, init 0 i pro carrion — žádný sploh penalty na carrion oproti fresh, jen univerzální decay). Per-tick increment všech foodů. `food_value` násobeno `(1 - DECAY_RATE_PER_SEC × age_sec).max(0)`. Při `value ≤ 0` despawn. Hypotéza: **scavenger niche degradation** — staré food je menej výživné, takže "free meal" z carrion má omezenou hodnotu. Selekce odmění čerstvé eat (rychlí scavengeři, blízkost k predaci) vs lazy ingestion poors.

  **Plán implementace (per body):**

  *Body 1 — aging:*
  - `Cell.age: u64` field. Init 0 v `from_genome` + `make_mating_child`.
  - `Cell::step` zvedne age o 1 per tick (před apply_energy_costs, aby věk byl konzistentní v rámci ticku).
  - `apply_energy_costs`: změna body cost line z `volume × body_cost_factor × dt` na `volume × body_cost_factor × (1 + AGE_DECAY_PER_SEC × age_sec) × dt` kde `age_sec = age as f32 / FIXED_TIMESTEP_HZ`.
  - **Pozor**: `FIXED_TIMESTEP_HZ` const už v lib.rs existuje (=60); použít přímo.
  - **Test**: `step_aging_increases_body_cost` — cell s age=0 a cell s age=600 (=10s) ve stejném těle, druhý drains víc energie.

  *Body 2 — mass / inerce:*
  - V `apply_brain_motor`: nahradit `body_proxy = effective_radius()` za `mass = volume().max(0.01)`. Linear `a_max = DRAG_COEFFICIENT × max_speed² / mass`. Angular `ang_acc = turn_signal × turn_rate / mass`. Pitch analogicky.
  - **Pozor: nadpis comment v existujícím Sprint 26 anisotropic drag** používá `body_proxy` notaci pro DIFFERENT účel (drag math). Drag math NEMĚNÍME — jen motor. Drag používá `body_length`/`body_width` přímo, ne body_proxy.
  - **Test**: `motor_scales_inversely_with_mass` — dva cells, one volume=1 (unit) druhý volume=2 (např. L=2, W=H=1), stejný thrust signal → druhý dosáhne polovičního a_max.

  *Body 3 — Brownian:*
  - Nová metoda `Cell::apply_brownian(&mut self, rng, dt)`: 2D složky vždy, z jen když relevant. Použít `gaussian(rng)` helper (už existuje v lib.rs).
  - Volaná v binárkách v hot loopu před `step` (pro každou cell, sdílený RNG).
  - Pro deterministic seed reproducibility headlessu: RNG pořadí musí být stable. Aplikace všem cells za sebou před step splňuje to.
  - **Pozor**: `√dt` ne `dt`. `velocity[i] += gaussian × THERMAL_NOISE × dt.sqrt()`.
  - **Test**: `brownian_perturbs_zero_velocity` — cell s velocity=0 po `apply_brownian` má některou složku ≠ 0 (statisticky téměř jistě). Velocity magnitude < THERMAL_NOISE × 5 × √dt v 99 % případů (sigma bound).

  *Body 4 — cooldown:*
  - `Cell.reproduce_cooldown_ticks: u32` field. Init 0.
  - `Cell::step` per-tick decrement: `if self.reproduce_cooldown_ticks > 0 { self.reproduce_cooldown_ticks -= 1; }`.
  - Přidat condition v `collect_fertile`: `c.reproduce_cooldown_ticks == 0`.
  - V `spawn_children_from_matings` (headless) a equivalent v main.rs predicate spojení predicate (`cell_predates_on_neighbor` pre-mating logic): po mating set `cell_a.reproduce_cooldown_ticks = MATING_COOLDOWN_TICKS; cell_b.reproduce_cooldown_ticks = MATING_COOLDOWN_TICKS;`.
  - **Test**: `cooldown_decrements_per_step`, `cooldown_blocks_immediate_remating` (helper test pro fertile filter).

  *Body 5 — carrion / food decay:*
  - `Food.age_ticks: u32` field. Init 0 v `Food::random` a v `Food { position: ... }` literálech v carrion drop.
  - Nová metoda `Food::age_step(dt) -> bool` (returns false if expired). Increment age, vrací `value_factor() > 0`.
  - `Food::value_factor() -> f32`: vrací `(1 - DECAY_RATE_PER_SEC × age_sec).max(0)` kde `age_sec = age_ticks as f32 / FIXED_TIMESTEP_HZ`.
  - V eat callsiteích: vynásobit `FOOD_VALUE × food_multiplier(richness) × food.value_factor()`.
  - V hot loopu binárek: po food gravity loop přidat aging loop. Despawn foody s `age_step` returning false (analogicky `apply_food_gravity`).
  - **Test**: `food_value_decays_with_age`, `food_expires_when_zero_value`.

- **Konstanty (smoke-tuned po A/B isolation, viz Výstup):**
  - `AGE_DECAY_PER_SEC = 0.001` — při age_sec=100 factor=1.1 (10 % extra). Z plánovaných 0.005.
  - `THERMAL_NOISE = 0.3` — gaussian × 0.3 × √(1/60) ≈ 0.04 stddev per tick. Z plánovaných 1.0.
  - `MATING_COOLDOWN_TICKS = 10` — 1/6 sec refractory. Z plánovaných 60.
  - `CARRION_DECAY_PER_SEC = 0.0005` — food expires v ~2000s (200 generací). Z plánovaných 0.05.
  - `mass = effective_radius()` — z plánovaného `volume()` po A/B test (volume příliš agresivní inerce penalty pro untrained brainy).

- **Výstup:**
  - `lib.rs`: nové konstanty (`AGE_DECAY_PER_SEC`, `THERMAL_NOISE`, `MATING_COOLDOWN_TICKS`, `CARRION_DECAY_PER_SEC`). `Cell.age: u64` + `Cell.reproduce_cooldown_ticks: u32` fields. `Cell::step` increment age + decrement cooldown. `apply_energy_costs` ramp na body cost přes `aging_factor`. `apply_brain_motor` denominator `mass = effective_radius()` (smoke-tuned z `volume()`). `Cell::apply_brownian(rng, dt, world_half_z)` method. `Food.age_ticks: u32` field. `Food::value_factor()` + `Food::age_step()` methods.
  - `src/bin/headless.rs`: `apply_brownian(rng, dt)` v hot loopu před `step`. `apply_food_gravity` rozšířen o age_step + retain. `eat_food` value × `food.value_factor()`. `collect_fertile` čeká cooldown == 0. `spawn_children_from_matings` set parents' cooldown.
  - `src/main.rs`: `apply_brownian_motion` Bevy system před `step_cells`. `apply_food_gravity` despawne expired food via Commands. `cell_eats_food` value × `food.value_factor()`. `cell_reproduces_on_threshold` check cooldown + set po mating. `MATING_COOLDOWN_TICKS` import.
  - **10 nových unit testů**: `step_aging_increases_body_cost`, `step_increments_age`, `cooldown_decrements_per_step`, `cooldown_does_not_underflow`, `motor_scales_inversely_with_mass`, `brownian_perturbs_zero_velocity`, `brownian_z_only_in_3d_world`, `food_value_decays_with_age`, `food_expires_when_zero_value`, `child_starts_with_zero_age_and_cooldown`. **55/55 testů pass** (54 stable + 1 pre-existing flake `random_brain_average_thrust_is_positive`, závisí na thread-local RNG, mimo Sprint 42 scope).
  - **Smoke run iterace (seed 0, 60 gen, headless):**
    - V1 (planned constants AGE=0.005 / NOISE=1.0 / COOLDOWN=60 / DECAY=0.05 / mass=volume): **extinkce gen 20**. Combined effect zabíjel replacement rate (1523 fertile-ticks → 3 births).
    - V2 (graduated AGE=0.002 / NOISE=0.3 / COOLDOWN=30 / DECAY=0.02 / mass=volume): **extinkce gen 40**.
    - V3 (mass→effective_radius): **extinkce gen 32**.
    - V4 (cooldown=10): stále gen 32. Cooldown not dominant.
    - V5 (all-zero baseline + mass=eff_r): pop 667 gen 60, healthy.
    - V6 (only DECAY=0.02 enabled): **extinkce gen 23**. Carrion decay je dominantní killer — food saturuje na low-pop, expire před snědením.
    - V7 (DECAY=0.0005, ostatní zero): pop 657 gen 60, healthy.
    - **V8 (final, all-on tuned)**: AGE=0.001, NOISE=0.3, COOLDOWN=10, DECAY=0.0005, mass=eff_r. **Pop 200 → 32 dip (gen 29) → 999 saturated (gen 59).** 3 lineages, 3445 predation events. `len/wid/hgt = 0.98/1.65/0.95` (Sprint 41 flat-plate selekce zachovaná). Oldest cell 59 gen, births=170 deaths=85 v gen 59.
  - **CSV identity nezachována** (čekáno).

- **Poznámky (po smoke iterace):**
  - **Carrion decay byla výrazný killer**. Důvod: na low-pop scenarios se food saturuje (zde 916 food vs 13 cells). Cells eat closest, takže staré food sedí dlouho a expire dřív, než ho někdo najde. Cells umírají hladem v "potravinou plném světě". 0.0005 (gentle) zachovává mechanismus jako field architecture pro budoucí carrion-specific decay.
  - **Mass = volume()** byl příliš agresivní. Volume scaling exponenciálně naškáluje inerci s tělesnou velikostí (vol ∝ r³). Untrained brainy s vol > 1.5 nemohou navigovat. `mass = effective_radius` (= aritmetický průměr r) zachovává inerce-by-size signal bez kvadratického cost shocku. Cleanest follow-up: po stabilizaci populace (Sprint 43+) pomalu posunout `mass` z `effective_radius` k `volume.sqrt()` k full `volume`.
  - **Refactoring opportunity**: 5 mechanismů v jednom sprintu udělalo A/B isolation potřebnou + náročnou. Poznatek: pro plně nezávislé mechanismy ano (každý lze testovat samostatně), ale combined effect vyžaduje careful tuning. Pro Sprint 43+ raději 1 mechanismus per sprint.
  - **Hypothesis status:**
    - ✓ Aging: works at 0.001 (mírnější než plánovaný 0.005). Cells dosahují gen 59 max age — funkční selekční gradient.
    - ✓ Mass: works at `effective_radius` (mírnější než plánovaný `volume`). Velké cells slower; chip cells faster (selekce stále ladí flat-plate vs chip tradeoff).
    - ✓ Brownian: works at 0.3. Brain robustness signál — random šum přes cells, neničí navigaci.
    - ✓ Cooldown: works at 10. Brief refractory; nezná-li to nikdo, žádný side effect.
    - ⚠ Carrion decay: works at 0.0005, ale efekt je téměř neviditelný (food prakticky vždy fresh). Real "carrion decomposition" by vyžadoval `Food.is_carrion: bool` flag a oddělený rate jen pro carrion-drops; Sprint 42 universal-decay je scope-cut.
  - **Future work (Sprint 43+):**
    - Carrion-specific decay: `Food.is_carrion: bool`, decay rate 0.05 jen pro carrion (ne map-spawn). Restoruje původní intent.
    - Aging visible v CSV: nový column `mean_age_at_death`. Bez něj nejde A/B porovnat aging trajectorii.
    - Ramp mass scale: incremental shift od `effective_radius` k `volume.sqrt()` (=`volume()^(1/2)`) přes několik sprintů, aby selekce držela krok.
    - Keel + ventro/dorzální asymetrie ze Sprintu 41 (původní body 2 sub-směry) — odložené z 42, ne v plánu pro 43+.

- **Poznámky:**
  - **Combined sprint risk**: 5 mechanismů najednou znamená vyšší extinkce-risk pro smoke run. Per-body fallback strategie:
    - Pokud `AGE_DECAY` extinktuje populaci → snížit na 0.001 (factor 1.1 při age=100s).
    - Pokud `mass` v motoru extinktuje (cells příliš pomalé) → použít `mass = volume.sqrt()` nebo `effective_radius²` místo full volume.
    - Pokud `THERMAL_NOISE = 1.0` rozbije brain control → snížit na 0.3.
    - Pokud `cooldown = 60` extinktuje (málo replicate v gen) → 30 nebo 20.
    - Pokud `CARRION_DECAY` extinktuje (food zmizí dřív, než ho cells najdou) → 0.02 nebo 0.01.
  - **CSV identity NEBUDE zachována**. Každý z 5 bodů mění RNG draw count nebo per-cell update logic. Žádný backward-compat.
  - **Pre-Sprint-42 baseline pro A/B**: Sprint 41 smoke run (gen 0–60, seed 0): `len=0.99, wid=1.50, hgt=1.15`, ratio 0.66, predation 4869 events gen 59, pop 1000 saturated. Pokud Sprint 42 výrazně sníží konečnou populaci, identifikuje se viník přes selektivní disable per-bodu.
  - **Hypotézy a co měřit (per body):**
    - **Aging**: gen-průměrná age při smrti (mean_lifetime). Pokud aging je sensitive driver, mean_lifetime by měl klesat oproti pre-S42 (více cells umírá age-induced než starvation-induced). Add metric to CSV pokud time permits.
    - **Mass**: `len/wid/hgt` ratio shift. Pre-Sprint-42 (post-Sprint-41) byla flat plate (wid_avg=1.5, vol≈1.5). Sprint 42 mass penalty by měl tlačit k volume<1 → menší cells, hgt může klesnout, len může mírně růst (chip = nižší volume).
    - **Brownian**: brain output stability. Naučená populace v gen 30+ by měla mít smaller per-tick output variance (multi-tick averaging via recurrent state ze Sprintu 28). Hard to measure bez nového CSV columnu.
    - **Cooldown**: `births_per_gen` distribution. Pre-S42 spike high (2 cells > REPRODUCE_THRESHOLD = mating event). Post-S42 cooldown rate-limits → smoother births curve.
    - **Carrion decay**: predation-related food spawn → eat correlation. Pre-S42 carrion eaten do ~5s after death. Post-S42 some carrion expires nepojídaný → food saturation klesne při high predation rate.
  - **What Sprint 42 NEMĚNÍ:** body axes (length/width/height ranges), shell, ellipsoid eat zone, brain I/O dimensions, drag math, gravity (Sprint 38), 3D environment fields, predation cone math, hazard zones. 5 nezávislých additivních mechanismů.
  - **Why bundle:** každý subsystem je per-bod jednoduchý (~30–60 LOC); rozdělit by znamenalo 5 sprint dokumentů s velkým overhead. User-driven request, scope explicitní.
