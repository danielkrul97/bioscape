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
