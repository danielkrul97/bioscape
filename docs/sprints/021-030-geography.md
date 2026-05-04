# Sprinty 21–30: Geography

Decade focused on **prostorové struktuře a niche differentiation**. Sprint 20 ukázal, že globálně homogenní svět vede k genetické monokultuře — jedna linie ovládne niku a evoluce se zastaví. Cílem této desítky je vytvořit prostředí, ve kterém má smysl být jiný: různá místa nutí různé strategie, subpopulace se mohou diferencovat, reprodukce nesetře genom přes celou planetu.

## Sprint 21 — spatial-foundation

- **Cíl:** zlomit globální homogenizaci linií ze Sprintu 20 přes deterministické prostorové pole, které moduluje food density. Bohatá místa vs. chudá místa → geografická specializace, lokální subpopulace, naděje na divergenci linií.

  **Plán:**
  - Nový typ `WorldMap` v `lib.rs` (Bevy-free), držící skalární pole `food_richness ∈ [0, 1]` na 64×64 mřížce.
  - **Value noise s smoothstep interpolací** z 8×8 random base grid, deterministicky ze `seed`. Vlastní implementace ~50 řádků, žádná nová crate dependency.
  - `WorldMap::sample(pos) -> f32` pro lookup; `WorldMap::new(seed, resolution, base_resolution, world_half) -> Self`.
  - **Spawn food** v `main.rs` + `headless.rs` sampluje richness a odmítne kandidátní pozici s pravděpodobností `(1 - richness)` — bohatá místa přitahují víc jídla.
  - **Visual overlay** v Bevy: Image asset z grayscale food_richness, Sprite na Z = -10, alpha 0.3, toggleable klávesou `M` (default visible).
  - **Headless** generuje WorldMap ze stejného seedu — reprodukovatelnost zachovaná.

- **Konstanty:**
  - `WORLD_MAP_RES: usize = 64`
  - `WORLD_MAP_BASE_RES: usize = 8` (=> ~240 sim units per "blob", ~8 blobs across 1920 width)
  - `WORLD_MAP_SEED: u64 = 1234`

- **Lib.rs API:**
  - `WorldMap::new(resolution, base_resolution, world_half, seed) -> Self`
  - `WorldMap::sample(pos: [f32; 2]) -> f32`

- **Výstup:**
  - **`WorldMap` typ v `lib.rs`** s value-noise generátorem (8×8 random base → 64×64 smoothstep bilinear interp), `sample(pos) -> f32`, deterministickým seedem. 4 testy: determinismus, různé seeds, range [0,1], boundary clamp. 21/21 testů.
  - **Bevy overlay** v `main.rs`: Image asset z grayscale field (zelená = bohaté, tmavá = chudé), Sprite z=-10 alpha 0.3, toggle klávesou M.
  - **Mechanika:** od rejection sampling přes food spawn (5 iterací, všechny extinct gen 70-110) jsme přešli k **food-value modulaci** — uniform spawn lokací, energie z jídla = `FOOD_VALUE × (FLOOR + AMP × richness)`. Konstanty `FLOOR=0.85, AMP=0.3` → range [0.85, 1.15] kolem baseline. Average ≈ 1.0, total food count se nemění.
  - **Pozorovaná dynamika (seed 0, 200 gen, food-value modulace):** 200 → bottleneck **18** (gen 40) → recovery → cap 1000 (gen 80). `lineages` 200 → **5** v gen 80.
- **Poznámky:**
  - **Negative result na cíl sprintu:** prostorová heterogenita **zhoršila diverzitu** (5 lineages vs 16 v Sprint 20 baseline), ne zlepšila. Důvod: hlubší bottleneck (18 vs 104) → silnější genetický drift → méně linií přežije. Cíl "zlomit homogenizaci" se nedaří.
  - **Tested permutace** všechny extinkční s rejection sampling food spawn:
    - v1: floor=0, amp=1, base_res=8 → extinct gen 110
    - v2: floor=0.3, amp=1.4, base_res=8 → extinct gen 90
    - v3: floor=0.3, amp=1.4, base_res=16 (smaller blobs) → extinct gen 70
    - v4: floor=0.6, amp=0.8, base_res=4 (continents) → extinct gen 90
    - v5: food-value modulace floor=0.4, amp=1.2 → extinct gen 100
    - v6: food-value modulace floor=0.85, amp=0.3 → cap (current default)
  - **Diagnóza root cause:** ~5 % random brainů je funkčních pohybovačů. Heterogenní food (ať distribuce nebo value) přidává variance v energii za jednotku času. Cells s nefunkčními mozky v "horší" pozici hladovějí rychleji → hlubší bottleneck → genetic drift dominuje selekci.
  - **Sprint 21 = infrastruktura, ne efekt.** WorldMap je hotový + visual overlay. Mírný kontrast zachová stabilitu, ale nezlomí monokulturu. Skutečné využití heterogenity vyžaduje **prerequisite cognitive priors** (Sprint 22+).
  - **Single octave** je záměrná simplifikace. Multi-octave přijde když budou potřeba menší detaily.
  - **Determinismus** WorldMap byte-identical mezi main + headless pro stejný seed (testem ověřeno).

## Sprint 22 — innate-brain-priors

- **Cíl:** odblokovat Sprint 21 přes vyšší podíl funkčních počátečních mozků. Sprint 21 selhal protože ~95 % random brainů je nefunkčních a heterogenní prostor jen zvýraznil bottleneck. Cílem je biased `Brain::random` + heading awareness — buňky startují s aktivním pohybem, ne random walkem.

  **Plán:**
  - **Architektura**: `BRAIN_INPUTS: 9 → 11`. Nové vstupy `inputs[9] = cos(heading), inputs[10] = sin(heading)` v obou binárkách (`brain_act`). Dovolí mozku v principu počítat body-frame food direction.
  - **Prior**: `INNATE_THRUST_BIAS: f32 = 2.0` přičten k `b2[1]` v `Brain::random`. Posune mean thrust output z ~0 (random walk) k ~+0.7 → defaultní pohyb dopředu.
  - **Strukturální food/smell priors** (cross/dot-product detektory přes hidden layer) zatím vynechány — vyžadovaly by ručně-tuned váhy a komplikované testování. Thrust bias + heading inputs jako minimum viable; pokud nestačí, Sprint 23+.
  - **Test**: na 200 random brainech mean thrust > 0.3 a >75 % má kladný thrust (dříve mean ~0, ~50 %).

- **Výstup:**
  - **`BRAIN_INPUTS = 11`** + nový test `random_brain_average_thrust_is_positive`. 22/22 testů.
  - **`INNATE_THRUST_BIAS = 2.0`** v `lib.rs`, přičten v `Brain::random` na `b2[1]` po Gaussian initu.
  - **`brain_act` v main.rs i headless.rs** populují `inputs[9..11]` z `cell.heading.cos/sin()`.
  - **Pozorovaná dynamika (seed 0, 200 gen):** 200 → **306** (gen 10, žádný bottleneck) → cap 1000 (gen 20) → emergent stable giants (size_avg 1.0 → 5.0) → samoregulace na ~200-300 cells (gen 100+). `spd_avg` 62 → **225**, vision 50 → 12 (cells nahrazují vision smellem). Lineages 200 → 1.
- **Poznámky:**
  - **Dramatický rozdíl** vs. Sprint 21 baseline (bottleneck 18). Cells okamžitě rostou — selekce má co selektovat protože mozky fungují od začátku.
  - **Emergent giant predator regime**: body_size 5 evolvuje z 1.0. Predace + samoregulace na ~250 cells. Není to cíl sprintu, ale ukazuje, že selekce funguje.
  - **Stále monokultura** (1 linie). Sprint 21 mírná heterogenita (`amp=0.3`) ji nezlomí. To je úkol Sprint 23.
  - **Heading inputs** otevírají dveře pro cross/dot-product detektory v Sprintu 23+ — brain teď v principu může spočítat "food ahead" vs "food behind" přes naučené hidden patterns. Hebbian na to možná dorazí sám.

## Sprint 23 — environmental-hazards

- **Cíl:** zlomit monokulturu ze Sprintu 22 přes nezávislou selekční páku — passive energy drain v "nebezpečných" zónách. Zóny pozitivně korelované s food richness (rich = dangerous, poor = safe) → trade-off niche, který nutí různé strategie napříč prostorem.

  **Plán:**
  - Hazard sample přes existující `WorldMap` (stejný noise, jiná interpretace) — žádná nová mapa.
  - Per-tick drain `HAZARD_DRAIN_PER_SEC × (HAZARD_FLOOR + HAZARD_AMP × noise) × dt` v každé buňce.
  - Mechanika čistě passive — žádná dynamická entita, žádná nová `Cell` field. Jen drain v `apply_hazards` system / metoda.
  - `Cell::step` nebo `PhysicsConfig` zůstávají netknuté — hazard drain se aplikuje externě, podobně jako predation.

- **Konstanty:**
  - `HAZARD_DRAIN_PER_SEC: f32 = 0.5` — base drain v nejhorší zóně. 0.5/sec = 5/gen, srovnatelně s vision cost.
  - `HAZARD_FLOOR: f32 = 0.0` — minimum hazard (safe zone)
  - `HAZARD_AMP: f32 = 1.0` — multiplier na noise

- **Výstup:**
  - `lib.rs`: 3 nové `pub const`. Žádný API change.
  - `main.rs`: nový system `apply_environmental_hazards` mezi `step_cells` a `rebuild_cell_grid`. Helper `hazard_drain(noise)`.
  - `bin/headless.rs`: nová metoda `World::apply_hazards`, helper `hazard_drain(noise)`. Volá se po `step` v `tick`.
  - **Dva běhy:** v1 s `DRAIN=2.0` extinct gen 40 (drain dominoval food bonus). v2 s `DRAIN=0.5` stabilní cap + diverzita.
  - **Pozorovaná dynamika (seed 0, 200 gen, v2):** 200 → cap 1000 (gen 20) → **persistent stable cap** s **20-21 liniemi** napříč celých 200 generací. `spd_avg` 60 → 80 (stabilní, NE giants), `size_avg` 1.0 → **1.5** (small! hazard činí velké body unaffordable), `vision_dev` 15.1 (heterogenní mezi liniemi → niche differentiation).
- **Poznámky:**
  - **Sprint 23 zlomil monokulturu.** 1 linie (Sprint 22) → 20-21 (Sprint 23). Persistentní napříč 200 gen — ne genetic drift, ale stable selection.
  - **Giants vymizeli:** hazard × body² cost činí velké cells neudržitelné. Sprint 22 emergent giant regime nahrazen "small generalist" + variance v vision (specialized). Trade-off mechanika funguje.
  - **Vysoká `vision_dev` (15)** ukazuje, že linie mají různé vision strategie — silný signál spatial niching. Pro potvrzení by Sprint 24 mohl přidat per-region analytics: zda jsou různé linie skutečně geograficky separované, nebo jen genome-distance separated.
  - **Tunování drain:** 0.5/sec sweet spot. 2.0 = extinkce, 0.1 nejspíš zanedbatelné. Mohlo by být jeptelnější jako fraction of food gain, ale konstantní funguje.
  - **Rich-dangerous interpretace** je intuitivní: bohaté biomy reálného světa (džungle) jsou taky nebezpečnější (predátoři, soutěž). Cells musí být efektivní aby z toho profitovali.

## Sprint 24+ — TBD

Možné směry:
- **Spatial speciation analytics** — CSV stats per region (svět rozdělen na N×N kvadrantů, lineage count + dominant genome per region). Přímo testovatelná hypotéza spatial niching.
- **Reprodukční izolace** přes `genome_distance(a, b) < threshold` (NEAT-style speciation) — pojistka pro persistenci diverzity.
- **Mobilní hrozby** — wandering predator entities, navigační AI challenge (brain musí zpracovat hrozbu jako další gradient).
- **Multi-food types** — různé typy jídla s různými energetickými profily, food-side niching.
- **Terrain drag** (třetí WorldMap vrstva) — pohyblivost varíuje s pozicí.
- **Asexuální fallback** při sparse mating density.
