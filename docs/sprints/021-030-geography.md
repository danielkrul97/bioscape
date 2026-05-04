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

- **Cíl:** odblokovat Sprint 21 přes vyšší podíl funkčních počátečních mozků. Sprint 21 selhal protože ~95 % random brainů je nefunkčních a heterogenní prostor jen zvýraznil bottleneck. Cílem je biased `Brain::random` tak, že většina nově narozených cells defaultně směřuje k jídlu — Hebbian + selekce pak tunují. Bez tohoto fixu žádný spatial sprint nemůže vykazovat diverzifikaci linií.

  **Plán:**
  - **Prior 1**: pozitivní bias na thrust output (`b2[1] += 0.5` při random init) — buňky startují s mírným pozitivním motorem, místo random walk.
  - **Prior 2**: kladné spojení food_dx → thrust, food_dy → turn (např. `w2[1][food_input_idx] += 0.3, w2[0][food_y_idx] += 0.3`). Defaultní heuristika "směrem k jídlu".
  - **Prior 3**: stejné pro smell gradient inputs (gradient_x → turn, gradient_y → thrust).
  - **Mutace nesnižuje priors disproportionately** — Gaussian noise sigma 0.2, prior signal ~0.3-0.5, takže selekce může priors přepsat ale defaultně zůstává.
  - Test: na 100 random brainech měřit avg `forward([1, 0, ..., 0])[thrust]` — měl by být kladný.

- **Výstup:** —

## Sprint 23 — re-test-spatial-with-priors

- **Cíl:** s funkčními počátečními mozky (Sprint 22) re-enable plný kontrast WorldMap (food_floor=0.4, amp=1.2 nebo dokonce rejection sampling). Otestovat, zda heterogenita teď tlačí lineages k geografické divergenci.
- **Výstup:** —

## Sprint 24+ — TBD

Možné směry:
- **Spatial speciation analytics** — CSV stats per region (svět rozdělen na N×N kvadrantů, lineage count + dominant genome per region).
- **Environmental hazards** vrstva — negative-pressure korelovaný inverzně s food_richness. Trade-off niky.
- **Reprodukční izolace** přes `genome_distance(a, b) < threshold` (NEAT-style speciation).
- **Terrain drag** (třetí WorldMap vrstva).
- **Asexuální fallback** při sparse mating density.
