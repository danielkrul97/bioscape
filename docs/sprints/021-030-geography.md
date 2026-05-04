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

## Sprint 24 — pheromone-signaling

- **Cíl:** zavést pheromone field jako mechanismus pro emergent komunikaci. Cells emitují skalární signál (`baseline` zdarma + brain-controlled `mod` s energy cost), detekují gradient. Otázka: vznikne adaptivní emisi/detekce, nebo selekce eliminuje signaling?

  **Plán:**
  - `BRAIN_INPUTS: 11 → 13` (pheromone gradient x/y), `BRAIN_OUTPUTS: 2 → 3` (emisi modulátor).
  - Reuse `SmellField` jako `pheromone` — stejný Jacobi diff + decay.
  - `emit_pheromones`: per cell `rate = BASELINE + BRAIN_MOD × max(0, output[2])`. Cost = `COST × BRAIN_MOD × max(0, output[2]) × dt`.
  - Tick order: `update_pheromone (decay)` → `brain_act (read gradient)` → `emit_pheromones (write field)`. Brain detekuje stav z konce minulého ticku — žádný self-feedback.
  - CSV: nová metrika `ph_emit` = mean `last_outputs[2].max(0)` napříč populací.

- **Konstanty:** `PHEROMONE_BASELINE_EMIT = 0.5`, `PHEROMONE_BRAIN_MOD = 1.0`, `PHEROMONE_COST_PER_RATE = 1.0`. Diffusion + decay stejné jako smell.

- **Výstup:**
  - Mechanismus implementován v lib.rs + obě binárky. 22/22 testů.
  - **Pozorovaná dynamika (seed 0, 200 gen):**
    - gen 30: `ph_emit = 0.319` (peak, cells aktivně emitují), populace 1000, lineages 22
    - gen 100: `ph_emit = 0.000` (selekce eliminovala emisi), populace 260, lineages 4
    - gen 200: `ph_emit = 0.000`, populace 235, lineages 1, size_avg 5.0 (giant regime)
- **Poznámky:**
  - **Komunikace nevznikla.** Selekce eliminovala active emission do gen 100. Free-rider problem: emitter platí cost, pole obohacuje všechny okolo. Bez specifické pressure (kin selection, signaling theory podmínky) emisi nevyplácí.
  - **Vedlejší efekt: baseline emission = "social sensor" pro predátory.** Každá živá cell přidává `BASELINE` do pole zdarma. Predátoři detekují gradient → najdou kořist → giant regime (size 5) dominuje.
  - **Sprint 23 diverzita zničena.** 20+ linií → 1. Predátorský exploit baseline pheromone překonal hazard niching. Po 200 gen jen monokultura giants.
  - **Biologicky správné chování** (komunikace je v reálu rare, vznikla evolučně jen za specifických podmínek), ale **ne to, co jsme chtěli**. Pro adaptivní signaling potřeba: kin recognition, mating signals, alarm calls s benefit pro vysílatele, nebo jiný explicit payoff.
  - **Možné fixy v Sprintu 25+:**
    - `BASELINE_EMIT = 0` → cells musí aktivně emitovat aby vznikl signál. Eliminuje predator exploit, ale field může být prázdné.
    - **Typed pheromones** (multi-channel) s explicitní rolí: alarm = decreases predator approach, mate call = increases reproduction radius.
    - **Cost na DETECTION** ne emission. Detekovat info stojí, vysílat zdarma — opačná dynamika.

## Sprint 25 — pheromone-mediated-mating

- **Cíl:** vyřešit free-rider problem ze Sprintu 24 explicitním payoffem pro emisi — **pouze cells, které aktivně emitují, mohou reprodukovat**. Plus odstranit `BASELINE_EMIT` (predator exploit Sprintu 24). Cells musí "volat" aby našly partnera.

  **Plán:**
  - `PHEROMONE_BASELINE_EMIT: 0.5 → 0.0` — žádné free-rider signaling.
  - `MATING_PHEROMONE_THRESHOLD = 0.2` — fertile filter v `reproduce` přidává podmínku `last_outputs[2] > THRESHOLD`.
  - `INNATE_PHEROMONE_BIAS = 1.0` — bias na `b2[2]` v `Brain::random` aby random brains byly už od gen 0 nad threshold (jinak ~25 % mating rate, extinction).

- **Výstup:**
  - V1 (no bias): extinkce gen 80 — random brains v polovině případů pod threshold, mating rate kvartován, populace nestihla recovery.
  - V2 (s bias 1.0): populace přežila. **Pozorovaná dynamika (seed 0, 200 gen):**
    - gen 10: `ph_emit = 0.588`, pop 105, lineages 54
    - gen 60: `ph_emit = 0.991` (saturace ~max), pop 952 (cap), lineages 5
    - gen 200: `ph_emit = 1.000`, pop 243, **lineages 1**, size_avg 5.0
- **Poznámky:**
  - **Mechanismus funguje:** mating-gated emission selektuje pro emisi. `ph_emit` MONOTÓNNĚ ROSTE z 0 na 1.0 (vs Sprint 24 kde padalo na 0). Selekce favorizuje signal — cells co nemluví, nereprodukují.
  - **Predator-eavesdropping problem zůstává.** Saturated emission (ph_emit = 1.0) znamená každá fertile cell je "loud" → predátoři je najdou přes pheromone gradient. Giant regime se vrací, lineages → 1.
  - **Klíčový research finding:** přes 4 sprinty (22/24/25-v1/25-v2) se **giants + monokultura** opakuje vždy, kdy existuje detektivovatelný cell-emitted signal. Sprint 23 (žádný pheromone) byl JEDINÝ s persistentní diverzitou (20 linií).
  - **Predátoři jsou v této simulaci zlatý strop.** Velký + rychlý cell s informací o ostatních cells dominuje vše. Niching mechaniky (hazard, pheromone) buď nestačí, nebo se obrátí proti.
  - **Možné Sprint 26+ směry pro skutečnou diverzitu:**
    - **Disable predation entirely** — vrátí se Sprint 23 dynamika. Pheromone pak může být skutečně mate-finding bez exploit.
    - **Predator gene** — cell musí explicitně být "predator type" aby mohla predovat. Generalists nepredují, jen scavenge.
    - **Pheromone receivers** — cells mají gen "kdo přijímá můj signal" (kin signaling). Predator nemá receiver, slepý k pheromone.
    - **Cost na DETEKCI místo emisi** — opačná dynamika, předator platí za sledování.

## Sprint 26 — body-morphogenesis

- **Cíl:** Tělo přestane být fixní teardrop. Replace `body_size: f32` → 2 osy (`body_length` podél heading, `body_width` kolmo) + volitelný frontální `spike_length`. Brain za běhu života kontroluje tvar přes 3 nové morph outputs (rate-limited, kontinuální). Cena: per-tick maintenance ∝ length×width + spike_length, plus okamžitý cost ∝ rychlost morfingu. Genotyp/fenotyp split: runtime morph mění `Phenotype` na cell, NE `Genome` — dítě dostane fresh phenotype z rodičovského gen template (žádný Lamarckismus). Hypotéza: morfologické niche (streamlined hunters / round ambushers / spike defenders) zlomí monokulturu Sprintu 25.

  **Plán:**
  - **Genome refactor**: `body_size` → `body_length`, `body_width`, `spike_length` (template). `MutationConfig` rozšířit na 3 sigmas. `Genome::random` startuje izotropně (`body_length == body_width` z jednoho rolu) — žádný prior na ellipse fenotyp.
  - **Cell**: nový field `phenotype: Phenotype { body_length, body_width, spike_length }`. `Phenotype::from_genome` při spawnu/reprodukci.
  - **Brain outputs: 3 → 6**: `[3] morph_length`, `[4] morph_width`, `[5] morph_spike`. `[0..2]` (turn/thrust/pheromone) nezměněny.
  - **Runtime morph step** (po `brain_act`, před `step`): `Phenotype::apply_morph(morph[3], MORPH_RATE, dt)` aplikuje signal × rate × dt na každou dim, clamp do MIN/MAX. Cost: `MORPH_COST_PER_DELTA × |actual_delta|`.
  - **Deadzone**: `MORPH_ACTIVATION_THRESHOLD = 0.7`. |signal| pod threshold → no-op. Filtruje šum z random brain biases (cca 38 % random outputs prochází), takže jen "vědomě silné" morph signály mění tvar.
  - **Anisotropic drag** v `Cell::step`: rozložit velocity na heading-paralelní (par) vs perpendicular (perp). Cross-section: forward motion cítí width, sideways cítí length. Pro length=width=s redukuje na původní izotropní semantiku.
  - **Maintenance cost** = `area × body_cost_factor + spike_length × SPIKE_COST_PER_SEC` (plus existující v² × cost_per_v_sq, vision, angular). Pro length=width=s area=s² == původní `body_size²` cost.
  - **Eat/collision (pragmatic)**: ne OBB. Effective_radius = `(length+width)/2` jako proxy v existujících circular checks. Predation size ratio i collision pair_r používají effective_radius.
  - **Spike mechanic**: `Cell::spike_bonus_against(target)` vrací `PREDATION_GAIN × spike × SPIKE_PREDATION_BONUS` pokud cosine(heading, vector_to_target) > `SPIKE_DOT_THRESHOLD`. Volá se v `predate` jako bonus k base gainu.
  - **Renderer**: non-uniform scale `Vec3::new(length, width, 1.0)` na sdíleném teardrop meshi. Spike: per-instance MeshTag bity 16..23 = `spike_norm × 255`. Vertex shader prodlouží vertex 1 (tip) o `spike_norm × MAX_SPIKE_WORLD_PX` v world-space (mimo body scale → spike length nezávislá na body asymetrii).
  - **CSV / Stats**: `len_avg`, `wid_avg`, `asp_avg = length/width`, `asp_dev`, `spk_avg`, `spk_max` namísto starých `size_avg`/`size_dev`.

- **Konstanty:** `MIN/MAX_BODY_LENGTH = 0.3..4.0`, `MIN/MAX_BODY_WIDTH = 0.3..4.0`, `MIN/MAX_SPIKE_LENGTH = 0.0..2.0`, `MORPH_RATE = 0.02`, `MORPH_ACTIVATION_THRESHOLD = 0.7`, `MORPH_COST_PER_DELTA = 2.0`, `SPIKE_COST_PER_SEC = 0.3`, `SPIKE_PREDATION_BONUS = 0.5`, `SPIKE_DOT_THRESHOLD = 0.7`, `BRAIN_OUTPUTS = 6`.

- **Výstup:**
  - **lib.rs**: `Phenotype` struct + impl (`from_genome`, `effective_radius`, `area`, `apply_morph`); `Cell` rozšířen o `phenotype`; `Cell::step` s anisotropic drag + area maintenance + spike maintenance; `Cell::apply_morph` per-tick wrapper; `Cell::spike_bonus_against` helper. 32/32 testů passes (4 nové: morph clamping, deadzone, genotyp/fenotyp split, anisotropic drag axis-dependence, spike frontal cone, spike maintenance drain).
  - **main.rs**: importy + `apply_cell_morph` system v schedule; brain_act, predate, eat, spawn_food, collisions, reproduce, death-fade, rebuild_grid migrované na phenotype; non-uniform scale v `cell_scale`; `spike_norm` helper; sync_transforms také rebuilduje MeshTag dle phenotype.
  - **headless.rs**: `apply_morph` step v tick chain; všechny scratch arrays migrované (`body_sizes_scratch` → `radii_scratch`, nové `spike_lengths_scratch`, `headings_scratch`); CSV header + values rozšířené.
  - **shader/cell_material**: `pack_cell_tag(hue, alpha, spike_norm)`; vertex shader extends vertex 1 o spike v world-space (`world_from_local[0]` jako heading direction).
  - **Tuning iterace** (smoke runy seed 0, 100 gen):
    - v1: `MORPH_RATE=0.5` → 200→34 cells (population oscillace 200→72→500→45). Příliš rychlé.
    - v2: `MORPH_RATE=0.1`, MCPD=8.0 → extinction gen 90 (morph cost burnoval cells).
    - v3: `MORPH_RATE=0.05`, MCPD=2.0 → extinction gen 35 (random brain noise dominantní).
    - v4: + deadzone `MORPH_ACTIVATION_THRESHOLD=0.5` → extinction gen 46 (filtroval, ale stále moc).
    - **v5/v6: `MORPH_RATE=0.02`, threshold=0.7, MCPD=2.0** → seed-dependent: seed 1, 7 přežily 200 gen (final 11/472 cells); seed 0, 2, 42 extinkce gen ~45. Stochastic instabilita.
  - **Pozorování seed 7 (přežilý, 200 gen):** `len_avg` 0.99 → 3.34, `wid_avg` 0.99 → 3.38 (giants), `asp_avg` 1.00 → 1.00 (round), `spk_avg` 0.05 → 0.46 (saturoval gen 50, pak klesl), `lineages` 200 → 2. Plus user vmoz Sprint 27 (attack gating) → `predation_events` až 10892/gen v gen 99.

- **Poznámky:**
  - **Mechanismus funguje, dynamika neuspela.** Acceptance kriteria selhala napříč seedami: `aspect_avg ≈ 1.0` (round dominuje), `aspect_dev < 0.2` (variance nízká), `lineages ≤ 2` (monokultura). Sprint 25 finding ("predator je zlatý strop") se opakuje — i s morfologickou flexibilitou cells konvergují k velkému-kulatému-spikatému predátorovi.
  - **Spike + frontal cone funguje** — některé seedy ukázaly `spk_max = 2.0` u predátorů. Ale spike je doplněk ke giantness, ne alternativní strategie.
  - **Anisotropic drag mathematicaly OK** — testem ověřeno, že length=width redukuje na isotropic. Nicméně selekce nepreferuje elongaci v této souboji (giants vyhrávají).
  - **Genotyp/fenotyp split** je čistý a unit-testem ověřený. Runtime morph nemodifikuje gen → dítě startuje od rodičovského template, ne od jeho aktuálního tvaru. Žádný Lamarckismus.
  - **Možná cesta dál (Sprint 27 attack gate, Sprint 28+):**
    - User paralelně přidal **Sprint 27 attack gating** (predace opt-in přes brain output[6]) — implementováno během Sprintu 26 v lib.rs + headless.rs + main.rs. Cíl je zlomit auto-predator dynamiku.
    - **Asymetrický body cost**: penalizovat length≠width méně než area, aby selekce favorizovala střídmou anisotropii.
    - **Drag-related advantage** pro elongated cells: bonus k thrustu when length > width (jako "fish shape"). Currently anisotropic drag dělá elongated cell faster forward, ale food-finding nepotřebuje raw speed nad rámec mating range.
    - **Spike-only predation**: aby spike měl smysl, base predation bez spike by byl neefektivní (low gain). Forced specialization.



- **Cíl:** předaci dát aktivní brain rozhodnutí. Doteď byla predace **automatická** kontaktní side-effect (size_a > 1.3 × size_b → drain/gain per tick). Sprint 27 to mění na **opt-in volbu**: cell musí přes `output[6] > ATTACK_THRESHOLD` "zaútočit", jinak je kontakt jen kolize. Hypotéza: gating na vědomé rozhodnutí + cost zlomí predátorský zlatý strop pozorovaný ve Sprintech 22–25, protože cells, které nikdy neútočí, neplatí cenu a niched scavenger fenotyp se dá udržet.

  **Plán:**
  - `BRAIN_OUTPUTS: 6 → 7`. Output[6] = attack signal (tanh-bound, gating přes `max(0.0, output) > THRESHOLD` jako u pheromone).
  - `predate()` v headless: skip celý outer loop pro daného attackera, pokud `last_outputs[6].max(0) <= ATTACK_THRESHOLD`. SIZE_RATIO i frontal-cone (Sprint 26 spike) gating zůstávají — útočit JE volba, ale úspěch pořád závisí na tělesných parametrech.
  - Nová tick fáze `pay_attack_cost`: per cell `energy -= ATTACK_COST_PER_SEC × max(0, output[6]) × dt`. Continuous, paid bez ohledu na to, jestli k predaci došlo. Drží "claws out" — bez ceny by selekce favorizovala vždy-zapnutý attack a gating by ztratil informační hodnotu.
  - `INNATE_ATTACK_BIAS = 0.0` v `Brain::random` (na rozdíl od `INNATE_PHEROMONE_BIAS = 1.0`). Záměrně ne-pushovat — chceme měřit, jestli selekce attack chování objeví sama. Sprint 25 ukázal, že biased default zafixuje saturaci v gen ~60.
  - Diagnostika v CSV: `atk_emit` (mean `last_outputs[6].max(0)` napříč pop) + `predation_events` (počet úspěšných drain/gain párů per gen). Allow tracking adoption rate.

- **Konstanty:** `BRAIN_OUTPUTS = 7`, `ATTACK_THRESHOLD = 0.2`, `ATTACK_COST_PER_SEC = 0.5`, `INNATE_ATTACK_BIAS = 0.0`.

- **Výstup:**
  - Implementováno v `lib.rs` (consts, bias, výstupní index) a `src/bin/headless.rs` (gate v `predate`, `pay_attack_cost`, CSV columns). Lib tests 31/31 pass.
  - **Renderer (`main.rs`) zatím nemigrovaný** — výsledek paralelní rozdělané práce na Sprint 26 morph; ladí se až po stabilizaci Sprintu 26 main.rs.
  - **Experimentální měření TBD.** Klíčové otázky pro vyhodnocení:
    1. Vyvine se nenulové `atk_emit` pod selekcí, nebo zůstane ~0 (predace utlumena)?
    2. Pokud ano, klesnou `predation_events` proti pre-Sprint-27 baseline (gating funguje)?
    3. Změní se `lineages` trajektorie? Cíl: zachovat víc než 1 linii oproti Sprintům 22–25.
    4. Jaký fenotyp dominuje attackerům — lze odvodit z `len_avg`/`wid_avg`/`spk_avg` korelovaného s `atk_emit`?

- **Poznámky:**
  - **Tradeoff vs. mating gate (Sprint 25).** Tam byl bias = 1.0, protože extinkce hrozila bez emise. Tady je bias = 0, protože extinkce hrozí *opačná* — z dominance predátorů. Symetrie default-sane ale o zrcadle.
  - **Hebbian update neopěvuje attack.** Currently jen `eat_food` vyvolává reward-modulated Hebbian. Successful predation (drain hit) by mohl analogicky postnout `last_outputs[6]` reward → attack chování by se učilo i v rámci jednoho života, ne jen genetickou selekcí. Záměrně pro Sprint 27 vynecháno: chceme nejdřív otestovat **čistou genetickou selekci** na attack output. Pokud nestačí (atk_emit zůstane ~0 napříč generacemi), Sprint 28 přidá Hebbian na predaci.
  - **Co nemění Sprint 27:** drain/gain ratio (3.0/1.5), size ratio (1.3), spike frontal cone (0.7). Všechno zůstává. Mění se jenom kdo se účastní.

## Sprint 28 — recurrent memory

- **Cíl:** dát buňkám krátkodobou paměť. Doteď je brain **stateless feed-forward MLP** — každý tick nezávislý forward pass, žádná schopnost integrovat informaci přes čas. Reálné nervové systémy jsou recurrent; bez recurrence se nedá evolvovat delayed responses, oscilátory, working memory. Hypotéza: paměť otevře nové behaviorální niky (vzpomínka kde bylo jídlo, perzistentní směr po kontaktu se stěnou, oscilatorní lov), které byly dosud principiálně nedosažitelné.

  **Plán:**
  - **Elman-style RNN.** Předchozí tick `last_hidden ∈ [-1, 1]^BRAIN_HIDDEN` se feeduje zpět do dalšího ticku jako dodatečné inputs. Žádný explicitní decay — pokud chce evoluce paměť dlouho, naučí se velké recurrent weights; pokud krátkou, malé.
  - **Refactor konstant:**
    - `BRAIN_INPUTS_SENSORY: usize = 13` — beze změny, food/cell/energy/heading/smell/pheromone gradients.
    - `BRAIN_RECURRENT: usize = BRAIN_HIDDEN` — 8 recurrent slotů, jeden na každý hidden neuron.
    - `BRAIN_INPUTS: usize = BRAIN_INPUTS_SENSORY + BRAIN_RECURRENT = 21` — total brain input width.
  - **brain_act** v binárkách: po naplnění senzoriky `inputs[0..13]` přidá `inputs[13..21] = cell.last_hidden`. `forward_with_state` zpracuje 21 inputs do 8 hidden → 7 outputs jako dříve.
  - **Reset paměti při reprodukci:** dítě dostane `last_hidden = [0; BRAIN_HIDDEN]`. Genome se dědí, paměť ne. Už takhle v `reproduce` je.
  - **Mutace + Hebbian** pracují na celém `w1` matrixu (21×8) bez rozlišení sensory vs recurrent — selekce/learning rozhodne, jak silné recurrent connections mají být.
  - **Diagnostika:** nový CSV sloupec `recurrent_io = mean(|last_hidden|)` napříč pop. Adoption metric — pokud zůstane ~0, paměť není pod selekčním tlakem nebo se utlumila; pokud roste, recurrent state se aktivně používá.

- **Konstanty:** `BRAIN_INPUTS_SENSORY = 13`, `BRAIN_RECURRENT = BRAIN_HIDDEN = 8`, `BRAIN_INPUTS = 21`. Žádné nové prahy ani costy.

- **Výstup:** TBD po experimentech. Klíčové otázky:
  1. **Adoption:** roste `recurrent_io` napříč generacemi, nebo zůstává ~0?
  2. **Substituce vision smyslem:** klesne `vis_avg` (vision je expensive), když cells získají schopnost si pamatovat polohu jídla z předchozích ticků?
  3. **Hebbian feedback dynamics:** klasický eligibility-trace problém — Hebbian update na recurrent weights může vytvořit positive-feedback smyčky. Pokud se objeví energy explosions / zacyklení, Sprint 29 přidá weight clipping nebo decay.
  4. **Diverzita:** otevírá paměť nové behaviorální niky? Cíl: víc než 1 linie napříč 200+ generací.

- **Poznámky:**
  - **Proč Elman a ne Jordan?** Jordan-style by feedval `last_outputs` (7 dims) místo `last_hidden` (8). Outputs jsou information bottleneck (turn, thrust, ph_emit, morph_*, attack) — málo a sématicky úzké. Hidden je informačně bohatší a bez specifického významu, takže evoluce má víc volnosti, jak ho použít.
  - **Proč BRAIN_RECURRENT = BRAIN_HIDDEN?** Symetrie. Každý hidden neuron má vlastní paměť slot. Mohl by být menší (subset hidden state jako "exposed memory") nebo větší (multiple feedback z minulosti) — ale 1:1 je nejjednodušší a kapacita ~8 slotů stačí pro pár desítek bitů working memory.
  - **Genom roste z 13×8=104 na 21×8=168 vah w1** (+62 %). Mutation noise + Hebbian budou pomalejší konvergovat, ale to je daň za schopnost. Můžeme měřit, jestli je rozdíl viditelný v rychlosti adaptace.
  - **Backwards compat:** dokud brain_act nezačne plnit `inputs[13..21]`, recurrent kanál je zero → identické chování s pre-Sprint-28. Implementace ve dvou krocích: (1) bump konstanty, (2) wire feedback. Pokud někde zůstane fáze (1) bez (2), simulace funguje jako dřív.

## Sprint 29 — emergent clustering

- **Cíl:** umožnit buňkám se shlukovat — ale **emergentně přes brain rozhodnutí**, ne přes mechanickou adheze. Doteď selekce tlačila proti shlukování (predace na kontakt, food competition, žádný group benefit). Přidávám dva ortogonální tahy, které dají brainu (a) **informaci** o lokálním zalidnění a (b) **důvod** být ve skupině. Selekce + Sprint 28 paměť rozhodnou, jestli clustering vznikne. Hypotéza: cells s density-aware brainem v selfish-herd režimu vyvinou flocking, protože samotná kořist je atraktivnější cíl než kořist v hejnu.
  
  **Plán:**
  - **Mechanism 1 — quorum sensing (information):** nový sensory input `local_density = tanh(n_neighbors_in_vision_radius / DENSITY_NORM_COUNT)`. `BRAIN_INPUTS_SENSORY: 13 → 14`, `BRAIN_INPUTS: 21 → 22`. Brain dostává skalární info bez emise — narozdíl od Sprintu 24/25 pheromone, density input nelze zneužít predátorem (nikdo „nevolá" do prostoru, info je jen lokálně přístupná).
  - **Mechanism 2 — selfish-herd dilution (incentive):** v `predate()` se `gain` násobí `1 / (1 + DILUTION_K × n_neighbors_prey_within_HERD_RADIUS)`. Drain prey beze změny — utrpení oběti se nemění, mění se atraktivita kořisti pro útočníka. Hamilton 1971 selfish-herd: být v hejnu = méně atraktivní cíl. Při K=0.5 a HERD_RADIUS=50 dává uniform distribuce pop=200 v 1920×1080 očekávaný `herd_count ≈ 0.76`, takže dilution při náhodné populaci ≈ 0.93 (skoro žádný plošný trest); reálný cluster s 5 close neighbors snižuje gain na 1/(1+2.5)=0.29.
  - **Diagnostika v CSV:**
    - `nn_dist_avg` — průměrná nearest-neighbor vzdálenost přes pop. Uniform reference v 1920×1080: `0.5·√(A/N) ≈ 720/√N`. Pop 200 → ref ≈ 51, pop 100 → ref ≈ 72. Hodnoty výrazně pod referencí = clustering.
    - `density_avg`, `density_dev` — průměr a odchylka `local_density` inputu napříč pop. Vysoké průměr = obecně husto, vysoká odchylka = bimodální (někdo v davu, někdo sám).
  - **Záměrně nedělám:** aggregation pheromone (predator-eavesdropping ze Sprintu 24), adhesion gen (mechanické řešení obchází brain), kin recognition přes `lineage_id` (perfektní tag, biologicky nerealistické), group hunting (komplikuje food + predace mechaniku).

- **Konstanty:** `BRAIN_INPUTS_SENSORY = 14`, `DENSITY_NORM_COUNT = 3.0`, `HERD_RADIUS = 50.0`, `DILUTION_K = 0.5`. Žádné nové prahy ani brain outputs. **Pozn.:** první iterace měla `DENSITY_NORM_COUNT=10` a `HERD_RADIUS=100` — extinct gen 47, protože density input zůstával u noise floor (typ. ~0.8 sousedů ve vision_radius nikdy nesaturoval) a dilution při uniform pop dampoval predaci na 40 % (predator economy zkolabovala). Recalibrace: noise floor → meaningful signal v rozmezí 0.2–0.5, dilution při uniform ≈ 0.93 (mírný), při clusteru < 0.3 (silný).

- **Výstup:**
  - Implementováno v `lib.rs` (consts + input doc), `src/bin/headless.rs` (count v `brain_act`, herd_counts pre-compute v `predate`, dilution multiplier, 3 nové CSV sloupce), `src/main.rs` (zrcadlí headless: count, herd_counts přes spatial grid, dilution).
  - Lib tests 32/32 pass. Full `cargo build --release` clean (lib + headless + main).
  - **Experimentální měření (4 seedy × 1500 gen, default map, mating_radius=200):**

    | sim | pop trajectory | early ratio (1-200) | late ratio (801-1500) | atk_emit final | lineages |
    |---:|---|---:|---:|---:|---:|
    | 0 | 200 → 21 → 15 | 1.47 | 1.04 | 0.75 | 7 |
    | 1 | 200 → 18 → 11 | 1.45 | 1.11 | 0.63 | 5 |
    | 2 | 200 → 19 → 16 | 1.62 | 1.08 | 0.51 | 6 |
    | 7 | 200 → 429 → 632 | 1.36 | 1.14 | 0.01 | 3 |

    `ratio = nn_dist_avg / uniform_reference` (uniform_ref = `720/√N`). ratio<1 = clustering, ratio>1 = dispersion.

    **Negative result na cíl sprintu:** clustering nevzniká. Across všech 4 seedů a 1500 gens je `ratio` setrvale ≥ 1.0 — buňky jsou rozprostřenější než uniform, ne shlukovaní. Min hodnoty (0.65–0.88) jsou transient při nízkém N (statistická fluktuace, ne selekce). Frekvence `ratio < 0.95` napříč generacemi: 13–19 % v low-pop sims (šum), 0.3 % v sim_7 (vysoká pop, malá variance).

    **Brzká disperze ratio 1.36–1.62** je nejvýraznější signál — buňky **aktivně utíkají od sebe** v gen 1–200, i přes dilution incentivu. `density_avg` klesá z ~0.10 (gen 1) na ~0.02 (gen 1500) napříč 3 ze 4 seedů — cells se rozprostírají s generacemi.

    **sim_7 jako outlier:** predace se vyhasla (`atk_emit` 0.79 → 0.01), pop stabilní 600+, lineages 3. „Peaceful equilibrium" — pacifismus je lokální optimum. Ale i tam ratio=1.14, žádný clustering.

  - **Diagnóza:** dilution incentive je **slabší než predator-avoidance disincentive**. First step k clusteru (přiblížení k cizí buňce) je nebezpečný — `inputs[2,3]` + `inputs[6]` (rel_size) signalizují potenciální predátor. Dilution benefit přijde až *po* shluknutí, ale cesta tam vede přes risk. Klasický local-minimum problém: globální optimum (cluster bezpečí) není dosažitelný gradientem od náhodné distribuce, protože každý jednotlivý krok zhoršuje fitness.

  - **Co by mohlo fungovat (Sprint 30+):**
    - **Kin recognition** přes hidden tag (matching on hue ranges nebo `lineage_id`) — cells rozliší příbuzné od cizinců, blízkost ke kin není riziková. Greenbeard gene mechanika.
    - **Local food sharing** — gain z jídla částečně sdílen v HERD_RADIUS. Group benefit *bez* potřeby být už ve clustru.
    - **Density-modulated mating** — pairs v dense areas reprodukují snáz. Selekční tlak na density rovnou přes fertility.
    - **Asexuální fallback** při nízké hustotě — daughters spawnou přímo u parenta → vznikají kin patches automaticky bez behaviorálního učení.

- **Poznámky:**
  - **Vztah ke Sprintu 27:** s `INNATE_ATTACK_BIAS = 0.0` cells defaultně neútočí, takže dilution v gen 0 nemá co modulovat. Selekční tlak na clustering zapne až jak se attack chování objeví. Doporučená sekvence experimentů: (a) Sprint 27 baseline 200gen, (b) Sprint 27+29 200gen, porovnání trajektorií `atk_emit`, `nn_dist_avg`, `lineages`.
  - **Vztah ke Sprintu 28:** density input poskytuje *prostorovou* informaci, recurrent state poskytuje *temporální*. Spolu mohou cells evolvovat „zůstaň blízko, když přicházejí draví giganti" — adaptivní flocking podmíněný kontextem. Bez Sprintu 28 by paměť nebyla a clustering musel být reaktivní (jen aktuální tick), s ní může být anticipativní.
  - **Trade-off informace vs. signal:** density input neemíruje, takže ho predátor nemůže odposlouchávat (oprava designové vady Sprintu 24). Ale brain potřebuje vidět sousedy, aby je počítal — vision_radius je gating. Slepé cells (low vision) clustering nikdy nepoznají. To je biologicky správně: vidět = hodnotit dav.
  - **Dilution faktor 0.5** je střelba od boku; pokud experimenty ukážou, že clustering nenastává, A/B s K=1.0 nebo K=2.0 je nejjednodušší tuning.

## Sprint 30 — damage-signal (self-preservation prior)

- **Cíl:** dát mozku vstup, který říká „někdo tě právě poškozuje" — chybějící senzorický kanál pro pud sebezáchovy. Doteď cell „cítí" jen vlastní energii (input[4]), což je integrovaná veličina; mezi „energie 100 a klesá pomalu od pohybu" a „energie 100 a někdo mě právě kouše" nedovedl rozlišit. Bez per-tick damage signálu nemohou vyvinout reactive útěk před predátorem ani směrovou aversi vůči hazard zónám. Hypotéza: damage signál umožní emergent flee/avoid chování v rámci jednoho života (přes Hebbian + recurrent paměť) i přes generace (selekce zvýhodní genomy, které na bolest reagují útěkem).

  **Plán:**
  - **Nový input `[14] = tanh(damage_accum × DAMAGE_NORMALIZATION_GAIN)`**. `BRAIN_INPUTS_SENSORY: 14 → 15`, `BRAIN_INPUTS: 22 → 23`. Genom `w1` matrix roste 22×8 → 23×8.
  - **„Damage" = výhradně nedobrovolná ztráta energie**: predation drain (`PREDATION_DRAIN_PER_TICK` na oběť) + hazard drain (`hazard_drain(noise) × dt`). NE movement / morph / vision / spike / pheromone / attack cost — ty platí cell sama přes svoje outputs, nemá smysl si je zpětně signalizovat jako útok.
  - **`Cell::damage_accum: f32`** field — single accumulator. Predation a hazard do něj v průběhu ticku přidávají; brain_act ho čte na začátku dalšího ticku jako input[14], pak resetuje na 0. 1-tick delay konzistentní s pheromone gradient + Sprint 28 recurrent (žádný self-feedback).
  - **Headless**: paralelní `damage_deltas_scratch` v `predate()`, přímý `+=` v `apply_hazards()`. Renderer: paralelní `damage_changes: HashMap<Entity, f32>` v predaci, přímý `+=` v `apply_environmental_hazards()`.
  - **Žádný innate bias**. Mirroruje Sprint 27 ATTACK design: chceme měřit, jestli selekce naučí pud sebezáchovy sama, ne ho inboardovat. Pokud `dmg_avg` zůstane stabilně > 0 napříč generacemi a `predation_events` neklesne, signál se nevyužívá; pokud `predation_events` poklesne v reakci na rostoucí `dmg_avg`, prey se učí utíkat.

- **Konstanty:** `BRAIN_INPUTS_SENSORY = 15`, `BRAIN_INPUTS = 23`, `DAMAGE_NORMALIZATION_GAIN = 0.5`. Single predation hit (drain=3.0/tick) → tanh(1.5) ≈ 0.90 (silný impuls). Stabilní hazard ~0.008/tick → tanh ≈ 0.004 (chronický hazard zůstává neviditelný — spatial avoidance se vyvíjí přes selekční gradient na pozici, ne přes per-tick signál). Multi-attacker pile-on saturuje k 1.0.

- **Výstup:**
  - `lib.rs`: nový `pub const DAMAGE_NORMALIZATION_GAIN`, bump `BRAIN_INPUTS_SENSORY`, `Cell::damage_accum` field, init v `from_genome` + `base_cell` test helperu. Brain input docstring rozšířen.
  - `src/bin/headless.rs`: `World.damage_deltas_scratch`, populate v `predate()`, akumulace v `apply_hazards()`, čtení/reset v `brain_act()` jako `inputs[14]`. CSV header + values + write_stats akumulátor `dmg_sum` a sloupec `dmg_avg`.
  - `src/main.rs`: zrcadlí headless — `damage_changes` HashMap v predation systemu, `cell.0.damage_accum +=` v `apply_environmental_hazards()`, `inputs[14]` populace + reset v `cells_brain_act()`. Reproduce site init.
  - 32/32 lib tests pass. `cargo build --release` clean (lib + headless + main).
  - **Smoke run (seed 0, 30 gen, headless):** populace 200 → 224 (gen 30), `dmg_avg` osciluje kolem 0.010–0.015 napříč generacemi 28–30 — damage propagace funguje, hodnota je řádově v souladu s expectation (cca 1 z 100 cells hit per tick × tanh attenuation). Žádný runtime panic, CSV format-stable.
  - **Plné experimentální měření TBD.** Klíčové otázky:
    1. **Adoption inputu:** roste `dmg_avg` napříč generacemi nad pasivní baseline (= pasivní hodnota z hazardu × pasivní pravděpodobnost predace v gen 0)?
    2. **Behaviorální response:** klesnou `predation_events`/gen po nástupu nenulového `dmg_avg`? Tj. naučí se prey reagovat (utéct, herd-up, morph defense)?
    3. **Vis ↑ vs Vis ↓:** vyvíjí se vyšší `vis_avg`? Damage signál bez směru (skalár) může zvýšit hodnotu vidění — kombinace „někdo mě kouše" + „kde je predátor" otevírá flee směr.
    4. **Lineage trajectory:** zlomí se monokultura ze Sprintů 22–25? Diverzifikuje prey strategie (fast-flee, spike-defend, herd-up)?
    5. **Hazard avoidance:** klesne fraction populace v high-noise zónách? (Spatial heatmap, ne přímo CSV — interpretace přes `mean_x/y` shift.)

- **Poznámky:**
  - **Paritní design s Sprint 27 attack gate.** Tam přidán BRAIN_OUTPUT[6] = „chci útočit" (offensive volba). Tady přidán BRAIN_INPUT[14] = „někdo útočí na mě" (defensive informace). Symetrie predator/prey informačního kanálu — útočník ohlašuje úmysl, oběť cítí dopad. V pre-Sprint-30 stavu cell „neviděla, že umírá" (kromě integrovaného energy ↓), takže reactive flee bylo principiálně neevolvovatelné.
  - **Voluntary cost se záměrně nezapočítává.** Cell, která rychle plave + emituje pheromone + morphuje, by jinak měla saturovaný damage signál ze svého vlastního kola, což je informační šum (cell o tom ví přes outputs). Damage = externí, nedobrovolný.
  - **Hazard drain je per-tick zlomek single predation hit.** Chronický hazard (~0.008 vs. predation 3.0 = ratio 1:375) zůstane v tanh(GAIN×x) prakticky neviditelný. To je záměrné: hazard zóna se má cells učit vyhýbat přes spatial selekční gradient (kdo do zóny zaleze, pomalu hladoví → potomci jsou jinde), ne přes per-tick reflex. „Burst damage" (predation) si zaslouží reflex; „attrition damage" (hazard) je population-level proces.
  - **Recurrent kanál (Sprint 28) hraje klíčovou roli.** Damage je spike v jednom ticku, ale prey potřebuje persistovat „pozor!" stav přes mnoho ticků, dokud nepřejde do bezpečí. Bez recurrent paměti by cell zapomněla po jednom ticku. S ním může vyvinout „alarm trace" — recurrent neuron, který se rozpálí při damage a chvíli doznívá.
  - **Hebbian update na damage event.** Currently jen `eat_food` triggeruje reward-modulated Hebbian. Ekvivalentní pathway pro damage (negativní reinforcement: zaktivované recurrent + sensory → escape behavior) by pomohl within-life learning. Záměrně pro Sprint 30 vynecháno — chceme nejdřív vidět, jestli čistá selekce + Sprint 28 paměť dovede vyvinout reactive flee. Pokud ne, **Sprint 31** přidá Hebbian-style anti-reinforcement na damage signal.
  - **Co Sprint 30 NEMĚNÍ:** dynamika predace, hazard mechanika, mating, morph, attack gating, recurrent paměť. Jen přidává jeden bit informace do brainu. Pokud nic ve výstupech nezareaguje, nikdo o nový kanál nezájem nemá a je to **negative result** ne bug.

## Sprint 31+ — TBD

Možné směry:
- **Hebbian na damage** — fallback ze Sprintu 30 pokud čistá genetická selekce reactive flee nevyvine; anti-reinforce sensory → action pattern při damage signálu.
- **Spatial speciation analytics** — CSV stats per region (svět rozdělen na N×N kvadrantů, lineage count + dominant genome per region). Přímo testovatelná hypotéza spatial niching.
- **Predator gating** — explicit "predator gene" / type, jen specialist eats; generalist nemůže. Eliminuje zlatou-strop dynamiku.
- **Reprodukční izolace** přes `genome_distance(a, b) < threshold` (NEAT-style speciation) — pojistka pro persistenci diverzity.
- **Mobilní hrozby** — wandering predator entities, navigační AI challenge (brain musí zpracovat hrozbu jako další gradient).
- **Multi-food types** — různé typy jídla s různými energetickými profily, food-side niching.
- **Terrain drag** (třetí WorldMap vrstva) — pohyblivost varíuje s pozicí.
- **Asexuální fallback** při sparse mating density.
- **Hebbian na predaci** — fallback ze Sprintu 27 pokud čistá genetická selekce nestačí; reinforce attack output při úspěšném drainu.
- **Recurrent stability** — fallback ze Sprintu 28 pokud Hebbian na recurrent vahy způsobí positive-feedback explosions: weight clipping, hidden-state decay, nebo Hebbian update jen na sensory část w1.
- **Adhesion fallback** — pokud Sprint 29 emergentní clustering selže, zvážit gen `adhesion_strength` + matching tag jako mechanické řešení; ale jen jako last resort, obchází brain rozhodování.
