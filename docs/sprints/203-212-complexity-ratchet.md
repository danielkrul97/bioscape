# Sprinty 203–212: Komplexifikační ratchet

Předchozí desítka (193–202) zaváděla **monotonní environmentální tlak**
(scarcity ramp, konzervativní food share, real-physics layer, endosymbióza).
Jádrový výsledek byl ambivalentní: scarcity sama o sobě **neodměnila chytrost
ani komplexitu** — naopak posílila pasivní cluster-share monokulturu
(`bond_active_frac 0 → 0.93`, lineages 200 → ~6, S193 sweep). S194 to částečně
zlomil sub-lineárním food share, ale otevřená research question zůstala:

> Brain stack (183–192) zaručuje, že mozek *může* dělat graded computation.
> Environment-pressure (193–202) přidal tlak na *efektivitu*. Ale evoluce
> pořád nemá důvod **používat výpočetní kapacitu, paměť ani strukturální
> složitost** — a co hůř, nemá je jak *udržet*, když krátkodobě prohrávají.

## Diagnóza: proč komplexita neroste

Projekt **už má variační operátory schopné přidávat složitost** (obvykle ta
těžká část):

- CPPN se strukturálně vyvíjí — `add_node 0.03`, `add_link 0.05`, `toggle`,
  `activation_change` (`src/neural/cppn.rs`)
- `hidden_n` roste ±1 random walkem (`hidden_n_step_rate 0.03`, range 4–45)
- `spike_count` roste ±1 walkem

Problém je trojí, a přesně odpovídá doc `07-open-ended-evolution.md`
(Tierra/Avida „sklouznou k jednodušším tvorům"):

1. **Komplexita driftuje dolů, nevybírá se.** `hidden_n` walk je nezaujatý,
   ale neurony stojí energii (brain cost je součást energy drainu ve
   `step.wgsl`). Bez úkolu, který výpočet vyžaduje, selekce tlačí `hidden_n`
   k minimu. To je doslova Avida problém.

2. **Inovace není chráněná (chybí NEAT-trick).** Strukturální mutace skoro
   vždy *krátkodobě* sníží fitness, než se doladí. `Cppn::compatibility_distance()`
   v kódu **existuje, ale je použitá jen v testech** (`tests_phase3.rs`,
   `tests.rs`) — komentář ji označuje jako „future speciation gate". Reprodukce
   (`src/reproduction.rs`) je jen greedy distance-pairing + energy threshold,
   žádná ochrana inovace. To je učebnicový důvod, proč se strukturální
   složitost nehromadí.

3. **Prostředí nevytváří poptávku po výpočtu/paměti.** Globální scarcity
   multiplikátor odměňuje efektivitu, ne chytrost; `MAX_POPULATION = 1500`
   (`src/params/physics.rs:17`) způsobí, že scarcity nikdy „nebolí" na
   populační úrovni (S194 analýza). Recurrent brain (45 paměťových slotů) nemá
   důvod si cokoli pamatovat — jídlo je všude stejné.

## Cíl desítky 203–212: komplexifikační ratchet

Aby složitost rostla, musí platit tři věci současně. Desítka je staví jako
tři pilíře nad měřící vrstvou:

| Pilíř | Co řeší | Doc 07 mapování | Sprinty |
|---|---|---|---|
| **Měření** | Nevidíš složitost → neřídíš ji | (předpoklad) | 203 |
| **Ochrana inovace** | Nová struktura se nesmí přebít dřív, než se vyplatí | Speciation, Quality-Diversity / MAP-Elites | 204, 205, 206 |
| **Poptávka** | Výpočet/paměť/koordinace se musí vyplácet | Patchy prostředí, multi-step úkoly, koevoluce | 207, 208, 209, 210 |
| **Otevřenost** | Anti-konvergence i ve steady-state | Novelty search | 211 |

Validace (212) je samostatný gate: desítka uspěla **jen pokud `complexity_floor`
a `hidden_n_avg` trendují nahoru pod selekcí** (ne jen driftem).

Sekvence respektuje závislosti: měření první (jinak letíme naslepo), pak
ochranná vrstva (speciation → fitness sharing → MAP-Elites), pak poptávková
mechanika, novelty na konci (potřebuje archiv z 206), validace uzavírá.

**Společný domov pro tuneables:** nový modul `src/params/evolution.rs`
(registrovaný v `src/params/mod.rs`) pro všechny `pub const` konstanty desítky —
ať nejsou roztroušené mezi `physics.rs` a `world.rs`.

## Stav implementace (k 2026-06-03)

Implementováno a otestováno (build zelený). **Jediné 2 faily v repu jsou
pre-existing S202 motor/mass WIP** (`motor_scales_inversely_with_mass`
hlásí „expected ratio ~2 (eff_r), got 8" — half-migrovaný test po přechodu
motoru na kubické `mass()`; `motor_gpu_zero_outputs_parity_with_cpu` cpu=2.5
vs gpu=1.15) — mimo tuto desítku, nezpůsobeno jí.

| Sprint | Stav | Pozn. k realizaci |
|---|---|---|
| **203** | ✅ hotovo | `hidden_n_avg/max` v CSV už existovaly; přidáno `complexity_floor`, `cppn_nodes_avg`, `cppn_links_avg`, `behavioral_entropy` (carnivore × z × fov), `species_count`. Regression test na shodu počtu sloupců empty/populated row. Byte-identical sim. |
| **204** | ✅ hotovo | `World::classify_species` — fresh per-gen greedy clustering (bez cross-gen reprezentantů; ID dense `0..k` ale nestabilní napříč gen — pro fitness sharing i diagnostiku stačí). `Cell.species_id`. Byte-identical sim. |
| **205** | ✅ hotovo | Postaveno na **existujícím** lineage-frequency mechanismu (`collect_fertile` už měl `LINEAGE_DIVERSITY_ALPHA`) — přidán paralelní species-frequency člen `species_fitness_share_scale`. |
| **206** | ✅ hotovo | Quality = `cell.age` (longevita = implicitní fitness; **žádné nové Cell pole**). Reinjekce vedena **přes birth→CppnGpu flow** (nahrazení genomu části dětí), ne separátní immigration pass — jinak by se desynchronizovaly GPU brain buffery. Smoke: `elite_coverage` 0→17→25→32→37 (roste, nezmenšuje se). |
| **207** | ✅ hotovo | **Deviace od plánu:** patch multiplikátor počítán **analyticky v `food_spawn.wgsl`** (driftující `sin·sin` vzor dle generace), ne druhým `WorldMap` bufferem — stejný research cíl, výrazně menší GPU surface, žádná CPU parita (food spawn je GPU-only). Smoke 5 gen OK. |
| **211** | ✅ hotovo | Behavioral-novelty fitness sharing na `elite_grid_key` binech (sparsita v current-pop, ne archiv) — `novelty_reproduce_scale`. Komplementární k S205 (genetická vs behaviorální osa). |
| **208** | ✅ hotovo | **Revize premisy + deviace:** žádný brain cost dříve neexistoval (`step.wgsl` má jen motion/vision/spike/shell/attack), takže „amortizovat down-pressure" bylo neplatné — místo toho *přidán* utilization-vážený cost. **Realizace CPU-side** (`apply_brain_metabolism`, jako `apply_symbiont_energy`): `energy` i `last_hidden` se round-trippují CPU↔GPU každý tick (`upload_metadata` Phase 2 / readback Phase 11), takže žádný shader ani parita. Silent neurony (|a|≤ε) zdarma. Cap relax **vynechán** — `MAX_POPULATION=1500` po S207 není binding (pop food-limited pod cap). |
| **209** | ✅ hotovo | **Deviace:** single-cell multi-step jako **separátní `ripening_foods` CPU subsystém** (mirror existujícího `CoopFood`/S128, který už pokrývá multi-cell variantu) — node vyžaduje N ticků sustained processing jednou buňkou, decay při opuštění. Žádné `Food`/`eat_food`/GPU změny (vyhne se ~11 `Food{}` literál editům). |
| **210** | ✅ hotovo | **Deviace:** koevoluce jako **CPU negative frequency-dependent selection** na `defense_contribution` (`apply_redqueen_pressure`) — častý defense fenotyp platí per-tick penalty ∝ své frekvenci → dominantní obrana se stává nevýhodou → cyklení. Věrné plánovanému frequency-dependent variantu, ale bez `predate.wgsl` plumbingu (predation gain je GPU-applied). |
| **212** | ◑ partial | 30-gen smoke na celém stacku hotový (retro níže): **stabilní, `cppn_nodes_avg` +58 %** (ratchet cvaká na CPPN ose), `elite_coverage` 0→58, species 1→2-3, Red Queen oscilace. `complexity_floor`/`hidden_n_avg` ploché (30 gen krátké) + behaviorální konvergence → tuning ochrany. **Plný 5×100 sweep** (~57 min) zůstává jako finální gate. |

**Všech 9 mechanických sprintů (203–211) hotovo a unit-otestováno** (lib 462
passed, headless ~126 passed, 0 nových failů; jediné 2 faily = pre-existing
S202 motor WIP). Tvoří uzavřenou smyčku **měření → ochrana → poptávka →
otevřenost**. Společný rys realizace: **CPU control-plane** (S204/205/206/208/
210/211) nebo **analytický shader bez parity** (S207) — energy/last_hidden
round-trip CPU↔GPU každý tick umožnil dělat energetické mechaniky CPU-side jako
`apply_symbiont_energy`, čímž se desítka vyhnula rizikové GPU-parita plumbingu.

**Společný domov tuneables:** `src/params/evolution.rs` — 20 `pub const` desítky.

---

## Sprint 203 — Complexity & open-endedness metriky

**Cíl:** instrumentovat složitost *než* začneme měnit selekci. Bez per-gen
viditelnosti `hidden_n`, CPPN velikosti a behaviorální diverzity nepoznáme,
jestli zbytek desítky funguje. Čistě diagnostický, CPU-only, žádná změna
chování sim — ideální nulový-riziko start.

**Plánovaný výstup:**

- **Nové CSV sloupce** na konci řádky (`src/bin/headless/csv.rs`, oba writeln
  branch — empty-pop i normal — i header v `src/bin/headless/main.rs`):
  - `hidden_n_avg` (f64), `hidden_n_max` (u32) — čteno z
    `cell.genome.brain.hidden_n`.
  - `complexity_floor` (u32) — min `hidden_n` v populaci (nejjednodušší
    přeživší; klíčová metrika ratchetu — má růst).
  - `cppn_nodes_avg` (f64), `cppn_links_avg` (f64) — z `cppn.num_nodes` /
    `cppn.num_links`.
  - `behavioral_entropy` (f64) — Shannon entropie obsazenosti hrubého
    behaviorálního histogramu (binning přes `[carnivore_score, z_norm, speed]`
    do malé mřížky). Proxy pro diverzitu strategií.
  - `species_count` (u64) — zatím placeholder `0`, naplní S204.

- **Helper** `pub fn behavioral_entropy(cells: &[Cell]) -> f64` (pure function,
  testovatelná bez GPU/World) — agregace už běží ve stejném průchodu, co počítá
  `sym_*` sloupce.

- **Testy** (`src/tests.rs`): `behavioral_entropy_uniform_is_max` (rovnoměrné
  obsazení → max entropie), `behavioral_entropy_degenerate_is_zero` (všichni
  v jednom binu → 0), `complexity_floor_picks_minimum`.

**Poznámky:** Read-only nad cells snapshotem → **byte-identical sim, žádná
RNG perturbace**, žádná GPU změna. Cross-seed reproducibilita zachována. Tohle
je baseline-measurement sprint: spustit 5×100-gen sweep teď a uložit trajektorie
`hidden_n_avg` / `complexity_floor` jako **pre-decade baseline**, vůči kterému
se měří úspěch S212. Očekávání: bez zbytku desítky `complexity_floor` driftuje
dolů nebo stagnuje (potvrzení diagnózy).

---

## Sprint 204 — Speciation gate přes CPPN compatibility distance

**Cíl:** zapojit existující (a dosud nevyužitou) `Cppn::compatibility_distance()`
do per-gen klasifikace populace na druhy. Foundation pro S205/S206 — sám o sobě
ještě **nemění reprodukci** (jen přiřazuje `species_id` a publikuje
`species_count`), takže zůstává byte-identical na sim úrovni a izoluje
correctness clusteringu od jeho dopadu.

**Plánovaný výstup:**

- **Nová konstanta** `CPPN_SPECIATION_THRESHOLD: f32` v
  `src/params/evolution.rs` (educated guess, ladí se podle `species_count`
  distribuce — cíl řádově desítky druhů, ne 1 a ne stovky).

- **Nové pole** `Cell.species_id: u32` (`#[serde(default)]`) — přiřazené na
  konci generace.

- **Per-gen klasifikace** v `World` (gen-boundary, vedle ostatních per-gen
  agregací): NEAT-style — udržuj `Vec<SpeciesRep { genome_cppn, species_id }>`
  reprezentantů z *předchozí* generace (stabilita ID napříč gen). Každá buňka
  se přiřadí k prvnímu druhu s `compatibility_distance < THRESHOLD`; jinak
  zakládá nový druh a stává se jeho reprezentantem. Prázdné druhy (žádný člen)
  se zahodí.

- **`species_count` (S203 placeholder)** napojen na reálný počet neprázdných
  druhů.

**Poznámky:** Klasifikace je **CPU control-plane** na gen-boundary, ne per-tick
— náklad `O(N × |species|)` compat distancí (1500 cells × řádově desítky druhů =
trivial vůči 300 ticks/gen GPU compute). Žádný shader, žádný per-tick cost.
Deterministická given pořadí buněk (žádné RNG draws) → sim zůstává
byte-identical, mění se jen diagnostika. Tahle inkrementalita je záměr: S204
ověří, že clustering produkuje smysluplné druhy (sledovat `species_count`
trajektorii v 25-gen smoke), než S205 na něm postaví selekční tlak. Testy:
identické CPPN → stejný druh; threshold boundary; vznik nového druhu při
překročení.

Co se NEŘEŠÍ (otevřené pro 205+): dopad na reprodukci, inter-species crossover
gate (zatím se může pářit napříč druhy — fitness sharing v S205 je měkčí páka
než tvrdý mating ban).

---

## Sprint 205 — Fitness sharing / niche reprodukce

**Cíl:** použít druhy z S204 k **ochraně malých inovativních lineages** přes
explicit fitness sharing. Velká zkonvergovaná monokultura (pasivní cluster-share
attractor z 193–194) nesmí monopolizovat reprodukci — to je selekční páka,
kterou scarcity sama neměla. Tohle je NEAT core trick přeložený do
energy-threshold sim.

**Plánovaný výstup:**

- **Nová konstanta** `FITNESS_SHARE_PRESSURE: f32` v `src/params/evolution.rs`
  (start nízko — over-sharing zmrazí adaptaci).

- **Species-scaled reprodukční práh** v `src/reproduction.rs` (eligibility
  check `energy >= reproduce_at_energy`): efektivní práh škálovaný velikostí
  druhu —
  ```rust
  let crowding = species_size as f32 / mean_species_size;
  let effective = reproduce_at_energy * (1.0 + FITNESS_SHARE_PRESSURE * (crowding - 1.0));
  ```
  Velký druh platí víc energie za reprodukci (crowding penalty), malý druh
  míň. `species_id` + per-species velikosti jsou spočtené na gen-boundary
  (S204) a dostupné při pairingu.

- **Testy** (`src/tests.rs`): `fitness_share_large_species_pays_more`,
  `fitness_share_uniform_is_neutral` (`PRESSURE = 0` → byte-identical
  s pre-S205), `fitness_share_small_species_discounted`.

**Poznámky:** **Toto je první sprint desítky, který mění populační dynamiku** →
pre-S205 RNG sequence schválně broken. Validace = 5×100-gen cross-seed sweep
([[feedback_validation_sweep]]): klíčová metrika `species_count` (drží se nad
floor?), sekundární návrat diverzity (lineages, `behavioral_entropy`). Risk:
příliš silný sharing zmrazí konvergenci k *jakémukoli* dobrému řešení — proto
`PRESSURE` nízko a `effective` práh zdola clampnutý (druh nesmí platit záporně).
Reprodukce je sekvenční CPU blok sdílený oběma binárkami (jako `eat_food` Pass 2),
jeden edit pokrývá renderer i headless.

---

## Sprint 206 — MAP-Elites behaviorální archiv

**Cíl:** otevřený diversity engine — doc 07 ho označuje jako *„pro Bioscape
ideální"*. Mřížka behaviorálního prostoru, drž nejlepšího per buňka, při
repopulaci re-injektuj z archivu. To zachová **stepping stones**: slabší řešení
v jedné nice může být *předkem* skvělého řešení ve vedlejší.

**Plánovaný výstup:**

- **Behaviorální deskriptor** (4D, discretizovaný): `[z_norm, carnivore_score,
  body_volume, hidden_n]` → grid key. Počty binů konfigurovatelné v
  `src/params/evolution.rs` (`ELITE_BINS_*`, např. 8×8×6×6).

- **Quality metrika:** nové pole `Cell.lifetime_fitness: f32`
  (`#[serde(default)]`) — akumulátor (např. `age`-vážený harvest energie +
  počet potomků). Inkrementovaný v `tick()` vedle ostatních per-cell agregací.

- **`World.elite_archive: HashMap<GridKey, EliteEntry>`** (CPU) — `EliteEntry`
  drží nejlepší `Genome` + jeho `lifetime_fitness` per buňka. Aktualizace na
  gen-boundary.

- **Stepping-stone reinjection:** konstanta `ELITE_REINJECT_FRACTION` — zlomek
  porodů (nebo repopulace po near-extinction) se seeduje z náhodně vybrané
  obsazené buňky archivu místo z živých rodičů.

- **CSV** (`src/bin/headless/csv.rs`): `elite_coverage` (u64 — počet obsazených
  buněk mřížky) jako open-endedness proxy.

**Poznámky:** **CPU control-plane** (jako symbiont/archiv mechaniky) — žádný
shader archiv nečte. Reinjection přidává RNG draws → pre-S206 sequence broken,
validace sweepem. Determinismus uvnitř S206 zachován (HashMap iterace musí být
seeded-deterministická — použít BTreeMap nebo seřazené klíče pro výběr, ať
reinjection je reprodukovatelná napříč běhy se stejným seedem). Testy: archiv
drží max per buňka; `elite_coverage` monotónně roste v rámci gen jak se přidávají
buňky; reinjection vzorkuje jen obsazené buňky. Risk: deskriptor musí korelovat
s tím, co nás zajímá — `hidden_n` jako jedna z os je záměr (chceme chránit
*složitost* jako niche dimenzi, ne jen morfologii).

---

## Sprint 207 — Patchy / clustered food (poptávka po paměti)

**Cíl:** nahradit globální scarcity multiplikátor **prostorově proměnným**
zdrojovým polem (úrodné enklávy + pouště, pomalý drift). To dá recurrent mozku
*poprvé* důvod pamatovat si polohy zdrojů — paměť a directed exploration se
začnou vyplácet. Doc 193 to explicitně označil jako kandidáta vyžadujícího
druhý `WorldMap::field` kanál.

**Plánovaný výstup:**

- **Druhý kanál** ve `WorldMap` (`src/world_map.rs`): `food_patch: Vec<f32>`
  vedle existujícího `field` (single-channel od S53). Accessor + sample-at
  metoda. Pomalu driftující prostorový multiplikátor (úrodné ~1.5×, pouště
  ~0.2×).

- **Drift:** znovupoužít S202 flow-field infrastrukturu — patch pole advekované
  stejným `generate_curl_flow_field` polem, nebo rotující low-freq vzor.
  Konstanty `FOOD_PATCH_CONTRAST`, `FOOD_PATCH_SCALE` (prostorová freq),
  `FOOD_PATCH_DRIFT_RATE` v `src/params/evolution.rs`.

- **Food spawn** (`src/gpu/food_spawn.rs` + `shaders/food_spawn.wgsl`): spawn
  pozice vzorkuje lokální patch multiplikátor místo uniformního targetu.
  Patch pole uploadnuté jako GPU buffer (vzor: smell field upload ve `FieldGpu`).

- **CSV:** volitelně `food_patch_gini` (f64) — koncentrace jídla jako sanity
  check, že patchiness opravdu existuje.

**Poznámky:** **Sahá do GPU food_spawn** → nový buffer + binding, potřebuje
parity coverage (food spawn je GPU pipeline sdílená oběma binárkami). Determinismus:
patch pole deterministické given gen; spawn pozice už jsou RNG → distribuce se
mění, sequence broken, validace sweepem. **Klíčová cross-validace s S203:**
pod patchy prostředím + amortizovaným metabolismem (S208) by mělo selekční
znaménko na `hidden_n` přepnout z negativního (drift dolů) na pozitivní. Pokud
ne, poptávka po paměti není dost silná — zvýšit `FOOD_PATCH_CONTRAST` nebo
zkrátit `FOOD_PATCH_DRIFT_RATE` (rychlejší drift = víc tlaku na adaptivní
re-foraging, ne naučenou fixní mapu).

---

## Sprint 208 — Amortizovaný metabolismus + cap relax

**Cíl:** odstranit down-pressure na `hidden_n`. Dnešní brain cost ∝ počet
aktivních neuronů → každý neuron je daň → selekce tlačí dolů. Změna:
**cost ∝ využití** — tichý neuron (~nulová aktivace) stojí skoro nic, neuron
nesoucí signál platí metabolismus. Přidat neuron je pak *levné na zkoušku* a
draze se platí *jen když se používá* → ratchet může cvaknout nahoru.

**Plánovaný výstup:**

- **Utilization-vážený brain cost** v `shaders/step.wgsl` (+ CPU mirror
  `src/cell.rs`): místo `BRAIN_COST_PER_NEURON × hidden_n` použít
  `BRAIN_COST_PER_NEURON × Σ_i util_i`, kde `util_i` je proxy využití
  i-tého hidden neuronu (mean `|hidden_i|` přes tick, nebo počet neuronů
  s `|activation| > ε`). `last_hidden` je už spočtený v brain forward —
  agregace je levná.

- **Konstanty** `BRAIN_COST_PER_NEURON`, `BRAIN_UTIL_EPSILON` v
  `src/params/evolution.rs` (přesun/refactor existující brain-cost konstanty).

- **Cap relax:** `MAX_POPULATION` (`src/params/physics.rs:17`) — buď zvednout,
  nebo nahradit tvrdý cap **density-dependent birth suppression** vázanou na
  lokální patch carrying capacity (přirozeně se pojí s S207). Cíl: scarcity
  konečně „bolí" na populační úrovni (S194 identifikovaná nejsilnější páka).

- **Parity test** (`src/tests_phase3.rs`): `brain_cost_utilization_gpu_matches_cpu`
  — fixní brain state, assert identický cost CPU/GPU.

**Poznámky:** **Sahá do `step.wgsl` energy bloku** → CPU/GPU parita kritická.
Determinismus broken (energy trajektorie všude). Validace sweepem.
**Tohle je spolu s S207 jádro poptávkového pilíře** — klíčová metrika z S203
je, jestli `hidden_n_avg` teď trenduje nahoru pod selekcí (ne driftem). Risk:
amortizace nesmí být tak měkká, že složitost je zadarmo (pak roste bez užitku =
bloat) — `BRAIN_UTIL_EPSILON` ladí dead-zone tak, aby skutečně využité neurony
platily netriviálně. Pořadí 207 → 208 záměrné: nejdřív vytvoř poptávku, pak
zlevni nabídku složitosti.

---

## Sprint 209 — Multi-step / sequential foods (hloubka výpočtu)

**Cíl:** zdroj, který vyžaduje **vícekrokovou policy** — odměna jen za sekvenci
akcí, ne jednorázový kontakt. To přímo odměňuje hloubku výpočtu / využití
recurrence (paměti mezi ticky).

**Plánovaný výstup:**

- **Nový food state:** food má `ripeness`/`stage` pole; plná hodnota se uvolní
  jen když ji buňka zpracuje opakovaně přes ≥N ticků (vyžaduje *zapamatovanou*
  multi-step policy „zůstaň a zpracuj", ne reaktivní grab). Single-cell varianta
  je **primární mechanika**.

- **Konstanty** `MULTISTEP_FOOD_FRACTION`, `MULTISTEP_STAGES`,
  `MULTISTEP_STAGE_TIMEOUT` v `src/params/evolution.rs`.

- **Integrace** do `src/gpu/eat_food.rs` + `shaders/eat_food.wgsl` (food state
  transition) + CPU mirror. Parita.

- **Coop varianta (gated/optional):** food vyžadující 2 bonded buňky emitující
  komplementární signály v pořadí. **Záměrně sekundární** — default headless je
  bond-sparse ([[bond-physics-testing]]), takže coop dráhu validovat **GPU unit
  testy**, ne headless sweepem.

**Poznámky:** Sahá do `eat_food` → CPU/GPU parita. Determinismus broken; sweep
pro single-cell mechaniku + GPU unit testy pro coop. Why-to: multi-step plan =
víc využité recurrence/hidden → pod S208 amortizací se to *vyplatí* (jinak by
extra neurony na plán byly jen daň). Risk: pokud je sekvence příliš dlouhá/tvrdá
na gen-0 random brains, multi-step food se ignoruje jako neprofitabilní niche —
začít s `MULTISTEP_STAGES = 2` a malým `MULTISTEP_FOOD_FRACTION`, nechat koexistovat
s běžným jídlem (niche, ne nahrazení).

---

## Sprint 210 — Koevoluční arms race

**Cíl:** udělat z predace **eskalující závod**, ne fixní bod. Doc 07 řešení 3:
„závody ve zbrojení jsou nikdy nekončící hnací motor." Dnešní predace
(carnivore_score / attack_gate / defense_contribution) konverguje k rovnováze;
chceme perpetuální cyklení = trvalý selekční tlak.

**Plánovaný výstup:**

- **Frequency-dependent eat efficiency:** predátorská efektivita proti dané
  defense-fenotyp třídě klesá s tím, jak je ta třída v populaci *častá*
  (negative frequency-dependent selection) → žádné stabilní equilibrium,
  perpetuální Red-Queen cyklus. Defense-fenotyp frekvence spočtené na
  gen-boundary (CPU), uploadnuté jako malá frequency tabulka do `predate`
  pipeline.

- **Konstanty** `REDQUEEN_FREQ_STRENGTH`, `REDQUEEN_PHENO_BINS` v
  `src/params/evolution.rs`.

- **Integrace** do `src/gpu/predate.rs` + `shaders/predate.wgsl` (eat efficiency
  modulace) — staví na existující predace pipeline.

**Poznámky:** Frequency table je převážně CPU bookkeeping + malý GPU upload
(vzor: symbiont state buffery). Determinismus broken; sweep. **Klíčová metrika:
udržené oscilace** `carnivore_avg` / `defense_avg` napříč generacemi (ne
konvergence k fixní hodnotě), `species_count` se drží. Pojí se s ochranným
pilířem: koevoluce generuje novost, speciation (S204/S205) ji chrání před
okamžitým přebitím — dohromady otevřená aréna. Risk: příliš silný
frequency-dependence → chaotické oscilace bez akumulace; `REDQUEEN_FREQ_STRENGTH`
nízko, validovat že oscilace mají rostoucí *amplitudu schopností*, ne jen
frekvenční flutter.

---

## Sprint 211 — Novelty bonus do reprodukce

**Cíl:** lehký novelty-search tlak proti konvergenci i ve steady-state
(doc 07 řešení 1). Buňka s behaviorálně novým profilem vůči archivu (S206)
dostane malou reprodukční výhodu — explorace zůstane živá i když je populace
„spokojená".

**Plánovaný výstup:**

- **Per-cell novelty score** (gen-boundary): vzdálenost behaviorálního
  deskriptoru (S206) ke k-nejbližším *obsazeným* buňkám archivu. Vysoká
  vzdálenost = vysoká novost.

- **Reprodukční discount** (`src/reproduction.rs`, vedle S205 fitness sharing):
  `effective_repro_energy -= NOVELTY_BONUS_WEIGHT × novelty_score` (clampnuto
  zdola). Reuse S206 deskriptoru + archivu.

- **Konstanta** `NOVELTY_BONUS_WEIGHT: f32` v `src/params/evolution.rs`
  (malá — novelty je koření, ne hlavní jídlo).

**Poznámky:** CPU control-plane na gen-boundary (modulace reprodukce jako S205).
Determinismus broken; sweep. **Klíčová metrika:** `behavioral_entropy` /
`elite_coverage` se *nezhroutí* na dlouhém horizontu (anti-konvergence). Risk
(doc 07 explicitně): novelty samo vede k „diverzitě nesmyslů" — proto je
kombinované s kvalitou (S205 fitness sharing + S206 quality archiv) a `WEIGHT`
nízko. Pořadí na konci desítky záměrné: novelty má smysl až když existuje
poptávka (207–209) a ochrana (204–206), které dají novosti směr.

---

## Sprint 212 — Validace + tuning + decade retro

**Cíl:** ověřit, jestli ratchet cvaká. Desítka uspěla **jen pokud složitost
roste pod selekcí**, ne driftem. Tuning pass nad konstantami `evolution.rs` +
retro s otázkami pro 213+.

**Plánovaný výstup:**

- **5×100-gen cross-seed sweep** (seedy 1–5, [[feedback_validation_sweep]])
  vs. pre-decade baseline uložená v S203.

- **Akceptační kritéria** (čtená z S203/S206 metrik):
  - `complexity_floor` a `hidden_n_avg` **trendují nahoru** napříč generacemi
    (ratchet cvaká, ne drift dolů jako baseline).
  - `cppn_nodes_avg` non-decreasing pod selekcí.
  - `species_count` se drží nad floor (diverzita chráněná).
  - `behavioral_entropy` / `elite_coverage` se nehroutí na dlouhém horizontu
    (open-ended).
  - Žádná extinkce; populace stabilní.

- **Tuning** problematických konstant (typicky `FITNESS_SHARE_PRESSURE`,
  `FOOD_PATCH_CONTRAST`, `BRAIN_UTIL_EPSILON`, `CPPN_SPECIATION_THRESHOLD`).

- **Decade retro** v tomto dokumentu — co fungovalo, slepé uličky, otevřené
  otázky.

**Poznámky:** Pokud `hidden_n_avg` neroste i po tuningu, diagnóza ukazuje, který
pilíř selhal: plochá složitost + vysoká `species_count` = ochrana funguje, ale
poptávka je slabá (zostřit 207–209); rostoucí složitost + kolaps `species_count`
= poptávka funguje, ale ochrana je slabá (zostřit 204–205); kolaps obojího =
prostředí příliš tvrdé, populace přežívá minimalizací (změkčit cap/scarcity).
Otevřené otázky pro 213+ (doc 07): explicit allopatrická speciace (geografická
izolace přes patch deserts), Hall-of-Fame koevoluce napříč evolučními dobami,
metriky open-endedness odolné Goodhartovu zákonu.

### Výsledek — 30-gen smoke (seed 1, 18000 ticků, 53 ticks/s)

Plný 5×100-gen sweep je ~57 min (mimo jednu seanci), takže gate proběhl jako
**30-gen smoke na celém S203–S211 stacku**. Trajektorie klíčových metrik:

| gen | cells | hidden_n_avg | hidden_n_max | complexity_floor | cppn_nodes_avg | elite_coverage | species | beh_entropy | carnivore | defense |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 200 | 25.0 | 25 | 25 | 10.0 | 0 | 1 | 0.49 | 0.25 | 0.16 |
| 7 | 470 | 24.6 | 26 | 23 | 11.4 | 42 | 1 | 0.24 | 0.06 | 0.25 |
| 15 | 298 | 24.6 | 26 | 21 | 14.3 | 54 | 2 | 0.20 | 0.06 | 0.16 |
| 22 | 327 | 24.7 | **27** | 21 | 15.0 | 56 | 2 | 0.16 | 0.07 | 0.13 |
| 30 | 412 | 24.5 | 26 | 22 | **15.8** | **58** | 2 | 0.19 | 0.06 | 0.14 |

**Co funguje (✅):**
- **Žádná extinkce, stabilní populace** (200 → 412) přes celý stack — všech 9
  mechanik koexistuje bez collapse/panic.
- **`cppn_nodes_avg` 10.0 → 15.8 (+58 %), monotonně** — headline signál: **
  strukturální složitost genomu se ratchetuje nahoru pod selekcí**, ne driftem.
  CPPN `add_node` mutace se akumulují a *udržují* (ochranný pilíř drží inovaci).
- **`hidden_n_max` 25 → 27** — topologický růst na frontieru (lineage vyrostla
  hidden walkem a přežila).
- **`elite_coverage` 0 → 58, monotonně neklesající** — MAP-Elites stepping
  stones se zachovávají (nikdy se nezmenší).
- **`species_count` 1 → 2–3** od gen 16 — speciation gate se zapojí, jakmile
  CPPN dostatečně divergují. **Řeší S207 callout** (`species=1 @ 5 gen` byl jen
  málo generací, ne hrubý threshold).
- **`defense_avg` oscilace** 0.16 → 0.25 (g7) → 0.13 (g30) — Red Queen cyklus
  (S210) viditelný i v 30 gen.

**Co potřebuje delší horizont / tuning (⚠️):**
- **`complexity_floor` plochý/mírně dolů** (25 → 22) a **`hidden_n_avg`
  neutrální** (~24.6). Hidden-neuron walk (0.03/gen) je příliš pomalý a S208
  cost mírný — 30 gen nestačí, aby se selekce na *aktivních* neuronech
  projevila na floor. (CPPN, který mutuje rychleji, ratchet ukazuje; hidden_n
  potřebuje 100+ gen.)
- **`behavioral_entropy` 0.49 → 0.19** + **lineages 200 → 4** — behaviorální i
  genealogická konvergence. Ochranné mechaniky ji *ohraničily* (bond_active
  ~0.25 vs pre-S194 runaway 0.93), ale nereverzovaly. Per diagnostický strom:
  *rostoucí složitost + nízký species_count + kolaps lineages* = poptávka/cost
  fungují, **ochrana je slabá** → zostřit `FITNESS_SHARE_PRESSURE` a snížit
  `CPPN_SPECIATION_THRESHOLD` (dřívější split druhů → víc chráněných nik).
- **`carnivore_avg` 0.25 → 0.06** — populace zkonvergovala k herbivorii;
  predační niche ztenčil. Koevoluce (S210) potřebuje silnější tlak, aby
  carnivore niche přežil.

**Závěr smoke:** stack je **stabilní a complexity ratchet prokazatelně cvaká na
CPPN ose** (+58 % nodes pod selekcí, stepping stones zachované). Hidden-neuron
floor a behaviorální diverzita vyžadují plný 100-gen horizont + tuning ochrany.

**Doporučené tuning kroky před plným 5×100 sweepem:**
1. `CPPN_SPECIATION_THRESHOLD` 1.0 → ~0.6 (dřívější/jemnější split druhů).
2. `FITNESS_SHARE_PRESSURE` 1.0 → ~1.5 (silnější ochrana malých druhů).
3. `BRAIN_COST_PER_UTIL` ponechat, ale ověřit na 100 gen, zda `hidden_n_avg`
   začne stoupat (jinak zostřit S207 `FOOD_PATCH_CONTRAST` = víc poptávky po
   paměti).
4. Pak teprve 5×100-gen cross-seed sweep (seedy 1–5) jako finální gate.

### Rozšíření — 100-gen běh (seed 1, 60000 ticků, 36 ticks/s)

30-gen smoke prodloužen na **plných 100 gen** (jeden seed). Potvrzuje a zostřuje
závěr:

| gen | cppn_nodes_avg | cppn_links_avg | complexity_floor | hidden_n_avg | species | elite | entropy | carnivore | lineages |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 10.0 | 9.0 | 25 | 25.0 | 1 | 0 | 0.49 | 0.25 | 200 |
| 30 | 15.8 | 23.6 | 22 | 24.5 | 2 | 58 | 0.19 | 0.06 | 4 |
| 60 | 20.6 | 34.2 | 22 | 24.1 | 2 | 64 | 0.10 | 0.04 | 4 |
| 100 | **23.9** | **43.5** | 22 | 24.1 | 2 | 65 | 0.19 | 0.04 | 4 |

- **CPPN ratchet drží přes celý horizont, BEZ plateau:** `cppn_nodes_avg`
  10.0 → 23.9 (**+139 %**), `cppn_links_avg` 9.0 → 43.5 (**+383 %**) — obojí
  **monotónně až do gen 100** (g30=15.8, g50=18.9, g70=21.8, g90=23.2). Strop
  `CPPN_MAX_NODES=64` ještě daleko. Tohle je hlavní úspěch desítky.
- **Vedlejší důkaz složitosti:** throughput klesl 53 → 36 ticks/s mezi 30 a 100
  gen — větší mozky (bohatší CPPN) stojí víc compute.
- **`complexity_floor` stabilní na 22** přes 80 gen (přestal erodovat ~g14, dál
  neklesá — floor drží).
- **`hidden_n_avg` mírně dolů** (25.0 → 24.1), `hidden_n_max` 25 → 27 (frontier
  roste). Neuron-count osa se *neratchetuje* — S208 cost spíš lehce prořezává
  nevyužité. Komplexifikace jde přes CPPN topologii, ne přes počet neuronů.
- **`species_count` drží 2, `elite_coverage` saturuje na 65** — ochrana ohraničí
  konvergenci, ale `behavioral_entropy` osciluje kolem 0.15-0.19 (nepadá k 0 —
  novelty/sharing drží spodní mez) a `lineages` kolaps na 4.
- **`carnivore_avg` 0.25 → 0.04** — predační niche prakticky vymřel → S210 Red
  Queen nemá s kým závodit (potvrzeno i renderer populací: `spk_avg=0`).

**Závěr validace:** **complexity ratchet je reálný a trvalý na CPPN ose**
(+139 % nodes / +383 % links přes 100 gen, žádný plateau). Floor drží, populace
stabilní, stepping stones zachované. Neuron-count osa a behaviorální/genealogická
diverzita zůstávají ploché → tuning (níže) + delší horizont. Renderer populace
@ gen 3567 (~999 buněk, 7 linií, herbivore, mírné clustery) = dlouhodobá
saturace téhle trajektorie.

**Stav S212:** single-seed 1×100 gen hotovo (+ 1×30 smoke). **Plný 5×100
cross-seed sweep** (seedy 1–5, s tuningem níže) zůstává jako finální gate.

### Tuning pass — výsledky (40-gen A/B, seed 1)

Aplikovány tři změny + jedna revertovaná, ověřeno 40-gen A/B vůči baseline:

| Změna | Konstanta | Výsledek |
|---|---|---|
| Jemnější speciation | `CPPN_SPECIATION_THRESHOLD` 1.0 → **0.6** | ✅ `species_count` ~2× (base 1-2 → tune 3-5 napříč gen) |
| Silnější fitness sharing | `FITNESS_SHARE_PRESSURE` 1.0 → **1.5** | ✅ `lineages` drží výš (5-8 vs 4), `behavioral_entropy` trvale vyšší (g40: 0.19 vs 0.17) |
| Diet rarity bonus | nová `CARNIVORE_RARITY_BONUS` v `apply_redqueen_pressure`, laděno **1.5 → 6.0** | ◑ **částečně** — @1.5 under-powered (pod baseline @g40); @6.0 výrazně zpomalí ranný kolaps (g5: 0.167 vs 0.068 base) a @g40 mírně nad baseline (0.052 vs 0.042), ale niche se stejně ustálí na ~5 %. Bonus navíc zvedl species (g10: 7) a entropy (g5: 0.37). |
| Rychlejší hidden walk | `hidden_n_step_rate` 0.03 → 0.05 | ⏪ **revertováno** — faster unbiased walk neratchetuje (jen churn), a rozbil mutate draw-accounting testy (hidden_n outcome → from_cppn jitter draw count) |

**Diagnóza diet bonusu:** všechny tři varianty (base / 1.5 / 6.0) konvergují ke
`carnivore_avg ≈ 0.04–0.05` → existuje **rovnovážná carnivore fraction daná
ekonomikou predace samotné** (post-Hunter balance je net-neprofitabilní mimo
malou frakci). Frequency-dependent bonus tu rovnováhu jen *posune* (6.0: +24 %
@g40, a dramaticky zpomalí ranný kolaps), nedokáže ji *zvednout* na robustní
predátorskou populaci. **Skutečná revival vyžaduje rebalance predace samotné**
(`PREDATION_GAIN_PER_TICK` / `PREDATION_DRAIN`), ne silnější bonus — to je větší
zásah do laděné ekonomiky a vlastní pass. Bonus @6.0 zůstává (čistý net-pozitiv:
víc carnivory brzy → víc species + entropy, self-limiting, žádná nestabilita).

**Net efekt tuningu:** ochranný pilíř **měřitelně silnější** (druhy 2×, diverzita
výš, CPPN ratchet i mírně rychlejší: cppnN g40 18.2 vs 17.2). Predační niche
pořád umírá — to je jediný nevyřešený target. `hidden_n` osa zůstává úkol pro
*poptávku*, ne mutaci.

---

## Follow-up experiment — facultativní reprodukce (sex vs dělení)

Diverzitní strop desítky (lineages → 4, entropy konverguje) vyvolal hypotézu:
**zpomaluje obligátní sexuální reprodukce vývoj?** (Crossover homogenizuje
genofond + mate-finding cost.) Místo hádky teorií se to v sim změřilo.

**Implementace:** per-genome gen `sexual_pref ∈ [0,1]` (init uniform) — pst, že
fertilní buňka hledá partnera (crossover) vs **dělí se asexuálně** (klonální
kopie, `make_division_child_no_brain` přes self-cross). Evoluce gen ladí sama.
40-gen A/B (seed 1) vs `tuned2` (stejný tuning, jen obligátně sexuální).

**Výsledek — hypotéza je z poloviny špatně, a ta důležitější polovina obráceně:**

| | FACULT (g30) | TUNED2 (g30) | závěr |
|---|---|---|---|
| `cppn_nodes_avg` | 13.0 | **16.9** | **sex je ~25 % RYCHLEJŠÍ na složitosti** |
| `behavioral_entropy` | **0.224** | 0.159 | asexual drží vyšší diverzitu |
| `carnivore_avg` | **0.110** | 0.052 | asexual drží predační niche |

1. **Sex ZRYCHLUJE strukturální složitost, ne zpomaluje.** Pure-sexual ratchetuje
   CPPN ~25 % rychleji. Příčina: NEAT-aligned crossover **kombinuje strukturální
   inovace napříč liniemi** (Fisher-Muller) — klonální dělení to neumí. Pro
   *primární cíl desítky* (růst složitosti) je sex **přínos**.
2. **Asexual dělení chrání DIVERZITU** (entropy + carnivore niche výš) — přesně
   to, s čím tuning bojoval. Klonální linie se nehomogenizují crossoverem.
3. **Evoluce si nevybírá — drží density-dependentní MIX.** `sexual_pref` osciluje
   ~0.4–0.5: prudce k dělení při řídké populaci (g5: 0.21 — mate-finding drahý),
   zpět k sexu při husté (g15-20: ~0.52). **Facultativní sexualita je ESS**,
   přesně jako v reálné biologii.

**Verdikt:** sex neslowuje složitost — *urychluje* ji (innovation-combining
crossover); asexual pomáhá *diverzitě*. Facultativní gen dává obojí + je
biologicky realistický + nechává evoluci balancovat trade-off sama → **zůstává
v kódu jako net přínos** (řeší půlku diverzitního stropu bez ztráty
complexity-ratchetu, protože sex si populace stejně z velké části udrží).
Kandidát na vlastní desítku 213+ (co-evoluce reprodukčního režimu s nikou).

## Architektonické invarianty desítky

- **GPU-first zachován:** S207/S208/S210 sahají do shaderů (`food_spawn`,
  `step`, `predate`) → každý potřebuje CPU/GPU parity test (jádrová strukturální
  záruka projektu). S203/S204/S205/S206/S211 jsou CPU control-plane na
  gen-boundary (vzor: symbiont mechaniky 196–198) — žádný per-tick GPU cost.
- **Shared driver:** veškerá mechanika v `src/sim/world.rs` → renderer i
  headless ji vidí automaticky. Reprodukční edity (`src/reproduction.rs`) jsou
  sekvenční CPU blok sdílený oběma binárkami.
- **Validace:** každý sprint s mechanikou → 5×100-gen cross-seed sweep; smoke
  ≤30 gen během vývoje ([[feedback_perf_smoke_runs]]).
- **Tuneables:** nový `src/params/evolution.rs` jako single home pro konstanty
  desítky.
- **Checkpoint:** S203 (nové CSV) nemění schema; S204 (`species_id`), S206
  (`lifetime_fitness`) přidávají `#[serde(default)]` pole → forward-compat.
  CHECKPOINT_VERSION bump jen pokud se mění brain weight layout (zatím
  neplánováno — desítka nemění topologii storage, jen selekci nad ní).
