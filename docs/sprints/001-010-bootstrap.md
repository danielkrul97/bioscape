# Sprinty 001–010 — Bootstrap

Bootstrap fáze projektu: minimální 2D scéna → fixed-tick simulační clock → headless běh → první genom.

## Sprint 01 — bevy-scaffold

- **Cíl:** minimální Bevy 2D scéna s buňkami pohybujícími se ve čtverci, simulační logika oddělená do `lib.rs` (aby šla později pohánět i headless).
- **Výstup:** `src/lib.rs` s `Cell::{random, step}`, `src/main.rs` s 200 buňkami a 2D kamerou, ořezaná Bevy feature sada bez `bevy_gilrs` a `audio` (nepotřebujeme gamepad ani zvuk, mizí závislost na `libudev` / `libasound`). Commity `1235e5c`, `b142e0e`.
- **Poznámky:** `step_cells` čte `Time<Real>` přes `time.delta_secs()` — vázané na FPS a nedeterministické. Řeší Sprint 02.

## Sprint 02 — sim-clock

- **Cíl:** oddělit simulační čas od wall clocku a zavést tříúrovňovou hierarchii **tick → generace → epocha**.
  - Sim systémy (`step_cells` a další) přesunout do `FixedUpdate` schedule a brát `dt` z `Time<Fixed>` (default 60 Hz). Tím získáme deterministický krok nezávislý na FPS — předpoklad reprodukovatelných experimentů z `docs/08`.
  - Rychlost runtime přes `Time<Virtual>::set_relative_speed`. Klávesy: Space = pause, `1` / `2` / `3` / `4` = 1× / 10× / 100× / max. Speed má smysl jen ve windowed binárce; headless vždy poběží naplno.
  - `SimClock { tick, generation, epoch, ticks_per_generation, generations_per_epoch }` jako plain struct v `lib.rs` (bez Bevy types, ať jde pohánět i z chystaného headless harness). V `main.rs` zabalený jako Bevy `Resource`.
  - Jeden `advance_clock` systém v `FixedUpdate` inkrementuje čítače a při překročení hran emituje Bevy eventy `GenerationEnded { gen }` a `EpochEnded { epoch }`. Důvod hybridu (čítače + eventy): rychlé per-tick systémy nemusí kontrolovat modulo, pomalá logika (selekce, snapshoty, klimatické cykly) jen reaguje na event. Sedí to na rozdělení rychlé učení / pomalá evoluce z `docs/02`.
  - Počáteční hodnoty (vyladí se empiricky): `Time<Fixed>` = 60 Hz, `ticks_per_generation` = 600 (≈ 10 s sim-času při 1×), `generations_per_epoch` = 100.
- **Výstup:**
  - `src/lib.rs`: `SimClock { tick, generation, epoch, ticks_per_generation, generations_per_epoch }` + `ClockTransitions` (`Option<u64>` pro každou hranici), `advance()` vrací přechody. Unit testy fixují boundary sémantiku.
  - `src/main.rs`: `step_cells` v `FixedUpdate`; `Time<Fixed>` na 60 Hz; `Clock(SimClock)` jako Bevy `Resource`; `advance_clock` v `FixedUpdate` emituje `GenerationEnded { generation }` a `EpochEnded { epoch }`.
  - `speed_input`: Space = pause/unpause, `1`/`2`/`3`/`4` = 1× / 10× / 100× / 1000× (zvolený strop pro „max"); `log_clock_events` zatím loguje hrany přes `info!`.
  - Konstanty: `FIXED_TIMESTEP_HZ = 60.0`, `TICKS_PER_GENERATION = 600`, `GENERATIONS_PER_EPOCH = 100`.
- **Poznámky:**
  - Per-organism věk (`born_tick` na `Cell`) **ne** — počká si na sprint o reprodukci/lifespan.
  - Globální generační hranice je úmyslně GA-style. Async selekce á la Tierra (`docs/02`) se znovu zváží, až bude reprodukce — `SimClock` pak buď zůstane jako environmentální čas (sezóny, klima) a generace se přesunou per-organism.
  - `set_relative_speed` na `Time<Virtual>` automaticky zrychluje i `Time<Fixed>` (víc `FixedUpdate` runů per frame) — proto stačí jeden multiplier pro celý sim.
  - `1000×` jako „max" je arbitrární strop; reálně je rychlost limitována Bevy `Time<Virtual>::max_delta` (default 0.25 s = ~15 fixed updateů per frame). Plný uncapped režim je téma chystaného headless harness.
  - Bevy 0.18 nepoužívá `Event`/`EventReader`/`EventWriter`/`add_event` pro buffered eventy — to je teď `Message`/`MessageReader`/`MessageWriter`/`add_message`. `Event` je rezervovaný pro observer-targeted eventy.

## Sprint 03 — stats-overlay

- **Cíl:** semitransparentní debug panel v pravém dolním rohu okna, ukazující stav simulace v reálném čase. Read-only, žádná interakce mimo toggle viditelnosti.
  - **UI stack:** nativní Bevy UI (`Node` + `Text`), bez `bevy_egui` (držíme minimální dep set). V `Cargo.toml` ověřit / doplnit chybějící features — current trimmed list má `default_font` a `2d_bevy_render`, ale `bevy_ui` / `bevy_ui_render` / `bevy_text` jsou nejspíš potřeba zapnout explicitně. Zjistí se při prvním buildu.
  - **Layout:** jeden root `Node` s `position_type: Absolute`, `right: Val::Px(8.0)`, `bottom: Val::Px(8.0)`, padding 8 px, `flex_direction: Column`, `BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.6))`, `BorderRadius::all(Val::Px(4.0))`. Světle šedý text (`Color::srgb(0.9, 0.9, 0.9)`) nad ~60 % černým overlayem sedí kontrastem na clear color `(0.05, 0.05, 0.08)`.
  - **Obsah** — jeden multi-line `Text` uzel (jednodušší než Text-per-řádek):
    - `tick`, `generation`, `epoch` ze `Clock`
    - `speed` — `relative_speed()` z `Time<Virtual>`, nebo `"paused"` při `is_paused()`
    - `cells` — počet entit `CellEntity` (`Query::iter().count()`)
    - `fps` — smoothed avg z `FrameTimeDiagnosticsPlugin` (přidat plugin do `App`)
  - **Update systém** `update_stats_overlay` v `Update` schedule. Panel je pozorovací, nepotřebuje deterministický tick — proto ne `FixedUpdate`.
  - **Klávesa `H`** — toggle přes `Display::None` ↔ `Display::Flex` na root `Node`. Důvod: evoluční runy panel obtěžuje, navíc snímky pro screenshoty bez HUDu.
- **Výstup:**
  - `Cargo.toml`: doplněny features `bevy_ui` + `bevy_ui_render` (`bevy_text` jde tranzitivně přes `2d_bevy_render` → `2d_api` → `common_api`).
  - `src/main.rs`: marker komponenty `StatsRoot` + `StatsText`, `FrameTimeDiagnosticsPlugin::default()` přidán k pluginům, root `Node` s `position_type: Absolute` + `right/bottom: 8 px` + `border_radius: 4 px` + `BackgroundColor` α 0.6, jeden child `Text` přes `children![]` macro.
  - Systémy: `setup_stats_overlay` (Startup), `update_stats_overlay` (Update, čte `Clock` / `Time<Virtual>` / `DiagnosticsStore` / count `CellEntity`), `toggle_stats_overlay` (Update, klávesa `H`).
  - Format textu: monospace 7-znakový label + hodnota (`tick   {}`, `gen    {}`, …). FiraMono jako default font dělá zarovnání zadarmo.
- **Poznámky:**
  - **Gotcha:** v Bevy 0.18 je `BorderRadius` **field na `Node`**, ne samostatná komponenta. Spawnovat ho jako `BorderRadius::all(...)` v tuple bundlu nepůjde (kompilátor kvičí na nesplněný `Bundle` trait). Stejně tak `border` apod.
  - `bevy_text` se aktivuje tranzitivně přes `common_api` ze `2d_bevy_render` — v `Cargo.toml` ho nemusíme uvádět explicitně.
  - `Single<&mut Text>` v `update_stats_overlay` panikne, kdyby StatsText neexistoval (např. dvakrát toggle z `Display::None` neskryje entitu, pořád existuje — OK). Pokud někdy budeme spawnovat / despawnovat panel, nahradit za `Query` + `single_mut().ok()`.
  - Bez `bevy_egui` záměrně. Pokud se nativní stack ukáže jako moc verbose (panel přes ~50 řádků), revidovat.
  - Reformat textu per-frame je pro krátký string OK; throttle na 5–10 Hz až kdyby to začalo být znát.
  - Headless harness (chystaný v dalších sprintech) tuhle vrstvu nezahrne — je to ryze rendering.

## Sprint 04 — first-genome

- **Cíl:** zavést první genom + dědičnost + neutrální mutaci, využít event `GenerationEnded` ze Sprintu 02 pro reprodukci celé populace najednou. **Bez selekčního tlaku** — jen ověřit, že kopírovací stroj (variation + heritability z `docs/01`) funguje. Selekce přijde Sprint 05.
  - **Genom:** `Genome { max_speed: f32, color_hue: f32 }` v `lib.rs` (bez Bevy). `max_speed` je behaviorální (ovlivňuje skutečný pohyb), `color_hue` je **neutrální marker** pro vizuální tracking populace — žádný behavior impact, jen barva. Odpovídá real-biology použití SNPs v non-coding regions.
  - **Cell:** přidá se `genome` field. Iniciální velocity = `random_direction × genome.max_speed` (rychlost se konečně projeví v pohybu, ne jen jako fixní ±60).
  - **Mutace:** `Genome::mutate(rng, σ_speed, σ_hue)` — gaussian noise (Box-Muller manuálně, bez `rand_distr` dep). `max_speed` clampnut na ≥ 1.0 (jinak buňka stojí), `color_hue` přes `rem_euclid(360.0)` (kruhové wrapping).
  - **Reprodukce:** `reproduce_on_generation_end` systém v `Update`, čte `GenerationEnded`. Sample N rodičů (with replacement, uniform), každý vyplodí jednoho potomka s mutovaným genomem. Despawn všech rodičů, spawn potomků na náhodných pozicích. Populace zůstane konstantní (200).
  - **Materiály:** sdílený `Mesh` přes resource `CellMesh`, ale **per-cell `ColorMaterial`** (každá buňka má unikátní barvu z `color_hue`). Bevy ref-counted `Handle` recykluje staré materiály při despawn.
  - **Stats panel rozšíření:** dva nové řádky `spd_avg` a `spd_dev` (mean / stddev `max_speed` napříč populací). Hue stats schválně vynechané — circular mean/stddev je zvláštní téma.
  - **Konstanty:** `SIGMA_SPEED = 5.0`, `SIGMA_HUE = 8.0` (empirický start, vyladí se).
- **Výstup:**
  - `src/lib.rs`: `Genome { max_speed, color_hue }` + `Genome::random` (uniform: speed 30–90, hue 0–360) + `Genome::mutate` (Gaussian σ na speed, σ na hue, clamp `MIN_SPEED = 1.0`, `rem_euclid(360)`). Pomocná `gaussian()` (Box-Muller). `Cell` rozšířen o `genome` field; nová `Cell::from_genome` (random pos + velocity = `random_dir × max_speed`); `Cell::random` napárované přes `Genome::random`. Dva nové unit testy.
  - `src/main.rs`: konstanty `SIGMA_SPEED`, `SIGMA_HUE`. Resource `CellMesh(Handle<Mesh>)` sdílí kruhový mesh mezi setupem a reprodukcí. Per-cell `ColorMaterial` přes `hue_to_color` (`Color::hsl(hue, 0.75, 0.55)`). Nový systém `reproduce_on_generation_end` v `Update` (čte `GenerationEnded`, sample with replacement, despawn rodičů, spawn potomků). Ordering `(reproduce, update_stats_overlay).chain()` zaručí, že stats vidí novou populaci.
  - Stats panel: dva nové řádky `spd_avg` (mean `max_speed`) a `spd_dev` (population stddev). Helper `mean_stddev`.
- **Poznámky:**
  - Reprodukce je úmyslně globální/synchronní GA pattern. Async á la Tierra (`docs/02`) odložené.
  - `rng = rand::rng()` je zatím non-deterministický globální stream. Seedovaná RNG přijde s headless harness.
  - Bez `rand_distr` crate — Box-Muller je 4 řádky a samostatná dep se nevyplatí.
  - Hue stats vynechané schválně (kruhová statistika). Visuální tracking přes barvu sám o sobě stačí.
  - Per-cell `ColorMaterial` znamená ~200 nových materiálů na generaci. `Handle` je ref-counted, takže staré mizí při despawn rodiče. Pokud by to začalo být znát, bin po N hue bucketech.
  - Selekce / fitness / energetika: out of scope. Sprint 05.

## Sprint 05 — energy-economy

- **Cíl:** první selekční tlak přes energetickou ekonomiku. Buňky jí, utrácí pohybem, nashromážděná energie váží šanci být rodičem v další generaci. `max_speed` má přestat jen driftovat a začít mířit k emergentnímu optimu (rychlejší = víc nasbírá, ale i víc utratí).
  - **Food:** `Food { position }` v `lib.rs` (sim-side, headless-friendly). V ECS jako `FoodEntity(Food)` se sdíleným meshem a materiálem (`FoodMesh`, `FoodMaterial` resources — bílý kruh, menší než buňka, z = −1 aby buňky byly nahoře).
  - **Spawn:** `spawn_food` v `FixedUpdate` udržuje target ~300 jídel. Pod targetem spawne `FOOD_SPAWN_RATE = 5` / tick. V `setup` se předem nasype celý target, ať buňky mají co jíst od prvního ticku.
  - **Energy cost:** `Cell::step(dt, world, energy_cost_per_distance)` — energie klesá úměrně vzdálenosti urazené v ticku (`distance × cost`). Faster cell = víc utrácí. `ENERGY_COST_PER_DISTANCE = 0.1`.
  - **Eating:** `Cell::try_eat(&food, eat_radius, food_value)` v `lib.rs` — sqrt-free distance² check. V `main.rs` systém `cell_eats_food` v `FixedUpdate` po `step_cells`, naive O(N×M), per-frame `eaten` flag aby jedno jídlo nesnědly dvě buňky najednou. Konstanty `FOOD_VALUE = 20`, `EAT_RADIUS = 8.0`.
  - **Selekce:** `reproduce_on_generation_end` mění uniform sampling na **roulette-wheel** vážený podle `cell.energy.max(0.0)`. Při total weight ≤ 0 (extinction event) fallback na uniform. Negativní energie ⇒ váha 0 ⇒ efektivní smrt bez explicitního despawn.
  - **Stats panel:** dva nové řádky `food` (count) a `e_avg` (mean energy populace).
  - **Buňky neumírají mid-generation** — energie může jít do záporu, despawn jen na gen-hraně (pattern ze Sprintu 04).
  - **Konstanty pro empirický start:** `INITIAL_ENERGY = 100`, `ENERGY_COST_PER_DISTANCE = 0.1`, `FOOD_VALUE = 20`, `FOOD_COUNT_TARGET = 300`, `FOOD_SPAWN_RATE = 5`, `EAT_RADIUS = 8.0`.
- **Výstup:**
  - `src/lib.rs`: `pub const INITIAL_ENERGY = 100.0`, nový `Food { position }` + `Food::random`, `Cell::step` rozšířen o `energy_cost_per_distance` (`distance × cost`), nový `Cell::try_eat(&food, radius, value) -> bool` se sqrt-free distance² check. 4 nové unit testy (energy drain, eat in/out radius).
  - `src/main.rs`: konstanty pro energy/food, komponenta `FoodEntity(Food)`, resources `FoodMesh` + `FoodMaterial`. `setup` pre-spawne `FOOD_COUNT_TARGET` jídel s z=−1 (pod buňkami). Nové systémy v `FixedUpdate`: `spawn_food` (replenish do targetu po `FOOD_SPAWN_RATE` per tick), `cell_eats_food` (naive O(N×M) s per-frame `eaten` flag pro každou potravinu). Order: `(advance_clock, step_cells, spawn_food, cell_eats_food).chain()`.
  - `reproduce_on_generation_end`: roulette-wheel přes `cell.energy.max(0.0)` (pomocná `sample_weighted`); fallback na uniform při `total ≤ 0`.
  - Stats panel: dva řádky navíc — `food` (count `FoodEntity`) a `e_avg` (mean energie populace).
- **Poznámky:**
  - Vědomě synchronní global generace (varianta A z diskuse). Async smrt/reprodukce při threshold = Sprint 06+.
  - Energie není dědičná (phenotype state, ne genotype). Každý potomek startuje s `INITIAL_ENERGY`.
  - Žádný cap na max energy — jedna buňka co najde food cluster může v další generaci výrazně dominovat. Pokud to bude moc dramatické, přidá se soft cap.
  - Food kolize je naivní O(N×M). Při N >> 1k nebo M >> 1k bude potřeba spatial hash, ale ne dřív.
  - Hue se pořád mění driftem (`SIGMA_HUE`). Pokud selekce zafunguje, vznikne korelace mezi hue a max_speed (lineage co měla rychlostní výhodu si svou hue protlačí) — uvidíme.
  - `cell_eats_food` collectne všechna jídla do Vec až ten frame — pokud food entit bude hodně (>10k), zvážit udržování spatial gridu jako resource místo iterativního `Query::iter`.

## Sprint 06 — vision-and-async

- **Cíl:** dvě věci najednou — **(A)** přidat vize a reaktivní pohyb (první perception-action smyčka, počátek toho, kvůli čemu projekt je) **(C)** přepnout reprodukci a smrt na **asynchronní per-organism lifecycle** (Tierra/Avida vibe z `docs/02`).

  **A) Vize:**
  - Nový gen `vision_radius` v `Genome`, mutovaný s `SIGMA_VISION`. Iniciální range 20–80 (cell radius je 5, world half-extent 400, takže vize 50 ≈ 12 % šířky světa).
  - **Cost vize:** lineárně s `vision_radius` (větší dohled = větší metabolic burn), `VISION_COST_PER_RADIUS = 0.05` per tick. Vyřešeno uvnitř `Cell::step`.
  - **Systém `cells_seek_food`** v `FixedUpdate` **před** `step_cells`: pro každou buňku najde nejbližší jídlo v dohledu (`d² ≤ vision_radius²`) a otočí velocity směrem k němu (hard lock-on, magnitude zachová `max_speed`). Bez jídla v dohledu velocity necháme.
  - Hard lock-on úmyslně. Smooth turning přes `turn_rate` gen je možná téma pozdější (Sprint 07/08), pokud bude lock-on působit moc dokonale.

  **C) Async lifecycle:**
  - **`cell_dies_on_zero_energy`** v `FixedUpdate` — buňky s energií ≤ 0 se despawnnou.
  - **`cell_reproduces_on_threshold`** v `FixedUpdate` — buňka s energií ≥ `REPRODUCE_THRESHOLD = 200` se rozdělí (binary fission). Parent energie se rozpůlí, child dostane druhou polovinu, child genome = mutace parenta. Child spawnne na pozici rodiče, random velocity směr × `child.max_speed`.
  - **`reproduce_on_generation_end` se odstraní.** Synchronní global generace skončila. `GenerationEnded`/`EpochEnded` eventy se pořád emitují (jsou cheap, hodí se na pozdější environmental cycles), ale neřídí reprodukci.
  - **Soft population cap** `MAX_POPULATION = 1000` v `cell_reproduces_on_threshold` (skip pokud over cap, prevence runaway). Hard floor neexistuje — extinction je možná a OK.

- **Konstanty:**
  - `SIGMA_SPEED = 3.0`, `SIGMA_HUE = 5.0`, `SIGMA_VISION = 3.0` — sníženo z původních (5/8/—). V async modelu se mutuje při **každé** reprodukci, ne 1× / generaci, takže menší sigma odpovídá zhruba stejné evoluční rychlosti.
  - `VISION_COST_PER_RADIUS = 0.05`, `REPRODUCE_THRESHOLD = 200`, `MAX_POPULATION = 1000`.

- **Stats panel:** přidat `vis_avg` a `vis_dev`. `cells` count se stane dynamickým a zajímavým.

- **FixedUpdate ordering:** `(advance_clock, cells_seek_food, step_cells, spawn_food, cell_eats_food, cell_reproduces_on_threshold, cell_dies_on_zero_energy).chain()`.

- **Výstup:**
  - `src/lib.rs`: `Genome` rozšířen o `vision_radius` (init range 20–80, `MIN_VISION = 1.0`). `Genome::mutate` má teď třetí parametr `sigma_vision`. `Cell::step` má čtvrtý parametr `vision_cost_per_radius` (energy drain `vision_radius × cost × dt` per tick). Aktualizované unit testy (5/5 pokrývá 3-sigma mutaci, kombinovaný energy drain z movement+vision, eat hit/miss).
  - `src/main.rs`: konstanty `SIGMA_VISION`, `VISION_COST_PER_RADIUS`, `REPRODUCE_THRESHOLD`, `MAX_POPULATION`. `SIGMA_SPEED` snížen z 5 → 3, `SIGMA_HUE` z 8 → 5 (kompenzace za vyšší frekvenci mutací v async režimu).
  - **Nový systém `cells_seek_food`** — najde nejbližší jídlo v `vision_radius²` a hard lock-on na velocity (magnitude = `max_speed`).
  - **Nový systém `cell_reproduces_on_threshold`** — buňka s energií ≥ 200 se rozdělí: parent rozpůlí energii, child genome = mutace, child spawnne na pozici parenta s random velocity směrem × `child.max_speed`. Soft cap `MAX_POPULATION = 1000` přes `budget`.
  - **Nový systém `cell_dies_on_zero_energy`** — despawn při `energy ≤ 0`.
  - **Odstraněno:** `reproduce_on_generation_end` + helper `sample_weighted`. `GenerationEnded`/`EpochEnded` eventy se pořád emitují (cheap + future-proofed pro klima/snapshoty), ale nikdo na ně nereaguje kromě `log_clock_events`.
  - Stats panel: 12 řádků (přibyly `vis_avg` a `vis_dev`). `cells` count je teď dynamický.
- **Poznámky:**
  - Async lifecycle znamená kolísavou populaci. Pokud se populace drží na soft cap, znamená to, že carrying capacity je vysoká (jídla je dost) — pak vyladit `FOOD_COUNT_TARGET` dolů. Pokud populace zaniká, naopak `FOOD_VALUE` nahoru nebo `ENERGY_COST_PER_DISTANCE` dolů.
  - `Genome::mutate` má teď 3 sigmas — signature roste. Při dalším genu refaktorovat na `MutationConfig` struct.
  - Vize je O(N×M), spolu s eat to je 2× O(N×M) per tick. Při N >> 1k bude potřeba spatial hash; před tím se nebude řešit.
  - `GenerationEnded`/`EpochEnded` zůstává jako "tep" simulace — užívá se zatím jen pro logging, ale je k dispozici.
  - Hard lock-on dělá z buněk lock-on missiles. Pokud se sim chová "moc cíleně" a chybí divokost, pak zvážit šum v turn-toward (jitter směru) nebo turn_rate gen.

## Sprint 07 — death-fade

- **Cíl:** vizuálně odlišit smrt — místo okamžitého despawn projde buňka fází `Dying`, kde 0.5 s shrinkne k 0 a fade-outne na alpha 0. Dying buňky během fáze **nejedí, nevidí, nereprodukují** — zůstávají jako fyzické entity (drift dál podle velocity), ale neovlivňují biologii.
  - **Komponenta `Dying { ticks_left: u32 }`** v `main.rs`.
  - **`cell_dies_on_zero_energy` upraveno:** místo `despawn` vloží `Dying { DEATH_FADE_TICKS }` přes `commands.entity(e).insert(...)`. Filter `Without<Dying>` aby se nevkládal opakovaně na už-umírající.
  - **Nový systém `tick_death_fade`** v `FixedUpdate`, na konci chainu: dekrementuje `ticks_left`, přepočítává `Transform.scale = progress` a `material.color.alpha = progress`, na `ticks_left == 0` despawn.
  - **Filtr `Without<Dying>`** v `cells_seek_food`, `cell_eats_food`, `cell_reproduces_on_threshold`. `step_cells` filter **nemá** — drift mrtvoly je vizuálně realistický a ovlivnění energie u umírajících buněk je no-op.
  - **`AlphaMode2d::Blend` na cell materiálech** — povinné, jinak alpha nefunguje. Bevy `From<Color> for ColorMaterial` přepne na Opaque pokud `color.alpha() == 1.0`. Helper `make_cell_material` (DRY napříč setup + reprodukce).
  - **Konstanta:** `DEATH_FADE_TICKS = 30` (0.5 s na 1×).
- **Výstup:**
  - `src/main.rs`: konstanta `DEATH_FADE_TICKS = 30`, marker komponenta `Dying { ticks_left }`, helper `make_cell_material` (`AlphaMode2d::Blend` explicitně, dvouřádkový WHY komentář kvůli `From<Color>` gotchovi). Použit v `setup` i `cell_reproduces_on_threshold`.
  - `cell_dies_on_zero_energy` přepsán: `Without<Dying>` filter + insert `Dying`. `cells_seek_food`, `cell_eats_food`, `cell_reproduces_on_threshold` mají `Without<Dying>` filter.
  - Nový systém `tick_death_fade` (FixedUpdate, na konci chainu): dekrement `ticks_left`, lerp `Transform.scale` a `material.color.alpha` na progress 1→0, na 0 despawn.
  - Import `bevy::sprite_render::AlphaMode2d`.
- **Poznámky:**
  - Stats panel `cells` count zahrnuje i Dying buňky. Pokud bude vadit, přidá se `Without<Dying>` filter v stats query — zatím ne (fade trvá jen 0.5 s, signál se neovlivní).
  - Mrtvé buňky se **nestávají jídlem** (varianta C nezvolena). Energie z nich odejde mimo systém.
  - Na 100× rychlosti je 0.5 s sim-času = 5 ms wall-clock = ~0–1 framů — fade prakticky neuvidíš. To je úmyslný kompromis, kosmetika rychlého režimu není priorita.
  - Fade kombinuje scale + alpha. Scale samotný by stačil, ale alpha + scale dohromady je biologicky čitelnější ("buňka zprůhlední a smrskne se"). Cena: `AlphaMode::Blend` místo Opaque na všech cell materiálech (drobný blend overhead, pro 200 cells negligible).

## Sprint 08 — first-brain

- **Cíl:** **(B)** evolvovatelný mozek místo hardcoded turn-to-food, **(C)** smooth turning přes `turn_rate` gen, **fullscreen window** (max rozlišení monitoru).

  **B) Tiny perceptron (single-layer):**
  - V `lib.rs` nový `Brain { weights: [[f32; 4]; 2], biases: [f32; 2] }` — 4 inputs → 2 outputs, tanh aktivace. 10 vah + 2 biasy = 12 parametrů.
  - **Inputs (4):** `[food_dx / vision_r, food_dy / vision_r, energy/threshold (clamped 0..1.5), speed/max_speed (clamped 0..1)]`. Pokud nic v dohledu, food vector = (0, 0).
  - **Outputs (2):** `turn_signal ∈ [-1, 1]` (raw tanh), `thrust_signal ∈ [-1, 1]` přemapováno na [0, 1].
  - `Brain::random` (váhy + biasy ~N(0, 1) přes existující `gaussian` helper), `Brain::mutate(sigma)` (gaussian perturbace všech vah/biasů).
  - V `main.rs` systém **`cells_brain_act`** nahradí `cells_seek_food`: perception (najít nejbližší food v `vision_radius`), forward pass, apply:
    - `new_angle = current_angle + turn_signal × turn_rate × dt`
    - `target_speed = thrust_norm × max_speed`
    - `velocity = (cos(new_angle), sin(new_angle)) × target_speed`
    - `Without<Dying>` filter zachován.

  **C) Smooth turning + `turn_rate` gen:**
  - `Genome.turn_rate: f32` (radians/sec), init `1.0..5.0`, `MIN_TURN_RATE = 0.1`.
  - Smooth turning emerguje z `turn_signal × turn_rate × dt` v `cells_brain_act`. **Žádný extra cost** za turn_rate zatím — energie se účtuje pouze za vzdálenost a vision radius. Pokud high turn_rate začne neúměrně dominovat, přidá se cost.

  **Genome refactor:** `mutate` má teď 5 sigmas → refaktor na **`MutationConfig` struct** v `lib.rs`. V `main.rs` const `MUTATION_CONFIG`. Konstanty `SIGMA_SPEED/HUE/VISION` zmizí jako standalone.

  **Window + dynamický world space:** `WindowMode::Windowed` (s rámečkem a title barem), startup-time `Window::set_maximized(true)`. Pevná rezoluce `1024×768` zmizí. World extent (`WORLD_HALF_EXTENT`) přestane být konstantou a stane se z něj resource `WorldExtent { half_x, half_y }` který se update-uje na `WindowResized` eventy. `Cell::step` / `Cell::random` / `Cell::from_genome` / `Food::random` v lib.rs berou `[f32; 2]` místo skaláru. Buňky a jídlo tak vyplňují celý prostor okna; po resize okna se prostor přizpůsobí.

  **Stats panel:** přidat `trn_avg` (mean `turn_rate` gene). Brain stats vynechané (high-dim, těžko shrnout v 1 lineu).

- **Konstanty:** `sigma_turn_rate = 0.3`, `sigma_brain = 0.2`. Init `turn_rate ∈ [1.0, 5.0]`.
- **Výstup:**
  - `src/lib.rs`: `Brain { weights: [[f32; 4]; 2], biases: [f32; 2] }` + `Brain::{random, forward, mutate}`. `MutationConfig` struct (5 sigmas). `Genome` rozšířen o `turn_rate` a `brain`, `Genome::mutate(rng, &MutationConfig)` (refaktor pryč od positional sigmas). Konstanty `BRAIN_INPUTS = 4`, `BRAIN_OUTPUTS = 2`, `MIN_TURN_RATE = 0.1`. Test `brain_forward_zero_inputs_outputs_tanh_of_biases` + helper `dummy_genome` / `zero_cfg` v testech (8/8 passing).
  - `src/main.rs`: konst `MUTATION_CONFIG: MutationConfig` (centralizovaná, byly to 3 standalone konstanty + nové dvě). Window: `WindowMode::Windowed` (default) + startup `Window::set_maximized(true)`. Stats panel řádek `trn_avg`.
  - **`cells_seek_food` → `cells_brain_act`** (FixedUpdate před `step_cells`): per cell perception (najít nejbližší food v `vision_radius²`), inputs `[food_dx/vis_r, food_dy/vis_r, energy/200 (clamp 0..1.5), speed/max_speed (clamp 0..1)]`, forward pass, apply `(turn_signal × turn_rate × dt)` na úhel velocity, `((thrust+1)/2 × max_speed)` jako magnitude.
  - `cell_reproduces_on_threshold` volá `mutate(rng, &MUTATION_CONFIG)`.
  - **Dynamický world space:** `WORLD_HALF_EXTENT` konstanta zmizí. Resource `WorldExtent { half_x, half_y }` se nastaví v `setup` z `Window::resolution`, system `track_window_resize` v `Update` čte `WindowResized` eventy a updatuje extent. `Cell::step` / `Cell::random` / `Cell::from_genome` / `Food::random` v lib.rs berou `[f32; 2]` místo skaláru. `step_cells` a `spawn_food` čtou resource a předávají do lib volání.
- **Poznámky:**
  - Single-layer perceptron je úmyslné minimum — žádná hidden layer, expressive power omezený (nedokáže XOR). Pokud uvidíme, že chování brainu je nerozeznatelné od náhody, přidá se hidden layer (Sprint 09+).
  - Brain je **fixed topology** (4×2). Variable topology (NEAT-style) je velký refaktor, odložené.
  - Initial brains ~N(0, 1) — většina počátečních buněk bude chaotická. Selekce má zafiltrovat ty, co náhodou dělají něco užitečného. To je celý smysl evoluce na vahách neuronky.
  - Žádný metabolic cost neuronky (12 ops × pár tisíc cells × 60 Hz = nevýznamné). Až budeme mít velké NN, přidá se "brain energy cost".
  - Brain inicializace **přes Gaussian** (ne uniform) — lepší distribuce vah pro NN než uniform [-1, 1].
  - BorderlessFullscreen vs exclusive Fullscreen: borderless je friendly k alt-tab, výkon srovnatelný; exclusive by potencionálně zamykl monitor a změnil resolution. Borderless je sane default.
  - Maximized window vs Fullscreen: po zpětné vazbě uživatele finální design je `Windowed` + `set_maximized(true)` — okno má rámeček/title bar a uživatel se s ním normálně chová. World extent dynamický přes `WorldExtent` resource a `WindowResized` event handler.

## Sprint 09 — interactions-and-carrion

- **Cíl:** **(A)** cell-cell kolize + carrion (mrtvá buňka → jídlo) + **(B)** sociální vnímání (brain vidí nejbližší sousední buňku). **Spatial hash grid** místo O(N²) — naive pair check by při růstu populace zabíjel CPU.

  **Spatial hash grid (foundation):**
  - Resource `CellGrid { cell_size, buckets: HashMap<(i32, i32), Vec<(Entity, [f32; 2])>> }` v `main.rs`. `GRID_CELL_SIZE = 100.0` (řádově max běžný `vision_radius`).
  - Systém **`rebuild_cell_grid`** v `FixedUpdate` po `step_cells` — wipe + insert z aktuálních `CellEntity` pozic (filtruje `Without<Dying>`).
  - Metoda `neighbors_within(pos, radius) -> Vec<(Entity, [f32; 2])>` iteruje `(2*ceil(radius/cell_size)+1)²` bucketů. Vrací `Vec` (alokace per query je drobná — pár entries × 200 cells × 60 Hz).

  **A) Cell-cell kolize:**
  - Systém **`resolve_cell_collisions`** v `FixedUpdate` po `rebuild_cell_grid`. Pro každou živou buňku: `grid.neighbors_within(pos, 2 * CELL_RADIUS)`, narrow phase použije CURRENT pozice (z lokálního `HashMap<Entity, [f32; 2]>`), pokud `d < 2 * CELL_RADIUS` push obě cells o `overlap / 2` podél normály.
  - **Position correction**, ne impulse — velocity zůstává nezměněná, cells si "klouznou" okolo sebe v hustém houfu. Bouncing může přibýt později, pokud bude potřeba.

  **A) Carrion:**
  - V `cell_dies_on_zero_energy` při insertu `Dying` zároveň spawn `CARRION_FOOD_COUNT = 2` `FoodEntity` na pozici mrtvé buňky (random offset ±`CELL_RADIUS` pro vizuální separaci jednotlivých particles).
  - Energie zůstává v ekosystému. Implicitní kanibalismus: živé buňky jí carrion jako jakékoliv jiné jídlo (brain neumí rozlišit).

  **B) Sociální vnímání:**
  - `BRAIN_INPUTS: 4 → 6` v `lib.rs`. `Brain.weights: [[f32; 6]; 2]` (10 → 14 parametrů).
  - V `cells_brain_act`: kromě nearest food najít i **nearest other cell** přes `CellGrid::neighbors_within(pos, vision_radius)`. Inputs:
    ```
    [0] food_dx / vis_r       (or 0 mimo dohled)
    [1] food_dy / vis_r       (or 0)
    [2] cell_dx / vis_r       (or 0 — nejbližší jiná buňka)
    [3] cell_dy / vis_r       (or 0)
    [4] energy / threshold    (clamp 0..1.5)
    [5] speed / max_speed     (clamp 0..1)
    ```
  - Brain teď může evolvovat sociální chování: vyhýbání, hejna, predace na carrion clusters.

  **FixedUpdate ordering:**
  ```
  advance_clock, cells_brain_act, step_cells,
  rebuild_cell_grid, resolve_cell_collisions,
  spawn_food, cell_eats_food,
  cell_reproduces_on_threshold,
  cell_dies_on_zero_energy, tick_death_fade
  ```

- **Konstanty:** `GRID_CELL_SIZE = 100.0`, `CARRION_FOOD_COUNT = 2`.
- **Výstup:**
  - `src/lib.rs`: `BRAIN_INPUTS: 4 → 6` (jediná změna v lib — `Brain.weights` shape mění tranzitivně, testy uses const, žádný update).
  - `src/main.rs`: nový type `GridBuckets = HashMap<(i32, i32), Vec<(Entity, [f32; 2])>>` (alias kvůli clippy type_complexity), resource `CellGrid { cell_size, buckets }` s metodami `key_of`, `rebuild`, `neighbors_within`. `init_resource::<CellGrid>()` v App.
  - **Nové systémy:** `rebuild_cell_grid` (po `step_cells`), `resolve_cell_collisions` (position correction přes half-overlap, narrow phase z lokálního `HashMap<Entity, [f32; 2]>` s aktuálními pozicemi).
  - **`cells_brain_act` rozšířeno:** dvě nové brain inputs (cell_dx/cell_dy nejbližší jiné buňky v dohledu), grid query přes `CellGrid::neighbors_within(pos, vision_r)`, self exclude přes Entity ID. Inputs `[food_dx, food_dy, cell_dx, cell_dy, energy, speed]` všechny normované.
  - **`cell_dies_on_zero_energy` rozšířeno:** kromě `Dying` insertu spawne 2 `FoodEntity` na pozici mrtvé buňky s random offsetem `±CELL_RADIUS` (carrion). Energie zůstává v ekosystému.
  - FixedUpdate ordering: `(advance_clock, cells_brain_act, step_cells, rebuild_cell_grid, resolve_cell_collisions, spawn_food, cell_eats_food, cell_reproduces_on_threshold, cell_dies_on_zero_energy, tick_death_fade).chain()`.
- **Poznámky:**
  - `cells_brain_act` čte grid postavený v PŘEDCHOZÍM ticku (rebuild běží AŽ po step). Pozice jsou 1 tick staré (~1 unit). Pro vision queries s radiem 50+ je bias <2 % — akceptovatelné.
  - `resolve_cell_collisions` neovlivňuje velocity — cells si jen "klouznou". Elastic impulse (bouncing s výměnou hybnosti) by byl pravá fyzika, ale teď nepřináší víc evoluční zajímavosti než position correction.
  - Food entity nejsou v gridu (zatím). `cell_eats_food` a `cells_brain_act` (food query) zůstávají O(N×M). Při M >> 1k přibude food grid samostatným sprintem.
  - Při GRID_CELL_SIZE = 100 a vision_radius do ~80 funguje 1-buněčný padding. Pokud by mutace pushla vision_radius nad 200, queries by ztrácely vzdálenější cíle — `neighbors_within` to řeší dynamicky `ceil(r / cell_size)` bucketů.
  - Carrion na stejném místě → 2 food particles překryté. Random offset rozprostře vizuálně. Při low population a hodně smrtech vznikne food cluster — atraktor pro hladovějící cells.
