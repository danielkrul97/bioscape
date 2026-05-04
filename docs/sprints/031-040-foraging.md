# Sprinty 31–40: Foraging + 3D Substrate

Decade tématicky bipolární. **Sprint 31** rozšiřuje 2D ekosystém přes spatial food clustering — výsledek Sprint 21 návratu po cognitive priors (22+) a morfologické flexibilitě (26). **Sprinty 32–37** představují fundamentální substrátový upgrade z 2D do 3D — Cell pozice/velocity 3D, body ellipsoid (length×width×height), z motion + 3D heading (yaw+pitch), volumetric environment infrastructure ready. Renderer port (Sprint 36) je odložen kvůli velikosti scope. Slug "foraging" zachovaný, ačkoli je decade širší než food specifika (3D substrátový upgrade odpovídá biologické změně dimenzionality habitatu — pelagický ↔ benthic přechod).

## Sprint 31 — food-spatial-clustering

- **Cíl:** dotáhnout Sprint 21 do funkčního spatial niching. WorldMap doteď ovlivňuje jen energetickou hodnotu jídla (`FOOD_VALUE × (FLOOR + AMP × richness)`, range [0.85, 1.15]) — uniform-spawn distribuce dělá z heterogenity jen šum, bez prostorové koncentrace zdrojů, kterou by selekce mohla využít. Sprint 31 přidá **mild rejection sampling** — bohaté zóny dostávají víc jídla, chudé méně, ale ne plný knockout (Sprint 21 v1–v5 plnou silou = extinkce). Hypotéza: s cognitive priors ze Sprintu 22+ a damage signálem ze Sprintu 30 se buňky naučí spatial preference (smell-following do rich zón, avoidance hazardních hot spots), což může reaktivovat lineage diversification ze Sprintu 23 v podmínkách aktivních predátorů a sociální dynamiky.

  **Plán:**
  - **Mechanika:** uniform candidate `Food::random(rng, world_half)`, pak per-candidate test `reject_food_for_richness(rng, richness) → bool` s probability `STRENGTH × (1 - richness)`. Při zamítnutí zkusit znovu (`MAX_SPAWN_ATTEMPTS = 5` retry budget). Po vyčerpání retry: spawn-or-skip per existující exclusion logiku (cell-too-close).
  - **Apply sites:**
    1. Initial food population — `World::new` (headless), `setup` (renderer). Bez retry by clustering občas nedotáhl initial target — proto MAX_SPAWN_ATTEMPTS smyčka kolem každého slotu.
    2. Continuous spawn — `World::spawn_food` (headless), `spawn_food` system (renderer). Drží stejný retry budget jako existující cell-exclusion check; rejection a exclusion sdílejí MAX_SPAWN_ATTEMPTS.
  - **Co se NEMĚNÍ:**
    - Carrion food drop — pozice = pozice mrtvé buňky, žádný richness filter (carrion následuje predaci, ne mapu).
    - Value modulation `FOOD_VALUE × (FLOOR + AMP × richness)` — zůstává jako safety net. I food v poor zone nese ~85 % baseline energie, takže výjimkové slot ne-rich-zone není "useless".
    - Hazard zone overlap — stále POSITIVNÍ korelace s richness (rich = dangerous, Sprint 23). Rejection sampling clustering tedy posiluje tradeoff: rich zone má víc jídla, ale i víc damage.

- **Konstanty:** `FOOD_REJECTION_STRENGTH = 0.3`. Sprint 21 zkoušel ekvivalent 1.0 (plná `1 - richness`) napříč v1–v5 → extinkce gen 70–110. 0.3 znamená:
  - Při richness=1 (nejbohatší zóna): 0 % zamítnutí.
  - Při richness=0 (nejchudší zóna): 30 % zamítnutí.
  - Po MAX_SPAWN_ATTEMPTS=5 retries pravděpodobnost, že chudá zóna *nikdy* nedostane food = 0.3⁵ = 0.24 % — prakticky nulová. Clustering ovlivňuje **ratio** spawn rate napříč zónami, ne absolute presence/absence.

- **Výstup:**
  - `lib.rs`: `pub const FOOD_REJECTION_STRENGTH = 0.3`, `pub fn reject_food_for_richness(rng, richness) -> bool` helper. 2 nové unit testy (`food_rejection_never_rejects_at_max_richness`, `food_rejection_rate_at_min_richness_matches_strength`). 34/34 testů pass.
  - `src/bin/headless.rs`: `World::new` initial food retry-loop, `spawn_food` rejection check před cell-exclusion. Reuse existujícího MAX_SPAWN_ATTEMPTS budgetu.
  - `src/main.rs`: zrcadlí headless — `setup` initial food retry, `spawn_food` system přidává `world_map: Res<WorldMapResource>` parametr a rejection check. Build clean.
  - **Smoke run (seed 0, 60 gen, headless):** populace 200 → oscilace 152–641 (gen 9–59), žádná extinkce. `noise_avg` v cell positions stabilně > baseline (0.5744): rozsah 0.572–0.605 napříč generacemi 19–59. Cells skutečně preferují rich zóny, jak očekáváno. Lineages 200 → 10 (standardní monokultura konvergence ze Sprintů 24+, ne důsledek clusteringu samotného). 9475 predation events v gen 29 → 5608 v gen 59 — predace pokračuje, dilution + damage signal aktivně modulují.
  - **Plné experimentální měření TBD.** Klíčové otázky:
    1. **Spatial niching:** roste `noise_avg` napříč generacemi nad baseline 0.5744 silněji než v pre-Sprint-31 (0.5–0.55)? Jak silně se cells koncentrují do rich zón?
    2. **Lineage trajectory:** zlomí se monokultura ze Sprintů 22–25? Tj. překoná spatial pressure scale × clustering tendence k jediné dominantní linii? Cíl: > 5 lineages na konci 200gen.
    3. **Vliv na damage signál:** korelace `dmg_avg` × `noise_avg`? Buňky v rich zonách jsou víc kousány? Pokud ano, damage input má co predikovat.
    4. **A/B vs. Sprint 30 baseline:** stejný seed, jeden běh s `STRENGTH=0.0` (pre-Sprint-31), druhý s 0.3, porovnat lineages, predation_events, vis_avg.
    5. **Sweep:** pokud mild clustering nestačí, zkusit `STRENGTH=0.5` (rich:poor spawn ratio ≈ 1:0.5 vs current 1:0.7). Watch for extinkční práh.

- **Poznámky:**
  - **Proč rejection-sampling, ne weighted-spawn?** Weighted distribution by vyžadovala precomputed CDF mřížku nebo per-candidate proportional sampling — extra paměť + složitost. Rejection sampling je „dumb but works": uniformní sample → boolean test → retry. MAX_SPAWN_ATTEMPTS budget eliminuje worst-case infinite loop a dělá clustering hladký, ne ostrý.
  - **Proč 0.3, ne středně-silné 0.6?** Sprint 21 v1 měl plnou sílu (= 1.0) a extinktoval gen 110. Lineární interpolace by čekala mezi-bod kolem 0.5 jako threshold. 0.3 je „pod prahem nebezpečí" s rezervou — mírný efekt s důvěrou v stabilitu. Pokud experimentální měření ukáže, že efekt je nedostatečný, A/B s 0.5 nebo 0.7 je triviální tuning. Sweep upward bezpečnější než downward (extinkce = ztráta dat).
  - **Carrion zůstává richness-agnostic záměrně.** Carrion follows death; if Sprint 31 selekce úspěšně tlačí cells do rich zón, carrion drop bude přirozeně biased k rich zonám (cells tam umírají). Nutit ho navíc přes mapu by maskovalo signal (jestli linie skutečně koncentruje vs. jestli jen carrion-bias dělá iluzi).
  - **Plus k Sprintu 23 hazard:** rich zone = high reward (víc food + vyšší value) × high risk (hazard drain). Sprint 30 damage signál teď dává brain možnost vědět, kdy je „v bohatství" trápen. Trojkombo richness/hazard/damage otevírá legitimní niche differentiation: efficient predátor v rich-dangerous, scavenger v poor-safe, mixovaný foraging v středu.
  - **Výhled:** pokud Sprint 31 trajectorie ukáže obnovu lineage diversity (víc než 5 linií), clustering je tu klíčový mechanismus a Sprint 32 může experimentovat s **multi-octave noise** (fine-grain food patches uvnitř coarse-grain biomes). Pokud lineages nadále kolabují k 1, problém není v food distribuci ale v predátorské dynamice — Sprint 32 by měl řešit predator gating (ne universal predator capability).

## Sprint 32 — substrate-3d-liftshift

- **Cíl:** převést všechny pozice, velocity a spatial datové struktury z `[f32;2]` na `[f32;3]` bez změny chování. Z-osa locked = 0 (initial spawn, brain output, bounce). Smyslem je oddělit **strukturální** změny od **sémantických** — vyřešit type-safety v jednom kroku, pak Sprint 33+ začne z čisté 3D základny. **Acceptance kritérium**: seed=0 headless 60 gen produkuje **identickou CSV** jako pre-Sprint-32.

  **Plán:**
  - `Cell.position`, `Cell.velocity`, `Food.position` typ `[f32; 2] → [f32; 3]`. Když `world_half[2] == 0` (Sprint 32 default), z-osa nedraw RNG, zachová pre-Sprint-32 reproducibility.
  - `Cell::step` integruje `position[2] += velocity[2] * dt` (vz=0 → no-op). Bounce na z-walls aktivní jen když `world_half[2] > 0`.
  - 3D distance squared (`+ dz²` vždy; pro z=0 příspěvek 0).
  - Spatial grid bucketing 2D → 3D — `(i32, i32, i32)` keys. Pro z=0 cells degenuje na single z-bucket → identical iteration order.
  - WorldMap a SmellField **zůstávají 2D** v lib.rs API (Sprint 35 promění je na 3D). Call sites projektují 3D pozici na xy.

- **Konstanty:** žádné nové. `WORLD_HALF[2] = 0.0` v binárkách.

- **Výstup:**
  - `lib.rs`: `Cell.position/velocity` 3D, `Food.position` 3D, `Cell::step` 3D-ready, `Cell::spike_bonus_against` 3D, `Cell::try_eat` 3D.
  - `src/bin/headless.rs` + `src/main.rs`: zrcadlí — `WORLD_HALF` / `WorldExtent` 3D, scratch buffers 3D, distance calcs 3D, projekce xy pro 2D fields.
  - **2 nové unit testy** (`step_3d_position_advances_with_z_velocity`, `z_locked_world_keeps_food_planar`). 36/36 testů pass.
  - **CSV identity ověřena**: seed 0 headless 60 gen, pre-Sprint-32 vs post-Sprint-32 výstup byte-identical (`diff` empty).

- **Poznámky:**
  - **Hypot vs naive sqrt**: `hypot(a, b)` ≠ `(a²+b²).sqrt()` na ULP úrovni. Sprint 32 ponechává hypot pro speed_norm calc (vz=0, identický výsledek). Sprint 33 přepne na 3D mag.
  - **v_mag_sq pro v² energy cost**: ponechán jako `vx² + vy²` v Sprint 32 (vz=0); Sprint 33 přidá `+ vz²`.
  - **Spatial grid 3D s z=0 overhead**: dz iterace v `for_each_in_radius` přidává triple-nested loop, ale pro z=0 cells je dz≠0 vždy empty bucket → identical visit order. Negligible perf.

## Sprint 33 — brain-io-3d (deferred z motion)

- **Cíl:** rozšířit brain dimenze pro 3D bez aktivace samotného z motion. `BRAIN_INPUTS_SENSORY: 15→20` (přidání food_dz, cell_dz, smell_dz, heading_z, ph_dz na konec, indices 15..20 — minimální disruption stávajícím indexům). `BRAIN_OUTPUTS: 7→8` (přidání turn_pitch). `Cell.pitch` + `Cell.pitch_velocity` fields. `forward_vector(yaw, pitch)` helper. 3D anisotropic drag math. Skutečné z motion (WORLD_HALF[2] > 0, pitch_velocity aktivní) odložené na Sprint 35.

  **Plán:**
  - Brain inputs comment + `pub const BRAIN_INPUTS_SENSORY = 20` (z 15). `pub const BRAIN_OUTPUTS = 8` (z 7).
  - Nový `Cell.pitch: f32` + `Cell.pitch_velocity: f32`. Init = 0 v `from_genome` + reproduce.
  - `forward_vector(yaw, pitch) -> [f32; 3]` helper v lib.rs. Pro pitch=0 redukuje na `(cos(y), sin(y), 0)` — backward kompat.
  - `Cell::step` 3D anisotropic drag: rozdělí velocity na along-forward + perpendicular (3D), drag_par váhuje `width`, drag_perp váhuje `length`.
  - `spike_bonus_against` cosine s 3D forward (Sprint 32 měl z=0 forward).
  - `Cell::step` energy cost: `v_mag_sq` teď `+ vz²` (Sprint 33+). Angular cost zůstává jen yaw² (kdyby zahrnovalo pitch², random brainy by měly 2× rotační drain → extinkce — Sprint 37 evaluuje).
  - Pitch clamp do ±π/12 (=15°) — tight conservative (Sprint 35 uvolní).
  - Brain_act in headless + main: populuje inputs[15] (food_dz / vision_r), [16] (cell_dz / vision_r), [17] = 0 (smell_dz, Sprint 35), [18] = sin(pitch) přes forward_vector helper, [19] = 0 (ph_dz, Sprint 35). Output[7] (pitch_signal) read ale **NE-aplikováno** na pitch_velocity — zůstává 0. Sprint 35 unlockne.
  - WORLD_HALF[2] = 0 — z dispersal odložené.

- **Konstanty:** `BRAIN_INPUTS_SENSORY = 20`, `BRAIN_OUTPUTS = 8`, `BRAIN_INPUTS = 28`. Žádné nové prahy.

- **Výstup:**
  - `lib.rs`: nové fields, helper, comment update. `Cell::step` 3D drag math.
  - `headless.rs` + `main.rs`: brain_act populuje 5 nových inputs (3 reálné + 2 nulové), čte ale neaplikuje output[7].
  - **Pozorovaná dynamika (seed 0, 30 gen, headless):** 200 → 529 cells, 19 lineages, 1380 predation events at gen 29. Stabilní population — brain dimenze rostly (28×8 + 8×8 vs pre-S33 23×8 + 7×8 = +48 weights), random brainy mají víc volných parametrů ale díky z=0 + pitch=0 jsou nové dimenze inert.

- **Poznámky:**
  - **Re-scoping z původního plánu**: original Sprint 33 plánoval aktivní z motion. Smoke testy ukázaly extinkci napříč mnoha tuninzích (z=270, 50, 20, 5). Cause: kombinace z dispersal cells/food + random brain pitch noise + brain dimension explosion. Cleaner: rozdělit Sprint 33 (I/O ready) a Sprint 35 (skutečný unlock) — minimum viable path.
  - **Heading semantics**: pre-Sprint-33 `inputs[9..11] = (cos(yaw), sin(yaw))`. Post-Sprint-33 = xy projekce 3D forward = `(cos(y)·cos(p), sin(y)·cos(p))`. Pro pitch=0 (Sprint 33 default) totožné. `inputs[18]` = `sin(pitch)` = 0.
  - **Recurrent slot index shift**: pre-Sprint-33 recurrent inputs[15..23] mapuje na last_hidden[0..8] s w1 rows 15..23. Post-Sprint-33 [20..28]. Nová Brain::random matrix má random weights na nových řádcích — brainy nejsou backward-kompatibilní s pre-Sprint-33 saved státem (žádný save/load v projektu, takže nedělá problém).

## Sprint 34 — body-3-axis-ellipsoid

- **Cíl:** rozšířit body morfologii o třetí osu (height) — ellipsoid `length × width × height`. Brain output[8] = morph_height (appended kvůli zachování existujících output indexů). Pro pre-Sprint-34 srovnatelnost: `body_height = 1.0` jako default činí tělo backward kompat (volume = length × width × 1 = area). Mutace + selekce mohou ladit asymetrii v 3D.

  **Plán:**
  - `Genome.body_height: f32` + `Phenotype.body_height: f32` + `MutationConfig.sigma_body_height: f32 = 0.05`. Random init `body_height = body_size` (== length, width — izotropní).
  - `MIN_BODY_HEIGHT = 0.3`, `MAX_BODY_HEIGHT = 4.0` (mirror length/width).
  - `Phenotype::effective_radius` = `(length + width + height) / 3.0` (3-osý průměr; pro length=width=height=s vrátí s).
  - `Phenotype::area` → `Phenotype::volume()` = `length × width × height`. Maintenance `cell.energy -= phenotype.volume() × body_cost_factor × dt`. Pro height=1 redukuje na area cost.
  - `Phenotype::apply_morph` rozšířen na `[f32; 4]` (length, width, height, spike). `Cell::apply_morph` mapuje výstupy `[outputs[3], outputs[4], outputs[8], outputs[5]]`.
  - `BRAIN_OUTPUTS = 9` (přidání morph_height na index 8).

- **Konstanty:** `MIN_BODY_HEIGHT = 0.3`, `MAX_BODY_HEIGHT = 4.0`. `MutationConfig.sigma_body_height = 0.05`.

- **Výstup:**
  - `lib.rs`: Genome + Phenotype + MutationConfig + apply_morph rozšířené o height. effective_radius/volume rovnice.
  - `headless.rs`: nový CSV sloupec `hgt_avg`. 36/36 testů pass (Phenotype literály ve testech aktualizovány).
  - **Pozorovaná dynamika (seed 0, 30 gen, headless):** 200 → 110 cells (gen 29), 12 lineages. `hgt_avg` 0.987 → 1.356 — selekce aktivně pracuje s height (mírně roste z baseline 1.0). `len_avg` 1.530, `wid_avg` 0.912 — asymmetric tělesa se objevují.

- **Poznámky:**
  - **Append index pattern**: morph_height na output[8] (za attack[6] a turn_pitch[7]) místo restrukturalizace. Cena: `Cell::apply_morph` čte non-contiguous indices `[3, 4, 8, 5]`. Výhoda: zachovává všechny stávající output indexy (`ATTACK_THRESHOLD` na [6], pitch na [7]).
  - **Sprint 33 anisotropic drag math** byla připravená pro 3D (perpendicular split na full 3D vector). Sprint 34 jen aktivuje height v drag rovnici skrze `width × height` cross-section pro forward, `length × height` pro side, `length × width` pro vertical. Pro length=width=height=s všechny váhy s² (isotropic).
  - **Genotyp/fenotyp split** beze změny — runtime morph mění jen `Phenotype.body_height`, ne `Genome.body_height`. Dítě dostane fresh phenotype z parent genomu.

## Sprint 35 — z-motion-unlock (volumetric environment minimum)

- **Cíl:** aktivovat skutečné z motion. Cells startují uniformly v z ∈ [-Z, Z], food také, brain output[7] (turn_pitch) řídí pitch_velocity, cells se pohybují ve 3D objemu. WorldMap + SmellField/Pheromone zůstávají 2D (xy projekce) — plné volumetric pole odložené na pozdější sprint kvůli scope. Tento sprint = "minimum viable 3D life".

  **Plán:**
  - `WORLD_HALF[2] = 2.0` v headless + `SIMULATION_HALF[2] = 2.0` v main. Velmi mírný 3D layer: cells v z ∈ [-2, 2], eat_radius (~8) pokryje plný z range, ale ellipsoid morfologie + pitch motion mají co dělat.
  - Brain_act in headless + main: aktivuje `pitch_acc = pitch_signal × turn_rate / body_proxy`, `cell.pitch_velocity += pitch_acc × dt`. Thrust podle 3D forward vektoru přes `forward_vector(yaw, pitch)` (Sprint 33 helper).
  - Pitch clamp do ±π/12 (=15°) — Sprint 37 evaluuje uvolnění.
  - Carrion drop respektuje cell.position[2] (clamped do world bounds).
  - `Cell::random` a `Food::random` aktivně využívají `world_half[2] > 0` cestu (1 RNG draw na z position).

- **Konstanty:** `WORLD_HALF[2] = 2.0`. Pitch clamp v `Cell::step` ±π/12.

- **Výstup:**
  - `lib.rs`: pitch clamp range upraven na ±π/12 (z původního ±π/2 v Sprint 33 plánu, který způsobil extinkci).
  - `headless.rs` + `main.rs`: pitch_velocity aktivováno v brain_act, WORLD_HALF[2] = 2.0.
  - **Pozorovaná dynamika (seed 0, 30 gen, headless):** 200 → 349 cells (gen 29), 17 lineages, 1163 predation events. Pop stabilní, `hgt_avg` 1.499 (selekce dál ladí ellipsoid). 3D motion plně funkční.
  - **Tuning iterace** (smoke runy seed 0, 30 gen, hledání stable z config):
    - z=270 (full 9× volume): extinct gen 4 (food density per volume drop 9×).
    - z=50: extinct gen 6.
    - z=20: extinct gen 8.
    - z=5 + pitch ±π/12: 1 cell at gen 30 (kolaps).
    - **z=2 + pitch ±π/12: 349 cells (current default).**

- **Poznámky:**
  - **Re-scoping z původního plánu**: original Sprint 35 měl 3D volume noise (64×64×16 anisotropic) + 3D Jacobi diffusion (32×32×16). Implementace by byla 500+ LOC. Pro tento sprint zachované 2D fields s xy projekcí — cells létají v 3D, ale environment cítí 2D ploche. Plné volumetric pole je výzva pro Sprint 38+.
  - **Pop kolaps při větším z**: random brainy s INNATE_THRUST_BIAS (forward) + random pitch_signal noise saturují pitch range; thrust se rozkládá na xy + z, cells v xy ztratí navigation v xy. Při z=2 (~half eat radius) je ten efekt minimální.
  - **Co Sprint 35 NEMĚNÍ**: WorldMap (stále 2D), SmellField (stále 2D), pheromone (stále 2D). Brain inputs[17] (smell_dz), [19] (ph_dz) zůstávají 0 — gradient v z neexistuje, dokud pole nejsou volumetric. Cells navigují z přes vision (food_dz, cell_dz already populated since Sprint 33) + memory (Sprint 28 recurrent).
  - **Pitch range conservativeness**: ±π/12 = 15°. Při full thrust to znamená max 26 % thrust v z, 96 % v xy. Bezpečné pro random brainy. Selekce může chtít víc; Sprint 37 ladí.

## Sprint 36 — 3d-renderer (DEFERRED)

- **Cíl:** převést Bevy renderer z 2D pipeline na 3D — `Camera2d → Camera3d`, `Mesh2d → Mesh3d`, custom `Material2d` shader (`cell_material.wgsl`) port na `Material3d`, sphere/ellipsoid mesh místo teardrop, lighting (DirectionalLight + AmbientLight), kameraový orbit/fly control, WorldMap overlay jako bottom plane texture. Spike rendering ve 3D = elongation podél 3D forward axisu.

- **Status: DEFERRED.** Implementační scope ~500+ LOC + WGSL port = sprint-sized session sám o sobě. Při kompletní 2D→3D session bylo prioritou simulační logika (Sprinty 32–35). Renderer port následuje:

  **Plán pro budoucí session:**
  - `cell_material.rs`: `Material2d` → `Material` (Bevy 0.18 Material3d). Shader binding bude potřebovat `bevy_pbr::mesh_functions` namísto `bevy_sprite::mesh2d_functions`. Vertex output struct potřebuje 3D `clip_position`.
  - `cell_material.wgsl`: vertex stage rewrite — `mesh_functions::get_world_from_local`, `mesh_position_local_to_world`, `mesh_position_world_to_clip`. Spike extension v world space podél 3D heading.
  - `main.rs::setup`: `commands.spawn(Camera3d)` na elevated angle s perspective projection. `DirectionalLight` + `AmbientLight` resource. Window setup beze změny.
  - Cell mesh: `Sphere::new(CELL_RADIUS).mesh()` místo teardrop. Spike rendering přes shader vertex extension (jako 2D, ale v 3D world frame).
  - Food mesh: `Sphere::new(FOOD_RADIUS).mesh()`.
  - `cell_scale`: `Vec3::new(length, height, width)` (Bevy 3D má Y up, takže length × width × height mapping pojde přehodit). Anisotropic ellipsoid.
  - WorldMap overlay: bottom plane (`Plane3d::new`) s texture z grayscale field. Z stratification: z=lowest_layer.
  - Camera control: orbit přes mouse drag, scroll = zoom. Implementace je ~50 LOC s `bevy_panorbit_camera` crate (nebo manuální).

- **Mezistav po Sprintech 32–35**: simulace produkuje 3D dynamiku (cells s position[2] ≠ 0, pitch motion, height morfologie), renderer pořád 2D — cells vidíme v xy projekci, z dimenze viz není. Headless + CSV jsou zdroj pravdy pro 3D measurement; renderer slouží 2D viz.

## Sprint 37 — measurement (DEFERRED)

- **Cíl:** plné experimentální měření 3D simulace. Hyperparameter sweep pro pitch range, z-volume, food density v 3D, body_height selection pressure. A/B porovnání 2D (Sprint 31 baseline) vs 3D (Sprint 35).

- **Status: DEFERRED.** Sprint 37 byl plánovaný jako measurement run (200+ gen × multiple seeds × hyperparameter combinations), ne implementační. Po Sprint 35 je infrastruktura pro tato měření připravená.

- **Klíčové otázky:**
  1. **Z-stratification**: vyvine se vertikální segregace nik (some cells preferují z>0, jiní z<0)? Měřitelné přes pop-binned `mean_z` per lineage.
  2. **Body-axis evolution**: jak se distribuuje `len_avg`, `wid_avg`, `hgt_avg` napříč generacemi? Vyvine se „flat-fish" (high length+width, low height) vs „eel" (high length, low width+height) niche?
  3. **Pitch usage**: kolik procent populace má `|pitch|` výrazně > 0 napříč generacemi? Nebo brain naučí pitch=0 jako optimum?
  4. **Diversifikace lineages**: vrátí se Sprint 23 paterně 20+ lineages (3D world dává víc nik), nebo monokultura ze Sprintů 22+ vyhrává?
  5. **A/B vs 2D**: stejný seed, jeden běh s `WORLD_HALF[2]=0` (2D), druhý s `WORLD_HALF[2]=2` (3D mírný). Porovnat lineages, predation_events, hgt_avg trajectory.

- **Hyperparameter sweep candidate**:
  - z-volume: 2.0 (current), 5.0, 10.0, 30.0
  - pitch_clamp: π/12 (current), π/8, π/6, π/4
  - food density (`WORLD_UNITS_PER_FOOD`): 2600 (current), 1300 (2× more food)
  - sigma_body_height: 0.05 (current), 0.1, 0.0 (no height mutation)

- **Recommended startup**: 200 gen × 5 seeds × current default = baseline. Pak 1-2 hyperparameter změny = prove principle.

## Sprint 38+ — TBD

Možné směry po dokončení 3D substrátu:
- **3D volumetric environment**: WorldMap → 64×64×16 3D value noise. SmellField → 32×32×16 3D Jacobi. Hazard + food richness 3D. Brain inputs[17] (smell_dz) a [19] (ph_dz) populated z 3D gradient.
- **3D renderer port**: Sprint 36 deferred work. Camera3d + custom Material3d + sphere mesh + lighting.
- **Vertical strata**: explicit "biome strata" v z (rich pelagic, sparse benthic). Forced niching skrze prostorovou heterogenitu v z.
- **Cells s body_height > body_length+width**: testovat, zda evoluce favorizuje extreme aspect ratios (long thin "wormy" tělesa, flat plate-like, tall cones). Vrhne to světlo na to, jestli má 3D shape genuinely jiné hodnoty než 2D shape.
- **Roll**: pokud experimenty ukážou, že yaw + pitch nestačí (cells potřebují roll pro inverted attack např.), přidat 3D angular velocity vector + quaternion heading.
- **Hebbian na pitch**: pokud Sprint 37 ukáže slabou pitch selection, fallback Hebbian update na pitch reward (eat_food + pitch directionality korelace).
