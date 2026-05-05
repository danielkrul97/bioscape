# Sprinty 73–82: Tissue emergence

Decade přechází z infrastructure / scale-up (63-72) k cílenému hledání
tipping pointu pro multicelularitu. Sprint 71 + 72 prokázaly, že
non-evolving macropredator persistuje (žádná extinkce), ale cells
opakovaně našly escape route přes pure speed (Sprint 71: outrun blízko
hunter speed; Sprint 72: outrun NAD hunter speed přes nelimitovaný
mutation drift). Sprint 73 zavedl strukturální `MAX_SPEED` cap →
spd_avg pinned at 193 (cap 200), ale bonding stále crashed na 0 do gen
299. Decade 73-82 hledá zbývající tunable, který přepne ekonomiku tak,
aby cluster path > solo path. Hypotéza: chyba je v cost-benefit poměru
hunt damage vs bond maintenance.

## Sprint 73 — `MAX_SPEED` cap + 1000-gen verification

- **Cíl:** přidat horní cap pro `genome.max_speed`, aby cells nemohly
  outrun hunteru pomocí mutation drift (Sprint 72 1000-gen ukázal cells
  evolovat speed na 344 vs HUNTER_MAX_SPEED=300 — cap absent →
  arms race degenerative).

  **Plán implementace:**

  *Body 1 — Lib const + clamp v `Genome::mutate`:*
  - Nový `pub const MAX_SPEED: f32 = 200.0` (mírně pod Sprint 71
    baseline 218 → HUNTER_MAX_SPEED=300 reálně neutekatelný).
  - `Genome::mutate`: změna `.max(MIN_SPEED)` → `.clamp(MIN_SPEED, MAX_SPEED)`.

  *Body 2 — Test guard:*
  - Existující `mutate_respects_genome_bounds` test rozšířen o
    `assert!(m.max_speed <= MAX_SPEED, "Sprint 73: speed cap respected")`.

- **Konstanty:**
  - `MAX_SPEED: f32 = 200.0` nový.

- **Výstup:**
  - `src/lib.rs`: const + clamp + test assertion.
  - **Test suite: 95/95 pass.**
  - **Long-run smoke seed=0, 1000 gen, default world, CPU:**
    - Wall-clock 834.5 s = **719 ticks/s**. Final pop **623** (Sprint 72: 390
      — cap zachoval populační zdraví, žádný extreme-speed energy crash).
    - **Cap funguje** ✓ — max `spd_avg = 193.08` @ gen 682 (cap 200), pinned.
    - **Hunters maintain pressure** end-to-end: 1500-2000 atks/gen
      napříč gen 100-1000, žádná extinkce.
    - **Bonding NEDRŽÍ:**

      | Gen | spd | asp | mBond | hunt_atks | immune_frac |
      |-----|-----|------|-------|-----------|-------------|
      | 50  | 148 | 3.5  | **0.039 (peak)** | 1523 | 0 |
      | 99  | 168 | 7.3  | 0.015 | 1597 | 0 |
      | 199 | 187 | 11.2 | 0.003 | 1497 | 0 |
      | 299 | 189 | 12.7 | **0.000** | 1981 | 0 |
      | 999 | 191 | 12.4 | 0.000 | 1538 | 0 |

- **Závěr — cap funguje, energy ekonomika ne:**
  - **Cap zlomil outrun.** spd_avg 193 vs HUNTER_MAX_SPEED 300 →
    cells nemůžou rychlostí utíkat. ✓
  - **Ale bonding stále crashed na 0** do gen 299. Problém není v
    rychlosti, ale v **fitness math**:

    | Strategy | Energy cost / gen |
    |----------|-------------------|
    | Solo cell, 1500 hunt atks/gen × 4.0 dmg × dt | ≈ 0.17 / cell |
    | Bonded 3-cell, BOND_MAINTENANCE_PER_SEC 0.1 × 10s × 3 | 3.0 / cell |

    Bonded cell platí **18× víc** než solo cell ztrácí na hunt damage.
    Selekce favorizuje solo, ne cluster. Cap zastavil outrun, ale
    *energetic incentive* pro cluster nikdy nepřišel.

- **Implikace pro Sprint 74+:**
  - **Rebalance hunt damage vs bond maintenance.** Konkrétní páky:
    1. `HUNTER_DAMAGE_PER_TICK` 4.0 → 12.0+ (3× lethaler)
    2. `HUNTER_TARGET_COUNT` 8 → 16 (2× víc atks)
    3. `BOND_MAINTENANCE_PER_SEC` 0.1 → 0.02 (5× cheaper bonding)
    Cíl: solo cost > bonded cost při typical bond_count = 3.

  - **Plus: hunter target persistence.** Aktuálně hunter každý tick
    re-vyhledává nearest target. Pokud target uteče, hunter okamžitě
    přepne. Real predátor by chvíli pursued. Sprint 74+ může přidat
    persistent target ID + chase duration → cells nemůžou jen
    momentárně utéct, hunter zůstane v close pursuit.

- **Poznámky:**
  - **Cell pop zdravější** (623 vs Sprint 72's 390). Bez extreme speed
    cells (337 spd → energy v² cost 91/s) jsou v energetic stress lower
    a víc cells přežívá k reprodukci. Cap má tedy positive side effect.
  - **asp_avg pořád 12.4** = max elongation. Cells ztratily „rychlost"
    ramp, ale streamlined body shape stále dominuje (energy savings
    z menšího cross-sectional drag). To je sustainable phenotype, jen
    bez klíčové defense.
  - **Co Sprint 73 NEŘEŠÍ (Sprint 74+):**
    - Energy ekonomika hunt vs bond. **Critical bottleneck.**
    - Hunter persistence / chase logic.
    - HUD overlay s real-time bond / hunter / immune_frac stats.
    - GPU collision wire-up, photic stratification, anizotropic collision.

## Sprint 74 — economy rebalance (hunt damage vs bond maintenance)

- **Cíl:** flip energy ekonomiku tak, aby bonded cluster path > solo path.
  Sprint 73 ukázal: solo cells trpí jen 0.17 energy/gen na hunt damage,
  ale bonded 3-cell platí 3.0/gen na maintenance → **18× inverted**.
  Sprint 74 zkouší 3 kombinované páky najednou:
  1. `HUNTER_DAMAGE_PER_TICK` 4 → **8** (2× lethaler hunt)
  2. `HUNTER_TARGET_COUNT` 8 → **12** (1.5× víc hunterů)
  3. `BOND_MAINTENANCE_PER_SEC` 0.1 → **0.05** (2× cheaper bond)

  Cíl: solo cost ~0.5/gen, bonded ~1.5/gen → ~3× inverted, blíže k
  break-even.

  **Plán:** parameter-only change, žádný structural code change.

- **Konstanty (lib.rs):**
  - `HUNTER_DAMAGE_PER_TICK` 4.0 → 8.0
  - `HUNTER_TARGET_COUNT` 8 → 12
  - `BOND_MAINTENANCE_PER_SEC` 0.1 → 0.05

- **Výstup:**
  - `src/lib.rs`: 3 const updates + comments.
  - **Test suite: 95/95 pass.**
  - **Long-run smoke seed=0, 1000 gen, default world, CPU:**
    - Wall-clock 756.7 s = **793 ticks/s** (S73 719). Final pop 614.
    - Hunt attacks 2500-3200/gen napříč gen 50-1000 (S73: 1500-2000) ✓.
    - **Bond density signal:**

      | Gen | spd | mBond | hunt_atks | immune_frac |
      |-----|-----|-------|-----------|-------------|
      | 36  | ~120 | **0.068 (peak, +50 % vs S73)** | ~3500 | 0 |
      | 49  | 137 | 0.022 | 3193 | 0 |
      | 99  | 184 | 0.009 | 2847 | 0 |
      | 199 | 189 | **0.000** | 3131 | 0 |
      | 999 | 189 | 0.000 | 2470 | 0 |
    - Peak `mean_bond_count = 0.068 @ gen 36` — **+51 % vs Sprint 73**.
    - Peak `immune_frac = 0.000` — **NIKDY** žádná cell s ≥3 bondů.

- **Závěr — částečný progress, fundamentální blok zůstává:**
  - **Bondová formace je vyšší** (peak 0.068 vs S73 0.045). Economy
    rebalance posunula needle správným směrem.
  - **Ale cluster path stále nedosáhnut.** Bondy se formují častěji,
    ale typicky jen 1 per cell. **immune_frac je 0** — žádná cell
    nikdy nezvládla nakumulovat 3+ bondy.
  - **Důvod není v ekonomice maintenance, ale v formation rate.**
    Bond formation vyžaduje:
    1. Same adhesion_type (8 typů → 1/8 šance random pair match)
    2. Oba cells `output[9] > BOND_FORM_THRESHOLD=0.2`
    3. Prolonged contact `BOND_FORM_TICKS=30` (0.5 s @60 Hz)

    Cells se pohybují speed 190 → contact je krátký, protnou se
    mihem. Plus `INNATE_BOND_BIAS=0` → random brainy nemají prior,
    selekce na bond_signal output[9] je slabá. Výsledek: bondy se
    formují sporadicky a dříve než cell stihne přidat druhý/třetí,
    původní prudí na overstretch.

- **Implikace pro Sprint 75+:**
  - **Atak na formation gating, ne na maintenance cost.** Páky:
    1. `INNATE_BOND_BIAS` 0.0 → **1.5** — všechny cells dávají
       output[9] > threshold by default. Bondy se formují bez
       brain-evolution preconditiony. Selekce může negativně tunit
       (cells co nechtějí bondovat se učí suppress).
    2. `BOND_FORM_TICKS` 30 → **10** (~0.17 s) — bond se vytvoří
       i z briefer contactu. Risk: nestabilní bondy formující se
       z náhodného mihem.
    3. `BOND_FORM_THRESHOLD` 0.2 → 0.0 — odstraní brain consent gate
       úplně. Filozoficky největší změna; bondy by formovaly výhradně
       physics (contact + same type).
  - Nejlepší first attempt: **#1 alone**. Filozoficky umírněné (cells
    pořád musí emit output[9], ale dostávají bias jako attack).
    Pokud nedotáhne immune_frac > 0.05, eskalace na #1 + #2 v Sprint 76.

- **Poznámky:**
  - **Cells konvergovaly k stejnému stable state** jako Sprint 73:
    spd 189-192 (cap pinned), asp 12.6 (extreme elongation), spk 0.05
    (vestigial), e_avg 73 (low-energy stable). Hunt damage zvýšen,
    ale cells absorbují (population 614 — comparable s S73's 623).
  - **Hunters zůstávají persistent** end-to-end: 2470-3193 atks/gen
    napříč 1000 gen. Macropredator design (Sprint 71) je robustní.
  - **Co Sprint 74 NEŘEŠÍ (Sprint 75+):**
    - Bond formation rate (real bottleneck — Sprint 75 priorita).
    - Hunter persistent chase logic.
    - HUD overlay.

## Sprint 75 — INNATE_BOND_BIAS = 1.5

- **Cíl:** odbourat bond formation gating. Sprint 74 ukázal, že bondy
  formace jsou **real bottleneck** (ne maintenance cost) — random brainy
  s `INNATE_BOND_BIAS=0` dávají `output[9] > 0.2` jen sporadicky, takže
  bondy se nestihnou hromadit do clusteru ≥3 (= immune k hunteru).
  Sprint 75 zvyšuje bias na 1.5 → většina cells emituje signal nad
  threshold by default. Selekce může pak negativně tunit (cells co
  nechtějí bondovat se učí brain weights pull b1[9] dolů). Mirror Sprint
  27 INNATE_ATTACK_BIAS philosophy.

- **Konstanty:**
  - `INNATE_BOND_BIAS` 0.0 → 1.5

- **Výstup:**
  - `src/lib.rs`: 1 const update + comment.
  - **Test suite: 95/95 pass.**
  - **Long-run smoke seed=0, 1000 gen, default world, CPU:**
    - Wall-clock 672.8 s = **892 ticks/s** (S74 793, +12 %). Final pop 631.
    - **Bond density:**

      | Gen | spd | asp | mBond | hunt_atks | immune_frac |
      |-----|-----|------|-------|-----------|-------------|
      | 31  | ~110 | ~1.7 | ~0.087 | ~5000 | **0.002 (peak!)** |
      | 38  | ~120 | ~2.0 | **0.088 (peak)** | ~4500 | ~0.001 |
      | 49  | 148 | 2.2  | 0.051 | 2485 | 0 |
      | 99  | 186 | 3.0  | 0.021 | 3015 | 0 |
      | 199 | 191 | 10.2 | 0.000 | 2150 | 0 |
      | 999 | 190 | 12.6 | 0.000 | 2855 | 0 |
    - Peak `mean_bond_count = 0.088 @ gen 38` — **+29 % vs Sprint 74,
      +96 % vs Sprint 73** baseline.
    - Peak `immune_frac = 0.002 @ gen 31` — **PRVNÍ non-zero immune_frac
      napříč Sprinty 73-75!** 0.2 % cells dosáhlo ≥3 bondů.

- **Závěr — průlom v signálu, ale ne v ekonomice:**
  - **Bias funguje mechanicky** ✓. Random brainy začínají bond signal
    nad threshold → physics-driven bond formation. Bond density narostla
    napříč všech 3 metrik (mBond, bond_active_frac, immune_frac).
  - **Ale selekce stále favorizuje solo.** Po gen 50 cells s bondy
    rapidně mizí (gen 199 mBond=0, gen 999 mBond=0). Bias dostává cells
    do bondů, ale **bonded lineages nereprodukují víc než solo**. Cluster
    advantage neexistuje — cells rychle losslužejí bond cost (formation +
    maintenance) bez compensating fitness benefit.
  - **Imbalance kořeny:** `HUNTER_BOND_IMMUNITY_THRESHOLD=3` je extrémní
    benchmark. Cell musí mít 3 bondy current jednou — to znamená contact
    s 3 různými same-type cells během krátké doby. S asp=2-3 (gen 30-50)
    je to možné, ale jakmile cells konverguje k asp=12 needles, contact
    je nemožný (1D čára nemá v xy projection prostor pro 3 sousedy).

- **Implikace pro Sprint 76+:**
  - **Lower threshold or direct reward.** Tři páky:
    1. `HUNTER_BOND_IMMUNITY_THRESHOLD` 3 → **2** — dramaticky snadnější
       cíl. Sprint 75 mělo 0.2 % @ 3-bond; @ 2-bond by se to mohlo
       posunout k 5-10 %. Single-line change.
    2. **Cluster food share** — když bonded cell eats food, share x %
       energy s bonded partners. Direct positive reinforcement bondingu.
       Větší scope (touch eat_food v lib + headless + main).
    3. `BOND_FORM_TICKS` 30 → **10** — bondy formují z briefer contacts
       → cells co se náhodně mihnou nestihnou bond. Risk: přefiltrované
       random clustery.
  - Kombinace **#1 + #3** je low-risk, single-file edit. **#2** je
    fundamentálnější ale invasive.

- **Poznámky:**
  - **Wall-clock +12 % rychlejší** vs Sprint 74 (892 vs 793 ticks/s).
    Důvod: víc bondů → cells lokálnější → cell_grid lookup cheaper.
  - **Asp konverguje k 12.6 NEZÁVISLE na bond bias.** Streamlined needles
    jsou attraktor pro foraging energy efficiency, ne anti-predator
    strategy. Bonded cells s asp 2-3 (gen 30-50) byly aircon-friendly
    pro cluster, ale jakmile asp evolovalo k 12, fyzicky nemohli bondovat.
    Body shape evolution race-condition s bond formation.
  - **Co Sprint 75 NEŘEŠÍ (Sprint 76+):**
    - Threshold pro immunity (3 je možná moc).
    - Direct cluster reward (energy sharing).
    - Anisotropic body penalty pro cluster-incompatible shapes.

## Sprint 76 — lower thresholds (immunity 3→2 + form_ticks 30→10)

- **Cíl:** snížit dosažitelný cíl pro multicelularitu. Sprint 75 dosáhlo
  immune_frac=0.002 @ 3-bond threshold = symbolický průlom, ale evolučně
  marginal. Sprint 76 zkouší dva současně:
  1. `HUNTER_BOND_IMMUNITY_THRESHOLD` 3 → **2** — pair / triad cluster
     stačí na immunity (ne plnokrevné quartet).
  2. `BOND_FORM_TICKS` 30 → **10** (~0.17 s) — bondy se formují i z
     krátkých contactů, aby cells s speed 190 nestihly utéct před
     formací.

- **Konstanty:**
  - `HUNTER_BOND_IMMUNITY_THRESHOLD` 3 → 2
  - `BOND_FORM_TICKS` 30 → 10

- **Výstup:**
  - `src/lib.rs`: 2 const updates + comments. Existující `0..3` bond
    setup v hunter testech zůstává valid (3 bondy stále ≥ threshold 2).
  - **Test suite: 95/95 pass.**
  - **Long-run smoke seed=0, 1000 gen, default world, CPU:**
    - Wall-clock 696.6 s = **861 ticks/s**. Final pop 575.
    - **MAJOR breakthrough:**

      | Gen | spd | asp | mBond | hunt_atks | immune_frac |
      |-----|-----|------|-------|-----------|-------------|
      | 2   | ~75 | ~1.0 | ~0.15 | ~5500 | **0.034 (peak!)** |
      | 27  | ~95 | ~1.5 | **0.210 (peak!)** | ~3500 | 0.025 |
      | 49  | 143 | 1.85 | **0.149** | 2633 | **0.031** |
      | 99  | 163 | 6.19 | 0.092 | 2564 | 0.008 |
      | 199 | 187 | 11.3 | 0.000 | 3216 | 0 |
      | 999 | 190 | 12.6 | 0.000 | 2610 | 0 |

    - **Peak `mBond=0.210` = 2.4× vs Sprint 75's 0.088, 4.7× vs S73 baseline.**
    - **Peak `immune_frac=0.034` = 17× vs Sprint 75's 0.002.** 3.4 %
      populace dosáhlo immunity (vs marginal 0.2 %).
    - **Sustained tissue regime gen 27-99:** mBond > 0.09, immune_frac
      pohybuje 0.008-0.034. Krátká, ale **proto-tissue real**.

- **Závěr — proto-tissue confirmed, ale pořád transient:**
  - **Mechanic průlom:** snížení threshold + form_ticks otevřel prostor
    pro cluster formation, který Sprint 73-75 absent. Cells s asp~1.5-2
    v gen 25-50 fyzicky vytvářely 2-3 bondové mini-clustery.
  - **Selekce stále nakonec bonded population eliminuje.** Gen 99 už
    asp=6.2, gen 199 asp=11.3 — streamlining race vyhrál nad clustering.
  - **Asp=12 needle attractor je strukturální problém.** Cells s asp 12
    fyzicky nemůžou clusterovat s 2 sousedy v rovině; bonded clustery
    z gen 25-50 (asp 1.5) byly geometricky možné, ale jakmile populace
    konvergovala k needle phenotype, cluster path se zavřel.

- **Implikace pro Sprint 77+:**
  - **Cap asp.** Tři páky k zamezení needle attractor:
    1. **`MIN_BODY_WIDTH` 0.3 → 0.8** (also `MIN_BODY_HEIGHT`) — cap
       asp na ~5 (length 4 / width 0.8). Single-line config change.
    2. Asp-aware bond formation gating — `if asp > 5: no bond`.
       Cells přímo selektovány proti extreme elongation pro cluster path.
    3. Drag model rebalance — currently anisotropic drag favorizuje
       streamlined; pokud rebalance ke „rounder is faster", asp evolves
       differently. Větší scope, fundamentální fyzika change.
  - **Doporučuju #1** (config change). Sprint 73 pattern (single
    const + 1000-gen verify) drží.

- **Poznámky:**
  - **Wall-clock 861 ticks/s** comparable s S74/S75 (793, 892). Hunt
    phase + collision phase oba zatížené, ale dohromady stable.
  - **Hunter pressure end-to-end** ✓: 2127-3216 atks/gen. Constant.
  - **Co Sprint 76 NEŘEŠÍ (Sprint 77+):**
    - Asp=12 needle attractor (real bottleneck pro tissue stability).
    - Cluster food share, persistent hunter chase, HUD.

## Sprint 77 — `MIN_BODY_WIDTH` 0.3 → 0.8 (cap asp ~5)

- **Cíl:** zlomit asp=12 needle attractor (Sprint 76 diagnóza). Hypotéza:
  cap asp na 5 přes MIN_BODY_WIDTH bump → cells si zachovají roundish
  body → cluster path zůstává geometricky viable napříč generací.

- **Konstanty:**
  - `MIN_BODY_WIDTH` 0.3 → 0.8 (asp_max = 4.0/0.8 = 5.0)
  - Test `morph_returns_total_absolute_delta` updated (delta 2.3 → 1.8
    kvůli novému clamp).

- **Výstup:**
  - `src/lib.rs`: 1 const update + 1 test fix.
  - **Test suite: 95/95 pass.**
  - **Long-run smoke seed=0, 1000 gen, default world, CPU:**
    - Wall-clock 613.8 s = **978 ticks/s** (S76 861, +14 % — smaller
      cells = lower compute cost). Final pop 398.
    - **Asp cap funguje** ✓: max asp = 4.885 @ gen 847 (vs S76's 12.73).
    - **Bond density LOWER vs Sprint 76:**

      | Gen | spd | asp | mBond | hunt_atks | immune_frac |
      |-----|-----|------|-------|-----------|-------------|
      | 3   | ~75 | ~1.0 | ~0.10 | ~5500 | **0.027 (peak)** |
      | 35  | ~110 | ~1.4 | **0.178 (peak)** | ~3500 | ~0.020 |
      | 49  | 151 | 1.6  | 0.061 | 2671 | 0.002 |
      | 99  | 182 | 3.9  | 0.015 | 2494 | 0.002 |
      | 199 | 191 | 4.7  | 0.000 | 2349 | 0 |
      | 999 | 193 | 4.7  | 0.000 | 2313 | 0 |
    - Peak `mBond=0.178` (S76: 0.210, **−15 %**).
    - Peak `immune_frac=0.027` (S76: 0.034, **−21 %**).

- **Závěr — hypotéza falsifikována:**
  - **Cap funguje mechanicky** (asp pinned 4.7 < cap 5.0), ale **nepomáhá
    bondingu**. Cluster path stále crashed po gen 200.
  - **Real bottleneck není body shape, ale fitness reward.** I s
    cluster-friendly bodies cells nedostávají dost benefitu z bondingu;
    selekce stále favorizuje solo path.
  - **Side effect: pop crash 575 → 398** (–31 %). Cells se širší body
    mají větší volume → vyšší maintenance + drag, méně cells přežívá
    do reprodukce.
  - **Cap zůstává v kódu** — prevence regrese k extrém asp je biologicky
    obhájitelná, ale není to wow effect.

- **Implikace pro Sprint 78+:**
  - **Cluster food share = direct fitness reward.** Pokud bonded cell eats
    food, share x % energy s bonded partners. Vytváří **explicit positive
    selection signal** pro bondování — bonded cells dostanou víc energie
    na osobu (multiplicative), což překlopí ekonomiku.
  - Implementace touch eat_food fáze v lib + headless + main. Větší scope
    než parameter tweaky, ale fundamentálnější.

- **Poznámky:**
  - **Wall-clock fastest yet** (978 ticks/s) — menší cells, menší volume,
    levnější collision detection.
  - **Asp pinned at 4.7 (just under cap 5)** — selekce pořád tlačí k
    streamline, jen je teď bounded. Streamlining-as-attractor je
    independent na bonding mechanic.
  - **Co Sprint 77 NEŘEŠÍ (Sprint 78+):**
    - Direct fitness reward pro bonding (cluster food share).
    - Hunter persistent chase, HUD, photic/thermal.

## Sprint 78 — cluster food share (BREAKTHROUGH)

- **Cíl:** přidat **direct fitness reward** pro bondování. Sprint 73-77
  ukázaly, že hunter immunity (přes bond defense) je nedostatečný benefit
  — bondy sice formují, ale selekce nepreserve bonded lineages. Sprint 78
  zkouší **explicit positive selection signal**: bonded cells sdílejí
  fraction eaten food s partnery, takže být v clusteru = víc energie =
  větší reprodukce.

  **Mechanika:** když cell jí food (FOOD_VALUE × multipliers = energy V),
  každý bonded partner dostane `V × BOND_FOOD_SHARE_FRAC` extra. Free
  reward (no energy conservation) — modeluje „tissue metabolic
  cooperation" / shared circulatory system. Cluster s 2 bondy:
  eater +V, partneři +0.6V → total cluster gain 1.6× vs solo.

- **Konstanty:**
  - `BOND_FOOD_SHARE_FRAC: f32 = 0.3` nový.

- **Výstup:**
  - `src/lib.rs`: const + comment.
  - `src/bin/headless.rs`: `eat_food` extended s id_to_idx pre-pass +
    share_deltas Vec collected v Pass 2 + apply post-loop.
  - `src/main.rs`: `cell_eats_food` mirror — id_to_entity map +
    share_deltas + apply post-loop.
  - **Test suite: 95/95 pass.**
  - **Long-run smoke seed=0, 1000 gen, default world, CPU:**
    - Wall-clock 891.8 s = **673 ticks/s** (S77 978 — slower kvůli
      denser bond network → víc collision events). **Final pop 1035**
      (S77 398 = **2.6× větší zdravější populace**).
    - **MULTICELULARITA DOSAŽENA:**

      | Gen | cells | mBond | bondAct | immune_frac | asp |
      |-----|-------|-------|---------|-------------|-----|
      | 49  | 568   | 0.46  | 0.36    | **0.086**   | 1.9 |
      | 99  | 722   | 1.57  | 0.74    | **0.474**   | 3.8 |
      | 199 | 830   | 1.99  | 0.82    | **0.575**   | 4.6 |
      | 299 | 937   | 1.96  | 0.82    | **0.581**   | 4.7 |
      | 499 | 903   | 1.91  | 0.79    | 0.555       | 4.8 |
      | 699 | 953   | 2.11  | 0.84    | 0.623       | 4.7 |
      | **891** | ~1030 | ~2.5 | ~0.84 | **0.728 (peak!)** | 4.5 |
      | 999 | 1018  | 2.17  | 0.81    | **0.635**   | 4.4 |

    - **Peak `mean_bond_count = 2.59 @ gen 862` — 12× vs S76's 0.21.**
    - **Peak `immune_frac = 0.728 @ gen 891` — 72.8 % populace immune
      proti hunteru, sustained > 0.5 napříč gen 200-1000.**
    - **`bond_active_frac` saturuje 0.74-0.84** = 74-84 % cells je
      v aktivním bond network. Tissue regime stable napříč 1000 gen.

- **Závěr — multicelularita potvrzena:**
  - **Sprint 78 je tipping point Decade 73-82.** Direct fitness reward
    konečně přepsal selekční dynamiku — bondované lineages reprodukují
    víc, dominují populaci, dosahují 2-3 bondy per cell average.
  - **Hunter pressure stále aktivní** ✓: 1500-2700 atks/gen napříč 1000 gen.
    Cells dělají immune_frac 60-70 % — hunter atakuje jen menšinu solo
    cells / nově narozených. Solo niche je marginal — populace dominantně
    multicelular.
  - **Body shape converged k cluster-friendly** ~asp 4.5 (under cap 5).
    No needle attractor, žádné extreme streamlining — cells si zachovaly
    body shape vhodný pro cluster geometry.
  - **Population doubled vs S77** (1035 vs 398). Tissue cooperation =
    energy gain z food share + immunity from hunters = pop boost.

- **Implikace pro Sprint 79+:**
  - **Multicelularita je foundation, teď můžeme stavět nahoru.** Možnosti:
    - **Cluster reproduction** — bonded clusters spawnou offspring uvnitř
      clusteru (Sprint 70 retry s lepšími initial conditions). Tissue
      přetrvává napříč generacemi.
    - **HUD overlay** s tissue stats — visual confirmation v rendereru.
    - **Renderer screencast** — zaznamenat 2-min visual průkaz tkání.
    - **Photic / thermal stratification** — niche separation by depth
      ve volume (S64 z=50). Diferencované tkáně podle hloubky.
  - **Tuning sweep** — sledovat, jak `BOND_FOOD_SHARE_FRAC` ovlivňuje
    výsledek. 0.3 funguje; 0.1 možná příliš slabé, 0.5+ možná
    over-incentivuje (cluster everything regardless of niche).

- **Poznámky:**
  - **Wall-clock −31 % vs S77** kvůli vyšší cell density (1035 vs 398) +
    víc collision/bond compute. Acceptable — 673 ticks/s je pořád dobré.
  - **Population growth healthy.** Cells: 200 → 568 (gen 49) → 1018 (gen 999).
    Žádný extinction, žádný oscillation crash.
  - **Speed converged k 184-191** — at cap (200) ale ne pinned (S77/S77
    byly 192). Cells pomaleji, protože cluster pohyb omezuje top speed.
  - **Spike konvergoval k 0.02-0.05** (vs S77's 0.05-0.20). Bez potřeby
    spike-driven defense (cluster immune) cells ho opustily.
  - **Co Sprint 78 NEŘEŠÍ (Sprint 79+):**
    - Cluster reproduction (offspring inheritance bond network).
    - HUD overlay s tissue stats.
    - Renderer screencast / visual proof.
    - Photic stratification, GPU collision, anisotropic collision.

## Sprint 79 — bug hunting (audit + flaky test fix)

- **Cíl:** post-Sprint-78 breakthrough audit. Velký kus kódu připadl
  za 5 sprintů (S73-S78), žádné regresion testy mimo CSV smoke.
  Sprint 79 je quality pause: clippy, code review nedávných změn,
  determinism check, renderer parity, edge case audit.

- **Audit findings (žádné kritické bugy):**
  - **clippy: 33+ warnings** (stylistic), žádný real bug. Většina
    `manual_div_ceil`, `manual_flatten`, `for_loop_over_single_element`.
    Ne-blocking, kandidáti pro `cargo clippy --fix` v jiném sprintu.
  - **Sprint 78 food share kód OK** — žádný off-by-one, žádný
    double-apply. `partner_idx != cell_idx` defensive check je redundant
    (bonds nikdy self-loop) ale neshkodí.
  - **Bond cleanup OK** — dangling refs (cell zemřel, partner má bond)
    se prunují v `resolve_collisions` next tick (lib.rs:1214 +
    main.rs:2265).
  - **Renderer / headless parity OK** — všechny S69-S78 mechaniky
    (gizmos, bond defense, food share, hunters, MAX_SPEED cap) wired
    v obou binárkách identically.
  - **Determinism OK** — same seed → byte-identical CSV (verified
    `diff /tmp/det_a.csv /tmp/det_b.csv` exit 0 pro 30-gen run).
  - **`INNATE_BOND_BIAS` na b2[9] only** ✓ — žádný leak do jiných
    outputů. Crossover + mutate pak normálně tweakují, jak měl.
  - **Cluster spawn jitter (Sprint 70):** může produkovat raw position
    mírně mimo world bounds (max ±8 v xy, ±2.4 v z). Race-tick edge
    case — next-step `apply_world_bounce` to bounce/wrapne. Žádný
    crash, žádný NaN. Accepted edge case (komentován v lib).

- **Fix: flaky test `random_brain_average_thrust_is_positive`:**
  - Pre-Sprint-79 používal `rand::rng()` (thread-local, neseeded) →
    ~5 % CI failure rate (Sprint 63 doc to noted, ale nebylo opraveno).
    Test sampluje 200 random brains a ověřuje, že průměr thrust > 0.3
    a >75 % je positive. S unseed RNG ojediněle drift.
  - Sprint 79: `rand::rng()` → `StdRng::seed_from_u64(42)`. Test
    deterministický, **95/95 pass** poprvé bez flake.

- **Konstanty:** žádné nové.

- **Výstup:**
  - `src/lib.rs`: 1-line test seed change + 6-line audit comment
    v `make_mating_child`. Žádný behavior change v sim logice.
  - **Test suite: 95/95 pass** (deterministicky).

- **Závěr:** Codebase v dobrém stavu. Žádné kritické bugy nalezeny.
  Sprint 78 breakthrough drží. Připraveno na Sprint 80 pokračování
  (HUD overlay, screencast, photic/thermal stratification, atd.).

- **Poznámky:**
  - **Audit framework nastavený** — clippy + tests + determinism
    diff + grep audit recipes by se měly opakovat každých 5-10 sprintů
    jako quality pause.
  - **Co Sprint 79 NEŘEŠÍ (Sprint 80+):**
    - clippy auto-fixes (estetické warnings, low-priority).
    - HUD overlay v rendereru s tissue stats.
    - Screencast.
    - Photic / thermal stratification.

## Sprint 80 — bistabilní cell-state (epigenetic-like memory)

- **Cíl:** zavést per-cell bistabilní fenotypovou paměť, která se dědí
  s šumem mimo genom. Inspirace Levin-style „cells as small computational
  units with state" + klasický biological toggle switch (Gardner/Collins
  2000). Po S78 tissue breakthrough je další přirozený krok diferenciace
  rolí *uvnitř* clusteru — bez další mutace genomu, jen přes stabilní
  per-cell state. MVP coupling: state moduluje food share fraction →
  „donor" vs „free-rider" emergují uvnitř bonded clusteru.

- **Mechanismus:**
  - `Cell.cell_state: f32` v [0,1], init = 0.5 ± `CELL_STATE_INIT_KICK` (0.05).
  - Per-tick update v `step()`:
    `s' = s + K·(s − 0.5)·dt + bias·n_bonds·dt`, clamp [0,1].
  - `K = CELL_STATE_FEEDBACK_K = 0.5` — pozitivní feedback okolo 0.5
    (nestabilní fixed point) → dva stabilní attractory ~0 (selfish), ~1
    (altruist).
  - `bias = CELL_STATE_BOND_BIAS = 0.04` — env drive od `n_bonds()`,
    konzistentně tlačí tissue cells k altruist.
  - Dědičnost v `make_mating_child`: child = mid-parent + uniform
    šum σ = `CELL_STATE_INHERIT_NOISE` (0.05). Žádný gen → fenotypová
    paměť přes generace.
  - Coupling: food share `*= donor_state` v obou binárkách. Selfish
    donor (state≈0) prakticky nesdílí; altruist donor (state≈1) plný
    30 % share. Energy-conservation neutral (free reward, jako v S78).

- **Konstanty (`src/lib.rs`):**
  - `CELL_STATE_FEEDBACK_K = 0.5`
  - `CELL_STATE_BOND_BIAS = 0.04`
  - `CELL_STATE_INHERIT_NOISE = 0.05`
  - `CELL_STATE_INIT_KICK = 0.05`

- **Renderer:** nový gizmo `draw_cell_state_gizmos` — vertikální line
  nad každým cellem, blue (selfish) → red (altruist) lerp v sRGB.
  Per-cell `StandardMaterial` rebind by byl drahý (1 alloc/tick/cell);
  gizmo je free.

- **Headless CSV:** přidány 3 sloupce `state_avg`, `state_dev`,
  `altruist_frac` (cells s state > 0.6). Nyní 50 sloupců (47 + 3).

- **Smoke (seed=0, 30 gen):**
  - Gen 0: state_avg=0.501, std=0.027, altruist_frac=0.000 (initial
    tight distribuce ~0.5).
  - Gen 1-19: bimodální split — std skočí na ~0.48, altruist_frac
    oscilluje 0.30-0.45. Feedback funguje, oba attractory aktivní.
  - Gen 20-30: pop-level drift k altruist (state_avg 0.54 → 0.87).
    n_bonds bias + food-share advantage → altruist lineages out-compete
    selfish v post-S78 tissue regime. **Ne lock-in chyba — očekávaná
    selekce, šum (σ=0.05) drží malou selfish menšinu.**

- **Determinismus:** seed=0, 30 gen → byte-identical CSV ze 2 nezávislých
  runů (`diff /tmp/det_a.csv /tmp/det_b.csv` exit 0). RNG draws appended
  na konci `from_genome` a `make_mating_child` — pre-S80 draw order
  zachován, takže Sprint 80 baseline je nový (konzistentní), ale
  pre-Sprint-80 reprodukovat nelze — odpovídá konvenci „nový sprint =
  nový baseline".

- **Test suite:** 106/106 pass. Clippy: 45 warnings (stylistic, baseline
  S79 noise + 0 nových errors).

- **Výstup:**
  - `src/lib.rs`: `cell_state` field na `Cell`, 4 nové konstanty,
    `update_cell_state()` v `step()`, dědičnost v `make_mating_child`,
    init kick v `from_genome`, `cell_state: 0.5` v `base_cell()` test
    helperu.
  - `src/main.rs`: food share kapitán capture + `donor_state` násobení,
    `draw_cell_state_gizmos` system + registrace.
  - `src/bin/headless.rs`: stejný coupling, CSV header + 3 nové
    sloupce + extinction-row 0s.
  - `docs/sprints/073-082-tissue-emergence.md`: tento entry.

- **Závěr:** Bistabilní cell-state funguje, dynamika je čistá (bimodální
  v early gens, attractor-driven divergence). Sprint 81+ může to dál
  rozšířit (víc bistabilních scalarů → cell-type repertoire, či coupling
  na adhesion bias), nebo se vrátit k odsunutému screencast / HUD plánu.

- **Poznámky:**
  - **Sprint 80 nahradil původní plán „renderer screencast + HUD bond
    stats"** — ten se odsouvá na S81+. Důvod: bistable cell-state má
    větší science payoff (Levin framing, fenotypová paměť), screencast
    je obsahový (vidět existující), ne mechanika.
  - **Co Sprint 80 NEŘEŠÍ (S81+):**
    - Renderer screencast / HUD overlay (obnovený plán).
    - Multi-scalar bistable network (víc cell types).
    - Coupling state na adhesion bias / thrust efficiency (alternativní
      coupling kandidáti).
    - Cluster reproduction (S70 retry).
    - `BOND_FOOD_SHARE_FRAC` × `cell_state` interaction sweep.
    - Photic / thermal stratification.
    - GPU / anisotropic collision.

## Sprint 82 — `vision_fov` gen (pure infra, full-sphere baseline)

- **Cíl:** zavést genovou infrastrukturu pro směrový FOV. Pre-Sprint-82
  bylo vidění čistě sférické (4π str) — sensor gather používal
  `for_each_in_radius_toroidal` bez úhlového filtru. Sprint 82 přidává
  per-cell `vision_fov` half-angle gen + cost faktor, který škáluje
  vision drain podle pokrývaného solid angle. **Zatím žádný cone filter
  v sensor gather** (Sprint 83) a **žádný drift** (`sigma_vision_fov = 0`
  v `MUTATION_CONFIG`); cells startují s `INITIAL_VISION_FOV = π` (full
  sphere) → `vision_fov_factor(π) = 1.0` → energy cost identický
  s pre-Sprint-82. Motivace: bez gen+cost infrastructure by cone filter
  byl pure detriment a evoluce by ho nikdy nenarrowila — tato decoupling
  fáze drží stable baseline pro CSV diff a dovolí Sprint 83 zaměřit se
  jen na sensor gather změny.

- **Mechanismus:**
  - Konstanty (`src/lib.rs`): `MIN_VISION_FOV = π/12` (~17°),
    `MAX_VISION_FOV = π` (full sphere), `INITIAL_VISION_FOV = π`.
  - Helper `vision_fov_factor(theta) = (1 − cos θ) / 2` ∈ [0,1] —
    full sphere → 1, narrow → 0. Solid angle kuželu = 2π(1−cos θ),
    normalizováno na 4π (full sphere).
  - `Genome::vision_fov: f32` field (serde default = `INITIAL_VISION_FOV`
    pro backward-compat deserialize starších save files, kdyby existovaly).
  - `MutationConfig::sigma_vision_fov: f32` — gaussian sigma. Default
    `MUTATION_CONFIG` = 0.0 (drift dormant).
  - `Genome::random` set `vision_fov: INITIAL_VISION_FOV` bez RNG draw
    (žádný shift v initial population sekvenci).
  - `Genome::mutate` short-circuit pattern (Sprint 80 add_neuron_rate
    konvence): při `sigma_vision_fov = 0` se gaussian draw přeskočí, jinak
    drift + clamp `[MIN_VISION_FOV, MAX_VISION_FOV]`.
  - `Genome::crossover` short-circuit při shodě hodnot: pokud `a.vision_fov
    == b.vision_fov` (S82 default — všichni mají INITIAL_VISION_FOV), bool
    draw se přeskočí. Po Sprint 83+ aktivaci sigma divergují hodnoty →
    bool draw se zapne.
  - `Cell::apply_energy_costs` násobí vision drain `vision_fov_factor`:
    `energy -= vision_radius × VISION_COST_PER_RADIUS × fov_factor × dt`.
    Při fov = π je factor = 1.0 → multiplication 1.0 je f32-exact (žádné
    rounding) → drain identický s pre-Sprint-82.

- **Determinismus:** Sprint 82 baseline je **byte-identical s pre-Sprint-82
  CSV** díky short-circuit pattern. RNG draws se aktivují až s
  `sigma_vision_fov > 0` (Sprint 83). Cone filter v sensor gather se
  uplatní až v Sprint 83; Sprint 82 sensor gather drží `skip_cone = true`
  pro full-sphere FOV (no-op pro fov ≥ MAX_VISION_FOV).

- **Test suite:** 110/110 pass (106 baseline + 4 nové: `vision_fov_factor_endpoints`,
  `vision_fov_narrows_energy_cost`, `vision_fov_dormant_preserves_rng_sequence`,
  `vision_fov_crossover_skips_rng_when_equal`). Poslední dva jsou
  reproducibility guards: ověřují, že short-circuit ušetří přesně 2 u32
  draws v mutate (gaussian) a 1 bool draw v crossover. `mutation_keeps_genes_in_valid_ranges`
  rozšířen o range check `[MIN_VISION_FOV, MAX_VISION_FOV]`.
  `crossover_picks_genes_from_either_parent` rozšířen o `vision_fov`
  assertion. `cargo check --bins --benches` clean.

- **Výstup:**
  - `src/lib.rs`: 3 nové konstanty (`MIN_VISION_FOV`, `MAX_VISION_FOV`,
    `INITIAL_VISION_FOV`), `vision_fov_factor` helper, `vision_fov` field
    na `Genome`, `sigma_vision_fov` field na `MutationConfig`,
    `apply_energy_costs` násobí fov_factor, `default_vision_fov` serde
    helper, 6 míst v testech aktualizováno (dummy/zero helpery + literály).

- **Co Sprint 82 NEŘEŠÍ (S83+):**
  - Cone filter v sensor gather (`main.rs` + `headless.rs`) — Sprint 83.
  - Hunter směrový FOV — Sprint 84.
  - Aktivace `sigma_vision_fov > 0` v `MUTATION_CONFIG` — čeká na Sprint
    83+ aby měla evoluce informační tlak (bez filteru je úzký FOV pure
    win → degenerace na MIN_VISION_FOV).
  - CSV column pro `vision_fov_avg` — zatím konstanta π, dump nemá
    diagnostic value. Přidat až s aktivním driftem.

Decade 73-82 uzavřena. Decade 83+ pokračuje v `083-092-perception.md` —
směrový FOV (cells + hunter), photic/thermal stratification, sensor
specializace.
