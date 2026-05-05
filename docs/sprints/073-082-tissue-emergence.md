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

## Sprinty 75–82 — open-ended

- **Sprint 75 (priorita):** `INNATE_BOND_BIAS` 0 → 1.5 — odbourat
  formation gating. Hypotéza: random brainy budou bondovat by default,
  selekce může pak rozhodnout (vs current „selekce musí bondování
  najít z 0"). Cíl: immune_frac > 0.05 do gen 500.
- **Sprint 75+:** Hunter persistent target + chase duration —
  predator-style „commit to one prey".
- **Sprint 75+:** HUD overlay s real-time bond stats v rendereru.
- **Sprint 75+:** Photic stratification (z-gradient light field +
  photoreceptor sensor input).
- **Sprint 75+:** Thermal stratification (z-gradient temperature).
- **Sprint 75+:** GPU collision shader (3D + adhesion + bonds mirror).
- **Sprint 75+:** Anisotropic cell collision (ellipsoid geometry).
- **Sprint 75+:** Cluster-aware reproduction tuning.
- **Sprint 75+:** Spatial autocorrelation adhesion_type clustering metric.
