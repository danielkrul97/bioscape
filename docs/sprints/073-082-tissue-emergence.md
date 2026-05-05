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

## Sprinty 74–82 — open-ended

- **Sprint 74 (priorita):** rebalance hunt vs bond cost. Single-line
  konstanty change + 1000-gen verify, jestli bond density rise
  k tissue regimu (immune_frac > 0.10).
- **Sprint 74+:** Hunter persistent target + chase duration —
  predator-style „commit to one prey".
- **Sprint 74+:** HUD overlay s real-time bond stats v rendereru.
- **Sprint 74+:** Photic stratification (z-gradient light field +
  photoreceptor sensor input).
- **Sprint 74+:** Thermal stratification (z-gradient temperature).
- **Sprint 74+:** GPU collision shader (3D + adhesion + bonds mirror).
- **Sprint 74+:** Anisotropic cell collision (ellipsoid geometry).
- **Sprint 74+:** Cluster-aware reproduction tuning (Sprint 70 retry
  s lepším cluster path).
- **Sprint 74+:** Spatial autocorrelation adhesion_type clustering metric
  (CSV) — empirický důkaz Steinberg sorting.
