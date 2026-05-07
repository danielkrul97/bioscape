use super::physics::TICKS_PER_GENERATION;

// Thermal stratification (Sprint 85). Lineární z-gradient: warm at top, cold
// at bottom (oceán-like). Coupling přes Q10 multiplikátor na all per-tick
// drains v `apply_energy_costs`. Niche separation emergne behaviorálně —
// cells co plavou dolů platí míň energie ale nemají žádnou další výhodu;
// selekce hledá optimum food-density × thermal-cost. Žádný temperature
// field grid — funkce z stačí.
/// Maximum teploty (z = +world_half[2], hladina). Sim units, ne reálné °C.
pub const THERMAL_TOP: f32 = 30.0;
/// Minimum teploty (z = −world_half[2], dno). Sim units.
pub const THERMAL_BOTTOM: f32 = 4.0;
/// Q10 koeficient — biologické rychlosti přibližně 2× per +10 sim-units
/// teploty. Standard biology Q10 ~ 2-3 (enzyme kinetics, metabolic rate).
pub const THERMAL_Q10: f32 = 2.0;
/// Referenční teplota — při T = THERMAL_REF_TEMP je `metabolism_factor = 1.0`,
/// drain identický s pre-Sprint-85. Volena uprostřed [BOTTOM, TOP] aby
/// průměrná cell drain ~ pre-Sprint-85 (ratio top:ref:bottom ≈ 2.46:1:0.41).
pub const THERMAL_REF_TEMP: f32 = 17.0;
/// Sprint 86: peak diurnal amplitude na hladině. Surface temperature osciluje
/// `THERMAL_TOP ± THERMAL_DIURNAL_AMP` v rámci 1 day = `TICKS_PER_GENERATION`
/// ticks (10 s real-time). Hloubka oscilace klesá lineárně k 0 na `THERMAL_BOTTOM`
/// (= mirror reálné termokliny: deep water buffered proti solárnímu cyklu).
pub const THERMAL_DIURNAL_AMP: f32 = 5.0;
/// Sprint 86: full diurnal cycle = 1 generation = 600 ticks = 10 s real-time.
/// Cells with ~1 gen lifespan experience exactly one day. Krátší period =
/// flicker, delší = cells málokdy zažijí přechod.
pub const THERMAL_DIURNAL_PERIOD_TICKS: u64 = TICKS_PER_GENERATION;
/// Sprint 86: peak seasonal amplitude (uniform shift všech depth). Period =
/// `CYCLE_GEN_PERIOD` (50 gen) — sdílený s food density cyklem, takže warm
/// season = abundant food (summer), cold season = scarce food (winter).
/// Coupling vytváří přírodní seasonal niche shift.
pub const THERMAL_SEASONAL_AMP: f32 = 4.0;
/// Sprint 87: range pro per-cell `thermal_optimum` gen. Init populace draws
/// uniform across [MIN, MAX] = [BOTTOM, TOP] → speciace mezi cold-prefer
/// a warm-prefer fenotypy. Range matches static gradient endpoints — cells
/// nemůžou preferovat teplotu mimo dostupný z-rozsah.
pub const MIN_THERMAL_OPTIMUM: f32 = THERMAL_BOTTOM;
pub const MAX_THERMAL_OPTIMUM: f32 = THERMAL_TOP;
/// Sprint 87: peak penalty rate při |dev|/13 = 1.0 (= max single-direction
/// deviation v [BOTTOM, TOP] od optima blízko opačného konce). Quadratic
/// penalty: `((temp - opt) / 13.0)² × PENALTY × dt`. Při PENALTY = 1.0:
/// extreme-deviation cell platí ~1.0 energy/sec navíc — comparable s body
/// maintenance cost. Independentní od metabolism factor (thermal stress je
/// extra cost, ne reduction enzymové aktivity).
pub const THERMAL_OPTIMUM_PENALTY: f32 = 1.0;
/// Sprint 31 spatial clustering: rejection sampling síla. Per uniformně
/// vzorkovaný food candidate je pravděpodobnost zamítnutí
/// `STRENGTH × (1 - richness)`. Při richness=1 (rich zone) nikdy nezamítá,
/// při richness=0 (poor zone) zamítá s pravděpodobností STRENGTH. Sprint 21
/// v1–v5 zkoušel plnou sílu (`1 × (1 - richness)`) → extinkce gen 70–110.
/// Sprint 100: 0.3 → 0.6 — silnější spatial gradient. Poor zone má 40 % šanci
/// spawnu per candidate (vs. 70 % před), což zostřuje hranici mezi rich a poor
/// zonami a posiluje selekční signál pro spatial preference. Stále hluboko pod
/// extinction threshold ze Sprintu 21 (1.0 = full reject v poor zone). Plus
/// value modulace `WORLD_MAP_FOOD_FLOOR/AMP` jako safety net (i food v poor
/// zone má hodnotu ~85 % baselinu). MAX_SPAWN_ATTEMPTS retry budget na uniform
/// resample zajišťuje, že clustering nikdy úplně nezablokuje spawn.
pub const FOOD_REJECTION_STRENGTH: f32 = 0.6;
// Environmentální hazard layer: passive energy drain v "nebezpečných" zónách.
// Zónová mapa jde z `WorldMap` noise — POSITIVNÍ korelace s food richness:
// bohaté oblasti = nebezpečné (high reward, high risk), chudé = bezpečné.
// Vytváří trade-off niche: efficient cell může těžit rich-dangerous, slabší
// se uchýlí do safe-poor a žije s méně food. Drain za sec při noise=1:
// HAZARD_FLOOR + HAZARD_AMP = celkový max. Ladí se pouze v binárkách.
pub const HAZARD_DRAIN_PER_SEC: f32 = 0.5;
pub const HAZARD_FLOOR: f32 = 0.0;
pub const HAZARD_AMP: f32 = 1.0;

/// Sprint 30 self-preservation: gain pro tanh(damage_accum × GAIN) brain input
/// [14]. Damage_accum = nedobrovolný drain z minulého ticku (predation +
/// hazard). Single predation hit (drain = `PREDATION_DRAIN_PER_TICK` = 3.0)
/// → tanh(1.5) ≈ 0.90 (silný signál). Stabilní hazard ~0.008/tick → tanh ≈
/// 0.004 (neviditelný — chronický hazard nemá být „damage event", spatial
/// avoidance se vyvíjí přes selekční gradient na pozici, ne přes tento input).
/// Multi-attacker pile-on saturuje k 1.0.
pub const DAMAGE_NORMALIZATION_GAIN: f32 = 0.5;
