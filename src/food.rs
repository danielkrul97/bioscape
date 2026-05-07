use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::*;

/// Sprint 92: food kind tagged enum. Differentiates plant (ambient spawn,
/// always available, baseline value), cell carrion (drops on cell death),
/// hunter carrion (drops on hunter death, richest reward). Eat efficiency
/// per kind je modulated by cell `genome.carnivore_score` ∈ [0, 1] —
/// herbivore digestion vs carnivore digestion trade-off.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum FoodKind {
    Plant = 0,
    Carrion = 1,
    HunterCarrion = 2,
}

impl Default for FoodKind {
    fn default() -> Self {
        FoodKind::Plant
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Food {
    pub position: [f32; 3],
    /// Sprint 42: ticks od spawnu. Drives decay of `value_factor`. Init 0
    /// pro fresh food i carrion (univerzální decay, žádný carrion-specific
    /// staleness offset).
    pub age_ticks: u32,
    /// Sprint 92: food kind. `serde(default)` returns Plant pro backward-
    /// compat s pre-S92 checkpointy.
    #[serde(default)]
    pub kind: FoodKind,
}

/// Sprint 92: base food value per kind. Carrion má vyšší value než plant
/// (concentrated biomass), hunter carrion ještě víc (apex predator drop).
pub const PLANT_FOOD_VALUE: f32 = 20.0;
pub const CARRION_FOOD_VALUE: f32 = 30.0;
pub const HUNTER_CARRION_FOOD_VALUE: f32 = 50.0;

/// Sprint 128: cooperative food node — high-value spawn, který nepřináší
/// nic dokud N cells během time window nedorazí. Vytváří fitness coupling
/// pro recruitment signaling: solo cells nedostanou nic, coordinated
/// trio dostane high reward → selekce na "I see food, signal peers".
pub const COOP_FOOD_REQUIRED_ARRIVALS: usize = 3;
/// Time okno (ticks) od spawnu. Po vypršení: despawn bez reward.
pub const COOP_FOOD_TIME_WINDOW_TICKS: u32 = 120;
/// Per-participant reward při úspěšné koordinaci. Asymetricky vysoký vůči
/// regular Plant food (20) — incentive justifying loiter cost.
pub const COOP_FOOD_REWARD_PER_CELL: f32 = 80.0;
/// Radius (sim units), v rámci kterého cell counts as "arrived". Větší než
/// regular eat radius (~20) — coop food má vizuální/aroma signal "here is
/// gathering point", cells nemusí stát přímo na něm.
pub const COOP_FOOD_ARRIVAL_RADIUS: f32 = 30.0;
/// Spawn pravděpodobnost per tick (Poisson-like). Kalibrováno tak, aby vznikalo
/// cca 10-15 coop nodes per generation (600 ticků). 0.02 → ~12 events/gen.
pub const COOP_FOOD_SPAWN_RATE_PER_TICK: f32 = 0.02;
/// Max simultaneous coop nodes ve světě. Cap pro ohraničení complexity
/// (a paměť).
pub const COOP_FOOD_MAX_CONCURRENT: usize = 8;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoopFood {
    pub position: [f32; 3],
    pub spawn_tick: u64,
    /// Set unique cell_ids, které byly v `ARRIVAL_RADIUS` aspoň jeden tick.
    /// Vec<u64> aby zachoval Serde + insertion order; lookup je O(N), ale
    /// N je malé (typicky < 10).
    pub arrivals: Vec<u64>,
    /// True pokud byl threshold dosažen → reward distribuován + bude
    /// despawnut na konci aktuálního ticku.
    pub triggered: bool,
}

impl CoopFood {
    pub fn new(position: [f32; 3], spawn_tick: u64) -> Self {
        Self {
            position,
            spawn_tick,
            arrivals: Vec::new(),
            triggered: false,
        }
    }

    /// True pokud věk > TIME_WINDOW. Caller volá po pokusu o trigger,
    /// aby triggered nodes (které trigger zvládly přesně v expiry frame)
    /// nebyly mylně klasifikovány jako "expired no reward".
    #[inline]
    pub fn is_expired(&self, current_tick: u64) -> bool {
        current_tick.saturating_sub(self.spawn_tick) >= COOP_FOOD_TIME_WINDOW_TICKS as u64
    }
}

/// Sprint 128: zaregistruj cell_id jako arrival. Insertion order zachovaná,
/// duplikáty ignorovány (cell může být v radius víc ticků). Vrací true pokud
/// byl id přidán, false pokud už evidoval.
pub fn register_coop_arrival(coop: &mut CoopFood, cell_id: u64) -> bool {
    if coop.arrivals.iter().any(|id| *id == cell_id) {
        return false;
    }
    coop.arrivals.push(cell_id);
    true
}

/// Sprint 128: pokus o trigger threshold. Vrací true pokud byl reward distribuován
/// (= aspoň REQUIRED arrivals + ne-yet-triggered). Caller propaguje return value
/// do per-gen counterů (coop_food_solved).
pub fn try_trigger_coop(coop: &mut CoopFood, cells: &mut [Cell]) -> bool {
    if coop.triggered || coop.arrivals.len() < COOP_FOOD_REQUIRED_ARRIVALS {
        return false;
    }
    for cell_id in &coop.arrivals {
        if let Some(cell) = cells.iter_mut().find(|c| c.cell_id == *cell_id) {
            cell.energy += COOP_FOOD_REWARD_PER_CELL;
        }
    }
    coop.triggered = true;
    true
}

/// Sprint 128: vyber random pozici uvnitř world bounds (toroidal world,
/// stejná logika jako `Food::random`). Pokud `world_half[2] == 0`, z-osa
/// vrací 0 — backward-compat s pre-S33 baseline.
pub fn random_coop_position(rng: &mut impl Rng, world_half: [f32; 3]) -> [f32; 3] {
    let z = if world_half[2] > 0.0 {
        rng.random_range(-world_half[2]..world_half[2])
    } else {
        0.0
    };
    [
        rng.random_range(-world_half[0]..world_half[0]),
        rng.random_range(-world_half[1]..world_half[1]),
        z,
    ]
}

/// Sprint 128: per-tick scan + arrival registration pro každý coop node.
/// Cell je v radius pokud (toroidal-aware) Euclidean distance ≤ ARRIVAL_RADIUS.
pub fn register_coop_arrivals_for_all(coops: &mut [CoopFood], cells: &[Cell], world_half: [f32; 3]) {
    let r2 = COOP_FOOD_ARRIVAL_RADIUS * COOP_FOOD_ARRIVAL_RADIUS;
    for coop in coops.iter_mut() {
        if coop.triggered {
            continue;
        }
        for cell in cells.iter() {
            let d = min_image_delta(coop.position, cell.position, world_half);
            let d2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            if d2 <= r2 {
                let _ = register_coop_arrival(coop, cell.cell_id);
            }
        }
    }
}

#[inline]
pub fn food_base_value(kind: FoodKind) -> f32 {
    match kind {
        FoodKind::Plant => PLANT_FOOD_VALUE,
        FoodKind::Carrion => CARRION_FOOD_VALUE,
        FoodKind::HunterCarrion => HUNTER_CARRION_FOOD_VALUE,
    }
}

/// Sprint 92: digestion efficiency per food kind × cell `carnivore_score`.
/// Continuous trade-off: 0 = pure herbivore (plant only), 1 = pure carnivore
/// (hunter carrion only), 0.5 = mixed (everything moderate).
///
/// - Plant + score 0.0 → 1.0 (full)
/// - Plant + score 1.0 → 0.0 (can't digest plants at all)
/// - HunterCarrion + score 0.0 → 0.0 (can't digest)
/// - HunterCarrion + score 1.0 → 1.0 (full)
/// - Carrion (cell) → 0.5 universally — semi-digestible by both diets
///   (compromise food, doesn't drive specialization)
#[inline]
pub fn eat_efficiency(kind: FoodKind, carnivore_score: f32) -> f32 {
    let s = carnivore_score.clamp(0.0, 1.0);
    match kind {
        FoodKind::Plant => 1.0 - s,
        FoodKind::Carrion => 0.5,
        FoodKind::HunterCarrion => s,
    }
}

impl Food {
    pub fn random(rng: &mut impl Rng, world_half: [f32; 3]) -> Self {
        // Sprint 32: z-osa conditional pro deterministický CSV; world_half[2]=0
        // → z=0 bez RNG draw.
        let z = if world_half[2] > 0.0 {
            rng.random_range(-world_half[2]..world_half[2])
        } else {
            0.0
        };
        Self {
            position: [
                rng.random_range(-world_half[0]..world_half[0]),
                rng.random_range(-world_half[1]..world_half[1]),
                z,
            ],
            age_ticks: 0,
            kind: FoodKind::Plant,
        }
    }

    /// Sprint 38: aplikuje gravitační drift food (sink). Pouze pokud je
    /// z-volume aktivní; jinak no-op (Sprint 32 z=0 setup).
    pub fn apply_gravity(&mut self, dt: f32, world_half_z: f32) {
        if world_half_z <= 0.0 {
            return;
        }
        self.position[2] = (self.position[2] - FOOD_SINK_RATE * dt).max(-world_half_z);
    }

    /// Sprint 42: lineární decay value factor podle stáří. Pro age=0 vrací 1.0,
    /// klesá lineárně k nule, pak clampnuto na 0.
    pub fn value_factor(&self) -> f32 {
        let age_sec = self.age_ticks as f32 / FIXED_TIMESTEP_HZ;
        (1.0 - CARRION_DECAY_PER_SEC * age_sec).max(0.0)
    }

    /// Sprint 42: age tick increment. Vrací `false` pokud food expiroval
    /// (value_factor ≤ 0) — caller despawne. Volá se po `apply_gravity`
    /// v hot loopu binárek.
    pub fn age_step(&mut self) -> bool {
        self.age_ticks = self.age_ticks.saturating_add(1);
        self.value_factor() > 0.0
    }
}
