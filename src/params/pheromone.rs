// Pheromone signaling layer. 2D scalar field jako SmellField, ale zdroje jsou
// cells. Sprint 25: BASELINE = 0 (žádné free-rider, žádný predator exploit z
// Sprint 24). Cells musí aktivně emitovat brain output[2] aby vznikl signál,
// **a aby byly způsobilé k reprodukci** — `MATING_PHEROMONE_THRESHOLD` gating.
// Brain detekuje gradient přes `inputs[11..13]`. Cost ∝ emise.
pub const PHEROMONE_GRID_RES: usize = 64;
/// Sprint 53: z-axis resolution pro pheromone field. Tenčí z-volume + lower
/// res = větší cell_size_z (32 vs 64) → matchne thin world aspect a šetří
/// memory bez ztráty rozlišení v xy.
pub const PHEROMONE_GRID_RES_Z: usize = 16;
/// Sprint 126: počet nezávislých pheromone kanálů. Multi-channel umožňuje
/// emergence diskriminované komunikace (cells emitují mixturu, sensors
/// rozliší). 3 = uprostřed rozsahu 3-8 — dost pro discrimination, méně
/// invazivní vs Brain dim expansion.
pub const N_PHEROMONE_CHANNELS: usize = 3;
/// Sprint 126: per-channel decay (1/s). ch0 = existing slow (mating-friendly),
/// ch1 medium, ch2 fast (bursty / temporal patterning).
pub const PHEROMONE_DECAY_PER_CH: [f32; N_PHEROMONE_CHANNELS] = [0.3, 1.5, 5.0];
/// Sprint 126: per-channel diffusion. Slow channels difunduji víc (cumulative
/// spread), rychlé méně (lokalizovaná spike).
pub const PHEROMONE_DIFFUSION_PER_CH: [f32; N_PHEROMONE_CHANNELS] = [0.15, 0.12, 0.08];
/// Backward-compat: ch0 (slow) decay/diffusion. GPU shaders + headless GPU
/// path stále používají single-channel scalar.
pub const PHEROMONE_DIFFUSION: f32 = PHEROMONE_DIFFUSION_PER_CH[0];
pub const PHEROMONE_DECAY: f32 = PHEROMONE_DECAY_PER_CH[0];
pub const PHEROMONE_BASELINE_EMIT: f32 = 0.0;
pub const PHEROMONE_BRAIN_MOD: f32 = 1.0;
pub const PHEROMONE_COST_PER_RATE: f32 = 1.0;
pub const PHEROMONE_SAMPLE_EPSILON: f32 = 10.0;
pub const PHEROMONE_NORMALIZATION_GAIN: f32 = 0.5;
/// Cell musí mít `last_outputs[2] > THRESHOLD` aby byla eligible pro mating.
/// Mating je tak podmíněn aktivní emisí — selektuje proti tichým cells, které
/// by jinak free-ride na public goods of pheromone field.
pub const MATING_PHEROMONE_THRESHOLD: f32 = 0.2;
pub const MAX_SPAWN_ATTEMPTS: usize = 5;
pub const CARRION_FOOD_COUNT: usize = 2;
