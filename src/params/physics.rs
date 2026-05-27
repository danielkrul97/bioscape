// Shared sim parameters consumed by both the Bevy renderer (`src/main.rs`)
// and the headless harness (`src/bin/headless.rs`). Single source of truth —
// tune here. Renderer-only and headless-only knobs stay in their binaries.

pub const FIXED_TIMESTEP_HZ: f32 = 60.0;
pub const TICKS_PER_GENERATION: u64 = 600;
pub const GENERATIONS_PER_EPOCH: u64 = 100;

pub const INITIAL_CELLS: usize = 200;
/// Sprint 64: 1000 → 2500 (proportional s z=20 → z=50 expansion). Cells
/// density per volume zachovaná: pre-Sprint-64 1.2e-5 cells/unit³, post:
/// stejně. CPU paralelní cesta drží > 60 FPS (Sprint 63 5k = 870 ticks/s).
///
/// Sprint 100: 2500 → 5000 (proportional s z=50 → z=100 expansion). Density
/// 1.2e-5 cells/unit³ zachovaná. 5k testováno v S63 benchmarku jako
/// 870 ticks/s na CPU paralelní cestě.
pub const MAX_POPULATION: usize = 1500;

pub const CELL_RADIUS: f32 = 5.0;
pub const EAT_RADIUS: f32 = 8.0;
pub const MATING_RADIUS: f32 = 200.0;

pub const DRAG_COEFFICIENT: f32 = 0.005;
/// Sprint 188: hard cap on per-cell velocity magnitude expressed as a
/// multiple of `genome.max_speed`. Bond spring forces are applied as
/// per-tick impulses without a `dt` scale (see `shaders/collision.wgsl`
/// + `world.rs::resolve_collisions` Phase 2), so an overstretched bond
/// briefly drives `|v|` well past what drag can balance — observed at
/// 3.6× `max_speed` in a real run. 1.5× preserves headroom for short
/// transients (bond release, predation impulse) while keeping run-away
/// out of the system. Applied AFTER all velocity-mutating phases so
/// brain, motor, brownian, collision, bond, and step contributions are
/// all bounded by the same envelope.
pub const VELOCITY_CAP_FACTOR: f32 = 1.5;
pub const ANGULAR_DRAG: f32 = 1.0;
pub const ENERGY_COST_PER_V_SQ: f32 = 0.0008;
pub const ANGULAR_ENERGY_COST: f32 = 0.05;
pub const VISION_COST_PER_RADIUS: f32 = 0.02;
pub const BODY_COST_FACTOR: f32 = 0.8;

pub const FOOD_VALUE: f32 = 20.0;
pub const FOOD_SPAWN_RATE: usize = 5;
/// Plant food density. Pre-Hunter: 2600 (rich). Post-Hunter rebalance: 6500
/// (40 % of pre baseline) — scarcer plants činí intra-cell predaci výnosnou
/// strategií, jinak by herbivore + outrun zůstal dominantní attractor.
pub const WORLD_UNITS_PER_FOOD: f32 = 6500.0;

/// Sprint 193: food scarcity ramp. Multiplier applied to `food_target`
/// alongside the seasonal cycle and FoodCrash shock. Starts at 1.0 in
/// generation 0 and linearly drops to `SCARCITY_FLOOR` by
/// `SCARCITY_RAMP_END_GEN`, then stays flat at the floor. Folded into
/// `World::density_factor` at end-of-generation so checkpoint + CSV
/// already track it.
///
/// Rationale: pre-S193 the 183-192 brain-stability stack guarantees the
/// brain *can* do graded computation, but the population still survives
/// on bonded-cluster food-share without using it. Halving plant density
/// over the first 100 generations creates a monotonic energetic gradient
/// that should reward smarter foraging policies without starving the
/// population to extinction (floor = 50 % baseline = still positive
/// energy budget for an efficient cell).
pub const SCARCITY_RAMP_END_GEN: u64 = 100;
pub const SCARCITY_FLOOR: f32 = 0.5;

/// Sprint 38: gravitační zrychlení (sim units / sec²) působící na cells.
/// Sprint 65: 5.0 → 0.0 (neutral buoyancy approximation). Pre-Sprint-65
/// vytvářelo selekční tlak směrem k „seď na dně" — cells postupně
/// sedimentovaly, akumulovaly se na floor reflective wall, vertikální
/// motion neměla evoluční benefit (úsilí plavat up = stejně sedneš dolů).
/// Po Sprintu 65 cell density == water density → vertikální motion je
/// 100 % brain-driven. Food sink (`FOOD_SINK_RATE`) zachován — food má
/// vyšší density než cells (benthic deposit semantika), cells musí
/// proaktivně dive za food.
pub const GRAVITY: f32 = 0.0;
/// Sprint 65: collision velocity damping. Restitution 0 = perfectly
/// inelastic — closing velocity podél separation normal je vynulovaná
/// (cells "stick" momentárně, pak se separují přes position depenetration).
/// 1.0 = elastic (perfect bounce). Soft biological cells = 0.0 default.
/// Pre-Sprint-65 cells měly delta_pos depenetration ale velocity neaffected
/// → po push-apart pokračovaly v closing motion → re-overlapped next tick
/// (oscilace + zbytečný compute).
pub const COLLISION_RESTITUTION: f32 = 0.0;
/// Sprint 38: terminal sink rate pro food (food nemá velocity field, pohybuje
/// se konstantní rychlostí dolů). Pomalejší než cells (které mohou plavat),
/// takže food drift k dnu = postupný „benthic deposit". 8 units/sec ~ 4 sec
/// průchod celé z-vrstvy (z=2).
pub const FOOD_SINK_RATE: f32 = 8.0;

/// Sprint 202 — inertial mass. `mass = body_l × body_w × body_h × DENSITY`.
/// Replaces the pre-S202 mass proxy `effective_radius = (l+w+h)/3` used by
/// `motor.wgsl` and treated as 1.0 elsewhere. DENSITY = 0.1 keeps the typical
/// cell mass (body ≈ 3³ = 27 → mass ≈ 2.7) within an order of magnitude of
/// the pre-S202 motor inertia coefficient so brain outputs that worked at the
/// old scale still produce comparable accelerations. Selection now sees a
/// cubic size penalty on responsiveness rather than linear, which is the
/// real-physics behavior the pre-S202 linear scaling was an approximation of.
pub const MASS_DENSITY: f32 = 0.1;

/// Sprint 202 — bulk-flow advection of diffusion fields (smell, pheromone,
/// vibration, thermal perturbation). Curl-noise vector field generated at
/// world init, advects field values upwind each diffuse step. Peak speed in
/// world units / sec — kept much smaller than typical cell `max_speed` (~100)
/// so flow biases scent trails without dragging cells.
pub const FLOW_MAGNITUDE: f32 = 8.0;
/// Spatial wavelength factor of the curl-noise flow pattern. Larger = bigger
/// gyres, smaller = turbulent finer eddies. `FLOW_SCALE = 0.003` → wavelength
/// ≈ 1/0.003 ≈ 330 world units, comparable to half the smaller world extent
/// so cells see large-scale gyres rather than per-tick noise.
pub const FLOW_SCALE: f32 = 0.003;

/// Sprint 202 — bond bending stiffness. Angle-spring between two bonds
/// sharing the same cell. Per (slot_a, slot_b) pair, rest angle is captured
/// at the moment the second bond forms; bend force pulls the bond pair back
/// toward that rest cosine. Conservative — stronger than this caused
/// oscillation in cluster geometry during testing.
pub const BOND_BENDING_STIFFNESS: f32 = 0.4;
/// Angular damping along the same pair. Critically damped at √(4·k) for a
/// unit-mass system; below that = underdamped (visible cluster wobble).
pub const BOND_BENDING_DAMPING: f32 = 0.6;

/// Sprint 202 — thermal advection perturbation. Magnitude (sim-units, same as
/// `THERMAL_*`) of warm-patch sources injected into a 3D perturbation field
/// that diffuses + advects with the same `FLOW` vector field as the smell /
/// pheromone channels. `step.wgsl` samples the perturbation at cell position
/// and adds it to the analytic `temperature_at_z` base so warm currents can
/// reach down into cold strata and vice versa.
pub const THERMAL_PERTURBATION_AMP: f32 = 3.0;
/// Number of warm patch sources injected per generation cycle. Sparse — the
/// field's advection is what carries them across the world, density doesn't
/// need to be high.
pub const THERMAL_PERTURBATION_SOURCES_PER_GEN: usize = 16;
/// Decay rate of the thermal perturbation field (1/s). Slower than smell
/// (0.3) — temperature carries further than scent.
pub const THERMAL_PERTURBATION_DECAY: f32 = 0.1;
/// Diffusion coefficient for the thermal perturbation field. Must stay
/// < 1/6 for stability of the 7-point Jacobi stencil.
pub const THERMAL_PERTURBATION_DIFFUSION: f32 = 0.12;
