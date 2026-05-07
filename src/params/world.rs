/// Half-extents simulačního světa (toroidal). Sdíleno mezi rendererem
/// (`src/main.rs`) a headlessem (`src/bin/headless.rs`) — single source
/// of truth, jinak by sim tiše divergoval při změně jen na jedné straně.
///
/// Sprint 53: WORLD_HALF[2] expanded z=2 → z=20. SmellField + WorldMap +
/// Pheromone jsou plně 3D (volumetric grid + 7-point Jacobi diffusion + 3D
/// gradient). Cells získávají vertikální environmental sensing (smell_grad_z,
/// pheromone_grad_z přes inputs[17,19]).
///
/// Sprint 64: z=20 → z=50 expansion + MAX_POPULATION 1000 → 2500
/// (proportional volumetric scaling, cells/unit³ ≈ 1.2e-5).
///
/// Sprint 100: z=50 → z=100 expansion + MAX_POPULATION 2500 → 5000
/// (drží density 1.2e-5 cells/unit³). Grid resolutions (`*_GRID_RES_Z=16`)
/// nezměněné — cell_size_z naroste 6.25 → 12.5, stále jemnější než xy
/// (cell_size_x=30, cell_size_y≈17), takže thin-world aspect rationale
/// ze S53 platí dál. Food count auto-scaluje přes `food_target` z_factor.
pub const WORLD_HALF: [f32; 3] = [960.0, 540.0, 100.0];

pub const WORLD_MAP_RES: usize = 64;
/// Sprint 53: WorldMap z-axis resolution.
pub const WORLD_MAP_RES_Z: usize = 16;
pub const WORLD_MAP_BASE_RES: usize = 8;
/// Sprint 53: base z-axis resolution pro WorldMap. Lower base → smoother
/// vertical noise (less high-frequency variation v thin z-volume).
pub const WORLD_MAP_BASE_RES_Z: usize = 4;
pub const WORLD_MAP_SEED: u64 = 1234;
// Food-value multiplier = FLOOR + AMP × noise(pos), noise ∈ [0,1].
// → multiplier ∈ [FLOOR, FLOOR+AMP]. Drives spatial selection on richness.
pub const WORLD_MAP_FOOD_FLOOR: f32 = 0.85;
pub const WORLD_MAP_FOOD_AMP: f32 = 0.3;
