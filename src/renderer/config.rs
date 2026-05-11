use bevy::diagnostic::DiagnosticPath;

pub(super) const DIAG_CELL_COUNT: DiagnosticPath = DiagnosticPath::const_new("sim/cell_count");
pub(super) const DIAG_FOOD_COUNT: DiagnosticPath = DiagnosticPath::const_new("sim/food_count");
pub(super) const DIAG_BRAIN_ACT: DiagnosticPath = DiagnosticPath::const_new("sim/brain_act_ms");
pub(super) const DIAG_BRAIN_GPU_RT: DiagnosticPath =
    DiagnosticPath::const_new("sim/brain_gpu_rt_ms");
pub(super) const DIAG_BROWNIAN: DiagnosticPath = DiagnosticPath::const_new("sim/brownian_ms");
pub(super) const DIAG_BROWNIAN_GPU_RT: DiagnosticPath =
    DiagnosticPath::const_new("sim/brownian_gpu_rt_ms");
pub(super) const DIAG_COLLISIONS: DiagnosticPath = DiagnosticPath::const_new("sim/collisions_ms");
pub(super) const DIAG_PREDATION: DiagnosticPath = DiagnosticPath::const_new("sim/predation_ms");
pub(super) const DIAG_EAT_FOOD: DiagnosticPath = DiagnosticPath::const_new("sim/eat_food_ms");
pub(super) const DIAG_SMELL: DiagnosticPath = DiagnosticPath::const_new("sim/smell_field_ms");
pub(super) const DIAG_PHEROMONE: DiagnosticPath =
    DiagnosticPath::const_new("sim/pheromone_field_ms");
pub(super) const DIAG_VIBRATION: DiagnosticPath =
    DiagnosticPath::const_new("sim/vibration_field_ms");
pub(super) const DIAG_GRID_REBUILD: DiagnosticPath =
    DiagnosticPath::const_new("sim/grid_rebuild_ms");
pub(super) const DIAG_SYNC_TRANSFORMS: DiagnosticPath =
    DiagnosticPath::const_new("sim/sync_transforms_ms");
pub(super) const DIAG_TICKS_PER_FRAME: DiagnosticPath =
    DiagnosticPath::const_new("sim/ticks_per_frame");
pub(super) const DIAG_RENDER_OVERHEAD: DiagnosticPath =
    DiagnosticPath::const_new("sim/render_overhead_ms");

// Renderer-only knobs. Sim parameters live in `bioscape` (lib.rs).
// Sprint 53: zmenšeno z 2.5 (Sprint 53 volumetric expansion 10× food count
// dělalo 2.5 mesh visuálně dominantní).
pub(super) const FOOD_RADIUS: f32 = 1.0;
pub(super) const DEATH_FADE_TICKS: u32 = 30;
pub(super) const GRID_CELL_SIZE: f32 = 100.0;
pub(super) const CAMERA_ZOOM_STEP: f32 = 0.1;
// Sprint 36: orbit Camera3d s ORTHOGRAPHIC projection. Distance je fixní;
// "zoom" modifikuje ortho scale (= world units per pixel), takže větší zoom
// out neudělá black void kolem scény (na rozdíl od perspective). Cells stále
// vypadají jako 3D body díky lighting + tilted angle, jen bez perspective
// foreshortening.
/// Fixní vzdálenost camera od target. Pro ortho neovlivňuje velikost cells,
/// jen znear/zfar clipping plane positioning. 3000 dává dostatek depth bufferu.
pub(super) const CAMERA_OFFSET_DISTANCE: f32 = 3000.0;
pub(super) const CAMERA_PITCH_INITIAL: f32 = 0.95; // ~55° from xy plane
/// Ortho scale (Bevy `OrthographicProjection.scale`): 1 world unit = 1 / scale
/// pixelů. Initial 1.2 dává mírný margin kolem world bounds (1920×1080 přesně
/// padne při scale=1.0, +20 % je rezerva pro tilted view).
pub(super) const CAMERA_SCALE_INITIAL: f32 = 1.2;
pub(super) const CAMERA_SCALE_MIN: f32 = 0.2; // hluboký zoom in (~6× větší cells)
pub(super) const CAMERA_SCALE_MAX: f32 = 2.0; // limit zoom out — vždy dohlédne ke kraji world
/// Pitch clamp tight near ±π/2 — `looking_at` s up vektorem +Z degeneruje při
/// pohledu kolmo dolů. 0.05 rad ≈ 2.9° margin.
pub(super) const CAMERA_PITCH_MIN: f32 = 0.05;
pub(super) const CAMERA_PITCH_MAX: f32 = std::f32::consts::FRAC_PI_2 - 0.05;
/// Mouse drag → orbit angle delta. Tuned pro 1080p screen — full screen drag
/// = ~π rotace.
pub(super) const ORBIT_SENSITIVITY: f32 = 0.005;

/// Sprint 91: shader asset path pro `BioMaterialExt`. Loaded přes AssetServer
/// při startu, hot-reload v dev mode.
pub(super) const BIO_SHADER_PATH: &str = "shaders/bio_material.wgsl";
