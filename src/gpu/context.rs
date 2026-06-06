use crate::{BRAIN_HIDDEN, BRAIN_INPUTS, BRAIN_OUTPUTS};
use std::sync::Arc;

/// Shared wgpu device + queue. Wrapped in `Arc` because `wgpu::Device` is
/// not `Clone`; every subsystem (`BrainGpu`, `FieldGpu`, `SpatialHashGpu`,
/// `StatsGpu`, …) clones the handles instead of allocating its own adapter.
/// The full-GPU tick pipeline relies on a single device — shared queue
/// submission order and command encoding only work with one underlying
/// device-lifetime.
#[derive(Clone)]
pub struct GpuContext {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
}

impl GpuContext {
    pub fn new() -> Result<Self, String> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .map_err(|e| format!("no suitable wgpu adapter: {e:?}"))?;
        // Surface the selected adapter at init: a fallback to a software
        // rasterizer (e.g. llvmpipe/lavapipe `Cpu` type) silently turns the
        // whole sim CPU-bound and is the difference between µs and ms per
        // dispatch — make it visible instead of guessing.
        let info = adapter.get_info();
        eprintln!(
            "gpu: adapter={:?} type={:?} backend={:?} driver={} {}",
            info.name, info.device_type, info.backend, info.driver, info.driver_info
        );
        if info.device_type == wgpu::DeviceType::Cpu {
            eprintln!(
                "gpu: WARNING software adapter selected — sim will be CPU-bound. \
                 Install/repair a hardware Vulkan ICD (e.g. NVIDIA), or set WGPU_ADAPTER_NAME."
            );
        }
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("bioscape-shared"),
            required_features: wgpu::Features::empty(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            trace: wgpu::Trace::Off,
            required_limits: wgpu::Limits {
                // Wave 6 bumped 14 → 16: sensor_gather adds heading +
                // pitch bindings for whisker raycast (bindings 14, 15)
                // alongside the existing 12 + maze_mask binding 13.
                // Wave L bumped 16 → 20: ch1/ch2 pheromone grids
                // (bindings 16, 17) for multi-channel sensor gather.
                // S213 bumped 20 → 22: predate binds bond_partner_idx
                // (binding 21) + lineage_ids (binding 22) for tissue (kin /
                // clonal) predation exclusion. Discrete GPUs expose far more
                // (NVIDIA ≥ 1M); 22 is safe.
                max_storage_buffers_per_shader_stage: 22,
                ..wgpu::Limits::default()
            },
            memory_hints: wgpu::MemoryHints::Performance,
        }))
        .map_err(|e| format!("device request failed: {e:?}"))?;
        Ok(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
        })
    }
}

/// Number of f32 weights per cell. Packing must match the WGSL shaders
/// (`brain_forward.wgsl`, `hebbian.wgsl`).
pub const BRAIN_WEIGHTS_PER_CELL: usize =
    BRAIN_HIDDEN * BRAIN_INPUTS + BRAIN_HIDDEN + BRAIN_OUTPUTS * BRAIN_HIDDEN + BRAIN_OUTPUTS;

// Layout offsets inside one per-cell weight block. Mirrored in the WGSL
// shaders; the compile-time asserts below pin the values that the shader
// constants assume.
const W1_OFFSET: usize = 0;
pub const B1_OFFSET: usize = BRAIN_HIDDEN * BRAIN_INPUTS;
pub const W2_OFFSET: usize = B1_OFFSET + BRAIN_HIDDEN;
pub const B2_OFFSET: usize = W2_OFFSET + BRAIN_OUTPUTS * BRAIN_HIDDEN;
// Wave 2 (whiskers): BRAIN_INPUTS 78→84 (6 maze-whisker slots appended).
// Offsets shift accordingly; the WGSL shader constants in
// `brain_forward.wgsl`, `hebbian.wgsl`, `populate_inputs.wgsl` are kept in
// lock-step with these values.
const _: () = assert!(W1_OFFSET == 0);
// S203: removed the 2 symbiont brain slots → BRAIN_INPUTS 86→84. All
// downstream offsets shift back by BRAIN_HIDDEN × 2 = 90.
// S213+: added 1 coop-call sensory slot → BRAIN_INPUTS 84→85; offsets shift
// forward by BRAIN_HIDDEN × 1 = 45.
const _: () = assert!(B1_OFFSET == 3825);
const _: () = assert!(W2_OFFSET == 3870);
// Active vibration emit: BRAIN_OUTPUTS 14→15 adds one row to w2 (45 floats)
// and one extra b2 entry. Downstream offsets shift accordingly.
const _: () = assert!(B2_OFFSET == 4545);
const _: () = assert!(BRAIN_WEIGHTS_PER_CELL == 4560);
