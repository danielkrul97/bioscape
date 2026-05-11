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
        let adapter = pollster::block_on(instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            },
        ))
        .ok_or_else(|| "no suitable wgpu adapter".to_string())?;
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("bioscape-shared"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits {
                    max_storage_buffers_per_shader_stage: 12,
                    ..wgpu::Limits::default()
                },
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
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
// V7 (vibration): BRAIN_INPUTS 74→78 (4 mechanosensory slots). Offsets
// shift accordingly; keep the WGSL shader constants in `brain_forward.wgsl`
// and `hebbian.wgsl` in lock-step with these values.
const _: () = assert!(W1_OFFSET == 0);
const _: () = assert!(B1_OFFSET == 3510);
const _: () = assert!(W2_OFFSET == 3555);
const _: () = assert!(B2_OFFSET == 4185);
const _: () = assert!(BRAIN_WEIGHTS_PER_CELL == 4199);

