use crate::{BRAIN_HIDDEN, BRAIN_INPUTS, BRAIN_OUTPUTS};
use std::sync::Arc;

// ============================================================================
// Sprint 47: shared GPU context — jeden device pro všechny subsystémy
// ============================================================================

/// Sprint 47: sdílený wgpu device + queue (Arc, protože `wgpu::Device` v 23
/// není `Clone`). Každý subsystém (BrainGpu, FieldGpu, SpatialHashGpu, StatsGpu)
/// si naklonuje `Arc::clone` handles z `GpuContext`. Předtím každý subsystém
/// vyrobil vlastní device → 4× duplicitních adapterů, žádné sdílení resourců.
/// Pro Sprint 47+ pipelines (full GPU tick) je shared device **nutnost** —
/// jedna queue submission ordering, jedna device-lifetime, shared command
/// encoding.
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

// ============================================================================
// Sprint 44: GPU brain forward batch
// ============================================================================

/// Počet f32 váh per cell — packing musí matchnout WGSL shader.
pub const BRAIN_WEIGHTS_PER_CELL: usize =
    BRAIN_HIDDEN * BRAIN_INPUTS + BRAIN_HIDDEN + BRAIN_OUTPUTS * BRAIN_HIDDEN + BRAIN_OUTPUTS;

// Layout offsety v rámci jednoho cell weights bloku — musí matchnout WGSL.
// Compile-time guard přes assert v `forward_batch`.
const W1_OFFSET: usize = 0;
pub const B1_OFFSET: usize = BRAIN_HIDDEN * BRAIN_INPUTS;
pub const W2_OFFSET: usize = B1_OFFSET + BRAIN_HIDDEN;
pub const B2_OFFSET: usize = W2_OFFSET + BRAIN_OUTPUTS * BRAIN_HIDDEN;
const _: () = assert!(W1_OFFSET == 0);
// Sprint 80 (HIDDEN 16 → 32 storage bump): B1 = 32*52 = 1664, W2 = 1664+32 =
// 1696, B2 = 1696+10*32 = 2016, WEIGHTS_PER_CELL = 2016+10 = 2026.
// Sprint 87 (BRAIN_INPUTS_SENSORY 20 → 21, BRAIN_INPUTS 52 → 53): B1 = 32*53 =
// 1696, W2 = 1696+32 = 1728, B2 = 1728+10*32 = 2048, WEIGHTS_PER_CELL = 2048+10
// = 2058.
// Sprint 103 (HIDDEN 32 → 50): B1 = 50*71 = 3550, W2 = 3550+50 = 3600,
// B2 = 3600+10*50 = 4100, WEIGHTS_PER_CELL = 4100+10 = 4110.
// Sprint 126 (multi-channel pheromones, BRAIN_INPUTS_SENSORY 21 → 27,
// BRAIN_INPUTS 71 → 77, BRAIN_OUTPUTS 10 → 12): B1 = 50*77 = 3850,
// W2 = 3850+50 = 3900, B2 = 3900+12*50 = 4500, WEIGHTS_PER_CELL = 4500+12 = 4512.
const _: () = assert!(B1_OFFSET == 3850);
const _: () = assert!(W2_OFFSET == 3900);
const _: () = assert!(B2_OFFSET == 4500);
const _: () = assert!(BRAIN_WEIGHTS_PER_CELL == 4512);

