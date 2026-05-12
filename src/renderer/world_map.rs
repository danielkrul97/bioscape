use bevy::asset::RenderAssetUsages;
use bevy::image::Image;
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use bioscape::{
    CYCLE_AMPLITUDE, CYCLE_GEN_PERIOD, HAZARD_AMP, HAZARD_DRAIN_PER_SEC, HAZARD_FLOOR, WorldMap,
    WORLD_MAP_FOOD_AMP, WORLD_MAP_FOOD_FLOOR, WORLD_UNITS_PER_FOOD,
};

use super::components::GenerationEnded;
use super::resources::{Clock, EventCalendarResource, FoodDensityFactor, WorldExtent};

pub(super) fn world_map_image(map: &WorldMap) -> Image {
    // Sprint 53: WorldMap je 3D. Ground plane overlay vykreslí xy-slice na
    // z = floor(nz/2) (canonical surface layer); food spawn taktéž samples
    // z=0 svět ⇒ middle z-slice.
    let nx = map.resolution[0];
    let ny = map.resolution[1];
    let nz = map.resolution[2];
    let z_slice = nz / 2;
    let mut data = Vec::with_capacity(nx * ny * 4);
    let low = [0.55_f32, 0.55, 0.55];
    let high = [0.92_f32, 0.92, 0.92];
    let field = map.field();
    let plane = nx * ny;
    for j in 0..ny {
        for i in 0..nx {
            let v = field[z_slice * plane + j * nx + i];
            let t = v.clamp(0.0, 1.0);
            let r = ((low[0] + t * (high[0] - low[0])) * 255.0).clamp(0.0, 255.0) as u8;
            let g = ((low[1] + t * (high[1] - low[1])) * 255.0).clamp(0.0, 255.0) as u8;
            let b = ((low[2] + t * (high[2] - low[2])) * 255.0).clamp(0.0, 255.0) as u8;
            data.push(r);
            data.push(g);
            data.push(b);
            data.push(255);
        }
    }
    Image::new(
        Extent3d {
            width: nx as u32,
            height: ny as u32,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        data,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::RENDER_WORLD,
    )
}

// Eat_food / spawn_food both compute the multiplier on the GPU directly
// from `WORLD_MAP_FOOD_FLOOR + WORLD_MAP_FOOD_AMP × richness` — the CPU
// helper is no longer called by either system but kept around in case a
// future stats / overlay path needs it.
#[allow(dead_code)]
pub(super) fn food_multiplier(noise: f32) -> f32 {
    WORLD_MAP_FOOD_FLOOR + WORLD_MAP_FOOD_AMP * noise
}

pub(super) fn hazard_drain(noise: f32) -> f32 {
    HAZARD_DRAIN_PER_SEC * (HAZARD_FLOOR + HAZARD_AMP * noise)
}

pub(super) fn food_target(extent: &WorldExtent, factor: f32) -> usize {
    // Sprint 53: scale s 3D objemem (mirror headless logiky).
    let area = (2.0 * extent.half_x) * (2.0 * extent.half_y);
    let z_extent = 2.0 * extent.half_z;
    let z_factor = (z_extent / 4.0).max(1.0);
    ((area / WORLD_UNITS_PER_FOOD) * factor.max(0.0) * z_factor) as usize
}

pub(super) fn update_food_density_cycle(
    mut events: MessageReader<GenerationEnded>,
    clock: Res<Clock>,
    calendar: Res<EventCalendarResource>,
    mut factor: ResMut<FoodDensityFactor>,
) {
    if events.read().next().is_none() {
        return;
    }
    let phase = (clock.0.generation as f32 / CYCLE_GEN_PERIOD as f32) * std::f32::consts::TAU;
    let seasonal = 1.0 + CYCLE_AMPLITUDE * phase.sin();
    // Sprint 113: FoodCrash multiplikátor (1.0 default).
    let shock_mult =
        bioscape::food_density_shock_multiplier(&calendar.0.events, clock.0.generation);
    factor.0 = seasonal * shock_mult;
}
