use bevy::asset::Asset;
use bevy::prelude::*;
use bevy::reflect::TypePath;
use bevy::render::render_resource::AsBindGroup;
use bevy::shader::ShaderRef;
use bevy::sprite_render::{AlphaMode2d, Material2d, Material2dPlugin};

const SHADER_ASSET_PATH: &str = "shaders/cell_material.wgsl";

#[derive(Asset, AsBindGroup, TypePath, Debug, Clone, Default)]
pub struct CellMaterial {}

impl Material2d for CellMaterial {
    fn vertex_shader() -> ShaderRef {
        SHADER_ASSET_PATH.into()
    }
    fn fragment_shader() -> ShaderRef {
        SHADER_ASSET_PATH.into()
    }
    fn alpha_mode(&self) -> AlphaMode2d {
        AlphaMode2d::Blend
    }
}

pub struct CellMaterialPlugin;

impl Plugin for CellMaterialPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(Material2dPlugin::<CellMaterial>::default());
    }
}

pub fn pack_cell_tag(hue_deg: f32, alpha: f32) -> u32 {
    let h = (hue_deg.rem_euclid(360.0) * (255.0 / 360.0)).round() as u32 & 0xFF;
    let a = (alpha.clamp(0.0, 1.0) * 255.0).round() as u32 & 0xFF;
    h | (a << 8)
}
