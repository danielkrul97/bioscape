use bevy::pbr::{ExtendedMaterial, MaterialExtension};
use bevy::prelude::*;
use bevy::render::render_resource::AsBindGroup;
use bevy::shader::ShaderRef;
use bioscape::Phenotype;

use super::config::BIO_SHADER_PATH;
use super::resources::AdhesionMaterials;

/// Sprint 91: empty marker extension nad `StandardMaterial` — žádné custom
/// uniformy. Tím se vyhne Bevy 0.18 ExtendedMaterial uniform binding layout
/// issue (binding 100 neprošlo validation v `pbr_opaque_mesh_pipeline`).
///
/// Pattern_kind se přepíná in-shader podle base material color.
#[derive(Asset, AsBindGroup, Reflect, Debug, Clone, Default)]
pub struct BioMaterialExt {}

impl MaterialExtension for BioMaterialExt {
    fn fragment_shader() -> ShaderRef {
        BIO_SHADER_PATH.into()
    }
    fn deferred_fragment_shader() -> ShaderRef {
        BIO_SHADER_PATH.into()
    }
}

/// Sprint 91: alias pro extended material handle type. `MaterialPlugin`
/// musí být registrován pro tento typ aby Bevy renderoval s naším shaderem.
pub(super) type BioMaterial = ExtendedMaterial<StandardMaterial, BioMaterialExt>;

/// Sprint 36: vrátí (případně vytvoří) StandardMaterial handle pro daný
/// lineage_id. Hue mapuje deterministicky přes `lineage_hue`. Cache zaručuje,
/// že cells se stejným lineage sdílejí jeden material — Bevy je instance
/// podle materialu pro draw call binning, takže shared material = 1 batch.
/// Sprint 69: 8 distinctních hues per `adhesion_type`, evenly spaced kolem
/// kruhu. Lazy-cache — handle vznikne při první cell s daným typem; pak
/// re-use. Same hue se zrcadlí do bond gizmo lines, takže shluk = barva
/// těla + barva bond lines = jednolitý vizuální chunk.
pub(super) fn adhesion_material(
    cache: &mut AdhesionMaterials,
    bio_materials: &mut Assets<BioMaterial>,
    adhesion_type: u8,
) -> Handle<BioMaterial> {
    let idx = (adhesion_type as usize) % 8;
    if let Some(h) = &cache.0[idx] {
        return h.clone();
    }
    let hue = idx as f32 * (360.0 / 8.0);
    // Sprint 85: saturation 0.85 → 1.0 — sytější body color.
    // Sprint 88: emissive ∝ hue color. Pod HDR + Bloom cells „bioluminescent".
    // Sprint 91: ExtendedMaterial s pattern_kind=0 (jelly membrane). Voronoi
    // procedural shader moduluje base_color + emissive na povrchu mesh.
    let color = Color::hsl(hue, 1.0, 0.50);
    let emissive_color = Color::hsl(hue, 1.0, 0.50);
    let emissive_linear = emissive_color.to_linear();
    let handle = bio_materials.add(BioMaterial {
        base: StandardMaterial {
            base_color: color,
            // Sprint 91: emissive max ~1.0 → shader detekuje jako CELL pattern.
            emissive: LinearRgba::new(
                emissive_linear.red,
                emissive_linear.green,
                emissive_linear.blue,
                1.0,
            ),
            perceptual_roughness: 0.5,
            ..default()
        },
        extension: BioMaterialExt {},
    });
    cache.0[idx] = Some(handle.clone());
    handle
}

/// Sprint 69: hue pro adhesion gizmo lines. Match s `adhesion_material`
/// (= rovnoměrné rozdělení 360°/8 = 45° per type).
pub(super) fn adhesion_hue(adhesion_type: u8) -> f32 {
    (adhesion_type as usize % 8) as f32 * (360.0 / 8.0)
}

/// Sprint 36: Quat z yaw + pitch pro orientaci ellipsoidu. Body's local +X
/// musí mířit ve forward direction = (cos(y)cos(p), sin(y)cos(p), sin(p)).
/// Quat::from_rotation_z(yaw) * Quat::from_rotation_y(pitch) splňuje
/// (1,0,0) → forward (viz `bioscape::forward_vector`).
pub(super) fn cell_rotation(yaw: f32, pitch: f32) -> Quat {
    Quat::from_rotation_z(yaw) * Quat::from_rotation_y(-pitch)
}

/// Sprint 36: 3-axis ellipsoid scale (length × width × height). Bevy non-uniform
/// scale aplikuje na unit-radius sphere, vytváří ellipsoid s poloosami
/// (L, W, H) podél x, y, z. Po `cell_rotation(yaw, pitch)` je local +X
/// alignovaný s forward vektorem buňky.
pub(super) fn cell_scale(phenotype: &Phenotype) -> Vec3 {
    Vec3::new(
        phenotype.body_length,
        phenotype.body_width,
        phenotype.body_height,
    )
}
