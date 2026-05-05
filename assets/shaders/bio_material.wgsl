// Sprint 91: procedurální fragment shader pro cells + hunter. Reuse PBR
// pipeline (lighting, fog, tonemapping, bloom přes HDR) přes
// ExtendedMaterial<StandardMaterial, BioMaterialExt>; tento shader override
// fragment a moduluje `pbr_input.material` před apply_pbr_lighting.
//
// Pattern_kind se nezprává uniformem (binding 100 vs Bevy 0.18 layout = pain),
// ale detekce ze `pbr_input.material.emissive.r`:
//   emissive.r > 2.0 → HUNTER (chitinous scales) — hunter má LinearRgba(3.5,0,0)
//   else            → CELL (jelly membrane)
//
// Texture coord = world_normal × scale → texture rotates s mesh.

#import bevy_pbr::{
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing, alpha_discard},
    forward_io::{VertexOutput, FragmentOutput},
}

fn hash31(p: vec3<f32>) -> f32 {
    let h = dot(p, vec3<f32>(127.1, 311.7, 74.7));
    return fract(sin(h) * 43758.5453);
}

fn hash33(p: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        hash31(p),
        hash31(p + vec3<f32>(1.0, 2.0, 3.0)),
        hash31(p + vec3<f32>(4.0, 5.0, 6.0)),
    );
}

// 3D Voronoi: vrací (F1, F2) — distance k nejbližšímu a druhému nejbližšímu cell
// pointu. F2 - F1 → border distance (peak na hraně mezi Voronoi cells).
fn voronoi(p: vec3<f32>) -> vec2<f32> {
    let cell = floor(p);
    let local = fract(p);
    var f1: f32 = 8.0;
    var f2: f32 = 8.0;
    for (var i: i32 = -1; i <= 1; i = i + 1) {
        for (var j: i32 = -1; j <= 1; j = j + 1) {
            for (var k: i32 = -1; k <= 1; k = k + 1) {
                let offset = vec3<f32>(f32(i), f32(j), f32(k));
                let neighbor = cell + offset;
                let h = hash33(neighbor);
                let point = offset + h - local;
                let d = dot(point, point);
                if d < f1 {
                    f2 = f1;
                    f1 = d;
                } else if d < f2 {
                    f2 = d;
                }
            }
        }
    }
    return vec2<f32>(sqrt(f1), sqrt(f2));
}

@fragment
fn fragment(
    vertex_output: VertexOutput,
    @builtin(front_facing) is_front: bool,
) -> FragmentOutput {
    var in = vertex_output;
    var pbr_input = pbr_input_from_standard_material(in, is_front);
    pbr_input.material.base_color =
        alpha_discard(pbr_input.material, pbr_input.material.base_color);

    // Detect hunter vs cell přes emissive intensity. Hunter má emissive.r >= 3.5,
    // cells max 1.0 (HSL lightness 0.5 = LinearRgba.r ≤ 1.0).
    let is_hunter = pbr_input.material.emissive.r > 2.0;

    // Procedurální coord = world_normal × scale. Hunter má denser pattern (14)
    // pro „scaly" look; cells coarser (6) pro membrane segments.
    let scale = select(6.0, 14.0, is_hunter);
    let p = in.world_normal * scale;
    let v = voronoi(p);
    let edge = smoothstep(0.0, 0.2, v.y - v.x);

    if (is_hunter) {
        // HUNTER: chitinous scales. Dark edges (between scales), bright
        // centers (scale plates). Partial metallic na edges → reflective armor.
        let scale_center = 1.0 - edge;
        pbr_input.material.base_color =
            vec4<f32>(pbr_input.material.base_color.rgb * mix(0.3, 1.2, scale_center),
                      pbr_input.material.base_color.a);
        pbr_input.material.emissive =
            vec4<f32>(pbr_input.material.emissive.rgb * mix(2.5, 0.5, edge),
                      pbr_input.material.emissive.a);
        pbr_input.material.perceptual_roughness = mix(0.2, 0.6, edge);
        pbr_input.material.metallic = mix(0.0, 0.4, edge);
    } else {
        // CELL: jelly membrane. Bright Voronoi edges (membrane web),
        // dim cores (cytoplasm). Emissive boost na edges → bioluminescence.
        let core_factor = 1.0 - edge;
        pbr_input.material.base_color =
            vec4<f32>(pbr_input.material.base_color.rgb * mix(0.4, 1.4, edge),
                      pbr_input.material.base_color.a);
        pbr_input.material.emissive =
            vec4<f32>(pbr_input.material.emissive.rgb * (1.0 + edge * 2.5),
                      pbr_input.material.emissive.a);
        pbr_input.material.perceptual_roughness = mix(0.3, 0.7, core_factor);
    }

    var out: FragmentOutput;
    out.color = apply_pbr_lighting(pbr_input);
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);
    return out;
}
