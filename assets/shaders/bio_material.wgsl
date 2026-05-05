// Sprint 91: procedurální fragment shader pro cells + hunter. Reuse PBR
// pipeline (lighting, fog, tonemapping, bloom přes HDR) přes
// ExtendedMaterial<StandardMaterial, BioMaterialExt>; tento shader override
// fragment a moduluje `pbr_input.material` před apply_pbr_lighting.
//
// Pattern_kind:
//   0 = CELL (jelly membrane: bright Voronoi edges, dim cores, smooth)
//   1 = HUNTER (chitinous scales: dark edges, bright centers, partial metallic)
//
// Texture coord = world_normal × scale → texture rotates with cell (looks 3D-baked
// na povrchu mesh).

#import bevy_pbr::{
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing, alpha_discard},
    forward_io::{VertexOutput, FragmentOutput},
}

struct BioParams {
    pattern_kind: u32,
    scale: f32,
    intensity: f32,
    _pad: f32,
}

@group(2) @binding(100) var<uniform> bio: BioParams;

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

    // Procedurální coordinate = world_normal × scale. Normal je unit vector
    // → world_position-independent → texture „nedrifuje" když cell letí
    // prostorem.
    let p = in.world_normal * bio.scale;
    let v = voronoi(p);
    let edge = smoothstep(0.0, 0.2, v.y - v.x);

    if (bio.pattern_kind == 0u) {
        // CELL: jelly membrane. Bright Voronoi edges (membrane web),
        // dim cores (cytoplasm). Emissive boost na edges → bioluminescence.
        let core_factor = 1.0 - edge;
        pbr_input.material.base_color =
            vec4<f32>(pbr_input.material.base_color.rgb * mix(0.4, 1.4, edge) * bio.intensity,
                      pbr_input.material.base_color.a);
        pbr_input.material.emissive =
            vec4<f32>(pbr_input.material.emissive.rgb * (1.0 + edge * 2.5),
                      pbr_input.material.emissive.a);
        pbr_input.material.perceptual_roughness = mix(0.3, 0.7, core_factor);
    } else {
        // HUNTER: chitinous scales. Dark edges (between scales), bright
        // centers (scale plates). Partial metallic na edges → reflective armor.
        let scale_center = 1.0 - edge;
        pbr_input.material.base_color =
            vec4<f32>(pbr_input.material.base_color.rgb * mix(0.3, 1.2, scale_center) * bio.intensity,
                      pbr_input.material.base_color.a);
        pbr_input.material.emissive =
            vec4<f32>(pbr_input.material.emissive.rgb * mix(2.5, 0.5, edge),
                      pbr_input.material.emissive.a);
        pbr_input.material.perceptual_roughness = mix(0.2, 0.6, edge);
        pbr_input.material.metallic = mix(0.0, 0.4, edge);
    }

    var out: FragmentOutput;
    out.color = apply_pbr_lighting(pbr_input);
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);
    return out;
}
