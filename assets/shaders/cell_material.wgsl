// Per-instance cell coloring without per-cell ColorMaterial.
//
// MeshTag (u32) carries:
//   bits  0..7  = hue index (0..255 → 0..360°)
//   bits  8..15 = alpha index (0..255 → 0..1)
//
// All cells share one Material2d handle and one mesh handle, so Bevy bins them
// into a single draw call. The tag travels with each instance through
// Mesh2dUniform and is read in the vertex stage; the resulting RGBA is then
// passed to fragment via a custom output, since the default VertexOutput does
// not carry instance_index.

#import bevy_sprite::{
    mesh2d_view_bindings::view,
    mesh2d_functions as mf,
}

#ifdef TONEMAP_IN_SHADER
#import bevy_core_pipeline::tonemapping
#endif

struct Vertex {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
};

struct VOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

fn hsl_to_rgb(h: f32, s: f32, l: f32) -> vec3<f32> {
    let c = (1.0 - abs(2.0 * l - 1.0)) * s;
    let hp = h / 60.0;
    let x = c * (1.0 - abs((hp % 2.0) - 1.0));
    var rgb: vec3<f32>;
    if      (hp < 1.0) { rgb = vec3<f32>(c, x, 0.0); }
    else if (hp < 2.0) { rgb = vec3<f32>(x, c, 0.0); }
    else if (hp < 3.0) { rgb = vec3<f32>(0.0, c, x); }
    else if (hp < 4.0) { rgb = vec3<f32>(0.0, x, c); }
    else if (hp < 5.0) { rgb = vec3<f32>(x, 0.0, c); }
    else               { rgb = vec3<f32>(c, 0.0, x); }
    let m = l - c * 0.5;
    return rgb + vec3<f32>(m);
}

@vertex
fn vertex(in: Vertex) -> VOut {
    var out: VOut;
    let world_from_local = mf::get_world_from_local(in.instance_index);
    let world_pos = mf::mesh2d_position_local_to_world(
        world_from_local,
        vec4<f32>(in.position, 1.0),
    );
    out.clip_position = mf::mesh2d_position_world_to_clip(world_pos);

    let tag = mf::get_tag(in.instance_index);
    let hue_byte   = f32(tag & 0xFFu);
    let alpha_byte = f32((tag >> 8u) & 0xFFu);
    let hue   = hue_byte * (360.0 / 255.0);
    let alpha = alpha_byte / 255.0;
    let rgb   = hsl_to_rgb(hue, 0.75, 0.55);
    out.color = vec4<f32>(rgb, alpha);
    return out;
}

@fragment
fn fragment(in: VOut) -> @location(0) vec4<f32> {
    var color = in.color;
#ifdef TONEMAP_IN_SHADER
    let tonemapped = tonemapping::tone_mapping(color, view.color_grading);
    color = vec4<f32>(tonemapped.rgb, color.a);
#endif
    return color;
}
