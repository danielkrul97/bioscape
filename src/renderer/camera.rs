use bevy::input::mouse::{MouseMotion, MouseScrollUnit, MouseWheel};
use bevy::prelude::*;

use super::config::*;
use super::resources::OrbitCamera;

/// Sprint 36: mouse drag rotuje kamerou kolem `target` (orbit) NEBO pannuje
/// `target` — left = orbit, middle = pan. Horizontální delta orbit → yaw,
/// vertical → pitch. Pan v "cursor pulls world" módu (drag right ⇒ target left).
pub(super) fn camera_orbit_input(
    buttons: Res<ButtonInput<MouseButton>>,
    mut motion: MessageReader<MouseMotion>,
    mut orbit: ResMut<OrbitCamera>,
) {
    // Sprint 73 chore: right button orbit alias. Blender/CAD-style users
    // očekávají rotaci na pravém tlačítku; left button zůstává jako
    // primary (Bevy default), right je pohodlná alternativa.
    let orbit_active =
        buttons.pressed(MouseButton::Left) || buttons.pressed(MouseButton::Right);
    let pan_active = buttons.pressed(MouseButton::Middle);
    if !orbit_active && !pan_active {
        // Drop accumulated motion when not actively dragging — jinak by
        // se delta nasčítaly a po stisku tlačítka by kamera skočila.
        motion.clear();
        return;
    }
    let mut delta = Vec2::ZERO;
    for ev in motion.read() {
        delta += ev.delta;
    }
    if delta == Vec2::ZERO {
        return;
    }
    if orbit_active {
        orbit.yaw = (orbit.yaw + delta.x * ORBIT_SENSITIVITY).rem_euclid(std::f32::consts::TAU);
        orbit.pitch =
            (orbit.pitch + delta.y * ORBIT_SENSITIVITY).clamp(CAMERA_PITCH_MIN, CAMERA_PITCH_MAX);
    } else if pan_active {
        // Pan target proti směru drag (cursor pulls world). Pan rovina v xy
        // podle yaw — right vector + forward vector v xy projekci.
        // Vertical screen drag (y) ≡ "do scény" → forward; Y screen jde dolů,
        // takže invertovat (y- = forward+).
        let cos_y = orbit.yaw.cos();
        let sin_y = orbit.yaw.sin();
        let forward_xy = Vec2::new(sin_y, cos_y);
        let right_xy = Vec2::new(cos_y, -sin_y);
        // Pan rychlost ∝ scale (víc zoomout = rychlejší pan, drag-distance
        // odpovídá viditelnému světu).
        let speed = orbit.scale;
        let world_xy = -right_xy * delta.x * speed + forward_xy * delta.y * speed;
        orbit.target.x += world_xy.x;
        orbit.target.y += world_xy.y;
    }
}

/// Sprint 36: mouse wheel zoom — adjustuje orthographic scale. Scroll up =
/// zoom in (menší scale = víc pixelů per world unit). Clamp brání zoom out
/// pryč ze scény (nebyly by vidět hranice světa, jen black void).
pub(super) fn camera_zoom_input(
    mut wheel: MessageReader<MouseWheel>,
    mut orbit: ResMut<OrbitCamera>,
) {
    let mut scroll = 0.0_f32;
    for ev in wheel.read() {
        scroll += match ev.unit {
            MouseScrollUnit::Line => ev.y,
            MouseScrollUnit::Pixel => ev.y / 50.0,
        };
    }
    if scroll == 0.0 {
        return;
    }
    let factor = (-scroll * CAMERA_ZOOM_STEP).exp();
    orbit.scale = (orbit.scale * factor).clamp(CAMERA_SCALE_MIN, CAMERA_SCALE_MAX);
}

/// Sprint 36: WASD/šipky pannují `OrbitCamera.target` v xy-plochy ve frame
/// kamery (W = posun "do scény", A = doleva). Pan rychlost ∝ distance
/// (víc zoomout = rychlejší pan), takže feel je konzistentní napříč zoom.
pub(super) fn camera_pan_input(
    keys: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    mut orbit: ResMut<OrbitCamera>,
) {
    let mut delta = Vec2::ZERO;
    if keys.pressed(KeyCode::ArrowLeft) || keys.pressed(KeyCode::KeyA) {
        delta.x -= 1.0;
    }
    if keys.pressed(KeyCode::ArrowRight) || keys.pressed(KeyCode::KeyD) {
        delta.x += 1.0;
    }
    if keys.pressed(KeyCode::KeyS) {
        delta.y -= 1.0;
    }
    if keys.pressed(KeyCode::KeyW) {
        delta.y += 1.0;
    }
    if delta == Vec2::ZERO {
        return;
    }
    // Pan rychlost ∝ scale — při zoom in pan jemně, při zoom out rychle.
    // 800 world units/s při scale=1.0 = ~3 sec přejet šíři screenu.
    let speed = orbit.scale * 800.0 * time.delta_secs();
    // Pan v rovině xy podle yaw orientace kamery: forward (do scény) je směr,
    // kterým camera kouká po xy projekci. Right = ⊥ k forwardu v xy.
    let cos_y = orbit.yaw.cos();
    let sin_y = orbit.yaw.sin();
    let forward_xy = Vec2::new(sin_y, cos_y);
    let right_xy = Vec2::new(cos_y, -sin_y);
    let world_xy = forward_xy * delta.y + right_xy * delta.x;
    orbit.target.x += world_xy.x * speed;
    orbit.target.y += world_xy.y * speed;
}

/// Sprint 36: aplikuje OrbitCamera state na Camera3d Transform a Projection
/// scale. Běží každý frame po input systemech, takže input změny se okamžitě
/// projeví ve view matici.
pub(super) fn update_orbit_camera_transform(
    orbit: Res<OrbitCamera>,
    camera: Single<(&mut Transform, &mut Projection), With<Camera3d>>,
) {
    let (mut transform, mut projection) = camera.into_inner();
    *transform = orbit.transform();
    if let Projection::Orthographic(ortho) = &mut *projection {
        ortho.scale = orbit.scale;
    }
}
