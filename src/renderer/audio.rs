//! Vibration sonification — the camera is treated as a virtual mechanoreceptor.
//!
//! Every Bevy `Update` we sample the `VibrationResource` field at the camera
//! position (amp + gradient), smooth the signals, and write them into fundsp
//! `Shared` atomics that drive a pink-noise → SVF lowpass → stereo-pan graph
//! running on a dedicated cpal output thread.
//!
//! Audio is renderer-only; the headless binary never builds this module.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use bevy::prelude::*;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use fundsp::prelude32::*;

use super::resources::{OrbitCamera, SimWorld};

/// `gradient_at` sampling step. Matches the existing brain-sensor stencil
/// (`crate::sensors`) so audio hears what the cells hear.
const GRAD_EPSILON: f32 = 1.0;

/// Brain inputs feed `tanh(amp * 10)` / `tanh(grad * 1000)`. We reuse those
/// gains so audible loudness tracks the same "what the cell would feel" curve.
const AMP_GAIN: f32 = 10.0;
const GRAD_GAIN: f32 = 1000.0;

const CUTOFF_BASE_HZ: f32 = 120.0;
const CUTOFF_RANGE_HZ: f32 = 4000.0;
const FILTER_Q: f32 = 0.7;

const SMOOTH_TAU_SEC: f32 = 0.08;

#[derive(Resource, Clone)]
pub(super) struct VibrationAudio {
    master_amp: Shared,
    cutoff_hz: Shared,
    pan_x: Shared,
    pub(super) enabled: Arc<AtomicBool>,
    pub(super) ui_volume: Arc<atomic_float::AtomicF32>,
    smoothed_amp: f32,
    smoothed_grad_mag: f32,
    smoothed_pan: f32,
    debug_frames: u32,
}

impl Default for VibrationAudio {
    fn default() -> Self {
        Self {
            master_amp: shared(0.0),
            cutoff_hz: shared(CUTOFF_BASE_HZ),
            pan_x: shared(0.0),
            enabled: Arc::new(AtomicBool::new(true)),
            ui_volume: Arc::new(atomic_float::AtomicF32::new(4.0)),
            smoothed_amp: 0.0,
            smoothed_grad_mag: 0.0,
            smoothed_pan: 0.0,
            debug_frames: 0,
        }
    }
}

/// `cpal::Stream` is `!Send` on some backends; keep it as a NonSend resource
/// so it lives on the main thread alongside winit.
#[allow(dead_code)]
struct AudioStream(cpal::Stream);

pub(super) struct BioscapeAudioPlugin;

impl Plugin for BioscapeAudioPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<VibrationAudio>()
            .add_systems(Startup, init_audio_stream)
            .add_systems(Update, (sample_vibration_at_camera, toggle_audio_input));
    }
}

fn init_audio_stream(world: &mut World) {
    let audio = world.resource::<VibrationAudio>().clone();
    match build_stream(&audio) {
        Ok((stream, info)) => {
            if let Err(e) = stream.play() {
                warn!("audio: stream.play() failed: {e}");
                return;
            }
            world.insert_non_send_resource(AudioStream(stream));
            info!(
                "audio: stream up — device={:?} sample_rate={} channels={} format={:?}",
                info.device_name, info.sample_rate, info.channels, info.format
            );
        }
        Err(e) => warn!("audio: init failed ({e}); GUI will run silent"),
    }
}

struct StreamInfo {
    device_name: String,
    sample_rate: u32,
    channels: u16,
    format: cpal::SampleFormat,
}

fn build_stream(audio: &VibrationAudio) -> Result<(cpal::Stream, StreamInfo), String> {
    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .ok_or_else(|| "no default output device".to_string())?;
    let device_name = device
        .description()
        .map(|d| d.name().to_string())
        .unwrap_or_else(|_| "<unnamed>".into());
    let supported = device
        .default_output_config()
        .map_err(|e| format!("default_output_config: {e}"))?;
    let sample_rate_u32 = supported.sample_rate();
    let channels_u16 = supported.channels();
    let sample_rate = sample_rate_u32 as f64;
    let channels = channels_u16 as usize;
    let sample_format = supported.sample_format();
    let stream_config: cpal::StreamConfig = supported.into();
    let stream_info = StreamInfo {
        device_name,
        sample_rate: sample_rate_u32,
        channels: channels_u16,
        format: sample_format,
    };

    let cutoff = audio.cutoff_hz.clone();
    let amp = audio.master_amp.clone();
    let pan = audio.pan_x.clone();
    let enabled = audio.enabled.clone();
    let ui_volume = audio.ui_volume.clone();

    let filt = (pink() | var(&cutoff) | constant(FILTER_Q)) >> lowpass();
    let panned = (filt * var(&amp) | var(&pan)) >> panner();
    let mut node = panned;
    node.set_sample_rate(sample_rate);
    node.reset();

    let err_fn = |err| eprintln!("audio stream error: {err}");

    let mut render = move |buf: &mut [f32]| {
        let on = enabled.load(Ordering::Relaxed);
        let vol = ui_volume.load(Ordering::Relaxed);
        for frame in buf.chunks_mut(channels) {
            let (l, r) = node.get_stereo();
            // tanh soft clip after volume so users can crank ui_volume >> 1
            // without the output blowing past full-scale into hard clipping.
            let (l, r) = if on {
                ((l * vol).tanh(), (r * vol).tanh())
            } else {
                (0.0, 0.0)
            };
            if channels == 1 {
                frame[0] = 0.5 * (l + r);
            } else {
                frame[0] = l;
                frame[1] = r;
                for s in &mut frame[2..] {
                    *s = 0.0;
                }
            }
        }
    };

    match sample_format {
        cpal::SampleFormat::F32 => device
            .build_output_stream(
                &stream_config,
                move |data: &mut [f32], _| render(data),
                err_fn,
                None,
            )
            .map(|s| (s, stream_info))
            .map_err(|e| format!("build_output_stream(f32): {e}")),
        other => Err(format!("unsupported sample format: {other:?}")),
    }
}

fn sample_vibration_at_camera(
    mut audio: ResMut<VibrationAudio>,
    sim_world: Res<SimWorld>,
    time: Res<Time>,
    orbit: Res<OrbitCamera>,
    camera: Query<&Transform, With<Camera3d>>,
) {
    let Ok(cam_xf) = camera.single() else {
        return;
    };
    // Read the live `World.vibration` field (the renderer-side
    // `VibrationResource` mirror is unscheduled since S178 and stays at zero).
    // Sample at the orbit focus, not the camera eye — the camera sits ~3000
    // units above the scene, far outside the vibration grid's thin z-slab
    // (|z| ≤ 100), so `sample` at the eye would always return 0.0.
    let pos = [orbit.target.x, orbit.target.y, 0.0];

    let amp_raw = sim_world.0.vibration.sample(pos).max(0.0);
    let grad = sim_world.0.vibration.gradient_at(pos, GRAD_EPSILON);
    let grad_vec = Vec3::new(grad[0], grad[1], grad[2]);
    let grad_mag = grad_vec.length();

    // Project gradient onto the camera's right vector so "louder on the left"
    // maps to stereo left regardless of camera yaw.
    let cam_right = cam_xf.right();
    let pan_axis = grad_vec.dot(*cam_right);

    let dt = time.delta_secs().max(1e-4);
    let alpha = 1.0 - (-dt / SMOOTH_TAU_SEC).exp();
    audio.smoothed_amp += alpha * (amp_raw - audio.smoothed_amp);
    audio.smoothed_grad_mag += alpha * (grad_mag - audio.smoothed_grad_mag);
    audio.smoothed_pan += alpha * (pan_axis - audio.smoothed_pan);

    let amp_norm = (audio.smoothed_amp * AMP_GAIN).tanh();
    let cutoff_hz =
        CUTOFF_BASE_HZ + CUTOFF_RANGE_HZ * (audio.smoothed_grad_mag * GRAD_GAIN).tanh();
    let pan_val = (audio.smoothed_pan * GRAD_GAIN).clamp(-1.0, 1.0);

    audio.master_amp.set(amp_norm);
    audio.cutoff_hz.set(cutoff_hz);
    audio.pan_x.set(pan_val);

    audio.debug_frames = audio.debug_frames.wrapping_add(1);
    if audio.debug_frames % 60 == 0 {
        debug!(
            "audio: pos=({:.1},{:.1},{:.1}) amp_raw={:.4} amp_norm={:.3} grad_mag={:.5} cutoff={:.0}Hz pan={:.2}",
            pos[0], pos[1], pos[2], amp_raw, amp_norm, grad_mag, cutoff_hz, pan_val
        );
    }
}

fn toggle_audio_input(keys: Res<ButtonInput<KeyCode>>, audio: Res<VibrationAudio>) {
    if keys.just_pressed(KeyCode::F8) {
        let now = audio.enabled.load(Ordering::Relaxed);
        audio.enabled.store(!now, Ordering::Relaxed);
        info!("audio: enabled = {}", !now);
    }
    let bump = if keys.just_pressed(KeyCode::F9) {
        -1.0
    } else if keys.just_pressed(KeyCode::F10) {
        1.0
    } else {
        0.0
    };
    if bump != 0.0 {
        let cur = audio.ui_volume.load(Ordering::Relaxed);
        let next = (cur + bump).clamp(0.0, 20.0);
        audio.ui_volume.store(next, Ordering::Relaxed);
        info!("audio: ui_volume = {next:.2}");
    }
}

mod atomic_float {
    use std::sync::atomic::{AtomicU32, Ordering};

    pub struct AtomicF32(AtomicU32);

    impl AtomicF32 {
        pub fn new(v: f32) -> Self {
            Self(AtomicU32::new(v.to_bits()))
        }
        pub fn load(&self, order: Ordering) -> f32 {
            f32::from_bits(self.0.load(order))
        }
        pub fn store(&self, v: f32, order: Ordering) {
            self.0.store(v.to_bits(), order);
        }
    }
}
