use bevy::prelude::*;
use bevy::render::view::screenshot::{save_to_disk, Screenshot};

use super::resources::ScreencastConfig;

pub(super) fn screencast_capture(
    mut commands: Commands,
    time: Res<Time<Real>>,
    cfg: Option<ResMut<ScreencastConfig>>,
    mut exit: MessageWriter<AppExit>,
) {
    // Sprint 97 follow-up: `Time<Real>` (wall clock), ne `Time<Virtual>` —
    // virtual má 50ms max_delta cap, takže pod heavy sim load by virtual
    // čas běžel 20× pomaleji než wall a 5min screencast by trval >1h.
    let Some(mut cfg) = cfg else { return; };
    let elapsed = time.elapsed_secs();
    if cfg.started_at.is_none() {
        cfg.started_at = Some(elapsed);
        cfg.last_capture = elapsed - cfg.interval_secs;
    }
    let started = cfg.started_at.unwrap();
    let dt_since_start = elapsed - started;
    if dt_since_start >= cfg.duration_secs {
        eprintln!("screencast: done, captured {} frames", cfg.frame_idx);
        exit.write(AppExit::Success);
        return;
    }
    if elapsed - cfg.last_capture < cfg.interval_secs {
        return;
    }
    let path = cfg.dir.join(format!("cap_{:05}.png", cfg.frame_idx));
    commands
        .spawn(Screenshot::primary_window())
        .observe(save_to_disk(path));
    cfg.frame_idx += 1;
    cfg.last_capture = elapsed;
}
