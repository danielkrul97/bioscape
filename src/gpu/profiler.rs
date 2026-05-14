use std::sync::Arc;
use std::time::Instant;

/// Serialize-and-time GPU profiler. When enabled, `mark()` drains all
/// submitted GPU work with a blocking `poll(Wait)`, then charges the wall
/// time since the previous `mark` / `frame_start` / `resume` to a named
/// bucket. Disabled (the default) makes every method an early-return — no
/// polls — so the production tick path is unchanged.
///
/// A bucket figure is CPU encode/submit + GPU execution of that stage.
/// `brain_act_gpu_full` already does a single `poll(Wait)` per tick and no
/// tick pipelining exists, so the extra polls only remove an overlap that
/// was never there — the per-shader ranking is faithful even though the
/// absolute numbers carry submit overhead.
pub struct GpuProfiler {
    enabled: bool,
    device: Arc<wgpu::Device>,
    last: Instant,
    frames: u64,
    buckets: Vec<Bucket>,
}

struct Bucket {
    name: &'static str,
    total_us: f64,
    calls: u64,
}

impl GpuProfiler {
    pub fn new(device: Arc<wgpu::Device>, enabled: bool) -> Self {
        Self {
            enabled,
            device,
            last: Instant::now(),
            frames: 0,
            buckets: Vec::new(),
        }
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    /// Begin a tick: drain the GPU, bump the frame counter, reset the
    /// stopwatch. The drain discards work submitted before this tick's
    /// first `mark` (e.g. the unmarked field-diffusion steps) so it is not
    /// charged to the first bucket.
    pub fn frame_start(&mut self) {
        if !self.enabled {
            return;
        }
        self.device.poll(wgpu::Maintain::Wait);
        self.frames += 1;
        self.last = Instant::now();
    }

    /// Drain the GPU and reset the stopwatch without bumping the frame
    /// counter — call at the start of a GPU region that follows unmarked
    /// phases, so the region's first `mark` measures only that region.
    pub fn resume(&mut self) {
        if !self.enabled {
            return;
        }
        self.device.poll(wgpu::Maintain::Wait);
        self.last = Instant::now();
    }

    /// Drain submitted GPU work and charge the elapsed time to `name`.
    pub fn mark(&mut self, name: &'static str) {
        if !self.enabled {
            return;
        }
        self.device.poll(wgpu::Maintain::Wait);
        let now = Instant::now();
        let dt = (now - self.last).as_secs_f64() * 1e6;
        self.last = now;
        match self.buckets.iter_mut().find(|b| b.name == name) {
            Some(b) => {
                b.total_us += dt;
                b.calls += 1;
            }
            None => self.buckets.push(Bucket {
                name,
                total_us: dt,
                calls: 1,
            }),
        }
    }

    /// Per-shader mean µs/tick, sorted by descending total time.
    pub fn report(&self) -> String {
        if self.frames == 0 {
            return "gpu-profile: no ticks recorded".to_string();
        }
        let mut rows: Vec<&Bucket> = self.buckets.iter().collect();
        rows.sort_by(|a, b| b.total_us.partial_cmp(&a.total_us).unwrap());
        let grand_total: f64 = self.buckets.iter().map(|b| b.total_us).sum();
        let frames = self.frames as f64;
        let mut out = format!(
            "gpu-profile: {} ticks, {:.1} µs/tick measured (submit + GPU exec)\n  {:<18} {:>9}  {:>6}  {:>11}\n",
            self.frames, grand_total / frames, "stage", "µs/tick", "%", "calls/tick",
        );
        for b in rows {
            out.push_str(&format!(
                "  {:<18} {:>9.1}  {:>5.1}%  {:>11.2}\n",
                b.name,
                b.total_us / frames,
                100.0 * b.total_us / grand_total,
                b.calls as f64 / frames,
            ));
        }
        out
    }
}
