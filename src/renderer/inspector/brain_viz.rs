//! Brain panel: three vertical columns of activation bars for the last
//! forward pass (`last_inputs` / `last_hidden` / `last_outputs`). Bar
//! color tracks signed magnitude — red for negative, white for zero,
//! green for positive — so the user can read activity sign at a glance.
//! Labels mark sensor / motor slots whose meaning is fixed by the
//! simulation contract (`src/sensors.rs` + `src/params/brain.rs`).

use bevy::prelude::*;
use bevy_egui::egui::{self, Color32, Pos2, Rect, RichText, Sense, Stroke, Ui, Vec2};
use bioscape::{Cell, BRAIN_HIDDEN, BRAIN_INPUTS, BRAIN_INPUTS_SENSORY, BRAIN_OUTPUTS};

use super::ActivationHistory;

const BAR_WIDTH: f32 = 130.0;
const BAR_HEIGHT: f32 = 14.0;
const BAR_GAP: f32 = 2.0;

pub(super) fn brain_panel(ui: &mut Ui, cell: &Cell) {
    ui.horizontal(|ui| {
        column(
            ui,
            "Inputs",
            &cell.last_inputs[..],
            BRAIN_INPUTS,
            input_label,
        );
        ui.separator();
        column(
            ui,
            "Hidden",
            &cell.last_hidden[..],
            BRAIN_HIDDEN,
            hidden_label,
        );
        ui.separator();
        column(
            ui,
            "Outputs",
            &cell.last_outputs[..],
            BRAIN_OUTPUTS,
            output_label,
        );
    });
}

fn column(
    ui: &mut Ui,
    title: &str,
    values: &[f32],
    count: usize,
    labeler: impl Fn(usize) -> String,
) {
    ui.vertical(|ui| {
        ui.label(RichText::new(title).strong());
        ui.label(RichText::new(format!("{} slots", count)).small().weak());
        egui::ScrollArea::vertical()
            .id_salt(title)
            .max_height(280.0)
            .auto_shrink([false, false])
            .show(ui, |ui| {
                for (idx, &v) in values.iter().take(count).enumerate() {
                    activation_row(ui, idx, v, &labeler(idx));
                }
            });
    });
}

fn activation_row(ui: &mut Ui, idx: usize, value: f32, label: &str) {
    ui.horizontal(|ui| {
        ui.label(
            RichText::new(format!("{:>2}", idx))
                .monospace()
                .small()
                .weak(),
        );
        draw_bar(ui, value);
        ui.label(RichText::new(format!("{:+.2}", value)).monospace().small());
        ui.label(RichText::new(label).small().weak());
    });
}

fn draw_bar(ui: &mut Ui, value: f32) {
    let (rect, _resp) = ui.allocate_exact_size(Vec2::new(BAR_WIDTH, BAR_HEIGHT), Sense::hover());
    let painter = ui.painter();
    // Background track + zero baseline. Track is a faint grey rounded
    // rect; baseline is the visual zero that signed bars grow out of.
    painter.rect_filled(rect, 2.0, Color32::from_gray(34));
    let mid_x = rect.center().x;
    painter.line_segment(
        [
            egui::pos2(mid_x, rect.top() + 1.0),
            egui::pos2(mid_x, rect.bottom() - 1.0),
        ],
        Stroke::new(1.0, Color32::from_gray(80)),
    );
    let clamped = value.clamp(-1.0, 1.0);
    let half = rect.width() * 0.5;
    let fill = (clamped.abs() * half).max(1.0);
    let (x0, x1) = if clamped >= 0.0 {
        (mid_x, mid_x + fill)
    } else {
        (mid_x - fill, mid_x)
    };
    let bar = Rect::from_min_max(
        egui::pos2(x0, rect.top() + 2.0),
        egui::pos2(x1, rect.bottom() - 2.0),
    );
    painter.rect_filled(bar, 1.5, activation_color(clamped));
    painter.rect_stroke(
        rect,
        2.0,
        Stroke::new(1.0, Color32::from_gray(58)),
        egui::StrokeKind::Inside,
    );
    let _ = BAR_GAP;
}

fn activation_color(value: f32) -> Color32 {
    let v = value.clamp(-1.0, 1.0);
    if v >= 0.0 {
        // White (zero) → green (+1). Lerp linearly per channel.
        let t = v;
        let r = lerp_u8(220, 80, t);
        let g = lerp_u8(220, 220, t);
        let b = lerp_u8(220, 90, t);
        Color32::from_rgb(r, g, b)
    } else {
        let t = -v;
        let r = lerp_u8(220, 230, t);
        let g = lerp_u8(220, 80, t);
        let b = lerp_u8(220, 90, t);
        Color32::from_rgb(r, g, b)
    }
}

fn lerp_u8(a: u8, b: u8, t: f32) -> u8 {
    let af = a as f32;
    let bf = b as f32;
    (af + (bf - af) * t).clamp(0.0, 255.0) as u8
}

/// Brain input slot → human-readable hint. Mirrors the layout documented
/// at the top of `src/params/brain.rs` and the slot writes in
/// `src/sensors.rs::populate_brain_inputs`. Slots past
/// `BRAIN_INPUTS_SENSORY` are recurrent feedback from the previous tick's
/// hidden layer.
fn input_label(slot: usize) -> String {
    if slot >= BRAIN_INPUTS_SENSORY {
        return format!("recurrent[{}]", slot - BRAIN_INPUTS_SENSORY);
    }
    match slot {
        0 => "food Δx".into(),
        1 => "food Δy".into(),
        2 => "cell Δx".into(),
        3 => "cell Δy".into(),
        4 => "energy".into(),
        5 => "speed".into(),
        6 => "rel size".into(),
        7 => "smell ∇x".into(),
        8 => "smell ∇y".into(),
        9 => "heading x".into(),
        10 => "heading y".into(),
        11 => "phero0 ∇x".into(),
        12 => "phero0 ∇y".into(),
        13 => "density".into(),
        14 => "damage".into(),
        15 => "food Δz".into(),
        16 => "cell Δz".into(),
        17 => "smell ∇z".into(),
        18 => "heading z".into(),
        19 => "phero0 ∇z".into(),
        20 => "temperature".into(),
        21 => "phero1 ∇x".into(),
        22 => "phero1 ∇y".into(),
        23 => "phero1 ∇z".into(),
        24 => "phero2 ∇x".into(),
        25 => "phero2 ∇y".into(),
        26 => "phero2 ∇z".into(),
        27 => "bond inbox 0".into(),
        28 => "bond inbox 1".into(),
        29 => "vibration ∇x".into(),
        30 => "vibration ∇y".into(),
        31 => "vibration ∇z".into(),
        32 => "vibration amp".into(),
        33 => "whisker +fwd".into(),
        34 => "whisker -fwd".into(),
        35 => "whisker +right".into(),
        36 => "whisker -right".into(),
        37 => "whisker +up".into(),
        38 => "whisker -down".into(),
        _ => String::new(),
    }
}

fn hidden_label(_slot: usize) -> String {
    String::new()
}

fn output_label(slot: usize) -> String {
    match slot {
        0 => "turn".into(),
        1 => "thrust".into(),
        2 => "phero0 emit".into(),
        3 => "morph length".into(),
        4 => "morph width".into(),
        5 => "morph spike".into(),
        6 => "attack".into(),
        7 => "pitch".into(),
        8 => "morph height".into(),
        9 => "bond signal".into(),
        10 => "phero1 emit".into(),
        11 => "phero2 emit".into(),
        12 => "bond msg 0".into(),
        13 => "bond msg 1".into(),
        _ => String::new(),
    }
}

// ─── Weight heatmaps ────────────────────────────────────────────────────────

const HEATMAP_CELL_W1: f32 = 4.0;
const HEATMAP_CELL_W2: f32 = 7.5;

pub(super) fn weights_panel(ui: &mut Ui, cell: &Cell) {
    let (w1_max, w2_max) = brain_weight_extents(cell);
    legend_row(ui, "color encodes signed weight; saturation at ±max|w|");
    ui.add_space(2.0);
    ui.label(
        RichText::new(format!(
            "w1  input → hidden    {} × {}   max|w| = {:.3}",
            BRAIN_INPUTS, BRAIN_HIDDEN, w1_max
        ))
        .strong(),
    );
    draw_weight_grid(
        ui,
        BRAIN_INPUTS,
        BRAIN_HIDDEN,
        HEATMAP_CELL_W1,
        w1_max,
        |row, col| cell.genome.brain.w1[row][col],
        |row, col, w| {
            format!(
                "hidden[{}] ← input[{}]  ({})\nweight = {:+.4}",
                row,
                col,
                input_label(col),
                w
            )
        },
    );
    ui.add_space(8.0);
    ui.label(
        RichText::new(format!(
            "w2  hidden → output   {} × {}   max|w| = {:.3}",
            BRAIN_HIDDEN, BRAIN_OUTPUTS, w2_max
        ))
        .strong(),
    );
    draw_weight_grid(
        ui,
        BRAIN_HIDDEN,
        BRAIN_OUTPUTS,
        HEATMAP_CELL_W2,
        w2_max,
        |row, col| cell.genome.brain.w2[row][col],
        |row, col, w| {
            format!(
                "output[{}] ({}) ← hidden[{}]\nweight = {:+.4}",
                row,
                output_label(row),
                col,
                w
            )
        },
    );
    ui.add_space(6.0);
    ui.collapsing(RichText::new("Biases").small().weak(), |ui| {
        ui.label(RichText::new("b1 (hidden):").small());
        bias_row(ui, &cell.genome.brain.b1[..BRAIN_HIDDEN]);
        ui.label(RichText::new("b2 (output):").small());
        bias_row(ui, &cell.genome.brain.b2[..BRAIN_OUTPUTS]);
    });
}

fn brain_weight_extents(cell: &Cell) -> (f32, f32) {
    let mut w1m: f32 = 1e-6;
    for row in cell.genome.brain.w1.iter() {
        for &w in row.iter() {
            w1m = w1m.max(w.abs());
        }
    }
    let mut w2m: f32 = 1e-6;
    for row in cell.genome.brain.w2.iter() {
        for &w in row.iter() {
            w2m = w2m.max(w.abs());
        }
    }
    (w1m, w2m)
}

fn draw_weight_grid(
    ui: &mut Ui,
    cols: usize,
    rows: usize,
    cell_size: f32,
    max_abs: f32,
    weight: impl Fn(usize, usize) -> f32,
    tooltip: impl Fn(usize, usize, f32) -> String,
) {
    let total = Vec2::new(cols as f32 * cell_size, rows as f32 * cell_size);
    let (rect, response) = ui.allocate_exact_size(total, Sense::hover());
    let painter = ui.painter();
    let inv = if max_abs > 1e-6 { 1.0 / max_abs } else { 0.0 };
    for r in 0..rows {
        for c in 0..cols {
            let w = weight(r, c);
            let norm = (w * inv).clamp(-1.0, 1.0);
            let color = activation_color(norm);
            let min = Pos2::new(
                rect.min.x + c as f32 * cell_size,
                rect.min.y + r as f32 * cell_size,
            );
            let max = Pos2::new(min.x + cell_size, min.y + cell_size);
            painter.rect_filled(Rect::from_min_max(min, max), 0.0, color);
        }
    }
    painter.rect_stroke(
        rect,
        0.0,
        Stroke::new(1.0, Color32::from_gray(60)),
        egui::StrokeKind::Outside,
    );
    if let Some(hover_pos) = response.hover_pos() {
        let local = hover_pos - rect.min;
        let c = (local.x / cell_size).floor() as i32;
        let r = (local.y / cell_size).floor() as i32;
        if c >= 0 && r >= 0 && (c as usize) < cols && (r as usize) < rows {
            let w = weight(r as usize, c as usize);
            response.on_hover_text(tooltip(r as usize, c as usize, w));
        }
    }
}

fn bias_row(ui: &mut Ui, biases: &[f32]) {
    let max_abs = biases.iter().fold(1e-6_f32, |a, &b| a.max(b.abs()));
    let inv = if max_abs > 1e-6 { 1.0 / max_abs } else { 0.0 };
    let cell_size = 10.0;
    let total = Vec2::new(biases.len() as f32 * cell_size, cell_size);
    let (rect, response) = ui.allocate_exact_size(total, Sense::hover());
    let painter = ui.painter();
    for (i, &b) in biases.iter().enumerate() {
        let norm = (b * inv).clamp(-1.0, 1.0);
        let min = Pos2::new(rect.min.x + i as f32 * cell_size, rect.min.y);
        let max = Pos2::new(min.x + cell_size, min.y + cell_size);
        painter.rect_filled(Rect::from_min_max(min, max), 0.0, activation_color(norm));
    }
    if let Some(hover_pos) = response.hover_pos() {
        let i = ((hover_pos.x - rect.min.x) / cell_size).floor() as i32;
        if i >= 0 && (i as usize) < biases.len() {
            response.on_hover_text(format!(
                "bias[{}] = {:+.4}    max|b| = {:.3}",
                i, biases[i as usize], max_abs
            ));
        }
    }
}

fn legend_row(ui: &mut Ui, caption: &str) {
    ui.horizontal(|ui| {
        let (rect, _) = ui.allocate_exact_size(Vec2::new(140.0, 12.0), Sense::hover());
        let painter = ui.painter();
        let n = 64;
        let step = rect.width() / n as f32;
        for i in 0..n {
            let t = i as f32 / (n - 1) as f32 * 2.0 - 1.0;
            let min = Pos2::new(rect.min.x + i as f32 * step, rect.min.y);
            let max = Pos2::new(min.x + step + 0.5, rect.max.y);
            painter.rect_filled(Rect::from_min_max(min, max), 0.0, activation_color(t));
        }
        painter.rect_stroke(
            rect,
            0.0,
            Stroke::new(1.0, Color32::from_gray(70)),
            egui::StrokeKind::Outside,
        );
        ui.label(RichText::new(" −max").monospace().small().weak());
        ui.label(RichText::new("0").monospace().small().weak());
        ui.label(RichText::new("+max ").monospace().small().weak());
        ui.label(RichText::new(caption).small().weak());
    });
}

// ─── Activation history mini-plot ──────────────────────────────────────────

const PLOT_HEIGHT: f32 = 120.0;
/// How many distinct hidden neurons to overlay. Picking by max activity
/// over the window means quiet neurons drop out automatically and the
/// plot stays legible at 45 hidden neurons.
const TOP_HIDDEN_LINES: usize = 8;

pub(super) fn history_panel(ui: &mut Ui, history: &ActivationHistory) {
    let len = history.hidden.len();
    if len < 2 {
        ui.label(
            RichText::new("Recording… (history fills as the cell ticks; freezes on death)")
                .small()
                .weak(),
        );
        return;
    }
    ui.label(
        RichText::new(format!(
            "window: {} ticks  ({:.2} s @ 60 Hz)",
            len,
            len as f32 / 60.0
        ))
        .small()
        .weak(),
    );
    ui.horizontal(|ui| {
        ui.vertical(|ui| {
            ui.label(RichText::new("Hidden (top by activity)").strong());
            draw_history_plot(ui, &history.hidden, BRAIN_HIDDEN, Some(TOP_HIDDEN_LINES));
        });
        ui.separator();
        ui.vertical(|ui| {
            ui.label(RichText::new("Outputs").strong());
            draw_history_plot(ui, &history.outputs, BRAIN_OUTPUTS, None);
            ui.add_space(2.0);
            output_legend(ui);
        });
    });
}

fn output_legend(ui: &mut Ui) {
    ui.horizontal_wrapped(|ui| {
        for i in 0..BRAIN_OUTPUTS {
            let color = line_color(i, BRAIN_OUTPUTS);
            let (rect, _) = ui.allocate_exact_size(Vec2::new(10.0, 10.0), Sense::hover());
            ui.painter().rect_filled(rect, 2.0, color);
            ui.label(
                RichText::new(format!("{}:{}", i, output_label(i)))
                    .small()
                    .weak(),
            );
        }
    });
}

fn draw_history_plot<const N: usize>(
    ui: &mut Ui,
    samples: &std::collections::VecDeque<[f32; N]>,
    count: usize,
    top_k: Option<usize>,
) {
    let size = Vec2::new(380.0, PLOT_HEIGHT);
    let (rect, response) = ui.allocate_exact_size(size, Sense::hover());
    let painter = ui.painter();
    painter.rect_filled(rect, 2.0, Color32::from_gray(20));
    // Horizontal reference lines at y = -1, 0, +1.
    let y_for = |v: f32| -> f32 {
        let t = (v.clamp(-1.0, 1.0) + 1.0) * 0.5;
        rect.max.y - t * rect.height()
    };
    for &v in &[-1.0_f32, 0.0, 1.0] {
        let y = y_for(v);
        let stroke = if v == 0.0 {
            Stroke::new(1.0, Color32::from_gray(80))
        } else {
            Stroke::new(0.5, Color32::from_gray(50))
        };
        painter.line_segment([Pos2::new(rect.min.x, y), Pos2::new(rect.max.x, y)], stroke);
    }
    let n_samples = samples.len();
    if n_samples < 2 {
        return;
    }
    let x_step = rect.width() / (n_samples - 1) as f32;
    let lines: Vec<usize> = match top_k {
        Some(k) => top_active_neurons(samples, count, k),
        None => (0..count).collect(),
    };
    for &neuron in &lines {
        let mut points: Vec<Pos2> = Vec::with_capacity(n_samples);
        for (i, sample) in samples.iter().enumerate() {
            let x = rect.min.x + i as f32 * x_step;
            let y = y_for(sample[neuron]);
            points.push(Pos2::new(x, y));
        }
        let color = line_color(neuron, count);
        painter.add(egui::Shape::line(points, Stroke::new(1.2, color)));
    }
    painter.rect_stroke(
        rect,
        2.0,
        Stroke::new(1.0, Color32::from_gray(60)),
        egui::StrokeKind::Outside,
    );
    if let Some(pos) = response.hover_pos() {
        let local_x = (pos.x - rect.min.x).clamp(0.0, rect.width());
        let i = (local_x / x_step).round() as usize;
        let i = i.min(n_samples - 1);
        let sample = &samples[i];
        let mut lines_txt = format!("t = -{}  (sample {}/{})\n", n_samples - 1 - i, i, n_samples);
        for &neuron in &lines {
            lines_txt.push_str(&format!("  [{}] {:+.3}\n", neuron, sample[neuron]));
        }
        response.on_hover_text(lines_txt);
        painter.line_segment(
            [
                Pos2::new(rect.min.x + i as f32 * x_step, rect.min.y),
                Pos2::new(rect.min.x + i as f32 * x_step, rect.max.y),
            ],
            Stroke::new(0.5, Color32::from_gray(110)),
        );
    }
}

fn top_active_neurons<const N: usize>(
    samples: &std::collections::VecDeque<[f32; N]>,
    count: usize,
    k: usize,
) -> Vec<usize> {
    let mut energy = vec![0.0_f32; count];
    for s in samples.iter() {
        for n in 0..count {
            energy[n] = energy[n].max(s[n].abs());
        }
    }
    let mut idx: Vec<usize> = (0..count).collect();
    idx.sort_by(|a, b| {
        energy[*b]
            .partial_cmp(&energy[*a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    idx.truncate(k.min(count));
    idx.sort();
    idx
}

/// Stable hue-shifted palette. Same `neuron` always maps to the same
/// color, no matter what `total` is — the offset comes from a golden-ratio
/// step so adjacent indices land far apart on the hue wheel.
fn line_color(neuron: usize, _total: usize) -> Color32 {
    const GOLDEN: f32 = 0.61803398875;
    let hue = (neuron as f32 * GOLDEN).fract();
    hsv_to_color(hue, 0.85, 1.0)
}

fn hsv_to_color(h: f32, s: f32, v: f32) -> Color32 {
    let i = (h * 6.0).floor() as i32;
    let f = h * 6.0 - i as f32;
    let p = v * (1.0 - s);
    let q = v * (1.0 - f * s);
    let t = v * (1.0 - (1.0 - f) * s);
    let (r, g, b) = match i.rem_euclid(6) {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    };
    Color32::from_rgb((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
}
