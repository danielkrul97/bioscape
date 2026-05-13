//! Main inspector window. Top half: live brain panel. Bottom half:
//! collapsible metadata sections (identity, physics, state, genome,
//! bonds, neural runtime). Footer: Save…/Copy buttons + last status
//! line. The window opens whenever a cell is selected and closes when
//! the user clears the selection (Escape, click-empty, or the [×]
//! button). A frozen snapshot keeps rendering after the cell dies, with
//! a "deceased" badge in the header.

use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};
use bioscape::{Cell, NeuronModel, BRAIN_HIDDEN, BRAIN_INPUTS, BRAIN_OUTPUTS};

use super::brain_viz;
use super::export;
use super::{ActivationHistory, PendingSave, SaveStatus, SelectedCell};

pub(super) fn draw_inspector_window(
    mut contexts: EguiContexts,
    mut selected: ResMut<SelectedCell>,
    mut pending: ResMut<PendingSave>,
    history: Res<ActivationHistory>,
    keys: Res<ButtonInput<KeyCode>>,
) -> Result {
    if !selected.is_active() {
        return Ok(());
    }
    let Some(cell) = selected.snapshot else {
        return Ok(());
    };
    let deceased = selected.deceased;
    let ctx = contexts.ctx_mut()?;

    let ctrl_s = keys.just_pressed(KeyCode::KeyS)
        && (keys.pressed(KeyCode::ControlLeft) || keys.pressed(KeyCode::ControlRight));

    let mut open = true;
    let mut close_requested = false;
    let mut save_requested = ctrl_s;
    let mut copy_requested = false;

    let header = if deceased {
        format!("Cell #{}  ·  deceased", cell.cell_id)
    } else {
        format!("Cell #{}", cell.cell_id)
    };

    egui::Window::new(header)
        .id(egui::Id::new("cell_inspector"))
        .open(&mut open)
        .default_pos([16.0, 16.0])
        .fixed_size([760.0, 720.0])
        .resizable(false)
        .collapsible(true)
        .vscroll(true)
        .show(ctx, |ui| {
            header_row(ui, &cell, deceased);
            ui.add_space(4.0);

            ui.collapsing(
                egui::RichText::new("Brain — live activations").strong(),
                |ui| {
                    brain_viz::brain_panel(ui, &cell);
                    ui.add_space(4.0);
                    brain_meta_row(ui, &cell);
                },
            )
            .header_response
            .on_hover_text("Last forward pass activations (live while alive).");

            egui::CollapsingHeader::new(egui::RichText::new("Brain — activation history").strong())
                .default_open(true)
                .show(ui, |ui| {
                    brain_viz::history_panel(ui, &history);
                });

            egui::CollapsingHeader::new(egui::RichText::new("Brain — weights").strong())
                .default_open(false)
                .show(ui, |ui| {
                    brain_viz::weights_panel(ui, &cell);
                });

            ui.separator();
            metadata_sections(ui, &cell);

            ui.separator();
            ui.horizontal(|ui| {
                if ui
                    .button("💾  Save JSON…")
                    .on_hover_text("Open a native file dialog to save the full cell snapshot.")
                    .clicked()
                {
                    save_requested = true;
                }
                if ui
                    .button("📋  Copy JSON")
                    .on_hover_text("Copy the full cell snapshot to the clipboard.")
                    .clicked()
                {
                    copy_requested = true;
                }
                ui.separator();
                if ui.button("Close").clicked() {
                    close_requested = true;
                }
                if let Some(status) = &pending.last_status {
                    ui.separator();
                    status_label(ui, status);
                }
            });
            ui.add_space(2.0);
            ui.label(
                egui::RichText::new(
                    "Tip: Esc to deselect · Ctrl+S to save · click empty space to clear",
                )
                .small()
                .weak(),
            );
        });

    if save_requested {
        export::spawn_save(&mut pending, &cell);
    }
    if copy_requested {
        pending.last_status = Some(export::copy_to_clipboard(ctx, &cell));
    }
    if !open || close_requested {
        selected.clear();
    }
    Ok(())
}

fn header_row(ui: &mut egui::Ui, cell: &Cell, deceased: bool) {
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(format!("lineage {}", cell.lineage_id)).monospace());
        ui.separator();
        ui.label(format!("age {}t", cell.age));
        ui.separator();
        ui.label(format!("energy {:.1}", cell.energy));
        ui.separator();
        ui.label(format!(
            "pos ({:.0}, {:.0}, {:.0})",
            cell.position[0], cell.position[1], cell.position[2]
        ));
        if deceased {
            ui.separator();
            ui.label(
                egui::RichText::new("[snapshot frozen]")
                    .color(egui::Color32::from_rgb(220, 130, 130))
                    .strong(),
            );
        }
    });
}

fn brain_meta_row(ui: &mut egui::Ui, cell: &Cell) {
    let model = match cell.genome.neuron_model {
        NeuronModel::Perceptron => "Perceptron (tanh MLP)",
        NeuronModel::Izhikevich => "Izhikevich (spiking)",
    };
    ui.horizontal_wrapped(|ui| {
        ui.label(egui::RichText::new(model).strong());
        ui.separator();
        ui.label(format!(
            "topology  {} → {} → {}",
            BRAIN_INPUTS, BRAIN_HIDDEN, BRAIN_OUTPUTS
        ));
        ui.separator();
        ui.label(format!("active hidden: {}", cell.genome.brain.hidden_n));
        ui.separator();
        ui.label(format!("learning rate: {:.4}", cell.genome.learning_rate));
        ui.separator();
        ui.label(format!(
            "trace decay/s: {:.3}",
            cell.genome.trace_decay_per_sec
        ));
        if matches!(cell.genome.neuron_model, NeuronModel::Izhikevich) {
            ui.separator();
            ui.label(format!(
                "STDP A±: {:+.3} / {:+.3}, τ {:.1}t",
                cell.genome.stdp_a_plus,
                cell.genome.stdp_a_minus,
                cell.genome.stdp_tau_ticks
            ));
        }
    });
}

fn metadata_sections(ui: &mut egui::Ui, cell: &Cell) {
    egui::CollapsingHeader::new(egui::RichText::new("Identity & lineage").strong())
        .default_open(true)
        .show(ui, |ui| {
            kv(ui, "cell_id", format!("{}", cell.cell_id));
            kv(ui, "lineage_id", format!("{}", cell.lineage_id));
            kv(
                ui,
                "lineage_birth_gen",
                format!("{}", cell.lineage_birth_gen),
            );
            kv(ui, "age (ticks)", format!("{}", cell.age));
            kv(
                ui,
                "reproduce_cooldown",
                format!("{} t", cell.reproduce_cooldown_ticks),
            );
        });

    egui::CollapsingHeader::new(egui::RichText::new("Kinematics").strong()).show(ui, |ui| {
        kv(
            ui,
            "position",
            format!(
                "({:.2}, {:.2}, {:.2})",
                cell.position[0], cell.position[1], cell.position[2]
            ),
        );
        kv(
            ui,
            "velocity",
            format!(
                "({:.2}, {:.2}, {:.2})",
                cell.velocity[0], cell.velocity[1], cell.velocity[2]
            ),
        );
        let speed = (cell.velocity[0].powi(2)
            + cell.velocity[1].powi(2)
            + cell.velocity[2].powi(2))
        .sqrt();
        kv(ui, "|velocity|", format!("{:.2}", speed));
        kv(ui, "heading (rad)", format!("{:.3}", cell.heading));
        kv(ui, "pitch (rad)", format!("{:.3}", cell.pitch));
        kv(
            ui,
            "angular vel",
            format!("{:.3}", cell.angular_velocity),
        );
        kv(ui, "pitch vel", format!("{:.3}", cell.pitch_velocity));
    });

    egui::CollapsingHeader::new(egui::RichText::new("Energy & state").strong()).show(ui, |ui| {
        kv(ui, "energy", format!("{:.3}", cell.energy));
        kv(ui, "damage_accum", format!("{:.3}", cell.damage_accum));
        kv(ui, "cell_state", format!("{:.3}", cell.cell_state));
        kv(
            ui,
            "under_attack_streak",
            format!("{}", cell.under_attack_streak),
        );
        kv(
            ui,
            "escape_cooldown",
            format!("{} t", cell.escape_cooldown_ticks),
        );
        kv(ui, "n_bonds", format!("{}", cell.n_bonds()));
    });

    egui::CollapsingHeader::new(egui::RichText::new("Body & phenotype").strong()).show(ui, |ui| {
        kv(
            ui,
            "body L×W×H",
            format!(
                "{:.2} × {:.2} × {:.2}",
                cell.phenotype.body_length,
                cell.phenotype.body_width,
                cell.phenotype.body_height
            ),
        );
        kv(
            ui,
            "effective r",
            format!("{:.2}", cell.phenotype.effective_radius()),
        );
        kv(
            ui,
            "shell thickness",
            format!("{:.3}", cell.phenotype.shell_thickness),
        );
        kv(
            ui,
            "spike_count",
            format!("{}", cell.phenotype.spike_count),
        );
        for i in 0..(cell.phenotype.spike_count.min(5) as usize) {
            let s = cell.phenotype.spikes[i];
            ui.label(format!(
                "  spike[{}]  len={:.2}  az={:+.2}  el={:+.2}  cmpx={:.2}",
                i, s.length, s.azimuth_offset, s.elevation_offset, s.complexity
            ));
        }
        kv(
            ui,
            "adhesion_type",
            format!("{}", cell.genome.adhesion_type),
        );
    });

    egui::CollapsingHeader::new(egui::RichText::new("Genome (key knobs)").strong()).show(
        ui,
        |ui| {
            kv(ui, "max_speed", format!("{:.2}", cell.genome.max_speed));
            kv(ui, "turn_rate", format!("{:.3}", cell.genome.turn_rate));
            kv(
                ui,
                "vision_radius",
                format!("{:.2}", cell.genome.vision_radius),
            );
            kv(
                ui,
                "vision_fov (rad)",
                format!("{:.3}", cell.genome.vision_fov),
            );
            kv(
                ui,
                "thermal_optimum",
                format!("{:.2}", cell.genome.thermal_optimum),
            );
            kv(
                ui,
                "carnivore_score",
                format!("{:.3}", cell.genome.carnivore_score),
            );
            kv(
                ui,
                "sensor_gains",
                format!(
                    "[{:.2}, {:.2}, {:.2}, {:.2}]",
                    cell.genome.sensor_gains[0],
                    cell.genome.sensor_gains[1],
                    cell.genome.sensor_gains[2],
                    cell.genome.sensor_gains[3]
                ),
            );
            kv(
                ui,
                "bond_stiffness / damping",
                format!(
                    "{:.2}  /  {:.2}",
                    cell.genome.bond_stiffness, cell.genome.bond_damping
                ),
            );
        },
    );

    egui::CollapsingHeader::new(egui::RichText::new("Bonds").strong()).show(ui, |ui| {
        let active: Vec<_> = cell
            .bonds
            .iter()
            .enumerate()
            .filter_map(|(i, b)| b.as_ref().map(|b| (i, b)))
            .collect();
        if active.is_empty() {
            ui.label(egui::RichText::new("no active bonds").weak());
            return;
        }
        for (i, b) in active {
            ui.label(format!(
                "  bond[{}]  → cell {}   rest {:.2}   k={:.2}  d={:.2}   age {}t",
                i, b.other_cell_id, b.rest_length, b.stiffness, b.damping, b.age_ticks
            ));
        }
    });
}

fn kv(ui: &mut egui::Ui, key: &str, value: String) {
    ui.horizontal(|ui| {
        ui.label(
            egui::RichText::new(format!("{:>22}", key))
                .monospace()
                .weak(),
        );
        ui.label(egui::RichText::new(value).monospace());
    });
}

fn status_label(ui: &mut egui::Ui, status: &SaveStatus) {
    match status {
        SaveStatus::Ok(path) => {
            let display = if path.as_os_str() == "(clipboard)" {
                "copied to clipboard".to_string()
            } else {
                format!("saved → {}", path.display())
            };
            ui.label(
                egui::RichText::new(display)
                    .color(egui::Color32::from_rgb(110, 200, 130)),
            );
        }
        SaveStatus::Cancelled => {
            ui.label(
                egui::RichText::new("save cancelled")
                    .color(egui::Color32::from_rgb(200, 180, 110)),
            );
        }
        SaveStatus::Failed(msg) => {
            ui.label(
                egui::RichText::new(format!("save failed: {}", msg))
                    .color(egui::Color32::from_rgb(220, 110, 110)),
            );
        }
    }
}
