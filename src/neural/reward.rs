//! Sprint 133: per-tick reward event log decoupling reward production
//! (eat / novelty / predation / damage / bond / mate events) from reward
//! consumption (GPU Hebbian apply dispatch). Pre-S133 the few existing
//! reward sources wrote directly into a `Vec<f32>` and dispatched immediately.
//! The accumulator keeps a typed event stream so later sprints can add new
//! sources without each site having to know about every dispatch path or
//! agree on a clamp / sum policy.

use serde::{Deserialize, Serialize};

/// Origin of a reward signal. Sprint 133 ships `EatFood` + `Novelty`; the
/// remaining variants are placeholders for sprints 134–135.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RewardKind {
    EatFood,
    Novelty,
    Predation,
    EscapedAttack,
    Damage,
    BondFormed,
    MateSignalAccepted,
    /// First cognitive-task reward: fires when a cell transitions out of a
    /// hazard zone after being damaged by it last tick. Rewards the
    /// sensing+motor decision (Hebbian credit assigned to the brain rule
    /// that produced the escape) — not the demographic survival outcome.
    HazardAvoided,
    /// Brief-encounter variant of `EscapedAttack`: fires when a cell took at
    /// least one tick of predation damage (under_attack_streak >= 1) and
    /// then escapes (damage = 0 this tick) WITHOUT crossing the longer
    /// `ESCAPE_STREAK_THRESHOLD` that gates `EscapedAttack`. Captures
    /// near-miss survival of a predator approach. Shares the escape
    /// cooldown with `EscapedAttack` — only one of the two fires per
    /// cooldown window.
    ThreatEscaped,
}

#[derive(Debug, Clone, Copy)]
pub struct RewardEvent {
    pub cell_idx: usize,
    pub kind: RewardKind,
    pub magnitude: f32,
}

/// Sprint 133 keeps `Replace` to match the pre-S133 direct-write semantics
/// (one event per cell per tick → byte-identical). Sprint 135 migrates
/// callers to `SumAndClamp` once damage / bond / mate sources are added.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlushMode {
    Replace,
    SumAndClamp,
}

/// Clamp window applied by `FlushMode::SumAndClamp`. Asymmetric magnitudes
/// per reward kind (attacker `+0.5`, victim `−1.0`, etc.) are allowed to
/// drift the per-cell sum within this window; values outside saturate.
pub const REWARD_CLAMP_MIN: f32 = -2.0;
pub const REWARD_CLAMP_MAX: f32 = 2.0;

#[derive(Debug, Default, Clone)]
pub struct RewardAccumulator {
    events: Vec<RewardEvent>,
}

impl RewardAccumulator {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self {
            events: Vec::with_capacity(cap),
        }
    }

    #[inline]
    pub fn push(&mut self, cell_idx: usize, kind: RewardKind, magnitude: f32) {
        self.events.push(RewardEvent {
            cell_idx,
            kind,
            magnitude,
        });
    }

    pub fn events(&self) -> &[RewardEvent] {
        &self.events
    }

    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    pub fn len(&self) -> usize {
        self.events.len()
    }

    pub fn clear(&mut self) {
        self.events.clear();
    }

    /// Drain queued events into `rewards`. Out-of-bounds `cell_idx` entries
    /// are silently skipped (caller is responsible for sizing `rewards` to
    /// the current cell count).
    pub fn flush_to(&mut self, rewards: &mut [f32], mode: FlushMode) {
        match mode {
            FlushMode::Replace => {
                for ev in self.events.drain(..) {
                    if let Some(slot) = rewards.get_mut(ev.cell_idx) {
                        *slot = ev.magnitude;
                    }
                }
            }
            FlushMode::SumAndClamp => {
                for ev in self.events.drain(..) {
                    if let Some(slot) = rewards.get_mut(ev.cell_idx) {
                        *slot += ev.magnitude;
                    }
                }
                for r in rewards.iter_mut() {
                    *r = r.clamp(REWARD_CLAMP_MIN, REWARD_CLAMP_MAX);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_records_event() {
        let mut acc = RewardAccumulator::new();
        acc.push(3, RewardKind::EatFood, 1.0);
        assert_eq!(acc.len(), 1);
        let ev = &acc.events()[0];
        assert_eq!(ev.cell_idx, 3);
        assert_eq!(ev.kind, RewardKind::EatFood);
        assert_eq!(ev.magnitude, 1.0);
    }

    #[test]
    fn replace_overwrites_per_cell() {
        let mut acc = RewardAccumulator::new();
        acc.push(0, RewardKind::EatFood, 1.0);
        acc.push(0, RewardKind::Novelty, 0.05);
        let mut rewards = vec![0.0; 1];
        acc.flush_to(&mut rewards, FlushMode::Replace);
        assert_eq!(rewards[0], 0.05);
        assert!(acc.is_empty());
    }

    #[test]
    fn sum_accumulates_per_cell() {
        let mut acc = RewardAccumulator::new();
        acc.push(0, RewardKind::EatFood, 1.0);
        acc.push(0, RewardKind::Novelty, 0.05);
        let mut rewards = vec![0.0; 1];
        acc.flush_to(&mut rewards, FlushMode::SumAndClamp);
        assert!((rewards[0] - 1.05).abs() < 1e-6);
    }

    #[test]
    fn sum_clamps_to_window() {
        let mut acc = RewardAccumulator::new();
        for _ in 0..10 {
            acc.push(0, RewardKind::EatFood, 1.0);
        }
        let mut rewards = vec![0.0; 1];
        acc.flush_to(&mut rewards, FlushMode::SumAndClamp);
        assert_eq!(rewards[0], REWARD_CLAMP_MAX);
    }

    #[test]
    fn sum_clamps_negative() {
        let mut acc = RewardAccumulator::new();
        for _ in 0..10 {
            acc.push(0, RewardKind::Damage, -1.0);
        }
        let mut rewards = vec![0.0; 1];
        acc.flush_to(&mut rewards, FlushMode::SumAndClamp);
        assert_eq!(rewards[0], REWARD_CLAMP_MIN);
    }

    #[test]
    fn out_of_bounds_silently_skipped() {
        let mut acc = RewardAccumulator::new();
        acc.push(99, RewardKind::EatFood, 1.0);
        acc.push(1, RewardKind::EatFood, 0.5);
        let mut rewards = vec![0.0; 2];
        acc.flush_to(&mut rewards, FlushMode::Replace);
        assert_eq!(rewards[0], 0.0);
        assert_eq!(rewards[1], 0.5);
    }

    #[test]
    fn flush_drains_events() {
        let mut acc = RewardAccumulator::new();
        acc.push(0, RewardKind::EatFood, 1.0);
        let mut rewards = vec![0.0; 1];
        acc.flush_to(&mut rewards, FlushMode::Replace);
        assert!(acc.is_empty());
        // Second flush is a no-op.
        let mut rewards2 = vec![0.0; 1];
        acc.flush_to(&mut rewards2, FlushMode::Replace);
        assert_eq!(rewards2[0], 0.0);
    }
}
