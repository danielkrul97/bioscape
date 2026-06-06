//! Sprint 203–212 "complexity ratchet" tuneables. Single home for the
//! decade's selection-shaping constants — speciation, fitness sharing,
//! MAP-Elites, patchy food, metabolism, coevolution, novelty — so they live
//! together rather than scattered across `physics.rs` / `world.rs`.

/// Sprint 204: CPPN compatibility distance below which two cells are deemed
/// the same species. NEAT-style threshold over the `δ = E/N + D/N + 0.4·W̄`
/// metric (`Cppn::compatibility_distance`). An educated guess — the goal is
/// tens of species at steady state (not 1, not hundreds). Tuned in Sprint 212
/// against the `species_count` CSV column; raise it to merge species, lower it
/// to split them.
// Sprint 212 tuning: 1.0 → 0.6. The 100-gen validation held at only 2 species
// (genealogical collapse to ~4 lineages); a finer threshold splits species
// earlier so the fitness-sharing gate protects more nascent niches.
pub const CPPN_SPECIATION_THRESHOLD: f32 = 0.6;

/// Sprint 205: fitness-sharing pressure on the reproduction energy threshold.
/// A cell in a species holding fraction `sf` of the population reproduces only
/// once its energy clears `reproduce_at_energy × (1 + PRESSURE × sf²)`. A
/// crowded species pays more; a small innovative species pays ~baseline,
/// protecting structural innovation (NEAT's core trick) and breaking the
/// passive-monoculture attractor that the 193–202 scarcity pressure could not
/// dislodge. Layered on top of the lineage-frequency term
/// (`LINEAGE_DIVERSITY_ALPHA`); 0.0 reproduces pre-S205 behaviour byte-for-byte.
/// Kept low to start — over-sharing freezes adaptation. Tuned in Sprint 212.
// Sprint 212 tuning: 1.0 → 1.5. Stronger protection of small/innovative species
// after the 100-gen run showed bounded-but-unreversed convergence.
pub const FITNESS_SHARE_PRESSURE: f32 = 1.5;

/// Sprint 206: MAP-Elites behavioural archive grid resolution, one factor per
/// descriptor axis. The archive keeps the longest-lived genome seen in each
/// `(z_norm × carnivore × body_volume × hidden_n)` cell, preserving stepping
/// stones (incl. simpler ancestors of later complexity) that selection would
/// otherwise discard. `hidden_n` is deliberately an axis — we want to protect
/// *complexity* as a niche dimension, not just morphology.
pub const ELITE_BINS_Z: usize = 4;
pub const ELITE_BINS_CARN: usize = 4;
pub const ELITE_BINS_VOL: usize = 4;
pub const ELITE_BINS_HIDDEN: usize = 4;

/// Body volume above which the volume descriptor axis saturates to its top
/// bin. A 1×1×1 cell has volume 1; large cells reach into the single digits.
pub const ELITE_VOL_CAP: f32 = 8.0;

/// Sprint 206: fraction of each generation's births that adopt a random
/// archived elite genome (mutated) instead of the crossover result —
/// stepping-stone reinjection. Routed through the normal birth path so the GPU
/// CPPN dispatch materialises the replaced genome identically. Kept low so
/// reinjection seeds diversity without flooding the gene pool. 0.0 disables it
/// (births stay pure crossover). Tuned in Sprint 212.
pub const ELITE_REINJECT_FRACTION: f32 = 0.05;

// Sprint 207: patchy resource field. Food spawn is biased by a slowly drifting
// analytic spatial pattern — fertile enclaves and barren deserts — so recurrent
// brains finally gain a reason to remember and re-find resource locations
// (directed exploration + memory start to pay). Computed in food_spawn.wgsl;
// food spawn is GPU-only, so there is no CPU parity mirror.

/// Strength of desert rejection: 0 = uniform (pre-S207), 1 = barren regions
/// spawn no food at all. Tuned in Sprint 212 against exploration / entropy.
pub const FOOD_PATCH_CONTRAST: f32 = 0.7;

/// Spatial frequency of the patch pattern (radians per world unit). ~0.006
/// gives a handful of fertile/barren regions across the world's xy extent.
pub const FOOD_PATCH_SCALE: f32 = 0.006;

/// Drift speed of the pattern (radians per generation) — patches migrate so a
/// fixed learned map decays, rewarding adaptive re-foraging over memorisation.
pub const FOOD_PATCH_DRIFT: f32 = 0.02;

/// Sprint 211: behavioural-novelty reproduction bonus. A cell whose strategy
/// (the `elite_grid_key` descriptor) is rare in the population reproduces at a
/// reduced energy threshold (`1 - weight·(1 - bin_freq)`), keeping exploration
/// alive even when the population has otherwise converged (anti-convergence;
/// doc 07 novelty search). Complementary to S205 species sharing — that protects
/// genetic innovation, this protects strategy exploration. 0.0 disables it.
///
/// S213+ tuning: 0.3 → 0.8. This is the lever that finally *reverses* (not just
/// bounds) the genealogical/behavioural-entropy collapse the 203–212 decade
/// left open. The collapse is competitive exclusion in a well-mixed world;
/// reproduction-side novelty is the one counter that works — unlike restricting
/// gene flow (assortative mating only slowed the CPPN ratchet) or an energy-side
/// crowding drain (which kills the marginal niches it means to protect). The
/// response is sharply non-monotonic: 0.6 is no better than 0.3, but ~0.8 lifts
/// a rare strategy's threshold discount to ~80%, enough to escape the exclusion
/// attractor. Validated 5×60-gen cross-seed (0/5 extinction): behavioural
/// entropy holds ~0.5 instead of collapsing to ~0.18, species ×2, and — because
/// the maintained diversity feeds Fisher-Muller crossover — the CPPN ratchet
/// *accelerates* ~4×. Drives population to `MAX_POPULATION` and CPPN toward
/// `CPPN_MAX_NODES`, so those caps become the new binding constraints.
pub const NOVELTY_BONUS_WEIGHT: f32 = 0.8;

// Sprint 208: utilization-weighted brain metabolism. There was no brain cost
// before this sprint, so `hidden_n` drifted neutrally; this adds a *use-based*
// cost so a neuron that carries signal pays a little while a silent (unused)
// neuron is free. Combined with S207 demand it gives complexity a cost/benefit
// gradient — used computation that earns food is net-positive (ratchet up),
// bloat that does nothing is pruned, but adding a quiet neuron to try is nearly
// free (no Avida-style collapse to the minimum). Applied CPU-side after the GPU
// step — energy and `last_hidden` round-trip to
// the CPU every tick, so no shader is involved.

/// Dead zone: hidden activations with |a| ≤ this cost nothing (a neuron is
/// "silent"). `last_hidden` is a tanh output in [-1, 1].
pub const BRAIN_UTIL_EPSILON: f32 = 0.1;

/// Energy per unit of brain utilization per second. Kept small so the brain
/// cost modulates rather than dominates host metabolism. 0.0 disables it
/// (byte-identical to pre-S208). Tuned in Sprint 212 against `hidden_n_avg`.
pub const BRAIN_COST_PER_UTIL: f32 = 0.05;

// Sprint 209: single-cell multi-step "ripening" food (see `RipeningFood`). A
// node a cell must process for several consecutive ticks before it pays out,
// rewarding a persistent goal-held-in-recurrence policy. Separate CPU subsystem
// mirroring `CoopFood`; the GPU eat path is untouched.

/// Per-tick Bernoulli spawn probability for a ripening node.
pub const RIPENING_FOOD_SPAWN_RATE_PER_TICK: f32 = 0.03;
/// Cap on simultaneously live ripening nodes.
pub const RIPENING_FOOD_MAX_CONCURRENT: usize = 12;
/// Consecutive processing ticks required to harvest (~0.5 s at 60 Hz).
pub const RIPENING_STAGES: u32 = 30;
/// Processing radius — a cell within this distance advances the node.
pub const RIPENING_RADIUS: f32 = 30.0;
/// Reward to the cell that completes a node. High enough to justify the wait
/// (4× plant baseline, matching the coop-food reward scale).
pub const RIPENING_REWARD: f32 = 80.0;
/// Progress lost per tick while unattended — makes completion require
/// persistence, not scattered fly-bys.
pub const RIPENING_DECAY: u32 = 1;
/// Lifetime (ticks since spawn) before an un-harvested node despawns.
pub const RIPENING_WINDOW_TICKS: u64 = 240;

// Sprint 210: coevolutionary arms race via negative frequency-dependent
// selection on the defence phenotype. A defence strategy that is common in the
// population is disadvantaged (as if predators had adapted to it), so whichever
// defence dominates becomes a liability — driving perpetual cycling instead of
// convergence to a fixed equilibrium (doc 07 Red Queen). Applied CPU-side,
// which keeps it out of the GPU predate path.

/// Number of bins the `defense_contribution` axis is partitioned into for the
/// frequency histogram.
pub const REDQUEEN_PHENO_BINS: usize = 8;

/// Energy/sec penalty at full crowding (a phenotype bin holding the whole
/// population). Scales linearly with the bin's population fraction, so rare
/// defences pay almost nothing. Kept small — a modulator, not life-support.
/// 0.0 disables it. Tuned in Sprint 212 against defence-trait oscillation.
pub const REDQUEEN_FREQ_STRENGTH: f32 = 0.3;

/// Sprint 212 tuning: diet-rarity bonus. The 100-gen validation showed the
/// predation niche collapsing (`carnivore_avg` 0.25 → 0.04), starving the S210
/// arms race of predators. A cell gains energy/sec ∝ its `carnivore_score` ×
/// the herbivore (prey) fraction — so carnivory pays most when prey is abundant
/// and self-limits as predators grow (negative frequency dependence on diet,
/// the complement of the defence penalty). Sustains a predator niche instead of
/// letting it go extinct. 0.0 disables it. Applied in `apply_redqueen_pressure`.
///
/// Sprint 212 tuning: 1.5 → 6.0. At 1.5 the bonus (∝ the already-collapsed
/// carnivore_score) was too weak to bootstrap the niche — carnivory still
/// crashed by gen 20. 6.0 quadruples the selection gradient on carnivore_score;
/// still self-limiting (the prey_frac factor shrinks the bonus as predators
/// crowd in), so worst case it oscillates rather than runs away.
pub const CARNIVORE_RARITY_BONUS: f32 = 6.0;

/// Sprint 212: `carnivore_score` below which a cell counts as "prey"
/// (herbivore-leaning) when measuring prey abundance for the diet-rarity bonus
/// above. Midpoint of the `[0, 1]` diet axis.
pub const PREY_CARNIVORE_SPLIT: f32 = 0.5;
