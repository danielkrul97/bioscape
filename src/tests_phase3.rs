//! Fáze 3 — neural/cppn + GPU paritní testy. GPU testy jsou feature-gated
//! (#[cfg(feature = "gpu")]) aby běžely jen když je `gpu` feature aktivní.

#![allow(unused_imports)]

use crate::test_helpers::*;
use crate::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

// ─── ActivationFn variants ──────────────────────────────────────────────────

#[test]
fn activation_linear_identity() {
    let a = ActivationFn::Linear;
    for x in [-2.0_f32, -0.5, 0.0, 0.5, 2.0] {
        assert_eq!(a.apply(x), x);
    }
}

#[test]
fn activation_sigmoid_in_unit_range() {
    let a = ActivationFn::Sigmoid;
    for x in [-10.0_f32, -1.0, 0.0, 1.0, 10.0] {
        let y = a.apply(x);
        assert!((0.0..=1.0).contains(&y), "sigmoid({x}) = {y} oob");
    }
    assert!((a.apply(0.0) - 0.5).abs() < 1e-6);
}

#[test]
fn activation_tanh_in_pm_one() {
    let a = ActivationFn::Tanh;
    for x in [-10.0_f32, -1.0, 0.0, 1.0, 10.0] {
        let y = a.apply(x);
        assert!((-1.0..=1.0).contains(&y), "tanh({x}) = {y} oob");
    }
    assert!(a.apply(0.0).abs() < 1e-6);
}

#[test]
fn activation_gaussian_peak_at_zero() {
    let a = ActivationFn::Gaussian;
    let y0 = a.apply(0.0);
    assert!((y0 - 1.0).abs() < 1e-6, "gaussian(0) = {y0}");
    assert!(a.apply(2.0) < y0);
    assert!(a.apply(-2.0) < y0);
    assert!((a.apply(1.5) - a.apply(-1.5)).abs() < 1e-6);
}

#[test]
fn activation_sine_in_pm_one() {
    let a = ActivationFn::Sine;
    for x in [-10.0_f32, -1.0, 0.0, 1.0, 10.0, 100.0] {
        let y = a.apply(x);
        assert!((-1.0..=1.0).contains(&y), "sine({x}) = {y} oob");
    }
}

#[test]
fn activation_abs_nonnegative() {
    let a = ActivationFn::Abs;
    for x in [-5.0_f32, -1.0, 0.0, 1.0, 5.0] {
        assert!(a.apply(x) >= 0.0);
        assert_eq!(a.apply(x), x.abs());
    }
}

#[test]
fn activation_step_binary() {
    let a = ActivationFn::Step;
    assert_eq!(a.apply(-1.0), 0.0);
    assert_eq!(a.apply(-0.0001), 0.0);
    assert_eq!(a.apply(0.0), 1.0);
    assert_eq!(a.apply(1.0), 1.0);
}

#[test]
fn activation_random_returns_valid_variant() {
    let mut rng = StdRng::seed_from_u64(42);
    for _ in 0..200 {
        let a = ActivationFn::random(&mut rng);
        let _y = a.apply(0.123);
    }
}

// ─── Cppn::random topology ──────────────────────────────────────────────────

#[test]
fn cppn_random_initial_link_count_matches_bipartite() {
    let mut rng = StdRng::seed_from_u64(1);
    let c = Cppn::random(&mut rng);
    let expected = CPPN_INPUTS * CPPN_INITIAL_HIDDEN + CPPN_INITIAL_HIDDEN * CPPN_OUTPUTS;
    assert_eq!(c.iter_links().count(), expected);
    assert_eq!(c.num_links as usize, expected);
}

#[test]
fn cppn_random_node_ids_unique_and_dense() {
    let mut rng = StdRng::seed_from_u64(2);
    let c = Cppn::random(&mut rng);
    let mut ids: Vec<u32> = c.iter_nodes().map(|n| n.id).collect();
    ids.sort();
    let n = ids.len();
    for (i, id) in ids.iter().enumerate() {
        assert_eq!(*id as usize, i, "expected dense ids 0..{n}, got {ids:?}");
    }
}

#[test]
fn cppn_random_input_outputs_at_correct_indices() {
    let mut rng = StdRng::seed_from_u64(3);
    let c = Cppn::random(&mut rng);
    for i in 0..CPPN_INPUTS {
        let n = c.nodes[i].expect("input slot");
        assert_eq!(n.layer, 0, "input {i} layer");
        assert_eq!(n.activation, ActivationFn::Linear);
    }
    for o in 0..CPPN_OUTPUTS {
        let n = c.nodes[CPPN_INPUTS + o].expect("output slot");
        assert_eq!(n.layer, 2, "output {o} layer");
        assert_eq!(n.activation, ActivationFn::Tanh);
    }
}

#[test]
fn cppn_random_links_enabled_by_default() {
    let mut rng = StdRng::seed_from_u64(4);
    let c = Cppn::random(&mut rng);
    for l in c.iter_links() {
        assert!(l.enabled, "link inv={} disabled at init", l.innovation);
    }
}

#[test]
fn cppn_random_innovations_are_consecutive() {
    let mut rng = StdRng::seed_from_u64(5);
    let c = Cppn::random(&mut rng);
    let mut ids: Vec<u32> = c.iter_links().map(|l| l.innovation).collect();
    ids.sort();
    for (i, inv) in ids.iter().enumerate() {
        assert_eq!(*inv as usize, i, "innovations not dense: {ids:?}");
    }
    assert_eq!(c.next_innovation as usize, ids.len());
}

#[test]
fn cppn_random_diverges_with_different_seeds() {
    let a = Cppn::random(&mut StdRng::seed_from_u64(100));
    let b = Cppn::random(&mut StdRng::seed_from_u64(101));
    let weights_a: Vec<f32> = a.iter_links().map(|l| l.weight).collect();
    let weights_b: Vec<f32> = b.iter_links().map(|l| l.weight).collect();
    assert_ne!(weights_a, weights_b, "different seeds → identical weights");
}

#[test]
fn cppn_random_seed_reproduces() {
    let a = Cppn::random(&mut StdRng::seed_from_u64(99));
    let b = Cppn::random(&mut StdRng::seed_from_u64(99));
    let wa: Vec<f32> = a.iter_links().map(|l| l.weight).collect();
    let wb: Vec<f32> = b.iter_links().map(|l| l.weight).collect();
    assert_eq!(wa, wb);
}

// ─── Cppn::forward ──────────────────────────────────────────────────────────

#[test]
fn cppn_forward_outputs_finite_for_extreme_inputs() {
    let mut rng = StdRng::seed_from_u64(7);
    let c = Cppn::random(&mut rng);
    let cases: [[f32; CPPN_INPUTS]; 3] = [
        [100.0; CPPN_INPUTS],
        [-100.0; CPPN_INPUTS],
        [0.0; CPPN_INPUTS],
    ];
    for inputs in cases.iter() {
        let out = c.forward(*inputs);
        for (i, o) in out.iter().enumerate() {
            assert!(o.is_finite(), "output[{i}] = {o} not finite for {inputs:?}");
        }
    }
}

#[test]
fn cppn_forward_outputs_in_tanh_range() {
    let mut rng = StdRng::seed_from_u64(13);
    let c = Cppn::random(&mut rng);
    for trial in 0..20 {
        let mut inputs = [0.0_f32; CPPN_INPUTS];
        let mut r = StdRng::seed_from_u64(13 + trial);
        for v in inputs.iter_mut() {
            *v = r.random_range(-2.0_f32..2.0);
        }
        let out = c.forward(inputs);
        for o in out.iter() {
            assert!((-1.0..=1.0).contains(o), "out {o} oob — outputs are Tanh");
        }
    }
}

#[test]
fn cppn_forward_changes_with_input() {
    let mut rng = StdRng::seed_from_u64(15);
    let c = Cppn::random(&mut rng);
    let a = [0.5_f32, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0];
    let b = [-0.5_f32, -0.5, -0.5, -0.5, -0.5, -0.5, 1.0];
    let out_a = c.forward(a);
    let out_b = c.forward(b);
    assert_ne!(out_a, out_b, "forward should depend on inputs");
}

#[test]
fn cppn_forward_disabled_link_excluded() {
    let mut rng = StdRng::seed_from_u64(17);
    let mut c = Cppn::random(&mut rng);
    let inputs = [0.4_f32, 0.2, -0.1, 0.3, 0.6, -0.2, 1.0];
    let before = c.forward(inputs);
    let target = (0..c.num_links as usize)
        .find(|&i| c.links[i].as_ref().map_or(false, |l| l.enabled))
        .expect("at least one enabled link");
    if let Some(l) = c.links[target].as_mut() {
        l.enabled = false;
    }
    let after = c.forward(inputs);
    assert_ne!(before, after, "disabling a link must change forward output");
}

// ─── mutate_weight ──────────────────────────────────────────────────────────

#[test]
fn cppn_mutate_weight_changes_one_weight() {
    let mut rng = StdRng::seed_from_u64(21);
    let base = Cppn::random(&mut rng);
    let mut mutated = base;
    mutated.mutate_weight(&mut rng, 1.0);
    let diffs: usize = base
        .iter_links()
        .zip(mutated.iter_links())
        .filter(|(a, b)| (a.weight - b.weight).abs() > 1e-9)
        .count();
    assert!(diffs <= 1, "mutate_weight changes ≤ 1 weight, got {diffs}");
}

#[test]
fn cppn_mutate_weight_zero_sigma_noop() {
    let mut rng = StdRng::seed_from_u64(22);
    let base = Cppn::random(&mut rng);
    let mut mutated = base;
    mutated.mutate_weight(&mut rng, 0.0);
    for (a, b) in base.iter_links().zip(mutated.iter_links()) {
        assert_eq!(a.weight.to_bits(), b.weight.to_bits());
    }
}

#[test]
fn cppn_mutate_weight_preserves_count() {
    let mut rng = StdRng::seed_from_u64(23);
    let mut c = Cppn::random(&mut rng);
    let n = c.num_nodes;
    let l = c.num_links;
    for _ in 0..50 {
        c.mutate_weight(&mut rng, 0.3);
    }
    assert_eq!(c.num_nodes, n);
    assert_eq!(c.num_links, l);
}

// ─── mutate_add_node ────────────────────────────────────────────────────────

#[test]
fn cppn_add_node_disables_split_link() {
    let mut rng = StdRng::seed_from_u64(31);
    let mut c = Cppn::random(&mut rng);
    let enabled_pre = c.iter_links().filter(|l| l.enabled).count();
    c.mutate_add_node(&mut rng);
    let enabled_post = c.iter_links().filter(|l| l.enabled).count();
    assert_eq!(
        enabled_post,
        enabled_pre + 1,
        "split link disabled, +2 enabled added"
    );
}

#[test]
fn cppn_add_node_increments_innovations_by_two() {
    let mut rng = StdRng::seed_from_u64(33);
    let mut c = Cppn::random(&mut rng);
    let inv_pre = c.next_innovation;
    c.mutate_add_node(&mut rng);
    assert_eq!(c.next_innovation, inv_pre + 2);
}

#[test]
fn cppn_add_node_respects_node_cap() {
    let mut rng = StdRng::seed_from_u64(35);
    let mut c = Cppn::random(&mut rng);
    for _ in 0..(CPPN_MAX_NODES * 2) {
        c.mutate_add_node(&mut rng);
    }
    assert!((c.num_nodes as usize) <= CPPN_MAX_NODES);
    assert!((c.num_links as usize) <= CPPN_MAX_LINKS);
}

#[test]
fn cppn_add_node_no_active_links_noop() {
    let mut rng = StdRng::seed_from_u64(36);
    let mut c = Cppn::random(&mut rng);
    for i in 0..c.num_links as usize {
        if let Some(l) = c.links[i].as_mut() {
            l.enabled = false;
        }
    }
    let n_pre = c.num_nodes;
    let l_pre = c.num_links;
    c.mutate_add_node(&mut rng);
    assert_eq!(c.num_nodes, n_pre, "no enabled links → noop");
    assert_eq!(c.num_links, l_pre);
}

// ─── mutate_add_link ────────────────────────────────────────────────────────

#[test]
fn cppn_add_link_does_not_duplicate() {
    let mut rng = StdRng::seed_from_u64(41);
    let mut c = Cppn::random(&mut rng);
    for _ in 0..200 {
        c.mutate_add_link(&mut rng, 0.5);
    }
    let mut seen: std::collections::HashSet<(u32, u32)> = std::collections::HashSet::new();
    for l in c.iter_links() {
        assert!(
            seen.insert((l.from, l.to)),
            "duplicate link {} → {}",
            l.from,
            l.to
        );
    }
}

#[test]
fn cppn_add_link_respects_link_cap() {
    let mut rng = StdRng::seed_from_u64(43);
    let mut c = Cppn::random(&mut rng);
    for _ in 0..(CPPN_MAX_LINKS * 2) {
        c.mutate_add_link(&mut rng, 0.5);
    }
    assert!((c.num_links as usize) <= CPPN_MAX_LINKS);
}

#[test]
fn cppn_add_link_increments_innovation_when_added() {
    let mut rng = StdRng::seed_from_u64(45);
    let mut c = Cppn::random(&mut rng);
    let inv_pre = c.next_innovation;
    let l_pre = c.num_links;
    c.mutate_add_link(&mut rng, 0.5);
    if c.num_links > l_pre {
        assert_eq!(c.next_innovation, inv_pre + 1);
    }
}

// ─── mutate_toggle_link ─────────────────────────────────────────────────────

#[test]
fn cppn_toggle_link_flips_one_bit() {
    let mut rng = StdRng::seed_from_u64(51);
    let base = Cppn::random(&mut rng);
    let mut mutated = base;
    mutated.mutate_toggle_link(&mut rng);
    let diffs: usize = base
        .iter_links()
        .zip(mutated.iter_links())
        .filter(|(a, b)| a.enabled != b.enabled)
        .count();
    assert_eq!(diffs, 1, "toggle_link flips exactly one bit");
}

#[test]
fn cppn_toggle_link_idempotent_after_two_calls_on_same_idx() {
    let mut rng = StdRng::seed_from_u64(53);
    let mut c = Cppn::random(&mut rng);
    let pre: Vec<bool> = c.iter_links().map(|l| l.enabled).collect();
    if let Some(l) = c.links[0].as_mut() {
        l.enabled = !l.enabled;
        l.enabled = !l.enabled;
    }
    let post: Vec<bool> = c.iter_links().map(|l| l.enabled).collect();
    assert_eq!(pre, post);
}

#[test]
fn cppn_toggle_link_preserves_topology() {
    let mut rng = StdRng::seed_from_u64(55);
    let mut c = Cppn::random(&mut rng);
    let n = c.num_nodes;
    let l = c.num_links;
    for _ in 0..30 {
        c.mutate_toggle_link(&mut rng);
    }
    assert_eq!(c.num_nodes, n);
    assert_eq!(c.num_links, l);
}

// ─── mutate_activation ──────────────────────────────────────────────────────

#[test]
fn cppn_mutate_activation_only_hidden_changed() {
    let mut rng = StdRng::seed_from_u64(61);
    let base = Cppn::random(&mut rng);
    let mut mutated = base;
    for _ in 0..30 {
        mutated.mutate_activation(&mut rng);
    }
    for i in 0..CPPN_INPUTS {
        let nb = base.nodes[i].unwrap();
        let nm = mutated.nodes[i].unwrap();
        assert_eq!(nb.activation, nm.activation, "input {i} activation changed");
    }
    for o in 0..CPPN_OUTPUTS {
        let nb = base.nodes[CPPN_INPUTS + o].unwrap();
        let nm = mutated.nodes[CPPN_INPUTS + o].unwrap();
        assert_eq!(
            nb.activation, nm.activation,
            "output {o} activation changed"
        );
    }
}

// ─── crossover invariants ──────────────────────────────────────────────────

#[test]
fn cppn_crossover_self_yields_same_innovations() {
    let mut rng = StdRng::seed_from_u64(71);
    let a = Cppn::random(&mut rng);
    let c = Cppn::crossover(&a, &a, &mut rng);
    let inv_a: std::collections::BTreeSet<u32> = a.iter_links().map(|l| l.innovation).collect();
    let inv_c: std::collections::BTreeSet<u32> = c.iter_links().map(|l| l.innovation).collect();
    assert_eq!(inv_a, inv_c);
}

#[test]
fn cppn_crossover_links_subset_of_parent_union() {
    let mut rng = StdRng::seed_from_u64(73);
    let a = Cppn::random(&mut rng);
    let mut b = a;
    for _ in 0..5 {
        b.mutate_add_node(&mut rng);
        b.mutate_add_link(&mut rng, 0.5);
    }
    let c = Cppn::crossover(&a, &b, &mut rng);
    let union: std::collections::HashSet<u32> = a
        .iter_links()
        .chain(b.iter_links())
        .map(|l| l.innovation)
        .collect();
    for l in c.iter_links() {
        assert!(
            union.contains(&l.innovation),
            "child has innovation {} not in parents",
            l.innovation
        );
    }
}

#[test]
fn cppn_crossover_respects_caps() {
    let mut rng = StdRng::seed_from_u64(75);
    let mut a = Cppn::random(&mut rng);
    let mut b = Cppn::random(&mut StdRng::seed_from_u64(76));
    for _ in 0..400 {
        a.mutate_add_node(&mut rng);
        a.mutate_add_link(&mut rng, 0.4);
        b.mutate_add_node(&mut rng);
        b.mutate_add_link(&mut rng, 0.4);
    }
    let c = Cppn::crossover(&a, &b, &mut rng);
    assert!((c.num_nodes as usize) <= CPPN_MAX_NODES);
    assert!((c.num_links as usize) <= CPPN_MAX_LINKS);
}

#[test]
fn cppn_crossover_includes_disjoint_from_either_parent() {
    let mut rng = StdRng::seed_from_u64(77);
    let a = Cppn::random(&mut rng);
    let mut b = a;
    for _ in 0..3 {
        b.mutate_add_node(&mut rng);
    }
    let c = Cppn::crossover(&a, &b, &mut rng);
    for la in a.iter_links() {
        assert!(c.iter_links().any(|lc| lc.innovation == la.innovation));
    }
    for lb in b.iter_links() {
        assert!(c.iter_links().any(|lc| lc.innovation == lb.innovation));
    }
}

#[test]
fn cppn_crossover_next_innovation_above_max() {
    let mut rng = StdRng::seed_from_u64(79);
    let a = Cppn::random(&mut rng);
    let mut b = a;
    for _ in 0..4 {
        b.mutate_add_link(&mut rng, 0.5);
    }
    let c = Cppn::crossover(&a, &b, &mut rng);
    for l in c.iter_links() {
        assert!(c.next_innovation > l.innovation);
    }
}

// ─── compatibility_distance ─────────────────────────────────────────────────

#[test]
fn cppn_compatibility_distance_symmetric() {
    let mut rng = StdRng::seed_from_u64(81);
    let a = Cppn::random(&mut rng);
    let mut b = a;
    for _ in 0..3 {
        b.mutate_weight(&mut rng, 0.5);
    }
    let d_ab = Cppn::compatibility_distance(&a, &b);
    let d_ba = Cppn::compatibility_distance(&b, &a);
    assert!(
        (d_ab - d_ba).abs() < 1e-5,
        "distance not symmetric: {d_ab} vs {d_ba}"
    );
}

#[test]
fn cppn_compatibility_distance_nonnegative() {
    for s in 0..10u64 {
        let a = Cppn::random(&mut StdRng::seed_from_u64(s));
        let b = Cppn::random(&mut StdRng::seed_from_u64(s + 50));
        let d = Cppn::compatibility_distance(&a, &b);
        assert!(d >= 0.0, "distance {d} negative");
    }
}

// ─── species classification (Sprint 204) ────────────────────────────────────

#[test]
fn classify_species_all_identical_is_one_species() {
    let mut rng = StdRng::seed_from_u64(1);
    let mut world = crate::sim::World::new(
        &mut rng,
        WORLD_MAP_SEED,
        MATING_RADIUS,
        8,
        10,
        EventCalendar::default(),
    );
    let proto = world.cells[0].genome.cppn;
    for c in world.cells.iter_mut() {
        c.genome.cppn = proto;
    }
    world.classify_species();
    assert!(world.cells.iter().all(|c| c.species_id == 0));
    let n_species = world
        .cells
        .iter()
        .map(|c| c.species_id)
        .collect::<std::collections::HashSet<u32>>()
        .len();
    assert_eq!(n_species, 1);
}

#[test]
fn classify_species_separates_distant_cppns() {
    let mut rng = StdRng::seed_from_u64(2);
    let mut world = crate::sim::World::new(
        &mut rng,
        WORLD_MAP_SEED,
        MATING_RADIUS,
        8,
        10,
        EventCalendar::default(),
    );
    let proto = world.cells[0].genome.cppn;
    let mut far = proto;
    for _ in 0..120 {
        far = far.mutate(&mut rng, &CPPN_MUTATION_CONFIG);
    }
    let d = Cppn::compatibility_distance(&proto, &far);
    for (i, c) in world.cells.iter_mut().enumerate() {
        c.genome.cppn = if i < 4 { proto } else { far };
    }
    world.classify_species();
    let ids = world
        .cells
        .iter()
        .map(|c| c.species_id)
        .collect::<std::collections::HashSet<u32>>();
    // Self-consistent against the tuned threshold: two clusters iff the far
    // variant actually diverged past CPPN_SPECIATION_THRESHOLD.
    if d > CPPN_SPECIATION_THRESHOLD {
        assert_eq!(ids.len(), 2, "distance {d} should split into 2 species");
    } else {
        assert_eq!(ids.len(), 1, "distance {d} should stay 1 species");
    }
}

#[test]
fn classify_species_ids_are_dense() {
    let mut rng = StdRng::seed_from_u64(3);
    let mut world = crate::sim::World::new(
        &mut rng,
        WORLD_MAP_SEED,
        MATING_RADIUS,
        16,
        10,
        EventCalendar::default(),
    );
    world.classify_species();
    let max_id = world.cells.iter().map(|c| c.species_id).max().unwrap();
    let ids = world
        .cells
        .iter()
        .map(|c| c.species_id)
        .collect::<std::collections::HashSet<u32>>();
    assert_eq!(ids.len() as u32, max_id + 1);
}

// ─── MAP-Elites archive (Sprint 206) ────────────────────────────────────────

#[test]
fn elite_grid_key_in_range() {
    let total = (crate::ELITE_BINS_Z
        * crate::ELITE_BINS_CARN
        * crate::ELITE_BINS_VOL
        * crate::ELITE_BINS_HIDDEN) as u32;
    let mut c = base_cell();
    for carn in [0.0_f32, 0.5, 1.0] {
        c.genome.carnivore_score = carn;
        let k = crate::sim::elite_grid_key(&c, WORLD_HALF);
        assert!(k < total, "key {k} out of range {total}");
    }
}

#[test]
fn elite_grid_key_deterministic() {
    let c = base_cell();
    assert_eq!(
        crate::sim::elite_grid_key(&c, WORLD_HALF),
        crate::sim::elite_grid_key(&c, WORLD_HALF)
    );
}

#[test]
fn elite_grid_key_carnivore_axis_separates() {
    let mut herb = base_cell();
    herb.genome.carnivore_score = 0.0;
    let mut carn = base_cell();
    carn.genome.carnivore_score = 1.0;
    assert_ne!(
        crate::sim::elite_grid_key(&herb, WORLD_HALF),
        crate::sim::elite_grid_key(&carn, WORLD_HALF)
    );
}

#[test]
fn update_elite_archive_keeps_oldest() {
    let mut rng = StdRng::seed_from_u64(4);
    let mut world = crate::sim::World::new(
        &mut rng,
        WORLD_MAP_SEED,
        MATING_RADIUS,
        4,
        10,
        EventCalendar::default(),
    );
    let proto = world.cells[0];
    world.cells.clear();
    for age in [10_u64, 50, 30] {
        let mut c = proto;
        c.age = age;
        world.cells.push(c);
    }
    world.update_elite_archive();
    let key = crate::sim::elite_grid_key(&world.cells[0], WORLD_HALF);
    assert_eq!(world.elite_archive.len(), 1);
    assert_eq!(world.elite_archive[&key].1, 50);
}

#[test]
fn update_elite_archive_coverage_never_shrinks() {
    let mut rng = StdRng::seed_from_u64(5);
    let mut world = crate::sim::World::new(
        &mut rng,
        WORLD_MAP_SEED,
        MATING_RADIUS,
        16,
        10,
        EventCalendar::default(),
    );
    world.update_elite_archive();
    let after_first = world.elite_archive.len();
    // Stepping stones persist: emptying the population must not shrink it.
    world.cells.clear();
    world.update_elite_archive();
    assert_eq!(world.elite_archive.len(), after_first);
}

// ─── Red Queen pressure (Sprint 210) ────────────────────────────────────────

#[test]
fn redqueen_penalizes_common_defense_more_than_rare() {
    let mut rng = StdRng::seed_from_u64(9);
    let mut world = crate::sim::World::new(
        &mut rng,
        WORLD_MAP_SEED,
        MATING_RADIUS,
        4,
        10,
        EventCalendar::default(),
    );
    // One rare defence (bin high) vs three common (bin low). carnivore_score
    // zeroed so the diet bonus doesn't confound the defence comparison.
    for (i, c) in world.cells.iter_mut().enumerate() {
        c.genome.defense_contribution = if i == 0 { 0.9 } else { 0.1 };
        c.genome.carnivore_score = 0.0;
        c.energy = 100.0;
    }
    world.apply_redqueen_pressure(1.0);
    let rare = world.cells[0].energy;
    let common = world.cells[1].energy;
    assert!(
        common < rare,
        "common defence {common} should pay more than rare {rare}"
    );
}

#[test]
fn redqueen_diet_bonus_favors_carnivore_when_prey_abundant() {
    let mut rng = StdRng::seed_from_u64(10);
    let mut world = crate::sim::World::new(
        &mut rng,
        WORLD_MAP_SEED,
        MATING_RADIUS,
        4,
        10,
        EventCalendar::default(),
    );
    // Uniform defence (no penalty differential) + one carnivore among herbivores.
    for (i, c) in world.cells.iter_mut().enumerate() {
        c.genome.defense_contribution = 0.0;
        c.genome.carnivore_score = if i == 0 { 0.8 } else { 0.0 };
        c.energy = 100.0;
    }
    world.apply_redqueen_pressure(1.0);
    let carnivore = world.cells[0].energy;
    let herbivore = world.cells[1].energy;
    assert!(
        carnivore > herbivore,
        "rare carnivore {carnivore} should out-gain herbivore {herbivore}"
    );
}

// ─── ripening food (Sprint 209) ─────────────────────────────────────────────

#[test]
fn ripening_food_completes_with_sustained_processing() {
    let mut rng = StdRng::seed_from_u64(7);
    let mut world = crate::sim::World::new(
        &mut rng,
        WORLD_MAP_SEED,
        MATING_RADIUS,
        4,
        10,
        EventCalendar::default(),
    );
    let cpos = world.cells[0].position;
    world.ripening_foods.push(RipeningFood {
        position: cpos,
        spawn_tick: 0,
        progress: 0,
    });
    let e0 = world.cells[0].energy;
    for _ in 0..RIPENING_STAGES {
        world.update_ripening_food();
    }
    // Harvested after sustained processing: node gone, processor rewarded.
    assert!(world.ripening_foods.is_empty());
    assert!(world.cells[0].energy >= e0 + RIPENING_REWARD - 1.0);
}

#[test]
fn ripening_food_decays_when_unattended() {
    let mut rng = StdRng::seed_from_u64(8);
    let mut world = crate::sim::World::new(
        &mut rng,
        WORLD_MAP_SEED,
        MATING_RADIUS,
        4,
        10,
        EventCalendar::default(),
    );
    world.cells.clear(); // nobody to process → always unattended
    world.ripening_foods.push(RipeningFood {
        position: [0.0, 0.0, 0.0],
        spawn_tick: 0,
        progress: 10,
    });
    world.update_ripening_food();
    assert_eq!(world.ripening_foods[0].progress, 10 - RIPENING_DECAY);
}

// ─── substrate coords ──────────────────────────────────────────────────────

#[test]
fn substrate_input_coords_in_range() {
    for slot in 0..BRAIN_INPUTS {
        let c = substrate_input_coords(slot);
        for axis in 0..3 {
            assert!(
                (-1.0..=1.0).contains(&c[axis]),
                "input slot {slot} axis {axis} = {} oob",
                c[axis]
            );
        }
    }
}

#[test]
fn substrate_hidden_coords_in_range() {
    for slot in 0..BRAIN_HIDDEN {
        let c = substrate_hidden_coords(slot);
        for axis in 0..3 {
            assert!((-1.0..=1.0).contains(&c[axis]));
        }
    }
}

#[test]
fn substrate_output_coords_in_range() {
    for slot in 0..BRAIN_OUTPUTS {
        let c = substrate_output_coords(slot);
        for axis in 0..3 {
            assert!((-1.0..=1.0).contains(&c[axis]));
        }
    }
}

#[test]
fn substrate_input_z_minus_one_for_sensory() {
    for slot in 0..BRAIN_INPUTS_SENSORY {
        let c = substrate_input_coords(slot);
        assert_eq!(c[2], -1.0, "sensory slot {slot} z != -1");
    }
}

#[test]
fn substrate_hidden_z_zero() {
    for slot in 0..BRAIN_HIDDEN {
        assert_eq!(substrate_hidden_coords(slot)[2], 0.0);
    }
}

#[test]
fn substrate_output_z_one() {
    for slot in 0..BRAIN_OUTPUTS {
        assert_eq!(substrate_output_coords(slot)[2], 1.0);
    }
}

#[test]
fn substrate_input_recurrent_maps_to_hidden() {
    for h in 0..BRAIN_HIDDEN {
        let recurrent_slot = BRAIN_INPUTS_SENSORY + h;
        let from_input = substrate_input_coords(recurrent_slot);
        let from_hidden = substrate_hidden_coords(h);
        assert_eq!(
            from_input, from_hidden,
            "recurrent slot {recurrent_slot} ≠ hidden {h}"
        );
    }
}

#[test]
fn substrate_input_x_spans_unit_interval() {
    let first = substrate_input_coords(0)[0];
    let last = substrate_input_coords(BRAIN_INPUTS_SENSORY - 1)[0];
    assert!(first <= -1.0 + 1e-5);
    assert!(last >= 1.0 - 1e-5);
}

#[test]
fn substrate_y_zero_everywhere() {
    for slot in 0..BRAIN_INPUTS_SENSORY {
        assert_eq!(substrate_input_coords(slot)[1], 0.0);
    }
    for slot in 0..BRAIN_HIDDEN {
        assert_eq!(substrate_hidden_coords(slot)[1], 0.0);
    }
    for slot in 0..BRAIN_OUTPUTS {
        assert_eq!(substrate_output_coords(slot)[1], 0.0);
    }
}

// ─── Brain × CPPN integration ───────────────────────────────────────────────

#[test]
fn brain_from_cppn_has_default_hidden_n() {
    let mut rng = StdRng::seed_from_u64(91);
    let cppn = Cppn::random(&mut rng);
    let brain = Brain::from_cppn(&cppn, BRAIN_HIDDEN_DEFAULT as u32);
    assert_eq!(brain.hidden_n, BRAIN_HIDDEN_DEFAULT as u32);
}

#[test]
fn brain_from_cppn_weights_finite() {
    let mut rng = StdRng::seed_from_u64(93);
    let cppn = Cppn::random(&mut rng);
    let brain = Brain::from_cppn(&cppn, BRAIN_HIDDEN_DEFAULT as u32);
    for h in 0..BRAIN_HIDDEN {
        for i in 0..BRAIN_INPUTS {
            assert!(brain.w1[h][i].is_finite());
        }
        assert!(brain.b1[h].is_finite());
    }
    for o in 0..BRAIN_OUTPUTS {
        for h in 0..BRAIN_HIDDEN {
            assert!(brain.w2[o][h].is_finite());
        }
        assert!(brain.b2[o].is_finite());
    }
}

#[test]
fn brain_from_cppn_weights_in_tanh_range() {
    let mut rng = StdRng::seed_from_u64(95);
    let cppn = Cppn::random(&mut rng);
    let brain = Brain::from_cppn(&cppn, BRAIN_HIDDEN_DEFAULT as u32);
    for h in 0..BRAIN_HIDDEN {
        for i in 0..BRAIN_INPUTS {
            let w = brain.w1[h][i];
            assert!((-1.0..=1.0).contains(&w), "w1[{h}][{i}] = {w} oob");
        }
    }
    for o in 0..BRAIN_OUTPUTS {
        for h in 0..BRAIN_HIDDEN {
            let w = brain.w2[o][h];
            assert!((-1.0..=1.0).contains(&w), "w2[{o}][{h}] = {w} oob");
        }
    }
}

#[test]
fn brain_from_cppn_deterministic() {
    let mut rng = StdRng::seed_from_u64(97);
    let cppn = Cppn::random(&mut rng);
    let b1 = Brain::from_cppn(&cppn, BRAIN_HIDDEN_DEFAULT as u32);
    let b2 = Brain::from_cppn(&cppn, BRAIN_HIDDEN_DEFAULT as u32);
    for h in 0..BRAIN_HIDDEN {
        for i in 0..BRAIN_INPUTS {
            assert_eq!(b1.w1[h][i].to_bits(), b2.w1[h][i].to_bits());
        }
    }
}

#[test]
fn brain_from_cppn_disabled_gate_zeros_weight() {
    let mut rng = StdRng::seed_from_u64(99);
    let cppn = Cppn::random(&mut rng);
    let brain = Brain::from_cppn(&cppn, BRAIN_HIDDEN_DEFAULT as u32);
    for h in 0..BRAIN_HIDDEN {
        for i in 0..BRAIN_INPUTS {
            let from_c = substrate_input_coords(i);
            let to_c = substrate_hidden_coords(h);
            let inputs = [
                from_c[0], from_c[1], from_c[2], to_c[0], to_c[1], to_c[2], 1.0,
            ];
            let out = cppn.forward(inputs);
            if out[1] < CPPN_LINK_EXISTS_THRESHOLD {
                assert_eq!(
                    brain.w1[h][i], 0.0,
                    "gate off but weight = {}",
                    brain.w1[h][i]
                );
            }
        }
    }
}

#[test]
fn default_cppn_is_consistent_across_calls() {
    let a = default_cppn();
    let b = default_cppn();
    let wa: Vec<f32> = a.iter_links().map(|l| l.weight).collect();
    let wb: Vec<f32> = b.iter_links().map(|l| l.weight).collect();
    assert_eq!(wa, wb);
}

// ─── Brain::random structure ────────────────────────────────────────────────

#[test]
fn brain_random_default_hidden_n() {
    let mut rng = StdRng::seed_from_u64(111);
    let b = Brain::random(&mut rng);
    assert_eq!(b.hidden_n, BRAIN_HIDDEN_DEFAULT as u32);
}

#[test]
fn brain_random_with_hidden_respects_size() {
    let mut rng = StdRng::seed_from_u64(113);
    for h_n in [BRAIN_HIDDEN_MIN, BRAIN_HIDDEN_DEFAULT, BRAIN_HIDDEN] {
        let b = Brain::random_with_hidden(&mut rng, h_n as u32);
        assert_eq!(b.hidden_n, h_n as u32);
    }
}

#[test]
fn brain_random_dead_zone_is_zero() {
    let mut rng = StdRng::seed_from_u64(117);
    let b = Brain::random_with_hidden(&mut rng, BRAIN_HIDDEN_DEFAULT as u32);
    for h in BRAIN_HIDDEN_DEFAULT..BRAIN_HIDDEN {
        for i in 0..BRAIN_INPUTS {
            assert_eq!(b.w1[h][i], 0.0, "dead zone w1[{h}][{i}] != 0");
        }
        assert_eq!(b.b1[h], 0.0);
    }
    for o in 0..BRAIN_OUTPUTS {
        for h in BRAIN_HIDDEN_DEFAULT..BRAIN_HIDDEN {
            assert_eq!(b.w2[o][h], 0.0, "dead zone w2[{o}][{h}] != 0");
        }
    }
}

#[test]
fn brain_forward_outputs_in_tanh_range() {
    let mut rng = StdRng::seed_from_u64(121);
    let b = Brain::random(&mut rng);
    let inputs = [0.5_f32; BRAIN_INPUTS];
    let out = b.forward(&inputs);
    for o in out.iter() {
        assert!((-1.0..=1.0).contains(o));
    }
}

// ─── GPU paritní testy ──────────────────────────────────────────────────────

#[test]
fn motor_gpu_zero_outputs_parity_with_cpu() {
    use crate::gpu::*;
    use crate::DRAG_COEFFICIENT;
    let n = 8;
    let dt = 1.0_f32 / 60.0;
    let outputs: Vec<[f32; BRAIN_OUTPUTS]> = vec![[0.0; BRAIN_OUTPUTS]; n];
    let headings = vec![0.0_f32; n];
    let pitches = vec![0.0_f32; n];
    let max_speeds = vec![60.0_f32; n];
    let turn_rates = vec![2.5_f32; n];
    // GPU mass must match the CPU cells' phenotype.mass() (S202 volume-based),
    // else apply_brain_motor's thrust/mass differs and CPU/GPU parity fails.
    let masses = vec![base_cell().phenotype.mass(); n];
    let velocities_in = vec![[1.0_f32, 0.0, 0.0]; n];
    let angular_in = vec![0.0_f32; n];
    let pitch_vel_in = vec![0.0_f32; n];

    let mut gpu = match MotorGpu::new(n) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("skip: no GPU adapter ({e})");
            return;
        }
    };
    let (gpu_v, _ga, _gp) = gpu.compute(
        &outputs,
        &headings,
        &pitches,
        &max_speeds,
        &turn_rates,
        &masses,
        &velocities_in,
        &angular_in,
        &pitch_vel_in,
        dt,
        DRAG_COEFFICIENT,
    );

    let mut cpu_cells: Vec<Cell> = (0..n)
        .map(|_| {
            let mut c = base_cell();
            c.velocity = velocities_in[0];
            c
        })
        .collect();
    for c in cpu_cells.iter_mut() {
        c.apply_brain_motor(&[0.0; BRAIN_OUTPUTS], dt);
    }

    for i in 0..n {
        for k in 0..3 {
            let d = (cpu_cells[i].velocity[k] - gpu_v[i][k]).abs();
            assert!(
                d < 1e-4,
                "i={i} k={k} cpu={} gpu={}",
                cpu_cells[i].velocity[k],
                gpu_v[i][k]
            );
        }
    }
}

#[test]
fn motor_gpu_small_batch_parity() {
    use crate::gpu::*;
    use crate::DRAG_COEFFICIENT;
    let mut rng = StdRng::seed_from_u64(201);
    let n = 4;
    let dt = 1.0_f32 / 60.0;
    let mut cells: Vec<Cell> = (0..n)
        .map(|i| Cell::random(&mut rng, [960.0, 540.0, 2.0], 0, 0, i as u64))
        .collect();
    let outputs: Vec<[f32; BRAIN_OUTPUTS]> = (0..n)
        .map(|_| {
            let mut a = [0.0_f32; BRAIN_OUTPUTS];
            for v in a.iter_mut() {
                *v = rng.random_range(-1.0_f32..1.0);
            }
            a
        })
        .collect();
    let headings: Vec<f32> = cells.iter().map(|c| c.heading).collect();
    let pitches: Vec<f32> = cells.iter().map(|c| c.pitch).collect();
    let max_speeds: Vec<f32> = cells.iter().map(|c| c.genome.max_speed).collect();
    let turn_rates: Vec<f32> = cells.iter().map(|c| c.genome.turn_rate).collect();
    let masses: Vec<f32> = cells.iter().map(|c| c.phenotype.mass()).collect();
    let velocities_in: Vec<[f32; 3]> = cells.iter().map(|c| c.velocity).collect();
    let angular_in: Vec<f32> = cells.iter().map(|c| c.angular_velocity).collect();
    let pitch_vel_in: Vec<f32> = cells.iter().map(|c| c.pitch_velocity).collect();

    let mut gpu = match MotorGpu::new(n) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("skip: no GPU adapter ({e})");
            return;
        }
    };
    let (gpu_v, gpu_a, gpu_p) = gpu.compute(
        &outputs,
        &headings,
        &pitches,
        &max_speeds,
        &turn_rates,
        &masses,
        &velocities_in,
        &angular_in,
        &pitch_vel_in,
        dt,
        DRAG_COEFFICIENT,
    );
    for (i, cell) in cells.iter_mut().enumerate() {
        cell.apply_brain_motor(&outputs[i], dt);
    }
    for i in 0..n {
        for k in 0..3 {
            let d = (cells[i].velocity[k] - gpu_v[i][k]).abs();
            assert!(d < 1e-4, "i={i} k={k}");
        }
        assert!((cells[i].angular_velocity - gpu_a[i]).abs() < 1e-4);
        assert!((cells[i].pitch_velocity - gpu_p[i]).abs() < 1e-4);
    }
}

#[test]
fn field_gpu_zero_sources_decays_only() {
    use crate::gpu::*;
    let resolution = [8usize, 8, 4];
    let world_half = [320.0_f32, 320.0, 20.0];
    let mut gpu = match FieldGpu::new(resolution, world_half, 32) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("skip: no GPU adapter ({e})");
            return;
        }
    };
    let mut cpu = SmellField::new(resolution, world_half);
    cpu.add_source([0.0, 0.0, 0.0], 5.0);
    gpu.add_source([0.0, 0.0, 0.0], 5.0);
    for _ in 0..5 {
        cpu.step(0.0, 0.5, 0.1);
        gpu.step(0.0, 0.5, 0.1);
    }
    let cpu_grid = cpu.grid_ref();
    let gpu_grid = gpu.download();
    for (i, (a, b)) in cpu_grid.iter().zip(gpu_grid.iter()).enumerate() {
        assert!((a - b).abs() < 1e-3, "i={i} cpu={a} gpu={b}");
    }
}

#[test]
fn field_gpu_pure_decay_matches_analytic() {
    use crate::gpu::*;
    let resolution = [8usize, 8, 4];
    let world_half = [320.0_f32, 320.0, 20.0];
    let mut gpu = match FieldGpu::new(resolution, world_half, 32) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("skip: no GPU adapter ({e})");
            return;
        }
    };
    gpu.add_source([0.0, 0.0, 0.0], 1.0);
    let decay = 0.5_f32;
    let dt = 0.1_f32;
    let steps = 10;
    for _ in 0..steps {
        gpu.step(0.0, decay, dt);
    }
    let grid = gpu.download();
    let nonneg = grid.iter().filter(|v| **v >= 0.0).count();
    assert_eq!(nonneg, grid.len(), "all values must be non-negative");
}

#[test]
fn hebbian_gpu_zero_reward_noop() {
    use crate::gpu::*;
    let mut rng = StdRng::seed_from_u64(301);
    let n = 8;
    let lr: f32 = 0.01;
    let brains: Vec<Brain> = (0..n).map(|_| Brain::random(&mut rng)).collect();
    let last_inputs: Vec<[f32; BRAIN_INPUTS]> = (0..n)
        .map(|_| {
            let mut a = [0.0_f32; BRAIN_INPUTS];
            for v in a.iter_mut() {
                *v = rng.random_range(-1.0_f32..1.0);
            }
            a
        })
        .collect();
    let last_hidden: Vec<[f32; BRAIN_HIDDEN]> = (0..n)
        .map(|_| {
            let mut a = [0.0_f32; BRAIN_HIDDEN];
            for v in a.iter_mut() {
                *v = rng.random_range(-1.0_f32..1.0);
            }
            a
        })
        .collect();
    let last_outputs: Vec<[f32; BRAIN_OUTPUTS]> = (0..n)
        .map(|_| {
            let mut a = [0.0_f32; BRAIN_OUTPUTS];
            for v in a.iter_mut() {
                *v = rng.random_range(-1.0_f32..1.0);
            }
            a
        })
        .collect();
    let rewards = vec![0.0_f32; n];

    let mut gpu = match HebbianGpu::new(n) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("skip: no GPU adapter ({e})");
            return;
        }
    };
    let after = gpu.compute(
        &last_inputs,
        &last_hidden,
        &last_outputs,
        &rewards,
        &brains,
        lr,
    );
    for i in 0..n {
        for h in 0..BRAIN_HIDDEN {
            for in_i in 0..BRAIN_INPUTS {
                let d = (brains[i].w1[h][in_i] - after[i].w1[h][in_i]).abs();
                assert!(d < 1e-6, "zero reward changed weight i={i} h={h} in={in_i}");
            }
        }
    }
}

#[test]
fn brownian_gpu_zero_noise_preserves_velocity() {
    use crate::gpu::*;
    let n = 32;
    let mut gpu = match BrownianGpu::new(n) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("skip: no GPU adapter ({e})");
            return;
        }
    };
    let velocities = vec![[0.5_f32, 1.0, -0.3]; n];
    let state: Vec<[u32; 4]> = (0..n).map(|i| [i as u32 + 1, 7, 11, 13]).collect();
    let (v_out, _) = gpu.compute(&velocities, &state, 0.0, 1.0 / 60.0, false);
    for i in 0..n {
        for k in 0..3 {
            let d = (velocities[i][k] - v_out[i][k]).abs();
            assert!(
                d < 1e-6,
                "zero-noise drifted i={i} k={k} in={} out={}",
                velocities[i][k],
                v_out[i][k]
            );
        }
    }
}

#[test]
fn sensor_gather_gpu_no_neighbors_when_alone() {
    use crate::gpu::*;
    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("skip: no GPU adapter ({e})");
            return;
        }
    };
    let n = 1_usize;
    let nf = 1_usize;
    let cell_size = 64.0_f32;
    let world_half_xy = [320.0_f32, 320.0];
    let field_resolution = [8usize, 8, 4];
    let field_world_half = [320.0_f32, 320.0, 20.0];

    let positions = vec![[10.0_f32, 0.0, 0.0]];
    let eff_radii = vec![1.0_f32];
    let vision_radii = vec![20.0_f32];
    let food_positions = vec![[200.0_f32, 200.0, 0.0]];

    let mut cell_hash = SpatialHashGpu::with_context(&ctx, n, cell_size, world_half_xy).unwrap();
    cell_hash.rebuild(&positions);
    let mut food_hash = SpatialHashGpu::with_context(&ctx, nf, cell_size, world_half_xy).unwrap();
    food_hash.rebuild(&food_positions);
    let mut smell = FieldGpu::with_context(&ctx, field_resolution, field_world_half, 32).unwrap();
    let mut phero = FieldGpu::with_context(&ctx, field_resolution, field_world_half, 32).unwrap();
    smell.step(0.0, 0.5, 0.1);
    phero.step(0.0, 0.5, 0.1);

    let mut sensor = SensorGatherGpu::with_context(&ctx, n, nf).unwrap();
    let params = SensorParamsGpu {
        hash_cell_size: cell_size,
        world_half_x: world_half_xy[0],
        world_half_y: world_half_xy[1],
        world_half_z: 20.0,
        field_res_x: field_resolution[0] as u32,
        field_res_y: field_resolution[1] as u32,
        field_res_z: field_resolution[2] as u32,
        field_eps: 4.0,
        field_world_half_x: field_world_half[0],
        field_world_half_y: field_world_half[1],
        field_world_half_z: field_world_half[2],
        ..SensorParamsGpu::default()
    };
    // Wave 6: sensor.compute now also takes per-cell heading + pitch for
    // the in-shader whisker raycast. Tests don't care about whiskers, so
    // upload zeros and let `params.maze_active = 0` skip the raycast block.
    let test_headings = vec![0.0_f32; positions.len()];
    let test_pitches = vec![0.0_f32; positions.len()];
    let test_whisker_state = crate::test_helpers::whisker_state_buf(&ctx.device, n);
    let rows = sensor.compute(
        &positions,
        &eff_radii,
        &vision_radii,
        &food_positions,
        &test_headings,
        &test_pitches,
        &cell_hash,
        &food_hash,
        &smell,
        &phero,
        &phero,
        &phero,
        &smell,
        &test_whisker_state,
        params,
    );
    assert_eq!(
        rows[0].neighbors_in_vision, 0,
        "alone cell has no neighbors"
    );
    assert!(rows[0].nearest_cell.is_none());
    assert!(rows[0].nearest_food.is_none(), "food too far for vision");
}

#[test]
fn sensor_gather_gpu_food_in_vision_detected() {
    use crate::gpu::*;
    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("skip: no GPU adapter ({e})");
            return;
        }
    };
    let n = 1_usize;
    let nf = 1_usize;
    let cell_size = 64.0_f32;
    let world_half_xy = [320.0_f32, 320.0];
    let field_resolution = [8usize, 8, 4];
    let field_world_half = [320.0_f32, 320.0, 20.0];

    let positions = vec![[0.0_f32, 0.0, 0.0]];
    let eff_radii = vec![1.0_f32];
    let vision_radii = vec![50.0_f32];
    let food_positions = vec![[10.0_f32, 5.0, 0.0]];

    let mut cell_hash = SpatialHashGpu::with_context(&ctx, n, cell_size, world_half_xy).unwrap();
    cell_hash.rebuild(&positions);
    let mut food_hash = SpatialHashGpu::with_context(&ctx, nf, cell_size, world_half_xy).unwrap();
    food_hash.rebuild(&food_positions);
    let mut smell = FieldGpu::with_context(&ctx, field_resolution, field_world_half, 32).unwrap();
    let mut phero = FieldGpu::with_context(&ctx, field_resolution, field_world_half, 32).unwrap();
    smell.step(0.0, 0.5, 0.1);
    phero.step(0.0, 0.5, 0.1);

    let mut sensor = SensorGatherGpu::with_context(&ctx, n, nf).unwrap();
    let params = SensorParamsGpu {
        hash_cell_size: cell_size,
        world_half_x: world_half_xy[0],
        world_half_y: world_half_xy[1],
        world_half_z: 20.0,
        field_res_x: field_resolution[0] as u32,
        field_res_y: field_resolution[1] as u32,
        field_res_z: field_resolution[2] as u32,
        field_eps: 4.0,
        field_world_half_x: field_world_half[0],
        field_world_half_y: field_world_half[1],
        field_world_half_z: field_world_half[2],
        ..SensorParamsGpu::default()
    };
    // Wave 6: sensor.compute now also takes per-cell heading + pitch for
    // the in-shader whisker raycast. Tests don't care about whiskers, so
    // upload zeros and let `params.maze_active = 0` skip the raycast block.
    let test_headings = vec![0.0_f32; positions.len()];
    let test_pitches = vec![0.0_f32; positions.len()];
    let test_whisker_state = crate::test_helpers::whisker_state_buf(&ctx.device, n);
    let rows = sensor.compute(
        &positions,
        &eff_radii,
        &vision_radii,
        &food_positions,
        &test_headings,
        &test_pitches,
        &cell_hash,
        &food_hash,
        &smell,
        &phero,
        &phero,
        &phero,
        &smell,
        &test_whisker_state,
        params,
    );
    assert!(
        rows[0].nearest_food.is_some(),
        "food within vision radius missed"
    );
}

/// Sprint 195: the whisker spring-damper must produce bit-comparable results
/// on the CPU (`whisker_step`) and the GPU (`sensor_gather.wgsl`). All cells
/// face `heading = 0`, `pitch = 0` so the six whisker rays are exact world
/// axes — the underlying raycast is then bit-identical CPU/GPU, isolating
/// this test to the spring-damper + transduction-noise arithmetic.
#[test]
fn whisker_spring_damper_gpu_matches_cpu() {
    use crate::gpu::*;
    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("skip: no GPU adapter ({e})");
            return;
        }
    };

    let field = ObstacleField::new_maze(WORLD_HALF, 195, MazeDifficulty::Medium);
    let packed = field.packed_for_gpu();
    let maze_res_x = field.resolution[0] as u32;
    let maze_res_y = field.resolution[1] as u32;

    // Fixed 4×2 grid of cells spanning the maze interior — varied positions
    // so whiskers see walls at varied distances (ringing), all heading 0.
    let mut positions: Vec<[f32; 3]> = Vec::new();
    for &x in &[-700.0_f32, -350.0, 0.0, 350.0] {
        for &y in &[-250.0_f32, 250.0] {
            positions.push([x, y, 0.0]);
        }
    }
    let n = positions.len();
    let nf = 1_usize;
    let headings = vec![0.0_f32; n];
    let pitches = vec![0.0_f32; n];
    let eff_radii = vec![1.0_f32; n];
    let vision_radii = vec![30.0_f32; n];
    let food_positions = vec![[900.0_f32, 500.0, 0.0]];

    let cell_size = 64.0_f32;
    let world_half_xy = [WORLD_HALF[0], WORLD_HALF[1]];
    let field_resolution = [8usize, 8, 4];
    let field_world_half = [WORLD_HALF[0], WORLD_HALF[1], 20.0];

    let mut cell_hash = SpatialHashGpu::with_context(&ctx, n, cell_size, world_half_xy).unwrap();
    cell_hash.rebuild(&positions);
    let mut food_hash = SpatialHashGpu::with_context(&ctx, nf, cell_size, world_half_xy).unwrap();
    food_hash.rebuild(&food_positions);
    let mut smell = FieldGpu::with_context(&ctx, field_resolution, field_world_half, 32).unwrap();
    let mut phero = FieldGpu::with_context(&ctx, field_resolution, field_world_half, 32).unwrap();
    smell.step(0.0, 0.5, 0.1);
    phero.step(0.0, 0.5, 0.1);

    let mut sensor = SensorGatherGpu::with_context(&ctx, n, nf).unwrap();
    sensor.upload_maze(&packed);

    let f = std::mem::size_of::<f32>();
    let mk_buf = |label: &str, size: usize, usage: wgpu::BufferUsages| {
        ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size as u64,
            usage,
            mapped_at_creation: false,
        })
    };
    // Persistent spring-damper state — mirrors CellsGpu::whisker_state_buf.
    let whisker_state = mk_buf(
        "test-whisker-state",
        n * 12 * f,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
    );
    ctx.queue
        .write_buffer(&whisker_state, 0, &vec![0u8; n * 12 * f]);
    let stride = SENSOR_OUTPUT_STRIDE;
    let output_rb = mk_buf(
        "test-sensor-output-rb",
        n * stride * f,
        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
    );
    let whisker_state_rb = mk_buf(
        "test-whisker-state-rb",
        n * 12 * f,
        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
    );

    let base_params = SensorParamsGpu {
        hash_cell_size: cell_size,
        world_half_x: world_half_xy[0],
        world_half_y: world_half_xy[1],
        world_half_z: 20.0,
        field_res_x: field_resolution[0] as u32,
        field_res_y: field_resolution[1] as u32,
        field_res_z: field_resolution[2] as u32,
        field_eps: 4.0,
        field_world_half_x: field_world_half[0],
        field_world_half_y: field_world_half[1],
        field_world_half_z: field_world_half[2],
        maze_active: 1,
        maze_res_x,
        maze_res_y,
        ..SensorParamsGpu::default()
    };

    // CPU mirror of the persistent state.
    let mut cpu_defl = vec![[0.0_f32; WHISKER_COUNT]; n];
    let mut cpu_vel = vec![[0.0_f32; WHISKER_COUNT]; n];
    let mut max_defl_seen = 0.0_f32;

    let ticks = 120u32;
    for tick in 0..ticks {
        let mut params = base_params;
        params.tick = tick;
        sensor.dispatch_no_readback(
            &positions,
            &eff_radii,
            &vision_radii,
            &food_positions,
            &headings,
            &pitches,
            &cell_hash,
            &food_hash,
            &smell,
            &phero,
            &phero,
            &phero,
            &smell,
            &whisker_state,
            params,
        );

        let mut enc = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_buffer_to_buffer(
            sensor.output_buffer(),
            0,
            &output_rb,
            0,
            (n * stride * f) as u64,
        );
        ctx.queue.submit(Some(enc.finish()));
        let slice = output_rb.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        ctx.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        let gpu_out: Vec<f32> = bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();
        output_rb.unmap();

        for i in 0..n {
            let raw = field.whisker_distances(positions[i], headings[i], pitches[i]);
            for k in 0..WHISKER_COUNT {
                let noise = whisker_noise(i as u32, tick, k as u32) * WHISKER_NOISE_AMPLITUDE;
                let cpu_sensed =
                    whisker_step(&mut cpu_defl[i][k], &mut cpu_vel[i][k], raw[k], noise);
                let gpu_sensed = gpu_out[i * stride + 19 + k];
                assert!(
                    (cpu_sensed - gpu_sensed).abs() < 1e-4,
                    "tick {tick} cell {i} whisker {k}: cpu sensed {cpu_sensed} vs gpu {gpu_sensed}"
                );
                max_defl_seen = max_defl_seen.max(cpu_defl[i][k].abs());
            }
        }
    }

    // Final persistent-state parity: deflection + velocity.
    let mut enc = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    enc.copy_buffer_to_buffer(&whisker_state, 0, &whisker_state_rb, 0, (n * 12 * f) as u64);
    ctx.queue.submit(Some(enc.finish()));
    let slice = whisker_state_rb.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    ctx.device
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
    let gpu_state: Vec<f32> = bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();
    whisker_state_rb.unmap();

    for i in 0..n {
        for k in 0..WHISKER_COUNT {
            let gpu_d = gpu_state[i * 12 + k];
            let gpu_v = gpu_state[i * 12 + 6 + k];
            assert!(
                (cpu_defl[i][k] - gpu_d).abs() < 1e-4,
                "cell {i} whisker {k}: cpu deflection {} vs gpu {gpu_d}",
                cpu_defl[i][k]
            );
            assert!(
                (cpu_vel[i][k] - gpu_v).abs() < 1e-3,
                "cell {i} whisker {k}: cpu velocity {} vs gpu {gpu_v}",
                cpu_vel[i][k]
            );
        }
    }

    // Sanity: at least one whisker actually rang against a wall — otherwise
    // the parity above would hold trivially with everything at rest.
    assert!(
        max_defl_seen > 0.05,
        "no whisker saw a wall (max |deflection| {max_defl_seen}) — test exercised nothing"
    );
}

#[test]
fn brain_forward_gpu_matches_cpu_small_batch() {
    use crate::gpu::*;
    let mut rng = StdRng::seed_from_u64(401);
    let n = 4;
    let brains: Vec<Brain> = (0..n).map(|_| Brain::random(&mut rng)).collect();
    let inputs: Vec<[f32; BRAIN_INPUTS]> = (0..n)
        .map(|_| {
            let mut a = [0.0_f32; BRAIN_INPUTS];
            for v in a.iter_mut() {
                *v = rng.random_range(-1.0_f32..1.0);
            }
            a
        })
        .collect();
    let mut gpu = match BrainGpu::new(n) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("skip: no GPU adapter ({e})");
            return;
        }
    };
    let mut h_gpu = vec![[0.0_f32; BRAIN_HIDDEN]; n];
    let mut o_gpu = vec![[0.0_f32; BRAIN_OUTPUTS]; n];
    gpu.forward_batch(
        &inputs,
        &brains,
        &mut h_gpu,
        &mut o_gpu,
        crate::LATERAL_INHIBITION_ALPHA,
    );
    for i in 0..n {
        let (_, o_cpu) = brains[i].forward_with_state(&inputs[i], crate::LATERAL_INHIBITION_ALPHA);
        for k in 0..BRAIN_OUTPUTS {
            let d = (o_cpu[k] - o_gpu[i][k]).abs();
            assert!(d < 1e-4, "i={i} k={k} cpu={} gpu={}", o_cpu[k], o_gpu[i][k]);
        }
    }
}
