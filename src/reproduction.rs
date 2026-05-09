use rand::Rng;

use crate::*;

/// Sprint 31: rejection test pro spatial food clustering. Vrací `true` =
/// kandidát zamítnout (zkusit jinou pozici). Probability rejection =
/// `FOOD_REJECTION_STRENGTH × (1 - richness)`. Volá se per uniformně
/// vzorkovaný kandidát; spawn loop drží retry budget (`MAX_SPAWN_ATTEMPTS`),
/// takže clustering jen ladí distribuci, neblokuje úplně.
pub fn reject_food_for_richness(rng: &mut impl Rng, richness: f32) -> bool {
    let r = richness.clamp(0.0, 1.0);
    rng.random::<f32>() < FOOD_REJECTION_STRENGTH * (1.0 - r)
}

/// Sprint 40: greedy O(N²) párování fertile cells na základě 3D distance.
/// Generic přes Idx (usize v headless, Entity v main) — helper dedupuje
/// pairing logiku, která byla pre-refactor identická v obou binárkách.
///
/// Used `std::HashSet` (SipHash, randomized seed) → `FxHashSet` (fixed seed,
/// 5–10× rychlejší hot ops). Iteration order výsledných párů zachován —
/// outer loop indexuje `fertile` přímo, paired set se používá jen na
/// `contains()`, takže RNG/CSV reproducibility intact.
pub fn pair_fertile<I>(
    fertile: &[(I, [f32; 3])],
    mating_r2: f32,
    budget: usize,
    world_half: [f32; 3],
) -> Vec<(I, I)>
where
    I: Copy + Eq + std::hash::Hash,
{
    let mut paired: rustc_hash::FxHashSet<I> =
        rustc_hash::FxHashSet::with_capacity_and_hasher(
            fertile.len(),
            rustc_hash::FxBuildHasher::default(),
        );
    let mut matings: Vec<(I, I)> = Vec::with_capacity(budget.min(fertile.len() / 2));
    for i in 0..fertile.len() {
        if matings.len() >= budget {
            break;
        }
        let (a, pos_a) = fertile[i];
        if paired.contains(&a) {
            continue;
        }
        let mut best: Option<(I, f32)> = None;
        for (j, &(b, pos_b)) in fertile.iter().enumerate() {
            if i == j || paired.contains(&b) {
                continue;
            }
            // Sprint 54: min-image distance pro toroidal world.
            let d = min_image_delta(pos_a, pos_b, world_half);
            let d2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            if d2 <= mating_r2 && best.is_none_or(|(_, bd2)| d2 < bd2) {
                best = Some((b, d2));
            }
        }
        if let Some((b, _)) = best {
            paired.insert(a);
            paired.insert(b);
            matings.push((a, b));
        }
    }
    matings
}

/// Sprint 40: vyrobí dítě z dvou rodičů (immutable refs — parent halving si
/// dělá caller před voláním). Random direction pro startovní heading +
/// crossover + mutate genomu, fresh phenotype z genomu (žádný Lamarckismus),
/// brain stav reset (last_*=0). Energy = a.energy + b.energy (caller už halved).
/// Sprint 66: caller poskytuje `cell_id` (World-level monotonic counter).
/// Sprint 70: vybere parent, do jehož bond clusteru se má spawn dítě. Priorita:
/// 1. Parent s bondy + adhesion_type matchující childovu (= dítě se chytne
///    do existujícího clusteru téhož typu).
/// 2. Pokud ani jeden parent nemá matchující adhesion, ale jeden má bondy,
///    spawn k němu (50 % šance, že se chytne — adhesion mismatch znamená,
///    že to není bond formation kandidát, ale aspoň blízká poloha).
/// 3. Pokud ani jeden parent nemá bondy, vrátí `None` → caller spawne na
///    midpoint (pre-Sprint-70 chování).
pub fn pick_cluster_parent<'a>(
    parent_a: &'a Cell,
    parent_b: &'a Cell,
    child_adhesion_type: u8,
) -> Option<&'a Cell> {
    let a_bonded = parent_a.n_bonds() > 0;
    let b_bonded = parent_b.n_bonds() > 0;
    let a_match = a_bonded && parent_a.genome.adhesion_type == child_adhesion_type;
    let b_match = b_bonded && parent_b.genome.adhesion_type == child_adhesion_type;
    if a_match {
        return Some(parent_a);
    }
    if b_match {
        return Some(parent_b);
    }
    if a_bonded {
        return Some(parent_a);
    }
    if b_bonded {
        return Some(parent_b);
    }
    None
}

pub fn make_mating_child(
    parent_a: &Cell,
    parent_b: &Cell,
    rng: &mut impl Rng,
    cell_id: u64,
) -> Cell {
    let mut child = make_mating_child_no_brain(parent_a, parent_b, rng, cell_id);
    child.genome.brain = Brain::from_cppn(&child.genome.cppn);
    child
}

/// Same RNG sequence as `make_mating_child` but leaves `Cell.genome.brain`
/// as `Brain::zeros()`. Callers that materialise the brain on the GPU
/// (`CppnGpu::dispatch`) skip the CPU `Brain::from_cppn` cost — the
/// dominant per-child reproduction work in `--gpu-full`.
pub fn make_mating_child_no_brain(
    parent_a: &Cell,
    parent_b: &Cell,
    rng: &mut impl Rng,
    cell_id: u64,
) -> Cell {
    // RNG draw order zachovává pre-refactor sekvenci: crossover/mutate FIRST,
    // pak direction. Změna pořadí by porušila CSV identity / reproducibility.
    let child_genome = Genome::crossover(&parent_a.genome, &parent_b.genome, rng)
        .mutate_no_brain(rng, &MUTATION_CONFIG);
    let direction = rng.random_range(0.0..TAU);
    // Sprint 70: cluster-aware jitter. Draw vždycky (i když ho nepoužijeme)
    // — RNG draw order pak zůstane consistent napříč all children, ne jen
    // bonded-parent větví. Z jitter je 0.3× kvůli užšímu z-rangi (±50 vs xy ±960).
    let jitter_x: f32 = rng.random_range(-CLUSTER_SPAWN_RADIUS..CLUSTER_SPAWN_RADIUS);
    let jitter_y: f32 = rng.random_range(-CLUSTER_SPAWN_RADIUS..CLUSTER_SPAWN_RADIUS);
    let jitter_z: f32 = rng.random_range(
        -CLUSTER_SPAWN_RADIUS * 0.3..CLUSTER_SPAWN_RADIUS * 0.3,
    );
    let mid_pos = [
        (parent_a.position[0] + parent_b.position[0]) * 0.5,
        (parent_a.position[1] + parent_b.position[1]) * 0.5,
        (parent_a.position[2] + parent_b.position[2]) * 0.5,
    ];
    // Sprint 70: pokud má kterýkoliv parent bondy + jeho adhesion_type matchuje
    // childovu, spawn dítě uvnitř jeho bond clusteru. Tím dochází k tipping
    // pointu mezi „cells occasionally bond" a „persistent multi-cell
    // organisms" — children rostou bond network místo aby ho jen redukovaly
    // skrz death (Sprint 67.1 + 69 ukázaly net formed-broken < 0).
    //
    // Sprint 79 audit: jitter může produkovat raw pozici mírně mimo world
    // bounds (max |Δ| = CLUSTER_SPAWN_RADIUS = 8 v xy, 2.4 v z). Následný
    // step() pre-tick aplikuje apply_world_bounce → toroidal xy wrap +
    // z reflective clamp. Jeden tick mezi spawn a step může grid lookup
    // vidět out-of-bounds pozici; for_each_in_radius_toroidal interně
    // používá min_image_delta, takže lookup je correct. No bug, just
    // race-tick edge case — accepted.
    let cluster_parent =
        pick_cluster_parent(parent_a, parent_b, child_genome.adhesion_type);
    let pos = match cluster_parent {
        Some(p) => [
            p.position[0] + jitter_x,
            p.position[1] + jitter_y,
            p.position[2] + jitter_z,
        ],
        None => mid_pos,
    };
    let child_phenotype = Phenotype::from_genome(&child_genome);
    Cell {
        position: pos,
        velocity: [
            direction.cos() * child_genome.max_speed,
            direction.sin() * child_genome.max_speed,
            0.0,
        ],
        angular_velocity: 0.0,
        pitch_velocity: 0.0,
        energy: parent_a.energy + parent_b.energy,
        heading: direction,
        pitch: 0.0,
        lineage_id: parent_a.lineage_id,
        lineage_birth_gen: parent_a.lineage_birth_gen,
        last_inputs: [0.0; BRAIN_INPUTS],
        last_hidden: [0.0; BRAIN_HIDDEN],
        last_outputs: [0.0; BRAIN_OUTPUTS],
        last_emit: [0.0; N_PHEROMONE_CHANNELS],
        burst_accum: [0.0; N_PHEROMONE_CHANNELS],
        pooled_hidden: [0.0; BRAIN_HIDDEN],
        damage_accum: 0.0,
        age: 0,
        // Sprint 42: child startuje s plnou cooldown — rodičovská cooldown
        // se nastaví v binárkách po `make_mating_child`, nezasáhne childa.
        reproduce_cooldown_ticks: 0,
        cell_id,
        // Sprint 66: child startuje bez bondů (čistý slate). Bondy se vytvoří
        // podle vlastního chování dítěte, neinheritují se po rodičích.
        bonds: [None; MAX_BONDS_PER_CELL],
        // Sprint 80: cell_state se DĚDÍ (mid-parent + uniform noise σ ≈
        // CELL_STATE_INHERIT_NOISE), na rozdíl od bondů. Tím vzniká
        // fenotypová paměť přes generace bez genetické změny — lineage
        // může držet altruist nebo selfish režim, dokud noise / drift
        // attractor nepřevrátí. Append na konci struct literálu zachovává
        // pre-Sprint-80 RNG draw order.
        cell_state: ((parent_a.cell_state + parent_b.cell_state) * 0.5
            + rng.random_range(-CELL_STATE_INHERIT_NOISE..CELL_STATE_INHERIT_NOISE))
            .clamp(0.0, 1.0),
        last_best_food_d2: f32::MAX,
        phenotype: child_phenotype,
        genome: child_genome,
    }
}

/// Sprint 66: differential-adhesion kernel pro jeden pár (i, j), aplikuje
/// se ze strany i. Vrací `[Δvx, Δvy, Δvz]` přírůstek na velocity_i (před
/// vynásobením `dt`). Same-type → soft attraction (positive coefficient,
/// pulls i toward j). Cross-type → mírná repulze (negative). Zapojí se až
/// **mimo** kontakt (d > pair_r), takže nekoliduje s collision depenetration.
/// Force shape: linearní falloff `(1 - d/R)`, kde R = `ADHESION_RANGE_FACTOR
/// × pair_r`. Mimo R → 0, takže není potřeba další distance gate v hot loop.
///
/// Vstup `delta_ji` je `pos_i - pos_j` (toroidal min-imaged); `dist`
/// je jeho délka (caller už spočítal). `pair_r` je kontaktní vzdálenost
/// (CELL_RADIUS × (radius_i + radius_j)). `same_type` rozlišuje cadherin
/// kompatibilitu.
pub fn adhesion_velocity_delta(
    delta_ji: [f32; 3],
    dist: f32,
    pair_r: f32,
    same_type: bool,
) -> [f32; 3] {
    if dist <= pair_r || dist <= 0.0 {
        return [0.0; 3];
    }
    let range = pair_r * ADHESION_RANGE_FACTOR;
    if dist >= range {
        return [0.0; 3];
    }
    // Linear falloff: 1 at d=pair_r, 0 at d=range.
    let falloff = (range - dist) / (range - pair_r);
    // Coefficient: positive same-type pulls i toward j (negative along delta_ji
    // = pos_i - pos_j). Cross-type negative coefficient flips sign → push apart.
    let coeff = if same_type {
        ADHESION_STRENGTH
    } else {
        ADHESION_STRENGTH * ADHESION_CROSS_TYPE
    };
    let inv_d = 1.0 / dist;
    let nx = delta_ji[0] * inv_d;
    let ny = delta_ji[1] * inv_d;
    let nz = delta_ji[2] * inv_d;
    let mag = -coeff * falloff;
    [mag * nx, mag * ny, mag * nz]
}

/// Sprint 66: spring-bond force pro jeden bond (drží cell_i, ukazuje na j).
/// Vrací `(velocity_delta_i, broken)` — broken=true pokud se bond v tomto
/// ticku trhá (overstretch). Caller zodpovídá za clear bondu. Damping
/// aplikujeme na rel velocity podél spring osy → utlumí oscilace bez
/// over-damping (kritické pro stabilní tissue).
///
/// `delta_ji` = `pos_i - pos_j` (toroidal min-imaged), `dist` jeho délka,
/// `vel_i`, `vel_j` aktuální velocities (caller předal). Vrací delta NA
/// velocity_i, j strana ji aplikuje sama z vlastního Bond slotu (Newton
/// 3rd law symmetric).
pub fn bond_velocity_delta(
    bond: &Bond,
    delta_ji: [f32; 3],
    dist: f32,
    vel_i: [f32; 3],
    vel_j: [f32; 3],
) -> ([f32; 3], bool) {
    let break_len = bond.rest_length * BOND_BREAK_FACTOR;
    if dist > break_len || dist <= f32::EPSILON {
        return ([0.0; 3], true);
    }
    let inv_d = 1.0 / dist;
    let nx = delta_ji[0] * inv_d;
    let ny = delta_ji[1] * inv_d;
    let nz = delta_ji[2] * inv_d;
    // Spring: extension = dist - rest. Pozitivní = roztažení → pulls i toward j
    // (force along -n_ji, kde n_ji ukazuje od j k i). Negativní = stlačení →
    // pushes i away from j (force along +n_ji).
    let extension = dist - bond.rest_length;
    // Sprint 68: per-bond stiffness/damping (uložené při formaci jako mean
    // obou cells' genome values). BOND_STIFFNESS / BOND_DAMPING konstanty
    // jen pro initial draw v Genome::random.
    let spring = -bond.stiffness * extension;
    // Damping: relativní velocity podél normálu. v_rel = v_i - v_j; closing
    // pair má v_rel·n < 0 (pos_i přibližuje k pos_j). Damping force opacuje
    // relative motion → -bond.damping × v_rel_n × n.
    let v_rel_n = (vel_i[0] - vel_j[0]) * nx
        + (vel_i[1] - vel_j[1]) * ny
        + (vel_i[2] - vel_j[2]) * nz;
    let damp = -bond.damping * v_rel_n;
    let mag = spring + damp;
    ([mag * nx, mag * ny, mag * nz], false)
}
