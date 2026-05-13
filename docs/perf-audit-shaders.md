# Perf audit — WGSL shadery

Statická analýza všech shaderů v `shaders/`. **Není to profil** — pořadí dopadu na FPS je odhad ze statického kódu. Doporučení: před refaktorem změř `wgpu` timestamp queries / renderdoc na 1k cells × 30 gen, abys ověřil, který pass dominuje.

Položky jsou seřazené **podle jednoduchosti fixu** — od triviálních (změna pár řádků) po architektonické refaktory.

---

## Sumarizace

Všechny audit items vyřešeny (S185-S190).

---

## Metodologie

Audit pokrývá všechny shadery v `shaders/`:
brain_forward, brain_forward_izhikevich, brownian, cell_stats, collision, cppn_from_cppn, eat_food, excitability, field_deposit, field_diffuse, food_spawn, hebbian, hebbian_apply_reward, hebbian_step, motor, populate_inputs, predate, sensor_gather, spatial_hash, stdp_apply, stdp_encode_pre, stdp_step, step, synaptic_scale.

**Co bylo zkoumáno:**
- Memory access patterns (atomic vs plain, coalescing, scattered loads)
- Branch divergence v hot loops
- Redundant computation (per-thread duplicate work, uniform values per-cell)
- Function-private arrays s dynamic indexingem (likely spill do DRAM)
- Aritmetické idiomy (sqrt+div vs inverseSqrt, Kahan vs plain)
- Dispatch geometry / workgroup occupancy

**Co NEBYLO zkoumáno** (vyžaduje profilování):
- Skutečný čas per-pipeline na konkrétním GPU
- Memory bandwidth saturation
- Subgroup-level divergence (vendor-specific)
- L1/L2 cache hit rates
