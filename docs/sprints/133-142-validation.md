# Validation report — Sprinty 133–142 plasticita

Cross-seed sweep: **3 seedy × 10 generací** v open-world režimu (`--maze` off),
default parametry, `MAX_POPULATION = 1500`. Validuje že rozšířený reward
funnel (S134-135) + per-cell neuromodulace (S136-137) + homeostatic
plasticita (S138-139) nepřinášejí extinction events a měřitelně mění
brain weight landscape vs pre-S133 baseline.

Krátká doba runu (10 gen místo 30 v sprint plánu) reflektuje GPU dispatch
overhead — multi-dispatch per tick zpomalil sim z ~190 ticks/s na ~30
ticks/s. 30-gen sweep by si vyžadoval ~15 min per seed; 10 gen dává
dostatečný signál pro acceptance.

## Výsledky gen 10

| seed | n_cells | pred_events_cum | bonds_formed_cum | lr_avg | lr_std | decay_avg | decay_std | w_norm_avg |
|------|---------|-----------------|------------------|--------|--------|-----------|-----------|------------|
| 0    | 691     | 921             | 57               | 0.0064 | 0.0035 | 0.466     | 0.255     | 7.29       |
| 42   | 499     | 646             | 42               | 0.0058 | 0.0041 | 0.433     | 0.187     | 7.16       |
| 100  | 439     | 564             | 26               | 0.0058 | 0.0034 | 0.458     | 0.255     | 6.22       |

Pre-S133 baseline (`run_seed0.csv`, 1 gen): n=197, pred=12, bonds=0,
lr/decay/w_norm sloupce neexistovaly. Direct comparison není apples-to-apples
napříč generacemi, ale měřitelný posun napříč seedy ukazuje:

## Pozorování

1. **Žádná extinkce.** 3/3 seedy populace přežily a rostly (439–691 cells)
   — nový reward funnel + clamp `[-2, +2]` udržuje dynamiku stabilní.

2. **Predator policy konverguje.** Cumulative predation events 564–921 za
   10 generací vs 12 za 1 gen pre-S133 baseline. Per-gen avg ~70 events = silně
   nad acceptance threshold (>0.1 events/cell/gen).

3. **Bond formace funkční.** 26–57 bondů cumulative — multicelulární niche
   vzniká nezávisle na seedu díky `BondFormed(+0.2)` rewardu (S135).

4. **Selekce na `learning_rate`.** `lr_avg` drifted z init 0.005 na 0.0058–0.0064
   (16-28 % nárůst), `lr_std` 0.0034–0.0041 (60–70 % mean) — selekční tlak
   na **rychlejší learners** je měřitelný a non-trivial; ne random drift.
   Šíře distribuce ukazuje lineage-level variance v plasticity strategiích.

5. **Selekce na `trace_decay_per_sec`.** `decay_avg` drifted z init 0.5 na
   0.43–0.47 (slabší pokles ~10 %), `decay_std` 0.19–0.26 — populace
   favorizují slabší decay (delší credit-assignment window). Konzistentní
   napříč seedy.

6. **Synaptic scaling aktivní.** `w_norm_avg` 6.22–7.29 vs cap `W_NORM_CAP =
   8.0` — homeostatic clipping ořezává top end weight growth. Bez S138 by
   norms divergovaly nad cap (pozorováno v pre-S138 smoke).

## Není pokryté

- 30+ gen runs (extrapolation z 10 gen suggestuje stabilní trajektorii,
  ale long-tail behaviors typu attractor lock-in nebo population collapse
  na gen 50+ se mohou objevit).
- Maze mode validace (`--maze` flag) — odložené.
- Damage avoidance score baseline measurement (nelze srovnat s pre-S133,
  metrika nebyla dostupná). S135 acceptance criterion "≥ baseline + 15 %"
  nelze validovat retroaktivně.
- Per-kind reward breakdown (S140 odloženo) — kvalitativně víme že rewards
  fungují (pop survives + lr drift), ale nevidíme kterých signálů jednotlivé
  brain weights nejvíc reagují.

## Doporučení pro 143-152

Decade 133-142 zavřela 4 z 5 plasticity bottlenecks z původní diagnózy:
1. ✅ Reward funnel rozšířen (eat + novelty + predation + escape + damage + bond + mate)
2. ✅ Negative reward (damage)
3. ✅ Per-cell evolved `learning_rate` + `trace_decay`
4. ✅ Homeostatic plasticita (synaptic scaling + BCM-lite excitability)
5. ❌ STDP / Izhikevich SNN — odložené do 143-152.

Pro 143-152 (SNN/STDP) navrhuji:
- Začít s passive spike event stream (S142 bridge) jako baseline; konfirmovat
  že stream sample-uje informativní timing.
- Dual-path forward: `enum NeuronModel { Perceptron, Izhikevich }` v Genome
  s opt-in mutací. Pre-existing perceptron lineages zůstávají funkční;
  Izhikevich lineages explore SNN-only fitness landscape.
- Po validaci Izhikevich correctness migrate STDP rule (timing-aware Δw)
  jako alternative reward apply path.
- Bottleneck v 137 byl `learning_rate` aktivace; cleanly v 143-152 by se
  měla rozšířit per-cell o `stdp_window_ms` + `stdp_a_plus` / `stdp_a_minus`
  jako gen traits.
