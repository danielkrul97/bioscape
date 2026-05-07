# Cooperative Food Packets — Selekční tlak pro emergentní komunikaci

Standalone feature work mimo decade. Reakce na verdict z `multi-channel-pheromones.md`:
multi-channel infrastructure funguje, ale bez fitness coupling cells abandonují
ch1/ch2 (300-gen × 3 seedy → ch1/ch2 ≈ 0.004, ch0 saturuje na 0.987). Tento
sprint zavádí **první ze 4 navržených tlaků** — cooperative food packets,
které vyžadují koordinovaný arrival N cells během time window.

## Cíl

Vytvořit closed loop:

```
emit → environment change → receiver behavior change → emitter benefit
```

High-value food node se neuvolní, dokud `COOP_FOOD_REQUIRED_ARRIVALS` cells
během time window nedorazí. Cells, které se naučí signalovat "come here"
(přes existing pheromone channels), recruit peers → coordination událost
úspěšná → reward sdílen napříč participanty. Solo cells nedosáhnou threshold
→ no reward = selekce na "see food, signal peers".

## Mechanismus

### Architektonická decize

`CoopFood` jako **separate struct** (ne `FoodKind` variant). Lifecycle
(waiting → triggered/expired) je jiný od regular food (consumed-by-eat),
spawn rate jiný (Poisson per-tick vs density target replenish), reward
distribuovaný **up-front** systémem (ne per-eat). Eat code regular food
zůstává nedotčený.

### Konstanty (lib.rs)

```rust
pub const COOP_FOOD_REQUIRED_ARRIVALS: usize = 3;
pub const COOP_FOOD_TIME_WINDOW_TICKS: u32 = 120; // 2 sec @ 60 Hz
pub const COOP_FOOD_REWARD_PER_CELL: f32 = 80.0;  // 4× Plant base
pub const COOP_FOOD_ARRIVAL_RADIUS: f32 = 30.0;   // > regular eat radius
pub const COOP_FOOD_SPAWN_RATE_PER_TICK: f32 = 0.02;
pub const COOP_FOOD_MAX_CONCURRENT: usize = 8;
```

Calibration rationale:
- 0.02 spawn × 600 ticks/gen ≈ 12 events per gen (matches 8 concurrent cap).
- Reward 80 vs Plant 20 = 4×. Asymetricky vysoký aby loiter cost (motion +
  brain emit) byl pokrytý i pro participanty co solo by snědli regular plant.
- ARRIVAL_RADIUS 30 > eat radius (~20). Coop food má vizuální/aroma signal,
  cells nemusí stát přímo na něm.

### `CoopFood` struct + lifecycle

```rust
pub struct CoopFood {
    pub position: [f32; 3],
    pub spawn_tick: u64,
    pub arrivals: Vec<u64>,      // unique cell_ids
    pub triggered: bool,
}
```

Lifecycle:
1. **Spawn** — per tick `rng.random::<f32>() < SPAWN_RATE_PER_TICK` při
   `coop_foods.len() < MAX_CONCURRENT`. Pozice uniform world bounds (žádný
   richness check — coop nodes nezávisí na food density mapě).
2. **Arrival** — `register_coop_arrivals_for_all` per tick: každý coop node
   sken cells, toroidal-aware Euclidean test ≤ ARRIVAL_RADIUS, unique cell_id
   přidá do `arrivals`. Insertion order, duplikáty ignored.
3. **Trigger** — `try_trigger_coop`: pokud `arrivals.len() >= REQUIRED && !triggered`
   → každý cell v arrivals dostane `REWARD_PER_CELL` energy, set `triggered = true`.
   Threshold-only — nad threshold víc cells = víc total reward (incentive pro
   recruiter, ne striktní equal-share).
4. **Cleanup** — pokud `triggered` OR `tick - spawn_tick >= TIME_WINDOW`
   → `swap_remove`. Counters `coop_food_solved_gen` / `coop_food_failed_gen`
   inkrementovány v cleanup pass.

### Sensor integrace

**Žádné brain dim změny.** Cells detekují coop food **stejnou cestou jako
regular food** přes `nearest_food` sensor input. V `brain_act` (CPU + GPU
varianty) se po regular food_grid sweep dělá lineární pass přes `coop_foods`
(typicky ≤ 8 nodes) se stejným vr2 + cone test. Cell tedy "vidí" nejbližšího
z {regular food, coop food}; coop má vyšší expected value, ale solo arrival
nedostane reward = trade-off naturálně vystavený na evoluci.

GPU `--gpu-full` path: coop pozice se konkatenují na konec `food_positions`
před GPU sensor shader dispatch — sdílený single array, nearest_food
selection probíhá uniformně.

### Eat blocking

Cell **nemůže eat solo** coop food — eat code prohledává `self.foods` (vec
regular Food), které coop_foods ne-obsahuje. Cell jen "navštěvuje" node;
reward distribuuje up-front systém v `update_coop_food`. Žádné race s
existing eat_food.

## CSV diagnostika

3 nové sloupce na konci řádku:

| sloupec                 | význam                                                            |
|-------------------------|-------------------------------------------------------------------|
| `coop_food_solved`      | count úspěšných events za generaci (triggered = reward)           |
| `coop_food_failed`      | count expired bez triggered za generaci                           |
| `coop_food_arrivals_avg`| mean `arrivals.len()` přes všechny zaniklé v gen (vč. expired)    |

Reset per-gen counterů v end-of-gen block (mirror existing `bonds_formed_gen`).

Header v `headless.rs`:
```
…,spike_total_length_avg,ticks_per_sec,coop_food_solved,coop_food_failed,coop_food_arrivals_avg
```

## Tests

3 nové unit tests v `mod tests`:

- `coop_food_lifecycle_no_arrivals_expires` — bez arrivals coop prošlý
  TIME_WINDOW vrací `is_expired = true`, `try_trigger_coop` vrátí false.
- `coop_food_threshold_triggers_reward` — 3 unique arrivals → trigger,
  každý cell dostane REWARD_PER_CELL, idempotent (druhý try nesmí znovu
  rozdat). Duplicate cell_id v `register_coop_arrival` ignored.
- `coop_food_below_threshold_no_reward` — 2 < REQUIRED → no trigger,
  energie beze změny, coop alive.

Lib test counts: **176 passed, 1 ignored** (z toho 173 původních + 3 nové).

## Smoke results (seed=42, 30 gens, CPU path)

Final agregáty:
- final cells: **945**
- final lineages: **9**
- total `coop_food_solved` (sum přes všechny gens): **24**
- total `coop_food_failed`: **343**
- ratio solved/total events: ~7 % (v early gens, před evolution adaptation)

Per-gen highlights (selected):

```
gen=0  cells=200  solved=0  failed=0  arr_avg=0.000
gen=4  cells=1500 solved=2  failed=11 arr_avg=0.615
gen=7  cells=1497 solved=3  failed=19 arr_avg=0.864
gen=9  cells=1076 solved=4  failed=12 arr_avg=1.250
gen=10 cells=898  solved=1  failed=8  arr_avg=0.556
gen=15 cells=820  solved=0  failed=12 arr_avg=0.417
gen=21 cells=872  solved=1  failed=10 arr_avg=0.364
gen=29 cells=946  solved=2  failed=10 arr_avg=0.833
gen=30 cells=945  solved=0  failed=15 arr_avg=0.200
```

Pozorování:
- **Triggering funguje**: solved > 0 v 14 z 30 generací — sanity že coop
  loop uzavřený.
- **Failed >> solved** — convergence ke koordinovanému recruitment behavior
  zatím nevidět (30 gens je málo). 300-gen experiment ukáže, jestli
  selekce začne `solved/failed` zvedat.
- **arr_avg < 1** většinu gens — cells obvykle navštíví coop solo,
  occasional duo (~0.5-1.2). Při solo arrival nestane nic.
- **Žádný regres** ostatních metric — populace stabilní (peak 1500 v gen 4
  daleko od MAX_POP, plateau ~870-950 později), bonds/predation dynamika
  beze změny.

## Otevřené otázky pro 300-gen experiment

1. **Konverguje populace k recruitment behavior?** Pokud ano, čekáme:
   - `coop_food_solved` roste s gen
   - `coop_food_arrivals_avg` roste směrem k 3.0 (REQUIRED) a možná dál
   - `ph_emit_ch1_avg` / `ph_emit_ch2_avg` přestávají klesat na ~0
     (signaling channels najdou raison d'être)
2. **Korelace `solved` ↔ `mean_bond_count`?** Bonded clustery jako
   structural recruitment substrate (bonds drží 3-5 cells blízko = perfect
   cooperative trio). Pokud korelace > 0.5 v late phase, tissue regime
   nepřímo vyhrává.
3. **Specialization mezi role recruiter / responder?** Variance brain
   output (ch1/ch2 emit) per cell (ne mean) měla by stoupat, pokud cells
   dělí role. Současná `ph_emit_*_dev` metric to detekuje.
4. **Free-rider problem?** Cell co prostě stane na coop nodu bez signalování
   také dostane reward. Jestli evoluce vybere "wait at random food spot",
   recruitment behavior nikdy nevznikne. Test: porovnat coop_food_solved
   napříč seedy — pokud variance vysoká, free-rider tactic dominuje
   stochastically.

## Soubory změněné

- `src/lib.rs` — `COOP_FOOD_*` konstanty, `CoopFood` struct, `register_coop_arrival`,
  `try_trigger_coop`, `random_coop_position`, `register_coop_arrivals_for_all`,
  3 unit testy.
- `src/bin/headless.rs` — `World.coop_foods` + per-gen counters, `spawn_coop_food`,
  `update_coop_food` per-tick metody, sensor gather rozšíření v CPU + GPU
  brain_act paths, `food_positions` injection pro `--gpu-full` sensor shader,
  CSV header + write_stats + empty-pop branch + per-gen reset.
- `src/main.rs` — `CoopFoodResource` Resource, `spawn_coop_food` + `update_coop_food`
  systems v FixedUpdate chain, sensor gather v `cells_brain_act` rozšířen o
  coop_positions vec.

## Implementační poznámky

- `try_trigger_coop` distribuuje reward přes `cells.iter_mut().find` — O(N)
  per cell_id, ale len = arrivals.len() (typicky < 10) × cells.len(). Pro
  větší pop bude lepší předpřipravit `id_to_idx` mapu, ale aktuální path
  není hot loop (max 8 coop × < 10 arrivals = < 80 lookup events per gen).
- Renderer (`main.rs`) má `update_coop_food` system bez reward distribuce —
  headless drží authoritative metric. Pro plný coupling rendereru je nutný
  separátní system s `Query<&mut CellEntity>` access; odloženo do pozdější
  iterace (renderer slouží primárně jako visualization, vědecké data jdou
  z headless CSV).
- GPU full path: pozice coop food konkatenovaná na konec `food_positions`,
  GPU sensor shader nezná rozdíl mezi regular a coop. Stejné chování jako
  CPU path → parity zachována pro brain forward.
