# Vibration Sensing — Motion-Driven Mechanosensory Field

Standalone feature work mimo decade — testuje hypotézu, že "trembling" pohyb
buněk, který lze pozorovat v rendereru, může sloužit jako vedlejší kanál
inter-cell komunikace, pokud má prostředí mechanickou paměť (šíření vibrace).

## Cíl

Předtím cells neměly žádný mechanosensorický kanál — veškerá inter-cell
komunikace probíhala přes:

1. **Pheromony** (3 broadcast pole, brain emits via `output[2,10,11]`),
2. **Bondové zprávy** (point-to-point přes spring bonds, `output[12,13]`),
3. **Vizuální detekce** (food/cell delta, density).

Hypotéza: protože buňky během pohybu vibrují (kombinace drag, brownian noise,
turn motor), motion-driven "stir" by mohl propagovat médiem jako tlaková
disturbance — a evoluce by mohla repurposnout motor patterns na carrying
signál. To je analog mechanosensoriky u prokaryot a primitivní eukaryot.

## Mechanismus

### Konstanty (`src/params/vibration.rs`)

```rust
pub const VIBRATION_GRID_RES: usize = 64;
pub const VIBRATION_GRID_RES_Z: usize = 16;
pub const VIBRATION_DIFFUSION: f32 = 0.15;          // < 1/6 stability
pub const VIBRATION_DECAY: f32 = 4.0;               // 1/s — fast dissipation
pub const VIBRATION_K_LINEAR: f32 = 1.0;            // emit z |v|/max_speed
pub const VIBRATION_K_ANGULAR: f32 = 0.5;           // emit z |ω|/turn_rate
pub const VIBRATION_NORMALIZATION_GAIN: f32 = 1.0;  // tanh on brain inputs
pub const VIBRATION_SAMPLE_EPSILON: f32 = 10.0;
pub const N_VIBRATION_INPUTS: usize = 4;            // grad_xyz + amp
```

### Emise (NEpoužívá brain output)

```text
emit = K_LINEAR * |v|/max_speed + K_ANGULAR * (|ω|+|ω_p|)/turn_rate
```

Klíčový design: emise není explicitně řízena brainem (jako pheromony jsou).
Je to vedlejší produkt pohybu — buňka, která rychleji cestuje nebo rotuje,
ruší medium víc. Energetický náklad za vibraci je nulový (energy už platí
za pohyb sám). Selekce tak musí repurposnout *existující* motor patterns.

Sdílený helper `bioscape::vibration_emit_for_cell` zajišťuje, že emisní
vzorec drží single-source v hlavičce + headless World, headless CSV
agregaci, i v renderer field updatu.

### Propagace

`SmellField` (3D scalar pole) s identickým 7-point Jacobi stencilem +
multiplikativní decay. Toroidal XY, bounded Z. Brain čte gradient + lokální
amplitudu — kombinace dává směr i sílu nejbližšího "hlučného" zdroje.

### Brain expansion

| const                  | pre-V7 | post-V7 |
|------------------------|--------|---------|
| `BRAIN_INPUTS_SENSORY` | 29     | **33**  |
| `BRAIN_INPUTS`         | 74     | **78**  |
| `BRAIN_OUTPUTS`        | 14     | 14      |
| `BRAIN_HIDDEN`         | 45     | 45      |

#### Slot mapping

Inputs (sensory):
- 0–26: existing (food/cell/smell/pheromone/density/damage/heading/energy/speed/temp)
- 27–28: bond message inbox (zachováno)
- **29, 30, 31**: vibration gradient xyz (NOVÉ)
- **32**: vibration amplitude — lokální field sample (NOVÉ)
- 33..78: recurrent (`BRAIN_RECURRENT = BRAIN_HIDDEN = 45`)

Outputs: beze změny. Vibrace nemá explicitní emit kanál (viz výše).

#### Sensor categories

Nová `SENSOR_CATEGORY_MECHANO = 3`, `N_SENSOR_CATEGORIES: 3 → 4`. Selekce
nezávisle ladí citlivost na vibrace přes `genome.sensor_gains[3]`. Cost
roste lineárně s gain (sdílený `SENSOR_GAIN_COST`). Pooling přes bond
network automaticky funguje (sloty 29–32 jsou zapsány v `sensor_slot_category`,
takže `pool_bonded_sensors` max-pooluje přes bonded peers).

### GPU path

CPU-only pro vibration field — žádný compute shader. WGSL `populate_inputs`
slots 29–32 píše 0.0 (GPU brain je effectively deaf k mechanosense). Bumply
ale jsme `BRAIN_INPUTS = 78` v `brain_forward.wgsl`, `hebbian.wgsl`,
`cppn_from_cppn.wgsl`, `motor.wgsl` + offset/weights asserts v
`src/gpu/context.rs`. Renderer-default je CPU brain (post-S132), takže
user-facing experiment běží correct path.

## CSV diagnostika

Nové sloupce v `src/bin/headless/csv.rs`:

| sloupec            | význam                                                        |
|--------------------|---------------------------------------------------------------|
| `vib_emit_avg`     | mean motion-driven emise per cell per tick (≈ pohybová aktivita) |
| `vib_amp_avg`      | mean field sample na cell pozici (= "hluk" lokálního média)   |
| `vib_grad_mag_avg` | mean `|∇vibration|` over cells — citlivost gradientu          |
| `gain_mech_avg`    | populace mean `genome.sensor_gains[MECHANO]` (selekční signal)|
| `gain_mech_dev`    | stddev téhož — specialization vs uniform shift                |

## Plánovaný headless test

```bash
# smoke (verifikace nenulových sloupců, healthy populace)
cargo run --release --bin headless 0 30 /tmp/vib_smoke.csv

# delší běh pro evoluční dynamiku
cargo run --release --bin headless 0 300 /tmp/vib_long.csv
```

Očekávané signály:

| pozorování                                  | interpretace                                  |
|---------------------------------------------|----------------------------------------------|
| `gain_mech_avg` roste přes gens             | pozitivní selekce — vibrace užitečná         |
| `gain_mech_avg` klesá k 0                   | cells šetří gain cost, signál není exploited |
| `gain_mech_avg ≈ 1.0` se širokým `gain_mech_dev` | bimodální — specialization mezi cells       |
| `vib_amp_avg` koreluje s `mean_bond_count`  | bonded clusters generují víc vibrace         |
| `vib_grad_mag_avg` koreluje s `bond_active_frac` | clusters vidí strukturovanější pole         |

Korelace mezi `vib_*` a bond/cluster metrikami se počítá post-hoc v
analytics notebooku (např. Python pandas). Single per-gen CSV file
postačuje.

## Mimo rozsah

- Plně GPU vibration field (compute shader). CPU SmellField step je
  ≤ 200 µs / tick (SIMD S117); perf headroom je dostatečný i bez GPU.
- "Aktivní relay" — bonded buňky neslouží jako vibration repeater. Sharing
  v rámci clusteru je řešen výhradně přes sensor pooling.
- Frekvenční rozlišení vibrací (multi-band spectral channel). Single
  scalar field; band-pass decomposition by byla samostatný feature.
- Wave equation (hyperbolický stencil). Diffusion approximation je
  dokumentovaný trade-off; reálná wave physics by zdvojnásobila paměť
  (velocity field) a vyžadovala CFL-stable scheme.

## Breaking changes

- `CHECKPOINT_VERSION 6 → 7`. V6 savefiles už nelze loadnout — load
  vrací explicit version mismatch error.
- `Genome.sensor_gains` shape `[f32; 3] → [f32; 4]`. Nová default = 1.0
  na index 3.
- Brain weight matice resize (w1 `[45][74]→[45][78]`, ostatní stejné).
- WGSL shader constants bumply (`BRAIN_INPUTS`, `BRAIN_INPUTS_SENSORY`,
  `BRAIN_OUTPUTS`, offsets, weights-per-cell).
