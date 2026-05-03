# Sprinty 001–010 — Bootstrap

Bootstrap fáze projektu: minimální 2D scéna → fixed-tick simulační clock → headless běh → první genom.

## Sprint 01 — bevy-scaffold

- **Cíl:** minimální Bevy 2D scéna s buňkami pohybujícími se ve čtverci, simulační logika oddělená do `lib.rs` (aby šla později pohánět i headless).
- **Výstup:** `src/lib.rs` s `Cell::{random, step}`, `src/main.rs` s 200 buňkami a 2D kamerou, ořezaná Bevy feature sada bez `bevy_gilrs` a `audio` (nepotřebujeme gamepad ani zvuk, mizí závislost na `libudev` / `libasound`). Commity `1235e5c`, `b142e0e`.
- **Poznámky:** `step_cells` čte `Time<Real>` přes `time.delta_secs()` — vázané na FPS a nedeterministické. Řeší Sprint 02.

## Sprint 02 — sim-clock

- **Cíl:** oddělit simulační čas od wall clocku a zavést tříúrovňovou hierarchii **tick → generace → epocha**.
  - Sim systémy (`step_cells` a další) přesunout do `FixedUpdate` schedule a brát `dt` z `Time<Fixed>` (default 60 Hz). Tím získáme deterministický krok nezávislý na FPS — předpoklad reprodukovatelných experimentů z `docs/08`.
  - Rychlost runtime přes `Time<Virtual>::set_relative_speed`. Klávesy: Space = pause, `1` / `2` / `3` / `4` = 1× / 10× / 100× / max. Speed má smysl jen ve windowed binárce; headless vždy poběží naplno.
  - `SimClock { tick, generation, epoch, ticks_per_generation, generations_per_epoch }` jako plain struct v `lib.rs` (bez Bevy types, ať jde pohánět i z chystaného headless harness). V `main.rs` zabalený jako Bevy `Resource`.
  - Jeden `advance_clock` systém v `FixedUpdate` inkrementuje čítače a při překročení hran emituje Bevy eventy `GenerationEnded { gen }` a `EpochEnded { epoch }`. Důvod hybridu (čítače + eventy): rychlé per-tick systémy nemusí kontrolovat modulo, pomalá logika (selekce, snapshoty, klimatické cykly) jen reaguje na event. Sedí to na rozdělení rychlé učení / pomalá evoluce z `docs/02`.
  - Počáteční hodnoty (vyladí se empiricky): `Time<Fixed>` = 60 Hz, `ticks_per_generation` = 600 (≈ 10 s sim-času při 1×), `generations_per_epoch` = 100.
- **Výstup:**
  - `src/lib.rs`: `SimClock { tick, generation, epoch, ticks_per_generation, generations_per_epoch }` + `ClockTransitions` (`Option<u64>` pro každou hranici), `advance()` vrací přechody. Unit testy fixují boundary sémantiku.
  - `src/main.rs`: `step_cells` v `FixedUpdate`; `Time<Fixed>` na 60 Hz; `Clock(SimClock)` jako Bevy `Resource`; `advance_clock` v `FixedUpdate` emituje `GenerationEnded { generation }` a `EpochEnded { epoch }`.
  - `speed_input`: Space = pause/unpause, `1`/`2`/`3`/`4` = 1× / 10× / 100× / 1000× (zvolený strop pro „max"); `log_clock_events` zatím loguje hrany přes `info!`.
  - Konstanty: `FIXED_TIMESTEP_HZ = 60.0`, `TICKS_PER_GENERATION = 600`, `GENERATIONS_PER_EPOCH = 100`.
- **Poznámky:**
  - Per-organism věk (`born_tick` na `Cell`) **ne** — počká si na sprint o reprodukci/lifespan.
  - Globální generační hranice je úmyslně GA-style. Async selekce á la Tierra (`docs/02`) se znovu zváží, až bude reprodukce — `SimClock` pak buď zůstane jako environmentální čas (sezóny, klima) a generace se přesunou per-organism.
  - `set_relative_speed` na `Time<Virtual>` automaticky zrychluje i `Time<Fixed>` (víc `FixedUpdate` runů per frame) — proto stačí jeden multiplier pro celý sim.
  - `1000×` jako „max" je arbitrární strop; reálně je rychlost limitována Bevy `Time<Virtual>::max_delta` (default 0.25 s = ~15 fixed updateů per frame). Plný uncapped režim je téma chystaného headless harness.
  - Bevy 0.18 nepoužívá `Event`/`EventReader`/`EventWriter`/`add_event` pro buffered eventy — to je teď `Message`/`MessageReader`/`MessageWriter`/`add_message`. `Event` je rezervovaný pro observer-targeted eventy.
