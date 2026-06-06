# Plán odstranění endosymbiontů

Cíl: kompletně odstranit symbiont feature z projektu.

## Proč

Po S204 pivotu je symbiont jen **data-only damage-resist příznak** (fotosyntéza
vypnutá — konstanty `SYMBIONT_PHOTO_*` nikdo nečte). Přitom stojí:

- ~44 KB na `Cell` (embeduje celý druhý `Genome` v `Option<Symbiont>`, i když `None`),
- 2 brain-input sloty (39 = `has_symbiont`, 40 = `deficit_norm`),
- 2 GPU buffery + bind sloty, 6 CSV sloupců, a několik RNG draws.

Za marginální mechaniku (flat 10% damage-resist vs flat body-cost). Odstranění
zjednoduší model a uvolní brain-input šířku.

## Pořadí fází (od nejméně provázaného k nejvíc)

### Fáze A — mechanické mazání (bez dopadu na RNG/paritu)

- **Renderer** (čistě vizuální): smazat `src/renderer/systems/symbionts.rs` celý;
  `SymbiontMarker` (`components.rs:33-36`); `SymbiontMesh`/`SymbiontMaterial`
  (`resources.rs:202,205`); registraci `sync_symbionts` (`renderer/mod.rs:188`);
  setup spawn/mesh/material (`setup.rs:241-248,289-295,485-486`); sim_tick spawn +
  importy (`sim_tick.rs:11,15,91-92,163-169`); `systems/mod.rs:12,18`.
- **Mrtvé pole** `World.sym_sheds_gen` (`world.rs:217`, vždy 0) + 4 init/reset
  sity (`world.rs:591,1002`, `sim_tick.rs:39`, `headless/main.rs:535`) + jeho CSV write.
- **GPU buffery**: `symbiont_has_buf`/`symbiont_deficit_buf`
  (`gpu/cells.rs:72,76,239-256,391-392,464-465,1036-1045`); scratch `sym_has`/
  `sym_deficit` (`gpu/scratch.rs:34-39,123-124`); populate_inputs bindy 14/15
  (`gpu/populate_inputs.rs:64-65,178-179` + loop `(0..16)`→`(0..14)`).
- `json_export.rs`, `gpu/stats.rs`, `headless/dump.rs` — bez referencí, nic.

### Fáze B — sim logika + RNG stream (přijmout změnu streamu)

- Smazat `apply_symbiont_energy` (`world.rs:3549-3567`) + call v `tick()` (`world.rs:1368`).
- Collapse damage-resist: hazards (`world.rs:2743-2747`), predace (`world.rs:3472-3479`),
  maze bump (`cell.rs:665`) → odebrat `* damage_resist_factor()`. Smazat
  `damage_resist_factor` (`cell.rs:282-288`).
- Init seeding (`world.rs:518-530`), predační capture (`world.rs:3427-3461`),
  dědičnost při reprodukci (`reproduction.rs:171-204,259`). **Tyto mažou RNG draws → viz riziko #4.**
- Symbiont age increment (`cell.rs:418-420,446-448`), checkpoint re-derive
  (`world.rs:945-953`), GPU upload scratch+call (`world.rs:1784-1791,1841`).
- Smazat `Symbiont` struct (`cell.rs:28-48`) + `Cell.symbiont` pole (`cell.rs:201-205`) +
  `from_genome` init (`cell.rs:361`) + `World.next_symbiont_lineage_id` (`world.rs:225`).
- Pozor: host `Cell.lineage_id`/`lineage_birth_gen` (`cell.rs:68-72`) NECHAT — to je
  něco jiného než `Symbiont.lineage_id`.

### Fáze C — brain inputs (NEJRIZIKOVĚJŠÍ — udělat atomicky)

- Odebrat `N_SYMBIONT_INPUTS` (`params/brain.rs:66`) → `BRAIN_INPUTS_SENSORY` 41→39,
  `BRAIN_INPUTS` 86→84 (Rust konstanty kaskádují samy přes všechna `[f32; BRAIN_INPUTS]` pole).
- Smazat zápisy do slotů 39/40: `sensors.rs:133-142` (CPU) **A** `populate_inputs.wgsl:50-55,214-221`
  (GPU). Ponechání WGSL zápisů by korumpovalo recurrent sloty 0/1.
- Ručně přepsat hardcoded WGSL konstanty (neadaptují se samy): `86u→84u`,
  `3870u→3780u` (=45×84), `41u→39u` v: `brain_forward`, `brain_forward_izhikevich`,
  `hebbian`, `hebbian_step`, `hebbian_apply_reward`, `synaptic_scale`, `excitability`,
  `stdp_step`, `stdp_apply`, `stdp_encode_pre`, `cppn_from_cppn`.
- Bump `CHECKPOINT_VERSION` 10→11 (`world.rs:83`) — `w1` se reshapuje 45×86→45×84,
  staré checkpointy nejdou načíst (version gate je odmítne).

### Fáze D — konstanty + CSV + testy

- Smazat všechny `SYMBIONT_*` / `*_PHOTO_*` konstanty + komentářový blok (`lib.rs:353-450`).
- CSV: smazat 6 sloupců `sym_count, sym_fraction, sym_lineage_count, sym_z_avg,
  sym_deficit_avg, sym_sheds` (0-based indexy 127–132) z **headeru** (`headless/main.rs:405`)
  **A** format-stringu (`headless/csv.rs:652`) **A** writer args (`csv.rs:804-810`) **A**
  akumulátorů (`csv.rs:115-120,349-358,646-649`) — header a writer držet v lockstepu.
- Smazat endosymbiosis/damage-resist test blok (`tests.rs:3395-3632`) + helpery;
  update `tests.rs:1040` (Cell literál), `test_helpers.rs:159`.

### Fáze E — validace

- Spustit GPU parity testy (populate_inputs + brain forward na 84-wide layoutu).
- **Regenerovat seeded baseliny** (RNG stream se změnil) — přegenerovat 5×100-gen
  cross-seed sweep jako NOVOU referenci, nediffovat proti starým běhům.

## 4 provázaná rizika (call-outy)

1. **Brain-input sloty** — WGSL hardcody se neadaptují; minutí jednoho shaderu =
   tichá CPU/GPU divergence brain forward (špatný w1 row-stride). Smazat slot-39/40
   zápisy v sensors.rs i populate_inputs.wgsl **současně** s posunem konstant.
2. **w1 layout / checkpoint** — `w1 = [[f32; BRAIN_INPUTS]; BRAIN_HIDDEN]` 45×86→45×84
   přetváří každý genom; bump `CHECKPOINT_VERSION`.
3. **Pořadí CSV sloupců** — header/format/args musí jít v lockstepu; test
   `empty_and_populated_rows_have_same_column_count` křížově porovnává jen dva writer
   řádky, header/writer drift NEodhalí — ověřit ručně.
4. **RNG-draw determinismus** — odebrání symbiont draws (init seeding = 1 `f32`/cell +
   `Genome::random` na ~50% buněk; dědičnost = draws/reprodukci; predační capture =
   `f32`/hit i když `P=0`) posune CELÝ deterministický stream. Žádný způsob, jak feature
   odebrat a zachovat byte-identické seedy — všechny baseliny přegenerovat.

## Doporučené pořadí provedení

A (mechanické) → B (sim logika, akceptovat změnu streamu) → C (brain inputs +
WGSL konstanty + CHECKPOINT_VERSION, vše naráz) → D (konstanty/CSV/testy) →
E (re-run GPU parity + přegenerovat baseliny).
