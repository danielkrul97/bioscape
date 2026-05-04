# Projekt

Cílem výzkumného projektu Bioscape je vytvořit simulaci evoluce, abych pochopil, jak vzniká inteligence. Projekt je napsán v Rustu. Výpočty se provádějí jak na CPU, tak na GPU.

**Architektura:**

- **Engine:** Bevy 0.18 (ECS, 2D rendering přes wgpu). `default-features = false` plus ručně vybraný feature set bez `bevy_gilrs` a `audio` — projekt nepotřebuje gamepad ani zvuk a tahle volba odstraňuje závislost na systémových `libudev-dev` / `libasound2-dev`.
- **Split kódu:**
  - `src/lib.rs` — čistá simulační logika (`Cell`, `step()`, `WorldMap`, …), bez Bevy. Drží taky všechny sdílené `pub const` sim parametry + `MUTATION_CONFIG` / `PHYSICS_CONFIG` — **single source of truth** pro renderer i headless. Nové tuneables, které ovlivňují simulaci, patří sem; renderer/headless-only knoby zůstávají ve svých binárkách.
  - `src/main.rs` — Bevy app, který drží svět v ECS a synchronizuje `Cell` stav s `Transform`em.
  - `src/bin/headless.rs` — bezokenní harness pro batch experimenty (CSV log per generaci, deterministický seed). Konzumuje stejné parametry z `lib.rs` jako `main.rs`, takže seed reprodukuje identický běh napříč rendererem a headlessem.
- **GPU:** Bevy táhne wgpu interně pro rendering. Pro vlastní compute kernely je v `Cargo.toml` opt-in feature `gpu` (přímý `wgpu` + `bytemuck` + `pollster`).
- **Dev smyčka:** `cargo run --features dev` zapne `bevy/dynamic_linking` pro rychlou inkrementální iteraci po prvním buildu.

**Zdroj pravdy pro biologii a evoluci:** Pokud prompt řeší evoluční mechanismy, genotyp/fenotyp, selekci, fitness, mutace nebo podobné koncepty, primárně čerpej z `docs/` — odráží to specifický framing tohoto výzkumu. Generické znalosti z tréninku použij jen jako doplněk a při konfliktu s `docs/` na to upozorni.

# Vývoj

- Vývoj vedeme po sprintech.
- 10 sprintů = jeden markdown dokument v `docs/sprints/`.
- Pojmenování: `NNN-MMM-slug.md`, kde `NNN` a `MMM` jsou zero-padded čísla prvního a posledního sprintu v souboru a `slug` je krátký název shrnující téma té desítky sprintů (např. `001-010-bootstrap.md`, `011-020-genome.md`).
- Každý sprint v dokumentu má sekci `## Sprint NN — krátký slug` se třemi řádky:
  - **Cíl:** co chceme za sprint dosáhnout.
  - **Výstup:** co reálně vzniklo (může odkazovat na commity / soubory).
  - **Poznámky:** volitelné — pozorování, slepé uličky, otázky do dalších sprintů.

# Code style

- Screenshoty VŽDY ukládej do složky screenshots/

## Commit messages

- Conventional Commits (`fix:`, `feat:`, `chore:`, …)
- Subject ≤ 50 znaků, bez tečky na konci
- Body jen když "proč" není zřejmé z diffů — max 2–3 řádky
- Žádné bullet-listy změněných souborů ani vyčerpávající popisy — kdo chce detail, přečte si diff

## Komentáře

- **Default: žádný komentář.** Dobře pojmenovaná proměnná, funkce a typ popisují *co* kód dělá — komentář to jen duplikuje a zastarává.
- Komentář piš **jen když vysvětluje WHY** — skrytý constraint, netriviální invariant, workaround pro konkrétní bug, překvapivé chování pro čtenáře.
- Pokud by odstranění komentáře nikoho nezmátlo, nepiš ho.
- **Stručně**: typicky 1–3 řádky. Odstavce v komentářích jsou varovný signál (kód by měl být čitelnější, ne okomentovanější).
- **Nepiš komentáře u self-explanatory kódu** — getter, mapovací funkce, zřejmá validace, pojmenovaný boolean výraz, typická CRUD operace. I krátký komentář tady jen přidává šum.
- **Nereferencuj kontext, který komentář přežije**: žádné „used by X", „added for the Y flow", „handles case from issue #123". To patří do PR description / commit message, ne do souboru.
- Komentáře piš anglicky.
