# Bioscape — výzkumná dokumentace

Rešerše vědeckých prací relevantních pro projekt Bioscape — simulaci evoluce, jejímž cílem je pochopit vznik inteligence.

Dokumenty jsou napsány srozumitelně pro laika, s analogiemi a bez zbytečné matematiky.

## Obsah

| Soubor | Téma |
|---|---|
| [00-uvod.md](00-uvod.md) | Úvod, motivace projektu, mapa dalších kapitol |
| [01-evoluce-zakladny.md](01-evoluce-zakladny.md) | Jak evoluce funguje (biologicky i v počítači), genotyp/fenotyp, evolvability |
| [02-umely-zivot.md](02-umely-zivot.md) | Tierra, Avida, Karl Sims, Lenia, Stanford DERL |
| [03-neuroevoluce.md](03-neuroevoluce.md) | NEAT, HyperNEAT, MAP-Elites — evoluce neuronových sítí |
| [04-bunky-a-morfogeneze.md](04-bunky-a-morfogeneze.md) | Cellular automata, Neural CA, Michael Levin a bioelektřina |
| [05-neurony-a-mozek.md](05-neurony-a-mozek.md) | Modely neuronu, spiking sítě, Izhikevich, neuromodulace |
| [06-inteligence-a-embodiment.md](06-inteligence-a-embodiment.md) | Co je inteligence, embodied cognition, Free Energy Principle |
| [07-open-ended-evolution.md](07-open-ended-evolution.md) | Proč se evoluce v simulacích zasekne a jak to obejít |
| [08-implementace-rust-gpu.md](08-implementace-rust-gpu.md) | Rust + GPU: wgpu, architektura, pořadí kroků |

## Doporučený pořadí čtení

1. **Pokud nevíš nic o oboru:** 00 → 01 → 06 → ostatní podle zájmu
2. **Pokud rovnou stavíš:** 00 → 02 → 03 → 08
3. **Pokud zajímá teoretický kontext:** 04 → 05 → 06 → 07
