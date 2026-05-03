# Úvod do projektu Bioscape

## O čem to celé je

Cíl je jednoduchý k popsání, ale šíleně těžký k provedení: **postavit simulaci, ve které se z primitivních věcí (něco jako buňky) vyvine inteligence — sami od sebe, bez toho, abychom jim řekli, jak se chovat**.

Když to bude fungovat, pochopíme něco zásadního o tom, *jak* se vůbec inteligence v přírodě objevila. Protože si pravdu řekněme — fakt nikdo neví. Máme hypotézy, máme fosílie, máme mozky k pitvání, ale nemáme „přehrávání evoluce" tlačítko. Bioscape je pokus o to tlačítko.

## Proč je to vůbec možné

Příroda nedělá žádné kouzlo. Používá jenom:

1. **Replikaci** — věci, co umí dělat své vlastní kopie
2. **Variaci** — kopie nejsou úplně přesné, mutují
3. **Selekci** — některé verze přežijí a kopírují se víc než jiné
4. **Čas** — hodně, hodně času (a hodně, hodně pokusů paralelně)

To je celé. Když máš tyhle čtyři věci a dost dlouho je necháš běžet v dostatečně bohatém prostředí, vznikají z toho ledviny, oči, hejna ryb, ekonomické trhy a já. Nebo aspoň tak to tvrdí evoluční biologové, a zatím všechno, co najdeme, jim dává za pravdu.

**Bioscape se snaží znovu vytvořit ty čtyři ingredience v počítači a pustit to běžet.** Klíčové slovo: *bohaté prostředí*. Bez něj se evoluce zasekne na lokálním optimu (typicky: „minimální organismus, co se ještě umí kopírovat, a víc už nepotřebuje").

## Co je v těchto dokumentech

Tahle složka `docs/` je rešerše vědeckých prací, které jsou pro projekt relevantní. Jsou napsány tak, aby je pochopil i člověk bez biologického nebo AI vzdělání — používáme analogie a žádné rovnice tam, kde nejsou nutné.

| Soubor | O čem |
|---|---|
| [01-evoluce-zakladny.md](01-evoluce-zakladny.md) | Jak vůbec evoluce funguje — biologicky a v počítači |
| [02-umely-zivot.md](02-umely-zivot.md) | Slavné simulace života: Tierra, Avida, Karl Sims, Lenia |
| [03-neuroevoluce.md](03-neuroevoluce.md) | Jak nechat evoluci postavit nervovou síť (NEAT, HyperNEAT) |
| [04-bunky-a-morfogeneze.md](04-bunky-a-morfogeneze.md) | Jak z jedné buňky vyroste tělo (Neural CA, Levin, bioelektřina) |
| [05-neurony-a-mozek.md](05-neurony-a-mozek.md) | Co je neuron, jak ho simulovat, spiking sítě |
| [06-inteligence-a-embodiment.md](06-inteligence-a-embodiment.md) | Co vůbec je inteligence, proč potřebuje tělo |
| [07-open-ended-evolution.md](07-open-ended-evolution.md) | Proč se evoluce v simulacích zasekne a jak to obejít |
| [08-implementace-rust-gpu.md](08-implementace-rust-gpu.md) | Jak to celé technicky postavit v Rustu na GPU |

## Tři velké otázky, které nás zajímají

1. **Jak se z chemie stane buňka?** (problém abiogeneze — pravděpodobně to nejtěžší)
2. **Jak se z buněk stanou těla a mozky?** (problém morfogeneze a vzniku nervových soustav)
3. **Jak se z mozků stane inteligence?** (problém kognice)

Většina existujících simulací řeší jenom jeden z těchto tří kroků. Bioscape ambiciózně mířína kombinaci. Možná to nezvládneme. Ale i když narazíme jen na první nebo druhý krok pořádně, bude to mít hodnotu.

## Filozofická poznámka, která je vlastně dost praktická

Existuje slavná kniha Kennetha Stanleyho a Joela Lehmana **„Why Greatness Cannot Be Planned"** — a její hlavní teze je, že **když máš na začátku jasný cíl („chci vyvinout inteligenci"), nejspíš ho nikdy nedosáhneš**, protože stepping stones (mezikroky) k velkým objevům vypadají z pohledu cíle jako nesmysly.

Příklad: Pokud bys chtěl vyvinout počítač a začal s vakuovými trubicemi, „nejlepší další krok" by nikdy nebyl „polovodičová součástka" — to vypadá jako úplně jiný obor. A přece to byl ten správný směr.

**Praktická konzekvence pro Bioscape:** namísto fitness funkce typu „odměna za inteligentní chování" musíme stavět prostředí, které samo *odměňuje rozmanitost a novost*. To je celá oblast zvaná **novelty search** a **quality-diversity** — viz [07-open-ended-evolution.md](07-open-ended-evolution.md).

## Zdroje (souhrn)

Detailní citace jsou v jednotlivých kapitolách. Hlavní odkazy:

- [Stanford HAI — Evolving Embodied Intelligence](https://hai.stanford.edu/news/how-bodies-get-smarts-simulating-evolution-embodied-intelligence)
- [Distill — Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/)
- [Why Greatness Cannot Be Planned — Stanley & Lehman](https://link.springer.com/book/10.1007/978-3-319-15524-1)
- [Avida Digital Evolution Platform](https://alife.org/encyclopedia/digital-evolution/avida/)
- [Karl Sims — Evolved Virtual Creatures](https://www.karlsims.com/evolved-virtual-creatures.html)
