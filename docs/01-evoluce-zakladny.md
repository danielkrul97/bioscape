# Evoluce: úplný základ

## Evoluce v jedné větě

**Když mají věci tendenci se kopírovat, kopie nejsou úplně přesné, a okolí nepustí všechny dál — z toho vzniká pokrok bez toho, aby ho kdokoli plánoval.**

To je celé. Žádný designer, žádný cíl, žádná inteligence v pozadí. Jenom tři pravidla a hodně času.

## Tři ingredience trochu podrobněji

### 1. Replikace (kopírování)

Něco musí umět udělat své vlastní kopie. V přírodě je to DNA — molekula, která má tvar dvojité šroubovice a obě poloviny si pasují jako zip. Když se zip rozevře, každá polovina si k sobě přitáhne nové stavební kameny a vznikne přesná kopie.

V počítači může být replikátor cokoli, co umí ze sebe vyrobit svou kopii: kus kódu, sada čísel (genom), grafová struktura...

### 2. Variace (mutace)

Kopie nejsou nikdy 100% přesné. V přírodě:
- **Bodové mutace** — náhodná chyba v jednom písmenku DNA
- **Crossover** — pohlavní rozmnožování míchá poloviny od dvou rodičů
- **Duplikace** — celé části genomu se omylem zkopírují dvakrát (mimochodem, velmi důležitý zdroj evoluční inovace)

V počítači mutace simulujeme přímočaře: s nějakou pravděpodobností změň náhodný bit/číslo/uzel.

### 3. Selekce (filtr)

Ne všechny kopie přežijí dost dlouho na to, aby udělaly další kopie. V přírodě je to drsné — nedostatek jídla, predátoři, choroby, nehody. V simulaci tomu říkáme **fitness function** — funkce, která řekne „tento jedinec je tak dobrý" a podle toho se rozhodne, kdo se rozmnoží.

⚠️ **Pozor**: Volba fitness funkce je často to nejtěžší rozhodnutí celé simulace. Špatná fitness produkuje organismy, co optimalizují fitness *na úkor* toho, co jsi vlastně chtěl. To je celá kapitola sama o sobě, viz [07-open-ended-evolution.md](07-open-ended-evolution.md).

## Genotyp vs. fenotyp

Klíčový rozdíl, který je dobré mít hned od začátku v hlavě:

- **Genotyp** = recept (DNA, kód, čísla v paměti)
- **Fenotyp** = upečený výsledek (skutečné tělo, chování, schopnosti)

Mezi nimi probíhá **vývoj** (development) — proces, který vezme recept a vyrobí z něj organismus. V přírodě to trvá týdny až roky (z oplodněného vajíčka člověk za 9 měsíců). V simulaci to může být cokoli od „okamžitě, pomocí mapování" po „simulace metabolismu po dobu N kroků".

**Tohle je důležité, protože evoluce mutuje *recept*, ale selekce odměňuje *upečenou věc*.** Drobná změna receptu může vést k velké změně výsledku (nebo k žádné, nebo k organismu, co se zhroutí ještě v děloze).

## Genetické algoritmy: evoluce v počítači

Klasický genetický algoritmus, jak ho vymyslel John Holland v 60. letech:

```
1. Vytvoř náhodnou populaci N jedinců
2. Loop:
   a. Vyhodnoť fitness každého jedince
   b. Vyber rodiče (preferuj fittest)
   c. Křížením a mutací vytvoř N nových potomků
   d. Nahraď populaci potomky
3. Vrať nejlepšího jedince
```

Tohle dnes funguje překvapivě dobře na spoustu inženýrských problémů (anténní designy NASA, optimalizace tras, atd.), ale **má jeden zásadní problém pro náš účel**: konverguje k jednomu řešení. Po pár generacích jsou všichni jedinci skoro stejní a další progress se zastaví.

V přírodě se to neděje, protože:
- Prostředí se mění (klimatické změny, jiné druhy se vyvíjejí)
- Selekce je lokální (predátor neporovnává všechny zajíce světa, ale jen ty, co potká)
- Existuje **niche partitioning** — různé druhy si rozdělí ekologické niky

Bioscape musí tohle nějak řešit — viz quality-diversity v [07-open-ended-evolution.md](07-open-ended-evolution.md).

## Co je „evolvability" a proč by tě měla zajímat

**Evolvability = schopnost organismu vyvíjet se dál.**

Některé reprezentace genomu jsou „křehké" — každá mutace je spíš škodlivá. Jiné jsou „pružné" — mutace často produkují nové, ale stále funkční varianty. Druhý typ se vyvíjí mnohem rychleji.

V přírodě má evolvability sama tendenci být selektována (organismy s dobrou evolvabilitou mají víc úspěšných potomků v dlouhém horizontu). V simulaci to musíme zařídit volbou *kódování* genomu.

**Příklady kódování pro Bioscape:**
- Přímé (direct): genom = váhy neuronů. Jednoduché, ale nemá strukturu.
- Vývojové (developmental): genom = recept, jak buňka roste a dělí se. Mnohem víc evolvable, ale složitější simulace. Viz [04-bunky-a-morfogeneze.md](04-bunky-a-morfogeneze.md).
- Generativní (CPPN, HyperNEAT): genom = funkce, která produkuje strukturu. Něco mezi. Viz [03-neuroevoluce.md](03-neuroevoluce.md).

## Důležité historické milníky (pro kontext)

| Rok | Co se stalo | Proč to bylo důležité |
|---|---|---|
| 1859 | Darwin: *On the Origin of Species* | Mechanismus selekce |
| 1953 | Watson & Crick: struktura DNA | Konečně jsme věděli, co je „recept" |
| 1962 | Lindgren a další: první simulace evoluce v počítači | Že to vůbec jde |
| 1991 | Tom Ray: **Tierra** | Self-replicating digital organisms |
| 1994 | Karl Sims: **Evolved Virtual Creatures** | Vývoj těl + mozků současně |
| 2002 | Stanley & Miikkulainen: **NEAT** | Evoluce topologie neuronové sítě |
| 2011 | Lehman & Stanley: **Novelty search** | Optimalizace bez explicitního cíle |
| 2020 | Mordvintsev et al.: **Growing Neural CA** | Trénované buňky, co umí regenerovat |
| 2021 | Gupta et al. (Stanford): **DERL — Embodied Intelligence** | Velkoškálová evoluce + učení současně |

Každý z těchto milníků má svou vlastní kapitolu nebo sekci dál v dokumentech.

## Zdroje

- [Stanford HAI: Simulating the Evolution of Embodied Intelligence](https://hai.stanford.edu/news/how-bodies-get-smarts-simulating-evolution-embodied-intelligence)
- [Embodied intelligence via learning and evolution (Nature Communications)](https://www.nature.com/articles/s41467-021-25874-z)
- [Why Greatness Cannot Be Planned — Stanley & Lehman](https://link.springer.com/book/10.1007/978-3-319-15524-1)
