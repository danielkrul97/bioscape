# Neuroevoluce: jak nechat evoluci postavit mozek

## Co to vůbec je

**Neuroevoluce = použití evolučních algoritmů k vytváření a vylepšování neuronových sítí.**

Místo backpropagation a gradient descent (jak se to dělá v normálním deep learningu) se neuronovky vyvíjejí pomocí mutací a selekce. Zní to možná pomalu — a často to *je* pomalejší — ale má to několik unikátních předností:

1. **Můžeš evolvovat i topologii sítě**, ne jen váhy
2. **Funguje to bez gradientu** — i tam, kde je fitness drsně diskrétní (organismus přežije nebo ne)
3. **Vyhne se to lokálním minimům** lépe než gradient
4. **Může být propojené s evolucí těla** — což je pro Bioscape klíčové

## Klasický postup

Každý jedinec v populaci je **neuronová síť** (různě reprezentovaná). Fitness se měří chováním v prostředí. Mutace mění váhy nebo strukturu. Selekce, crossover, opakovat.

Hlavní rozhodnutí: **jak zakódovat síť do genomu?**

---

## NEAT (Stanley & Miikkulainen, 2002)

**NEAT = NeuroEvolution of Augmenting Topologies**

Toto je kanonický algoritmus neuroevoluce. Pokud jsi měl číst jediný paper z této kapitoly, je to [Stanley & Miikkulainen 2002](https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf).

### Tři klíčové inovace

#### 1. Začni jednoduše, rosti komplexnost postupně

Většina neuroevolučních algoritmů před NEATem startovala s náhodnou velkou sítí. NEAT startuje s **minimální sítí** (žádné skryté neurony, jen vstupy přímo na výstupy) a komplexnost přidává jen tehdy, když ji evoluce „odměňuje".

Proč to funguje: každá nová struktura (neuron, spojení) ze začátku snižuje fitness (síť ji ještě neumí používat). Kdyby se přidávalo všechno najednou, nestihlo by se to vyladit. NEAT přidává po jedné věci a chrání ji, dokud se ladí.

#### 2. Historical markings (gen identifikátory)

Když si dva rodiče vyměňují části sítě (crossover), je problém: jak víš, která spojení v rodiči A „odpovídají" kterému v rodiči B? Topologie se může lišit.

**Řešení:** každé strukturální spojení má **innovation number** — pořadové číslo, kdy v evolučně historii vzniklo. Při crossoveru se geny se stejným číslem párují, ostatní se zdědí od fit rodiče. Elegantní řešení historicky velmi obtížného problému.

#### 3. Speciation (druhy)

Nové struktury jsou zpočátku **horší** než zaběhnuté. Kdybychom je vystavili globální konkurenci, hned by vyhynuly. NEAT proto rozděluje populaci do **druhů** podle podobnosti genomů — jedinec konkuruje hlavně se svými druhovými soudruhy. To dává inovacím čas se vyladit.

### Co NEAT umí

- Vyřešil úlohy, které předtím evoluční algoritmy nezvládaly (double pole balancing)
- Použit v hrách (NERO — agenti, kteří se učí ze zkušenosti hráče)
- Stovky variant a follow-upů

🔗 [Stanley NEAT homepage](https://www.cs.ucf.edu/~kstanley/neat.html)
🔗 [Wikipedia: NEAT](https://en.wikipedia.org/wiki/Neuroevolution_of_augmenting_topologies)

---

## HyperNEAT (Stanley et al., 2009)

NEAT má jeden vážný problém: pro velké sítě (řekněme stovky tisíc neuronů jako v mozku) by genom byl nepoužitelně velký.

**HyperNEAT to řeší pomocí indirektního kódování.** Místo aby genom říkal *„neuron 17 má váhu 0.34 na neuron 42"*, říká *„existuje funkce f(x₁,y₁,x₂,y₂), která pro každé dvě pozice neuronů spočítá váhu mezi nimi"*.

Tato funkce f je sama neuronovka — říká se jí **CPPN (Compositional Pattern Producing Network)**. Co je super: CPPN umí produkovat **regulérní vzory** (symetrie, opakování, gradienty), což jsou přesně ty vzory, které vidíme v mozcích reálných organismů.

### Důsledky

- **Genom je malý**, fenotyp (skutečná síť) je velký
- **Geometrické pravidelnosti** přicházejí přirozeně (levo-pravá symetrie nervové soustavy)
- **Síť lze přeškálovat** — naučenou strukturu můžeš vyrenderovat ve větší rozlišení a často to ještě funguje

### Pro Bioscape

HyperNEAT/CPPN je extrémně relevantní, protože **stejný princip funguje i pro morfologii**. Tělo organismu je v zásadě 3D vzor, který se dá popsat funkcí pozice. Toho využívá kombinovaná evoluce těla + mozku.

🔗 [HyperNEAT: The First Five Years](https://www.researchgate.net/publication/287307129_Hyperneat_The_first_five_years)
🔗 [Stanley CPPN paper](https://axon.cs.byu.edu/~dan/778/papers/NeuroEvolution/stanley3**.pdf)

---

## Novelty Search (Lehman & Stanley, 2011)

Tohle si zaslouží vlastní kapitolu (viz [07-open-ended-evolution.md](07-open-ended-evolution.md)), ale ve zkratce:

**Novelty Search obrací standardní evoluční logiku na hlavu: neoptimalizuje fitness, ale novost.**

Místo aby selektoval jedince podle jejich výkonnosti, selektuje je podle toho, **jak se jejich chování liší od všeho, co bylo už viděno**. Zní to bláznivě, ale na úlohách s podvodným fitness landscape (kde lokální optimum ti brání objevit globální optimum) to často funguje *lépe* než přímá optimalizace.

**Příklad:** robot, co se snaží dostat z bludiště. Standardní fitness = vzdálenost k cíli → robot uvízne v rohu nejbližším cíli, kde je zeď. Novelty search → robot zkouší různé cesty, *protože se liší od předchozích*, a nakonec se dostane k cíli.

---

## Quality-Diversity (Mouret, Clune et al.)

Dvě hlavní rodiny algoritmů:

### Novelty Search with Local Competition (NSLC)
Selekce kombinuje novelty (najdi nové chování) a local competition (ale buď taky dobrý ve své nice).

### MAP-Elites (Mouret & Clune, 2015)

**Idea:** rozděl prostor chování do mřížky buněk. V každé buňce udržuj jen nejlepšího jedince, který do ní spadl. Tím dostaneš mapu „pro každý styl chování zde máš nejlepšího reprezentanta".

**Pro Bioscape mimořádně relevantní.** Místo abychom selektovali jedince globálně podle jediné fitness (a tím konvergovali k jednomu řešení), můžeme udržovat **různorodou populaci** v různých nikách — víc analogické tomu, co se děje v přírodě.

🔗 [Quality-Diversity research overview](https://quality-diversity.github.io/)
🔗 [Quality Diversity: A New Frontier (Pugh, Soros, Stanley)](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2016.00040/full)

---

## Praktická poznámka: kdy zvolit co

| Situace | Doporučený přístup |
|---|---|
| Klasická supervised loss + lots of data | Backprop, ne neuroevoluce |
| RL s spojitými akcemi a malou sítí | Neuroevoluce může konkurovat |
| Vývoj struktury sítě, ne jen vah | NEAT |
| Velké sítě s pravidelnostmi | HyperNEAT / CPPN |
| Otevřená evoluce, žádný explicit fitness | Novelty Search, MAP-Elites |
| Evoluce těla i mozku současně | NEAT-like + tělesné kódování (Sims, DERL) |

**Pro Bioscape pravděpodobně:** kombinace HyperNEAT-style indirektního kódování (mozek + tělo z jedné CPPN) s MAP-Elites pro udržení diverzity. Konkrétně to ale závisí na tom, jakou architekturu prostředí zvolíme — viz [08-implementace-rust-gpu.md](08-implementace-rust-gpu.md).

## Zdroje

- [Stanley & Miikkulainen 2002 — Original NEAT](https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf)
- [HyperNEAT review](https://www.researchgate.net/publication/287307129_Hyperneat_The_first_five_years)
- [Quality-Diversity overview](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2016.00040/full)
- [MAP-Elites Introduction](https://szhaovas.github.io/2022-09-15-me/)
- [Systematic review of NEAT successors](https://www.cse.unr.edu/~sushil/class/gas/papers/evco_a_00282.pdf)
