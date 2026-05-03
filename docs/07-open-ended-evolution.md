# Open-Ended Evolution: jak udělat, aby se evoluce nezasekla

## Problém

Když pustíš genetický algoritmus s pevnou fitness funkcí, **téměř vždy konverguje**. Po stovkách až tisících generací jsou všichni jedinci skoro stejní. Další progress se zastaví.

V přírodě se to **neděje**. Příroda už 3,8 miliardy let neustále produkuje nové druhy, nové schopnosti, nové strategie. Co máme jinak?

Tomuto problému se říká **open-ended evolution (OEE)** — jak nechat evoluci, aby **nikdy nepřestala produkovat něco nového**.

---

## Proč se evoluce v simulacích zastaví

Tři hlavní důvody:

### 1. Pevný cíl
Jakmile máme metric „fitness = něco konkrétního", evoluce konverguje k optimu té metriky. **Co měříme, to dostaneme** — i když to není to, co jsme vlastně chtěli (Goodhartův zákon).

### 2. Pevné prostředí
Když se prostředí nemění, neexistuje tlak na další evoluci. Optimální strategie je nalezena → stop.

### 3. Žádná koevoluce
V přírodě se predátoři přizpůsobují kořisti, která se přizpůsobuje predátorům — **závody ve zbrojení**. Tohle je nikdy nekončící hnací motor. V uzavřené simulaci s pasivním prostředím tohle chybí.

---

## Tierra/Avida: zaseknutí v praxi

Jeden z důvodů, proč Tierra a Avida (viz [02-umely-zivot.md](02-umely-zivot.md)) jsou „klasiky pro učebnice", ale **nezpůsobily revoluci**:

> *„Artificial life systems currently under study, such as Tierra or Avida, if anything, show the opposite trend toward simpler and more highly optimised creatures."* — z [open-ended evolution review](https://direct.mit.edu/artl/article/22/3/408/2841/Open-Ended-Evolution-Perspectives-from-the-OEE)

Místo aby se programy stávaly komplexnější, **stávají se jednoduššími** — najdou nejmenší self-replicator a tam se usadí.

---

## Řešení 1: Novelty Search (Lehman & Stanley, 2011)

**Princip:** Nepoužívej fitness vůbec. Místo toho **odměňuj jedince za to, že se jejich chování liší od všeho předchozího**.

### Implementace

1. Udržuj **archiv** všech zajímavých chování dosud viděných
2. Pro nového jedince vypočítej **vzdálenost** jeho chování k k-nejbližším v archivu
3. Vysoká vzdálenost = vysoké novelty score = vyšší šance na reprodukci
4. Jedinci s opravdu novou chováním se přidají do archivu

### Proč to funguje

V deceptive prostředích (kde gradient fitness vede do slepé uličky) je **explorace zajímavějších oblastí** efektivnější než „lezení nahoru po fitness".

### Slavný příklad

Robot v bludišti, který se má dostat k cíli. Standardní fitness (vzdálenost k cíli) ho zavede do rohu nejbližšího cíli, kde je zeď → uvíznutí. Novelty search ho nutí zkoušet jiné cesty (i ty, co zdánlivě „od cíle vedou pryč"), a nakonec najde celou cestu.

🔗 [Novelty Search Theoretical Perspective](https://hal.science/hal-02561846/file/NS_theory.pdf)

---

## Řešení 2: Quality-Diversity (Mouret, Clune)

Novelty search má slabinu: vede k **diverzitě nesmyslů**. „Tento jedinec dělá něco unikátního, ale je to k ničemu." 

QD algoritmy řeší tohle tak, že **kombinují diverzitu s kvalitou**: chceš velkou rozmanitost dobrých řešení, ne velkou rozmanitost čehokoli.

### MAP-Elites (Mouret & Clune, 2015)

**Idea:** rozděl behaviorální prostor do mřížky buněk. V každé buňce udržuj **jen toho nejlepšího** jedince, který do ní spadl.

```
Příklad: Vyvíjím chodící robot.
Behaviorální dimenze: výška × rychlost
Mřížka 10×10 = 100 buněk

V každé buňce ten nejvýkonnější (e.g. nejstabilnější, nejvíc 
energeticky efektivní) robot s dotyčnou kombinací výšky/rychlosti.

Výsledek: 100 různých robotů, každý expert ve své nice.
```

### Proč je to mocné

1. **Mapuješ celý design space**, nejen vrchol
2. **Stepping stones jsou zachované** — slabší řešení v jiné nice mohou být *předkové* skvělých řešení v další nice
3. **Robust k mutaci** — diverzita je vestavěná
4. **Pro Bioscape ideální**, protože nás zajímají různé strategie/morfologie/inteligence, ne jeden „vítěz"

🔗 [MAP-Elites Introduction](https://szhaovas.github.io/2022-09-15-me/)
🔗 [Quality-Diversity overview](https://quality-diversity.github.io/)
🔗 [Quality Diversity: A New Frontier (Pugh, Soros, Stanley)](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2016.00040/full)

---

## Řešení 3: Koevoluce a ekologie

V přírodě je **prostředí samo živé**. Predátoři, kořist, paraziti, hostitelé — všichni se navzájem evolvují. Tohle vytváří **otevřenou aréna**, ze které není „výstup".

### Implementace v simulaci

- **Více populací**, které se navzájem ovlivňují (predátoři vs. kořist)
- **Hall of Fame** — nový jedinec se měří proti minulým jedincům, ne jen proti aktuální populaci
- **Coevolutionary tournaments** — soutěží proti soupeřům z evolučně různých dob

Karl Sims dělal coevolutionary tournament v jeho 1994 práci — tvoři soutěžili o kostku, což produkovalo **mnohem zajímavější chování** než single-agent fitness.

### Pro Bioscape

Multi-agent prostředí s kořistí + predátorem (s možností evoluce do obou rolí) by mělo poskytnout dlouhotrvající evoluční tlak. Ekosystém je daleko lepší než „úloha".

---

## Řešení 4: Rostoucí prostředí

Co když prostředí samo dělá evoluci? **Zdroje, terén, klima** se mění s časem. To je prakticky to, co se děje na Zemi.

V simulaci to může vypadat jako:

- **Zdroje** docházejí v lokalitách, vznikají jinde → migrace
- **Klimatické cykly** — ledové doby přicházejí a odcházejí
- **Geologické změny** — kontinenty se rozdělují, isolované populace se diferencují (allopatrická speciace)

Tohle je drahé výpočetně, ale možná to je nejvyšší koncept, jak se přiblížit přírodním procesům.

---

## „Why Greatness Cannot Be Planned" — Stanley & Lehman

Tato kniha je manifesto open-endedness. Hlavní teze:

> **Velké objevy se neudělají tím, že si stanovíme za cíl je objevit.** Stepping stones k velkým inovacím jsou často nezřejmé a zdánlivě nesouvisející s cílem.

### Příklady z reálu

- **Polovodiče vznikly z výzkumu vakuových trubic, ale ne tím, že někdo „cílil na polovodič"** — vznikly jako side-product výzkumu krystalů.
- **Internet vznikl jako vojenský komunikační projekt**, ne jako „hub pro lidskou kreativitu", což z něj nakonec vzniklo.
- **GPU pro deep learning** — GPU byly stavěny pro hry, jejich užití pro AI je úplně mimo původní záměr.

### Důsledky pro AI/evoluci

- Přímé optimalizace fitness „buď inteligentní" nikdy nebudou inteligentní (deceptive landscape)
- **Hraj si, exploruj, sbírej různorodé úspěchy** — to je jediná cesta
- AI by měla mít vnitřní motivaci k exploraci, ne externí cíl

🔗 [Why Greatness Cannot Be Planned (Springer)](https://link.springer.com/book/10.1007/978-3-319-15524-1)
🔗 [Review of "Why Greatness Cannot Be Planned"](https://engineeringideas.substack.com/p/review-of-why-greatness-cannot-be)

---

## Praktická doporučení pro Bioscape

| Princip | Implementace |
|---|---|
| Žádná explicitní „buď chytrý" fitness | Selekce přes přežití (jídlo, predátor, množení) |
| Diverzita za každou cenu | MAP-Elites nad behaviorálními dimenzemi |
| Koevoluce | Multi-agent prostředí, predátor/kořist |
| Měnící se prostředí | Zdroje, sezóny, klimatické cykly |
| Stepping stones | Archív zajímavých řešení (jako v novelty search) |

---

## Otevřené výzkumné otázky

- Jak měřit, jestli simulace je „opravdu" open-ended? Existující metriky (komplexita genomu, behaviorální entropie, množství druhů) jsou všechny napadnutelné.
- Lze ukázat, že OEE nutně vede ke vzniku inteligence, nebo to je jen nutná podmínka?
- Jaké je minimum prostředí, ze kterého ještě OEE vznikne? (Žádná pravá odpověď zatím není)

## Zdroje

- [Open-Ended Evolution: Perspectives from York Workshop](https://direct.mit.edu/artl/article/22/3/408/2841/Open-Ended-Evolution-Perspectives-from-the-OEE)
- [Open-Ended Artificial Evolution (paper)](https://www.researchgate.net/publication/220606221_Open-Ended_Artificial_Evolution)
- [Why Greatness Cannot Be Planned](https://link.springer.com/book/10.1007/978-3-319-15524-1)
- [Quality-Diversity overview (Pugh et al.)](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2016.00040/full)
- [Novelty Search Theoretical Perspective](https://hal.science/hal-02561846/file/NS_theory.pdf)
