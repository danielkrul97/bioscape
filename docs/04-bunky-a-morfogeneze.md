# Buňky a morfogeneze: jak z bodu vyroste tělo

## Otázka, která je důležitější, než vypadá

Jak je možné, že z **jedné jediné buňky** (oplodněné vajíčko) vyroste tělo, které má pět prstů na ruce, jedny játra (ne dvě a ne nula), a mozek se spoustou částí na správných místech?

Tomu se říká **morfogeneze** a její mechanismus je jeden z nejdůležitějších a stále ne úplně pochopených problémů biologie. **Pro Bioscape je to klíčová otázka,** protože pokud chceme evolvovat těla a mozky, musíme mít nějaký mechanismus, kterým z jednoduchého genomu vyroste komplexní organismus.

---

## Cellular Automata (CA): historicky první přístup

### Game of Life (Conway, 1970)

Asi nejznámější příklad. Mřížka buněk, každá buňka je živá nebo mrtvá. V každém kroku:

- Živá buňka přežije, pokud má 2 nebo 3 živé sousedy
- Mrtvá buňka oživne, pokud má přesně 3 živé sousedy
- Jinak buňka umírá / zůstává mrtvá

Z těchto **čtyř pravidel** vznikne fascinující diverzita: stabilní tvary, oscilátory, putující „glidery", vznikají kanóny střílející glidery, postupně i Turingovsky úplné konstrukce. **Game of Life je Turing-complete** — můžeš v něm postavit počítač.

**Pro nás důležité:** komplexnost neporochází z komplexity pravidel, ale z **iterace jednoduchých pravidel s prostorovou interakcí**. Tohle je naprosto fundamentální poznatek.

🔗 [Conway's Game of Life — Wikipedia](https://en.wikipedia.org/wiki/Conway's_Game_of_Life)
🔗 [Turing-completeness of GoL (proof)](https://theoremoftheday.org/LogicAndComputerScience/Life/TotDLife.pdf)

### Lenia (Bert Chan, 2018)

Spojitá zobecněná verze Game of Life — viz kapitola [02-umely-zivot.md](02-umely-zivot.md). Důležité: Lenia ukazuje, že CA nemusí být binární (živý/mrtvý), pravidla mohou pracovat se spojitými stavy, a vzniklé struktury mohou být neuvěřitelně organické.

---

## Neural Cellular Automata (Mordvintsev et al., 2020)

**Tohle je možná nejdůležitější paper v této kapitole.** Distill paper „Growing Neural Cellular Automata" autorů Mordvintsev, Randazzo, Niklasson, Levin (ano, Michael Levin) je fenomenálně srozumitelně napsaný a interaktivní.

### Idea

Vezmi cellular automaton, ale místo ručně psaných pravidel **nech update rule, aby byl neuronovka.** Trénuj ji backpropagation tak, aby:

1. Počínaje jednou „seed" buňkou
2. Po opakovaném aplikování pravidla
3. Vyrostl předem daný tvar (např. obrázek emoji ještěrky)

A k tomu **navíc** trénuj síť, aby:

4. Po **poškození** (vystřihni z obrázku půlku) se sám zase zregeneroval

### Co je na tom úžasné

- **Pravidlo je pouze lokální** — buňka zná jenom své okolí (3×3)
- **Žádný externí controller** — není žádný „shora dirigent", který by říkal každé buňce, co má dělat
- **Robustní** — poškozený tvar se opraví, protože každá buňka „ví" (skrze geometrii sousedů), kde v cílovém tvaru je
- **Dá se evolvovat** — i když původní paper používal gradient descent, princip je plně kompatibilní s evolucí

### Pro Bioscape

Tohle je možná nejlepší známý přístup k **vývojovému kódování těla**. Genom = neuronovka popisující update rule. Fenotyp = výsledný tvar po N krocích simulace. Mutace = změna vah neuronovky. **Naprosto evolvable.**

Navazující práce:
- **Growing Isotropic NCA** (2022) — invariance vůči rotaci
- **Growing Steerable NCA** (2023) — externí signály (jako hormony) řídí vývoj

🔗 [Distill — Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/)
🔗 [Notes on Growing NCA (Hugo Cisneros)](https://hugocisneros.com/notes/mordvintsevgrowingneuralcellular2020/)
🔗 [Growing Isotropic NCA](https://direct.mit.edu/isal/proceedings/isal2022/34/65/112305)

---

## Michael Levin: bioelektrika a kognice na úrovni buněk

Tohle je možná nejvíc mind-blowing výzkum v celé této oblasti. Michael Levin (Tufts) tvrdí — a má pro to silnou empirickou evidenci — že **buňky komunikují elektrickou aktivitou nejen v mozku, ale po celém těle**, a tato komunikace dělá rozhodnutí o tom, jak má organismus vypadat.

### Klíčová tvrzení Levinovy laboratoře

#### 1. Bioelektřina je univerzální komunikační vrstva

Všechny buňky (ne jen neurony) mají **membránový potenciál** a **iontové kanály**. Buňky si přes tyto signály předávají informace o tom, jaká část těla mají vyrostnout. Levin to nazývá **bioelektrickou „cognitive glue"**.

#### 2. Pattern memory je bioelektrická

Když planárii (zploštělou červa) rozřežeš na 3 kusy, každý kus regeneruje celé tělo — s hlavou na správné straně, ocasem na druhé. Levin ukázal, že **paměť** o tom, kde má být hlava, není v genech ani v lokální chemii — je v **bioelektrickém vzoru** rozprostřeném po celém těle.

Manipulace tohoto vzoru (drogy, optogenetika) způsobuje:
- Planárie s **dvěma hlavami**
- **Žáby s extra očima** (na zádech, v břiše)
- **Změnu druhové paměti** — manipulovaný organismus si „pamatuje" jinou anatomii

#### 3. Xenoboti

V roce 2020 Levin ve spolupráci s Joshuou Bongardem publikoval **xenoboty** — „živé roboty" sestrojené z buněk frog skin. Co je zajímavé:

- **Design xenobotů byl evolvován v simulaci** (evoluční algoritmus + simulace fyziky), pak postaven biology
- Xenoboti sami *od sebe* dělali věci, které nikdo neprogramoval — pohybovali se, sdružovali, dokonce **vykazovali primitivní replikaci** (nahrabávali si do sebe další buňky)

### Pro Bioscape

Levinova práce naznačuje, že **kontrola morfogeneze má hlubokou vrstvu, která je v zásadě „proto-kognitivní"** — buňky si zjednodušeně řečeno „vyjednávají" o tom, kdo bude co dělat.

To otevírá zajímavou možnost pro design:
- Místo oddělit „tělo" a „mozek" jako samostatné struktury,
- Zacházet s **každou buňkou jako s malou výpočetní jednotkou s vlastním stavem**,
- Komunikace mezi buňkami a vznik vyšších struktur (orgány, mozek) jako emergent property.

🔗 [Michael Levin — Wikipedia](https://en.wikipedia.org/wiki/Michael_Levin_(biologist))
🔗 [Bioelectric networks: cognitive glue paper](https://www.researchgate.net/publication/370899618_Bioelectric_networks_the_cognitive_glue_enabling_evolutionary_scaling_from_physiology_to_mind)
🔗 [Levin Lab publications](https://drmichaellevin.org/publications/)
🔗 [Levin profile in The Biologist (xenobots)](https://thebiologist.rsb.org.uk/biologist-features/professor-michael-levin-interview)

---

## Praktické poznatky pro Bioscape

### 1. Buňka jako základní výpočetní jednotka
Modelovat individuální „buňky" jako agenty s:
- Vnitřním stavem (vektor čísel)
- Lokální komunikací (sousedi)
- Update rule (sdílená neuronovka — všechny buňky stejná)

### 2. Tělo = stabilní vzor v dynamickém systému
Místo „nakresli tělo" má smysl modelovat tělo jako **rovnovážný atraktor** procesu. To dává automatickou regeneraci a robustnost vůči mutacím.

### 3. Diferenciace přes lokální signály
Buňky se „rozhodnou" být kůží, svalem nebo neuronem na základě svého lokálního prostředí. **Stejný genom, různé chování** — to je klíč k mnohobuněčnosti.

### 4. Evoluce update rule, ne tvaru
Genom kóduje *pravidla*, ne *výsledek*. To je obrovsky výpočetně efektivní (genom může být malý) a evolvable (mutace pravidel je gradient-friendly).

## Otevřené otázky (zajímavé pro rešerši dál)

- Lze v jediném prostředí evolvovat **i replikaci**, ne jen tvarování? (V přírodě vajíčko + spermie + vývoj. V Neural CA zatím jen vývoj.)
- Jak modelovat **metabolismus** — zdroje, růst, smrt — aniž by simulace nebyla šíleně pomalá?
- Lze nějak spojit Lenia (chemie) s Neural CA (vývojové signály)?

## Zdroje

- [Mordvintsev et al. — Growing Neural Cellular Automata (Distill)](https://distill.pub/2020/growing-ca/)
- [Conway's Game of Life](https://en.wikipedia.org/wiki/Conway's_Game_of_Life)
- [Lenia](https://en.wikipedia.org/wiki/Lenia)
- [Levin Lab](https://drmichaellevin.org/)
- [Bioelectric networks paper](https://www.researchgate.net/publication/370899618_Bioelectric_networks_the_cognitive_glue_enabling_evolutionary_scaling_from_physiology_to_mind)
