# Umělý život (Artificial Life / ALife)

> „Život, ale ne tak, jak ho známe."

Umělý život je obor, který studuje **život jako abstraktní jev**, ne jako konkrétní biochemii. Otázka zní: *Kdyby existovaly úplně jiné chemie nebo úplně jiné planety, jaké společné rysy by všechen život měl?* Nejlepší způsob, jak na to odpovědět, je pokusit se nějakou formu života postavit (v počítači), která **nepoužívá** uhlík, vodu a DNA.

Tohle je přímo srdce projektu Bioscape. Pojďme si projít nejdůležitější existující systémy.

---

## Tierra (Tom Ray, 1991)

**Co to je:** Virtuální počítač, ve kterém běží malé programy. Programy soutěží o čas procesoru a paměť. Mají schopnost se kopírovat (= dělat své vlastní kopie do volné paměti) a občas se při kopírování stane chyba (= mutace).

**Co se stalo:** Tom Ray nasadil jediný „startovní" program (ručně napsaný, dlouhý 80 instrukcí) a nechal to běžet. Z toho jednoho programu se evolucí postupně vytvořily:

- **Kratší programy** (rychlejší replikace)
- **Parazité** — programy, co neumí kopírovat sami sebe, ale ukradnou kopírovací mašinu od jiného
- **Imunita** — hostitelé vyvinuli obranu proti parazitům
- **Hyperparazité** — parazité parazitů
- **Sociální spolupráce** — programy, co se kopírují jen ve skupinách

Všechno tohle vzniklo **bez plánování**. To je úžasné.

**Pro nás důležité poučení:** I šíleně jednoduchý systém (paměť + replikace + mutace) produkuje překvapivě bohatou ekologii. Ale: po nějaké době se evoluce zastaví — programy najdou „dostatečně dobrý" způsob přežití a zůstanou tam.

🔗 [Tierra na Wikipedii](https://en.wikipedia.org/wiki/Tierra_(computer_simulation))

---

## Avida (Adami, Ofria et al., 1998–dosud)

**Co to je:** Pokročilejší následník Tiery. Stále se jedná o samoreplikující se programy v paměti, ale:

- Jsou na 2D mřížce (mají sousedy → ekologie)
- Můžou se učit „triky" — např. provádět logické operace (AND, OR, XOR), za které dostávají energii navíc
- Vědci s tím opravdu testují biologické hypotézy

**Co je na tom slavné:** V roce 2003 Lenski a kol. publikovali v *Nature* studii, kde ukázali, jak v Avidě vznikla schopnost provést **EQU** (logická ekvivalence — relativně složitá operace) ne jako výsledek přímého výběru pro EQU, ale jako *vedlejší produkt* selekce pro jednodušší operace. Stepping stones v praxi.

**Pro nás důležité poučení:** Pokud organismy odměňuješ za **dílčí dovednosti**, můžeš dostat komplexní chování, na které bys přímou selekcí nedosáhl.

🔗 [Avida Digital Evolution Platform](https://alife.org/encyclopedia/digital-evolution/avida/)
🔗 [Avida: Software Platform for Computational Evolutionary Biology](https://www.researchgate.net/publication/232808314_Avida_A_Software_Platform_for_Research_in_Computational_Evolutionary_Biology)

---

## Karl Sims — Evolved Virtual Creatures (1994)

Tohle je naprostá klasika. Pokud jsi viděl ty staré 90s videa, kde se 3D klikaté tvory plouhají vodou nebo bojují o zelenou kostku — to je Karl Sims.

**Co to je:** Sims simuloval **3D fyzikální svět** (gravitace, kontakty, viskozita) a v něm tvory. Každý tvor měl:
- **Tělo** — sada kvádrů spojených klouby. Strukturu těla popisoval **graf** (uzel = část těla, hrana = jak je připojená).
- **Mozek** — neuronová síť uvnitř těla, která řídila svaly v kloubech. Topologie sítě byla také kódována v genomu.

**Klíčová věc:** **tělo i mozek se vyvíjely současně**. Genom byl jeden směrovaný graf, který popisoval rekurzivně, jak se tělo skládá a jak v něm probíhá řízení.

**Co se evolvovalo:**
- Plavání ve vodě — různé strategie (úhoři, krabi, řasy)
- Chůze po zemi
- Skoky
- Sledování cíle (světla)
- Boj o kostku — dva tvoři proti sobě, kdo dřív získá kontrolu

**Pro nás důležité poučení:** **Morfologie a kontrola se musí vyvíjet pohromadě.** Když máš pevné tělo a evolvuješ jen mozek (jak to dělá většina robotiky), zahodíš obrovskou část designového prostoru.

🔗 [Evolved Virtual Creatures — domovská stránka](https://www.karlsims.com/evolved-virtual-creatures.html)
🔗 [Sims 1994 SIGGRAPH paper](https://karlsims.com/papers/siggraph94.pdf)

---

## Lenia (Bert Chan, 2018)

**Co to je:** Spojitá zobecněná verze Game of Life. Místo mřížky 0/1 buněk pracuje se spojitou hodnotou (intenzita) a spojitým časem. Pravidla jsou popsána funkcí růstu/úbytku závislou na vážené sumě okolí.

**Co se objevilo:** Stovky **„životních forem"** — kapičky, které se pohybují, dělí, slučují, rotují, dokonce i organismy s něčím jako symetrií, končetinami a vnitřními „orgány". Vypadá to úchvatně. Některé struktury jsou tak komplexní, že jim Chan dává jména a popisuje je jako biology by popisoval druhy.

**Pro nás důležité:**
- Spojité Cellular Automata jsou mocné medium pro simulaci „virtuální chemie"
- I bez explicitní replikace v nich vznikají perzistující struktury
- Otázka: *Lze v Lenii dosáhnout pravé evoluce?* Je to otevřený výzkum (zatím nikdo nevyrobil verzi Lenie s plnohodnotnou replikací + dědičností + selekcí, co by reflektovalo „skutečnou evoluci")

🔗 [Lenia na Wikipedii](https://en.wikipedia.org/wiki/Lenia)
🔗 [Lenia: Biology of Artificial Life (paper)](https://www.researchgate.net/publication/336712387_Lenia_Biology_of_Artificial_Life)

---

## Polyworld, Geb, Critterding a další

Existuje tucet menších systémů, které stojí za zmínku:

- **Polyworld** (Larry Yaeger, 90s) — 2D svět s tvory, kteří mají neuronové sítě, jí, množí se, útočí. Pěkná experimentální platforma.
- **Geb / Cyberlife Creatures** — komerční hra "Creatures" (90s) měla překvapivě hluboký model neurologie a genetiky.
- **3D Virtual Creatures Evolution (3DVCE)** a podobné implementace Karla Simse, ale v dnešních enginech.

---

## Stanford DERL — Embodied Intelligence via Learning and Evolution (2021)

Toto je čerstvý velký krok dopředu, vyšlo v *Nature Communications*.

**Co to je:** Tým Agrim Gupty (Stanford) postavil masivně paralelní simulaci „unimal" — kloubových tvorů (jako trochu zjednodušení Sims) v 3D fyzice. Trénovali je **kombinací evoluce a reinforcement learningu**:

- Evoluce hraje s **tělem** (morfologií)
- RL učí **mozek** (kontrolu) — neuronová síť se učí během života jednoho jedince
- Naučené chování se nedědí (Lamarck NE), ale dědí se **morfologie, která se naučí rychleji**

**Klíčové výsledky:**
- **Morfologický Baldwin effect:** Evoluce konvergovala k tělům, která se učí rychleji. Co se kdysi muselo „naučit pracně", se nyní v dalších generacích spustí téměř okamžitě (i když to není v genech přímo).
- **Komplexita prostředí pomáhá:** Tvoři vyvinutí v komplexnějším prostředí byli lepší v *všech* úkolech, i v těch jednoduchých.
- **Tělo má smysl pro inteligenci:** Některé „chytré chování" je důsledek pasivní fyziky těla (správně tvarovaná noha sama dopadne stabilně, mozek nemusí počítat).

**Pro Bioscape:** Tohle je možná nejbližší existující práce k cíli projektu. Klíčové architektonické rozhodnutí: **odděl, co se evolvuje (pomalu, mezi generacemi), od toho, co se učí (rychle, v rámci života)**.

🔗 [Embodied intelligence via learning and evolution — Nature Communications](https://www.nature.com/articles/s41467-021-25874-z)
🔗 [Stanford HAI write-up](https://hai.stanford.edu/news/how-bodies-get-smarts-simulating-evolution-embodied-intelligence)

---

## Co si z této kapitoly odnést

1. **Přímá replikace v paměti funguje** (Tierra, Avida) → vznikne ekologie, ale narazí na strop
2. **Vývoj těla + mozku současně je nutnost** (Sims) → bez toho zahazuješ design space
3. **Spojitý prostor je mocný, ale neevolvuje se sám** (Lenia) → potřebuje strukturu navíc
4. **Kombinovat evoluci a učení během života** (DERL) → nejmodernější přístup
5. **Komplexní prostředí > komplexní fitness funkce** → nech bohatství světa dělat selekci
