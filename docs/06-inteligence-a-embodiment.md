# Inteligence a embodiment (proč mozek potřebuje tělo)

## Co vůbec je inteligence?

Nikdo nemá perfektní definici. Ale několik užitečných:

- **Schopnost dosahovat cílů v širokém spektru prostředí** (Legg & Hutter, 2007)
- **Schopnost komprimovat zkušenost a generovat predikce** (Marcus Hutter, AIXI)
- **Adaptivní chování pod nejistotou** (Friston a další)
- **Schopnost transferu** — naučit se rychle něco nového na základě staré zkušenosti

V biologii je to spíš škála než binárka. Bakterie mají primitivní formy inteligence (chemotaxe — pohyb směrem k jídlu). Hmyz má víc. Savci ještě víc. Lidé jsou na špici toho, co známe.

**Pro Bioscape je důležitý poznatek:** inteligence se v evoluci **neobjevila naráz**. Vznikla jako sekvence stepping stones, kde každá úroveň otevřela prostor pro další.

---

## Embodiment: hypotéza, že tělo je nutnost

**Embodiment hypothesis (ztělesněná kognice)** říká, že inteligence **nemůže být chápána izolovaně od těla a prostředí**. Mozek není počítač, který si můžeš vyjmout a spustit jinde — je to **regulátor těla**, který bez svého těla a vstupů ze světa není ničím.

### Klíčové argumenty pro embodiment

#### 1. Mozek se vyvinul pro pohyb

Sea squirt (sasanka mořská) má v larvální fázi mozek a plave. Když dospěje, **přisedne k podkladu a sní svůj vlastní mozek** — protože ho už nepotřebuje. To není legrace, to je literálně to, co dělá.

Hluboký poznatek: **mozky existují kvůli pohybu.** Bez nutnosti rozhodovat o akci by celá struktura nervové soustavy byla zbytečná. Statické rostliny nepotřebují mozek.

#### 2. Tělo je „outsourced computation"

Spousta „chytrého chování" není v mozku, ale v **mechanice těla**:
- Pasivní chodci dokážou jít z kopce **bez motorů a bez kontroleru** — jejich kinematika to dělá za ně
- Hmyzí krky se otáčejí stabilně díky **mechanické struktuře**, ne neuronové kontrole
- Lidská chůze využívá pružnosti šlach — mozek dává hrubé povely, šlachy „dořeší" detaily

To Stanford DERL (viz [02-umely-zivot.md](02-umely-zivot.md)) přímo demonstruje: evoluce **vybírá tělaalternati která usnadňují učení**, ne nejvýkonnější mozky.

#### 3. Senzomotorické smyčky utvářejí myšlení

Naše abstraktní pojmy mají kořeny v tělesné zkušenosti:
- „Nahoru" = dobré (nahoru rosteme, vyhráváme), „dolů" = špatné
- „Teplý" mezilidský vztah, „chladný" člověk
- „Chápu" = chytám, držím rukou

Lakoff & Johnson, *Metaphors We Live By* — tahle metaforická vrstva není jen lingvistický trik, je to ukotvení abstrakce v sensorimotorice.

🔗 [Embodied intelligence via learning and evolution (Nature)](https://www.nature.com/articles/s41467-021-25874-z)
🔗 [Embodied Intelligence on PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8494941/)

---

## Free Energy Principle (Karl Friston)

Tohle je velká moderní teorie, která se snaží unifikovat **vnímání, akci a učení** pod jediným principem. Je dost matematicky náročná, ale myšlenka je krásná:

### Hlavní teze

**Každý živý systém minimalizuje překvapení (surprise)** — rozdíl mezi tím, co očekává, a tím, co vnímá.

To může dělat dvěma způsoby:

1. **Aktualizací svého modelu světa** (vnímání, učení) — „aha, svět je jiný než jsem myslel"
2. **Akcí ve světě** (chování) — „udělám něco, aby svět odpovídal mému modelu"

Tomuto se říká **active inference** — aktivní inference. Vnímání a akce jsou dvě strany téže mince.

### Příklad

Jsi hladový. Tvůj model světa říká „brzy budu jíst". Realita říká „nejím". Free energy je vysoká.

- **Perception update:** „aha, asi nepřijde jídlo, musím revidovat očekávání" (učení o dlouhých prodlevách)
- **Action:** „jdu si pro jídlo" (změním realitu, aby odpovídala modelu)

Mozek si volí cestu menšího odporu (přibližně). Z tohoto plyne celé chování.

### Pro Bioscape

Free Energy Principle nabízí **vnitřní motivaci** pro agenty bez explicitní fitness funkce. Agent je „úspěšný", když dobře predikuje svět → přežije, množí se. Zde je krásné napojení na evoluci: **lepší prediktivní model = lepší fitness.**

To může být způsob, jak elegantně spojit „mozek" a „chování" v jediném principu.

🔗 [Free Energy Principle — Wikipedia](https://en.wikipedia.org/wiki/Free_energy_principle)
🔗 [Active Inference: The Free Energy Principle (book, MIT Press)](https://direct.mit.edu/books/oa-monograph/5299/Active-InferenceThe-Free-Energy-Principle-in-Mind)
🔗 [Friston's "Rough Guide to the Brain"](https://www.fil.ion.ucl.ac.uk/~karl/The%20free-energy%20principle%20-%20a%20rough%20guide%20to%20the%20brain.pdf)
🔗 [From Neuroscience to AI: Free Energy Principle](https://www.researchgate.net/publication/397380587_From_Neuroscience_to_Artificial_Intelligence_Karl_Friston's_Free_Energy_Principle_and_the_Rise_of_Active_Inference)

---

## Evoluce a inteligence: Baldwin effect

**Baldwin effect** je pojmenovaný po Jamesi Marku Baldwinovi (1896) a popisuje vztah evoluce a učení.

### Idea

Učení samo o sobě se *nedědí* (chování naučené v životě se nepředá potomkům — to by byl Lamarckismus, který v přírodě nefunguje). Ale **schopnost se rychleji učit konkrétní věc se dědit může**.

### Mechanismus

1. Agenti, kteří se naučí prospěšné chování, přežijí lépe
2. Mezi nimi přežijí ještě lépe ti, kteří se to **naučí rychleji**
3. Mezi nimi pak ti, kteří mají **vrozenou predispozici** k rychlému naučení
4. Časem se chování přesune z „naučeného" k „částečně vrozenému"

### Stanford DERL: morfologický Baldwin

DERL paper ukazuje variantu: nejen mozek, ale i **morfologie** (tvar těla) se evolvuje tak, aby usnadnila učení. Agent s pružnýma nohama se naučí chodit rychleji než agent s prkennými, takže evoluce vybírá pružné nohy.

To je extrémně relevantní pro Bioscape: **necháme tělo a kontrolu evolvovat současně, ale plasticitu během života trénujeme RL nebo STDP**.

---

## Co dělá nějaké prostředí „stimulujícím" pro vznik inteligence?

Tohle je v podstatě otázka, kterou musíme vyřešit pro Bioscape **jako designové rozhodnutí**.

### Vlastnosti, co podle výzkumu pomáhají

1. **Bohatá smyslová modalita** — zrak, sluch, hmat. Víc kanálů → víc důvodů zpracovávat informace.
2. **Pohyblivost** — agent musí volit akce. Pasivní pozorovatel intelligenci nepotřebuje.
3. **Variabilita** — prostředí, co se mění, vyžaduje **adaptaci**, ne jen optimalizaci.
4. **Sociální interakce** — jiní agenti jsou nejtěžší a nejbohatší stimulus. Predátoři, kořist, soukmenovci, soutěž o partnery.
5. **Dlouhý časový horizont** — odměny vzdálené v čase od akcí vyžadují plánování.
6. **Sparse rewards s občasnou kořistí** — pokud je všechno snadné, stačí instinkt; pokud je všechno nemožné, vymře. Edge of chaos.

### „Sociální mozek" hypotéza

Velikost mozku u primátů koreluje **s velikostí sociální skupiny**, ne s velikostí teritoria nebo komplexitou stravy. Hypotéza Robina Dunbara: **mozek se vyvinul hlavně proto, aby si poradil s ostatními mozky.**

Pro Bioscape: **multi-agent prostředí může být důležitější než realistická fyzika.**

---

## Praktické důsledky pro design Bioscape

1. **Tělo a mozek vyvíjet pohromadě** (Sims, DERL — viz [02-umely-zivot.md](02-umely-zivot.md))
2. **Bohaté multi-agent prostředí** s pohybem, jídlem, predátory
3. **Žádná explicitní fitness „buď chytrý"** — selekce přes přežití
4. **Možná free-energy / active-inference jako vnitřní motivace** agenta
5. **Plasticita během života** (učení) + selekce mezi generacemi (evoluce)

## Otevřené otázky

- Vznikne v simulaci sociální chování spontánně, nebo je nutné ho podpořit designem prostředí?
- Lze měřit „inteligenci" populace bez explicitního testu? (Komprese, novelty produkce, behaviorální komplexita)
- Free-energy v praxi: jak ji efektivně počítat pro tisíce agentů na GPU?

## Zdroje

- [Embodied intelligence via learning and evolution (Nature Communications)](https://www.nature.com/articles/s41467-021-25874-z)
- [Stanford HAI write-up](https://hai.stanford.edu/news/how-bodies-get-smarts-simulating-evolution-embodied-intelligence)
- [Active Inference (MIT Press, open)](https://direct.mit.edu/books/oa-monograph/5299/Active-InferenceThe-Free-Energy-Principle-in-Mind)
- [Free Energy Principle for Perception and Action (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8871280/)
