# Neurony a mozek

## Neuron, jak ho zná každý

Mozek se skládá z buněk zvaných **neurony**. Každý neuron má:

- **Dendrity** — krátké výběžky, kterými přijímá signály od jiných neuronů
- **Tělo (soma)** — kde se signály sčítají
- **Axon** — dlouhý výběžek, kterým posílá signál dál
- **Synapse** — místa kontaktu axonu jednoho neuronu s dendritem druhého

Když dendrity přijmou dost stimulace, neuron **„vypálí" (fires)** — pošle elektrický impuls (akční potenciál, *spike*) přes axon. Ten impuls projde přes synapse k dalším neuronům, které mohou taky vypálit, atd. Tomuto se říká **šíření aktivity**.

Síla synapse (jak moc impuls od neuronu A ovlivní neuron B) se mění s časem podle aktivity. Tomu říkáme **synaptická plasticita** a je to základ učení a paměti.

---

## Tři úrovně modelu neuronu

V počítači můžeš modelovat neuron na různých úrovních detailu. Zde je trade-off **biologická věrnost** vs. **výpočetní cena**.

### 1. Perceptron (umělý neuron — nejjednodušší)

```
output = activation(sum(weight_i * input_i) + bias)
```

To je celý moderní deep learning. Žádný čas, žádné spiky, jenom čísla.

**Plus:** Šíleně rychlé, paralelizovatelné, gradient-friendly.
**Mínus:** Není to, co dělá biologický mozek. Nezachycuje časovou dynamiku, energii, lokální chemii.

### 2. Spiking Neural Networks (SNN) — biologičtější

Místo spojitých čísel produkuje neuron diskrétní spiky v čase. Klasický model:

#### Leaky Integrate-and-Fire (LIF)
Neuron má membránový potenciál, který se hromadí z příchozích spiků a postupně samovolně vyteká pryč. Když překročí práh → spike, reset potenciálu.

#### Hodgkin-Huxley (HH)
Detailní model akčního potenciálu z roku 1952, popisuje iontové kanály diferenciálními rovnicemi. **Nositel Nobelovy ceny.** Velmi přesný, ale výpočetně drahý.

#### Izhikevich (2003) — sweet spot
Dvouvariabilní model, který umí reprodukovat všechny hlavní typy neuronového „firingu" s podobnou věrností jako Hodgkin-Huxley, ale **mnohem levnější**. Konkrétně:

> *"Tens of thousands of spiking cortical neurons in real time on a desktop PC"* (citace z [Izhikevich's paper](https://www.izhikevich.org/publications/spikes.pdf))

**Pro Bioscape je Izhikevich pravděpodobně nejlepší volba**, pokud chceme jít cestou spiking sítí.

🔗 [Simple Model of Spiking Neurons (Izhikevich 2003)](https://www.izhikevich.org/publications/spikes.pdf)
🔗 [Which Model to Use? (Izhikevich 2004)](https://www.izhikevich.org/publications/whichmod.pdf)

### 3. Detailní biofyzika (NEURON simulator a podobné)

Modelují morfologii dendritů, distribuci kanálů, atd. Pro detailní výzkum konkrétních neuronů, ale pro evoluci stovek tisíc agentů nepoužitelné.

---

## Proč by tě měly zajímat spiking sítě

V deep learningu vyhrává perceptron, takže přímá otázka: **proč v Bioscape přemýšlet o SNN?**

Důvody:

### 1. Energetická efektivita
SNN jsou **sparsní v čase** — neuron je tichý většinu času, posílá spike jen občas. Real biological brain spotřebuje ~20 W. GPT-4 spotřebuje řádově **víc**. Jakmile budeme provozovat statisíce agentů paralelně, tohle začne hrát roli.

### 2. Časová dynamika
Spiky mají timing. Vznikají rytmy, oscilace, fázové vztahy. **Časová informace** je v biologii klíčová pro vnímání rychlosti, predikci, koordinaci pohybu. Perceptron tohle úplně ztrácí (musí se to obejít rekurencí).

### 3. Lokální učící pravidla
SNN se přirozeně učí přes **STDP (spike-timing dependent plasticity)** — synapse se posílí, když pre-synaptický neuron spike-uje krátce **před** postsynaptickým. To je čistě lokální pravidlo, žádný backpropagation, žádný globální cílový signál.

To je obrovská výhoda pro evoluci v simulaci: nepotřebuješ propagovat fitness gradient přes celou síť, jenom necháš lokální plasticity běžet.

### 4. Hardware konzistence
Neuromorphic hardware (Intel Loihi, IBM TrueNorth, SpiNNaker) je optimalizovaný pro SNN. Potenciální budoucnost — pokud bys chtěl Bioscape jednou škálovat na speciální hardware.

🔗 [Bio-Inspired Evolutionary SNN (Frontiers)](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2019.01085/full)
🔗 [Improving Izhikevich Model](https://arxiv.org/pdf/1910.11380)

---

## Jak vznikají struktury v mozku

Mozek není homogenní masa. Má vrstvy (cortex 6 vrstev), regiony s různou specializací (zrakový kortex, motorický kortex...), a struktury jako hippocampus, mozkový kmen, atd.

**Jak tohle všechno vznikne z jednoho receptu?** Klíčové mechanismy:

1. **Gradienty morfogenů** — chemické signály vytvářejí v rostoucím embryu „mapy", podle kterých se neurony rozhodují, kým budou
2. **Self-organization přes activity** — neurony, co spíkají společně, se navzájem propojují (Hebbovo pravidlo)
3. **Kompetice o cíle** — víc axonů soutěží o omezenou „synaptickou plochu" na cílových neuronech
4. **Pruning** — zpočátku se vytvoří víc spojení, neúspěšná zaniknou

Tohle je krásně analogické postupům v [04-bunky-a-morfogeneze.md](04-bunky-a-morfogeneze.md). **Mozek je v zásadě další epizoda morfogeneze** — jen místo končetin a orgánů se „kresí" propojení.

---

## Neuromodulace: chemická vrstva nad elektrickou

V biologickém mozku není všechno jen elektrika. Existují **neuromodulátory** — látky jako dopamin, serotonin, noradrenalin, acetylcholin — které **mění globální „režim"** neuronů: jak rychle se učí, jak silně reagují, jak moc explorují vs. exploit.

Pro Bioscape: pokud chceš modelovat něco jako **odměnu, motivaci, pozornost**, neuromodulace je téměř povinná. Klasický perceptron tohle nemá.

---

## Praktické rozhodnutí pro Bioscape

Trade-off matrix:

| Model | Speed | Biologická věrnost | Plasticity | Vhodné pro |
|---|---|---|---|---|
| Perceptron | ⭐⭐⭐⭐⭐ | ⭐ | gradient-only | Když chceš RL/backprop |
| LIF | ⭐⭐⭐⭐ | ⭐⭐⭐ | STDP | Velké populace |
| Izhikevich | ⭐⭐⭐ | ⭐⭐⭐⭐ | STDP | **Sweet spot** |
| Hodgkin-Huxley | ⭐ | ⭐⭐⭐⭐⭐ | STDP | Detail výzkum |

**Doporučení:** začni s Izhikevich + STDP. Pokud bude pomalé, fallback na LIF. Kombinuj s neuromodulací (jednoduchý globální signál „odměna"/„novost") pro modulaci učení.

## Otevřené výzkumné otázky

- Jak nechat **topologii sítě** evolvovat společně s **dynamikou neuronu**?
- Jak modelovat **dlouhodobou paměť** přes generace (epigenetika? zděděná struktura?)
- Lze postavit „protomozek" čistě z buněk Neural CA, kde se některé buňky diferencují na neurony emergentně?

## Zdroje

- [Izhikevich — Simple Model of Spiking Neurons](https://www.izhikevich.org/publications/spikes.pdf)
- [Izhikevich — Which Model to Use](https://www.izhikevich.org/publications/whichmod.pdf)
- [Bio-Inspired Evolutionary Model of SNN](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2019.01085/full)
- [Investigation of SNN with Izhikevich Model](https://www.mdpi.com/2227-7390/10/4/612)
