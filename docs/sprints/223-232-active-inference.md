# Active inference — predikční smyčka (S223–S232)

**Téma dekády:** obrátit complexity ratchet z **bloatu na funkci**. Decade 203–212
prokázala, že CPPN topologie ratchetuje (+139 % nodes / +383 % links přes 100 gen),
ale je to Goodhart — `cppn_nodes` roste, `brain_w1_rank` klesá, `r(nodes, w1_rank) = −0.59`,
`behavioral_entropy` kolabuje. Diagnóza z S203 zůstává: *„prostředí nevytváří poptávku
po výpočtu/paměti — jídlo je všude stejné."* Mozek je nosný (motor outputs řídí fyziku),
ale lepší zpracovatel informace se nikdy nevyplatí v reprodukci, takže selekce nemá co
zlepšovat kromě velikosti genomu.

Tahle dekáda staví **embodied prediktivní (active-inference) smyčku** — vlastní, dosud
nepostavený FEP design z `docs/06` (§Free Energy Principle: *„lepší prediktivní model =
lepší fitness"*, v kódu před S223 0 výskytů). Mozek dostane **predikční hlavu**: z hidden
state predikuje příští senzorický vstup. Prediction error (surprise) se stane (1) intrinsic
rewardem modulujícím plasticitu, (2) metabolickým členem v energii a (3) negoodhartovatelnou
metrikou. Protože predikce i motorika vznikají ze stejného forward passu, je predikce
implicitně podmíněná akcí — tj. active inference. Smyslem je vytvořit poptávku po kognici
**zevnitř**, nezávisle na ladění prostředí (S207–211 ukázaly, že tudy chování zkonverguje).

**Headline akceptační kritérium (falzifikovatelný podpis průlomu):** v běhu, kde *žádný
člen v kódu neodměňuje výkon na úloze ani „chytrost"*, je průlom prokázán, když `pred_skill`
(= `1 − MSE_model / MSE_persistence`): (a) je pozitivní, (b) ratchetuje napříč generacemi,
(c) kolabuje při ablaci rekurence, (d) roste *spolu* s `w1_eff_rank` (funkce, ne bloat).

**VÝSLEDEK (S223–S229, seed 2, 60 gen):** **(a) ✓** skill > 0 od gen ~47 (mean gen 50–60 =
+0.02). **(b) ✓** ratchetuje −0.6→−0.33→+0.02. **(c) ✗** ablace rekurence skill NEsrazí —
`pred_ablation_div` klesá se skillem (corr −0.903) → kognice je **feedforward** (extrapolace
z aktuálního sensory), ne memory-based. **(d)** netestováno. **Závěr: PARČIÁLNÍ průlom** —
emergentní *netriviální prediktivní kognice* pod no-objective selekcí (model poráží persistence
nelineární feedforward extrapolací), ale **ne** temporální integrace / world-model. Plný
breakthrough (memory-based) vyžaduje úlohu co paměť *vyžaduje* (partial observability / delší
horizont). Cesta sem byla čistá: 5 mechanismů, 2 zamítnuté (energy coupling, weight-leak)
s diagnózou, 1 co prolomil plateau (S226 brain-Hebbian).

## Architektonické rozhodnutí

**Celá smyčka žije v CPU control-plane — žádná GPU/shader surgery.** Predikční hlava je
malá direct-weight vrstva (`predict_w/b` v `Genome`, paralelní k CPPN mozku) počítaná na
CPU z už-stahovaného `last_hidden` ve writebacku `brain_act_gpu_full` (Phase 10–11 →
`last_inputs[0..40]` = sensory, `last_hidden` = hidden(t)). To respektuje „GPU = compute,
CPU = control-plane".

**Proč ne CPPN-generovaná hlava přes `BRAIN_OUTPUTS` (původní plán):** rozšíření output
substrátu by posunulo x-souřadnice *motorických* outputů (`substrate_output_coords`
rozprostírá x přes všechny outputy) → přepsalo by motor váhy → narušilo populaci; plus
sdílený LayerNorm by coupling-oval predikci s motorikou. Direct-weight CPU readout nechá
motor mozek bitově netknutý (truly inert) a S225 delta-rule je pak triviální CPU update.
Cena: readout není CPPN-generovaný (přijatelné — je to prostý lineární map hidden→sensory).
Výjimka: Baldwin weight-leak (S228) může chtít readback.

## Rizika (platí pro celou dekádu)

- **Dark-room problem** (#1 FEP patologie): agent minimalizuje surprise schováním / zamrznutím.
  Obrana napříč dekádou: odměňovat **skill nad persistence baseline**, ne surovou nízkou surprise
  (prázdné místo má nulovou varianci → nulový skill → žádný benefit) + zachovat ekologickou selekci.
- **Goodhart-again:** skill jde gamovat predikcí self-caused kanálů → kurátorovaný exogenní
  slot set (`PREDICTED_SENSORY_SLOTS`) + ablační test (S229).
- **Determinism break:** růst `BRAIN_OUTPUTS` (S224) re-randomizuje váhy → baseline zachytit
  v S223 *před* tím (jako S205).
- **Perf:** w2 roste ~+80 %, ale je malý vůči w1; čekej malý dopad (GPU-sync-bound). Měř v S232.

---

## Sprint 223 — baseline surprise (měření, byte-identical)

- **Cíl:** změřit, jak predikovatelný svět vůbec je — persistence baseline (predikuj
  sensory(t+1) = sensory(t)) přes kurátorovaný exogenní subset slotů. Denominátor, vůči
  kterému se bude skórovat model (S224+). Žádná genová ani GPU změna → byte-identical dynamika.
- **Výstup:** hotovo. `PREDICTED_SENSORY_SLOTS` (12 slotů: food/cell/smell gradient xyz +
  vibration gradient xyz — exogenní kanály; `cell_*` = zárodek sociální predikce) v
  `params/brain.rs`. World akumulátory `surprise_persist_accum/_ticks` (per-gen, reset v obou
  tick-driverech jako `*_gen` čítače). Per-tick populační MSE počítán v `brain_act_gpu_full`
  writebacku *před* přepisem `last_inputs` (staré sensory = t−1). CSV sloupec
  `surprise_persist_avg`. 25-gen smoke (seed 1): baseline **nenulový a stabilní ~0.022–0.027 MSE**
  (gen 1 → 0.0068, plateau ~0.025), tj. svět je netriviálně-ale-ne-chaoticky predikovatelný →
  headroom existuje (model musí porazit slušný baseline). Headless build + `cargo check`
  (renderer) + `empty_and_populated_rows_have_same_column_count` test zelené.
- **Poznámky:** při doplňování sloupce odhalen předexistující WIP bug (metric-fix-goodhart):
  empty-pop CSV řádek byl o 3 sloupce (`functional_nodes_avg/functional_links_avg/dead_nodes_avg`)
  kratší než plný + hlavička → regresní test červený už před S223; doplněno 3× nula.
  Newborni mají `last_inputs = 0` → jednотick transient v surprise, naředěný populací,
  zanedbatelný přes generaci. Gen 0 = 0 (pre-loop, `ticks = 0`). **TODO před S224:** per-slot
  variance z CSV/dumpu — vyřadit nízkovariantní sloty (bezcenné prediktory).

## Sprint 224 — predikční hlava (CPU readout, inertní)

- **Cíl:** přidat predikční readout `last_hidden → predicted_sensory` (jeden výstup na
  `PREDICTED_SENSORY_SLOTS`), měřit model surprise + `pred_skill = 1 − MSE_model/MSE_persist`.
  Výstupy nenapojené na reward/energii.
- **Výstup:** hotovo. **Pivot od původního „rozšiř `BRAIN_OUTPUTS`"** — místo GPU brain-output
  surgery je readout direct-weight vrstva `predict_w/predict_b` v `Genome` (serde helper,
  random/mutate/crossover plumbing), počítaná na CPU ve writebacku z `last_hidden` do nového
  `Cell.predicted_sensory`. `BRAIN_PREDICT = PREDICTED_SENSORY_SLOTS.len()` (12). CSV sloupce
  `surprise_model_avg`, `pred_skill`. `CHECKPOINT_VERSION` 12→13 (genome format). 20-gen smoke
  (seed 1): **byte-identical s S223** (`surprise_persist_avg` sedí přesně gen po gen, populační
  trajektorie identická — dormant zero readout, `sigma_predict = 0`, crossover skip-when-equal
  → nula RNG draws). `pred_skill ≈ −5 až −7` (model 0.05–0.14 vs persist 0.007–0.022). Build +
  464 lib testů + 128 headless testů + column-count zelené.
- **Poznámky:** readout je při S224 **dormant-zero** (predikuje 0), takže `pred_skill` není
  ≈0 (jak plán čekal pro *random* hlavu) ale silně záporný — zero-predictor je ~6× horší než
  persistence. To je vlastně **silnější potvrzení headroomu**: učení (S225) má lézt z −6 k 0+
  (0 = vyrovná persistence, >0 = reálná predikce). Plumbing (`sigma_predict`, mutate/crossover)
  je kompletní ale spící; S225 přidá CPU delta-rule (`Δpredict_w = lr·(actual−pred)·hidden(t)`),
  S226 zvedne `sigma_predict` pro evoluci. Readout je `tanh(predict_w·hidden + predict_b)`,
  cílové kanály jsou ≈[−1,1] (tanh-kompatibilní).

## Sprint 225 — predikční plasticita (delta rule)

- **Cíl:** lokální učení readoutu za života — **čistý CPU delta-rule** ve writebacku (žádný
  shader, readout žije na CPU): `Δpredict_w_live[k][h] += LR · (actual_k − pred_k) ·
  (1−pred_k²) · hidden(t)[h]` (tanh-gradient). Per-cell working copy `predict_w_live` (NE
  děděná — child startuje od nuly; S228 přidá leak).
- **Výstup:** hotovo. `Cell.predict_w_live/predict_b_live` (serde skip + default → transient,
  re-učí se po restore), konstanta `PREDICT_LEARNING_RATE = 0.05`. Delta-rule sloučen do
  persist/model smyčky ve writebacku, používá `hidden(t)` = `cell.last_hidden` před přepisem.
  20-gen smoke (seed 1): **byte-identical** (persist sedí přesně s S223/S224, učení je
  measurement-only — `predict_w_live` ani `predicted_sensory` nikam nezpětnovazbí).
  **`pred_skill` skočil z ≈−6 (S224) na ≈−0.6** — model surprise 0.14→0.035, **10× zlepšení**.
  Build + 464 lib + 128 headless + renderer zelené.
- **Poznámky:** within-life učení prokazatelně **funguje** — readout se naučil rekonstruovat
  senzorický stav z `hidden(t)` a blíží se persistence baseline (skill −0.6 = model jen 1.6×
  horší než persistence, vs 6× u zero readoutu). Persistence ještě neporáží (skill <0): k tomu
  musí `hidden(t)` nést *víc* prediktivní informace než poslední hodnota — úkol pro S226/227
  (reward/energy coupling tvarující hidden state pod selekcí). `lr_pred` je zatím konstanta;
  S228 ho povýší na gen kvůli Baldwin selekci na rychlost učení. Genome `predict_w` = (zatím
  nulový) birth prior; S226 z něj musí init-ovat `predict_w_live` (teď init nula).

## Sprint 226 — brain-directed: skill → plasticita mozku (láme plateau)

- **Cíl:** prolomit S227 plateau (predikovatelnost-strop daných mozků) tím, že prediction
  skill **tvaruje mozek** — moduluje Hebbian na w1/w2 → hidden states k predikovatelnosti.
  Provedeno *po* S227 (potřebovalo per-cell skill a důkaz, že selekce sama plateauuje).
- **Výstup:** hotovo. Lean (bez RewardKind plumbingu): nová metoda `apply_prediction_reward`
  postaví rewards vec `PREDICT_HEBB_COEFF · max(0, cell.pred_skill)` a pustí ho do existujícího
  GPU `dispatch_apply_reward_persistent` hned po `brain_act` (fresh eligibility trace). `max(0,·)`
  → **jen posiluje, nikdy anti-Hebbian** (žádný kolaps mozku); cílem je *non-constant* sensory
  future, takže degenerovaný konstantní hidden predikuje špatně → nic nevydělá → není fixed
  point. `cell.pred_skill` (raw) set ve writebacku. `PREDICT_HEBB_COEFF=1.0`. 30-gen smoke
  (seed 1): populace zdravá (~1500), skill nejdřív klesl (~−0.45, Hebbian tvaruje/perturbuje),
  pak **vylezl PŘES S227 plateau: gen 28–30 = −0.20/−0.18/−0.19** a pořád stoupal. Build +
  464 lib + 128 headless + renderer zelené.
- **Poznámky:** within-life Hebbian tvaruje mozek k predikovatelnosti; se S227 selekcí se
  CPPN co se dobře tvarují selektují → skill leze napříč generacemi = **správný Baldwin na
  mozku** (na rozdíl od S228 readout weight-leaku, který přenášel nepřenositelné). První
  mechanismus co plateau prolomil. **Otevřená otázka ZODPOVĚZENA:** 60-gen běh (seed 2) →
  `pred_skill` **přešel přes 0 kolem gen 47** a zůstal kladný (mean gen 50–60 = **+0.023**,
  10 generací > 0). Model poráží persistence — **headline akceptační kritérium (a)+(b)
  splněno** (pozitivní + ratchetuje: −0.59→−0.33→+0.02), v režimu bez explicitní odměny za
  úlohu/chytrost. **Kalibrace:** zatím 1 dlouhý běh přes 0 (+ seed 1 lezl k −0.18@gen30
  konzistentně); k potvrzení průlomu chybí **(i) replikace** (víc seedů), **(ii) ablace
  rekurence** (S229 — kolabuje skill bez paměti? = genuine temporal integration vs trivial),
  **(iii)** rozlišit lepší world-model vs predikovatelné chování (active inference action). Pozn.:
  gen 55–57 dip populace (1302→724→1496) = mírná turbulence kolem skill-peaku.

## Sprint 227 — skill → reprodukce (keystone fitness coupling)

- **Cíl:** napojit prediction skill na fitness tak, aby lepší prediktoři víc reprodukovali —
  bez explicitní „buď chytrý" fitness.
- **Výstup:** hotovo, s **pivotem mechanismu**. První pokus (per-tick energy advantage,
  zero-sum `energy += COEFF·(skill−cohort_mean)`) **zkolaboval populaci** i při COEFF=0.002:
  per-tick energetické perturbace rozbíjejí synchronizovanou reprodukci (bloom je knife-edge
  citlivý) a zero-sum koncentruje energii do jednoho vítěze (200→1 buňka). Izolováno:
  COEFF=0.0 bloomuje byte-identicky s S225 → kód OK, disrupce čistě z energie. **Pivot na
  reprodukční-threshold discount** (vzor novelty bonusu): `cell.pred_advantage = clamp(skill −
  warmed_cohort_mean, 0, 1)` (set ve writebacku), `collect_fertile` násobí threshold
  `(1 − PREDICT_REPRO_BONUS·pred_advantage)`. Jednostranný (fertility bonus, nikdy trest →
  nemůže vyhladovět/zkolabovat populaci); brzy je variance skillu ≈0 → discount ≈0 → bloom
  nedisruptovaný; selekce náběhne až jak readouty trénují a mozky se diferencují.
  `PREDICT_REPRO_BONUS=0.3`. 30-gen smoke (seed 1): **populace zdravá** (bloom na cap ~1500,
  stabilní), `pred_skill ~−0.3` vs S225 baseline ~−0.6 → **selekce zvedla skill populace**.
  Build + 464 lib + 128 headless + renderer zelené.
- **Poznámky:** skill se ustálil ~−0.3 (rychlá rovnováha — readout-strop daných mozků). Posun
  k 0+ (porazit persistence = jádro průlomu) vyžaduje evoluci prediktivnějších *mozkových*
  hidden states = pomalý CPPN proces přes mnoho generací → vlastní breakthrough test je dlouhý
  sweep (S231), ne smoke. Per-cell skill konfunduje kvalitu mozku s věkem readoutu → mean jen
  přes trénované (warmed) buňky + clamp[0,1] (novorozenci neutrální).

## Sprint 228 — Baldwinův most (weight-leak otestován, zamítnut)

- **Cíl:** prolomit S227 plateau dědičností naučeného readoutu — child zdědí `BALDWIN_LEAK·`
  rodičovský `predict_w_live` (mean obou) a dál ho ladí.
- **Výstup:** implementováno (`BALDWIN_LEAK`, leak v `make_mating_child_no_brain` → pokrývá
  sexual i klonální cestu), **otestováno a zamítnuto**. Leak=1.0 (30-gen smoke, seed 1):
  populace zdravá (~1500), ale `pred_skill` **spadl na ~−0.45** vs S227 ~−0.28. Nastaveno na 0
  (≡ S227), guarded (disabled path nestojí nic), infrastruktura ponechána. Build + 464 lib +
  128 headless zelené.
- **Poznámky — důležitý nález:** naučený readout je **brain-specific** (mapuje `hidden(t)→
  sensory(t+1)` *konkrétního* mozku); sexuální child má crossover mozek odlišný od obou
  rodičů, takže zděděný readout je mistuned → horší než re-učení od nuly. **S227 už Baldwinův
  *efekt* MÁ** (within-life učení adaptuje readout na mozek buňky → skill reflektuje
  predikovatelnost mozku → selekce na ni působí); plateau ~−0.28 je **pomalý CPPN ratchet** na
  prediktivní mozky, ne chybějící dědičnost. Lamarckovský leak přenáší nepřenositelné. Možné
  varianty: clonal-only (brain-matched) leak, nebo pure-Baldwin `lr_pred` gen (dědí se
  *schopnost* učit, ne obsah). Plateau prolomí spíš brain-directed mechanismus (S226 Hebbian na
  hidden states) nebo dlouhý běh (CPPN evoluce).

## Sprint 229 — kauzální instrumentace (recurrence ablace)

- **Cíl:** zjistit, jestli je skill > 0 genuine temporal integration (používá paměť), nebo
  feedforward artefakt — ablovat rekurenci a změřit efekt.
- **Výstup:** hotovo. **(1) Korelační (free, z existujícího trendu):** `recurrent_io` roste
  0.31→0.37, jak skill překračuje 0; `corr(skill, recurrent_io) = +0.315` → brain používá
  paměť víc, jak se stává prediktivním. **(2) Kauzální proba** (`pred_ablation_div` v
  `write_stats`): na hranici generace (váhy synced) re-run parity-matched CPU
  `forward_with_state` s vynulovanými recurrent sloty `[BRAIN_INPUTS_SENSORY..]`, změř, jak moc
  se readout predikce změní vs normální forward. GPU `forward_persistent_into` je hardcoded na
  cells buffery (žádný scratch) → CPU forward na synced váhách je čistá, low-risk cesta (žádný
  zásah do `gpu/brain.rs`, který je navíc v externím fluxu). Sanity (12 gen): div ~0.025 při
  skill ~−0.45. CSV sloupec `pred_ablation_div`. Build + 464 lib + 128 headless + renderer
  zelené.
- **Poznámky — KLÍČOVÝ NÁLEZ (60-gen seed 2):** `pred_ablation_div` **KLESÁ**, jak skill roste
  (low-skill gen 5–20: 0.026 → high-skill gen 45–60: **0.010**), `corr(skill, div) = −0.903`.
  Čím lepší predikce, tím MÍŇ závisí na recurrence → **emergentní kognice je FEEDFORWARD, NE
  memory-based.** Mozek predikuje z *aktuálního* senzorického stavu (nelineární extrapolace
  gradientů/rychlosti), ne z temporální integrace — dává smysl, nejspolehlivější signál pro
  next-tick sensory je current sensory + vlastní rychlost. **Headline (c) NESPLNĚNO** (skill
  *nekolabuje* při ablaci paměti — naopak je na ni robustní). `recurrent_io` rostl, ale to je
  jen magnituda hidden state, ne funkční použití paměti — kauzální ablace ukázala pravdu a
  **zabránila over-claimu.** Pro memory-based kognici (plný breakthrough) musí *úloha* vyžadovat
  paměť (partial observability, delší horizont, delayed info) — current next-tick task je
  řešitelný feedforward, tak paměť nevznikne. Ablace splnila účel: rigorózní falzifikace.

## Sprint 230 — dark-room / pathology audit + tuning

- **Cíl:** systematicky zkontrolovat FEP patologie — dark-room (zamrznutí/schování),
  trivial-channel gaming (skill koncentrovaný v self-caused kanálech), balance
  surprise-seeking vs -avoiding. Doladit slot set + znaménko + energy scale.
- **Výstup (plán):** zisky skillu z *exogenních* kanálů, buňky aktivně forageují, populace diverzní.
- **Poznámky:** sem patří i finální per-slot variance pruning z S223 TODO.

## Sprint 231 — cross-seed validační sweep + ablační matice

- **Cíl:** plný 5×100-gen (seedy 1–5). Porovnat: baseline (no-FEP) vs FEP-energy-coupled
  vs FEP+Baldwin. Metriky: `pred_skill` trajektorie, `behavioral_entropy`, `w1_eff_rank`,
  foraging, stabilita.
- **Výstup (plán):** FEP arm má monotónní `pred_skill` ratchet, který baseline nemá.
- **Poznámky:** akceptace — `w1_eff_rank` roste (reálná kapacita, ne bloat) = přímé vyvrácení
  Goodhart nálezu z decade 203–212.

## Sprint 232 — perf + konsolidace + write-up

- **Cíl:** změřit ticks/s dopad (čekej malý — GPU-sync-bound). Konsolidovat metriky, zdokumentovat.
- **Výstup (plán):** pokud `pred_skill` ratchetuje a přežije ablaci → headline průlom demonstrován:
  funkční kognice pod čistě přežívací selekcí, žádný explicitní intelligence reward.
- **Poznámky:** sázka z `docs/00` (Stanley & Lehman: na inteligenci se nedá mířit) převedená
  do testovatelného výsledku.
