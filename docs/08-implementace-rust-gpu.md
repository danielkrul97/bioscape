# Implementace v Rustu na GPU

## Proč Rust + GPU

### Proč Rust
- **Paměťová bezpečnost bez garbage collectoru** — žádné nečekané pauzy
- **Zero-cost abstractions** — výkon jako C/C++, ergonomie podstatně lepší
- **Skvělá konkurence** — `rayon`, `tokio`, atd. Bezpečné paralelizace
- **Roste GPU ekosystém** — `wgpu`, `rust-gpu`, `cust` (CUDA), `vulkano`, `cudarc`

### Proč GPU
Evoluční simulace je **embarrassingly parallel**:

- Tisíce/miliony agentů vyhodnocujeme současně
- Každá buňka v Neural CA aplikuje stejné pravidlo (jednoznačně SIMD)
- Mutace, crossover, výpočet fitness — vše paralelní

CPU s 16 jádry vs. GPU s tisíci shader cores — pro tento typ úlohy GPU vyhraje řádově.

---

## Hlavní cesty: wgpu vs. CUDA vs. ostatní

### wgpu (doporučeno pro start)

[wgpu](https://wgpu.rs/) je rust knihovna implementující WebGPU standard. Funguje **napříč všemi platformami**:

- Windows, Linux, macOS
- Mobile (iOS, Android)
- Web přes WASM
- Backend: Vulkan / Metal / DX12 / OpenGL ES

**Compute shadery se píšou v WGSL** (WebGPU Shading Language) — moderní, čistý jazyk podobný Rustu.

**Plus:**
- Portable napříč hardwarem (NVIDIA, AMD, Apple Silicon, Intel)
- Žádná závislost na CUDA toolchainu
- Hot reload shaderů (klíčové pro experimentování s evolucí)
- Rust-first ekosystém

**Mínus:**
- Trochu nižší peak výkon než nativní CUDA
- WGSL má méně low-level funkcí než CUDA C++

🔗 [wgpu.rs](https://wgpu.rs/)
🔗 [Rust GPU Programming with wgpu (2026 guide)](https://rustify.rs/articles/rust-gpu-computing-wgpu-2026)
🔗 [High Performance GPGPU with Rust and wgpu (DEV.to)](https://dev.to/jaysmito101/high-performance-gpgpu-with-rust-and-wgpu-4l9i)

### CUDA přes `cudarc` nebo `cust`

Pokud máš jen NVIDIA hardware a chceš maximální výkon:

- **cudarc** — moderní, idiomatic Rust wrapper na CUDA driver API
- **cust** — alternativní binding

**Plus:** maximální výkon, přístup ke všem CUDA features, NVIDIA tooling
**Mínus:** Vendor lock-in, složitější setup, žádný macOS

### rust-gpu (rust kód jako shader)

Projekt **rust-gpu** umí kompilovat Rust přímo do SPIR-V (Vulkan shader bytecode). To znamená, že **stejný Rust kód běží i na GPU**, sdílíš struktury mezi CPU a GPU stranou.

**Plus:** Type safety napříč CPU/GPU, sdílené struktury
**Mínus:** Stále experimentální, pomalejší kompilace

V roce 2026 se rust-gpu transition do community ownership (z Embark Studios), je to živý projekt.

🔗 [Rust GPU community announcement](https://rust-gpu.github.io/blog/transition-announcement/)

---

## Architekturální rozhodnutí pro Bioscape

### Klíčový problém: kolik práce na GPU vs. CPU

Pravidlo palce:
- **Na GPU:** vše, co zahrnuje paralelní výpočet nad mnoha agenty/buňkami současně (fyzika, neuronové sítě, Neural CA, fitness evaluace)
- **Na CPU:** orchestrace, IO, vizualizace, řízení experimentů, mutace genomu (často poměrně složité a serial)

### Layout dat: SoA vs. AoS

Na GPU **vždycky SoA** (Structure of Arrays). Místo:

```rust
struct Agent { pos: Vec3, vel: Vec3, energy: f32, ... }
let agents: Vec<Agent>;  // BAD pro GPU
```

Spíš:

```rust
struct Population {
    pos_x: Vec<f32>,
    pos_y: Vec<f32>,
    pos_z: Vec<f32>,
    vel_x: Vec<f32>,
    // ...
    energy: Vec<f32>,
}
```

Důvod: GPU shadery čtou data **v souvislých blocích pro celý warp** (skupina 32+ threadů). SoA = každý thread čte z jednoho pole, koalescovaný přístup → mnohem rychlejší.

### Buffer management

Hlavní bolest práce s GPU. V wgpu:

- **Storage buffers** pro persistent data (populace, neuronové váhy, mřížka prostředí)
- **Uniform buffers** pro frame parametry (čas, generační číslo, RNG seed)
- **Staging buffers** pro upload/download mezi CPU a GPU

**Tip:** Drž data na GPU **co nejdéle**. Stahování zpátky je drahé. Pokud je možné, dělej i mutaci, selekci a celý loop na GPU.

---

## Konkrétní design pro Bioscape (jeden návrh)

### Vrstva 1: prostředí

2D nebo 3D grid (pro start 2D je možná dost). Buňky obsahují:
- **Stav** (energie, chemikálie, světlo, atd.)
- **Update rule** (může být fixní nebo evolvable per region)

Implementace: storage buffer s velikostí W×H×Channels. Každý compute shader pass aktualizuje stav.

### Vrstva 2: agenti

Tisíce agentů, každý s:
- **Pozicí** v gridu
- **Tělem** (Neural CA výsledek nebo jednoduchá morfologie)
- **Mozkem** (neuronka — buď klasická, nebo SNN)
- **Genomem** — recept pro tělo i mozek
- **Energií, věkem, paměťkou**

Implementace: SoA buffers. Pevná maximální kapacita, recyklace slotů (slot allocator).

### Vrstva 3: evoluce

V hlavním loopu:
1. Simuluj prostředí jeden tick
2. Pro každého agenta: smysly → mozek → akce
3. Uplatni akce na svět (jíst, pohnout, množit)
4. Vyřaď mrtvé, přidej narozené (s mutacemi)
5. Občas: behavior diversity check (pro MAP-Elites archive)

### Hlavní pain pointy

- **Bezbolestná synchronizace** — víc compute passes, fence/barrier správně
- **Branch divergence** — když agenti dělají různé věci, GPU jede pomalu (warpy se rozcházejí). Trick: **batch agents by behavior** před každým passem
- **Rozhodování bez branching** — místo `if (in_water) ... else ...` se dělá maska a oba výpočty
- **Dynamic kapacity** — pevné velikosti bufferů jsou friend, dynamické jsou enemy

---

## Knihovny, které se hodí

### Compute / GPU
- [`wgpu`](https://crates.io/crates/wgpu) — main GPU framework
- [`bytemuck`](https://crates.io/crates/bytemuck) — bezpečné cast Rust struktur do byte buffers
- [`encase`](https://crates.io/crates/encase) — automatic GPU layout (řeší padding rules WGSL)
- [`pollster`](https://crates.io/crates/pollster) — block-on async (wgpu má async API)

### Vizualizace
- [`bevy`](https://crates.io/crates/bevy) — game engine, použitelný i jako simulátor s vizualizací. Má wgpu vestavěné, ECS, render pipelines.
- [`egui`](https://crates.io/crates/egui) — immediate mode UI, skvělé pro debug panely
- [`macroquad`](https://crates.io/crates/macroquad) — jednodušší než Bevy, dobré pro prototypy

### Numerika
- [`glam`](https://crates.io/crates/glam) — vektorová matematika, GPU-friendly layouts
- [`nalgebra`](https://crates.io/crates/nalgebra) — bohatší, ale těžší
- [`rand`](https://crates.io/crates/rand) + [`rand_xoshiro`](https://crates.io/crates/rand_xoshiro) — PRNG, pro GPU se obvykle implementuje vlastní v shaderu

### Profilování
- [`tracy`](https://crates.io/crates/tracing-tracy) — frame profiler
- `RenderDoc` — debug GPU draw/dispatch calls
- wgpu má `wgpu-profiler` pro per-pass timing

🔗 [GPU Computing in Rust — Are We Learning Yet](https://www.arewelearningyet.com/gpu-computing/)
🔗 [Massively Parallel Fun with GPUs (substack)](https://getcode.substack.com/p/massively-parallel-fun-with-gpus)

---

## Pořadí kroků doporučené pro projekt

### Fáze 0: vyčistit hlavu, postavit kostru
- Hello world wgpu
- Compute shader, který udělá něco trivialního (ráj nakažlivost mřížky)
- Render layer pro vizualizaci 2D mřížky

### Fáze 1: prostředí + agenti bez evoluce
- 2D grid s difuzí zdrojů
- Agenti s pevnými, ručně psanými strategiemi
- Vidíš agenty na obrazovce, jak se pohybují, jí, množí (s ručně psaným rozmnožováním)

### Fáze 2: evoluce klasickou neuronkou
- Genom = váhy malé feedforward sítě
- Mutace, selekce
- Sleduj fitness/diverzitu v čase
- Validuj: chování opravdu evolvuje?

### Fáze 3: tělo přes Neural CA
- Genom = pravidla CA
- Z jedné buňky vyroste tvar agenta
- Tvar ovlivní jak agent „funguje" v prostředí
- Validuj: morfologická diverzita roste?

### Fáze 4: ekologie a OEE
- MAP-Elites archive
- Multi-agent dynamika (predátor/kořist)
- Měnící se prostředí
- Cíl: simulace, co produkuje něco nového i po milionu generací

### Fáze 5: pokročilé
- Spiking sítě, STDP plasticita
- Active inference / free-energy modeled agents
- 3D fyzikální prostředí (pokud výpočetně utáhne)

---

## Otevřené technické otázky

- **Determinismus** — pro reprodukovatelné experimenty potřebujeme deterministické GPU passy. Možné, ale vyžaduje opatrnost (atomic operations mají nedefinované pořadí).
- **Long-running stability** — simulace běžící dny: leaky paměti, drift PRNG state, atd.
- **Save/load** — checkpointování celé populace + state prostředí. Velké, ale nutné.
- **Parametr-tuning** — meta-evoluce parametrů simulace? Nebo grid search?

## Zdroje

- [wgpu.rs](https://wgpu.rs/)
- [Are We Learning Yet — GPU computing](https://www.arewelearningyet.com/gpu-computing/)
- [Rust GPU Programming with wgpu (2026)](https://rustify.rs/articles/rust-gpu-computing-wgpu-2026)
- [High Performance GPGPU with Rust and wgpu](https://dev.to/jaysmito101/high-performance-gpgpu-with-rust-and-wgpu-4l9i)
- [Rust for GPU Programming Complete Guide](https://tillcode.com/rust-for-gpu-programming-wgpu-and-rust-gpu/)
- [WebGPU compute exploration (10 examples, GitHub)](https://github.com/scttfrdmn/webgpu-compute-exploration)
- [Rust GPU community ownership announcement](https://rust-gpu.github.io/blog/transition-announcement/)
