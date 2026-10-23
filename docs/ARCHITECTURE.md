# Architecture: Layers, Packages and Consumers

How the engine's code is organized and how other projects connect to it. Written for collaborators who need the map before writing code. The functional design (what `extract(A, V)` does, stage by stage) is in [DESIGN.md](DESIGN.md); decisions are dated in [DECISIONS.md](DECISIONS.md).

> **Status:** target architecture, approved by the owner on 2026-10-23. No code yet: the first code arrives by seeding from short_extract (§7).

---

## 1. Position: one library, several consumers

```mermaid
flowchart TB
    subgraph LIB["video_extraction_engine (this repository)"]
        CT["contracts<br/>ports, A / V / dataset schemas,<br/>lens spec format, ids, run records"]
        MA["model access<br/>catalog, router, adapters,<br/>response cache"]
        EX["extraction<br/>stages L0–L4, runner,<br/>dataset writer and reader,<br/>benchmark harness, CLI"]
    end
    APP["short_extract<br/>search agent"]
    BEN["benchmarks"]
    TRN["training experiments"]
    DS[("dataset<br/>Parquet / Arrow + manifest")]

    EX --> MA
    EX --> CT
    MA --> CT
    EX -- "writes" --> DS
    APP -- "model access for its roles" --> MA
    APP -- "lens specs, schemas" --> CT
    APP -. "loads" .-> DS
    BEN --> EX
    TRN -. "reads" .-> DS
```

| Layer | Owns | Must not |
|---|---|---|
| **contracts** | ports (`conversation`; later `video_understand`, `vision_generate`, `embed_*`), media ports, the `ResponseStore` port, the A / V / dataset schemas, the lens spec format, ids, run records | import anything else in the library, any provider SDK, or any I/O library |
| **model access** | model catalog, router, the SDK and LangChain adapter families, the response cache wrapper and its file store | know stages, datasets or lenses |
| **extraction** | stages L0–L4 + embed, the stage runner, the dataset writer and reader, the benchmark harness, the CLI | call a provider SDK directly; contain lens or corpus vocabulary |

**What the library never knows:** any consumer. No sessions, no web framework, no Postgres, no task queue, no consumer's services. A consumer depends on the library; never the other way round.

---

## 2. Packages (proposed)

One installable distribution, `video-extraction-engine`, with one import package and a subpackage per layer. Flat layout, no inner `src/`, managed with uv.

```
video_extraction_engine/
  contracts/     ports, schemas (A, V, dataset, lens spec), ids, clock, run records, ResponseStore
  models/        catalog, router, adapters (sdk/, langchain/), cache/ (wrapper + file store)
  extraction/    stages/ (normalize, signals, segment, perceive, lens_pass, embed, qc),
                 runner/, dataset/ (writer, reader, manifest), bench/, alignment/
  cli.py         `vxe extract …`, `vxe bench …`, `vxe cache stats|wipe`
tests/
docs/
```

Optional extras keep the base install light: `[media]` (ffmpeg bindings, scene detection, audio), `[speech]` (Whisper + VAD), `[langchain]` (the LangChain adapter family).

### 2.1 Dependency rules (import-linter)

```mermaid
flowchart LR
    CLI["cli"] --> EX["extraction"]
    EX --> MA["models"]
    EX --> CT["contracts"]
    MA --> CT
```

| Contract | Rule |
|---|---|
| Layers | `cli` → `extraction` → `models` → `contracts` |
| Contracts are pure | `contracts` imports no provider SDK, LangChain, media library, or I/O framework |
| Provider SDKs and LangChain | only inside `models` adapters |
| Media libraries (ffmpeg bindings, scene detection, audio, Whisper) | only inside `extraction.stages` behind media ports |
| LangGraph | nowhere |
| No consumer code | no import of any consumer package, web framework, ORM or queue library |

A **vocabulary test** scans engine code (and prompt, JSON and text files shipped with it) for known lens and corpus terms. Example lens specs and corpus profiles used by tests live under `tests/fixtures/`, outside the scan, and contain no footage.

---

## 3. Public API

| Surface | Shape |
|---|---|
| **Function** | `extract(context: ExtractionContext, corpus: Corpus, *, models: ModelProfile, runner: Runner = LocalRunner(), out: Path, cache: ResponseStore | None = None) -> Dataset` |
| **Schemas** | `ExtractionContext` (A), `Corpus` + `CorpusProfile` (V), `LensSpec`, `ModelProfile`, `DatasetManifest`: Pydantic models, versioned with the dataset schema |
| **Dataset** | `Dataset.open(path)` → typed readers per table (Arrow); `manifest` |
| **Model access** | `ConversationRouter`, `ConversationRequest` / `ConversationResult`, `with_cache(port, store, settings)`; the same API consumers use for their own roles |
| **CLI** | `vxe extract --context a.yaml --corpus v.yaml --out ./ds`; `vxe bench …`; `vxe cache stats|wipe` |

Changing a public schema or port is a **release** with a version bump; the dataset manifest records the schema version so readers can refuse or migrate old datasets.

---

## 4. Inside `extract()`

```mermaid
flowchart LR
    A["A: context"] --> X["extract()"]
    V["V: corpus"] --> X
    X --> PL["plan<br/>stages × units × lenses,<br/>input hashes, cost estimate"]
    PL --> RU["stage runner<br/>idempotent by input hash,<br/>resumable, budget-capped"]
    RU --> ST["stages<br/>L0 → L1 → L2 → L3a →<br/>L3b → embed → L4"]
    ST -- "model ports" --> MA["model access<br/>(+ response cache)"]
    ST -- "media ports" --> MED["media tools<br/>ffmpeg, shot detection,<br/>audio, speech"]
    ST --> W["dataset writer"]
    W --> DS[("dataset")]
    BE["benchmark harness"] --> X
    CLI["CLI"] --> X
```

- **Plan:** expands A × V into tasks and their input hashes; tasks whose run record already exists are skipped. The plan gives the cost estimate checked against the budget.
- **Runner:** executes tasks with bounded parallelism and shared per-provider rate limits. Writes the run record before scheduling dependants, so a crash resumes cleanly. `LocalRunner` first; a distributed runner later, behind the same port (DESIGN.md §5.2).
- **Working store:** intermediate artifacts (normalized copies, chunk files, per-task outputs, run records) live in a local work directory next to the output, keyed by input hash. It's a cache: deleting it costs only recomputation.
- **Writer:** turns task outputs into the dataset tables and the manifest.

---

## 5. Model access

```mermaid
flowchart LR
    R["caller<br/>(stage, or a consumer's role)"] -- "ConversationRequest<br/>+ cache_class, family" --> C["response cache<br/>(off / record / replay)"]
    C -- "miss" --> RT["ConversationRouter<br/>catalog → provider"]
    C -. "ResponseStore port" .-> FS[("file store (built in)<br/>or a consumer's store")]
    RT --> F1["SDK family<br/>openai · anthropic"]
    RT --> F2["LangChain family<br/>ChatDeepSeek · ChatAnthropic"]
    F1 --> P["provider"]
    F2 --> P
```

- **Catalog:** one entry per model: provider, wire protocol, context window, structured-output method, the reasoning switches (`ReasoningControl`), the LangChain class. Reasoning is off unless a request asks for it.
- **Two adapter families behind one port:** the SDK family (official `openai` and `anthropic` SDKs) and the LangChain family. Both share a base that owns the key check, output budget, timing, error mapping, stop-reason normalization and the result.
- **Keys** come from the environment only (`HIVE_API_KEY`, `DEEPSEEK_API_KEY`, `ANTHROPIC_API_KEY`); never in files, never logged.
- **Response cache:** keyed by the request (including model and method) per cache class; replays are marked `cached`, cost 0, adapter version suffixed `+cache`. The store is a port: the library ships a file store; a consumer may plug in its own (short_extract uses Valkey).
- `video_understand`, `vision_generate` and `embed_*` are added the same way when E1 needs them.

This stack is seeded from short_extract, where it's built and tested today (§7).

---

## 6. How consumers connect

- **Dependency:** consumers install the library from its git repository at a **release tag** (uv git source). A contract change means a library release, then a version bump in the consumer.
- **short_extract (the search agent):**
  - uses model access for its Interpreter, Responder and digest roles, exactly as before the move;
  - reads lens specs with the library's loader (query mapping, specificity rules, rubric);
  - never runs stages: it **loads datasets** into Postgres + pgvector with its own loader;
  - provides a Valkey `ResponseStore` for its cache.
- **Benchmarks and training:** use `extract` and `Dataset.open` directly; results stay local per the rights rule (DESIGN.md §8).

---

## 7. Seeding plan

The engine's first code comes from short_extract, keeping its history:

| Step | Repository | Work |
|---|---|---|
| 1. Restructure in place | short_extract | Gather contracts into `engine/core` and model access into `engine/models` (transitional folders), introduce the `ResponseStore` port, new import contracts. Behavior-preserving. |
| 2. Seed this library | this repo | `git subtree split` of `engine/core` + `engine/models`, merged here; rename packages to the layout in §2; add tooling (pytest, ruff, import-linter contracts, vocabulary test) and CI. Tag **`v0.1.0`**. |
| 3. Depend on the library | short_extract | Replace the local copies with this library at `v0.1.0`. Behavior-preserving. |

Then extraction (E1 onward, DESIGN.md §7) is built here.

**Acceptance for step 2:** the moved tests pass here unchanged in count; `lint-imports` and `ruff` pass; a cache record → replay round trip works with the file store; no file in the repository contains footage, frames or keys.

---

## 8. Open decisions

- Package and CLI names (proposed: import package `video_extraction_engine`, CLI `vxe`).
- Dataset layout details, A / V schema details, lens spec file format (DESIGN.md §9).
- Distributed runner technology, once corpus size needs it.
- Video, text and embedding models (DESIGN.md §9).
