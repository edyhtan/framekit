# Decision Journal

Chronological log of design decisions. Newest last. Each entry: what was decided, why, and where it is recorded. The current design is always [DESIGN.md](DESIGN.md) and [ARCHITECTURE.md](ARCHITECTURE.md); this file explains how they got there.

Times are local (UTC−4). Decisions made before this repository existed are in [short_extract's journal](https://github.com/edyhtan/short_extract/blob/main/docs/DECISIONS.md); the ones this engine inherits are summarized in the first entry.

---

## 2026-10-23

### 18:30 — Repository created: video_extraction_engine
- **Decision** (owner): corpus extraction lives in its own repository, named **`video_extraction_engine`**, as a general-purpose library and CLI: `extract(A, V) → dataset`. Its design and architecture documents start from the extraction parts of short_extract's DESIGN.md and ARCHITECTURE.md, which now point here instead of describing extraction themselves.
- **Why:** the owner wants the extraction layer reusable beyond the search app, for benchmarking and experimental model training. A general library must not depend on an app's persistence or services.
- **Recorded in:** README.md; CLAUDE.md; DESIGN.md; ARCHITECTURE.md (all new).

### 18:31 — Decisions carried over from short_extract
These were decided in short_extract (dates are from its journal) and bind this engine from day one:
- **Function and output** (2026-10-23): `extract(A, V) → dataset`. A is the extraction context, V the corpus (profile + unit references, never copies). The output is a **full dataset** (segments, text, vectors, provenance) as Parquet / Arrow plus a manifest, not vectors alone. → DESIGN.md §2.
- **The library owns the engine** (2026-10-23): contracts (ports, A / V / dataset schemas, lens spec format) and model access (catalog, router, adapter families, response cache). Consumers depend on it at a release tag. → ARCHITECTURE.md §1, §6.
- **Lens specs are declarative data** (2026-10-23): schema, vocabulary, prompts and rubric as data files, passed in A and also read by consumers. → DESIGN.md §4.
- **Perception separate from interpretation** (2026-09-27): L3a records observable facts only; every interpretation is a text-only lens pass (L3b). → DESIGN.md §0.3, §3.
- **Fixed, versioned vocabularies per lens** (2026-09-27); no free-text labels. → DESIGN.md §4.
- **Units are the reference, never split as a reference;** chunk files are derived processing artifacts (2026-09-27 baseline; chunk files later). → DESIGN.md §1, §5.3.
- **Lens- and corpus-free engine code**, checked by a vocabulary test; **model-agnostic** ports with `model_id` on every output, new versions instead of overwrites. → DESIGN.md §0; CLAUDE.md.
- **Rights modes** (`byo_media` | `hosted` | `embed`); for `byo_media`, no footage, frames, thumbnails, ripping tools or downloaders in the repo; datasets stay local, and training follows the same rule (2026-10-23). → DESIGN.md §8.
- **Model access as built in short_extract:** one `conversation` port with SDK and LangChain adapter families (2026-09-29); reasoning off unless requested; the response cache with off / record / replay per call class (2026-10-14), whose store becomes a port (2026-10-15). → ARCHITECTURE.md §5.
- **Seeding:** the library starts from short_extract's contracts and model access via `git subtree split`, keeping history, tagged `v0.1.0` (2026-10-23). → ARCHITECTURE.md §7.
- **PRs are squash-merged; never stack PRs** (owner's workflow rule). → CLAUDE.md "Workflow".

### 18:32 — Package layout proposed (not decided)
- **Proposal:** one distribution, `video-extraction-engine`, import package `video_extraction_engine` with `contracts`, `models` and `extraction` subpackages, and a `vxe` CLI; heavy media and speech dependencies as optional extras. Import contracts enforce `cli → extraction → models → contracts`.
- **Status:** open until seeding (E0); names are proposals.
- **Recorded in:** ARCHITECTURE.md §2, §8.

### 18:55 — Packaging: the Python library convention, src layout
- **Decision** (owner): the engine is packaged as a standard Python **library**, not an app: one `pyproject.toml` with the `hatchling` build backend (wheel + sdist), **src layout** (`src/video_extraction_engine/`), loose dependency ranges, heavy dependencies as optional extras (`[media]`, `[speech]`, `[langchain]`), `py.typed`, semantic-version git tags. `uv.lock` is committed for the engine's own development only. Consumers install it from git at a tag.
- **Supersedes:** the flat layout in the earlier entry "Package layout proposed (not decided)" (names stay proposals).
- **Why:** the engine will be installed by other projects, including **experimental model-training projects** (owner, 2026-10-23). The library convention keeps it installable next to their own stacks: ranges instead of pins avoid version clashes with training frameworks, extras keep media and speech dependencies optional, and the src layout makes tests run against the installed package, as consumers see it.
- **Consequence for training:** experiments read datasets as plain Parquet / Arrow with whatever training stack they choose; the engine never depends on a training framework. Experiment code lives in the consumer.
- **Recorded in:** ARCHITECTURE.md §2, §6, §7, §8; DESIGN.md §2.2, §9; CLAUDE.md "Key decisions".

### 19:03 — Renamed to framekit: a general library, not a single-purpose engine
- **Decision** (owner): the repository and library are renamed from `video_extraction_engine` to **`framekit`** (repo `edyhtan/framekit`; distribution, import package and CLI all `framekit`). It's framed as a **general library of AI building blocks for video datasets**: `framekit.contracts`, `framekit.models` (model access) and `framekit.extraction` are modules of one package, each usable on its own.
  - **Modules, not packages:** new capabilities are added as subpackages, with their own import contract and, if heavy, their own extra. A separate package only when a consumer needs one part without the rest and extras can't provide it.
- **Why:** the old name described only one of the three parts; model access is general LLM access used by the search app too, and training experiments will use the library beyond extraction. The owner wants a general library but not many small packages.
- **Supersedes:** the repository name in "Repository created: video_extraction_engine", and the proposed names (`video_extraction_engine`, `vxe`) in "Package layout proposed (not decided)".
- **Open:** an unrelated, active package named `framekit` exists on PyPI. Installing from git is unaffected; publishing would need another distribution name, and a project must not install both.
- **Recorded in:** README.md; CLAUDE.md "Project", "Key decisions"; DESIGN.md title and intro; ARCHITECTURE.md §1, §2, §6, §7, §8.

### 19:15 — Model access: own adapters on the official SDKs only; no LangChain family, no [langchain] extra
- **Decision** (owner): framekit's model access is our own adapters on the official `openai` and `anthropic` SDKs, nothing else. The LangChain adapter family and the `[langchain]` extra are dropped from the plan; short_extract removed LangChain the same day (its PR #17), and the OpenAI-compatible adapter gained `function_calling` structured output (a forced tool call), which was the one thing LangChain provided.
- **Why:** an adapter library sees every API key and prompt, so its supply chain matters more than its features. LiteLLM had two PyPI compromises in 2026 (March and October 21, the latter a credential stealer), and the LangChain family alone added 20 packages. Our adapters are a thin, tested layer; the hard parts (catalog, reasoning switches, error mapping, cache) are ours anyway.
- **Who owns frameworks:** the owner asked why framekit, not the app, would own LangChain. Answer recorded: it shouldn't. A framework used by one consumer belongs in that consumer, wrapping framekit's port; framekit carries only what its own modules and every consumer need. The earlier plan put it here only to keep one catalog, and that reason goes away with the family.
- **Recorded in:** ARCHITECTURE.md §1, §2, §2.1, §5, §6; DESIGN.md §7; CLAUDE.md "Architecture rule".

### 19:24 — Vision parameters are their own structured data (`vision_params`)
- **Decision** (owner): how a model sees video is a structured parameter of its own, not a few loose numbers. **Vision profiles** are named and versioned, and A assigns one per stage that sends pictures (L2, L3a; consumers reuse them at query time). A profile has four parts:
  - `frame_selection`: candidates, anchors, filters, budget;
  - `image`: size, aspect, encoding;
  - `delivery`: images, storyboard, or native video.
- **Details:**
  - **Strategies are versioned code** in a new module, `framekit.vision`; profiles only select and parameterize them.
  - **Model capabilities are checked at load time;** nothing is silently resampled.
  - **Resolved profiles** go into the manifest and each stage's input hash.
  - **The timestamps of the frames actually sent** are recorded per scene.
  - **Defaults come from a benchmark sweep.**
- **Why:** frame choice changes the dataset as much as the model does (fast action can fall between uniformly sampled frames), and frames are most of perception's cost. As structured data, it can be compared in benchmarks, controlled in training experiments, and reproduced exactly.
- **Context gathered:** native-video APIs sample uniformly (fps, clip range, resolution); frame-list input allows any strategy: shot-based, change-based dedupe, motion- or audio-weighted, sharpness filtering, coarse-to-fine, query-aware, storyboard tiling. L1 signals (shots, motion, audio onsets) make adaptive selection cheap here.
- **Open:** default profiles (from a sweep); the schema of the other extraction settings, which follow the same pattern.
- **Recorded in:** DESIGN.md §2 (A), §2.3 (new), §3 L2 / L3a / Cost, §6.2, §9; ARCHITECTURE.md §1, §2, §2.1, §3; DESIGN.md §0.1 diagram; README.md; CLAUDE.md "Key decisions".

### 19:39 — A is the given search space; the plan is separate; benchmarking augmentation strategies
- **Decision** (owner):
  - **A is the description of the search space,** the user's input or an already-augmented input, saying what should become findable in the corpus: dimensions (lens specs), evidence (required perception modules), resolution (target moment length), scope, and query style (example queries).
  - **Augmentation is out of scope for framekit.** framekit takes A as given and never creates or augments it.
- **Why:** the owner's goal for this abstraction is an artifact for **benchmarking augmentation strategies**: different ways of producing A from a raw intent, compared on the same corpus. That only works if framekit is a fixed machine that A passes through.
- **Consequences:**
  - **A and the plan are separate.** How A is realized (vision, segmentation, signals, perception and lens-pass settings, embedding, models, quality control, budget, formatting; governance inherited from V) is an optional **plan**, with defaults, overrides, and documented derivation rules from A. The signature becomes `extract(A, V, plan=…)`. This supersedes the earlier framing of A as "the extraction context" holding these settings, including the vision-profile wording earlier today.
  - **The manifest records A verbatim with its hash,** the resolved plan, and the derivation rules applied.
  - **Benchmark fairness:** the harness compares A's under a pinned plan, or plans under a fixed A, and refuses comparisons where the other side differs. Strategy labels are recorded; L0–L3a are reused when only lenses differ; the gold queries and moments are fixed and independent of A; randomness is pinned.
  - **Scoring is a plug-in** (owner agreed): framekit ships a neutral **baseline retriever** (fixed embedding model, recall@k, gold-moment rank) as the default; consumers can plug in their real search agent.
- **Recorded in:** DESIGN.md intro, §0.2, §2, §2.2, §2.3 wording, §2.4 (new), §3 Embed / Cost, §5.1, §6.2, §9; ARCHITECTURE.md §1, §2, §3, §4, §8; README.md; CLAUDE.md "Project", "Architecture rule", "Key decisions".

### 21:34 — v0.1.0: contracts and model access seeded from short_extract; repository made public
- **Built:**
  - `framekit.contracts`: the conversation port types and the `ResponseStore` port.
  - `framekit.models`: catalog, router, the OpenAI-compatible and Anthropic adapters on the official SDKs, and the response cache.
  - A new `FileResponseStore` for standalone runs. 89 tests; 5 import contracts; vocabulary test; CI on locked and newest dependency versions; wheel and sdist build; a fresh install imports the public API.
- **How it was seeded:** the plan's step 1 (restructure short_extract in place) was skipped. The files were taken straight from short_extract with **`git filter-repo`**, keeping their history (15 commits, including earlier file names), then reshaped into `src/framekit/`. `git subtree split` wouldn't have worked, since the files were spread over several packages.
- **Changes from the app's version:**
  - **No environment-bound settings:** `ModelSettings` and `CacheSettings` are plain Pydantic models that consumers build. The app keeps its own `CLIPSEARCH_*` variables.
  - **The router's app-specific helpers** (`from_environment`, `for_search`) stay in short_extract.
  - `with_cache` now requires a store: no default backend, so the library never reaches for a server.
  - The Valkey store and its stats/wipe CLI stay in short_extract.
- **Repository made public** (owner, 2026-10-23), so consumers' Docker builds can install it without credentials. It contains no footage, frames, thumbnails or keys; the seeded history was scanned for key-like strings first.
- **Open:** the code licence (DESIGN.md §8.3 says MIT or Apache-2.0); until it's chosen, no licence is granted.
- **Recorded in:** README.md "Status"; ARCHITECTURE.md status, §3, §7; DESIGN.md §7; CLAUDE.md "Workflow", "Current milestone".

### 21:36 — CI: separate unit-test and lint workflows
- **Decision** (owner, now that the repository is public): CI is two workflows, each with its own status badge in the README.
  - **`tests`:** pytest and the wheel/sdist build, on **locked** and **newest** dependency versions. The newest run is what a consumer installing framekit today gets.
  - **`lint`:** ruff, plus the import contracts, which enforce the layer rules.
  - Both run on pull requests and on pushes to `main`, with read-only permissions.
- **Why:** contributors and users of a public library see at a glance whether the code passes its tests and its rules, and a lint failure is told apart from a test failure.
- **Recorded in:** `.github/workflows/tests.yml`, `.github/workflows/lint.yml` (replacing `ci.yml`); README.md badges; CLAUDE.md "Workflow".

---

## Still open
See DESIGN.md §9: video, text and embedding models; A / V / plan schema details and the lens spec file format; default vision profiles; baseline scorer details and gold queries; dataset layout; a distribution name if ever published to PyPI; distributed runner; per-lens taxonomies; the code licence.
