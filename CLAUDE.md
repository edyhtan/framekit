# CLAUDE.md

## Project
framekit: a non-commercial, general-purpose Python library of AI building blocks for video datasets. One package with modules: `contracts` (ports, schemas), `models` (model access: catalog, router, adapters, response cache) and `extraction`. New capabilities become new modules, not new packages, unless there's a concrete reason to split. The extraction module's main function is `extract(A, V) → dataset`: **A** is the extraction context (lens specs, perception modules, segmentation, embedding spec, model profile, budget), **V** is a video corpus (corpus profile + unit references). The dataset holds segments, text, vectors and provenance (Parquet / Arrow + manifest).

Consumers: the video clip search agent ([short_extract](https://github.com/edyhtan/short_extract)), benchmarks, and experimental model training. The library never depends on a consumer.

- **Test corpus (example, swappable):** classic Tom and Jerry shorts — `docs/DESIGN.md` Appendix A.
- **First lens (example):** emotions & dynamics — `docs/DESIGN.md` Appendix B.

Functional design: `docs/DESIGN.md`. Code layout and seeding plan: `docs/ARCHITECTURE.md`. Why it looks this way: `docs/DECISIONS.md`. Read them before architectural work.

## Decision journal (required)
Every design decision is recorded in `docs/DECISIONS.md`, the timeline journal, in the same change that applies it.
- Append a new entry at the end, under today's date heading, with local time and a short title.
- Each entry states **Decision**, **Why** (if a reason was given) and **Recorded in** (the doc sections changed).
- Never rewrite past entries. If a decision is reversed, add a new entry that points to the one it supersedes.
- Discussions that inform but don't decide may be logged as "Context gathered (no decision)".

## Diagrams in documents (required)
Always use rendered diagrams, never ASCII art: Mermaid code blocks (`flowchart` for pipelines and data flow, `erDiagram` for data, `stateDiagram-v2`, `sequenceDiagram`, `gantt`). Plain-text folder trees may stay in normal code blocks.

## Architecture rule (most important)
Layers: **contracts** (pure: ports, schemas, lens spec format) · **model access** (catalog, router, adapters, response cache) · **extraction** (stages, runner, dataset, benchmarks, CLI). See `docs/ARCHITECTURE.md` §1–§2.
- Library code and the dataset schema never reference lens concepts (emotion, dynamic, situation, gag, …) or corpus concepts (short, era, Tom, …). They enter only as data: lens specs in A, the corpus profile in V.
- No lens-specific columns: lens output lives in `annotations` / `phrasings`, validated against the lens spec.
- If a feature seems to need a lens- or corpus-specific branch, extend the lens spec format or the corpus profile instead, and discuss first.
- **No model-access libraries** (LangChain, LiteLLM, …): our own adapters on the official provider SDKs only, to keep supply-chain exposure low in code that holds API keys. Adding one needs discussion and a journal entry.
- **Model-agnostic:** no provider SDK calls, and no provider or library types, outside model adapters. Stages call ports (`video_understand`, `vision_generate`, `conversation`, `embed_*`). Models are chosen per role in the model profile. Every model output records `model_id`; switching models creates new versions and never overwrites.
- **No consumer knowledge:** no sessions, web framework, ORM, Postgres or task queue in this repo.

## Key decisions (do not change without discussion)
- **Units are the reference and are never split as a reference.** Every timestamp is on the unit timeline. Chunk files cut from the normalized copy are derived, cacheable processing artifacts; outputs are mapped back to the unit timeline; the source is never modified.
- **Perception and interpretation are separate passes.** L3a records observable facts only (core record + optional modules). Every interpretation is a text-only lens pass (L3b).
- **Fixed, versioned vocabularies per lens.** Never write free-text labels; validate against the lens spec.
- **Lens specs are declarative data** (schema, vocabulary, prompts, rubric), passed in A.
- **Vision parameters are their own structured data** (`vision_params`, `docs/DESIGN.md` §2.3): named, versioned vision profiles assigned per stage (frame selection, image prep, delivery). Strategies are versioned code in `framekit.vision`; profiles only select and parameterize them. Resolved profiles go in the manifest and the stage input hash; frames actually sent are recorded per scene.
- **The output is a full dataset**, not vectors alone: segments, text, vectors, provenance.
- **Idempotent, resumable runs** keyed by input hash; the response cache makes replays free and deterministic.
- **Packaged as a Python library** (`docs/ARCHITECTURE.md` §2): `pyproject.toml` with `hatchling`, **src layout** (`src/framekit/`), loose dependency ranges, heavy dependencies as optional extras, `py.typed`, semver git tags. No training framework as a dependency; consumers (including training experiments) bring their own.

## Constraints
- **Rights mode per corpus** (`byo_media` | `hosted` | `embed`). Tom and Jerry is `byo_media`: no footage, frames, thumbnails, ripping tools or YouTube downloaders in the repo, ever (`docs/DESIGN.md` §8). Datasets from a `byo_media` corpus stay local; training on them follows the same rule. Test fixtures use synthetic media only.
- Keep original source files untouched; analysis uses normalized copies.
- API keys come from `HIVE_API_KEY` / `DEEPSEEK_API_KEY` / `ANTHROPIC_API_KEY` in the local environment only: never commit them, never print or read their values.

## Workflow
- Changes go on feature branches and reach `main` through pull requests. PRs are squash-merged, so never stack PRs: each targets `main`.
- Code checks (once code exists): `uv run pytest`, `uv run lint-imports`, `uv run ruff check .`. Keep all three passing; never weaken an import-linter contract or the vocabulary test to make code pass.

## Current milestone
E0: seed contracts and model access from short_extract with `git subtree split`, tag `v0.1.0` (`docs/ARCHITECTURE.md` §7). Then E1: first dataset on 5 units with two lenses (`docs/DESIGN.md` §7).

## Open decisions
See `docs/DESIGN.md` §9. Ask before picking models, the lens spec file format, the dataset layout, a distributed runner, or a lens's base taxonomy.
