# CLAUDE.md

## Project
framekit: a non-commercial, general-purpose Python library of AI building blocks for video datasets. One package with modules: `contracts` (ports, pure shared types), `models` (model access: catalog, router, adapters, response cache), `vision` (`vision_params`: schema, checks, frame planning, pixel filters) and `extraction` (schemas, stages, runner, dataset, benchmarks), with the `framekit` command on top. New capabilities become new modules, not new packages, unless there's a concrete reason to split. The extraction module's main function is `extract(A, V) → dataset`: **A** is the given description of the search space (intent, lens specs, target moment length, scope, example queries), **V** is a video corpus (corpus profile + unit references), and an optional **plan** says how A is realized (vision, segmentation, models, quality rules, budget, formatting). The dataset holds segments, text, vectors and provenance (Parquet / Arrow + manifest).

Consumers: the video clip search agent ([short_extract](https://github.com/edyhtan/short_extract)), benchmarks, and experimental model training. The library never depends on a consumer.

- **Test corpus (example, swappable):** classic Tom and Jerry shorts — `docs/DESIGN.md` Appendix A.
- **First lens (example):** emotions & dynamics — `docs/DESIGN.md` Appendix B.

Functional design: `docs/DESIGN.md`. Every extraction setting: `docs/settings.md` (keep it in sync when a setting changes). Code layout and seeding plan: `docs/ARCHITECTURE.md`. Why it looks this way: `docs/DECISIONS.md`. Read them before architectural work.

## Decision journal (required)
Every design decision is recorded in `docs/DECISIONS.md`, the timeline journal, in the same change that applies it.
- Append a new entry at the end, under today's date heading, with local time and a short title.
- Each entry states **Decision**, **Why** (if a reason was given) and **Recorded in** (the doc sections changed).
- Never rewrite past entries. If a decision is reversed, add a new entry that points to the one it supersedes.
- Discussions that inform but don't decide may be logged as "Context gathered (no decision)".

## Diagrams in documents (required)
Always use rendered diagrams, never ASCII art: Mermaid code blocks (`flowchart` for pipelines and data flow, `erDiagram` for data, `stateDiagram-v2`, `sequenceDiagram`, `gantt`). Plain-text folder trees may stay in normal code blocks.

## Architecture rule (most important)
Layers: **contracts** (pure: ports, shared types) · **model access** (catalog, router, adapters, response cache) and **vision** (the `vision_params` schema, strategies, frame planning, pixel filters), independent siblings that never import each other · **extraction** (the schemas and lens spec format, stages, runner, dataset, benchmarks) · **cli**. See `docs/ARCHITECTURE.md` §1–§2.
- Library code and the dataset schema never reference lens concepts (emotion, dynamic, situation, gag, …) or corpus concepts (short, era, Tom, …). They enter only as data: lens specs in A, the corpus profile in V, overrides in the plan.
- No lens-specific columns: lens output lives in `annotations` / `phrasings`, validated against the lens spec.
- If a feature seems to need a lens- or corpus-specific branch, extend the lens spec format or the corpus profile instead, and discuss first.
- **No model-access libraries** (LangChain, LiteLLM, …): our own adapters on the official provider SDKs only, to keep supply-chain exposure low in code that holds API keys. Adding one needs discussion and a journal entry.
- **Model-agnostic:** no provider SDK calls, and no provider or library types, outside model adapters. Stages call ports (`video_understand`, `vision_generate`, `conversation`, `embed_*`). Models are chosen per role in the model profile. Every model output records `model_id`; switching models creates new versions and never overwrites.
- **No consumer knowledge:** no sessions, web framework, ORM, Postgres or task queue in this repo.

## Key decisions (do not change without discussion)
- **Units are the reference and are never split as a reference.** Every timestamp is on the unit timeline. Chunk files cut from the normalized copy are derived, cacheable processing artifacts; outputs are mapped back to the unit timeline; the source is never modified.
- **Perception and interpretation are separate passes.** L3a records observable facts only (core record + optional modules). Every interpretation is a text-only lens pass (L3b).
- **Fixed, versioned vocabularies per lens.** Never write free-text labels; validate against the lens spec.
- **A is given, never augmented here.** framekit doesn't create or augment A; producing A is outside the library, so augmentation strategies can be benchmarked by running their A's through the same pinned plan (`docs/DESIGN.md` §2.4). Plans derive from A only through documented, recorded rules.
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
- framekit has no environment-bound settings: configuration is plain data (Pydantic models) that consumers build. Only provider keys (and base-URL overrides) are read from the environment, by name from the provider configs (the built-in catalog's, or a registry's).
- Changes go on feature branches and reach `main` through pull requests. PRs are squash-merged, and merged branches are deleted automatically (GitHub then retargets any PR based on one to `main`). Don't stack PRs unless the owner asks for it; a stack keeps every PR above the bottom one as a draft, and is rebased onto `main` after each merge, since a squash merge leaves the next PR conflicting in `docs/DECISIONS.md` until then.
- Code checks: `uv run pytest` (the media and end-to-end tests need ffmpeg: on PATH, or `FRAMEKIT_TEST_FFMPEG=/path`; CI requires it), `uv run lint-imports`, `uv run ruff check .` (CI: the `tests` workflow runs pytest and the build on locked and newest dependencies; the `lint` workflow runs ruff, including the `S` security rules, and the import contracts; the `security` workflow runs `uv audit`, zizmor and CodeQL on PRs, `main` and daily). Keep all three passing; never weaken an import-linter contract or the vocabulary test to make code pass.
- Supply chain (`docs/ARCHITECTURE.md` §2.2): pin every GitHub Action to a full commit SHA with a `# vX.Y.Z` comment, set `persist-credentials: false` on checkout, give every workflow and job least-privilege `permissions:`, and pin tool versions (uv, zizmor). Dependencies resolve with a 7-day release cooldown (`exclude-newer`); a newer security fix gets a narrow `exclude-newer-package` exception with a comment and a removal date. Fix ruff `S` findings, or keep one with an inline `# noqa: Sxxx` and the reason. No secrets in CI.

## Current milestone
E0 done (v0.1.0: `contracts`, `models`). On `main` since: `vision`, and the E1 code (2026-10-24): `extraction` end to end and the `framekit` command, tested on generated test patterns with a scripted model. Next, the first real dataset: 5 units with two lenses and a gold set (`docs/DESIGN.md` §7); it needs the model choices first (ask) and the provider keys.

## Open decisions
See `docs/DESIGN.md` §9. Ask before picking models, changing the lens spec format or the dataset layout (both v1, 2026-10-24), picking a distributed runner, or a lens's base taxonomy.
