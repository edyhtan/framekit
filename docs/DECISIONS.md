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

### 22:07 — Code licence: MIT
- **Decision** (owner): framekit's code is licensed under **MIT**, chosen over Apache-2.0. The copyright line reads `edyhtan`, the owner's GitHub handle.
- **Where it's recorded:** the `LICENSE` file at the repository root, and in the package metadata (`license = "MIT"` and `license-files`), so the wheel and sdist carry it.
- **Why:** framekit is public and meant to be reused by others (benchmarks, training experiments, collaborators). Without a licence, nobody else may legally use it. MIT is the shortest permissive licence and matches small Python libraries (pydantic and the `anthropic` SDK use it). It's compatible with every framekit dependency (pydantic MIT, `anthropic` MIT, `openai` Apache-2.0).
- **Unchanged:** annotation packs, if ever published, stay CC-BY with no footage (DESIGN.md §8).
- **Closes:** the open item "the code licence" in the v0.1.0 entry above.
- **Recorded in:** `LICENSE` (new); `pyproject.toml`; README.md "Licence"; DESIGN.md §8.3.

### 22:31 — Extraction settings reference (docs/settings.md)
- **Decision** (owner): every setting that shapes an extraction run is written down in one reference, **`docs/settings.md`**, which becomes the spec for the E1 schemas. It contains:
  - a high-level view, without the `extract(A, V)` notation: the six questions a run answers (what should be findable, which videos, how the video is looked at, what one data point is, what is written and by which models, what rules apply), plus execution;
  - the commands (`framekit extract`, `bench` in both directions, `cache`) and the Python calls;
  - complete example files: search space, corpus, plan by group, gold set;
  - the output and its manifest;
  - which groups are recorded and which are hashed.
- **Status tags:** every setting is tagged `[built]`, `[designed]`, `[discussed]`, `[open]` or `[fixed]`. All numbers are placeholders until benchmark sweeps set them.
- **First written here** (agreed in conversation, now `[discussed]`):
  - derivation rules v1;
  - views per segment;
  - the keyframes profile, fixed time windows, and beat limits under perception;
  - entity policy, per-stage budgets, `on_exceed`, and formatting layout outside the hash;
  - a corpus `subset` and an alignment override;
  - gold-set overlap matching, and a ban on gold queries in search spaces;
  - default work/cache/output folders outside any repository, each with a `.gitignore` of `*`;
  - always-resume runs, with `--rerun` to force a stage.
- **Cross-checked** by three independent reviewers (docs coverage, rules, the agreed list). Their fixes are applied:
  - models are named only under `models:`; embedding refers to roles;
  - native video takes only uniform sampling;
  - rows are never dropped;
  - the plan can't widen the scope;
  - gold moments fit the declared resolution;
  - each self-consistency run gets its own seed.
- **Doc fixes:**
  - DESIGN.md §2.3's example used dedupe, a frame budget and JPEG settings with `native_video`, which breaks the section's own rule. It's corrected.
  - ARCHITECTURE.md §3 still named the CLI flag `--context` (the old framing); it's now `--space`, with `cache_settings`, `dry_run` and the runner options added.
- **Open** (settings.md §8):
  - whether frames may be sent to remote models for a bring-your-own corpus (`model_inputs`);
  - adding temperature and seed to `ConversationRequest` and the cache key, so repeated runs don't replay one answer;
  - whether a "drop" policy should ever exist.
- **Recorded in:** docs/settings.md (new); DESIGN.md §2.3, §2.4, §9; ARCHITECTURE.md §3; README.md "Documents"; CLAUDE.md "Project".

### 22:36 — Ruleset protecting `main`
- **Decision** (owner): framekit's default branch is protected by an active GitHub ruleset, **"protect main"**, with no bypass actors (not even admins). Its rules:
  - **no force pushes** and **no deletion** of `main`;
  - **a pull request is required** (0 approvals, single maintainer);
  - **these checks must pass**, and only GitHub Actions may report them: `lint and layer rules`, `unit tests (locked dependencies)`, `unit tests (newest dependencies)`.
  - "Require branches to be up to date" is off, so not every PR needs a rebase. All merge methods stay allowed; merge commits are needed to keep seeded history.
- **Why:** the repository is public. History on `main` must never be rewritten, and nothing should land without CI.
- **Recorded in:** GitHub repository settings (ruleset 25767064; created directly, so recorded here afterwards); README.md "Development". New CI checks (the `security` workflow) are added to the ruleset once they have run on `main`.

### 22:41 — Security checks: supply-chain hardening for the public library
- **Decision** (owner, approving the proposed rollout): framekit gets a security baseline in CI and in its dependency resolution.
  - **`security` workflow** on PRs, pushes to `main` and daily: `uv audit` of `uv.lock` (OSV advisories); **zizmor** on `.github/` (offline audits, fails on medium severity or higher); **CodeQL** for Python and for the workflows (`actions`). Only the CodeQL job may write, to upload its results.
  - **Workflow hygiene** in all three workflows: every action pinned to a full commit SHA with a `# vX.Y.Z` comment; `persist-credentials: false` on checkout; least-privilege `permissions:` per workflow and per job; uv pinned through `setup-uv` (`0.13.5`), zizmor pinned (`1.31.1`); matrix values reach shell scripts through `env:`. No job uses a secret.
  - **Release cooldown:** `[tool.uv] exclude-newer = "7 days"`. A security fix newer than that gets a narrow `exclude-newer-package` exception with a comment and a removal date. The `newest` test run now means the newest releases outside the cooldown.
  - **Dependabot** for uv and GitHub Actions: weekly, 7-day cooldown for version updates, minor and patch updates grouped. Dependabot's own cooldown doesn't delay security updates, but `exclude-newer` still does: a fix younger than 7 days needs the narrow `exclude-newer-package` exception before Dependabot or `uv lock` can pick it up.
  - **Ruff `S` (bandit) rules.** Tests may use `assert` (S101); the hardcoded-secret rules stay on in tests too. The one finding in library code, an `assert` in the OpenAI-compatible adapter's `call` (the path `generate` takes when the provider streams), became the explicit `ModelCallError` check that `stream` already uses, with a test (an `assert` disappears under `python -O`).
  - **`SECURITY.md`:** private reporting through GitHub's "Report a vulnerability", supported versions (0.1.x), scope, modest response expectations.
- **Why:** framekit is public and its adapters handle provider keys. Recent attacks came through freshly published releases (LiteLLM's compromised PyPI releases were live for about two hours before an advisory) and through rewritten action tags (76 of 77 trivy-action tags). Advisory scanners only catch what has an advisory; the cooldown and SHA pins reduce exposure before one exists.
- **Lock changes** (re-locked under the cooldown; `uv audit` is clean before and after, so none of the skipped releases was a security fix): anthropic 1.17.0 → 1.15.0, openai 3.37.0 → 3.33.0, httpx2 and httpcore2 2.16.0 → 2.13.1, jiter 0.18.0 → 0.17.0, ruff 0.17.2 → 0.17.1. All tests pass on the new lock (90, with the new one).
- **Needs the owner** (repository settings, not code): secret scanning and push protection; private vulnerability reporting (`SECURITY.md` relies on it); Dependabot alerts and security updates; immutable releases and a ruleset protecting `v*` tags, before tagging the next release; adding this workflow's checks to the `main` ruleset once they have run on `main` (the ruleset itself is in the entry above); the Actions policy requiring SHA-pinned actions. Code scanning's default setup must stay off, or the CodeQL workflow's uploads are rejected.
- **Open:** behavioural malware detection for new dependency releases (e.g. Socket, GuardDog), which doesn't wait for an advisory: the main remaining gap.
- **Recorded in:** `.github/workflows/security.yml` (new), `tests.yml`, `lint.yml`; `.github/dependabot.yml` (new); `SECURITY.md` (new); `pyproject.toml` (`[tool.uv]`, ruff); `uv.lock`; ARCHITECTURE.md §2 (lockfile row), §2.2 (new); README.md badges, "Development", "Documents"; CLAUDE.md "Workflow".

## 2026-10-24

### 03:16 — `framekit.vision`, first code: vision parameters as checked, resolved data
- **Decision** (owner: "create vision param now", taking up the proposed first step): `framekit.vision` gets its first code, covering everything that needs no pixels and no model choice.
  - **Schema:** `VisionParams`, `VisionProfile` and their parts. They're Pydantic models: frozen, and unknown keys are rejected. `from_mapping` / `to_mapping` use the shape of settings.md §5.2. YAML or JSON parsing stays with the caller, so there's no new dependency.
  - **Where it lives:** `framekit.vision.params`, not `contracts`. The "Contracts are pure" import contract forbids Pydantic there, and contracts are never weakened.
  - **Import contracts:** the layers contract becomes `framekit.vision | framekit.models` above `framekit.contracts`: independent siblings, checked in both directions. It replaces "models → contracts" and keeps all of it. The provider-SDK contract now covers `vision` too.
  - **Strategy registry:** strategies are named, versioned code. Each has a kind (`candidates`, `anchor`, `filter`, `allocation`), a settings schema and a `needs_pixels` flag.
    - Kinds are separate namespaces: the `audio_onsets` anchor and the `audio_onsets` allocation signal are two strategies.
    - Registering an existing kind + name + version raises.
    - A reference without a version resolves to the newest version, and the resolved form records it.
    - Version 1 registers candidates `uniform` and `shot_based`; anchors `shot_boundaries` and `audio_onsets`; filters `dedupe` and `sharpness`; allocation signals `motion` and `audio_onsets`. The two filters need pixels: they're checked and recorded now, and run in the media stage.
  - **References** come in three shapes, all with the same meaning: a name (`name@N` adds a version), `{name: {settings}}`, or inline `{strategy: name, settings…}` (the candidates shape in the docs).
  - **Load-time checks** raise one `VisionParamsError` that lists every problem with its path:
    - unknown or misplaced strategies, and settings that fail a strategy's schema;
    - native video with anything but uniform candidates, `max_side_px` and `aspect`;
    - views that don't fit `max_frames_per_scene`;
    - any `aspect` other than `preserve`, and a quality setting for png;
    - stages that name a missing profile.
  - **Checks that need outside facts are separate functions:** `check_against_source` (the analysis copy's fps and long side) and `check_against_model`. A model's limits are described by `ModelVisionCaps`, a dataclass in `framekit.vision`, because the model catalog doesn't declare vision capabilities yet.
  - **Resolution:**
    - versions are pinned and defaults filled in: delivery `images`; `jpeg` at quality `85` for images and storyboards; a 2×2 storyboard grid; allocation `window_s` 1.0. All are placeholders;
    - the form is canonical JSON: sorted keys, no unset values, anchors and allocation signals sorted as sets, filters kept in order;
    - SHA-256 digests are computed per profile (content only, not the name) and for the whole block.
  - **Pixel-free planning** (`plan_frames`):
    - anchors are kept first;
    - the rest of the cap (the tighter of `max_frames` and `max_frames_per_scene`) goes to the candidates with the highest normalized allocation weight;
    - `min_gap_s` holds between kept frames;
    - `overview_plus_detail` views are planned;
    - native video gives nominal uniform timestamps, with a note;
    - pixel filters come back as pending;
    - `split_by_budget` and `per_module` views are checked and recorded but not planned: planning them raises an explicit error, never a silent fallback.
- **Choices made while building** (each is easy to revisit):
  - **Budget ties go to the candidate farthest from the frames already kept, then to the earliest,** rather than simply the earliest. With no signal, earliest-first would spend the whole budget at the start of a scene; spreading keeps coverage and is still deterministic.
  - **`candidates` is required for native video too,** so the fps the provider samples at is always recorded.
  - **Numeric strategy settings have no defaults** (`fps`, `pick`, a filter's `max_distance` or `window_s`): a missing setting fails instead of being guessed. Settings with a single possible value (`method: phash`, `keep: best_per_window`) and allocation windows do have defaults.
  - **A storyboard tile is one frame of `max_side_px`.** The grid image's long side is that times the larger grid dimension, and that's what the model check compares with the model's limit.
  - **In views, anchors count toward the scene cap** and belong to the overview.
  - **`shot_based` `last` is 1 ms before the cut,** which falls inside the shot's last frame at any frame rate.
- **Why:** this is the first step that doesn't depend on the open choices (the video model, and default values from a sweep). With the schema, checks and digests in place, the E1 plan, manifest and stage input hashes can build on them, and query-time consumers can already plan frames.
- **Open:** default values (from a benchmark sweep); vision capabilities in the model catalog; how the media stage feeds pixel-filter results back into the budget; planning for `split_by_budget` and `per_module` views.
- **Recorded in:** `src/framekit/vision/` (new); `tests/test_vision_params.py`, `tests/test_vision_planning.py`, `tests/vision_examples.py` (new); `pyproject.toml` (import contracts); `src/framekit/__init__.py`; DESIGN.md §2.3, §7 (E1); settings.md status, tags, §5.2; ARCHITECTURE.md status, §1, §2, §2.1, §3; README.md module table, "Vision parameters" (new); CLAUDE.md "Project", "Architecture rule", "Current milestone".

### 03:40 — Your own adapters and providers: a model registry and a public adapter template
- **Decision** (owner, approving the proposed "supported template" step of model access): consumers add their own models, providers and adapters in their own code, without a change to framekit.
  - **`ModelRegistry`** (`framekit.models.registry`): an immutable object holding providers (by name), models (by `model_id`) and adapter classes (by wire protocol). Its immutability is shallow: entries are shared with the built-ins and between registries, so they're replaced (`dataclasses.replace` plus `extend(..., override=True)`), never mutated. It's checked when built and reports every problem in one `ValueError`: the arguments' shapes, names (non-empty strings), key variables, prices (`Decimal`, since a float would fail only after the call), every model's provider is configured, every provider's protocol has an adapter, and every adapter class can work (concrete, with an `adapter_version`, `make_client` and `from_environment` as classmethods). `extend(providers=, models=, adapters=, override=False)` returns a new registry; a name that already exists is an error unless `override=True`. `resolve`, `create_adapter` (an unknown provider or protocol is a non-retryable `ModelCallError`, never a `KeyError`) and `listing`. A registry pickles: `DEFAULT_REGISTRY` by reference, any other by value.
  - **`DEFAULT_REGISTRY`** serves the built-in `PROVIDERS`, `CATALOG` and `PROTOCOLS` (`PROTOCOLS` moved from `router` to `registry`). `ConversationRouter` gains a keyword `registry=`; a passed `adapter_factory` (any but `None`) still builds the adapters. The router calls it for every provider it uses, so a factory for one hand-built adapter hands the others to `registry.create_adapter`.
  - **Error hook:** `ExternalAIAdapterBase.sdk` is optional (default `None`), and the base maps exceptions through an overridable `map_error(exc)`: an `APIError` of `sdk` goes through `map_sdk_error`, anything else propagates as a bug. A `ModelCallError` raised by `call` or `iterate` always passes through unchanged; `map_error` never sees it. `map_error` returns a `ModelCallError` or `None`; anything else is a `TypeError` chained to the original. `make_client` is no longer abstract, so an adapter built directly doesn't need it.
  - **Optional keys:** `ProviderConfig.key_env` may be `None` (a keyless server). The adapter is then built with `PLACEHOLDER_KEY` (`"framekit-no-key"`), never with `None`: given no key, the `openai` and `anthropic` SDKs fall back to `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` or `ANTHROPIC_AUTH_TOKEN` and would send a real cloud key to a custom host.
  - **Public API:** `framekit.models.adapters` exports `ExternalAIAdapterBase`, `Reply`, `PLACEHOLDER_KEY` and the request helpers (`conversation`, `system_text`, `structured_mode`, `reasoning_active`, `reasoning_fields`, `merge_fields`, `output_budget`, `normalize_stop_reason`, `map_sdk_error`); with `ModelRegistry` and `DEFAULT_REGISTRY` they're covered by semantic versioning. The built-in adapter classes stay internal. The base gets a `client` property (assignable too, so adapters don't reach for `_client`), refuses a missing or empty `adapter_version` when an adapter is built, before any call is paid for, and its default `aclose` handles clients with `aclose()` (like `httpx`) and with a synchronous `close()`.
- **Why:** a probe that plugged a user adapter into the router showed five rough edges: models were only resolved from the global `CATALOG`; a base without `sdk` turned any error, even a clean `ModelCallError`, into an `AttributeError`; the base's error mapping assumed the `openai` and `anthropic` SDKs; `from_environment` always required a key variable; and the base wasn't public API. Each is fixed above, and the probe's error case now reports the clean `ModelCallError`.
- **Reviewed:** independent reviewers checked compatibility, adversarial inputs and the docs, and their fixes are applied above: the `ModelCallError` pass-through and `map_error`'s return value are enforced, not just documented; a registry validates argument shapes, names, prices and adapter classes; a falsy `adapter_factory` is still used; `client` is assignable; `aclose` covers more client shapes; registries pickle. In the docs: hand-built adapters and their factories, `make_client` as a classmethod, shallow immutability, the Bedrock fallback relying on an internal class, two more security gaps stated, and the cache key's limits in the README.
- **Kept compatible:** every existing test passes unchanged. `CATALOG`, `PROVIDERS`, `PROTOCOLS`, `create_adapter`, `catalog_listing` and `ConversationRouter(settings, adapter_factory)` keep working, and `framekit.models.router` still re-exports the names it had before the registry (`CATALOG`, `PROVIDERS`, `PROTOCOLS`, `ModelCallError` and the built-in adapter classes). The default registry reads the built-in dicts live, so code that edits them in place still reaches the default router; that's now documented as unsupported, in favour of `extend`. Two changes, neither used by a known consumer: rebinding those module names (`router.CATALOG = {...}`, or `monkeypatch.setattr` on them) no longer changes what a router uses, since the default registry holds the dicts themselves; and an adapter class without `adapter_version` now fails when it's built instead of after its first call.
- **Not verified:** Claude on Amazon Bedrock as config only. Bedrock's Mantle endpoint speaks the Messages API, but the `anthropic` SDK's own Bedrock client sends a Bedrock API key as `Authorization: Bearer`, while the Anthropic adapter sends `x-api-key`. The guide says to probe it, and that an adapter building the SDK's Bedrock client is the way if config is rejected.
- **Deferred:**
  - providers from configuration files (`from_data`);
  - the three security fixes for config-driven providers: stripping SDK environment headers (`OPENAI_CUSTOM_HEADERS`, `ANTHROPIC_CUSTOM_HEADERS`, `OPENAI_ORG_ID`, `OPENAI_PROJECT_ID`, which are still sent to custom hosts), region and hostname validation of base URLs (including the scheme: an `http://` URL sends the key in clear text), and binding keys to hosts (any provider config may name any key variable). All three are known gaps, stated in the guide and `SECURITY.md`;
  - a host fingerprint in response-cache keys (until then, registries that map one `model_id` to different hosts need separate cache namespaces);
  - cloud adapters (e.g. Bedrock or Vertex authentication);
  - provider locality (local or remote).
- **Recorded in:** `src/framekit/models/registry.py` (new), `router.py`, `catalog.py`, `__init__.py`, `adapters/base.py`, `adapters/__init__.py`; `tests/test_custom_adapters.py` (new); `pyproject.toml` (the "Provider SDKs only inside the adapters" contract now covers `framekit.models.registry`); docs/adapters.md (new); README.md "Configure", "Models", "Your own models and providers" (new), "Response cache", "Documents"; ARCHITECTURE.md §1, §2 (layout), §3, §5; `SECURITY.md` "Scope"; CLAUDE.md "Workflow".

### 04:10 — Vision parameters: fixes after review, before merge
- **Decision** (fixes to the 03:16 code from independent reviews; each is easy to revisit):
  - **One error lists every problem, for real.** Each strategy reference is checked where it sits (known for its slot, settings valid), next to the structural checks. The rules that tie settings together (native video, views, png quality, storyboard grid, listed once) run on whatever parsed. A rule is skipped only where a setting it reads has a problem of its own, which is listed instead. Problems are grouped by part, in schema order.
  - **Strict values:** a number must be a number (YAML `yes` or `"4"` fails) and finite; a whole number is fine where a decimal is expected; list settings must be lists. A reference keeps its checked settings, not the raw input. `views.detail.around` needs at least one signal; `views.detail.span_s` is at least 1 ms.
  - **Read-only all the way down:** strategy settings are a read-only, hashable dict and lists are tuples, so parsed profiles are hashable; `vision_profiles`, `stages` and the resolved `profiles` are read-only too. A resolved profile keeps the canonical JSON its digest was computed from, and `.mapping` returns a copy, so the digest always describes what runs.
  - **The same frame is the same millisecond.** This supersedes the 1 ms merge of 03:16: uniform candidates stay distinct up to 1000 fps, and `shot_based` `last` (1 ms before a cut) and an anchor on the cut are two frames. Timestamps and the span's edges are rounded to the nearest millisecond, halves up; a span must hold at least 1 ms.
  - **Budget fill.** This supersedes 03:16's "ties go to the candidate farthest from the frames already kept": highest weight first; candidates of equal weight in time order, as many as `min_gap_s` allows, at a regular spacing; a spread subset (farthest from the frames kept, ties to the earliest) only when the slots left can't take them all. A cap is now filled whenever `min_gap_s` allows it. Planning is O(n log n): an uncapped hour at 30 fps plans in 0.3 s here, 2.4 s before.
  - **Views.** This refines 03:16's "anchors count toward the scene cap and belong to the overview": the anchors count toward `overview.max_frames`, so the detail views keep their frames unless the anchors alone outnumber the overview's. The load-time check compares the views with the tighter of `max_frames_per_scene` and `max_frames`, as planning does.
  - **Detail views are centred on their peak:** the middle of a run of candidates of equal weight, higher than the candidates on either side. Windowed signals make such a plateau around each event, so the middle is the event, or a cluster's middle. Before, ties went to the earliest candidate: a view sat up to half an allocation window early and could miss its event. A second view goes to another peak, not down the slope of the first. A peak cut by the span's edge is centred on its part inside the span: with an allocation `window_s` over twice `span_s`, an event near the edge can fall outside its view (documented).
  - **Inputs:** in `Signals`, a time listed twice counts once (no zero-length shot), and wrong types raise `TypeError` naming the field; motion works on any scale (weights relative to the strongest sample). `SourceLimits` and `ModelVisionCaps` check their values, and the outside checks reject wrong argument types with `TypeError`. One strategy gives at most 2,000,000 candidates per span.
  - **Docs:** ARCHITECTURE.md still put the Pydantic A / V / plan / dataset schemas in `contracts`, where the import contracts forbid them (no Pydantic, and the plan embeds `VisionParams`). Their place is now marked open. DESIGN.md §2.3 says `dedupe@1`, not `phash@1`. The clip range of native video is given when a stage runs, not in a profile.
  - **Tests:** a fixture isolates the strategy registry, so the suite passes in any order and twice in one process. A golden digest pins the documented example's canonical form; changing it changes every recorded digest and needs an entry here.
- **Why:** the reviews found that a structural error hid every other problem in its profile; that "frozen" stopped at the top level, so a resolved profile could change under its digest; that the 1 ms rule dropped real frames; that the farthest-first fill kept fewer frames than `min_gap_s` allows; that detail views sat beside their peaks; and that ARCHITECTURE.md placed the plan schema where the import contracts forbid it.
- **Open:** where the Pydantic schemas live (decide before E1); keeping `budget.max_frames` across the spans of a unit (planning applies it per span).
- **Recorded in:** `src/framekit/vision/` (params, resolve, planning, builtin, signals, checks, the package docstring); `tests/conftest.py` (new), `tests/test_vision_params.py`, `tests/test_vision_planning.py`; DESIGN.md §0.1, §2.3, §9; settings.md §5.2; ARCHITECTURE.md §1, §2, §2.1, §3, §8; README.md "Vision parameters".

### 04:15 — Gemini integration paused (context gathered)
- **Context gathered** (desk research and a code reading, 2026-10-24; no live probe, since no Gemini key was available):
  - **Text roles:** Google's OpenAI-compatible endpoint (`https://generativelanguage.googleapis.com/v1beta/openai/`, labelled beta) can serve the text-only `conversation` port through the existing OpenAI-compatible adapter, with one provider entry and no new dependency. Three fixes are needed first:
    - current Gemini 3.x and 4 models can't switch thinking off, which conflicts with "reasoning off unless asked" and its test;
    - `completion_tokens` reportedly leaves out thinking tokens, so cost would be undercounted unless they're inferred from `total_tokens`;
    - a blocked prompt can return no message.
  - **Images:** the endpoint accepts base64 images, but the port can't carry images yet.
  - **Native video:** impossible through the compatible endpoint, which has no video input and no fps, clip-range or resolution settings. It needs a `video_understand` port plus a native adapter, either on the official `google-genai` SDK as an optional extra (about 16 more packages, including a second HTTP stack) or on REST over httpx2. For a `byo_media` corpus it must also send `store=false`, use a paid-tier key, and put media identity in the cache key.
  - **Owner choices left open:** whether "always reasons" models are allowed; Interactions API vs generateContent (only the latter has 50%-off batch and explicit caching); and whether Gemini is the video model at all, which should wait for a timestamp-precision test on a gold set.
- **Decision** (owner): Gemini integration is paused. Nothing for it is built: no provider entry, no adapter, no dependency.
- **Recorded in:** DESIGN.md §9 ("Video model").

### 04:31 — `ModelSource`: one structure for model, provider, credential and adapter
- **Decision** (owner: "one single structure to identify both model and credentials (or a custom adapter)"; name chosen by the owner: `ModelSource`):
  - **`ModelSource(name, model, provider=None, credential=None, adapter=None)`** (`framekit.models.sources`) says which model, where it's served, with which key and through which adapter. Requests and `ModelSettings.default_model` name a source; the router argument is `sources=[...]`.
  - `model` is a registry model id (e.g. a built-in) or your own `ModelSpec`, whose `model_id` is what's sent. `provider` is a `ProviderConfig` or a registry provider name, and wins over the spec's own `provider` field. `credential` is a string, a `SecretStr` or a callable (called once per router). `adapter` is an adapter class, never an instance, since the router opens and closes clients per unit of work.
  - **The registry stays** as the lower-level container: everything that isn't a source resolves through it, so the built-in catalog and environment keys work as before, and sources are checked against it when the router is built (every problem in one `ValueError`). A source reusing a catalog id overrides it for that router only. An `adapter_factory` builds registry models only; sources build their own.
  - **Explicit keys:** the adapter template gains `from_key(config, api_key, settings)`; `from_environment` reads the key and calls it, so one override covers both paths. Precedence: the source's credential, then the provider's `key_env`, then the keyless placeholder.
  - **Credentials are never recorded:** stored as a `SecretStr` (hidden in reprs), passed only to `make_client`, kept out of results, cache keys and fingerprints, and a source holding one refuses to be pickled.
  - **Provenance and cache:** `ConversationResult.source` names the source that served a call (`None` for registry models). The response cache keys a source by its model id plus a public fingerprint (protocol, base URL, adapter class and version), so two hosts serving the same model id never share answers. Registry models keep their plain model id, so existing cache keys are unchanged.
- **Why:** one object per model-and-access instead of three router arguments (`registry=`, a separate `credentials=`, `adapter_factory=`), and the explicit-key request (keys as arguments, with the environment as the fallback) is covered by the same structure. It also gives the host fingerprint the 03:40 entry deferred, for sources.
- **Supersedes:** the separate `credentials=` router argument proposed in conversation (never built), and, for sources, the deferred "host fingerprint in response-cache keys" in the 03:40 entry.
- **Recorded in:** `src/framekit/models/sources.py` (new), `router.py`, `cache/port.py`, `adapters/base.py` (`from_key`), `contracts/conversation.py` (`ConversationResult.source`), `models/__init__.py`; `tests/test_model_sources.py` (new); `pyproject.toml` (the provider-SDK contract covers `sources`); docs/adapters.md (now led by `ModelSource`), README.md "Configure" and "Your own models and providers", ARCHITECTURE.md §3.

### 05:37 — Model access for extraction: images, sampling, embeddings
- **Decision** (needed by extraction, which the owner asked to finish: "finish the framekit.extraction layers"):
  - `ConversationRequest` gains `images` (`ImagePart`: bytes and a media type, jpeg, png or webp), `temperature` and `seed`. Both adapters send images before the prompt's text (Anthropic base64 image blocks; OpenAI-compatible `data:` URLs) and pass the sampling settings; Anthropic's Messages API has no seed, so its adapter leaves it out.
  - Catalog entries declare `vision`, `max_images` and `max_image_side_px`. A request with images to a model whose entry has `vision=False` fails before any call (non-retryable). The Claude values are from platform.claude.com/docs/en/build-with-claude/vision (2026-10-24): Haiku 4.5 takes 100 images per request at a 1,568 px native long edge; Sonnet 5 and Opus 5 take 600, capped here at 2,000 px because a request with more than 20 images rejects any larger side.
  - **The cache key** adds the images' digests (never their bytes), `temperature` and `seed`, each only when set, so every existing key is unchanged.
  - **Embeddings (`embed_text`):** `EmbeddingRequest` / `EmbeddingResult` / `EmbeddingPort` in `contracts`; the OpenAI-compatible adapter serves `/embeddings` (vectors in request order, the count checked, errors through the shared mapper); `router.embed()`; the response cache passes embeddings through uncached. An adapter without embeddings keeps the template's default, a non-retryable error.
  - **The file response store** keeps entries in `<namespace>/<class>/` folders (taken from the cache keys), with `stats()` and `clear(cache_class=, namespace=)`, so one class can be counted or wiped. Entries recorded in the old flat layout are counted and wiped but no longer read; nothing released used it.
- **Why:** L2 and L3a send frames; self-consistency runs must be distinct requests with distinct cache keys (settings.md §8, decision 2); the dataset needs text vectors.
- **Supersedes:** settings.md §8 decision 2 (temperature and seed in the request and the cache key), now done.
- **Recorded in:** `src/framekit/contracts/` (`conversation.py`, `embedding.py` new, `__init__.py`); `src/framekit/models/` (`catalog.py`, `router.py`, `adapters/base.py`, `adapters/openai_compatible.py`, `adapters/anthropic.py`, `adapters/__init__.py`, `cache/entries.py`, `cache/port.py`, `cache/file_store.py`); `tests/test_model_media.py` (new), `tests/test_cache.py`, `tests/test_custom_adapters.py`; ARCHITECTURE.md §5; settings.md §5.2, §8; README.md "Response cache".

### 14:40 — Extraction v1, part 1: where the schemas live, and the lens spec format
- **Decision** (owner: "finish the framekit.extraction layers"; then "break it down into shippable components", so extraction lands as a series of PRs, this being the first). Each choice is v1 and revisitable; CLAUDE.md asked to discuss the lens spec format before picking it, so it's listed here for review.
  - **Where the schemas live:** `framekit.extraction.schemas` (decides the open question of ARCHITECTURE.md §8). `contracts` stays free of Pydantic; the plan will embed `VisionParams`, so the schemas sit above `vision` and `models`. Consumers that only read lens specs or datasets won't need the media extra.
  - **Lens spec format v1:** YAML or JSON (a file, or a folder with `lens.yaml`), prompts inline or `{file: path}` inside the spec's folder (a spec from someone else can't make framekit read, and send to a model, any other file). Typed fields: `label`, `labels`, `integer`, `number`, `boolean`, `object`, `list`, and `map` keyed by the corpus profile's entities (plus `extra_keys`) or a vocabulary; any field may be `optional`; no free-text type; labels lower snake case. Validators: `required`, `exclusive`. `consumer` is carried verbatim for search agents, never read by extraction.
  - **What a lens pass will send and check:** the fields compile into a JSON Schema (`scene`, `beats` by index, `phrasings`); every reply is checked against the fields, the vocabularies, the beat indexes and the phrasing count, whatever the provider enforced. Annotations become one row per label, or one row with the JSON value for other fields.
  - **The spec's digest** covers its canonical form, prompts included, so any change to what a lens pass sends or checks changes it.
  - **Loading conventions** shared by every schema to come: unknown keys fail, values are strict (numbers are numbers, ids are strings), lists are stored as tuples, and one `ConfigError` lists every problem with its path. PyYAML joins the base install, read with the safe loader only.
- **Why:** lenses are data (DESIGN.md §4), and both extraction and consumers (the search agent's query mapping and judge) read the same spec; a fixed, typed format keeps labels out of free text.
- **Recorded in:** `src/framekit/extraction/` (new: `__init__.py`, `schemas/__init__.py`, `schemas/common.py`, `schemas/lens.py`), `src/framekit/__init__.py`; `tests/test_lens_specs.py`, `tests/extraction_helpers.py` (new); `pyproject.toml` (PyYAML; the layers contract gains `extraction`; yaml only in `extraction`; the provider-SDK contract covers `extraction`); DESIGN.md §0.1, §4, §7, §9; settings.md status, §3, §3.1 (new), §8; ARCHITECTURE.md status, §1, §2, §2.1, §3, §8; README.md module table, "Extraction (preview)"; CLAUDE.md "Project", "Architecture rule", "Open decisions"; SECURITY.md "Scope".

### 14:45 — Extraction v1, part 2: the run inputs (search space, corpus, plan, gold set)
- **Decision** (the second PR of the extraction series; v1 choices, each revisitable):
  - **Search space (A):** lens references are paths (relative to the space's file) or loaded specs, and each spec's version must match the space's; the digest covers the space as given plus every lens spec's digest. `scope.excluded_ranges` is `from_corpus` or `none`.
  - **Corpus (V):** declared metadata fields may sit directly on a unit entry and are checked against the profile (types or allowed values); unit ids are strings of letters, digits, `_`, `-` and `.` (they become file names; YAML `042` is refused, not read as a number); relative paths are read from the corpus file's folder. The digest leaves paths out (they differ between machines).
  - **Plan:** every setting of docs/settings.md §5, with documented defaults (placeholders). Choices: segmentation has two methods, `model` (the L2 role) and `shot_groups` (no model), and `fixed_windows` alone is refused because perception runs on scenes; models are never defaulted; `vision_profiles` and `stages` sit at the plan's top level and each replaces its default (a storyboard overview for L2, up to 16 frames per scene for L3a); self-consistency needs one distinct seed per run; `governance.model_inputs` has no default (how resolution enforces it comes with resolution); `units`, `scenes`, `runs` and `qc` can't be left out of `formatting.tables`.
  - **Gold set:** fixed per corpus; `overlapping_queries` finds a search space's example that repeats a gold query (the same words, or 80% in common).
- **Why:** the run's inputs are data a user writes; they're checked completely before anything else is built on them, so every later stage reads valid, read-only values.
- **Recorded in:** `src/framekit/extraction/schemas/` (`space.py`, `corpus.py`, `plan.py`, `gold.py` new; `__init__.py`), `src/framekit/extraction/__init__.py`, `src/framekit/__init__.py`; `tests/test_run_inputs.py` (new), `tests/extraction_helpers.py`; settings.md status, tags, §2 (the import), §3, §4, §5 intro, §5.2–§5.5 (the settings), §6; DESIGN.md §2.3, §7, §9; ARCHITECTURE.md status, §3, §8; README.md module table, "Extraction (preview)"; SECURITY.md "Scope".

### 14:50 — Extraction v1, part 3: the task store, budget and rate limits, and the dataset format
- **Decision** (the third PR of the extraction series; v1, revisitable; CLAUDE.md asked to discuss the dataset layout before picking it, so it's listed here for review):
  - **Dataset layout v1:** Parquet per table and unit (`tables/<table>/unit-<id>.parquet`, or one file per table), fixed column schemas per table, free-form values (unit metadata, perception records, annotation values) as JSON text, vectors as float32 lists, and `manifest.json` written last with every file's row count and sha256. `Dataset.open(path, verify=)` refuses an unknown schema version and checks the checksums. A folder holding another dataset is never overwritten.
  - **The task store:** each step of a run is a task keyed by a digest of its inputs, stored as its run record plus output in the work folder (`tasks/<stage>/<xx>/<hash>.json`, written atomically) before anything uses it. A finished task is reused; `rerun=` forces a stage; a failed task is never stored, a flagged one is (so a rerun doesn't pay for it again).
  - **Model calls:** a retryable error (transport, rate limit, overload) is retried with exponential backoff up to `provider_retries`; per-provider rate limits are shared by every stage; the budget is checked before each call against the adapters' cost estimates. When a cap is reached, `stop` fails the next call; `finish_unit_then_stop` lets the units in progress finish (the runs will start no new unit).
  - **pyarrow `>=23.0.1`** joins the base install, the first release without PYSEC-2026-113 (CVE-2026-25087, found by the dependency audit on the combined PR), with no upper bound: its majors come every few months without breaking the table and Parquet calls framekit makes (checked on 23.0.1 and 26.0.0), and the newest-dependencies CI job would catch one that did.
- **Why:** consumers (the search agent's loader, training experiments) read datasets with any Parquet reader, so the format is fixed and self-describing; idempotent tasks make runs resumable and reruns free where nothing changed (DESIGN.md §5.1).
- **Recorded in:** `src/framekit/extraction/runner.py`, `dataset.py`, `schemas/manifest.py` (new), `schemas/__init__.py`; `tests/test_runner.py`, `tests/test_dataset.py` (new); `pyproject.toml` (pyarrow; the pyarrow and yaml contract); settings.md §7, §8; DESIGN.md §2.1, §9; ARCHITECTURE.md §2.1, §3, §8; CLAUDE.md "Open decisions".

### 14:55 — Extraction v1, part 4: resolution (derivation rules v1 and the checks before any spend)
- **Decision** (the fourth PR of the extraction series; v1, revisitable):
  - **Derivation rules v1**, deterministic and each recorded with its inputs: perception modules (core + the search space's evidence + every lens's needs; a plan may add, never remove), beat lengths (max = the longest moment, min 0.5 s), fixed windows (the shortest moment, with `mode: both`), the phrasing style (the space's examples, else each lens's style), the scope (it can only narrow the corpus), the lenses. The vision profiles aren't derived: they're explicit, so nothing adapts them behind the user's back.
  - **Checks, every problem in one `ConfigError`:** unknown unit ids in the scope; a non-identity `alignment_override` (alignment isn't built); a model for each role that runs (L2 with `segmentation.method: model`, and L3a; the lens-pass and embedding roles join with their stage), known to the router (catalog, registry or `ModelSource`); frame-sending roles need a vision model whose profile fits its `max_images` and `max_image_side_px`; native video is refused until a `video_understand` port exists; candidate frame rates can't exceed the analysis copy's.
  - **Governance:** for a `byo_media` corpus, a run that sends frames needs the plan to say `governance.model_inputs` (no default); `local_only` is checked against each frame-sending role's base URL (localhost, or a loopback, private or link-local IP); `image_vectors` and `training_use` stay `local_only`. Other corpora default to `provider_transient`. This settles how settings.md §8 decision 1 is enforced; which answer to use stays per plan, the owner's call.
  - **Recorded:** the resolved plan in canonical form with three digests: everything (`plan_digest`), without budget, governance and layout (`content_digest`), and also without what the derivation rules wrote (`fairness_digest`, which the benchmark harness will compare).
- **Why:** nothing should be spent on a run that can't finish or isn't allowed; and plans derive from A only through documented, recorded rules (DESIGN.md §2.4).
- **Recorded in:** `src/framekit/extraction/resolve.py` (new); `tests/test_resolve.py`, `tests/model_helpers.py` (new); settings.md §4, §5.1, §5.4, §5.5, §8; DESIGN.md §9; ARCHITECTURE.md §4, §5.

### 15:00 — Extraction v1, part 5: media work (ffmpeg, signal detectors, frames, pixel filters)
- **Decision** (the fifth PR of the extraction series; v1, revisitable):
  - **ffmpeg is a system binary**, run as a subprocess with argument lists (never a shell), found on PATH or given as a path; framekit needs only `ffmpeg`, not ffprobe (probing parses `ffmpeg -i`: duration, the first video stream with cover art skipped and rotation applied, the first audio stream). Its version will be recorded in each manifest.
  - **The analysis copy** (`normalize@1`): constant fps, at most `height_px` tall and never upscaled, square pixels with the displayed aspect, H.264 8-bit 4:2:0 with keyframes every 2 s, mono or stereo AAC, no metadata, subtitles or data streams. The source is only read.
  - **L1 detectors are named, versioned code**, like vision strategies, and pluggable (`register_detector`). Built in, needing only ffmpeg and numpy: `ffmpeg_scene@1` (cuts over a scene-score threshold; cuts closer than `min_shot_s` compete, the stronger stays), `spectral_flux@1` (onsets: peaks of the spectral flux, computed block by block so memory stays flat; RMS loudness per window) and `frame_difference@1` (mean absolute difference of small gray frames). No speech model or audio tagger is built in: a plan asking for one names a registered detector.
  - **Frames:** a timestamp maps to the copy's frame on screen then (read a quarter frame early, so the decoder lands on it exactly); two timestamps on one frame are read once. **Pixel filters run after the budget** (`framekit.vision.pixels`): anchors are never dropped, and version 1 doesn't refill the budget; `dedupe@1` compares with the last frame kept (pHash, Hamming distance), `sharpness@1` keeps the sharpest frame per window (Laplacian variance). Delivery as images or storyboard grids (empty tiles black); JPEG quality 1–100 maps linearly onto ffmpeg's MJPEG scale.
  - **numpy is the `media` extra**, enforced: only media work and the pixel filters import it, so the schemas and the dataset reader run without it. CI installs ffmpeg from Ubuntu's archive and fails, never skips, the tests that need it (`FRAMEKIT_REQUIRE_FFMPEG=1`).
- **Why:** deterministic signals and exact frame selection are what perception is built on (DESIGN.md §2.3, §3 L1); keeping ffmpeg as a binary and numpy optional keeps the dependency footprint small.
- **Recorded in:** `src/framekit/extraction/media/` (new: `ffmpeg.py`, `detectors.py`, `frames.py`, `__init__.py`), `src/framekit/vision/pixels.py` (new), `src/framekit/vision/__init__.py`; `tests/test_media.py`, `tests/media_helpers.py` (new); `pyproject.toml` (the media extra; numpy only in media work and the pixel filters); `.github/workflows/tests.yml` (ffmpeg); settings.md §5.2; DESIGN.md §2.3, §3 (L1), §7, §9; ARCHITECTURE.md §1, §2, §2.1, §3; README.md module table, "Vision parameters", "Development"; CLAUDE.md "Workflow"; SECURITY.md "Scope".

### 15:05 — Extraction v1, part 6: what models are asked, and how replies are checked
- **Decision** (the sixth PR of the extraction series; v1, revisitable):
  - **framekit's own prompts** (`prompts.py`): L2 segmentation, L3a perception, and the frame around each lens's prompts in L3b. Provider-neutral text that never names a lens or corpus concept (names and hints come from the corpus profile, lens prompts from the lens spec); a change to any text is a new prompt version (`segment@1`, `perceive@1`, `lens_pass@1`), which will be part of the input hash of the stage that sends it. Images are announced with their times (storyboards tile by tile).
  - **The perception record, schema 1:** the core record (`summary`, `setting`, `entities`, `props`, `beats` with `t: [start, end]`, `who` as a list, `action`, `audio`) plus the `camera`, `timing` and `sound` modules written by the model; `speech` comes from L1, never from the model. Checks: every time inside the scene (±0.5 s) and ordered, names from the corpus profile (or marked unknown), every module present.
  - **Replies:** the JSON object is the whole reply, or a code fence's content, or the first object in surrounding prose, found with plain string operations (no backtracking regex).
  - **Scene repair** (`segmenting.py`): a model's proposed scenes become a clean cover of each processable span (excluded ranges cut out): ordered and contiguous, boundaries snapped to the nearest cut within the tolerance, scenes under the minimum merged into a neighbour when the result stays within the maximum, scenes over the maximum split near even points (at a cut when one is close), every repair reported as a note. Shot groups build scenes without a model; fixed windows are arithmetic.
- **Why:** perception records only what's observable, and every value a model returns is checked by code before anything stores it (DESIGN.md §0.3, §3).
- **Recorded in:** `src/framekit/extraction/prompts.py`, `records.py`, `segmenting.py` (new); `tests/test_model_io.py`, `tests/test_segmenting.py` (new); `pyproject.toml` (prompt text may exceed the line length); DESIGN.md §3 (L3a).

---

## Still open
See DESIGN.md §9: video, text and embedding models; default vision profiles; baseline scorer details and gold queries; dataset layout; a distribution name if ever published to PyPI; distributed runner; per-lens taxonomies; frames to remote models for bring-your-own corpora; a drop policy (settings.md §8); pixel-filter results feeding back into the frame budget, planning for `split_by_budget` and `per_module` views, and keeping `max_frames` across the spans of a unit (2026-10-24).
