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

---

## Still open
See DESIGN.md §9: video, text and embedding models; A / V schema details and the lens spec file format; dataset layout; package and CLI names; distributed runner; per-lens taxonomies.
