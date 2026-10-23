# framekit

A general-purpose Python library of AI building blocks for video datasets. One package, with modules that can be used on their own:

| Module | What it gives you |
|---|---|
| `framekit.contracts` | provider-neutral ports, ids, run records, and the schemas shared by the other modules |
| `framekit.models` | model access: a catalog, a router, adapters for several LLM providers, and a record / replay response cache |
| `framekit.vision` | how a model sees video: frame selection, image preparation and delivery, driven by structured `vision_params` |
| `framekit.extraction` | turns a video corpus into a dataset (below), plus a benchmark harness and the CLI |

New capabilities are added as new modules in this package, not as new packages, until there's a concrete reason to split (docs/ARCHITECTURE.md §2).

### Extraction

The extraction module turns a video corpus into a dataset:

```python
dataset = extract(A, V, plan=plan, out=path)
```

- **A**, the search space: what should become findable in the corpus. Given input: intent, lens specs (vocabulary, schema, prompts, rubric) as data, target moment length, scope, example queries. framekit doesn't create A, so different ways of producing it can be benchmarked on the same corpus.
- **plan**: how A is realized (vision profiles, segmentation, models, quality rules, budget, formatting). Optional; defaults apply.
- **V**, the corpus: a corpus profile plus references to the video files (never copies).
- **Dataset:** segments with timestamps, text (perception records, annotations, phrasings), vectors per embedding model, and provenance, as Parquet / Arrow tables plus a manifest.

Used by the video clip search agent ([short_extract](https://github.com/edyhtan/short_extract)), and for benchmarks and experimental model training.

## Status

Design only. The first code (contracts and model access) is seeded from short_extract; extraction stages follow. See the roadmap in [docs/DESIGN.md](docs/DESIGN.md) §7.

## Documents

| Document | What it covers |
|---|---|
| [docs/DESIGN.md](docs/DESIGN.md) | what extraction does: inputs, the dataset, stages L0–L4, lens specs, runs, evaluation, roadmap, rights |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | how the code is layered, the public API, model access, how consumers connect, the seeding plan |
| [docs/DECISIONS.md](docs/DECISIONS.md) | why, in date order |
| [CLAUDE.md](CLAUDE.md) | rules for contributors and coding agents |

## Rights

Not affiliated with any rights holder. For corpora in `byo_media` mode, users supply their own copies; this repository contains no footage, frames, thumbnails, ripping tools or downloaders, and datasets built from such corpora stay local.
