"""Extraction: a search space (A) and a corpus (V), with an optional plan, become a dataset
of segments, text, vectors and provenance (docs/DESIGN.md §2–§6).

    from framekit.extraction import Corpus, ExtractionPlan, LocalRunner, SearchSpace, extract

    dataset = extract(
        SearchSpace.load("search-space.yaml"),
        Corpus.load("corpus.yaml"),
        plan=ExtractionPlan.load("plan.yaml"),
        out="~/framekit-data/datasets/first",
        runner=LocalRunner(concurrency=8),
    )
    dataset.scenes, dataset.rows("perception")

Built so far: runs through L0–L3a (the lens-free part: the analysis copy, signals, scenes,
observable facts). Lens passes and vectors, the `framekit` command and the benchmark
harness follow in milestone E1.

The schemas, resolution and the dataset reader need only the base install; running
extraction needs the media extra (`pip install "framekit[media]"`, for numpy) and an
ffmpeg binary. Every setting: docs/settings.md.
"""

from typing import Any

from framekit.extraction.dataset import SCHEMAS, Dataset
from framekit.extraction.resolve import ResolvedRun, RoleBinding, resolve_run
from framekit.extraction.runner import BudgetExceeded, LocalRunner
from framekit.extraction.schemas import (
    MODULES,
    ConfigError,
    Corpus,
    CorpusProfile,
    DatasetManifest,
    ExtractionPlan,
    FieldSpec,
    GoldSet,
    LensSpec,
    SearchSpace,
)

# Running extraction needs numpy (the media extra): these load on first use, so the
# schemas and the dataset reader work without it.
_LAZY = {
    "extract": "framekit.extraction.pipeline",
    "extract_async": "framekit.extraction.pipeline",
    "CostEstimate": "framekit.extraction.estimate",
    "Detector": "framekit.extraction.media",
    "MediaContext": "framekit.extraction.media",
    "register_detector": "framekit.extraction.media",
    "detectors": "framekit.extraction.media",
}


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module 'framekit.extraction' has no attribute {name!r}")
    import importlib

    try:
        loaded = importlib.import_module(module)
    except ImportError as exc:
        raise ImportError(
            f"framekit.extraction.{name} needs the media extra: pip install "
            f'"framekit[media]" ({exc.name} is missing)'
        ) from exc
    return getattr(loaded, name)


__all__ = [
    "MODULES",
    "SCHEMAS",
    "BudgetExceeded",
    "ConfigError",
    "Corpus",
    "CorpusProfile",
    "CostEstimate",
    "Dataset",
    "DatasetManifest",
    "Detector",
    "ExtractionPlan",
    "FieldSpec",
    "GoldSet",
    "LensSpec",
    "LocalRunner",
    "MediaContext",
    "ResolvedRun",
    "RoleBinding",
    "SearchSpace",
    "detectors",
    "extract",
    "extract_async",
    "register_detector",
    "resolve_run",
]
