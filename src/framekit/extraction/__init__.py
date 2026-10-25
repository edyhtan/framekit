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
    dataset.scenes, dataset.rows("annotations")

The schemas, resolution and the dataset reader need only the base install; running
extraction and the benchmark harness needs the media extra (`pip install
"framekit[media]"`, for numpy) and an ffmpeg binary. Every setting: docs/settings.md.
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
    Interpretation,
    LensSpec,
    Models,
    Observation,
    Policy,
    SearchSpace,
    Structure,
)

# Running extraction needs numpy (the media extra): these load on first use, so the
# schemas and the dataset reader work without it.
_LAZY = {
    "extract": "framekit.extraction.pipeline",
    "extract_async": "framekit.extraction.pipeline",
    "CostEstimate": "framekit.extraction.estimate",
    "benchmark": "framekit.extraction.bench",
    "benchmark_async": "framekit.extraction.bench",
    "baseline_retriever": "framekit.extraction.bench",
    "BaselineRetriever": "framekit.extraction.bench",
    "Report": "framekit.extraction.bench",
    "Scorer": "framekit.extraction.bench",
    "ScoringContext": "framekit.extraction.bench",
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
    "BaselineRetriever",
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
    "Interpretation",
    "LensSpec",
    "LocalRunner",
    "MediaContext",
    "Models",
    "Observation",
    "Policy",
    "Report",
    "ResolvedRun",
    "RoleBinding",
    "Scorer",
    "ScoringContext",
    "SearchSpace",
    "Structure",
    "baseline_retriever",
    "benchmark",
    "benchmark_async",
    "detectors",
    "extract",
    "extract_async",
    "register_detector",
    "resolve_run",
]
