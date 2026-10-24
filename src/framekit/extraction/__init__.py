"""Extraction: a search space (A) and a corpus (V), with an optional plan, become a dataset
of segments, text, vectors and provenance (docs/DESIGN.md §2–§6).

Built so far: the inputs of a run, as checked schemas: lens specs (`LensSpec`), the search
space (`SearchSpace`), the corpus (`Corpus`), the plan (`ExtractionPlan`) and the benchmark
gold set (`GoldSet`). Resolution, media work, the stages and the dataset follow (milestone
E1).

    space = SearchSpace.load("search-space.yaml")   # ConfigError lists every problem
    corpus = Corpus.load("corpus.yaml")
    plan = ExtractionPlan.load("plan.yaml")         # or ExtractionPlan(): the defaults
"""

from framekit.extraction.schemas import (
    MODULES,
    ConfigError,
    Corpus,
    CorpusProfile,
    ExtractionPlan,
    FieldSpec,
    GoldSet,
    LensSpec,
    SearchSpace,
)

__all__ = [
    "MODULES",
    "ConfigError",
    "Corpus",
    "CorpusProfile",
    "ExtractionPlan",
    "FieldSpec",
    "GoldSet",
    "LensSpec",
    "SearchSpace",
]
