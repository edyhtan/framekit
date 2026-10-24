"""Extraction: a search space (A) and a corpus (V), with an optional plan, become a dataset
of segments, text, vectors and provenance (docs/DESIGN.md §2–§6).

Built so far: lens specs (`LensSpec`): the declarative format of what a lens pass writes,
its loader and checks, and the JSON Schema it compiles to. The search space, corpus,
plan, resolution, media work, stages and the dataset follow (milestone E1).
"""

from framekit.extraction.schemas import MODULES, ConfigError, FieldSpec, LensSpec

__all__ = ["MODULES", "ConfigError", "FieldSpec", "LensSpec"]
