"""The extraction schemas: A (`SearchSpace`), V (`Corpus`), the plan (`ExtractionPlan`), lens
specs (`LensSpec`) and the gold set (`GoldSet`).

Pydantic models, strict and read-only once parsed. Each loads from a YAML or JSON file
(`.load(path)`) or plain data (`.from_mapping(data)`) and raises one `ConfigError` listing
every problem with its path. They live here, above `framekit.vision` and `framekit.models`,
because `framekit.contracts` stays free of Pydantic and the plan embeds `VisionParams`.
"""

from framekit.extraction.schemas.common import MODULES, ConfigError, canonical_json, digest
from framekit.extraction.schemas.corpus import (
    AlignmentSegment,
    Corpus,
    CorpusProfile,
    ManifestEntry,
    UnitRef,
)
from framekit.extraction.schemas.gold import GoldMatch, GoldMoment, GoldQuery, GoldSet
from framekit.extraction.schemas.lens import FieldSpec, LensPrompts, LensSpec, LensValidator
from framekit.extraction.schemas.plan import (
    DEFAULT_VISION,
    MODEL_ROLES,
    TABLES,
    ExtractionPlan,
    RoleModel,
)
from framekit.extraction.schemas.space import LensRef, SearchSpace

__all__ = [
    "DEFAULT_VISION",
    "MODEL_ROLES",
    "MODULES",
    "TABLES",
    "AlignmentSegment",
    "ConfigError",
    "Corpus",
    "CorpusProfile",
    "ExtractionPlan",
    "FieldSpec",
    "GoldMatch",
    "GoldMoment",
    "GoldQuery",
    "GoldSet",
    "LensPrompts",
    "LensRef",
    "LensSpec",
    "LensValidator",
    "ManifestEntry",
    "RoleModel",
    "SearchSpace",
    "UnitRef",
    "canonical_json",
    "digest",
]
