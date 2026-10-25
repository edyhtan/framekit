"""The extraction schemas: A (`SearchSpace`), V (`Corpus`), the plan (`ExtractionPlan`, made
of five components: `Observation`, `Structure`, `Interpretation`, `Models`, `Policy`), lens
specs (`LensSpec`), the gold set (`GoldSet`) and the dataset manifest (`DatasetManifest`).

Pydantic models, strict and read-only once parsed. Each loads from a YAML or JSON file
(`.load(path)`) or plain data (`.from_mapping(data)`) and raises one `ConfigError` listing
every problem with its path. They live here, above `framekit.frames` and `framekit.models`,
because `framekit.contracts` stays free of Pydantic and the plan embeds `FrameParams`.
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
from framekit.extraction.schemas.manifest import SCHEMA_VERSION, DatasetManifest
from framekit.extraction.schemas.plan import (
    COMPONENTS,
    DEFAULT_FRAMES,
    MODEL_ROLES,
    TABLES,
    Component,
    ExtractionPlan,
    Interpretation,
    Models,
    Observation,
    Policy,
    RoleModel,
    Structure,
)
from framekit.extraction.schemas.space import LensRef, SearchSpace

__all__ = [
    "COMPONENTS",
    "DEFAULT_FRAMES",
    "MODEL_ROLES",
    "MODULES",
    "SCHEMA_VERSION",
    "TABLES",
    "AlignmentSegment",
    "Component",
    "ConfigError",
    "Corpus",
    "CorpusProfile",
    "DatasetManifest",
    "ExtractionPlan",
    "FieldSpec",
    "GoldMatch",
    "GoldMoment",
    "GoldQuery",
    "GoldSet",
    "Interpretation",
    "LensPrompts",
    "LensRef",
    "LensSpec",
    "LensValidator",
    "ManifestEntry",
    "Models",
    "Observation",
    "Policy",
    "RoleModel",
    "SearchSpace",
    "Structure",
    "UnitRef",
    "canonical_json",
    "digest",
]
