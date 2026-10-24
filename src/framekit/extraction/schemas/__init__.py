"""The extraction schemas, so far lens specs (`LensSpec`).

Pydantic models, strict and read-only once parsed. Each loads from a YAML or JSON file
(`.load(path)`) or plain data (`.from_mapping(data)`) and raises one `ConfigError` listing
every problem with its path. They live here, above `framekit.vision` and `framekit.models`,
because `framekit.contracts` stays free of Pydantic and the plan will embed `VisionParams`.
"""

from framekit.extraction.schemas.common import MODULES, ConfigError, canonical_json, digest
from framekit.extraction.schemas.lens import FieldSpec, LensPrompts, LensSpec, LensValidator

__all__ = [
    "MODULES",
    "ConfigError",
    "FieldSpec",
    "LensPrompts",
    "LensSpec",
    "LensValidator",
    "canonical_json",
    "digest",
]
