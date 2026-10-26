"""The dataset manifest (DESIGN.md §2.1; docs/settings.md §7): what a dataset is and how it
was made, in `manifest.json` next to the tables.

It's written last, after every table, so a folder without one is an unfinished dataset.
Readers refuse a schema version they don't know. Unknown keys are kept (a newer framekit
may add some within the same schema version).
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = 2  # 2: plan v2 (grouped), frames_digest (was vision_digest)
FILE_NAME = "manifest.json"


class _Open(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)


class TableFile(_Open):
    path: str  # relative to the dataset folder, with forward slashes
    rows: int
    sha256: str


class TableInfo(_Open):
    rows: int
    files: tuple[TableFile, ...] = ()


class UnitStatus(_Open):
    unit_id: str
    status: str  # complete | partial | failed | excluded
    reason: str = ""


class DatasetManifest(_Open):
    schema_version: int
    dataset_id: str
    created_at: str  # ISO 8601, UTC
    framekit_version: str
    label: str | None = None  # the benchmark variant label, when made by the harness
    complete: bool  # every unit in scope finished
    search_space: dict[str, Any]  # {verbatim, digest}
    corpus: dict[str, Any]  # {name, digest, rights_mode, units: {id: source fingerprint}}
    plan: dict[str, Any]  # {resolved, digest, content_digest, fairness_digest, components}
    derivations: tuple[dict[str, Any], ...] = ()
    lenses: tuple[dict[str, Any], ...] = ()  # {name, version, digest}
    models: dict[str, Any] = Field(default_factory=dict)  # role -> {model_id, provider, ...}
    frames_digest: str = ""
    governance: dict[str, Any] = Field(default_factory=dict)
    units: tuple[UnitStatus, ...] = ()
    tables: dict[str, TableInfo] = Field(default_factory=dict)
    cost: dict[str, Any] = Field(default_factory=dict)  # estimated USD, per stage
    tools: dict[str, Any] = Field(default_factory=dict)  # e.g. the ffmpeg version
    notes: tuple[str, ...] = ()


__all__ = ["FILE_NAME", "SCHEMA_VERSION", "DatasetManifest", "TableFile", "TableInfo", "UnitStatus"]
