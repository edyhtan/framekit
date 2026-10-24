"""The dataset (DESIGN.md §2.1; docs/settings.md §7): Parquet tables plus `manifest.json`.

Layout (schema version 1):

    <dataset>/
      manifest.json                    written last: no manifest, no finished dataset
      .gitignore                       `*`: a dataset from a byo_media corpus stays local
      tables/<table>/unit-<id>.parquet one file per unit (layout.partition_by: unit_id)
      tables/<table>.parquet           or one file per table (partition_by: none)

Every table has a fixed schema per schema version (`SCHEMAS`). Free-form values (unit
metadata, perception records, annotation values) are JSON text; vectors are lists of
float32. Readers need only pyarrow (or any Parquet reader): `Dataset.open(path)`.
"""

import hashlib
import json
import shutil
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from framekit.extraction.runner import atomic_write, ensure_private_dir
from framekit.extraction.schemas.manifest import (
    FILE_NAME,
    SCHEMA_VERSION,
    DatasetManifest,
    TableFile,
    TableInfo,
)
from framekit.extraction.schemas.plan import TABLES

_S, _F, _I, _B = pa.string(), pa.float64(), pa.int32(), pa.bool_()

SCHEMAS: dict[str, pa.Schema] = {
    "units": pa.schema(
        [
            ("unit_id", _S),
            ("status", _S),
            ("reason", _S),
            ("file_name", _S),
            ("source_fingerprint", _S),
            ("duration_s", _F),
            ("width", _I),
            ("height", _I),
            ("fps", _F),
            ("has_audio", _B),
            ("copy_width", _I),
            ("copy_height", _I),
            ("copy_fps", _F),
            ("metadata", _S),
            ("excluded_ranges", pa.list_(pa.list_(_F))),
            ("synopsis", _S),
        ]
    ),
    "shots": pa.schema(
        [("shot_id", _S), ("unit_id", _S), ("index", _I), ("start", _F), ("end", _F)]
    ),
    "scenes": pa.schema(
        [
            ("scene_id", _S),
            ("unit_id", _S),
            ("index", _I),
            ("start", _F),
            ("end", _F),
            ("summary", _S),
            ("method", _S),
            ("status", _S),
        ]
    ),
    "beats": pa.schema(
        [
            ("beat_id", _S),
            ("scene_id", _S),
            ("unit_id", _S),
            ("index", _I),
            ("start", _F),
            ("end", _F),
            ("who", pa.list_(_S)),
            ("action", _S),
            ("audio", _S),
        ]
    ),
    "windows": pa.schema(
        [("window_id", _S), ("unit_id", _S), ("index", _I), ("start", _F), ("end", _F)]
    ),
    "signals": pa.schema(
        [
            ("unit_id", _S),
            ("signal", _S),
            ("t", _F),
            ("end", _F),
            ("value", _F),
            ("label", _S),
            ("text", _S),
            ("detector", _S),
        ]
    ),
    "perception": pa.schema(
        [
            ("scene_id", _S),
            ("unit_id", _S),
            ("record_schema", _I),
            ("modules", pa.list_(_S)),
            ("summary", _S),
            ("record", _S),
            ("status", _S),
            ("run_id", _S),
            ("model_id", _S),
            ("prompt_version", _S),
        ]
    ),
    "frames_sent": pa.schema(
        [
            ("unit_id", _S),
            ("scene_id", _S),
            ("stage", _S),
            ("image_index", _I),
            ("tile_index", _I),
            ("planned_s", _F),
            ("frame_s", _F),
            ("frame_index", _I),
            ("anchor", _B),
            ("width", _I),
            ("height", _I),
            ("image_width", _I),
            ("image_height", _I),
            ("format", _S),
            ("image_sha256", _S),
            ("profile", _S),
            ("profile_digest", _S),
        ]
    ),
    "annotations": pa.schema(
        [
            ("target_type", _S),
            ("target_id", _S),
            ("scene_id", _S),
            ("unit_id", _S),
            ("lens", _S),
            ("lens_version", _S),
            ("run_index", _I),
            ("field", _S),
            ("label", _S),
            ("value", _S),
            ("weight", _F),
            ("run_id", _S),
        ]
    ),
    "phrasings": pa.schema(
        [
            ("phrasing_id", _S),
            ("target_type", _S),
            ("target_id", _S),
            ("scene_id", _S),
            ("unit_id", _S),
            ("lens", _S),
            ("lens_version", _S),
            ("run_index", _I),
            ("text", _S),
            ("source", _S),
            ("run_id", _S),
        ]
    ),
    "vectors": pa.schema(
        [
            ("target_type", _S),
            ("target_id", _S),
            ("unit_id", _S),
            ("embedding_model", _S),
            ("dimension", _I),
            ("vector", pa.list_(pa.float32())),
            ("text_sha256", _S),
            ("run_id", _S),
        ]
    ),
    "runs": pa.schema(
        [
            ("run_id", _S),
            ("stage", _S),
            ("unit_id", _S),
            ("target_id", _S),
            ("input_hash", _S),
            ("status", _S),
            ("attempts", _I),
            ("model_id", _S),
            ("provider", _S),
            ("adapter_version", _S),
            ("started_at", _S),
            ("finished_at", _S),
            ("latency_ms", pa.int64()),
            ("input_tokens", pa.int64()),
            ("output_tokens", pa.int64()),
            ("reasoning_tokens", pa.int64()),
            ("cost_usd", _S),
            ("cached", _B),
            ("reused", _B),
            ("lens", _S),
            ("run_index", _I),
            ("error", _S),
        ]
    ),
    "qc": pa.schema(
        [
            ("check", _S),
            ("target_type", _S),
            ("target_id", _S),
            ("unit_id", _S),
            ("scene_id", _S),
            ("lens", _S),
            ("run_index", _I),
            ("status", _S),
            ("detail", _S),
        ]
    ),
}
JSON_COLUMNS = {"units": ("metadata",), "perception": ("record",), "annotations": ("value",)}
if set(SCHEMAS) != set(TABLES):  # one schema per table the plan can name
    raise RuntimeError("dataset schemas and plan tables differ")


def _json(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def to_table(name: str, rows: Iterable[Mapping[str, Any]]) -> pa.Table:
    """Rows (dicts) as an Arrow table with the table's fixed schema: missing columns are
    null, unknown keys fail, free-form columns become JSON text."""
    schema = SCHEMAS[name]
    names = set(schema.names)
    columns: dict[str, list[Any]] = {field: [] for field in schema.names}
    json_columns = JSON_COLUMNS.get(name, ())
    for row in rows:
        unknown = set(row) - names
        if unknown:
            raise ValueError(f"{name}: unknown columns {sorted(unknown)}")
        for field in schema.names:
            value = row.get(field)
            columns[field].append(_json(value) if field in json_columns else value)
    return pa.table(columns, schema=schema)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_tables(
    out: Path,
    tables: Mapping[str, list[Mapping[str, Any]]],
    *,
    include: Iterable[str],
    partition_by: str,
    compression: str,
) -> dict[str, TableInfo]:
    """Write the tables (replacing any from an earlier write of the same dataset) and
    return what was written, with checksums."""
    root = out / "tables"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    codec = None if compression == "none" else compression
    written: dict[str, TableInfo] = {}
    for name in include:
        rows = list(tables.get(name, ()))
        groups: dict[str, list[Mapping[str, Any]]] = {}
        if partition_by == "unit_id":
            for row in rows:
                groups.setdefault(str(row.get("unit_id") or "_none"), []).append(row)
        else:
            groups = {"": rows}
        files = []
        for key, group in sorted(groups.items()):
            if partition_by == "unit_id":
                path = root / name / f"unit-{key}.parquet"
                path.parent.mkdir(parents=True, exist_ok=True)
            else:
                path = root / f"{name}.parquet"
            pq.write_table(to_table(name, group), path, compression=codec)
            files.append(
                TableFile(
                    path=path.relative_to(out).as_posix(), rows=len(group), sha256=_sha256(path)
                )
            )
        written[name] = TableInfo(rows=len(rows), files=tuple(files))
    return written


def write_manifest(out: Path, manifest: DatasetManifest) -> None:
    atomic_write(out / FILE_NAME, manifest.model_dump_json(indent=1))


class Dataset:
    """A dataset folder, read: the manifest and one Arrow table per name."""

    def __init__(self, path: Path, manifest: DatasetManifest) -> None:
        self.path = path
        self.manifest = manifest

    @classmethod
    def open(cls, path: str | Path, *, verify: bool = False) -> "Dataset":
        """Read `manifest.json` (a folder without one is an unfinished dataset). Refuses a
        schema version it doesn't know; `verify` also checks every file's sha256."""
        folder = Path(path).expanduser()
        file = folder / FILE_NAME
        if not file.is_file():
            raise FileNotFoundError(f"{folder}: no {FILE_NAME} (not a finished dataset)")
        manifest = DatasetManifest.model_validate_json(file.read_text(encoding="utf-8"))
        if manifest.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"{folder}: dataset schema version {manifest.schema_version}; this framekit "
                f"reads version {SCHEMA_VERSION}"
            )
        dataset = cls(folder, manifest)
        if verify:
            problems = dataset.verify()
            if problems:
                raise ValueError(f"{folder}: " + "; ".join(problems))
        return dataset

    def verify(self) -> list[str]:
        """Files whose content no longer matches the manifest's checksums."""
        problems = []
        for name, info in self.manifest.tables.items():
            for item in info.files:
                path = self.path / item.path
                if not path.is_file():
                    problems.append(f"{name}: {item.path} is missing")
                elif _sha256(path) != item.sha256:
                    problems.append(f"{name}: {item.path} changed since it was written")
        return problems

    @property
    def table_names(self) -> tuple[str, ...]:
        return tuple(self.manifest.tables)

    def table(self, name: str) -> pa.Table:
        """One table, every file of it concatenated (an empty table with the schema when
        it has no rows)."""
        if name not in SCHEMAS:
            raise KeyError(f"no table {name!r} (tables: {', '.join(SCHEMAS)})")
        info = self.manifest.tables.get(name)
        if info is None:
            raise KeyError(f"this dataset wasn't written with the {name!r} table")
        parts = [pq.read_table(self.path / item.path, schema=SCHEMAS[name]) for item in info.files]
        if not parts:
            return SCHEMAS[name].empty_table()
        return pa.concat_tables(parts)

    def rows(self, name: str) -> list[dict[str, Any]]:
        """A table as dicts, with JSON columns decoded."""
        rows = self.table(name).to_pylist()
        for column in JSON_COLUMNS.get(name, ()):
            for row in rows:
                if row[column] is not None:
                    row[column] = json.loads(row[column])
        return rows

    def __getattr__(self, name: str) -> pa.Table:
        if name in SCHEMAS:
            return self.table(name)
        raise AttributeError(name)

    def __repr__(self) -> str:
        counts = ", ".join(f"{n}={i.rows}" for n, i in self.manifest.tables.items())
        return f"Dataset({self.path}, {counts})"


def prepare_output(out: Path, identity: Mapping[str, str]) -> None:
    """Make the output folder, refusing one that holds a different dataset: datasets are
    never overwritten, only rewritten by a rerun of the same inputs."""
    file = out / FILE_NAME
    if file.is_file():
        try:
            existing = json.loads(file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = {}
        found = {
            "search_space": (existing.get("search_space") or {}).get("digest"),
            "corpus": (existing.get("corpus") or {}).get("digest"),
            "plan": (existing.get("plan") or {}).get("digest"),
        }
        if found != dict(identity):
            raise FileExistsError(
                f"{out} holds another dataset (different search space, corpus or plan); "
                "write to a new folder: datasets are never overwritten"
            )
    elif out.exists() and any(p.name not in (".gitignore", "tables") for p in out.iterdir()):
        raise FileExistsError(f"{out} isn't empty and holds no dataset; use a new folder")
    ensure_private_dir(out)


__all__ = ["SCHEMAS", "Dataset", "prepare_output", "to_table", "write_manifest", "write_tables"]
