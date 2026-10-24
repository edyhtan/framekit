"""V, the corpus (DESIGN.md §1, §8.2; docs/settings.md §4): which videos, pointed to and
never copied, with the corpus profile that is everything framekit knows about them.

    corpus:
      profile:
        name: demo_corpus
        unit_definition: "one complete film per file"
        metadata_fields:                  # per-unit fields: str | int | float | bool, or
          title: str                      # a list of allowed values
          year: int
          source_quality: [good, fair, poor]
        entities: [Ada, Brook]            # names perception should use consistently
        perception_hints: ["almost no dialogue"]
        rights_mode: byo_media            # byo_media | hosted | embed
      subset: [u001, u002]                # optional: the units this corpus run covers
      manifest:                           # optional: what each unit should be
        - unit_id: u001
          expected_duration_s: 412.3
          excluded_ranges: [[0.0, 9.5]]   # title cards, logos: never processed
      units:
        - unit_id: u001
          path: D:/media/u001.mkv         # relative paths: from this file's folder
          title: First film               # declared metadata fields go here directly
          year: 1951

A unit is the reference and is never split as one: every time in the dataset is on its
timeline. The source file is never modified.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, model_validator

from framekit.extraction.schemas.common import (
    ConfigError,
    Frozen,
    FrozenDict,
    Model,
    Name,
    Tupled,
    UnitId,
    digest,
    parse,
    plain,
    read_mapping,
    unwrap,
)

WHAT = "the corpus"
RightsMode = Literal["byo_media", "hosted", "embed"]
MetadataType = Literal["str", "int", "float", "bool"]
Range = tuple[float, float]


class CorpusProfile(Model):
    name: Name
    unit_definition: str = ""
    # Per-unit fields: a type, or the list of allowed values.
    metadata_fields: Annotated[
        dict[Name, MetadataType | Annotated[tuple[str, ...], Tupled]], Frozen
    ] = FrozenDict()
    entities: Annotated[tuple[str, ...], Tupled] = ()
    perception_hints: Annotated[tuple[str, ...], Tupled] = ()
    rights_mode: RightsMode


class ManifestEntry(Model):
    unit_id: UnitId
    expected_duration_s: float | None = Field(default=None, gt=0)
    fingerprint: str | None = None  # recorded; matching by fingerprint isn't built yet
    excluded_ranges: Annotated[tuple[Range, ...], Tupled] = ()


class AlignmentSegment(Model):
    """`t_user = t_ref * drift + offset_s` over the reference range `ref`."""

    ref: Annotated[Range, Tupled]
    offset_s: float = 0.0
    drift: float = Field(default=1.0, gt=0)

    def is_identity(self) -> bool:
        return self.offset_s == 0 and self.drift == 1


class UnitRef(Model):
    unit_id: UnitId
    path: str = Field(min_length=1)
    metadata: Annotated[dict[str, Any], Frozen] = FrozenDict()
    alignment_override: Annotated[tuple[AlignmentSegment, ...], Tupled] = ()

    @model_validator(mode="before")
    @classmethod
    def _metadata_inline(cls, data: Any) -> Any:
        """Keys that aren't unit fields are metadata (the corpus checks them against the
        profile), so a unit can say `title: ...` directly."""
        if not isinstance(data, Mapping):
            return data
        own = set(cls.model_fields)
        extra = {k: v for k, v in data.items() if k not in own}
        metadata = data.get("metadata", {})
        if not extra or not isinstance(metadata, Mapping) or set(extra) & set(metadata):
            return data  # nothing to move, or left for the field checks to report
        return {**{k: v for k, v in data.items() if k in own}, "metadata": {**metadata, **extra}}


class Corpus(Model):
    profile: CorpusProfile
    subset: Annotated[tuple[UnitId, ...], Tupled] | None = None
    manifest: Annotated[tuple[ManifestEntry, ...], Tupled] = ()
    units: Annotated[tuple[UnitRef, ...], Tupled]

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], base_dir: str | Path | None = None) -> "Corpus":
        """A corpus from plain data. Relative unit paths are read from `base_dir` (default:
        the working directory) and stored as absolute paths."""
        raw = plain(unwrap(data, "corpus"))
        corpus = parse(cls, raw, WHAT, extra=lambda c: c.problems())
        base = Path(base_dir or ".").expanduser().resolve()
        units = tuple(
            unit.model_copy(update={"path": str((base / Path(unit.path).expanduser()).resolve())})
            for unit in corpus.units
        )
        return corpus.model_copy(update={"units": units})

    @classmethod
    def load(cls, path: str | Path) -> "Corpus":
        file = Path(path).expanduser()
        return cls.from_mapping(read_mapping(file, WHAT), base_dir=file.parent)

    def problems(self) -> list[str]:
        """Checks across fields: unique ids, the subset and manifest refer to known units,
        metadata matches the profile, ranges are ordered and non-negative."""
        problems = []
        ids = [u.unit_id for u in self.units]
        if not ids:
            problems.append("units: at least one unit is needed")
        problems += [f"units: {u!r} is listed twice" for u in sorted(set(ids)) if ids.count(u) > 1]
        known = set(ids)
        for u in self.subset or ():
            if u not in known:
                problems.append(f"subset: {u!r} is not one of the units")
        manifest_ids = [m.unit_id for m in self.manifest]
        problems += [
            f"manifest: {u!r} is listed twice"
            for u in sorted(set(manifest_ids))
            if manifest_ids.count(u) > 1
        ]
        for i, entry in enumerate(self.manifest):
            for j, (start, end) in enumerate(entry.excluded_ranges):
                if not 0 <= start < end:
                    problems.append(
                        f"manifest[{i}].excluded_ranges[{j}]: expected [start, end] with "
                        "0 <= start < end"
                    )
        fields = self.profile.metadata_fields
        for i, unit in enumerate(self.units):
            for key, value in unit.metadata.items():
                if key not in fields:
                    problems.append(
                        f"units[{i}].{key}: not a unit field or a metadata field declared in "
                        "profile.metadata_fields"
                    )
                else:
                    problem = _metadata_problem(fields[key], value)
                    if problem:
                        problems.append(f"units[{i}].{key}: {problem}")
            for j, segment in enumerate(unit.alignment_override):
                start, end = segment.ref
                if not 0 <= start < end:
                    problems.append(
                        f"units[{i}].alignment_override[{j}].ref: expected [start, end] with "
                        "0 <= start < end"
                    )
        return problems

    def manifest_for(self, unit_id: str) -> ManifestEntry | None:
        return next((m for m in self.manifest if m.unit_id == unit_id), None)

    @property
    def digest(self) -> str:
        """sha256 of the corpus as framekit sees it: profile, subset, manifest, and each
        unit's id, metadata and alignment. File paths are left out (they differ between
        machines); each unit's content fingerprint is recorded per unit instead."""
        units = [
            {
                "unit_id": u.unit_id,
                "metadata": plain(u.metadata),
                "alignment_override": plain(u.alignment_override),
            }
            for u in self.units
        ]
        return digest(
            {
                "profile": plain(self.profile),
                "subset": plain(self.subset),
                "manifest": plain(self.manifest),
                "units": units,
            }
        )


def _metadata_problem(kind: Any, value: Any) -> str | None:
    if isinstance(kind, tuple):
        return None if value in kind else f"{value!r} is not one of {', '.join(kind)}"
    ok = {
        "str": isinstance(value, str),
        "int": isinstance(value, int) and not isinstance(value, bool),
        "float": isinstance(value, int | float) and not isinstance(value, bool),
        "bool": isinstance(value, bool),
    }[kind]
    return None if ok else f"expected {kind}, got {value!r}"


def check_corpus(corpus: Corpus) -> None:
    """Raise ConfigError if a corpus built in code has cross-field problems."""
    problems = corpus.problems()
    if problems:
        raise ConfigError(problems, WHAT)


__all__ = [
    "AlignmentSegment",
    "Corpus",
    "CorpusProfile",
    "ManifestEntry",
    "UnitRef",
    "check_corpus",
]
