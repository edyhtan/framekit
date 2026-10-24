"""A, the search space (DESIGN.md §2.4; docs/settings.md §3): what should become findable.

A is **given input**: written by hand or produced by an augmentation strategy outside
framekit, never edited here. The manifest records it verbatim with its digest, which
covers the lens specs it names (their prompts included).

    search_space:
      version: 1
      intent: Moments where a plan falls apart.
      lenses:
        - {spec: lenses/pace, version: 0.1.0}      # a folder or file, relative to this file
      evidence: {modules: [sound]}                 # optional
      resolution: {moment_length_s: {min: 3, max: 6}}
      scope:                                       # can only narrow the corpus
        units: {include: all, exclude: []}
        excluded_ranges: from_corpus               # or none
      query_style:
        examples: ["my team keeps arguing about the same thing"]
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, PrivateAttr

from framekit.extraction.schemas.common import (
    ConfigError,
    Model,
    Module,
    Tupled,
    UnitId,
    Version,
    digest,
    parse,
    plain,
    read_mapping,
    unwrap,
)
from framekit.extraction.schemas.lens import LensSpec

WHAT = "the search space"


class LensRef(Model):
    """A lens of the search space: a spec path (a folder or file, relative to the search
    space's file) or a loaded `LensSpec`, and the version it must have."""

    spec: str | LensSpec
    version: Version


class Evidence(Model):
    modules: Annotated[tuple[Module, ...], Tupled] = ()


class MomentLength(Model):
    min: float = Field(gt=0, le=3600)
    max: float = Field(gt=0, le=3600)


class Resolution(Model):
    moment_length_s: MomentLength


class UnitScope(Model):
    include: Literal["all"] | Annotated[tuple[UnitId, ...], Tupled] = "all"
    exclude: Annotated[tuple[UnitId, ...], Tupled] = ()


class Scope(Model):
    units: UnitScope = UnitScope()
    excluded_ranges: Literal["from_corpus", "none"] = "from_corpus"


class QueryStyle(Model):
    examples: Annotated[tuple[str, ...], Tupled] = ()


class SearchSpace(Model):
    version: Literal[1] = 1
    intent: str = ""
    lenses: Annotated[tuple[LensRef, ...], Tupled]
    evidence: Evidence = Evidence()
    resolution: Resolution
    scope: Scope = Scope()
    query_style: QueryStyle = QueryStyle()

    _verbatim: dict[str, Any] | None = PrivateAttr(default=None)

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, Any], base_dir: str | Path | None = None
    ) -> "SearchSpace":
        """A search space from plain data (`{search_space: {...}}` or the bare mapping).
        Lens spec paths are read relative to `base_dir` (default: the working directory)."""
        raw = plain(unwrap(data, "search_space"))
        space = parse(cls, raw, WHAT, extra=_space_problems)
        space._verbatim = raw
        return space.with_lenses_loaded(base_dir)

    @classmethod
    def load(cls, path: str | Path) -> "SearchSpace":
        file = Path(path).expanduser()
        return cls.from_mapping(read_mapping(file, WHAT), base_dir=file.parent)

    def with_lenses_loaded(self, base_dir: str | Path | None = None) -> "SearchSpace":
        """The same search space with every lens spec loaded (paths read relative to
        `base_dir`), each checked against the version the space names."""
        problems: list[str] = []
        refs = []
        base = Path(base_dir or ".")
        for i, ref in enumerate(self.lenses):
            spec = ref.spec
            if isinstance(spec, str):
                try:
                    spec = LensSpec.load(base / spec)
                except ConfigError as exc:
                    problems += [f"lenses[{i}].spec: {p}" for p in exc.problems]
                    continue
            if spec.version != ref.version:
                problems.append(
                    f"lenses[{i}].version: the space asks for {spec.name} {ref.version}, "
                    f"the spec is {spec.version}"
                )
            refs.append(LensRef(spec=spec, version=ref.version))
        names = [r.spec.name for r in refs if isinstance(r.spec, LensSpec)]
        problems += [
            f"lenses: {name!r} is listed twice"
            for name in sorted(set(names))
            if names.count(name) > 1
        ]
        if problems:
            raise ConfigError(problems, WHAT)
        loaded = self.model_copy(update={"lenses": tuple(refs)})
        loaded._verbatim = self._verbatim
        return loaded

    @property
    def lens_specs(self) -> tuple[LensSpec, ...]:
        specs = tuple(ref.spec for ref in self.lenses if isinstance(ref.spec, LensSpec))
        if len(specs) != len(self.lenses):
            raise ValueError("lens specs aren't loaded: use with_lenses_loaded() first")
        return specs

    @property
    def verbatim(self) -> dict[str, Any]:
        """The search space as given (the file's mapping, or the model's own data)."""
        if self._verbatim is not None:
            return plain(self._verbatim)
        return plain({k: v for k, v in self.model_dump(mode="json").items()})

    @property
    def digest(self) -> str:
        """sha256 of A: the space as given plus the digest of every lens spec it names."""
        return digest(
            {"space": self.verbatim, "lenses": {s.key: s.digest for s in self.lens_specs}}
        )


def _space_problems(space: SearchSpace) -> list[str]:
    problems = []
    if not space.lenses:
        problems.append("lenses: at least one lens is needed")
    length = space.resolution.moment_length_s
    if length.min > length.max:
        problems.append("resolution.moment_length_s: min is more than max")
    include = space.scope.units.include
    if include != "all" and not include:
        problems.append("scope.units.include: `all` or a list of unit ids")
    if include != "all":
        overlap = sorted(set(include) & set(space.scope.units.exclude))
        problems += [f"scope.units: {u!r} is both included and excluded" for u in overlap]
    return problems


__all__ = [
    "Evidence",
    "LensRef",
    "MomentLength",
    "QueryStyle",
    "Resolution",
    "Scope",
    "SearchSpace",
    "UnitScope",
]
