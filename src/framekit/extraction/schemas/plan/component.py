"""A plan component: one of the five groups of the plan (docs/settings.md §5).

Each component has its own version, its own checks and a digest of its canonical form, so
it can be loaded, checked, compared and reused on its own:

    observation = Observation.load("observation.yaml")   # ConfigError lists every problem
    observation.digest                                    # equal digests: equal settings
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar, Self

from framekit.extraction.schemas.common import Model, digest, parse, plain, read_mapping, unwrap


class Component(Model):
    """One group of the plan. Subclasses set `KEY` (the group's name in a plan file) and a
    `version` field, and may add checks across their own settings in `problems()`."""

    KEY: ClassVar[str]

    def problems(self) -> list[str]:
        """Checks across the component's own settings, with paths inside the component."""
        return []

    @property
    def digest(self) -> str:
        """sha256 of the component's canonical form, defaults filled in."""
        return digest(self)

    @classmethod
    def what(cls) -> str:
        return f"the {cls.KEY} component"

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> Self:
        """The component from plain data: `{<KEY>: {...}}` or the bare mapping. Settings it
        doesn't give keep their defaults; one ConfigError lists every problem."""
        raw = dict(plain(unwrap(data, cls.KEY)))
        return parse(cls, raw, cls.what(), extra=lambda c: c.problems())

    @classmethod
    def load(cls, path: str | Path) -> Self:
        return cls.from_mapping(read_mapping(Path(path).expanduser(), cls.what()))


__all__ = ["Component"]
