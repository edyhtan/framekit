"""The strategy registry: named, versioned frame-selection strategies.

Strategies are code, settings are data (DESIGN.md §2.3). A frame profile names a strategy
for each slot and gives its settings; the registry says what each name means:

- **kind**: the slot it fills. `candidates` (where frames may come from), `anchor` (frames
  that are always kept), `filter` (removes frames, in order) or `allocation` (a signal that
  weights where the frame budget goes). Kinds are separate namespaces: the `audio_onsets`
  anchor and the `audio_onsets` allocation signal are two strategies.
- **params**: a Pydantic model that validates the settings (unknown keys fail) and fills
  their defaults.
- **needs_pixels**: whether it must look at decoded frames. Those are validated and
  recorded now and run in the media stage; the others run in pixel-free planning
  (`framekit.frames.planning`) through `run`.

A name never changes meaning: a changed algorithm registers a new version (`dedupe@2`), so
datasets made with `dedupe@1` stay reproducible. Registering an existing kind, name and
version again raises. A reference without a version resolves to the newest one, and the
resolved form records it.

`run` signatures, by kind (times in seconds on the unit timeline):
- `candidates`, `anchor`: `run(params, span, signals) -> list[float]`
- `filter` (pixel-free only): `run(params, times, signals) -> list[float]` (the times kept)
- `allocation`: `run(params, signals, times) -> list[float]` (one weight >= 0 per time)
"""

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, get_args

from pydantic import BaseModel

StrategyKind = Literal["candidates", "anchor", "filter", "allocation"]
KINDS: tuple[str, ...] = get_args(StrategyKind)
KIND_LABELS = {
    "candidates": "candidates strategy",
    "anchor": "anchor",
    "filter": "filter",
    "allocation": "allocation signal",
}

NAME = re.compile(r"[a-z][a-z0-9_]*")
REFERENCE = re.compile(r"(?P<name>[a-z][a-z0-9_]*)(?:@(?P<version>[1-9][0-9]*))?")
# Inline references ({strategy: uniform, fps: 1}) use this key, so no setting may have it.
RESERVED_SETTING = "strategy"


class UnknownStrategyError(LookupError):
    """A strategy name or version that isn't registered for the slot it's used in."""


@dataclass(frozen=True)
class Strategy:
    name: str
    version: int
    kind: StrategyKind
    params: type[BaseModel]
    needs_pixels: bool = False
    run: Callable[..., Any] | None = None  # pixel-free implementation; None if needs_pixels
    description: str = ""

    @property
    def key(self) -> str:
        return f"{self.name}@{self.version}"


_REGISTRY: dict[tuple[str, str, int], Strategy] = {}


def split_reference(reference: str) -> tuple[str, int | None]:
    """`"dedupe"` → `("dedupe", None)`; `"dedupe@1"` → `("dedupe", 1)`."""
    match = REFERENCE.fullmatch(reference) if isinstance(reference, str) else None
    if match is None:
        raise ValueError(
            f"not a strategy name: {reference!r} (expected a name like 'uniform' or 'uniform@1')"
        )
    version = match.group("version")
    return match.group("name"), int(version) if version else None


def register_strategy(strategy: Strategy) -> Strategy:
    """Add a strategy. Raises ValueError if this kind, name and version already exist."""
    if strategy.kind not in KINDS:
        raise ValueError(f"unknown strategy kind {strategy.kind!r} (kinds: {', '.join(KINDS)})")
    if not NAME.fullmatch(strategy.name):
        raise ValueError(f"strategy names are lower_snake_case: {strategy.name!r}")
    if strategy.version < 1:
        raise ValueError(f"strategy versions start at 1: {strategy.key}")
    if RESERVED_SETTING in strategy.params.model_fields:
        raise ValueError(f"{strategy.key}: a setting may not be called {RESERVED_SETTING!r}")
    if not strategy.needs_pixels and strategy.run is None:
        raise ValueError(f"{strategy.key}: a pixel-free strategy needs `run`")
    slot = (strategy.kind, strategy.name, strategy.version)
    if slot in _REGISTRY:
        raise ValueError(
            f"{KIND_LABELS[strategy.kind]} {strategy.key} is already registered; "
            "a changed algorithm registers a new version"
        )
    _REGISTRY[slot] = strategy
    return strategy


def _article(label: str) -> str:
    return f"{'an' if label[0] in 'aeiou' else 'a'} {label}"


def _versions(kind: str, name: str) -> list[int]:
    return sorted(v for (k, n, v) in _REGISTRY if k == kind and n == name)


def _names(kind: str) -> list[str]:
    return sorted({n for (k, n, _) in _REGISTRY if k == kind})


def get_strategy(kind: StrategyKind, name: str, version: int | None = None) -> Strategy:
    """The strategy for a slot. `name` may carry its version (`"dedupe@1"`); without one,
    the newest version. Raises UnknownStrategyError with what is available."""
    if kind not in KINDS:
        raise ValueError(f"unknown strategy kind {kind!r} (kinds: {', '.join(KINDS)})")
    base, inline_version = split_reference(name)
    if version is not None and inline_version is not None and version != inline_version:
        raise ValueError(f"two versions given for {base!r}: {inline_version} and {version}")
    version = version if version is not None else inline_version
    label = KIND_LABELS[kind]
    versions = _versions(kind, base)
    if not versions:
        other = [KIND_LABELS[k] for k in KINDS if k != kind and _versions(k, base)]
        known = ", ".join(_names(kind)) or "none"
        if other:
            raise UnknownStrategyError(
                f"{base!r} is {' and '.join(map(_article, other))}, not {_article(label)} "
                f"(known: {known})"
            )
        raise UnknownStrategyError(f"unknown {label} {base!r} (known: {known})")
    if version is None:
        version = versions[-1]
    if version not in versions:
        raise UnknownStrategyError(
            f"{label} {base}@{version} is not registered "
            f"(versions: {', '.join(map(str, versions))})"
        )
    return _REGISTRY[(kind, base, version)]


def strategies(kind: StrategyKind | None = None) -> list[Strategy]:
    """Every registered strategy (of one kind, if given), sorted by kind, name, version."""
    return [
        strategy
        for (k, _, _), strategy in sorted(_REGISTRY.items(), key=lambda item: item[0])
        if kind is None or k == kind
    ]
