"""Resolution: frame parameters in the canonical form the manifest records.

`resolve(params)` pins every strategy to `name@version`, fills every default (strategy
settings, delivery, image encoding, storyboard grid) and computes a stable SHA-256 digest
of the canonical JSON, per profile and for the whole block. The digests feed the dataset
manifest and the input hash of each stage that uses a profile (DESIGN.md §2.3).

Canonical means: sorted keys, no None, no empty collections, strategies pinned; anchors
and allocation signals are sets (sorted), filters keep their order. The same parameters
give the same digest whatever the key order or spelling of the input (`fps: 1` and
`fps: 1.0`, `uniform` and `uniform@1` once version 1 is the newest). A profile's digest
covers its content, not its name, so renaming a profile doesn't rerun a stage.

Resolved parameters are read-only: the profiles, their settings, and the canonical text the
digest was computed from. `.mapping` returns a fresh copy of that text as plain data.
"""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

from framekit.frames.params import (
    Budget,
    Delivery,
    DetailView,
    FrameParams,
    FrameProfile,
    FrameSelection,
    FrozenDict,
    ImagePrep,
    StoryboardGrid,
    StrategyRef,
    Views,
    profile_mapping,
    thaw,
)
from framekit.frames.registry import StrategyKind, get_strategy

# Image encoding when a profile leaves it unset (not for native_video, where the provider
# decodes the clip). Placeholders like every default until a benchmark sweep sets them.
DEFAULT_FORMAT: Final = "jpeg"
DEFAULT_QUALITY: Final = 85  # jpeg and webp


@dataclass(frozen=True)
class ResolvedProfile:
    name: str
    profile: FrameProfile  # every strategy pinned, every default filled
    canonical: str  # the canonical JSON
    digest: str  # its sha256

    @property
    def mapping(self) -> dict[str, Any]:
        """The canonical form as plain data (a fresh copy)."""
        result: dict[str, Any] = json.loads(self.canonical)
        return result

    def canonical_json(self) -> str:
        return self.canonical


@dataclass(frozen=True)
class ResolvedFrameParams:
    params: FrameParams  # resolved: every profile pinned and filled
    profiles: Mapping[str, ResolvedProfile]  # read-only
    stages: Mapping[str, str]  # stage → profile name, read-only
    canonical: str  # the canonical JSON of the whole block, for the manifest
    digest: str  # its sha256

    @property
    def mapping(self) -> dict[str, Any]:
        """The canonical form as plain data (a fresh copy)."""
        result: dict[str, Any] = json.loads(self.canonical)
        return result

    def to_mapping(self) -> dict[str, Any]:
        return self.mapping

    def canonical_json(self) -> str:
        return self.canonical

    def for_stage(self, stage: str) -> ResolvedProfile:
        if stage not in self.stages:
            raise KeyError(f"no frame profile assigned to stage {stage!r}")
        return self.profiles[self.stages[stage]]


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def pin(ref: StrategyRef, kind: StrategyKind) -> StrategyRef:
    """The reference with its version pinned and its settings validated, defaults filled."""
    strategy = get_strategy(kind, ref.name, ref.version)
    settings = strategy.params.model_validate(thaw(ref.params)).model_dump(
        mode="json", exclude_none=True
    )
    return StrategyRef(name=strategy.name, version=strategy.version, params=settings)


def _pin_set(refs: tuple[StrategyRef, ...], kind: StrategyKind) -> tuple[StrategyRef, ...]:
    """Anchors and signals are sets: pinned and sorted, so their order doesn't matter."""
    return tuple(sorted((pin(ref, kind) for ref in refs), key=lambda ref: (ref.name, ref.version)))


def resolve_profile(profile: FrameProfile, name: str = "") -> ResolvedProfile:
    fs = profile.frame_selection
    frame_selection = FrameSelection(
        candidates=pin(fs.candidates, "candidates"),
        anchors=_pin_set(fs.anchors, "anchor"),
        filters=tuple(pin(ref, "filter") for ref in fs.filters),
        budget=Budget(
            max_frames=fs.budget.max_frames,
            max_frames_per_scene=fs.budget.max_frames_per_scene,
            allocate_by=_pin_set(fs.budget.allocate_by, "allocation"),
            min_gap_s=fs.budget.min_gap_s,
        ),
    )
    image, delivery = profile.image, profile.delivery
    if delivery.mode != "native_video":
        fmt = image.format or DEFAULT_FORMAT
        quality = image.quality
        if quality is None and fmt in ("jpeg", "webp"):
            quality = DEFAULT_QUALITY
        image = ImagePrep(
            max_side_px=image.max_side_px, aspect=image.aspect, format=fmt, quality=quality
        )
    if delivery.mode == "storyboard" and delivery.storyboard is None:
        delivery = Delivery(mode="storyboard", storyboard=StoryboardGrid())
    views = profile.views
    if views is not None and views.detail is not None:
        detail = views.detail
        views = Views(
            mode=views.mode,
            overview=views.overview,
            detail=DetailView(
                around=_pin_set(detail.around, "allocation"),
                span_s=detail.span_s,
                max_views=detail.max_views,
                max_frames=detail.max_frames,
            ),
        )
    resolved = FrameProfile(
        version=profile.version,
        frame_selection=frame_selection,
        image=image,
        delivery=delivery,
        views=views,
    )
    text = canonical_json(_sorted(profile_mapping(resolved)))
    return ResolvedProfile(name=name, profile=resolved, canonical=text, digest=digest(text))


def resolve(params: FrameParams | Mapping[str, Any]) -> ResolvedFrameParams:
    """Pins and fills every profile; plain data is checked first (FrameParamsError)."""
    if not isinstance(params, FrameParams):
        params = FrameParams.from_mapping(params)
    profiles = {
        name: resolve_profile(profile, name)
        for name, profile in sorted(params.profiles.items())
    }
    stages = {stage: entry.profile for stage, entry in sorted(params.stages.items())}
    text = canonical_json(
        _sorted(
            {
                "profiles": {name: p.mapping for name, p in profiles.items()},
                "stages": {stage: {"profile": name} for stage, name in stages.items()},
            }
        )
    )
    resolved_params = FrameParams(
        profiles={name: p.profile for name, p in profiles.items()},
        stages=params.stages,
    )
    return ResolvedFrameParams(
        params=resolved_params,
        profiles=FrozenDict(profiles),
        stages=FrozenDict(stages),
        canonical=text,
        digest=digest(text),
    )


def _sorted(value: Any) -> Any:
    """Mappings with sorted keys, recursively (lists keep their order); empty mappings and
    lists dropped, as in the YAML shape."""
    if isinstance(value, Mapping):
        return {
            key: _sorted(value[key])
            for key in sorted(value)
            if value[key] is not None and value[key] != {} and value[key] != []
        }
    if isinstance(value, list | tuple):
        return [_sorted(item) for item in value]
    return value
