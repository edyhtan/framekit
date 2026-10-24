"""Resolution: vision parameters in the canonical form the manifest records.

`resolve(params)` pins every strategy to `name@version`, fills every default (strategy
settings, delivery, image encoding, storyboard grid) and computes a stable SHA-256 digest
of the canonical JSON, per profile and for the whole block. The digests feed the dataset
manifest and the input hash of each stage that uses a profile (DESIGN.md §2.3).

Canonical means: sorted keys, no None, no empty collections, strategies pinned; anchors
and allocation signals are sets (sorted), filters keep their order. The same parameters
give the same digest whatever the key order or spelling of the input (`fps: 1` and
`fps: 1.0`, `uniform` and `uniform@1` once version 1 is the newest). A profile's digest
covers its content, not its name, so renaming a profile doesn't rerun a stage.
"""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from framekit.vision.params import (
    Budget,
    Delivery,
    DetailView,
    FrameSelection,
    ImagePrep,
    StoryboardGrid,
    StrategyRef,
    Views,
    VisionParams,
    VisionProfile,
    profile_mapping,
)
from framekit.vision.registry import StrategyKind, get_strategy

# Image encoding when a profile leaves it unset (not for native_video, where the provider
# decodes the clip). Placeholders like every default until a benchmark sweep sets them.
DEFAULT_FORMAT = "jpeg"
DEFAULT_QUALITY = 85  # jpeg and webp


@dataclass(frozen=True)
class ResolvedProfile:
    name: str
    profile: VisionProfile  # every strategy pinned, every default filled
    mapping: dict[str, Any]  # canonical
    digest: str  # sha256 of the canonical JSON

    def canonical_json(self) -> str:
        return canonical_json(self.mapping)


@dataclass(frozen=True)
class ResolvedVisionParams:
    params: VisionParams  # resolved: every profile pinned and filled
    profiles: dict[str, ResolvedProfile]
    stages: dict[str, str]  # stage → profile name
    mapping: dict[str, Any]  # canonical, for the manifest
    digest: str  # sha256 of the whole block's canonical JSON

    def to_mapping(self) -> dict[str, Any]:
        return json.loads(self.canonical_json())  # a deep copy

    def canonical_json(self) -> str:
        return canonical_json(self.mapping)

    def for_stage(self, stage: str) -> ResolvedProfile:
        if stage not in self.stages:
            raise KeyError(f"no vision profile assigned to stage {stage!r}")
        return self.profiles[self.stages[stage]]


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def pin(ref: StrategyRef, kind: StrategyKind) -> StrategyRef:
    """The reference with its version pinned and its settings validated, defaults filled."""
    strategy = get_strategy(kind, ref.name, ref.version)
    settings = strategy.params.model_validate(ref.params).model_dump(
        mode="json", exclude_none=True
    )
    return StrategyRef(name=strategy.name, version=strategy.version, params=settings)


def _pin_set(refs: list[StrategyRef], kind: StrategyKind) -> list[StrategyRef]:
    return sorted((pin(ref, kind) for ref in refs), key=lambda ref: (ref.name, ref.version))


def resolve_profile(profile: VisionProfile, name: str = "") -> ResolvedProfile:
    fs = profile.frame_selection
    frame_selection = FrameSelection(
        candidates=pin(fs.candidates, "candidates"),
        anchors=_pin_set(fs.anchors, "anchor"),
        filters=[pin(ref, "filter") for ref in fs.filters],
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
    resolved = VisionProfile(
        version=profile.version,
        frame_selection=frame_selection,
        image=image,
        delivery=delivery,
        views=views,
    )
    mapping = _sorted(profile_mapping(resolved))
    return ResolvedProfile(name=name, profile=resolved, mapping=mapping, digest=digest(mapping))


def resolve(params: VisionParams | Mapping[str, Any]) -> ResolvedVisionParams:
    """Pins and fills every profile; plain data is checked first (VisionParamsError)."""
    if not isinstance(params, VisionParams):
        params = VisionParams.from_mapping(params)
    profiles = {
        name: resolve_profile(profile, name)
        for name, profile in sorted(params.vision_profiles.items())
    }
    stages = {stage: entry.vision for stage, entry in sorted(params.stages.items())}
    mapping = _sorted(
        {
            "vision_profiles": {name: p.mapping for name, p in profiles.items()},
            "stages": {stage: {"vision": name} for stage, name in stages.items()},
        }
    )
    resolved_params = VisionParams(
        vision_profiles={name: p.profile for name, p in profiles.items()},
        stages=params.stages,
    )
    return ResolvedVisionParams(
        params=resolved_params,
        profiles=profiles,
        stages=stages,
        mapping=mapping,
        digest=digest(mapping),
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
