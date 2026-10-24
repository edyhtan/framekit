"""Checks that need outside facts, run when those facts are known.

- `check_against_source`: the analysis copy (L0) a profile will read. No profile may ask for
  more frames per second or more pixels than the copy has; nothing is upsampled.
- `check_against_model`: the vision limits of the model a stage will call. Capabilities are
  checked, never guessed: nothing is silently resampled or dropped (DESIGN.md §0.2).

Both return a list of problems (`path: message`), empty when all is well; raise
`VisionParamsError(problems)` to stop. The model catalog doesn't declare vision
capabilities yet, so callers describe a model with `ModelVisionCaps`.
"""

import math
from collections.abc import Iterator
from dataclasses import dataclass
from numbers import Integral, Real

from framekit.vision.params import VisionParams, VisionProfile
from framekit.vision.resolve import (
    ResolvedProfile,
    ResolvedVisionParams,
    resolve,
    resolve_profile,
)


def _whole(owner: str, field: str, value: object, *, optional: bool = False) -> None:
    """A count or a size: a whole number >= 1 (None too, when optional)."""
    if optional and value is None:
        return
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{owner}.{field} must be a whole number, got {value!r}")
    if value < 1:
        raise ValueError(f"{owner}.{field} must be >= 1, got {value!r}")


@dataclass(frozen=True)
class SourceLimits:
    """What the analysis copy of a unit has: frame rate and frame size in pixels."""

    fps: float
    width_px: int
    height_px: int

    def __post_init__(self) -> None:
        if isinstance(self.fps, bool) or not isinstance(self.fps, Real):
            raise TypeError(f"SourceLimits.fps must be a number, got {self.fps!r}")
        if not (math.isfinite(self.fps) and self.fps > 0):
            raise ValueError(f"SourceLimits.fps must be finite and > 0, got {self.fps!r}")
        _whole("SourceLimits", "width_px", self.width_px)
        _whole("SourceLimits", "height_px", self.height_px)

    @property
    def long_side_px(self) -> int:
        return max(self.width_px, self.height_px)


@dataclass(frozen=True)
class ModelVisionCaps:
    """What a model accepts. None means no stated limit."""

    native_video: bool = False
    max_images: int | None = None  # per request
    max_image_side_px: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.native_video, bool):
            raise TypeError(
                f"ModelVisionCaps.native_video must be True or False, got {self.native_video!r}"
            )
        _whole("ModelVisionCaps", "max_images", self.max_images, optional=True)
        _whole("ModelVisionCaps", "max_image_side_px", self.max_image_side_px, optional=True)


Profiles = VisionParams | ResolvedVisionParams | VisionProfile | ResolvedProfile
ONE_PROFILE = (
    "takes one profile, a VisionProfile or a ResolvedProfile (for a whole block, check the "
    "profile of each stage: resolve(params).for_stage(stage))"
)


def _profiles(params: Profiles) -> Iterator[tuple[str, VisionProfile]]:
    """(path prefix, resolved profile) pairs: checks see the defaults that will apply."""
    if isinstance(params, VisionParams):
        params = resolve(params)
    if isinstance(params, VisionProfile):
        params = resolve_profile(params)
    if isinstance(params, ResolvedVisionParams):
        for name, resolved in params.profiles.items():
            yield f"vision_profiles.{name}.", resolved.profile
    elif isinstance(params, ResolvedProfile):
        yield (f"vision_profiles.{params.name}." if params.name else ""), params.profile
    else:
        raise TypeError(
            "expected VisionParams, ResolvedVisionParams, VisionProfile or ResolvedProfile, "
            f"got {type(params).__name__}"
        )


def check_against_source(params: Profiles, source: SourceLimits) -> list[str]:
    """Problems if any profile asks for more fps or pixels than the analysis copy has."""
    if not isinstance(source, SourceLimits):
        raise TypeError(f"check_against_source needs SourceLimits, got {type(source).__name__}")
    problems = []
    for prefix, profile in _profiles(params):
        fps = profile.frame_selection.candidates.params.get("fps")
        if isinstance(fps, int | float) and fps > source.fps:
            problems.append(
                f"{prefix}frame_selection.candidates.fps: {fps:g} fps is more than the "
                f"analysis copy's {source.fps:g} fps"
            )
        overview = profile.views.overview if profile.views is not None else None
        if overview is not None and overview.fps is not None and overview.fps > source.fps:
            problems.append(
                f"{prefix}views.overview.fps: {overview.fps:g} fps is more than the "
                f"analysis copy's {source.fps:g} fps"
            )
        if profile.image.max_side_px > source.long_side_px:
            problems.append(
                f"{prefix}image.max_side_px: {profile.image.max_side_px} px is more than the "
                f"analysis copy's long side ({source.long_side_px} px)"
            )
    return problems


def frame_cap(profile: VisionProfile) -> int | None:
    """The most frames one planned span can send, or None when nothing bounds it."""
    if profile.delivery.mode == "native_video":
        return None
    fs = profile.frame_selection
    cap = fs.budget.cap()
    if cap is not None:
        return cap
    views = profile.views
    if (
        views is not None
        and views.mode == "overview_plus_detail"
        and views.overview is not None
        and views.detail is not None
        and not fs.anchors
    ):
        return views.overview.max_frames + views.detail.max_views * views.detail.max_frames
    return None


def check_against_model(
    profile: VisionProfile | ResolvedProfile, caps: ModelVisionCaps
) -> list[str]:
    """Problems if the model can't take what the profile sends: native video on a model
    without it, more images than it accepts (from the profile's own frame cap), or images
    larger than it accepts. A storyboard tile is one frame of `image.max_side_px`, so a grid
    image's long side is that times the larger grid dimension."""
    if not isinstance(profile, VisionProfile | ResolvedProfile):
        raise TypeError(f"check_against_model {ONE_PROFILE}, got {type(profile).__name__}")
    if not isinstance(caps, ModelVisionCaps):
        raise TypeError(f"check_against_model needs ModelVisionCaps, got {type(caps).__name__}")
    ((prefix, profile),) = _profiles(profile)
    problems = []
    delivery = profile.delivery
    if delivery.mode == "native_video":
        if not caps.native_video:
            problems.append(
                f"{prefix}delivery.mode: the model takes no native video; "
                "use images or storyboard delivery"
            )
    elif caps.max_images is not None:
        frames = frame_cap(profile)
        if frames is None:
            problems.append(
                f"{prefix}frame_selection.budget: no frame cap, but the model accepts at most "
                f"{caps.max_images} images; set max_frames_per_scene"
            )
        else:
            tiles = 1
            if delivery.mode == "storyboard" and delivery.storyboard is not None:
                tiles = delivery.storyboard.rows * delivery.storyboard.cols
            images = math.ceil(frames / tiles)
            if images > caps.max_images:
                unit = f"{images} images ({frames} frames)" if tiles > 1 else f"{frames} images"
                problems.append(
                    f"{prefix}frame_selection.budget: up to {unit} per request, but the model "
                    f"accepts at most {caps.max_images}"
                )
    if caps.max_image_side_px is not None and delivery.mode != "native_video":
        side = profile.image.max_side_px
        grid = delivery.storyboard if delivery.mode == "storyboard" else None
        if grid is not None:
            side *= max(grid.rows, grid.cols)
        if side > caps.max_image_side_px:
            what = "a storyboard grid's long side" if grid is not None else "image.max_side_px"
            problems.append(
                f"{prefix}image.max_side_px: {what} is {side} px, but the model accepts at "
                f"most {caps.max_image_side_px} px"
            )
    return problems
