"""Frames: which frames a model sees, and how they're prepared and delivered
(docs/DESIGN.md §2.3), as data.

The schema and its load-time checks, resolution to the canonical form the manifest records
(with digests), the strategy registry, and pixel-free frame planning. The filters that need
pixels are in `framekit.frames.pixels` (numpy, the media extra); extraction's media stage
reads the frames, runs them, and prepares and delivers the images.

    params = FrameParams.from_mapping(                  # FrameParamsError lists problems
        {"vision_profiles": plan["vision_profiles"], "stages": plan["stages"]}
    )
    resolved = resolve(params)                           # pinned, defaults filled, digest
    problems = check_against_source(resolved, SourceLimits(fps=12, width_px=960,
                                                           height_px=720))
    frames = plan_frames(resolved.for_stage("L3a"), (12.0, 48.5), Signals(...))

No media library, no model call: strategies are named, versioned code; profiles only pick
and parameterize them.
"""

from framekit.frames import builtin as _builtin  # noqa: F401  registers the built-ins
from framekit.frames.checks import (
    ModelVisionCaps,
    SourceLimits,
    check_against_model,
    check_against_source,
    frame_cap,
)
from framekit.frames.params import (
    Budget,
    Delivery,
    DetailView,
    FrameParams,
    FrameParamsError,
    FrameProfile,
    FrameSelection,
    ImagePrep,
    OverviewView,
    StageFrames,
    StoryboardGrid,
    StrategyRef,
    Views,
)
from framekit.frames.planning import FramePlan, PlannedView, plan_frames
from framekit.frames.registry import (
    Strategy,
    StrategyKind,
    UnknownStrategyError,
    get_strategy,
    register_strategy,
    strategies,
)
from framekit.frames.resolve import (
    ResolvedFrameParams,
    ResolvedProfile,
    resolve,
    resolve_profile,
)
from framekit.frames.signals import Signals, Span

__all__ = [
    "Budget",
    "Delivery",
    "DetailView",
    "FramePlan",
    "FrameSelection",
    "ImagePrep",
    "ModelVisionCaps",
    "OverviewView",
    "PlannedView",
    "ResolvedProfile",
    "ResolvedFrameParams",
    "Signals",
    "SourceLimits",
    "Span",
    "StageFrames",
    "StoryboardGrid",
    "Strategy",
    "StrategyKind",
    "StrategyRef",
    "UnknownStrategyError",
    "Views",
    "FrameParams",
    "FrameParamsError",
    "FrameProfile",
    "check_against_model",
    "check_against_source",
    "frame_cap",
    "get_strategy",
    "plan_frames",
    "register_strategy",
    "resolve",
    "resolve_profile",
    "strategies",
]
