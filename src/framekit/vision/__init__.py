"""Vision: how a model sees video (`vision_params`, docs/DESIGN.md §2.3), as data.

Built so far: the schema and its load-time checks, resolution to the canonical form the
manifest records (with digests), the strategy registry, and pixel-free frame planning.
Filters that need pixels, image preparation and delivery run in the media stage (E1).

    params = VisionParams.from_mapping(plan["vision"])   # VisionParamsError lists problems
    resolved = resolve(params)                           # pinned, defaults filled, digest
    problems = check_against_source(resolved, SourceLimits(fps=12, width_px=960,
                                                           height_px=720))
    plan = plan_frames(resolved.for_stage("L3a"), (12.0, 48.5), Signals(...))

No media library, no model call: strategies are named, versioned code; profiles only pick
and parameterize them.
"""

from framekit.vision import builtin as _builtin  # noqa: F401  registers the built-ins
from framekit.vision.checks import (
    ModelVisionCaps,
    SourceLimits,
    check_against_model,
    check_against_source,
    frame_cap,
)
from framekit.vision.params import (
    Budget,
    Delivery,
    DetailView,
    FrameSelection,
    ImagePrep,
    OverviewView,
    StageVision,
    StoryboardGrid,
    StrategyRef,
    Views,
    VisionParams,
    VisionParamsError,
    VisionProfile,
)
from framekit.vision.planning import FramePlan, PlannedView, plan_frames
from framekit.vision.registry import (
    Strategy,
    StrategyKind,
    UnknownStrategyError,
    get_strategy,
    register_strategy,
    strategies,
)
from framekit.vision.resolve import (
    ResolvedProfile,
    ResolvedVisionParams,
    resolve,
    resolve_profile,
)
from framekit.vision.signals import Signals, Span

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
    "ResolvedVisionParams",
    "Signals",
    "SourceLimits",
    "Span",
    "StageVision",
    "StoryboardGrid",
    "Strategy",
    "StrategyKind",
    "StrategyRef",
    "UnknownStrategyError",
    "Views",
    "VisionParams",
    "VisionParamsError",
    "VisionProfile",
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
