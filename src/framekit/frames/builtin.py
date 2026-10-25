"""Built-in strategies, version 1. Registered when `framekit.frames` is imported.

Every number is a setting with no default unless only one choice exists, so a missing
setting fails at load time instead of being guessed (DESIGN.md §2.3). Allocation windows
have a default: it's recorded in the resolved form like any other setting.
"""

from bisect import bisect_left, bisect_right
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from framekit.frames.registry import Strategy, register_strategy
from framekit.frames.signals import Signals, Span

# shot_based "last": this far before the cut, which falls inside the shot's last frame
# whatever the frame rate. Planning works in whole milliseconds and only merges times in
# the same millisecond, so this frame and an anchor on the cut stay two frames.
LAST_FRAME_OFFSET_S = 0.001
# Candidates one strategy may produce for one span; past this, plan shorter spans (1 h at
# 500 fps is 1.8 million).
MAX_CANDIDATES = 2_000_000


class StrategyParams(BaseModel):
    """Base for strategy settings: frozen, unknown keys fail, numbers are strict (no bool or
    str) and finite."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)


class NoParams(StrategyParams):
    pass


# --- candidates -------------------------------------------------------------------------


class UniformParams(StrategyParams):
    fps: float = Field(gt=0, le=1000)


def uniform(params: UniformParams, span: Span, signals: Signals) -> list[float]:
    """Every 1/fps seconds from the span's start, end excluded."""
    start, end = span
    count = int((end - start) * params.fps) + 2  # the planner drops anything at or past end
    if count > MAX_CANDIDATES:
        raise ValueError(
            f"uniform at {params.fps:g} fps over {end - start:g} s is {count:,} candidates, "
            f"more than {MAX_CANDIDATES:,}: plan shorter spans"
        )
    return [start + k / params.fps for k in range(count)]


class ShotBasedParams(StrategyParams):
    pick: Literal["first", "middle", "last"]


def shot_based(params: ShotBasedParams, span: Span, signals: Signals) -> list[float]:
    """One frame per shot, from the shot boundaries inside the span: its first frame, its
    middle, or its last frame. The span's edges close the first and last shots."""
    start, end = span
    cuts = [t for t in signals.shot_boundaries if start < t < end]
    edges = [start, *cuts, end]
    picks = []
    for shot_start, shot_end in zip(edges, edges[1:], strict=False):
        if params.pick == "first":
            picks.append(shot_start)
        elif params.pick == "middle":
            picks.append((shot_start + shot_end) / 2)
        else:
            picks.append(max(shot_start, shot_end - LAST_FRAME_OFFSET_S))
    return picks


# --- anchors ----------------------------------------------------------------------------


def _within(times: Sequence[float], span: Span) -> list[float]:
    return [t for t in times if span[0] <= t < span[1]]


def shot_boundary_anchors(params: NoParams, span: Span, signals: Signals) -> list[float]:
    return _within(signals.shot_boundaries, span)


def audio_onset_anchors(params: NoParams, span: Span, signals: Signals) -> list[float]:
    return _within(signals.audio_onsets, span)


# --- filters (need pixels: validated and recorded now, run in the media stage) ----------


class DedupeParams(StrategyParams):
    method: Literal["phash"] = "phash"
    max_distance: int = Field(ge=0, le=64)  # Hamming distance between 64-bit hashes


class SharpnessParams(StrategyParams):
    keep: Literal["best_per_window"] = "best_per_window"
    window_s: float = Field(gt=0, le=60)


# --- allocation signals -----------------------------------------------------------------


class SignalWindowParams(StrategyParams):
    window_s: float = Field(default=1.0, gt=0, le=60)  # centred on each candidate


def motion_weights(params: SignalWindowParams, signals: Signals, times: Sequence[float]):
    """Mean motion of the samples within window_s / 2 of each time (0 when none), relative
    to the strongest sample: the planner normalizes every signal anyway, and the sums stay
    finite whatever the scale."""
    top = max((v for _, v in signals.motion), default=0.0)
    if top <= 0:
        return [0.0] * len(times)
    sample_times = [t for t, _ in signals.motion]
    values = [v / top for _, v in signals.motion]
    half = params.window_s / 2
    weights = []
    for t in times:
        lo, hi = bisect_left(sample_times, t - half), bisect_right(sample_times, t + half)
        weights.append(sum(values[lo:hi]) / (hi - lo) if hi > lo else 0.0)
    return weights


def audio_onset_weights(params: SignalWindowParams, signals: Signals, times: Sequence[float]):
    """Number of audio onsets within window_s / 2 of each time."""
    onsets = signals.audio_onsets
    half = params.window_s / 2
    return [
        float(bisect_right(onsets, t + half) - bisect_left(onsets, t - half)) for t in times
    ]


BUILTIN = (
    Strategy("uniform", 1, "candidates", UniformParams, run=uniform,
             description="every 1/fps seconds from the span's start"),
    Strategy("shot_based", 1, "candidates", ShotBasedParams, run=shot_based,
             description="the first, middle or last frame of each shot"),
    Strategy("shot_boundaries", 1, "anchor", NoParams, run=shot_boundary_anchors,
             description="a frame at every shot cut (L1)"),
    Strategy("audio_onsets", 1, "anchor", NoParams, run=audio_onset_anchors,
             description="a frame at every audio onset (L1)"),
    Strategy("dedupe", 1, "filter", DedupeParams, needs_pixels=True,
             description="drops near-duplicate frames: perceptual hashes within "
                         "max_distance (Hamming)"),
    Strategy("sharpness", 1, "filter", SharpnessParams, needs_pixels=True,
             description="keeps the sharpest frame per window (drops blur and smear frames)"),
    Strategy("motion", 1, "allocation", SignalWindowParams, run=motion_weights,
             description="mean motion near each candidate (L1)"),
    Strategy("audio_onsets", 1, "allocation", SignalWindowParams, run=audio_onset_weights,
             description="audio onsets near each candidate (L1)"),
)

for _strategy in BUILTIN:
    register_strategy(_strategy)
