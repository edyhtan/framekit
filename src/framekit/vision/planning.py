"""Pixel-free planning: the timestamps a vision profile selects for one span.

`plan_frames(profile, span, signals)` runs everything that needs no pixels: candidates,
anchors, pixel-free filters, the budget and views. Filters that need pixels (`dedupe`,
`sharpness`) are returned as `pending_filters` for the media stage. Pure and deterministic:
the same profile, span and signals give the same plan. Every timestamp is on the unit
timeline, rounded to milliseconds.

How the budget is spent (views.mode `single`, or no views):
1. **Anchors are kept first** (deduplicated within 1 ms). If they alone exceed the cap, the
   earliest ones are kept and a note says so.
2. **Remaining slots are filled from the candidates**, highest allocation weight first. A
   candidate's weight is the sum of the `allocate_by` signals near it, each normalized to
   its maximum over the candidates. Ties go to the candidate farthest from the frames
   already kept (frames spread out when no signal decides), then to the earliest.
3. **min_gap_s** holds between every two kept frames, except anchor against anchor; no two
   frames are ever within 1 ms.

The cap is the tightest of `budget.max_frames` and `budget.max_frames_per_scene` (a scene
lies inside a unit, so it can exceed neither).

`overview_plus_detail` views: after the anchors, the overview takes up to
`overview.max_frames` from a uniform grid at `overview.fps` across the span (spread the same
way); then each detail view, strongest first, takes up to `detail.max_frames` inside a
window of `span_s` centred on one of the strongest `around` peaks (non-overlapping, up to
`max_views`). The scene cap bounds the total. `native_video` profiles return the uniform
timestamps for the record, with a note: the provider samples the clip itself.
"""

import math
from bisect import bisect_left, insort
from collections.abc import Sequence
from dataclasses import dataclass
from heapq import heapify, heappop, heappush
from typing import Literal

from framekit.vision.params import StrategyRef, VisionProfile
from framekit.vision.registry import StrategyKind, get_strategy
from framekit.vision.resolve import ResolvedProfile, resolve_profile
from framekit.vision.signals import Signals, Span

SAME_FRAME_MS = 1  # timestamps this close are one frame
INF = math.inf


@dataclass(frozen=True)
class PlannedView:
    kind: Literal["single", "overview", "detail", "native_video"]
    start_s: float
    end_s: float
    timestamps: tuple[float, ...]


@dataclass(frozen=True)
class FramePlan:
    timestamps: tuple[float, ...]  # sorted, unit timeline, milliseconds
    pending_filters: tuple[StrategyRef, ...]  # pinned filters that need pixels, in order
    notes: tuple[str, ...]
    views: tuple[PlannedView, ...]  # how the timestamps split into looks


def _ms(seconds: float) -> int:
    return round(seconds * 1000)


def _s(ms: int) -> float:
    return ms / 1000


def _strategy(kind: StrategyKind, ref: StrategyRef):
    strategy = get_strategy(kind, ref.name, ref.version)
    return strategy, strategy.params.model_validate(ref.params)


def _merge_same_frame(times: Sequence[int]) -> list[int]:
    """Sorted, keeping the earliest of timestamps within 1 ms of each other."""
    out: list[int] = []
    for t in sorted(times):
        if not out or t - out[-1] > SAME_FRAME_MS:
            out.append(t)
    return out


def _times(kind: StrategyKind, ref: StrategyRef, span: Span, signals: Signals) -> list[int]:
    strategy, params = _strategy(kind, ref)
    if strategy.run is None:
        raise NotImplementedError(f"{strategy.key} needs pixels; it can't be planned here")
    start, end = _ms(span[0]), _ms(span[1])
    found = (_ms(t) for t in strategy.run(params, span, signals) if math.isfinite(t))
    return _merge_same_frame([t for t in found if start <= t < end])


def _weights(refs: Sequence[StrategyRef], pool: Sequence[int], signals: Signals) -> list[float]:
    total = [0.0] * len(pool)
    times = [_s(t) for t in pool]
    for ref in refs:
        strategy, params = _strategy("allocation", ref)
        weights = [float(w) for w in strategy.run(params, signals, times)]
        if len(weights) != len(times) or any(not (w >= 0 and math.isfinite(w)) for w in weights):
            raise ValueError(f"{strategy.key} must return one finite weight >= 0 per time")
        top = max(weights, default=0.0)
        if top > 0:
            total = [a + w / top for a, w in zip(total, weights, strict=True)]
    return total


def _fill(
    pool: Sequence[int],
    weights: Sequence[float],
    kept: Sequence[int],
    slots: int | None,
    gap_ms: int,
) -> list[int]:
    """Greedy fill: highest weight, then farthest from every frame kept so far, then
    earliest; skipping candidates within min_gap (or 1 ms) of a kept frame.

    Lazy greedy: a candidate's distance to the kept frames only shrinks, so a stored heap
    key is an upper bound; a stale top is re-keyed and pushed back.
    """
    if not pool or (slots is not None and slots <= 0):
        return []
    placed = sorted(kept)

    def nearest(t: int) -> float:
        i = bisect_left(placed, t)
        before = t - placed[i - 1] if i > 0 else INF
        after = placed[i] - t if i < len(placed) else INF
        return min(before, after)

    heap = [(-w, -nearest(t), t) for t, w in zip(pool, weights, strict=True)]
    heapify(heap)
    chosen: list[int] = []
    while heap and (slots is None or len(chosen) < slots):
        neg_weight, neg_distance, t = heappop(heap)
        distance = nearest(t)
        if distance <= SAME_FRAME_MS or distance < gap_ms:
            continue  # a conflict never clears: distances only shrink
        if -distance != neg_distance:
            heappush(heap, (neg_weight, -distance, t))
            continue
        chosen.append(t)
        insort(placed, t)
    return chosen


def _peaks(pool: Sequence[int], weights: Sequence[float], spacing_ms: int, count: int) -> list[int]:
    """The strongest candidates (weight > 0) at least `spacing_ms` apart, strongest first;
    ties go to the earliest."""
    centres: list[int] = []
    for i in sorted(range(len(pool)), key=lambda i: (-weights[i], pool[i])):
        if len(centres) == count or weights[i] <= 0:
            break
        if all(abs(pool[i] - c) >= spacing_ms for c in centres):
            centres.append(pool[i])
    return centres


def _check_span(span: Span) -> tuple[float, float]:
    start, end = float(span[0]), float(span[1])
    if not (math.isfinite(start) and math.isfinite(end)) or start < 0 or end <= start:
        raise ValueError(f"span must be (start_s, end_s) with 0 <= start < end, got {span!r}")
    return start, end


def plan_frames(
    profile: VisionProfile | ResolvedProfile, span: Span, signals: Signals | None = None
) -> FramePlan:
    """The frames `profile` selects in `span` (seconds on the unit timeline, end excluded),
    from the L1 `signals`. See the module docstring for the rules."""
    resolved = profile if isinstance(profile, ResolvedProfile) else resolve_profile(profile)
    p = resolved.profile
    span = _check_span(span)
    signals = signals or Signals()
    start, end = _ms(span[0]), _ms(span[1])
    fs = p.frame_selection
    notes: list[str] = []

    pool = _times("candidates", fs.candidates, span, signals)
    if p.delivery.mode == "native_video":
        fps = fs.candidates.params.get("fps")
        notes.append(
            f"native_video: the provider samples the clip itself at {fps:g} fps; "
            "these timestamps are nominal"
        )
        stamps = tuple(_s(t) for t in pool)
        view = PlannedView("native_video", _s(start), _s(end), stamps)
        return FramePlan(stamps, (), tuple(notes), (view,))

    pending: list[StrategyRef] = []
    for ref in fs.filters:
        strategy, params = _strategy("filter", ref)
        if pending or strategy.needs_pixels or strategy.run is None:
            pending.append(ref)  # order matters: everything after a pixel filter waits too
            continue
        kept = {_ms(t) for t in strategy.run(params, [_s(t) for t in pool], signals)}
        pool = [t for t in pool if t in kept]
    if pending:
        notes.append(
            f"pixel filters pending ({', '.join(ref.key for ref in pending)}): the media "
            "stage runs them on the candidates before the budget; these timestamps assume "
            "every candidate survives"
        )

    anchors: list[int] = []
    for ref in fs.anchors:
        anchors += _times("anchor", ref, span, signals)
    anchors = _merge_same_frame(anchors)
    cap = fs.budget.cap()
    if cap is not None and len(anchors) > cap:
        notes.append(
            f"{len(anchors)} anchors exceed the cap of {cap} frames; kept the earliest {cap}"
        )
        anchors = anchors[:cap]
    gap_ms = _ms(fs.budget.min_gap_s or 0.0)
    kept = list(anchors)

    def room(wanted: int | None) -> int | None:
        if cap is None:
            return wanted
        left = cap - len(kept)
        return left if wanted is None else min(wanted, left)

    views: list[PlannedView] = []
    mode = p.views.mode if p.views is not None else "single"
    if mode == "single":
        weights = _weights(fs.budget.allocate_by, pool, signals)
        kept += _fill(pool, weights, kept, room(None), gap_ms)
        views.append(PlannedView("single", _s(start), _s(end), tuple(_s(t) for t in sorted(kept))))
    elif mode == "overview_plus_detail" and p.views is not None:
        overview, detail = p.views.overview, p.views.detail
        if overview is None or detail is None:  # excluded by validation
            raise ValueError("overview_plus_detail needs views.overview and views.detail")
        grid = pool
        if overview.fps is not None:
            grid = _times("candidates", StrategyRef(name="uniform", version=1,
                                                    params={"fps": overview.fps}), span, signals)
        slots = room(overview.max_frames)
        picked = _fill(grid, [0.0] * len(grid), kept, slots, gap_ms)
        if slots is not None and slots < overview.max_frames:
            notes.append(f"scene cap of {cap} frames: the overview got {len(picked)} of "
                         f"{overview.max_frames} frames")
        kept += picked
        views.append(PlannedView("overview", _s(start), _s(end),
                                 tuple(_s(t) for t in sorted(anchors + picked))))
        weights = _weights(detail.around, pool, signals)
        centres = _peaks(pool, weights, _ms(detail.span_s), detail.max_views)
        if not centres:
            notes.append("no peaks in the detail signals "
                         f"({', '.join(ref.key for ref in detail.around)}): no detail views")
        half = _ms(detail.span_s) / 2
        for centre in centres:
            lo, hi = max(start, math.ceil(centre - half)), min(end, math.ceil(centre + half))
            inside = [i for i, t in enumerate(pool) if lo <= t < hi]
            slots = room(detail.max_frames)
            picked = _fill([pool[i] for i in inside], [weights[i] for i in inside],
                           kept, slots, gap_ms)
            if slots is not None and slots < detail.max_frames:
                notes.append(f"scene cap of {cap} frames: the detail view at {_s(centre):g} s "
                             f"got {len(picked)} of {detail.max_frames} frames")
            kept += picked
            views.append(PlannedView("detail", _s(lo), _s(hi),
                                     tuple(_s(t) for t in sorted(picked))))
    else:
        raise NotImplementedError(
            f"views.mode {mode!r} is validated and recorded but not planned yet "
            "(planned: single, overview_plus_detail)"
        )
    return FramePlan(
        timestamps=tuple(_s(t) for t in sorted(kept)),
        pending_filters=tuple(pending),
        notes=tuple(notes),
        views=tuple(views),
    )
