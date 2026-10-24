"""Pixel-free planning: the timestamps a vision profile selects for one span.

`plan_frames(profile, span, signals)` runs everything that needs no pixels: candidates,
anchors, pixel-free filters, the budget and views. Filters that need pixels (`dedupe`,
`sharpness`) are returned as `pending_filters` for the media stage. Pure and deterministic:
the same profile, span and signals give the same plan.

**Whole milliseconds.** Every timestamp is on the unit timeline, rounded to the nearest
millisecond (halves up), and so are the span's edges: a plan's timestamps lie in the
rounded [start, end). Times in the same millisecond are one frame; 1 ms apart, two.

How the budget is spent (views.mode `single`, or no views):
1. **Anchors are kept first.** If they alone exceed the cap, the earliest ones are kept and a
   note says so.
2. **The rest goes to the candidates with the highest allocation weight.** A candidate's
   weight is the sum of the `allocate_by` signals near it, each normalized to its maximum
   over the candidates. Candidates of equal weight are taken in time order, which keeps the
   most frames `min_gap_s` allows, at a regular spacing. Only when the slots left can't take
   all of them is a spread subset of those chosen instead: each the farthest from the frames
   kept so far, ties to the earliest. So a cap never bunches frames at the start, and the
   plan has as many frames as the cap and `min_gap_s` allow together.
3. **min_gap_s** holds between every two kept frames, except anchor against anchor.

The cap is the tighter of `budget.max_frames` and `budget.max_frames_per_scene`, applied to
the planned span (a scene lies inside a unit, so it can exceed neither). Keeping
`max_frames` across the spans of a unit is the caller's job.

`overview_plus_detail` views: the anchors belong to the overview, which adds frames from a
uniform grid at `overview.fps` (or from the candidates) up to `overview.max_frames` in all,
spread over the span. Then each detail view takes up to `detail.max_frames` in a window of
`span_s` centred on one of the strongest peaks of the `around` signals. A peak is a run of
candidates of equal weight, higher than the candidates on either side, centred on its
middle (a peak cut by the span's edge is centred on the part inside the span); a peak
closer than `span_s` to a stronger one gets no view of its own. The cap bounds the total.
`native_video` profiles return the uniform timestamps for the record, with a note: the
provider samples the clip itself.
"""

import math
from bisect import bisect_left, bisect_right
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import groupby
from typing import Any, Literal

from pydantic import BaseModel

from framekit.vision.params import StrategyRef, VisionProfile, thaw
from framekit.vision.registry import Strategy, StrategyKind, get_strategy
from framekit.vision.resolve import ResolvedProfile, resolve_profile
from framekit.vision.signals import Signals, Span

INF = math.inf
Window = tuple[int, int]  # [start, end) in milliseconds


@dataclass(frozen=True)
class PlannedView:
    kind: Literal["single", "overview", "detail", "native_video"]
    start_s: float
    end_s: float
    timestamps: tuple[float, ...]


@dataclass(frozen=True)
class FramePlan:
    timestamps: tuple[float, ...]  # sorted, unit timeline, whole milliseconds
    pending_filters: tuple[StrategyRef, ...]  # pinned filters that need pixels, in order
    notes: tuple[str, ...]
    views: tuple[PlannedView, ...]  # how the timestamps split into looks


def _ms(seconds: float) -> int:
    """Seconds to whole milliseconds, halves rounded up."""
    return math.floor(seconds * 1000 + 0.5)


def _s(ms: float) -> float:
    return ms / 1000


def _stamps(times: Iterable[int]) -> tuple[float, ...]:
    return tuple(_s(t) for t in sorted(times))


def _span(span: Span) -> Window:
    try:
        start, end = (float(edge) for edge in span)
    except (TypeError, ValueError):
        raise ValueError(f"span must be (start_s, end_s), got {span!r}") from None
    if not (math.isfinite(start) and math.isfinite(end)) or start < 0 or end <= start:
        raise ValueError(f"span must be (start_s, end_s) with 0 <= start < end, got {span!r}")
    window = _ms(start), _ms(end)
    if window[1] <= window[0]:
        raise ValueError(
            f"span must hold a whole millisecond (timestamps are whole milliseconds), "
            f"got {span!r}"
        )
    return window


def _strategy(kind: StrategyKind, ref: StrategyRef) -> tuple[Strategy, BaseModel]:
    strategy = get_strategy(kind, ref.name, ref.version)
    return strategy, strategy.params.model_validate(thaw(ref.params))


def _run(strategy: Strategy) -> Callable[..., Any]:
    if strategy.run is None:
        raise NotImplementedError(f"{strategy.key} needs pixels; it can't be planned here")
    return strategy.run


def _times(kind: StrategyKind, ref: StrategyRef, window: Window, signals: Signals) -> list[int]:
    """The strategy's times inside the window: whole milliseconds, sorted, each once."""
    strategy, settings = _strategy(kind, ref)
    start, end = window
    found = _run(strategy)(settings, (_s(start), _s(end)), signals)
    return sorted({t for t in (_ms(x) for x in found if math.isfinite(x)) if start <= t < end})


def _weights(refs: Sequence[StrategyRef], pool: Sequence[int], signals: Signals) -> list[float]:
    total = [0.0] * len(pool)
    times = [_s(t) for t in pool]
    for ref in refs:
        strategy, settings = _strategy("allocation", ref)
        weights = [float(w) for w in _run(strategy)(settings, signals, times)]
        if len(weights) != len(times) or any(not (w >= 0 and math.isfinite(w)) for w in weights):
            raise ValueError(f"{strategy.key} must return one finite weight >= 0 per time")
        top = max(weights, default=0.0)
        if top > 0:
            total = [a + w / top for a, w in zip(total, weights, strict=True)]
    return total


class _Taken:
    """Which pool candidates are taken (a Fenwick tree over pool indices): any taken in a
    range of indices, in O(log n)."""

    def __init__(self, size: int):
        self._tree = [0] * (size + 1)

    def add(self, index: int) -> None:
        index += 1
        while index < len(self._tree):
            self._tree[index] += 1
            index += index & -index

    def _before(self, index: int) -> int:
        total = 0
        while index > 0:
            total += self._tree[index]
            index -= index & -index
        return total

    def any(self, lo: int, hi: int) -> bool:
        return hi > lo and self._before(hi) > self._before(lo)


def _fill(
    pool: Sequence[int],
    weights: Sequence[float],
    kept: Sequence[int],
    slots: int | None,
    gap_ms: int,
) -> list[int]:
    """Up to `slots` (None: no limit) more frames from `pool` (sorted), none closer than
    min_gap (and never in the same millisecond) to a kept frame or to each other.

    Highest weight first. Candidates of equal weight are taken in time order, which keeps
    the most min_gap allows; when they don't all fit in the slots left, a spread subset of
    them is taken instead (`_spread`)."""
    if not pool or (slots is not None and slots <= 0):
        return []
    reach = max(gap_ms, 1)  # two frames are at least this far apart
    fixed = sorted(kept)
    taken = _Taken(len(pool))
    chosen: list[int] = []

    def free(t: int) -> bool:
        i = bisect_left(fixed, t - reach + 1)
        if i < len(fixed) and fixed[i] < t + reach:
            return False
        return not taken.any(bisect_left(pool, t - reach + 1), bisect_left(pool, t + reach))

    levels: Iterable[Iterable[int]]
    if min(weights) == max(weights):
        levels = [range(len(pool))]
    else:
        order = sorted(range(len(pool)), key=lambda i: (-weights[i], pool[i]))
        levels = (level for _, level in groupby(order, key=weights.__getitem__))
    for level in levels:
        fits: list[int] = []
        for i in level:  # in time order
            if (not fits or pool[i] - pool[fits[-1]] >= reach) and free(pool[i]):
                fits.append(i)
        left = None if slots is None else slots - len(chosen)
        if left is not None and len(fits) > left:
            chosen += _spread([pool[i] for i in fits], sorted([*fixed, *chosen]), left)
            break
        for i in fits:
            taken.add(i)
        chosen += [pool[i] for i in fits]
        if left is not None and len(fits) == left:
            break
    return chosen


def _spread(candidates: Sequence[int], placed: Sequence[int], count: int) -> list[int]:
    """`count` of `candidates` (sorted, none in conflict with `placed` or each other), one
    at a time: the one farthest from every frame placed so far, ties to the earliest. Works
    on the gaps between placed frames: O(count log n)."""
    heap: list[tuple[float, int, int, int, int, float, float]] = []

    def queue(lo_t: float, hi_t: float, i: int, j: int) -> None:
        """Queue the best of candidates[i:j], the ones between frames at lo_t and hi_t."""
        if i >= j:
            return
        if lo_t == -INF:
            best, distance = i, (INF if hi_t == INF else hi_t - candidates[i])
        elif hi_t == INF:
            best, distance = j - 1, candidates[j - 1] - lo_t
        else:  # the farthest is one of the two next to the middle; ties to the earlier
            middle = bisect_left(candidates, (lo_t + hi_t) / 2, i, j)
            best, distance = -1, -INF
            for k in (middle - 1, middle):
                if i <= k < j and min(candidates[k] - lo_t, hi_t - candidates[k]) > distance:
                    best, distance = k, min(candidates[k] - lo_t, hi_t - candidates[k])
        heappush(heap, (-distance, candidates[best], best, i, j, lo_t, hi_t))

    bounds: list[float] = [-INF, *placed, INF]
    for lo_t, hi_t in zip(bounds, bounds[1:], strict=False):
        queue(lo_t, hi_t, bisect_right(candidates, lo_t), bisect_left(candidates, hi_t))
    picked: list[int] = []
    while heap and len(picked) < count:
        _, t, k, i, j, lo_t, hi_t = heappop(heap)
        picked.append(t)
        queue(lo_t, t, i, k)
        queue(t, hi_t, k + 1, j)
    return picked


def _peaks(
    pool: Sequence[int], weights: Sequence[float], spacing_ms: int, count: int
) -> list[float]:
    """Centres (ms) of the strongest peaks, strongest first (ties: earliest), at least
    `spacing_ms` apart. A peak is a run of candidates of equal weight > 0, higher than the
    candidates on either side; its centre is the run's middle. Windowed weights make a
    plateau around each event, so the middle is the event (or a cluster's middle)."""
    runs: list[tuple[float, int, float]] = []
    i, n = 0, len(pool)
    while i < n:
        j = i
        while j + 1 < n and weights[j + 1] == weights[i]:
            j += 1
        before = weights[i - 1] if i > 0 else -INF
        after = weights[j + 1] if j + 1 < n else -INF
        if weights[i] > max(before, after, 0.0):
            runs.append((-weights[i], pool[i], (pool[i] + pool[j]) / 2))
        i = j + 1
    centres: list[float] = []
    for _, _, centre in sorted(runs):
        if len(centres) == count:
            break
        if all(abs(centre - c) >= spacing_ms for c in centres):
            centres.append(centre)
    return centres


def plan_frames(
    profile: VisionProfile | ResolvedProfile, span: Span, signals: Signals | None = None
) -> FramePlan:
    """The frames `profile` selects in `span` (seconds on the unit timeline, end excluded),
    from the L1 `signals`. See the module docstring for the rules."""
    resolved = profile if isinstance(profile, ResolvedProfile) else resolve_profile(profile)
    p = resolved.profile
    window = start, end = _span(span)
    signals = Signals() if signals is None else signals
    fs = p.frame_selection
    notes: list[str] = []

    pool = _times("candidates", fs.candidates, window, signals)
    if p.delivery.mode == "native_video":
        fps = fs.candidates.params.get("fps")
        notes.append(
            f"native_video: the provider samples the clip itself at {fps:g} fps; "
            "these timestamps are nominal"
        )
        stamps = _stamps(pool)
        view = PlannedView("native_video", _s(start), _s(end), stamps)
        return FramePlan(stamps, (), tuple(notes), (view,))

    pending: list[StrategyRef] = []
    for ref in fs.filters:
        strategy, settings = _strategy("filter", ref)
        if pending or strategy.needs_pixels or strategy.run is None:
            pending.append(ref)  # order matters: everything after a pixel filter waits too
            continue
        survivors = {_ms(t) for t in _run(strategy)(settings, [_s(t) for t in pool], signals)}
        pool = [t for t in pool if t in survivors]
    if pending:
        notes.append(
            f"pixel filters pending ({', '.join(ref.key for ref in pending)}): the media "
            "stage runs them on the candidates before the budget; these timestamps assume "
            "every candidate survives"
        )

    anchors = sorted({t for ref in fs.anchors for t in _times("anchor", ref, window, signals)})
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
        views.append(PlannedView("single", _s(start), _s(end), _stamps(kept)))
    elif mode == "overview_plus_detail" and p.views is not None:
        overview, detail = p.views.overview, p.views.detail
        if overview is None or detail is None:  # excluded by validation
            raise ValueError("overview_plus_detail needs views.overview and views.detail")
        grid = pool
        if overview.fps is not None:
            uniform = StrategyRef(name="uniform", version=1, params={"fps": overview.fps})
            grid = _times("candidates", uniform, window, signals)
        if len(anchors) > overview.max_frames:
            notes.append(
                f"the overview holds {len(anchors)} anchors, more than its "
                f"{overview.max_frames} frames: anchors are always kept"
            )
        wanted = max(0, overview.max_frames - len(anchors))
        slots = room(wanted)
        picked = _fill(grid, [0.0] * len(grid), kept, slots, gap_ms)
        if slots is not None and slots < wanted:
            notes.append(
                f"scene cap of {cap} frames: the overview got {len(anchors) + len(picked)} "
                f"of {overview.max_frames} frames"
            )
        kept += picked
        views.append(PlannedView("overview", _s(start), _s(end), _stamps(anchors + picked)))
        weights = _weights(detail.around, pool, signals)
        span_ms = _ms(detail.span_s)
        centres = _peaks(pool, weights, span_ms, detail.max_views)
        if not centres:
            notes.append(
                "no peaks in the detail signals "
                f"({', '.join(ref.key for ref in detail.around)}): no detail views"
            )
        for centre in centres:
            lo = max(start, math.ceil(centre - span_ms / 2))
            hi = min(end, math.ceil(centre + span_ms / 2))
            i, j = bisect_left(pool, lo), bisect_left(pool, hi)
            slots = room(detail.max_frames)
            picked = _fill(pool[i:j], weights[i:j], kept, slots, gap_ms)
            if slots is not None and slots < detail.max_frames:
                notes.append(
                    f"scene cap of {cap} frames: the detail view at {_s(centre):g} s "
                    f"got {len(picked)} of {detail.max_frames} frames"
                )
            kept += picked
            views.append(PlannedView("detail", _s(lo), _s(hi), _stamps(picked)))
    else:
        raise NotImplementedError(
            f"views.mode {mode!r} is validated and recorded but not planned yet "
            "(planned: single, overview_plus_detail)"
        )
    return FramePlan(
        timestamps=_stamps(kept),
        pending_filters=tuple(pending),
        notes=tuple(notes),
        views=tuple(views),
    )
