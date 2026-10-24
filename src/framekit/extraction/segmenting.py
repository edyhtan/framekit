"""Segments on the unit timeline, as pure functions: the parts of a unit that are processed,
shots from cuts, scenes from shot groups or from a model's proposal, fixed windows.

Deterministic: the same inputs give the same segments, and every repair of a model's
proposal is reported as a note.
"""

import math
from collections.abc import Sequence

from framekit.extraction.schemas.plan import SceneLength

Span = tuple[float, float]
Scene = tuple[float, float, str]  # start, end, summary

MIN_SCENE_S = 0.5  # nothing shorter is a scene, whatever the plan says


def _r(t: float) -> float:
    return round(t, 3)


def processable_spans(duration_s: float, excluded: Sequence[Span]) -> list[Span]:
    """The unit minus its excluded ranges (title cards, logos), as sorted spans."""
    spans: list[Span] = []
    cursor = 0.0
    for start, end in sorted(excluded):
        start, end = max(0.0, start), min(duration_s, end)
        if start > cursor:
            spans.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < duration_s:
        spans.append((cursor, duration_s))
    return [(_r(a), _r(b)) for a, b in spans if b - a >= MIN_SCENE_S]


def shots_from_cuts(cuts: Sequence[float], duration_s: float) -> list[Span]:
    edges = [0.0, *sorted(t for t in cuts if 0 < t < duration_s), duration_s]
    return [(_r(a), _r(b)) for a, b in zip(edges, edges[1:], strict=False) if b > a]


def _split(start: float, end: float, max_s: float, cuts: Sequence[float]) -> list[float]:
    """Boundaries that split [start, end] into pieces of at most max_s: near even division
    points, moved to the closest cut within a quarter piece when there is one."""
    pieces = math.ceil((end - start) / max_s - 1e-9)
    if pieces <= 1:
        return []
    length = (end - start) / pieces
    bounds = []
    for k in range(1, pieces):
        target = start + k * length
        near = [c for c in cuts if abs(c - target) <= length / 4 and start < c < end]
        bounds.append(min(near, key=lambda c: (abs(c - target), c)) if near else target)
    return sorted(set(bounds))


def group_shots(span: Span, cuts: Sequence[float], length: SceneLength) -> list[Span]:
    """Scenes for `segmentation.method: shot_groups`: consecutive shots grouped until a
    scene reaches the target length, never past the max (a longer shot is split evenly),
    and a short remainder merged into the scene before it when that stays within the max."""
    start, end = span
    inner = [c for c in sorted(cuts) if start < c < end]
    scenes: list[Span] = []
    scene_start = start
    edges = [*inner, end]
    for k, edge in enumerate(edges):
        size = edge - scene_start
        nxt = edges[k + 1] if k + 1 < len(edges) else None
        if size >= length.target or nxt is None or nxt - scene_start > length.max:
            if size > length.max:
                bounds = [scene_start, *_split(scene_start, edge, length.max, inner), edge]
                scenes += list(zip(bounds, bounds[1:], strict=False))
            else:
                scenes.append((scene_start, edge))
            scene_start = edge
    return _merge_short([(_r(a), _r(b)) for a, b in scenes], length)


def _merge_short(scenes: list[Span], length: SceneLength) -> list[Span]:
    """Scenes under the min merged into a neighbour (the shorter one) when the result stays
    within the max."""
    merged = list(scenes)
    changed = True
    while changed and len(merged) > 1:
        changed = False
        for i, (a, b) in enumerate(merged):
            if b - a >= length.min:
                continue
            options = []
            if i > 0:
                options.append((merged[i - 1][1] - merged[i - 1][0], i - 1))
            if i + 1 < len(merged):
                options.append((merged[i + 1][1] - merged[i + 1][0], i + 1))
            for _, j in sorted(options):
                lo, hi = min(i, j), max(i, j)
                if merged[hi][1] - merged[lo][0] <= length.max:
                    merged[lo : hi + 1] = [(merged[lo][0], merged[hi][1])]
                    changed = True
                    break
            if changed:
                break
    return merged


def snap(t: float, cuts: Sequence[float], tolerance_s: float) -> float:
    """The closest cut within the tolerance (ties: the earlier), or t itself."""
    near = [c for c in cuts if abs(c - t) <= tolerance_s]
    return min(near, key=lambda c: (abs(c - t), c)) if near else t


def repair_scenes(
    proposed: Sequence[Scene],
    spans: Sequence[Span],
    cuts: Sequence[float],
    length: SceneLength,
    *,
    snap_to_cuts: bool,
    tolerance_s: float,
) -> tuple[list[Scene], list[str]]:
    """A model's proposed scenes made into a clean cover of each processable span: ordered,
    contiguous, boundaries snapped to cuts, too-short scenes merged and too-long ones split.
    Every repair is reported."""
    notes: list[str] = []
    valid = sorted(
        (float(a), float(b), s)
        for a, b, s in proposed
        if math.isfinite(a) and math.isfinite(b) and a < b
    )
    if len(valid) < len(proposed):
        notes.append(f"{len(proposed) - len(valid)} proposed scenes had no valid time span")
    scenes: list[Scene] = []
    for start, end in spans:
        inside = [(max(a, start), min(b, end), s) for a, b, s in valid if a < end and b > start]
        if not inside:
            notes.append(f"no proposed scene in [{start}, {end}]: shot groups used there")
            scenes += [(a, b, "") for a, b in group_shots((start, end), cuts, length)]
            continue
        bounds = [start]
        for a, _, _ in inside[1:]:
            moved = snap(a, cuts, tolerance_s) if snap_to_cuts else a
            if moved != a:
                notes.append(f"boundary {a:.3f} snapped to the cut at {moved:.3f}")
            if bounds[-1] + MIN_SCENE_S <= moved <= end - MIN_SCENE_S:
                bounds.append(moved)
        bounds.append(end)
        if inside[0][0] - start > 1 or end - inside[-1][1] > 1:
            notes.append(f"the proposal didn't cover [{start}, {end}] fully: edges extended")
        pieces = list(zip(bounds, bounds[1:], strict=False))
        merged = _merge_short(pieces, length)
        if len(merged) < len(pieces):
            notes.append(f"{len(pieces) - len(merged)} scenes under {length.min:g} s merged")
        final: list[Span] = []
        for a, b in merged:
            if b - a > length.max:
                cut = [a, *_split(a, b, length.max, cuts), b]
                notes.append(f"a {b - a:.1f} s scene split in {len(cut) - 1}")
                final += list(zip(cut, cut[1:], strict=False))
            else:
                final.append((a, b))
        # Each proposed summary goes to the final scene holding its scene's midpoint.
        texts: dict[int, list[str]] = {}
        for a, b, summary in inside:
            middle = (a + b) / 2
            k = next((i for i, (x, y) in enumerate(final) if x <= middle < y), len(final) - 1)
            if summary:
                texts.setdefault(k, []).append(summary)
        for k, (a, b) in enumerate(final):
            scenes.append((_r(a), _r(b), "; ".join(dict.fromkeys(texts.get(k, [])))))
    return scenes, notes


def fixed_windows(spans: Sequence[Span], window_s: float, stride_s: float) -> list[Span]:
    """Windows of window_s every stride_s inside each span (the last one ends at the span's
    end; spans shorter than a window give one window)."""
    windows: list[Span] = []
    for start, end in spans:
        if end - start <= window_s:
            windows.append((_r(start), _r(end)))
            continue
        k = 0
        while True:
            a = start + k * stride_s
            if a + window_s >= end:
                windows.append((_r(max(start, end - window_s)), _r(end)))
                break
            windows.append((_r(a), _r(a + window_s)))
            k += 1
    return list(dict.fromkeys(windows))


__all__ = [
    "fixed_windows",
    "group_shots",
    "processable_spans",
    "repair_scenes",
    "shots_from_cuts",
    "snap",
]
