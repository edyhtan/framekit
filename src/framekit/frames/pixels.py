"""Pixel filters: the frame-selection filters that need decoded frames (`dedupe`, `sharpness`).

Planning (`plan_frames`) returns these as `pending_filters`; the media stage decodes the
planned frames and runs them here, in the profile's order, after the budget. They only
remove frames: version 1 doesn't refill the budget with other candidates (DESIGN.md §9).
**Anchors are never dropped.**

Needs numpy (`pip install "framekit[media]"`); importing `framekit.frames` alone doesn't.
Each filter is registered under its strategy's name and version, so `dedupe@1` keeps one
meaning; a changed algorithm registers a new version, here and in the strategy registry.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from pydantic import BaseModel

from framekit.frames.params import StrategyRef, thaw
from framekit.frames.registry import get_strategy

HASH_SIDE = 32  # pHash: the frame is reduced to 32 x 32, its DCT kept at 8 x 8
HASH_BITS = 8


@dataclass(frozen=True)
class PixelFrame:
    time_s: float  # on the unit timeline
    gray: np.ndarray  # 2-D uint8 luma
    anchor: bool = False  # anchors are never dropped


# A filter gets its checked settings and the frames in time order, and says which to keep.
PixelFilter = Callable[[Any, Sequence[PixelFrame]], list[bool]]
_FILTERS: dict[tuple[str, int], PixelFilter] = {}


def register_pixel_filter(name: str, version: int, run: PixelFilter) -> None:
    """The pixel implementation of a registered `needs_pixels` filter strategy."""
    key = (name, version)
    if key in _FILTERS:
        raise ValueError(f"pixel filter {name}@{version} is already registered")
    strategy = get_strategy("filter", name, version)
    if not strategy.needs_pixels:
        raise ValueError(f"{strategy.key} runs in planning; it needs no pixel implementation")
    _FILTERS[key] = run


def apply_pixel_filters(
    filters: Sequence[StrategyRef], frames: Sequence[PixelFrame]
) -> tuple[list[PixelFrame], list[str]]:
    """The frames left after each pending filter in order, and a note per filter that
    dropped any. Frames are kept in time order."""
    kept = sorted(frames, key=lambda f: f.time_s)
    notes = []
    for ref in filters:
        strategy = get_strategy("filter", ref.name, ref.version)
        run = _FILTERS.get((strategy.name, strategy.version))
        if run is None:
            raise LookupError(f"no pixel implementation for filter {strategy.key}")
        settings: BaseModel = strategy.params.model_validate(thaw(ref.params))
        mask = run(settings, kept)
        if len(mask) != len(kept):
            raise ValueError(
                f"{strategy.key} returned {len(mask)} decisions for {len(kept)} frames"
            )
        survivors = [f for f, keep in zip(kept, mask, strict=True) if keep or f.anchor]
        if len(survivors) < len(kept):
            notes.append(
                f"{strategy.key} dropped {len(kept) - len(survivors)} of {len(kept)} frames"
            )
        kept = survivors
    return kept, notes


# --- measures ---------------------------------------------------------------------------


def _area_resize(gray: np.ndarray, side: int) -> np.ndarray:
    """Average-pool a frame to side x side (frames smaller than that are enlarged by
    repetition first). Deterministic and dependency-free."""
    frame = gray.astype(np.float64)
    height, width = frame.shape
    if height < side:
        frame = np.repeat(frame, -(-side // height), axis=0)
    if width < side:
        frame = np.repeat(frame, -(-side // width), axis=1)
    height, width = frame.shape
    rows = np.linspace(0, height, side + 1).astype(int)
    cols = np.linspace(0, width, side + 1).astype(int)
    sums = np.add.reduceat(np.add.reduceat(frame, rows[:-1], axis=0), cols[:-1], axis=1)
    counts = np.outer(np.diff(rows), np.diff(cols))
    return sums / counts


def _dct_matrix(n: int) -> np.ndarray:
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    matrix = np.cos(np.pi * (2 * i + 1) * k / (2 * n)) * np.sqrt(2 / n)
    matrix[0] /= np.sqrt(2)
    return matrix


_DCT = _dct_matrix(HASH_SIDE)


def phash(gray: np.ndarray) -> int:
    """64-bit perceptual hash: the frame pooled to 32 x 32, its 2-D DCT, and one bit per
    coefficient of the top-left 8 x 8 block: above the block's median or not."""
    pooled = _area_resize(gray, HASH_SIDE)
    coefficients = _DCT @ pooled @ _DCT.T
    block = coefficients[:HASH_BITS, :HASH_BITS].ravel()
    bits = block > np.median(block)
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def sharpness(gray: np.ndarray) -> float:
    """Variance of the 4-neighbour Laplacian: higher is sharper (blur and smear frames score
    low). Computed on the frame as decoded for sending, so frames of one profile compare."""
    frame = gray.astype(np.float64)
    if frame.shape[0] < 3 or frame.shape[1] < 3:
        return 0.0
    laplacian = (
        frame[1:-1, :-2]
        + frame[1:-1, 2:]
        + frame[:-2, 1:-1]
        + frame[2:, 1:-1]
        - 4 * frame[1:-1, 1:-1]
    )
    return float(laplacian.var())


# --- the built-in filters, version 1 ----------------------------------------------------


def dedupe_v1(settings: Any, frames: Sequence[PixelFrame]) -> list[bool]:
    """In time order, drop a frame whose pHash is within `max_distance` (Hamming) of the
    last frame kept: consecutive near-duplicates go, a composition that comes back later
    stays. Anchors are kept and become the frame compared against."""
    keep = []
    last: int | None = None
    for frame in frames:
        code = phash(frame.gray)
        duplicate = last is not None and hamming(code, last) <= settings.max_distance
        if frame.anchor or not duplicate:
            keep.append(True)
            last = code
        else:
            keep.append(False)
    return keep


def sharpness_v1(settings: Any, frames: Sequence[PixelFrame]) -> list[bool]:
    """Windows of `window_s` on the unit timeline (from 0): in each, the sharpest frame that
    isn't an anchor is kept (ties to the earliest); anchors are always kept."""
    best: dict[int, tuple[float, int]] = {}
    for i, frame in enumerate(frames):
        if frame.anchor:
            continue
        window = int(frame.time_s // settings.window_s)
        score = sharpness(frame.gray)
        if window not in best or score > best[window][0]:
            best[window] = (score, i)
    chosen = {i for _, i in best.values()}
    return [frame.anchor or i in chosen for i, frame in enumerate(frames)]


register_pixel_filter("dedupe", 1, dedupe_v1)
register_pixel_filter("sharpness", 1, sharpness_v1)

__all__ = [
    "PixelFrame",
    "apply_pixel_filters",
    "dedupe_v1",
    "hamming",
    "phash",
    "register_pixel_filter",
    "sharpness",
    "sharpness_v1",
]
