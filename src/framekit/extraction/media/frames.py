"""Frames for a model: read the planned frames, run the pixel filters, prepare and deliver.

The steps after pixel-free planning (`framekit.vision.plan_frames`), in DESIGN.md §2.3's
order: decode each planned timestamp from the analysis copy at the profile's image size,
apply the pending pixel filters (anchors are never dropped), encode, and deliver as images
or storyboard grids. What was sent is returned for the `frames_sent` table.
"""

import asyncio
import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from framekit.contracts import ImagePart
from framekit.extraction.media.ffmpeg import FFmpeg, MediaInfo, frame_index, scaled_size
from framekit.vision import FramePlan, ResolvedProfile
from framekit.vision.pixels import PixelFrame, apply_pixel_filters

MEDIA_TYPES = {"jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}


@dataclass(frozen=True)
class Frame:
    planned_s: float  # the timestamp planning chose
    frame_s: float  # the time of the analysis copy's frame shown then
    index: int  # that frame's number in the copy
    size: tuple[int, int]  # (width, height) as prepared
    rgb: bytes
    anchor: bool

    def gray(self) -> np.ndarray:
        """Luma (BT.601 weights), for the pixel filters."""
        width, height = self.size
        pixels = np.frombuffer(self.rgb, np.uint8).reshape(height, width, 3).astype(np.float32)
        return (pixels @ np.array([0.299, 0.587, 0.114], np.float32)).round().astype(np.uint8)


@dataclass(frozen=True)
class SentImage:
    image: ImagePart
    frames: tuple[Frame, ...]  # in tile order for a storyboard, else one frame
    size: tuple[int, int]

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.image.data).hexdigest()


@dataclass(frozen=True)
class Delivered:
    images: tuple[SentImage, ...]
    frames: tuple[Frame, ...]  # every frame sent, in time order
    notes: tuple[str, ...]


async def read_frames(
    ffmpeg: FFmpeg,
    copy: Path,
    info: MediaInfo,
    timestamps: Sequence[float],
    anchors: set[float],
    max_side_px: int,
    limit: asyncio.Semaphore,
) -> tuple[list[Frame], list[str]]:
    """The copy's frames at the planned timestamps, at the profile's size. Two timestamps on
    the same frame of the copy are read once (a note says so)."""
    if not info.fps or not info.width or not info.height:
        raise ValueError("the analysis copy has no frame rate or size")
    size = scaled_size(info.width, info.height, max_side_px)
    wanted: dict[int, tuple[float, bool]] = {}
    for t in sorted(timestamps):
        index = frame_index(t, info.fps)
        if index in wanted:
            continue
        wanted[index] = (t, t in anchors)
    notes = []
    if len(wanted) < len(timestamps):
        notes.append(
            f"{len(timestamps) - len(wanted)} planned timestamps fell on frames already "
            f"chosen (copy at {info.fps:g} fps)"
        )
    last = max(0, round(info.duration_s * info.fps) - 1)

    async def one(index: int, planned: float, anchor: bool) -> Frame:
        index = min(index, last)
        async with limit:
            rgb = await ffmpeg.frame_rgb(copy, index, info.fps, size)
        return Frame(planned, round(index / info.fps, 6), index, size, rgb, anchor)

    frames = await asyncio.gather(*(one(i, t, a) for i, (t, a) in wanted.items()))
    return sorted(frames, key=lambda f: f.planned_s), notes


def filter_frames(plan: FramePlan, frames: Sequence[Frame]) -> tuple[list[Frame], list[str]]:
    """The plan's pending pixel filters, in order; anchors stay."""
    if not plan.pending_filters:
        return list(frames), []
    by_time = {f.planned_s: f for f in frames}
    pixel = [PixelFrame(f.planned_s, f.gray(), f.anchor) for f in frames]
    kept, notes = apply_pixel_filters(plan.pending_filters, pixel)
    return [by_time[p.time_s] for p in kept], notes


def _grid(frames: Sequence[Frame], rows: int, cols: int) -> tuple[bytes, tuple[int, int]]:
    """Frames tiled left to right, top to bottom; empty tiles stay black."""
    width, height = frames[0].size
    canvas = np.zeros((rows * height, cols * width, 3), np.uint8)
    for k, frame in enumerate(frames):
        r, c = divmod(k, cols)
        tile = np.frombuffer(frame.rgb, np.uint8).reshape(height, width, 3)
        canvas[r * height : (r + 1) * height, c * width : (c + 1) * width] = tile
    return canvas.tobytes(), (cols * width, rows * height)


async def deliver(
    ffmpeg: FFmpeg,
    profile: ResolvedProfile,
    frames: Sequence[Frame],
    limit: asyncio.Semaphore,
) -> list[SentImage]:
    """Encode frames as the profile says: one image per frame (`images`), or grids of
    rows x cols frames (`storyboard`)."""
    image = profile.profile.image
    delivery = profile.profile.delivery
    fmt = image.format or "jpeg"
    groups: list[tuple[list[Frame], bytes, tuple[int, int]]] = []
    if delivery.mode == "storyboard":
        grid = delivery.storyboard
        rows, cols = (grid.rows, grid.cols) if grid is not None else (2, 2)
        for start in range(0, len(frames), rows * cols):
            chunk = list(frames[start : start + rows * cols])
            rgb, size = _grid(chunk, rows, cols)
            groups.append((chunk, rgb, size))
    elif delivery.mode == "images":
        groups = [([f], f.rgb, f.size) for f in frames]
    else:
        raise ValueError(f"delivery {delivery.mode!r} sends no frames (native video isn't built)")

    async def encode(rgb: bytes, size: tuple[int, int]) -> bytes:
        async with limit:
            return await ffmpeg.encode_image(rgb, size, fmt, image.quality)

    data = await asyncio.gather(*(encode(rgb, size) for _, rgb, size in groups))
    media_type = MEDIA_TYPES[fmt]
    return [
        SentImage(ImagePart(blob, media_type), tuple(chunk), size)  # type: ignore[arg-type]
        for (chunk, _, size), blob in zip(groups, data, strict=True)
    ]


async def frames_for_span(
    ffmpeg: FFmpeg,
    copy: Path,
    info: MediaInfo,
    profile: ResolvedProfile,
    plan: FramePlan,
    anchors: set[float],
    limit: asyncio.Semaphore,
) -> Delivered:
    """Planned timestamps → frames read, filtered, prepared and delivered."""
    frames, notes = await read_frames(
        ffmpeg, copy, info, plan.timestamps, anchors, profile.profile.image.max_side_px, limit
    )
    kept, filter_notes = filter_frames(plan, frames)
    images = await deliver(ffmpeg, profile, kept, limit) if kept else []
    return Delivered(tuple(images), tuple(kept), (*plan.notes, *notes, *filter_notes))


__all__ = [
    "Delivered",
    "Frame",
    "SentImage",
    "deliver",
    "filter_frames",
    "frames_for_span",
    "read_frames",
]
