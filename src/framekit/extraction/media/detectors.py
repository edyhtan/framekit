"""L1 signal detectors (DESIGN.md §3 L1): deterministic, no AI model in the built-ins.

A detector is named, versioned code, like a vision strategy: the plan picks it by name
(`signals.shots.detector: ffmpeg_scene`, or `ffmpeg_scene@1`), and its key goes into L1's
input hash, so a changed algorithm gets a new version and old datasets stay reproducible.

Kinds, and what `run(settings, media)` returns (times in seconds on the unit timeline):
- `shots`: the cut times, sorted;
- `audio`: `{"onsets": [t, ...], "loudness": [(t, dBFS), ...]}`;
- `motion`: `[(t, value >= 0), ...]`;
- `speech`: `[{"start", "end", "text"}, ...]`;
- `audio_tags`: `[{"start", "end", "label", "score"}, ...]`.

Built in (version 1): `ffmpeg_scene` (shots), `spectral_flux` (audio) and
`frame_difference` (motion). There's no built-in speech model or audio tagger (both are
open choices): register your own detector (e.g. wrapping Whisper with VAD) and name it in
the plan.
"""

import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, get_args

import numpy as np

from framekit.extraction.media.ffmpeg import FFmpeg, MediaInfo
from framekit.vision.registry import split_reference

DetectorKind = Literal["shots", "audio", "motion", "speech", "audio_tags"]
DETECTOR_KINDS: tuple[str, ...] = get_args(DetectorKind)


@dataclass(frozen=True)
class MediaContext:
    """What a detector reads: the analysis copy, its facts, and the ffmpeg to decode it."""

    path: Path
    info: MediaInfo
    ffmpeg: FFmpeg


@dataclass(frozen=True)
class Detector:
    kind: DetectorKind
    name: str
    version: int
    run: Callable[[Any, MediaContext], Awaitable[Any]]
    description: str = ""

    @property
    def key(self) -> str:
        return f"{self.name}@{self.version}"


_DETECTORS: dict[tuple[str, str, int], Detector] = {}


def register_detector(detector: Detector) -> Detector:
    """Add a detector. A kind, name and version can be registered once."""
    if detector.kind not in DETECTOR_KINDS:
        raise ValueError(f"unknown detector kind {detector.kind!r} ({', '.join(DETECTOR_KINDS)})")
    split_reference(detector.name)  # the same naming rule as vision strategies
    if detector.version < 1:
        raise ValueError(f"detector versions start at 1: {detector.key}")
    slot = (detector.kind, detector.name, detector.version)
    if slot in _DETECTORS:
        raise ValueError(f"{detector.kind} detector {detector.key} is already registered")
    _DETECTORS[slot] = detector
    return detector


def get_detector(kind: str, reference: str) -> Detector:
    """`name` (the newest version) or `name@version`; LookupError says what's known."""
    name, version = split_reference(reference)
    versions = sorted(v for (k, n, v) in _DETECTORS if k == kind and n == name)
    if not versions:
        known = sorted({n for (k, n, _) in _DETECTORS if k == kind})
        raise LookupError(
            f"no {kind} detector {name!r} is registered (known: {', '.join(known) or 'none'})"
        )
    chosen = versions[-1] if version is None else version
    if chosen not in versions:
        raise LookupError(f"{kind} detector {name}@{chosen} is not registered")
    return _DETECTORS[(kind, name, chosen)]


def detectors(kind: str | None = None) -> list[Detector]:
    return [d for slot, d in sorted(_DETECTORS.items()) if kind is None or slot[0] == kind]


# --- built-ins, version 1 ---------------------------------------------------------------


async def ffmpeg_scene(settings: Any, media: MediaContext) -> list[float]:
    """Cuts where ffmpeg's scene score (on frames 160 px wide) exceeds `threshold`. Cuts
    closer than `min_shot_s` compete: the stronger one stays (ties: the earlier). Cuts
    within `min_shot_s` of either end of the unit are dropped."""
    scores = await media.ffmpeg.scene_scores(media.path)
    end = media.info.duration_s
    candidates = sorted(
        (
            (score, t)
            for t, score in scores
            if score > settings.threshold
            and settings.min_shot_s <= t <= end - settings.min_shot_s
            and t > 0
        ),
        key=lambda item: (-item[0], item[1]),
    )
    cuts: list[float] = []
    for _, t in candidates:
        if all(abs(t - kept) >= settings.min_shot_s for kept in cuts):
            cuts.append(t)
    return sorted(round(t, 3) for t in cuts)


AUDIO_RATE_HZ = 16000
FLUX_FRAME = 1024  # 64 ms
FLUX_HOP = 256  # 16 ms
ONSET_MIN_GAP_S = 0.05


async def spectral_flux(settings: Any, media: MediaContext) -> dict[str, list]:
    """Onsets: peaks of the spectral flux (positive change of the log-magnitude spectrum,
    Hann frames of 64 ms every 16 ms, mono at 16 kHz). A peak is the maximum within ±48 ms
    and above the mean of the surrounding 0.5 s plus `(1 - sensitivity) × 2` standard
    deviations of the whole flux; onsets are at least 50 ms apart. Loudness: RMS in dBFS
    per `loudness.window_s`."""
    if not media.info.has_audio:
        return {"onsets": [], "loudness": []}
    pcm = await media.ffmpeg.audio_samples(media.path, AUDIO_RATE_HZ)
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float64) / 32768.0
    return {
        "onsets": _onsets(samples, settings.onsets.sensitivity),
        "loudness": _loudness(samples, settings.loudness.window_s),
    }


FLUX_BLOCK = 2048  # frames per FFT block, so memory stays flat on long units


def spectral_flux_curve(samples: np.ndarray) -> np.ndarray:
    """The flux of every frame (0 for the first), computed block by block."""
    count = 1 + (samples.size - FLUX_FRAME) // FLUX_HOP
    window = np.hanning(FLUX_FRAME)
    parts = [np.zeros(1)]
    previous = None
    for start in range(0, count, FLUX_BLOCK):
        stop = min(count, start + FLUX_BLOCK)
        index = np.arange(FLUX_FRAME)[None, :] + FLUX_HOP * np.arange(start, stop)[:, None]
        spectrum = np.log1p(np.abs(np.fft.rfft(samples[index] * window, axis=1)) * 10)
        stacked = spectrum if previous is None else np.vstack([previous[None, :], spectrum])
        parts.append(np.maximum(np.diff(stacked, axis=0), 0).sum(axis=1))
        previous = spectrum[-1]
    return np.concatenate(parts)


def _sliding_max(values: np.ndarray, before: int, after: int) -> np.ndarray:
    """For each i, the max of values[i - before : i + after + 1] (the edges padded)."""
    padded = np.pad(values, (before, after), constant_values=-np.inf)
    return np.lib.stride_tricks.sliding_window_view(padded, before + after + 1).max(axis=1)


def _onsets(samples: np.ndarray, sensitivity: float) -> list[float]:
    if samples.size < FLUX_FRAME * 2:
        return []
    flux = spectral_flux_curve(samples)
    if flux.max() <= 1e-9:
        return []
    radius = 3  # ±48 ms
    context = max(1, round(0.25 * AUDIO_RATE_HZ / FLUX_HOP))  # ±0.25 s
    kernel = np.ones(2 * context + 1) / (2 * context + 1)
    local_mean = np.convolve(np.pad(flux, context, mode="edge"), kernel, mode="valid")
    threshold = np.maximum(local_mean + (1 - sensitivity) * 2 * flux.std(), 0.05 * flux.max())
    # A peak: the largest within ±radius and strictly larger than everything before it in
    # the window (on a plateau, only the first frame), above the threshold.
    earlier = _sliding_max(np.concatenate([[-np.inf], flux[:-1]]), radius - 1, 0)
    later = _sliding_max(flux, 0, radius)
    peaks = np.flatnonzero((flux > earlier) & (flux >= later) & (flux >= threshold))
    onsets: list[float] = []
    for i in peaks[peaks > 0]:
        t = (i * FLUX_HOP + FLUX_FRAME / 2) / AUDIO_RATE_HZ
        if not onsets or t - onsets[-1] >= ONSET_MIN_GAP_S:
            onsets.append(round(float(t), 3))
    return onsets


def _loudness(samples: np.ndarray, window_s: float) -> list[tuple[float, float]]:
    step = max(1, round(window_s * AUDIO_RATE_HZ))
    values = []
    for k in range(math.ceil(samples.size / step)):
        chunk = samples[k * step : (k + 1) * step]
        rms = float(np.sqrt(np.mean(chunk**2))) if chunk.size else 0.0
        db = 20 * math.log10(rms) if rms > 1e-10 else -200.0
        values.append((round((k * step + chunk.size / 2) / AUDIO_RATE_HZ, 3), round(db, 2)))
    return values


MOTION_WIDTH = 64
MOTION_SAMPLES_PER_VALUE = 4


async def frame_difference(settings: Any, media: MediaContext) -> list[tuple[float, float]]:
    """Mean absolute difference between consecutive frames (64 px wide, gray, 0–1 scale),
    sampled 4 times per value, one value per 1 / `rate_hz` seconds at the window's middle.
    A cut counts as motion too."""
    width = MOTION_WIDTH
    info = media.info
    height = max(2, round((info.height or 36) * width / (info.width or 64)))
    fps = settings.rate_hz * MOTION_SAMPLES_PER_VALUE
    raw, count = await media.ffmpeg.gray_frames(media.path, fps, (width, height))
    if count < 2:
        return []
    frames = np.frombuffer(raw[: count * width * height], np.uint8).reshape(count, height, width)
    diffs = np.abs(np.diff(frames.astype(np.float32), axis=0)).mean(axis=(1, 2)) / 255.0
    middles = (np.arange(diffs.size) + 1.0) / fps  # between frame k and k + 1
    window = 1.0 / settings.rate_hz
    values = []
    for k in range(math.ceil(info.duration_s / window)):
        inside = (middles >= k * window) & (middles < (k + 1) * window)
        if inside.any():
            t = min((k + 0.5) * window, info.duration_s)
            values.append((round(t, 3), round(float(diffs[inside].mean()), 6)))
    return values


register_detector(
    Detector(
        "shots", "ffmpeg_scene", 1, ffmpeg_scene, "ffmpeg's scene-change score over a threshold"
    )
)
register_detector(
    Detector("audio", "spectral_flux", 1, spectral_flux, "spectral-flux onsets and RMS loudness")
)
register_detector(
    Detector(
        "motion",
        "frame_difference",
        1,
        frame_difference,
        "mean absolute difference between consecutive frames",
    )
)

__all__ = [
    "DETECTOR_KINDS",
    "Detector",
    "DetectorKind",
    "MediaContext",
    "detectors",
    "get_detector",
    "register_detector",
]
