"""L1 · Deterministic signals (DESIGN.md §3): cuts, audio onsets and loudness, motion, and
speech or audio tags when a detector for them is registered. No AI model in the built-ins.
"""

import asyncio
import math
from typing import Any

from framekit.extraction.media.detectors import MediaContext, get_detector
from framekit.extraction.media.ffmpeg import MediaError
from framekit.extraction.schemas.common import plain
from framekit.extraction.stages.context import StageContext, UnitMedia


def _times(values: Any, what: str) -> list[float]:
    times = sorted({round(float(t), 3) for t in values if math.isfinite(float(t))})
    if any(t < 0 for t in times):
        raise MediaError(f"{what}: negative times")
    return times


def _pairs(values: Any, what: str) -> list[list[float]]:
    pairs = []
    for item in values:
        t, v = item
        pairs.append([round(float(t), 3), float(v)])
    return sorted(pairs)


def _segments(values: Any, keys: tuple[str, ...]) -> list[dict[str, Any]]:
    found = []
    for item in values:
        segment = {key: item[key] for key in keys}
        segment["start"], segment["end"] = float(segment["start"]), float(segment["end"])
        found.append(segment)
    return sorted(found, key=lambda s: (s["start"], s["end"]))


async def detect(ctx: StageContext, media: UnitMedia) -> tuple[dict[str, Any], dict[str, Any]]:
    """Every configured detector over the analysis copy; their outputs checked and sorted."""
    settings = ctx.run.plan.observation.signals
    chosen = {
        "shots": (get_detector("shots", settings.shots.detector), settings.shots),
        "audio": (get_detector("audio", settings.audio.onsets.method), settings.audio),
        "motion": (get_detector("motion", settings.motion.method), settings.motion),
    }
    if settings.speech is not None:
        chosen["speech"] = (get_detector("speech", settings.speech.model), settings.speech)
    if settings.audio.tagger is not None:
        chosen["audio_tags"] = (
            get_detector("audio_tags", settings.audio.tagger.model),
            settings.audio.tagger,
        )
    keys = {kind: detector.key for kind, (detector, _) in chosen.items()}
    inputs = {"copy": media.copy_fingerprint, "settings": plain(settings), "detectors": keys}
    context = MediaContext(media.copy_path, media.copy, ctx.ffmpeg)

    async def run(kind: str) -> tuple[str, Any]:
        detector, detector_settings = chosen[kind]
        async with ctx.ex.media_slots:
            return kind, await detector.run(detector_settings, context)

    async def compute(_: str) -> tuple[dict[str, Any], dict[str, Any]]:
        found = dict(await asyncio.gather(*(run(kind) for kind in chosen)))
        output: dict[str, Any] = {
            "shots": _times(found["shots"], "shots"),
            "onsets": _times(found["audio"].get("onsets", ()), "onsets"),
            "loudness": _pairs(found["audio"].get("loudness", ()), "loudness"),
            "motion": _pairs(found["motion"], "motion"),
        }
        if "speech" in found:
            output["speech"] = _segments(found["speech"], ("start", "end", "text"))
        if "audio_tags" in found:
            output["audio_tags"] = _segments(
                found["audio_tags"], ("start", "end", "label", "score")
            )
        return output, {"unit_id": media.unit_id, "target_id": media.unit_id, "detectors": keys}

    return await ctx.ex.task("L1", inputs, compute)


def signal_rows(unit_id: str, output: dict[str, Any], keys: dict[str, str]) -> list[dict[str, Any]]:
    """L1 output as rows of the long `signals` table."""
    rows: list[dict[str, Any]] = []

    def row(signal: str, t: float, **extra: Any) -> dict[str, Any]:
        return {
            "unit_id": unit_id,
            "signal": signal,
            "t": t,
            "end": None,
            "value": None,
            "label": None,
            "text": None,
            "detector": keys.get(_kind(signal)),
            **extra,
        }

    rows += [row("shot_cut", t) for t in output.get("shots", ())]
    rows += [row("audio_onset", t) for t in output.get("onsets", ())]
    rows += [row("loudness_dbfs", t, value=v) for t, v in output.get("loudness", ())]
    rows += [row("motion", t, value=v) for t, v in output.get("motion", ())]
    rows += [
        row("speech", s["start"], end=s["end"], text=s["text"]) for s in output.get("speech", ())
    ]
    rows += [
        row("audio_tag", s["start"], end=s["end"], label=s["label"], value=float(s["score"]))
        for s in output.get("audio_tags", ())
    ]
    return rows


def _kind(signal: str) -> str:
    return {
        "shot_cut": "shots",
        "audio_onset": "audio",
        "loudness_dbfs": "audio",
        "motion": "motion",
        "speech": "speech",
        "audio_tag": "audio_tags",
    }[signal]


__all__ = ["detect", "signal_rows"]
