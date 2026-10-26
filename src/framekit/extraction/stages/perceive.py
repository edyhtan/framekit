"""L3a · Perception (DESIGN.md §3): observable facts per scene, lens-free and shared.

Per scene: frames planned by L3a's frame profile (on the scene span, with the L1 signals),
read, filtered and delivered; the scene's signals, the synopsis, neighbouring summaries and
the corpus profile's names and hints as context. The reply must match the record schema
for the run's modules; an unusable reply is retried with its problems (as a new request),
and after the allowed attempts the scene is flagged and keeps no record. The library
checks of L4 (times, order, names, modules) run on every record.
"""

from typing import Any

from framekit.extraction.media.frames import frames_for_span
from framekit.extraction.prompts import PERCEIVE_PROMPT, perceive_request
from framekit.extraction.records import (
    RECORD_SCHEMA,
    check_record,
    record_schema,
    shape_problems,
)
from framekit.extraction.runner import CallRecord
from framekit.extraction.schemas.common import digest
from framekit.extraction.stages.context import (
    Attempts,
    SceneRow,
    StageContext,
    UnitMedia,
    anchor_times,
    reply_json,
    request_for,
    retry_prompt,
    signals_for,
)
from framekit.extraction.stages.segment import frame_rows, grid_of
from framekit.frames import plan_frames

MAX_ONSETS_IN_PROMPT = 60
MOTION_PEAKS_IN_PROMPT = 5


def scene_signals(
    signals: dict[str, Any], span: tuple[float, float], wanted: tuple[str, ...]
) -> dict[str, Any]:
    """The L1 signals inside one scene, short enough for a prompt."""
    start, end = span
    found: dict[str, Any] = {}
    if "shots" in wanted:
        found["shot_cuts"] = [t for t in signals.get("shots", ()) if start < t < end]
    if "audio" in wanted:
        onsets = [t for t in signals.get("onsets", ()) if start <= t < end]
        found["audio_onsets"] = onsets[:MAX_ONSETS_IN_PROMPT]
    if "motion" in wanted:
        motion = [(t, v) for t, v in signals.get("motion", ()) if start <= t < end]
        peaks = sorted(motion, key=lambda m: (-m[1], m[0]))[:MOTION_PEAKS_IN_PROMPT]
        found["motion_peaks"] = [[t, round(v, 4)] for t, v in sorted(peaks)]
    if "speech" in wanted and "speech" in signals:
        found["speech"] = [s for s in signals["speech"] if s["start"] < end and s["end"] > start]
    return found


def speech_module(signals: dict[str, Any], span: tuple[float, float]) -> list[dict[str, Any]]:
    """The `speech` module comes from L1, never from the model."""
    start, end = span
    return [
        {"t": [s["start"], s["end"]], "text": s["text"]}
        for s in signals.get("speech", ())
        if s["start"] < end and s["end"] > start
    ]


async def perceive(
    ctx: StageContext,
    media: UnitMedia,
    signals: dict[str, Any],
    scene: SceneRow,
    synopsis: str | None,
    neighbors: tuple[str, ...],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """(output, run record). The output holds `record` (or None), `status`, `problems`,
    `checks` (the L4 library checks) and `frames`."""
    run = ctx.run
    settings = run.plan.interpretation.perception
    binding = run.roles["L3a"]
    profile = run.profile("L3a")
    max_attempts = run.plan.policy.quality_control.invalid_output.max_attempts
    context_signals = scene_signals(signals, scene.span, settings.context.signals)
    use_synopsis = synopsis if settings.context.synopsis else None
    inputs = {
        "version": PERCEIVE_PROMPT,
        "record_schema": RECORD_SCHEMA,
        "copy": media.copy_fingerprint,
        "scene": [scene.start, scene.end],
        "signals": digest(context_signals),
        "speech": digest(speech_module(signals, scene.span)) if "speech" in run.modules else None,
        "synopsis": use_synopsis,
        "neighbors": list(neighbors),
        "profile": profile.digest,
        "modules": list(run.modules),
        "beats": list(run.beats),
        "entities": list(ctx.entities),
        "unknown": settings.entities.unknown,
        "hints": list(ctx.hints),
        "model": binding.identity(),
        "max_attempts": max_attempts,
    }

    async def compute(_: str) -> tuple[dict[str, Any], dict[str, Any]]:
        planned = plan_frames(profile, scene.span, signals_for(signals))
        anchors = anchor_times(profile, scene.span, signals_for(signals))
        delivered = await frames_for_span(
            ctx.ffmpeg, media.copy_path, media.copy, profile, planned, anchors, ctx.ex.media_slots
        )
        system, prompt = perceive_request(
            span=scene.span,
            modules=run.modules,
            beats=run.beats,
            entities=ctx.entities,
            hints=ctx.hints,
            synopsis=use_synopsis,
            neighbors=neighbors,
            signals=context_signals,
            images=[[f.frame_s for f in sent.frames] for sent in delivered.images],
            grid=grid_of(profile),
        )
        schema = record_schema(run.modules)
        attempts = Attempts()
        record, problems, text = None, [], prompt
        for _ in range(max_attempts):
            request = request_for(
                binding,
                system=system,
                prompt=text,
                schema=schema,
                stage="L3a",
                images=tuple(sent.image for sent in delivered.images),
            )
            result = await ctx.ex.call("L3a", binding.provider.name, request)
            attempts.calls.append(CallRecord.of(result))
            data, problems = reply_json(result)
            if data is not None:
                problems = shape_problems(data, schema)
            if not problems:
                record = data
                break
            text = retry_prompt(prompt, problems, result.text)
        status = "ok" if record is not None else "flagged"
        checks: dict[str, list[str]] = {}
        if record is not None:
            if "speech" in run.modules:
                record["speech"] = speech_module(signals, scene.span)
            checks = check_record(
                record,
                span=scene.span,
                modules=run.modules,
                entities=ctx.entities,
                unknown=settings.entities.unknown,
            )
        output = {
            "record": record,
            "status": status,
            "problems": problems if record is None else [],
            "checks": checks,
            "frames": frame_rows(
                delivered, profile, stage="L3a", unit_id=media.unit_id, scene_id=scene.scene_id
            ),
            "notes": list(delivered.notes),
        }
        fields = attempts.fields(unit_id=media.unit_id, target_id=scene.scene_id, status=status)
        return output, fields

    return await ctx.ex.task("L3a", inputs, compute)


def beat_rows(scene: SceneRow, record: dict[str, Any] | None) -> list[dict[str, Any]]:
    """The record's beats as rows of the `beats` table."""
    rows = []
    for i, beat in enumerate((record or {}).get("beats") or ()):
        t = beat.get("t") or [None, None]
        rows.append(
            {
                "beat_id": f"{scene.scene_id}_b{i:03d}",
                "scene_id": scene.scene_id,
                "unit_id": scene.unit_id,
                "index": i,
                "start": float(t[0]),
                "end": float(t[1]),
                "who": [str(w) for w in beat.get("who") or ()],
                "action": str(beat.get("action") or ""),
                "audio": beat.get("audio"),
            }
        )
    return rows


__all__ = ["beat_rows", "perceive", "scene_signals", "speech_module"]
