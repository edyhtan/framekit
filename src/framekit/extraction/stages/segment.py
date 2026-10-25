"""L2 · Scene segmentation (DESIGN.md §3): a synopsis and scenes on the unit timeline.

`segmentation.method: model`: the L2 model sees frames of the whole unit (its vision
profile; excluded ranges left out) and proposes a synopsis and scenes, which are repaired
into a clean cover of the processable parts (snapped to cuts, too-short scenes merged,
too-long ones split; each repair noted). An unusable reply after the allowed attempts falls
back to shot groups, flagged. `shot_groups`: scenes from consecutive shots, no model.
"""

import dataclasses
from typing import Any

from framekit.extraction.media.frames import Delivered, frames_for_span
from framekit.extraction.prompts import SEGMENT_PROMPT, segment_request
from framekit.extraction.records import segmentation_schema, shape_problems
from framekit.extraction.runner import CallRecord
from framekit.extraction.schemas.common import digest, plain
from framekit.extraction.segmenting import group_shots, processable_spans, repair_scenes
from framekit.extraction.stages.context import (
    Attempts,
    StageContext,
    UnitMedia,
    anchor_times,
    reply_json,
    request_for,
    retry_prompt,
    signals_for,
)
from framekit.vision import ResolvedProfile, plan_frames


def frame_rows(
    delivered: Delivered,
    profile: ResolvedProfile,
    *,
    stage: str,
    unit_id: str,
    scene_id: str | None,
) -> list[dict[str, Any]]:
    """The frames actually sent, for the `frames_sent` table (no pixels, only times, sizes
    and the sent image's digest)."""
    fmt = profile.profile.image.format or "jpeg"
    rows = []
    for image_index, sent in enumerate(delivered.images):
        for tile_index, frame in enumerate(sent.frames):
            rows.append(
                {
                    "unit_id": unit_id,
                    "scene_id": scene_id,
                    "stage": stage,
                    "image_index": image_index,
                    "tile_index": tile_index,
                    "planned_s": frame.planned_s,
                    "frame_s": frame.frame_s,
                    "frame_index": frame.index,
                    "anchor": frame.anchor,
                    "width": frame.size[0],
                    "height": frame.size[1],
                    "image_width": sent.size[0],
                    "image_height": sent.size[1],
                    "format": fmt,
                    "image_sha256": sent.sha256,
                    "profile": profile.name,
                    "profile_digest": profile.digest,
                }
            )
    return rows


def grid_of(profile: ResolvedProfile) -> tuple[int, int] | None:
    delivery = profile.profile.delivery
    if delivery.mode != "storyboard":
        return None
    grid = delivery.storyboard
    return (grid.rows, grid.cols) if grid is not None else (2, 2)


async def segment(
    ctx: StageContext, media: UnitMedia, signals: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """(output, run record or None when no model ran). The output holds `method`,
    `synopsis`, `scenes` ([start, end, summary]), `notes`, `frames` and `status`."""
    settings = ctx.run.plan.segmentation
    length = settings.scene_length_s
    spans = processable_spans(media.copy.duration_s, media.unit.excluded_ranges)
    cuts = signals.get("shots", [])

    def shot_groups() -> list[list[Any]]:
        return [[a, b, ""] for span in spans for a, b in group_shots(span, cuts, length)]

    if not spans:
        return {
            "method": settings.method,
            "synopsis": None,
            "scenes": [],
            "frames": [],
            "notes": ["nothing to process: the excluded ranges cover the unit"],
            "status": "ok",
        }, None
    if settings.method == "shot_groups":
        return {
            "method": "shot_groups",
            "synopsis": None,
            "scenes": shot_groups(),
            "frames": [],
            "notes": [],
            "status": "ok",
        }, None

    binding = ctx.run.roles["L2"]
    profile = ctx.run.profile("L2")
    max_attempts = ctx.run.plan.quality_control.invalid_output.max_attempts
    inputs = {
        "version": SEGMENT_PROMPT,
        "copy": media.copy_fingerprint,
        "signals": digest({k: signals.get(k) for k in ("shots", "onsets", "motion")}),
        "spans": spans,
        "profile": profile.digest,
        "segmentation": plain(settings),
        "model": binding.identity(),
        "entities": list(ctx.entities),
        "hints": list(ctx.hints),
        "max_attempts": max_attempts,
    }

    async def compute(_: str) -> tuple[dict[str, Any], dict[str, Any]]:
        covering = (spans[0][0], spans[-1][1])
        planned = plan_frames(profile, covering, signals_for(signals))
        inside = tuple(t for t in planned.timestamps if any(a <= t < b for a, b in spans))
        planned = dataclasses.replace(planned, timestamps=inside)
        anchors = anchor_times(profile, covering, signals_for(signals))
        delivered = await frames_for_span(
            ctx.ffmpeg, media.copy_path, media.copy, profile, planned, anchors, ctx.ex.media_slots
        )
        system, prompt = segment_request(
            duration_s=media.copy.duration_s,
            spans=spans,
            cuts=cuts,
            images=[[f.frame_s for f in sent.frames] for sent in delivered.images],
            grid=grid_of(profile),
            entities=ctx.entities,
            hints=ctx.hints,
            scene_length=plain(length),
        )
        schema = segmentation_schema()
        attempts = Attempts()
        reply, problems, text = None, [], prompt
        for _ in range(max_attempts):
            request = request_for(
                binding,
                system=system,
                prompt=text,
                schema=schema,
                stage="L2",
                images=tuple(sent.image for sent in delivered.images),
            )
            result = await ctx.ex.call("L2", binding.provider.name, request)
            attempts.calls.append(CallRecord.of(result))
            data, problems = reply_json(result)
            if data is not None:
                problems = shape_problems(data, schema)
                if not problems and not data["scenes"]:
                    problems = ["no scenes"]
            if not problems:
                reply = data
                break
            text = retry_prompt(prompt, problems, result.text)
        notes = list(delivered.notes)
        if reply is None:
            scenes = shot_groups()
            synopsis, status = None, "flagged"
            notes += [
                f"the L2 reply was unusable after {len(attempts.calls)} attempts: shot groups used",
                *problems,
            ]
        else:
            proposed = [(s["start"], s["end"], s["summary"]) for s in reply["scenes"]]
            repaired, repair_notes = repair_scenes(
                proposed,
                spans,
                cuts,
                length,
                snap_to_cuts=settings.snap_to == "shot_boundaries",
                tolerance_s=settings.snap_tolerance_s,
            )
            scenes = [list(s) for s in repaired]
            synopsis, status = reply["synopsis"], "ok"
            notes += repair_notes
        output = {
            "method": "model",
            "synopsis": synopsis,
            "scenes": scenes,
            "frames": frame_rows(
                delivered, profile, stage="L2", unit_id=media.unit_id, scene_id=None
            ),
            "notes": notes,
            "status": status,
            "problems": problems if reply is None else [],
        }
        return output, attempts.fields(
            unit_id=media.unit_id, target_id=media.unit_id, status=status
        )

    return await ctx.ex.task("L2", inputs, compute)


__all__ = ["frame_rows", "grid_of", "segment"]
