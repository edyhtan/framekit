"""L3b · Lens passes (DESIGN.md §3, §4): facts → annotations and phrasings, text only.

For each lens of the search space and each scene with a perception record: the lens's
prompts, framed by framekit's rules, over the record (never the video, never another lens's
output). The reply is checked against the lens spec (fields, vocabularies, beat indexes,
phrasing count); an unusable one is retried with its problems, then flagged. With
self-consistency, each run has its own seed, so the runs are distinct requests.
"""

from typing import Any

from framekit.extraction.prompts import LENS_PROMPT, lens_request
from framekit.extraction.runner import CallRecord
from framekit.extraction.schemas.common import digest
from framekit.extraction.schemas.lens import LensSpec, annotation_rows
from framekit.extraction.stages.context import (
    Attempts,
    SceneRow,
    StageContext,
    reply_json,
    request_for,
    retry_prompt,
)


def lens_context(scene: SceneRow, record: dict[str, Any], synopsis: str | None) -> dict[str, Any]:
    """What a lens pass reads: the scene's span, the synopsis and the facts, with each beat
    numbered (beats are annotated by that index)."""
    facts = dict(record)
    facts["beats"] = [{"index": i, **beat} for i, beat in enumerate(record.get("beats") or ())]
    return {
        "scene": {"start": scene.start, "end": scene.end},
        "synopsis": synopsis,
        "facts": facts,
    }


async def lens_pass(
    ctx: StageContext,
    scene: SceneRow,
    record: dict[str, Any],
    synopsis: str | None,
    lens: LensSpec,
    run_index: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """(output, run record). The output holds `annotation` (the checked reply, or None),
    `status` and `problems`."""
    run = ctx.run
    settings = run.plan.lens_passes
    binding = run.roles["L3b"]
    targets = lens.targets_in(settings.targets)
    count = settings.phrasings.count
    seed = settings.self_consistency.seeds[run_index]
    max_attempts = run.plan.quality_control.invalid_output.max_attempts
    beats = len(record.get("beats") or ())
    style = "" if run.style_examples else lens.prompts.phrasing_style
    context = lens_context(scene, record, synopsis)
    inputs = {
        "version": LENS_PROMPT,
        "lens": lens.digest,
        "context": digest(context),
        "targets": list(targets),
        "phrasings": count,
        "style_examples": list(run.style_examples),
        "style": style,
        "entities": list(ctx.entities),
        "model": {**binding.identity(), "seed": seed},
        "run_index": run_index,
        "max_attempts": max_attempts,
    }

    async def compute(_: str) -> tuple[dict[str, Any], dict[str, Any]]:
        system, prompt = lens_request(
            name=lens.name,
            description=lens.description,
            interpretation=lens.prompts.interpretation,
            phrasing_prompt=lens.prompts.phrasing,
            style_examples=run.style_examples,
            style=style,
            phrasings=count,
            beats="beat" in targets and beats > 0,
            context=context,
        )
        schema = lens.output_schema(targets=targets, phrasings=count, entities=ctx.entities)
        if "beat" in targets and beats == 0:
            schema["properties"]["beats"]["maxItems"] = 0
        attempts = Attempts()
        annotation, problems, text = None, [], prompt
        for _ in range(max_attempts):
            request = request_for(
                binding, system=system, prompt=text, schema=schema, stage="L3b", seed=seed
            )
            result = await ctx.ex.call("L3b", binding.provider.name, request)
            attempts.calls.append(CallRecord.of(result))
            data, problems = reply_json(result)
            if data is not None:
                problems = lens.check_output(
                    data, targets=targets, phrasings=count, entities=ctx.entities, beats=beats
                )
            if not problems:
                annotation = data
                break
            text = retry_prompt(prompt, problems, result.text)
        status = "ok" if annotation is not None else "flagged"
        output = {
            "annotation": annotation,
            "status": status,
            "problems": problems if annotation is None else [],
        }
        fields = attempts.fields(
            unit_id=scene.unit_id,
            target_id=scene.scene_id,
            status=status,
            lens=lens.key,
            run_index=run_index,
        )
        return output, fields

    return await ctx.ex.task("L3b", inputs, compute)


def lens_rows(
    scene: SceneRow,
    lens: LensSpec,
    run_index: int,
    annotation: dict[str, Any],
    run_id: str,
    weight: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """A checked annotation as (`annotations` rows, `phrasings` rows)."""
    base = {
        "unit_id": scene.unit_id,
        "scene_id": scene.scene_id,
        "lens": lens.name,
        "lens_version": lens.version,
        "run_index": run_index,
        "run_id": run_id,
        "weight": weight,
    }
    rows: list[dict[str, Any]] = []
    if "scene" in annotation:
        rows += [
            {**base, "target_type": "scene", "target_id": scene.scene_id, **row}
            for row in annotation_rows(lens.fields, annotation["scene"])
        ]
    for beat in annotation.get("beats") or ():
        beat_id = f"{scene.scene_id}_b{int(beat['index']):03d}"
        fields = {k: v for k, v in beat.items() if k != "index"}
        rows += [
            {**base, "target_type": "beat", "target_id": beat_id, **row}
            for row in annotation_rows(lens.fields, fields)
        ]
    phrasings = [
        {
            "phrasing_id": f"{scene.scene_id}_{lens.name}_r{run_index}_p{k:02d}",
            "target_type": "scene",
            "target_id": scene.scene_id,
            "unit_id": scene.unit_id,
            "scene_id": scene.scene_id,
            "lens": lens.name,
            "lens_version": lens.version,
            "run_index": run_index,
            "text": text.strip(),
            "source": "generated",
            "run_id": run_id,
        }
        for k, text in enumerate(annotation.get("phrasings") or ())
    ]
    return rows, phrasings


def agreement(lens: LensSpec, annotations: list[dict[str, Any]]) -> float | None:
    """How much the self-consistency runs agree on one scene, 0–1: per field, the mean
    Jaccard similarity of the label sets (label and labels fields) or equality (other
    fields) over every pair of runs; then the mean over fields. None with fewer than two
    runs."""
    if len(annotations) < 2:
        return None
    scores = []
    for name, field in lens.fields.items():
        values = [(a.get("scene") or {}).get(name) for a in annotations]
        pair_scores = []
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                a, b = values[i], values[j]
                if field.type in ("label", "labels"):
                    sa = set(a if isinstance(a, list) else [a] if a is not None else [])
                    sb = set(b if isinstance(b, list) else [b] if b is not None else [])
                    union = sa | sb
                    pair_scores.append(len(sa & sb) / len(union) if union else 1.0)
                else:
                    pair_scores.append(1.0 if a == b else 0.0)
        scores.append(sum(pair_scores) / len(pair_scores))
    return round(sum(scores) / len(scores), 4) if scores else None


__all__ = ["agreement", "lens_context", "lens_pass", "lens_rows"]
