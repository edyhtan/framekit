"""framekit's own prompts: L2 segmentation, L3a perception, and the frame around each
lens's prompts in L3b. Provider-neutral text; they never name a lens or corpus concept
(lens prompts come from the lens spec, names and hints from the corpus profile).

A change to any text here is a new prompt version (the constants below), which is part of
the input hash of the stage that sends it.
"""

import json
from collections.abc import Mapping, Sequence
from typing import Any

SEGMENT_PROMPT = "segment@1"
PERCEIVE_PROMPT = "perceive@1"
LENS_PROMPT = "lens_pass@1"


def _times(values: Sequence[float], limit: int = 400) -> str:
    shown = ", ".join(f"{t:.2f}" for t in values[:limit])
    more = f" (and {len(values) - limit} more)" if len(values) > limit else ""
    return (shown + more) or "none detected"


def image_legend(images: Sequence[Sequence[float]], grid: tuple[int, int] | None) -> str:
    """Which times each attached image shows (in order)."""
    lines = []
    for k, times in enumerate(images, 1):
        if grid is None:
            lines.append(f"Image {k}: the frame at {times[0]:.3f} s")
        else:
            rows, cols = grid
            lines.append(
                f"Image {k}: a grid of {rows} x {cols} frames, left to right then top to "
                f"bottom, at {', '.join(f'{t:.3f}' for t in times)} s"
                + (" (the remaining tiles are empty)" if len(times) < rows * cols else "")
            )
    return "\n".join(lines)


# --- L2: segmentation -------------------------------------------------------------------


SEGMENT_SYSTEM = """You split a video into scenes. You see frames sampled from the whole video, in time order; each frame's time on the video's timeline is given in seconds.

A scene is a continuous stretch of action in one place or situation. Return:
- a synopsis: a few sentences on what happens in the whole video, in order, describing only what is shown and heard;
- the scenes, in order, each with its start and end time in seconds and a one-sentence summary.

Rules:
- Scenes cover each listed part of the video exactly, without gaps or overlaps.
- Scene lengths are between {min_s:g} and {max_s:g} seconds, about {target_s:g} seconds where the content allows.
- Prefer boundaries at the listed shot cuts.
- Name people or characters only as listed below; describe anyone else without a name."""


def segment_request(
    *,
    duration_s: float,
    spans: Sequence[tuple[float, float]],
    cuts: Sequence[float],
    images: Sequence[Sequence[float]],
    grid: tuple[int, int] | None,
    entities: Sequence[str],
    hints: Sequence[str],
    scene_length: Mapping[str, float],
) -> tuple[str, str]:
    """(system, prompt) for one L2 call."""
    system = SEGMENT_SYSTEM.format(
        min_s=scene_length["min"], max_s=scene_length["max"], target_s=scene_length["target"]
    )
    parts = [
        f"Video length: {duration_s:.2f} s.",
        "Parts to cover: " + ", ".join(f"[{a:.2f}, {b:.2f}]" for a, b in spans) + ".",
        f"Shot cuts (s): {_times(cuts)}.",
        "Names to use: " + (", ".join(entities) if entities else "none given") + ".",
    ]
    if hints:
        parts.append("About this collection: " + "; ".join(hints) + ".")
    parts += ["", image_legend(images, grid), "", "Reply with the JSON object only."]
    return system, "\n".join(parts)


# --- L3a: perception --------------------------------------------------------------------


PERCEIVE_SYSTEM = """You record what can be observed in one scene of a video: what is visible and audible, never interpretation. Write actions and expressions as they appear (a grin, a scream, a door slammed), not inferred motives, intentions or judgments.

You see frames from the scene in time order, each labeled with its time on the video's timeline in seconds, and signals measured from the video.

The record:
- summary: one or two sentences on what happens;
- setting: where it happens;
- entities: who appears, using only the listed names; anyone else is "unknown" (or "unknown: a short description");
- props: notable objects;
- beats: short units of action in order, each with its start and end time inside the scene, who acts, the action, and what is heard at that moment (null if nothing notable). Beats last between {beat_min:g} and {beat_max:g} seconds.{modules}"""

MODULE_TEXT = {
    "camera": "camera: each shot's time span, its size (e.g. close-up, wide) and movement (e.g. static, pan, zoom)",
    "timing": "timing: holds and pauses (spans with a short description) and the overall rhythm",
    "sound": "sound: music cues (spans with a description), silences (spans), and moments where a sound lines up with an action (time and description)",
}


def perceive_request(
    *,
    span: tuple[float, float],
    modules: Sequence[str],
    beats: tuple[float, float],
    entities: Sequence[str],
    hints: Sequence[str],
    synopsis: str | None,
    neighbors: Sequence[str],
    signals: Mapping[str, Any],
    images: Sequence[Sequence[float]],
    grid: tuple[int, int] | None,
) -> tuple[str, str]:
    """(system, prompt) for one L3a call."""
    extra = [MODULE_TEXT[m] for m in ("camera", "timing", "sound") if m in modules]
    module_text = "\n\nAlso record:\n- " + "\n- ".join(extra) if extra else ""
    system = PERCEIVE_SYSTEM.format(beat_min=beats[0], beat_max=beats[1], modules=module_text)
    start, end = span
    parts = [f"Scene: from {start:.3f} s to {end:.3f} s."]
    parts.append("Names to use: " + (", ".join(entities) if entities else "none given") + ".")
    if hints:
        parts.append("About this collection: " + "; ".join(hints) + ".")
    if synopsis:
        parts.append(f"Synopsis of the whole video: {synopsis}")
    for label, summary in zip(("Previous scene", "Next scene"), neighbors, strict=False):
        if summary:
            parts.append(f"{label}: {summary}")
    if signals:
        parts.append("Signals (times in seconds): " + json.dumps(signals, separators=(", ", ": ")))
    parts += ["", image_legend(images, grid), "", "Reply with the JSON object only."]
    return system, "\n".join(parts)


# --- L3b: lens passes -------------------------------------------------------------------


LENS_SYSTEM = """You annotate one scene of a video for the lens "{name}"{description}. You don't see the video: you read a factual record of the scene, written beforehand.

{interpretation}

Rules:
- Use only the labels the schema lists for each field; never invent a label.
- Base every annotation on the record alone.{beats}{phrasings}"""


def lens_request(
    *,
    name: str,
    description: str,
    interpretation: str,
    phrasing_prompt: str,
    style_examples: Sequence[str],
    style: str,
    phrasings: int,
    beats: bool,
    context: Mapping[str, Any],
) -> tuple[str, str]:
    """(system, prompt) for one L3b call."""
    beat_text = (
        "\n- Under `beats`, annotate each beat of the record by its index, with the same fields."
        if beats
        else ""
    )
    phrasing_text = ""
    if phrasings:
        phrasing_text = (
            f"\n- Under `phrasings`, write {phrasings} different searches a person would type to "
            "find this moment: the situation they're thinking of, not a description of the "
            "footage."
        )
        if phrasing_prompt:
            phrasing_text += f" {phrasing_prompt.strip()}"
        if style_examples:
            phrasing_text += (
                " Write them in the style of these searches: "
                + "; ".join(f'"{e}"' for e in style_examples)
                + "."
            )
        elif style:
            phrasing_text += f" Style: {style.strip()}."
    system = LENS_SYSTEM.format(
        name=name,
        description=f" ({description.strip()})" if description.strip() else "",
        interpretation=interpretation.strip(),
        beats=beat_text,
        phrasings=phrasing_text,
    )
    prompt = (
        json.dumps(context, ensure_ascii=False, indent=1) + "\n\nReply with the JSON object only."
    )
    return system, prompt


__all__ = [
    "LENS_PROMPT",
    "PERCEIVE_PROMPT",
    "SEGMENT_PROMPT",
    "image_legend",
    "lens_request",
    "perceive_request",
    "segment_request",
]
