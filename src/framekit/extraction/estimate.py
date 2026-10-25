"""The cost estimate shown before a run (`dry_run=True`).

An upper-bound estimate from the probed units, the plan and the catalog prices: scenes are
counted at the target length, outputs at their full `max_tokens`, images at the number of
28 x 28 px patches they cover (the way Claude counts image tokens; other providers differ),
retries and replays from the response cache not counted. Prices are per the model
catalog; nothing here calls a model.
"""

import math
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from framekit.extraction.media.ffmpeg import MediaInfo, copy_size, scaled_size
from framekit.extraction.resolve import ResolvedRun
from framekit.extraction.segmenting import processable_spans
from framekit.models.catalog import estimate_cost
from framekit.vision import ResolvedProfile, frame_cap

PATCH_PX = 28
PROMPT_TOKENS = {"L2": 1500, "L3a": 1500, "L3b": 1200}
TOKENS_PER_TEXT = 40  # an embedded summary or phrasing


@dataclass
class StageEstimate:
    calls: int = 0
    images: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: Decimal = Decimal(0)


@dataclass
class CostEstimate:
    units: int
    seconds: float
    scenes: int
    stages: dict[str, StageEstimate] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def total_usd(self) -> Decimal:
        return sum((s.cost_usd for s in self.stages.values()), Decimal(0))

    def to_mapping(self) -> dict[str, Any]:
        return {
            "units": self.units,
            "seconds": round(self.seconds, 1),
            "scenes": self.scenes,
            "total_usd": str(self.total_usd.quantize(Decimal("0.0001"))),
            "stages": {
                name: {
                    "calls": s.calls,
                    "images": s.images,
                    "input_tokens": s.input_tokens,
                    "output_tokens": s.output_tokens,
                    "cost_usd": str(s.cost_usd.quantize(Decimal("0.0001"))),
                }
                for name, s in self.stages.items()
            },
            "notes": list(self.notes),
        }

    def __str__(self) -> str:
        lines = [
            f"Estimate (upper bound): {self.units} units, {self.seconds / 60:.1f} min, "
            f"~{self.scenes} scenes",
            f"{'stage':<8}{'calls':>8}{'images':>8}{'in tokens':>12}{'out tokens':>12}{'USD':>10}",
        ]
        for name, s in self.stages.items():
            lines.append(
                f"{name:<8}{s.calls:>8}{s.images:>8}{s.input_tokens:>12}{s.output_tokens:>12}"
                f"{s.cost_usd:>10.4f}"
            )
        lines.append(f"{'total':<48}{self.total_usd:>10.4f}")
        lines += [f"note: {n}" for n in self.notes]
        return "\n".join(lines)


def _image_tokens(width: int, height: int) -> int:
    return math.ceil(width / PATCH_PX) * math.ceil(height / PATCH_PX)


def _frames(profile: ResolvedProfile, seconds: float) -> int:
    cap = frame_cap(profile.profile)
    fps = profile.profile.frame_selection.candidates.params.get("fps")
    uncapped = math.ceil(seconds * fps) if fps else math.ceil(seconds)  # shot_based: ~1/s
    return min(cap, uncapped) if cap is not None else uncapped


def _images(profile: ResolvedProfile, frames: int, copy: tuple[int, int]) -> tuple[int, int]:
    """(images, image tokens) for that many frames of a copy of that size."""
    width, height = scaled_size(copy[0], copy[1], profile.profile.image.max_side_px)
    delivery = profile.profile.delivery
    if delivery.mode == "storyboard" and delivery.storyboard is not None:
        tiles = delivery.storyboard.rows * delivery.storyboard.cols
        images = math.ceil(frames / tiles)
        size = (width * delivery.storyboard.cols, height * delivery.storyboard.rows)
        return images, images * _image_tokens(*size)
    return frames, frames * _image_tokens(width, height)


def estimate(run: ResolvedRun, probes: dict[str, MediaInfo]) -> CostEstimate:
    """The estimate for the units that probed (their facts in `probes`)."""
    plan = run.plan
    target = plan.segmentation.scene_length_s.target
    result = CostEstimate(units=len(probes), seconds=0.0, scenes=0)
    stage_names = [r for r in ("L2", "L3a", "L3b") if r in run.roles]
    if "embed_text" in run.roles:
        stage_names.append("embed")
    result.stages = {name: StageEstimate() for name in stage_names}
    lens_runs = len(run.lenses) * plan.lens_passes.self_consistency.runs
    for unit in run.units:
        info = probes.get(unit.unit_id)
        if info is None:
            continue
        spans = processable_spans(info.duration_s, unit.excluded_ranges)
        seconds = sum(b - a for a, b in spans)
        result.seconds += seconds
        scenes = sum(max(1, math.ceil((b - a) / target)) for a, b in spans)
        result.scenes += scenes
        copy = copy_size(info, plan.normalize.height_px)
        if "L2" in result.stages:
            images, tokens = _images(run.profile("L2"), _frames(run.profile("L2"), seconds), copy)
            stage = result.stages["L2"]
            stage.calls += 1
            stage.images += images
            stage.input_tokens += PROMPT_TOKENS["L2"] + tokens
            stage.output_tokens += run.roles["L2"].max_tokens
        stage = result.stages["L3a"]
        frames = _frames(run.profile("L3a"), min(target, seconds))
        images, tokens = _images(run.profile("L3a"), frames, copy)
        stage.calls += scenes
        stage.images += scenes * images
        stage.input_tokens += scenes * (PROMPT_TOKENS["L3a"] + tokens)
        stage.output_tokens += scenes * run.roles["L3a"].max_tokens
        stage = result.stages["L3b"]
        stage.calls += scenes * lens_runs
        stage.input_tokens += (
            scenes * lens_runs * (PROMPT_TOKENS["L3b"] + run.roles["L3a"].max_tokens)
        )
        stage.output_tokens += scenes * lens_runs * run.roles["L3b"].max_tokens
        if "embed" in result.stages:
            texts = scenes * (1 + lens_runs * plan.lens_passes.phrasings.count)
            batch = plan.embedding.text.batch_size if plan.embedding.text else 64
            stage = result.stages["embed"]
            stage.calls += math.ceil(texts / batch)
            stage.input_tokens += texts * TOKENS_PER_TEXT
    for name, stage in result.stages.items():
        role = "embed_text" if name == "embed" else name
        spec = run.roles[role].spec
        stage.cost_usd = estimate_cost(spec, stage.input_tokens, stage.output_tokens)
    missing = [u.unit_id for u in run.units if u.unit_id not in probes]
    if missing:
        result.notes.append(f"not estimated (couldn't be read): {', '.join(missing)}")
    if plan.budget.max_cost_usd is not None and result.total_usd > Decimal(
        str(plan.budget.max_cost_usd)
    ):
        result.notes.append(
            f"the estimate exceeds budget.max_cost_usd (${plan.budget.max_cost_usd:g}); "
            f"on_exceed: {plan.budget.on_exceed}"
        )
    return result


__all__ = ["CostEstimate", "StageEstimate", "estimate"]
