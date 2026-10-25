"""What every stage gets: the resolved run, the execution state, ffmpeg and the work folder,
plus the per-unit facts stages hand to each other."""

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from framekit.contracts import ConversationRequest, ConversationResult
from framekit.extraction.media.ffmpeg import FFmpeg, MediaInfo
from framekit.extraction.records import parse_json_reply
from framekit.extraction.resolve import ResolvedRun, RoleBinding, ScopedUnit
from framekit.extraction.runner import CallRecord, Execution
from framekit.vision import ResolvedProfile, Signals
from framekit.vision.params import thaw
from framekit.vision.registry import get_strategy

QUICK_CHUNK = 1 << 20  # 1 MiB


@dataclass
class StageContext:
    run: ResolvedRun
    ex: Execution
    ffmpeg: FFmpeg
    work_dir: Path

    @property
    def entities(self) -> tuple[str, ...]:
        return self.run.corpus.profile.entities

    @property
    def hints(self) -> tuple[str, ...]:
        return self.run.corpus.profile.perception_hints

    @property
    def media_dir(self) -> Path:
        return self.work_dir / "media"


@dataclass(frozen=True)
class UnitMedia:
    unit: ScopedUnit
    source: MediaInfo
    source_fingerprint: str
    copy_path: Path
    copy: MediaInfo
    copy_fingerprint: str

    @property
    def unit_id(self) -> str:
        return self.unit.unit_id


@dataclass(frozen=True)
class SceneRow:
    scene_id: str
    unit_id: str
    index: int
    start: float
    end: float
    summary: str

    @property
    def span(self) -> tuple[float, float]:
        return (self.start, self.end)


@dataclass
class Attempts:
    """The calls of one task (attempts after invalid output included), for its run record."""

    calls: list[CallRecord] = field(default_factory=list)

    def fields(self, **extra: Any) -> dict[str, Any]:
        from decimal import Decimal

        last = self.calls[-1] if self.calls else None
        return {
            "attempts": len(self.calls),
            "model_id": last.model_id if last else None,
            "provider": last.provider if last else None,
            "adapter_version": last.adapter_version if last else None,
            "latency_ms": sum(c.latency_ms for c in self.calls),
            "input_tokens": sum(c.input_tokens for c in self.calls),
            "output_tokens": sum(c.output_tokens for c in self.calls),
            "reasoning_tokens": sum(c.reasoning_tokens or 0 for c in self.calls),
            "cost_usd": str(sum((Decimal(c.cost_usd) for c in self.calls), Decimal(0))),
            "cached": bool(self.calls) and all(c.cached for c in self.calls),
            **extra,
        }


def quick_fingerprint(path: Path) -> str:
    """A content fingerprint of a (possibly huge) file without reading all of it: its size
    and three 1 MiB samples (start, middle, end). Machine-independent, unlike a path."""
    size = path.stat().st_size
    digest = hashlib.sha256(str(size).encode())
    with path.open("rb") as handle:
        for offset in sorted({0, max(0, size // 2 - QUICK_CHUNK // 2), max(0, size - QUICK_CHUNK)}):
            handle.seek(offset)
            digest.update(handle.read(QUICK_CHUNK))
    return "q1:" + digest.hexdigest()


def signals_for(output: dict[str, Any]) -> Signals:
    """L1 output as the planner reads it."""
    return Signals(
        shot_boundaries=output.get("shots", ()),
        audio_onsets=output.get("onsets", ()),
        motion=[tuple(m) for m in output.get("motion", ())],
    )


def _ms(t: float) -> float:
    return math.floor(t * 1000 + 0.5) / 1000


def anchor_times(
    profile: ResolvedProfile, span: tuple[float, float], signals: Signals
) -> set[float]:
    """The anchor timestamps of a span, rounded like the planner's, so planned frames can be
    told apart: anchors are never dropped by pixel filters."""
    times: set[float] = set()
    start, end = _ms(span[0]), _ms(span[1])
    for ref in profile.profile.frame_selection.anchors:
        strategy = get_strategy("anchor", ref.name, ref.version)
        if strategy.run is None:
            continue
        settings = strategy.params.model_validate(thaw(ref.params))
        for t in strategy.run(settings, span, signals):
            ms = _ms(t)
            if start <= ms < end:
                times.add(ms)
    return times


def request_for(
    binding: RoleBinding,
    *,
    system: str,
    prompt: str,
    schema: dict[str, Any] | None,
    stage: str,
    images: tuple = (),
    seed: int | None = None,
) -> ConversationRequest:
    return ConversationRequest(
        prompt,
        max_tokens=binding.max_tokens,
        model_id=binding.model_id,
        reasoning=binding.reasoning,
        system=system,
        response_schema=schema,
        cache_class=stage,
        images=images,
        temperature=binding.temperature,
        seed=seed if seed is not None else binding.seed,
    )


def retry_prompt(prompt: str, problems: list[str], text: str) -> str:
    """The prompt again, with what was wrong with the last reply (so the retry is a new
    request, never a replay of the same answer from the response cache)."""
    listed = "\n".join(f"- {p}" for p in problems[:20])
    return (
        f"{prompt}\n\nYour previous reply couldn't be used:\n{listed}\n"
        f"Previous reply (start): {text[:300]!r}\nReply again with a corrected JSON object only."
    )


def reply_json(result: ConversationResult) -> tuple[Any, list[str]]:
    """The reply's JSON, or the problems that make it unusable."""
    if result.stop_reason == "length":
        return None, ["the reply was cut off at max_tokens"]
    try:
        return parse_json_reply(result.text), []
    except ValueError as exc:
        return None, [str(exc)]


__all__ = [
    "Attempts",
    "SceneRow",
    "StageContext",
    "UnitMedia",
    "anchor_times",
    "quick_fingerprint",
    "reply_json",
    "request_for",
    "retry_prompt",
    "signals_for",
]
