"""L1 signals as pixel-free planning reads them: plain data on the unit timeline.

Times are seconds on the unit timeline (never on a chunk file). Strategies name the
signals they read; `Signals` only carries them.
"""

import math
from collections.abc import Iterable
from dataclasses import dataclass

Span = tuple[float, float]  # (start_s, end_s) on the unit timeline, end excluded


def _times(name: str, values: Iterable[float]) -> tuple[float, ...]:
    out = tuple(sorted(float(t) for t in values))
    if any(not math.isfinite(t) or t < 0 for t in out):
        raise ValueError(f"{name}: times must be finite and >= 0")
    return out


@dataclass(frozen=True)
class Signals:
    """The deterministic L1 signals of one unit (DESIGN.md §3 L1). Every field is optional;
    a strategy that needs a missing signal simply finds no events.

    - `shot_boundaries`: times of shot cuts, in seconds;
    - `audio_onsets`: times of audio onsets, in seconds;
    - `motion`: `(time_s, value)` samples of motion strength, value >= 0 (any scale: the
      planner normalizes each signal before combining them).

    Lists are accepted and stored as sorted tuples.
    """

    shot_boundaries: tuple[float, ...] = ()
    audio_onsets: tuple[float, ...] = ()
    motion: tuple[tuple[float, float], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "shot_boundaries", _times("shot_boundaries", self.shot_boundaries))
        object.__setattr__(self, "audio_onsets", _times("audio_onsets", self.audio_onsets))
        samples = tuple(sorted((float(t), float(v)) for t, v in self.motion))
        for t, v in samples:
            if not (math.isfinite(t) and math.isfinite(v)) or t < 0 or v < 0:
                raise ValueError("motion: samples must be finite (time_s >= 0, value >= 0)")
        object.__setattr__(self, "motion", samples)
