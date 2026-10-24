"""L1 signals as pixel-free planning reads them: plain data on the unit timeline.

Times are seconds on the unit timeline (never on a chunk file). Strategies name the
signals they read; `Signals` only carries them.
"""

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from numbers import Real

Span = tuple[float, float]  # (start_s, end_s) on the unit timeline, end excluded


def _items(field: str, values: object) -> list[object]:
    if isinstance(values, str | bytes) or not isinstance(values, Iterable):
        raise TypeError(f"{field}: expected a list, got {values!r}")
    return list(values)


def _number(field: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field}: expected a number, got {value!r}")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{field}: must be finite and >= 0, got {value!r}")
    return number


def _times(field: str, values: object) -> tuple[float, ...]:
    """Sorted, each time once: two events at the same time are one event."""
    return tuple(sorted({_number(field, t) for t in _items(field, values)}))


def _samples(values: object) -> tuple[tuple[float, float], ...]:
    samples = []
    for sample in _items("motion", values):
        pair = [] if isinstance(sample, str | bytes) else _items("motion", sample)
        if len(pair) != 2:
            raise TypeError(f"motion: each sample is (time_s, value), got {sample!r}")
        samples.append((_number("motion time", pair[0]), _number("motion value", pair[1])))
    return tuple(sorted(samples))


@dataclass(frozen=True)
class Signals:
    """The deterministic L1 signals of one unit (DESIGN.md §3 L1). Every field is optional;
    a strategy that needs a missing signal simply finds no events.

    - `shot_boundaries`: times of shot cuts, in seconds;
    - `audio_onsets`: times of audio onsets, in seconds;
    - `motion`: `(time_s, value)` samples of motion strength, value >= 0 (any scale: the
      planner normalizes each signal before combining them).

    Any list (or other iterable) is accepted and stored as a sorted tuple; a time listed
    twice counts once. Wrong types raise TypeError, negative or non-finite numbers
    ValueError, naming the field.
    """

    shot_boundaries: Sequence[float] = ()  # stored as a sorted tuple
    audio_onsets: Sequence[float] = ()
    motion: Sequence[tuple[float, float]] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "shot_boundaries", _times("shot_boundaries", self.shot_boundaries))
        object.__setattr__(self, "audio_onsets", _times("audio_onsets", self.audio_onsets))
        object.__setattr__(self, "motion", _samples(self.motion))
