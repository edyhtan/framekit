"""Observation (question 3, docs/settings.md §5.2): how the video is looked at. The
analysis copy (L0), the signals (L1), the frames each model sees (`framekit.frames`), and
long units."""

from collections.abc import Mapping
from typing import Any, ClassVar, Literal, Self

from pydantic import Field

from framekit.extraction.schemas.common import ConfigError, Model, parse, plain, unwrap
from framekit.extraction.schemas.plan.component import Component
from framekit.frames import FrameParams, FrameParamsError

# The default frame profiles (placeholders, like every number): a storyboard overview of
# the whole unit for L2, and up to 16 frames per scene for L3a.
DEFAULT_FRAMES: dict[str, Any] = {
    "profiles": {
        "scene_overview": {
            "version": 1,
            "frame_selection": {
                "candidates": {"strategy": "uniform", "fps": 0.5},
                "budget": {"max_frames_per_scene": 36, "min_gap_s": 1.0},
            },
            "image": {"max_side_px": 384, "format": "jpeg", "quality": 85},
            "delivery": {"mode": "storyboard", "storyboard": {"rows": 3, "cols": 3}},
        },
        "scene_detail": {
            "version": 1,
            "frame_selection": {
                "candidates": {"strategy": "uniform", "fps": 4},
                "anchors": ["shot_boundaries"],
                "filters": [
                    {"dedupe": {"method": "phash", "max_distance": 6}},
                    {"sharpness": {"keep": "best_per_window", "window_s": 0.5}},
                ],
                "budget": {
                    "max_frames_per_scene": 16,
                    "allocate_by": ["motion", "audio_onsets"],
                    "min_gap_s": 0.25,
                },
            },
            "image": {"max_side_px": 768, "format": "jpeg", "quality": 85},
            "delivery": {"mode": "images"},
        },
    },
    "stages": {"L2": {"profile": "scene_overview"}, "L3a": {"profile": "scene_detail"}},
}


# --- source preparation -----------------------------------------------------------------


class AudioNormalize(Model):
    sample_rate_hz: int = Field(default=32000, ge=8000, le=192000)
    channels: Literal[1, 2] = 1


class Encoding(Model):
    codec: Literal["h264"] = "h264"
    crf: int = Field(default=23, ge=0, le=51)
    preset: Literal["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow"] = (
        "veryfast"
    )


class ColorSettings(Model):
    normalize: Literal["auto", "off"] = "auto"  # auto: 8-bit 4:2:0, BT.709 range tags


class NormalizeSettings(Model):
    """L0: the analysis copy. Never upscaled: a source smaller than `height_px` keeps its
    height. Keyframes every 2 s, so frames can be read from anywhere quickly."""

    fps: float = Field(default=12, gt=0, le=120)
    height_px: int = Field(default=720, ge=120, le=4320)
    aspect: Literal["preserve"] = "preserve"
    audio: AudioNormalize = AudioNormalize()
    encoding: Encoding = Encoding()
    deinterlace: Literal["auto", "off"] = "auto"  # auto: only frames flagged interlaced
    color: ColorSettings = ColorSettings()


# --- signals and long units -------------------------------------------------------------


class ShotSettings(Model):
    detector: str = "ffmpeg_scene"  # a registered shot detector, `name` or `name@version`
    threshold: float = Field(default=0.3, ge=0, le=1)
    min_shot_s: float = Field(default=0.3, ge=0, le=60)


class OnsetSettings(Model):
    method: str = "spectral_flux"
    sensitivity: float = Field(default=0.5, ge=0, le=1)


class LoudnessSettings(Model):
    window_s: float = Field(default=0.1, gt=0, le=10)


class TaggerSettings(Model):
    """Audio tags (L1). No tagger is built in: `model` names a registered detector."""

    model: str
    labels: str | None = None
    threshold: float = Field(default=0.3, ge=0, le=1)


class AudioSignals(Model):
    onsets: OnsetSettings = OnsetSettings()
    loudness: LoudnessSettings = LoudnessSettings()
    tagger: TaggerSettings | None = None


class VadSettings(Model):
    enabled: bool = True  # VAD on: no transcripts hallucinated over music
    threshold: float = Field(default=0.5, ge=0, le=1)


class SpeechSettings(Model):
    """Speech segments (L1). No speech model is built in: `model` names a registered
    detector."""

    model: str
    language: str = "auto"
    vad: VadSettings = VadSettings()


class MotionSettings(Model):
    method: str = "frame_difference"
    rate_hz: float = Field(default=1, gt=0, le=30)


class SignalSettings(Model):
    shots: ShotSettings = ShotSettings()
    audio: AudioSignals = AudioSignals()
    speech: SpeechSettings | None = None
    motion: MotionSettings = MotionSettings()


class MergeSettings(Model):
    role: Literal["L2_merge"] = "L2_merge"


class LongUnits(Model):
    """Chunked processing of long units (DESIGN.md §5.3) isn't built yet: a unit longer
    than `enabled_over_s` stops with that reason. Raise it to process a unit in one pass."""

    chunk_s: float = Field(default=600, gt=0)
    overlap_s: float = Field(default=90, ge=0)
    snap_to: Literal["shot_boundaries"] = "shot_boundaries"
    enabled_over_s: float = Field(default=900, gt=0)
    merge: MergeSettings = MergeSettings()


def frames_from(given: Any, path: str) -> tuple[FrameParams | None, list[str]]:
    """The `frames` block of a plan: each part given (`profiles`, `stages`) replaces its
    default; the other keeps it. Problems carry `path` (where the block sits) in front."""
    if given is None:
        return None, []
    if not isinstance(given, Mapping):
        return None, [f"{path}: expected a mapping with profiles and stages"]
    try:
        return FrameParams.from_mapping({**DEFAULT_FRAMES, **given}), []
    except FrameParamsError as exc:
        return None, [f"{path}.{problem}" for problem in exc.problems]


# --- the component ----------------------------------------------------------------------


class Observation(Component):
    """How the video is looked at: `normalize` (the analysis copy), `signals` (L1), `frames`
    (which frames each model sees, and how they're prepared and delivered) and
    `long_units`."""

    KEY: ClassVar[str] = "observation"

    version: Literal[1] = 1
    normalize: NormalizeSettings = NormalizeSettings()
    signals: SignalSettings = SignalSettings()
    frames: FrameParams = FrameParams.from_mapping(DEFAULT_FRAMES)
    long_units: LongUnits = LongUnits()

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> Self:
        raw = dict(plain(unwrap(data, cls.KEY)))
        frames, problems = frames_from(raw.pop("frames", None), "frames")
        try:
            observation = parse(cls, raw, cls.what(), extra=lambda c: c.problems())
        except ConfigError as exc:
            raise ConfigError([*problems, *exc.problems], cls.what()) from None
        if problems:
            raise ConfigError(problems, cls.what())
        return observation.model_copy(update={"frames": frames}) if frames else observation
