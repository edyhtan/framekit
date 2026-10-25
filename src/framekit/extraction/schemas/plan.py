"""The plan (DESIGN.md §2.4; docs/settings.md §5): how a search space is realized.

Optional: every setting has a documented default, and anything the plan file sets
overrides it (a group given in part keeps the defaults of the rest). Values the
derivation rules fill from the search space are marked `derived` or `from_space`; the
resolved plan (`framekit.extraction.resolve`) has them filled in and records each rule.
Every number is a placeholder until a benchmark sweep sets it.

Models are never defaulted: each role the run uses names its model in `models`.
Execution (concurrency, rate limits, retries, folders, cache) isn't part of the plan: it
changes speed and cost, never the dataset (`LocalRunner`).
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field

from framekit.extraction.schemas.common import (
    ConfigError,
    Frozen,
    FrozenDict,
    Model,
    Module,
    Tupled,
    parse,
    plain,
    read_mapping,
    unwrap,
)
from framekit.frames import FrameParams, FrameParamsError

WHAT = "the plan"

STAGES = ("L0", "L1", "L2", "L3a", "L3b", "embed", "L4")
MODEL_ROLES = ("L2", "L3a", "L3b", "L2_merge", "embed_text", "embed_image")
TABLES = (
    "units",
    "shots",
    "scenes",
    "beats",
    "windows",
    "signals",
    "perception",
    "frames_sent",
    "annotations",
    "phrasings",
    "vectors",
    "runs",
    "qc",
)
REQUIRED_TABLES = ("units", "scenes", "runs", "qc")  # the dataset is unreadable without them
QC_CHECKS = (
    "timestamps_in_scene",
    "timestamps_ordered",
    "entities_in_profile",
    "modules_complete",
    "vocabulary",
    "schema",
    "validators",
    "agreement",
)

# The default frame profiles (placeholders, like every number): a storyboard overview of
# the whole unit for L2, and up to 16 frames per scene for L3a.
DEFAULT_FRAMES: dict[str, Any] = {
    "vision_profiles": {
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
    "stages": {"L2": {"vision": "scene_overview"}, "L3a": {"vision": "scene_detail"}},
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


# --- observation ------------------------------------------------------------------------


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


# --- structure --------------------------------------------------------------------------


class SceneLength(Model):
    min: float = Field(default=30, gt=0)
    target: float = Field(default=45, gt=0)
    max: float = Field(default=60, gt=0)


class FixedWindows(Model):
    window_s: float | None = Field(default=None, gt=0)  # derived: the shortest moment length
    stride_s: float = Field(default=1, gt=0)


class Segmentation(Model):
    """L2. `method: model` asks the L2 model for a synopsis and scene boundaries (snapped to
    shot cuts); `shot_groups` groups consecutive shots into scenes of `scene_length_s`
    without a model (and without a synopsis)."""

    mode: Literal["scenes", "fixed_windows", "both"] = "scenes"
    method: Literal["model", "shot_groups"] = "model"
    scene_length_s: SceneLength = SceneLength()
    snap_to: Literal["shot_boundaries", "none"] = "shot_boundaries"
    snap_tolerance_s: float = Field(default=2.0, ge=0, le=30)
    fixed_windows: FixedWindows = FixedWindows()


# --- interpretation ---------------------------------------------------------------------


class Beats(Model):
    min_s: float | None = Field(default=None, gt=0)  # derived when unset
    max_s: float | None = Field(default=None, gt=0)


class PerceptionContext(Model):
    synopsis: bool = True
    signals: Annotated[tuple[Literal["shots", "audio", "speech", "motion"], ...], Tupled] = (
        "shots",
        "audio",
    )
    neighbors: int = Field(default=0, ge=0, le=3)  # adjacent scenes' summaries


class EntitySettings(Model):
    source: Literal["corpus_profile"] = "corpus_profile"
    # mark_unknown: a name not in the profile's list is written as "unknown"; allow: any name.
    unknown: Literal["mark_unknown", "allow"] = "mark_unknown"


class Perception(Model):
    modules: Literal["derived"] | Annotated[tuple[Module, ...], Tupled] = "derived"
    beats: Beats = Beats()
    context: PerceptionContext = PerceptionContext()
    entities: EntitySettings = EntitySettings()
    record_schema: Literal[1] = 1
    prompt_version: Literal[1] = 1


class Phrasings(Model):
    count: int = Field(default=8, ge=0, le=20)
    style: Literal["from_space", "from_lens"] = "from_space"


class SelfConsistency(Model):
    runs: int = Field(default=2, ge=1, le=5)
    seeds: Annotated[tuple[int, ...], Tupled] = (7, 8)
    on_disagreement: Literal["flag", "down_weight"] = "flag"


class LensPasses(Model):
    lenses: Literal["from_space"] = "from_space"
    targets: Annotated[tuple[Literal["scene", "beat"], ...], Tupled] = ("scene", "beat")
    phrasings: Phrasings = Phrasings()
    self_consistency: SelfConsistency = SelfConsistency()
    input: Literal["perception_records"] = "perception_records"


class TextEmbedding(Model):
    role: Literal["embed_text"] = "embed_text"
    inputs: Annotated[tuple[Literal["scene_summary", "phrasings"], ...], Tupled] = (
        "scene_summary",
        "phrasings",
    )
    template: str = "{text}"
    batch_size: int = Field(default=64, ge=1, le=2048)


class ImageEmbedding(Model):
    role: Literal["embed_image"] = "embed_image"


class Embedding(Model):
    text: TextEmbedding | None = TextEmbedding()
    image: ImageEmbedding | None = None  # no image embedding port yet: rejected when set
    normalize: bool = True


class RoleModel(Model):
    model_id: str = Field(min_length=1)
    max_tokens: int | None = Field(default=None, ge=1, le=65536)
    temperature: float | None = Field(default=None, ge=0, le=2)
    seed: int | None = None
    reasoning: bool | None = None


class ModelDefaults(Model):
    reasoning: bool = False  # reasoning is billed as output tokens: off unless asked
    temperature: float | None = Field(default=0.2, ge=0, le=2)
    structured_method: Literal["catalog"] = "catalog"  # the catalog's method per model


class Models(Model):
    defaults: ModelDefaults = ModelDefaults()
    L2: RoleModel | None = None
    L3a: RoleModel | None = None
    L3b: RoleModel | None = None
    L2_merge: RoleModel | None = None
    embed_text: RoleModel | None = None
    embed_image: RoleModel | None = None

    def role(self, name: str) -> RoleModel | None:
        model: RoleModel | None = getattr(self, name)
        return model


# --- control policy, formatting, governance ---------------------------------------------


class InvalidOutput(Model):
    max_attempts: int = Field(default=2, ge=1, le=5)
    then: Literal["flag", "down_weight"] = "flag"


class QCThresholds(Model):
    agreement_min: float = Field(default=0.6, ge=0, le=1)


class QualityControl(Model):
    checks: Annotated[tuple[Literal[QC_CHECKS], ...], Tupled] = QC_CHECKS  # type: ignore[valid-type]
    invalid_output: InvalidOutput = InvalidOutput()
    thresholds: QCThresholds = QCThresholds()


class BudgetSettings(Model):
    """Caps on estimated model spend, in USD. Not part of any input hash: a budget decides
    whether and when a run completes, never what it computes."""

    max_cost_usd: float | None = Field(default=None, ge=0)
    per_stage_usd: Annotated[dict[Literal["L2", "L3a", "L3b", "embed"], float], Frozen] = (
        FrozenDict()
    )
    on_exceed: Literal["stop", "finish_unit_then_stop"] = "finish_unit_then_stop"


class Layout(Model):
    partition_by: Literal["unit_id", "none"] = "unit_id"
    compression: Literal["zstd", "snappy", "none"] = "zstd"


class Formatting(Model):
    schema_version: Literal[1] = 1
    tables: Annotated[tuple[Literal[TABLES], ...], Tupled] = TABLES  # type: ignore[valid-type]
    layout: Layout = Layout()


class Governance(Model):
    """What may leave the machine. Inherited from the corpus's rights mode: a plan can only
    make it stricter. `model_inputs` says whether frames may be sent to a model provider
    for inference (`provider_transient`) or only to models on this machine or network
    (`local_only`); for a byo_media corpus there's no default, the plan must say."""

    rights_mode: Literal["inherited"] = "inherited"
    image_vectors: Literal["local_only", "per_licence"] = "local_only"
    training_use: Literal["local_only", "per_licence"] = "local_only"
    annotation_pack: bool = False
    model_inputs: Literal["provider_transient", "local_only"] | None = None


class ExtractionPlan(Model):
    version: Literal[1] = 1
    derivation_rules: Literal["v1"] = "v1"
    normalize: NormalizeSettings = NormalizeSettings()
    vision: FrameParams = FrameParams.from_mapping(DEFAULT_FRAMES)
    signals: SignalSettings = SignalSettings()
    long_units: LongUnits = LongUnits()
    segmentation: Segmentation = Segmentation()
    perception: Perception = Perception()
    lens_passes: LensPasses = LensPasses()
    embedding: Embedding = Embedding()
    models: Models = Models()
    quality_control: QualityControl = QualityControl()
    budget: BudgetSettings = BudgetSettings()
    formatting: Formatting = Formatting()
    governance: Governance = Governance()

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ExtractionPlan":
        """A plan from plain data (`{plan: {...}}` or the bare mapping). The vision block is
        `vision_profiles` and `stages` at the top level, as in docs/settings.md §5.2; each
        one given replaces its default (the profiles, or the stage assignments)."""
        raw = dict(plain(unwrap(data, "plan")))
        problems: list[str] = []
        if "vision" in raw:
            problems.append("vision: give `vision_profiles` and `stages` at the top level")
        frame_keys = {k: raw.pop(k) for k in ("vision_profiles", "stages") if k in raw}
        vision = None
        if frame_keys:
            block = {**DEFAULT_FRAMES, **frame_keys}
            try:
                vision = FrameParams.from_mapping(block)
            except FrameParamsError as exc:
                problems += list(exc.problems)
        try:
            plan = parse(cls, raw, WHAT, extra=lambda p: p.problems())
        except ConfigError as exc:
            raise ConfigError([*problems, *exc.problems], WHAT) from None
        if problems:
            raise ConfigError(problems, WHAT)
        return plan.model_copy(update={"vision": vision}) if vision is not None else plan

    @classmethod
    def load(cls, path: str | Path) -> "ExtractionPlan":
        return cls.from_mapping(read_mapping(Path(path).expanduser(), WHAT))

    def problems(self) -> list[str]:
        """Checks across the plan's own settings (checks against the search space, the
        corpus and the models run when the plan is resolved)."""
        problems = []
        length = self.segmentation.scene_length_s
        if not length.min <= length.target <= length.max:
            problems.append("segmentation.scene_length_s: expected min <= target <= max")
        if self.segmentation.mode == "fixed_windows":
            problems.append(
                "segmentation.mode: fixed_windows alone leaves no scenes to perceive "
                "(perception and lens passes run on scenes): use both"
            )
        beats = self.perception.beats
        if beats.min_s is not None and beats.max_s is not None and beats.min_s > beats.max_s:
            problems.append("perception.beats: min_s is more than max_s")
        consistency = self.lens_passes.self_consistency
        if len(consistency.seeds) != consistency.runs:
            problems.append(
                f"lens_passes.self_consistency.seeds: {consistency.runs} runs need "
                f"{consistency.runs} seeds, got {len(consistency.seeds)}"
            )
        elif len(set(consistency.seeds)) != len(consistency.seeds):
            problems.append("lens_passes.self_consistency.seeds: distinct runs need distinct seeds")
        targets = self.lens_passes.targets
        if not targets or len(set(targets)) != len(targets):
            problems.append("lens_passes.targets: scene and/or beat, each once")
        tables = self.formatting.tables
        problems += [
            f"formatting.tables: {name} is needed (provenance and reading)"
            for name in REQUIRED_TABLES
            if name not in tables
        ]
        if len(set(tables)) != len(tables):
            problems.append("formatting.tables: a table is listed twice")
        checks = self.quality_control.checks
        if len(set(checks)) != len(checks):
            problems.append("quality_control.checks: a check is listed twice")
        if self.embedding.text is not None and "{text}" not in self.embedding.text.template:
            problems.append("embedding.text.template: must contain {text}")
        return problems


__all__ = [
    "DEFAULT_FRAMES",
    "MODEL_ROLES",
    "QC_CHECKS",
    "STAGES",
    "TABLES",
    "Beats",
    "BudgetSettings",
    "Embedding",
    "ExtractionPlan",
    "Formatting",
    "Governance",
    "LensPasses",
    "LongUnits",
    "Models",
    "NormalizeSettings",
    "Perception",
    "QualityControl",
    "RoleModel",
    "Segmentation",
    "SignalSettings",
    "TextEmbedding",
]
