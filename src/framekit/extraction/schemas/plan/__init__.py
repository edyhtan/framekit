"""The plan (DESIGN.md §2.4; docs/settings.md §5): how a search space is realized.

Five components, each its own versioned group with its own checks and digest:

- `observation`: how the video is looked at (the analysis copy, signals, frames, long units);
- `structure`: what one data point is (scenes, fixed windows, beats);
- `interpretation`: what is written about each data point (facts, lens passes, vectors);
- `models`: which model writes what, one per role;
- `policy`: what rules apply (quality control, budget, formatting, governance).

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
from typing import Any, Literal

from framekit.extraction.schemas.common import (
    ConfigError,
    Model,
    parse,
    plain,
    read_mapping,
    unwrap,
)
from framekit.extraction.schemas.plan.component import Component
from framekit.extraction.schemas.plan.interpretation import (
    Embedding,
    EntitySettings,
    ImageEmbedding,
    Interpretation,
    LensPasses,
    Perception,
    PerceptionContext,
    Phrasings,
    SelfConsistency,
    TextEmbedding,
)
from framekit.extraction.schemas.plan.models import MODEL_ROLES, ModelDefaults, Models, RoleModel
from framekit.extraction.schemas.plan.observation import (
    DEFAULT_FRAMES,
    AudioNormalize,
    AudioSignals,
    ColorSettings,
    Encoding,
    LongUnits,
    LoudnessSettings,
    MergeSettings,
    MotionSettings,
    NormalizeSettings,
    Observation,
    OnsetSettings,
    ShotSettings,
    SignalSettings,
    SpeechSettings,
    TaggerSettings,
    VadSettings,
    frames_from,
)
from framekit.extraction.schemas.plan.policy import (
    QC_CHECKS,
    REQUIRED_TABLES,
    TABLES,
    BudgetSettings,
    Formatting,
    Governance,
    InvalidOutput,
    Layout,
    Policy,
    QCThresholds,
    QualityControl,
)
from framekit.extraction.schemas.plan.structure import (
    Beats,
    FixedWindows,
    SceneLength,
    Segmentation,
    Structure,
)

WHAT = "the plan"

STAGES = ("L0", "L1", "L2", "L3a", "L3b", "embed", "L4")
COMPONENTS: dict[str, type[Component]] = {
    "observation": Observation,
    "structure": Structure,
    "interpretation": Interpretation,
    "models": Models,
    "policy": Policy,
}

# Plan version 1 kept these at the top level; version 2 groups them. A version 1 file gets
# one problem per key, saying where the key moved.
MOVED_IN_V2 = {
    "normalize": "observation.normalize",
    "vision_profiles": "observation.frames.profiles (stages assign one with `profile:`)",
    "stages": "observation.frames.stages (`{vision: name}` is now `{profile: name}`)",
    "signals": "observation.signals",
    "long_units": "observation.long_units",
    "segmentation": "structure.segmentation",
    "perception": "interpretation.perception (its `beats` is now structure.beats)",
    "lens_passes": "interpretation.lens_passes",
    "embedding": "interpretation.embedding",
    "quality_control": "policy.quality_control",
    "budget": "policy.budget",
    "formatting": "policy.formatting",
    "governance": "policy.governance",
}


class ExtractionPlan(Model):
    version: Literal[2] = 2
    derivation_rules: Literal["v1"] = "v1"
    observation: Observation = Observation()
    structure: Structure = Structure()
    interpretation: Interpretation = Interpretation()
    models: Models = Models()
    policy: Policy = Policy()

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ExtractionPlan":
        """A plan from plain data (`{plan: {...}}` or the bare mapping), in the shape of
        docs/settings.md §5. In `observation.frames`, each part given (`profiles`,
        `stages`) replaces its default; the other keeps it."""
        raw = dict(plain(unwrap(data, "plan")))
        old = [key for key in MOVED_IN_V2 if key in raw]
        if raw.get("version") == 1 or old:
            raise ConfigError(
                [
                    "version: this is a version 1 plan; version 2 groups the settings into "
                    "observation, structure, interpretation, models and policy "
                    "(docs/settings.md §5)",
                    *(f"{key}: now {MOVED_IN_V2[key]}" for key in old),
                ],
                WHAT,
            )
        problems: list[str] = []
        frames = None
        observation = raw.get("observation")
        if isinstance(observation, Mapping) and "frames" in observation:
            observation = dict(observation)
            frames, problems = frames_from(observation.pop("frames"), "observation.frames")
            raw["observation"] = observation
        try:
            plan = parse(cls, raw, WHAT, extra=lambda p: p.problems())
        except ConfigError as exc:
            raise ConfigError([*problems, *exc.problems], WHAT) from None
        if problems:
            raise ConfigError(problems, WHAT)
        if frames is None:
            return plan
        observed = plan.observation.model_copy(update={"frames": frames})
        return plan.model_copy(update={"observation": observed})

    @classmethod
    def load(cls, path: str | Path) -> "ExtractionPlan":
        return cls.from_mapping(read_mapping(Path(path).expanduser(), WHAT))

    def components(self) -> dict[str, Component]:
        """The five components, by their names in a plan file."""
        return {name: getattr(self, name) for name in COMPONENTS}

    def component_digests(self) -> dict[str, str]:
        """Each component's digest, as given (derived values still marked `derived`): two
        plans with the same digest for a component have the same settings there."""
        return {name: component.digest for name, component in self.components().items()}

    def problems(self) -> list[str]:
        """Each component's own checks, with the component's name in front (checks against
        the search space, the corpus and the models run when the plan is resolved)."""
        return [
            f"{name}.{problem}"
            for name, component in self.components().items()
            for problem in component.problems()
        ]


__all__ = [
    "COMPONENTS",
    "DEFAULT_FRAMES",
    "MODEL_ROLES",
    "MOVED_IN_V2",
    "QC_CHECKS",
    "REQUIRED_TABLES",
    "STAGES",
    "TABLES",
    "AudioNormalize",
    "AudioSignals",
    "Beats",
    "BudgetSettings",
    "ColorSettings",
    "Component",
    "Embedding",
    "Encoding",
    "EntitySettings",
    "ExtractionPlan",
    "FixedWindows",
    "Formatting",
    "Governance",
    "ImageEmbedding",
    "Interpretation",
    "InvalidOutput",
    "Layout",
    "LensPasses",
    "LongUnits",
    "LoudnessSettings",
    "MergeSettings",
    "ModelDefaults",
    "Models",
    "MotionSettings",
    "NormalizeSettings",
    "Observation",
    "OnsetSettings",
    "Perception",
    "PerceptionContext",
    "Phrasings",
    "Policy",
    "QCThresholds",
    "QualityControl",
    "RoleModel",
    "SceneLength",
    "Segmentation",
    "SelfConsistency",
    "ShotSettings",
    "SignalSettings",
    "SpeechSettings",
    "Structure",
    "TaggerSettings",
    "TextEmbedding",
    "VadSettings",
]
