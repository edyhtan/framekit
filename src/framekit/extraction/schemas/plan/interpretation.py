"""Interpretation (question 5, docs/settings.md §5.4): what is written about each data
point. Observable facts (L3a), lens labels and query-like phrasings (L3b), and vectors."""

from typing import Annotated, ClassVar, Literal

from pydantic import Field

from framekit.extraction.schemas.common import Model, Module, Tupled
from framekit.extraction.schemas.plan.component import Component


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


# --- the component ----------------------------------------------------------------------


class Interpretation(Component):
    """What is written about each data point: `perception` (observable facts, L3a),
    `lens_passes` (lens labels and phrasings, L3b) and `embedding` (vectors)."""

    KEY: ClassVar[str] = "interpretation"

    version: Literal[1] = 1
    perception: Perception = Perception()
    lens_passes: LensPasses = LensPasses()
    embedding: Embedding = Embedding()

    def problems(self) -> list[str]:
        problems = []
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
        if self.embedding.text is not None and "{text}" not in self.embedding.text.template:
            problems.append("embedding.text.template: must contain {text}")
        return problems
