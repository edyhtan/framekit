"""Policy (question 6, docs/settings.md §5.5): what rules apply. Quality control, the
budget, the dataset's format, and governance (what may leave the machine)."""

from typing import Annotated, ClassVar, Literal

from pydantic import Field

from framekit.extraction.schemas.common import Frozen, FrozenDict, Model, Tupled
from framekit.extraction.schemas.plan.component import Component

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


# --- quality control, budget, formatting, governance ---------------------------------------------


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
    schema_version: Literal[2] = 2  # the dataset's (the manifest's SCHEMA_VERSION)
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


# --- the component ----------------------------------------------------------------------


class Policy(Component):
    """What rules apply: `quality_control`, `budget`, `formatting` and `governance`."""

    KEY: ClassVar[str] = "policy"

    version: Literal[1] = 1
    quality_control: QualityControl = QualityControl()
    budget: BudgetSettings = BudgetSettings()
    formatting: Formatting = Formatting()
    governance: Governance = Governance()

    def problems(self) -> list[str]:
        tables = self.formatting.tables
        problems = [
            f"formatting.tables: {name} is needed (provenance and reading)"
            for name in REQUIRED_TABLES
            if name not in tables
        ]
        if len(set(tables)) != len(tables):
            problems.append("formatting.tables: a table is listed twice")
        checks = self.quality_control.checks
        if len(set(checks)) != len(checks):
            problems.append("quality_control.checks: a check is listed twice")
        return problems
