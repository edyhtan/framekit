"""Models (question 5, docs/settings.md §5.4): which model writes what. One model per role
(L2, L3a, L3b, embeddings), each with its sampling settings."""

from typing import ClassVar, Literal

from pydantic import Field

from framekit.extraction.schemas.common import Model
from framekit.extraction.schemas.plan.component import Component

MODEL_ROLES = ("L2", "L3a", "L3b", "L2_merge", "embed_text", "embed_image")


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


class Models(Component):
    """Who writes it: one model per role, and the defaults every role starts from. Models
    are never defaulted: each role the run uses names its model."""

    KEY: ClassVar[str] = "models"

    version: Literal[1] = 1
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
