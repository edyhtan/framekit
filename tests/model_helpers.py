"""Test models for the extraction tests, served through `ModelSource`s by a local (loopback)
provider: no API calls."""

from decimal import Decimal
from typing import Any, ClassVar

from framekit.models import ModelSource, ModelSpec, ProviderConfig
from framekit.models.adapters import ExternalAIAdapterBase


class StubAdapter(ExternalAIAdapterBase):
    """An adapter that is never called: resolution only reads its class."""

    adapter_version: ClassVar[str] = "stub-1"

    @classmethod
    def make_client(cls, api_key: str, base_url: str | None, timeout_s: float) -> Any:
        return object()

    def build_params(self, spec: ModelSpec, request: Any, max_tokens: int) -> dict[str, Any]:
        return {}

    async def call(self, spec: ModelSpec, params: dict[str, Any]) -> Any:
        raise AssertionError("resolution never calls a model")


LOCAL = ProviderConfig("scripted", "openai", base_url="http://127.0.0.1:9/v1")
VISION = ModelSpec(
    "scripted",
    "seer-1",
    "Scripted vision",
    Decimal("1"),
    Decimal("5"),
    vision=True,
    max_images=50,
    max_image_side_px=1600,
)
TEXT = ModelSpec("scripted", "writer-1", "Scripted text", Decimal("0.3"), Decimal("1.2"))
EMBEDDER = ModelSpec("scripted", "vectors-1", "Scripted embeddings", Decimal("0.02"), Decimal("0"))


def sources(adapter: type = StubAdapter) -> list[ModelSource]:
    """The test models, served by a local (loopback) provider."""
    return [
        ModelSource("seer", model=VISION, provider=LOCAL, credential="test", adapter=adapter),
        ModelSource("writer", model=TEXT, provider=LOCAL, credential="test", adapter=adapter),
        ModelSource("vectors", model=EMBEDDER, provider=LOCAL, credential="test", adapter=adapter),
    ]
