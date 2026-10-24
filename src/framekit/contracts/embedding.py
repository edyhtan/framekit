"""The `embed_text` port: provider-neutral types for turning texts into vectors.

`framekit.models` implements it (the router's `embed`); which model serves the role is
configuration. Plain dataclasses only, so no provider or library type crosses the port.
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol


@dataclass(frozen=True)
class EmbeddingRequest:
    texts: tuple[str, ...]  # one vector per text, in order
    model_id: str | None = None  # None: the router's default model


@dataclass(frozen=True)
class EmbeddingResult:
    provider: str
    model_id: str  # the model that answered: vectors are only comparable within one model
    vectors: tuple[tuple[float, ...], ...]  # one per text, in the request's order
    dimension: int
    input_tokens: int
    latency_ms: int
    cost_usd: Decimal  # estimate from the catalog's prices
    adapter_version: str
    # The `ModelSource` that served the call (its name); None: a registry model.
    source: str | None = None


class EmbeddingPort(Protocol):
    async def embed(self, request: EmbeddingRequest) -> EmbeddingResult: ...

    async def aclose(self) -> None: ...
