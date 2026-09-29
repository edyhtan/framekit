"""Model capability ports (DESIGN.md §0, model-agnostic rule).

The domain calls capabilities, never providers. Adapters in
`clipsearch_adapters.external_connections` implement these; which model serves a role
is configuration.
"""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal, Protocol


@dataclass(frozen=True)
class ChatMessage:
    role: Literal["user", "assistant"]
    content: str


@dataclass(frozen=True)
class TextGenerateRequest:
    prompt: str  # the new user message
    max_tokens: int
    model_id: str | None = None  # None: the adapter's configured default
    history: tuple[ChatMessage, ...] = ()  # earlier turns, oldest first


@dataclass(frozen=True)
class TextGenerateResult:
    provider: str
    model_id: str
    text: str
    stop_reason: str | None
    input_tokens: int
    output_tokens: int
    latency_ms: int
    cost_usd: Decimal  # estimate from the adapter's price table
    request_id: str | None
    adapter_version: str
    # Reasoning models: the model's reasoning, kept apart from the answer.
    reasoning_text: str | None = None
    reasoning_tokens: int | None = None
    # `stop_reason` is normalized (see STOP_REASONS); this is the provider's own value.
    provider_stop_reason: str | None = None
    # Streaming only: time from sending the request to the first piece of output.
    first_token_ms: int | None = None


@dataclass(frozen=True)
class TextDelta:
    """One streamed piece of output: part of the reasoning, or part of the answer."""

    kind: Literal["reasoning", "text"]
    text: str


# Provider-neutral stop reasons used in TextGenerateResult.stop_reason.
STOP_REASONS = ("end", "length", "refusal", "tool_use", "content_filter", "other")


class ModelCallError(Exception):
    """A model call failed. `retryable` says whether trying again can help."""

    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


class TextGeneratePort(Protocol):
    """One generator per job attempt or chat turn: it may open connections, and `aclose`
    releases them."""

    async def generate(self, request: TextGenerateRequest) -> TextGenerateResult: ...

    def stream(self, request: TextGenerateRequest) -> AsyncIterator[TextDelta | TextGenerateResult]:
        """Yield `TextDelta`s as output arrives, then exactly one final `TextGenerateResult`."""
        ...

    async def aclose(self) -> None: ...
