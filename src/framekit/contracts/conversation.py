"""The `conversation` port: provider-neutral types for one model conversation.

Callers use capabilities, never providers (DESIGN.md §0.2). `framekit.models` implements
this port; which model serves a role is configuration. Plain dataclasses only, so no
provider or library type crosses the port.
"""

from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Literal, Protocol


@dataclass(frozen=True)
class ChatMessage:
    role: Literal["user", "assistant"]
    content: str


@dataclass(frozen=True)
class ConversationRequest:
    prompt: str  # the new user message
    max_tokens: int
    model_id: str | None = None  # None: the adapter's configured default
    history: tuple[ChatMessage, ...] = ()  # earlier turns, oldest first
    # Reasoning is billed as output tokens: off unless asked for. Models that can't
    # switch it off reason anyway; the result says whether it was on.
    reasoning: bool = False
    system: str | None = None  # instructions sent ahead of the conversation
    # Structured output: a JSON Schema (plain data). The reply text is then JSON meant to
    # match it; callers always validate it themselves, whatever the adapter enforced.
    response_schema: Mapping[str, Any] | None = None
    # What kind of request this is (e.g. "lens_pass", "chat"), for the model
    # response cache's per-class policy. None: the "default" class.
    cache_class: str | None = None


@dataclass(frozen=True)
class ConversationResult:
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
    reasoning_on: bool = False
    # How a response_schema was applied: "json_schema", "json_mode", "function_calling"
    # or "prompted" (schema in the instructions only). None: no schema was asked for.
    structured_mode: str | None = None
    # True when the reply was replayed from the model response cache, not a provider call
    # (then cost is 0 and timings are the replay's own).
    cached: bool = False
    # The `ModelSource` that served the call (its name). None: a model from the router's
    # registry (e.g. a built-in catalog model).
    source: str | None = None


@dataclass(frozen=True)
class TextDelta:
    """One streamed piece of output: part of the reasoning, or part of the answer."""

    kind: Literal["reasoning", "text"]
    text: str


# Provider-neutral stop reasons used in ConversationResult.stop_reason.
STOP_REASONS = ("end", "length", "refusal", "tool_use", "content_filter", "other")


class ModelCallError(Exception):
    """A model call failed. `retryable` says whether trying again can help."""

    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


class ConversationPort(Protocol):
    """One port per unit of work (a task attempt, a chat turn): it may open connections,
    and `aclose` releases them."""

    async def generate(self, request: ConversationRequest) -> ConversationResult: ...

    def stream(self, request: ConversationRequest) -> AsyncIterator[TextDelta | ConversationResult]:
        """Yield `TextDelta`s as output arrives, then exactly one final `ConversationResult`."""
        ...

    async def aclose(self) -> None: ...
