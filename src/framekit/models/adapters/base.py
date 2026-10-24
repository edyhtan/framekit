"""One interface for every external AI provider behind the `conversation` port.

A provider is described by a `ProviderConfig` in the catalog (which wire protocol it
speaks, its key, URL and quirks). Each wire protocol has one `ExternalAIAdapterBase`
subclass that only builds the request, makes the call and parses the reply. Everything
else (key check, output budget, timing, error mapping, stop-reason normalization, the
result) is shared here.
"""

import json
import os
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from types import ModuleType
from typing import Any, ClassVar

from framekit.contracts.conversation import (
    ConversationRequest,
    ConversationResult,
    ModelCallError,
    TextDelta,
)
from framekit.models.catalog import (
    ModelSettings,
    ModelSpec,
    ProviderConfig,
    estimate_cost,
)

REASONING_TEXT_LIMIT = 4000  # characters kept for display


@dataclass
class Reply:
    """What a protocol adapter extracts from one response, before normalization."""

    text: str = ""
    reasoning: str = ""
    model_id: str | None = None
    stop_reason: str | None = None  # provider's raw value
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int | None = None
    request_id: str | None = None


def conversation(request: ConversationRequest) -> list[dict[str, str]]:
    """Earlier turns (oldest first) followed by the new user message."""
    return [{"role": m.role, "content": m.content} for m in request.history] + [
        {"role": "user", "content": request.prompt}
    ]


def structured_mode(
    spec: ModelSpec, request: ConversationRequest, supported: frozenset[str]
) -> str | None:
    """How this call applies `response_schema`: the model's method from the catalog if this
    adapter supports it, else "prompted" (schema in the instructions only)."""
    if request.response_schema is None:
        return None
    return spec.structured_method if spec.structured_method in supported else "prompted"


def system_text(request: ConversationRequest, mode: str | None) -> str | None:
    """The system instructions, plus the schema when the provider doesn't enforce it.

    JSON mode enforces valid JSON but not the shape, and "prompted" enforces nothing, so
    both carry the schema in the instructions. (DeepSeek's JSON mode also requires the
    word "json" in the prompt.)"""
    parts = [request.system] if request.system else []
    if mode in ("prompted", "json_mode"):
        schema = json.dumps(request.response_schema, separators=(",", ":"))
        parts.append(
            "Reply with a single JSON object that matches this JSON Schema, "
            f"and nothing else (no prose, no code fences):\n{schema}"
        )
    return "\n\n".join(parts) or None


def reasoning_active(spec: ModelSpec, request: ConversationRequest) -> bool:
    """Whether this call will reason: asked for, or the model can't switch it off."""
    control = spec.reasoning
    if control is None:
        return False
    return request.reasoning or control.off is None


def reasoning_fields(spec: ModelSpec, reasoning: bool) -> dict[str, Any]:
    """The request fields that switch this model's reasoning on or off."""
    control = spec.reasoning
    if control is None:
        return {}
    return control.on if reasoning or control.off is None else control.off


def merge_fields(*parts: dict[str, Any]) -> dict[str, Any]:
    """Merge request-field dicts; nested dicts (e.g. `extra_body`) are merged, not replaced."""
    merged: dict[str, Any] = {}
    for part in parts:
        for key, value in part.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = merge_fields(merged[key], value)
            else:
                merged[key] = value
    return merged


def output_budget(spec: ModelSpec, requested: int, profile_cap: int, *, reasoning: bool) -> int:
    """Tokens to request: the answer budget, plus the reasoning allowance when reasoning is
    on, capped."""
    control = spec.reasoning
    if not (reasoning and control):
        return min(requested, profile_cap)
    return min(requested + control.allowance, control.max_tokens_cap or profile_cap)


_STOP_REASON_MAP = {
    "end_turn": "end",  # Anthropic
    "stop_sequence": "end",
    "stop": "end",  # OpenAI format
    "max_tokens": "length",
    "length": "length",
    "refusal": "refusal",
    "tool_use": "tool_use",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "content_filter",
}


def normalize_stop_reason(raw: str | None, *, hit_limit: bool) -> str:
    # Some providers report "stop" even when the output hit the limit mid-answer.
    if hit_limit:
        return "length"
    return _STOP_REASON_MAP.get(raw or "", "other")


def map_sdk_error(exc: Exception, sdk: ModuleType, config: ProviderConfig) -> ModelCallError:
    """Translate an SDK exception into ModelCallError. The `anthropic` and `openai` SDKs
    are generated by the same toolchain and share these exception names."""
    if isinstance(exc, sdk.AuthenticationError | sdk.PermissionDeniedError):
        return ModelCallError(f"not authorized: {exc.message}", retryable=False)
    if isinstance(exc, sdk.BadRequestError | sdk.UnprocessableEntityError | sdk.NotFoundError):
        return ModelCallError(f"request rejected: {exc.message}", retryable=False)
    if isinstance(exc, sdk.RateLimitError):
        return ModelCallError(f"rate limited: {exc.message}", retryable=True)
    if isinstance(exc, sdk.APIStatusError):
        permanent = exc.status_code in config.permanent_statuses
        return ModelCallError(
            f"API error {exc.status_code}: {exc.message}",
            retryable=exc.status_code >= 500 and not permanent,
        )
    if isinstance(exc, sdk.APIConnectionError):  # includes timeouts
        return ModelCallError(f"connection failed: {exc}", retryable=True)
    raise exc  # not an SDK error: a bug, let it surface


def require_api_key(config: ProviderConfig) -> str:
    """The provider's key from the environment; a missing key fails only this provider."""
    api_key = os.environ.get(config.key_env)
    if not api_key:
        raise ModelCallError(
            f"{config.key_env} is not set in this service's environment", retryable=False
        )
    return api_key


class ExternalAIAdapterBase(ABC):
    """Shared request lifecycle; subclasses implement one wire protocol."""

    sdk: ClassVar[ModuleType]
    adapter_version: ClassVar[str]
    # Structured-output methods this adapter can apply; anything else falls back to
    # "prompted".
    structured_modes: ClassVar[frozenset[str]] = frozenset({"prompted"})

    def __init__(self, config: ProviderConfig, client: Any, settings: ModelSettings) -> None:
        self.config = config
        self._client = client
        self._settings = settings

    @classmethod
    def from_environment(
        cls, config: ProviderConfig, settings: ModelSettings
    ) -> "ExternalAIAdapterBase":
        api_key = require_api_key(config)
        client = cls.make_client(api_key, config.resolved_base_url(), settings.timeout_s)
        return cls(config, client, settings)

    @classmethod
    @abstractmethod
    def make_client(cls, api_key: str, base_url: str | None, timeout_s: float) -> Any: ...

    async def aclose(self) -> None:
        """Close the SDK client and its connection pool."""
        close = getattr(self._client, "close", None)
        if close is not None:
            await close()

    @abstractmethod
    def build_params(
        self, spec: ModelSpec, request: ConversationRequest, max_tokens: int
    ) -> dict[str, Any]: ...

    @abstractmethod
    async def call(self, spec: ModelSpec, params: dict[str, Any]) -> Reply:
        """Send the request and parse the response (streamed or not) into a Reply."""

    async def iterate(
        self, spec: ModelSpec, params: dict[str, Any]
    ) -> AsyncIterator[TextDelta | Reply]:
        """Yield output pieces as they arrive, then the final Reply.

        Default for protocols that don't stream: one call, then its reasoning and text as
        single pieces. Streaming protocols override this."""
        reply = await self.call(spec, params)
        if reply.reasoning:
            yield TextDelta("reasoning", reply.reasoning)
        if reply.text:
            yield TextDelta("text", reply.text)
        yield reply

    async def generate(self, spec: ModelSpec, request: ConversationRequest) -> ConversationResult:
        reasoning = reasoning_active(spec, request)
        max_tokens = output_budget(
            spec, request.max_tokens, self._settings.max_tokens_cap, reasoning=reasoning
        )
        params = self.build_params(spec, request, max_tokens)
        started = time.perf_counter()
        try:
            reply = await self.call(spec, params)
        except self.sdk.APIError as exc:
            raise map_sdk_error(exc, self.sdk, self.config) from exc
        return self._result(
            spec,
            reply,
            max_tokens,
            started,
            first_token_at=None,
            reasoning=reasoning,
            mode=structured_mode(spec, request, self.structured_modes),
        )

    async def stream(
        self, spec: ModelSpec, request: ConversationRequest
    ) -> AsyncIterator[TextDelta | ConversationResult]:
        """Same lifecycle as `generate`, yielding pieces as they arrive, then the result."""
        reasoning = reasoning_active(spec, request)
        max_tokens = output_budget(
            spec, request.max_tokens, self._settings.max_tokens_cap, reasoning=reasoning
        )
        params = self.build_params(spec, request, max_tokens)
        started = time.perf_counter()
        first_token_at: float | None = None
        reply: Reply | None = None
        try:
            async for piece in self.iterate(spec, params):
                if isinstance(piece, TextDelta):
                    first_token_at = first_token_at or time.perf_counter()
                    yield piece
                else:
                    reply = piece
        except self.sdk.APIError as exc:
            raise map_sdk_error(exc, self.sdk, self.config) from exc
        if reply is None:  # a protocol bug, not a provider error
            raise ModelCallError("the adapter produced no final reply", retryable=False)
        yield self._result(
            spec,
            reply,
            max_tokens,
            started,
            first_token_at=first_token_at,
            reasoning=reasoning,
            mode=structured_mode(spec, request, self.structured_modes),
        )

    def _result(
        self,
        spec: ModelSpec,
        reply: Reply,
        max_tokens: int,
        started: float,
        *,
        first_token_at: float | None,
        reasoning: bool,
        mode: str | None = None,
    ) -> ConversationResult:
        latency_ms = round((time.perf_counter() - started) * 1000)
        first_token_ms = round((first_token_at - started) * 1000) if first_token_at else None
        return ConversationResult(
            provider=self.config.name,
            model_id=reply.model_id or spec.model_id,
            text=reply.text,
            stop_reason=normalize_stop_reason(
                reply.stop_reason, hit_limit=reply.output_tokens >= max_tokens
            ),
            provider_stop_reason=reply.stop_reason,
            input_tokens=reply.input_tokens,
            output_tokens=reply.output_tokens,
            latency_ms=latency_ms,
            cost_usd=estimate_cost(spec, reply.input_tokens, reply.output_tokens),
            request_id=reply.request_id,
            adapter_version=self.adapter_version,
            reasoning_text=reply.reasoning[:REASONING_TEXT_LIMIT] or None,
            reasoning_tokens=reply.reasoning_tokens,
            first_token_ms=first_token_ms,
            reasoning_on=reasoning,
            structured_mode=mode,
        )
