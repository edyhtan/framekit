"""One interface for every external AI provider behind the `conversation` port.

A provider is described by a `ProviderConfig` (which wire protocol it speaks, its key, URL
and quirks). Each wire protocol has one `ExternalAIAdapterBase` subclass that only builds
the request, makes the call and parses the reply. Everything else (key check, output
budget, timing, error mapping, stop-reason normalization, the result) is shared here.

`ExternalAIAdapterBase` is also the template for adapters you write yourself; it, `Reply`,
`PLACEHOLDER_KEY` and the helpers exported from `framekit.models.adapters` are public API
(docs/adapters.md).
"""

import base64
import inspect
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
    ImagePart,
    ModelCallError,
    TextDelta,
)
from framekit.contracts.embedding import EmbeddingRequest, EmbeddingResult
from framekit.models.catalog import (
    ModelSettings,
    ModelSpec,
    ProviderConfig,
    estimate_cost,
)

REASONING_TEXT_LIMIT = 4000  # characters kept for display

# The key a provider without one (`ProviderConfig.key_env=None`, e.g. a local server) is
# built with. Never None: given no key, the `openai` and `anthropic` SDKs fall back to
# their own variables (OPENAI_API_KEY, ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, ...) and
# would send a real cloud key to the provider's host.
PLACEHOLDER_KEY = "framekit-no-key"


@dataclass
class Reply:
    """What a protocol adapter extracts from one response, before normalization.

    `stop_reason` is the provider's raw value (the base normalizes it). `model_id` is the
    model that actually answered, when the response says; otherwise the requested model
    is recorded."""

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


def image_data_url(image: ImagePart) -> str:
    """An image as a `data:` URL (base64), the form OpenAI-compatible APIs take."""
    return f"data:{image.media_type};base64,{base64.b64encode(image.data).decode('ascii')}"


def sampling_fields(request: ConversationRequest, *, seed: bool = True) -> dict[str, Any]:
    """The request's temperature and seed, when set (a protocol without a seed passes
    seed=False and leaves it out)."""
    fields: dict[str, Any] = {}
    if request.temperature is not None:
        fields["temperature"] = request.temperature
    if seed and request.seed is not None:
        fields["seed"] = request.seed
    return fields


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
    are generated by the same toolchain and share these exception names. An exception of
    any other kind is re-raised unchanged."""
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
    if not config.key_env:
        raise ModelCallError(
            f"provider {config.name!r} has an empty key_env: name the variable that holds "
            "its key, or use key_env=None for a provider that takes no key",
            retryable=False,
        )
    api_key = os.environ.get(config.key_env)
    if not api_key:
        raise ModelCallError(
            f"{config.key_env} is not set in this service's environment", retryable=False
        )
    return api_key


def _version_problem(version: object, adapter: str) -> str | None:
    if isinstance(version, str) and version.strip():
        return None
    return (
        f"{adapter} must set adapter_version, a non-empty string recorded in every result, "
        f'e.g. adapter_version: ClassVar[str] = "my-adapter-0.1.0" (got {version!r})'
    )


def adapter_class_problems(cls: type) -> list[str]:
    """What keeps an `ExternalAIAdapterBase` subclass from working, found before it's used:
    `from_environment` checks it before building a client, and a `ModelRegistry` when it's
    built. Internal (not part of the template's public API)."""
    problems = []
    if inspect.isabstract(cls):
        missing = ", ".join(sorted(getattr(cls, "__abstractmethods__", ())))
        problems.append(f"{cls.__name__} doesn't implement {missing}")
    version_problem = _version_problem(getattr(cls, "adapter_version", None), cls.__name__)
    if version_problem:
        problems.append(version_problem)
    for name in ("make_client", "from_environment"):
        if not isinstance(inspect.getattr_static(cls, name, None), classmethod | staticmethod):
            problems.append(f"{cls.__name__}.{name} must be a classmethod (@classmethod)")
    return problems


class ExternalAIAdapterBase(ABC):
    """The adapter template: one subclass per wire protocol (or per client).

    The base owns the call lifecycle: the output budget and reasoning switches, timing
    (including time to first token), stop-reason normalization, the cost estimate from the
    model's prices, and the `ConversationResult` with its provenance. A subclass only builds
    the request, sends it and parses the reply, and maps its client's errors.

    Required in a subclass:

    - `adapter_version` (`ClassVar[str]`, non-empty): recorded in every result. Bump it
      whenever the requests the adapter sends, or the way it reads replies, change. An
      adapter without one can't be built (`TypeError`).
    - `build_params(spec, request, max_tokens) -> dict`: the request for one call. The
      helpers in `framekit.models.adapters` (`conversation`, `system_text`,
      `structured_mode`, `reasoning_fields`, `merge_fields`, ...) do the common parts.
    - `call(spec, params) -> Reply`: send the request and parse the response.

    Optional:

    - `make_client(api_key, base_url, timeout_s)`, a classmethod: lets `from_environment`,
      and so a `ModelRegistry`, build the adapter from its `ProviderConfig`. Without it,
      build the adapter yourself, `MyAdapter(config, client, settings)`, and hand the
      router an `adapter_factory`.
    - `from_environment(config, settings)`, a classmethod: reads the key named by
      `config.key_env` (`PLACEHOLDER_KEY` when that's None), then calls `make_client`.
    - `iterate(spec, params)`: real streaming. The default makes one `call` and yields its
      reasoning and text as single pieces.
    - `map_error(exc)`: translate your client's exceptions into `ModelCallError`. It never
      sees a `ModelCallError`: one raised by `call` or `iterate` always propagates
      unchanged, the same object.
    - `aclose()`: release the client. The default calls the client's `aclose()`, or else
      its `close()`, and awaits the result when it's awaitable.
    - `structured_modes`: the structured-output methods `build_params` applies; any other
      method falls back to "prompted".
    - `sdk`: an `openai`/`anthropic`-style SDK module, whose `APIError`s are then mapped by
      `map_sdk_error`. None (the default): no SDK mapping.

    In your methods, `self.client` is the client and `self.config` the `ProviderConfig`.
    Assigning `self.client` replaces the client the adapter uses and closes.

    This class, `Reply`, `PLACEHOLDER_KEY` and the helpers exported from
    `framekit.models.adapters` are public API under semantic versioning: an incompatible
    change to them is a new minor version while framekit is 0.x, and a new major version
    after that. Guide: docs/adapters.md.
    """

    sdk: ClassVar[ModuleType | None] = None
    adapter_version: ClassVar[str]
    # Structured-output methods this adapter can apply; anything else falls back to
    # "prompted".
    structured_modes: ClassVar[frozenset[str]] = frozenset({"prompted"})

    def __init__(self, config: ProviderConfig, client: Any, settings: ModelSettings) -> None:
        # Checked here, not when the first reply is read: by then the call is paid for.
        problem = _version_problem(getattr(self, "adapter_version", None), type(self).__name__)
        if problem:
            raise TypeError(problem)
        self.config = config
        self._client = client
        self._settings = settings

    @property
    def client(self) -> Any:
        """The client this adapter sends its requests with (and closes in `aclose`)."""
        return self._client

    @client.setter
    def client(self, client: Any) -> None:
        self._client = client

    @classmethod
    def from_environment(
        cls, config: ProviderConfig, settings: ModelSettings
    ) -> "ExternalAIAdapterBase":
        """Build the adapter with the provider's key from the environment, or with
        `PLACEHOLDER_KEY` when the provider takes no key (`key_env` is None).

        An adapter class that can't work (abstract, no `adapter_version`, `make_client` not
        a classmethod) is a `TypeError`, raised before any client is made."""
        problems = adapter_class_problems(cls)
        if problems:
            raise TypeError("; ".join(problems))
        api_key = PLACEHOLDER_KEY if config.key_env is None else require_api_key(config)
        return cls.from_key(config, api_key, settings)

    @classmethod
    def from_key(
        cls, config: ProviderConfig, api_key: str, settings: ModelSettings
    ) -> "ExternalAIAdapterBase":
        """Build the adapter with this key: a `ModelSource`'s credential, or the key
        `from_environment` read. Override this (not `from_environment`) to build the client
        another way, so both paths use it.

        The key goes only to `make_client`: it's never recorded in results, cache keys or
        error messages. An empty key is a non-retryable `ModelCallError`."""
        problems = adapter_class_problems(cls)
        if problems:
            raise TypeError("; ".join(problems))
        if not isinstance(api_key, str) or not api_key:
            raise ModelCallError(
                f"provider {config.name!r}: the credential is empty", retryable=False
            )
        client = cls.make_client(api_key, config.resolved_base_url(), settings.timeout_s)
        return cls(config, client, settings)

    @classmethod
    def make_client(cls, api_key: str, base_url: str | None, timeout_s: float) -> Any:
        """The client `from_environment` builds the adapter with. Optional: an adapter
        built directly (`MyAdapter(config, client, settings)`) doesn't need it."""
        raise NotImplementedError(
            f"{cls.__name__} doesn't implement make_client, so it can't be built from the "
            "environment or by a ModelRegistry: implement make_client, or build it directly "
            f"as {cls.__name__}(config, client, settings)"
        )

    async def aclose(self) -> None:
        """Close the client and its connection pool: its `aclose()` if it has one (an
        `httpx` client, say), else its `close()`, awaited when that returns an awaitable.
        A client with neither is left as it is."""
        for name in ("aclose", "close"):
            close = getattr(self._client, name, None)
            if callable(close):
                result = close()
                if inspect.isawaitable(result):
                    await result
                return

    def map_error(self, exc: Exception) -> ModelCallError | None:
        """The `ModelCallError` to raise for an exception from `call` or `iterate`, or None
        to let the exception propagate unchanged.

        Never called for a `ModelCallError`: `generate` and `stream` let one through
        unchanged. Default: an `APIError` of `sdk` is translated by `map_sdk_error`;
        anything else (every exception, when `sdk` is None) propagates as it is, since it's
        a bug to surface. Override it to translate your own client's exceptions, and return
        `super().map_error(exc)` for the rest. Returning anything but a `ModelCallError` or
        None (or `exc` itself, which also lets it propagate) is a bug in the adapter, and
        raises a `TypeError` chained to `exc`."""
        if isinstance(exc, ModelCallError):
            return None
        api_error = getattr(self.sdk, "APIError", None)
        if self.sdk is None or not isinstance(api_error, type) or not isinstance(exc, api_error):
            return None
        return map_sdk_error(exc, self.sdk, self.config)

    def _replacement(self, exc: Exception) -> Exception | None:
        """What `generate` and `stream` raise instead of `exc` (chained to it), from
        `map_error`; None to re-raise `exc` itself."""
        mapped = self.map_error(exc)
        if mapped is None or mapped is exc:
            return None
        if isinstance(mapped, ModelCallError):
            return mapped
        return TypeError(
            f"{type(self).__name__}.map_error must return a ModelCallError or None, "
            f"got {mapped!r}"
        )

    @abstractmethod
    def build_params(
        self, spec: ModelSpec, request: ConversationRequest, max_tokens: int
    ) -> dict[str, Any]:
        """The request for one call, with `max_tokens` as its output budget."""

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

    def check_request(self, spec: ModelSpec, request: ConversationRequest) -> None:
        """Refuse, before any call, what the model can't take: images for a model whose
        catalog entry doesn't accept them. A non-retryable `ModelCallError`."""
        if request.images and not spec.vision:
            raise ModelCallError(
                f"model {spec.model_id!r} doesn't accept images (its catalog entry has "
                "vision=False)",
                retryable=False,
            )

    async def embed(self, spec: ModelSpec, request: EmbeddingRequest) -> EmbeddingResult:
        """Texts to vectors. Optional: an adapter for a protocol without embeddings keeps
        this default, a non-retryable `ModelCallError`."""
        raise ModelCallError(
            f"{type(self).__name__} doesn't serve embeddings (model {spec.model_id!r})",
            retryable=False,
        )

    async def generate(self, spec: ModelSpec, request: ConversationRequest) -> ConversationResult:
        self.check_request(spec, request)
        reasoning = reasoning_active(spec, request)
        max_tokens = output_budget(
            spec, request.max_tokens, self._settings.max_tokens_cap, reasoning=reasoning
        )
        params = self.build_params(spec, request, max_tokens)
        started = time.perf_counter()
        try:
            reply = await self.call(spec, params)
        except ModelCallError:
            raise  # always unchanged, whatever map_error does
        except Exception as exc:
            replacement = self._replacement(exc)
            if replacement is None:
                raise
            raise replacement from exc
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
        self.check_request(spec, request)
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
        except ModelCallError:
            raise  # always unchanged, whatever map_error does
        except Exception as exc:
            replacement = self._replacement(exc)
            if replacement is None:
                raise
            raise replacement from exc
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
