"""OpenAI-compatible providers for the `text_generate` port (official `openai` SDK, async).

One adapter, configured per provider:
- **hive**: Hive's self-hosted LLMs (api-cdn.thehive.ai). Hive documents the streaming
  path only (server-sent events, usage always in the final chunk), so requests stream
  and the adapter assembles the reply.
- **deepseek**: DeepSeek's own OpenAI-format endpoint (api.deepseek.com), non-streaming.

Credentials come from the environment; nothing here stores or logs a key.
"""

import os
import time
from dataclasses import dataclass, field
from typing import Any

import openai

from clipsearch_adapters.models.catalog import ModelSpec, TextModelSettings, estimate_cost
from clipsearch_domain.common.ports import ModelCallError, TextGenerateRequest, TextGenerateResult

ADAPTER_VERSION = "0.3.0"
REASONING_TEXT_LIMIT = 4000  # characters kept for display


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    base_url: str
    key_env: str
    stream: bool
    base_url_env: str | None = None  # optional override, e.g. a regional endpoint
    extra_headers: dict[str, str] = field(default_factory=dict)
    # Provider-specific statuses that mean "don't retry" beyond the usual 4xx.
    permanent_statuses: frozenset[int] = frozenset()

    def resolved_base_url(self) -> str:
        if self.base_url_env and os.environ.get(self.base_url_env):
            return os.environ[self.base_url_env]
        return self.base_url


HIVE = ProviderConfig(
    name="hive",
    base_url="https://api-cdn.thehive.ai/api/v3/",
    base_url_env="HIVE_BASE_URL",  # Virginia customers: https://api-va1.thehive.ai/api/v3/
    key_env="HIVE_API_KEY",
    stream=True,
    extra_headers={"Accept": "text/event-stream"},
    permanent_statuses=frozenset({405}),  # "organization paused" / out of balance
)

DEEPSEEK = ProviderConfig(
    name="deepseek",
    base_url="https://api.deepseek.com",
    key_env="DEEPSEEK_API_KEY",
    stream=False,
    permanent_statuses=frozenset({402}),  # out of balance
)


@dataclass
class _Reply:
    text: str = ""
    reasoning: str = ""
    model_id: str | None = None
    stop_reason: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int | None = None
    request_id: str | None = None


def _reasoning_delta(delta: Any) -> str | None:
    # Not part of the OpenAI schema; the SDK keeps unknown fields in `model_extra`.
    return (getattr(delta, "model_extra", None) or {}).get("reasoning_content")


def _reasoning_tokens(usage: Any) -> int | None:
    details = getattr(usage, "completion_tokens_details", None)
    return getattr(details, "reasoning_tokens", None) if details else None


def output_budget(spec: ModelSpec, requested: int, profile_cap: int) -> int:
    """Tokens to request: the answer budget plus the model's reasoning allowance, capped."""
    cap = spec.max_tokens_cap or profile_cap
    return min(requested + spec.reasoning_allowance, cap)


class OpenAICompatibleGenerator:
    def __init__(self, provider: ProviderConfig, client: Any, settings: TextModelSettings) -> None:
        self._provider = provider
        self._client = client
        self._settings = settings

    @classmethod
    def from_environment(
        cls, provider: ProviderConfig, settings: TextModelSettings
    ) -> "OpenAICompatibleGenerator":
        api_key = os.environ.get(provider.key_env)
        if not api_key:
            raise ModelCallError(
                f"{provider.key_env} is not set in the worker's environment", retryable=False
            )
        client = openai.AsyncOpenAI(
            api_key=api_key, base_url=provider.resolved_base_url(), timeout=settings.timeout_s
        )
        return cls(provider, client, settings)

    async def generate(self, spec: ModelSpec, request: TextGenerateRequest) -> TextGenerateResult:
        max_tokens = output_budget(spec, request.max_tokens, self._settings.max_tokens_cap)
        params: dict[str, Any] = {
            "model": spec.model_id,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": request.prompt}],
            "stream": self._provider.stream,
            **spec.request_extras,
        }
        if self._provider.extra_headers:
            params["extra_headers"] = dict(self._provider.extra_headers)

        started = time.perf_counter()
        try:
            response = await self._client.chat.completions.create(**params)
            reply = await self._read_stream(response) if self._provider.stream else _read(response)
        except (openai.AuthenticationError, openai.PermissionDeniedError) as exc:
            raise ModelCallError(f"not authorized: {exc.message}", retryable=False) from exc
        except (
            openai.BadRequestError,
            openai.UnprocessableEntityError,
            openai.NotFoundError,
        ) as exc:
            raise ModelCallError(f"request rejected: {exc.message}", retryable=False) from exc
        except openai.RateLimitError as exc:
            raise ModelCallError(f"rate limited: {exc.message}", retryable=True) from exc
        except openai.APIStatusError as exc:
            permanent = exc.status_code in self._provider.permanent_statuses
            raise ModelCallError(
                f"API error {exc.status_code}: {exc.message}",
                retryable=exc.status_code >= 500 and not permanent,
            ) from exc
        except openai.APIConnectionError as exc:
            raise ModelCallError(f"connection failed: {exc}", retryable=True) from exc
        latency_ms = round((time.perf_counter() - started) * 1000)

        # Some providers report "stop" even when the output hit the limit mid-answer.
        stop_reason = "length" if reply.output_tokens >= max_tokens else reply.stop_reason
        return TextGenerateResult(
            provider=self._provider.name,
            model_id=reply.model_id or spec.model_id,
            text=reply.text,
            stop_reason=stop_reason,
            input_tokens=reply.input_tokens,
            output_tokens=reply.output_tokens,
            latency_ms=latency_ms,
            cost_usd=estimate_cost(spec, reply.input_tokens, reply.output_tokens),
            request_id=reply.request_id,
            adapter_version=ADAPTER_VERSION,
            reasoning_text=reply.reasoning[:REASONING_TEXT_LIMIT] or None,
            reasoning_tokens=reply.reasoning_tokens,
        )

    @staticmethod
    async def _read_stream(stream: Any) -> _Reply:
        reply, parts, thoughts = _Reply(), [], []
        http_response = getattr(stream, "response", None)
        if http_response is not None:
            reply.request_id = http_response.headers.get("x-request-id")
        async for chunk in stream:
            reply.model_id = reply.model_id or getattr(chunk, "model", None)
            reply.request_id = reply.request_id or getattr(chunk, "id", None)
            for choice in chunk.choices or []:
                if choice.delta and choice.delta.content:
                    parts.append(choice.delta.content)
                if choice.delta and (thought := _reasoning_delta(choice.delta)):
                    thoughts.append(thought)
                if choice.finish_reason:
                    reply.stop_reason = choice.finish_reason
            if chunk.usage:  # Hive always sends usage in the final chunk
                reply.input_tokens = chunk.usage.prompt_tokens or 0
                reply.output_tokens = chunk.usage.completion_tokens or 0
                reply.reasoning_tokens = _reasoning_tokens(chunk.usage)
        reply.text = "".join(parts)
        reply.reasoning = "".join(thoughts)
        return reply


def _read(completion: Any) -> _Reply:
    choice = completion.choices[0] if completion.choices else None
    usage = completion.usage
    return _Reply(
        text=(choice.message.content or "") if choice else "",
        reasoning=(_reasoning_delta(choice.message) or "") if choice else "",
        reasoning_tokens=_reasoning_tokens(usage),
        model_id=completion.model,
        stop_reason=choice.finish_reason if choice else None,
        input_tokens=usage.prompt_tokens if usage else 0,
        output_tokens=usage.completion_tokens if usage else 0,
        request_id=getattr(completion, "_request_id", None) or completion.id,
    )
