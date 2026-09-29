"""`text_generate` port on the Anthropic Messages API (official `anthropic` SDK, async).

Prototype scope: one short, non-streaming message per call. Credentials come from the
environment (`ANTHROPIC_API_KEY`); nothing here stores or logs a key.
"""

import os
import time
from dataclasses import dataclass, field
from decimal import Decimal
from functools import lru_cache
from typing import Any

import anthropic
from pydantic_settings import BaseSettings, SettingsConfigDict

from clipsearch_domain.common.ports import (
    ModelCallError,
    TextGenerateRequest,
    TextGenerateResult,
)

PROVIDER = "anthropic"
ADAPTER_VERSION = "0.1.0"
MILLION = Decimal(1_000_000)


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    label: str
    input_usd_per_mtok: Decimal
    output_usd_per_mtok: Decimal
    # Extra request fields this model needs for a short, cheap probe.
    request_extras: dict[str, Any] = field(default_factory=dict)
    betas: tuple[str, ...] = ()


# Prices: Anthropic first-party API rates per 1M tokens (skill cache 2026-06-24).
CATALOG: dict[str, ModelSpec] = {
    spec.model_id: spec
    for spec in (
        ModelSpec("claude-haiku-4-5", "Claude Haiku 4.5", Decimal("1"), Decimal("5")),
        ModelSpec(
            "claude-sonnet-5",
            "Claude Sonnet 5",
            Decimal("2"),
            Decimal("10"),
            request_extras={"output_config": {"effort": "low"}},
        ),
        ModelSpec(
            "claude-opus-5",
            "Claude Opus 5",
            Decimal("5"),
            Decimal("25"),
            # Thinking is on by default on Opus 5; low effort keeps a probe cheap.
            # Server-side refusal fallback, as recommended for Opus 5 code.
            request_extras={"output_config": {"effort": "low"}, "fallbacks": "default"},
            betas=("server-side-fallback-2026-07-01",),
        ),
    )
}


class AnthropicSettings(BaseSettings):
    """Prototype model profile for the test AI worker (`CLIPSEARCH_AI_TEST_*`)."""

    model_config = SettingsConfigDict(
        env_prefix="CLIPSEARCH_AI_TEST_", env_file=".env", extra="ignore"
    )

    default_model: str = "claude-haiku-4-5"
    max_tokens_cap: int = 256
    timeout_s: float = 30.0


@lru_cache
def get_anthropic_settings() -> AnthropicSettings:
    return AnthropicSettings()


def estimate_cost(spec: ModelSpec, input_tokens: int, output_tokens: int) -> Decimal:
    cost = (
        Decimal(input_tokens) * spec.input_usd_per_mtok
        + Decimal(output_tokens) * spec.output_usd_per_mtok
    ) / MILLION
    return cost.quantize(Decimal("0.000001"))


class AnthropicTextGenerator:
    def __init__(self, client: Any, settings: AnthropicSettings) -> None:
        self._client = client
        self._settings = settings

    @classmethod
    def from_environment(cls) -> "AnthropicTextGenerator":
        # Prototype: the key comes from a local environment variable. Without it the SDK
        # only fails at request time, with a TypeError, so check up front.
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise ModelCallError(
                "ANTHROPIC_API_KEY is not set in the worker's environment", retryable=False
            )
        settings = get_anthropic_settings()
        return cls(anthropic.AsyncAnthropic(timeout=settings.timeout_s), settings)

    def resolve(self, model_id: str | None) -> ModelSpec:
        chosen = model_id or self._settings.default_model
        if chosen not in CATALOG:
            raise ModelCallError(f"model {chosen!r} is not in the catalog", retryable=False)
        return CATALOG[chosen]

    async def generate(self, request: TextGenerateRequest) -> TextGenerateResult:
        spec = self.resolve(request.model_id)
        params: dict[str, Any] = {
            "model": spec.model_id,
            "max_tokens": min(request.max_tokens, self._settings.max_tokens_cap),
            "messages": [{"role": "user", "content": request.prompt}],
            **spec.request_extras,
        }
        started = time.perf_counter()
        try:
            if spec.betas:
                message = await self._client.beta.messages.create(betas=list(spec.betas), **params)
            else:
                message = await self._client.messages.create(**params)
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as exc:
            raise ModelCallError(f"not authorized: {exc.message}", retryable=False) from exc
        except (anthropic.NotFoundError, anthropic.BadRequestError) as exc:
            raise ModelCallError(f"request rejected: {exc.message}", retryable=False) from exc
        except anthropic.RateLimitError as exc:
            raise ModelCallError(f"rate limited: {exc.message}", retryable=True) from exc
        except anthropic.APIStatusError as exc:
            retryable = exc.status_code >= 500
            raise ModelCallError(
                f"API error {exc.status_code}: {exc.message}", retryable=retryable
            ) from exc
        except anthropic.APIConnectionError as exc:
            raise ModelCallError(f"connection failed: {exc}", retryable=True) from exc
        latency_ms = round((time.perf_counter() - started) * 1000)

        text = "".join(block.text for block in message.content if block.type == "text")
        usage = message.usage
        return TextGenerateResult(
            provider=PROVIDER,
            model_id=message.model,
            text=text,
            stop_reason=message.stop_reason,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            latency_ms=latency_ms,
            cost_usd=estimate_cost(spec, usage.input_tokens, usage.output_tokens),
            request_id=getattr(message, "_request_id", None),
            adapter_version=ADAPTER_VERSION,
        )


def catalog_listing() -> list[dict[str, Any]]:
    default = get_anthropic_settings().default_model
    return [
        {
            "model_id": spec.model_id,
            "label": spec.label,
            "input_usd_per_mtok": str(spec.input_usd_per_mtok),
            "output_usd_per_mtok": str(spec.output_usd_per_mtok),
            "default": spec.model_id == default,
        }
        for spec in CATALOG.values()
    ]
