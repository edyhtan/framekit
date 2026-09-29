"""Anthropic provider for the `text_generate` port (official `anthropic` SDK, async).

Prototype scope: one short, non-streaming message per call. Credentials come from the
environment (`ANTHROPIC_API_KEY`); nothing here stores or logs a key.
"""

import os
import time
from typing import Any

import anthropic

from clipsearch_adapters.models.catalog import ModelSpec, TextModelSettings, estimate_cost
from clipsearch_domain.common.ports import ModelCallError, TextGenerateRequest, TextGenerateResult

PROVIDER = "anthropic"
ADAPTER_VERSION = "0.2.0"


class AnthropicTextGenerator:
    def __init__(self, client: Any, settings: TextModelSettings) -> None:
        self._client = client
        self._settings = settings

    @classmethod
    def from_environment(cls, settings: TextModelSettings) -> "AnthropicTextGenerator":
        # Without a key the SDK only fails at request time, with a TypeError; check up front.
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise ModelCallError(
                "ANTHROPIC_API_KEY is not set in the worker's environment", retryable=False
            )
        return cls(anthropic.AsyncAnthropic(timeout=settings.timeout_s), settings)

    async def generate(self, spec: ModelSpec, request: TextGenerateRequest) -> TextGenerateResult:
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
            raise ModelCallError(
                f"API error {exc.status_code}: {exc.message}", retryable=exc.status_code >= 500
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
