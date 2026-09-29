"""Anthropic Messages protocol (official `anthropic` SDK, async).

Claude is called through Anthropic's own SDK, not an OpenAI-compatible shim, so typed
content blocks, thinking, effort, beta features and refusal fallbacks keep working.
"""

from typing import Any, ClassVar

import anthropic

from clipsearch_adapters.external_connections.adapters.external_ai_adapter_base import (
    ExternalAIAdapterBase,
    Reply,
)
from clipsearch_adapters.external_connections.catalog import ModelSpec
from clipsearch_domain.common.ports import TextGenerateRequest


class AnthropicAdapter(ExternalAIAdapterBase):
    sdk: ClassVar = anthropic
    adapter_version: ClassVar[str] = "0.4.0"

    @classmethod
    def make_client(cls, api_key: str, base_url: str | None, timeout_s: float) -> Any:
        kwargs: dict[str, Any] = {"api_key": api_key, "timeout": timeout_s}
        if base_url:
            kwargs["base_url"] = base_url
        return anthropic.AsyncAnthropic(**kwargs)

    def build_params(
        self, spec: ModelSpec, request: TextGenerateRequest, max_tokens: int
    ) -> dict[str, Any]:
        return {
            "model": spec.model_id,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": request.prompt}],
            **spec.request_extras,
        }

    async def call(self, spec: ModelSpec, params: dict[str, Any]) -> Reply:
        if spec.betas:
            message = await self._client.beta.messages.create(betas=list(spec.betas), **params)
        else:
            message = await self._client.messages.create(**params)
        return Reply(
            text="".join(b.text for b in message.content if b.type == "text"),
            # Thinking text is empty unless the request asks for a summarized display.
            reasoning="".join(
                getattr(b, "thinking", "") or "" for b in message.content if b.type == "thinking"
            ),
            model_id=message.model,
            stop_reason=message.stop_reason,
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
            request_id=getattr(message, "_request_id", None),
        )
