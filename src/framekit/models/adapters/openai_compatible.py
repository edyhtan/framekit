"""OpenAI Chat Completions protocol (official `openai` SDK, async).

Serves every OpenAI-compatible provider (Hive, DeepSeek's own API, …); providers differ
only in their `ProviderConfig`. Streaming is used where the provider documents it
(Hive: server-sent events, usage always in the final chunk).
"""

from typing import Any, ClassVar

import openai

from clipsearch_adapters.external_connections.adapters.external_ai_adapter_base import (
    ExternalAIAdapterBase,
    Reply,
)
from clipsearch_adapters.external_connections.catalog import ModelSpec
from clipsearch_domain.common.ports import TextGenerateRequest


def _reasoning_delta(delta: Any) -> str | None:
    # Not part of the OpenAI schema; the SDK keeps unknown fields in `model_extra`.
    return (getattr(delta, "model_extra", None) or {}).get("reasoning_content")


def _reasoning_tokens(usage: Any) -> int | None:
    details = getattr(usage, "completion_tokens_details", None)
    return getattr(details, "reasoning_tokens", None) if details else None


class OpenAICompatibleAdapter(ExternalAIAdapterBase):
    sdk: ClassVar = openai
    adapter_version: ClassVar[str] = "0.4.0"

    @classmethod
    def make_client(cls, api_key: str, base_url: str | None, timeout_s: float) -> Any:
        return openai.AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout_s)

    def build_params(
        self, spec: ModelSpec, request: TextGenerateRequest, max_tokens: int
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": spec.model_id,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": request.prompt}],
            "stream": self.config.stream,
            **spec.request_extras,
        }
        if self.config.extra_headers:
            params["extra_headers"] = dict(self.config.extra_headers)
        return params

    async def call(self, spec: ModelSpec, params: dict[str, Any]) -> Reply:
        response = await self._client.chat.completions.create(**params)
        return await self._read_stream(response) if self.config.stream else _read(response)

    @staticmethod
    async def _read_stream(stream: Any) -> Reply:
        reply, parts, thoughts = Reply(), [], []
        http_response = getattr(stream, "response", None)
        if http_response is not None:
            reply.request_id = http_response.headers.get("x-request-id")
        try:
            async for chunk in stream:
                OpenAICompatibleAdapter._read_chunk(reply, chunk, parts, thoughts)
        finally:
            # Return the connection even if the stream fails mid-way.
            close = getattr(stream, "close", None)
            if close is not None:
                await close()
        reply.text = "".join(parts)
        reply.reasoning = "".join(thoughts)
        return reply

    @staticmethod
    def _read_chunk(reply: Reply, chunk: Any, parts: list[str], thoughts: list[str]) -> None:
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


def _read(completion: Any) -> Reply:
    choice = completion.choices[0] if completion.choices else None
    usage = completion.usage
    return Reply(
        text=(choice.message.content or "") if choice else "",
        reasoning=(_reasoning_delta(choice.message) or "") if choice else "",
        reasoning_tokens=_reasoning_tokens(usage),
        model_id=completion.model,
        stop_reason=choice.finish_reason if choice else None,
        input_tokens=usage.prompt_tokens if usage else 0,
        output_tokens=usage.completion_tokens if usage else 0,
        request_id=getattr(completion, "_request_id", None) or completion.id,
    )
