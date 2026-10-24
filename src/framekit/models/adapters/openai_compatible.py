"""OpenAI Chat Completions protocol (official `openai` SDK, async).

Serves every OpenAI-compatible provider (Hive, DeepSeek's own API, …); providers differ
only in their `ProviderConfig`. Streaming is used where the provider documents it
(Hive: server-sent events, usage always in the final chunk).

Structured output by `function_calling` sends the schema as the one tool the model must
call; the tool call's arguments are the reply's JSON. Those calls are never streamed:
the JSON is only usable whole.
"""

import time
from collections.abc import AsyncIterator
from typing import Any, ClassVar

import openai

from framekit.contracts.conversation import ConversationRequest, ModelCallError, TextDelta
from framekit.contracts.embedding import EmbeddingRequest, EmbeddingResult
from framekit.models.adapters.base import (
    ExternalAIAdapterBase,
    Reply,
    conversation,
    image_data_url,
    merge_fields,
    reasoning_active,
    reasoning_fields,
    sampling_fields,
    structured_mode,
    system_text,
)
from framekit.models.catalog import ModelSpec, estimate_cost


def _reasoning_delta(delta: Any) -> str | None:
    # Not part of the OpenAI schema; the SDK keeps unknown fields in `model_extra`.
    return (getattr(delta, "model_extra", None) or {}).get("reasoning_content")


def _reasoning_tokens(usage: Any) -> int | None:
    details = getattr(usage, "completion_tokens_details", None)
    return getattr(details, "reasoning_tokens", None) if details else None


# The single tool a `function_calling` request forces the model to call.
TOOL_NAME = "response"


def _response_format(request: ConversationRequest, mode: str | None) -> dict[str, Any]:
    if mode == "function_calling":
        schema = dict(request.response_schema or {})
        return {
            "tools": [
                {"type": "function", "function": {"name": TOOL_NAME, "parameters": schema}}
            ],
            "tool_choice": {"type": "function", "function": {"name": TOOL_NAME}},
        }
    if mode == "json_schema":
        schema = dict(request.response_schema or {})
        return {
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": schema},
            }
        }
    if mode == "json_mode":
        return {"response_format": {"type": "json_object"}}
    return {}


class OpenAICompatibleAdapter(ExternalAIAdapterBase):
    sdk: ClassVar = openai
    adapter_version: ClassVar[str] = "0.7.0"
    structured_modes: ClassVar[frozenset[str]] = frozenset(
        {"json_schema", "json_mode", "function_calling", "prompted"}
    )

    @classmethod
    def make_client(cls, api_key: str, base_url: str | None, timeout_s: float) -> Any:
        return openai.AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout_s)

    def build_params(
        self, spec: ModelSpec, request: ConversationRequest, max_tokens: int
    ) -> dict[str, Any]:
        mode = structured_mode(spec, request, self.structured_modes)
        system = system_text(request, mode)
        messages: list[dict[str, Any]] = list(conversation(request))
        if request.images:  # images first, then the prompt's text, in one user message
            messages[-1] = {
                "role": "user",
                "content": [
                    *({"type": "image_url", "image_url": {"url": image_data_url(i)}}
                      for i in request.images),
                    {"type": "text", "text": request.prompt},
                ],
            }
        if system:
            messages = [{"role": "system", "content": system}, *messages]
        params = merge_fields(
            {
                "model": spec.model_id,
                "max_tokens": max_tokens,
                "messages": messages,
                "stream": self.config.stream and mode != "function_calling",
            },
            spec.request_extras,
            reasoning_fields(spec, reasoning_active(spec, request)),
            _response_format(request, mode),
            sampling_fields(request),
        )
        if self.config.extra_headers:
            params["extra_headers"] = dict(self.config.extra_headers)
        return params

    async def embed(self, spec: ModelSpec, request: EmbeddingRequest) -> EmbeddingResult:
        """Texts to vectors through the OpenAI-compatible `/embeddings` endpoint."""
        started = time.perf_counter()
        try:
            response = await self.client.embeddings.create(
                model=spec.model_id, input=list(request.texts)
            )
        except ModelCallError:
            raise
        except Exception as exc:
            mapped = self.map_error(exc)
            if mapped is None:
                raise
            raise mapped from exc
        vectors = tuple(
            tuple(float(x) for x in item.embedding)
            for item in sorted(response.data, key=lambda item: item.index)
        )
        if len(vectors) != len(request.texts):
            raise ModelCallError(
                f"the provider returned {len(vectors)} vectors for {len(request.texts)} texts",
                retryable=False,
            )
        tokens = getattr(getattr(response, "usage", None), "prompt_tokens", 0) or 0
        return EmbeddingResult(
            provider=self.config.name,
            model_id=getattr(response, "model", None) or spec.model_id,
            vectors=vectors,
            dimension=len(vectors[0]) if vectors else 0,
            input_tokens=tokens,
            latency_ms=round((time.perf_counter() - started) * 1000),
            cost_usd=estimate_cost(spec, tokens, 0),
            adapter_version=self.adapter_version,
        )

    async def call(self, spec: ModelSpec, params: dict[str, Any]) -> Reply:
        if not params["stream"]:
            return _read(await self._client.chat.completions.create(**params))
        reply: Reply | None = None
        async for piece in self.iterate(spec, params):
            if isinstance(piece, Reply):
                reply = piece
        if reply is None:  # a protocol bug, not a provider error (same check as `stream`)
            raise ModelCallError("the adapter produced no final reply", retryable=False)
        return reply

    async def iterate(
        self, spec: ModelSpec, params: dict[str, Any]
    ) -> AsyncIterator[TextDelta | Reply]:
        if not params["stream"]:
            async for piece in super().iterate(spec, params):
                yield piece
            return
        stream = await self._client.chat.completions.create(**params)
        reply, parts, thoughts = Reply(), [], []
        http_response = getattr(stream, "response", None)
        if http_response is not None:
            reply.request_id = http_response.headers.get("x-request-id")
        try:
            async for chunk in stream:
                for delta in self._read_chunk(reply, chunk):
                    (thoughts if delta.kind == "reasoning" else parts).append(delta.text)
                    yield delta
        finally:
            # Return the connection even if the stream fails or the caller stops reading.
            close = getattr(stream, "close", None)
            if close is not None:
                await close()
        reply.text = "".join(parts)
        reply.reasoning = "".join(thoughts)
        yield reply

    @staticmethod
    def _read_chunk(reply: Reply, chunk: Any) -> list[TextDelta]:
        deltas: list[TextDelta] = []
        reply.model_id = reply.model_id or getattr(chunk, "model", None)
        reply.request_id = reply.request_id or getattr(chunk, "id", None)
        for choice in chunk.choices or []:
            if choice.delta and (thought := _reasoning_delta(choice.delta)):
                deltas.append(TextDelta("reasoning", thought))
            if choice.delta and choice.delta.content:
                deltas.append(TextDelta("text", choice.delta.content))
            if choice.finish_reason:
                reply.stop_reason = choice.finish_reason
        if chunk.usage:  # Hive always sends usage in the final chunk
            reply.input_tokens = chunk.usage.prompt_tokens or 0
            reply.output_tokens = chunk.usage.completion_tokens or 0
            reply.reasoning_tokens = _reasoning_tokens(chunk.usage)
        return deltas


def _tool_arguments(message: Any) -> str | None:
    """The JSON arguments of the forced tool call, if the reply made one."""
    for call in getattr(message, "tool_calls", None) or []:
        if call.function and call.function.name == TOOL_NAME:
            return call.function.arguments or ""
    return None


def _read(completion: Any) -> Reply:
    choice = completion.choices[0] if completion.choices else None
    usage = completion.usage
    text = (choice.message.content or "") if choice else ""
    if choice and (arguments := _tool_arguments(choice.message)) is not None:
        text = arguments  # the caller validates this JSON against its schema
    return Reply(
        text=text,
        reasoning=(_reasoning_delta(choice.message) or "") if choice else "",
        reasoning_tokens=_reasoning_tokens(usage),
        model_id=completion.model,
        stop_reason=choice.finish_reason if choice else None,
        input_tokens=usage.prompt_tokens if usage else 0,
        output_tokens=usage.completion_tokens if usage else 0,
        request_id=getattr(completion, "_request_id", None) or completion.id,
    )
