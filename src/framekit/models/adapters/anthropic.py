"""Anthropic Messages protocol (official `anthropic` SDK, async).

Claude is called through Anthropic's own SDK, not an OpenAI-compatible shim, so typed
content blocks, thinking, effort, beta features and refusal fallbacks keep working.
"""

import base64
from typing import Any, ClassVar

import anthropic

from framekit.contracts.conversation import ConversationRequest
from framekit.models.adapters.base import (
    ExternalAIAdapterBase,
    Reply,
    conversation,
    merge_fields,
    reasoning_active,
    reasoning_fields,
    sampling_fields,
    structured_mode,
    system_text,
)
from framekit.models.catalog import ModelSpec


class AnthropicAdapter(ExternalAIAdapterBase):
    sdk: ClassVar = anthropic
    adapter_version: ClassVar[str] = "0.6.0"

    @classmethod
    def make_client(cls, api_key: str, base_url: str | None, timeout_s: float) -> Any:
        kwargs: dict[str, Any] = {"api_key": api_key, "timeout": timeout_s}
        if base_url:
            kwargs["base_url"] = base_url
        return anthropic.AsyncAnthropic(**kwargs)

    def build_params(
        self, spec: ModelSpec, request: ConversationRequest, max_tokens: int
    ) -> dict[str, Any]:
        # Structured output is "prompted" here: the schema goes into the system prompt.
        system = system_text(request, structured_mode(spec, request, self.structured_modes))
        messages: list[dict[str, Any]] = list(conversation(request))
        if request.images:  # image blocks first, then the prompt's text (Anthropic's advice)
            messages[-1] = {
                "role": "user",
                "content": [
                    *(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": image.media_type,
                                "data": base64.b64encode(image.data).decode("ascii"),
                            },
                        }
                        for image in request.images
                    ),
                    {"type": "text", "text": request.prompt},
                ],
            }
        return merge_fields(
            {"model": spec.model_id, "max_tokens": max_tokens, "messages": messages},
            {"system": system} if system else {},
            spec.request_extras,
            reasoning_fields(spec, reasoning_active(spec, request)),
            sampling_fields(request, seed=False),  # the Messages API has no seed
        )

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
