"""`conversation` port implementation: catalog → provider config → protocol adapter.

Every caller goes through the same adapters, built on the official `openai` and
`anthropic` SDKs.

Lifetime: **one router per unit of work** (a task attempt, a chat turn). It creates
adapters (and their clients and connections) lazily per provider, and `aclose` closes
them all when the unit of work ends.
A missing key for one provider only fails the requests that need that provider.
"""

import logging
from collections.abc import AsyncIterator, Callable
from typing import Protocol

from framekit.contracts.conversation import (
    ConversationRequest,
    ConversationResult,
    ModelCallError,
    TextDelta,
)
from framekit.models.adapters.anthropic import AnthropicAdapter
from framekit.models.adapters.base import (
    ExternalAIAdapterBase,
)
from framekit.models.adapters.openai_compatible import (
    OpenAICompatibleAdapter,
)
from framekit.models.catalog import (
    CATALOG,
    PROVIDERS,
    ModelSettings,
    ModelSpec,
)


class ProviderAdapter(Protocol):
    async def generate(
        self, spec: ModelSpec, request: ConversationRequest
    ) -> ConversationResult: ...

    def stream(
        self, spec: ModelSpec, request: ConversationRequest
    ) -> AsyncIterator[TextDelta | ConversationResult]: ...

    async def aclose(self) -> None: ...


# (provider name, settings) -> adapter
AdapterFactory = Callable[[str, ModelSettings], ProviderAdapter]

log = logging.getLogger(__name__)

# Which template (protocol adapter) serves each wire protocol named in a ProviderConfig.
PROTOCOLS: dict[str, type[ExternalAIAdapterBase]] = {
    "openai": OpenAICompatibleAdapter,
    "anthropic": AnthropicAdapter,
}

def create_adapter(provider: str, settings: ModelSettings) -> ExternalAIAdapterBase:
    config = PROVIDERS.get(provider)
    if config is None:
        raise ModelCallError(f"no provider configured as {provider!r}", retryable=False)
    return PROTOCOLS[config.protocol].from_environment(config, settings)


class ConversationRouter:
    def __init__(
        self,
        settings: ModelSettings | None = None,
        adapter_factory: AdapterFactory = create_adapter,
    ) -> None:
        self._settings = settings or ModelSettings()
        self._factory = adapter_factory
        self._adapters: dict[str, ProviderAdapter] = {}

    def resolve(self, model_id: str | None) -> ModelSpec:
        chosen = model_id or self._settings.default_model
        if chosen not in CATALOG:
            raise ModelCallError(f"model {chosen!r} is not in the catalog", retryable=False)
        return CATALOG[chosen]

    def _adapter(self, provider: str) -> ProviderAdapter:
        if provider not in self._adapters:
            self._adapters[provider] = self._factory(provider, self._settings)
        return self._adapters[provider]

    async def generate(self, request: ConversationRequest) -> ConversationResult:
        spec = self.resolve(request.model_id)
        return await self._adapter(spec.provider).generate(spec, request)

    async def stream(
        self, request: ConversationRequest
    ) -> AsyncIterator[TextDelta | ConversationResult]:
        spec = self.resolve(request.model_id)
        async for piece in self._adapter(spec.provider).stream(spec, request):
            yield piece

    async def aclose(self) -> None:
        """Close every adapter this router opened. Safe to call more than once."""
        adapters, self._adapters = list(self._adapters.values()), {}
        for adapter in adapters:
            try:
                await adapter.aclose()
            except Exception:  # closing must never mask the caller's own outcome
                log.warning("closing a model adapter failed", exc_info=True)

    async def __aenter__(self) -> "ConversationRouter":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()
