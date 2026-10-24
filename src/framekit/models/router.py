"""`conversation` port implementation: registry → model spec → provider config → protocol
adapter.

Every caller goes through the same adapters: the built-ins on the official `openai` and
`anthropic` SDKs, plus any a `ModelRegistry` adds (docs/adapters.md).

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
    TextDelta,
)
from framekit.models.adapters.base import (
    ExternalAIAdapterBase,
)
from framekit.models.catalog import (
    ModelSettings,
    ModelSpec,
)
from framekit.models.registry import (
    DEFAULT_REGISTRY,
    ModelRegistry,
)
from framekit.models.registry import (
    PROTOCOLS as PROTOCOLS,  # re-exported: router.PROTOCOLS predates the registry
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


def create_adapter(provider: str, settings: ModelSettings) -> ExternalAIAdapterBase:
    """Build the adapter for a built-in provider (`DEFAULT_REGISTRY.create_adapter`)."""
    return DEFAULT_REGISTRY.create_adapter(provider, settings)


class ConversationRouter:
    """Serves the `conversation` port from a model registry.

    `registry` (default: `DEFAULT_REGISTRY`, the built-ins) resolves models and builds
    adapters. An `adapter_factory`, if given, builds the adapters instead (tests, or
    adapters built by hand); the registry still resolves the models."""

    def __init__(
        self,
        settings: ModelSettings | None = None,
        adapter_factory: AdapterFactory | None = None,
        *,
        registry: ModelRegistry | None = None,
    ) -> None:
        self._settings = settings or ModelSettings()
        self._registry = registry if registry is not None else DEFAULT_REGISTRY
        self._factory: AdapterFactory = adapter_factory or self._registry.create_adapter
        self._adapters: dict[str, ProviderAdapter] = {}

    @property
    def registry(self) -> ModelRegistry:
        return self._registry

    def resolve(self, model_id: str | None) -> ModelSpec:
        """The spec for `model_id`, or for the settings' default model when it's None."""
        return self._registry.resolve(model_id or self._settings.default_model)

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
