"""`conversation` port implementation: model source or registry → model spec → provider
config → protocol adapter.

Every caller goes through the same adapters: the built-ins on the official `openai` and
`anthropic` SDKs, plus any a `ModelSource` or a `ModelRegistry` adds (docs/adapters.md).

Lifetime: **one router per unit of work** (a task attempt, a chat turn). It creates
adapters (and their clients and connections) lazily per provider, and `aclose` closes
them all when the unit of work ends.
A missing key for one provider only fails the requests that need that provider.
"""

import dataclasses
import logging
from collections.abc import AsyncIterator, Callable, Iterable
from typing import Protocol

# The `X as X` imports are re-exports: names this module had before the registry existed,
# still importable from here. Rebinding them here (`router.CATALOG = {...}`) no longer
# changes what a router uses: the default registry holds the dicts themselves
# (docs/adapters.md, "The registry").
from framekit.contracts.conversation import (
    ConversationRequest,
    ConversationResult,
    TextDelta,
)
from framekit.contracts.conversation import ModelCallError as ModelCallError
from framekit.models.adapters.anthropic import AnthropicAdapter as AnthropicAdapter
from framekit.models.adapters.base import (
    ExternalAIAdapterBase,
)
from framekit.models.adapters.openai_compatible import (
    OpenAICompatibleAdapter as OpenAICompatibleAdapter,
)
from framekit.models.catalog import CATALOG as CATALOG
from framekit.models.catalog import PROVIDERS as PROVIDERS
from framekit.models.catalog import (
    ModelSettings,
    ModelSpec,
)
from framekit.models.registry import (
    DEFAULT_REGISTRY,
    ModelRegistry,
)
from framekit.models.registry import PROTOCOLS as PROTOCOLS
from framekit.models.sources import ModelSource, ResolvedSource, resolve_sources


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
    """Build the adapter for a built-in provider (`DEFAULT_REGISTRY.create_adapter`). It
    knows only the built-ins: for a registry of your own, use its `create_adapter`."""
    return DEFAULT_REGISTRY.create_adapter(provider, settings)


class ConversationRouter:
    """Serves the `conversation` port from model sources and a model registry.

    `sources` (`ModelSource`s) are the usual way to add models: each names a model, where
    it's served, its credential and, optionally, an adapter. A request's `model_id` (or the
    settings' `default_model`) names a source; anything else is resolved by the `registry`
    (default: `DEFAULT_REGISTRY`, the built-in catalog, keys from the environment). Sources
    are checked when the router is built: every problem is reported in one `ValueError`.

    An `adapter_factory`, if given, builds the adapters for registry models instead (tests,
    or adapters built by hand); sources always build their own. The factory is called once
    for every provider the router uses, so a factory for one adapter of your own returns
    `registry.create_adapter(provider, settings)` for the others."""

    def __init__(
        self,
        settings: ModelSettings | None = None,
        adapter_factory: AdapterFactory | None = None,
        *,
        registry: ModelRegistry | None = None,
        sources: Iterable[ModelSource] = (),
    ) -> None:
        self._settings = settings or ModelSettings()
        self._registry = registry if registry is not None else DEFAULT_REGISTRY
        self._factory: AdapterFactory = (
            adapter_factory if adapter_factory is not None else self._registry.create_adapter
        )
        self._sources: dict[str, ResolvedSource] = resolve_sources(
            sources,
            providers=self._registry.providers,
            models=self._registry.models,
            adapters=self._registry.adapters,
        )
        self._adapters: dict[str, ProviderAdapter] = {}

    @property
    def registry(self) -> ModelRegistry:
        """The registry this router resolves models with (read-only: pass another one to
        the constructor)."""
        return self._registry

    @property
    def sources(self) -> dict[str, ResolvedSource]:
        """The sources this router serves, checked, by name (a copy)."""
        return dict(self._sources)

    def resolve(self, model_id: str | None) -> ModelSpec:
        """The spec for `model_id` (a source's name, or a registry model id), or for the
        settings' default model when it's None."""
        name = model_id or self._settings.default_model
        source = self._sources.get(name)
        return source.spec if source is not None else self._registry.resolve(name)

    def cache_identity(self, model_id: str | None) -> str:
        """What the response cache keys a call by: the model id sent, plus the source's
        public fingerprint for a source, so two hosts serving the same model id never share
        cached answers. Registry models keep their plain model id (unchanged keys)."""
        name = model_id or self._settings.default_model
        source = self._sources.get(name)
        if source is not None:
            return f"{source.spec.model_id}@{source.fingerprint}"
        return self._registry.resolve(name).model_id

    def _adapter_for(self, name: str) -> tuple[ModelSpec, ProviderAdapter, str | None]:
        source = self._sources.get(name)
        if source is not None:
            key = f"source:{name}"
            if key not in self._adapters:
                self._adapters[key] = source.build(self._settings)
            return source.spec, self._adapters[key], name
        spec = self._registry.resolve(name)
        if spec.provider not in self._adapters:
            self._adapters[spec.provider] = self._factory(spec.provider, self._settings)
        return spec, self._adapters[spec.provider], None

    async def generate(self, request: ConversationRequest) -> ConversationResult:
        name = request.model_id or self._settings.default_model
        spec, adapter, source = self._adapter_for(name)
        result = await adapter.generate(spec, request)
        return dataclasses.replace(result, source=source) if source else result

    async def stream(
        self, request: ConversationRequest
    ) -> AsyncIterator[TextDelta | ConversationResult]:
        name = request.model_id or self._settings.default_model
        spec, adapter, source = self._adapter_for(name)
        async for piece in adapter.stream(spec, request):
            if source and isinstance(piece, ConversationResult):
                piece = dataclasses.replace(piece, source=source)
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
