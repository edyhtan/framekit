"""`text_generate` port implementation: catalog → provider config → protocol adapter.

Lifetime: **one router per job attempt**. It creates provider adapters (and their SDK
clients and connections) lazily, and `aclose` closes them all when the attempt ends.
A missing key for one provider only fails the requests that need that provider.
"""

import logging
from collections.abc import Callable
from typing import Protocol

from clipsearch_adapters.external_connections.adapters.anthropic_text import AnthropicAdapter
from clipsearch_adapters.external_connections.adapters.external_ai_adapter_base import (
    ExternalAIAdapterBase,
)
from clipsearch_adapters.external_connections.adapters.openai_compatible import (
    OpenAICompatibleAdapter,
)
from clipsearch_adapters.external_connections.catalog import (
    CATALOG,
    PROVIDERS,
    ModelSpec,
    TextModelSettings,
    get_text_model_settings,
)
from clipsearch_domain.common.ports import ModelCallError, TextGenerateRequest, TextGenerateResult


class ProviderAdapter(Protocol):
    async def generate(
        self, spec: ModelSpec, request: TextGenerateRequest
    ) -> TextGenerateResult: ...

    async def aclose(self) -> None: ...


AdapterFactory = Callable[[str, TextModelSettings], ProviderAdapter]

log = logging.getLogger(__name__)

# Which template (protocol adapter) serves each wire protocol named in a ProviderConfig.
PROTOCOLS: dict[str, type[ExternalAIAdapterBase]] = {
    "openai": OpenAICompatibleAdapter,
    "anthropic": AnthropicAdapter,
}


def create_adapter(provider: str, settings: TextModelSettings) -> ExternalAIAdapterBase:
    config = PROVIDERS.get(provider)
    if config is None:
        raise ModelCallError(f"no provider configured as {provider!r}", retryable=False)
    return PROTOCOLS[config.protocol].from_environment(config, settings)


class TextGenerateRouter:
    def __init__(
        self, settings: TextModelSettings, adapter_factory: AdapterFactory = create_adapter
    ) -> None:
        self._settings = settings
        self._factory = adapter_factory
        self._adapters: dict[str, ProviderAdapter] = {}

    @classmethod
    def from_environment(cls) -> "TextGenerateRouter":
        return cls(get_text_model_settings())

    def resolve(self, model_id: str | None) -> ModelSpec:
        chosen = model_id or self._settings.default_model
        if chosen not in CATALOG:
            raise ModelCallError(f"model {chosen!r} is not in the catalog", retryable=False)
        return CATALOG[chosen]

    def _adapter(self, provider: str) -> ProviderAdapter:
        if provider not in self._adapters:
            self._adapters[provider] = self._factory(provider, self._settings)
        return self._adapters[provider]

    async def generate(self, request: TextGenerateRequest) -> TextGenerateResult:
        spec = self.resolve(request.model_id)
        return await self._adapter(spec.provider).generate(spec, request)

    async def aclose(self) -> None:
        """Close every adapter this router opened. Safe to call more than once."""
        adapters, self._adapters = list(self._adapters.values()), {}
        for adapter in adapters:
            try:
                await adapter.aclose()
            except Exception:  # closing must never mask the job's own outcome
                log.warning("closing a model adapter failed", exc_info=True)

    async def __aenter__(self) -> "TextGenerateRouter":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()
