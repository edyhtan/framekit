"""`text_generate` port implementation: catalog → provider config → protocol adapter.

Adapters are created lazily, so a missing key for one provider only fails the requests
that need that provider.
"""

from collections.abc import Callable
from typing import Protocol

from clipsearch_adapters.models.catalog import (
    CATALOG,
    ModelSpec,
    TextModelSettings,
    get_text_model_settings,
)
from clipsearch_adapters.models.providers import create_adapter
from clipsearch_domain.common.ports import ModelCallError, TextGenerateRequest, TextGenerateResult


class ProviderAdapter(Protocol):
    async def generate(
        self, spec: ModelSpec, request: TextGenerateRequest
    ) -> TextGenerateResult: ...


AdapterFactory = Callable[[str, TextModelSettings], ProviderAdapter]


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
