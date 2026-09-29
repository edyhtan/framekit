"""`text_generate` port implementation that routes each request to its provider's adapter.

Provider adapters are created lazily, so a missing key for one provider only fails the
requests that need that provider.
"""

from collections.abc import Callable
from typing import Protocol

from clipsearch_adapters.models.anthropic_text import AnthropicTextGenerator
from clipsearch_adapters.models.catalog import (
    CATALOG,
    ModelSpec,
    TextModelSettings,
    get_text_model_settings,
)
from clipsearch_adapters.models.openai_compatible import (
    DEEPSEEK,
    HIVE,
    OpenAICompatibleGenerator,
)
from clipsearch_domain.common.ports import ModelCallError, TextGenerateRequest, TextGenerateResult


class ProviderAdapter(Protocol):
    async def generate(
        self, spec: ModelSpec, request: TextGenerateRequest
    ) -> TextGenerateResult: ...


ProviderFactory = Callable[[TextModelSettings], ProviderAdapter]

PROVIDERS: dict[str, ProviderFactory] = {
    "hive": lambda settings: OpenAICompatibleGenerator.from_environment(HIVE, settings),
    "deepseek": lambda settings: OpenAICompatibleGenerator.from_environment(DEEPSEEK, settings),
    "anthropic": AnthropicTextGenerator.from_environment,
}


class TextGenerateRouter:
    def __init__(
        self,
        settings: TextModelSettings,
        providers: dict[str, ProviderFactory] | None = None,
    ) -> None:
        self._settings = settings
        self._factories = providers or PROVIDERS
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
            factory = self._factories.get(provider)
            if factory is None:
                raise ModelCallError(f"no adapter for provider {provider!r}", retryable=False)
            self._adapters[provider] = factory(self._settings)
        return self._adapters[provider]

    async def generate(self, request: TextGenerateRequest) -> TextGenerateResult:
        spec = self.resolve(request.model_id)
        return await self._adapter(spec.provider).generate(spec, request)
