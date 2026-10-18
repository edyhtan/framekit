"""`conversation` port implementation: catalog → provider config → adapter family.

Two adapter families serve the same port: the SDK family (official `openai` /
`anthropic` SDKs; corpus work, the ai_test job, /prototype/chat) and the LangChain family
(conversational search roles). A request picks one with `ConversationRequest.family`.

Lifetime: **one router per job attempt or chat turn**. It creates adapters (and their
clients and connections) lazily per (provider, family), and `aclose` closes them all when
the attempt or turn ends.
A missing key for one provider only fails the requests that need that provider.
"""

import logging
from collections.abc import AsyncIterator, Callable
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
from clipsearch_domain.common.ports import (
    ConversationRequest,
    ConversationResult,
    ModelCallError,
    TextDelta,
)


class ProviderAdapter(Protocol):
    async def generate(
        self, spec: ModelSpec, request: ConversationRequest
    ) -> ConversationResult: ...

    def stream(
        self, spec: ModelSpec, request: ConversationRequest
    ) -> AsyncIterator[TextDelta | ConversationResult]: ...

    async def aclose(self) -> None: ...


# (provider name, settings, adapter family) -> adapter
AdapterFactory = Callable[[str, TextModelSettings, str | None], ProviderAdapter]

log = logging.getLogger(__name__)

# Default ("SDK") family: which template (protocol adapter) serves each wire protocol
# named in a ProviderConfig. Used for corpus work, the ai_test job and /prototype/chat.
PROTOCOLS: dict[str, type[ExternalAIAdapterBase]] = {
    "openai": OpenAICompatibleAdapter,
    "anthropic": AnthropicAdapter,
}

# Adapter families a request can ask for (`ConversationRequest.family`); None is the SDK
# family above. "langchain" serves the conversational search roles.
FAMILIES = (None, "langchain")


def create_adapter(
    provider: str, settings: TextModelSettings, family: str | None = None
) -> ExternalAIAdapterBase:
    config = PROVIDERS.get(provider)
    if config is None:
        raise ModelCallError(f"no provider configured as {provider!r}", retryable=False)
    if family is None:
        return PROTOCOLS[config.protocol].from_environment(config, settings)
    if family == "langchain":
        # Imported here so processes that only use the SDK family never load LangChain.
        from clipsearch_adapters.external_connections.langchain.chat_adapter import (
            LangChainChatAdapter,
        )

        return LangChainChatAdapter.from_environment(config, settings)
    raise ModelCallError(f"unknown adapter family {family!r}", retryable=False)


class ConversationRouter:
    def __init__(
        self, settings: TextModelSettings, adapter_factory: AdapterFactory = create_adapter
    ) -> None:
        self._settings = settings
        self._factory = adapter_factory
        self._adapters: dict[tuple[str, str | None], ProviderAdapter] = {}

    @classmethod
    def from_environment(cls) -> "ConversationRouter":
        return cls(get_text_model_settings())

    @classmethod
    def for_search(cls) -> "ConversationRouter":
        """A router for search turns: their roles set per-role token budgets (spec §8.4),
        capped by `search_max_tokens_cap` instead of the ai_test probe cap."""
        settings = get_text_model_settings()
        cap = settings.search_max_tokens_cap
        return cls(settings.model_copy(update={"max_tokens_cap": cap}))

    def resolve(self, model_id: str | None) -> ModelSpec:
        chosen = model_id or self._settings.default_model
        if chosen not in CATALOG:
            raise ModelCallError(f"model {chosen!r} is not in the catalog", retryable=False)
        return CATALOG[chosen]

    def _adapter(self, provider: str, family: str | None) -> ProviderAdapter:
        key = (provider, family)
        if key not in self._adapters:
            self._adapters[key] = self._factory(provider, self._settings, family)
        return self._adapters[key]

    async def generate(self, request: ConversationRequest) -> ConversationResult:
        spec = self.resolve(request.model_id)
        return await self._adapter(spec.provider, request.family).generate(spec, request)

    async def stream(
        self, request: ConversationRequest
    ) -> AsyncIterator[TextDelta | ConversationResult]:
        spec = self.resolve(request.model_id)
        async for piece in self._adapter(spec.provider, request.family).stream(spec, request):
            yield piece

    async def aclose(self) -> None:
        """Close every adapter this router opened. Safe to call more than once."""
        adapters, self._adapters = list(self._adapters.values()), {}
        for adapter in adapters:
            try:
                await adapter.aclose()
            except Exception:  # closing must never mask the job's own outcome
                log.warning("closing a model adapter failed", exc_info=True)

    async def __aenter__(self) -> "ConversationRouter":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()
