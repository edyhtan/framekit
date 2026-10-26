"""Model access: the catalog (providers and models as data), the model registry, the
router that serves the `conversation` port, the protocol adapters, and the response cache.

    async with ConversationRouter(ModelSettings(default_model=...)) as router:
        result = await router.generate(ConversationRequest("Hello", max_tokens=64))

Your own models go in as `ModelSource`s: one structure for the model, where it's served,
its credential and, optionally, your adapter (`ConversationRouter(settings, sources=[...])`).
A `ModelRegistry` (`DEFAULT_REGISTRY.extend(...)`) is the lower-level container; the adapter
template is in `framekit.models.adapters`. Guide: docs/adapters.md.
"""

from framekit.models.catalog import (
    CATALOG,
    PROVIDERS,
    LongPromptPrices,
    ModelSettings,
    ModelSpec,
    ProviderConfig,
    ReasoningControl,
    catalog_listing,
    estimate_cost,
)
from framekit.models.registry import DEFAULT_REGISTRY, ModelRegistry
from framekit.models.router import ConversationRouter, create_adapter
from framekit.models.sources import ModelSource

__all__ = [
    "CATALOG",
    "DEFAULT_REGISTRY",
    "PROVIDERS",
    "ConversationRouter",
    "LongPromptPrices",
    "ModelRegistry",
    "ModelSettings",
    "ModelSource",
    "ModelSpec",
    "ProviderConfig",
    "ReasoningControl",
    "catalog_listing",
    "create_adapter",
    "estimate_cost",
]
