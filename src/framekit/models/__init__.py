"""Model access: the catalog (providers and models as data), the model registry, the
router that serves the `conversation` port, the protocol adapters, and the response cache.

    async with ConversationRouter(ModelSettings(default_model=...)) as router:
        result = await router.generate(ConversationRequest("Hello", max_tokens=64))

Your own providers, models and adapters go into a registry (`DEFAULT_REGISTRY.extend(...)`)
that you pass to the router (`ConversationRouter(settings, registry=...)`); the adapter
template is in `framekit.models.adapters`. Guide: docs/adapters.md.
"""

from framekit.models.catalog import (
    CATALOG,
    PROVIDERS,
    ModelSettings,
    ModelSpec,
    ProviderConfig,
    ReasoningControl,
    catalog_listing,
    estimate_cost,
)
from framekit.models.registry import DEFAULT_REGISTRY, ModelRegistry
from framekit.models.router import ConversationRouter, create_adapter

__all__ = [
    "CATALOG",
    "DEFAULT_REGISTRY",
    "PROVIDERS",
    "ConversationRouter",
    "ModelRegistry",
    "ModelSettings",
    "ModelSpec",
    "ProviderConfig",
    "ReasoningControl",
    "catalog_listing",
    "create_adapter",
    "estimate_cost",
]
