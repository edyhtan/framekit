"""Model access: the catalog (providers and models as data), the router that serves the
`conversation` port, the protocol adapters, and the response cache.

    async with ConversationRouter(ModelSettings(default_model=...)) as router:
        result = await router.generate(ConversationRequest("Hello", max_tokens=64))
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
from framekit.models.router import ConversationRouter, create_adapter

__all__ = [
    "CATALOG",
    "PROVIDERS",
    "ConversationRouter",
    "ModelSettings",
    "ModelSpec",
    "ProviderConfig",
    "ReasoningControl",
    "catalog_listing",
    "create_adapter",
    "estimate_cost",
]
