"""The model registry: the providers, models and protocol adapters a router may use.

`DEFAULT_REGISTRY` serves the built-ins (`PROVIDERS`, `CATALOG` and `PROTOCOLS`). Add your
own with `extend`, which returns a new registry and leaves the default untouched:

    registry = DEFAULT_REGISTRY.extend(
        providers=[ProviderConfig("local", "openai", base_url="http://127.0.0.1:8000/v1")],
        models=[ModelSpec("local", "my-model", "My model", Decimal(0), Decimal(0))],
    )
    async with ConversationRouter(settings, registry=registry) as router:
        ...

A registry is checked when it's built (every model's provider is configured, every
provider's protocol has an adapter), and reports every problem at once. Guide:
docs/adapters.md.
"""

from collections.abc import Callable, Iterable, Mapping
from types import MappingProxyType
from typing import Any

from framekit.contracts.conversation import ModelCallError
from framekit.models.adapters.anthropic import AnthropicAdapter
from framekit.models.adapters.base import ExternalAIAdapterBase
from framekit.models.adapters.openai_compatible import OpenAICompatibleAdapter
from framekit.models.catalog import (
    CATALOG,
    PROVIDERS,
    ModelSettings,
    ModelSpec,
    ProviderConfig,
    model_listing,
)

AdapterClass = type[ExternalAIAdapterBase]

# Which template (protocol adapter) serves each wire protocol named in a ProviderConfig:
# the built-ins. Add protocols with ModelRegistry.extend(adapters=...), not here.
PROTOCOLS: dict[str, AdapterClass] = {
    "openai": OpenAICompatibleAdapter,
    "anthropic": AnthropicAdapter,
}


class ModelRegistry:
    """Providers (name → `ProviderConfig`), models (model_id → `ModelSpec`) and protocol
    adapters (protocol → adapter class).

    Immutable: `extend` returns a new registry. Built from scratch, a registry holds only
    what it's given (no built-ins):

        ModelRegistry(providers=[...], models=[...], adapters={"my-protocol": MyAdapter})
    """

    __slots__ = ("_adapters", "_models", "_providers")
    _providers: Mapping[str, ProviderConfig]
    _models: Mapping[str, ModelSpec]
    _adapters: Mapping[str, AdapterClass]

    def __init__(
        self,
        *,
        providers: Iterable[ProviderConfig] = (),
        models: Iterable[ModelSpec] = (),
        adapters: Mapping[str, AdapterClass] | None = None,
    ) -> None:
        problems: list[str] = []
        self._freeze(
            _merged_providers({}, providers, override=False, problems=problems),
            _merged_models({}, models, override=False, problems=problems),
            _merged_adapters({}, adapters, override=False, problems=problems),
            problems,
        )

    @classmethod
    def _over(
        cls,
        providers: dict[str, ProviderConfig],
        models: dict[str, ModelSpec],
        adapters: dict[str, AdapterClass],
    ) -> "ModelRegistry":
        """A registry over existing dicts, without copying them: in-place edits to the
        dicts show through. Only for the built-ins (`DEFAULT_REGISTRY`)."""
        registry = cls.__new__(cls)
        registry._freeze(providers, models, adapters, [])
        return registry

    def _freeze(
        self,
        providers: dict[str, ProviderConfig],
        models: dict[str, ModelSpec],
        adapters: dict[str, AdapterClass],
        problems: list[str],
    ) -> None:
        problems = problems + _consistency_problems(providers, models, adapters)
        if problems:
            raise ValueError(
                f"invalid model registry ({len(problems)} problem"
                f"{'s' if len(problems) > 1 else ''}):\n"
                + "\n".join(f"- {problem}" for problem in problems)
            )
        object.__setattr__(self, "_providers", MappingProxyType(providers))
        object.__setattr__(self, "_models", MappingProxyType(models))
        object.__setattr__(self, "_adapters", MappingProxyType(adapters))

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("a ModelRegistry is immutable: extend() returns a new one")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("a ModelRegistry is immutable: extend() returns a new one")

    def __copy__(self) -> "ModelRegistry":
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> "ModelRegistry":
        return self

    def __repr__(self) -> str:
        return (
            f"ModelRegistry(providers={sorted(self._providers)}, "
            f"models={len(self._models)}, adapters={sorted(self._adapters)})"
        )

    @property
    def providers(self) -> Mapping[str, ProviderConfig]:
        """Provider configs by name (read-only)."""
        return self._providers

    @property
    def models(self) -> Mapping[str, ModelSpec]:
        """Model specs by model_id (read-only)."""
        return self._models

    @property
    def adapters(self) -> Mapping[str, AdapterClass]:
        """Adapter classes by wire protocol (read-only)."""
        return self._adapters

    def resolve(self, model_id: str) -> ModelSpec:
        """The spec of a model; an unknown model is a non-retryable `ModelCallError`."""
        spec = self._models.get(model_id)
        if spec is None:
            raise ModelCallError(f"model {model_id!r} is not in the catalog", retryable=False)
        return spec

    def create_adapter(self, provider: str, settings: ModelSettings) -> ExternalAIAdapterBase:
        """Build the adapter for a provider, with its key from the environment. An unknown
        provider or protocol, or a missing key, is a non-retryable `ModelCallError`."""
        config = self._providers.get(provider)
        if config is None:
            raise ModelCallError(f"no provider configured as {provider!r}", retryable=False)
        adapter = self._adapters.get(config.protocol)
        if adapter is None:
            raise ModelCallError(
                f"provider {provider!r} speaks protocol {config.protocol!r}, "
                "which has no adapter",
                retryable=False,
            )
        return adapter.from_environment(config, settings)

    def listing(self, default_model: str | None = None) -> list[dict[str, Any]]:
        """This registry's models, for clients choosing one (no prices)."""
        return model_listing(self._models.values(), default_model)

    def extend(
        self,
        *,
        providers: Iterable[ProviderConfig] = (),
        models: Iterable[ModelSpec] = (),
        adapters: Mapping[str, AdapterClass] | None = None,
        override: bool = False,
    ) -> "ModelRegistry":
        """A new registry: this one's entries plus these. A name that's already here
        raises `ValueError` unless `override=True`, which replaces it in the new registry.

        This registry is unchanged. The new one is a snapshot: later in-place edits of the
        built-in dicts don't reach it."""
        problems: list[str] = []
        registry = type(self).__new__(type(self))
        registry._freeze(
            _merged_providers(self._providers, providers, override=override, problems=problems),
            _merged_models(self._models, models, override=override, problems=problems),
            _merged_adapters(self._adapters, adapters, override=override, problems=problems),
            problems,
        )
        return registry


# The helpers below collect problems instead of raising, so a registry reports them all
# at once.


def _merged_providers(
    base: Mapping[str, ProviderConfig],
    configs: Iterable[ProviderConfig],
    *,
    override: bool,
    problems: list[str],
) -> dict[str, ProviderConfig]:
    return _merged(
        base,
        configs,
        ProviderConfig,
        "provider",
        lambda config: config.name,
        override=override,
        problems=problems,
    )


def _merged_models(
    base: Mapping[str, ModelSpec],
    specs: Iterable[ModelSpec],
    *,
    override: bool,
    problems: list[str],
) -> dict[str, ModelSpec]:
    return _merged(
        base,
        specs,
        ModelSpec,
        "model",
        lambda spec: spec.model_id,
        override=override,
        problems=problems,
    )


def _merged[T](
    base: Mapping[str, T],
    items: Iterable[T],
    kind: type[T],
    label: str,
    name_of: Callable[[T], str],
    *,
    override: bool,
    problems: list[str],
) -> dict[str, T]:
    """`base` plus `items`, keyed by `name_of(item)`."""
    merged = dict(base)
    if isinstance(items, kind):  # e.g. providers=config instead of providers=[config]
        problems.append(f"{label}s must be a list of {kind.__name__}s, not a single one")
        return merged
    given: set[str] = set()
    for item in items:
        if not isinstance(item, kind):
            problems.append(f"{label}s: expected a {kind.__name__}, got {type(item).__name__}")
            continue
        name = name_of(item)
        if name in given:
            problems.append(f"{label} {name!r} is given twice")
        elif name in merged and not override:
            problems.append(f"{label} {name!r} already exists (pass override=True to replace it)")
        given.add(name)
        merged[name] = item
    return merged


def _merged_adapters(
    base: Mapping[str, AdapterClass],
    adapters: Mapping[str, AdapterClass] | None,
    *,
    override: bool,
    problems: list[str],
) -> dict[str, AdapterClass]:
    merged = dict(base)
    for protocol, adapter in (adapters or {}).items():
        if not (isinstance(adapter, type) and callable(getattr(adapter, "from_environment", None))):
            problems.append(
                f"the adapter for protocol {protocol!r} must be an adapter class "
                f"(an ExternalAIAdapterBase subclass), got {adapter!r}"
            )
            continue
        if protocol in merged and not override:
            problems.append(
                f"protocol {protocol!r} already has an adapter "
                "(pass override=True to replace it)"
            )
        merged[protocol] = adapter
    return merged


def _consistency_problems(
    providers: Mapping[str, ProviderConfig],
    models: Mapping[str, ModelSpec],
    adapters: Mapping[str, AdapterClass],
) -> list[str]:
    problems = [
        f"provider {name!r} speaks protocol {config.protocol!r}, which has no adapter"
        for name, config in providers.items()
        if config.protocol not in adapters
    ]
    problems += [
        f"model {model_id!r} names provider {spec.provider!r}, which isn't configured"
        for model_id, spec in models.items()
        if spec.provider not in providers
    ]
    return problems


# The built-ins. It reads PROVIDERS, CATALOG and PROTOCOLS live, so code that edits those
# dicts in place keeps reaching the default router (unsupported: use extend instead).
DEFAULT_REGISTRY = ModelRegistry._over(PROVIDERS, CATALOG, PROTOCOLS)
