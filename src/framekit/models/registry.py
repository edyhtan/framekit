"""The model registry: the providers, models and protocol adapters a router may use.

`DEFAULT_REGISTRY` serves the built-ins (`PROVIDERS`, `CATALOG` and `PROTOCOLS`). Add your
own with `extend`, which returns a new registry and leaves the default untouched:

    registry = DEFAULT_REGISTRY.extend(
        providers=[ProviderConfig("local", "openai", base_url="http://127.0.0.1:8000/v1")],
        models=[ModelSpec("local", "my-model", "My model", Decimal(0), Decimal(0))],
    )
    async with ConversationRouter(settings, registry=registry) as router:
        ...

A registry is checked when it's built and reports every problem at once, in one
`ValueError`: the shape of each argument, names (non-empty strings), key variables, prices,
that every model's provider is configured and every provider's protocol has an adapter, and
that every adapter class can work (concrete, with an `adapter_version`, `make_client` and
`from_environment` classmethods). Guide: docs/adapters.md.
"""

import inspect
from collections.abc import Callable, Iterable, Mapping
from decimal import Decimal
from types import MappingProxyType
from typing import Any

from framekit.contracts.conversation import ModelCallError
from framekit.models.adapters.anthropic import AnthropicAdapter
from framekit.models.adapters.base import ExternalAIAdapterBase, adapter_class_problems
from framekit.models.adapters.openai_compatible import OpenAICompatibleAdapter
from framekit.models.catalog import (
    CATALOG,
    PROVIDERS,
    LongPromptPrices,
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

    Immutable, shallowly: its mappings can't be changed, and `extend` returns a new
    registry. The entries themselves are shared (with the built-ins, and between
    registries), so never mutate one in place, e.g. a provider's `extra_headers`: replace
    it, with `dataclasses.replace(...)` and `extend(..., override=True)`. Built from
    scratch, a registry holds only what it's given (no built-ins):

        ModelRegistry(providers=[...], models=[...], adapters={"my-protocol": MyAdapter})

    A registry can be pickled, e.g. for worker processes: `DEFAULT_REGISTRY` by reference
    (it stays the built-ins of the process that loads it), any other by value, with its
    adapter classes by reference (so they must be importable).
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
        """A registry over existing dicts, without copying them or checking their entries
        (only that they're consistent): in-place edits to the dicts show through. For the
        built-ins (`DEFAULT_REGISTRY`) and for unpickling."""
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

    def __reduce__(self) -> str | tuple[Any, ...]:
        if self is DEFAULT_REGISTRY:
            return "DEFAULT_REGISTRY"  # by reference: the loading process's built-ins
        return (
            _unpickle,
            (type(self), dict(self._providers), dict(self._models), dict(self._adapters)),
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(providers={sorted(map(str, self._providers))}, "
            f"models={len(self._models)}, adapters={sorted(map(str, self._adapters))})"
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
        provider or protocol, or a missing key, is a non-retryable `ModelCallError`.

        An adapter class without `make_client` can't be built here: that's a
        `NotImplementedError`, a bug in the code that uses the registry (build such an
        adapter yourself, in an `adapter_factory`)."""
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


def _unpickle(
    cls: type[ModelRegistry],
    providers: dict[str, ProviderConfig],
    models: dict[str, ModelSpec],
    adapters: dict[str, AdapterClass],
) -> ModelRegistry:
    return cls._over(providers, models, adapters)


# The helpers below collect problems instead of raising, so a registry reports them all
# at once. An entry whose name is unusable is left out; any other problem is reported and
# the entry kept, so it doesn't also show up as a missing provider or adapter.


def _is_name(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _shape(value: object) -> str:
    return "None" if value is None else type(value).__name__


def _provider_problems(config: ProviderConfig) -> list[str]:
    problems = []
    if not _is_name(config.protocol):
        problems.append(
            f"provider {config.name!r}: protocol must be a non-empty string, "
            f"got {config.protocol!r}"
        )
    if config.key_env is not None and not _is_name(config.key_env):
        problems.append(
            f"provider {config.name!r}: key_env must name the environment variable that "
            f"holds its key, or be None for a provider without one, got {config.key_env!r}"
        )
    return problems


def _model_problems(spec: ModelSpec) -> list[str]:
    problems = []
    if not _is_name(spec.provider):
        problems.append(
            f"model {spec.model_id!r}: provider must be a non-empty string, "
            f"got {spec.provider!r}"
        )
    for field in ("input_usd_per_mtok", "output_usd_per_mtok"):
        price = getattr(spec, field)
        if not isinstance(price, Decimal | int):  # a float fails only after the call
            problems.append(
                f"model {spec.model_id!r}: {field} must be a Decimal, e.g. "
                f'Decimal("0.30"), got {price!r}'
            )
    return problems + _long_prompt_problems(spec)


def _long_prompt_problems(spec: ModelSpec) -> list[str]:
    tier = spec.long_prompt
    if tier is None:
        return []
    name = f"model {spec.model_id!r}: long_prompt"
    if not isinstance(tier, LongPromptPrices):
        return [f"{name} must be a LongPromptPrices or None, got {tier!r}"]
    problems = []
    above = tier.above_tokens
    if isinstance(above, bool) or not isinstance(above, int) or above < 1:
        problems.append(f"{name}.above_tokens must be a whole number >= 1, got {above!r}")
    for field in ("input_usd_per_mtok", "output_usd_per_mtok"):
        price = getattr(tier, field)
        if not isinstance(price, Decimal | int):
            problems.append(f"{name}.{field} must be a Decimal, got {price!r}")
    return problems


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
        "name",
        lambda config: config.name,
        _provider_problems,
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
        "model_id",
        lambda spec: spec.model_id,
        _model_problems,
        override=override,
        problems=problems,
    )


def _merged[T](
    base: Mapping[str, T],
    items: Iterable[T],
    kind: type[T],
    label: str,
    name_field: str,
    name_of: Callable[[T], object],
    item_problems: Callable[[T], list[str]],
    *,
    override: bool,
    problems: list[str],
) -> dict[str, T]:
    """`base` plus `items`, keyed by `name_of(item)`."""
    merged = dict(base)
    expected = f"{label}s must be a list of {kind.__name__}s"
    if isinstance(items, kind):  # e.g. providers=config instead of providers=[config]
        problems.append(f"{expected}, not a single one")
        return merged
    if items is None or isinstance(items, str | bytes | Mapping):
        problems.append(f"{expected}, got {_shape(items)}")
        return merged
    try:
        iterator = iter(items)
    except TypeError:
        problems.append(f"{expected}, got {_shape(items)}")
        return merged
    given: set[str] = set()
    for item in iterator:
        if not isinstance(item, kind):
            problems.append(f"{label}s: expected a {kind.__name__}, got {_shape(item)}")
            continue
        name = name_of(item)
        if not isinstance(name, str) or not name.strip():
            problems.append(f"a {label}'s {name_field} must be a non-empty string, got {name!r}")
            continue
        problems += item_problems(item)
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
    if adapters is None:
        return merged
    if not isinstance(adapters, Mapping):
        problems.append(
            "adapters must be a mapping of protocol names to adapter classes, e.g. "
            f'{{"my-protocol": MyAdapter}}, got {_shape(adapters)}'
        )
        return merged
    for protocol, adapter in adapters.items():
        if not _is_name(protocol):
            problems.append(f"a protocol name must be a non-empty string, got {protocol!r}")
            continue
        problems += _adapter_problems(protocol, adapter)
        if protocol in merged and not override:
            problems.append(
                f"protocol {protocol!r} already has an adapter "
                "(pass override=True to replace it)"
            )
        merged[protocol] = adapter
    return merged


def _adapter_problems(protocol: str, adapter: object) -> list[str]:
    if not (isinstance(adapter, type) and callable(getattr(adapter, "from_environment", None))):
        return [
            f"the adapter for protocol {protocol!r} must be an adapter class "
            f"(an ExternalAIAdapterBase subclass), got {adapter!r}"
        ]
    if issubclass(adapter, ExternalAIAdapterBase):
        found = adapter_class_problems(adapter)
    else:  # duck-typed: any class with a from_environment that builds an adapter
        found = [f"{adapter.__name__} is abstract"] if inspect.isabstract(adapter) else []
    return [f"the adapter for protocol {protocol!r}: {problem}" for problem in found]


def _consistency_problems(
    providers: Mapping[str, ProviderConfig],
    models: Mapping[str, ModelSpec],
    adapters: Mapping[str, AdapterClass],
) -> list[str]:
    # Fields that aren't usable names are reported by the checks above.
    problems = [
        f"provider {name!r} speaks protocol {config.protocol!r}, which has no adapter"
        for name, config in providers.items()
        if _is_name(config.protocol) and config.protocol not in adapters
    ]
    problems += [
        f"model {model_id!r} names provider {spec.provider!r}, which isn't configured"
        for model_id, spec in models.items()
        if _is_name(spec.provider) and spec.provider not in providers
    ]
    return problems


# The built-ins. It reads PROVIDERS, CATALOG and PROTOCOLS live, so code that edits those
# dicts in place keeps reaching the default router (unsupported: use extend instead).
# Rebinding the names (`router.PROTOCOLS = {...}`) doesn't: this holds the dicts themselves.
DEFAULT_REGISTRY = ModelRegistry._over(PROVIDERS, CATALOG, PROTOCOLS)
