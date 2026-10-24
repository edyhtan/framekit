"""Model sources: one structure for which model, where it's served, with which key, and
through which adapter.

    ConversationRouter(settings, sources=[
        ModelSource("claude", model="claude-haiku-4-5", credential=SecretStr(my_key)),
        ModelSource("local", model=ModelSpec("box", "qwen3-32b", "Qwen3 32B", Decimal(0),
                                             Decimal(0)),
                    provider=ProviderConfig("box", "openai", base_url="http://box:8000/v1")),
        ModelSource("inhouse", model=..., provider=..., adapter=GatewayAdapter,
                    credential=lambda: vault.read("gateway")),
    ])

A request names a source by `model_id` (and `ModelSettings.default_model` may name one). The
router's registry still serves everything else: the built-in catalog models keep working
under their own ids, with their keys from the environment. A source reusing a catalog id
overrides that model for this router.

**Credentials are never recorded.** A source keeps its credential as a pydantic `SecretStr`
(hidden in reprs and logs), passes it only to the adapter's `make_client`, and refuses to be
pickled while it holds one. Results, cache keys and manifests carry the source's name and
its public fingerprint (protocol, base URL, adapter and version, model id), never the key.
Guide: docs/adapters.md.
"""

import dataclasses
import hashlib
import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from pydantic import SecretStr

from framekit.contracts.conversation import ModelCallError
from framekit.models.adapters.base import ExternalAIAdapterBase, adapter_class_problems
from framekit.models.catalog import ModelSettings, ModelSpec, ProviderConfig

AdapterClass = type[ExternalAIAdapterBase]
# A key, or a callable returning one: called each time a router builds the source's
# adapter (once per unit of work), so rotating tokens and secret managers work.
Credential = str | SecretStr | Callable[[], str | SecretStr]


@dataclass(frozen=True)
class ModelSource:
    """A model you can call: what it is, where it's served, with which key, through which
    adapter.

    - `name`: what requests (`ConversationRequest.model_id`) and plan roles refer to.
    - `model`: a model id from the router's registry (e.g. a built-in catalog model), or
      your own `ModelSpec`. Its `model_id` is what's sent to the provider.
    - `provider`: where it's served: a `ProviderConfig`, or the name of a provider in the
      router's registry. Default: the model's own provider. The source's provider wins over
      the spec's `provider` field.
    - `credential`: a string, a `SecretStr`, or a callable returning one. Default: the
      provider's `key_env` variable, or no key for a provider without one.
    - `adapter`: your adapter class (docs/adapters.md). Default: the registry's adapter for
      the provider's protocol.
    """

    name: str
    model: str | ModelSpec
    provider: str | ProviderConfig | None = None
    credential: Credential | None = field(default=None, repr=False, compare=False)
    adapter: AdapterClass | None = None

    def __post_init__(self) -> None:
        if isinstance(self.credential, str):  # hidden from reprs and logs from here on
            object.__setattr__(self, "credential", SecretStr(self.credential))

    def __reduce__(self) -> Any:
        if self.credential is not None:
            raise TypeError(
                f"ModelSource {self.name!r} holds a credential and can't be pickled: give "
                "worker processes their keys through their own environment or secret store"
            )
        return (_rebuild, (self.name, self.model, self.provider, self.adapter))


def _rebuild(
    name: str,
    model: str | ModelSpec,
    provider: str | ProviderConfig | None,
    adapter: AdapterClass | None,
) -> ModelSource:
    return ModelSource(name, model, provider, adapter=adapter)


@dataclass(frozen=True)
class ResolvedSource:
    """A source checked against a registry: everything a router needs to call it."""

    name: str
    spec: ModelSpec  # the effective spec (its provider is the source's provider)
    config: ProviderConfig
    adapter: AdapterClass
    credential: Credential | None = field(repr=False, compare=False)
    fingerprint: str  # public: no credential, no source name

    def build(self, settings: ModelSettings) -> ExternalAIAdapterBase:
        """A fresh adapter (and client) for one unit of work."""
        if self.credential is None:
            return self.adapter.from_environment(self.config, settings)
        from_key = getattr(self.adapter, "from_key", None)
        if from_key is None:
            raise ModelCallError(
                f"source {self.name!r}: adapter {self.adapter.__name__} can't take an "
                "explicit credential (it has no from_key)",
                retryable=False,
            )
        return from_key(self.config, _secret(self.name, self.credential), settings)


def _secret(name: str, credential: Credential) -> str:
    value = credential() if callable(credential) else credential
    if isinstance(value, SecretStr):
        value = value.get_secret_value()
    if not isinstance(value, str) or not value:
        raise ModelCallError(
            f"source {name!r}: the credential is empty or not a string", retryable=False
        )
    return value


def fingerprint(spec: ModelSpec, config: ProviderConfig, adapter: AdapterClass) -> str:
    """A short, stable, public identity of what serves a source: the protocol, the base
    URL, the adapter class and version, and the model id sent. No names, no credentials.
    Two sources reaching the same model at the same place share it (and cached answers)."""
    identity = {
        "adapter": f"{adapter.__module__}.{adapter.__qualname__}",
        "adapter_version": getattr(adapter, "adapter_version", None),
        "base_url": config.resolved_base_url(),
        "model_id": spec.model_id,
        "protocol": config.protocol,
    }
    blob = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def resolve_sources(
    sources: Iterable[ModelSource],
    *,
    providers: Mapping[str, ProviderConfig],
    models: Mapping[str, ModelSpec],
    adapters: Mapping[str, AdapterClass],
) -> dict[str, ResolvedSource]:
    """Check sources against a registry's entries. Every problem is reported at once, in
    one `ValueError`."""
    resolved: dict[str, ResolvedSource] = {}
    problems: list[str] = []
    for source in sources:
        if not isinstance(source, ModelSource):
            problems.append(f"expected a ModelSource, got {type(source).__name__}")
            continue
        found = _resolve_one(source, providers, models, adapters)
        if isinstance(found, list):
            problems += found
        elif source.name in resolved:
            problems.append(f"source {source.name!r} is given twice")
        else:
            resolved[source.name] = found
    if problems:
        raise ValueError(
            f"invalid model sources ({len(problems)} problem"
            f"{'s' if len(problems) > 1 else ''}):\n"
            + "\n".join(f"- {problem}" for problem in problems)
        )
    return resolved


def _resolve_one(
    source: ModelSource,
    providers: Mapping[str, ProviderConfig],
    models: Mapping[str, ModelSpec],
    adapters: Mapping[str, AdapterClass],
) -> ResolvedSource | list[str]:
    where = f"source {source.name!r}"
    problems: list[str] = []
    if not (isinstance(source.name, str) and source.name.strip()):
        return [f"source name must be a non-empty string, got {source.name!r}"]

    spec: ModelSpec | None = None
    if isinstance(source.model, ModelSpec):
        spec = source.model
        if not (isinstance(spec.model_id, str) and spec.model_id.strip()):
            problems.append(f"{where}: the model's model_id must be a non-empty string")
    elif isinstance(source.model, str):
        spec = models.get(source.model)
        if spec is None:
            problems.append(f"{where}: model {source.model!r} is not in the registry")
    else:
        problems.append(f"{where}: model must be a model id or a ModelSpec")

    config: ProviderConfig | None = None
    if isinstance(source.provider, ProviderConfig):
        config = source.provider
        if not (isinstance(config.protocol, str) and config.protocol.strip()):
            problems.append(f"{where}: the provider's protocol must be a non-empty string")
    elif isinstance(source.provider, str):
        config = providers.get(source.provider)
        if config is None:
            problems.append(f"{where}: provider {source.provider!r} is not in the registry")
    elif source.provider is None:
        if spec is not None:
            config = providers.get(spec.provider)
            if config is None:
                problems.append(
                    f"{where}: the model's provider {spec.provider!r} is not in the "
                    "registry: give the source a provider"
                )
    else:
        problems.append(f"{where}: provider must be a ProviderConfig, a provider name or None")

    adapter: AdapterClass | None = source.adapter
    if adapter is not None:
        if not (isinstance(adapter, type) and issubclass(adapter, ExternalAIAdapterBase)):
            problems.append(f"{where}: adapter must be an ExternalAIAdapterBase subclass")
            adapter = None
        else:
            problems += [f"{where}: {problem}" for problem in adapter_class_problems(adapter)]
    elif config is not None:
        adapter = adapters.get(config.protocol)
        if adapter is None:
            problems.append(
                f"{where}: protocol {config.protocol!r} has no adapter: give the source "
                "an adapter"
            )

    credential = source.credential
    usable = callable(credential) or (
        isinstance(credential, SecretStr) and bool(credential.get_secret_value())
    )
    if credential is not None and not usable:
        problems.append(f"{where}: credential must be a non-empty string, SecretStr or callable")

    if problems or spec is None or config is None or adapter is None:
        return problems or [f"{where}: can't be resolved"]
    if spec.provider != config.name:
        spec = dataclasses.replace(spec, provider=config.name)
    return ResolvedSource(
        name=source.name,
        spec=spec,
        config=config,
        adapter=adapter,
        credential=credential,
        fingerprint=fingerprint(spec, config, adapter),
    )
