"""Your own adapters, providers and models: the adapter template, the model registry,
keyless providers, error mapping, and backwards compatibility with the built-in dicts.

No real API calls: the custom adapter talks to a fake gateway client, and the real SDK
clients are only constructed, never used.
"""

import pickle
from dataclasses import replace
from decimal import Decimal
from typing import Any, ClassVar

import openai
import pytest
from test_adapters import api_error
from test_cache import MemoryStore

import framekit.models.adapters as template
import framekit.models.registry as registry_module
import framekit.models.router as router_module
from framekit.contracts import (
    ConversationRequest,
    ConversationResult,
    ModelCallError,
    TextDelta,
)
from framekit.models import (
    CATALOG,
    DEFAULT_REGISTRY,
    PROVIDERS,
    ConversationRouter,
    ModelRegistry,
    ModelSettings,
    ModelSpec,
    ProviderConfig,
    catalog_listing,
    create_adapter,
)
from framekit.models.adapters import (
    PLACEHOLDER_KEY,
    ExternalAIAdapterBase,
    Reply,
    conversation,
    merge_fields,
    reasoning_active,
    reasoning_fields,
    structured_mode,
    system_text,
)
from framekit.models.adapters.anthropic import AnthropicAdapter
from framekit.models.adapters.openai_compatible import OpenAICompatibleAdapter
from framekit.models.cache import CacheSettings, request_key, with_cache
from framekit.models.router import PROTOCOLS

# --- a user-written adapter: an in-house gateway with its own client and errors ----------


class GatewayError(Exception):
    """The gateway client's own exception type (not an SDK's)."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


class FakeGatewayClient:
    """Stands in for the gateway's client. A prompt "fail:<status>" makes it fail."""

    built: ClassVar[list["FakeGatewayClient"]] = []

    def __init__(self, api_key: str, base_url: str | None, timeout_s: float) -> None:
        self.api_key, self.base_url, self.timeout_s = api_key, base_url, timeout_s
        self.payloads: list[dict[str, Any]] = []
        self.closed = False
        FakeGatewayClient.built.append(self)

    async def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.payloads.append(payload)
        prompt = payload["messages"][-1]["content"]
        if prompt.startswith("fail:"):
            raise GatewayError(int(prompt.removeprefix("fail:")), "gateway said no")
        return {
            "text": f"{payload['model']} says: {prompt}",
            "finish": "stop",
            "usage": {"in": 7, "out": 3},
            "id": "gw-1",
        }

    async def close(self) -> None:
        self.closed = True


class GatewayAdapter(ExternalAIAdapterBase):
    adapter_version: ClassVar[str] = "gw-0.1.0"

    @classmethod
    def make_client(cls, api_key: str, base_url: str | None, timeout_s: float) -> Any:
        return FakeGatewayClient(api_key, base_url, timeout_s)

    def build_params(
        self, spec: ModelSpec, request: ConversationRequest, max_tokens: int
    ) -> dict[str, Any]:
        system = system_text(request, structured_mode(spec, request, self.structured_modes))
        messages = conversation(request)
        if system:
            messages = [{"role": "system", "content": system}, *messages]
        return merge_fields(
            {"model": spec.model_id, "messages": messages, "max_tokens": max_tokens},
            spec.request_extras,
            reasoning_fields(spec, reasoning_active(spec, request)),
        )

    async def call(self, spec: ModelSpec, params: dict[str, Any]) -> Reply:
        data = await self.client.complete(params)
        return Reply(
            text=data["text"],
            stop_reason=data["finish"],
            input_tokens=data["usage"]["in"],
            output_tokens=data["usage"]["out"],
            request_id=data["id"],
        )

    def map_error(self, exc: Exception) -> ModelCallError | None:
        if isinstance(exc, GatewayError):  # rate limits and server errors may pass
            retryable = exc.status == 429 or exc.status >= 500
            return ModelCallError(f"gateway {exc.status}: {exc}", retryable=retryable)
        return super().map_error(exc)


# Keyless (key_env defaults to None), like a server on a private network.
GATEWAY = ProviderConfig("my-gateway", "gateway", base_url="https://llm.internal.example/v1")
SPEC = ModelSpec("my-gateway", "inhouse/llm-1", "In-house LLM", Decimal("2"), Decimal("10"))
REGISTRY = DEFAULT_REGISTRY.extend(
    providers=[GATEWAY], models=[SPEC], adapters={"gateway": GatewayAdapter}
)
SETTINGS = ModelSettings(default_model="inhouse/llm-1", max_tokens_cap=512)


@pytest.fixture(autouse=True)
def fresh_gateway_clients():
    FakeGatewayClient.built.clear()
    yield
    FakeGatewayClient.built.clear()


def req(prompt="hi", **fields):
    return ConversationRequest(prompt, **({"max_tokens": 50} | fields))


def gateway_router():
    return ConversationRouter(SETTINGS, registry=REGISTRY)


# --- the template, end to end through a registry router ----------------------------------


async def test_generate_records_provenance_and_cost():
    async with gateway_router() as port:
        result = await port.generate(req("hello there", system="be brief"))

    [client] = FakeGatewayClient.built
    assert client.closed  # the router closed what it opened
    [payload] = client.payloads
    assert payload["messages"][0] == {"role": "system", "content": "be brief"}
    assert payload["model"] == "inhouse/llm-1" and payload["max_tokens"] == 50
    assert result.text == "inhouse/llm-1 says: hello there"
    assert (result.provider, result.model_id, result.adapter_version) == (
        "my-gateway",
        "inhouse/llm-1",
        "gw-0.1.0",
    )
    assert (result.input_tokens, result.output_tokens) == (7, 3)
    assert result.cost_usd == Decimal("0.000044")  # 7 x $2 + 3 x $10 per 1M tokens
    assert (result.stop_reason, result.provider_stop_reason) == ("end", "stop")
    assert result.request_id == "gw-1" and result.latency_ms >= 0
    assert result.cached is False and result.reasoning_on is False
    assert result.structured_mode is None and result.first_token_ms is None


async def test_stream_falls_back_to_one_call():
    async with gateway_router() as port:
        *deltas, final = [piece async for piece in port.stream(req("stream me"))]
    assert deltas == [TextDelta("text", "inhouse/llm-1 says: stream me")]
    assert isinstance(final, ConversationResult) and final.text == deltas[0].text
    assert final.provider == "my-gateway" and final.first_token_ms is not None
    assert len(FakeGatewayClient.built[0].payloads) == 1


async def test_structured_output_falls_back_to_prompted_for_a_custom_adapter():
    schema = {"type": "object", "properties": {"answer": {"type": "integer"}}}
    spec = replace(SPEC, structured_method="json_schema")  # not in its structured_modes
    registry = DEFAULT_REGISTRY.extend(
        providers=[GATEWAY], models=[spec], adapters={"gateway": GatewayAdapter}
    )
    async with ConversationRouter(SETTINGS, registry=registry) as port:
        result = await port.generate(req(response_schema=schema))
    system = FakeGatewayClient.built[0].payloads[0]["messages"][0]
    assert system["role"] == "system" and '"answer"' in system["content"]
    assert result.structured_mode == "prompted"


# --- errors -------------------------------------------------------------------------------


@pytest.mark.parametrize(("status", "retryable"), [(503, True), (429, True), (400, False)])
async def test_map_error_override_translates_the_clients_own_errors(status, retryable):
    async with gateway_router() as port:
        with pytest.raises(ModelCallError) as generated:
            await port.generate(req(f"fail:{status}"))
        with pytest.raises(ModelCallError) as streamed:
            [piece async for piece in port.stream(req(f"fail:{status}"))]
    for info in (generated, streamed):
        assert info.value.retryable is retryable and f"gateway {status}" in str(info.value)
        assert isinstance(info.value.__cause__, GatewayError)


class RaisingAdapter(ExternalAIAdapterBase):
    """No `sdk`, no `make_client`, no `map_error`: built directly, raises what it's told."""

    adapter_version: ClassVar[str] = "raising-1"

    def build_params(self, spec, request, max_tokens):
        return {"model": spec.model_id}

    async def call(self, spec, params):
        raise self.client


async def drain(stream):
    return [piece async for piece in stream]


async def generate_and_stream_errors(adapter):
    errors = []
    for run in (
        lambda: adapter.generate(SPEC, req()),
        lambda: drain(adapter.stream(SPEC, req())),
    ):
        try:
            await run()
        except Exception as exc:  # the test inspects whatever came out
            errors.append(exc)
    return errors


async def test_model_call_error_from_call_passes_through_unchanged():
    error = ModelCallError("quota used up", retryable=False)
    adapter = RaisingAdapter(GATEWAY, error, SETTINGS)
    assert await generate_and_stream_errors(adapter) == [error, error]  # the same object
    assert error.__cause__ is None


class CatchAllAdapter(RaisingAdapter):
    """Maps every exception it's given, without `super().map_error`."""

    def map_error(self, exc):
        return ModelCallError(f"gateway failed: {exc}", retryable=True)


class CatchAllStreamingAdapter(CatchAllAdapter):
    async def iterate(self, spec, params):
        yield TextDelta("text", "partial")
        raise self.client


@pytest.mark.parametrize("adapter_class", [CatchAllAdapter, CatchAllStreamingAdapter])
async def test_a_model_call_error_passes_through_even_a_catch_all_map_error(adapter_class):
    error = ModelCallError("request rejected: bad schema", retryable=False)
    adapter = adapter_class(GATEWAY, error, SETTINGS)
    assert await generate_and_stream_errors(adapter) == [error, error]  # the same object
    assert error.retryable is False and error.__cause__ is None  # not made retryable

    other = OSError("connection reset")  # anything else still goes through the override
    adapter = adapter_class(GATEWAY, other, SETTINGS)
    for mapped in await generate_and_stream_errors(adapter):
        assert isinstance(mapped, ModelCallError) and mapped.retryable is True
        assert mapped.__cause__ is other


@pytest.mark.parametrize("returned", ["boom", False, RuntimeError("not a ModelCallError")])
async def test_map_error_returning_anything_else_is_a_type_error(returned):
    class WrongReturn(RaisingAdapter):
        def map_error(self, exc):
            return returned

    original = ValueError("the real error")
    errors = await generate_and_stream_errors(WrongReturn(GATEWAY, original, SETTINGS))
    assert len(errors) == 2
    for error in errors:
        assert isinstance(error, TypeError)
        assert "WrongReturn.map_error must return a ModelCallError or None" in str(error)
        assert error.__cause__ is original  # the real error isn't lost


async def test_map_error_returning_the_exception_itself_lets_it_propagate():
    class PassThrough(RaisingAdapter):
        def map_error(self, exc):
            return exc

    original = ValueError("the real error")
    adapter = PassThrough(GATEWAY, original, SETTINGS)
    assert await generate_and_stream_errors(adapter) == [original, original]
    assert original.__cause__ is None  # never chained to itself


async def test_without_an_sdk_other_errors_surface_as_they_are():
    bug = RuntimeError("a bug in the adapter")
    adapter = RaisingAdapter(GATEWAY, bug, SETTINGS)
    assert ExternalAIAdapterBase.sdk is None and adapter.map_error(bug) is None
    assert await generate_and_stream_errors(adapter) == [bug, bug]  # no AttributeError


async def test_sdk_none_set_explicitly_keeps_errors_clean():
    # The probe's case: an adapter that sets `sdk = None` and raises ModelCallError.
    class NoSdkAdapter(GatewayAdapter):
        sdk = None

        async def call(self, spec, params):
            raise ModelCallError("clean error", retryable=False)

    def factory(provider, settings):
        return NoSdkAdapter(GATEWAY, None, settings)

    async with ConversationRouter(SETTINGS, factory, registry=REGISTRY) as port:
        with pytest.raises(ModelCallError, match="clean error"):
            await port.generate(req())


class SdkAdapter(RaisingAdapter):
    sdk = openai  # an openai-style SDK: its APIErrors go through map_sdk_error


async def test_an_adapter_naming_an_sdk_gets_the_shared_mapping():
    adapter = SdkAdapter(GATEWAY, api_error(openai, "RateLimitError", 429), SETTINGS)
    [generated, streamed] = await generate_and_stream_errors(adapter)
    for error in (generated, streamed):
        assert isinstance(error, ModelCallError) and error.retryable is True
        assert isinstance(error.__cause__, openai.RateLimitError)
    assert adapter.map_error(ModelCallError("x", retryable=True)) is None
    assert adapter.map_error(ValueError("not an SDK error")) is None


# --- keys and clients ----------------------------------------------------------------------


async def test_a_keyless_provider_gets_the_placeholder_never_an_ambient_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "ambient-openai-key-must-not-leak")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ambient-anthropic-key-must-not-leak")
    async with gateway_router() as port:
        await port.generate(req())
    [client] = FakeGatewayClient.built
    assert client.api_key == PLACEHOLDER_KEY == "framekit-no-key"
    assert client.base_url == "https://llm.internal.example/v1"
    assert client.timeout_s == SETTINGS.timeout_s


@pytest.mark.parametrize(
    ("adapter_class", "protocol"),
    [(OpenAICompatibleAdapter, "openai"), (AnthropicAdapter, "anthropic")],
)
async def test_keyless_providers_never_let_the_sdks_pick_up_their_own_keys(
    monkeypatch, adapter_class, protocol
):
    # The SDKs fall back to these variables when given no key; framekit never gives none.
    ambient = "ambient-cloud-credential"
    for variable in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        monkeypatch.setenv(variable, ambient)
    local = ProviderConfig("local", protocol, base_url="http://127.0.0.1:8000/v1")
    adapter = adapter_class.from_environment(local, ModelSettings())
    try:
        assert adapter.client.api_key == PLACEHOLDER_KEY
        assert getattr(adapter.client, "auth_token", None) is None
        assert not any(ambient in value for value in adapter.client.auth_headers.values())
    finally:
        await adapter.aclose()


def test_a_provider_with_a_key_variable_still_requires_it(monkeypatch):
    keyed = replace(GATEWAY, key_env="MY_GATEWAY_KEY")
    registry = DEFAULT_REGISTRY.extend(
        providers=[keyed], models=[SPEC], adapters={"gateway": GatewayAdapter}
    )
    monkeypatch.delenv("MY_GATEWAY_KEY", raising=False)
    with pytest.raises(ModelCallError) as info:
        registry.create_adapter("my-gateway", ModelSettings())
    assert info.value.retryable is False and "MY_GATEWAY_KEY" in str(info.value)

    monkeypatch.setenv("MY_GATEWAY_KEY", "gateway-test-value")
    adapter = registry.create_adapter("my-gateway", ModelSettings())
    assert adapter.client.api_key == "gateway-test-value"


def test_an_adapter_without_make_client_is_built_directly():
    with pytest.raises(NotImplementedError, match="make_client"):
        RaisingAdapter.from_environment(GATEWAY, ModelSettings())
    registry = DEFAULT_REGISTRY.extend(
        providers=[GATEWAY], models=[SPEC], adapters={"gateway": RaisingAdapter}
    )
    with pytest.raises(NotImplementedError, match="RaisingAdapter"):
        registry.create_adapter("my-gateway", ModelSettings())
    assert RaisingAdapter(GATEWAY, "a client", ModelSettings()).client == "a client"


def test_an_empty_key_variable_is_named_as_the_problem():
    # A registry refuses key_env="" (test_validation_checks_names_and_fields); an adapter
    # built from such a config directly says what's wrong instead of naming no variable.
    with pytest.raises(ModelCallError) as info:
        GatewayAdapter.from_environment(replace(GATEWAY, key_env=""), ModelSettings())
    assert info.value.retryable is False
    assert "provider 'my-gateway' has an empty key_env" in str(info.value)


async def test_a_subclass_may_assign_self_client():
    class Assigning(GatewayAdapter):
        def __init__(self, config, client, settings):
            super().__init__(config, client, settings)
            self.client = FakeGatewayClient("replacement", None, 1.0)

    adapter = Assigning(GATEWAY, FakeGatewayClient("original", None, 1.0), SETTINGS)
    assert adapter.client.api_key == "replacement"
    await adapter.generate(SPEC, req())
    await adapter.aclose()
    original, replacement = FakeGatewayClient.built
    assert replacement.payloads and replacement.closed  # used and closed: the same client
    assert not original.payloads and not original.closed


class AcloseOnlyClient:
    """Shaped like an `httpx.AsyncClient`: `aclose()`, no `close()`."""

    def __init__(self):
        self.calls = []

    async def aclose(self):
        self.calls.append("aclose")


class SyncCloseClient:
    """Shaped like a `requests.Session`: a synchronous `close()`."""

    def __init__(self):
        self.calls = []

    def close(self):
        self.calls.append("close")


class BothCloseClient(AcloseOnlyClient, SyncCloseClient):
    pass


@pytest.mark.parametrize(
    ("client_class", "expected"),
    [
        (AcloseOnlyClient, ["aclose"]),
        (SyncCloseClient, ["close"]),
        (BothCloseClient, ["aclose"]),  # aclose() is preferred
    ],
)
async def test_the_default_aclose_closes_common_client_shapes(client_class, expected, caplog):
    clients = []

    class Adapter(GatewayAdapter):
        @classmethod
        def make_client(cls, api_key, base_url, timeout_s):
            clients.append(client_class())
            return clients[-1]

        async def call(self, spec, params):
            return Reply(text="ok", stop_reason="stop")

    registry = DEFAULT_REGISTRY.extend(
        providers=[GATEWAY], models=[SPEC], adapters={"gateway": Adapter}
    )
    async with ConversationRouter(SETTINGS, registry=registry) as port:
        await port.generate(req())
    [client] = clients
    assert client.calls == expected
    assert "closing a model adapter failed" not in caplog.text
    await RaisingAdapter(GATEWAY, object(), SETTINGS).aclose()  # neither: left as it is


class NoVersionAdapter(GatewayAdapter):
    adapter_version = ""  # empty: nothing to record in the results


class AbstractAdapter(ExternalAIAdapterBase):
    """Forgot `call`. Records the clients `make_client` builds."""

    adapter_version: ClassVar[str] = "abstract-1"
    clients: ClassVar[list[FakeGatewayClient]] = []

    @classmethod
    def make_client(cls, api_key, base_url, timeout_s):
        client = FakeGatewayClient(api_key, base_url, timeout_s)
        cls.clients.append(client)
        return client

    def build_params(self, spec, request, max_tokens):
        return {}


def test_an_adapter_that_cannot_work_fails_before_any_call_or_client():
    class Unversioned(ExternalAIAdapterBase):  # forgot adapter_version
        def build_params(self, spec, request, max_tokens):
            return {}

        async def call(self, spec, params):
            raise AssertionError("never called")

    # Built directly (e.g. in an adapter_factory): refused when the adapter is built, not
    # when the first reply is read, after the call is paid for.
    with pytest.raises(TypeError, match="Unversioned must set adapter_version"):
        Unversioned(GATEWAY, None, SETTINGS)
    with pytest.raises(TypeError, match="NoVersionAdapter must set adapter_version"):
        NoVersionAdapter(GATEWAY, None, SETTINGS)

    # From the environment: refused before make_client builds a client nobody would close.
    AbstractAdapter.clients.clear()
    with pytest.raises(TypeError, match="AbstractAdapter doesn't implement call"):
        AbstractAdapter.from_environment(GATEWAY, SETTINGS)
    with pytest.raises(TypeError, match="NoVersionAdapter must set adapter_version"):
        NoVersionAdapter.from_environment(GATEWAY, SETTINGS)
    assert AbstractAdapter.clients == [] and FakeGatewayClient.built == []


# --- the registry ------------------------------------------------------------------------


def test_unknown_provider_and_unknown_protocol_are_model_call_errors(monkeypatch):
    with pytest.raises(ModelCallError) as info:
        REGISTRY.create_adapter("nope", ModelSettings())
    assert info.value.retryable is False and "'nope'" in str(info.value)

    # Only reachable by editing the built-in dicts in place (a validated registry refuses
    # it), but it must still be a ModelCallError, never a bare KeyError.
    monkeypatch.setitem(PROVIDERS, "odd", ProviderConfig("odd", "carrier-pigeon"))
    with pytest.raises(ModelCallError) as info:
        create_adapter("odd", ModelSettings())
    assert info.value.retryable is False and "'carrier-pigeon'" in str(info.value)


async def test_a_model_whose_provider_is_missing_fails_cleanly(monkeypatch):
    ghost = replace(SPEC, provider="ghost", model_id="ghost-model")
    monkeypatch.setitem(CATALOG, ghost.model_id, ghost)
    async with ConversationRouter(ModelSettings()) as port:
        with pytest.raises(ModelCallError, match="'ghost'"):
            await port.generate(req(model_id="ghost-model"))


def test_validation_reports_every_problem_at_once():
    with pytest.raises(ValueError) as info:
        ModelRegistry(
            providers=[ProviderConfig("p1", "nowire"), ProviderConfig("p2", "openai")],
            models=[
                replace(SPEC, model_id="m1", provider="ghost"),
                replace(SPEC, model_id="m2"),  # names my-gateway, not configured here
                replace(SPEC, model_id="m3", provider="p2"),
                replace(SPEC, model_id="m3", provider="p2"),  # m3 again
            ],
            adapters={"openai": OpenAICompatibleAdapter, "broken": object()},
        )
    message = str(info.value)
    assert message.startswith("invalid model registry (5 problems):")
    for fragment in ("'p1'", "'nowire'", "'ghost'", "'my-gateway'", "'m3' is given twice"):
        assert fragment in message
    assert "'broken'" in message


def test_validation_catches_wrong_types():
    with pytest.raises(ValueError, match="expected a ProviderConfig, got dict"):
        ModelRegistry(providers=[{"name": "x", "protocol": "openai"}])
    with pytest.raises(ValueError, match="a list of ModelSpecs, not a single one"):
        DEFAULT_REGISTRY.extend(providers=[GATEWAY], models=SPEC)


def test_validation_reports_wrong_argument_shapes_as_problems():
    with pytest.raises(ValueError) as info:
        ModelRegistry(providers=None, models="inhouse/llm-1", adapters=GatewayAdapter)
    message = str(info.value)
    assert message.startswith("invalid model registry (3 problems):")
    assert "providers must be a list of ProviderConfigs, got None" in message
    assert "models must be a list of ModelSpecs, got str" in message
    assert "adapters must be a mapping of protocol names to adapter classes" in message

    with pytest.raises(ValueError, match="providers must be a list of ProviderConfigs, got dict"):
        DEFAULT_REGISTRY.extend(providers=PROVIDERS)
    for wrong in ([("gateway", GatewayAdapter)], {GatewayAdapter}):  # pairs, a set
        with pytest.raises(ValueError, match="adapters must be a mapping"):
            DEFAULT_REGISTRY.extend(adapters=wrong)


def test_validation_checks_names_and_fields():
    with pytest.raises(ValueError) as info:
        ModelRegistry(
            providers=[
                ProviderConfig(None, "gateway"),
                ProviderConfig("", "gateway"),
                replace(GATEWAY, protocol=None),
                replace(GATEWAY, name="keyed", key_env=""),
            ],
            models=[
                replace(SPEC, model_id=None),
                replace(SPEC, model_id="m-float", input_usd_per_mtok=0.3),  # read after the call
                replace(SPEC, model_id="m-blank", provider=" "),
            ],
            adapters={None: GatewayAdapter, "": GatewayAdapter, "gateway": GatewayAdapter},
        )
    message = str(info.value)
    assert message.startswith("invalid model registry (9 problems):")
    for fragment in (
        "a provider's name must be a non-empty string, got None",
        "a provider's name must be a non-empty string, got ''",
        "provider 'my-gateway': protocol must be a non-empty string, got None",
        "provider 'keyed': key_env must name the environment variable that holds its key",
        "a model's model_id must be a non-empty string, got None",
        "model 'm-float': input_usd_per_mtok must be a Decimal",
        "model 'm-blank': provider must be a non-empty string, got ' '",
        "a protocol name must be a non-empty string, got None",
        "a protocol name must be a non-empty string, got ''",
    ):
        assert fragment in message


def test_validation_refuses_adapter_classes_that_cannot_work():
    class InstanceMakeClient(GatewayAdapter):
        def make_client(self, api_key, base_url, timeout_s):  # forgot @classmethod
            return FakeGatewayClient(api_key, base_url, timeout_s)

    with pytest.raises(ValueError) as info:
        DEFAULT_REGISTRY.extend(
            providers=[
                GATEWAY,
                replace(GATEWAY, name="gw-2", protocol="gw-2"),
                replace(GATEWAY, name="gw-3", protocol="gw-3"),
            ],
            models=[SPEC],
            adapters={
                "gateway": AbstractAdapter,
                "gw-2": NoVersionAdapter,
                "gw-3": InstanceMakeClient,
            },
        )
    message = str(info.value)
    assert message.startswith("invalid model registry (3 problems):")
    assert "protocol 'gateway': AbstractAdapter doesn't implement call" in message
    assert "protocol 'gw-2': NoVersionAdapter must set adapter_version" in message
    assert "protocol 'gw-3': InstanceMakeClient.make_client must be a classmethod" in message
    assert "which has no adapter" not in message  # reported as broken, not as missing

    # Without make_client at all, an adapter class is accepted: the registry still resolves
    # its models when an adapter_factory builds the adapters
    # (test_an_adapter_without_make_client_is_built_directly).
    accepted = DEFAULT_REGISTRY.extend(adapters={"raising": RaisingAdapter})
    assert accepted.adapters["raising"] is RaisingAdapter


def test_repr_survives_odd_names_edited_into_the_built_ins(monkeypatch):
    # Only reachable by editing the built-in dicts in place: a registry refuses such names.
    monkeypatch.setitem(PROVIDERS, None, GATEWAY)
    assert "'None'" in repr(DEFAULT_REGISTRY)


def test_a_registry_pickles_and_the_default_stays_the_built_ins():
    restored = pickle.loads(pickle.dumps(REGISTRY))  # noqa: S301 - our own bytes
    assert restored is not REGISTRY
    assert dict(restored.providers) == dict(REGISTRY.providers)
    assert dict(restored.models) == dict(REGISTRY.models)
    assert dict(restored.adapters) == dict(REGISTRY.adapters)
    assert restored.resolve("inhouse/llm-1") == SPEC
    # By reference: in the process that loads it, it's that process's live built-ins.
    assert pickle.loads(pickle.dumps(DEFAULT_REGISTRY)) is DEFAULT_REGISTRY  # noqa: S301


def test_a_registry_from_scratch_holds_only_what_it_is_given():
    mine = ModelRegistry(providers=[GATEWAY], models=[SPEC], adapters={"gateway": GatewayAdapter})
    assert mine.resolve("inhouse/llm-1") is SPEC
    with pytest.raises(ModelCallError, match="not in the catalog"):
        mine.resolve("claude-haiku-4-5")
    assert set(mine.providers) == {"my-gateway"} and set(mine.adapters) == {"gateway"}
    assert "my-gateway" in repr(mine)


def test_extend_refuses_to_replace_unless_asked():
    hive_here = replace(PROVIDERS["hive"], base_url="https://hive.example/v3/")
    with pytest.raises(ValueError, match="provider 'hive' already exists"):
        DEFAULT_REGISTRY.extend(providers=[hive_here])
    with pytest.raises(ValueError, match="model 'deepseek-flash' already exists"):
        DEFAULT_REGISTRY.extend(models=[CATALOG["deepseek-flash"]])
    with pytest.raises(ValueError, match="protocol 'openai' already has an adapter"):
        DEFAULT_REGISTRY.extend(adapters={"openai": GatewayAdapter})

    replaced = DEFAULT_REGISTRY.extend(
        providers=[hive_here], adapters={"openai": GatewayAdapter}, override=True
    )
    assert replaced.providers["hive"].base_url == "https://hive.example/v3/"
    assert replaced.adapters["openai"] is GatewayAdapter
    assert DEFAULT_REGISTRY.providers["hive"] is PROVIDERS["hive"]  # the original is intact
    assert DEFAULT_REGISTRY.adapters["openai"] is OpenAICompatibleAdapter

    with pytest.raises(ValueError, match="model 'inhouse/llm-1' is given twice"):
        REGISTRY.extend(models=[SPEC, SPEC], override=True)  # ambiguous even with override


def test_a_registry_is_immutable():
    with pytest.raises(AttributeError, match="immutable"):
        REGISTRY.models = {}  # type: ignore[misc]
    with pytest.raises(TypeError):
        REGISTRY.models["x"] = SPEC  # type: ignore[index]
    with pytest.raises(TypeError):
        DEFAULT_REGISTRY.providers["x"] = GATEWAY  # type: ignore[index]


def test_listing_shows_the_registrys_own_models():
    rows = {row["model_id"]: row for row in REGISTRY.listing(default_model="inhouse/llm-1")}
    assert rows["inhouse/llm-1"] == {
        "provider": "my-gateway",
        "model_id": "inhouse/llm-1",
        "label": "In-house LLM",
        "default": True,
        "reasoning": "none",
    }
    assert set(rows) == set(CATALOG) | {"inhouse/llm-1"}
    assert catalog_listing("claude-haiku-4-5") == DEFAULT_REGISTRY.listing("claude-haiku-4-5")


async def test_a_model_added_to_one_registry_is_invisible_elsewhere(monkeypatch):
    assert "inhouse/llm-1" not in CATALOG and "inhouse/llm-1" not in DEFAULT_REGISTRY.models
    assert all(row["model_id"] != "inhouse/llm-1" for row in catalog_listing())
    with pytest.raises(ModelCallError, match="'inhouse/llm-1' is not in the catalog"):
        ConversationRouter(SETTINGS).resolve(None)  # the default router doesn't know it

    other = DEFAULT_REGISTRY.extend(
        providers=[GATEWAY],
        models=[replace(SPEC, model_id="inhouse/llm-2")],
        adapters={"gateway": GatewayAdapter},
    )
    with pytest.raises(ModelCallError):
        ConversationRouter(SETTINGS, registry=other).resolve("inhouse/llm-1")

    # An extended registry is a snapshot: in-place edits of the built-ins don't reach it.
    late = replace(CATALOG["deepseek-flash"], model_id="late-model")
    monkeypatch.setitem(CATALOG, late.model_id, late)
    assert "late-model" in DEFAULT_REGISTRY.models and "late-model" not in REGISTRY.models


# --- backwards compatibility --------------------------------------------------------------


def test_the_public_names_keep_working():
    assert ConversationRouter().registry is DEFAULT_REGISTRY
    with pytest.raises(AttributeError):
        ConversationRouter().registry = REGISTRY  # read-only: pass it to the constructor
    assert PROTOCOLS is registry_module.PROTOCOLS  # router.PROTOCOLS is the same dict
    # Names the router module had before the registry existed are still importable from it.
    for name, canonical in {
        "CATALOG": CATALOG,
        "PROVIDERS": PROVIDERS,
        "ModelCallError": ModelCallError,
        "AnthropicAdapter": AnthropicAdapter,
        "OpenAICompatibleAdapter": OpenAICompatibleAdapter,
        "create_adapter": create_adapter,
    }.items():
        assert getattr(router_module, name) is canonical
    assert dict(DEFAULT_REGISTRY.adapters) == PROTOCOLS
    assert dict(DEFAULT_REGISTRY.models) == CATALOG
    assert dict(DEFAULT_REGISTRY.providers) == PROVIDERS
    assert set(template.__all__) == {
        "PLACEHOLDER_KEY",
        "ExternalAIAdapterBase",
        "Reply",
        "conversation",
        "map_sdk_error",
        "merge_fields",
        "normalize_stop_reason",
        "output_budget",
        "reasoning_active",
        "reasoning_fields",
        "structured_mode",
        "system_text",
    }
    assert all(getattr(template, name) is not None for name in template.__all__)


async def test_in_place_edits_of_the_built_ins_still_reach_the_default_router(monkeypatch):
    # Unsupported (use ModelRegistry.extend), but kept working for existing code: the
    # default registry reads PROVIDERS, CATALOG and PROTOCOLS live.
    monkeypatch.setitem(PROVIDERS, GATEWAY.name, GATEWAY)
    monkeypatch.setitem(PROTOCOLS, "gateway", GatewayAdapter)
    monkeypatch.setitem(CATALOG, SPEC.model_id, SPEC)
    async with ConversationRouter(SETTINGS) as port:
        result = await port.generate(req())
    assert (result.provider, result.adapter_version) == ("my-gateway", "gw-0.1.0")
    assert any(row["model_id"] == SPEC.model_id for row in catalog_listing())


async def test_a_catalog_edit_with_a_positional_adapter_factory_still_works(monkeypatch):
    # The pattern consumers used before registries existed.
    monkeypatch.setitem(CATALOG, SPEC.model_id, SPEC)
    built = []

    def factory(provider, settings):
        built.append(provider)
        if provider == GATEWAY.name:
            return GatewayAdapter(GATEWAY, FakeGatewayClient("k", None, 1.0), settings)
        return create_adapter(provider, settings)  # the built-ins

    async with ConversationRouter(SETTINGS, factory) as port:
        result = await port.generate(req())
    assert built == ["my-gateway"] and result.provider == "my-gateway"


async def test_a_passed_adapter_factory_wins_over_the_registry():
    # The factory is called for every provider the router uses: it builds its own adapter
    # and leaves the rest to the registry (docs/adapters.md, "Adapters built by hand").
    other = replace(GATEWAY, name="other-gateway")
    registry = REGISTRY.extend(
        providers=[other],
        models=[replace(SPEC, provider=other.name, model_id="inhouse/llm-other")],
    )
    built = []

    def factory(provider, settings):
        built.append(provider)
        if provider == GATEWAY.name:
            client = FakeGatewayClient("from-factory", None, 1.0)
            return GatewayAdapter(GATEWAY, client, settings)
        return registry.create_adapter(provider, settings)

    async with ConversationRouter(SETTINGS, factory, registry=registry) as port:
        mine = await port.generate(req())
        theirs = await port.generate(req(model_id="inhouse/llm-other"))
    assert built == ["my-gateway", "other-gateway"]
    assert (mine.provider, theirs.provider) == ("my-gateway", "other-gateway")
    keys = [client.api_key for client in FakeGatewayClient.built]
    assert keys == ["from-factory", PLACEHOLDER_KEY]  # the second built by the registry


async def test_a_falsy_adapter_factory_is_still_used():
    class MemoFactory(dict):
        """Remembers the adapters it built: empty, so falsy, when the router is built."""

        def __call__(self, provider, settings):
            if provider not in self:
                client = FakeGatewayClient("memo", None, 1.0)
                self[provider] = GatewayAdapter(GATEWAY, client, settings)
            return self[provider]

    factory = MemoFactory()
    assert not factory
    async with ConversationRouter(SETTINGS, factory, registry=REGISTRY) as port:
        await port.generate(req())
    assert set(factory) == {"my-gateway"}
    assert [client.api_key for client in FakeGatewayClient.built] == ["memo"]


# --- the response cache with a registry router --------------------------------------------


async def test_the_cache_keys_on_the_model_the_registry_resolves():
    store = MemoryStore()
    ports = with_cache(gateway_router, settings=CacheSettings(mode="record"), store=store)
    request = req("cache me")
    for _ in range(2):
        port = ports()
        result = await port.generate(request)
        await port.aclose()
    assert result.cached and result.adapter_version == "gw-0.1.0+cache"
    assert result.cost_usd == 0 and result.text == "inhouse/llm-1 says: cache me"
    assert len(FakeGatewayClient.built) == 1  # only the first run reached the gateway
    assert list(store.data) == [request_key(request, "inhouse/llm-1", "v1")]


async def test_registries_resolving_different_default_models_do_not_share_entries():
    other = DEFAULT_REGISTRY.extend(
        providers=[GATEWAY],
        models=[replace(SPEC, model_id="inhouse/llm-2")],
        adapters={"gateway": GatewayAdapter},
    )
    other_settings = SETTINGS.model_copy(update={"default_model": "inhouse/llm-2"})
    store, cache = MemoryStore(), CacheSettings(mode="record")
    first = with_cache(gateway_router, settings=cache, store=store)()
    second = with_cache(
        lambda: ConversationRouter(other_settings, registry=other), settings=cache, store=store
    )()
    a = await first.generate(req("same prompt"))  # model_id=None: each router's default
    b = await second.generate(req("same prompt"))
    assert not a.cached and not b.cached
    assert (a.model_id, b.model_id) == ("inhouse/llm-1", "inhouse/llm-2")
    assert b.text == "inhouse/llm-2 says: same prompt" and len(store.data) == 2
    await first.aclose()
    await second.aclose()
