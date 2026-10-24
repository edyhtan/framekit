"""Model sources: one structure for which model, where it's served, with which key, and
through which adapter (docs/adapters.md).

No real API calls: the custom adapter talks to a fake gateway client, and the real SDK
clients are only constructed, never used.
"""

import pickle
from dataclasses import replace

import pytest
from pydantic import SecretStr
from test_cache import MemoryStore
from test_custom_adapters import GATEWAY, SPEC, FakeGatewayClient, GatewayAdapter

from framekit.contracts import ConversationRequest, ConversationResult, ModelCallError, TextDelta
from framekit.models import ConversationRouter, ModelSettings, ModelSource, ProviderConfig
from framekit.models.adapters import PLACEHOLDER_KEY
from framekit.models.cache import CacheSettings, with_cache
from framekit.models.sources import fingerprint

SECRET = "sk-test-not-a-real-key-123456"  # noqa: S105  a fake key, for the tests
SETTINGS = ModelSettings(default_model="inhouse", max_tokens_cap=512)


@pytest.fixture(autouse=True)
def fresh_gateway_clients():
    FakeGatewayClient.built.clear()
    yield
    FakeGatewayClient.built.clear()


def req(prompt="hi", **fields):
    return ConversationRequest(prompt, **({"max_tokens": 50} | fields))


def inhouse(**fields):
    base = {"name": "inhouse", "model": SPEC, "provider": GATEWAY, "adapter": GatewayAdapter}
    return ModelSource(**(base | fields))


# --- one structure: model, provider, credential, adapter --------------------------------


async def test_a_custom_adapter_source_serves_calls_and_stamps_its_name():
    router = ConversationRouter(SETTINGS, sources=[inhouse(credential=SECRET)])
    async with router:
        result = await router.generate(req("hello", model_id="inhouse"))
    [client] = FakeGatewayClient.built
    assert client.api_key == SECRET and client.closed
    assert result.source == "inhouse" and result.provider == "my-gateway"
    assert result.model_id == "inhouse/llm-1" and result.adapter_version == "gw-0.1.0"


async def test_the_default_model_may_name_a_source():
    async with ConversationRouter(SETTINGS, sources=[inhouse()]) as router:
        result = await router.generate(req("hello"))  # no model_id: the default
    assert result.source == "inhouse"
    assert FakeGatewayClient.built[0].api_key == PLACEHOLDER_KEY  # keyless gateway


async def test_streaming_stamps_the_source_on_the_final_result():
    async with ConversationRouter(SETTINGS, sources=[inhouse()]) as router:
        pieces = [p async for p in router.stream(req("stream me"))]
    assert isinstance(pieces[0], TextDelta)
    assert isinstance(pieces[-1], ConversationResult) and pieces[-1].source == "inhouse"


def test_a_built_in_model_with_my_own_key_never_uses_the_ambient_one(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ambient-key")
    source = ModelSource("claude", model="claude-haiku-4-5", credential=SecretStr(SECRET))
    router = ConversationRouter(ModelSettings(), sources=[source])
    adapter = router.sources["claude"].build(ModelSettings())
    assert adapter.client.api_key == SECRET
    assert router.resolve("claude").model_id == "claude-haiku-4-5"


def test_without_a_credential_the_providers_variable_is_used(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-the-environment")
    source = ModelSource("c", model="claude-haiku-4-5")
    router = ConversationRouter(ModelSettings(), sources=[source])
    adapter = router.sources["c"].build(ModelSettings())
    assert adapter.client.api_key == "from-the-environment"


async def test_a_callable_credential_is_called_once_per_router():
    calls = []

    def vault():
        calls.append(1)
        return f"token-{len(calls)}"

    source = inhouse(credential=vault)
    for _ in range(2):
        async with ConversationRouter(SETTINGS, sources=[source]) as router:
            await router.generate(req("a"))
            await router.generate(req("b"))  # same router: same adapter, no new call
    assert len(calls) == 2
    assert [c.api_key for c in FakeGatewayClient.built] == ["token-1", "token-2"]


async def test_an_empty_credential_from_a_callable_fails_cleanly():
    router = ConversationRouter(SETTINGS, sources=[inhouse(credential=lambda: "")])
    with pytest.raises(ModelCallError) as info:
        await router.generate(req())
    assert not info.value.retryable and "credential" in str(info.value)


def test_a_source_may_route_a_built_in_model_through_another_provider():
    proxy = ProviderConfig("corp-proxy", "anthropic", key_env=None,
                           base_url="https://llm-proxy.corp.example")
    router = ConversationRouter(ModelSettings(), sources=[
        ModelSource("claude-via-proxy", model="claude-haiku-4-5", provider=proxy),
    ])
    spec = router.resolve("claude-via-proxy")
    assert spec.provider == "corp-proxy" and spec.model_id == "claude-haiku-4-5"
    adapter = router.sources["claude-via-proxy"].build(ModelSettings())
    assert str(adapter.client.base_url).startswith("https://llm-proxy.corp.example")


def test_a_source_reusing_a_catalog_id_overrides_it_for_this_router_only():
    override = ModelSource(
        "claude-haiku-4-5", model=SPEC, provider=GATEWAY, adapter=GatewayAdapter
    )
    router = ConversationRouter(ModelSettings(), sources=[override])
    assert router.resolve("claude-haiku-4-5").provider == "my-gateway"
    assert ConversationRouter(ModelSettings()).resolve("claude-haiku-4-5").provider == "anthropic"


async def test_an_adapter_factory_builds_registry_models_never_sources():
    built = []

    def factory(provider, settings):
        built.append(provider)
        raise AssertionError("not for sources")

    async with ConversationRouter(SETTINGS, factory, sources=[inhouse()]) as router:
        await router.generate(req())
    assert built == []


# --- checked when the router is built ----------------------------------------------------


def test_every_problem_is_reported_at_once():
    sources = [
        ModelSource("a", model="no-such-model"),
        ModelSource("b", model="claude-haiku-4-5", provider="no-such-provider"),
        ModelSource("c", model=replace(SPEC, provider="unregistered")),
        ModelSource("d", model=SPEC, provider=GATEWAY, adapter=dict),  # type: ignore[arg-type]
        ModelSource("e", model=SPEC, provider=GATEWAY),  # protocol "gateway" has no adapter
        ModelSource("f", model="claude-haiku-4-5", credential=""),
        inhouse(name="g"),
        inhouse(name="g"),
        "not a source",  # type: ignore[list-item]
    ]
    with pytest.raises(ValueError) as info:
        ConversationRouter(ModelSettings(), sources=sources)
    message = str(info.value)
    for expected in [
        "'no-such-model' is not in the registry",
        "provider 'no-such-provider' is not in the registry",
        "provider 'unregistered' is not in the registry: give the source a provider",
        "adapter must be an ExternalAIAdapterBase subclass",
        "protocol 'gateway' has no adapter",
        "credential must be a non-empty string",
        "source 'g' is given twice",
        "expected a ModelSource, got str",
    ]:
        assert expected in message, expected
    assert "9 problems" not in message and "8 problems" in message


# --- credentials are never recorded --------------------------------------------------------


def test_the_credential_never_shows_in_reprs_fingerprints_or_cache_identities():
    source = inhouse(credential=SECRET)
    router = ConversationRouter(SETTINGS, sources=[source])
    resolved = router.sources["inhouse"]
    for text in (repr(source), str(source), repr(resolved), repr(router.sources),
                 resolved.fingerprint, router.cache_identity("inhouse")):
        assert SECRET not in text
    assert isinstance(source.credential, SecretStr)  # stored as a secret from the start


def test_a_source_holding_a_credential_refuses_to_be_pickled():
    with pytest.raises(TypeError, match="can't be pickled"):
        pickle.dumps(inhouse(credential=SECRET))
    plain = inhouse()
    assert pickle.loads(pickle.dumps(plain)) == plain  # noqa: S301  our own bytes, just made


def test_sources_differing_only_in_credential_are_equal():
    assert inhouse(credential="one") == inhouse(credential="two")


# --- cache: one identity per place a model is served -------------------------------------


def test_registry_models_keep_their_plain_cache_identity():
    router = ConversationRouter(ModelSettings())
    assert router.cache_identity("claude-haiku-4-5") == "claude-haiku-4-5"


def test_the_same_model_on_two_hosts_has_two_identities_and_one_host_has_one():
    other_host = replace(GATEWAY, base_url="https://other.internal.example/v1")
    router = ConversationRouter(SETTINGS, sources=[
        inhouse(name="a"),
        inhouse(name="b", provider=other_host),
        inhouse(name="c", credential="a-different-key"),
    ])
    a, b, c = (router.cache_identity(name) for name in "abc")
    assert a != b and a == c and a.startswith("inhouse/llm-1@")
    assert router.sources["a"].fingerprint == fingerprint(SPEC, GATEWAY, GatewayAdapter)


async def test_the_cache_replays_per_source_and_never_across_hosts():
    store = MemoryStore()
    other_host = replace(GATEWAY, base_url="https://other.internal.example/v1")
    sources = [inhouse(name="a"), inhouse(name="b", provider=other_host)]
    port = with_cache(lambda: ConversationRouter(SETTINGS, sources=sources),
                      settings=CacheSettings(mode="record"), store=store)
    for name in ("a", "a", "b"):
        p = port()
        result = await p.generate(req("same prompt", model_id=name))
        await p.aclose()
    assert result.cached is False  # "b" never reuses "a"'s answer
    assert len(FakeGatewayClient.built) == 2  # one call each for "a" and "b"
    assert len(store.data) == 2
