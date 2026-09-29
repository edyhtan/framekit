"""The single model-adapter interface: shared lifecycle, both protocols, providers, router.

No real API calls: SDK clients are replaced by fakes with the same shape.
"""

from decimal import Decimal
from types import SimpleNamespace

import anthropic
import httpx2
import openai
import pytest

from clipsearch_adapters.models.anthropic_text import AnthropicAdapter
from clipsearch_adapters.models.base import (
    ProtocolAdapter,
    ProviderConfig,
    map_sdk_error,
    normalize_stop_reason,
    output_budget,
)
from clipsearch_adapters.models.catalog import CATALOG, TextModelSettings
from clipsearch_adapters.models.openai_compatible import OpenAICompatibleAdapter
from clipsearch_adapters.models.providers import (
    ANTHROPIC,
    DEEPSEEK,
    HIVE,
    PROTOCOLS,
    PROVIDERS,
    create_adapter,
)
from clipsearch_adapters.models.router import TextGenerateRouter
from clipsearch_domain.common.ports import STOP_REASONS, ModelCallError, TextGenerateRequest

HIVE_MODEL = "deepseek-ai/deepseek-v4.1-flash"


# --- fakes -------------------------------------------------------------------------------


class FakeCreate:
    def __init__(self, result=None, error=None):
        self.result, self.error, self.calls = result, error, []

    async def create(self, **params):
        self.calls.append(params)
        if self.error:
            raise self.error
        return self.result


def anthropic_client(result=None, error=None):
    client = SimpleNamespace(messages=FakeCreate(result, error))
    client.beta = SimpleNamespace(messages=FakeCreate(result, error))
    return client


def openai_client(result=None, error=None):
    return SimpleNamespace(chat=SimpleNamespace(completions=FakeCreate(result, error)))


def claude_message(model="claude-haiku-4-5", stop="end_turn", out=12, thinking=None):
    content = [SimpleNamespace(type="text", text="Hello! It finds moments fast.")]
    if thinking is not None:
        content.insert(0, SimpleNamespace(type="thinking", thinking=thinking))
    return SimpleNamespace(
        model=model,
        content=content,
        stop_reason=stop,
        usage=SimpleNamespace(input_tokens=21, output_tokens=out),
        _request_id="req_claude",
    )


def completion(model="deepseek-flash", finish="stop", out=9):
    return SimpleNamespace(
        id="cmpl_1",
        model=model,
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="Hi! It finds clips.", model_extra={}),
                finish_reason=finish,
            )
        ],
        usage=SimpleNamespace(prompt_tokens=18, completion_tokens=out),
        _request_id="req_ds",
    )


def chunk(content=None, finish=None, usage=None, reasoning=None):
    if content is None and finish is None and reasoning is None:
        choices = []
    else:
        extra = {"reasoning_content": reasoning} if reasoning else {}
        delta = SimpleNamespace(content=content, model_extra=extra)
        choices = [SimpleNamespace(delta=delta, finish_reason=finish)]
    return SimpleNamespace(id="chunk_1", model=HIVE_MODEL, choices=choices, usage=usage)


def usage(prompt, completion_tokens, reasoning=None):
    details = SimpleNamespace(reasoning_tokens=reasoning) if reasoning is not None else None
    return SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion_tokens,
        completion_tokens_details=details,
    )


class FakeStream:
    """Async iterator over chunks, like openai.AsyncStream, with the HTTP response attached."""

    def __init__(self, chunks):
        self._chunks = list(chunks)
        self.response = SimpleNamespace(headers={"x-request-id": "req_hive"})

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for c in self._chunks:
            yield c


def hive_stream():
    return FakeStream(
        [
            chunk(reasoning="The user wants a greeting. "),
            chunk(reasoning="Keep it short."),
            chunk("Hello"),
            chunk(" from Hive!"),
            chunk(finish="stop"),
            chunk(usage=usage(15, 6, reasoning=4)),  # usage arrives last
        ]
    )


def api_error(sdk, name, status):
    request = httpx2.Request("POST", "https://api.example.com/v1")
    return getattr(sdk, name)("boom", response=httpx2.Response(status, request=request), body=None)


def req(model_id=None, max_tokens=200, prompt="hi"):
    return TextGenerateRequest(prompt=prompt, max_tokens=max_tokens, model_id=model_id)


# --- one interface -----------------------------------------------------------------------


def test_every_provider_maps_to_a_protocol_adapter():
    assert set(PROVIDERS) == {spec.provider for spec in CATALOG.values()}
    for config in PROVIDERS.values():
        assert issubclass(PROTOCOLS[config.protocol], ProtocolAdapter)
    assert (HIVE.protocol, DEEPSEEK.protocol, ANTHROPIC.protocol) == (
        "openai",
        "openai",
        "anthropic",
    )


@pytest.mark.parametrize(
    ("raw", "hit_limit", "expected"),
    [
        ("end_turn", False, "end"),
        ("stop", False, "end"),
        ("stop_sequence", False, "end"),
        ("max_tokens", False, "length"),
        ("length", False, "length"),
        ("refusal", False, "refusal"),
        ("tool_use", False, "tool_use"),
        ("tool_calls", False, "tool_use"),
        ("content_filter", False, "content_filter"),
        ("pause_turn", False, "other"),
        (None, False, "other"),
        ("stop", True, "length"),  # Hive says "stop" when the output hit the limit
    ],
)
def test_stop_reasons_are_normalized(raw, hit_limit, expected):
    assert normalize_stop_reason(raw, hit_limit=hit_limit) == expected
    assert expected in STOP_REASONS


def test_output_budget_adds_reasoning_allowance_and_caps():
    assert output_budget(CATALOG[HIVE_MODEL], 128, 256) == 896  # 128 + 768
    assert output_budget(CATALOG[HIVE_MODEL], 256, 256) == 1024  # model cap
    assert output_budget(CATALOG["deepseek-flash"], 300, 256) == 256  # profile cap
    assert output_budget(CATALOG["claude-haiku-4-5"], 100, 256) == 100


ERROR_CASES = [
    ("RateLimitError", 429, True),
    ("InternalServerError", 500, True),
    ("APIStatusError", 503, True),
    ("AuthenticationError", 401, False),
    ("PermissionDeniedError", 403, False),
    ("BadRequestError", 400, False),
    ("NotFoundError", 404, False),
    ("UnprocessableEntityError", 422, False),
]


@pytest.mark.parametrize("sdk", [anthropic, openai], ids=["anthropic", "openai"])
@pytest.mark.parametrize(("name", "status", "retryable"), ERROR_CASES)
def test_one_error_mapper_serves_both_sdks(sdk, name, status, retryable):
    config = ProviderConfig(name="x", protocol="openai", key_env="X")
    assert map_sdk_error(api_error(sdk, name, status), sdk, config).retryable is retryable


@pytest.mark.parametrize("sdk", [anthropic, openai], ids=["anthropic", "openai"])
def test_connection_errors_are_retryable(sdk):
    error = sdk.APIConnectionError(request=httpx2.Request("POST", "https://x"))
    config = ProviderConfig(name="x", protocol="openai", key_env="X")
    assert map_sdk_error(error, sdk, config).retryable is True


@pytest.mark.parametrize(("config", "status"), [(HIVE, 405), (DEEPSEEK, 402)])
def test_provider_specific_permanent_statuses(config, status):
    assert map_sdk_error(api_error(openai, "APIStatusError", status), openai, config).retryable is (
        False
    )


@pytest.mark.parametrize("provider", ["hive", "deepseek", "anthropic"])
def test_missing_key_is_not_retryable_for_every_provider(monkeypatch, provider):
    monkeypatch.delenv(PROVIDERS[provider].key_env, raising=False)
    with pytest.raises(ModelCallError) as info:
        create_adapter(provider, TextModelSettings())
    assert info.value.retryable is False and PROVIDERS[provider].key_env in str(info.value)


def test_unknown_provider_is_rejected():
    with pytest.raises(ModelCallError):
        create_adapter("nope", TextModelSettings())


def test_hive_base_url_default_and_override(monkeypatch):
    monkeypatch.delenv("HIVE_BASE_URL", raising=False)
    assert HIVE.resolved_base_url() == "https://api-cdn.thehive.ai/api/v3/"
    monkeypatch.setenv("HIVE_BASE_URL", "https://api-va1.thehive.ai/api/v3/")
    assert HIVE.resolved_base_url() == "https://api-va1.thehive.ai/api/v3/"


# --- OpenAI protocol: Hive (streaming) and DeepSeek (non-streaming) ----------------------


async def test_hive_streams_and_assembles_reply_reasoning_and_usage():
    client = openai_client(hive_stream())
    adapter = OpenAICompatibleAdapter(HIVE, client, TextModelSettings())
    result = await adapter.generate(CATALOG[HIVE_MODEL], req(max_tokens=100))

    [params] = client.chat.completions.calls
    assert params["model"] == HIVE_MODEL
    assert params["stream"] is True and params["max_tokens"] == 868  # 100 + 768
    assert params["extra_headers"] == {"Accept": "text/event-stream"}
    assert result.provider == "hive" and result.text == "Hello from Hive!"
    assert result.reasoning_text == "The user wants a greeting. Keep it short."
    assert (result.input_tokens, result.output_tokens, result.reasoning_tokens) == (15, 6, 4)
    assert (result.stop_reason, result.provider_stop_reason) == ("end", "stop")
    assert result.request_id == "req_hive" and result.cost_usd == Decimal("0.000012")


async def test_hive_output_hitting_the_limit_is_length():
    stream = FakeStream(
        [chunk(reasoning="thinking..."), chunk(finish="stop"), chunk(usage=usage(30, 868))]
    )
    adapter = OpenAICompatibleAdapter(HIVE, openai_client(stream), TextModelSettings())
    result = await adapter.generate(CATALOG[HIVE_MODEL], req(max_tokens=100))
    assert result.text == "" and result.stop_reason == "length"
    assert result.provider_stop_reason == "stop"


async def test_deepseek_non_streaming_completion():
    client = openai_client(completion())
    adapter = OpenAICompatibleAdapter(DEEPSEEK, client, TextModelSettings(max_tokens_cap=64))
    result = await adapter.generate(CATALOG["deepseek-flash"], req())

    [params] = client.chat.completions.calls
    assert params["stream"] is False and params["max_tokens"] == 64
    assert "extra_headers" not in params
    assert result.provider == "deepseek" and result.text == "Hi! It finds clips."
    assert (result.stop_reason, result.provider_stop_reason) == ("end", "stop")
    assert result.request_id == "req_ds" and result.cost_usd == Decimal("0.000016")


async def test_openai_protocol_errors_go_through_the_shared_mapper():
    error = api_error(openai, "APIStatusError", 405)
    adapter = OpenAICompatibleAdapter(HIVE, openai_client(error=error), TextModelSettings())
    with pytest.raises(ModelCallError) as info:
        await adapter.generate(CATALOG[HIVE_MODEL], req())
    assert info.value.retryable is False and "405" in str(info.value)


# --- Anthropic protocol ------------------------------------------------------------------


async def test_anthropic_message_is_mapped_into_the_same_result():
    client = anthropic_client(claude_message())
    adapter = AnthropicAdapter(ANTHROPIC, client, TextModelSettings(max_tokens_cap=64))
    result = await adapter.generate(CATALOG["claude-haiku-4-5"], req())

    [params] = client.messages.calls
    assert params["model"] == "claude-haiku-4-5" and params["max_tokens"] == 64
    assert result.provider == "anthropic" and result.text == "Hello! It finds moments fast."
    assert (result.stop_reason, result.provider_stop_reason) == ("end", "end_turn")
    assert (result.input_tokens, result.output_tokens) == (21, 12)
    assert result.request_id == "req_claude" and result.cost_usd == Decimal("0.000081")


async def test_anthropic_thinking_blocks_become_reasoning():
    client = anthropic_client(claude_message(thinking="Short greeting needed."))
    adapter = AnthropicAdapter(ANTHROPIC, client, TextModelSettings())
    result = await adapter.generate(CATALOG["claude-haiku-4-5"], req())
    assert result.reasoning_text == "Short greeting needed."


async def test_anthropic_max_tokens_is_length():
    client = anthropic_client(claude_message(stop="max_tokens", out=5))
    adapter = AnthropicAdapter(ANTHROPIC, client, TextModelSettings())
    result = await adapter.generate(CATALOG["claude-haiku-4-5"], req())
    assert (result.stop_reason, result.provider_stop_reason) == ("length", "max_tokens")


async def test_opus_uses_beta_endpoint_with_fallbacks_and_low_effort():
    client = anthropic_client(claude_message(model="claude-opus-5"))
    adapter = AnthropicAdapter(ANTHROPIC, client, TextModelSettings())
    await adapter.generate(CATALOG["claude-opus-5"], req(max_tokens=32))
    assert client.messages.calls == []
    [params] = client.beta.messages.calls
    assert params["betas"] == ["server-side-fallback-2026-07-01"]
    assert params["fallbacks"] == "default" and params["output_config"] == {"effort": "low"}


async def test_anthropic_errors_go_through_the_shared_mapper():
    error = api_error(anthropic, "RateLimitError", 429)
    adapter = AnthropicAdapter(ANTHROPIC, anthropic_client(error=error), TextModelSettings())
    with pytest.raises(ModelCallError) as info:
        await adapter.generate(CATALOG["claude-haiku-4-5"], req())
    assert info.value.retryable is True


# --- router ------------------------------------------------------------------------------


class RecordingAdapter:
    def __init__(self, provider):
        self.provider, self.specs = provider, []

    async def generate(self, spec, request):
        self.specs.append(spec.model_id)
        return SimpleNamespace(provider=self.provider)


async def test_router_sends_each_model_to_its_provider():
    adapters = {name: RecordingAdapter(name) for name in PROVIDERS}
    router = TextGenerateRouter(TextModelSettings(), lambda name, _s: adapters[name])
    assert (await router.generate(req())).provider == "hive"  # default model
    assert (await router.generate(req(model_id="deepseek-flash"))).provider == "deepseek"
    assert (await router.generate(req(model_id="claude-sonnet-5"))).provider == "anthropic"
    assert adapters["hive"].specs == [HIVE_MODEL]


async def test_router_missing_key_only_affects_that_provider():
    claude = RecordingAdapter("anthropic")

    def factory(name, _settings):
        if name == "hive":
            raise ModelCallError("HIVE_API_KEY is not set", retryable=False)
        return claude

    router = TextGenerateRouter(TextModelSettings(), factory)
    with pytest.raises(ModelCallError):
        await router.generate(req())
    assert (await router.generate(req(model_id="claude-haiku-4-5"))).provider == "anthropic"


async def test_router_rejects_unknown_model():
    router = TextGenerateRouter(TextModelSettings(), lambda name, _s: RecordingAdapter(name))
    with pytest.raises(ModelCallError) as info:
        await router.generate(req(model_id="nope"))
    assert info.value.retryable is False
