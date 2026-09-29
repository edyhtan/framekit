"""Reasoning is an explicit option: off unless asked for, switched per model from the catalog."""

import pytest
from test_chat import FakeGenerator, final_result, use_generator
from test_llm_adapters import (
    HIVE_MODEL,
    FakeStream,
    anthropic_client,
    chunk,
    claude_message,
    completion,
    openai_client,
    req,
    usage,
)

from clipsearch_adapters.external_connections.adapters.anthropic_text import AnthropicAdapter
from clipsearch_adapters.external_connections.adapters.external_ai_adapter_base import (
    merge_fields,
)
from clipsearch_adapters.external_connections.adapters.openai_compatible import (
    OpenAICompatibleAdapter,
)
from clipsearch_adapters.external_connections.catalog import (
    ANTHROPIC,
    CATALOG,
    DEEPSEEK,
    HIVE,
    TextModelSettings,
)


def hive_stream():
    return FakeStream([chunk("4", finish="stop"), chunk(usage=usage(10, 2))])


async def hive_params(model_id, reasoning):
    client = openai_client(hive_stream())
    adapter = OpenAICompatibleAdapter(HIVE, client, TextModelSettings())
    result = await adapter.generate(CATALOG[model_id], req(max_tokens=100, reasoning=reasoning))
    [params] = client.chat.completions.calls
    return params, result


async def test_hive_deepseek_reasoning_is_off_by_default_and_capped_to_the_answer():
    params, result = await hive_params(HIVE_MODEL, reasoning=False)
    assert params["extra_body"] == {"reasoning_effort": "none"}
    assert params["max_tokens"] == 100  # no reasoning allowance
    assert result.reasoning_on is False


async def test_hive_deepseek_reasoning_on_uses_the_default_and_the_allowance():
    params, result = await hive_params(HIVE_MODEL, reasoning=True)
    assert "extra_body" not in params
    assert params["max_tokens"] == 868  # 100 + 768
    assert result.reasoning_on is True


async def test_glm_uses_its_own_off_switch():
    params, _ = await hive_params("zai-org/glm-5.3-flash", reasoning=False)
    assert params["extra_body"] == {"reasoning_effort": "low"}  # "none" backfires on GLM


@pytest.mark.parametrize(
    ("reasoning", "expected"),
    [
        (False, {"extra_body": {"thinking": {"type": "disabled"}}}),
        (True, {"extra_body": {"thinking": {"type": "enabled"}}, "reasoning_effort": "low"}),
    ],
)
async def test_deepseek_api_switches_thinking_explicitly(reasoning, expected):
    client = openai_client(completion())
    adapter = OpenAICompatibleAdapter(DEEPSEEK, client, TextModelSettings())
    await adapter.generate(CATALOG["deepseek-flash"], req(reasoning=reasoning))
    [params] = client.chat.completions.calls
    assert {k: params[k] for k in expected} == expected


async def anthropic_params(model_id, reasoning, max_tokens=100):
    client = anthropic_client(claude_message(model=model_id))
    adapter = AnthropicAdapter(ANTHROPIC, client, TextModelSettings())
    request = req(max_tokens=max_tokens, reasoning=reasoning)
    await adapter.generate(CATALOG[model_id], request)
    calls = client.beta.messages.calls or client.messages.calls
    return calls[0]


@pytest.mark.parametrize("model_id", ["claude-sonnet-5", "claude-opus-5"])
async def test_claude_5_thinking_is_disabled_unless_asked_and_effort_is_kept(model_id):
    off = await anthropic_params(model_id, reasoning=False)
    assert off["thinking"] == {"type": "disabled"}
    assert off["output_config"] == {"effort": "low"}  # merged, not replaced
    assert off["max_tokens"] == 100

    on = await anthropic_params(model_id, reasoning=True)
    assert on["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert on["max_tokens"] == 868


async def test_haiku_thinking_budget_fits_below_max_tokens():
    off = await anthropic_params("claude-haiku-4-5", reasoning=False)
    assert "thinking" not in off and off["max_tokens"] == 100

    on = await anthropic_params("claude-haiku-4-5", reasoning=True, max_tokens=256)
    assert on["thinking"] == {"type": "enabled", "budget_tokens": 1024}
    assert on["max_tokens"] == 1280 and on["thinking"]["budget_tokens"] < on["max_tokens"]


def test_merge_fields_merges_nested_request_fields():
    merged = merge_fields({"a": 1, "extra_body": {"x": 1}}, {"extra_body": {"y": 2}, "b": 2})
    assert merged == {"a": 1, "b": 2, "extra_body": {"x": 1, "y": 2}}


def test_every_catalog_model_can_switch_reasoning_off():
    assert {spec.reasoning_mode for spec in CATALOG.values()} == {"optional"}


# --- api ---------------------------------------------------------------------------------


async def test_models_listing_says_how_each_model_reasons(client):
    body = (await client.get("/prototype/ai-test/models")).json()
    assert {m["reasoning"] for m in body["models"]} <= {"optional", "always", "none"}


async def test_chat_passes_the_reasoning_option_and_reports_it(client):
    gen = FakeGenerator([final_result(reasoning_on=True)])
    use_generator(client, gen)
    response = await client.post("/prototype/chat", json={"message": "hi", "reasoning": True})
    assert gen.requests[0].reasoning is True
    assert '"reasoning_on": true' in response.text

    gen = FakeGenerator([final_result()])
    use_generator(client, gen)
    await client.post("/prototype/chat", json={"message": "hi"})
    assert gen.requests[0].reasoning is False  # off unless asked for
