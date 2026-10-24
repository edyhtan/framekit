"""Images, sampling and embeddings through the conversation and embedding ports.

No real API calls: the SDK clients are fakes with the same shape.
"""

import hashlib
import json
from dataclasses import replace
from decimal import Decimal
from types import SimpleNamespace

import openai
import pytest
from test_adapters import (
    FakeCreate,
    anthropic_client,
    api_error,
    claude_message,
    completion,
    openai_client,
)
from test_cache import MemoryStore

from framekit.contracts import (
    ConversationRequest,
    EmbeddingRequest,
    ImagePart,
    ModelCallError,
)
from framekit.models import (
    CATALOG,
    ConversationRouter,
    ModelSettings,
    ModelSource,
    ModelSpec,
    ProviderConfig,
)
from framekit.models.adapters.anthropic import AnthropicAdapter
from framekit.models.adapters.openai_compatible import OpenAICompatibleAdapter
from framekit.models.cache import CacheSettings, request_key, with_cache
from framekit.models.catalog import ANTHROPIC, DEEPSEEK

JPEG = ImagePart(b"\xff\xd8\xff fake jpeg bytes", "image/jpeg")
PNG = ImagePart(b"\x89PNG fake png bytes", "image/png")
SETTINGS = ModelSettings(max_tokens_cap=512)


def req(prompt="describe", **fields):
    return ConversationRequest(prompt, **({"max_tokens": 50} | fields))


# --- capability: images only to models that take them -------------------------------------


async def test_images_to_a_model_without_vision_fail_before_any_call():
    client = openai_client(completion())
    adapter = OpenAICompatibleAdapter(DEEPSEEK, client, SETTINGS)
    with pytest.raises(ModelCallError) as info:
        await adapter.generate(CATALOG["deepseek-flash"], req(images=(JPEG,)))
    assert not info.value.retryable and "doesn't accept images" in str(info.value)
    assert client.chat.completions.calls == []  # nothing was sent


def test_the_catalog_marks_claude_models_as_taking_images():
    assert {m for m, spec in CATALOG.items() if spec.vision} == {
        "claude-haiku-4-5",
        "claude-sonnet-5",
        "claude-opus-5",
    }


# --- images on the wire ------------------------------------------------------------------


async def test_openai_compatible_sends_images_first_then_the_text():
    client = openai_client(completion())
    adapter = OpenAICompatibleAdapter(DEEPSEEK, client, SETTINGS)
    spec = replace(CATALOG["deepseek-flash"], vision=True)
    await adapter.generate(spec, req("what is shown?", images=(JPEG, PNG), system="facts only"))
    [params] = client.chat.completions.calls
    assert params["messages"][0] == {"role": "system", "content": "facts only"}
    parts = params["messages"][-1]["content"]
    assert [p["type"] for p in parts] == ["image_url", "image_url", "text"]
    assert parts[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert parts[2] == {"type": "text", "text": "what is shown?"}


async def test_anthropic_sends_base64_image_blocks_then_the_text():
    client = anthropic_client(claude_message())
    adapter = AnthropicAdapter(ANTHROPIC, client, SETTINGS)
    await adapter.generate(CATALOG["claude-haiku-4-5"], req("what is shown?", images=(JPEG,)))
    [params] = client.messages.calls
    [image, text] = params["messages"][-1]["content"]
    assert image["type"] == "image" and image["source"]["type"] == "base64"
    assert image["source"]["media_type"] == "image/jpeg" and image["source"]["data"]
    assert text == {"type": "text", "text": "what is shown?"}


async def test_without_images_the_message_stays_plain_text():
    client = openai_client(completion())
    adapter = OpenAICompatibleAdapter(DEEPSEEK, client, SETTINGS)
    await adapter.generate(CATALOG["deepseek-flash"], req("hi"))
    [params] = client.chat.completions.calls
    assert params["messages"][-1] == {"role": "user", "content": "hi"}


# --- sampling ----------------------------------------------------------------------------


async def test_temperature_and_seed_reach_openai_compatible_providers():
    client = openai_client(completion())
    adapter = OpenAICompatibleAdapter(DEEPSEEK, client, SETTINGS)
    await adapter.generate(CATALOG["deepseek-flash"], req(temperature=0.2, seed=7))
    [params] = client.chat.completions.calls
    assert params["temperature"] == 0.2 and params["seed"] == 7


async def test_anthropic_takes_temperature_and_leaves_out_the_seed():
    client = anthropic_client(claude_message())
    adapter = AnthropicAdapter(ANTHROPIC, client, SETTINGS)
    await adapter.generate(CATALOG["claude-haiku-4-5"], req(temperature=0.4, seed=7))
    [params] = client.messages.calls
    assert params["temperature"] == 0.4 and "seed" not in params


async def test_unset_sampling_sends_nothing():
    client = openai_client(completion())
    adapter = OpenAICompatibleAdapter(DEEPSEEK, client, SETTINGS)
    await adapter.generate(CATALOG["deepseek-flash"], req())
    [params] = client.chat.completions.calls
    assert "temperature" not in params and "seed" not in params


# --- cache keys --------------------------------------------------------------------------


def old_key(request, model_id, namespace, method="generate"):
    """The key as it was before images and sampling existed."""
    shape = {
        "method": method,
        "model_id": model_id,
        "system": request.system,
        "history": [[m.role, m.content] for m in request.history],
        "prompt": request.prompt,
        "max_tokens": request.max_tokens,
        "reasoning": request.reasoning,
        "response_schema": request.response_schema,
    }
    blob = json.dumps(shape, sort_keys=True, separators=(",", ":"), default=str)
    return f"llmcache:{namespace}:{request.cache_class or 'default'}:" + hashlib.sha256(
        blob.encode()
    ).hexdigest()


def test_keys_of_requests_without_images_or_sampling_are_unchanged():
    request = req("hi", system="s", cache_class="lens_pass")
    assert request_key(request, "m", "v1") == old_key(request, "m", "v1")


def test_images_temperature_and_seed_each_change_the_key():
    base = req("hi")
    keys = {
        request_key(base, "m", "v1"),
        request_key(req("hi", images=(JPEG,)), "m", "v1"),
        request_key(req("hi", images=(PNG,)), "m", "v1"),
        request_key(req("hi", images=(JPEG, PNG)), "m", "v1"),
        request_key(req("hi", temperature=0.2), "m", "v1"),
        request_key(req("hi", seed=7), "m", "v1"),
        request_key(req("hi", seed=8), "m", "v1"),
    }
    assert len(keys) == 7
    same_bytes = ImagePart(bytes(JPEG.data), "image/jpeg")
    assert request_key(req("hi", images=(same_bytes,)), "m", "v1") == request_key(
        req("hi", images=(JPEG,)), "m", "v1"
    )


def test_the_image_bytes_never_appear_in_a_key():
    image = ImagePart(b"SECRET-FRAME-CONTENT" * 10, "image/jpeg")
    assert "SECRET-FRAME" not in request_key(req(images=(image,)), "m", "v1")


# --- embeddings --------------------------------------------------------------------------


EMBED_SPEC = ModelSpec("box", "embedder-1", "Embedder", Decimal("0.02"), Decimal("0"))
BOX = ProviderConfig("box", "openai", base_url="http://127.0.0.1:8000/v1")


def embedding_response(vectors, model="embedder-1", tokens=12):
    data = [SimpleNamespace(index=i, embedding=list(v)) for i, v in enumerate(vectors)]
    data.reverse()  # providers may return them in any order: sorted by index
    return SimpleNamespace(data=data, model=model, usage=SimpleNamespace(prompt_tokens=tokens))


def embedding_client(result=None, error=None):
    client = openai_client()
    client.embeddings = FakeCreate(result, error)
    return client


async def test_openai_compatible_embeddings_come_back_in_request_order():
    client = embedding_client(embedding_response([(1.0, 0.0), (0.0, 1.0), (0.6, 0.8)]))
    adapter = OpenAICompatibleAdapter(BOX, client, SETTINGS)
    result = await adapter.embed(EMBED_SPEC, EmbeddingRequest(("a", "b", "c")))
    [params] = client.embeddings.calls
    assert params == {"model": "embedder-1", "input": ["a", "b", "c"]}
    assert result.vectors == ((1.0, 0.0), (0.0, 1.0), (0.6, 0.8)) and result.dimension == 2
    assert result.input_tokens == 12 and result.cost_usd == Decimal("0.000000")
    assert result.provider == "box" and result.adapter_version == "0.7.0"


async def test_embedding_errors_go_through_the_shared_mapper():
    client = embedding_client(error=api_error(openai, "RateLimitError", 429))
    adapter = OpenAICompatibleAdapter(BOX, client, SETTINGS)
    with pytest.raises(ModelCallError) as info:
        await adapter.embed(EMBED_SPEC, EmbeddingRequest(("a",)))
    assert info.value.retryable


async def test_a_vector_count_mismatch_is_an_error():
    client = embedding_client(embedding_response([(1.0, 0.0)]))
    adapter = OpenAICompatibleAdapter(BOX, client, SETTINGS)
    with pytest.raises(ModelCallError, match="1 vectors for 2 texts"):
        await adapter.embed(EMBED_SPEC, EmbeddingRequest(("a", "b")))


async def test_anthropic_serves_no_embeddings():
    adapter = AnthropicAdapter(ANTHROPIC, anthropic_client(), SETTINGS)
    with pytest.raises(ModelCallError, match="doesn't serve embeddings") as info:
        await adapter.embed(CATALOG["claude-haiku-4-5"], EmbeddingRequest(("a",)))
    assert not info.value.retryable


class EmbeddingAdapter(OpenAICompatibleAdapter):
    """The OpenAI-compatible adapter with a fake client, for router tests."""

    @classmethod
    def make_client(cls, api_key, base_url, timeout_s):
        return embedding_client(embedding_response([(0.6, 0.8)]))


async def test_the_router_embeds_through_a_source_and_stamps_it():
    source = ModelSource("vectors", model=EMBED_SPEC, provider=BOX, adapter=EmbeddingAdapter)
    async with ConversationRouter(SETTINGS, sources=[source]) as router:
        result = await router.embed(EmbeddingRequest(("text",), model_id="vectors"))
    assert result.source == "vectors" and result.vectors == ((0.6, 0.8),)


async def test_the_response_cache_passes_embeddings_through():
    source = ModelSource("vectors", model=EMBED_SPEC, provider=BOX, adapter=EmbeddingAdapter)
    port = with_cache(
        lambda: ConversationRouter(SETTINGS, sources=[source]),
        settings=CacheSettings(mode="record"),
        store=MemoryStore(),
    )()
    result = await port.embed(EmbeddingRequest(("text",), model_id="vectors"))
    await port.aclose()
    assert result.vectors == ((0.6, 0.8),)
