"""Model response cache: record model calls once, replay them later (any request class)."""

import pytest
from helpers import final_result

from framekit.contracts.conversation import (
    ChatMessage,
    ConversationRequest,
    ModelCallError,
    TextDelta,
)
from framekit.models.cache import (
    CacheSettings,
    CachingConversationPort,
    ClassPolicy,
    Entry,
    FileResponseStore,
    request_key,
    with_cache,
)


class MemoryStore:
    def __init__(self, room=True):
        self.data, self.ttls, self.room = {}, {}, room

    async def get(self, key):
        return self.data.get(key)

    async def put(self, key, value, ttl_s):
        if not self.room:
            return False
        self.data[key], self.ttls[key] = value, ttl_s
        return True


class Model:
    """The inner port: counts real calls."""

    def __init__(self, text="Hello there", error=None):
        self.text, self.error = text, error
        self.generated, self.streamed, self.closed = 0, 0, 0

    def resolve(self, model_id):
        class Spec:
            pass

        spec = Spec()
        spec.model_id = model_id or "default-model"
        return spec

    async def generate(self, request):
        self.generated += 1
        if self.error:
            raise self.error
        return final_result(text=self.text)

    async def stream(self, request):
        self.streamed += 1
        yield TextDelta("reasoning", "thinking")
        for word in self.text.split():
            yield TextDelta("text", word + " ")
        if self.error:
            raise self.error
        yield final_result(text=self.text)

    async def aclose(self):
        self.closed += 1


def req(**overrides):
    fields = {"prompt": "hi", "max_tokens": 64, "cache_class": "search.respond"}
    return ConversationRequest(**(fields | overrides))


def cached_port(model, store=None, **settings):
    return CachingConversationPort(
        model, store or MemoryStore(), CacheSettings(**({"mode": "record"} | settings))
    )


# --- policy and keys ---------------------------------------------------------------------


def test_policy_per_class_and_enabled():
    settings = CacheSettings(
        mode="off", ttl_s=100, classes={"ai_test": ClassPolicy(mode="record", ttl_s=5)}
    )
    assert settings.policy("ai_test") == ("record", 5)
    assert settings.policy("search.respond") == ("off", 100)
    assert settings.policy(None) == ("off", 100)  # the "default" class
    assert settings.enabled()
    assert not CacheSettings(mode="off").enabled()


def test_off_everywhere_wraps_nothing():
    factory = object
    assert with_cache(factory, settings=CacheSettings(mode="off"), store=MemoryStore()) is factory


def test_key_changes_with_everything_that_shapes_a_reply():
    base = req(system="s", history=(ChatMessage("user", "a"),), response_schema={"type": "object"})
    key = request_key(base, "m", "v1")
    assert key == request_key(
        req(system="s", history=(ChatMessage("user", "a"),), response_schema={"type": "object"}),
        "m",
        "v1",
    )
    assert key.startswith("llmcache:v1:search.respond:")
    variants = [
        request_key(
            req(system="t", history=base.history, response_schema=base.response_schema), "m", "v1"
        ),
        request_key(req(system="s", response_schema=base.response_schema), "m", "v1"),
        request_key(req(system="s", history=base.history), "m", "v1"),
        request_key(
            req(
                system="s",
                history=base.history,
                response_schema=base.response_schema,
                max_tokens=65,
            ),
            "m",
            "v1",
        ),
        request_key(
            req(
                system="s",
                history=base.history,
                response_schema=base.response_schema,
                reasoning=True,
            ),
            "m",
            "v1",
        ),
        request_key(base, "other-model", "v1"),
        request_key(base, "m", "v2"),
        request_key(base, "m", "v1", method="stream"),
    ]
    assert len(set(variants) | {key}) == len(variants) + 1


# --- record and replay -------------------------------------------------------------------


async def test_generate_records_then_replays_without_a_call():
    model = Model()
    port = cached_port(model)
    first = await port.generate(req())
    again = await port.generate(req())
    assert model.generated == 1
    assert first.cached is False and again.cached is True
    assert again.text == first.text and again.input_tokens == first.input_tokens
    assert str(again.cost_usd) == "0" and again.adapter_version.endswith("+cache")


async def test_stream_replays_the_same_pieces_in_order():
    model = Model("one two three")
    port = cached_port(model)
    first = [p async for p in port.stream(req())]
    again = [p async for p in port.stream(req())]
    assert model.streamed == 1
    pieces = lambda run: [(p.kind, p.text) for p in run[:-1]]  # noqa: E731
    assert pieces(again) == pieces(first)
    assert again[-1].cached is True and again[-1].first_token_ms is not None


async def test_generate_and_stream_entries_are_separate():
    model = Model()
    port = cached_port(model)
    await port.generate(req())
    [p async for p in port.stream(req())]
    assert (model.generated, model.streamed) == (1, 1)


async def test_replay_mode_never_calls_the_provider():
    store = MemoryStore()
    await cached_port(Model(), store).generate(req())  # recorded earlier
    model = Model()
    port = cached_port(model, store, mode="replay")
    assert (await port.generate(req())).cached
    with pytest.raises(ModelCallError) as info:
        await port.generate(req(prompt="never recorded"))
    assert info.value.retryable is False and "replay mode" in str(info.value)
    assert model.generated == 0


async def test_class_policies():
    model = Model()
    port = cached_port(model, classes={"search.respond": ClassPolicy(mode="off")})
    await port.generate(req())
    await port.generate(req())
    assert model.generated == 2  # this class is never cached
    await port.generate(req(cache_class="ai_test"))
    await port.generate(req(cache_class="ai_test"))
    assert model.generated == 3  # the default (record) applies to the others


async def test_failures_and_oversized_replies_are_not_stored():
    store = MemoryStore()
    with pytest.raises(ModelCallError):
        await cached_port(Model(error=ModelCallError("down", retryable=True)), store).generate(
            req()
        )
    with pytest.raises(ModelCallError):
        [
            p
            async for p in cached_port(
                Model(error=ModelCallError("x", retryable=True)), store
            ).stream(req())
        ]
    assert store.data == {}
    await cached_port(Model("x" * 5000), store, max_entry_bytes=10).generate(req())
    assert store.data == {}


async def test_ttl_follows_the_class_policy():
    store = MemoryStore()
    port = cached_port(Model(), store, ttl_s=100, classes={"search.digest": ClassPolicy(ttl_s=9)})
    await port.generate(req())
    await port.generate(req(cache_class="search.digest"))
    assert sorted(store.ttls.values()) == [9, 100]


def test_entries_round_trip():
    entry = Entry(final_result(text="hé"), [("text", "hé", 12)])
    back = Entry.decode(entry.encode())
    assert back.result == entry.result and back.deltas == [("text", "hé", 12)]


def test_on_for_any_class_wraps_the_factory():
    model = Model()
    settings = CacheSettings(classes={"ai_test": ClassPolicy(mode="record")})
    port = with_cache(lambda: model, settings=settings, store=MemoryStore())()
    assert isinstance(port, CachingConversationPort)


# --- the file store ----------------------------------------------------------------------


async def test_file_store_round_trip_lifetime_and_clear(tmp_path, monkeypatch):
    store = FileResponseStore(tmp_path / "cache")
    assert await store.get("llmcache:v1:x:abc") is None
    assert await store.put("llmcache:v1:x:abc", b"reply", ttl_s=60)
    assert await store.get("llmcache:v1:x:abc") == b"reply"  # ':' in keys is fine

    import framekit.models.cache.file_store as file_store

    later = file_store.time.time() + 120
    monkeypatch.setattr(file_store.time, "time", lambda: later)
    assert await store.get("llmcache:v1:x:abc") is None  # expired, and removed
    monkeypatch.undo()

    await store.put("a", b"1", ttl_s=60)
    await store.put("b", b"2", ttl_s=60)
    assert store.clear() == 2 and await store.get("a") is None


async def test_file_store_failures_are_misses_and_skipped_writes(tmp_path):
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("x")
    store = FileResponseStore(blocker)  # a file where the directory should be
    assert await store.put("k", b"v", ttl_s=60) is False
    assert await store.get("k") is None


async def test_a_session_of_calls_replays_from_the_file_store(tmp_path):
    settings = CacheSettings(mode="record")
    model = Model()
    port = with_cache(lambda: model, settings=settings, store=FileResponseStore(tmp_path))()
    first = await port.generate(req())
    replay = with_cache(
        lambda: model, settings=CacheSettings(mode="replay"), store=FileResponseStore(tmp_path)
    )()
    again = await replay.generate(req())
    assert model.generated == 1 and again.cached and again.text == first.text


async def test_the_file_store_counts_and_wipes_by_class(tmp_path):
    store = FileResponseStore(tmp_path)
    await store.put("llmcache:v1:L3a:aaa", b"x" * 10, ttl_s=60)
    await store.put("llmcache:v1:L3b:bbb", b"y", ttl_s=60)
    await store.put("llmcache:v1:L3b:ccc", b"z", ttl_s=-1)  # already expired
    await store.put("anything else", b"w", ttl_s=60)
    stats = store.stats()
    assert stats["v1/L3a"]["entries"] == 1
    assert stats["v1/L3b"] == {"entries": 2, "bytes": 2 * (8 + 1), "expired": 1}
    assert store.clear(cache_class="L3b") == 2
    assert await store.get("llmcache:v1:L3a:aaa") == b"x" * 10
    assert set(store.stats()) == {"_other/_other", "v1/L3a"}


async def test_cache_class_names_never_escape_the_folder(tmp_path):
    store = FileResponseStore(tmp_path / "cache")
    await store.put("llmcache:..:../../x:key", b"v", ttl_s=60)
    assert all(p.is_relative_to(tmp_path / "cache") for p in tmp_path.rglob("*") if p.is_file())
    assert await store.get("llmcache:..:../../x:key") == b"v"
