"""Model response cache: record model calls once, replay them later (any request class)."""

import fnmatch
import json

import pytest
import redis
from test_chat import final_result, parse_sse
from test_search_api import new_session, turn

from clipsearch_adapters.model_cache import CachingConversationPort, with_cache
from clipsearch_adapters.model_cache import __main__ as cli
from clipsearch_adapters.model_cache.entries import Entry, request_key
from clipsearch_adapters.model_cache.settings import ClassPolicy, ModelCacheSettings
from clipsearch_adapters.model_cache.store import ValkeyResponseStore
from clipsearch_domain.common.ports import (
    ChatMessage,
    ConversationRequest,
    ModelCallError,
    TextDelta,
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
        model, store or MemoryStore(), ModelCacheSettings(**({"mode": "record"} | settings))
    )


# --- policy and keys ---------------------------------------------------------------------


def test_policy_per_class_and_enabled():
    settings = ModelCacheSettings(
        mode="off", ttl_s=100, classes={"ai_test": ClassPolicy(mode="record", ttl_s=5)}
    )
    assert settings.policy("ai_test") == ("record", 5)
    assert settings.policy("search.respond") == ("off", 100)
    assert settings.policy(None) == ("off", 100)  # the "default" class
    assert settings.enabled()
    assert not ModelCacheSettings(mode="off").enabled()


def test_off_everywhere_wraps_nothing():
    factory = object
    assert with_cache(factory, settings=ModelCacheSettings(mode="off")) is factory


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
        request_key(
            req(
                system="s",
                history=base.history,
                response_schema=base.response_schema,
                family="langchain",
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


# --- the Valkey store --------------------------------------------------------------------


class FakeValkey:
    def __init__(self, used=0, limit=0, broken=False):
        self.data, self.ex, self.used, self.limit, self.broken = {}, {}, used, limit, broken

    async def get(self, key):
        if self.broken:
            raise redis.ConnectionError("down")
        return self.data.get(key)

    async def set(self, key, value, ex):
        if self.broken:
            raise redis.ConnectionError("down")
        self.data[key], self.ex[key] = value, ex

    async def info(self, section):
        if self.broken:
            raise redis.ConnectionError("down")
        return {"used_memory": self.used, "maxmemory": self.limit}


async def test_valkey_store_sets_a_lifetime_and_respects_the_memory_guard():
    client = FakeValkey(used=10, limit=100)
    store = ValkeyResponseStore(client, memory_guard=0.8)
    assert await store.put("k", b"v", 60) and client.ex["k"] == 60
    assert await store.get("k") == b"v"

    full = ValkeyResponseStore(FakeValkey(used=81, limit=100), memory_guard=0.8)
    assert await full.put("k", b"v", 60) is False  # would risk blocking the task queue


async def test_valkey_store_outages_degrade_to_misses():
    store = ValkeyResponseStore(FakeValkey(broken=True), memory_guard=0.8)
    assert await store.get("k") is None
    assert await store.put("k", b"v", 60) is False
    model = Model()
    port = cached_port(model, store)
    assert (await port.generate(req())).cached is False  # the call still works


def test_cli_stats_and_wipe_by_class(monkeypatch, capsys):
    class SyncFake:
        def __init__(self):
            self.data = {
                b"llmcache:v1:search.respond:a": b"xx",
                b"llmcache:v1:search.respond:b": b"x",
                b"llmcache:v1:ai_test:c": b"xxx",
            }

        def scan_iter(self, match, count):
            return [k for k in self.data if fnmatch.fnmatchcase(k.decode(), match)]

        def strlen(self, key):
            return len(self.data[key.encode()])

        def delete(self, *keys):
            return sum(self.data.pop(k.encode(), None) is not None for k in keys)

    fake = SyncFake()
    monkeypatch.setattr(cli, "sync_client", lambda url: fake)
    assert cli.main(["stats"]) == 0
    out = capsys.readouterr().out
    assert "entries=3" in out and "search.respond" in out and "ai_test" in out
    assert cli.main(["wipe", "--class", "search.respond"]) == 0
    assert list(fake.data) == [b"llmcache:v1:ai_test:c"]


# --- a whole session, replayed -----------------------------------------------------------


class StreamingModel(Model):
    async def generate(self, request):  # the Interpreter
        self.generated += 1
        reading = {"message_type": "chat", "reading": "a greeting"}
        return final_result(text=json.dumps(reading))


async def test_a_session_replays_from_the_cache_with_no_model_calls(client, session_factory):
    from sqlalchemy import select

    from clipsearch_domain.jobs.entities import Run

    model = StreamingModel("Hi! How can I help?")
    store = MemoryStore()
    state = client._transport.app.state
    state.conversation_factory = with_cache(
        lambda: model, settings=ModelCacheSettings(mode="record"), store=store
    )

    async def run_session():
        session_id = await new_session(client)
        events = []
        for n, message in enumerate(["hello", "what can you do?"]):
            response = await turn(client, session_id, client_turn_id=f"r{n}", message=message)
            events.append(parse_sse(response.text))
        return events

    first = await run_session()
    calls_after_first = (model.generated, model.streamed)
    second = await run_session()

    assert (model.generated, model.streamed) == calls_after_first == (2, 2)
    texts = lambda run: ["".join(d["text"] for n, d in t if n == "text") for t in run]  # noqa: E731
    assert texts(second) == texts(first)
    done = second[-1][-1][1]
    assert [c["cached"] for c in done["calls"]] == [True, True]
    async with session_factory() as db:
        replayed = [
            r
            for r in (await db.execute(select(Run))).scalars()
            if r.adapter_version.endswith("+cache")
        ]
    assert len(replayed) == 4 and all(str(r.cost_usd) in ("0", "0.000000") for r in replayed)
