"""The caching wrapper around a conversation port (normally the ConversationRouter).

Off / record / replay per cache class. Replays are marked: `cached=True`, cost 0, and the
adapter version suffixed `+cache`, so stored records never mistake a replay for a call."""

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import replace
from decimal import Decimal
from typing import Any

from framekit.contracts.conversation import (
    ConversationPort,
    ConversationRequest,
    ConversationResult,
    ModelCallError,
    TextDelta,
)
from framekit.contracts.response_store import ResponseStore
from framekit.models.cache.entries import Entry, request_key
from framekit.models.cache.settings import CacheSettings

log = logging.getLogger(__name__)
CACHED_SUFFIX = "+cache"


class CachingConversationPort:
    def __init__(self, inner: Any, store: ResponseStore, settings: CacheSettings) -> None:
        self._inner = inner
        self._store = store
        self._settings = settings

    def _model_id(self, request: ConversationRequest) -> str:
        resolve = getattr(self._inner, "resolve", None)  # the router knows the default model
        if resolve is not None:
            return resolve(request.model_id).model_id
        return request.model_id or "default"

    def _key(self, request: ConversationRequest, method: str) -> str:
        return request_key(request, self._model_id(request), self._settings.namespace, method)

    def _replayed(self, entry: Entry, started: float, first_at: float | None) -> ConversationResult:
        """The recorded result, marked as a replay: no cost, this replay's own timings."""
        original = entry.result
        return replace(
            original,
            cached=True,
            cost_usd=Decimal("0"),
            adapter_version=f"{original.adapter_version}{CACHED_SUFFIX}",
            latency_ms=round((time.perf_counter() - started) * 1000),
            first_token_ms=round((first_at - started) * 1000) if first_at else None,
        )

    def _miss_in_replay(self, request: ConversationRequest) -> ModelCallError:
        return ModelCallError(
            f"model cache miss in replay mode (class {request.cache_class or 'default'!r})",
            retryable=False,
        )

    async def _save(self, key: str, entry: Entry, ttl_s: int) -> None:
        blob = entry.encode()
        if len(blob) > self._settings.max_entry_bytes:
            log.info("model cache: reply too large to store (%d bytes)", len(blob))
            return
        await self._store.put(key, blob, ttl_s)

    async def generate(self, request: ConversationRequest) -> ConversationResult:
        mode, ttl_s = self._settings.policy(request.cache_class)
        if mode == "off":
            return await self._inner.generate(request)
        started = time.perf_counter()
        key = self._key(request, "generate")
        if (blob := await self._store.get(key)) is not None:
            return self._replayed(Entry.decode(blob), started, None)
        if mode == "replay":
            raise self._miss_in_replay(request)
        result = await self._inner.generate(request)
        await self._save(key, Entry(result), ttl_s)
        return result

    async def stream(
        self, request: ConversationRequest
    ) -> AsyncIterator[TextDelta | ConversationResult]:
        mode, ttl_s = self._settings.policy(request.cache_class)
        if mode == "off":
            async for piece in self._inner.stream(request):
                yield piece
            return
        started = time.perf_counter()
        key = self._key(request, "stream")
        if (blob := await self._store.get(key)) is not None:
            entry = Entry.decode(blob)
            first_at = None
            for kind, text, at_ms in entry.deltas:
                if self._settings.replay_pace == "original":
                    await asyncio.sleep(max(0.0, at_ms / 1000 - (time.perf_counter() - started)))
                first_at = first_at or time.perf_counter()
                yield TextDelta(kind, text)  # type: ignore[arg-type]
            yield self._replayed(entry, started, first_at)
            return
        if mode == "replay":
            raise self._miss_in_replay(request)
        deltas: list[tuple[str, str, int]] = []
        async for piece in self._inner.stream(request):
            if isinstance(piece, TextDelta):
                deltas.append(
                    (piece.kind, piece.text, round((time.perf_counter() - started) * 1000))
                )
            else:
                await self._save(key, Entry(piece, deltas), ttl_s)
            yield piece

    def resolve(self, model_id: str | None) -> Any:
        return self._inner.resolve(model_id)

    async def aclose(self) -> None:
        await self._inner.aclose()  # the store's client is shared, not closed here


def with_cache(
    factory: Callable[[], Any], *, settings: CacheSettings, store: ResponseStore
) -> Callable[[], ConversationPort]:
    """Wrap a port factory with the cache. With caching off for every class, the factory
    is returned unchanged (no overhead, the store is never touched)."""
    if not settings.enabled():
        return factory

    def build() -> ConversationPort:
        return CachingConversationPort(factory(), store, settings)

    return build
