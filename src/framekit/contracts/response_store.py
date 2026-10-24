"""The `ResponseStore` port: where the model response cache keeps its entries.

framekit ships a file store (`framekit.models.cache.FileResponseStore`); a consumer can
plug in its own (e.g. a key-value server). Stores never raise to the caller: a failed
read is a miss and a failed write is a skipped write, so caching can't break a call.
"""

from typing import Protocol


class ResponseStore(Protocol):
    async def get(self, key: str) -> bytes | None:
        """The stored value, or None if absent, expired or unreadable."""
        ...

    async def put(self, key: str, value: bytes, ttl_s: int) -> bool:
        """Store for `ttl_s` seconds; False if it wasn't stored."""
        ...
