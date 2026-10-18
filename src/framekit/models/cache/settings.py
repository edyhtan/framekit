"""Cache policy: a default mode and lifetime, overridable per cache class."""

from functools import lru_cache
from typing import Literal

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

Mode = Literal["off", "record", "replay"]
DEFAULT_CLASS = "default"


class ClassPolicy(BaseModel):
    mode: Mode | None = None  # None: the default mode
    ttl_s: int | None = None  # None: the default lifetime


class ModelCacheSettings(BaseSettings):
    """`CLIPSEARCH_LLM_CACHE_MODE=record`, and per class, e.g.
    `CLIPSEARCH_LLM_CACHE_CLASSES='{"search.respond": {"mode": "off"}}'`."""

    model_config = SettingsConfigDict(
        env_prefix="CLIPSEARCH_LLM_CACHE_", env_file=".env", extra="ignore"
    )

    mode: Mode = "off"
    classes: dict[str, ClassPolicy] = {}
    ttl_s: int = 7 * 24 * 3600
    # Changing the namespace invalidates every entry at once (e.g. after a prompt rework).
    namespace: str = "v1"
    max_entry_bytes: int = 256_000  # compressed; larger replies are not stored
    # Stop writing once Valkey uses this share of its maxmemory (it refuses writes when
    # full, which would also block the task queue).
    memory_guard: float = 0.8
    # Replayed streams: "instant", or "original" to keep the recorded pacing.
    replay_pace: Literal["instant", "original"] = "instant"

    def policy(self, cache_class: str | None) -> tuple[Mode, int]:
        override = self.classes.get(cache_class or DEFAULT_CLASS, ClassPolicy())
        return override.mode or self.mode, override.ttl_s or self.ttl_s

    def enabled(self) -> bool:
        """Whether any class can use the cache (if not, nothing is wrapped at all)."""
        return self.mode != "off" or any(p.mode not in (None, "off") for p in self.classes.values())


@lru_cache
def get_model_cache_settings() -> ModelCacheSettings:
    return ModelCacheSettings()
