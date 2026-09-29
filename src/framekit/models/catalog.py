"""Model catalog and prototype model profile for the `text_generate` port.

Each entry names its provider; the router sends requests to that provider's adapter.
Prices are per 1M tokens, used only to record an estimated cost (never displayed).
"""

from dataclasses import dataclass, field
from decimal import Decimal
from functools import lru_cache
from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict

MILLION = Decimal(1_000_000)


@dataclass(frozen=True)
class ModelSpec:
    provider: str
    model_id: str
    label: str
    input_usd_per_mtok: Decimal
    output_usd_per_mtok: Decimal
    # Extra request fields this model needs for a short, cheap probe.
    request_extras: dict[str, Any] = field(default_factory=dict)
    betas: tuple[str, ...] = ()
    # Models that always reason before answering get extra output room for it: the job's
    # max_tokens is the answer budget, and this allowance is added on top.
    reasoning_allowance: int = 0
    max_tokens_cap: int | None = None  # overrides the profile-wide cap


_SPECS = (
    # Hive-hosted models: standard (non-discounted) rates from thehive.ai/pricing, 2026-09-29.
    ModelSpec(
        "hive",
        "deepseek-ai/deepseek-v4.1-flash",
        "DeepSeek V4.1 Flash (Hive)",
        Decimal("0.30"),
        Decimal("1.20"),
        # Reasons on every request on Hive (thinking can't be disabled; checked 2026-09-29).
        reasoning_allowance=768,
        max_tokens_cap=1024,
    ),
    ModelSpec(
        "hive",
        "zai-org/glm-5.3-flash",
        "GLM-5.3 Flash (Hive)",
        Decimal("0.15"),
        Decimal("0.50"),
        reasoning_allowance=768,
        max_tokens_cap=1024,
    ),
    # DeepSeek's own API: peak-hour cache-miss rates (api-docs.deepseek.com, 2026-09-29).
    ModelSpec(
        "deepseek",
        "deepseek-flash",
        "DeepSeek V4.1 Flash (DeepSeek API)",
        Decimal("0.3"),
        Decimal("1.2"),
    ),
    # Anthropic first-party rates (claude-api skill cache, 2026-06-24).
    ModelSpec("anthropic", "claude-haiku-4-5", "Claude Haiku 4.5", Decimal("1"), Decimal("5")),
    ModelSpec(
        "anthropic",
        "claude-sonnet-5",
        "Claude Sonnet 5",
        Decimal("2"),
        Decimal("10"),
        request_extras={"output_config": {"effort": "low"}},
    ),
    ModelSpec(
        "anthropic",
        "claude-opus-5",
        "Claude Opus 5",
        Decimal("5"),
        Decimal("25"),
        # Thinking is on by default on Opus 5; low effort keeps a probe cheap.
        # Server-side refusal fallback, as recommended for Opus 5 code.
        request_extras={"output_config": {"effort": "low"}, "fallbacks": "default"},
        betas=("server-side-fallback-2026-07-01",),
    ),
)
CATALOG: dict[str, ModelSpec] = {spec.model_id: spec for spec in _SPECS}


class TextModelSettings(BaseSettings):
    """Prototype model profile for the test AI worker (`CLIPSEARCH_AI_TEST_*`)."""

    model_config = SettingsConfigDict(
        env_prefix="CLIPSEARCH_AI_TEST_", env_file=".env", extra="ignore"
    )

    default_model: str = "deepseek-ai/deepseek-v4.1-flash"
    max_tokens_cap: int = 256
    timeout_s: float = 30.0


@lru_cache
def get_text_model_settings() -> TextModelSettings:
    return TextModelSettings()


def estimate_cost(spec: ModelSpec, input_tokens: int, output_tokens: int) -> Decimal:
    cost = (
        Decimal(input_tokens) * spec.input_usd_per_mtok
        + Decimal(output_tokens) * spec.output_usd_per_mtok
    ) / MILLION
    return cost.quantize(Decimal("0.000001"))


def catalog_listing() -> list[dict[str, Any]]:
    default = get_text_model_settings().default_model
    return [
        {
            "provider": spec.provider,
            "model_id": spec.model_id,
            "label": spec.label,
            "default": spec.model_id == default,
        }
        for spec in CATALOG.values()
    ]
