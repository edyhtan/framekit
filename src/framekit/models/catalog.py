"""Everything the model layer needs to know, as data: providers, models and the profile.

- `PROVIDERS`: how to reach each provider (protocol, URL, key variable, streaming, …)
- `CATALOG`: the selectable models, each naming its provider
- `TextModelSettings`: the prototype model profile (default model, caps, timeout)

Adding an OpenAI-compatible provider is one `ProviderConfig` plus its models, here.
Prices are per 1M tokens, used only to record an estimated cost (never displayed).
"""

import os
from dataclasses import dataclass, field
from decimal import Decimal
from functools import lru_cache
from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict

MILLION = Decimal(1_000_000)


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    protocol: str  # "openai" (Chat Completions) or "anthropic" (Messages)
    key_env: str
    base_url: str | None = None  # None: the SDK's default endpoint
    base_url_env: str | None = None  # optional override, e.g. a regional endpoint
    stream: bool = False
    extra_headers: dict[str, str] = field(default_factory=dict)
    # Provider-specific statuses that mean "don't retry" beyond the usual 4xx.
    permanent_statuses: frozenset[int] = frozenset()

    def resolved_base_url(self) -> str | None:
        if self.base_url_env and os.environ.get(self.base_url_env):
            return os.environ[self.base_url_env]
        return self.base_url


PROVIDERS: dict[str, ProviderConfig] = {
    config.name: config
    for config in (
        ProviderConfig(
            name="hive",
            protocol="openai",
            key_env="HIVE_API_KEY",
            base_url="https://api-cdn.thehive.ai/api/v3/",
            base_url_env="HIVE_BASE_URL",  # Virginia customers: https://api-va1.thehive.ai/api/v3/
            stream=True,  # the only path Hive documents
            extra_headers={"Accept": "text/event-stream"},
            permanent_statuses=frozenset({405}),  # organization paused / out of balance
        ),
        ProviderConfig(
            name="deepseek",
            protocol="openai",
            key_env="DEEPSEEK_API_KEY",
            base_url="https://api.deepseek.com",
            permanent_statuses=frozenset({402}),  # out of balance
        ),
        ProviderConfig(
            name="anthropic",
            protocol="anthropic",
            key_env="ANTHROPIC_API_KEY",
        ),
    )
}

HIVE, DEEPSEEK, ANTHROPIC = PROVIDERS["hive"], PROVIDERS["deepseek"], PROVIDERS["anthropic"]


@dataclass(frozen=True)
class ReasoningControl:
    """How a model's reasoning is switched on and off, as extra request fields.

    Reasoning is billed as output tokens, so it's off unless the request asks for it.
    The switch differs per model, even on one provider, so it's data, not code.
    """

    on: dict[str, Any] = field(default_factory=dict)
    off: dict[str, Any] | None = None  # None: this model can't switch reasoning off
    # While reasoning is on: output room added on top of the answer budget, and the cap
    # that replaces the profile-wide one.
    allowance: int = 0
    max_tokens_cap: int | None = None


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
    reasoning: ReasoningControl | None = None  # None: the model doesn't reason

    @property
    def reasoning_mode(self) -> str:
        """`optional`, `always` or `none`, for clients choosing the option."""
        if self.reasoning is None:
            return "none"
        return "always" if self.reasoning.off is None else "optional"


_SPECS = (
    # Hive-hosted models: standard (non-discounted) rates from thehive.ai/pricing, 2026-09-29.
    ModelSpec(
        "hive",
        "deepseek-ai/deepseek-v4.1-flash",
        "DeepSeek V4.1 Flash (Hive)",
        Decimal("0.30"),
        Decimal("1.20"),
        # Reasons by default on Hive. Measured 2026-09-29: `reasoning_effort: "none"` turns
        # it off (2 output tokens instead of 27); DeepSeek's `thinking: disabled` is ignored.
        reasoning=ReasoningControl(
            off={"extra_body": {"reasoning_effort": "none"}}, allowance=768, max_tokens_cap=1024
        ),
    ),
    ModelSpec(
        "hive",
        "zai-org/glm-5.3-flash",
        "GLM-5.3 Flash (Hive)",
        Decimal("0.15"),
        Decimal("0.50"),
        # Measured 2026-09-29 (3 runs each): `"low"` gives 0-1 reasoning characters, while
        # `"none"` backfires and reasons until the limit. Hive doesn't report its
        # reasoning tokens for this model.
        reasoning=ReasoningControl(
            off={"extra_body": {"reasoning_effort": "low"}}, allowance=768, max_tokens_cap=1024
        ),
    ),
    # DeepSeek's own API: peak-hour cache-miss rates (api-docs.deepseek.com, 2026-09-29).
    # Thinking is on by default at effort "high" (api-docs.deepseek.com/guides/thinking_mode);
    # not verified live (no key yet).
    ModelSpec(
        "deepseek",
        "deepseek-flash",
        "DeepSeek V4.1 Flash (DeepSeek API)",
        Decimal("0.3"),
        Decimal("1.2"),
        reasoning=ReasoningControl(
            on={"extra_body": {"thinking": {"type": "enabled"}}, "reasoning_effort": "low"},
            off={"extra_body": {"thinking": {"type": "disabled"}}},
            allowance=768,
            max_tokens_cap=1024,
        ),
    ),
    # Anthropic first-party rates (claude-api skill cache, 2026-06-24).
    ModelSpec(
        "anthropic",
        "claude-haiku-4-5",
        "Claude Haiku 4.5",
        Decimal("1"),
        Decimal("5"),
        # Off by default; when on, the thinking budget must be at least 1024 and below
        # max_tokens, so the allowance is 1024 on top of the answer.
        reasoning=ReasoningControl(
            on={"thinking": {"type": "enabled", "budget_tokens": 1024}},
            off={},
            allowance=1024,
            max_tokens_cap=1280,
        ),
    ),
    ModelSpec(
        "anthropic",
        "claude-sonnet-5",
        "Claude Sonnet 5",
        Decimal("2"),
        Decimal("10"),
        request_extras={"output_config": {"effort": "low"}},
        # Adaptive thinking is on by default on Sonnet 5 and Opus 5, with its text hidden
        # unless `display: summarized` is asked for. Off needs an explicit `disabled`,
        # which Opus 5 accepts at effort `high` or lower (we use `low`).
        reasoning=ReasoningControl(
            on={"thinking": {"type": "adaptive", "display": "summarized"}},
            off={"thinking": {"type": "disabled"}},
            allowance=768,
            max_tokens_cap=1024,
        ),
    ),
    ModelSpec(
        "anthropic",
        "claude-opus-5",
        "Claude Opus 5",
        Decimal("5"),
        Decimal("25"),
        # Low effort keeps a probe cheap. Server-side refusal fallback, as recommended for
        # Opus 5 code.
        request_extras={"output_config": {"effort": "low"}, "fallbacks": "default"},
        betas=("server-side-fallback-2026-07-01",),
        reasoning=ReasoningControl(
            on={"thinking": {"type": "adaptive", "display": "summarized"}},
            off={"thinking": {"type": "disabled"}},
            allowance=768,
            max_tokens_cap=1024,
        ),
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
            "reasoning": spec.reasoning_mode,
        }
        for spec in CATALOG.values()
    ]
