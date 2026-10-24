"""Everything the model layer needs to know, as data: providers, models and the profile.

- `PROVIDERS`: how to reach each provider (protocol, URL, key variable, streaming, …)
- `CATALOG`: the selectable models, each naming its provider
- `ModelSettings`: the defaults a router uses (default model, output cap, timeout)

These dicts are the built-ins, served by `framekit.models.DEFAULT_REGISTRY`. Your own
providers and models go into a registry of your own (`DEFAULT_REGISTRY.extend(...)`,
docs/adapters.md), not into these dicts: editing them in place still reaches the default
router, but it's unsupported and changes every router in the process.
Prices are per 1M tokens, used only to record an estimated cost (never displayed).
"""

import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from pydantic import BaseModel

MILLION = Decimal(1_000_000)


@dataclass(frozen=True)
class ProviderConfig:
    """How to reach one provider. Plain data: the built-ins are in `PROVIDERS`; your own
    go into a `ModelRegistry`. framekit sends the provider's key to its base URL, so point
    a provider only at a host you trust (docs/adapters.md, "Security")."""

    name: str
    # The wire protocol, which picks the adapter: "openai" (Chat Completions) or
    # "anthropic" (Messages) built in, or one a registry adds.
    protocol: str
    # The environment variable holding the key. None: the provider takes no key (e.g. a
    # local server); its adapter is then built with PLACEHOLDER_KEY, never with None.
    key_env: str | None = None
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
    # How to apply a response schema: "json_schema" (provider enforces the schema),
    # "json_mode" (valid JSON only), "function_calling", or "prompted" (instructions only).
    # Adapters fall back to "prompted" for methods they can't apply. Set from probes.
    structured_method: str = "prompted"
    # Context window in tokens, from the provider's documentation; None until recorded.
    context_window: int | None = None
    # Accepts images (`ConversationRequest.images`), per the provider's documentation.
    # A request with images to a model without it fails before any call.
    vision: bool = False
    # Image limits per request, from the provider's documentation (None: not stated).
    # Extraction checks vision profiles against them before any spend, so nothing is
    # downscaled or rejected by the provider halfway through a run.
    max_images: int | None = None
    max_image_side_px: int | None = None  # the long edge the model reads without resizing

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
        # Probes 2026-09-30: schema via a forced tool call returned valid JSON; Hive
        # rejects a native json_schema response_format (400), so the method is
        # function_calling (sent non-streamed). json_mode and "prompted" also work.
        structured_method="function_calling",
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
        # Probes 2026-09-30: same results as above (tool calling valid in 17 output tokens;
        # native json_schema response_format rejected with 400).
        structured_method="function_calling",
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
        # Context windows for Claude: claude-api skill model table (2026). Structured output
        # stays "prompted" until probed with a key.
        context_window=200_000,
        vision=True,  # Claude models take image input (Messages API image blocks)
        # platform.claude.com/docs/en/build-with-claude/vision (2026-10-24): 100 images per
        # request for 200k-context models; 1,568 px native long edge on this tier (larger
        # images are downscaled; over 20 images, 2,000 px is a hard limit).
        max_images=100,
        max_image_side_px=1568,
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
        context_window=1_000_000,
        vision=True,
        # Same page: 600 images per request for other models; native long edge 2,576 px,
        # but a request with over 20 images rejects any side over 2,000 px, so 2,000.
        max_images=600,
        max_image_side_px=2000,
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
        context_window=1_000_000,
        vision=True,
        max_images=600,
        max_image_side_px=2000,
    ),
)
CATALOG: dict[str, ModelSpec] = {spec.model_id: spec for spec in _SPECS}


class ModelSettings(BaseModel):
    """What a router needs besides its model registry. Plain data: consumers build it from
    their own configuration (framekit reads no settings from the environment, only provider
    keys and optional base-URL overrides named in the provider configs)."""

    default_model: str = "deepseek-ai/deepseek-v4.1-flash"
    max_tokens_cap: int = 256  # caps every request's output budget
    timeout_s: float = 30.0


def estimate_cost(spec: ModelSpec, input_tokens: int, output_tokens: int) -> Decimal:
    cost = (
        Decimal(input_tokens) * spec.input_usd_per_mtok
        + Decimal(output_tokens) * spec.output_usd_per_mtok
    ) / MILLION
    return cost.quantize(Decimal("0.000001"))


def model_listing(
    specs: Iterable[ModelSpec], default_model: str | None = None
) -> list[dict[str, Any]]:
    """Models as rows for clients choosing one (no prices). Shared by `catalog_listing` and
    `ModelRegistry.listing`."""
    return [
        {
            "provider": spec.provider,
            "model_id": spec.model_id,
            "label": spec.label,
            "default": spec.model_id == default_model,
            "reasoning": spec.reasoning_mode,
        }
        for spec in specs
    ]


def catalog_listing(default_model: str | None = None) -> list[dict[str, Any]]:
    """The built-in models, for clients choosing one (no prices). The same rows as
    `DEFAULT_REGISTRY.listing(default_model)`; a registry of your own lists its models with
    its own `listing`."""
    return model_listing(CATALOG.values(), default_model)
