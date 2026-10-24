"""Shared test helpers."""

from decimal import Decimal

from framekit.contracts import ConversationResult

HIVE_MODEL = "deepseek-ai/deepseek-v4.1-flash"


def final_result(**overrides):
    fields = {
        "provider": "hive",
        "model_id": HIVE_MODEL,
        "text": "Hello!",
        "stop_reason": "end",
        "input_tokens": 12,
        "output_tokens": 5,
        "latency_ms": 900,
        "cost_usd": Decimal("0.00001"),
        "request_id": "req_1",
        "adapter_version": "0.4.0",
        "reasoning_tokens": 2,
        "provider_stop_reason": "stop",
        "first_token_ms": 300,
    }
    fields.update(overrides)
    return ConversationResult(**fields)
