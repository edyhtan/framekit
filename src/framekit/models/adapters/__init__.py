"""One adapter per wire protocol, on the providers' official SDKs (the only place they're
imported):

- `base`: the shared call lifecycle (`ExternalAIAdapterBase`): key check, output budget,
  timing, error mapping, stop-reason normalization, the result
- `openai_compatible`: OpenAI Chat Completions (Hive, DeepSeek, any compatible provider)
- `anthropic`: Anthropic Messages

**Writing your own adapter** (docs/adapters.md): subclass `ExternalAIAdapterBase`, set
`adapter_version`, implement `build_params` and `call`, and register it with a
`framekit.models.ModelRegistry`. The names in `__all__` below are the template's public
API, under semantic versioning; everything else in these modules is internal.
"""

from framekit.models.adapters.base import (
    PLACEHOLDER_KEY,
    ExternalAIAdapterBase,
    Reply,
    conversation,
    image_data_url,
    map_sdk_error,
    merge_fields,
    normalize_stop_reason,
    output_budget,
    reasoning_active,
    reasoning_fields,
    sampling_fields,
    structured_mode,
    system_text,
)

__all__ = [
    "PLACEHOLDER_KEY",
    "ExternalAIAdapterBase",
    "Reply",
    "conversation",
    "image_data_url",
    "sampling_fields",
    "map_sdk_error",
    "merge_fields",
    "normalize_stop_reason",
    "output_budget",
    "reasoning_active",
    "reasoning_fields",
    "structured_mode",
    "system_text",
]
