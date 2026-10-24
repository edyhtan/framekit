"""One adapter per wire protocol, on the providers' official SDKs (the only place they're
imported):

- `base`: the shared call lifecycle (`ExternalAIAdapterBase`): key check, output budget,
  timing, error mapping, stop-reason normalization, the result
- `openai_compatible`: OpenAI Chat Completions (Hive, DeepSeek, any compatible provider)
- `anthropic`: Anthropic Messages
"""
