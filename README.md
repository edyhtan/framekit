# framekit

[![tests](https://github.com/edyhtan/framekit/actions/workflows/tests.yml/badge.svg)](https://github.com/edyhtan/framekit/actions/workflows/tests.yml) [![lint](https://github.com/edyhtan/framekit/actions/workflows/lint.yml/badge.svg)](https://github.com/edyhtan/framekit/actions/workflows/lint.yml) [![security](https://github.com/edyhtan/framekit/actions/workflows/security.yml/badge.svg)](https://github.com/edyhtan/framekit/actions/workflows/security.yml)

AI building blocks for video datasets, in Python. One package with modules you can use on their own:

| Module | What it gives you | Status |
|---|---|---|
| `framekit.contracts` | provider-neutral types: the conversation port, the response-store port | **v0.1.0** |
| `framekit.models` | model access: a catalog of providers and models (extendable with your own), a router, adapters on the official `openai` and `anthropic` SDKs, a record / replay response cache | **v0.1.0** |
| `framekit.vision` | how a model sees video: frame selection, image preparation and delivery | **partial** on `main`: schema, checks, pixel-free planning ([settings](docs/settings.md) §5.2) |
| `framekit.extraction` | turns a video corpus into a dataset of segments, text, vectors and provenance, plus a benchmark harness and a CLI | designed ([design](docs/DESIGN.md)) |

Used by the video clip search agent ([short_extract](https://github.com/edyhtan/short_extract)), and built for benchmarks and experimental model training.

**Contents:** [Install](#install) · [Configure](#configure) · [Quick start](#quick-start) · [Usage guide](#usage-guide) · [Models](#models) · [Your own models and providers](#your-own-models-and-providers) · [Response cache](#response-cache) · [Vision parameters](#vision-parameters) · [Extraction (preview)](#extraction-preview) · [Development](#development) · [Documents](#documents) · [Licence](#licence)

---

## Install

framekit is installed from this repository, pinned to a release tag. It needs Python 3.12+.

**uv** (`pyproject.toml`):

```toml
[project]
dependencies = ["framekit"]

[tool.uv.sources]
framekit = { git = "https://github.com/edyhtan/framekit", tag = "v0.1.0" }
```

**pip:**

```bash
pip install "framekit @ git+https://github.com/edyhtan/framekit@v0.1.0"
```

**Images without git** (e.g. slim Docker images) can use the tag's source archive instead. It's still pinned, and hash-locked by uv:

```toml
[tool.uv.sources]
framekit = { url = "https://github.com/edyhtan/framekit/archive/refs/tags/v0.1.0.tar.gz" }
```

Dependencies are kept to the providers' official SDKs plus pydantic, with loose version ranges, so framekit fits next to your own stack. It never depends on LangChain, LiteLLM or any other model-access library (supply-chain policy, [DECISIONS](docs/DECISIONS.md)).

## Configure

framekit reads **no settings from the environment** except provider keys and an optional endpoint override. Everything else is plain data that you pass in.

| Variable | Used for |
|---|---|
| `HIVE_API_KEY` | models served by Hive (OpenAI-compatible, streaming) |
| `DEEPSEEK_API_KEY` | DeepSeek's own API |
| `ANTHROPIC_API_KEY` | Claude models |
| `HIVE_BASE_URL` | optional Hive endpoint override (e.g. a regional one) |

A missing key fails only the calls to that provider, with a non-retryable error. Keep keys in your shell or a git-ignored `.env`; never commit them.

**Or pass a key explicitly**, which wins over the environment: a `ModelSource` with `credential=` (a string, a `SecretStr`, or a callable for a secret manager), [below](#your-own-models-and-providers). Applications usually load their `.env` themselves (e.g. `python-dotenv` or `pydantic-settings`): framekit never reads files on its own.

## Quick start

```python
import asyncio

from framekit.contracts import ConversationRequest
from framekit.models import ConversationRouter, ModelSettings


async def main() -> None:
    settings = ModelSettings(default_model="claude-haiku-4-5", max_tokens_cap=1024)
    async with ConversationRouter(settings) as router:          # closes connections on exit
        result = await router.generate(
            ConversationRequest("Describe a slapstick chase in one sentence.", max_tokens=200)
        )
    print(result.text)
    print(result.model_id, result.input_tokens, result.output_tokens, result.stop_reason)


asyncio.run(main())
```

One router per unit of work (a task, a chat turn): it opens connections lazily, one per provider, and closes them all when the `async with` block ends.

## Usage guide

### Requests

`ConversationRequest` carries everything about one call. Only `prompt` and `max_tokens` are required.

| Field | Meaning |
|---|---|
| `prompt` | the new user message |
| `max_tokens` | the answer budget (capped by `ModelSettings.max_tokens_cap`) |
| `model_id` | a catalog model; `None` uses `ModelSettings.default_model` |
| `history` | earlier turns, oldest first: a tuple of `ChatMessage("user" or "assistant", text)` |
| `system` | instructions sent ahead of the conversation |
| `reasoning` | `False` by default: reasoning is billed as output tokens, so it's on only when you ask |
| `response_schema` | a JSON Schema (plain dict) for structured output |
| `cache_class` | a label for the response cache's per-class policy (e.g. `"lens_pass"`) |

> **Output cap:** `ModelSettings.max_tokens_cap` (default **256**) caps every request's `max_tokens`. Raise it for longer answers. When reasoning is on, the model's reasoning allowance is added on top, within that model's own cap.

### Conversations and system prompts

```python
from framekit.contracts import ChatMessage, ConversationRequest

request = ConversationRequest(
    "And who wins in the end?",
    max_tokens=300,
    system="You describe cartoon scenes factually, in plain English.",
    history=(
        ChatMessage("user", "What happens in the kitchen scene?"),
        ChatMessage("assistant", "The cat chases the mouse around the table."),
    ),
)
```

Every call is stateless: send the history you want the model to see.

### Streaming, with optional reasoning

`stream` yields `TextDelta` pieces as they arrive, then exactly one final `ConversationResult`.

```python
from framekit.contracts import ConversationRequest, ConversationResult, TextDelta

request = ConversationRequest("Plan a three-step prank.", max_tokens=300, reasoning=True)
async with ConversationRouter(settings) as router:
    async for piece in router.stream(request):
        if isinstance(piece, TextDelta):
            print(piece.text, end="")         # piece.kind is "reasoning" or "text"
        else:
            final: ConversationResult = piece
print("\nreasoning was on:", final.reasoning_on, "| first token after", final.first_token_ms, "ms")
```

Each model switches reasoning in its own way (an effort field on Hive, `thinking` on Claude), and framekit applies the right fields from the catalog. The model's reasoning text is kept separate from the answer (`result.reasoning_text`).

### Structured output

Pass a JSON Schema; the reply text is then JSON meant to match it. **Always validate it yourself**, because providers differ in how strictly they enforce schemas.

```python
import json

schema = {
    "type": "object",
    "properties": {"mood": {"type": "string"}, "intensity": {"type": "integer"}},
    "required": ["mood", "intensity"],
}
result = await router.generate(
    ConversationRequest("Rate the mood of: 'the trap snaps on his hand'.",
                        max_tokens=100, response_schema=schema)
)
data = json.loads(result.text)          # then validate, e.g. with a Pydantic model
print(result.structured_mode)           # how the schema was applied (see below)
```

framekit picks the method each model supports (`ModelSpec.structured_method`), and reports it in `result.structured_mode`:

| Method | How | Used by |
|---|---|---|
| `function_calling` | the schema becomes the one tool the model must call; its arguments are the reply (sent non-streamed) | Hive models (a native JSON-Schema response format is rejected there) |
| `json_schema` | the provider enforces the schema | providers that support it |
| `json_mode` | valid JSON guaranteed; shape described in the instructions | — |
| `prompted` | the schema goes into the instructions only | the default, e.g. Claude until probed |

A reply cut off at the token limit (`stop_reason == "length"`) is never complete JSON: retry with a larger budget.

### Results

| Field | Meaning |
|---|---|
| `text`, `reasoning_text` | the answer, and the reasoning when the model shares it |
| `stop_reason` | normalized: `end`, `length`, `refusal`, `tool_use`, `content_filter`, `other` (`provider_stop_reason` keeps the raw value; a cut-off reply is reported as `length` even when the provider says otherwise) |
| `input_tokens`, `output_tokens`, `reasoning_tokens` | usage |
| `latency_ms`, `first_token_ms` | total time, and time to the first streamed piece |
| `cost_usd` | an **estimate** from the catalog's price table |
| `model_id`, `provider`, `adapter_version`, `request_id` | provenance: record these with every output |
| `reasoning_on`, `structured_mode`, `cached` | what actually happened |

### Errors

Every provider failure becomes one exception type, `ModelCallError`, with a `retryable` flag:

```python
from framekit.contracts import ModelCallError

try:
    result = await router.generate(request)
except ModelCallError as error:
    if error.retryable:      # rate limits, timeouts, 5xx: try again later
        ...
    else:                    # bad request, missing key, no balance: fix, don't retry
        raise
```

## Models

The catalog (`framekit.models.CATALOG`) lists the selectable models as data: provider, prices, reasoning switches, structured-output method, context window.

| `model_id` | Provider | Reasoning | Structured output |
|---|---|---|---|
| `deepseek-ai/deepseek-v4.1-flash` (default) | Hive | optional | `function_calling` |
| `zai-org/glm-5.3-flash` | Hive | optional | `function_calling` |
| `deepseek-flash` | DeepSeek API | optional | `prompted` |
| `claude-haiku-4-5` | Anthropic | optional | `prompted` |
| `claude-sonnet-5` | Anthropic | optional | `prompted` |
| `claude-opus-5` | Anthropic | optional | `prompted` |

```python
from framekit.models import CATALOG, catalog_listing

catalog_listing(default_model="claude-haiku-4-5")   # for a model picker: no prices
CATALOG["claude-sonnet-5"].context_window             # 1_000_000
```

These are the built-ins. To add your own, see the next section.

## Your own models and providers

Add models and providers in your own code, without changing framekit. One structure, a **`ModelSource`**, says which model, where it's served, with which key, and through which adapter; pass sources to the router and refer to them by name. (On `main`, for the release after v0.1.0.)

```python
import os
from decimal import Decimal

from pydantic import SecretStr

from framekit.models import ConversationRouter, ModelSettings, ModelSource, ModelSpec, ProviderConfig

sources = [
    ModelSource("claude", model="claude-haiku-4-5",                        # a built-in model,
                credential=SecretStr(os.environ["MY_CLAUDE_KEY"])),       # with your own key
    ModelSource("local", model=ModelSpec("box", "my-local-model", "My local model",
                                         Decimal("0"), Decimal("0")),
                provider=ProviderConfig("box", "openai",                  # keyless local server
                                        base_url="http://127.0.0.1:8000/v1")),
]
async with ConversationRouter(ModelSettings(default_model="local"), sources=sources) as router:
    ...  # your sources and the built-in catalog, side by side
```

- **Credentials are never recorded:** kept as a `SecretStr`, passed only to the client; results record the source's name (`result.source`), and the response cache keys a source by a public fingerprint of where it's served, so two hosts never share cached answers.

- **Any OpenAI-compatible server** (a hosted provider, a self-hosted vLLM server, Ollama's `/v1` endpoint) is config only, as above. Probe it before relying on it: reasoning switches, structured output and streaming differ from server to server.
- **Another protocol or client** needs an adapter: subclass `ExternalAIAdapterBase` from `framekit.models.adapters`, set `adapter_version`, implement `build_params` and `call`, and map your client's errors to `ModelCallError` (override `map_error`, or raise `ModelCallError` in `call`). Pass the class as the source's `adapter=`. The base does the rest: output budget, reasoning switches, timing, stop reasons, cost, provenance.
- **Lower level:** a `ModelRegistry` (`DEFAULT_REGISTRY.extend(...)`, `ConversationRouter(..., registry=...)`) holds providers, models and adapter classes; sources are checked against it.
- **Only point a provider at a host you trust**, over `https` unless it's on the same machine. framekit sends the provider's key to its `base_url`, and the SDKs also send some of their own environment variables there (known gaps, described in the guide).

Guide, with a complete runnable adapter: **[docs/adapters.md](docs/adapters.md)**.

## Response cache

Record a model call once and replay it later, instantly and for free. This is useful for long tests, benchmarks that rerun only later stages, and reproducible datasets.

```python
from framekit.contracts import ConversationRequest
from framekit.models import ConversationRouter, ModelSettings
from framekit.models.cache import CacheSettings, ClassPolicy, FileResponseStore, with_cache

store = FileResponseStore(".cache/models")
request = ConversationRequest("Describe the scene.", max_tokens=200, cache_class="lens_pass")

# First run: record. Calls the provider on a miss and stores the reply.
recording = with_cache(
    lambda: ConversationRouter(ModelSettings(max_tokens_cap=1024)),   # a factory: one router per use
    settings=CacheSettings(
        mode="record",                                               # off | record | replay
        classes={"chat": ClassPolicy(mode="off")},                   # per cache class
    ),
    store=store,
)
port = recording()
first = await port.generate(request)
await port.aclose()

# Later runs: replay only. The same request is answered from the store, with no call.
replaying = with_cache(
    lambda: ConversationRouter(ModelSettings(max_tokens_cap=1024)),
    settings=CacheSettings(mode="replay"),
    store=store,
)
port = replaying()
again = await port.generate(request)
await port.aclose()
assert again.cached and again.text == first.text and again.cost_usd == 0
```

| Mode | Behaviour |
|---|---|
| `off` | no caching (the default; `with_cache` then returns your factory unchanged) |
| `record` | replay hits; on a miss, call the provider and store the reply |
| `replay` | hits only; a miss raises `ModelCallError` (for tests that must never spend money) |

- **What counts as the same request:** the cache key covers everything in the request that shapes a reply: the resolved model, system prompt, history, message, budget, reasoning switch, schema, and whether it was streamed. Change any of them and it's a miss. It doesn't cover the provider or its host, so registries that map one `model_id` to different hosts need separate namespaces ([docs/adapters.md](docs/adapters.md#provenance-and-the-response-cache)).
- **Marked replays:** replays come back with `cached=True`, `cost_usd=0`, and `adapter_version` ending in `+cache`, so stored records never mistake a replay for a real call.
- **Streams:** they replay their pieces in order; use `CacheSettings(replay_pace="original")` to keep the recorded timing.
- **Settings:** `ttl_s` (default 7 days), `namespace` (change it to invalidate everything at once), `max_entry_bytes` (larger replies aren't stored).
- **The file store** keeps entries in a folder per namespace and cache class, so `store.stats()` counts them per class and `store.clear(cache_class="...")` wipes one class.
- **Your own store:** implement the `ResponseStore` protocol from `framekit.contracts`: `async get(key) -> bytes | None` and `async put(key, value, ttl_s) -> bool`. A store must never raise: a failed read is a miss, and a failed write is a skipped write. The search agent, for example, stores entries in Valkey.

## Vision parameters

How a model sees video (which frames, at what size, delivered how) is structured, versioned data: named **vision profiles**, assigned per stage. `framekit.vision` checks them, records them in a canonical form with a digest, and plans frame timestamps from cheap signals (shot cuts, audio onsets, motion) without decoding any video. Not released yet: it's on `main`, after v0.1.0.

```python
from framekit.vision import Signals, VisionParams, plan_frames, resolve

params = VisionParams.from_mapping({                 # parse your YAML or JSON first
    "vision_profiles": {
        "detail": {
            "version": 1,
            "frame_selection": {
                "candidates": {"strategy": "uniform", "fps": 4},
                "anchors": ["shot_boundaries"],
                "budget": {"max_frames_per_scene": 16, "allocate_by": ["motion"]},
            },
            "image": {"max_side_px": 768},
        },
    },
    "stages": {"L3a": {"vision": "detail"}},
})                                                    # VisionParamsError lists every problem
resolved = resolve(params)                            # versions pinned, defaults filled
print(resolved.digest)                                # same settings, same digest
plan = plan_frames(resolved.for_stage("L3a"), (12.0, 48.5),
                   Signals(shot_boundaries=[12.0, 20.4], motion=[(30.0, 4.2)]))
print(plan.timestamps)                                # seconds on the video's own timeline
```

- **Checks before any spend, all reported at once:** unknown strategies or keys, a strategy in the wrong slot, missing settings, `true` or `"4"` where a number belongs, `native_video` with anything but uniform sampling, views that overflow the frame budget. `check_against_source` and `check_against_model` add the analysis copy's and the model's limits.
- **Read-only once checked:** parsed and resolved parameters can't be changed in place, so the digest always describes what runs.
- **Strategies are versioned code:** `uniform`, `shot_based`, `shot_boundaries`, `audio_onsets`, `dedupe`, `sharpness`, `motion`; a changed algorithm gets a new version (`dedupe@2`), so old datasets stay reproducible. Filters that need pixels (`dedupe`, `sharpness`) are recorded now and run with the media stage (E1).
- Every setting, check and default: [docs/settings.md](docs/settings.md) §5.2.

## Extraction (preview)

Not built yet; this is where framekit is going (milestone E1). An extraction run answers six questions:
1. What should people be able to find?
2. Which videos?
3. How is the video looked at?
4. What is one data point?
5. What is written about each data point, and by which models?
6. What rules apply?

The result is a dataset with a manifest that records every answer.

```bash
framekit extract --space search-space.yaml --corpus corpus.yaml --plan plan.yaml \
                 --out ~/framekit-data/datasets/my-dataset --dry-run
framekit bench   --corpus corpus.yaml --plan plan.yaml \
                 --space raw=space-raw.yaml --space llm=space-llm.yaml --gold gold/queries.yaml
```

- **[docs/settings.md](docs/settings.md):** every setting, with complete example files and their status.
- **[docs/DESIGN.md](docs/DESIGN.md):** how extraction works, stage by stage.

## Development

```bash
uv sync                   # Python 3.12, dev tools included
uv run pytest             # unit tests
uv run lint-imports       # layer rules: contracts stay pure, SDKs only in adapters, no model libraries
uv run ruff check .       # lint
uv build                  # wheel + sdist
```

- **CI:** `tests` runs the suite and the build against both the locked and the newest dependency versions; `lint` runs ruff (including the bandit `S` rules) and the layer rules; `security` audits `uv.lock` for known vulnerabilities, checks the workflows with zizmor and runs CodeQL (also daily).
- **Supply chain:** dependencies resolve with a 7-day release cooldown (`exclude-newer`), every action is pinned to a commit SHA, and Dependabot proposes updates weekly (ARCHITECTURE.md §2.2).
- **Contributing:** changes reach `main` only through pull requests with passing checks. The branch is protected: no force pushes, no merges without CI.
- **Security:** report vulnerabilities privately, as described in [SECURITY.md](SECURITY.md).
- **Before architectural work,** read [CLAUDE.md](CLAUDE.md) for the rules: no lens or corpus vocabulary in library code, model-agnostic ports, a decision journal entry for every design decision.

## Documents

| Document | What it covers |
|---|---|
| [docs/DESIGN.md](docs/DESIGN.md) | what extraction does: inputs, the dataset, stages L0–L4, lens specs, runs, evaluation, roadmap, rights |
| [docs/settings.md](docs/settings.md) | every extraction setting in one place: the six questions a run answers, commands, the search space, corpus, plan and gold-set files, the output, open decisions |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | how the code is layered, the public API, model access, CI and security checks, how consumers connect, the seeding plan |
| [docs/adapters.md](docs/adapters.md) | your own models, providers and adapters: the model registry, the adapter template with a runnable example, errors, keyless servers, streaming, provenance, security |
| [docs/DECISIONS.md](docs/DECISIONS.md) | why, in date order |
| [CLAUDE.md](CLAUDE.md) | rules for contributors and coding agents |
| [SECURITY.md](SECURITY.md) | how to report a vulnerability privately, supported versions, scope |

## Licence

Code: [MIT](LICENSE). Annotation packs, if published, are CC-BY and contain no footage (docs/DESIGN.md §8).

## Rights

Not affiliated with any rights holder. For corpora in `byo_media` mode, users supply their own copies; this repository contains no footage, frames, thumbnails, ripping tools or downloaders, and datasets built from such corpora stay local.
