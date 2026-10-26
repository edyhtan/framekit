"""Test models for the extraction tests, served through `ModelSource`s by a local (loopback)
provider: no API calls. `ScriptedAdapter` answers every stage from the request itself, so
requests go through the real router, response cache and adapter template."""

import hashlib
import json
import math
import re
from decimal import Decimal
from typing import Any, ClassVar

from framekit.contracts import EmbeddingRequest, EmbeddingResult
from framekit.models import LongPromptPrices, ModelSource, ModelSpec, ProviderConfig
from framekit.models.adapters import ExternalAIAdapterBase, Reply

WORDS = ("alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel")
_SCENE = re.compile(r"Scene: from ([\d.]+) s to ([\d.]+) s\.")
_PARTS = re.compile(r"Parts to cover: (.*)\.")
_CUTS = re.compile(r"Shot cuts \(s\): ([^\n]*)\.")
_NAMES = re.compile(r"Names to use: ([^\n]*)\.")


def word_for(start: float) -> str:
    """The word a scene's facts carry: picked from its start time, so tests can query it."""
    return WORDS[int(start // 8) % len(WORDS)]


def example(schema: dict[str, Any], seed: int, path: str = "") -> Any:
    """A value matching a lens-pass JSON Schema (the scripted lens pass's answer)."""
    if "anyOf" in schema:
        return example(schema["anyOf"][0], seed, path)
    kind = schema.get("type")
    if kind == "object":
        if "properties" in schema:
            return {k: example(v, seed, f"{path}.{k}") for k, v in schema["properties"].items()}
        return {}
    if kind == "array":
        count = schema.get("minItems", 1 if "enum" in schema.get("items", {}) else 0)
        return [example(schema["items"], seed + i, f"{path}[{i}]") for i in range(count)]
    if kind == "string":
        options = schema.get("enum")
        return options[seed % len(options)] if options else f"text {path}"
    if kind == "integer":
        return int(schema.get("minimum", 0))
    if kind == "number":
        return float(schema.get("minimum", 0))
    if kind == "boolean":
        return seed % 2 == 0
    return None


def scripted_text(request: Any) -> str:
    """What the scripted model answers, by stage (the request's cache class)."""
    stage = request.cache_class
    if stage == "L2":
        parts = json.loads("[" + _PARTS.search(request.prompt).group(1) + "]")
        cuts_text = _CUTS.search(request.prompt).group(1)
        cuts = [] if cuts_text == "none detected" else [float(c) for c in cuts_text.split(", ")]
        scenes = []
        for start, end in parts:
            edges = [start, *[c for c in cuts if start < c < end], end]
            scenes += [
                {"start": a, "end": b, "summary": f"the {word_for(a)} pattern"}
                for a, b in zip(edges, edges[1:], strict=False)
            ]
        return json.dumps({"synopsis": "test patterns, one after another", "scenes": scenes})
    if stage == "L3a":
        start, end = (float(x) for x in _SCENE.search(request.prompt).groups())
        names = _NAMES.search(request.prompt).group(1).split(", ")
        word = word_for(start)
        middle = round((start + end) / 2, 3)
        record = {
            "summary": f"a {word} test pattern fills the frame",
            "setting": "a test card",
            "entities": names[:1] if names != ["none given"] else [],
            "props": ["colour bars"],
            "beats": [
                {
                    "t": [start, middle],
                    "who": names[:1],
                    "action": f"the {word} pattern holds",
                    "audio": "a beep",
                },
                {
                    "t": [middle, end],
                    "who": [],
                    "action": "the pattern keeps moving",
                    "audio": None,
                },
            ],
        }
        properties = request.response_schema["properties"]
        for module in ("camera", "timing", "sound"):
            if module in properties:
                record[module] = example(properties[module], 0)
        return json.dumps(record)
    if stage == "L3b":
        answer = example(request.response_schema, request.seed or 0)
        context = json.loads(request.prompt.split("\n\nReply with", 1)[0])
        word = context["facts"]["summary"].split()[1]
        if "phrasings" in answer:
            answer["phrasings"] = [
                f"looking for {word} moment {i}" for i in range(len(answer["phrasings"]))
            ]
        if "beats" in answer:
            beats = context["facts"]["beats"]
            item = request.response_schema["properties"]["beats"]["items"]
            answer["beats"] = [
                {**example(item, request.seed or 0), "index": beat["index"]} for beat in beats
            ]
        return json.dumps(answer)
    raise AssertionError(f"unexpected stage {stage!r}")


class ScriptedAdapter(ExternalAIAdapterBase):
    """A model that answers every stage from the request itself."""

    adapter_version: ClassVar[str] = "scripted-1"
    requests: ClassVar[list[Any]] = []
    fail_first: ClassVar[set[str]] = set()  # stages whose first reply is unusable

    @classmethod
    def make_client(cls, api_key: str, base_url: str | None, timeout_s: float) -> Any:
        return object()

    def build_params(self, spec: ModelSpec, request: Any, max_tokens: int) -> dict[str, Any]:
        return {"request": request, "max_tokens": max_tokens}

    async def call(self, spec: ModelSpec, params: dict[str, Any]) -> Reply:
        request = params["request"]
        ScriptedAdapter.requests.append(request)
        retry = "Your previous reply couldn't be used" in request.prompt
        if request.cache_class in ScriptedAdapter.fail_first and not retry:
            text = "sorry, no JSON here"
        else:
            text = scripted_text(request)
        return Reply(text=text, stop_reason="stop", input_tokens=1000, output_tokens=200)

    async def embed(self, spec: ModelSpec, request: EmbeddingRequest) -> EmbeddingResult:
        vectors = tuple(bag_of_words(text) for text in request.texts)
        return EmbeddingResult(
            provider=self.config.name,
            model_id=spec.model_id,
            vectors=vectors,
            dimension=len(vectors[0]) if vectors else 0,
            input_tokens=10 * len(request.texts),
            latency_ms=1,
            cost_usd=Decimal("0.000001"),
            adapter_version=self.adapter_version,
        )


def bag_of_words(text: str, size: int = 64) -> tuple[float, ...]:
    vector = [0.0] * size
    for word in re.findall(r"[a-z]+", text.lower()):
        slot = int(hashlib.sha256(word.encode()).hexdigest(), 16) % size
        vector[slot] += 1.0
    norm = math.sqrt(sum(x * x for x in vector)) or 1.0
    return tuple(x / norm for x in vector)


LOCAL = ProviderConfig("scripted", "openai", base_url="http://127.0.0.1:9/v1")
VISION = ModelSpec(
    "scripted",
    "seer-1",
    "Scripted vision",
    Decimal("1"),
    Decimal("5"),
    vision=True,
    max_images=50,
    max_image_side_px=1600,
)
# Takes only its default sampling (no temperature) and charges more for prompts over 1,000
# tokens, as Claude Haiku 5.5 does over 100,000.
STRICT = ModelSpec(
    "scripted",
    "strict-1",
    "Scripted vision, no temperature",
    Decimal("1"),
    Decimal("5"),
    vision=True,
    max_images=50,
    max_image_side_px=1600,
    sampling=False,
    long_prompt=LongPromptPrices(1_000, Decimal("5"), Decimal("25")),
)
TEXT = ModelSpec("scripted", "writer-1", "Scripted text", Decimal("0.3"), Decimal("1.2"))
EMBEDDER = ModelSpec("scripted", "vectors-1", "Scripted embeddings", Decimal("0.02"), Decimal("0"))


def sources(adapter: type = ScriptedAdapter) -> list[ModelSource]:
    """The test models, served by a local (loopback) provider."""
    return [
        ModelSource("seer", model=VISION, provider=LOCAL, credential="test", adapter=adapter),
        ModelSource("strict", model=STRICT, provider=LOCAL, credential="test", adapter=adapter),
        ModelSource("writer", model=TEXT, provider=LOCAL, credential="test", adapter=adapter),
        ModelSource("vectors", model=EMBEDDER, provider=LOCAL, credential="test", adapter=adapter),
    ]
