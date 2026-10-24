"""Helpers for the extraction tests: synthetic media, a scripted model, input files.

No footage: videos are ffmpeg test patterns and tones generated per test. No API calls: the
model is a scripted adapter served through a `ModelSource`, so the real router, cache and
adapter template run.
"""

import hashlib
import json
import math
import os
import re
import shutil
import subprocess  # noqa: S404  (tests drive a local ffmpeg)
from decimal import Decimal
from pathlib import Path
from typing import Any, ClassVar

import pytest
import yaml

from framekit.contracts import EmbeddingRequest, EmbeddingResult
from framekit.models import ModelSource, ModelSpec, ProviderConfig
from framekit.models.adapters import ExternalAIAdapterBase, Reply

# --- ffmpeg --------------------------------------------------------------------------


def ffmpeg_binary() -> str | None:
    """ffmpeg for the tests: $FRAMEKIT_TEST_FFMPEG (a path), else `ffmpeg` on PATH."""
    explicit = os.environ.get("FRAMEKIT_TEST_FFMPEG")
    if explicit and Path(explicit).is_file():
        return explicit
    return shutil.which("ffmpeg")


# CI sets FRAMEKIT_REQUIRE_FFMPEG=1, so a missing ffmpeg fails these tests there instead of
# skipping them silently.
needs_ffmpeg = pytest.mark.skipif(
    ffmpeg_binary() is None and os.environ.get("FRAMEKIT_REQUIRE_FFMPEG") != "1",
    reason="ffmpeg isn't installed",
)

SEGMENT_SOURCES = (
    "testsrc2=size={w}x{h}:rate={fps}",
    "smptebars=size={w}x{h}:rate={fps}",
    "color=c=0x3050a0:size={w}x{h}:rate={fps}",
    "rgbtestsrc=size={w}x{h}:rate={fps}",
)


def make_video(
    path: Path,
    *,
    cuts: tuple[float, ...] = (8.0, 16.0),
    duration: float = 24.0,
    beeps: tuple[float, ...] = (2.0, 5.0, 11.0, 19.0),
    size: tuple[int, int] = (640, 360),
    fps: int = 24,
    audio: bool = True,
) -> Path:
    """A video of visually distinct test patterns switching at `cuts`, with short 880 Hz
    beeps at `beeps` over silence."""
    binary = ffmpeg_binary()
    assert binary is not None
    edges = [0.0, *cuts, duration]
    inputs: list[str] = []
    for k, (a, b) in enumerate(zip(edges, edges[1:], strict=False)):
        source = SEGMENT_SOURCES[k % len(SEGMENT_SOURCES)].format(w=size[0], h=size[1], fps=fps)
        inputs += ["-f", "lavfi", "-i", f"{source}:duration={b - a}"]
    count = len(edges) - 1
    graph = "".join(f"[{k}:v]" for k in range(count)) + f"concat=n={count}:v=1:a=0[v]"
    args = [binary, "-hide_banner", "-nostdin", "-y", *inputs]
    maps = ["-map", "[v]"]
    if audio:
        gate = "+".join(f"between(t,{t},{t + 0.08})" for t in beeps) or "0"
        expression = f"0.8*sin(2*PI*880*t)*({gate})"
        args += ["-f", "lavfi", "-i", f"aevalsrc='{expression}':s=48000:d={duration}"]
        maps += ["-map", f"{count}:a", "-c:a", "aac"]
    args += [
        "-filter_complex",
        graph,
        *maps,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-t",
        str(duration),
        str(path),
    ]
    subprocess.run(args, check=True, capture_output=True)  # noqa: S603  (fixed arguments)
    return path


# --- a scripted model --------------------------------------------------------------------

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
    """A model that answers every stage from the request itself, and embeds text as a
    bag of hashed words (so texts sharing words are close)."""

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
TEXT = ModelSpec("scripted", "writer-1", "Scripted text", Decimal("0.3"), Decimal("1.2"))
EMBEDDER = ModelSpec("scripted", "vectors-1", "Scripted embeddings", Decimal("0.02"), Decimal("0"))


def sources() -> list[ModelSource]:
    """The scripted models, served by a local (loopback) provider."""
    return [
        ModelSource(
            "seer", model=VISION, provider=LOCAL, credential="test", adapter=ScriptedAdapter
        ),
        ModelSource(
            "writer", model=TEXT, provider=LOCAL, credential="test", adapter=ScriptedAdapter
        ),
        ModelSource(
            "vectors", model=EMBEDDER, provider=LOCAL, credential="test", adapter=ScriptedAdapter
        ),
    ]


# --- inputs --------------------------------------------------------------------------

PACE_LENS = {
    "lens": {
        "name": "pace",
        "version": "0.1.0",
        "description": "How fast a moment feels.",
        "requires_modules": ["timing"],
        "fields": {
            "tempo": {"type": "label", "vocabulary": ["slow", "steady", "fast"]},
            "cues": {"type": "labels", "vocabulary": ["pause", "rush", "hold"], "min": 1, "max": 2},
            "level": {"type": "integer", "min": 1, "max": 5},
        },
        "prompts": {
            "interpretation": {"file": "interpret.md"},
            "phrasing": "Searches a person would type.",
            "phrasing_style": "short, first person",
        },
        "validators": [{"kind": "required", "fields": ["tempo"]}],
        "consumer": {"judge_rubric": "Does the pace match?"},
    }
}

COLOUR_LENS = {
    "lens": {
        "name": "colour",
        "version": "1.0.0",
        "targets": ["scene"],
        "fields": {"palette": {"type": "label", "vocabulary": ["warm", "cool", "mixed"]}},
        "prompts": {"interpretation": "Say which palette dominates."},
    }
}


def write_yaml(path: Path, data: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def write_inputs(
    root: Path,
    video: Path,
    *,
    lenses: tuple[str, ...] = ("pace",),
    plan: dict[str, Any] | None = None,
    space: dict[str, Any] | None = None,
    rights_mode: str = "byo_media",
) -> dict[str, Path]:
    """Lens specs, a search space, a corpus and a plan as files, like a user's."""
    write_yaml(root / "lenses" / "pace" / "lens.yaml", PACE_LENS)
    (root / "lenses" / "pace" / "interpret.md").write_text(
        "Rate how fast the moment feels.", encoding="utf-8"
    )
    write_yaml(root / "lenses" / "colour.yaml", COLOUR_LENS)
    refs = {
        "pace": {"spec": "lenses/pace", "version": "0.1.0"},
        "colour": {"spec": "lenses/colour.yaml", "version": "1.0.0"},
    }
    space_data = {
        "search_space": {
            "version": 1,
            "intent": "test patterns",
            "lenses": [refs[name] for name in lenses],
            "resolution": {"moment_length_s": {"min": 3, "max": 6}},
            "query_style": {"examples": ["when everything moves at once"]},
            **(space or {}),
        }
    }
    corpus_data = {
        "corpus": {
            "profile": {
                "name": "synthetic",
                "unit_definition": "one generated test video per file",
                "metadata_fields": {"title": "str", "quality": ["good", "fair"]},
                "entities": ["Ada", "Brook"],
                "perception_hints": ["generated test patterns"],
                "rights_mode": rights_mode,
            },
            "manifest": [
                {"unit_id": "u1", "expected_duration_s": 24.0, "excluded_ranges": [[0.0, 1.0]]}
            ],
            "units": [
                {"unit_id": "u1", "path": str(video), "title": "Patterns", "quality": "good"}
            ],
        }
    }
    plan_data = {"plan": base_plan() | (plan or {})}
    return {
        "space": write_yaml(root / "search-space.yaml", space_data),
        "corpus": write_yaml(root / "corpus.yaml", corpus_data),
        "plan": write_yaml(root / "plan.yaml", plan_data),
    }


def base_plan() -> dict[str, Any]:
    """A small, fast plan on the scripted models."""
    return {
        "version": 1,
        "normalize": {"fps": 8, "height_px": 240},
        "vision_profiles": {
            "overview": {
                "version": 1,
                "frame_selection": {
                    "candidates": {"strategy": "uniform", "fps": 0.5},
                    "budget": {"max_frames_per_scene": 9},
                },
                "image": {"max_side_px": 160},
                "delivery": {"mode": "storyboard", "storyboard": {"rows": 3, "cols": 3}},
            },
            "detail": {
                "version": 1,
                "frame_selection": {
                    "candidates": {"strategy": "uniform", "fps": 2},
                    "anchors": ["shot_boundaries"],
                    "filters": [{"dedupe": {"max_distance": 4}}, {"sharpness": {"window_s": 1.0}}],
                    "budget": {"max_frames_per_scene": 4, "allocate_by": ["motion"]},
                },
                "image": {"max_side_px": 200, "format": "jpeg", "quality": 80},
            },
        },
        "stages": {"L2": {"vision": "overview"}, "L3a": {"vision": "detail"}},
        "segmentation": {"scene_length_s": {"min": 4, "target": 8, "max": 12}},
        "lens_passes": {"phrasings": {"count": 3}},
        "models": {
            "L2": {"model_id": "seer", "max_tokens": 1024},
            "L3a": {"model_id": "seer", "max_tokens": 1024},
            "L3b": {"model_id": "writer", "max_tokens": 512},
            "embed_text": {"model_id": "vectors"},
        },
        "governance": {"model_inputs": "local_only"},
    }
