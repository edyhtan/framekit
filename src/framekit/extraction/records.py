"""Perception records (DESIGN.md §3 L3a), schema version 1: observable facts per scene.

A **core record** plus optional **modules**. The model writes the core and the `camera`,
`timing` and `sound` modules; `speech` comes from L1 (a speech detector), never from the
model. Times are seconds on the unit timeline.

    {
      "summary": "...", "setting": "...", "entities": ["..."], "props": ["..."],
      "beats": [{"t": [152.0, 156.5], "who": ["..."], "action": "...", "audio": null}],
      "camera": {"shots": [{"t": [s, e], "size": "...", "movement": "..."}]},
      "timing": {"holds": [{"t": [s, e], "description": "..."}],
                 "pauses": [{"t": [s, e], "description": "..."}], "rhythm": "..."},
      "sound": {"music": [{"t": [s, e], "description": "..."}],
                "silences": [{"t": [s, e]}], "sync": [{"t": s, "description": "..."}]},
      "speech": [{"t": [s, e], "text": "..."}]
    }
"""

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

RECORD_SCHEMA = 1
MODEL_MODULES = ("camera", "timing", "sound")  # written by the model
UNKNOWN = "unknown"

_SPAN = {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2}
_TEXT = {"type": "string"}


def _object(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def _spans(**extra: Any) -> dict[str, Any]:
    return {"type": "array", "items": _object({"t": _SPAN, **extra})}


MODULE_SCHEMAS: dict[str, dict[str, Any]] = {
    "camera": _object({"shots": _spans(size=_TEXT, movement=_TEXT)}),
    "timing": _object(
        {"holds": _spans(description=_TEXT), "pauses": _spans(description=_TEXT), "rhythm": _TEXT}
    ),
    "sound": _object(
        {
            "music": _spans(description=_TEXT),
            "silences": _spans(),
            "sync": {
                "type": "array",
                "items": _object({"t": {"type": "number"}, "description": _TEXT}),
            },
        }
    ),
}

BEAT = _object(
    {
        "t": _SPAN,
        "who": {"type": "array", "items": _TEXT},
        "action": _TEXT,
        "audio": {"type": ["string", "null"]},
    }
)


def record_schema(modules: Sequence[str]) -> dict[str, Any]:
    """The JSON Schema of an L3a reply: the core record plus the model-written modules."""
    properties: dict[str, Any] = {
        "summary": _TEXT,
        "setting": _TEXT,
        "entities": {"type": "array", "items": _TEXT},
        "props": {"type": "array", "items": _TEXT},
        "beats": {"type": "array", "items": BEAT},
    }
    for module in MODEL_MODULES:
        if module in modules:
            properties[module] = MODULE_SCHEMAS[module]
    return _object(properties)


def segmentation_schema() -> dict[str, Any]:
    """The JSON Schema of an L2 reply: a synopsis and scenes with start, end and summary."""
    scene = _object({"start": {"type": "number"}, "end": {"type": "number"}, "summary": _TEXT})
    return _object({"synopsis": _TEXT, "scenes": {"type": "array", "items": scene}})


# --- reading replies --------------------------------------------------------------------


def parse_json_reply(text: str) -> Any:
    """The JSON object in a model's reply: the whole text, or the first `{...}` in it (code
    fences and stray prose around it are ignored). Raises ValueError when there's none."""
    stripped = text.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, re.DOTALL)
    if fenced:
        stripped = fenced.group(1)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    start = stripped.find("{")
    if start >= 0:
        try:
            value, _ = json.JSONDecoder().raw_decode(stripped[start:])
            return value
        except json.JSONDecodeError:
            pass
    raise ValueError("the reply holds no JSON object")


# --- checks -----------------------------------------------------------------------------


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _is_span(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 2 and all(map(_is_number, value))


def _shape_problems(value: Any, schema: Mapping[str, Any], path: str) -> list[str]:
    """A small checker for the shapes above (objects, arrays, strings, numbers, null)."""
    kind = schema.get("type")
    kinds = kind if isinstance(kind, list) else [kind]
    if value is None:
        return [] if "null" in kinds else [f"{path}: missing (null)"]
    if "object" in kinds:
        if not isinstance(value, Mapping):
            return [f"{path}: expected an object"]
        props = schema["properties"]
        problems = [f"{path}.{k}: not expected" for k in value if k not in props]
        for key, sub in props.items():
            if key not in value:
                problems.append(f"{path}.{key}: missing")
            else:
                problems += _shape_problems(value[key], sub, f"{path}.{key}")
        return problems
    if "array" in kinds:
        if not isinstance(value, list):
            return [f"{path}: expected a list"]
        if "minItems" in schema and len(value) < schema["minItems"]:
            return [f"{path}: expected at least {schema['minItems']} items"]
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            return [f"{path}: expected at most {schema['maxItems']} items"]
        problems = []
        for i, item in enumerate(value):
            problems += _shape_problems(item, schema["items"], f"{path}[{i}]")
        return problems
    if "string" in kinds:
        return [] if isinstance(value, str) else [f"{path}: expected text"]
    if "number" in kinds:
        return [] if _is_number(value) else [f"{path}: expected a number"]
    return []


def shape_problems(value: Any, schema: Mapping[str, Any]) -> list[str]:
    return _shape_problems(value, schema, "record")


def record_times(record: Mapping[str, Any]) -> list[tuple[str, list[float]]]:
    """Every (path, [start, end]) or (path, [t]) in a record, for the timestamp checks."""
    found: list[tuple[str, list[float]]] = []
    for i, beat in enumerate(record.get("beats") or ()):
        if isinstance(beat, Mapping) and _is_span(beat.get("t")):
            found.append((f"beats[{i}].t", list(beat["t"])))
    for module in (*MODEL_MODULES, "speech"):
        block = record.get(module)
        items: list[tuple[str, Any]] = []
        if isinstance(block, Mapping):
            for key, value in block.items():
                if isinstance(value, list):
                    items += [(f"{module}.{key}[{i}]", item) for i, item in enumerate(value)]
        elif isinstance(block, list):
            items = [(f"{module}[{i}]", item) for i, item in enumerate(block)]
        for path, item in items:
            if not isinstance(item, Mapping):
                continue
            t = item.get("t")
            if _is_span(t):
                found.append((f"{path}.t", list(t)))
            elif _is_number(t):
                found.append((f"{path}.t", [t]))
    return found


def check_record(
    record: Mapping[str, Any],
    *,
    span: tuple[float, float],
    modules: Sequence[str],
    entities: Sequence[str],
    unknown: str,
    tolerance_s: float = 0.5,
) -> dict[str, list[str]]:
    """The library checks of L4 on one record, by check name: times inside the scene and
    ordered, entities from the corpus profile, every module present."""
    start, end = span
    inside, ordered = [], []
    for path, times in record_times(record):
        if any(t < start - tolerance_s or t > end + tolerance_s for t in times):
            inside.append(f"{path}: {times} is outside the scene [{start}, {end}]")
        if len(times) == 2 and times[0] > times[1]:
            ordered.append(f"{path}: starts after it ends")
    beats = [
        b for b in record.get("beats") or () if isinstance(b, Mapping) and _is_span(b.get("t"))
    ]
    starts = [b["t"][0] for b in beats]
    if starts != sorted(starts):
        ordered.append("beats: not in time order")
    names: list[tuple[str, str]] = [
        (f"entities[{i}]", name) for i, name in enumerate(record.get("entities") or ())
    ]
    for i, beat in enumerate(beats):
        names += [(f"beats[{i}].who[{j}]", n) for j, n in enumerate(beat.get("who") or ())]
    known = set(entities)
    strangers = []
    if known and unknown == "mark_unknown":
        for path, name in names:
            if isinstance(name, str) and name not in known and not name.startswith(UNKNOWN):
                strangers.append(f"{path}: {name!r} is not in the profile's entity list")
    missing = [f"{m}: missing" for m in modules if m not in ("core",) and m not in record]
    return {
        "timestamps_in_scene": inside,
        "timestamps_ordered": ordered,
        "entities_in_profile": strangers,
        "modules_complete": missing,
    }


__all__ = [
    "MODEL_MODULES",
    "MODULE_SCHEMAS",
    "RECORD_SCHEMA",
    "check_record",
    "parse_json_reply",
    "record_schema",
    "record_times",
    "segmentation_schema",
    "shape_problems",
]
