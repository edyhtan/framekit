"""Lens specs (DESIGN.md §4): what a lens pass writes, as declarative data.

A lens is a spec file plus prompt files, versioned. framekit validates the spec at load
time and never interprets its labels: it compiles the fields into a JSON Schema for the
model, checks every reply against them, and stores the result in `annotations` and
`phrasings` without understanding it.

File shape (YAML or JSON; a file, or a folder holding `lens.yaml`, `lens.yml` or
`lens.json`):

    lens:
      name: pace                       # lower snake case
      version: 0.1.0
      description: How fast a moment feels.
      example_queries: ["a slow, quiet morning"]
      requires_modules: [timing]       # perception modules beyond core
      targets: [scene, beat]           # what gets annotations (default: both)
      fields:                          # the output schema; vocabularies are fixed
        tempo: {type: label, vocabulary: [slow, steady, fast]}
        cues:  {type: labels, vocabulary: [pause, rush, hold], max: 2}
        level: {type: integer, min: 1, max: 5}
      prompts:
        interpretation: {file: prompts/interpret.md}   # or inline text
        phrasing: "Write searches a person would type to find this moment."
        phrasing_style: "short, first person"          # used when the space has no examples
      validators:
        - {kind: required, fields: [tempo]}
        - {kind: exclusive, field: cues, labels: [pause, rush]}
      consumer: {...}                  # carried verbatim for consumers (search agents)

Field types: `label` (one value of `vocabulary`), `labels` (several, unique; `min`/`max`
count them), `integer` and `number` (`min`/`max` bound the value), `boolean`, `object`
(`fields`), `list` (`items`; `min`/`max` count them) and `map` (`keys: entities`, the corpus
profile's entity names plus `extra_keys`, or `keys: vocabulary`; `values` for each). Any
field may be `optional` (null or left out). There is no free-text type: labels come from a
vocabulary, never from the model.
"""

from collections.abc import Mapping, Sequence
from math import isfinite
from pathlib import Path
from typing import Annotated, Any, Literal, Optional

from pydantic import AfterValidator, Field, StringConstraints

from framekit.extraction.schemas.common import (
    ConfigError,
    Frozen,
    FrozenDict,
    Model,
    Module,
    Name,
    Tupled,
    Version,
    digest,
    parse,
    plain,
    read_mapping,
    unwrap,
)

WHAT = "the lens spec"
SPEC_FILES = ("lens.yaml", "lens.yml", "lens.json")
Label = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9_]*$", max_length=64)]
FieldType = Literal["label", "labels", "integer", "number", "boolean", "object", "list", "map"]
Target = Literal["scene", "beat"]

# Which settings each field type takes (everything else must stay unset).
TAKES: dict[str, set[str]] = {
    "label": {"vocabulary"},
    "labels": {"vocabulary", "min", "max"},
    "integer": {"min", "max"},
    "number": {"min", "max"},
    "boolean": set(),
    "object": {"fields"},
    "list": {"items", "min", "max"},
    "map": {"keys", "vocabulary", "extra_keys", "values"},
}
SETTINGS = ("vocabulary", "min", "max", "fields", "items", "keys", "extra_keys", "values")


class FieldSpec(Model):
    type: FieldType
    description: str = ""
    optional: bool = False  # may be null or left out
    vocabulary: Annotated[tuple[Label, ...], Tupled] = ()
    min: float | None = None
    max: float | None = None
    fields: Annotated[dict[Name, "FieldSpec"], AfterValidator(FrozenDict)] | None = None
    items: Optional["FieldSpec"] = None  # noqa: UP045  (a forward reference)
    keys: Literal["entities", "vocabulary"] | None = None
    extra_keys: Annotated[tuple[Label, ...], Tupled] = ()
    values: Optional["FieldSpec"] = None  # noqa: UP045


FieldSpec.model_rebuild()


class PromptFile(Model):
    file: str


class LensPrompts(Model):
    interpretation: str = Field(min_length=1)
    phrasing: str = ""
    phrasing_style: str = ""  # the fallback style when the search space has no examples


class LensValidator(Model):
    """A rule beyond the field schema, checked in L4. `required`: these fields are never
    empty. `exclusive`: in one `labels` field, at most one of these labels."""

    kind: Literal["required", "exclusive"]
    fields: Annotated[tuple[Name, ...], Tupled] = ()
    field: Name | None = None
    labels: Annotated[tuple[Label, ...], Tupled] = ()


class LensSpec(Model):
    name: Name
    version: Version
    description: str = ""
    example_queries: Annotated[tuple[str, ...], Tupled] = ()
    requires_modules: Annotated[tuple[Module, ...], Tupled] = ()
    targets: Annotated[tuple[Target, ...], Tupled] = ("scene", "beat")
    fields: Annotated[dict[Name, FieldSpec], AfterValidator(FrozenDict)]
    prompts: LensPrompts
    validators: Annotated[tuple[LensValidator, ...], Tupled] = ()
    gold: str | None = None  # an annotation gold set, carried for benchmarks
    # Carried verbatim for consumers (query mapping, specificity rules, retrieval weights,
    # judge rubric, critique mapping, abstention thresholds); extraction never reads it.
    consumer: Annotated[Any, Frozen] = FrozenDict()

    @property
    def digest(self) -> str:
        """sha256 of the spec's canonical form, prompts included: any change to what a lens
        pass sends or checks changes it, and with it the input hash of its lens passes."""
        return digest(self.model_dump(mode="json"))

    @property
    def key(self) -> str:
        return f"{self.name}@{self.version}"

    # --- loading ------------------------------------------------------------------------

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, Any], base_dir: str | Path | None = None
    ) -> "LensSpec":
        """A spec from plain data. Prompts given as `{file: path}` are read from `base_dir`
        (default: the working directory) and must stay inside it."""
        data = dict(unwrap(data, "lens"))
        problems: list[str] = []
        prompts = data.get("prompts")
        if isinstance(prompts, Mapping):
            data["prompts"] = {
                key: _read_prompt(value, f"prompts.{key}", base_dir, problems)
                for key, value in prompts.items()
            }
        if problems:
            raise ConfigError(problems, WHAT)
        return parse(cls, data, WHAT, extra=_spec_problems)

    @classmethod
    def load(cls, path: str | Path) -> "LensSpec":
        """A spec from a file, or a folder holding `lens.yaml`, `lens.yml` or `lens.json`."""
        location = Path(path).expanduser()
        if location.is_dir():
            found = [location / name for name in SPEC_FILES if (location / name).is_file()]
            if not found:
                raise ConfigError([f"{location}: no {', '.join(SPEC_FILES)} in this folder"], WHAT)
            location = found[0]
        try:
            return cls.from_mapping(read_mapping(location, WHAT), base_dir=location.parent)
        except ConfigError as exc:
            raise ConfigError([f"{location}: {p}" for p in exc.problems], WHAT) from None

    # --- what a lens pass sends and checks ----------------------------------------------

    def targets_in(self, targets: Sequence[str]) -> tuple[str, ...]:
        """The plan's targets this lens annotates (scene first)."""
        return tuple(t for t in ("scene", "beat") if t in targets and t in self.targets)

    def output_schema(
        self, *, targets: Sequence[str], phrasings: int, entities: Sequence[str]
    ) -> dict[str, Any]:
        """The JSON Schema of one lens-pass reply for one scene: `scene` (the fields),
        `beats` (the fields per beat, by index) and `phrasings`."""
        targets = self.targets_in(targets)
        properties: dict[str, Any] = {}
        if "scene" in targets:
            properties["scene"] = _object_schema(self.fields, entities)
        if "beat" in targets:
            beat = _object_schema(self.fields, entities)
            beat["properties"] = {"index": {"type": "integer", "minimum": 0}, **beat["properties"]}
            beat["required"] = ["index", *beat["required"]]
            properties["beats"] = {"type": "array", "items": beat}
        if phrasings:
            properties["phrasings"] = {
                "type": "array",
                "items": {"type": "string"},
                "minItems": phrasings,
                "maxItems": phrasings,
            }
        return {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        }

    def check_output(
        self,
        output: Any,
        *,
        targets: Sequence[str],
        phrasings: int,
        entities: Sequence[str],
        beats: int,
    ) -> list[str]:
        """Problems with one lens-pass reply (empty when it can be stored): the shape, every
        label in its vocabulary, every value in range, the beat indexes, the phrasings."""
        targets = self.targets_in(targets)
        if not isinstance(output, Mapping):
            return ["(root): expected an object"]
        expected = {"scene"} & set(targets) | ({"beats"} if "beat" in targets else set())
        expected |= {"phrasings"} if phrasings else set()
        problems = [f"{key}: not expected" for key in output if key not in expected]
        problems += [f"{key}: missing" for key in sorted(expected) if key not in output]
        if "scene" in targets and "scene" in output:
            problems += check_fields(self.fields, output["scene"], "scene", entities)
        if "beat" in targets and isinstance(output.get("beats"), list):
            seen: set[int] = set()
            for i, item in enumerate(output["beats"]):
                path = f"beats[{i}]"
                if not isinstance(item, Mapping):
                    problems.append(f"{path}: expected an object")
                    continue
                index = item.get("index")
                if not _whole(index) or not 0 <= index < beats:
                    problems.append(f"{path}.index: expected a beat index from 0 to {beats - 1}")
                elif index in seen:
                    problems.append(f"{path}.index: beat {index} annotated twice")
                else:
                    seen.add(index)
                rest = {k: v for k, v in item.items() if k != "index"}
                problems += check_fields(self.fields, rest, path, entities)
        elif "beat" in targets and "beats" in output:
            problems.append("beats: expected a list")
        if phrasings and "phrasings" in output:
            found = output["phrasings"]
            if not isinstance(found, list) or not all(
                isinstance(p, str) and p.strip() for p in found
            ):
                problems.append("phrasings: expected a list of non-empty strings")
            elif len(found) != phrasings:
                problems.append(f"phrasings: expected {phrasings}, got {len(found)}")
        return problems

    def check_rules(self, annotation: Mapping[str, Any], path: str) -> list[str]:
        """The spec's validators on one target's annotation (L4 lens checks)."""
        problems = []
        for rule in self.validators:
            if rule.kind == "required":
                for name in rule.fields:
                    if _empty(annotation.get(name)):
                        problems.append(f"{path}.{name}: required by the lens, but empty")
            elif rule.field is not None:
                found = [
                    label for label in annotation.get(rule.field) or () if label in rule.labels
                ]
                if len(found) > 1:
                    problems.append(
                        f"{path}.{rule.field}: {', '.join(found)} can't appear together"
                    )
        return problems


# --- loading helpers --------------------------------------------------------------------


def _read_prompt(value: Any, path: str, base_dir: str | Path | None, problems: list[str]) -> Any:
    if not isinstance(value, Mapping):
        return value
    if set(value) != {"file"} or not isinstance(value["file"], str):
        problems.append(f"{path}: a prompt is text, or {{file: path}}")
        return value
    base = Path(base_dir or ".").resolve()
    file = (base / value["file"]).resolve()
    # A spec may come from someone else; its prompt files must stay inside its folder, so a
    # spec can't make framekit read (and send to a model) any other file.
    if not file.is_relative_to(base):
        problems.append(f"{path}: {value['file']!r} is outside the spec's folder")
        return value
    try:
        return file.read_text(encoding="utf-8")
    except OSError as exc:
        problems.append(f"{path}: {value['file']!r} can't be read ({exc.strerror or exc})")
        return value


def _spec_problems(spec: LensSpec) -> list[str]:
    problems: list[str] = []
    if not spec.fields:
        problems.append("fields: at least one field is needed")
    for name, field in spec.fields.items():
        problems += _field_problems(field, f"fields.{name}")
    if not spec.targets:
        problems.append("targets: at least one target is needed")
    if len(set(spec.targets)) != len(spec.targets):
        problems.append("targets: listed twice")
    for i, rule in enumerate(spec.validators):
        path = f"validators[{i}]"
        if rule.kind == "required":
            if not rule.fields or rule.field is not None or rule.labels:
                problems.append(f"{path}: `required` takes `fields` only")
            problems += [
                f"{path}.fields: {name!r} is not a field of the lens"
                for name in rule.fields
                if name not in spec.fields
            ]
        else:
            target = spec.fields.get(rule.field or "")
            if rule.fields or target is None or target.type != "labels":
                problems.append(f"{path}: `exclusive` takes `field` (a labels field) and `labels`")
            elif len(rule.labels) < 2:
                problems.append(f"{path}.labels: at least two labels are needed")
            else:
                problems += [
                    f"{path}.labels: {label!r} is not in the vocabulary of {rule.field!r}"
                    for label in rule.labels
                    if label not in target.vocabulary
                ]
    return problems


def _field_problems(field: FieldSpec, path: str) -> list[str]:
    problems = [
        f"{path}.{setting}: not used by type {field.type}"
        for setting in SETTINGS
        if setting not in TAKES[field.type] and _is_set(getattr(field, setting))
    ]
    if field.type in ("label", "labels") or (field.type == "map" and field.keys == "vocabulary"):
        if not field.vocabulary:
            problems.append(f"{path}.vocabulary: a vocabulary is needed")
        elif len(set(field.vocabulary)) != len(field.vocabulary):
            problems.append(f"{path}.vocabulary: a label is listed twice")
    if field.type in ("labels", "list"):
        for bound in ("min", "max"):
            value = getattr(field, bound)
            if value is not None and (value != int(value) or value < 0):
                problems.append(f"{path}.{bound}: a count, a whole number >= 0")
    if field.type == "integer":
        for bound in ("min", "max"):
            value = getattr(field, bound)
            if value is not None and value != int(value):
                problems.append(f"{path}.{bound}: a whole number for type integer")
    if field.min is not None and field.max is not None and field.min > field.max:
        problems.append(f"{path}: min is more than max")
    if field.type == "labels" and field.min is not None and field.min > len(field.vocabulary):
        problems.append(f"{path}.min: more labels than the vocabulary has")
    if field.type == "object":
        if not field.fields:
            problems.append(f"{path}.fields: an object needs fields")
        for name, sub in (field.fields or {}).items():
            problems += _field_problems(sub, f"{path}.fields.{name}")
    if field.type == "list":
        if field.items is None:
            problems.append(f"{path}.items: a list needs items")
        else:
            problems += _field_problems(field.items, f"{path}.items")
    if field.type == "map":
        if field.keys is None:
            problems.append(f"{path}.keys: entities or vocabulary")
        if field.keys == "entities" and field.vocabulary:
            problems.append(f"{path}.vocabulary: not used with keys: entities")
        if field.keys == "vocabulary" and field.extra_keys:
            problems.append(f"{path}.extra_keys: only with keys: entities")
        if field.values is None:
            problems.append(f"{path}.values: a map needs values")
        else:
            problems += _field_problems(field.values, f"{path}.values")
    return problems


def _is_set(value: Any) -> bool:
    return value is not None and value != () and value != {}


# --- JSON Schema ------------------------------------------------------------------------


def _object_schema(fields: Mapping[str, FieldSpec], entities: Sequence[str]) -> dict[str, Any]:
    # Every property is listed as required (optional ones are nullable), the shape most
    # structured-output modes accept; check_fields also accepts an optional one left out.
    return {
        "type": "object",
        "properties": {name: field_schema(f, entities) for name, f in fields.items()},
        "required": list(fields),
        "additionalProperties": False,
    }


def field_schema(field: FieldSpec, entities: Sequence[str]) -> dict[str, Any]:
    """One field as JSON Schema (plain data)."""
    schema: dict[str, Any]
    if field.type == "label":
        schema = {"type": "string", "enum": list(field.vocabulary)}
    elif field.type == "labels":
        schema = {"type": "array", "items": {"type": "string", "enum": list(field.vocabulary)}}
        schema |= _counts(field)
    elif field.type in ("integer", "number"):
        schema = {"type": field.type}
        if field.min is not None:
            schema["minimum"] = int(field.min) if field.type == "integer" else field.min
        if field.max is not None:
            schema["maximum"] = int(field.max) if field.type == "integer" else field.max
    elif field.type == "boolean":
        schema = {"type": "boolean"}
    elif field.type == "object":
        schema = _object_schema(field.fields or {}, entities)
    elif field.type == "list":
        items = _needed(field.items, "items")
        schema = {"type": "array", "items": field_schema(items, entities)} | _counts(field)
    else:
        keys = map_keys(field, entities)
        value = field_schema(_needed(field.values, "values"), entities)
        if keys is None:
            schema = {"type": "object", "additionalProperties": value}
        else:
            schema = {
                "type": "object",
                "properties": {key: value for key in keys},
                "additionalProperties": False,
            }
    if field.description:
        schema["description"] = field.description
    if field.optional:
        schema = {"anyOf": [schema, {"type": "null"}]}
    return schema


def _needed(part: "FieldSpec | None", name: str) -> FieldSpec:
    """A part the load-time checks guarantee (a list's items, a map's values)."""
    if part is None:
        raise ValueError(f"a field without {name}: load lens specs with LensSpec.from_mapping")
    return part


def _counts(field: FieldSpec) -> dict[str, int]:
    counts = {}
    if field.min is not None:
        counts["minItems"] = int(field.min)
    if field.max is not None:
        counts["maxItems"] = int(field.max)
    return counts


def map_keys(field: FieldSpec, entities: Sequence[str]) -> list[str] | None:
    """The keys a map allows, or None for any key (an entity map when the corpus profile
    lists no entities: then the profile doesn't constrain names)."""
    if field.keys == "vocabulary":
        return list(field.vocabulary)
    if not entities:
        return None
    return [*entities, *(k for k in field.extra_keys if k not in entities)]


# --- checking replies -------------------------------------------------------------------


def check_fields(
    fields: Mapping[str, FieldSpec], value: Any, path: str, entities: Sequence[str]
) -> list[str]:
    """Problems with an object of lens fields (unknown keys, missing fields, bad values)."""
    if not isinstance(value, Mapping):
        return [f"{path}: expected an object"]
    problems = [f"{path}.{key}: not a field of the lens" for key in value if key not in fields]
    for name, field in fields.items():
        if name not in value:
            if not field.optional:
                problems.append(f"{path}.{name}: missing")
            continue
        problems += check_value(field, value[name], f"{path}.{name}", entities)
    return problems


def check_value(field: FieldSpec, value: Any, path: str, entities: Sequence[str]) -> list[str]:
    if value is None:
        return [] if field.optional else [f"{path}: missing (null)"]
    kind = field.type
    if kind == "label":
        if not isinstance(value, str) or value not in field.vocabulary:
            return [f"{path}: {value!r} is not in the vocabulary"]
        return []
    if kind == "labels":
        if not isinstance(value, list):
            return [f"{path}: expected a list of labels"]
        problems = [
            f"{path}: {item!r} is not in the vocabulary"
            for item in value
            if not isinstance(item, str) or item not in field.vocabulary
        ]
        if len(set(map(str, value))) != len(value):
            problems.append(f"{path}: a label is listed twice")
        return problems + _count_problems(field, len(value), path)
    if kind in ("integer", "number"):
        ok = _whole(value) if kind == "integer" else _number(value)
        if not ok:
            return [f"{path}: expected {'a whole number' if kind == 'integer' else 'a number'}"]
        if (field.min is not None and value < field.min) or (
            field.max is not None and value > field.max
        ):
            return [f"{path}: {value} is outside {field.min}..{field.max}"]
        return []
    if kind == "boolean":
        return [] if isinstance(value, bool) else [f"{path}: expected true or false"]
    if kind == "object":
        return check_fields(field.fields or {}, value, path, entities)
    if kind == "list":
        if not isinstance(value, list):
            return [f"{path}: expected a list"]
        items = _needed(field.items, "items")
        problems = _count_problems(field, len(value), path)
        for i, item in enumerate(value):
            problems += check_value(items, item, f"{path}[{i}]", entities)
        return problems
    if not isinstance(value, Mapping):
        return [f"{path}: expected an object"]
    values = _needed(field.values, "values")
    allowed = map_keys(field, entities)
    problems = []
    for key, item in value.items():
        if allowed is not None and key not in allowed:
            problems.append(f"{path}: key {key!r} is not allowed")
            continue
        problems += check_value(values, item, f"{path}.{key}", entities)
    return problems


def _count_problems(field: FieldSpec, count: int, path: str) -> list[str]:
    if field.min is not None and count < field.min:
        return [f"{path}: at least {int(field.min)} expected, got {count}"]
    if field.max is not None and count > field.max:
        return [f"{path}: at most {int(field.max)} expected, got {count}"]
    return []


def _whole(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    return isinstance(value, float) and isfinite(value) and value == int(value)


def _number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int | float) and isfinite(value)


def _empty(value: Any) -> bool:
    return value is None or value == [] or value == {} or value == ()


def annotation_rows(fields: Mapping[str, FieldSpec], annotation: Mapping[str, Any]) -> list[dict]:
    """One checked annotation as `annotations` rows: one row per label of a `label` or
    `labels` field (the label in `label`), one row with the JSON `value` for any other
    field. Labels nested in objects stay in `value`, already checked."""
    rows: list[dict[str, Any]] = []
    for name, field in fields.items():
        value = annotation.get(name)
        if value is None:
            continue
        if field.type == "label":
            rows.append({"field": name, "label": value, "value": None})
        elif field.type == "labels":
            rows += [{"field": name, "label": label, "value": None} for label in value]
        else:
            rows.append({"field": name, "label": None, "value": plain(value)})
    return rows


__all__ = [
    "FieldSpec",
    "LensPrompts",
    "LensSpec",
    "LensValidator",
    "annotation_rows",
    "check_fields",
    "check_value",
    "field_schema",
    "map_keys",
]
