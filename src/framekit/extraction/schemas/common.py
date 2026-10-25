"""Shared pieces of the extraction schemas: strict read-only models, one error that lists
every problem, digests, and loading YAML or JSON files.

The schemas follow the conventions of `framekit.frames.params`: unknown keys fail, numbers
must be numbers (YAML `yes` or `"4"` fails; a whole number is fine where a decimal is
expected) and finite, lists are stored as tuples, and one `ConfigError` lists every problem
with its path, so a file is fixed in one pass.
"""

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Annotated, Any, Literal, get_args

import yaml
from pydantic import BaseModel, BeforeValidator, ConfigDict, StringConstraints, ValidationError

from framekit.frames.params import FrozenDict, freeze, problems_from_validation

# Perception modules (DESIGN.md §3 L3a): `core` is always on; lenses and the search space
# ask for the others.
Module = Literal["core", "camera", "timing", "sound", "speech"]
MODULES: tuple[str, ...] = get_args(Module)

# Lens names, metadata fields: lower snake case.
Name = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]*$", max_length=64)]
# Unit ids: they become file names, so letters, digits, `_`, `-` and `.` only.
UnitId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$", max_length=128)]
# A version string: `1.0.0`, `0.1`, `2`.
Version = Annotated[str, StringConstraints(pattern=r"^[0-9]+(\.[0-9]+){0,2}$")]


class ConfigError(ValueError):
    """Inputs that can't be used (a search space, corpus, plan, lens spec or gold set),
    with every problem and its path."""

    def __init__(self, problems: Iterable[str], what: str = "the inputs"):
        self.problems: tuple[str, ...] = tuple(problems)
        self.what = what
        count = len(self.problems)
        super().__init__(
            f"{count} problem{'s' if count != 1 else ''} in {what}:\n"
            + "\n".join(f"- {p}" for p in self.problems)
        )


def tupled(value: Any) -> Any:
    """Lists (and lists inside lists) become tuples; mappings are left to their models."""
    if isinstance(value, list | tuple):
        return tuple(tupled(item) for item in value)
    return value


Tupled = BeforeValidator(tupled)
Frozen = BeforeValidator(freeze)  # any plain data, read-only all the way down


class Model(BaseModel):
    """Base for the extraction schemas: unknown keys fail, values are strict and finite,
    and a parsed model is read-only."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)


def canonical_json(value: Any) -> str:
    """Sorted keys, no spaces, UTF-8: the form every digest is computed from."""
    return json.dumps(
        plain(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def digest(value: Any) -> str:
    """sha256 of a value's canonical JSON (a str is hashed as it is)."""
    text = value if isinstance(value, str) else canonical_json(value)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def plain(value: Any) -> Any:
    """Writable JSON-shaped data: models dumped, read-only dicts and tuples copied."""
    if isinstance(value, BaseModel):
        return plain(value.model_dump(mode="json"))
    if isinstance(value, Mapping):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [plain(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    return value


def read_mapping(path: str | Path, what: str) -> dict[str, Any]:
    """A YAML (`.yaml`, `.yml`) or JSON (`.json`) file whose top level is a mapping. YAML is
    read with the safe loader, so a file can never construct Python objects."""
    file = Path(path).expanduser()
    try:
        text = file.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError([f"{file}: can't be read ({exc.strerror or exc})"], what) from None
    try:
        if file.suffix.lower() == ".json":
            data = json.loads(text)
        elif file.suffix.lower() in (".yaml", ".yml"):
            data = yaml.safe_load(text)
        else:
            raise ConfigError([f"{file}: expected a .yaml, .yml or .json file"], what)
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ConfigError([f"{file}: not valid {file.suffix[1:].upper()}: {exc}"], what) from None
    if not isinstance(data, dict):
        raise ConfigError([f"{file}: the top level must be a mapping"], what)
    return data


def unwrap(data: Mapping[str, Any], envelope: str) -> Mapping[str, Any]:
    """`{envelope: {...}}` (the file shape of docs/settings.md) or the bare mapping."""
    if (
        isinstance(data, Mapping)
        and set(data) == {envelope}
        and isinstance(data[envelope], Mapping)
    ):
        return data[envelope]
    return data


def parse[M: BaseModel](
    model: type[M], data: Any, what: str, extra: Callable[[M], list[str]] | None = None
) -> M:
    """Validate plain data into `model`, raising one ConfigError with every problem. `extra`
    adds checks that run once the structure is valid."""
    try:
        parsed = model.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(problems_from_validation(exc), what) from None
    if extra is not None:
        problems = extra(parsed)
        if problems:
            raise ConfigError(problems, what)
    return parsed


__all__ = [
    "MODULES",
    "ConfigError",
    "Frozen",
    "FrozenDict",
    "Model",
    "Module",
    "Name",
    "Tupled",
    "UnitId",
    "Version",
    "canonical_json",
    "digest",
    "freeze",
    "parse",
    "plain",
    "read_mapping",
    "tupled",
    "unwrap",
]
