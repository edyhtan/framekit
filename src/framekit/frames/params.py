"""Frame parameters (DESIGN.md §2.3): which frames a model sees, and how they're prepared
and delivered, as data.

A **frame profile** is a named, versioned block with four parts: `frame_selection`
(candidates, anchors, filters, budget), `image` (size and encoding), `delivery` (images,
storyboard or native video) and optional `views` (several looks per segment). The plan
assigns one profile per stage that sends pictures (`stages: {L3a: {profile: scene_detail}}`).

Strategies are referenced by name, as in the YAML of docs/settings.md §5.2:
- a bare name: `shot_boundaries`, or with its version: `shot_boundaries@1`;
- a single-key mapping: `{dedupe: {method: phash, max_distance: 6}}`;
- inline (the candidates shape): `{strategy: uniform, fps: 4}`.

`FrameParams.from_mapping(data)` checks everything at load time and raises one
`FrameParamsError` listing every problem with its path: the structure (unknown keys, types,
ranges), each strategy (known for its slot, settings valid for it) and the rules that tie
settings together (native video, views, png quality, ...). A rule is skipped only where a
setting it reads has a problem of its own, which is listed instead.

Values are strict: numbers must be numbers (not `true` or `"4"`) and finite; a whole number
is fine where a decimal is expected. Parsed parameters are read-only all the way down, so
what was checked and digested is what runs. Checks that need outside facts are separate
functions: `check_against_source` (the analysis copy) and `check_against_model` (a model's
vision limits), in `framekit.frames.checks`.
"""

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Annotated, Any, Literal, NoReturn

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    ValidatorFunctionWrapHandler,
    field_validator,
    model_validator,
)
from pydantic_core import ErrorDetails, PydanticCustomError

from framekit.frames.registry import (
    KIND_LABELS,
    RESERVED_SETTING,
    StrategyKind,
    UnknownStrategyError,
    get_strategy,
    split_reference,
)

# native_video: the provider samples the clip itself, at a uniform rate.
NATIVE_VIDEO_CANDIDATES = "uniform"
NATIVE_VIDEO_RULE = (
    "not allowed with native_video delivery (the provider samples the clip uniformly: "
    "only uniform candidates, image.max_side_px and image.aspect apply)"
)
REF_SHAPES = (
    "a strategy is a name ('uniform', 'uniform@1'), a single-key mapping "
    "({name: {settings}}) or an inline mapping ({strategy: name, settings...})"
)
PROBLEMS_ERROR = "frame_problems"  # our error type inside pydantic's ValidationError

Loc = tuple[str | int, ...]
Problem = tuple[Loc, str]  # (path, message)

Name = Annotated[str, StringConstraints(pattern=r"^[A-Za-z][A-Za-z0-9_-]*$")]


class FrameParamsError(ValueError):
    """Frame parameters that can't be used, with every problem and its path."""

    def __init__(self, problems: Iterable[str]):
        self.problems: tuple[str, ...] = tuple(problems)
        count = len(self.problems)
        super().__init__(
            f"{count} problem{'s' if count != 1 else ''} in frame parameters:\n"
            + "\n".join(f"- {p}" for p in self.problems)
        )


# --- read-only data ---------------------------------------------------------------------


class FrozenDict(dict[str, Any]):
    """A read-only, hashable dict (strategy settings, the profiles of a block). Equality and
    JSON work as for a dict; changing it raises TypeError."""

    __slots__ = ()

    def _read_only(self, *args: Any, **kwargs: Any) -> NoReturn:
        raise TypeError(f"{type(self).__name__} is read-only")

    __setitem__ = __delitem__ = clear = pop = popitem = _read_only  # type: ignore[assignment]
    setdefault = update = __ior__ = _read_only  # type: ignore[assignment]

    def __hash__(self) -> int:  # type: ignore[override]
        return hash(frozenset(self.items()))

    def __reduce__(self) -> tuple[type["FrozenDict"], tuple[dict[str, Any]]]:
        return (type(self), (dict(self),))


def freeze(value: Any) -> Any:
    """A read-only copy of plain data: mappings become FrozenDicts, lists become tuples."""
    if isinstance(value, Mapping):
        return FrozenDict({key: freeze(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(freeze(item) for item in value)
    return value


def thaw(value: Any) -> Any:
    """Plain, writable data again: mappings become dicts, tuples become lists."""
    if isinstance(value, Mapping):
        return {key: thaw(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [thaw(item) for item in value]
    return value


class _Model(BaseModel):
    # strict: no bool or str where a number is expected; allow_inf_nan: finite numbers only.
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)


def _listed() -> Any:
    """An optional list setting: given as a list, stored as a read-only tuple."""
    return Field(default=(), strict=False)


def _a_list(value: Any) -> Any:
    if not isinstance(value, list | tuple):
        raise ValueError("Input should be a valid list")
    return value


def _a_signal_list(value: Any) -> Any:
    # Checked before the items: a length check after them would count only valid items.
    if not _a_list(value):
        raise ValueError("at least one signal is needed")
    return value


AsList = BeforeValidator(_a_list)


# --- problems ---------------------------------------------------------------------------


def format_path(loc: Iterable[Any]) -> str:
    """`("a", "b", 0, "c")` → `"a.b[0].c"`."""
    path = ""
    for part in loc:
        if isinstance(part, int):
            path += f"[{part}]"
        elif isinstance(part, str) and part.startswith("["):
            path += part
        else:
            path += f".{part}" if path else str(part)
    return path


def _format(problem: Problem) -> str:
    loc, message = problem
    return f"{format_path(loc) or '(root)'}: {message}"


def _message(err: ErrorDetails) -> str:
    return err["msg"].removeprefix("Value error, ")


def _expand(errors: Iterable[ErrorDetails]) -> list[Problem]:
    """One problem per pydantic error; ours carry their own list, relative to where they
    were raised."""
    problems: list[Problem] = []
    for err in errors:
        loc = tuple(err["loc"])
        if err["type"] == PROBLEMS_ERROR:
            ctx = err.get("ctx") or {}
            problems += [((*loc, *sub), message) for sub, message in ctx["problems"]]
        else:
            problems.append((loc, _message(err)))
    return problems


def problems_from_validation(exc: ValidationError) -> list[str]:
    """One line per problem, `path: message`, from a pydantic ValidationError."""
    return [_format(problem) for problem in _expand(exc.errors(include_url=False))]


def _fail(problems: Sequence[Problem]) -> NoReturn:
    raise PydanticCustomError(
        PROBLEMS_ERROR,
        "{summary}",
        {"summary": "; ".join(map(_format, problems)), "problems": list(problems)},
    )


def _raise_problems(problems: Sequence[Problem]) -> None:
    if problems:
        _fail(problems)


# --- strategy references ----------------------------------------------------------------


class StrategyRef(_Model):
    """A strategy picked for a slot, with its settings: checked against the strategy once
    the reference sits in a profile (defaults are filled at resolution). `version` None
    means the newest registered version; resolution pins it."""

    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    version: int | None = Field(default=None, ge=1)
    params: Annotated[dict[str, Any], BeforeValidator(thaw), AfterValidator(freeze)] = (
        FrozenDict()
    )

    @property
    def key(self) -> str:
        return self.name if self.version is None else f"{self.name}@{self.version}"

    def to_mapping(self) -> str | dict[str, Any]:
        """The compact YAML shape: a bare name, or `{name: {settings}}`."""
        return {self.key: thaw(self.params)} if self.params else self.key

    def to_inline(self) -> dict[str, Any]:
        """The inline YAML shape used for candidates: `{strategy: name, settings...}`."""
        return {RESERVED_SETTING: self.key, **thaw(self.params)}


def parse_strategy_ref(value: Any) -> Any:
    """Normalizes the three YAML shapes into StrategyRef fields."""
    if isinstance(value, StrategyRef):
        return value
    if isinstance(value, str):
        name, version = split_reference(value)
        return {"name": name, "version": version, "params": {}}
    if isinstance(value, Mapping):
        if RESERVED_SETTING in value:
            name, version = split_reference(value[RESERVED_SETTING])
            params = {k: v for k, v in value.items() if k != RESERVED_SETTING}
            return {"name": name, "version": version, "params": params}
        if len(value) == 1:
            ((key, settings),) = value.items()
            name, version = split_reference(key)
            if settings is None:
                settings = {}
            if not isinstance(settings, Mapping):
                raise ValueError(f"the settings of {key!r} must be a mapping")
            return {"name": name, "version": version, "params": dict(settings)}
    raise ValueError(REF_SHAPES)


def _ref_name(value: Any) -> str | None:
    """The strategy a reference names, in any shape, or None if it doesn't parse."""
    try:
        parsed = parse_strategy_ref(value)
    except ValueError:
        return None
    return parsed.name if isinstance(parsed, StrategyRef) else parsed["name"]


def _in_slot(
    kind: StrategyKind, *, inline: bool = False
) -> Callable[[StrategyRef], StrategyRef]:
    """The check for one slot: the strategy exists for this kind and its settings pass its
    schema. The reference keeps the checked settings as given (resolution fills defaults).
    Settings paths follow the YAML shape: `candidates.fps` inline, `filters[0].dedupe.x`."""

    def check(ref: StrategyRef) -> StrategyRef:
        try:
            strategy = get_strategy(kind, ref.name, ref.version)
        except UnknownStrategyError as exc:
            _fail([((), str(exc))])
        try:
            settings = strategy.params.model_validate(thaw(ref.params))
        except ValidationError as exc:
            where: Loc = () if inline else (ref.key,)
            _fail(
                [
                    ((*where, *err["loc"]), f"{_message(err)} ({strategy.key})")
                    for err in exc.errors(include_url=False)
                ]
            )
        checked = settings.model_dump(mode="json", exclude_unset=True)
        return ref.model_copy(update={"params": freeze(checked)})

    return check


_parse = BeforeValidator(parse_strategy_ref)
CandidatesRef = Annotated[StrategyRef, _parse, AfterValidator(_in_slot("candidates", inline=True))]
AnchorRef = Annotated[StrategyRef, _parse, AfterValidator(_in_slot("anchor"))]
FilterRef = Annotated[StrategyRef, _parse, AfterValidator(_in_slot("filter"))]
SignalRef = Annotated[StrategyRef, _parse, AfterValidator(_in_slot("allocation"))]


# --- the profile ------------------------------------------------------------------------


class Budget(_Model):
    """How many frames survive and where they go. Unset caps mean no cap."""

    max_frames: int | None = Field(default=None, ge=1, le=100_000)  # per unit
    max_frames_per_scene: int | None = Field(default=None, ge=1, le=10_000)
    allocate_by: Annotated[tuple[SignalRef, ...], AsList] = _listed()  # views.mode single
    min_gap_s: float | None = Field(default=None, ge=0, le=3600)

    def is_empty(self) -> bool:
        return (
            self.max_frames is None
            and self.max_frames_per_scene is None
            and not self.allocate_by
            and self.min_gap_s is None
        )

    def cap(self) -> int | None:
        """The tightest frame cap that applies to one planned span (a scene lies inside a
        unit, so it can't exceed either cap)."""
        caps = [c for c in (self.max_frames, self.max_frames_per_scene) if c is not None]
        return min(caps) if caps else None


class FrameSelection(_Model):
    candidates: CandidatesRef  # where frames may come from
    anchors: Annotated[tuple[AnchorRef, ...], AsList] = _listed()  # always kept
    filters: Annotated[tuple[FilterRef, ...], AsList] = _listed()  # in order
    budget: Budget = Budget()


class ImagePrep(_Model):
    max_side_px: int = Field(gt=0, le=16_384)
    aspect: Literal["preserve"] = "preserve"  # fixed rule: never crop or stretch
    format: Literal["jpeg", "png", "webp"] | None = None
    quality: int | None = Field(default=None, ge=1, le=100)  # jpeg and webp

    @field_validator("aspect", mode="before")
    @classmethod
    def _always_preserve(cls, value: Any) -> Any:
        if value != "preserve":
            raise ValueError(
                f"aspect is always 'preserve' (frames are never cropped or stretched), "
                f"got {value!r}"
            )
        return value


class StoryboardGrid(_Model):
    """Frames tiled into one image: fewer images, less detail per frame. Each tile is a
    frame prepared by `image` (max_side_px applies per tile)."""

    rows: int = Field(default=2, ge=1, le=8)
    cols: int = Field(default=2, ge=1, le=8)


class Delivery(_Model):
    mode: Literal["images", "storyboard", "native_video"] = "images"
    storyboard: StoryboardGrid | None = None  # mode storyboard only


class OverviewView(_Model):
    fps: float | None = Field(default=None, gt=0, le=1000)  # None: the profile's candidates
    max_frames: int = Field(ge=1, le=10_000)  # the anchors count toward it


class DetailView(_Model):
    # the allocation signals whose peaks get a view
    around: Annotated[tuple[SignalRef, ...], BeforeValidator(_a_signal_list)] = Field(strict=False)
    span_s: float = Field(ge=0.001, le=3600)  # at least a millisecond, the planning grid
    max_views: int = Field(ge=1, le=100)
    max_frames: int = Field(ge=1, le=10_000)  # per detail view


class Views(_Model):
    mode: Literal["single", "split_by_budget", "overview_plus_detail", "per_module"]
    overview: OverviewView | None = None
    detail: DetailView | None = None


class FrameProfile(_Model):
    version: int = Field(ge=1)
    frame_selection: FrameSelection
    image: ImagePrep
    delivery: Delivery = Delivery()
    views: Views | None = None

    @model_validator(mode="wrap")
    @classmethod
    def _check(cls, data: Any, handler: ValidatorFunctionWrapHandler) -> "FrameProfile":
        """Every problem at once: when the structure fails, the rules that tie settings
        together still run on the settings that parsed."""
        parts = list(cls.model_fields)
        try:
            profile = handler(data)
        except ValidationError as exc:
            errors = exc.errors(include_url=False)
            rules = _profile_problems(_Settings(data, [err["loc"] for err in errors]))
            _fail(_by_part(_expand(errors) + rules, parts))
        _raise_problems(_by_part(_profile_problems(_Settings(profile)), parts))
        return profile

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "FrameProfile":
        """One profile from plain data; raises FrameParamsError listing every problem."""
        try:
            return cls.model_validate(thaw(data))
        except ValidationError as exc:
            raise FrameParamsError(problems_from_validation(exc)) from None

    def to_mapping(self) -> dict[str, Any]:
        """The YAML shape (docs/settings.md §5.2), without unset values."""
        return profile_mapping(self)


class StageFrames(_Model):
    profile: Name


class FrameParams(_Model):
    """The plan's frames block (`observation.frames`): named profiles and the profile each
    stage uses."""

    profiles: Annotated[dict[Name, FrameProfile], AfterValidator(FrozenDict)] = (
        FrozenDict()
    )
    stages: Annotated[dict[Name, StageFrames], AfterValidator(FrozenDict)] = FrozenDict()

    @model_validator(mode="wrap")
    @classmethod
    def _check(cls, data: Any, handler: ValidatorFunctionWrapHandler) -> "FrameParams":
        try:
            params = handler(data)
        except ValidationError as exc:
            errors = exc.errors(include_url=False)
            stages = _stage_problems(_Settings(data, [err["loc"] for err in errors]))
            _fail(_expand(errors) + stages)
        _raise_problems(_stage_problems(_Settings(params)))
        return params

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "FrameParams":
        """Parses and checks the shape of docs/settings.md §5.2 (keys `profiles` and
        `stages`; YAML or JSON parsing is the caller's job). Raises one FrameParamsError
        listing every problem with its path."""
        try:
            return cls.model_validate(thaw(data))
        except ValidationError as exc:
            raise FrameParamsError(problems_from_validation(exc)) from None

    def to_mapping(self) -> dict[str, Any]:
        return _clean(
            {
                "profiles": {
                    name: profile_mapping(profile)
                    for name, profile in self.profiles.items()
                },
                "stages": {stage: {"profile": e.profile} for stage, e in self.stages.items()},
            }
        )

    def profile_for(self, stage: str) -> FrameProfile:
        if stage not in self.stages:
            raise KeyError(f"no frame profile assigned to stage {stage!r}")
        return self.profiles[self.stages[stage].profile]


# --- rules that tie settings together ---------------------------------------------------


class _Unknown:
    def __repr__(self) -> str:
        return "UNKNOWN"


UNKNOWN: Any = _Unknown()  # a setting that has a problem of its own


def _is_set(value: Any) -> bool:
    """Whether a setting is given: not None, not empty; a mapping or a model only if one of
    its own settings is given."""
    if value is None:
        return False
    if isinstance(value, BaseModel):
        return any(_is_set(getattr(value, name)) for name in type(value).model_fields)
    if isinstance(value, Mapping):
        return any(_is_set(item) for item in value.values())
    if isinstance(value, list | tuple):
        return len(value) > 0
    return True


class _Settings:
    """Settings as the rules read them: the parsed model or, when parsing failed, the raw
    data, where a value with a problem of its own (at it, inside it, or in what holds it)
    reads as UNKNOWN."""

    def __init__(self, data: Any, bad: Iterable[Sequence[str | int]] = ()):
        self._data = data
        self._bad = [tuple(loc) for loc in bad]

    def raw(self, *path: str | int) -> Any:
        node = self._data
        for key in path:
            if isinstance(node, BaseModel):
                fields = type(node).model_fields
                node = getattr(node, key) if isinstance(key, str) and key in fields else None
            elif isinstance(node, Mapping):
                node = node.get(key)
            else:
                return None
            if node is None:
                return None
        return node

    def clean(self, *path: str | int, inside: bool = True) -> bool:
        """No problem at `path` or in what holds it (nor inside it, unless `inside` is
        False)."""
        return not any(
            path[: len(loc)] == loc or (inside and loc[: len(path)] == path) for loc in self._bad
        )

    def get(self, *path: str, default: Any = None) -> Any:
        """The setting at `path`: `default` when unset, UNKNOWN when it has a problem."""
        if not self.clean(*path):
            return UNKNOWN
        value = self.raw(*path)
        return default if value is None else value

    def is_set(self, *path: str) -> bool:
        return _is_set(self.raw(*path))

    def ref_name(self, *path: str) -> str | None:
        """The strategy named at `path`, if the reference parses (its settings may not)."""
        return _ref_name(self.raw(*path))

    def ref_names(self, *path: str) -> list[tuple[int, str]]:
        """(index, strategy name) of each listed reference that parses."""
        items = self.raw(*path)
        if not isinstance(items, list | tuple) or not self.clean(*path, inside=False):
            return []
        names = ((index, _ref_name(item)) for index, item in enumerate(items))
        return [(index, name) for index, name in names if name is not None]


def _by_part(problems: list[Problem], parts: list[str]) -> list[Problem]:
    """Problems grouped by the part they're in, in schema order (stable within a part)."""

    def rank(problem: Problem) -> int:
        part = problem[0][0] if problem[0] else None
        return parts.index(part) if isinstance(part, str) and part in parts else len(parts)

    return sorted(problems, key=rank)


LISTED_ONCE: tuple[tuple[tuple[str, ...], StrategyKind], ...] = (
    (("frame_selection", "anchors"), "anchor"),
    (("frame_selection", "budget", "allocate_by"), "allocation"),
    (("views", "detail", "around"), "allocation"),
)
NOT_WITH_NATIVE_VIDEO: tuple[tuple[str, ...], ...] = (
    ("frame_selection", "anchors"),
    ("frame_selection", "filters"),
    ("frame_selection", "budget"),
    ("image", "format"),
    ("image", "quality"),
    ("views",),
)


def _profile_problems(s: _Settings) -> list[Problem]:
    """The rules that tie a profile's settings together, with paths relative to it."""
    problems: list[Problem] = []
    for path, kind in LISTED_ONCE:
        seen: set[str] = set()
        for index, name in s.ref_names(*path):
            if name in seen:
                problems.append(((*path, index), f"{KIND_LABELS[kind]} {name!r} is listed twice"))
            seen.add(name)
    if s.get("image", "format") == "png" and s.is_set("image", "quality"):
        problems.append((("image", "quality"), "png is lossless; quality applies to jpeg and webp"))
    mode = s.get("delivery", "mode", default="images")
    if mode not in (UNKNOWN, "storyboard") and s.is_set("delivery", "storyboard"):
        problems.append((("delivery", "storyboard"), "a grid applies only to mode storyboard"))
    if mode == "native_video":
        candidates = s.ref_name("frame_selection", "candidates")
        if candidates is not None and candidates != NATIVE_VIDEO_CANDIDATES:
            problems.append(
                (("frame_selection", "candidates"), f"{candidates!r} {NATIVE_VIDEO_RULE}")
            )
        problems += [(path, NATIVE_VIDEO_RULE) for path in NOT_WITH_NATIVE_VIDEO if s.is_set(*path)]
    elif mode is not UNKNOWN and s.is_set("views"):
        problems += _views_problems(s)
    return problems


def _views_problems(s: _Settings) -> list[Problem]:
    problems: list[Problem] = []
    mode = s.get("views", "mode")
    for part in ("overview", "detail"):
        present = s.is_set("views", part)
        if mode == "overview_plus_detail" and not present:
            problems.append((("views", part), "required for mode overview_plus_detail"))
        elif mode not in (UNKNOWN, "overview_plus_detail") and present:
            problems.append((("views", part), "applies only to mode overview_plus_detail"))
    overview = s.get("views", "overview", "max_frames")
    views = s.get("views", "detail", "max_views")
    frames = s.get("views", "detail", "max_frames")
    caps = {
        name: s.get("frame_selection", "budget", name)
        for name in ("max_frames_per_scene", "max_frames")
    }
    if any(n is UNKNOWN for n in (overview, views, frames, *caps.values())):
        return problems
    given = {name: cap for name, cap in caps.items() if cap is not None}
    if given and None not in (overview, views, frames):
        name = min(given, key=given.__getitem__)  # the tighter cap (per scene on a tie)
        total = overview + views * frames
        if total > given[name]:
            problems.append(
                (
                    ("views",),
                    f"overview ({overview}) + detail ({views} x {frames}) = {total} frames, "
                    f"more than frame_selection.budget.{name} ({given[name]})",
                )
            )
    return problems


def _stage_problems(s: _Settings) -> list[Problem]:
    """Stages that name a missing profile. A profile with problems of its own exists."""
    profiles, stages = s.raw("profiles"), s.raw("stages")
    profiles = {} if profiles is None else profiles
    if not isinstance(profiles, Mapping) or not s.clean("profiles", inside=False):
        return []
    if not isinstance(stages, Mapping) or not s.clean("stages", inside=False):
        return []
    names = sorted(name for name in profiles if isinstance(name, str))
    problems: list[Problem] = []
    for stage in stages:
        name = s.get("stages", stage, "profile")
        if isinstance(name, str) and name not in names:
            problems.append(
                (
                    ("stages", stage, "profile"),
                    f"no frame profile named {name!r} (profiles: {', '.join(names) or 'none'})",
                )
            )
    return problems


# --- the YAML shape ---------------------------------------------------------------------


def profile_mapping(profile: FrameProfile) -> dict[str, Any]:
    fs = profile.frame_selection
    budget = fs.budget.model_dump(exclude={"allocate_by"})
    budget["allocate_by"] = [ref.to_mapping() for ref in fs.budget.allocate_by]
    views = None
    if profile.views is not None:
        views = profile.views.model_dump(exclude={"detail"})
        if profile.views.detail is not None:
            detail = profile.views.detail.model_dump(exclude={"around"})
            detail["around"] = [ref.to_mapping() for ref in profile.views.detail.around]
            views["detail"] = detail
    return _clean(
        {
            "version": profile.version,
            "frame_selection": {
                "candidates": fs.candidates.to_inline(),
                "anchors": [ref.to_mapping() for ref in fs.anchors],
                "filters": [ref.to_mapping() for ref in fs.filters],
                "budget": budget,
            },
            "image": profile.image.model_dump(),
            "delivery": profile.delivery.model_dump(),
            "views": views,
        }
    )


def _clean(value: Any) -> Any:
    """Plain data without None or empty collections, recursively; lists keep their order."""
    if isinstance(value, Mapping):
        out = {}
        for key, item in value.items():
            item = _clean(item)
            if item is None or (isinstance(item, dict | list) and not item):
                continue
            out[key] = item
        return out
    if isinstance(value, list | tuple):
        return [_clean(item) for item in value]
    return value
