"""Vision parameters (`vision_params`, DESIGN.md §2.3): how a model sees video, as data.

A **vision profile** is a named, versioned block with four parts: `frame_selection`
(candidates, anchors, filters, budget), `image` (size and encoding), `delivery` (images,
storyboard or native video) and optional `views` (several looks per segment). The plan
assigns one profile per stage that sends pictures (`stages: {L3a: {vision: scene_detail}}`).

Strategies are referenced by name, as in the YAML of docs/settings.md §5.2:
- a bare name: `shot_boundaries`, or with its version: `shot_boundaries@1`;
- a single-key mapping: `{dedupe: {method: phash, max_distance: 6}}`;
- inline (the candidates shape): `{strategy: uniform, fps: 4}`.

`VisionParams.from_mapping(data)` checks everything at load time and raises one
`VisionParamsError` listing every problem with its path. Checks that need outside facts
are separate functions: `check_against_source` (the analysis copy) and
`check_against_model` (a model's vision limits), in `framekit.vision.checks`.
"""

from collections.abc import Iterable, Mapping
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticCustomError

from framekit.vision.registry import (
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
PROBLEMS_ERROR = "vision_problems"  # our error type inside pydantic's ValidationError

Name = Annotated[str, StringConstraints(pattern=r"^[A-Za-z][A-Za-z0-9_-]*$")]


class VisionParamsError(ValueError):
    """Vision parameters that can't be used, with every problem and its path."""

    def __init__(self, problems: Iterable[str]):
        self.problems: tuple[str, ...] = tuple(problems)
        count = len(self.problems)
        super().__init__(
            f"{count} problem{'s' if count != 1 else ''} in vision parameters:\n"
            + "\n".join(f"- {p}" for p in self.problems)
        )


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --- strategy references ----------------------------------------------------------------


class StrategyRef(_Model):
    """A strategy picked for a slot, with its settings as given. `version` None means the
    newest registered version; resolution pins it."""

    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    version: int | None = Field(default=None, ge=1)
    params: dict[str, Any] = {}

    @property
    def key(self) -> str:
        return self.name if self.version is None else f"{self.name}@{self.version}"

    def to_mapping(self) -> str | dict[str, Any]:
        """The compact YAML shape: a bare name, or `{name: {settings}}`."""
        return {self.key: dict(self.params)} if self.params else self.key

    def to_inline(self) -> dict[str, Any]:
        """The inline YAML shape used for candidates: `{strategy: name, settings...}`."""
        return {RESERVED_SETTING: self.key, **self.params}


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


Ref = Annotated[StrategyRef, BeforeValidator(parse_strategy_ref)]


# --- the profile ------------------------------------------------------------------------


class Budget(_Model):
    """How many frames survive and where they go. Unset caps mean no cap."""

    max_frames: int | None = Field(default=None, ge=1, le=100_000)  # per unit
    max_frames_per_scene: int | None = Field(default=None, ge=1, le=10_000)
    allocate_by: list[Ref] = []  # allocation signals; used when views.mode is single
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
    candidates: Ref  # where frames may come from
    anchors: list[Ref] = []  # frames that are always kept
    filters: list[Ref] = []  # in order
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
    max_frames: int = Field(ge=1, le=10_000)


class DetailView(_Model):
    around: list[Ref] = Field(min_length=1)  # allocation signals whose peaks get a view
    span_s: float = Field(gt=0, le=3600)
    max_views: int = Field(ge=1, le=100)
    max_frames: int = Field(ge=1, le=10_000)  # per detail view


class Views(_Model):
    mode: Literal["single", "split_by_budget", "overview_plus_detail", "per_module"]
    overview: OverviewView | None = None
    detail: DetailView | None = None


class VisionProfile(_Model):
    version: int = Field(ge=1)
    frame_selection: FrameSelection
    image: ImagePrep
    delivery: Delivery = Delivery()
    views: Views | None = None

    @model_validator(mode="after")
    def _check(self) -> "VisionProfile":
        _raise_problems(profile_problems(self))
        return self

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "VisionProfile":
        """One profile from plain data; raises VisionParamsError listing every problem."""
        try:
            return cls.model_validate(data)
        except ValidationError as exc:
            raise VisionParamsError(problems_from_validation(exc)) from None

    def to_mapping(self) -> dict[str, Any]:
        """The YAML shape (docs/settings.md §5.2), without unset values."""
        return profile_mapping(self)


class StageVision(_Model):
    vision: Name


class VisionParams(_Model):
    """The plan's vision block: named profiles and the profile each stage uses."""

    vision_profiles: dict[Name, VisionProfile] = {}
    stages: dict[Name, StageVision] = {}

    @model_validator(mode="after")
    def _check(self) -> "VisionParams":
        stages = {stage: entry.vision for stage, entry in self.stages.items()}
        _raise_problems(stage_problems(stages, self.vision_profiles))
        return self

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "VisionParams":
        """Parses and checks the shape of docs/settings.md §5.2 (keys `vision_profiles` and
        `stages`; YAML or JSON parsing is the caller's job). Raises one VisionParamsError
        listing every problem with its path."""
        try:
            return cls.model_validate(data)
        except ValidationError as exc:
            problems = problems_from_validation(exc)
        # The model checks stage references only once every profile is valid; check them
        # on the raw data too, so one error lists everything.
        for problem in _raw_stage_problems(data):
            if problem not in problems:
                problems.append(problem)
        raise VisionParamsError(problems)

    def to_mapping(self) -> dict[str, Any]:
        return _clean(
            {
                "vision_profiles": {
                    name: profile_mapping(profile)
                    for name, profile in self.vision_profiles.items()
                },
                "stages": {stage: {"vision": e.vision} for stage, e in self.stages.items()},
            }
        )

    def profile_for(self, stage: str) -> VisionProfile:
        if stage not in self.stages:
            raise KeyError(f"no vision profile assigned to stage {stage!r}")
        return self.vision_profiles[self.stages[stage].vision]


# --- load-time checks -------------------------------------------------------------------


def profile_problems(profile: VisionProfile) -> list[str]:
    """Every problem in one structurally valid profile, with paths relative to it."""
    fs = profile.frame_selection
    problems = _ref_problems("frame_selection.candidates", "candidates", fs.candidates)
    problems += _refs_problems("frame_selection.anchors", "anchor", fs.anchors, unique=True)
    problems += _refs_problems("frame_selection.filters", "filter", fs.filters, unique=False)
    problems += _refs_problems(
        "frame_selection.budget.allocate_by", "allocation", fs.budget.allocate_by, unique=True
    )
    if profile.views is not None and profile.views.detail is not None:
        problems += _refs_problems(
            "views.detail.around", "allocation", profile.views.detail.around, unique=True
        )
    if profile.image.format == "png" and profile.image.quality is not None:
        problems.append("image.quality: png is lossless; quality applies to jpeg and webp")
    if profile.delivery.storyboard is not None and profile.delivery.mode != "storyboard":
        problems.append("delivery.storyboard: a grid applies only to mode storyboard")
    if profile.delivery.mode == "native_video":
        problems += _native_video_problems(profile)
    elif profile.views is not None:
        problems += _views_problems(profile, profile.views)
    return problems


def _ref_problems(
    path: str, kind: StrategyKind, ref: StrategyRef, settings_path: str | None = None
) -> list[str]:
    try:
        strategy = get_strategy(kind, ref.name, ref.version)
    except UnknownStrategyError as exc:
        return [f"{path}: {exc}"]
    try:
        strategy.params.model_validate(ref.params)
    except ValidationError as exc:
        base = settings_path or path
        return [
            f"{format_path((base, *err['loc']))}: {err['msg']} ({strategy.key})"
            for err in exc.errors(include_url=False)
        ]
    return []


def _refs_problems(
    path: str, kind: StrategyKind, refs: list[StrategyRef], *, unique: bool
) -> list[str]:
    problems: list[str] = []
    seen: set[str] = set()
    for index, ref in enumerate(refs):
        item = f"{path}[{index}]"
        problems += _ref_problems(item, kind, ref, settings_path=f"{item}.{ref.key}")
        if unique and ref.name in seen:
            problems.append(f"{item}: {KIND_LABELS[kind]} {ref.name!r} is listed twice")
        seen.add(ref.name)
    return problems


def _native_video_problems(profile: VisionProfile) -> list[str]:
    fs, image = profile.frame_selection, profile.image
    problems = []
    if fs.candidates.name != NATIVE_VIDEO_CANDIDATES:
        problems.append(f"frame_selection.candidates: {fs.candidates.name!r} {NATIVE_VIDEO_RULE}")
    for part, present in (
        ("frame_selection.anchors", bool(fs.anchors)),
        ("frame_selection.filters", bool(fs.filters)),
        ("frame_selection.budget", not fs.budget.is_empty()),
        ("image.format", image.format is not None),
        ("image.quality", image.quality is not None),
        ("views", profile.views is not None),
    ):
        if present:
            problems.append(f"{part}: {NATIVE_VIDEO_RULE}")
    return problems


def _views_problems(profile: VisionProfile, views: Views) -> list[str]:
    problems = []
    if views.mode == "overview_plus_detail":
        for part in ("overview", "detail"):
            if getattr(views, part) is None:
                problems.append(f"views.{part}: required for mode overview_plus_detail")
    else:
        for part in ("overview", "detail"):
            if getattr(views, part) is not None:
                problems.append(f"views.{part}: applies only to mode overview_plus_detail")
    cap = profile.frame_selection.budget.max_frames_per_scene
    if views.overview is not None and views.detail is not None and cap is not None:
        overview, detail = views.overview.max_frames, views.detail
        total = overview + detail.max_views * detail.max_frames
        if total > cap:
            problems.append(
                f"views: overview ({overview}) + detail ({detail.max_views} x "
                f"{detail.max_frames}) = {total} frames, more than "
                f"frame_selection.budget.max_frames_per_scene ({cap})"
            )
    return problems


def stage_problems(stages: Mapping[str, str], profiles: Iterable[str]) -> list[str]:
    names = sorted(profiles)
    return [
        f"stages.{stage}.vision: no vision profile named {name!r} "
        f"(profiles: {', '.join(names) or 'none'})"
        for stage, name in stages.items()
        if name not in names
    ]


def _raw_stage_problems(data: Any) -> list[str]:
    if not isinstance(data, Mapping):
        return []
    profiles, stages = data.get("vision_profiles", {}), data.get("stages", {})
    if not isinstance(profiles, Mapping) or not isinstance(stages, Mapping):
        return []
    refs = {
        stage: entry["vision"]
        for stage, entry in stages.items()
        if isinstance(entry, Mapping) and isinstance(entry.get("vision"), str)
    }
    return stage_problems(refs, [name for name in profiles if isinstance(name, str)])


def _raise_problems(problems: list[str]) -> None:
    if problems:
        raise PydanticCustomError(
            PROBLEMS_ERROR, "{summary}", {"summary": "; ".join(problems), "problems": problems}
        )


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


def problems_from_validation(exc: ValidationError) -> list[str]:
    """One line per problem, `path: message`, from a pydantic ValidationError."""
    problems: list[str] = []
    for err in exc.errors(include_url=False):
        base = format_path(err["loc"])
        if err["type"] == PROBLEMS_ERROR:
            problems += [f"{base}.{p}" if base else p for p in err["ctx"]["problems"]]
        else:
            message = err["msg"].removeprefix("Value error, ")
            problems.append(f"{base or '(root)'}: {message}")
    return problems


# --- the YAML shape ---------------------------------------------------------------------


def profile_mapping(profile: VisionProfile) -> dict[str, Any]:
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
    """Drops None and empty collections, recursively; keeps list order."""
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
