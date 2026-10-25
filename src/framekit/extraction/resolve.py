"""Resolve A + V + plan into one checked, recorded description of a run (DESIGN.md §2.4,
§4 "Resolve"), before anything is spent.

- **Derivation rules v1** fill what the plan leaves to the search space, deterministically,
  and each one applied is recorded: perception modules, beat lengths, fixed windows, the
  phrasing style, the lenses, the units in scope.
- **Checks**, every problem in one `ConfigError`: the units in scope, alignment, signal
  detectors, a model for every role the run uses (vision models for frame-sending roles,
  checked against each frame profile), embeddings, frame rates against the analysis copy,
  and governance (what may leave the machine).
- **Recorded:** the resolved plan in canonical form with its digests, for the manifest and
  the benchmark harness's fairness check.
"""

import ipaddress
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from framekit.extraction.schemas.common import ConfigError, digest, plain
from framekit.extraction.schemas.corpus import Corpus, UnitRef
from framekit.extraction.schemas.lens import LensSpec
from framekit.extraction.schemas.plan import ExtractionPlan, RoleModel
from framekit.extraction.schemas.space import SearchSpace
from framekit.frames import (
    ModelVisionCaps,
    ResolvedFrameParams,
    ResolvedProfile,
    SourceLimits,
    check_against_model,
    check_against_source,
    resolve,
)
from framekit.models.adapters.base import ExternalAIAdapterBase
from framekit.models.catalog import ModelSpec, ProviderConfig

WHAT = "the extraction run"
DEFAULT_BEAT_MIN_S = 0.5  # a placeholder, like every number
DEFAULT_MAX_TOKENS = {"L2": 4096, "L3a": 2048, "L3b": 1024, "L2_merge": 2048}
FRAME_ROLES = ("L2", "L3a")  # roles that send frames to their model
# The plan's paths the derivation rules write: masked when the harness compares the plans
# of different search spaces (only these may differ).
DERIVED_PATHS = (
    ("interpretation", "perception", "modules"),
    ("structure", "beats"),
    ("structure", "segmentation", "fixed_windows", "window_s"),
    ("interpretation", "lens_passes", "lenses"),
    ("interpretation", "lens_passes", "phrasings", "style"),
    ("scope",),
)
# Not part of what a run computes: masked from the content digest.
NOT_CONTENT = (("policy", "budget"), ("policy", "governance"), ("policy", "formatting", "layout"))


@dataclass(frozen=True)
class RoleBinding:
    """The model a role uses and how it's called."""

    role: str
    model_id: str  # what requests name: a source's name or a registry model id
    spec: ModelSpec
    provider: ProviderConfig
    adapter: type
    max_tokens: int
    temperature: float | None
    seed: int | None
    reasoning: bool
    source: str | None = None

    def manifest(self) -> dict[str, Any]:
        return {
            "model_id": self.spec.model_id,
            "requested": self.model_id,
            "provider": self.provider.name,
            "source": self.source,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "seed": self.seed,
            "reasoning": self.reasoning,
        }

    def identity(self) -> dict[str, Any]:
        """What the input hash of a stage using this role covers."""
        return {
            "model_id": self.spec.model_id,
            "provider": self.provider.name,
            "base_url": self.provider.base_url,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "seed": self.seed,
            "reasoning": self.reasoning,
        }


@dataclass(frozen=True)
class ScopedUnit:
    unit: UnitRef
    excluded_ranges: tuple[tuple[float, float], ...]
    expected_duration_s: float | None

    @property
    def unit_id(self) -> str:
        return self.unit.unit_id


@dataclass(frozen=True)
class ResolvedRun:
    space: SearchSpace
    corpus: Corpus
    plan: ExtractionPlan
    lenses: tuple[LensSpec, ...]
    modules: tuple[str, ...]
    beats: tuple[float, float]
    windows: tuple[float, float] | None  # (window_s, stride_s) when windows are made
    style_examples: tuple[str, ...]  # phrasing style from the space; empty: the lens's style
    units: tuple[ScopedUnit, ...]
    roles: Mapping[str, RoleBinding]
    frames: ResolvedFrameParams
    derivations: tuple[dict[str, Any], ...]
    resolved_plan: dict[str, Any]
    plan_digest: str  # everything resolved
    content_digest: str  # without budget, governance and layout
    fairness_digest: str  # also without what the derivation rules wrote
    model_inputs: str  # provider_transient | local_only
    component_digests: Mapping[str, str] = field(default_factory=dict)  # as given
    notes: tuple[str, ...] = field(default=())

    def profile(self, stage: str) -> ResolvedProfile:
        return self.frames.for_stage(stage)

    @property
    def space_digest(self) -> str:
        return self.space.digest


def _mask(data: dict[str, Any], paths: tuple[tuple[str, ...], ...]) -> dict[str, Any]:
    result = plain(data)
    for path in paths:
        node = result
        for key in path[:-1]:
            node = node.get(key, {}) if isinstance(node, dict) else {}
        if isinstance(node, dict):
            node.pop(path[-1], None)
    return result


def _is_local(config: ProviderConfig) -> bool:
    """A provider on this machine or a private network: an IP literal that's loopback,
    private or link-local, or `localhost`. Anything else counts as remote."""
    url = config.resolved_base_url()
    if not url:
        return False  # the SDK's default endpoint: the provider's cloud
    host = urlparse(url).hostname or ""
    if host == "localhost":
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return address.is_loopback or address.is_private or address.is_link_local


def _lookup(router: Any, model_id: str) -> tuple[ModelSpec, ProviderConfig, type, str | None]:
    """A model's spec, provider and adapter class, as the router will resolve them."""
    source = router.sources.get(model_id)
    if source is not None:
        return source.spec, source.config, source.adapter, model_id
    registry = router.registry
    spec = registry.resolve(model_id)
    config = registry.providers.get(spec.provider)
    if config is None:
        raise LookupError(f"provider {spec.provider!r} of {model_id!r} is not configured")
    adapter = registry.adapters.get(config.protocol)
    if adapter is None:
        raise LookupError(f"no adapter for protocol {config.protocol!r}")
    return spec, config, adapter, None


def _serves_embeddings(adapter: type) -> bool:
    return getattr(adapter, "embed", None) is not getattr(ExternalAIAdapterBase, "embed", None)


def resolve_run(
    space: SearchSpace, corpus: Corpus, plan: ExtractionPlan | None, *, router: Any
) -> ResolvedRun:
    """Resolve and check everything before any spend. Raises one ConfigError listing every
    problem. `router` is the ConversationRouter the run will use (its sources and registry
    say which models exist and where they're served)."""
    plan = plan if plan is not None else ExtractionPlan()
    # The plan's components, by the names the checks and messages use.
    observation, structure = plan.observation, plan.structure
    interpretation, policy = plan.interpretation, plan.policy
    problems: list[str] = []
    problems += [f"plan: {p}" for p in plan.problems()]
    problems += [f"corpus: {p}" for p in corpus.problems()]
    try:
        lenses = space.lens_specs
    except ValueError as exc:
        raise ConfigError([f"search space: {exc}"], WHAT) from None
    derivations: list[dict[str, Any]] = []
    notes: list[str] = []

    # --- scope: which units, which ranges ------------------------------------------------
    known = {u.unit_id: u for u in corpus.units}
    pool = list(corpus.subset) if corpus.subset is not None else [u.unit_id for u in corpus.units]
    include = space.scope.units.include
    for u in () if include == "all" else include:
        if u not in known:
            problems.append(f"search space: scope.units.include: {u!r} is not in the corpus")
        elif u not in pool:
            problems.append(
                f"search space: scope.units.include: {u!r} is outside the corpus subset "
                "(the scope can only narrow it)"
            )
    for u in space.scope.units.exclude:
        if u not in known:
            problems.append(f"search space: scope.units.exclude: {u!r} is not in the corpus")
    chosen = [
        u for u in pool if (include == "all" or u in include) and u not in space.scope.units.exclude
    ]
    if not chosen:
        problems.append("search space: scope: no units left to process")
    units = []
    for unit_id in chosen:
        unit = known[unit_id]
        entry = corpus.manifest_for(unit_id)
        ranges = (
            entry.excluded_ranges if entry and space.scope.excluded_ranges == "from_corpus" else ()
        )
        if any(not segment.is_identity() for segment in unit.alignment_override):
            problems.append(
                f"corpus: units[{unit_id}].alignment_override: aligning a file to a reference "
                "timeline isn't built yet (DESIGN.md §8.2); each file's own timeline is used"
            )
        units.append(
            ScopedUnit(unit, tuple(sorted(ranges)), entry.expected_duration_s if entry else None)
        )
    derivations.append(
        {
            "rule": "scope",
            "target": "units",
            "value": chosen,
            "from": {"corpus_subset": plain(corpus.subset), "scope": plain(space.scope)},
        }
    )

    # --- perception modules -----------------------------------------------------------
    derived = {"core", *space.evidence.modules}
    by_lens = {lens.name: sorted(lens.requires_modules) for lens in lenses}
    for needed in by_lens.values():
        derived |= set(needed)
    asked = interpretation.perception.modules
    explicit = () if asked == "derived" else asked
    modules = tuple(
        m for m in ("core", "camera", "timing", "sound", "speech") if m in derived | set(explicit)
    )
    derivations.append(
        {
            "rule": "modules",
            "target": "interpretation.perception.modules",
            "value": list(modules),
            "from": {
                "evidence": list(space.evidence.modules),
                "lenses": by_lens,
                "plan": list(explicit),
            },
        }
    )
    if "speech" in modules and observation.signals.speech is None:
        problems.append(
            "plan: interpretation.perception.modules: speech needs a speech detector in "
            "observation.signals.speech "
            "(none is built in; register one)"
        )

    # --- beats and windows from the resolution ---------------------------------------
    moment = space.resolution.moment_length_s
    beat_max = (
        structure.beats.max_s if structure.beats.max_s is not None else moment.max
    )
    beat_min = (
        structure.beats.min_s
        if structure.beats.min_s is not None
        else min(DEFAULT_BEAT_MIN_S, beat_max)
    )
    if beat_max > moment.max:
        problems.append(
            f"plan: structure.beats.max_s: {beat_max:g} s is longer than the search space's "
            f"longest moment ({moment.max:g} s)"
        )
    if beat_min > beat_max:
        problems.append("plan: structure.beats: min_s is more than max_s")
    derivations.append(
        {
            "rule": "beats",
            "target": "structure.beats",
            "value": {"min_s": beat_min, "max_s": beat_max},
            "from": {"moment_length_s": plain(moment), "plan": plain(structure.beats)},
        }
    )
    windows = None
    if structure.segmentation.mode == "both":
        fixed = structure.segmentation.fixed_windows
        window_s = fixed.window_s if fixed.window_s is not None else moment.min
        if not moment.min <= window_s <= moment.max:
            problems.append(
                f"plan: structure.segmentation.fixed_windows.window_s: {window_s:g} s is "
                f"outside the search space's moment length ({moment.min:g}-{moment.max:g} s)"
            )
        windows = (window_s, fixed.stride_s)
        derivations.append(
            {
                "rule": "windows",
                "target": "structure.segmentation.fixed_windows",
                "value": {"window_s": window_s, "stride_s": fixed.stride_s},
                "from": {"moment_length_s": plain(moment)},
            }
        )

    # --- lenses and phrasing style ---------------------------------------------------
    derivations.append(
        {
            "rule": "lenses",
            "target": "interpretation.lens_passes.lenses",
            "value": [{"name": s.name, "version": s.version, "digest": s.digest} for s in lenses],
            "from": "search_space.lenses",
        }
    )
    examples = space.query_style.examples
    from_space = interpretation.lens_passes.phrasings.style == "from_space" and bool(examples)
    style_examples = tuple(examples) if from_space else ()
    derivations.append(
        {
            "rule": "phrasing_style",
            "target": "interpretation.lens_passes.phrasings.style",
            "value": "space_examples" if from_space else "lens_style",
            "from": {"plan": interpretation.lens_passes.phrasings.style, "examples": len(examples)},
        }
    )
    lens_targets = interpretation.lens_passes.targets
    if "beat" in lens_targets and not any("beat" in s.targets for s in lenses):
        notes.append("interpretation.lens_passes.targets has beat, but no lens annotates beats")

    # --- signal detectors ------------------------------------------------------------
    try:
        from framekit.extraction.media import get_detector
    except ImportError as exc:
        problems.append(
            f'extraction needs the media extra: pip install "framekit[media]" ({exc.name} '
            "is missing)"
        )
    else:
        signals = observation.signals
        wanted = [
            ("shots", signals.shots.detector, "shots.detector"),
            ("audio", signals.audio.onsets.method, "audio.onsets.method"),
            ("motion", signals.motion.method, "motion.method"),
        ]
        if signals.speech is not None:
            wanted.append(("speech", signals.speech.model, "speech.model"))
        if signals.audio.tagger is not None:
            wanted.append(("audio_tags", signals.audio.tagger.model, "audio.tagger.model"))
        for kind, reference, path in wanted:
            try:
                get_detector(kind, reference)
            except (LookupError, ValueError) as exc:
                problems.append(f"plan: observation.signals.{path}: {exc}")

    # --- frame profiles -------------------------------------------------------------
    frames = resolve(observation.frames)
    frame_roles = [r for r in FRAME_ROLES if r == "L3a" or structure.segmentation.method == "model"]
    for role in frame_roles:
        if role not in frames.stages:
            problems.append(
                f"plan: observation.frames.stages.{role}: {role} sends frames, so it needs a "
                "frame profile"
            )
            continue
        profile = frames.for_stage(role)
        if profile.profile.delivery.mode == "native_video":
            problems.append(
                f"plan: observation.frames.profiles.{profile.name}.delivery.mode: native video "
                "needs the video_understand port, which isn't built yet (use images or "
                "storyboard)"
            )
        limits = SourceLimits(observation.normalize.fps, 16384, 16384)
        problems += [f"plan: observation.frames.{p}" for p in check_against_source(profile, limits)]

    # --- models ----------------------------------------------------------------------
    needed_roles = ["L3a", "L3b"]
    if structure.segmentation.method == "model":
        needed_roles.insert(0, "L2")
    if interpretation.embedding.text is not None:
        needed_roles.append("embed_text")
    if interpretation.embedding.image is not None:
        problems.append(
            "plan: interpretation.embedding.image: image embeddings need an embed_image port, "
            "which isn't built yet (set interpretation.embedding.image to null)"
        )
    if plan.models.L2_merge is not None:
        notes.append("models.L2_merge is for chunked long units, which aren't built yet: unused")
    defaults = plan.models.defaults
    roles: dict[str, RoleBinding] = {}
    for role in needed_roles:
        chosen_model: RoleModel | None = plan.models.role(role)
        if chosen_model is None:
            hint = (
                " (or set structure.segmentation.method: shot_groups for scenes without a model)"
                if role == "L2"
                else " (or set interpretation.embedding.text: null for no vectors)"
                if role == "embed_text"
                else ""
            )
            problems.append(f"plan: models.{role}: this run needs a model for {role}{hint}")
            continue
        try:
            spec, config, adapter, source = _lookup(router, chosen_model.model_id)
        except Exception as exc:  # an unknown model or provider, from the router
            problems.append(f"plan: models.{role}.model_id: {exc}")
            continue
        embedding = role.startswith("embed")
        binding = RoleBinding(
            role=role,
            model_id=chosen_model.model_id,
            spec=spec,
            provider=config,
            adapter=adapter,
            max_tokens=0 if embedding else (chosen_model.max_tokens or DEFAULT_MAX_TOKENS[role]),
            temperature=None
            if embedding
            else (
                chosen_model.temperature
                if chosen_model.temperature is not None
                else defaults.temperature
            ),
            seed=None if embedding else chosen_model.seed,
            reasoning=False
            if embedding
            else (
                chosen_model.reasoning if chosen_model.reasoning is not None else defaults.reasoning
            ),
            source=source,
        )
        roles[role] = binding
        if embedding and not _serves_embeddings(adapter):
            problems.append(
                f"plan: models.{role}: {adapter.__name__} (provider {config.name}) serves no "
                "embeddings"
            )
        if role in frame_roles and role in frames.stages:
            if not spec.vision:
                problems.append(
                    f"plan: models.{role}: {spec.model_id} doesn't take images (its catalog "
                    "entry has vision=False), and this role sends frames"
                )
            else:
                caps = ModelVisionCaps(
                    max_images=spec.max_images, max_image_side_px=spec.max_image_side_px
                )
                problems += [
                    f"plan: observation.frames.{p} (model {spec.model_id})"
                    for p in check_against_model(frames.for_stage(role), caps)
                ]

    # --- governance ------------------------------------------------------------------
    rights = corpus.profile.rights_mode
    governance = policy.governance
    model_inputs = governance.model_inputs
    if rights == "byo_media":
        for setting in ("image_vectors", "training_use"):
            if getattr(governance, setting) != "local_only":
                problems.append(
                    f"plan: policy.governance.{setting}: a byo_media corpus keeps this local_only "
                    "(a plan can only make governance stricter)"
                )
        if model_inputs is None and any(r in roles for r in frame_roles):
            problems.append(
                "plan: policy.governance.model_inputs: required for a byo_media corpus, because "
                "this "
                "run sends frames to models ("
                + ", ".join(r for r in frame_roles if r in roles)
                + "). Set provider_transient to allow sending frames to the model provider "
                "for inference, or local_only to allow only models on this machine or "
                "network (DESIGN.md §8.1; settings.md §8)"
            )
    if model_inputs is None:
        model_inputs = "provider_transient"
    if model_inputs == "local_only":
        for role in frame_roles:
            binding = roles.get(role)
            if binding is not None and not _is_local(binding.provider):
                problems.append(
                    f"plan: policy.governance.model_inputs is local_only, but models.{role} "
                    f"({binding.spec.model_id}) is served by {binding.provider.name}, which "
                    "isn't a local or private address"
                )

    if problems:
        raise ConfigError(problems, WHAT)

    # --- the resolved plan, recorded ----------------------------------------------------
    resolved = plan.model_dump(mode="json")
    resolved["observation"]["frames"] = frames.mapping
    resolved["interpretation"]["perception"]["modules"] = list(modules)
    resolved["structure"]["beats"] = {"min_s": beat_min, "max_s": beat_max}
    if windows is not None:
        resolved["structure"]["segmentation"]["fixed_windows"] = {
            "window_s": windows[0],
            "stride_s": windows[1],
        }
    resolved["interpretation"]["lens_passes"]["lenses"] = [
        {"name": s.name, "version": s.version, "digest": s.digest} for s in lenses
    ]
    resolved["interpretation"]["lens_passes"]["phrasings"]["style"] = (
        {"examples": list(style_examples)} if style_examples else "from_lens"
    )
    resolved["policy"]["governance"]["model_inputs"] = model_inputs
    resolved["models"] = {role: b.identity() for role, b in roles.items()}
    resolved["scope"] = {"units": chosen}
    content = _mask(resolved, NOT_CONTENT)
    return ResolvedRun(
        space=space,
        corpus=corpus,
        plan=plan,
        lenses=lenses,
        modules=modules,
        beats=(beat_min, beat_max),
        windows=windows,
        style_examples=style_examples,
        units=tuple(units),
        roles=roles,
        frames=frames,
        derivations=tuple(derivations),
        resolved_plan=resolved,
        plan_digest=digest(resolved),
        content_digest=digest(content),
        fairness_digest=digest(_mask(content, DERIVED_PATHS)),
        model_inputs=model_inputs,
        component_digests=plan.component_digests(),
        notes=tuple(notes),
    )


__all__ = ["DERIVED_PATHS", "ResolvedRun", "RoleBinding", "ScopedUnit", "resolve_run"]
