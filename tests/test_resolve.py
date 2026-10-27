"""Resolution (docs/settings.md §5.1): derivation rules, every check before any spend, digests.
No media and no model calls."""

import copy

import pytest
from extraction_helpers import COLOUR_LENS, PACE_LENS, base_plan, merged
from model_helpers import sources

from framekit.extraction import ConfigError, Corpus, ExtractionPlan, LensSpec, SearchSpace
from framekit.extraction.resolve import resolve_run
from framekit.models import ConversationRouter, ModelSettings


def lens(data):
    data = copy.deepcopy(data)
    data["lens"]["prompts"]["interpretation"] = "x"
    return LensSpec.from_mapping(data)


def space(*lenses, **extra):
    return SearchSpace.from_mapping(
        {
            "lenses": [{"spec": spec, "version": spec.version} for spec in lenses],
            "resolution": {"moment_length_s": {"min": 3, "max": 6}},
            **extra,
        }
    )


def corpus(rights="byo_media", **profile):
    return Corpus.from_mapping(
        {
            "profile": {"name": "demo", "rights_mode": rights, **profile},
            "units": [{"unit_id": "u1", "path": "a.mp4"}, {"unit_id": "u2", "path": "b.mp4"}],
            "manifest": [{"unit_id": "u1", "excluded_ranges": [[0, 2]]}],
        }
    )


def plan(**overrides):
    return ExtractionPlan.from_mapping(merged(base_plan(), overrides))


def router():
    return ConversationRouter(ModelSettings(), sources=sources())


def resolve(space_, corpus_=None, plan_=None):
    return resolve_run(space_, corpus_ or corpus(), plan_ or plan(), router=router())


def problems_of(*args):
    with pytest.raises(ConfigError) as info:
        resolve(*args)
    return "\n".join(info.value.problems)


def test_derivation_rules_fill_the_plan_and_are_recorded():
    run = resolve(space(lens(PACE_LENS), evidence={"modules": ["sound"]}))
    assert run.modules == ("core", "timing", "sound")
    assert run.beats == (0.5, 6.0)
    rules = {d["rule"] for d in run.derivations}
    assert rules == {"scope", "modules", "beats", "lenses", "phrasing_style"}
    resolved = run.resolved_plan
    assert resolved["interpretation"]["perception"]["modules"] == ["core", "timing", "sound"]
    assert resolved["structure"]["beats"] == {"min_s": 0.5, "max_s": 6.0}
    assert {d["rule"]: d["target"] for d in run.derivations}["beats"] == "structure.beats"
    assert run.component_digests == plan().component_digests()  # as given, recorded
    detail = resolved["observation"]["frames"]["profiles"]["detail"]
    assert detail["frame_selection"]["candidates"]["strategy"] == "uniform@1"  # pinned


def test_a_plan_may_add_modules_but_never_drop_one():
    camera = {"perception": {"modules": ["camera"]}}
    run = resolve(space(lens(PACE_LENS)), None, plan(interpretation=camera))
    assert run.modules == ("core", "camera", "timing")


def test_scope_narrows_and_excluded_ranges_come_from_the_corpus():
    run = resolve(space(lens(COLOUR_LENS), scope={"units": {"exclude": ["u2"]}}))
    assert [u.unit_id for u in run.units] == ["u1"]
    assert run.units[0].excluded_ranges == ((0.0, 2.0),)
    text = problems_of(space(lens(COLOUR_LENS), scope={"units": {"include": ["u9"]}}))
    assert "'u9' is not in the corpus" in text


def test_beats_longer_than_the_finest_moment_are_refused():
    text = problems_of(space(lens(COLOUR_LENS)), None, plan(structure={"beats": {"max_s": 9}}))
    assert "structure.beats.max_s: 9 s is longer than the search space's longest" in text


def test_every_role_the_run_uses_needs_a_known_model():
    data = base_plan()
    data["models"] = {"L3a": {"model_id": "nope"}}
    text = problems_of(space(lens(COLOUR_LENS)), None, ExtractionPlan.from_mapping(data))
    assert "models.L2: this run needs a model for L2 (or set structure.segmentation.method" in text
    assert "models.L3a.model_id" in text and "nope" in text
    assert "models.L3b: this run needs a model" in text
    assert "models.embed_text: this run needs a model for embed_text (or set interpretation" in text


def test_frame_sending_roles_need_a_vision_model_within_its_limits():
    models = dict(base_plan()["models"], L3a={"model_id": "writer"})
    text = problems_of(space(lens(COLOUR_LENS)), None, plan(models=models))
    assert "writer-1 doesn't take images" in text
    profiles = base_plan()["observation"]["frames"]["profiles"]
    profiles["detail"]["frame_selection"]["budget"]["max_frames_per_scene"] = 80
    frames = {"frames": {"profiles": profiles}}
    text = problems_of(space(lens(COLOUR_LENS)), None, plan(observation=frames))
    assert "observation.frames.profiles.detail" in text
    assert "the model accepts at most 50" in text


def test_native_video_is_refused_until_built():
    profiles = base_plan()["observation"]["frames"]["profiles"]
    profiles["overview"] = {
        "version": 1,
        "frame_selection": {"candidates": {"strategy": "uniform", "fps": 1}},
        "image": {"max_side_px": 160},
        "delivery": {"mode": "native_video"},
    }
    data = base_plan()
    data["observation"]["frames"]["profiles"] = profiles  # replaced, not merged
    text = problems_of(space(lens(COLOUR_LENS)), None, ExtractionPlan.from_mapping(data))
    assert "observation.frames.profiles.overview.delivery.mode: native video needs" in text


def test_speech_needs_a_speech_detector():
    text = problems_of(space(lens(COLOUR_LENS), evidence={"modules": ["speech"]}))
    assert "speech needs a speech detector" in text


def test_local_only_frames_refuse_a_remote_provider():
    models = dict(base_plan()["models"], L3a={"model_id": "claude-haiku-4-5"})
    text = problems_of(space(lens(COLOUR_LENS)), None, plan(models=models))
    assert "policy.governance.model_inputs is local_only, but models.L3a" in text


def test_a_model_that_takes_no_temperature_leaves_the_default_out():
    models = dict(base_plan()["models"], L3a={"model_id": "strict"})
    run = resolve(space(lens(COLOUR_LENS)), None, plan(models=models))
    assert run.roles["L3a"].temperature is None and run.roles["L3b"].temperature == 0.2
    assert (
        "models.L3a: strict-1 takes no temperature, so models.defaults.temperature isn't applied"
        in run.notes
    )


def test_a_temperature_for_a_model_that_takes_none_is_refused():
    models = dict(base_plan()["models"], L3a={"model_id": "strict", "temperature": 0.5})
    text = problems_of(space(lens(COLOUR_LENS)), None, plan(models=models))
    assert "plan: models.L3a.temperature: strict-1 takes no temperature" in text


def test_byo_media_keeps_vectors_and_training_local():
    text = problems_of(
        space(lens(COLOUR_LENS)),
        None,
        plan(policy={"governance": {"model_inputs": "local_only", "training_use": "per_licence"}}),
    )
    assert "policy.governance.training_use: a byo_media corpus keeps this local_only" in text


def test_a_hosted_corpus_may_send_frames_by_default():
    unset = {"governance": {"model_inputs": None}}
    run = resolve(space(lens(COLOUR_LENS)), corpus("hosted"), plan(policy=unset))
    assert run.model_inputs == "provider_transient"


def test_lens_only_differences_keep_the_fairness_digest():
    one = resolve(space(lens(COLOUR_LENS)))
    two = resolve(space(lens(COLOUR_LENS), lens(PACE_LENS)))
    assert one.plan_digest != two.plan_digest  # derived modules and lenses differ...
    assert one.fairness_digest == two.fairness_digest  # ...and only those
    other = resolve(
        space(lens(COLOUR_LENS)), None, plan(observation={"normalize": {"fps": 6}})
    )
    assert other.fairness_digest != one.fairness_digest
    budget = resolve(space(lens(COLOUR_LENS)), None, plan(policy={"budget": {"max_cost_usd": 3}}))
    assert budget.content_digest == one.content_digest  # a budget changes nothing computed


def test_unknown_detectors_are_refused():
    text = problems_of(
        space(lens(COLOUR_LENS)),
        None,
        plan(observation={"signals": {"shots": {"detector": "transnet"}}}),
    )
    assert "observation.signals.shots.detector: no shots detector 'transnet'" in text


def test_embeddings_need_an_adapter_that_serves_them():
    models = dict(base_plan()["models"], embed_text={"model_id": "claude-haiku-4-5"})
    text = problems_of(space(lens(COLOUR_LENS)), None, plan(models=models))
    assert "AnthropicAdapter (provider anthropic) serves no embeddings" in text


def test_image_embeddings_are_refused_until_built():
    image = {"embedding": {"image": {}}}
    text = problems_of(space(lens(COLOUR_LENS)), None, plan(interpretation=image))
    assert "image embeddings need an embed_image port" in text


# --- levels (policy.levels) --------------------------------------------------------------


def test_a_run_without_a_search_space_stops_where_the_plan_says():
    run = resolve(None, None, plan(policy={"levels": ["L0", "L1", "L2"]}))
    assert run.levels == ("L0", "L1", "L2") and run.read_only == frozenset()
    assert set(run.roles) == {"L2"}  # no model for levels the run doesn't reach
    assert run.space is None and run.space_digest is None
    assert run.lenses == () and run.beats is None and run.modules == ("core",)
    assert [u.unit_id for u in run.units] == ["u1", "u2"]  # the corpus, unnarrowed
    assert run.units[0].excluded_ranges == ((0, 2),)  # still from the corpus
    assert run.computes("L2") and not run.computes("L3a")


def test_listed_levels_only_read_the_earlier_levels_they_need():
    run = resolve(space(lens(PACE_LENS)), None, plan(policy={"levels": ["L3b"]}))
    assert run.levels == ("L0", "L1", "L2", "L3a", "L3b")  # embed isn't reached
    assert run.read_only == {"L0", "L1", "L2", "L3a"}
    assert set(run.roles) == {"L2", "L3a", "L3b"}  # read-only roles: their hashes name them
    assert run.computes_role("L3b") and not run.computes_role("L3a")
    assert any("L0, L1, L2, L3a only read from the task store" in n for n in run.notes)


def test_only_levels_that_run_send_frames():
    no_policy = {"governance": {"model_inputs": None}}
    reading = resolve(space(lens(PACE_LENS)), None, plan(policy={**no_policy, "levels": ["L3b"]}))
    assert reading.model_inputs == "provider_transient"  # nothing is sent: nothing to decide
    found = problems_of(space(lens(PACE_LENS)), None, plan(policy=no_policy))
    assert "model_inputs: required" in found


def test_without_a_search_space_what_needs_it_must_come_from_the_plan():
    found = problems_of(None, None, plan())
    assert "L3b (lens passes) needs a search space" in found
    assert "structure.beats.max_s: L3a writes beats" in found
    windows = {"segmentation": {"mode": "both"}}
    found = problems_of(None, None, plan(structure=windows, policy={"levels": ["L2"]}))
    assert "fixed_windows.window_s: with no search space" in found

    pinned = plan(
        structure={"beats": {"max_s": 4}},
        interpretation={"perception": {"modules": ["timing", "sound"]}},
        policy={"levels": ["L0", "L1", "L2", "L3a"]},
    )
    run = resolve(None, None, pinned)
    assert run.beats == (0.5, 4) and run.modules == ("core", "timing", "sound")
    assert set(run.roles) == {"L2", "L3a"}


def test_levels_are_checked_when_the_plan_loads():
    for levels, problem in (
        ([], "policy.levels: list at least one level"),
        (["L2", "L2"], "policy.levels: a level is listed twice"),
        (["L5"], "policy.levels"),
    ):
        with pytest.raises(ConfigError) as info:
            plan(policy={"levels": levels})
        assert any(problem in p for p in info.value.problems)
