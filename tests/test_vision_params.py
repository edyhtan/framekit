"""Vision parameters: schema, load-time checks, resolution and digests, outside checks,
and the strategy registry (docs/DESIGN.md §2.3, docs/settings.md §5.2)."""

import json

import pytest
from pydantic import BaseModel, ValidationError
from vision_examples import OLD_NATIVE_VIDEO_EXAMPLE, params_with, profile, settings_example

from framekit.vision import (
    ModelVisionCaps,
    SourceLimits,
    Strategy,
    UnknownStrategyError,
    VisionParams,
    VisionParamsError,
    VisionProfile,
    check_against_model,
    check_against_source,
    get_strategy,
    register_strategy,
    resolve,
    resolve_profile,
    strategies,
)
from framekit.vision.builtin import NoParams


def problems_of(data):
    with pytest.raises(VisionParamsError) as caught:
        VisionParams.from_mapping(data)
    return list(caught.value.problems)


def problems_of_profile(**parts):
    return problems_of(params_with(p=profile(**parts)))


# --- the documented example -------------------------------------------------------------


def test_the_settings_reference_example_parses_validates_and_resolves():
    params = VisionParams.from_mapping(settings_example())
    resolved = resolve(params)

    assert set(resolved.profiles) == {"scene_overview", "scene_detail", "keyframes"}
    assert resolved.stages == {
        "L2": "scene_overview",
        "L3a": "scene_detail",
        "embed_image": "keyframes",
    }
    detail = resolved.for_stage("L3a").mapping
    assert detail["frame_selection"]["candidates"] == {"strategy": "uniform@1", "fps": 4.0}
    assert detail["frame_selection"]["anchors"] == ["shot_boundaries@1"]
    assert [next(iter(f)) for f in detail["frame_selection"]["filters"]] == [
        "dedupe@1",
        "sharpness@1",
    ]
    assert detail["frame_selection"]["budget"]["allocate_by"] == [
        {"audio_onsets@1": {"window_s": 1.0}},
        {"motion@1": {"window_s": 1.0}},
    ]
    assert len(resolved.digest) == 64
    assert params.profile_for("L2").delivery.mode == "native_video"


def test_resolution_fills_defaults_the_example_leaves_out():
    resolved = resolve(settings_example())

    keyframes = resolved.profiles["keyframes"].mapping
    assert keyframes["delivery"] == {"mode": "images"}
    overview = resolved.profiles["scene_overview"].mapping
    assert overview["image"] == {"max_side_px": 512, "aspect": "preserve"}  # native video
    plain = resolve(params_with(p=profile())).profiles["p"].mapping
    assert plain["image"] == {"max_side_px": 512, "aspect": "preserve", "format": "jpeg",
                              "quality": 85}
    board = resolve(params_with(p=profile(delivery={"mode": "storyboard"}))).profiles["p"]
    assert board.mapping["delivery"] == {"mode": "storyboard",
                                         "storyboard": {"rows": 2, "cols": 2}}


def test_the_old_native_video_example_fails_with_a_precise_message():
    problems = problems_of(OLD_NATIVE_VIDEO_EXAMPLE)

    prefix = "vision_profiles.scene_overview."
    assert [p.split(":")[0] for p in problems] == [
        prefix + "frame_selection.filters",
        prefix + "frame_selection.budget",
        prefix + "image.format",
        prefix + "image.quality",
    ]
    assert all("not allowed with native_video delivery" in p for p in problems)
    assert "only uniform candidates, image.max_side_px and image.aspect apply" in problems[0]


def test_the_error_message_lists_every_problem():
    with pytest.raises(VisionParamsError) as caught:
        VisionParams.from_mapping(OLD_NATIVE_VIDEO_EXAMPLE)

    message = str(caught.value)
    assert message.startswith("4 problems in vision parameters:")
    assert message.count("\n- ") == 4
    assert isinstance(caught.value, ValueError)


# --- strategy references ----------------------------------------------------------------


@pytest.mark.parametrize(
    "anchor",
    ["shot_boundaries", "shot_boundaries@1", {"shot_boundaries": {}}, {"shot_boundaries": None},
     {"strategy": "shot_boundaries"}],
)
def test_every_documented_strategy_shape_means_the_same(anchor):
    data = params_with(p=profile(frame_selection={
        "candidates": {"uniform@1": {"fps": 2}}, "anchors": [anchor]}))
    reference = params_with(p=profile(frame_selection={
        "candidates": {"strategy": "uniform", "fps": 2}, "anchors": ["shot_boundaries"]}))

    assert resolve(data).digest == resolve(reference).digest


@pytest.mark.parametrize("bad", [42, ["uniform"], {"a": {}, "b": {}}, "Uniform", "uniform@0",
                                 {"uniform": 3}])
def test_a_malformed_strategy_reference_fails(bad):
    problems = problems_of_profile(frame_selection={"candidates": bad})

    assert len(problems) == 1
    assert problems[0].startswith("vision_profiles.p.frame_selection.candidates")


def test_an_unknown_strategy_fails_with_the_known_names():
    (problem,) = problems_of_profile(frame_selection={"candidates": {"strategy": "unifrom"}})

    assert problem == (
        "vision_profiles.p.frame_selection.candidates: unknown candidates strategy 'unifrom' "
        "(known: shot_based, uniform)"
    )


def test_an_unknown_strategy_version_fails():
    (problem,) = problems_of_profile(frame_selection={
        "candidates": {"strategy": "uniform@9", "fps": 2}})

    assert "candidates strategy uniform@9 is not registered (versions: 1)" in problem


def test_a_strategy_in_the_wrong_slot_fails():
    problems = problems_of_profile(frame_selection={
        "candidates": {"strategy": "dedupe", "max_distance": 4},
        "anchors": ["motion"],
        "filters": ["shot_boundaries"],
    })

    assert problems == [
        "vision_profiles.p.frame_selection.candidates: 'dedupe' is a filter, not a candidates "
        "strategy (known: shot_based, uniform)",
        "vision_profiles.p.frame_selection.anchors[0]: 'motion' is an allocation signal, not "
        "an anchor (known: audio_onsets, shot_boundaries)",
        "vision_profiles.p.frame_selection.filters[0]: 'shot_boundaries' is an anchor, not a "
        "filter (known: dedupe, sharpness)",
    ]


def test_strategy_settings_are_checked_against_the_strategy():
    problems = problems_of_profile(frame_selection={
        "candidates": {"strategy": "shot_based"},
        "filters": [{"dedupe": {"max_distance": -1, "methd": "phash"}}],
        "budget": {"allocate_by": [{"motion": {"window_s": 0}}]},
    })

    assert problems == [
        "vision_profiles.p.frame_selection.candidates.pick: Field required (shot_based@1)",
        "vision_profiles.p.frame_selection.filters[0].dedupe.max_distance: Input should be "
        "greater than or equal to 0 (dedupe@1)",
        "vision_profiles.p.frame_selection.filters[0].dedupe.methd: Extra inputs are not "
        "permitted (dedupe@1)",
        "vision_profiles.p.frame_selection.budget.allocate_by[0].motion.window_s: Input "
        "should be greater than 0 (motion@1)",
    ]


def test_a_missing_required_setting_fails():
    (problem,) = problems_of_profile(frame_selection={"candidates": "uniform"})

    assert problem == (
        "vision_profiles.p.frame_selection.candidates.fps: Field required (uniform@1)"
    )


def test_anchors_and_allocation_signals_may_not_repeat():
    problems = problems_of_profile(frame_selection={
        "candidates": {"strategy": "uniform", "fps": 2},
        "anchors": ["shot_boundaries", "shot_boundaries@1"],
        "budget": {"allocate_by": ["motion", "motion"]},
    })

    assert problems == [
        "vision_profiles.p.frame_selection.anchors[1]: anchor 'shot_boundaries' is listed twice",
        "vision_profiles.p.frame_selection.budget.allocate_by[1]: allocation signal 'motion' "
        "is listed twice",
    ]


def test_filters_may_repeat_and_keep_their_order():
    data = params_with(p=profile(frame_selection={
        "candidates": {"strategy": "uniform", "fps": 2},
        "filters": [{"sharpness": {"window_s": 1}}, {"dedupe": {"max_distance": 4}},
                    {"sharpness": {"window_s": 0.5}}],
    }))

    filters = resolve(data).profiles["p"].mapping["frame_selection"]["filters"]
    assert [next(iter(f)) for f in filters] == ["sharpness@1", "dedupe@1", "sharpness@1"]


# --- structure ----------------------------------------------------------------------------


def test_extra_keys_are_rejected_everywhere():
    data = settings_example()
    data["profiles"] = {}
    detail = data["vision_profiles"]["scene_detail"]
    detail["image"]["qualty"] = 90
    detail["frame_selection"]["anchor"] = ["shot_boundaries"]
    detail["views"]["detail"]["spans"] = 2
    data["stages"]["L2"]["model"] = "x"

    problems = problems_of(data)

    assert sorted(p.split(":")[0] for p in problems) == [
        "profiles",
        "stages.L2.model",
        "vision_profiles.scene_detail.frame_selection.anchor",
        "vision_profiles.scene_detail.image.qualty",
        "vision_profiles.scene_detail.views.detail.spans",
    ]
    assert all(p.endswith("Extra inputs are not permitted") for p in problems)


def test_one_error_lists_problems_from_every_profile_and_stage():
    data = settings_example()
    data["vision_profiles"]["keyframes"]["image"]["max_side_px"] = 0  # structural
    data["vision_profiles"]["scene_overview"]["frame_selection"]["anchors"] = ["shot_boundaries"]
    data["stages"]["L3b"] = {"vision": "missing"}

    problems = problems_of(data)

    assert [p.split(":")[0] for p in problems] == [
        "vision_profiles.scene_overview.frame_selection.anchors",
        "vision_profiles.keyframes.image.max_side_px",
        "stages.L3b.vision",
    ]


def test_aspect_is_always_preserve():
    (problem,) = problems_of_profile(image={"max_side_px": 512, "aspect": "crop"})

    assert problem == (
        "vision_profiles.p.image.aspect: aspect is always 'preserve' (frames are never "
        "cropped or stretched), got 'crop'"
    )
    assert VisionParams.from_mapping(params_with(p=profile(
        image={"max_side_px": 512, "aspect": "preserve"})))


def test_png_takes_no_quality():
    (problem,) = problems_of_profile(image={"max_side_px": 512, "format": "png", "quality": 90})

    assert problem == (
        "vision_profiles.p.image.quality: png is lossless; quality applies to jpeg and webp"
    )
    assert VisionParams.from_mapping(params_with(p=profile(
        image={"max_side_px": 512, "format": "webp", "quality": 90})))


def test_a_storyboard_grid_needs_storyboard_delivery():
    (problem,) = problems_of_profile(delivery={"mode": "images", "storyboard": {"rows": 3}})

    assert problem == (
        "vision_profiles.p.delivery.storyboard: a grid applies only to mode storyboard"
    )
    assert VisionParams.from_mapping(params_with(p=profile(
        delivery={"mode": "storyboard", "storyboard": {"rows": 3, "cols": 4}})))


def test_the_input_must_be_a_mapping():
    assert problems_of(["vision_profiles"]) == [
        "(root): Input should be a valid dictionary or instance of VisionParams"
    ]


def test_a_single_profile_can_be_loaded_on_its_own():
    loaded = VisionProfile.from_mapping(profile())
    assert loaded.frame_selection.candidates.params == {"fps": 2}

    with pytest.raises(VisionParamsError) as caught:
        VisionProfile.from_mapping(profile(image={"max_side_px": 512, "format": "gif"}))
    assert caught.value.problems == ("image.format: Input should be 'jpeg', 'png' or 'webp'",)


def test_direct_construction_runs_the_same_checks():
    with pytest.raises(ValidationError, match="not allowed with native_video"):
        VisionProfile.model_validate(OLD_NATIVE_VIDEO_EXAMPLE["vision_profiles"]["scene_overview"])


def test_to_mapping_round_trips():
    params = VisionParams.from_mapping(settings_example())
    again = VisionParams.from_mapping(params.to_mapping())

    assert again == params
    detail = params.to_mapping()["vision_profiles"]["scene_detail"]["frame_selection"]
    assert detail["candidates"] == {"strategy": "uniform", "fps": 4.0}
    assert detail["anchors"] == ["shot_boundaries"]


# --- native video -----------------------------------------------------------------------


def test_native_video_accepts_uniform_candidates_and_a_size():
    native = profile(image={"max_side_px": 512, "aspect": "preserve"},
                     delivery={"mode": "native_video"})
    assert VisionParams.from_mapping(params_with(p=native))


def test_native_video_rejects_other_candidates_anchors_and_views():
    problems = problems_of_profile(
        frame_selection={"candidates": {"strategy": "shot_based", "pick": "first"},
                         "anchors": ["shot_boundaries"],
                         "budget": {"allocate_by": ["motion"]}},
        delivery={"mode": "native_video"},
        views={"mode": "single"},
    )

    assert [p.split(":")[0] for p in problems] == [
        "vision_profiles.p.frame_selection.candidates",
        "vision_profiles.p.frame_selection.anchors",
        "vision_profiles.p.frame_selection.budget",
        "vision_profiles.p.views",
    ]
    assert problems[0].startswith(
        "vision_profiles.p.frame_selection.candidates: 'shot_based' not allowed with native_video"
    )


# --- views --------------------------------------------------------------------------------


def views(overview_frames=6, max_views=2, detail_frames=5, mode="overview_plus_detail"):
    return {
        "mode": mode,
        "overview": {"fps": 1, "max_frames": overview_frames},
        "detail": {"around": ["motion"], "span_s": 3, "max_views": max_views,
                   "max_frames": detail_frames},
    }


def capped(cap):
    return {"candidates": {"strategy": "uniform", "fps": 4},
            "budget": {"max_frames_per_scene": cap}}


def test_views_must_fit_in_the_scene_budget():
    (problem,) = problems_of_profile(views=views(6, 2, 6), frame_selection=capped(16))

    assert problem == (
        "vision_profiles.p.views: overview (6) + detail (2 x 6) = 18 frames, more than "
        "frame_selection.budget.max_frames_per_scene (16)"
    )


def test_views_that_fill_the_budget_exactly_are_accepted():
    assert VisionParams.from_mapping(params_with(p=profile(
        views=views(6, 2, 5), frame_selection=capped(16))))


def test_views_without_a_scene_cap_are_not_summed():
    assert VisionParams.from_mapping(params_with(p=profile(views=views(60, 10, 50))))


def test_overview_plus_detail_needs_both_parts():
    problems = problems_of_profile(views={"mode": "overview_plus_detail"})

    assert problems == [
        "vision_profiles.p.views.overview: required for mode overview_plus_detail",
        "vision_profiles.p.views.detail: required for mode overview_plus_detail",
    ]


def test_other_view_modes_take_no_overview_or_detail():
    problems = problems_of_profile(views=views(mode="single"))

    assert problems == [
        "vision_profiles.p.views.overview: applies only to mode overview_plus_detail",
        "vision_profiles.p.views.detail: applies only to mode overview_plus_detail",
    ]
    for mode in ("single", "split_by_budget", "per_module"):
        assert VisionParams.from_mapping(params_with(p=profile(views={"mode": mode})))


def test_detail_signals_are_checked_like_allocation_signals():
    bad = views()
    bad["detail"]["around"] = ["shot_boundaries"]

    (problem,) = problems_of_profile(views=bad)

    assert problem.startswith(
        "vision_profiles.p.views.detail.around[0]: 'shot_boundaries' is an anchor, not an "
        "allocation signal"
    )


# --- stages -------------------------------------------------------------------------------


def test_stages_must_name_an_existing_profile():
    data = settings_example()
    data["stages"]["L3a"] = {"vision": "scene_detial"}

    assert problems_of(data) == [
        "stages.L3a.vision: no vision profile named 'scene_detial' "
        "(profiles: keyframes, scene_detail, scene_overview)"
    ]


def test_unused_profiles_are_allowed():
    data = settings_example()
    del data["stages"]["embed_image"]

    resolved = resolve(VisionParams.from_mapping(data))

    assert "keyframes" in resolved.profiles
    with pytest.raises(KeyError):
        resolved.for_stage("embed_image")


# --- digests ------------------------------------------------------------------------------


def test_the_same_input_gives_the_same_digest():
    assert resolve(settings_example()).digest == resolve(settings_example()).digest


def test_key_order_and_spelling_do_not_change_the_digest():
    reordered = settings_example()
    reordered = {"stages": dict(reversed(reordered["stages"].items())),
                 "vision_profiles": dict(reversed(reordered["vision_profiles"].items()))}
    detail = reordered["vision_profiles"]["scene_detail"]
    reordered["vision_profiles"]["scene_detail"] = dict(reversed(detail.items()))
    detail = reordered["vision_profiles"]["scene_detail"]
    detail["frame_selection"]["candidates"] = {"fps": 4.0, "strategy": "uniform@1"}
    detail["frame_selection"]["budget"]["allocate_by"] = [
        {"audio_onsets": {"window_s": 1}}, "motion@1"]
    detail["image"] = {"quality": 85, "format": "jpeg", "aspect": "preserve", "max_side_px": 768}

    assert resolve(reordered).digest == resolve(settings_example()).digest


def test_a_changed_setting_changes_the_digest_of_its_profile_only():
    changed = settings_example()
    changed["vision_profiles"]["scene_detail"]["frame_selection"]["budget"]["min_gap_s"] = 0.3

    before, after = resolve(settings_example()), resolve(changed)

    assert after.digest != before.digest
    assert after.profiles["scene_detail"].digest != before.profiles["scene_detail"].digest
    assert after.profiles["keyframes"].digest == before.profiles["keyframes"].digest


def test_filter_order_is_part_of_the_digest():
    swapped = settings_example()
    filters = swapped["vision_profiles"]["scene_detail"]["frame_selection"]["filters"]
    filters.reverse()

    assert resolve(swapped).digest != resolve(settings_example()).digest


def test_a_profile_digest_ignores_the_profile_name():
    data = settings_example()
    data["vision_profiles"]["renamed"] = data["vision_profiles"].pop("keyframes")
    data["stages"]["embed_image"] = {"vision": "renamed"}

    renamed, original = resolve(data), resolve(settings_example())

    assert renamed.profiles["renamed"].digest == original.profiles["keyframes"].digest
    assert renamed.digest != original.digest  # the block records the names


def test_the_resolved_form_loads_back_to_the_same_digest():
    resolved = resolve(settings_example())

    again = resolve(VisionParams.from_mapping(resolved.to_mapping()))

    assert again.digest == resolved.digest
    assert again.mapping == resolved.mapping


def test_the_canonical_json_is_sorted_and_has_no_nulls():
    text = resolve(settings_example()).canonical_json()

    assert "null" not in text
    assert ": " not in text and ", " not in text
    assert text == json.dumps(json.loads(text), sort_keys=True, separators=(",", ":"))


def test_a_profile_resolves_on_its_own_with_the_same_digest():
    params = VisionParams.from_mapping(settings_example())

    alone = resolve_profile(params.vision_profiles["scene_detail"])

    assert alone.digest == resolve(params).profiles["scene_detail"].digest


# --- the analysis copy ----------------------------------------------------------------------


def test_the_source_check_passes_when_the_copy_is_large_enough():
    copy = SourceLimits(fps=12, width_px=1280, height_px=720)

    assert check_against_source(VisionParams.from_mapping(settings_example()), copy) == []


def test_the_source_check_rejects_more_fps_or_pixels_than_the_copy_has():
    copy = SourceLimits(fps=3, width_px=480, height_px=720)  # portrait: long side 720

    problems = check_against_source(resolve(settings_example()), copy)

    assert problems == [
        "vision_profiles.scene_detail.frame_selection.candidates.fps: 4 fps is more than "
        "the analysis copy's 3 fps",
        "vision_profiles.scene_detail.image.max_side_px: 768 px is more than the analysis "
        "copy's long side (720 px)",
    ]


def test_the_source_check_covers_the_overview_fps():
    data = profile(views=views(), frame_selection=capped(16))
    data["views"]["overview"]["fps"] = 30

    problems = check_against_source(VisionProfile.from_mapping(data),
                                    SourceLimits(fps=12, width_px=960, height_px=720))

    assert problems == ["views.overview.fps: 30 fps is more than the analysis copy's 12 fps"]


def test_source_limits_must_be_positive():
    with pytest.raises(ValueError, match="must be > 0"):
        SourceLimits(fps=0, width_px=960, height_px=720)


# --- the model ------------------------------------------------------------------------------


def test_the_model_check_passes_for_a_capable_model():
    resolved = resolve(settings_example())
    caps = ModelVisionCaps(native_video=True, max_images=20, max_image_side_px=1024)

    for name in resolved.profiles:
        assert check_against_model(resolved.profiles[name], caps) == []


def test_native_video_needs_a_model_that_takes_video():
    overview = resolve(settings_example()).for_stage("L2")

    assert check_against_model(overview, ModelVisionCaps(native_video=False)) == [
        "vision_profiles.scene_overview.delivery.mode: the model takes no native video; "
        "use images or storyboard delivery"
    ]


def test_more_frames_than_the_model_accepts_fail():
    detail = VisionParams.from_mapping(settings_example()).vision_profiles["scene_detail"]

    assert check_against_model(detail, ModelVisionCaps(max_images=10)) == [
        "frame_selection.budget: up to 16 images per request, but the model accepts at most 10"
    ]


def test_a_profile_without_a_frame_cap_fails_on_a_limited_model():
    assert check_against_model(VisionProfile.from_mapping(profile()),
                               ModelVisionCaps(max_images=10)) == [
        "frame_selection.budget: no frame cap, but the model accepts at most 10 images; "
        "set max_frames_per_scene"
    ]


def test_views_without_anchors_bound_the_frames_on_their_own():
    bounded = VisionProfile.from_mapping(profile(views=views(6, 2, 5)))

    assert check_against_model(bounded, ModelVisionCaps(max_images=16)) == []
    assert check_against_model(bounded, ModelVisionCaps(max_images=15)) != []


def test_a_storyboard_sends_fewer_but_larger_images():
    board = VisionProfile.from_mapping(profile(
        frame_selection=capped(16),
        delivery={"mode": "storyboard", "storyboard": {"rows": 2, "cols": 3}},
    ))

    assert check_against_model(board, ModelVisionCaps(max_images=3)) == []
    assert check_against_model(board, ModelVisionCaps(max_images=2)) == [
        "frame_selection.budget: up to 3 images (16 frames) per request, but the model "
        "accepts at most 2"
    ]
    assert check_against_model(board, ModelVisionCaps(max_image_side_px=1024)) == [
        "image.max_side_px: a storyboard grid's long side is 1536 px, but the model accepts "
        "at most 1024 px"
    ]


def test_images_larger_than_the_model_accepts_fail():
    detail = resolve(settings_example()).profiles["scene_detail"]

    assert check_against_model(detail, ModelVisionCaps(max_image_side_px=512)) == [
        "vision_profiles.scene_detail.image.max_side_px: image.max_side_px is 768 px, but "
        "the model accepts at most 512 px"
    ]


# --- the registry -------------------------------------------------------------------------


def test_the_built_in_strategies_are_registered():
    found = {(s.kind, s.key, s.needs_pixels) for s in strategies()}

    assert {
        ("candidates", "uniform@1", False),
        ("candidates", "shot_based@1", False),
        ("anchor", "shot_boundaries@1", False),
        ("anchor", "audio_onsets@1", False),
        ("filter", "dedupe@1", True),
        ("filter", "sharpness@1", True),
        ("allocation", "motion@1", False),
        ("allocation", "audio_onsets@1", False),
    } <= found
    assert all(s.kind == "filter" for s in strategies("filter"))


def test_get_strategy_takes_a_version_inline_or_apart():
    assert get_strategy("filter", "dedupe@1") is get_strategy("filter", "dedupe", 1)
    assert get_strategy("filter", "dedupe").version == 1
    with pytest.raises(UnknownStrategyError, match="not registered"):
        get_strategy("filter", "dedupe", 2)
    with pytest.raises(ValueError, match="unknown strategy kind"):
        get_strategy("candidate", "uniform")  # type: ignore[arg-type]


def test_a_registered_name_and_version_can_not_be_registered_again():
    with pytest.raises(ValueError, match="already registered"):
        register_strategy(Strategy("dedupe", 1, "filter", NoParams, needs_pixels=True))


def test_a_new_version_is_what_unversioned_references_resolve_to():
    def every_shot(params, span, signals):
        return list(signals.shot_boundaries)

    register_strategy(Strategy("test_newest", 1, "anchor", NoParams, run=every_shot))
    data = params_with(p=profile(frame_selection={
        "candidates": {"strategy": "uniform", "fps": 2}, "anchors": ["test_newest"]}))
    pinned_v1 = resolve(params_with(p=profile(frame_selection={
        "candidates": {"strategy": "uniform", "fps": 2}, "anchors": ["test_newest@1"]})))
    assert resolve(data).digest == pinned_v1.digest

    register_strategy(Strategy("test_newest", 2, "anchor", NoParams, run=every_shot))

    assert resolve(data).profiles["p"].mapping["frame_selection"]["anchors"] == ["test_newest@2"]
    assert resolve(data).digest != pinned_v1.digest  # same input, a new meaning: a new digest


@pytest.mark.parametrize(
    ("strategy", "message"),
    [
        (Strategy("Bad", 1, "anchor", NoParams, run=print), "lower_snake_case"),
        (Strategy("test_zero", 0, "anchor", NoParams, run=print), "start at 1"),
        (Strategy("test_kind", 1, "anchors", NoParams, run=print), "unknown strategy kind"),  # type: ignore[arg-type]
        (Strategy("test_norun", 1, "anchor", NoParams), "needs `run`"),
    ],
)
def test_registration_rejects_malformed_strategies(strategy, message):
    with pytest.raises(ValueError, match=message):
        register_strategy(strategy)


def test_no_setting_may_be_called_strategy():
    class Clashing(BaseModel):
        strategy: str = "x"

    with pytest.raises(ValueError, match="may not be called 'strategy'"):
        register_strategy(Strategy("test_clash", 1, "anchor", Clashing, run=print))
