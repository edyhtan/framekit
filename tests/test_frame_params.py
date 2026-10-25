"""Frame parameters: schema, load-time checks, resolution and digests, outside checks,
and the strategy registry (docs/DESIGN.md §2.3, docs/settings.md §5.2)."""

import json

import pytest
from frame_examples import OLD_NATIVE_VIDEO_EXAMPLE, params_with, profile, settings_example
from pydantic import BaseModel, ValidationError

from framekit.frames import (
    FrameParams,
    FrameParamsError,
    FrameProfile,
    ModelVisionCaps,
    SourceLimits,
    Strategy,
    UnknownStrategyError,
    check_against_model,
    check_against_source,
    get_strategy,
    register_strategy,
    resolve,
    resolve_profile,
    strategies,
)
from framekit.frames.builtin import NoParams


def problems_of(data):
    with pytest.raises(FrameParamsError) as caught:
        FrameParams.from_mapping(data)
    return list(caught.value.problems)


def problems_of_profile(**parts):
    return problems_of(params_with(p=profile(**parts)))


# --- the documented example -------------------------------------------------------------


def test_the_settings_reference_example_parses_validates_and_resolves():
    params = FrameParams.from_mapping(settings_example())
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

    prefix = "profiles.scene_overview."
    assert [p.split(":")[0] for p in problems] == [
        prefix + "frame_selection.filters",
        prefix + "frame_selection.budget",
        prefix + "image.format",
        prefix + "image.quality",
    ]
    assert all("not allowed with native_video delivery" in p for p in problems)
    assert "only uniform candidates, image.max_side_px and image.aspect apply" in problems[0]


def test_the_error_message_lists_every_problem():
    with pytest.raises(FrameParamsError) as caught:
        FrameParams.from_mapping(OLD_NATIVE_VIDEO_EXAMPLE)

    message = str(caught.value)
    assert message.startswith("4 problems in frame parameters:")
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
    assert problems[0].startswith("profiles.p.frame_selection.candidates")


def test_an_unknown_strategy_fails_with_the_known_names():
    (problem,) = problems_of_profile(frame_selection={"candidates": {"strategy": "unifrom"}})

    assert problem == (
        "profiles.p.frame_selection.candidates: unknown candidates strategy 'unifrom' "
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
        "profiles.p.frame_selection.candidates: 'dedupe' is a filter, not a candidates "
        "strategy (known: shot_based, uniform)",
        "profiles.p.frame_selection.anchors[0]: 'motion' is an allocation signal, not "
        "an anchor (known: audio_onsets, shot_boundaries)",
        "profiles.p.frame_selection.filters[0]: 'shot_boundaries' is an anchor, not a "
        "filter (known: dedupe, sharpness)",
    ]


def test_strategy_settings_are_checked_against_the_strategy():
    problems = problems_of_profile(frame_selection={
        "candidates": {"strategy": "shot_based"},
        "filters": [{"dedupe": {"max_distance": -1, "methd": "phash"}}],
        "budget": {"allocate_by": [{"motion": {"window_s": 0}}]},
    })

    assert problems == [
        "profiles.p.frame_selection.candidates.pick: Field required (shot_based@1)",
        "profiles.p.frame_selection.filters[0].dedupe.max_distance: Input should be "
        "greater than or equal to 0 (dedupe@1)",
        "profiles.p.frame_selection.filters[0].dedupe.methd: Extra inputs are not "
        "permitted (dedupe@1)",
        "profiles.p.frame_selection.budget.allocate_by[0].motion.window_s: Input "
        "should be greater than 0 (motion@1)",
    ]


def test_a_missing_required_setting_fails():
    (problem,) = problems_of_profile(frame_selection={"candidates": "uniform"})

    assert problem == (
        "profiles.p.frame_selection.candidates.fps: Field required (uniform@1)"
    )


def test_anchors_and_allocation_signals_may_not_repeat():
    problems = problems_of_profile(frame_selection={
        "candidates": {"strategy": "uniform", "fps": 2},
        "anchors": ["shot_boundaries", "shot_boundaries@1"],
        "budget": {"allocate_by": ["motion", "motion"]},
    })

    assert problems == [
        "profiles.p.frame_selection.anchors[1]: anchor 'shot_boundaries' is listed twice",
        "profiles.p.frame_selection.budget.allocate_by[1]: allocation signal 'motion' "
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
    data["profile"] = {}
    detail = data["profiles"]["scene_detail"]
    detail["image"]["qualty"] = 90
    detail["frame_selection"]["anchor"] = ["shot_boundaries"]
    detail["views"]["detail"]["spans"] = 2
    data["stages"]["L2"]["model"] = "x"

    problems = problems_of(data)

    assert sorted(p.split(":")[0] for p in problems) == [
        "profile",
        "profiles.scene_detail.frame_selection.anchor",
        "profiles.scene_detail.image.qualty",
        "profiles.scene_detail.views.detail.spans",
        "stages.L2.model",
    ]
    assert all(p.endswith("Extra inputs are not permitted") for p in problems)


def test_one_error_lists_problems_from_every_profile_and_stage():
    data = settings_example()
    data["profiles"]["keyframes"]["image"]["max_side_px"] = 0  # structural
    data["profiles"]["scene_overview"]["frame_selection"]["anchors"] = ["shot_boundaries"]
    data["stages"]["L3b"] = {"profile": "missing"}

    problems = problems_of(data)

    assert [p.split(":")[0] for p in problems] == [
        "profiles.scene_overview.frame_selection.anchors",
        "profiles.keyframes.image.max_side_px",
        "stages.L3b.profile",
    ]


def test_one_error_lists_structural_and_rule_problems_of_the_same_profile():
    problems = problems_of_profile(
        frame_selection={"candidates": "dedupe", "anchors": ["motion"],
                         "filters": [{"sharpness": {"window_s": -1}}]},
        image={"max_side_px": 0, "fromat": "jpeg", "format": "png", "quality": 3},
        delivery={"mode": "images", "storyboard": {"rows": 2, "cols": 2}},
    )

    assert problems == [
        "profiles.p.frame_selection.candidates: 'dedupe' is a filter, not a candidates "
        "strategy (known: shot_based, uniform)",
        "profiles.p.frame_selection.anchors[0]: 'motion' is an allocation signal, not "
        "an anchor (known: audio_onsets, shot_boundaries)",
        "profiles.p.frame_selection.filters[0].sharpness.window_s: Input should be "
        "greater than 0 (sharpness@1)",
        "profiles.p.image.max_side_px: Input should be greater than 0",
        "profiles.p.image.fromat: Extra inputs are not permitted",
        "profiles.p.image.quality: png is lossless; quality applies to jpeg and webp",
        "profiles.p.delivery.storyboard: a grid applies only to mode storyboard",
    ]


def test_a_typo_does_not_hide_the_native_video_rules():
    data = settings_example()
    overview = data["profiles"]["scene_overview"]
    overview["image"]["qualty"] = 80
    overview["frame_selection"]["filters"] = [{"dedupe": {"max_distance": 6}}]

    assert [p.split(":")[0] for p in problems_of(data)] == [
        "profiles.scene_overview.frame_selection.filters",
        "profiles.scene_overview.image.qualty",
    ]


def test_a_rule_is_skipped_where_a_setting_it_reads_has_a_problem_of_its_own():
    overflowing_views = views(6, 2, 5)  # 16 frames

    assert problems_of_profile(  # the views rules depend on the delivery mode
        frame_selection=capped(10), delivery={"mode": "native-video"}, views=overflowing_views,
    ) == [
        "profiles.p.delivery.mode: Input should be 'images', 'storyboard' or "
        "'native_video'"
    ]
    assert problems_of_profile(image={"max_side_px": 512, "format": "gif", "quality": 3}) == [
        "profiles.p.image.format: Input should be 'jpeg', 'png' or 'webp'"
    ]


def test_aspect_is_always_preserve():
    (problem,) = problems_of_profile(image={"max_side_px": 512, "aspect": "crop"})

    assert problem == (
        "profiles.p.image.aspect: aspect is always 'preserve' (frames are never "
        "cropped or stretched), got 'crop'"
    )
    assert FrameParams.from_mapping(params_with(p=profile(
        image={"max_side_px": 512, "aspect": "preserve"})))


def test_png_takes_no_quality():
    (problem,) = problems_of_profile(image={"max_side_px": 512, "format": "png", "quality": 90})

    assert problem == (
        "profiles.p.image.quality: png is lossless; quality applies to jpeg and webp"
    )
    assert FrameParams.from_mapping(params_with(p=profile(
        image={"max_side_px": 512, "format": "webp", "quality": 90})))


def test_a_storyboard_grid_needs_storyboard_delivery():
    (problem,) = problems_of_profile(delivery={"mode": "images", "storyboard": {"rows": 3}})

    assert problem == (
        "profiles.p.delivery.storyboard: a grid applies only to mode storyboard"
    )
    assert FrameParams.from_mapping(params_with(p=profile(
        delivery={"mode": "storyboard", "storyboard": {"rows": 3, "cols": 4}})))


def test_the_input_must_be_a_mapping():
    assert problems_of(["profiles"]) == [
        "(root): Input should be a valid dictionary or instance of FrameParams"
    ]


def test_a_single_profile_can_be_loaded_on_its_own():
    loaded = FrameProfile.from_mapping(profile())
    assert loaded.frame_selection.candidates.params == {"fps": 2}

    with pytest.raises(FrameParamsError) as caught:
        FrameProfile.from_mapping(profile(image={"max_side_px": 512, "format": "gif"}))
    assert caught.value.problems == ("image.format: Input should be 'jpeg', 'png' or 'webp'",)


def test_direct_construction_runs_the_same_checks():
    with pytest.raises(ValidationError, match="not allowed with native_video"):
        FrameProfile.model_validate(OLD_NATIVE_VIDEO_EXAMPLE["profiles"]["scene_overview"])


def test_to_mapping_round_trips():
    params = FrameParams.from_mapping(settings_example())
    again = FrameParams.from_mapping(params.to_mapping())

    assert again == params
    detail = params.to_mapping()["profiles"]["scene_detail"]["frame_selection"]
    assert detail["candidates"] == {"strategy": "uniform", "fps": 4.0}
    assert detail["anchors"] == ["shot_boundaries"]


# --- strict values ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("part", "path", "message"),
    [
        ({"version": True}, "version", "Input should be a valid integer"),
        ({"version": "1"}, "version", "Input should be a valid integer"),
        ({"image": {"max_side_px": "512"}}, "image.max_side_px", "Input should be a valid integer"),
        ({"image": {"max_side_px": 512, "quality": True}}, "image.quality",
         "Input should be a valid integer"),
        ({"frame_selection": {"candidates": {"strategy": "uniform", "fps": "4"}}},
         "frame_selection.candidates.fps", "Input should be a valid number (uniform@1)"),
        ({"frame_selection": {"candidates": {"strategy": "uniform", "fps": True}}},
         "frame_selection.candidates.fps", "Input should be a valid number (uniform@1)"),
        ({"frame_selection": {"candidates": {"strategy": "uniform", "fps": 2},
                              "filters": [{"dedupe": {"max_distance": True}}]}},
         "frame_selection.filters[0].dedupe.max_distance",
         "Input should be a valid integer (dedupe@1)"),
        ({"frame_selection": {"candidates": {"strategy": "uniform", "fps": 2},
                              "budget": {"min_gap_s": "0.25"}}},
         "frame_selection.budget.min_gap_s", "Input should be a valid number"),
    ],
)
def test_numbers_must_be_numbers_not_booleans_or_strings(part, path, message):
    assert problems_of_profile(**part) == [f"profiles.p.{path}: {message}"]


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_numbers_must_be_finite(bad):
    problems = problems_of_profile(frame_selection={
        "candidates": {"strategy": "uniform", "fps": bad}, "budget": {"min_gap_s": bad}})

    assert problems == [
        "profiles.p.frame_selection.candidates.fps: Input should be a finite number "
        "(uniform@1)",
        "profiles.p.frame_selection.budget.min_gap_s: Input should be a finite number",
    ]


def test_a_reference_keeps_its_checked_settings_and_resolution_fills_the_rest():
    params = FrameParams.from_mapping(params_with(p=profile(frame_selection={
        "candidates": {"strategy": "uniform", "fps": 4},  # a whole number for a decimal
        "filters": [{"dedupe": {"max_distance": 4}}],
    })))

    fs = params.to_mapping()["profiles"]["p"]["frame_selection"]
    assert fs["candidates"] == {"strategy": "uniform", "fps": 4.0}
    assert type(fs["candidates"]["fps"]) is float
    assert fs["filters"] == [{"dedupe": {"max_distance": 4}}]
    pinned = resolve(params).profiles["p"].mapping["frame_selection"]["filters"]
    assert pinned == [{"dedupe@1": {"max_distance": 4, "method": "phash"}}]


def test_list_settings_must_be_lists():
    assert problems_of_profile(frame_selection={
        "candidates": {"strategy": "uniform", "fps": 2}, "anchors": "shot_boundaries"}) == [
        "profiles.p.frame_selection.anchors: Input should be a valid list"
    ]


# --- read-only --------------------------------------------------------------------------------


def test_parsed_parameters_are_read_only():
    params = FrameParams.from_mapping(settings_example())
    detail = params.profiles["scene_detail"]

    with pytest.raises(TypeError, match="read-only"):
        detail.frame_selection.candidates.params["fps"] = 1
    with pytest.raises(TypeError, match="read-only"):
        params.profiles["other"] = detail
    with pytest.raises(AttributeError):
        detail.frame_selection.anchors.append("audio_onsets")  # type: ignore[attr-defined]
    with pytest.raises(ValidationError, match="frozen"):
        detail.frame_selection.budget.min_gap_s = 1.0  # type: ignore[misc]


def test_parsed_and_resolved_parameters_are_hashable():
    params = FrameParams.from_mapping(settings_example())

    assert hash(params) == hash(FrameParams.from_mapping(settings_example()))
    assert hash(resolve(params).for_stage("L3a")) == hash(resolve(params).for_stage("L3a"))
    assert len(set(params.profiles["scene_detail"].frame_selection.filters)) == 2


def test_changing_the_resolved_mapping_changes_nothing_recorded():
    resolved = resolve(settings_example())
    text, digest = resolved.canonical_json(), resolved.digest

    resolved.mapping["profiles"]["keyframes"]["version"] = 99
    detail = resolved.for_stage("L3a")
    detail.mapping["frame_selection"]["candidates"]["fps"] = 1

    assert resolved.canonical_json() == text and resolved.digest == digest
    assert detail.mapping["frame_selection"]["candidates"]["fps"] == 4.0
    with pytest.raises(TypeError, match="read-only"):
        resolved.profiles["x"] = detail  # type: ignore[index]


# --- native video -----------------------------------------------------------------------


def test_native_video_accepts_uniform_candidates_and_a_size():
    native = profile(image={"max_side_px": 512, "aspect": "preserve"},
                     delivery={"mode": "native_video"})
    assert FrameParams.from_mapping(params_with(p=native))


def test_native_video_rejects_other_candidates_anchors_and_views():
    problems = problems_of_profile(
        frame_selection={"candidates": {"strategy": "shot_based", "pick": "first"},
                         "anchors": ["shot_boundaries"],
                         "budget": {"allocate_by": ["motion"]}},
        delivery={"mode": "native_video"},
        views={"mode": "single"},
    )

    assert [p.split(":")[0] for p in problems] == [
        "profiles.p.frame_selection.candidates",
        "profiles.p.frame_selection.anchors",
        "profiles.p.frame_selection.budget",
        "profiles.p.views",
    ]
    assert problems[0].startswith(
        "profiles.p.frame_selection.candidates: 'shot_based' not allowed with native_video"
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
        "profiles.p.views: overview (6) + detail (2 x 6) = 18 frames, more than "
        "frame_selection.budget.max_frames_per_scene (16)"
    )


def test_views_must_fit_the_tighter_of_the_two_caps():
    budget = {"candidates": {"strategy": "uniform", "fps": 4},
              "budget": {"max_frames": 10, "max_frames_per_scene": 16}}

    (problem,) = problems_of_profile(views=views(6, 2, 5), frame_selection=budget)

    assert problem == (
        "profiles.p.views: overview (6) + detail (2 x 5) = 16 frames, more than "
        "frame_selection.budget.max_frames (10)"
    )


def test_detail_views_need_at_least_one_signal():
    no_signals = views()
    no_signals["detail"]["around"] = []

    assert problems_of_profile(views=no_signals) == [
        "profiles.p.views.detail.around: at least one signal is needed"
    ]


def test_a_detail_span_is_at_least_a_millisecond():
    tiny = views()
    tiny["detail"]["span_s"] = 0.0004

    assert problems_of_profile(views=tiny) == [
        "profiles.p.views.detail.span_s: Input should be greater than or equal to 0.001"
    ]


def test_views_that_fill_the_budget_exactly_are_accepted():
    assert FrameParams.from_mapping(params_with(p=profile(
        views=views(6, 2, 5), frame_selection=capped(16))))


def test_views_without_a_scene_cap_are_not_summed():
    assert FrameParams.from_mapping(params_with(p=profile(views=views(60, 10, 50))))


def test_overview_plus_detail_needs_both_parts():
    problems = problems_of_profile(views={"mode": "overview_plus_detail"})

    assert problems == [
        "profiles.p.views.overview: required for mode overview_plus_detail",
        "profiles.p.views.detail: required for mode overview_plus_detail",
    ]


def test_other_view_modes_take_no_overview_or_detail():
    problems = problems_of_profile(views=views(mode="single"))

    assert problems == [
        "profiles.p.views.overview: applies only to mode overview_plus_detail",
        "profiles.p.views.detail: applies only to mode overview_plus_detail",
    ]
    for mode in ("single", "split_by_budget", "per_module"):
        assert FrameParams.from_mapping(params_with(p=profile(views={"mode": mode})))


def test_detail_signals_are_checked_like_allocation_signals():
    bad = views()
    bad["detail"]["around"] = ["shot_boundaries"]

    (problem,) = problems_of_profile(views=bad)

    assert problem.startswith(
        "profiles.p.views.detail.around[0]: 'shot_boundaries' is an anchor, not an "
        "allocation signal"
    )


# --- stages -------------------------------------------------------------------------------


def test_stages_must_name_an_existing_profile():
    data = settings_example()
    data["stages"]["L3a"] = {"profile": "scene_detial"}

    assert problems_of(data) == [
        "stages.L3a.profile: no frame profile named 'scene_detial' "
        "(profiles: keyframes, scene_detail, scene_overview)"
    ]


def test_unused_profiles_are_allowed():
    data = settings_example()
    del data["stages"]["embed_image"]

    resolved = resolve(FrameParams.from_mapping(data))

    assert "keyframes" in resolved.profiles
    with pytest.raises(KeyError):
        resolved.for_stage("embed_image")


# --- digests ------------------------------------------------------------------------------


def test_the_same_input_gives_the_same_digest():
    assert resolve(settings_example()).digest == resolve(settings_example()).digest


def test_key_order_and_spelling_do_not_change_the_digest():
    reordered = settings_example()
    reordered = {"stages": dict(reversed(reordered["stages"].items())),
                 "profiles": dict(reversed(reordered["profiles"].items()))}
    detail = reordered["profiles"]["scene_detail"]
    reordered["profiles"]["scene_detail"] = dict(reversed(detail.items()))
    detail = reordered["profiles"]["scene_detail"]
    detail["frame_selection"]["candidates"] = {"fps": 4.0, "strategy": "uniform@1"}
    detail["frame_selection"]["budget"]["allocate_by"] = [
        {"audio_onsets": {"window_s": 1}}, "motion@1"]
    detail["image"] = {"quality": 85, "format": "jpeg", "aspect": "preserve", "max_side_px": 768}

    assert resolve(reordered).digest == resolve(settings_example()).digest


def test_the_digest_of_the_documented_example_is_pinned():
    # Manifests and stage input hashes compare digests across runs and machines. If this
    # changes, every recorded digest changes: that needs a DECISIONS entry.
    assert resolve(settings_example()).digest == (
        "b5a76d732ffec037cf5beb22dcd2cd1b22e6c627676cba44dd4ae5253ecb5532"
    )


def test_a_changed_setting_changes_the_digest_of_its_profile_only():
    changed = settings_example()
    changed["profiles"]["scene_detail"]["frame_selection"]["budget"]["min_gap_s"] = 0.3

    before, after = resolve(settings_example()), resolve(changed)

    assert after.digest != before.digest
    assert after.profiles["scene_detail"].digest != before.profiles["scene_detail"].digest
    assert after.profiles["keyframes"].digest == before.profiles["keyframes"].digest


def test_filter_order_is_part_of_the_digest():
    swapped = settings_example()
    filters = swapped["profiles"]["scene_detail"]["frame_selection"]["filters"]
    filters.reverse()

    assert resolve(swapped).digest != resolve(settings_example()).digest


def test_a_profile_digest_ignores_the_profile_name():
    data = settings_example()
    data["profiles"]["renamed"] = data["profiles"].pop("keyframes")
    data["stages"]["embed_image"] = {"profile": "renamed"}

    renamed, original = resolve(data), resolve(settings_example())

    assert renamed.profiles["renamed"].digest == original.profiles["keyframes"].digest
    assert renamed.digest != original.digest  # the block records the names


def test_the_resolved_form_loads_back_to_the_same_digest():
    resolved = resolve(settings_example())

    again = resolve(FrameParams.from_mapping(resolved.to_mapping()))

    assert again.digest == resolved.digest
    assert again.mapping == resolved.mapping


def test_the_canonical_json_is_sorted_and_has_no_nulls():
    text = resolve(settings_example()).canonical_json()

    assert "null" not in text
    assert ": " not in text and ", " not in text
    assert text == json.dumps(json.loads(text), sort_keys=True, separators=(",", ":"))


def test_a_profile_resolves_on_its_own_with_the_same_digest():
    params = FrameParams.from_mapping(settings_example())

    alone = resolve_profile(params.profiles["scene_detail"])

    assert alone.digest == resolve(params).profiles["scene_detail"].digest


# --- the analysis copy ----------------------------------------------------------------------


def test_the_source_check_passes_when_the_copy_is_large_enough():
    copy = SourceLimits(fps=12, width_px=1280, height_px=720)

    assert check_against_source(FrameParams.from_mapping(settings_example()), copy) == []


def test_the_source_check_rejects_more_fps_or_pixels_than_the_copy_has():
    copy = SourceLimits(fps=3, width_px=480, height_px=720)  # portrait: long side 720

    problems = check_against_source(resolve(settings_example()), copy)

    assert problems == [
        "profiles.scene_detail.frame_selection.candidates.fps: 4 fps is more than "
        "the analysis copy's 3 fps",
        "profiles.scene_detail.image.max_side_px: 768 px is more than the analysis "
        "copy's long side (720 px)",
    ]


def test_the_source_check_covers_the_overview_fps():
    data = profile(views=views(), frame_selection=capped(16))
    data["views"]["overview"]["fps"] = 30

    problems = check_against_source(FrameProfile.from_mapping(data),
                                    SourceLimits(fps=12, width_px=960, height_px=720))

    assert problems == ["views.overview.fps: 30 fps is more than the analysis copy's 12 fps"]


@pytest.mark.parametrize(
    ("fps", "width", "height", "error", "message"),
    [
        (0, 960, 720, ValueError, "fps must be finite and > 0"),
        (float("inf"), 960, 720, ValueError, "fps must be finite and > 0"),
        (True, 960, 720, TypeError, "fps must be a number"),
        (12, 1.5, 720, TypeError, "width_px must be a whole number"),
        (12, 960, True, TypeError, "height_px must be a whole number"),
        (12, 0, 720, ValueError, "width_px must be >= 1"),
    ],
)
def test_source_limits_must_be_finite_positive_numbers(fps, width, height, error, message):
    with pytest.raises(error, match=message):
        SourceLimits(fps=fps, width_px=width, height_px=height)


# --- the model ------------------------------------------------------------------------------


def test_the_model_check_passes_for_a_capable_model():
    resolved = resolve(settings_example())
    caps = ModelVisionCaps(native_video=True, max_images=20, max_image_side_px=1024)

    for name in resolved.profiles:
        assert check_against_model(resolved.profiles[name], caps) == []


def test_native_video_needs_a_model_that_takes_video():
    overview = resolve(settings_example()).for_stage("L2")

    assert check_against_model(overview, ModelVisionCaps(native_video=False)) == [
        "profiles.scene_overview.delivery.mode: the model takes no native video; "
        "use images or storyboard delivery"
    ]


def test_more_frames_than_the_model_accepts_fail():
    detail = FrameParams.from_mapping(settings_example()).profiles["scene_detail"]

    assert check_against_model(detail, ModelVisionCaps(max_images=10)) == [
        "frame_selection.budget: up to 16 images per request, but the model accepts at most 10"
    ]


def test_a_profile_without_a_frame_cap_fails_on_a_limited_model():
    assert check_against_model(FrameProfile.from_mapping(profile()),
                               ModelVisionCaps(max_images=10)) == [
        "frame_selection.budget: no frame cap, but the model accepts at most 10 images; "
        "set max_frames_per_scene"
    ]


def test_views_without_anchors_bound_the_frames_on_their_own():
    bounded = FrameProfile.from_mapping(profile(views=views(6, 2, 5)))

    assert check_against_model(bounded, ModelVisionCaps(max_images=16)) == []
    assert check_against_model(bounded, ModelVisionCaps(max_images=15)) != []


def test_a_storyboard_sends_fewer_but_larger_images():
    board = FrameProfile.from_mapping(profile(
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
        "profiles.scene_detail.image.max_side_px: image.max_side_px is 768 px, but "
        "the model accepts at most 512 px"
    ]


@pytest.mark.parametrize(
    ("caps", "error", "message"),
    [
        ({"max_images": 0}, ValueError, "max_images must be >= 1"),
        ({"max_images": -5}, ValueError, "max_images must be >= 1"),
        ({"max_images": 2.5}, TypeError, "max_images must be a whole number"),
        ({"max_image_side_px": True}, TypeError, "max_image_side_px must be a whole number"),
        ({"native_video": "yes"}, TypeError, "native_video must be True or False"),
    ],
)
def test_model_limits_must_be_whole_numbers_of_at_least_one(caps, error, message):
    with pytest.raises(error, match=message):
        ModelVisionCaps(**caps)


def test_the_model_check_takes_one_profile():
    params = FrameParams.from_mapping(settings_example())

    for block in (params, resolve(params)):
        with pytest.raises(TypeError, match="takes one profile.*for_stage"):
            check_against_model(block, ModelVisionCaps())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="SourceLimits"):
        check_against_source(params, ModelVisionCaps())  # type: ignore[arg-type]


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


@pytest.mark.usefixtures("isolated_registry")
def test_a_registered_name_and_version_can_not_be_registered_again():
    with pytest.raises(ValueError, match="already registered"):
        register_strategy(Strategy("dedupe", 1, "filter", NoParams, needs_pixels=True))


@pytest.mark.usefixtures("isolated_registry")
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
@pytest.mark.usefixtures("isolated_registry")
def test_registration_rejects_malformed_strategies(strategy, message):
    with pytest.raises(ValueError, match=message):
        register_strategy(strategy)


@pytest.mark.usefixtures("isolated_registry")
def test_no_setting_may_be_called_strategy():
    class Clashing(BaseModel):
        strategy: str = "x"

    with pytest.raises(ValueError, match="may not be called 'strategy'"):
        register_strategy(Strategy("test_clash", 1, "anchor", Clashing, run=print))
