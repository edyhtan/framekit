"""Pixel-free frame planning: candidates, anchors, budget, allocation and views, on the
unit timeline, deterministic."""

import pytest
from vision_examples import profile, settings_example

from framekit.vision import (
    Signals,
    Strategy,
    VisionParams,
    VisionProfile,
    plan_frames,
    register_strategy,
    resolve,
)
from framekit.vision.builtin import NoParams


def load(**parts):
    return VisionProfile.from_mapping(profile(**parts))


def selection(candidates, **rest):
    return {"candidates": candidates, **rest}


UNIFORM_4 = {"strategy": "uniform", "fps": 4}


# --- candidates -------------------------------------------------------------------------


def test_uniform_candidates_are_evenly_spaced_from_the_span_start_end_excluded():
    plan = plan_frames(load(frame_selection=selection(UNIFORM_4)), (2.0, 4.0))

    assert plan.timestamps == (2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75)
    assert plan.notes == ()


def test_timestamps_are_rounded_to_milliseconds_on_the_unit_timeline():
    plan = plan_frames(load(frame_selection=selection({"strategy": "uniform", "fps": 3})),
                       (100.0, 101.0))

    assert plan.timestamps == (100.0, 100.333, 100.667)


@pytest.mark.parametrize(
    ("pick", "expected"),
    [
        ("first", (10.0, 14.0, 20.0)),
        ("middle", (12.0, 17.0, 25.0)),
        ("last", (13.999, 19.999, 29.999)),
    ],
)
def test_shot_based_picks_one_frame_per_shot_inside_the_span(pick, expected):
    signals = Signals(shot_boundaries=[5.0, 10.0, 14.0, 20.0, 31.0])
    keyframe = load(frame_selection=selection({"strategy": "shot_based", "pick": pick}))

    assert plan_frames(keyframe, (10.0, 30.0), signals).timestamps == expected


def test_frames_one_millisecond_apart_stay_apart():
    thousand = load(frame_selection=selection({"strategy": "uniform", "fps": 1000}))

    assert len(plan_frames(thousand, (0.0, 1.0)).timestamps) == 1000


def test_the_last_frame_of_a_shot_and_its_cut_are_two_frames():
    last_and_cut = load(frame_selection=selection(
        {"strategy": "shot_based", "pick": "last"}, anchors=["shot_boundaries"]))

    plan = plan_frames(last_and_cut, (0.0, 10.0), Signals(shot_boundaries=[2.0, 5.0, 8.0]))

    assert plan.timestamps == (1.999, 2.0, 4.999, 5.0, 7.999, 8.0, 9.999)


def test_the_span_is_rounded_to_milliseconds_like_the_timestamps():
    plan = plan_frames(load(frame_selection=selection(UNIFORM_4)), (0.0005, 1.0))

    assert plan.timestamps == (0.001, 0.251, 0.501, 0.751)  # never before the span's start


# --- anchors and the budget -----------------------------------------------------------------


def test_anchors_are_always_kept_and_merged_within_a_millisecond():
    signals = Signals(shot_boundaries=[1.1, 2.0], audio_onsets=[1.1004, 3.3])
    anchored = load(frame_selection=selection(
        {"strategy": "uniform", "fps": 1},
        anchors=["shot_boundaries", "audio_onsets"],
        budget={"max_frames_per_scene": 3},
    ))

    plan = plan_frames(anchored, (0.0, 5.0), signals)

    assert plan.timestamps == (1.1, 2.0, 3.3)  # anchors fill the cap; 1.1004 merged into 1.1


def test_anchors_alone_over_the_cap_keep_the_earliest_with_a_note():
    signals = Signals(shot_boundaries=[0.5, 1.5, 2.5, 3.5, 4.5])
    anchored = load(frame_selection=selection(
        UNIFORM_4, anchors=["shot_boundaries"], budget={"max_frames_per_scene": 3}))

    plan = plan_frames(anchored, (0.0, 5.0), signals)

    assert plan.timestamps == (0.5, 1.5, 2.5)
    assert plan.notes == ("5 anchors exceed the cap of 3 frames; kept the earliest 3",)


def test_the_cap_is_respected_and_counts_anchors():
    signals = Signals(shot_boundaries=[2.0])
    capped = load(frame_selection=selection(
        UNIFORM_4, anchors=["shot_boundaries"], budget={"max_frames_per_scene": 5}))

    plan = plan_frames(capped, (0.0, 10.0), signals)

    assert len(plan.timestamps) == 5
    assert 2.0 in plan.timestamps


def test_the_tighter_of_the_two_caps_applies():
    capped = load(frame_selection=selection(
        UNIFORM_4, budget={"max_frames": 3, "max_frames_per_scene": 8}))

    assert len(plan_frames(capped, (0.0, 10.0)).timestamps) == 3


def test_without_signals_the_budget_spreads_frames_across_the_span():
    capped = load(frame_selection=selection(UNIFORM_4, budget={"max_frames_per_scene": 3}))

    plan = plan_frames(capped, (0.0, 10.0))

    assert plan.timestamps == (0.0, 4.75, 9.75)  # earliest, then farthest from those kept


def test_without_a_cap_min_gap_keeps_the_most_frames_at_a_regular_spacing():
    gapped = load(frame_selection=selection(UNIFORM_4, budget={"min_gap_s": 0.3}))

    plan = plan_frames(gapped, (0.0, 10.0))

    assert plan.timestamps == tuple(k / 2 for k in range(20))


def test_a_cap_is_filled_whenever_min_gap_leaves_enough_frames():
    capped = load(frame_selection=selection(
        UNIFORM_4, budget={"min_gap_s": 0.5, "max_frames_per_scene": 18}))

    plan = plan_frames(capped, (0.0, 10.0))

    assert len(plan.timestamps) == 18
    gaps = [b - a for a, b in zip(plan.timestamps, plan.timestamps[1:], strict=False)]
    assert min(gaps) >= 0.5 - 1e-9


def test_min_gap_holds_between_kept_frames_including_anchors():
    signals = Signals(shot_boundaries=[1.1])
    gapped = load(frame_selection=selection(
        UNIFORM_4, anchors=["shot_boundaries"], budget={"min_gap_s": 0.6}))

    plan = plan_frames(gapped, (0.0, 3.0), signals)

    assert 1.1 in plan.timestamps
    gaps = [b - a for a, b in zip(plan.timestamps, plan.timestamps[1:], strict=False)]
    assert min(gaps) >= 0.6 - 1e-9


def test_anchors_are_kept_even_closer_than_the_min_gap():
    signals = Signals(shot_boundaries=[1.0, 1.1])
    gapped = load(frame_selection=selection(
        UNIFORM_4, anchors=["shot_boundaries"], budget={"min_gap_s": 1.0}))

    plan = plan_frames(gapped, (0.0, 2.0), signals)

    assert {1.0, 1.1} <= set(plan.timestamps)
    assert all(abs(t - 1.0) >= 1.0 or t in (1.0, 1.1) for t in plan.timestamps)


def test_allocation_prefers_the_region_where_the_signal_is_dense():
    signals = Signals(audio_onsets=[7.0, 7.1, 7.2, 7.3, 7.4, 7.5, 2.0])
    allocated = load(frame_selection=selection(UNIFORM_4, budget={
        "max_frames_per_scene": 4, "allocate_by": ["audio_onsets"], "min_gap_s": 0.25}))

    plan = plan_frames(allocated, (0.0, 10.0), signals)

    assert len(plan.timestamps) == 4
    assert all(6.5 <= t <= 8.0 for t in plan.timestamps)


def test_allocation_by_motion_follows_the_strongest_motion():
    signals = Signals(motion=[(t / 2, 9.0 if 12 <= t <= 14 else 0.5) for t in range(20)])
    allocated = load(frame_selection=selection(UNIFORM_4, budget={
        "max_frames_per_scene": 3, "allocate_by": ["motion"]}))

    plan = plan_frames(allocated, (0.0, 10.0), signals)

    assert all(5.5 <= t <= 7.5 for t in plan.timestamps)


def test_signals_are_normalized_before_they_are_combined():
    # motion is on a huge scale; onsets still count as much as motion does.
    signals = Signals(motion=[(1.0, 1000.0)], audio_onsets=[8.0])
    allocated = load(frame_selection=selection(UNIFORM_4, budget={
        "max_frames_per_scene": 2, "allocate_by": ["motion", "audio_onsets"]}))

    first, second = plan_frames(allocated, (0.0, 10.0), signals).timestamps

    assert 0.5 <= first <= 1.5 and 7.5 <= second <= 8.5  # one frame near each signal


# --- filters ------------------------------------------------------------------------------


def test_pixel_filters_are_returned_pending_in_order():
    plan = plan_frames(resolve(settings_example()).for_stage("L3a"), (0.0, 30.0))

    assert [ref.key for ref in plan.pending_filters] == ["dedupe@1", "sharpness@1"]
    assert plan.pending_filters[0].params == {"method": "phash", "max_distance": 6}
    assert plan.notes[0].startswith("pixel filters pending (dedupe@1, sharpness@1)")


@pytest.mark.usefixtures("isolated_registry")
def test_pixel_free_filters_run_until_the_first_pixel_filter():
    def drop_whole_seconds(params, times, signals):
        return [t for t in times if t != int(t)]

    register_strategy(Strategy("test_drop_whole_seconds", 1, "filter", NoParams,
                               run=drop_whole_seconds))
    filtered = load(frame_selection=selection(
        {"strategy": "uniform", "fps": 2},
        filters=["test_drop_whole_seconds", {"dedupe": {"max_distance": 4}},
                 "test_drop_whole_seconds"],
    ))

    plan = plan_frames(filtered, (0.0, 3.0))

    assert plan.timestamps == (0.5, 1.5, 2.5)
    assert [ref.key for ref in plan.pending_filters] == ["dedupe@1", "test_drop_whole_seconds@1"]


# --- views --------------------------------------------------------------------------------


def detail_profile(cap=16, overview_frames=6, max_views=2, detail_frames=5, anchors=()):
    return load(
        frame_selection=selection(
            UNIFORM_4, anchors=list(anchors),
            budget={"max_frames_per_scene": cap, "min_gap_s": 0.25}),
        views={
            "mode": "overview_plus_detail",
            "overview": {"fps": 1, "max_frames": overview_frames},
            "detail": {"around": ["motion", "audio_onsets"], "span_s": 3,
                       "max_views": max_views, "max_frames": detail_frames},
        },
    )


PEAKS = Signals(
    audio_onsets=[15.0, 15.2, 15.4, 33.0],
    motion=[(float(t), 5.0 if 32 <= t <= 34 else 0.2) for t in range(0, 45)],
)


def test_overview_plus_detail_spreads_an_overview_and_zooms_on_the_peaks():
    plan = plan_frames(detail_profile(), (0.0, 45.0), PEAKS)

    overview, *details = plan.views
    assert overview.kind == "overview"
    assert len(overview.timestamps) == 6
    assert all(t == int(t) for t in overview.timestamps)  # the overview's own 1 fps grid
    assert max(overview.timestamps) - min(overview.timestamps) >= 30
    assert [d.kind for d in details] == ["detail", "detail"]
    windows = sorted((d.start_s, d.end_s) for d in details)
    assert windows[0][0] <= 15.0 < windows[0][1] and windows[1][0] <= 33.0 < windows[1][1]
    for d in details:
        assert len(d.timestamps) == 5
        assert all(d.start_s <= t < d.end_s for t in d.timestamps)
    assert len(plan.timestamps) == 16
    assert plan.timestamps == tuple(sorted(plan.timestamps))


def test_detail_windows_do_not_overlap_and_stay_inside_the_span():
    # the strongest peak is at the span start; 2.5 s is too close to it for its own view
    signals = Signals(audio_onsets=[0.2, 0.4, 0.6, 1.0, 1.2, 2.5, 5.0, 5.1, 19.9])
    plan = plan_frames(detail_profile(cap=21, max_views=3), (0.0, 20.0), signals)

    details = sorted((v.start_s, v.end_s) for v in plan.views if v.kind == "detail")
    assert len(details) == 3
    assert details[0][0] == 0.0 and details[-1][1] == 20.0
    for (_, end), (start, _) in zip(details, details[1:], strict=False):
        assert start >= end - 1e-9


def test_without_detail_signal_peaks_there_are_no_detail_views():
    plan = plan_frames(detail_profile(), (0.0, 45.0))

    assert [v.kind for v in plan.views] == ["overview"]
    assert plan.notes == (
        "no peaks in the detail signals (audio_onsets@1, motion@1): no detail views",
    )


def test_in_views_the_anchors_count_toward_the_overview():
    signals = Signals(shot_boundaries=[5.0, 10.0, 20.0], audio_onsets=PEAKS.audio_onsets,
                      motion=PEAKS.motion)

    plan = plan_frames(detail_profile(anchors=["shot_boundaries"]), (0.0, 45.0), signals)

    overview, *details = plan.views
    assert len(overview.timestamps) == 6 and {5.0, 10.0, 20.0} <= set(overview.timestamps)
    assert [len(d.timestamps) for d in details] == [5, 5]  # the details lose nothing
    assert len(plan.timestamps) == 16
    assert plan.notes == ()


def test_anchors_beyond_the_overview_leave_the_details_less_room():
    cuts = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 20.0, 25.0]  # 8 anchors; the overview has 6 frames
    signals = Signals(shot_boundaries=cuts, audio_onsets=PEAKS.audio_onsets, motion=PEAKS.motion)

    plan = plan_frames(detail_profile(anchors=["shot_boundaries"]), (0.0, 45.0), signals)

    overview, *details = plan.views
    assert overview.timestamps == tuple(cuts)
    assert [len(d.timestamps) for d in details] == [5, 3]
    assert len(plan.timestamps) == 16
    assert plan.notes == (
        "the overview holds 8 anchors, more than its 6 frames: anchors are always kept",
        "scene cap of 16 frames: the detail view at 15.25 s got 3 of 5 frames",
    )


def one_view(around, span_s, max_views=1):
    return load(
        frame_selection=selection(UNIFORM_4),
        views={"mode": "overview_plus_detail", "overview": {"max_frames": 2},
               "detail": {"around": around, "span_s": span_s, "max_views": max_views,
                          "max_frames": 4}},
    )


def detail_windows(plan):
    return [(v.start_s, v.end_s) for v in plan.views if v.kind == "detail"]


@pytest.mark.parametrize(("window_s", "span_s"), [(1, 3), (4, 2), (10, 2)])
def test_a_detail_view_is_centred_on_its_peak(window_s, span_s):
    # window_s > span_s too: every candidate within window_s / 2 of the onset ties
    view = one_view([{"audio_onsets": {"window_s": window_s}}], span_s)

    plan = plan_frames(view, (0.0, 60.0), Signals(audio_onsets=[20.0]))

    assert detail_windows(plan) == [(20.0 - span_s / 2, 20.0 + span_s / 2)]


def test_a_detail_view_is_centred_on_a_motion_spike():
    view = one_view([{"motion": {"window_s": 8}}], 2)

    plan = plan_frames(view, (0.0, 60.0), Signals(motion=[(20.0, 9.0)]))

    assert detail_windows(plan) == [(19.0, 21.0)]


def test_one_broad_peak_gets_one_detail_view():
    view = one_view([{"audio_onsets": {"window_s": 10}}], 2, max_views=2)

    plan = plan_frames(view, (0.0, 60.0), Signals(audio_onsets=[20.0]))

    assert detail_windows(plan) == [(19.0, 21.0)]


def test_a_second_detail_view_goes_to_another_peak_not_down_the_slope():
    shape = {30: 3.0, 31: 4.0, 32: 5.0, 33: 5.0, 34: 5.0, 35: 4.0, 36: 3.0, 10: 2.0}
    hill = Signals(motion=[(float(t), shape.get(t, 0.5)) for t in range(45)])

    plan = plan_frames(one_view(["motion"], 3, max_views=2), (0.0, 45.0), hill)

    assert detail_windows(plan) == [(31.5, 34.5), (8.5, 11.5)]


def test_view_modes_not_planned_yet_say_so():
    split = load(views={"mode": "split_by_budget"})

    with pytest.raises(NotImplementedError, match="split_by_budget"):
        plan_frames(split, (0.0, 10.0))


def test_a_single_view_mode_plans_like_no_views():
    single = load(frame_selection=selection(UNIFORM_4, budget={"max_frames_per_scene": 4}),
                  views={"mode": "single"})
    plain = load(frame_selection=selection(UNIFORM_4, budget={"max_frames_per_scene": 4}))

    assert plan_frames(single, (0.0, 8.0)).timestamps == plan_frames(plain, (0.0, 8.0)).timestamps


# --- native video, inputs, determinism ------------------------------------------------------


def test_native_video_returns_nominal_uniform_timestamps_with_a_note():
    plan = plan_frames(resolve(settings_example()).for_stage("L2"), (60.0, 64.0))

    assert plan.timestamps == (60.0, 61.0, 62.0, 63.0)
    assert plan.views[0].kind == "native_video"
    assert plan.notes == (
        "native_video: the provider samples the clip itself at 1 fps; these timestamps are "
        "nominal",
    )


def test_a_profile_and_its_resolved_form_plan_the_same():
    params = VisionParams.from_mapping(settings_example())
    raw = params.vision_profiles["scene_detail"]

    assert plan_frames(raw, (0.0, 45.0), PEAKS) == plan_frames(
        resolve(params).for_stage("L3a"), (0.0, 45.0), PEAKS)


def test_planning_is_deterministic():
    detail = resolve(settings_example()).for_stage("L3a")
    as_lists = Signals(audio_onsets=list(reversed(PEAKS.audio_onsets)), motion=list(PEAKS.motion))

    first = plan_frames(detail, (0.0, 45.0), PEAKS)

    assert all(plan_frames(detail, (0.0, 45.0), PEAKS) == first for _ in range(3))
    assert plan_frames(detail, (0.0, 45.0), as_lists) == first


@pytest.mark.parametrize(
    "span", [(5.0, 5.0), (6.0, 5.0), (-1.0, 3.0), (0.0, float("nan")), (1.0, 1.0004), (1.0,)]
)
def test_an_empty_or_negative_span_fails(span):
    with pytest.raises(ValueError, match="span must"):
        plan_frames(load(), span)


def test_signals_must_be_finite_and_non_negative():
    with pytest.raises(ValueError, match="audio_onsets"):
        Signals(audio_onsets=[-1.0])
    with pytest.raises(ValueError, match="motion"):
        Signals(motion=[(1.0, -0.5)])


def test_signals_name_the_field_that_has_the_wrong_type():
    with pytest.raises(TypeError, match="shot_boundaries: expected a list"):
        Signals(shot_boundaries="123")
    with pytest.raises(TypeError, match=r"motion: each sample is \(time_s, value\)"):
        Signals(motion=[(1.0,)])
    with pytest.raises(TypeError, match="audio_onsets: expected a number"):
        Signals(audio_onsets=[True])


def test_a_time_listed_twice_counts_once():
    signals = Signals(shot_boundaries=[1, 2, 5, 5, 9])
    middles = load(frame_selection=selection({"strategy": "shot_based", "pick": "middle"}))

    assert signals.shot_boundaries == (1.0, 2.0, 5.0, 9.0)
    assert plan_frames(middles, (0.0, 10.0), signals).timestamps == (0.5, 1.5, 3.5, 7.0, 9.5)


def test_motion_on_any_scale_plans():
    allocated = load(frame_selection=selection(UNIFORM_4, budget={
        "max_frames_per_scene": 3, "allocate_by": ["motion"]}))

    plan = plan_frames(allocated, (0.0, 3.0), Signals(motion=[(1.0, 1e308), (2.0, 1e308)]))

    assert plan.timestamps == (0.5, 1.5, 2.5)


def test_a_span_with_too_many_candidates_fails_clearly():
    with pytest.raises(ValueError, match="more than 2,000,000: plan shorter spans"):
        plan_frames(load(frame_selection=selection(UNIFORM_4)), (0.0, 1e6))
