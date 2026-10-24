"""Segments as pure functions: processable spans, shot groups, repairs, windows."""

from framekit.extraction.schemas.plan import SceneLength
from framekit.extraction.segmenting import (
    fixed_windows,
    group_shots,
    processable_spans,
    repair_scenes,
    shots_from_cuts,
    snap,
)

LENGTH = SceneLength(min=30, target=45, max=60)


def test_excluded_ranges_are_cut_out():
    assert processable_spans(100, [(0, 9.5), (90, 100)]) == [(9.5, 90.0)]
    assert processable_spans(100, [(40, 50), (45, 60)]) == [(0.0, 40.0), (60.0, 100.0)]
    assert processable_spans(10, [(0, 10)]) == []


def test_shots_come_from_cuts():
    assert shots_from_cuts([3.0, 7.5], 10) == [(0.0, 3.0), (3.0, 7.5), (7.5, 10.0)]


def test_shots_group_into_scenes_of_the_target_length():
    cuts = [10.0 * k for k in range(1, 20)]
    scenes = group_shots((0, 200), cuts, LENGTH)
    assert scenes[0] == (0.0, 50.0)
    assert all(LENGTH.min <= b - a <= LENGTH.max for a, b in scenes)
    assert scenes[-1][1] == 200.0
    assert all(a == prev_b for (_, prev_b), (a, _) in zip(scenes, scenes[1:], strict=False))


def test_a_long_shot_is_split_evenly():
    assert group_shots((0, 200), [], LENGTH) == [(0, 50), (50, 100), (100, 150), (150, 200)]


def test_snapping_takes_the_closest_cut_within_the_tolerance():
    assert snap(10.4, [9.0, 10.0, 11.0], 1.0) == 10.0
    assert snap(10.5, [10.0, 11.0], 1.0) == 10.0  # a tie: the earlier
    assert snap(14.0, [10.0], 1.0) == 14.0


def test_a_proposal_is_repaired_into_a_clean_cover():
    proposed = [(0.0, 31.0, "one"), (31.2, 40.0, "two"), (40.0, 150.0, "three")]
    scenes, notes = repair_scenes(
        proposed, [(0.0, 150.0)], [31.0, 75.0, 110.0], LENGTH, snap_to_cuts=True, tolerance_s=2
    )
    assert scenes[0][0] == 0.0 and scenes[-1][1] == 150.0
    assert all(b - a <= LENGTH.max for a, b, _ in scenes)
    assert any("snapped" in n for n in notes)
    assert any("merged" in n for n in notes)  # the 9 s "two" was too short
    assert any("split" in n for n in notes)  # "three" was 110 s
    assert scenes[0][2]  # summaries follow their scenes


def test_a_span_with_no_proposal_falls_back_to_shot_groups():
    scenes, notes = repair_scenes(
        [], [(0.0, 90.0)], [45.0], LENGTH, snap_to_cuts=True, tolerance_s=2
    )
    assert scenes == [(0.0, 45.0, ""), (45.0, 90.0, "")]
    assert "shot groups used" in notes[0]


def test_windows_cover_each_span():
    assert fixed_windows([(0.0, 7.0)], 3, 2) == [(0.0, 3.0), (2.0, 5.0), (4.0, 7.0)]
    assert fixed_windows([(0.0, 2.0)], 3, 1) == [(0.0, 2.0)]
