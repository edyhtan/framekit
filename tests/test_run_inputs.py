"""The run inputs (docs/settings.md §3–§6): the search space, the corpus, the plan and the gold
set, as checked schemas."""

import copy

import pytest
from extraction_helpers import COLOUR_LENS, base_plan, write_yaml

from framekit.extraction import (
    ConfigError,
    Corpus,
    ExtractionPlan,
    GoldSet,
    SearchSpace,
)
from framekit.extraction.schemas.gold import overlapping_queries

# --- the search space ------------------------------------------------------------------


def test_a_search_space_loads_its_lenses_and_checks_their_versions(tmp_path):
    write_yaml(tmp_path / "lenses" / "colour.yaml", COLOUR_LENS)
    space = SearchSpace.from_mapping(
        {
            "search_space": {
                "lenses": [{"spec": "lenses/colour.yaml", "version": "1.0.0"}],
                "resolution": {"moment_length_s": {"min": 3, "max": 6}},
            }
        },
        base_dir=tmp_path,
    )
    assert [s.key for s in space.lens_specs] == ["colour@1.0.0"]
    assert space.verbatim["lenses"][0]["spec"] == "lenses/colour.yaml"  # as given
    with pytest.raises(ConfigError, match="the space asks for colour 2.0.0"):
        SearchSpace.from_mapping(
            {
                "lenses": [{"spec": "lenses/colour.yaml", "version": "2.0.0"}],
                "resolution": {"moment_length_s": {"min": 3, "max": 6}},
            },
            base_dir=tmp_path,
        )


def test_the_space_digest_changes_with_a_lens_spec(tmp_path):
    write_yaml(tmp_path / "lenses" / "colour.yaml", COLOUR_LENS)
    data = {
        "lenses": [{"spec": "lenses/colour.yaml", "version": "1.0.0"}],
        "resolution": {"moment_length_s": {"min": 3, "max": 6}},
    }
    first = SearchSpace.from_mapping(data, base_dir=tmp_path).digest
    changed = copy.deepcopy(COLOUR_LENS)
    changed["lens"]["prompts"]["interpretation"] = "Something else."
    write_yaml(tmp_path / "lenses" / "colour.yaml", changed)
    assert SearchSpace.from_mapping(data, base_dir=tmp_path).digest != first


def test_space_problems_are_listed():
    with pytest.raises(ConfigError) as info:
        SearchSpace.from_mapping(
            {
                "lenses": [],
                "resolution": {"moment_length_s": {"min": 6, "max": 3}},
                "scope": {"units": {"include": ["a"], "exclude": ["a"]}},
            }
        )
    problems = info.value.problems
    assert "lenses: at least one lens is needed" in problems
    assert "resolution.moment_length_s: min is more than max" in problems
    assert "scope.units: 'a' is both included and excluded" in problems


# --- the corpus ------------------------------------------------------------------------


def corpus_data(**unit):
    return {
        "corpus": {
            "profile": {
                "name": "demo",
                "rights_mode": "hosted",
                "metadata_fields": {"title": "str", "year": "int", "grade": ["a", "b"]},
            },
            "units": [{"unit_id": "u1", "path": "videos/one.mp4", **unit}],
        }
    }


def test_unit_metadata_can_sit_on_the_unit_and_paths_become_absolute(tmp_path):
    corpus = Corpus.from_mapping(corpus_data(title="One", year=1951, grade="a"), base_dir=tmp_path)
    unit = corpus.units[0]
    assert dict(unit.metadata) == {"title": "One", "year": 1951, "grade": "a"}
    assert unit.path == str((tmp_path / "videos" / "one.mp4").resolve())


def test_metadata_is_checked_against_the_profile():
    with pytest.raises(ConfigError) as info:
        Corpus.from_mapping(corpus_data(title=7, year=True, grade="c", colour="red"))
    problems = "\n".join(info.value.problems)
    assert "units[0].title: expected str" in problems
    assert "units[0].year: expected int" in problems  # true is not a number
    assert "units[0].grade: 'c' is not one of a, b" in problems
    assert "units[0].colour: not a unit field" in problems


def test_unit_ids_are_strings_so_yaml_numbers_fail():
    data = corpus_data()
    data["corpus"]["units"][0]["unit_id"] = 42  # YAML `unit_id: 042` is a number (34!)
    with pytest.raises(ConfigError, match="unit_id"):
        Corpus.from_mapping(data)


def test_the_corpus_digest_leaves_paths_out(tmp_path):
    one = Corpus.from_mapping(corpus_data(), base_dir=tmp_path / "a")
    two = Corpus.from_mapping(corpus_data(), base_dir=tmp_path / "b")
    assert one.digest == two.digest


# --- the plan --------------------------------------------------------------------------


def test_the_default_plan_needs_no_file():
    plan = ExtractionPlan()
    assert plan.segmentation.method == "model" and plan.models.L3a is None
    assert set(plan.vision.stages) == {"L2", "L3a"}


def test_a_plan_overrides_defaults_group_by_group():
    plan = ExtractionPlan.from_mapping({"plan": {"normalize": {"fps": 8}}})
    assert plan.normalize.fps == 8 and plan.normalize.height_px == 720


def test_frame_profiles_in_a_plan_replace_the_defaults():
    plan = ExtractionPlan.from_mapping(base_plan())
    assert set(plan.vision.vision_profiles) == {"overview", "detail"}
    assert plan.vision.stages["L3a"].vision == "detail"


def test_plan_problems_are_listed_with_their_paths():
    data = base_plan() | {
        "normalize": {"fps": "12"},
        "bogus": 1,
        "stages": {"L3a": {"vision": "nope"}},
    }
    with pytest.raises(ConfigError) as info:
        ExtractionPlan.from_mapping(data)
    problems = info.value.problems
    assert any(p.startswith("stages.L3a.vision") for p in problems)
    assert "normalize.fps: Input should be a valid number" in problems
    assert "bogus: Extra inputs are not permitted" in problems


def test_cross_setting_checks():
    data = base_plan() | {
        "segmentation": {"mode": "fixed_windows"},
        "lens_passes": {"self_consistency": {"runs": 3, "seeds": [1, 2]}},
        "formatting": {"tables": ["scenes"]},
        "embedding": {"text": {"template": "no placeholder"}},
    }
    with pytest.raises(ConfigError) as info:
        ExtractionPlan.from_mapping(data)
    problems = "\n".join(info.value.problems)
    assert "fixed_windows alone" in problems
    assert "3 runs need 3 seeds" in problems
    assert "formatting.tables: units is needed" in problems
    assert "must contain {text}" in problems


# --- the gold set ----------------------------------------------------------------------


def test_a_gold_set_loads_and_rejects_repeated_queries():
    gold = GoldSet.from_mapping(
        {
            "gold": {
                "corpus": "demo",
                "queries": [
                    {
                        "query": "a slow morning",
                        "moments": [{"unit_id": "u1", "start": 1, "end": 4}],
                    },
                    {"query": "life is hard"},
                ],
            }
        }
    )
    assert gold.match.min_overlap == 0.5
    with pytest.raises(ConfigError, match="the same query"):
        GoldSet.from_mapping(
            {
                "corpus": "demo",
                "queries": [{"query": "A slow morning!"}, {"query": "a slow morning"}],
            }
        )


def test_examples_that_repeat_a_gold_query_are_found():
    gold = GoldSet.from_mapping(
        {"corpus": "demo", "queries": [{"query": "my team keeps arguing about the same thing"}]}
    )
    found = overlapping_queries(
        ("My team keeps arguing about the same thing!", "a quiet walk"), gold
    )
    assert len(found) == 1 and "repeats the gold query" in found[0]
