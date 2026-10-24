"""The extraction schemas: A, V, the plan, lens specs, the gold set (docs/settings.md §3–§6)."""

import copy

import pytest
from extraction_helpers import COLOUR_LENS, PACE_LENS, base_plan, write_yaml

from framekit.extraction import (
    ConfigError,
    Corpus,
    ExtractionPlan,
    GoldSet,
    LensSpec,
    SearchSpace,
)
from framekit.extraction.schemas.gold import overlapping_queries
from framekit.extraction.schemas.lens import annotation_rows

# --- lens specs ------------------------------------------------------------------------


def pace(tmp_path):
    write_yaml(tmp_path / "pace" / "lens.yaml", PACE_LENS)
    (tmp_path / "pace" / "interpret.md").write_text("Rate the pace.", encoding="utf-8")
    return LensSpec.load(tmp_path / "pace")


def test_a_lens_spec_loads_from_a_folder_with_its_prompt_file(tmp_path):
    lens = pace(tmp_path)
    assert lens.key == "pace@0.1.0"
    assert lens.prompts.interpretation == "Rate the pace."
    assert lens.requires_modules == ("timing",)
    assert lens.consumer["judge_rubric"] == "Does the pace match?"  # carried, unread


def test_the_digest_covers_the_prompts(tmp_path):
    first = pace(tmp_path).digest
    (tmp_path / "pace" / "interpret.md").write_text("Rate it differently.", encoding="utf-8")
    assert LensSpec.load(tmp_path / "pace").digest != first


def test_prompt_files_must_stay_inside_the_spec_folder(tmp_path):
    data = copy.deepcopy(PACE_LENS)
    data["lens"]["prompts"]["interpretation"] = {"file": "../secret.txt"}
    (tmp_path / "secret.txt").write_text("not a prompt", encoding="utf-8")
    write_yaml(tmp_path / "spec" / "lens.yaml", data)
    with pytest.raises(ConfigError, match="outside the spec's folder"):
        LensSpec.load(tmp_path / "spec")


def test_every_problem_in_a_lens_spec_is_listed():
    data = {
        "name": "Bad Name",
        "version": "one",
        "fields": {
            "a": {"type": "label"},  # no vocabulary
            "b": {"type": "integer", "vocabulary": ["x"], "min": 5, "max": 1},
            "c": {"type": "list"},  # no items
            "d": {
                "type": "map",
                "keys": "entities",
                "values": {"type": "boolean"},
                "vocabulary": ["k"],
            },
        },
        "prompts": {"interpretation": "x"},
    }
    with pytest.raises(ConfigError) as info:
        LensSpec.from_mapping(data)
    text = str(info.value)
    assert "name" in text and "version" in text
    data["name"], data["version"] = "ok", "1.0"
    with pytest.raises(ConfigError) as info:
        LensSpec.from_mapping(data)
    problems = info.value.problems
    assert any("fields.a.vocabulary" in p for p in problems)
    assert any("fields.b.vocabulary: not used by type integer" in p for p in problems)
    assert any("fields.b: min is more than max" in p for p in problems)
    assert any("fields.c.items" in p for p in problems)
    assert any("fields.d.vocabulary: not used with keys: entities" in p for p in problems)


def test_labels_are_identifiers_never_free_text():
    data = copy.deepcopy(COLOUR_LENS)
    data["lens"]["fields"]["palette"]["vocabulary"] = ["warm", "Very Cool!"]
    with pytest.raises(ConfigError, match="palette"):
        LensSpec.from_mapping(data)


def lens_with_map():
    return LensSpec.from_mapping(
        {
            "name": "people",
            "version": "1",
            "fields": {
                "who": {
                    "type": "map",
                    "keys": "entities",
                    "extra_keys": ["viewer"],
                    "values": {
                        "type": "list",
                        "max": 2,
                        "items": {
                            "type": "object",
                            "fields": {
                                "label": {"type": "label", "vocabulary": ["calm", "tense"]},
                                "level": {"type": "integer", "min": 1, "max": 5},
                            },
                        },
                    },
                },
                "note": {"type": "boolean", "optional": True},
            },
            "prompts": {"interpretation": "Who feels what."},
        }
    )


def test_the_output_schema_lists_entities_and_extra_keys():
    lens = lens_with_map()
    schema = lens.output_schema(targets=["scene"], phrasings=2, entities=["Ada"])
    who = schema["properties"]["scene"]["properties"]["who"]
    assert set(who["properties"]) == {"Ada", "viewer"} and who["additionalProperties"] is False
    assert schema["properties"]["phrasings"]["minItems"] == 2
    assert "beats" not in schema["properties"]
    note = schema["properties"]["scene"]["properties"]["note"]
    assert note["anyOf"][1] == {"type": "null"}  # optional: nullable


def test_replies_are_checked_against_the_lens():
    lens = lens_with_map()
    good = {
        "scene": {"who": {"Ada": [{"label": "calm", "level": 2}]}, "note": None},
        "phrasings": ["a", "b"],
    }
    assert lens.check_output(good, targets=["scene"], phrasings=2, entities=["Ada"], beats=0) == []
    bad = {
        "scene": {"who": {"Zed": [], "viewer": [{"label": "angry", "level": 9}] * 3}},
        "phrasings": ["a"],
        "extra": 1,
    }
    problems = lens.check_output(bad, targets=["scene"], phrasings=2, entities=["Ada"], beats=0)
    assert "extra: not expected" in problems
    assert any("key 'Zed' is not allowed" in p for p in problems)
    assert any("'angry' is not in the vocabulary" in p for p in problems)
    assert any("9 is outside 1.0..5.0" in p for p in problems)
    assert any("at most 2 expected, got 3" in p for p in problems)
    assert "phrasings: expected 2, got 1" in problems


def test_entity_maps_take_any_name_when_the_profile_lists_none():
    lens = lens_with_map()
    reply = {"scene": {"who": {"Someone": []}}, "phrasings": ["a", "b"]}
    assert lens.check_output(reply, targets=["scene"], phrasings=2, entities=[], beats=0) == []


def test_annotations_become_rows_one_per_label():
    lens = LensSpec.from_mapping(
        PACE_LENS | {"lens": {**PACE_LENS["lens"], "prompts": {"interpretation": "x"}}}
    )
    rows = annotation_rows(lens.fields, {"tempo": "fast", "cues": ["pause", "hold"], "level": 3})
    assert rows == [
        {"field": "tempo", "label": "fast", "value": None},
        {"field": "cues", "label": "pause", "value": None},
        {"field": "cues", "label": "hold", "value": None},
        {"field": "level", "label": None, "value": 3},
    ]


def test_validators_are_checked_at_load_and_on_annotations():
    data = copy.deepcopy(COLOUR_LENS)
    data["lens"]["validators"] = [{"kind": "exclusive", "field": "palette", "labels": ["warm"]}]
    with pytest.raises(ConfigError, match="a labels field"):
        LensSpec.from_mapping(data)
    lens = LensSpec.from_mapping(
        {
            **PACE_LENS["lens"],
            "prompts": {"interpretation": "x"},
            "validators": [{"kind": "exclusive", "field": "cues", "labels": ["pause", "rush"]}],
        }
    )
    assert lens.check_rules({"cues": ["pause", "rush"]}, "scene") == [
        "scene.cues: pause, rush can't appear together"
    ]


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


def test_vision_profiles_in_a_plan_replace_the_defaults():
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
