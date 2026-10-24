"""Lens specs (docs/settings.md §3.1): the format, its loader and checks, the JSON Schema it
compiles to, and how replies are checked against it."""

import copy

import pytest
from extraction_helpers import COLOUR_LENS, PACE_LENS, write_yaml

from framekit.extraction import ConfigError, LensSpec
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
