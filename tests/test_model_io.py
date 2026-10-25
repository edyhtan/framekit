"""What models are asked and how replies are checked: the prompts, the perception record
schema and its checks, and finding the JSON object in a reply."""

import json

import pytest

from framekit.extraction.prompts import (
    image_legend,
    lens_request,
    perceive_request,
    segment_request,
)
from framekit.extraction.records import (
    check_record,
    parse_json_reply,
    record_schema,
    segmentation_schema,
    shape_problems,
)

# --- the perception record -------------------------------------------------------------


def record(**changes):
    base = {
        "summary": "a ball rolls under a table",
        "setting": "a kitchen",
        "entities": ["Ada"],
        "props": ["ball"],
        "beats": [
            {"t": [10.0, 12.0], "who": ["Ada"], "action": "rolls the ball", "audio": None},
            {"t": [12.0, 15.0], "who": ["unknown: a child"], "action": "looks", "audio": "a thud"},
        ],
    }
    return base | changes


def test_the_schema_holds_the_core_and_only_the_modules_the_model_writes():
    schema = record_schema(["core", "timing", "speech"])
    assert set(schema["properties"]) == {
        "summary",
        "setting",
        "entities",
        "props",
        "beats",
        "timing",
    }
    assert schema["additionalProperties"] is False  # speech comes from L1, never the model


def test_a_valid_record_passes_every_check():
    assert shape_problems(record(), record_schema(["core"])) == []
    checks = check_record(
        record(), span=(10.0, 15.0), modules=["core"], entities=["Ada"], unknown="mark_unknown"
    )
    assert checks == {
        "timestamps_in_scene": [],
        "timestamps_ordered": [],
        "entities_in_profile": [],
        "modules_complete": [],
    }


def test_shape_problems_name_their_paths():
    broken = record(beats=[{"t": [1.0], "who": "Ada", "action": 3, "audio": None}], extra=1)
    problems = shape_problems(broken, record_schema(["core"]))
    assert "record.extra: not expected" in problems
    assert "record.beats[0].t: expected at least 2 items" in problems
    assert "record.beats[0].who: expected a list" in problems
    assert "record.beats[0].action: expected text" in problems


def test_times_names_and_modules_are_checked():
    late = record(
        beats=[
            {"t": [14.0, 13.0], "who": ["Zed"], "action": "a", "audio": None},
            {"t": [11.0, 30.0], "who": [], "action": "b", "audio": None},
        ]
    )
    checks = check_record(
        late, span=(10.0, 15.0), modules=["core", "sound"], entities=["Ada"], unknown="mark_unknown"
    )
    assert checks["timestamps_in_scene"] == [
        "beats[1].t: [11.0, 30.0] is outside the scene [10.0, 15.0]"
    ]
    assert "beats[0].t: starts after it ends" in checks["timestamps_ordered"]
    assert "beats: not in time order" in checks["timestamps_ordered"]
    assert checks["entities_in_profile"] == [
        "beats[0].who[0]: 'Zed' is not in the profile's entity list"
    ]
    assert checks["modules_complete"] == ["sound: missing"]
    relaxed = check_record(
        late, span=(10.0, 15.0), modules=["core"], entities=["Ada"], unknown="allow"
    )
    assert relaxed["entities_in_profile"] == []


def test_the_segmentation_reply_shape():
    reply = {"synopsis": "s", "scenes": [{"start": 0, "end": 5.5, "summary": "x"}]}
    assert shape_problems(reply, segmentation_schema()) == []
    assert shape_problems({"synopsis": "s"}, segmentation_schema()) == ["record.scenes: missing"]


# --- the prompts -----------------------------------------------------------------------


def test_the_perception_prompt_carries_the_scene_names_modules_and_frame_times():
    system, prompt = perceive_request(
        span=(10.0, 15.0),
        modules=["core", "camera"],
        beats=(0.5, 6.0),
        entities=["Ada"],
        hints=["no dialogue"],
        synopsis="a day",
        neighbors=("before", ""),
        signals={"shot_cuts": [12.0]},
        images=[[10.0], [12.5]],
        grid=None,
    )
    assert "between 0.5 and 6 seconds" in system and "camera:" in system
    assert "Scene: from 10.000 s to 15.000 s." in prompt and "Names to use: Ada." in prompt
    assert "About this collection: no dialogue." in prompt and "Previous scene: before" in prompt
    assert "Image 2: the frame at 12.500 s" in prompt


def test_a_storyboard_legend_lists_each_tile_time():
    legend = image_legend([[1.0, 2.0, 3.0]], grid=(2, 2))
    assert "a grid of 2 x 2 frames" in legend and "1.000, 2.000, 3.000" in legend
    assert "the remaining tiles are empty" in legend


def test_the_segmentation_prompt_lists_the_parts_cuts_and_lengths():
    system, prompt = segment_request(
        duration_s=60.0,
        spans=[(1.0, 60.0)],
        cuts=[8.0, 16.0],
        images=[[2.0]],
        grid=None,
        entities=[],
        hints=[],
        scene_length={"min": 4, "target": 8, "max": 12},
    )
    assert "between 4 and 12 seconds, about 8 seconds" in system
    assert "Parts to cover: [1.00, 60.00]." in prompt and "Shot cuts (s): 8.00, 16.00." in prompt


def test_the_lens_prompt_frames_the_lens_and_its_phrasings():
    system, prompt = lens_request(
        name="pace",
        description="How fast a moment feels.",
        interpretation="Rate the pace.",
        phrasing_prompt="Searches a person would type.",
        style_examples=["a slow morning"],
        style="",
        phrasings=3,
        beats=True,
        context={"scene": {"start": 1, "end": 2}},
    )
    assert 'lens "pace" (How fast a moment feels.)' in system and "Rate the pace." in system
    assert "write 3 different searches" in system and '"a slow morning"' in system
    assert "Under `beats`" in system
    assert json.loads(prompt.split("\n\nReply with")[0]) == {"scene": {"start": 1, "end": 2}}


# --- model replies ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "reply",
    [
        '{"a": 1}',
        '```json\n{"a": 1}\n```',
        '```\n{"a": 1}\n```',
        '```json {"a": 1} ```',
        'Here is the record:\n{"a": 1}\nThat is all.',
    ],
)
def test_the_json_object_is_found_in_fences_and_prose(reply):
    assert parse_json_reply(reply) == {"a": 1}


def test_a_reply_without_json_is_refused_quickly():
    with pytest.raises(ValueError, match="no JSON object"):
        parse_json_reply("```" + " " * 100_000 + "no closing fence")
