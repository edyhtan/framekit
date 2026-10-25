"""`extract` end to end on a synthetic video with a scripted model: the dataset, reruns,
the response cache, retries, the budget, governance, the cost estimate.

The model is `ScriptedAdapter`, served through `ModelSource`s, so requests go through the
real router, cache and adapter template; nothing leaves the machine.
"""

import shutil

import pytest
import yaml
from extraction_helpers import base_plan, write_inputs, write_yaml
from media_helpers import ffmpeg_binary, make_video, needs_ffmpeg
from model_helpers import ScriptedAdapter, sources

from framekit.extraction import (
    ConfigError,
    Corpus,
    Dataset,
    ExtractionPlan,
    LocalRunner,
    SearchSpace,
)
from framekit.extraction.estimate import CostEstimate
from framekit.extraction.pipeline import extract
from framekit.models.cache import CacheSettings, FileResponseStore

pytestmark = needs_ffmpeg


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    return make_video(tmp_path_factory.mktemp("media") / "u1.mp4")


@pytest.fixture(autouse=True)
def fresh_script():
    ScriptedAdapter.requests.clear()
    ScriptedAdapter.fail_first = set()
    yield
    ScriptedAdapter.fail_first = set()


def load(files):
    return (
        SearchSpace.load(files["space"]),
        Corpus.load(files["corpus"]),
        ExtractionPlan.load(files["plan"]),
    )


def run(tmp_path, files, out="dataset", **options):
    space, corpus, plan = load(files)
    runner = options.pop("runner", None) or LocalRunner(
        ffmpeg=ffmpeg_binary(), work_dir=tmp_path / "work", concurrency=4
    )
    return extract(
        space, corpus, plan=plan, out=tmp_path / out, runner=runner, sources=sources(), **options
    )


def stages(requests):
    return [r.cache_class for r in requests]


def test_extract_writes_a_complete_dataset(tmp_path, video):
    files = write_inputs(tmp_path, video)
    dataset = run(tmp_path, files)

    assert isinstance(dataset, Dataset)
    manifest = dataset.manifest
    assert manifest.complete and [u.status for u in manifest.units] == ["complete"]
    assert dataset.rows("units")[0]["metadata"] == {"title": "Patterns", "quality": "good"}

    scenes = dataset.rows("scenes")
    assert [(s["start"], s["end"]) for s in scenes] == [(1.0, 8.0), (8.0, 16.0), (16.0, 24.0)]
    assert {s["method"] for s in scenes} == {"model"}  # excluded [0, 1] left out
    assert [round(s["start"]) for s in dataset.rows("shots")] == [0, 8, 16]
    assert {s["signal"] for s in dataset.rows("signals")} >= {"shot_cut", "audio_onset", "motion"}

    perception = dataset.rows("perception")
    assert all(p["status"] == "ok" and "timing" in p["record"] for p in perception)
    assert len(dataset.rows("beats")) == 6

    sent = dataset.rows("frames_sent")
    assert {f["stage"] for f in sent} == {"L2", "L3a"}
    assert all(len(f["image_sha256"]) == 64 for f in sent)  # digests, never pixels
    l3a = [r for r in ScriptedAdapter.requests if r.cache_class == "L3a"]
    assert len(l3a) == 3 and all(r.images for r in l3a)
    assert all(img.media_type == "image/jpeg" for r in l3a for img in r.images)

    runs = dataset.rows("runs")
    assert {r["stage"] for r in runs} == {"L0", "L1", "L2", "L3a", "L3b", "embed"}
    assert all(r["status"] in ("ok", "flagged") for r in runs)
    checks = {q["check"] for q in dataset.rows("qc")}
    assert {"schema", "timestamps_in_scene", "entities_in_profile", "modules_complete"} <= checks

    derivations = {d["rule"]: d["value"] for d in manifest.derivations}
    assert derivations["modules"] == ["core", "timing"]  # the lens needs timing
    assert derivations["beats"] == {"min_s": 0.5, "max_s": 6.0}
    assert manifest.governance["model_inputs"] == "local_only"
    assert manifest.governance["stays_local"] is True
    assert manifest.models["L3a"]["model_id"] == "seer-1"
    assert manifest.search_space["verbatim"]["intent"] == "test patterns"
    assert manifest.tools["ffmpeg"].startswith("ffmpeg version")
    assert dataset.verify() == []
    assert (tmp_path / "dataset" / ".gitignore").read_text() == "*\n"
    assert (tmp_path / "work" / ".gitignore").read_text() == "*\n"


def test_a_rerun_reuses_every_task(tmp_path, video):
    files = write_inputs(tmp_path, video)
    first = run(tmp_path, files)
    calls = len(ScriptedAdapter.requests)
    again = run(tmp_path, files)
    assert len(ScriptedAdapter.requests) == calls  # no model call the second time
    assert again.manifest.cost["tasks"]["computed"] == 0
    assert again.manifest.dataset_id == first.manifest.dataset_id
    assert again.rows("perception") == first.rows("perception")


def test_a_dry_run_estimates_and_runs_nothing(tmp_path, video):
    files = write_inputs(tmp_path, video)
    estimate = run(tmp_path, files, dry_run=True)
    assert isinstance(estimate, CostEstimate)
    assert estimate.units == 1 and estimate.scenes == 3
    assert set(estimate.stages) == {"L2", "L3a", "L3b", "embed"}
    assert estimate.stages["L3a"].images == 3 * 4 and estimate.total_usd > 0
    assert ScriptedAdapter.requests == [] and not (tmp_path / "dataset").exists()
    assert "Estimate (upper bound)" in str(estimate)


def test_the_response_cache_replays_a_whole_run(tmp_path, video):
    files = write_inputs(tmp_path, video)
    store = FileResponseStore(tmp_path / "cache")
    first = run(tmp_path, files, cache=store, cache_settings=CacheSettings(mode="record"))
    calls = len(ScriptedAdapter.requests)
    shutil.rmtree(tmp_path / "work")  # forget every task: only the cache is left
    replayed = run(
        tmp_path, files, out="replayed", cache=store, cache_settings=CacheSettings(mode="replay")
    )
    assert len(ScriptedAdapter.requests) == calls  # every model call replayed
    assert replayed.rows("perception") == first.rows("perception")
    assert all(r["cached"] for r in replayed.rows("runs") if r["stage"] in ("L2", "L3a"))
    assert set(store.stats()) == {"v1/L2", "v1/L3a", "v1/L3b"}


def test_an_unusable_reply_is_retried_with_its_problems(tmp_path, video):
    ScriptedAdapter.fail_first = {"L3a"}
    dataset = run(tmp_path, write_inputs(tmp_path, video))
    retries = [r for r in ScriptedAdapter.requests if "couldn't be used" in r.prompt]
    assert len(retries) == 3 and "holds no JSON object" in retries[0].prompt
    assert all(p["status"] == "ok" for p in dataset.rows("perception"))
    assert {r["attempts"] for r in dataset.rows("runs") if r["stage"] == "L3a"} == {2}


def test_after_the_allowed_attempts_a_scene_is_flagged_never_dropped(tmp_path, video):
    ScriptedAdapter.fail_first = {"L3a"}
    files = write_inputs(
        tmp_path, video, plan={"quality_control": {"invalid_output": {"max_attempts": 1}}}
    )
    dataset = run(tmp_path, files)
    assert [p["status"] for p in dataset.rows("perception")] == ["flagged"] * 3
    assert len(dataset.rows("scenes")) == 3  # kept, flagged
    flags = [q for q in dataset.rows("qc") if q["check"] == "schema" and q["status"] == "flag"]
    assert len(flags) == 3


def test_shot_groups_need_no_segmentation_model(tmp_path, video):
    plan = {
        "segmentation": {
            "method": "shot_groups",
            "scene_length_s": {"min": 4, "target": 8, "max": 12},
        }
    }
    dataset = run(tmp_path, write_inputs(tmp_path, video, plan=plan))
    assert "L2" not in stages(ScriptedAdapter.requests)
    assert {s["method"] for s in dataset.rows("scenes")} == {"shot_groups"}


def test_windows_are_written_when_asked_for(tmp_path, video):
    plan = {"segmentation": {"mode": "both", "scene_length_s": {"min": 4, "target": 8, "max": 12}}}
    dataset = run(tmp_path, write_inputs(tmp_path, video, plan=plan))
    windows = dataset.rows("windows")
    assert windows[0]["start"] == 1.0 and windows[0]["end"] == 4.0  # 3 s: the shortest moment
    derivations = {d["rule"]: d["value"] for d in dataset.manifest.derivations}
    assert derivations["windows"] == {"window_s": 3.0, "stride_s": 1.0}


def test_a_budget_stops_model_work(tmp_path, video):
    plan = {"budget": {"max_cost_usd": 0.000001, "on_exceed": "stop"}}
    dataset = run(tmp_path, write_inputs(tmp_path, video, plan=plan))
    assert not dataset.manifest.complete
    assert dataset.manifest.units[0].status in ("partial", "failed")
    assert "budget reached" in dataset.manifest.units[0].reason


def test_a_byo_media_corpus_needs_an_explicit_frames_policy(tmp_path, video):
    files = write_inputs(tmp_path, video)
    data = yaml.safe_load(files["plan"].read_text())
    del data["plan"]["governance"]
    write_yaml(files["plan"], data)
    with pytest.raises(ConfigError, match="governance.model_inputs: required for a byo_media"):
        run(tmp_path, files)


def test_a_missing_key_stops_the_run_before_any_work(tmp_path, video, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    models = dict(base_plan()["models"], L3a={"model_id": "claude-haiku-4-5"})
    files = write_inputs(
        tmp_path, video, plan={"models": models, "governance": {}}, rights_mode="hosted"
    )
    estimate = run(tmp_path, files, dry_run=True)
    assert "a real run would stop here: models.L3a: ANTHROPIC_API_KEY" in str(estimate)
    with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY is not set"):
        run(tmp_path, files)
    assert not (tmp_path / "work").exists()  # no analysis copy, no task


def test_a_folder_holding_another_dataset_is_refused(tmp_path, video):
    run(tmp_path, write_inputs(tmp_path, video))
    files = write_inputs(tmp_path, video, lenses=("colour",))
    with pytest.raises(FileExistsError, match="holds another dataset"):
        run(tmp_path, files)


def test_frame_profiles_are_checked_against_the_analysis_copy(tmp_path, video):
    profiles = base_plan()["vision_profiles"]
    profiles["detail"]["image"]["max_side_px"] = 400
    files = write_inputs(
        tmp_path,
        video,
        plan={"normalize": {"fps": 8, "height_px": 120}, "vision_profiles": profiles},
    )
    with pytest.raises(ConfigError, match="analysis copy's long side"):
        run(tmp_path, files)  # 400 px frames from a 214 x 120 copy would be upscaled
    assert ScriptedAdapter.requests == []  # refused before any spend
