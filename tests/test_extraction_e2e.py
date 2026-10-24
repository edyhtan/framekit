"""`extract` end to end on a synthetic video with a scripted model: the dataset, reruns,
the response cache, retries, the budget, governance, the benchmark harness and the CLI.

The model is `ScriptedAdapter`, served through `ModelSource`s, so requests go through the
real router, cache and adapter template; nothing leaves the machine.
"""

import json
import shutil

import pytest
import yaml
from extraction_helpers import (
    ScriptedAdapter,
    base_plan,
    ffmpeg_binary,
    make_video,
    needs_ffmpeg,
    sources,
    write_inputs,
    write_yaml,
)

from framekit.cli import main
from framekit.extraction import (
    ConfigError,
    Corpus,
    Dataset,
    ExtractionPlan,
    GoldSet,
    LocalRunner,
    SearchSpace,
)
from framekit.extraction.bench import benchmark
from framekit.extraction.estimate import CostEstimate
from framekit.extraction.pipeline import extract
from framekit.models.cache import CacheSettings, ClassPolicy, FileResponseStore

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

    perception = dataset.rows("perception")
    assert all(p["status"] == "ok" and "timing" in p["record"] for p in perception)
    assert len(dataset.rows("beats")) == 6

    annotations = dataset.rows("annotations")
    assert {a["lens"] for a in annotations} == {"pace"}
    assert {a["target_type"] for a in annotations} == {"scene", "beat"}
    assert {a["run_index"] for a in annotations} == {0, 1}  # two self-consistency runs
    assert all(
        a["label"] in ("slow", "steady", "fast", "pause", "rush", "hold")
        for a in annotations
        if a["field"] in ("tempo", "cues")
    )
    assert len(dataset.rows("phrasings")) == 3 * 2 * 3  # scenes x runs x count
    vectors = dataset.rows("vectors")
    assert len(vectors) == 3 + 18 and {v["embedding_model"] for v in vectors} == {"vectors-1"}

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
    assert {"schema", "timestamps_in_scene", "entities_in_profile", "agreement"} <= checks

    derivations = {d["rule"]: d["value"] for d in manifest.derivations}
    assert derivations["modules"] == ["core", "timing"]  # the lens needs timing
    assert derivations["beats"] == {"min_s": 0.5, "max_s": 6.0}
    assert derivations["phrasing_style"] == "space_examples"
    assert manifest.governance["model_inputs"] == "local_only"
    assert manifest.governance["stays_local"] is True
    assert manifest.models["L3a"]["model_id"] == "seer-1"
    assert manifest.search_space["verbatim"]["intent"] == "test patterns"
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
    assert again.rows("annotations") == first.rows("annotations")


def test_adding_a_lens_reruns_only_lens_passes_and_embeddings(tmp_path, video):
    files = write_inputs(tmp_path, video)
    run(tmp_path, files)
    before = len(ScriptedAdapter.requests)
    files = write_inputs(tmp_path, video, lenses=("pace", "colour"))
    dataset = run(tmp_path, files, out="dataset-2")
    new = stages(ScriptedAdapter.requests[before:])
    assert set(new) == {"L3b"} and len(new) == 3 * 2  # the colour lens: scenes x runs
    assert {a["lens"] for a in dataset.rows("annotations")} == {"pace", "colour"}
    assert dataset.manifest.cost["tasks"]["reused"] > 0


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
    assert replayed.rows("annotations") == first.rows("annotations")
    assert all(r["cached"] for r in replayed.rows("runs") if r["stage"] in ("L2", "L3a", "L3b"))
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
    assert len(flags) == 3 and dataset.rows("annotations") == []


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


def test_a_folder_holding_another_dataset_is_refused(tmp_path, video):
    run(tmp_path, write_inputs(tmp_path, video))
    files = write_inputs(tmp_path, video, lenses=("colour",))
    with pytest.raises(FileExistsError, match="holds another dataset"):
        run(tmp_path, files)


def test_vision_profiles_are_checked_against_the_analysis_copy(tmp_path, video):
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


GOLD = {
    "gold": {
        "version": 1,
        "corpus": "synthetic",
        "match": {"min_overlap": 0.5},
        "queries": [
            {
                "query": "bravo pattern moment",
                "moments": [{"unit_id": "u1", "start": 9.0, "end": 12.0}],
            },
            {
                "query": "charlie pattern",
                "moments": [{"unit_id": "u1", "start": 17.0, "end": 20.0}],
            },
            {"query": "nothing to find here"},
        ],
    }
}


def test_the_benchmark_compares_search_spaces_with_a_pinned_plan(tmp_path, video):
    files = write_inputs(tmp_path, video)
    space, corpus, plan = load(files)
    files_two = write_inputs(tmp_path / "two", video, lenses=("pace", "colour"))
    gold = GoldSet.from_mapping(GOLD)
    report = benchmark(
        {"one-lens": space, "two-lenses": SearchSpace.load(files_two["space"])},
        corpus,
        plan=plan,
        gold=gold,
        out=tmp_path / "bench",
        sources=sources(),
        runner=LocalRunner(ffmpeg=ffmpeg_binary(), concurrency=4),
    )
    assert [v.label for v in report.variants] == ["one-lens", "two-lenses"]
    for variant in report.variants:
        assert variant.error is None and variant.complete
        assert variant.metrics["queries_scored"] == 2
        assert variant.metrics["recall@1"] == 1.0
    second = report.variants[1].cost["tasks"]
    assert second["reused"] > 0  # L0-L3a ran once for both
    saved = json.loads((tmp_path / "bench" / "report.json").read_text())
    assert saved["kind"] == "search_spaces"
    assert "| one-lens |" in (tmp_path / "bench" / "report.md").read_text()


def test_the_benchmark_refuses_unfair_comparisons(tmp_path, video):
    files = write_inputs(tmp_path, video)
    space, corpus, plan = load(files)
    gold = GoldSet.from_mapping(GOLD)
    with pytest.raises(ValueError, match="pin the plan"):
        benchmark({"a": space}, corpus, plan=plan, space=space, gold=gold, out=tmp_path / "b")
    leaky = SearchSpace.from_mapping(
        {**space.verbatim, "query_style": {"examples": ["Bravo pattern moment!"]}},
        base_dir=tmp_path,
    )
    with pytest.raises(ConfigError, match="repeats the gold query"):
        benchmark({"a": leaky}, corpus, plan=plan, gold=gold, out=tmp_path / "b", sources=sources())


def test_the_cli_runs_a_dry_run_and_reads_the_cache(tmp_path, video, capsys, monkeypatch):
    files = write_inputs(tmp_path, video)
    monkeypatch.syspath_prepend(str(__import__("pathlib").Path(__file__).parent))
    code = main(
        [
            "extract",
            "--space",
            str(files["space"]),
            "--corpus",
            str(files["corpus"]),
            "--plan",
            str(files["plan"]),
            "--out",
            str(tmp_path / "cli"),
            "--dry-run",
            "--ffmpeg",
            ffmpeg_binary(),
            "--sources",
            "extraction_helpers:sources",
        ]
    )
    assert code == 0 and "Estimate (upper bound)" in capsys.readouterr().out
    assert main(["cache", "stats", "--cache-dir", str(tmp_path / "empty")]) == 0
    assert "the cache is empty" in capsys.readouterr().out


def test_cli_config_errors_are_reported_not_raised(tmp_path, capsys):
    bad = write_yaml(tmp_path / "space.yaml", {"search_space": {"lenses": []}})
    code = main(
        ["extract", "--space", str(bad), "--corpus", str(bad), "--out", str(tmp_path / "x")]
    )
    assert code == 1 and "problem" in capsys.readouterr().err


def test_cache_policies_per_class_come_from_the_cli():
    from framekit.cli import parse_cache_modes, parse_rate

    assert parse_cache_modes(["record", "L3b=replay"]) == ("record", {"L3b": "replay"})
    assert parse_rate("hive=120/m") == ("hive", 2.0)
    assert ClassPolicy(mode="replay").mode == "replay"
