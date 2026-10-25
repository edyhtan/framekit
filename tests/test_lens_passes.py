"""Lens passes (L3b) and vectors end to end on a synthetic video with a scripted model:
annotations from fixed vocabularies, phrasings, self-consistency, embeddings, and reruns
that touch only what a new lens needs."""

import shutil

import pytest
from extraction_helpers import write_inputs
from media_helpers import ffmpeg_binary, make_video, needs_ffmpeg
from model_helpers import ScriptedAdapter, sources

from framekit.extraction import Corpus, ExtractionPlan, LocalRunner, SearchSpace
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


def run(tmp_path, files, out="dataset", **options):
    runner = LocalRunner(ffmpeg=ffmpeg_binary(), work_dir=tmp_path / "work", concurrency=4)
    return extract(
        SearchSpace.load(files["space"]),
        Corpus.load(files["corpus"]),
        plan=ExtractionPlan.load(files["plan"]),
        out=tmp_path / out,
        runner=runner,
        sources=sources(),
        **options,
    )


def stages(requests):
    return [r.cache_class for r in requests]


def test_lens_passes_write_annotations_phrasings_and_vectors(tmp_path, video):
    dataset = run(tmp_path, write_inputs(tmp_path, video))
    assert dataset.manifest.complete

    annotations = dataset.rows("annotations")
    assert {a["lens"] for a in annotations} == {"pace"}
    assert {a["target_type"] for a in annotations} == {"scene", "beat"}
    assert {a["run_index"] for a in annotations} == {0, 1}  # two self-consistency runs
    assert all(
        a["label"] in ("slow", "steady", "fast", "pause", "rush", "hold")
        for a in annotations
        if a["field"] in ("tempo", "cues")
    )
    levels = [a for a in annotations if a["field"] == "level"]
    assert levels and all(a["label"] is None and a["value"] == 1 for a in levels)
    assert len(dataset.rows("phrasings")) == 3 * 2 * 3  # scenes x runs x count
    vectors = dataset.rows("vectors")
    assert len(vectors) == 3 + 18 and {v["embedding_model"] for v in vectors} == {"vectors-1"}
    assert all(abs(sum(x * x for x in v["vector"]) - 1) < 1e-5 for v in vectors)  # normalized

    l3b = [r for r in ScriptedAdapter.requests if r.cache_class == "L3b"]
    assert len(l3b) == 3 * 2 and {r.seed for r in l3b} == {7, 8}  # one seed per run
    assert all(not r.images for r in l3b)  # text only: lenses read facts, never frames
    assert '"when everything moves at once"' in l3b[0].system  # the space's examples as style

    runs = dataset.rows("runs")
    assert {r["stage"] for r in runs} == {"L0", "L1", "L2", "L3a", "L3b", "embed"}
    checks = {q["check"] for q in dataset.rows("qc")}
    assert {"vocabulary", "validators", "agreement"} <= checks
    derivations = {d["rule"]: d["value"] for d in dataset.manifest.derivations}
    assert derivations["phrasing_style"] == "space_examples"


def test_runs_that_disagree_are_flagged_or_down_weighted(tmp_path, video):
    files = write_inputs(
        tmp_path,
        video,
        plan={
            "lens_passes": {
                "phrasings": {"count": 3},
                "self_consistency": {"runs": 2, "seeds": [7, 8], "on_disagreement": "down_weight"},
            }
        },
    )
    dataset = run(tmp_path, files)
    agreement = [q for q in dataset.rows("qc") if q["check"] == "agreement"]
    assert len(agreement) == 3 and {q["status"] for q in agreement} == {"down_weight"}
    weights = {a["weight"] for a in dataset.rows("annotations")}
    assert weights and all(w < 0.6 for w in weights)  # seeds 7 and 8 pick different labels


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


def test_the_response_cache_replays_lens_passes_too(tmp_path, video):
    files = write_inputs(tmp_path, video)
    store = FileResponseStore(tmp_path / "cache")
    first = run(tmp_path, files, cache=store, cache_settings=CacheSettings(mode="record"))
    calls = len(ScriptedAdapter.requests)
    shutil.rmtree(tmp_path / "work")
    replayed = run(
        tmp_path, files, out="replayed", cache=store, cache_settings=CacheSettings(mode="replay")
    )
    assert len(ScriptedAdapter.requests) == calls
    assert replayed.rows("annotations") == first.rows("annotations")
    assert set(store.stats()) == {"v1/L2", "v1/L3a", "v1/L3b"}


def test_an_unusable_lens_reply_is_retried(tmp_path, video):
    ScriptedAdapter.fail_first = {"L3b"}
    dataset = run(tmp_path, write_inputs(tmp_path, video))
    assert dataset.manifest.complete
    assert {r["attempts"] for r in dataset.rows("runs") if r["stage"] == "L3b"} == {2}
