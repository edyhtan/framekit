"""The benchmark harness end to end on a synthetic video with a scripted model: variants
share their work, the baseline retriever scores them on a gold set, and unfair comparisons
are refused before any work."""

import json

import pytest
from extraction_helpers import base_plan, merged, write_inputs
from media_helpers import ffmpeg_binary, make_video, needs_ffmpeg
from model_helpers import ScriptedAdapter, sources

from framekit.extraction import (
    ConfigError,
    Corpus,
    ExtractionPlan,
    GoldSet,
    LocalRunner,
    SearchSpace,
)
from framekit.extraction.bench import benchmark

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
    assert report.varying_components == ()  # one pinned plan
    markdown = (tmp_path / "bench" / "report.md").read_text()
    assert "| one-lens |" in markdown and "differ between variants: none." in markdown


def test_comparing_plans_names_the_component_that_differs(tmp_path, video):
    files = write_inputs(tmp_path, video)
    space, corpus, plan = load(files)
    more = {"interpretation": {"lens_passes": {"phrasings": {"count": 5}}}}
    report = benchmark(
        {"three": plan, "five": ExtractionPlan.from_mapping(merged(base_plan(), more))},
        corpus,
        space=space,
        gold=GoldSet.from_mapping(GOLD),
        out=tmp_path / "bench",
        sources=sources(),
        runner=LocalRunner(ffmpeg=ffmpeg_binary(), concurrency=4),
    )
    assert report.kind == "plans" and report.varying_components == ("interpretation",)
    one, two = (v.components for v in report.variants)
    assert [name for name in one if one[name] != two[name]] == ["interpretation"]
    assert report.variants[1].cost["tasks"]["reused"] > 0  # L0-L3a shared


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
