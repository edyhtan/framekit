"""Resolution: derivation rules, every check before any spend, digests (no media needed)."""

import copy
from pathlib import Path

import pyarrow as pa
import pytest
from extraction_helpers import COLOUR_LENS, PACE_LENS, base_plan, sources

from framekit.extraction import (
    ConfigError,
    Corpus,
    Dataset,
    ExtractionPlan,
    LensSpec,
    SearchSpace,
    resolve_run,
)
from framekit.extraction.dataset import SCHEMAS, to_table, write_manifest, write_tables
from framekit.extraction.pipeline import build_router
from framekit.extraction.runner import LocalRunner, RateLimiter, Spend
from framekit.extraction.schemas.manifest import DatasetManifest
from framekit.extraction.schemas.plan import BudgetSettings
from framekit.models.cache import FileResponseStore


def lens(data):
    data = copy.deepcopy(data)
    data["lens"]["prompts"]["interpretation"] = "x"
    return LensSpec.from_mapping(data)


def space(*lenses, **extra):
    return SearchSpace.from_mapping(
        {
            "lenses": [{"spec": spec, "version": spec.version} for spec in lenses],
            "resolution": {"moment_length_s": {"min": 3, "max": 6}},
            **extra,
        }
    )


def corpus(rights="byo_media", **profile):
    return Corpus.from_mapping(
        {
            "profile": {"name": "demo", "rights_mode": rights, **profile},
            "units": [{"unit_id": "u1", "path": "a.mp4"}, {"unit_id": "u2", "path": "b.mp4"}],
            "manifest": [{"unit_id": "u1", "excluded_ranges": [[0, 2]]}],
        }
    )


def plan(**overrides):
    return ExtractionPlan.from_mapping(base_plan() | overrides)


def router():
    return build_router(None, LocalRunner(), None, sources())


def resolve(space_, corpus_=None, plan_=None):
    return resolve_run(space_, corpus_ or corpus(), plan_ or plan(), router=router())


def problems_of(*args):
    with pytest.raises(ConfigError) as info:
        resolve(*args)
    return "\n".join(info.value.problems)


def test_derivation_rules_fill_the_plan_and_are_recorded():
    run = resolve(space(lens(PACE_LENS), evidence={"modules": ["sound"]}))
    assert run.modules == ("core", "timing", "sound")
    assert run.beats == (0.5, 6.0)
    rules = {d["rule"] for d in run.derivations}
    assert rules == {"scope", "modules", "beats", "lenses", "phrasing_style"}
    assert run.resolved_plan["perception"]["modules"] == ["core", "timing", "sound"]
    detail = run.resolved_plan["vision"]["vision_profiles"]["detail"]
    assert detail["frame_selection"]["candidates"]["strategy"] == "uniform@1"  # pinned


def test_a_plan_may_add_modules_but_never_drop_one():
    run = resolve(space(lens(PACE_LENS)), None, plan(perception={"modules": ["camera"]}))
    assert run.modules == ("core", "camera", "timing")


def test_scope_narrows_and_excluded_ranges_come_from_the_corpus():
    run = resolve(space(lens(COLOUR_LENS), scope={"units": {"exclude": ["u2"]}}))
    assert [u.unit_id for u in run.units] == ["u1"]
    assert run.units[0].excluded_ranges == ((0.0, 2.0),)
    text = problems_of(space(lens(COLOUR_LENS), scope={"units": {"include": ["u9"]}}))
    assert "'u9' is not in the corpus" in text


def test_beats_longer_than_the_finest_moment_are_refused():
    text = problems_of(space(lens(COLOUR_LENS)), None, plan(perception={"beats": {"max_s": 9}}))
    assert "longer than the search space's longest moment" in text


def test_every_role_the_run_uses_needs_a_known_model():
    data = base_plan()
    data["models"] = {"L3a": {"model_id": "nope"}}
    text = problems_of(space(lens(COLOUR_LENS)), None, ExtractionPlan.from_mapping(data))
    assert "models.L2: this run needs a model for L2 (or set segmentation.method" in text
    assert "models.L3a.model_id" in text and "nope" in text
    assert "models.L3b: this run needs a model" in text
    assert "models.embed_text: this run needs a model for embed_text (or set embedding" in text


def test_frame_sending_roles_need_a_vision_model_within_its_limits():
    models = dict(base_plan()["models"], L3a={"model_id": "writer"})
    text = problems_of(space(lens(COLOUR_LENS)), None, plan(models=models))
    assert "writer-1 doesn't take images" in text
    profiles = base_plan()["vision_profiles"]
    profiles["detail"]["frame_selection"]["budget"]["max_frames_per_scene"] = 80
    text = problems_of(space(lens(COLOUR_LENS)), None, plan(vision_profiles=profiles))
    assert "the model accepts at most 50" in text


def test_native_video_and_image_embeddings_are_refused_until_built():
    profiles = base_plan()["vision_profiles"]
    profiles["overview"] = {
        "version": 1,
        "frame_selection": {"candidates": {"strategy": "uniform", "fps": 1}},
        "image": {"max_side_px": 160},
        "delivery": {"mode": "native_video"},
    }
    text = problems_of(
        space(lens(COLOUR_LENS)), None, plan(vision_profiles=profiles, embedding={"image": {}})
    )
    assert "native video needs the video_understand port" in text
    assert "image embeddings need an embed_image port" in text


def test_embeddings_need_an_adapter_that_serves_them():
    models = dict(base_plan()["models"], embed_text={"model_id": "claude-haiku-4-5"})
    text = problems_of(space(lens(COLOUR_LENS)), None, plan(models=models))
    assert "AnthropicAdapter (provider anthropic) serves no embeddings" in text


def test_unknown_detectors_and_speech_without_a_detector():
    text = problems_of(
        space(lens(COLOUR_LENS), evidence={"modules": ["speech"]}),
        None,
        plan(signals={"shots": {"detector": "transnet"}}),
    )
    assert "signals.shots.detector: no shots detector 'transnet'" in text
    assert "speech needs a speech detector" in text


def test_local_only_frames_refuse_a_remote_provider():
    models = dict(base_plan()["models"], L3a={"model_id": "claude-haiku-4-5"})
    text = problems_of(space(lens(COLOUR_LENS)), None, plan(models=models))
    assert "governance.model_inputs is local_only, but models.L3a" in text


def test_byo_media_keeps_vectors_and_training_local():
    text = problems_of(
        space(lens(COLOUR_LENS)),
        None,
        plan(governance={"model_inputs": "local_only", "training_use": "per_licence"}),
    )
    assert "governance.training_use: a byo_media corpus keeps this local_only" in text


def test_a_hosted_corpus_may_send_frames_by_default():
    run = resolve(space(lens(COLOUR_LENS)), corpus("hosted"), plan(governance={}))
    assert run.model_inputs == "provider_transient"


def test_lens_only_differences_keep_the_fairness_digest():
    one = resolve(space(lens(COLOUR_LENS)))
    two = resolve(space(lens(COLOUR_LENS), lens(PACE_LENS)))
    assert one.plan_digest != two.plan_digest  # derived modules and lenses differ...
    assert one.fairness_digest == two.fairness_digest  # ...and only those
    other = resolve(space(lens(COLOUR_LENS)), None, plan(normalize={"fps": 6, "height_px": 240}))
    assert other.fairness_digest != one.fairness_digest
    budget = resolve(space(lens(COLOUR_LENS)), None, plan(budget={"max_cost_usd": 3}))
    assert budget.content_digest == one.content_digest  # a budget changes nothing computed


# --- runner pieces ---------------------------------------------------------------------


def test_the_budget_stops_new_work_and_fails_calls_only_with_stop():
    from decimal import Decimal

    from framekit.extraction.runner import BudgetExceeded

    gentle = Spend(BudgetSettings(max_cost_usd=1, on_exceed="finish_unit_then_stop"))
    gentle.add("L3a", Decimal("1.5"))
    gentle.check("L3a")  # units in progress finish
    assert gentle.stopped and "budget reached" in gentle.reason
    strict = Spend(BudgetSettings(per_stage_usd={"L3b": 0.5}, on_exceed="stop"))
    strict.add("L3b", Decimal("0.6"))
    strict.check("L3a")  # another stage still runs
    with pytest.raises(BudgetExceeded, match="L3b"):
        strict.check("L3b")


def test_rate_limits_must_be_positive():
    with pytest.raises(ValueError, match="must be > 0"):
        RateLimiter({"hive": 0})


# --- the dataset files -----------------------------------------------------------------


def test_tables_have_fixed_schemas():
    with pytest.raises(ValueError, match="unknown columns"):
        to_table("scenes", [{"scene_id": "s", "colour": "red"}])
    table = to_table("annotations", [{"field": "f", "value": {"a": 1}}])
    assert table.column("value").to_pylist() == ['{"a":1}']  # free-form values are JSON


def manifest(**extra):
    fields = {
        "schema_version": 1,
        "dataset_id": "ds_x",
        "created_at": "now",
        "framekit_version": "0",
        "complete": True,
        "search_space": {},
        "corpus": {},
        "plan": {},
    }
    return DatasetManifest(**(fields | extra))


@pytest.mark.parametrize("partition_by", ["unit_id", "none"])
def test_tables_round_trip_in_either_layout(tmp_path, partition_by):
    rows = {
        "scenes": [
            {"scene_id": f"u{k}_s000", "unit_id": f"u{k}", "index": 0, "start": 0.0, "end": 5.0}
            for k in (1, 2)
        ]
    }
    written = write_tables(
        tmp_path, rows, include=["scenes", "qc"], partition_by=partition_by, compression="zstd"
    )
    write_manifest(tmp_path, manifest(tables=written))
    dataset = Dataset.open(tmp_path, verify=True)
    assert dataset.scenes.num_rows == 2 and dataset.qc.num_rows == 0
    assert dataset.qc.schema == SCHEMAS["qc"]
    files = [f.path for f in written["scenes"].files]
    expected = (
        ["tables/scenes/unit-u1.parquet", "tables/scenes/unit-u2.parquet"]
        if partition_by == "unit_id"
        else ["tables/scenes.parquet"]
    )
    assert files == expected


def test_a_changed_file_or_an_unknown_schema_version_is_refused(tmp_path):
    written = write_tables(
        tmp_path, {"scenes": []}, include=["scenes"], partition_by="none", compression="none"
    )
    write_manifest(tmp_path, manifest(tables=written))
    import pyarrow.parquet as pq

    pq.write_table(pa.table({"x": [1]}), tmp_path / "tables" / "scenes.parquet")
    with pytest.raises(ValueError, match="changed since it was written"):
        Dataset.open(tmp_path, verify=True)
    write_manifest(tmp_path, manifest(schema_version=9))
    with pytest.raises(ValueError, match="schema version 9"):
        Dataset.open(tmp_path)
    with pytest.raises(FileNotFoundError, match="not a finished dataset"):
        Dataset.open(Path(tmp_path) / "elsewhere")


# --- the file store's class layout ------------------------------------------------------


async def test_the_file_store_counts_and_wipes_by_class(tmp_path):
    store = FileResponseStore(tmp_path)
    await store.put("llmcache:v1:L3a:aaa", b"x" * 10, ttl_s=60)
    await store.put("llmcache:v1:L3b:bbb", b"y", ttl_s=60)
    await store.put("llmcache:v1:L3b:ccc", b"z", ttl_s=-1)  # already expired
    await store.put("anything else", b"w", ttl_s=60)
    stats = store.stats()
    assert stats["v1/L3a"]["entries"] == 1
    assert stats["v1/L3b"] == {"entries": 2, "bytes": 2 * (8 + 1), "expired": 1}
    assert store.clear(cache_class="L3b") == 2
    assert await store.get("llmcache:v1:L3a:aaa") == b"x" * 10
    assert set(store.stats()) == {"_other/_other", "v1/L3a"}


async def test_cache_class_names_never_escape_the_folder(tmp_path):
    store = FileResponseStore(tmp_path / "cache")
    await store.put("llmcache:..:../../x:key", b"v", ttl_s=60)
    assert all(p.is_relative_to(tmp_path / "cache") for p in tmp_path.rglob("*") if p.is_file())
    assert await store.get("llmcache:..:../../x:key") == b"v"
