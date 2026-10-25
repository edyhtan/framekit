"""`extract(A, V, plan)` (DESIGN.md §2, §4): resolve, check, run the stages per unit, write
the dataset.

1. **Resolve** A, V and the plan (derivation rules, every check) — `ConfigError` before
   any spend.
2. **Preflight:** probe every unit's source (no decoding): its facts and content
   fingerprint, the expected duration, long units, and each frame profile against the
   analysis copy the unit will get. A dry run stops here with the cost estimate.
3. **Run** each unit: L0 → L1 → L2 → L3a per scene → L3b per scene, lens and run → embed;
   L4's checks on every result. Units run in parallel; every step is a task keyed by its
   input hash, so a rerun reuses everything unchanged.
4. **Write** the tables, then the manifest.
"""

import asyncio
import os
from collections import defaultdict
from collections.abc import Awaitable, Iterable, Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

from framekit import __version__
from framekit.contracts import ModelCallError, ResponseStore
from framekit.extraction.dataset import Dataset, prepare_output, write_manifest, write_tables
from framekit.extraction.estimate import CostEstimate, estimate
from framekit.extraction.media.ffmpeg import FFmpeg, MediaError, MediaInfo, copy_size
from framekit.extraction.prompts import PERCEIVE_PROMPT
from framekit.extraction.records import RECORD_SCHEMA
from framekit.extraction.resolve import ResolvedRun, ScopedUnit, resolve_run
from framekit.extraction.runner import (
    BudgetExceeded,
    Execution,
    LocalRunner,
    Spend,
    TaskStore,
    ensure_private_dir,
    now,
    total_cost,
)
from framekit.extraction.schemas.common import ConfigError, digest, plain
from framekit.extraction.schemas.corpus import Corpus
from framekit.extraction.schemas.manifest import SCHEMA_VERSION, DatasetManifest, UnitStatus
from framekit.extraction.schemas.plan import STAGES, ExtractionPlan
from framekit.extraction.schemas.space import SearchSpace
from framekit.extraction.segmenting import fixed_windows, processable_spans, shots_from_cuts
from framekit.extraction.stages.context import SceneRow, StageContext, quick_fingerprint
from framekit.extraction.stages.embed import embed_texts
from framekit.extraction.stages.lens_pass import agreement, lens_pass, lens_rows
from framekit.extraction.stages.normalize import normalize
from framekit.extraction.stages.perceive import beat_rows, perceive
from framekit.extraction.stages.qc import (
    agreement_qc,
    lens_qc,
    perception_qc,
    qc_row,
    segmentation_qc,
)
from framekit.extraction.stages.segment import segment
from framekit.extraction.stages.signals import detect, signal_rows
from framekit.frames import SourceLimits, check_against_source
from framekit.models import ConversationRouter, ModelSettings, ModelSource
from framekit.models.cache import CacheSettings, with_cache
from framekit.models.registry import ModelRegistry

DURATION_TOLERANCE = (2.0, 0.02)  # a source within 2 s or 2% of the expected duration matches
MODEL_STAGES = {"L2", "L3a", "L3b", "embed"}


@dataclass
class UnitResult:
    unit_id: str
    status: str = "complete"  # complete | partial | failed | excluded | skipped
    reason: str = ""
    rows: dict[str, list[dict[str, Any]]] = field(default_factory=lambda: defaultdict(list))
    records: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    source_fingerprint: str | None = None

    def problem(self, status: str, reason: str) -> None:
        """Worsen the status (never improve it) and keep the first reason."""
        order = ("complete", "partial", "failed")
        if self.status in order and order.index(status) > order.index(self.status):
            self.status = status
        if not self.reason:
            self.reason = reason


async def _safe[T](awaitable: Awaitable[T]) -> tuple[T | None, BaseException | None]:
    """The result, or the error a step may end with (model, media, budget) without
    stopping the unit's other steps."""
    try:
        return await awaitable, None
    except (ModelCallError, MediaError, BudgetExceeded) as exc:
        return None, exc


def _record_row(record: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "run_id",
        "stage",
        "unit_id",
        "target_id",
        "input_hash",
        "status",
        "attempts",
        "model_id",
        "provider",
        "adapter_version",
        "started_at",
        "finished_at",
        "latency_ms",
        "input_tokens",
        "output_tokens",
        "reasoning_tokens",
        "cost_usd",
        "cached",
        "reused",
        "lens",
        "run_index",
        "error",
    )
    return {key: record.get(key) for key in keys}


def _neighbors(scenes: list[SceneRow], index: int, count: int) -> tuple[str, ...]:
    if count == 0:
        return ()
    before = " | ".join(s.summary for s in scenes[max(0, index - count) : index] if s.summary)
    after = " | ".join(s.summary for s in scenes[index + 1 : index + 1 + count] if s.summary)
    return (before, after)


async def process_unit(
    ctx: StageContext, unit: ScopedUnit, source: MediaInfo, fingerprint: str
) -> UnitResult:
    run = ctx.run
    plan = run.plan
    checks = plan.quality_control.checks
    then = plan.quality_control.invalid_output.then
    result = UnitResult(unit.unit_id, source_fingerprint=fingerprint)
    rows = result.rows
    try:
        media, record = await normalize(ctx, unit, source, fingerprint)
        result.records.append(record)
        signals, record = await detect(ctx, media)
        result.records.append(record)
    except (MediaError, BudgetExceeded) as exc:
        result.problem("failed", f"{type(exc).__name__}: {exc}")
        rows["units"].append(_unit_row(unit, source, None, fingerprint, result, None))
        return result

    rows["signals"] += signal_rows(unit.unit_id, signals, record.get("detectors") or {})
    duration = media.copy.duration_s
    rows["shots"] += [
        {
            "shot_id": f"{unit.unit_id}_sh{i:04d}",
            "unit_id": unit.unit_id,
            "index": i,
            "start": a,
            "end": b,
        }
        for i, (a, b) in enumerate(shots_from_cuts(signals["shots"], duration))
    ]
    if run.windows is not None:
        window_s, stride_s = run.windows
        spans = processable_spans(duration, unit.excluded_ranges)
        rows["windows"] += [
            {
                "window_id": f"{unit.unit_id}_w{i:05d}",
                "unit_id": unit.unit_id,
                "index": i,
                "start": a,
                "end": b,
            }
            for i, (a, b) in enumerate(fixed_windows(spans, window_s, stride_s))
        ]

    segmented, error = await _safe(segment(ctx, media, signals))
    if error is not None or segmented is None:
        result.problem("failed", f"L2: {type(error).__name__}: {error}")
        rows["units"].append(_unit_row(unit, source, media.copy, fingerprint, result, None))
        return result
    output, record = segmented
    if record is not None:
        result.records.append(record)
    result.notes += [f"{unit.unit_id} L2: {n}" for n in output.get("notes", ())]
    rows["frames_sent"] += output.get("frames", [])
    rows["qc"] += segmentation_qc(unit.unit_id, output, checks, then)
    synopsis = output.get("synopsis")
    scenes = [
        SceneRow(f"{unit.unit_id}_s{i:03d}", unit.unit_id, i, float(a), float(b), str(s))
        for i, (a, b, s) in enumerate(output["scenes"])
    ]

    # L3a, every scene at once (the runner bounds the calls in flight)
    perceptions = await asyncio.gather(
        *(
            _safe(
                perceive(
                    ctx,
                    media,
                    signals,
                    scene,
                    synopsis,
                    _neighbors(scenes, i, plan.perception.context.neighbors),
                )
            )
            for i, scene in enumerate(scenes)
        )
    )
    records_by_scene: dict[str, dict[str, Any]] = {}
    for scene, (found, error) in zip(scenes, perceptions, strict=True):
        status = "ok"
        if error is not None or found is None:
            result.problem("partial", f"L3a {scene.scene_id}: {type(error).__name__}: {error}")
            rows["qc"].append(
                qc_row(
                    "schema",
                    target_type="scene",
                    target_id=scene.scene_id,
                    unit_id=unit.unit_id,
                    scene_id=scene.scene_id,
                    status="flag",
                    detail=f"no perception: {error}",
                )
            )
            status = "missing"
            rows["perception"].append(_perception_row(run, scene, None, status, None))
        else:
            out, record = found
            result.records.append(record)
            rows["frames_sent"] += out.get("frames", [])
            rows["qc"] += perception_qc(scene, out, checks, then)
            result.notes += [f"{scene.scene_id} L3a: {n}" for n in out.get("notes", ())]
            status = out["status"]
            rows["perception"].append(
                _perception_row(run, scene, out.get("record"), status, record)
            )
            if out.get("record") is not None:
                records_by_scene[scene.scene_id] = out["record"]
                rows["beats"] += beat_rows(scene, out["record"])
        rows["scenes"].append(
            {
                "scene_id": scene.scene_id,
                "unit_id": unit.unit_id,
                "index": scene.index,
                "start": scene.start,
                "end": scene.end,
                "summary": scene.summary,
                "method": output.get("method"),
                "status": status,
            }
        )

    # L3b: every scene with a record × every lens × every self-consistency run
    runs = plan.lens_passes.self_consistency.runs
    jobs = [
        (scene, lens, k)
        for scene in scenes
        if scene.scene_id in records_by_scene
        for lens in run.lenses
        for k in range(runs)
    ]
    passes = await asyncio.gather(
        *(
            _safe(lens_pass(ctx, scene, records_by_scene[scene.scene_id], synopsis, lens, k))
            for scene, lens, k in jobs
        )
    )
    by_scene_lens: dict[tuple[str, str], list[tuple[int, dict[str, Any], dict[str, Any]]]] = (
        defaultdict(list)
    )
    for (scene, lens, k), (found, error) in zip(jobs, passes, strict=True):
        if error is not None or found is None:
            result.problem(
                "partial", f"L3b {scene.scene_id} {lens.key}: {type(error).__name__}: {error}"
            )
            rows["qc"].append(
                qc_row(
                    "schema",
                    target_type="scene",
                    target_id=scene.scene_id,
                    unit_id=unit.unit_id,
                    scene_id=scene.scene_id,
                    lens=lens.name,
                    run_index=k,
                    status="flag",
                    detail=f"no lens pass: {error}",
                )
            )
            continue
        out, record = found
        result.records.append(record)
        rows["qc"] += lens_qc(scene, lens, k, out, checks, then)
        if out.get("annotation") is not None:
            by_scene_lens[(scene.scene_id, lens.name)].append((k, out["annotation"], record))
    threshold = plan.quality_control.thresholds.agreement_min
    on_disagreement = plan.lens_passes.self_consistency.on_disagreement
    scene_by_id = {s.scene_id: s for s in scenes}
    lens_by_name = {lens.name: lens for lens in run.lenses}
    for (scene_id, lens_name), done in by_scene_lens.items():
        scene, lens = scene_by_id[scene_id], lens_by_name[lens_name]
        score = agreement(lens, [annotation for _, annotation, _ in done])
        weight = 1.0
        if score is not None and "agreement" in checks:
            rows["qc"].append(agreement_qc(scene, lens, score, threshold, on_disagreement))
            if score < threshold and on_disagreement == "down_weight":
                weight = score
        for k, annotation, record in done:
            annotation_list, phrasing_list = lens_rows(
                scene, lens, k, annotation, record["run_id"], weight
            )
            rows["annotations"] += annotation_list
            rows["phrasings"] += phrasing_list

    # embed: scene summaries (the record's, else L2's) and phrasings
    if "embed_text" in run.roles and plan.embedding.text is not None:
        inputs = plan.embedding.text.inputs
        items: list[tuple[str, str, str]] = []
        if "scene_summary" in inputs:
            for scene in scenes:
                text = (records_by_scene.get(scene.scene_id) or {}).get("summary") or scene.summary
                if text:
                    items.append(("scene", scene.scene_id, text))
        if "phrasings" in inputs:
            items += [
                ("phrasing", p["phrasing_id"], p["text"]) for p in rows["phrasings"] if p["text"]
            ]
        embedded, error = await _safe(embed_texts(ctx, unit.unit_id, items))
        if error is not None or embedded is None:
            result.problem("partial", f"embed: {type(error).__name__}: {error}")
        else:
            vectors, records = embedded
            rows["vectors"] += vectors
            result.records += records

    rows["units"].append(_unit_row(unit, source, media.copy, fingerprint, result, synopsis))
    return result


def _perception_row(
    run: ResolvedRun,
    scene: SceneRow,
    record: dict[str, Any] | None,
    status: str,
    run_record: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "scene_id": scene.scene_id,
        "unit_id": scene.unit_id,
        "record_schema": RECORD_SCHEMA,
        "modules": list(run.modules),
        "summary": (record or {}).get("summary"),
        "record": record,
        "status": status,
        "run_id": run_record.get("run_id") if run_record else None,
        "model_id": run_record.get("model_id") if run_record else None,
        "prompt_version": PERCEIVE_PROMPT,
    }


def _unit_row(
    unit: ScopedUnit,
    source: MediaInfo | None,
    copy: MediaInfo | None,
    fingerprint: str | None,
    result: UnitResult,
    synopsis: str | None,
) -> dict[str, Any]:
    return {
        "unit_id": unit.unit_id,
        "status": result.status,
        "reason": result.reason,
        "file_name": Path(unit.unit.path).name,
        "source_fingerprint": fingerprint,
        "duration_s": source.duration_s if source else None,
        "width": source.width if source else None,
        "height": source.height if source else None,
        "fps": source.fps if source else None,
        "has_audio": source.has_audio if source else None,
        "copy_width": copy.width if copy else None,
        "copy_height": copy.height if copy else None,
        "copy_fps": copy.fps if copy else None,
        "metadata": plain(unit.unit.metadata),
        "excluded_ranges": [list(r) for r in unit.excluded_ranges],
        "synopsis": synopsis,
    }


@dataclass(frozen=True)
class Preflight:
    probes: dict[str, MediaInfo]
    fingerprints: dict[str, str]
    excluded: dict[str, str]  # unit_id -> reason (unreadable, unmatched, too long)


async def preflight(run: ResolvedRun, ffmpeg: FFmpeg, slots: asyncio.Semaphore) -> Preflight:
    """Probe every unit's source; raise ConfigError when a frame profile asks for more than
    a unit's analysis copy will have (nothing is upsampled)."""
    plan = run.plan

    async def one(unit: ScopedUnit) -> tuple[str, MediaInfo | None, str | None, str]:
        path = Path(unit.unit.path)
        try:
            async with slots:
                info = await ffmpeg.probe(path)
        except MediaError as exc:
            return unit.unit_id, None, None, str(exc)
        return unit.unit_id, info, quick_fingerprint(path), ""

    found = await asyncio.gather(*(one(u) for u in run.units))
    probes, fingerprints, excluded = {}, {}, {}
    problems = []
    units = {u.unit_id: u for u in run.units}
    for unit_id, info, fingerprint, error in found:
        if info is None or fingerprint is None:
            excluded[unit_id] = f"unreadable: {error}"
            continue
        expected = units[unit_id].expected_duration_s
        if expected is not None:
            tolerance = max(DURATION_TOLERANCE[0], expected * DURATION_TOLERANCE[1])
            if abs(info.duration_s - expected) > tolerance:
                excluded[unit_id] = (
                    f"unmatched: {info.duration_s:.2f} s, the manifest expects {expected:.2f} s"
                )
                continue
        if info.duration_s > plan.long_units.enabled_over_s:
            excluded[unit_id] = (
                f"long unit ({info.duration_s:.0f} s over long_units.enabled_over_s = "
                f"{plan.long_units.enabled_over_s:g} s): chunked processing isn't built yet; "
                "raise enabled_over_s to process it in one pass"
            )
            continue
        width, height = copy_size(info, plan.normalize.height_px)
        limits = SourceLimits(plan.normalize.fps, width, height)
        for role in ("L2", "L3a"):
            if role in run.roles and role in run.vision.stages:
                problems += [
                    f"unit {unit_id}: {p}" for p in check_against_source(run.profile(role), limits)
                ]
        probes[unit_id] = info
        fingerprints[unit_id] = fingerprint
    if problems:
        raise ConfigError(problems, "the frame profiles against the analysis copies")
    return Preflight(probes, fingerprints, excluded)


def missing_keys(run: ResolvedRun, router: ConversationRouter) -> list[str]:
    """The roles whose provider key isn't in the environment (checked for presence only,
    never read further), so a run stops before any media work rather than at its first
    model call. A source with its own credential, or a keyless provider, needs none."""
    missing = []
    for role, binding in run.roles.items():
        source = router.sources.get(binding.model_id) if binding.source else None
        if source is not None and source.credential is not None:
            continue
        variable = binding.provider.key_env
        if variable and not os.environ.get(variable):
            missing.append(
                f"models.{role}: {variable} is not set (the key for provider "
                f"{binding.provider.name})"
            )
    return missing


def build_router(
    run: ResolvedRun | None,
    runner: LocalRunner,
    registry: ModelRegistry | None,
    sources: Iterable[ModelSource],
) -> ConversationRouter:
    cap = 256
    if run is not None:
        cap = max([b.max_tokens for b in run.roles.values()] + [cap])
    settings = ModelSettings(max_tokens_cap=cap, timeout_s=runner.model_timeout_s)
    return ConversationRouter(settings, registry=registry, sources=list(sources))


async def extract_async(
    space: SearchSpace,
    corpus: Corpus,
    *,
    out: str | Path,
    plan: ExtractionPlan | None = None,
    runner: LocalRunner | None = None,
    cache: ResponseStore | None = None,
    cache_settings: CacheSettings | None = None,
    dry_run: bool = False,
    sources: Iterable[ModelSource] = (),
    registry: ModelRegistry | None = None,
    rerun: Iterable[str] = (),
    label: str | None = None,
) -> Dataset | CostEstimate:
    """`extract` for callers already in an event loop."""
    runner = runner or LocalRunner()
    sources = list(sources)
    out = Path(out).expanduser().resolve()
    unknown = sorted(set(rerun) - set(STAGES))
    if unknown:
        raise ValueError(f"rerun: unknown stages {unknown} (stages: {', '.join(STAGES)})")
    lookup = build_router(None, runner, registry, sources)
    run = resolve_run(space, corpus, plan, router=lookup)
    keys = missing_keys(run, lookup)
    ffmpeg = FFmpeg(runner.ffmpeg)
    checked = await preflight(run, ffmpeg, asyncio.Semaphore(runner.media_slots()))
    if dry_run:
        found = estimate(run, checked.probes)
        found.notes += [f"{u} excluded: {why}" for u, why in sorted(checked.excluded.items())]
        found.notes += [f"a real run would stop here: {k}" for k in keys]
        return found
    if keys:
        raise ConfigError(keys, "the model keys")

    identity = {"search_space": space.digest, "corpus": corpus.digest, "plan": run.plan_digest}
    prepare_output(out, identity)
    work_dir = ensure_private_dir(Path(runner.work_dir or out.with_name(out.name + ".work")))
    router = build_router(run, runner, registry, sources)
    if cache is not None:
        port = with_cache(
            lambda: router, settings=cache_settings or CacheSettings(mode="record"), store=cache
        )()
    else:
        port = router
    spend = Spend(run.plan.budget)
    ex = Execution(runner, TaskStore(work_dir), port, spend, rerun=frozenset(rerun))
    ctx = StageContext(run, ex, ffmpeg, work_dir)
    unit_slots = asyncio.Semaphore(runner.media_slots())

    async def guarded(unit: ScopedUnit) -> UnitResult:
        if unit.unit_id in checked.excluded:
            result = UnitResult(
                unit.unit_id, status="excluded", reason=checked.excluded[unit.unit_id]
            )
            result.rows["units"].append(_unit_row(unit, None, None, None, result, None))
            return result
        async with unit_slots:
            if spend.stopped:
                result = UnitResult(unit.unit_id, status="skipped", reason=spend.reason)
                info = checked.probes[unit.unit_id]
                result.rows["units"].append(
                    _unit_row(unit, info, None, checked.fingerprints[unit.unit_id], result, None)
                )
                return result
            return await process_unit(
                ctx, unit, checked.probes[unit.unit_id], checked.fingerprints[unit.unit_id]
            )

    try:
        results = await asyncio.gather(*(guarded(u) for u in run.units))
    finally:
        await port.aclose()

    tables: dict[str, list[dict[str, Any]]] = defaultdict(list)
    records: list[dict[str, Any]] = []
    notes = list(run.notes)
    for result in results:
        for name, table_rows in result.rows.items():
            tables[name] += table_rows
        records += result.records
        notes += result.notes
    tables["runs"] = [_record_row(r) for r in records]
    formatting = run.plan.formatting
    written = write_tables(
        out,
        tables,
        include=formatting.tables,
        partition_by=formatting.layout.partition_by,
        compression=formatting.layout.compression,
    )
    per_stage: dict[str, Decimal] = defaultdict(Decimal)
    for record in records:
        per_stage[record["stage"]] += Decimal(str(record.get("cost_usd") or 0))
    manifest = DatasetManifest(
        schema_version=SCHEMA_VERSION,
        dataset_id="ds_" + digest(identity)[:20],
        created_at=now(),
        framekit_version=__version__,
        label=label,
        complete=all(r.status == "complete" for r in results),
        search_space={"verbatim": space.verbatim, "digest": space.digest},
        corpus={
            "name": corpus.profile.name,
            "digest": corpus.digest,
            "rights_mode": corpus.profile.rights_mode,
            "units": {u: f for u, f in checked.fingerprints.items()},
        },
        plan={
            "resolved": run.resolved_plan,
            "digest": run.plan_digest,
            "content_digest": run.content_digest,
            "fairness_digest": run.fairness_digest,
        },
        derivations=tuple(run.derivations),
        lenses=tuple(
            {"name": s.name, "version": s.version, "digest": s.digest} for s in run.lenses
        ),
        models={role: binding.manifest() for role, binding in run.roles.items()},
        vision_digest=run.vision.digest,
        governance={
            **plain(run.plan.governance),
            "rights_mode": corpus.profile.rights_mode,
            "model_inputs": run.model_inputs,
            "stays_local": corpus.profile.rights_mode == "byo_media",
        },
        units=tuple(
            UnitStatus(unit_id=r.unit_id, status=r.status, reason=r.reason) for r in results
        ),
        tables=written,
        cost={
            "estimated": True,
            "this_run_usd": str(spend.total),
            "total_usd": str(total_cost(records)),
            "per_stage_usd": {k: str(v) for k, v in sorted(per_stage.items())},
            "tasks": {"computed": ex.computed, "reused": ex.reused},
        },
        tools={"ffmpeg": await ffmpeg.version(), "execution": runner.options()},
        notes=tuple(notes),
    )
    write_manifest(out, manifest)
    return Dataset.open(out)


def extract(
    space: SearchSpace,
    corpus: Corpus,
    *,
    out: str | Path,
    plan: ExtractionPlan | None = None,
    runner: LocalRunner | None = None,
    cache: ResponseStore | None = None,
    cache_settings: CacheSettings | None = None,
    dry_run: bool = False,
    sources: Iterable[ModelSource] = (),
    registry: ModelRegistry | None = None,
    rerun: Iterable[str] = (),
    label: str | None = None,
) -> Dataset | CostEstimate:
    """A + V (+ plan) → a dataset in `out` (DESIGN.md §2). Returns the opened `Dataset`, or
    the `CostEstimate` with `dry_run=True` (nothing is run or written then).

    - `plan`: None for the defaults plus derivation rules; models are never defaulted, so a
      real run needs at least `models` in a plan.
    - `runner`: execution options (`LocalRunner`); never change the dataset.
    - `cache` + `cache_settings`: the model response cache (a store alone records).
    - `sources` / `registry`: models beyond the built-in catalog (`ModelSource`).
    - `rerun`: stages to run again even where their tasks already ran (e.g. `{"L3b"}`).
    - `label`: recorded in the manifest (the benchmark harness sets it).
    """
    return asyncio.run(
        extract_async(
            space,
            corpus,
            out=out,
            plan=plan,
            runner=runner,
            cache=cache,
            cache_settings=cache_settings,
            dry_run=dry_run,
            sources=sources,
            registry=registry,
            rerun=rerun,
            label=label,
        )
    )


__all__ = [
    "Preflight",
    "UnitResult",
    "build_router",
    "extract",
    "extract_async",
    "missing_keys",
    "preflight",
    "process_unit",
]
