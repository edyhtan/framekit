"""The benchmark harness (DESIGN.md §2.4, §6.2): compare search spaces with a pinned plan,
or plans with a pinned search space, on one corpus and one fixed gold set.

- **Fairness first:** with search spaces as the variants, every resolved plan must be the
  same except where the derivation rules wrote (`fairness_digest`); with plans as the
  variants, the search space is one object. Gold queries must never repeat a search
  space's example queries.
- **Shared work:** every variant runs with the same work folder, so tasks with the same
  inputs (L0–L3a when only lenses differ) run once.
- **Scoring is a plug-in:** `scorer(dataset, gold, context) -> metrics`. The default,
  `baseline_retriever`, embeds each gold query with the dataset's own text-embedding model,
  ranks scenes by their best cosine similarity (summary or phrasings), and reports recall@k,
  the mean reciprocal rank and the gold moment's rank. It's neutral and the same for every
  variant, not the real search agent; consumers plug in theirs.
"""

import asyncio
import dataclasses
import json
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from framekit.contracts import EmbeddingRequest, ResponseStore
from framekit.extraction.dataset import Dataset
from framekit.extraction.pipeline import build_router, extract_async
from framekit.extraction.resolve import resolve_run
from framekit.extraction.runner import LocalRunner, atomic_write, ensure_private_dir
from framekit.extraction.schemas.common import ConfigError
from framekit.extraction.schemas.corpus import Corpus
from framekit.extraction.schemas.gold import GoldSet, overlapping_queries
from framekit.extraction.schemas.plan import ExtractionPlan
from framekit.extraction.schemas.space import SearchSpace
from framekit.models import ModelSource
from framekit.models.cache import CacheSettings
from framekit.models.registry import ModelRegistry

Embed = Callable[[list[str]], Awaitable[list[list[float]]]]


@dataclass(frozen=True)
class ScoringContext:
    """What a scorer may use besides the dataset: `embed(texts)` with the dataset's own
    text-embedding model (the same model for every variant)."""

    embed: Embed


class Scorer(Protocol):
    name: str

    async def __call__(
        self, dataset: Dataset, gold: GoldSet, context: ScoringContext
    ) -> dict[str, float]: ...


def overlap(a: tuple[float, float], gold: tuple[float, float]) -> float:
    """How much of the gold moment a segment covers, 0–1."""
    inside = min(a[1], gold[1]) - max(a[0], gold[0])
    return max(0.0, inside) / (gold[1] - gold[0])


@dataclass(frozen=True)
class BaselineRetriever:
    ks: tuple[int, ...] = (1, 5, 10)
    name: str = "baseline_retriever"

    async def __call__(
        self, dataset: Dataset, gold: GoldSet, context: ScoringContext
    ) -> dict[str, float]:
        scenes = dataset.rows("scenes")
        vectors = dataset.rows("vectors") if "vectors" in dataset.table_names else []
        if not vectors:
            raise ValueError("the dataset has no vectors: the baseline retriever needs them")
        phrasing_scene = {
            p["phrasing_id"]: p["scene_id"]
            for p in (dataset.rows("phrasings") if "phrasings" in dataset.table_names else [])
        }
        owners, matrix = [], []
        for row in vectors:
            scene_id = (
                row["target_id"]
                if row["target_type"] == "scene"
                else (phrasing_scene.get(row["target_id"]))
            )
            if scene_id is not None:
                owners.append(scene_id)
                matrix.append(row["vector"])
        index = {s["scene_id"]: s for s in scenes}
        vectors_np = _unit_rows(np.asarray(matrix, dtype=np.float64))
        queries = [q for q in gold.queries if q.moments]
        if not queries:
            return {"queries_scored": 0.0}
        embedded = _unit_rows(np.asarray(await context.embed([q.query for q in queries])))
        similarity = embedded @ vectors_np.T  # queries x vectors
        ranks: list[int | None] = []
        for qi, query in enumerate(queries):
            best: dict[str, float] = {}
            for vi, scene_id in enumerate(owners):
                score = float(similarity[qi, vi])
                if score > best.get(scene_id, -2.0):
                    best[scene_id] = score
            ordered = sorted(best, key=lambda s: (-best[s], s))
            rank = None
            for position, scene_id in enumerate(ordered, 1):
                scene = index[scene_id]
                if any(
                    scene["unit_id"] == m.unit_id
                    and overlap((scene["start"], scene["end"]), (m.start, m.end))
                    >= gold.match.min_overlap
                    for m in query.moments
                ):
                    rank = position
                    break
            ranks.append(rank)
        found = [r for r in ranks if r is not None]
        metrics: dict[str, float] = {"queries_scored": float(len(queries))}
        for k in self.ks:
            metrics[f"recall@{k}"] = sum(1 for r in found if r <= k) / len(queries)
        metrics["mrr"] = sum(1 / r for r in found) / len(queries)
        metrics["median_rank"] = float(np.median(found)) if found else float("nan")
        metrics["not_found"] = float(len(queries) - len(found))
        return metrics


def _unit_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0, 1, norms)


baseline_retriever = BaselineRetriever()


@dataclass
class VariantResult:
    label: str
    path: str | None
    metrics: dict[str, float] = field(default_factory=dict)
    cost: dict[str, Any] = field(default_factory=dict)
    complete: bool = False
    error: str | None = None


@dataclass
class Report:
    kind: str  # "search_spaces" (plan pinned) or "plans" (search space pinned)
    corpus: str
    gold_digest: str
    scorer: str
    pinned_digest: str
    variants: list[VariantResult]

    def to_mapping(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def table(self) -> str:
        """A Markdown table: one row per variant."""
        keys = sorted({k for v in self.variants for k in v.metrics})
        head = ["variant", *keys, "cost USD (est.)", "tasks run / reused", "complete"]
        lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        for v in self.variants:
            tasks = v.cost.get("tasks", {})
            cells = [
                v.label,
                *(f"{v.metrics[k]:.3f}" if k in v.metrics else "" for k in keys),
                v.cost.get("this_run_usd", ""),
                f"{tasks.get('computed', '')} / {tasks.get('reused', '')}",
                "yes" if v.complete else f"no{': ' + v.error if v.error else ''}",
            ]
            lines.append("| " + " | ".join(str(c) for c in cells) + " |")
        return "\n".join(lines)

    def write(self, folder: str | Path) -> None:
        folder = ensure_private_dir(Path(folder))
        atomic_write(folder / "report.json", json.dumps(self.to_mapping(), indent=1))
        atomic_write(
            folder / "report.md",
            f"# Benchmark: {self.kind}\n\nCorpus `{self.corpus}`, scorer `{self.scorer}`, "
            f"gold `{self.gold_digest[:12]}`, pinned `{self.pinned_digest[:12]}`.\n\n"
            + self.table()
            + "\n",
        )


async def benchmark_async(
    variants: Mapping[str, SearchSpace] | Mapping[str, ExtractionPlan],
    corpus: Corpus,
    *,
    gold: GoldSet,
    out: str | Path,
    plan: ExtractionPlan | None = None,
    space: SearchSpace | None = None,
    scorer: Scorer = baseline_retriever,
    runner: LocalRunner | None = None,
    cache: ResponseStore | None = None,
    cache_settings: CacheSettings | None = None,
    sources: Iterable[ModelSource] = (),
    registry: ModelRegistry | None = None,
) -> Report:
    """`benchmark` for callers already in an event loop."""
    if not variants:
        raise ValueError("no variants to compare")
    values = list(variants.values())
    sources = list(sources)
    if all(isinstance(v, SearchSpace) for v in values):
        kind = "search_spaces"
        if space is not None:
            raise ValueError("comparing search spaces: pin the plan (plan=...), not a space")
        pairs = [(label, v, plan) for label, v in variants.items()]
    elif all(isinstance(v, ExtractionPlan) for v in values):
        kind = "plans"
        if space is None:
            raise ValueError("comparing plans: pin the search space (space=...)")
        pairs = [(label, space, v) for label, v in variants.items()]
    else:
        raise TypeError("variants are all SearchSpace or all ExtractionPlan")
    for label in variants:
        if not label or "/" in label or "\\" in label or label.startswith("."):
            raise ValueError(f"variant label {label!r}: a plain folder name")

    problems = []
    if gold.corpus != corpus.profile.name:
        problems.append(f"gold.corpus is {gold.corpus!r}, the corpus is {corpus.profile.name!r}")
    for label, s, _ in pairs:
        problems += [f"{label}: {p}" for p in overlapping_queries(s.query_style.examples, gold)]
    runner = runner or LocalRunner()
    lookup = build_router(None, runner, registry, sources)
    resolved = {label: resolve_run(s, corpus, p, router=lookup) for label, s, p in pairs}
    if kind == "search_spaces":
        digests = {r.fairness_digest for r in resolved.values()}
        if len(digests) > 1:
            problems.append(
                "the variants' resolved plans differ beyond the derivation rules: compare "
                "search spaces with one pinned plan"
            )
        pinned = next(iter(resolved.values())).fairness_digest
    else:
        pinned = space.digest if space is not None else ""
    if isinstance(scorer, BaselineRetriever):
        models = {
            (r.roles["embed_text"].spec.model_id if "embed_text" in r.roles else None)
            for r in resolved.values()
        }
        if None in models:
            problems.append("baseline_retriever needs text vectors: set models.embed_text")
        elif len(models) > 1:
            problems.append(
                "baseline_retriever compares variants with one text-embedding model; the "
                f"variants use {sorted(models)}"
            )
    if problems:
        raise ConfigError(problems, "the benchmark")

    out = ensure_private_dir(Path(out).expanduser().resolve())
    shared = (
        runner
        if runner.work_dir is not None
        else dataclasses.replace(runner, work_dir=out / "work")
    )
    results = []
    for label, s, p in pairs:
        result = VariantResult(label, None)
        try:
            dataset = await extract_async(
                s,
                corpus,
                plan=p,
                out=out / "datasets" / label,
                runner=shared,
                cache=cache,
                cache_settings=cache_settings,
                sources=sources,
                registry=registry,
                label=label,
            )
        except Exception as exc:  # one failed variant shouldn't lose the others' results
            result.error = f"{type(exc).__name__}: {exc}"
            results.append(result)
            continue
        if not isinstance(dataset, Dataset):  # never: not a dry run
            raise TypeError("extract returned no dataset")
        result.path = str(dataset.path)
        result.cost = dict(dataset.manifest.cost)
        result.complete = dataset.manifest.complete
        try:
            result.metrics = await scorer(
                dataset, gold, _context(resolved[label], shared, registry, sources)
            )
        except Exception as exc:
            result.error = f"scoring: {type(exc).__name__}: {exc}"
        results.append(result)
    report = Report(
        kind=kind,
        corpus=corpus.profile.name,
        gold_digest=gold.digest,
        scorer=getattr(scorer, "name", type(scorer).__name__),
        pinned_digest=pinned,
        variants=results,
    )
    report.write(out)
    return report


def _context(
    run: Any, runner: LocalRunner, registry: ModelRegistry | None, sources: list[ModelSource]
) -> ScoringContext:
    binding = run.roles.get("embed_text")

    async def embed(texts: list[str]) -> list[list[float]]:
        if binding is None:
            raise ValueError("no text-embedding model in the plan")
        router = build_router(run, runner, registry, sources)
        try:
            result = await router.embed(EmbeddingRequest(tuple(texts), model_id=binding.model_id))
        finally:
            await router.aclose()
        return [list(v) for v in result.vectors]

    return ScoringContext(embed=embed)


def benchmark(
    variants: Mapping[str, SearchSpace] | Mapping[str, ExtractionPlan],
    corpus: Corpus,
    *,
    gold: GoldSet,
    out: str | Path,
    plan: ExtractionPlan | None = None,
    space: SearchSpace | None = None,
    scorer: Scorer = baseline_retriever,
    runner: LocalRunner | None = None,
    cache: ResponseStore | None = None,
    cache_settings: CacheSettings | None = None,
    sources: Iterable[ModelSource] = (),
    registry: ModelRegistry | None = None,
) -> Report:
    """Run every variant through `extract` and score it against the gold set; writes the
    datasets and `report.json` / `report.md` under `out`. One side varies: search spaces
    with `plan=` pinned, or plans with `space=` pinned."""
    return asyncio.run(
        benchmark_async(
            variants,
            corpus,
            gold=gold,
            out=out,
            plan=plan,
            space=space,
            scorer=scorer,
            runner=runner,
            cache=cache,
            cache_settings=cache_settings,
            sources=sources,
            registry=registry,
        )
    )


__all__ = [
    "BaselineRetriever",
    "Report",
    "Scorer",
    "ScoringContext",
    "VariantResult",
    "baseline_retriever",
    "benchmark",
    "benchmark_async",
    "overlap",
]
