"""The `framekit` command (docs/settings.md §2).

    framekit extract --space search-space.yaml --corpus corpus.yaml [--plan plan.yaml]
                     --out DIR [--work-dir DIR] [--concurrency N] [--rate-limit P=4/s]
                     [--provider-retries N] [--cache-dir DIR] [--cache-mode MODE|CLASS=MODE]
                     [--dry-run] [--rerun STAGE] [--ffmpeg PATH] [--sources MODULE:NAME]
    framekit bench   --corpus corpus.yaml --gold gold.yaml --out DIR
                     (--plan plan.yaml --space LABEL=space.yaml ...   compare search spaces)
                     (--space space.yaml --plan LABEL=plan.yaml ...   compare plans)
    framekit cache   stats|wipe --cache-dir DIR [--class CLASS]

Models beyond the built-in catalog come from your own code: `--sources my.module:SOURCES`
imports a list of `ModelSource`s (or a function returning one). Plan files only name
models; they never hold keys or endpoints.
"""

import argparse
import importlib
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from framekit import __version__

RATE_UNITS = {"s": 1.0, "m": 60.0, "h": 3600.0}


def parse_rate(text: str) -> tuple[str, float]:
    """`hive=4/s`, `anthropic=120/m` or `local=2` → (provider, requests per second)."""
    name, _, value = text.partition("=")
    if not name or not value:
        raise argparse.ArgumentTypeError(f"{text!r}: expected PROVIDER=RATE, e.g. hive=4/s")
    amount, _, unit = value.partition("/")
    try:
        rate = float(amount) / RATE_UNITS[unit or "s"]
    except (ValueError, KeyError):
        raise argparse.ArgumentTypeError(f"{text!r}: a rate is N, N/s, N/m or N/h") from None
    if rate <= 0:
        raise argparse.ArgumentTypeError(f"{text!r}: the rate must be > 0")
    return name, rate


def parse_cache_modes(values: Sequence[str]) -> tuple[str, dict[str, str]]:
    """`record` (the default mode) and `L3b=replay` (one class) → (mode, {class: mode})."""
    modes = ("off", "record", "replay")
    default, classes = "off", {}
    for value in values:
        name, sep, mode = value.partition("=")
        if not sep:
            name, mode = "", name
        if mode not in modes:
            raise SystemExit(f"--cache-mode {value!r}: the mode is one of {', '.join(modes)}")
        if name:
            classes[name] = mode
        else:
            default = mode
    return default, classes


def load_sources(reference: str | None) -> list[Any]:
    """`module:NAME`: a list of ModelSource, or a function returning one."""
    if not reference:
        return []
    module_name, _, attribute = reference.partition(":")
    if not module_name or not attribute:
        raise SystemExit(f"--sources {reference!r}: expected MODULE:NAME")
    sys.path.insert(0, str(Path.cwd()))
    found = getattr(importlib.import_module(module_name), attribute)
    sources = found() if callable(found) else found
    return list(sources)


def _labelled(values: Sequence[str], flag: str) -> dict[str, str]:
    pairs = {}
    for value in values:
        label, sep, path = value.partition("=")
        if not sep or not label or not path:
            raise SystemExit(f"{flag} {value!r}: expected LABEL=FILE when comparing")
        if label in pairs:
            raise SystemExit(f"{flag}: the label {label!r} is used twice")
        pairs[label] = path
    return pairs


def _execution(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("execution (speed and cost only, never the dataset)")
    group.add_argument("--work-dir", help="analysis copies and the task store")
    group.add_argument("--concurrency", type=int, default=8, help="model calls in flight")
    group.add_argument("--media-concurrency", type=int, help="ffmpeg processes at once")
    group.add_argument(
        "--rate-limit",
        type=parse_rate,
        action="append",
        default=[],
        metavar="PROVIDER=N/s",
        help="requests per provider (repeatable)",
    )
    group.add_argument(
        "--provider-retries",
        type=int,
        default=3,
        help="retries of transport, rate-limit and overload errors",
    )
    group.add_argument("--timeout", type=float, default=120.0, help="seconds per model call")
    group.add_argument("--cache-dir", help="the model response cache folder")
    group.add_argument(
        "--cache-mode",
        action="append",
        default=[],
        metavar="MODE|CLASS=MODE",
        help="off, record or replay; per class: L3b=replay (repeatable)",
    )
    group.add_argument("--ffmpeg", default="ffmpeg", help="the ffmpeg binary")
    group.add_argument(
        "--sources",
        metavar="MODULE:NAME",
        help="your ModelSources (a list, or a function returning one)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="framekit", description="AI building blocks for video datasets."
    )
    parser.add_argument("--version", action="version", version=f"framekit {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="log progress")
    commands = parser.add_subparsers(dest="command", required=True)

    extract = commands.add_parser("extract", help="a search space and a corpus → a dataset")
    extract.add_argument("--space", required=True, help="search-space.yaml (A)")
    extract.add_argument("--corpus", required=True, help="corpus.yaml (V)")
    extract.add_argument("--plan", help="plan.yaml (optional: defaults + derivation rules)")
    extract.add_argument("--out", required=True, help="the dataset folder")
    extract.add_argument("--dry-run", action="store_true", help="check and estimate cost only")
    extract.add_argument(
        "--rerun",
        action="append",
        default=[],
        metavar="STAGE",
        help="run a stage again (L0 L1 L2 L3a L3b embed; repeatable)",
    )
    extract.add_argument("--label", help="recorded in the manifest")
    _execution(extract)

    bench = commands.add_parser("bench", help="compare search spaces or plans on a gold set")
    bench.add_argument("--corpus", required=True)
    bench.add_argument("--gold", required=True, help="gold/queries.yaml")
    bench.add_argument(
        "--space", action="append", default=[], help="one pinned file, or LABEL=FILE per variant"
    )
    bench.add_argument(
        "--plan", action="append", default=[], help="one pinned file, or LABEL=FILE per variant"
    )
    bench.add_argument(
        "--scorer",
        default="baseline_retriever",
        help="baseline_retriever, or MODULE:NAME of your scorer",
    )
    bench.add_argument("--out", required=True, help="the report folder")
    _execution(bench)

    cache = commands.add_parser("cache", help="the model response cache")
    cache.add_argument("action", choices=("stats", "wipe"))
    cache.add_argument("--cache-dir", required=True)
    cache.add_argument("--class", dest="cache_class", help="only this cache class (e.g. L3b)")
    cache.add_argument("--namespace", help="only this cache namespace")
    return parser


def _runner(args: argparse.Namespace) -> Any:
    from framekit.extraction import LocalRunner

    return LocalRunner(
        concurrency=args.concurrency,
        media_concurrency=args.media_concurrency,
        rate_limits=dict(args.rate_limit),
        provider_retries=args.provider_retries,
        work_dir=args.work_dir,
        model_timeout_s=args.timeout,
        ffmpeg=args.ffmpeg,
    )


def _cache(args: argparse.Namespace) -> tuple[Any, Any]:
    from framekit.models.cache import CacheSettings, ClassPolicy, FileResponseStore

    if not args.cache_dir:
        if args.cache_mode:
            raise SystemExit("--cache-mode needs --cache-dir")
        return None, None
    default, classes = parse_cache_modes(args.cache_mode or ["record"])
    settings = CacheSettings(
        mode=default,  # type: ignore[arg-type]
        classes={name: ClassPolicy(mode=mode) for name, mode in classes.items()},  # type: ignore[arg-type]
    )
    return FileResponseStore(Path(args.cache_dir).expanduser()), settings


def run_extract(args: argparse.Namespace) -> int:
    from framekit.extraction import Corpus, ExtractionPlan, SearchSpace
    from framekit.extraction.pipeline import extract

    store, settings = _cache(args)
    result = extract(
        SearchSpace.load(args.space),
        Corpus.load(args.corpus),
        plan=ExtractionPlan.load(args.plan) if args.plan else None,
        out=args.out,
        runner=_runner(args),
        cache=store,
        cache_settings=settings,
        dry_run=args.dry_run,
        sources=load_sources(args.sources),
        rerun=args.rerun,
        label=args.label,
    )
    if args.dry_run:
        print(result)
        return 0
    manifest = result.manifest  # type: ignore[union-attr]
    print(f"dataset: {result.path}")  # type: ignore[union-attr]
    for unit in manifest.units:
        print(f"  {unit.unit_id}: {unit.status}{' (' + unit.reason + ')' if unit.reason else ''}")
    rows = ", ".join(f"{name} {info.rows}" for name, info in manifest.tables.items())
    print(f"rows: {rows}")
    print(f"estimated cost this run: ${manifest.cost.get('this_run_usd')}")
    return 0 if manifest.complete else 2


def run_bench(args: argparse.Namespace) -> int:
    from framekit.extraction import Corpus, ExtractionPlan, GoldSet, SearchSpace
    from framekit.extraction.bench import baseline_retriever, benchmark

    spaces, plans = args.space, args.plan
    if len(plans) == 1 and "=" not in plans[0] and spaces:
        variants: Any = {k: SearchSpace.load(v) for k, v in _labelled(spaces, "--space").items()}
        pinned = {"plan": ExtractionPlan.load(plans[0])}
    elif len(spaces) == 1 and "=" not in spaces[0] and plans:
        variants = {k: ExtractionPlan.load(v) for k, v in _labelled(plans, "--plan").items()}
        pinned = {"space": SearchSpace.load(spaces[0])}
    else:
        raise SystemExit(
            "pin one side: --plan FILE with --space LABEL=FILE ..., or --space FILE with "
            "--plan LABEL=FILE ..."
        )
    scorer: Any = baseline_retriever
    if args.scorer != "baseline_retriever":
        module_name, _, attribute = args.scorer.partition(":")
        sys.path.insert(0, str(Path.cwd()))
        scorer = getattr(importlib.import_module(module_name), attribute)
    store, settings = _cache(args)
    report = benchmark(
        variants,
        Corpus.load(args.corpus),
        gold=GoldSet.load(args.gold),
        out=args.out,
        scorer=scorer,
        runner=_runner(args),
        cache=store,
        cache_settings=settings,
        sources=load_sources(args.sources),
        **pinned,
    )
    print(report.table())
    print(f"report: {Path(args.out) / 'report.md'}")
    return 0 if all(v.error is None for v in report.variants) else 2


def run_cache(args: argparse.Namespace) -> int:
    from framekit.models.cache import FileResponseStore

    store = FileResponseStore(Path(args.cache_dir).expanduser())
    if args.action == "stats":
        stats = store.stats()
        if args.cache_class:
            stats = {k: v for k, v in stats.items() if k.endswith("/" + args.cache_class)}
        print(json.dumps(stats, indent=1) if stats else "the cache is empty")
        return 0
    removed = store.clear(cache_class=args.cache_class, namespace=args.namespace)
    print(f"removed {removed} entries")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    from framekit.extraction import ConfigError

    try:
        if args.command == "extract":
            return run_extract(args)
        if args.command == "bench":
            return run_bench(args)
        return run_cache(args)
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 1
    except (FileExistsError, FileNotFoundError, ValueError, ImportError) as exc:
        print(f"framekit: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
