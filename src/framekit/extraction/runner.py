"""Running extraction (DESIGN.md §5): tasks keyed by input hash, the task store, model calls
with rate limits, retries and the budget.

**Idempotent and resumable.** Each task's input hash covers its inputs, its stage version
and its model; its outcome is written to the task store (the work directory) as a run
record plus its output before anything depends on it. A finished task is never run again;
a changed input runs only what depends on it. An interrupted run continues from the store.

`LocalRunner` holds the execution options: they change speed and cost, never the dataset.
A distributed runner would implement the same `Runner` protocol later.
"""

import asyncio
import json
import logging
import os
import tempfile
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Protocol

from framekit.contracts import (
    ConversationRequest,
    ConversationResult,
    EmbeddingRequest,
    EmbeddingResult,
    ModelCallError,
)
from framekit.extraction.schemas.common import digest
from framekit.extraction.schemas.plan import BudgetSettings

log = logging.getLogger(__name__)
GITIGNORE = "*\n"  # every folder framekit creates: never committed by accident


def ensure_private_dir(path: Path) -> Path:
    """Create a folder (and parents) holding a `.gitignore` with `*`: work folders hold
    analysis copies, which count as footage for a byo_media corpus."""
    path.mkdir(parents=True, exist_ok=True)
    ignore = path / ".gitignore"
    if not ignore.exists():
        ignore.write_text(GITIGNORE, encoding="utf-8")
    return path


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=path.suffix)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(temp, path)
    except BaseException:
        Path(temp).unlink(missing_ok=True)
        raise


class TaskStore:
    """Task outcomes on disk: `tasks/<stage>/<hash[:2]>/<hash>.json` holding the run record
    and the output. It's a cache: deleting it costs only recomputation."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def path(self, stage: str, input_hash: str) -> Path:
        return self.root / "tasks" / stage / input_hash[:2] / f"{input_hash}.json"

    def get(self, stage: str, input_hash: str) -> dict[str, Any] | None:
        path = self.path(stage, input_hash)
        try:
            entry: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return entry

    def put(self, stage: str, input_hash: str, record: Mapping[str, Any], output: Any) -> None:
        entry = {"record": dict(record), "output": output}
        atomic_write(self.path(stage, input_hash), json.dumps(entry, ensure_ascii=False))


class BudgetExceeded(RuntimeError):
    def __init__(self, stage: str, spent: Decimal, cap: Decimal, scope: str) -> None:
        super().__init__(
            f"budget reached: {scope} spend ${spent:.4f} of ${cap:.4f} (stage {stage})"
        )
        self.stage = stage


class Spend:
    """Estimated model spend this run, per stage, against the plan's budget. Checked before
    each call, so a run overshoots by at most the calls already in flight.

    When a cap is reached, no new unit starts. `on_exceed: stop` also fails the next call
    of every unit in progress (those units end partial); `finish_unit_then_stop` lets the
    units in progress finish."""

    def __init__(self, budget: BudgetSettings) -> None:
        self.budget = budget
        self.per_stage: dict[str, Decimal] = {}
        self.stopped = False  # set when a cap is reached: no new units start
        self.reason = ""

    @property
    def total(self) -> Decimal:
        return sum(self.per_stage.values(), Decimal(0))

    def _exceeded(self, stage: str) -> BudgetExceeded | None:
        cap = self.budget.max_cost_usd
        if cap is not None and self.total >= Decimal(str(cap)):
            return BudgetExceeded(stage, self.total, Decimal(str(cap)), "total")
        group = stage_group(stage)
        stage_cap = self.budget.per_stage_usd.get(group)
        spent = sum((v for k, v in self.per_stage.items() if stage_group(k) == group), Decimal(0))
        if stage_cap is not None and spent >= Decimal(str(stage_cap)):
            return BudgetExceeded(stage, spent, Decimal(str(stage_cap)), group)
        return None

    def check(self, stage: str) -> None:
        exceeded = self._exceeded(stage)
        if exceeded is None:
            return
        self.stopped = True
        self.reason = str(exceeded)
        if self.budget.on_exceed == "stop":
            raise exceeded

    def add(self, stage: str, cost: Decimal) -> None:
        self.per_stage[stage] = self.per_stage.get(stage, Decimal(0)) + cost


def stage_group(stage: str) -> str:
    return "embed" if stage.startswith("embed") else stage


class RateLimiter:
    """At most `rate` requests per second per provider, shared by every stage."""

    def __init__(self, rates: Mapping[str, float]) -> None:
        for name, rate in rates.items():
            if not rate > 0:
                raise ValueError(f"rate limit for {name!r} must be > 0 requests per second")
        self.rates = dict(rates)
        self._next: dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def wait(self, provider: str) -> None:
        rate = self.rates.get(provider)
        if rate is None:
            return
        async with self._lock:
            moment = time.monotonic()
            start = max(moment, self._next.get(provider, moment))
            self._next[provider] = start + 1 / rate
        if start > moment:
            await asyncio.sleep(start - moment)


@dataclass(frozen=True)
class CallRecord:
    """One model call as the `runs` table records it (no request content)."""

    model_id: str
    provider: str
    adapter_version: str
    latency_ms: int
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int | None
    cost_usd: str
    cached: bool
    stop_reason: str | None
    source: str | None = None

    @classmethod
    def of(cls, result: ConversationResult) -> "CallRecord":
        return cls(
            model_id=result.model_id,
            provider=result.provider,
            adapter_version=result.adapter_version,
            latency_ms=result.latency_ms,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            reasoning_tokens=result.reasoning_tokens,
            cost_usd=str(result.cost_usd),
            cached=result.cached,
            stop_reason=result.stop_reason,
            source=result.source,
        )

    @classmethod
    def of_embedding(cls, result: EmbeddingResult) -> "CallRecord":
        return cls(
            model_id=result.model_id,
            provider=result.provider,
            adapter_version=result.adapter_version,
            latency_ms=result.latency_ms,
            input_tokens=result.input_tokens,
            output_tokens=0,
            reasoning_tokens=None,
            cost_usd=str(result.cost_usd),
            cached=False,
            stop_reason=None,
            source=result.source,
        )


class Runner(Protocol):
    """What extraction needs from a runner (LocalRunner today, a distributed one later)."""

    concurrency: int
    provider_retries: int
    model_timeout_s: float
    work_dir: Path | None
    ffmpeg: str

    def media_slots(self) -> int: ...


@dataclass
class LocalRunner:
    """One process, async I/O, bounded parallelism (DESIGN.md §5.2).

    - `concurrency`: model calls in flight at once;
    - `media_concurrency`: ffmpeg processes at once (default: half the CPUs, at least 2);
    - `rate_limits`: requests per second per provider name, e.g. `{"anthropic": 2.0}`;
    - `provider_retries`: retries of a retryable model error (transport, rate limit,
      overload), with exponential backoff from `retry_backoff_s`;
    - `work_dir`: analysis copies and the task store (default: `<out>.work` next to the
      output); never inside a repository for a byo_media corpus;
    - `model_timeout_s`: per model call;
    - `ffmpeg`: the ffmpeg binary, a name on PATH or a path.
    """

    concurrency: int = 8
    media_concurrency: int | None = None
    rate_limits: Mapping[str, float] = field(default_factory=dict)
    provider_retries: int = 3
    retry_backoff_s: float = 1.0
    work_dir: Path | str | None = None
    model_timeout_s: float = 120.0
    ffmpeg: str = "ffmpeg"

    def __post_init__(self) -> None:
        if self.concurrency < 1:
            raise ValueError("concurrency must be at least 1")
        if self.provider_retries < 0:
            raise ValueError("provider_retries must be >= 0")
        if self.media_concurrency is not None and self.media_concurrency < 1:
            raise ValueError("media_concurrency must be at least 1")
        RateLimiter(self.rate_limits)  # checks the rates
        if self.work_dir is not None:
            self.work_dir = Path(self.work_dir).expanduser()

    def media_slots(self) -> int:
        return self.media_concurrency or max(2, (os.cpu_count() or 4) // 2)

    def options(self) -> dict[str, Any]:
        """For the manifest's log of how the run was executed (never hashed)."""
        return {
            "runner": type(self).__name__,
            "concurrency": self.concurrency,
            "media_concurrency": self.media_slots(),
            "rate_limits": dict(self.rate_limits),
            "provider_retries": self.provider_retries,
            "model_timeout_s": self.model_timeout_s,
        }


@dataclass
class Execution:
    """The state of one run: the store, the limits and the spend."""

    runner: LocalRunner
    store: TaskStore
    port: Any  # the conversation port (router, or the caching wrapper around it)
    spend: Spend
    rerun: frozenset[str] = frozenset()
    model_slots: asyncio.Semaphore = field(init=False)
    media_slots: asyncio.Semaphore = field(init=False)
    limiter: RateLimiter = field(init=False)
    reused: int = 0
    computed: int = 0

    def __post_init__(self) -> None:
        self.model_slots = asyncio.Semaphore(self.runner.concurrency)
        self.media_slots = asyncio.Semaphore(self.runner.media_slots())
        self.limiter = RateLimiter(self.runner.rate_limits)

    async def task(
        self,
        stage: str,
        inputs: Mapping[str, Any],
        compute: Callable[[str], Awaitable[tuple[Any, dict[str, Any]]]],
        *,
        valid: Callable[[Any], bool] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """The output and run record of a task: from the store when it already ran with
        these inputs (and `valid(output)` holds, e.g. its file still exists), else
        computed now and stored. `compute(input_hash)` returns (output, record fields)."""
        input_hash = digest({"stage": stage, **inputs})
        if stage not in self.rerun:
            entry = self.store.get(stage, input_hash)
            if (
                entry is not None
                and entry["record"].get("status") in ("ok", "flagged")
                and (valid is None or valid(entry["output"]))
            ):
                self.reused += 1
                return entry["output"], {**entry["record"], "reused": True}
        started = now()
        output, fields = await compute(input_hash)
        record = {
            "run_id": f"run_{input_hash[:20]}",
            "stage": stage,
            "input_hash": input_hash,
            "started_at": started,
            "finished_at": now(),
            "status": "ok",
            **fields,
        }
        if record["status"] in ("ok", "flagged"):
            self.store.put(stage, input_hash, record, output)
        self.computed += 1
        return output, {**record, "reused": False}

    async def call(
        self, stage: str, provider: str, request: ConversationRequest
    ) -> ConversationResult:
        """One model call: budget check, rate limit, a slot, then retries of retryable
        errors with exponential backoff."""
        attempt = 0
        while True:
            self.spend.check(stage)
            await self.limiter.wait(provider)
            try:
                async with self.model_slots:
                    result: ConversationResult = await self.port.generate(request)
            except ModelCallError as exc:
                if not exc.retryable or attempt >= self.runner.provider_retries:
                    raise
                attempt += 1
                delay = self.runner.retry_backoff_s * 2 ** (attempt - 1)
                log.info("retrying %s on %s in %.1f s: %s", stage, provider, delay, exc)
                await asyncio.sleep(delay)
                continue
            self.spend.add(stage, result.cost_usd)
            return result

    async def embed(self, stage: str, provider: str, request: EmbeddingRequest) -> EmbeddingResult:
        attempt = 0
        while True:
            self.spend.check(stage)
            await self.limiter.wait(provider)
            try:
                async with self.model_slots:
                    result: EmbeddingResult = await self.port.embed(request)
            except ModelCallError as exc:
                if not exc.retryable or attempt >= self.runner.provider_retries:
                    raise
                attempt += 1
                await asyncio.sleep(self.runner.retry_backoff_s * 2 ** (attempt - 1))
                continue
            self.spend.add(stage, result.cost_usd)
            return result


def total_cost(records: Iterable[Mapping[str, Any]]) -> Decimal:
    return sum((Decimal(str(r.get("cost_usd") or 0)) for r in records), Decimal(0))


__all__ = [
    "BudgetExceeded",
    "CallRecord",
    "Execution",
    "LocalRunner",
    "RateLimiter",
    "Runner",
    "Spend",
    "TaskStore",
    "ensure_private_dir",
    "total_cost",
]
