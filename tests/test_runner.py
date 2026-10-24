"""Running extraction (docs/DESIGN.md §5.1): tasks keyed by input hash in the task store,
model calls with retries, rate limits and the budget."""

from decimal import Decimal

import pytest
from helpers import final_result

from framekit.contracts import ConversationRequest, ModelCallError
from framekit.extraction.runner import (
    BudgetExceeded,
    Execution,
    LocalRunner,
    RateLimiter,
    Spend,
    TaskStore,
)
from framekit.extraction.schemas.plan import BudgetSettings


def test_the_budget_stops_new_work_and_fails_calls_only_with_stop():
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


def execution(tmp_path, port=None, budget=None, **runner):
    return Execution(
        LocalRunner(work_dir=tmp_path, retry_backoff_s=0, **runner),
        TaskStore(tmp_path),
        port,
        Spend(budget or BudgetSettings()),
    )


async def test_a_task_runs_once_and_is_then_reused(tmp_path):
    runs = []

    async def compute(input_hash):
        runs.append(input_hash)
        return {"answer": 42}, {"unit_id": "u1", "target_id": "u1"}

    ex = execution(tmp_path)
    first, record = await ex.task("L1", {"copy": "abc"}, compute)
    again, reused = await ex.task("L1", {"copy": "abc"}, compute)
    other, _ = await ex.task("L1", {"copy": "abd"}, compute)  # another input: another task
    assert first == again == other == {"answer": 42} and len(runs) == 2
    assert record["reused"] is False and reused["reused"] is True
    assert record["run_id"] == reused["run_id"] and record["status"] == "ok"
    assert (ex.computed, ex.reused) == (2, 1)
    assert TaskStore(tmp_path).get("L1", record["input_hash"])["output"] == {"answer": 42}


async def test_rerun_and_invalid_outputs_compute_again_and_failures_are_never_stored(tmp_path):
    runs = []

    async def compute(_):
        runs.append(1)
        return {"file": "copy.mp4"}, {}

    ex = execution(tmp_path)
    await ex.task("L0", {"source": "x"}, compute)
    await ex.task("L0", {"source": "x"}, compute, valid=lambda output: False)  # file gone
    rerun = Execution(ex.runner, ex.store, None, ex.spend, rerun=frozenset({"L0"}))
    await rerun.task("L0", {"source": "x"}, compute)
    assert len(runs) == 3

    async def failing(_):
        return None, {"status": "failed"}

    _, record = await ex.task("L2", {"x": 1}, failing)
    assert record["status"] == "failed" and ex.store.get("L2", record["input_hash"]) is None


class FlakyPort:
    def __init__(self, failures, retryable=True):
        self.failures, self.retryable, self.calls = failures, retryable, 0

    async def generate(self, request):
        self.calls += 1
        if self.calls <= self.failures:
            raise ModelCallError("overloaded", retryable=self.retryable)
        return final_result(cost_usd=Decimal("0.5"))


async def test_retryable_errors_are_retried_and_spend_is_counted(tmp_path):
    port = FlakyPort(failures=2)
    ex = execution(tmp_path, port, provider_retries=3)
    result = await ex.call("L3a", "hive", ConversationRequest("hi", max_tokens=10))
    assert result.text == "Hello!" and port.calls == 3
    assert ex.spend.per_stage == {"L3a": Decimal("0.5")}


async def test_errors_that_retrying_can_not_fix_fail_at_once(tmp_path):
    port = FlakyPort(failures=1, retryable=False)
    ex = execution(tmp_path, port, provider_retries=3)
    with pytest.raises(ModelCallError, match="overloaded"):
        await ex.call("L3a", "hive", ConversationRequest("hi", max_tokens=10))
    assert port.calls == 1


async def test_a_reached_budget_stops_calls_with_on_exceed_stop(tmp_path):
    port = FlakyPort(failures=0)
    ex = execution(tmp_path, port, budget=BudgetSettings(max_cost_usd=0.5, on_exceed="stop"))
    await ex.call("L3a", "hive", ConversationRequest("hi", max_tokens=10))
    with pytest.raises(BudgetExceeded):
        await ex.call("L3a", "hive", ConversationRequest("hi", max_tokens=10))
    assert port.calls == 1 and ex.spend.stopped


def test_runner_options_are_checked_and_logged():
    with pytest.raises(ValueError, match="concurrency"):
        LocalRunner(concurrency=0)
    with pytest.raises(ValueError, match="must be > 0"):
        LocalRunner(rate_limits={"hive": -1})
    options = LocalRunner(concurrency=3, rate_limits={"hive": 2.0}).options()
    assert options["concurrency"] == 3 and options["rate_limits"] == {"hive": 2.0}
