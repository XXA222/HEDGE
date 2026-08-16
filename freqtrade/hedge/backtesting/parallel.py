from __future__ import annotations

from collections.abc import Iterable
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ProcessPoolExecutor,
    wait,
)
from dataclasses import dataclass
from typing import Any

from freqtrade.hedge.performance.resource_governor import (
    AdaptiveResourceGovernor,
    configure_worker_numeric_threads,
    multiprocessing_context,
    worker_numeric_environment,
)

from .contracts import BacktestEvaluation, Candidate
from .runner import HedgeBacktestRunner


_WORKER_RUNNER: HedgeBacktestRunner | None = None


def _init_worker(runner: HedgeBacktestRunner) -> None:
    global _WORKER_RUNNER
    _WORKER_RUNNER = runner
    configure_worker_numeric_threads()


def _evaluate_candidate(candidate: Candidate) -> BacktestEvaluation:
    runner = _WORKER_RUNNER
    if runner is None:  # pragma: no cover - defensive process initializer guard
        raise RuntimeError("Hedge optimization worker was not initialized")
    return runner.evaluate(candidate)


@dataclass(frozen=True, slots=True)
class ParallelEvaluationResult:
    evaluations: tuple[BacktestEvaluation, ...]
    worker_count: int
    executor_kind: str = "serial"
    adaptive: bool = False
    resource_samples: tuple[dict[str, Any], ...] = ()


def _snapshot_row(snapshot, desired: int, active: int) -> dict[str, Any]:
    return {
        "system_cpu_percent": snapshot.system_cpu_percent,
        "process_cpu_percent": snapshot.process_cpu_percent,
        "logical_cpus": snapshot.logical_cpus,
        "physical_cpus": snapshot.physical_cpus,
        "affinity_cpus": snapshot.affinity_cpus,
        "memory_available_bytes": snapshot.effective_available_memory_bytes,
        "desired_workers": desired,
        "active_workers": active,
    }


def evaluate_parallel(
    *,
    runner: HedgeBacktestRunner,
    candidates: Iterable[Candidate],
    workers: int = 0,
) -> ParallelEvaluationResult:
    """Evaluate independent candidates with adaptive process parallelism.

    One symbol's event replay is intentionally serial because wallet/order state is
    causally dependent across bars.  Candidate evaluations are independent and are
    therefore distributed across processes to bypass the CPython GIL.

    ``workers=0`` is adaptive, ``workers=-1`` uses the resource-aware maximum,
    ``workers=1`` is explicitly serial, and positive values above one are upper
    bounds.
    """
    materialized = tuple(candidates)
    if workers < -1:
        raise ValueError("workers must be -1, 0, or a positive integer")
    if len(materialized) < 2 or workers == 1:
        return ParallelEvaluationResult(
            evaluations=tuple(runner.evaluate(item) for item in materialized),
            worker_count=1,
            executor_kind="serial",
            adaptive=False,
        )

    governor = AdaptiveResourceGovernor()
    first_snapshot = governor.snapshot()
    initial = governor.recommended_workers(
        tasks=len(materialized),
        requested=workers,
        current_workers=0,
        snapshot=first_snapshot,
    )
    if initial <= 1:
        return ParallelEvaluationResult(
            evaluations=tuple(runner.evaluate(item) for item in materialized),
            worker_count=1,
            executor_kind="serial-resource-limited",
            adaptive=workers == 0,
            resource_samples=(_snapshot_row(first_snapshot, initial, 0),),
        )

    # Upper bound is computed without load throttling so the pool can scale up later
    # if desktop load falls. Processes are spawned lazily as tasks are submitted.
    upper = governor.recommended_workers(
        tasks=len(materialized),
        requested=-1 if workers == 0 else workers,
        current_workers=0,
        snapshot=first_snapshot,
    )
    upper = max(initial, upper)
    iterator = iter(materialized)
    futures: dict[Future[BacktestEvaluation], Candidate] = {}
    results: list[BacktestEvaluation] = []
    samples: list[dict[str, Any]] = [_snapshot_row(first_snapshot, initial, 0)]
    peak_active = 0

    def submit_until(executor: ProcessPoolExecutor, desired: int) -> None:
        nonlocal peak_active
        while len(futures) < desired:
            try:
                candidate = next(iterator)
            except StopIteration:
                break
            future = executor.submit(_evaluate_candidate, candidate)
            futures[future] = candidate
        peak_active = max(peak_active, len(futures))

    with worker_numeric_environment():
        with ProcessPoolExecutor(
            max_workers=upper,
            mp_context=multiprocessing_context(),
            initializer=_init_worker,
            initargs=(runner,),
        ) as executor:
            submit_until(executor, initial)
            try:
                while futures:
                    done, _ = wait(tuple(futures), return_when=FIRST_COMPLETED)
                    for future in done:
                        futures.pop(future)
                        results.append(future.result())

                    remaining = len(materialized) - len(results)
                    if remaining <= 0:
                        continue
                    snapshot = governor.snapshot()
                    desired = governor.recommended_workers(
                        tasks=remaining,
                        requested=workers,
                        current_workers=len(futures),
                        snapshot=snapshot,
                    )
                    samples.append(_snapshot_row(snapshot, desired, len(futures)))
                    # If other applications became busy, existing evaluations finish
                    # naturally; we simply stop feeding the pool until active work falls
                    # below the new target.
                    submit_until(executor, desired)
            except Exception:
                for future in futures:
                    future.cancel()
                raise

    results.sort(key=lambda item: (item.candidate.ordinal, item.candidate.candidate_id))
    return ParallelEvaluationResult(
        evaluations=tuple(results),
        worker_count=max(1, peak_active),
        executor_kind="process",
        adaptive=workers == 0,
        resource_samples=tuple(samples),
    )
