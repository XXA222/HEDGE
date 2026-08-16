from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from freqtrade.hedge.performance.resource_governor import (
    AdaptiveResourceGovernor,
    configure_worker_numeric_threads,
    multiprocessing_context,
    worker_numeric_environment,
)

from .hyperopt import DeterministicTrial, HedgeHyperoptRunner, HedgeHyperoptTrialResult


_PREPARED: Any = None
_BASE_CONFIG: Mapping[str, Any] | None = None
_OUTPUT_DIRECTORY: Path | None = None
_SEED = 42


def _worker_init() -> None:
    configure_worker_numeric_threads()


def _worker_run(number: int) -> HedgeHyperoptTrialResult:
    prepared = _PREPARED
    base_config = _BASE_CONFIG
    output_directory = _OUTPUT_DIRECTORY
    if prepared is None or base_config is None or output_directory is None:
        raise RuntimeError("native hyperopt fork state was not initialized")
    runner = HedgeHyperoptRunner()
    trial = DeterministicTrial(_SEED + number)

    def backtest(trial_config: Mapping[str, Any]):
        export = output_directory / f"epoch-{number:05d}.json"
        run = prepared.run(
            dict(trial_config),
            export_path=export,
            persist_artifact=False,
        )
        if run.native_artifact is None:
            raise RuntimeError("Hedge backtest did not produce a native artifact")
        return run.native_artifact

    return runner.evaluate(
        trial,
        number=number,
        base_config=base_config,
        backtest=backtest,
    )


@dataclass(frozen=True, slots=True)
class NativeParallelHyperoptResult:
    results: tuple[HedgeHyperoptTrialResult, ...]
    backend: str
    peak_workers: int
    resource_samples: tuple[dict[str, object], ...]


def evaluate_native_hyperopt(  # noqa: C901 - hyperopt evaluation boundary
    *,
    prepared: Any,
    base_config: Mapping[str, Any],
    output_directory: Path,
    epochs: int,
    seed: int,
    workers: int = 0,
) -> NativeParallelHyperoptResult:
    """Evaluate independent epochs with Linux fork/COW and adaptive worker feed."""
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if workers < -1:
        raise ValueError("workers must be -1, 0, or a positive integer")

    numbers = tuple(range(epochs))
    if epochs == 1 or workers == 1:
        runner = HedgeHyperoptRunner()
        rows = []
        for number in numbers:
            trial = DeterministicTrial(seed + number)

            def backtest(trial_config: Mapping[str, Any], *, number: int = number):
                export = output_directory / f"epoch-{number:05d}.json"
                run = prepared.run(
                    dict(trial_config),
                    export_path=export,
                    persist_artifact=False,
                )
                if run.native_artifact is None:
                    raise RuntimeError("Hedge backtest did not produce a native artifact")
                return run.native_artifact

            rows.append(
                runner.evaluate(
                    trial,
                    number=number,
                    base_config=base_config,
                    backtest=backtest,
                )
            )
        return NativeParallelHyperoptResult(tuple(rows), "serial", 1, ())

    context = multiprocessing_context()
    if context.get_start_method() != "fork":
        # Prepared pandas/numpy data is intentionally shared copy-on-write in Docker.
        # Spawn would serialize/copy it, defeating the memory policy.
        return evaluate_native_hyperopt(
            prepared=prepared,
            base_config=base_config,
            output_directory=output_directory,
            epochs=epochs,
            seed=seed,
            workers=1,
        )

    governor = AdaptiveResourceGovernor()
    first = governor.snapshot()
    initial = governor.recommended_workers(
        tasks=epochs,
        requested=workers,
        current_workers=0,
        snapshot=first,
    )
    samples: list[dict[str, object]] = [
        {
            "system_cpu_percent": first.system_cpu_percent,
            "source": first.source,
            "desired_workers": initial,
            "active_workers": 0,
            "memory_available_bytes": first.effective_available_memory_bytes,
        }
    ]
    if initial <= 1:
        return evaluate_native_hyperopt(
            prepared=prepared,
            base_config=base_config,
            output_directory=output_directory,
            epochs=epochs,
            seed=seed,
            workers=1,
        )

    upper = governor.recommended_workers(
        tasks=epochs,
        requested=-1 if workers == 0 else workers,
        current_workers=0,
        snapshot=first,
    )
    upper = max(initial, upper)

    global _PREPARED, _BASE_CONFIG, _OUTPUT_DIRECTORY, _SEED
    _PREPARED = prepared
    _BASE_CONFIG = base_config
    _OUTPUT_DIRECTORY = output_directory
    _SEED = seed

    iterator = iter(numbers)
    futures: dict[Future[HedgeHyperoptTrialResult], int] = {}
    results: list[HedgeHyperoptTrialResult] = []
    peak = 0

    def submit_until(executor: ProcessPoolExecutor, desired: int) -> None:
        nonlocal peak
        while len(futures) < desired:
            try:
                number = next(iterator)
            except StopIteration:
                break
            futures[executor.submit(_worker_run, number)] = number
        peak = max(peak, len(futures))

    try:
        with worker_numeric_environment():
            with ProcessPoolExecutor(
                max_workers=upper,
                mp_context=context,
                initializer=_worker_init,
            ) as executor:
                submit_until(executor, initial)
                while futures:
                    done, _ = wait(tuple(futures), return_when=FIRST_COMPLETED)
                    for future in done:
                        futures.pop(future)
                        results.append(future.result())
                    remaining = epochs - len(results)
                    if remaining <= 0:
                        continue
                    snap = governor.snapshot()
                    desired = governor.recommended_workers(
                        tasks=remaining,
                        requested=workers,
                        current_workers=len(futures),
                        snapshot=snap,
                    )
                    samples.append(
                        {
                            "system_cpu_percent": snap.system_cpu_percent,
                            "source": snap.source,
                            "desired_workers": desired,
                            "active_workers": len(futures),
                            "memory_available_bytes": snap.effective_available_memory_bytes,
                        }
                    )
                    submit_until(executor, desired)
    finally:
        _PREPARED = None
        _BASE_CONFIG = None
        _OUTPUT_DIRECTORY = None

    results.sort(key=lambda item: item.number)
    return NativeParallelHyperoptResult(
        results=tuple(results),
        backend="process-fork-adaptive",
        peak_workers=max(1, peak),
        resource_samples=tuple(samples),
    )
