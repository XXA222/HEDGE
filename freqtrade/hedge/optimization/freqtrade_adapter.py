"""Freqtrade historical-data adapter for the generic Hedge optimization engine."""

from __future__ import annotations

import shutil
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, cast

from freqtrade.hedge.optimization.artifacts import OptimizationArtifacts, export_optimization_result
from freqtrade.hedge.optimization.config import parse_optimization_config
from freqtrade.hedge.optimization.engine import (
    EvaluationContext,
    OptimizationEngine,
    TrialEvaluator,
)
from freqtrade.hedge.optimization.types import OptimizationResult


@dataclass(frozen=True, slots=True)
class HedgeOptimizationRun:
    result: OptimizationResult
    artifacts: OptimizationArtifacts
    dataset_pair: str
    dataset_timeframe: str
    dataset_start: datetime
    dataset_end: datetime
    dataset_bar_count: int
    parallel_backend: str = "serial"
    peak_workers: int = 1
    resource_samples: tuple[Mapping[str, object], ...] = ()


_REQUIRED_OPTIMIZATION_DATASET_FIELDS = (
    "pair",
    "timeframe",
    "start",
    "end",
    "bar_count",
    "data_fingerprint",
    "missing_candle_count",
)


def _require_optimization_dataset_contract(dataset: Any, *, source: str) -> Any:
    """Validate the minimum dataset contract consumed by the optimization adapter.

    Custom and test backtest runners are intentionally supported, but they must expose
    the same safety-relevant facts as ``HedgeBacktestDataset``.  Missing gap metadata
    must fail closed instead of being silently interpreted as a gap-free dataset.
    """

    missing = tuple(
        name for name in _REQUIRED_OPTIMIZATION_DATASET_FIELDS if not hasattr(dataset, name)
    )
    if missing:
        joined = ", ".join(missing)
        raise TypeError(f"{source} backtest dataset is missing required field(s): {joined}")

    missing_candle_count = dataset.missing_candle_count
    if isinstance(missing_candle_count, bool) or not isinstance(missing_candle_count, int):
        raise TypeError(f"{source} dataset missing_candle_count must be an integer")
    if missing_candle_count < 0:
        raise ValueError(f"{source} dataset missing_candle_count cannot be negative")

    if isinstance(dataset.bar_count, bool) or not isinstance(dataset.bar_count, int):
        raise TypeError(f"{source} dataset bar_count must be an integer")
    if dataset.bar_count < 1:
        raise ValueError(f"{source} dataset bar_count must be positive")

    if not isinstance(dataset.data_fingerprint, str):
        raise TypeError(f"{source} dataset data_fingerprint must be a string")
    if not isinstance(dataset.pair, str) or not dataset.pair:
        raise TypeError(f"{source} dataset pair must be a non-empty string")
    if not isinstance(dataset.timeframe, str) or not dataset.timeframe:
        raise TypeError(f"{source} dataset timeframe must be a non-empty string")
    return dataset


def window_timerange(
    timestamps: Sequence[datetime],
    context: EvaluationContext,
) -> str | None:
    """Return a millisecond timerange for a walk-forward test slice."""

    window = context.window
    if window is None:
        return None
    if not timestamps or window.test.stop > len(timestamps):
        raise ValueError("walk-forward test window exceeds dataset timestamps")
    start = timestamps[window.test.start]
    if window.test.stop < len(timestamps):
        stop = timestamps[window.test.stop]
    elif len(timestamps) >= 2:
        stop = timestamps[-1] + (timestamps[-1] - timestamps[-2])
    else:
        stop = timestamps[-1] + timedelta(minutes=1)
    start_ms = int(start.timestamp() * 1000)
    stop_ms = int(stop.timestamp() * 1000)
    if stop_ms <= start_ms:
        raise ValueError("walk-forward timerange must be positive")
    return f"{start_ms}-{stop_ms}"


def run_freqtrade_hedge_optimization(  # noqa: C901
    config: dict[str, Any],
    *,
    backtest_runner: Callable[..., Any] | None = None,
) -> HedgeOptimizationRun:
    """Optimize Hedge planner/matcher parameters on downloaded Freqtrade data.

    The first deterministic probe binds the study to the actual analyzed signal,
    OHLCV, and funding-event fingerprint.  Every trial then uses the normal
    ``run_freqtrade_hedge_backtest`` path; no alternate fill engine is introduced.
    """

    prepared = None
    if backtest_runner is None:
        from freqtrade.optimize.hedge_backtesting import prepare_freqtrade_hedge_backtest

        prepared = prepare_freqtrade_hedge_backtest(deepcopy(config))
    default_output = Path(str(config.get("user_data_dir", "user_data"))) / "hyperopt_results"
    optimization = parse_optimization_config(
        config,
        default_output_directory=default_output,
    )
    output = optimization.output_directory.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    probe_path = output / ".dataset-probe.json"
    try:
        if prepared is not None:
            probe = prepared.run(deepcopy(config), export_path=probe_path, persist_artifact=False)
        else:
            if backtest_runner is None:  # pragma: no cover - defensive
                raise RuntimeError("optimization backtest runner was not initialized")
            probe = backtest_runner(deepcopy(config), export_path=probe_path, export_events=False)
    finally:
        for path in (probe_path, probe_path.with_suffix(probe_path.suffix + ".sha256")):
            if path.exists():
                path.unlink()

    probe_dataset = _require_optimization_dataset_contract(probe.dataset, source="probe")
    if probe_dataset.missing_candle_count:
        raise ValueError(
            "research optimization requires a gap-free compact dataset for O(1) timestamp indexing"
        )

    from freqtrade.exchange import timeframe_to_seconds
    from freqtrade.hedge.backtesting.memory import RegularTimestampSequence

    timestamps = RegularTimestampSequence(
        start=probe_dataset.start,
        step_seconds=timeframe_to_seconds(probe_dataset.timeframe),
        length=probe_dataset.bar_count,
    )

    trial_directory = output / ".trial-artifacts"
    trial_directory.mkdir(parents=True, exist_ok=True)

    def evaluator(  # noqa: C901 - adapter evaluation boundary
        trial_config: Mapping[str, Any],
        context: EvaluationContext,
    ) -> Mapping[str, object]:
        candidate = deepcopy(dict(trial_config))
        timerange = window_timerange(timestamps, context)
        row_slice = None
        if context.window is not None:
            if prepared is not None:
                row_slice = slice(context.window.test.start, context.window.test.stop)
            else:
                if timerange is None:  # pragma: no cover - defensive contract
                    raise RuntimeError("walk-forward evaluation did not produce a timerange")
                candidate["timerange"] = timerange
        artifact = trial_directory / (
            f"trial-{context.trial_id:06d}-eval-{context.evaluation_index:04d}.json"
        )
        try:
            if prepared is not None:
                run = prepared.run(
                    candidate,
                    export_path=artifact,
                    row_slice=row_slice,
                    persist_artifact=False,
                )
            else:
                if backtest_runner is None:  # pragma: no cover - defensive
                    raise RuntimeError("optimization backtest runner was not initialized")
                run = backtest_runner(candidate, export_path=artifact, export_events=False)
            run_dataset = _require_optimization_dataset_contract(run.dataset, source="trial")
            if run_dataset.missing_candle_count:
                raise ValueError("trial backtest returned a dataset with missing candles")
            if run_dataset.data_fingerprint == "":
                raise ValueError("trial backtest returned an empty data fingerprint")
            if run_dataset.pair != probe_dataset.pair:
                raise ValueError("trial backtest changed the managed dataset pair")
            if run_dataset.timeframe != probe_dataset.timeframe:
                raise ValueError("trial backtest changed the dataset timeframe")
            expected_bars = (
                probe_dataset.bar_count if context.window is None else context.window.test.length
            )
            if run_dataset.bar_count != expected_bars:
                raise ValueError(
                    "trial backtest returned an unexpected number of bars: "
                    f"expected={expected_bars}; actual={run_dataset.bar_count}"
                )
            # A full-range baseline trial must be byte-for-byte bound to the
            # initial analyzed dataset.  Walk-forward slices and explicit stress
            # scenarios intentionally transform replay inputs (timerange, funding
            # multiplier), so their fingerprints are expected to differ.
            if (
                context.window is None
                and context.stress_scenario.name == "baseline"
                and run_dataset.data_fingerprint != probe_dataset.data_fingerprint
            ):
                raise ValueError(
                    "baseline trial dataset fingerprint drifted from the optimization probe; "
                    "optimization parameters must not mutate analyzed market/signal inputs"
                )
            return run.result.report
        finally:
            for path in (artifact, artifact.with_suffix(artifact.suffix + ".sha256")):
                if path.exists():
                    path.unlink()

    engine = OptimizationEngine(
        base_config=config,
        optimization_config=optimization,
        evaluator=cast(TrialEvaluator, evaluator),
        dataset_fingerprint=probe_dataset.data_fingerprint,
        dataset_size=probe_dataset.bar_count,
        timestamps=timestamps,
    )
    try:
        result = engine.run()
        artifacts = export_optimization_result(result, output)
    finally:
        shutil.rmtree(trial_directory, ignore_errors=True)
    return HedgeOptimizationRun(
        result=result,
        artifacts=artifacts,
        dataset_pair=probe_dataset.pair,
        dataset_timeframe=probe_dataset.timeframe,
        dataset_start=probe_dataset.start,
        dataset_end=probe_dataset.end,
        dataset_bar_count=probe_dataset.bar_count,
        parallel_backend=engine.parallel_backend,
        peak_workers=engine.peak_workers,
        resource_samples=tuple(engine.resource_samples),
    )
