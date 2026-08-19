#!/usr/bin/env python3
"""Train, optimize and independently backtest a causal ETH HPRL dual-leg strategy.

The workflow uses the supplied two-year ETH candles only.  It uses 1h as the decision bar and
uses completed 1m/15m/1h/8h/1d candles as backward-asof features.  Validation and holdout test
periods are chronological and never participate in model selection.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
import shutil
import sys
import time
import zipfile
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from freqtrade.hedge.hprl.checkpoint import load_checkpoint, save_checkpoint
from freqtrade.hedge.hprl.config import (
    HPRLActionConfig,
    HPRLConfig,
    HPRLCostConfig,
    HPRLEnvironmentConfig,
    HPRLMemoryConfig,
    HPRLRewardConfig,
    HPRLTrainingConfig,
)
from freqtrade.hedge.hprl.data import TensorMarketDataset
from freqtrade.hedge.hprl.device import require_torch
from freqtrade.hedge.hprl.env import VectorizedHedgeEnv
from freqtrade.hedge.hprl.evaluation import evaluate_trading
from freqtrade.hedge.hprl.registry import available_algorithms
from freqtrade.hedge.hprl.runtime import build_online_runtime
from freqtrade.hedge.strategies.hprl_eth_dual_leg import HprlEthDualLegStrategy


TIMEFRAMES = ("1m", "15m", "1h", "8h", "1d")
PERIODS_PER_YEAR = {"1m": 525_600, "15m": 35_040, "1h": 8_760, "8h": 1_095, "1d": 365}
SYMBOL = "ETH/USDT:USDT"


@dataclass(frozen=True, slots=True)
class Candidate:
    learning_rate: float
    gamma: float
    tau: float
    hidden_dim: int
    reward_drawdown: float
    reward_turnover: float


def _json(value: object) -> object:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dataclass_fields__"):
        return {key: _json(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(_json(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _find_data_file(root: Path, timeframe: str) -> Path:
    candidates = sorted(root.rglob(f"eth-{timeframe}.csv"))
    if not candidates:
        raise FileNotFoundError(f"ETH {timeframe} CSV not found below {root}")
    return candidates[0]


def _read_candles(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - {str(column).lower() for column in frame.columns}
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    frame.columns = [str(column).lower() for column in frame.columns]
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    for name in ("open", "high", "low", "close", "volume"):
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    frame = frame.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"])
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    if frame.empty or (frame[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError(f"{path} contains invalid OHLC values")
    return frame.reset_index(drop=True)


def build_feature_frame(data_root: Path, *, primary: str) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    if primary not in TIMEFRAMES:
        raise ValueError(f"primary timeframe must be one of {TIMEFRAMES}")
    candles = {timeframe: _read_candles(_find_data_file(data_root, timeframe)) for timeframe in TIMEFRAMES}
    frame = candles[primary][["timestamp", "close", "volume"]].copy()
    feature_columns: list[str] = []
    for timeframe, source in candles.items():
        close = source["close"].astype(float)
        feature = pd.DataFrame({"timestamp": source["timestamp"]})
        columns = {
            f"{timeframe}_ret_1": close.pct_change(1),
            f"{timeframe}_ret_4": close.pct_change(4),
            f"{timeframe}_vol_16": close.pct_change().rolling(16).std(),
            f"{timeframe}_z_32": (close - close.rolling(32).mean()) / close.rolling(32).std(),
        }
        # A candle at timestamp t is not considered observable until the following decision bar.
        for name, values in columns.items():
            feature[name] = values.shift(1)
            feature_columns.append(name)
        frame = pd.merge_asof(
            frame.sort_values("timestamp"), feature.sort_values("timestamp"),
            on="timestamp", direction="backward", allow_exact_matches=True,
        )
    frame["forward_return"] = frame["close"].shift(-1) / frame["close"] - 1.0
    frame["available_notional"] = (frame["close"] * frame["volume"]).clip(lower=1.0)
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    if len(frame) < 2_000:
        raise ValueError("insufficient clean feature rows after causal warmup")
    fingerprints = {timeframe: str(_find_data_file(data_root, timeframe)) for timeframe in TIMEFRAMES}
    return frame, feature_columns, fingerprints


def dataset_from_frame(frame: pd.DataFrame, feature_columns: list[str]) -> TensorMarketDataset:
    torch = require_torch()
    features = torch.tensor(frame[feature_columns].to_numpy(dtype=np.float32), dtype=torch.float32).unsqueeze(1)
    returns = torch.tensor(frame["forward_return"].to_numpy(dtype=np.float32), dtype=torch.float32).unsqueeze(1)
    available = torch.tensor(frame["available_notional"].to_numpy(dtype=np.float32), dtype=torch.float32).unsqueeze(1)
    return TensorMarketDataset(features=features, forward_returns=returns, available_notional=available, symbols=(SYMBOL,)).validate()


def make_config(
    algorithm: str,
    candidate: Candidate,
    args: argparse.Namespace,
    *,
    parallel_envs: int,
) -> HPRLConfig:
    action = HPRLActionConfig(position_levels=(0.0, 0.05, 0.12, 0.25, 0.40), leverage=args.leverage)
    environment = HPRLEnvironmentConfig(
        initial_equity=args.initial_equity,
        parallel_envs=parallel_envs,
        annualization_periods=PERIODS_PER_YEAR[args.primary_timeframe],
        runtime_checks=False,
        info_mode="training",
        action=action,
        costs=HPRLCostConfig(maker_fee_bps=args.maker_fee_bps, taker_fee_bps=args.taker_fee_bps, base_slippage_bps=args.slippage_bps),
        reward=HPRLRewardConfig(drawdown=candidate.reward_drawdown, turnover=candidate.reward_turnover),
    )
    training = HPRLTrainingConfig(
        algorithm=algorithm,
        seed=args.seed,
        device=args.device,
        replay_device=args.replay_device,
        batch_size=args.batch_size,
        replay_capacity=args.replay_capacity,
        warmup_steps=args.warmup_steps,
        gradient_steps=args.gradient_steps,
        gamma=candidate.gamma,
        tau=candidate.tau,
        learning_rate=candidate.learning_rate,
        hidden_dim=candidate.hidden_dim,
        hidden_depth=args.hidden_depth,
        mixed_precision=args.mixed_precision,
        metrics_interval=args.metrics_interval,
        compile_mode="off" if args.compile_mode == "off" else args.compile_mode,
        expected_updates=args.final_steps,
        hardware_profile=args.hardware_profile,
        optimizer_backend="auto",
        replay_prefetch=True,
    )
    return HPRLConfig(environment=environment, training=training, memory=HPRLMemoryConfig(dataset_mode="auto"))


def objective(metrics: dict[str, float]) -> float:
    if metrics["liquidations"]:
        return -1000.0 - float(metrics["liquidations"])
    return metrics["net_return"] - 1.5 * metrics["max_drawdown"] - 0.002 * metrics["turnover"]


def training_health(training: object) -> tuple[bool, dict[str, float]]:
    """Return the recorded health evidence and reject a policy-collapse candidate."""
    metrics = getattr(training, "last_metrics", {})
    values = {
        str(name): float(value)
        for name, value in dict(metrics).items()
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    }
    collapsed = (
        values.get("training_health_ready", 0.0) >= 1.0
        and values.get("training_health_collapsed", 0.0) >= 1.0
    )
    return collapsed, values


def evaluate_agent(agent: object, dataset: TensorMarketDataset, config: HPRLConfig) -> tuple[dict[str, float], list[dict[str, object]]]:
    torch = require_torch()
    env_config = replace(config.environment, parallel_envs=1, info_mode="full")
    environment = VectorizedHedgeEnv(dataset, env_config, device=config.training.device, memory_config=config.memory)
    strategy = HprlEthDualLegStrategy()
    observation, _ = environment.reset()
    equity = [float(env_config.initial_equity)]
    turnover = fees = funding = 0.0
    liquidations = 0
    actions: list[dict[str, object]] = []
    try:
        while True:
            with torch.no_grad():
                action = agent.act(observation, deterministic=True)
            step = environment.step(action)
            info = step.info
            policy = [float(value) for value in action[0].detach().float().cpu().tolist()]
            directive = strategy.directive_from_policy_action(policy)
            equity.append(max(float(info["equity"][0].item()), 1e-9))
            turnover += float(info["turnover_ratio"][0].item())
            fees += float(info["fee_cost"][0].item()) + float(info["slippage_cost"][0].item()) + float(info["market_impact_cost"][0].item())
            funding += float(info["funding_pnl_ratio"][0].item())
            liquidations += int(bool(info["autoreset_mask"][0].item()))
            actions.append({"policy_action": policy, "directive": asdict(directive), "equity": equity[-1]})
            observation = step.observation
            if bool(info["time_done"]):
                break
    finally:
        environment.close(aggressive=True)
    metrics = asdict(evaluate_trading(
        equity, periods_per_year=env_config.annualization_periods, turnover=turnover,
        fees=fees, funding=funding, liquidations=liquidations,
    ))
    return {key: float(value) for key, value in metrics.items()}, actions


def train_and_validate(
    algorithm: str,
    candidate: Candidate,
    train_dataset: TensorMarketDataset,
    validation_dataset: TensorMarketDataset,
    args: argparse.Namespace,
    *,
    steps: int,
) -> tuple[dict[str, Any], object | None, HPRLConfig | None]:
    started = time.monotonic()
    config = make_config(algorithm, candidate, args, parallel_envs=args.parallel_envs)
    runtime = None
    try:
        runtime = build_online_runtime(train_dataset, config)
        training = runtime.trainer.run(steps)
        metrics, _ = evaluate_agent(runtime.agent, validation_dataset, config)
        collapsed, health = training_health(training)
        record = {
            "status": "REJECTED" if collapsed else "PASS",
            "algorithm": algorithm,
            "candidate": asdict(candidate),
            "training": asdict(training),
            "training_health": health,
            "validation": metrics,
            "objective": objective(metrics),
            "seconds": time.monotonic() - started,
        }
        if collapsed:
            record["rejection_reason"] = "training_health_policy_collapse"
            return record, None, None
        return record, runtime.agent, config
    except Exception as exc:  # The experiment matrix must continue after one algorithm failure.
        return ({"status": "ERROR", "algorithm": algorithm, "candidate": asdict(candidate), "error": f"{type(exc).__name__}: {exc}", "seconds": time.monotonic() - started}, None, None)
    finally:
        if runtime is not None:
            runtime.close(aggressive=True)


def candidate_grid(trials: int, seed: int) -> list[Candidate]:
    conservative = Candidate(3e-5, 0.99, 0.003, 128, 0.40, 0.0015)
    all_candidates = [
        Candidate(*values)
        for values in itertools.product(
            (1e-5, 3e-5, 1e-4, 3e-4), (0.985, 0.99, 0.995), (0.003, 0.005, 0.01),
            (128, 256), (0.25, 0.40), (0.0005, 0.0015),
        )
    ]
    random.Random(seed).shuffle(all_candidates)
    return [conservative, *[item for item in all_candidates if item != conservative]][:trials]


def split_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_end = int(len(frame) * 0.60)
    validation_end = int(len(frame) * 0.80)
    purge = 1
    return (
        frame.iloc[:train_end - purge].copy(),
        frame.iloc[train_end:validation_end - purge].copy(),
        frame.iloc[validation_end:].copy(),
    )


def write_summary(output: Path, payload: dict[str, Any]) -> None:
    final = payload["final_holdout_backtest"]
    best = payload["best_configuration"]
    lines = [
        "# ETH two-year HPRL strategy research", "",
        f"- Status: `{payload['status']}`", f"- Winner: `{best['algorithm']}`", f"- Objective: `{best['validation_objective']:.6f}`", "",
        "| Metric | Holdout result |", "|---|---:|",
        f"| Net return | {final['net_return']:.4%} |",
        f"| Sharpe | {final['sharpe']:.4f} |",
        f"| Sortino | {final['sortino']:.4f} |",
        f"| Calmar | {final['calmar']:.4f} |",
        f"| Max drawdown | {final['max_drawdown']:.4%} |",
        f"| CVaR | {final['cvar']:.4%} |",
        f"| Turnover | {final['turnover']:.4f} |",
        f"| Fees/slippage/impact | {final['fees']:.4f} |",
        f"| Liquidations | {final['liquidations']:.0f} |", "",
        "The holdout period was not used by algorithm selection or parameter optimization.",
    ]
    if not payload["qualification"]["accepted"]:
        lines.extend(("", "**Not qualified:** the continued final training triggered policy/gradient health collapse. The holdout result is diagnostic only and the checkpoint is saved with a `rejected-` prefix."))
    (output / "BACKTEST-SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def package(output: Path) -> Path:
    scripts = output / "scripts"
    scripts.mkdir(exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    for relative in (
        "tools/train_hprl_eth_two_year.py", "tools/Run-HPRL-Eth-TwoYear-Research.ps1",
        "freqtrade/hedge/strategies/hprl_eth_dual_leg.py",
    ):
        source = root / relative
        if source.is_file():
            shutil.copy2(source, scripts / source.name)
    archive = output.parent / f"HEDGE-HPRL-ETH-two-year-{output.name}.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as handle:
        for file in sorted(output.rglob("*")):
            if file.is_file():
                handle.write(file, file.relative_to(output).as_posix())
    return archive


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--primary-timeframe", choices=TIMEFRAMES, default="1h")
    parser.add_argument("--algorithms", default=",".join(available_algorithms()))
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--baseline-steps", type=int, default=20_000)
    parser.add_argument("--optimization-steps", type=int, default=30_000)
    parser.add_argument("--final-steps", type=int, default=100_000)
    parser.add_argument("--parallel-envs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--replay-capacity", type=int, default=200_000)
    parser.add_argument("--warmup-steps", type=int, default=2_000)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--hidden-depth", type=int, default=2)
    parser.add_argument("--metrics-interval", type=int, default=250)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--replay-device", default="cpu")
    parser.add_argument("--compile-mode", default="off")
    parser.add_argument("--hardware-profile", default="auto")
    parser.add_argument("--mixed-precision", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--initial-equity", type=float, default=10_000.0)
    parser.add_argument("--leverage", type=float, default=1.0)
    parser.add_argument("--maker-fee-bps", type=float, default=2.0)
    parser.add_argument("--taker-fee-bps", type=float, default=5.0)
    parser.add_argument("--slippage-bps", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    positive = (args.trials, args.baseline_steps, args.optimization_steps, args.final_steps, args.parallel_envs, args.batch_size, args.replay_capacity, args.warmup_steps, args.gradient_steps, args.hidden_depth, args.metrics_interval)
    if min(positive) < 1 or args.replay_capacity < args.batch_size:
        raise SystemExit("training workload values must be positive and replay capacity >= batch size")
    algorithms = tuple(dict.fromkeys(item.strip().lower().replace("-", "_") for item in args.algorithms.split(",") if item.strip()))
    unknown = set(algorithms) - set(available_algorithms())
    if not algorithms or unknown:
        raise SystemExit(f"unknown/empty algorithms: {sorted(unknown)}")
    data_root = args.data_root.expanduser().resolve()
    if not data_root.is_dir():
        raise SystemExit(f"data root does not exist: {data_root}")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = (args.output_dir or Path(__file__).resolve().parents[1] / "artifacts" / "hprl-eth-two-year" / stamp).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    frame, feature_columns, source_files = build_feature_frame(data_root, primary=args.primary_timeframe)
    train_frame, validation_frame, test_frame = split_frame(frame)
    train_dataset = dataset_from_frame(train_frame, feature_columns)
    validation_dataset = dataset_from_frame(validation_frame, feature_columns)
    combined_dataset = dataset_from_frame(pd.concat((train_frame, validation_frame), ignore_index=True), feature_columns)
    test_dataset = dataset_from_frame(test_frame, feature_columns)
    _write_json(output / "dataset.json", {"sources": source_files, "primary_timeframe": args.primary_timeframe, "feature_columns": feature_columns, "rows": {"train": len(train_frame), "validation": len(validation_frame), "test": len(test_frame)}, "timestamp_ranges": {"train": [str(train_frame.timestamp.iloc[0]), str(train_frame.timestamp.iloc[-1])], "validation": [str(validation_frame.timestamp.iloc[0]), str(validation_frame.timestamp.iloc[-1])], "test": [str(test_frame.timestamp.iloc[0]), str(test_frame.timestamp.iloc[-1])]} })

    baseline_candidate = Candidate(3e-5, 0.99, 0.003, 128, 0.40, 0.0015)
    baseline: list[dict[str, Any]] = []
    print(f"[BASELINE] algorithms={','.join(algorithms)}", flush=True)
    for algorithm in algorithms:
        record, _, _ = train_and_validate(algorithm, baseline_candidate, train_dataset, validation_dataset, args, steps=args.baseline_steps)
        baseline.append(record)
        print(f"[{record['status']}] baseline {algorithm}", flush=True)
    successful = [row for row in baseline if row["status"] == "PASS"]
    if not successful:
        _write_json(output / "report.json", {"status": "FAIL", "baseline": baseline})
        return 1
    winner = max(successful, key=lambda row: float(row["objective"]))["algorithm"]
    trials: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    checkpoint = output / "best-validation.pt"
    print(f"[OPTIMIZE] winner={winner} trials={args.trials}", flush=True)
    for index, candidate in enumerate(candidate_grid(args.trials, args.seed), start=1):
        record, agent, config = train_and_validate(winner, candidate, train_dataset, validation_dataset, args, steps=args.optimization_steps)
        record["trial"] = index
        trials.append(record)
        if record["status"] == "PASS" and (best is None or float(record["objective"]) > float(best["objective"])):
            assert agent is not None and config is not None
            save_checkpoint(checkpoint, agent, {"algorithm": winner, "candidate": asdict(candidate), "validation": record["validation"], "objective": record["objective"], "feature_columns": feature_columns})
            best = record
        print(f"[{record['status']}] trial={index}/{args.trials}", flush=True)
    if best is None:
        _write_json(output / "report.json", {"status": "FAIL", "baseline": baseline, "optimization_trials": trials})
        return 1

    best_candidate = Candidate(**best["candidate"])
    final_config = make_config(winner, best_candidate, args, parallel_envs=args.parallel_envs)
    final_runtime = build_online_runtime(combined_dataset, final_config)
    try:
        metadata = load_checkpoint(checkpoint, final_runtime.agent)
        final_training = final_runtime.trainer.run(args.final_steps)
        final_metrics, actions = evaluate_agent(final_runtime.agent, test_dataset, final_config)
        final_collapsed, final_health = training_health(final_training)
        model_name = "rejected-hprl-eth-strategy.pt" if final_collapsed else "best-hprl-eth-strategy.pt"
        save_checkpoint(output / model_name, final_runtime.agent, {"algorithm": winner, "candidate": asdict(best_candidate), "selection_metadata": metadata, "final_training": asdict(final_training), "training_health": final_health, "holdout_backtest": final_metrics, "qualified": not final_collapsed, "feature_columns": feature_columns})
    finally:
        final_runtime.close(aggressive=True)
    with (output / "holdout-actions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("step", "long_policy", "short_policy", "equity", "directive"))
        writer.writeheader()
        for index, row in enumerate(actions):
            writer.writerow({"step": index, "long_policy": row["policy_action"][0], "short_policy": row["policy_action"][1], "equity": row["equity"], "directive": json.dumps(_json(row["directive"]), ensure_ascii=False)})
    payload: dict[str, Any] = {"schema": "hedge-hprl-eth-two-year-research-v1", "status": "REJECTED" if final_collapsed else "PASS", "generated_at": datetime.now(UTC).isoformat(), "baseline": baseline, "optimization_trials": trials, "best_configuration": {"algorithm": winner, "candidate": asdict(best_candidate), "validation_objective": best["objective"], "validation": best["validation"], "hprl_config": asdict(final_config)}, "final_training": asdict(final_training), "final_training_health": final_health, "final_holdout_backtest": final_metrics, "qualification": {"accepted": not final_collapsed, "reason": "training_health_policy_collapse" if final_collapsed else "passed"}}
    _write_json(output / "report.json", payload)
    _write_json(output / "best-parameters.json", payload["best_configuration"])
    write_summary(output, payload)
    archive = package(output)
    print(f"Report: {output / 'report.json'}")
    print(f"Summary: {output / 'BACKTEST-SUMMARY.md'}")
    print(f"Package: {archive}")
    return 0 if not final_collapsed else 2


if __name__ == "__main__":
    raise SystemExit(main())
