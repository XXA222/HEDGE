#!/usr/bin/env python3
"""Research-grade ETH HPRL training with learning-integrity qualification.

This workflow is intentionally fail-closed.  Candidate selection uses chronological validation,
behavior/activity qualification, numerical health, risk gates, multi-seed robustness and optional
walk-forward confirmation before any final holdout is opened.  Economic reward is never modified
merely to force the policy to trade.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
import zipfile
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from freqtrade.hedge.hprl.checkpoint import save_checkpoint
from freqtrade.hedge.hprl.checkpoint_resume import load_warm_start_weights
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
from freqtrade.hedge.hprl.diagnostics import NonFiniteTransitionError, write_artifact_manifest
from freqtrade.hedge.hprl.device import require_torch
from freqtrade.hedge.hprl.env import VectorizedHedgeEnv
from freqtrade.hedge.hprl.exchange_risk import load_exchange_risk_evidence
from freqtrade.hedge.hprl.evaluation import evaluate_trading
from freqtrade.hedge.hprl.qualification import (
    QualificationStatus,
    QualificationThresholds,
    actions_from_evaluation_rows,
    aggregate_seed_qualification,
    policy_activity,
    qualify_candidate,
    search_is_degenerate,
    select_winner,
    trading_objective,
)
from freqtrade.hedge.hprl.registry import available_algorithms
from freqtrade.hedge.hprl.runtime import build_online_runtime
from freqtrade.hedge.strategies.hprl_eth_dual_leg import HprlEthDualLegStrategy


TIMEFRAMES = ("1m", "15m", "1h", "8h", "1d")
PERIODS_PER_YEAR = {"1m": 525_600, "15m": 35_040, "1h": 8_760, "8h": 1_095, "1d": 365}
TIMEFRAME_SECONDS = {"1m": 60, "15m": 900, "1h": 3_600, "8h": 28_800, "1d": 86_400}
FEATURE_TOLERANCE = {
    "1m": pd.Timedelta(minutes=2),
    "15m": pd.Timedelta(minutes=30),
    "1h": pd.Timedelta(hours=2),
    "8h": pd.Timedelta(hours=16),
    "1d": pd.Timedelta(days=2),
}
SYMBOL = "ETH/USDT:USDT"
BASE_COMMIT = "395493fd23cb31ad63af0ebd2c72c612c7967293"
SCHEMA = "hedge-hprl-eth-learning-integrity-v2"


@dataclass(frozen=True, slots=True)
class Candidate:
    learning_rate: float
    gamma: float
    tau: float
    hidden_dim: int
    reward_drawdown: float
    reward_turnover: float


@dataclass(frozen=True, slots=True)
class SplitFrames:
    train: pd.DataFrame
    validation: pd.DataFrame
    combined: pd.DataFrame
    holdout: pd.DataFrame
    train_end: int
    validation_end: int
    purge: int


@dataclass(frozen=True, slots=True)
class ExpandingFold:
    train_end: int
    validation_start: int
    validation_end: int


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
    path.write_text(json.dumps(_json(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _find_unique_data_file(root: Path, timeframe: str) -> Path:
    candidates = sorted(root.rglob(f"eth-{timeframe}.csv"))
    if not candidates:
        raise FileNotFoundError(f"ETH {timeframe} CSV not found below {root}")
    if len(candidates) != 1:
        joined = ", ".join(str(path) for path in candidates[:10])
        raise ValueError(
            f"multiple ETH {timeframe} sources found; source authority is ambiguous: {joined}"
        )
    return candidates[0]


def _read_candles(path: Path, *, timeframe: str, max_gap_fraction: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = pd.read_csv(path)
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - {str(column).lower() for column in raw.columns}
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    raw.columns = [str(column).lower() for column in raw.columns]
    original_rows = len(raw)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True, errors="coerce")
    for name in ("open", "high", "low", "close", "volume"):
        raw[name] = pd.to_numeric(raw[name], errors="coerce")
    invalid_required = raw[["timestamp", "open", "high", "low", "close", "volume"]].isna().any(axis=1)
    invalid_rows = int(invalid_required.sum())
    if invalid_rows:
        raise ValueError(f"{path} contains {invalid_rows} invalid/NaN required candle rows")
    duplicate_count = int(raw["timestamp"].duplicated(keep=False).sum())
    if duplicate_count:
        raise ValueError(f"{path} contains duplicate timestamps: {duplicate_count}")
    frame = raw.sort_values("timestamp").reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"{path} is empty")
    ohlc = frame[["open", "high", "low", "close"]]
    if (ohlc <= 0).any().any():
        raise ValueError(f"{path} contains non-positive OHLC values")
    if (frame["volume"] < 0).any():
        raise ValueError(f"{path} contains negative volume")
    zero_volume_count = int((frame["volume"] == 0).sum())
    zero_volume_fraction = zero_volume_count / max(1, len(frame))
    if (frame["high"] < frame[["open", "close", "low"]].max(axis=1)).any():
        raise ValueError(f"{path} contains logically invalid high prices")
    if (frame["low"] > frame[["open", "close", "high"]].min(axis=1)).any():
        raise ValueError(f"{path} contains logically invalid low prices")
    expected = TIMEFRAME_SECONDS[timeframe]
    deltas = frame["timestamp"].diff().dt.total_seconds().dropna()
    gap_mask = deltas > expected * 1.5
    gap_count = int(gap_mask.sum())
    gap_fraction = gap_count / max(1, len(deltas))
    if gap_fraction > max_gap_fraction:
        raise ValueError(
            f"{path} gap fraction {gap_fraction:.6f} exceeds limit {max_gap_fraction:.6f}"
        )
    manifest = {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "rows": len(frame),
        "original_rows": original_rows,
        "first_timestamp": frame["timestamp"].iloc[0].isoformat(),
        "last_timestamp": frame["timestamp"].iloc[-1].isoformat(),
        "duplicate_count": duplicate_count,
        "invalid_rows": invalid_rows,
        "gap_count": gap_count,
        "gap_fraction": gap_fraction,
        "max_gap_seconds": float(deltas.max()) if len(deltas) else 0.0,
        "expected_interval_seconds": expected,
        "zero_volume_count": zero_volume_count,
        "zero_volume_fraction": zero_volume_fraction,
    }
    return frame, manifest


def _read_funding(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = pd.read_csv(path)
    frame.columns = [str(column).lower() for column in frame.columns]
    timestamp_name = "timestamp" if "timestamp" in frame.columns else "funding_time" if "funding_time" in frame.columns else None
    rate_name = "funding_rate" if "funding_rate" in frame.columns else "fundingrate" if "fundingrate" in frame.columns else None
    if timestamp_name is None or rate_name is None:
        raise ValueError("funding CSV requires timestamp/funding_time and funding_rate columns")
    frame["timestamp"] = pd.to_datetime(frame[timestamp_name], utc=True, errors="coerce")
    frame["funding_rate"] = pd.to_numeric(frame[rate_name], errors="coerce")
    if frame[["timestamp", "funding_rate"]].isna().any().any():
        raise ValueError("funding CSV contains invalid rows")
    if frame["timestamp"].duplicated().any():
        raise ValueError("funding CSV contains duplicate timestamps")
    if not np.isfinite(frame["funding_rate"].to_numpy(dtype=float)).all():
        raise ValueError("funding rates must be finite")
    frame = frame[["timestamp", "funding_rate"]].sort_values("timestamp").reset_index(drop=True)
    manifest = {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "rows": len(frame),
        "first_timestamp": frame["timestamp"].iloc[0].isoformat(),
        "last_timestamp": frame["timestamp"].iloc[-1].isoformat(),
    }
    return frame, manifest


def build_feature_frame(
    data_root: Path,
    *,
    primary: str,
    max_gap_fraction: float,
    funding_file: Path | None,
    require_funding: bool,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    if primary not in TIMEFRAMES:
        raise ValueError(f"primary timeframe must be one of {TIMEFRAMES}")
    candles: dict[str, pd.DataFrame] = {}
    source_manifest: dict[str, Any] = {}
    for timeframe in TIMEFRAMES:
        source_path = _find_unique_data_file(data_root, timeframe)
        candle, manifest = _read_candles(
            source_path,
            timeframe=timeframe,
            max_gap_fraction=max_gap_fraction,
        )
        candles[timeframe] = candle
        source_manifest[timeframe] = manifest

    frame = candles[primary][["timestamp", "close", "volume"]].copy().sort_values("timestamp")
    feature_columns: list[str] = []
    feature_age: dict[str, dict[str, float]] = {}
    for timeframe, source in candles.items():
        close = source["close"].astype(float)
        feature = pd.DataFrame({"source_timestamp": source["timestamp"]})
        columns = {
            f"{timeframe}_ret_1": close.pct_change(1),
            f"{timeframe}_ret_4": close.pct_change(4),
            f"{timeframe}_vol_16": close.pct_change().rolling(16).std(),
            f"{timeframe}_z_32": (close - close.rolling(32).mean()) / close.rolling(32).std(),
        }
        for name, values in columns.items():
            feature[name] = values.shift(1)
            feature_columns.append(name)
        feature = feature.sort_values("source_timestamp")
        frame = pd.merge_asof(
            frame,
            feature,
            left_on="timestamp",
            right_on="source_timestamp",
            direction="backward",
            allow_exact_matches=True,
            tolerance=FEATURE_TOLERANCE[timeframe],
        )
        ages = (frame["timestamp"] - frame["source_timestamp"]).dt.total_seconds()
        valid_ages = ages.dropna()
        feature_age[timeframe] = {
            "max_age_seconds": float(valid_ages.max()) if len(valid_ages) else math.inf,
            "mean_age_seconds": float(valid_ages.mean()) if len(valid_ages) else math.inf,
            "missing_after_tolerance": int(frame["source_timestamp"].isna().sum()),
            "tolerance_seconds": FEATURE_TOLERANCE[timeframe].total_seconds(),
        }
        frame = frame.drop(columns=["source_timestamp"])

    frame["forward_return"] = frame["close"].shift(-1) / frame["close"] - 1.0
    raw_available_notional = frame["close"] * frame["volume"]
    raw_liquidity = raw_available_notional.to_numpy(dtype=float)
    if not np.isfinite(raw_liquidity).all():
        raise ValueError("primary dataset contains non-finite available_notional")
    if (raw_available_notional < 0).any():
        raise ValueError("primary dataset contains negative available_notional")
    positive_liquidity_count = int((raw_available_notional > 0).sum())
    if positive_liquidity_count == 0:
        raise ValueError("primary dataset contains no positive-liquidity candles")
    zero_liquidity_count = int((raw_available_notional == 0).sum())
    # Zero-volume candles are valid exchange observations. TensorMarketDataset and the execution
    # cost model require a strictly-positive denominator, so retain the observation but use the
    # historical 1 USDT safety floor for the liquidity proxy. This makes any attempted trade on
    # the zero-volume bar maximally conservative while preserving the market timeline.
    available_notional_floor = 1.0
    frame["available_notional"] = raw_available_notional.clip(lower=available_notional_floor)
    liquidity_manifest = {
        "proxy": "close_x_volume",
        "available_notional_floor": available_notional_floor,
        "positive_liquidity_count": positive_liquidity_count,
        "zero_liquidity_count": zero_liquidity_count,
        "zero_liquidity_fraction": zero_liquidity_count / max(1, len(frame)),
        "floored_count": zero_liquidity_count,
    }

    funding_manifest: dict[str, Any] | None = None
    if funding_file is not None:
        funding, funding_manifest = _read_funding(funding_file.expanduser().resolve())
        # Funding is an event payment, not a forward-filled state.  Only exact event timestamps pay.
        frame = frame.merge(funding, on="timestamp", how="left")
        frame["funding_rate"] = frame["funding_rate"].fillna(0.0)
    else:
        frame["funding_rate"] = 0.0
        if require_funding:
            raise ValueError("--require-funding was set but no --funding-file was supplied")

    frame = frame.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    if len(frame) < 2_000:
        raise ValueError("insufficient clean feature rows after causal warmup/tolerance filtering")
    if not np.isfinite(frame[feature_columns + ["forward_return", "available_notional", "funding_rate"]].to_numpy(dtype=float)).all():
        raise ValueError("feature frame contains non-finite values after cleaning")

    manifest = {
        "sources": source_manifest,
        "funding": funding_manifest,
        "funding_enabled": funding_manifest is not None,
        "feature_age": feature_age,
        "liquidity": liquidity_manifest,
    }
    return frame, feature_columns, manifest


def dataset_from_frame(frame: pd.DataFrame, feature_columns: list[str]) -> TensorMarketDataset:
    torch = require_torch()
    features = torch.tensor(
        frame[feature_columns].to_numpy(dtype=np.float32), dtype=torch.float32
    ).unsqueeze(1)
    returns = torch.tensor(
        frame["forward_return"].to_numpy(dtype=np.float32), dtype=torch.float32
    ).unsqueeze(1)
    available = torch.tensor(
        frame["available_notional"].to_numpy(dtype=np.float32), dtype=torch.float32
    ).unsqueeze(1)
    funding = torch.tensor(
        frame["funding_rate"].to_numpy(dtype=np.float32), dtype=torch.float32
    ).unsqueeze(1)
    return TensorMarketDataset(
        features=features,
        forward_returns=returns,
        funding_rates=funding,
        available_notional=available,
        symbols=(SYMBOL,),
    ).validate()


def split_frame(frame: pd.DataFrame, *, purge: int = 1) -> SplitFrames:
    if purge < 1:
        raise ValueError("purge must be >= 1 for next-bar labels")
    train_end = int(len(frame) * 0.60)
    validation_end = int(len(frame) * 0.80)
    if train_end <= purge or validation_end - train_end <= purge:
        raise ValueError("dataset is too short for chronological split")
    train = frame.iloc[: train_end - purge].copy()
    validation = frame.iloc[train_end : validation_end - purge].copy()
    # Final retraining uses every pre-holdout row except the purge immediately before holdout.
    # This avoids the previous artificial train/validation concat gap.
    combined = frame.iloc[: validation_end - purge].copy()
    holdout = frame.iloc[validation_end:].copy()
    return SplitFrames(train, validation, combined, holdout, train_end, validation_end, purge)


def make_config(
    algorithm: str,
    candidate: Candidate,
    args: argparse.Namespace,
    *,
    seed: int,
    parallel_envs: int,
) -> HPRLConfig:
    action = HPRLActionConfig(
        position_levels=(0.0, 0.05, 0.12, 0.25, 0.40),
        leverage=args.leverage,
    )
    environment = HPRLEnvironmentConfig(
        initial_equity=args.initial_equity,
        parallel_envs=parallel_envs,
        annualization_periods=PERIODS_PER_YEAR[args.primary_timeframe],
        runtime_checks=args.runtime_checks,
        info_mode="training",
        action=action,
        costs=HPRLCostConfig(
            maker_fee_bps=args.maker_fee_bps,
            taker_fee_bps=args.taker_fee_bps,
            base_slippage_bps=args.slippage_bps,
        ),
        reward=HPRLRewardConfig(
            drawdown=candidate.reward_drawdown,
            turnover=candidate.reward_turnover,
        ),
    )
    training = HPRLTrainingConfig(
        algorithm=algorithm,
        seed=seed,
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
        runtime_checks=args.runtime_checks,
        metrics_interval=args.metrics_interval,
        compile_mode=args.compile_mode,
        expected_updates=args.final_steps,
        hardware_profile=args.hardware_profile,
        optimizer_backend="auto",
        replay_prefetch=True,
        health_fail_mode="stop",
        health_capture_trace=True,
        fast_td3_actor_temperature=args.fast_td3_actor_temperature,
        fast_td3_actor_output_mode=args.fast_td3_actor_output_mode,
        fast_td3_tier_exploration_epsilon=args.fast_td3_tier_exploration_epsilon,
    )
    return HPRLConfig(
        environment=environment,
        training=training,
        memory=HPRLMemoryConfig(dataset_mode="auto"),
    )


def training_health(training: object) -> dict[str, float]:
    values: dict[str, float] = {}
    for name, value in dict(getattr(training, "last_metrics", {})).items():
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values[str(name)] = float(value)
    if bool(getattr(training, "early_stopped", False)):
        values["training_health_collapsed"] = 1.0
    return values


def bounded_training_steps(requested_steps: int, dataset: TensorMarketDataset, *, max_market_sweeps: float) -> tuple[int, dict[str, float]]:
    rows = int(dataset.features.shape[0])
    realizable = max(1, rows - 1)
    maximum = max(1, int(math.floor(realizable * float(max_market_sweeps))))
    actual = min(int(requested_steps), maximum)
    return actual, {
        "requested_environment_steps": int(requested_steps),
        "environment_steps_cap": maximum,
        "unique_market_transitions_per_sweep": realizable,
        "requested_market_sweeps": float(requested_steps) / realizable,
        "planned_market_sweeps": float(actual) / realizable,
    }


def evaluate_agent(
    agent: object,
    dataset: TensorMarketDataset,
    config: HPRLConfig,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    torch = require_torch()
    env_config = replace(config.environment, parallel_envs=1, info_mode="full")
    environment = VectorizedHedgeEnv(
        dataset,
        env_config,
        device=config.training.device,
        memory_config=config.memory,
    )
    strategy = HprlEthDualLegStrategy()
    observation, _ = environment.reset()
    equity = [float(env_config.initial_equity)]
    turnover = fees = funding = 0.0
    synthetic_bankruptcies = 0
    actions: list[dict[str, object]] = []
    try:
        while True:
            with torch.no_grad():
                requested_action = agent.act(observation, deterministic=True)
            step = environment.step(requested_action)
            info = step.info
            requested_tensor = info.get("requested_policy_action", requested_action)
            requested = [float(value) for value in requested_tensor[0].detach().float().cpu().tolist()]
            executed_tensor = info.get("executed_action", requested_action)
            executed = [float(value) for value in executed_tensor[0].detach().float().cpu().tolist()]
            directive = strategy.directive_from_policy_action(executed)
            equity.append(max(float(info["equity"][0].item()), 1e-9))
            turnover += float(info["turnover_ratio"][0].item())
            fees += (
                float(info["fee_cost"][0].item())
                + float(info["slippage_cost"][0].item())
                + float(info["market_impact_cost"][0].item())
            )
            funding += float(info["funding_pnl_ratio"][0].item())
            synthetic_bankruptcies += int(bool(info["autoreset_mask"][0].item()))
            actions.append(
                {
                    "policy_action": executed,
                    "requested_action": requested,
                    "directive": asdict(directive),
                    "equity": equity[-1],
                    "quantization_distance": float(info.get("quantization_distance", torch.zeros(1, device=environment.device))[0].item()),
                    "constraint_distance": float(info.get("constraint_distance", torch.zeros(1, device=environment.device))[0].item()),
                    "transition_limited": bool(info.get("transition_limited", torch.zeros(1, dtype=torch.bool, device=environment.device))[0].item()),
                    "risk_limited": bool(info.get("risk_limited", torch.zeros(1, dtype=torch.bool, device=environment.device))[0].item()),
                    "projected": bool(info.get("projected", torch.zeros(1, dtype=torch.bool, device=environment.device))[0].item()),
                    "gross_margin_ratio": float(info.get("gross_margin_ratio", torch.zeros(1, device=environment.device))[0].item()),
                }
            )
            observation = step.observation
            if bool(info["time_done"]):
                break
    finally:
        environment.close(aggressive=True)
    metrics = asdict(
        evaluate_trading(
            equity,
            periods_per_year=env_config.annualization_periods,
            turnover=turnover,
            fees=fees,
            funding=funding,
            liquidations=synthetic_bankruptcies,
        )
    )
    return {key: float(value) for key, value in metrics.items()}, actions


def summarize_evaluation_behavior(rows: Sequence[Mapping[str, object]]) -> dict[str, float]:
    if not rows:
        return {
            "decisions": 0.0,
            "projection_fraction": 0.0,
            "transition_limited_fraction": 0.0,
            "risk_limited_fraction": 0.0,
            "mean_quantization_distance": 0.0,
            "mean_constraint_distance": 0.0,
            "mean_gross_margin_ratio": 0.0,
            "requested_executed_l1": 0.0,
        }
    count = len(rows)
    requested_executed = []
    for row in rows:
        requested = row["requested_action"]
        executed = row["policy_action"]
        requested_executed.append(sum(abs(float(a) - float(b)) for a, b in zip(requested, executed, strict=True)))
    return {
        "decisions": float(count),
        "projection_fraction": sum(bool(row.get("projected", False)) for row in rows) / count,
        "transition_limited_fraction": sum(bool(row.get("transition_limited", False)) for row in rows) / count,
        "risk_limited_fraction": sum(bool(row.get("risk_limited", False)) for row in rows) / count,
        "mean_quantization_distance": sum(float(row.get("quantization_distance", 0.0)) for row in rows) / count,
        "mean_constraint_distance": sum(float(row.get("constraint_distance", 0.0)) for row in rows) / count,
        "mean_gross_margin_ratio": sum(float(row.get("gross_margin_ratio", 0.0)) for row in rows) / count,
        "requested_executed_l1": sum(requested_executed) / count,
    }


def evaluate_fixed_policy(
    dataset: TensorMarketDataset,
    config: HPRLConfig,
    action_rows: Iterable[Sequence[float]],
) -> dict[str, float]:
    torch = require_torch()
    env_config = replace(config.environment, parallel_envs=1, info_mode="full")
    environment = VectorizedHedgeEnv(dataset, env_config, device=config.training.device, memory_config=config.memory)
    observation, _ = environment.reset()
    del observation
    equity = [float(env_config.initial_equity)]
    turnover = fees = funding = 0.0
    bankruptcies = 0
    iterator = iter(action_rows)
    try:
        while True:
            try:
                action = next(iterator)
            except StopIteration as exc:
                raise ValueError("benchmark action sequence ended before dataset") from exc
            tensor = torch.tensor([action], dtype=torch.float32, device=environment.device)
            step = environment.step(tensor)
            info = step.info
            equity.append(max(float(info["equity"][0].item()), 1e-9))
            turnover += float(info["turnover_ratio"][0].item())
            fees += float(info["fee_cost"][0].item()) + float(info["slippage_cost"][0].item()) + float(info["market_impact_cost"][0].item())
            funding += float(info["funding_pnl_ratio"][0].item())
            bankruptcies += int(bool(info["autoreset_mask"][0].item()))
            if bool(info["time_done"]):
                break
    finally:
        environment.close(aggressive=True)
    return {
        key: float(value)
        for key, value in asdict(
            evaluate_trading(
                equity,
                periods_per_year=env_config.annualization_periods,
                turnover=turnover,
                fees=fees,
                funding=funding,
                liquidations=bankruptcies,
            )
        ).items()
    }


def benchmark_suite(frame: pd.DataFrame, dataset: TensorMarketDataset, config: HPRLConfig) -> dict[str, Any]:
    steps = len(frame) - 1
    flat = [[0.0, 0.0] for _ in range(steps)]
    long_heavy = [[1.0, 0.0] for _ in range(steps)]
    short_heavy = [[0.0, 1.0] for _ in range(steps)]
    probe_long = [[0.25, 0.0] for _ in range(steps)]
    balanced_light_hedge = [[0.5, 0.5] for _ in range(steps)]
    trend: list[list[float]] = []
    signal_name = "1h_ret_4"
    if signal_name in frame:
        for value in frame[signal_name].iloc[:steps]:
            trend.append([0.5, 0.0] if float(value) >= 0.0 else [0.0, 0.5])
    else:
        trend = flat
    policies = {
        "flat": flat,
        "fixed_long_heavy": long_heavy,
        "fixed_short_heavy": short_heavy,
        "fixed_long_probe": probe_long,
        "static_balanced_light_hedge": balanced_light_hedge,
        "simple_trend": trend,
    }
    result = {}
    for name, actions in policies.items():
        metrics = evaluate_fixed_policy(dataset, config, actions)
        result[name] = {"metrics": metrics, "objective": trading_objective(metrics)}

    # Unconstrained market reference: 100% unlevered spot ETH buy-and-hold.  It is intentionally
    # separate from the HPRL 40%-margin envelope and is never used as an action-policy candidate.
    closes = frame["close"].to_numpy(dtype=float)
    entry_cost_ratio = (
        float(config.environment.costs.taker_fee_bps)
        + float(config.environment.costs.base_slippage_bps)
    ) / 10_000.0
    initial_equity = float(config.environment.initial_equity)
    invested_equity = initial_equity * (1.0 - entry_cost_ratio)
    spot_equity = [initial_equity] + [
        invested_equity * float(value / closes[0]) for value in closes[1:]
    ]
    spot_metrics = {
        key: float(value)
        for key, value in asdict(
            evaluate_trading(
                spot_equity,
                periods_per_year=config.environment.annualization_periods,
                turnover=1.0,
                fees=float(config.environment.initial_equity) * entry_cost_ratio,
                funding=0.0,
                liquidations=0,
            )
        ).items()
    }
    result["spot_buy_hold_reference"] = {
        "metrics": spot_metrics,
        "objective": trading_objective(spot_metrics),
        "semantics": "unlevered_spot_market_reference_outside_hprl_margin_envelope",
    }
    return result


def _exception_record(exc: Exception) -> dict[str, Any]:
    payload: dict[str, Any] = {"type": type(exc).__name__, "message": str(exc)}
    if isinstance(exc, NonFiniteTransitionError):
        payload["diagnostics"] = exc.to_dict()
    elif hasattr(exc, "to_dict"):
        try:
            payload["diagnostics"] = exc.to_dict()
        except Exception:
            pass
    return payload


def train_and_validate(
    algorithm: str,
    candidate: Candidate,
    train_dataset: TensorMarketDataset,
    validation_dataset: TensorMarketDataset,
    args: argparse.Namespace,
    *,
    seed: int,
    steps: int,
    thresholds: QualificationThresholds,
) -> tuple[dict[str, Any], object | None, HPRLConfig | None]:
    started = time.monotonic()
    config = make_config(algorithm, candidate, args, seed=seed, parallel_envs=args.parallel_envs)
    actual_steps, workload = bounded_training_steps(steps, train_dataset, max_market_sweeps=args.max_market_sweeps)
    runtime = None
    try:
        runtime = build_online_runtime(train_dataset, config)
        training = runtime.trainer.run(actual_steps)
        metrics, actions = evaluate_agent(runtime.agent, validation_dataset, config)
        health = training_health(training)
        activity_actions = actions_from_evaluation_rows(actions)
        decision = qualify_candidate(
            metrics=metrics,
            health=health,
            actions=activity_actions,
            thresholds=thresholds,
            flat_objective=0.0,
            level_count=config.environment.action.level_count,
        )
        record = {
            "status": decision.status.value,
            "accepted": decision.accepted,
            "algorithm": algorithm,
            "seed": seed,
            "candidate": asdict(candidate),
            "workload": workload,
            "training": asdict(training),
            "training_health": health,
            "validation": metrics,
            "activity": asdict(decision.activity),
            "behavior_diagnostics": summarize_evaluation_behavior(actions),
            "qualification": decision.to_dict(),
            "objective": decision.objective,
            "seconds": time.monotonic() - started,
        }
        if not decision.accepted:
            record["rejection_reasons"] = list(decision.reasons)
            return record, None, None
        return record, runtime.agent, config
    except Exception as exc:
        return (
            {
                "status": "ERROR",
                "accepted": False,
                "algorithm": algorithm,
                "seed": seed,
                "candidate": asdict(candidate),
                "workload": workload,
                "error": _exception_record(exc),
                "seconds": time.monotonic() - started,
            },
            None,
            None,
        )
    finally:
        if runtime is not None:
            runtime.close(aggressive=True)


def candidate_grid(trials: int, seed: int) -> list[Candidate]:
    conservative = Candidate(3e-5, 0.99, 0.003, 128, 0.40, 0.0015)
    all_candidates = [
        Candidate(*values)
        for values in itertools.product(
            (1e-5, 3e-5, 1e-4, 3e-4),
            (0.985, 0.99, 0.995),
            (0.003, 0.005, 0.01),
            (128, 256),
            (0.25, 0.40),
            (0.0005, 0.0015),
        )
    ]
    random.Random(seed).shuffle(all_candidates)
    return [conservative, *[item for item in all_candidates if item != conservative]][:trials]


def parse_seeds(text: str) -> tuple[int, ...]:
    values = tuple(dict.fromkeys(int(item.strip()) for item in text.split(",") if item.strip()))
    if not values or any(value < 0 for value in values):
        raise ValueError("seeds must contain one or more non-negative integers")
    return values


def aggregate_group(records: list[dict[str, Any]], thresholds: QualificationThresholds) -> dict[str, Any]:
    seed_rows = [
        {
            "accepted": bool(row.get("accepted", False)),
            "status": row.get("status"),
            "reasons": row.get("rejection_reasons", ()),
            "objective": row.get("objective", float("-inf")),
        }
        for row in records
    ]
    robustness = aggregate_seed_qualification(seed_rows, thresholds=thresholds)
    accepted_records = [row for row in records if row.get("accepted")]
    objectives = [float(row["objective"]) for row in accepted_records]
    return {
        "accepted": bool(robustness["accepted"]),
        "status": QualificationStatus.PASS.value if robustness["accepted"] else QualificationStatus.REJECTED_ROBUSTNESS.value,
        "objective": median(objectives) if objectives else float("-inf"),
        "robustness": robustness,
        "seed_results": records,
    }


def build_expanding_folds(length: int, *, folds: int, purge: int) -> tuple[ExpandingFold, ...]:
    if folds <= 0:
        return ()
    min_train = max(2_000, int(length * 0.45))
    remaining = length - min_train - folds * purge
    validation_size = remaining // folds
    if validation_size < 256:
        raise ValueError("pre-holdout data is too short for requested walk-forward folds")
    result = []
    train_end = min_train
    for _ in range(folds):
        validation_start = train_end + purge
        validation_end = validation_start + validation_size
        if validation_end > length:
            break
        result.append(ExpandingFold(train_end, validation_start, validation_end))
        train_end += validation_size
    if len(result) != folds:
        raise ValueError("could not construct requested walk-forward folds")
    return tuple(result)


def run_walk_forward(
    frame: pd.DataFrame,
    feature_columns: list[str],
    algorithm: str,
    candidate: Candidate,
    args: argparse.Namespace,
    thresholds: QualificationThresholds,
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    folds = build_expanding_folds(len(frame), folds=args.walk_forward_folds, purge=args.purge)
    results = []
    for fold_index, fold in enumerate(folds, start=1):
        train_frame = frame.iloc[: fold.train_end].copy()
        validation_frame = frame.iloc[fold.validation_start : fold.validation_end].copy()
        train_dataset = dataset_from_frame(train_frame, feature_columns)
        validation_dataset = dataset_from_frame(validation_frame, feature_columns)
        fold_seed_results = []
        for seed in seeds[: args.walk_forward_seed_count]:
            record, _, _ = train_and_validate(
                algorithm,
                candidate,
                train_dataset,
                validation_dataset,
                args,
                seed=seed,
                steps=args.walk_forward_steps,
                thresholds=thresholds,
            )
            fold_seed_results.append(record)
        aggregate = aggregate_group(fold_seed_results, thresholds)
        results.append(
            {
                "fold": fold_index,
                "indices": asdict(fold),
                "aggregate": aggregate,
            }
        )
    accepted = bool(results) and all(row["aggregate"]["accepted"] for row in results)
    objectives = [float(row["aggregate"]["objective"]) for row in results if math.isfinite(float(row["aggregate"]["objective"]))]
    return {
        "accepted": accepted,
        "status": QualificationStatus.PASS.value if accepted else QualificationStatus.REJECTED_ROBUSTNESS.value,
        "folds": results,
        "median_objective": median(objectives) if objectives else float("-inf"),
        "worst_objective": min(objectives) if objectives else float("-inf"),
    }


def write_action_csv(path: Path, actions: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "step",
                "requested_long",
                "requested_short",
                "executed_long",
                "executed_short",
                "equity",
                "directive",
            ),
        )
        writer.writeheader()
        for index, row in enumerate(actions):
            requested = row["requested_action"]
            executed = row["policy_action"]
            writer.writerow(
                {
                    "step": index,
                    "requested_long": requested[0],
                    "requested_short": requested[1],
                    "executed_long": executed[0],
                    "executed_short": executed[1],
                    "equity": row["equity"],
                    "directive": json.dumps(_json(row["directive"]), ensure_ascii=False),
                }
            )


def run_identity(root: Path, args: argparse.Namespace, seeds: tuple[int, ...]) -> dict[str, Any]:
    torch = require_torch()
    return {
        "schema": SCHEMA,
        "generated_at": datetime.now(UTC).isoformat(),
        "source_commit": _git_head(root),
        "expected_base_commit": BASE_COMMIT,
        "argv": sys.argv,
        "cwd": os.getcwd(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": getattr(torch, "__version__", "unknown"),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": getattr(torch.version, "cuda", None),
        "device_argument": args.device,
        "replay_device_argument": args.replay_device,
        "seeds": seeds,
    }


def write_summary(output: Path, payload: dict[str, Any]) -> None:
    final = payload.get("final_holdout_backtest") or {}
    qualification = payload.get("qualification") or {}
    winner = payload.get("selected_algorithm") or "NONE"
    candidate = payload.get("selected_candidate") or {}
    lines = [
        "# ETH HPRL Learning Integrity research summary",
        "",
        f"- Status: `{payload['status']}`",
        f"- Selected algorithm: `{winner}`",
        f"- Holdout role: `{payload['holdout_role']}`",
        f"- Final accepted: `{qualification.get('accepted', False)}`",
        f"- Release qualified: `{qualification.get('release_qualified', False)}`",
        "",
        "## Selected candidate",
        "",
        "```json",
        json.dumps(candidate, indent=2, sort_keys=True),
        "```",
        "",
        "## Final holdout",
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| Net return | {float(final.get('net_return', 0.0)):.4%} |",
        f"| Sharpe | {float(final.get('sharpe', 0.0)):.4f} |",
        f"| Sortino | {float(final.get('sortino', 0.0)):.4f} |",
        f"| Calmar | {float(final.get('calmar', 0.0)):.4f} |",
        f"| Max drawdown | {float(final.get('max_drawdown', 0.0)):.4%} |",
        f"| CVaR | {float(final.get('cvar', 0.0)):.4%} |",
        f"| Turnover | {float(final.get('turnover', 0.0)):.4f} |",
        f"| Fees/slippage/impact | {float(final.get('fees', 0.0)):.4f} |",
        f"| Funding PnL ratio sum | {float(final.get('funding', 0.0)):.6f} |",
        f"| Synthetic bankruptcies | {float(final.get('liquidations', 0.0)):.0f} |",
        "",
        "> `liquidations` in the HPRL simulator currently represents synthetic bankruptcy/autoreset events, not a Binance maintenance-margin liquidation engine.",
    ]
    reasons = qualification.get("reasons", [])
    if reasons:
        lines.extend(("", "## Qualification reasons", "", *[f"- {reason}" for reason in reasons]))
    (output / "BACKTEST-SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def package(output: Path) -> Path:
    archive = output.parent / f"HEDGE-HPRL-ETH-learning-integrity-{output.name}.zip"
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
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--trials", type=int, default=18)
    parser.add_argument("--optimization-confirm-top-k", type=int, default=3)
    parser.add_argument("--baseline-steps", type=int, default=30_000)
    parser.add_argument("--optimization-steps", type=int, default=50_000)
    parser.add_argument("--optimization-confirm-steps", type=int, default=50_000)
    parser.add_argument("--final-steps", type=int, default=100_000)
    parser.add_argument("--max-market-sweeps", type=float, default=8.0)
    parser.add_argument("--walk-forward-folds", type=int, default=3)
    parser.add_argument("--walk-forward-steps", type=int, default=30_000)
    parser.add_argument("--walk-forward-seed-count", type=int, default=2)
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
    parser.add_argument("--runtime-checks", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--initial-equity", type=float, default=10_000.0)
    parser.add_argument("--leverage", type=float, default=1.0)
    parser.add_argument("--maker-fee-bps", type=float, default=2.0)
    parser.add_argument("--taker-fee-bps", type=float, default=5.0)
    parser.add_argument("--slippage-bps", type=float, default=0.5)
    parser.add_argument("--fast-td3-actor-temperature", type=float, default=2.0)
    parser.add_argument("--fast-td3-actor-output-mode", choices=("sigmoid", "softsign", "tanh"), default="softsign")
    parser.add_argument("--fast-td3-tier-exploration-epsilon", type=float, default=0.10)
    parser.add_argument("--purge", type=int, default=1)
    parser.add_argument("--max-gap-fraction", type=float, default=0.01)
    parser.add_argument("--funding-file", type=Path)
    parser.add_argument("--require-funding", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--holdout-role", choices=("diagnostic", "blind"), default="diagnostic")
    parser.add_argument("--exchange-risk-mode", choices=("synthetic", "verified"), default="synthetic")
    parser.add_argument("--exchange-risk-evidence", type=Path)
    parser.add_argument("--final-init", choices=("fresh", "warm_start_weights"), default="fresh")
    parser.add_argument("--min-non-flat-decisions", type=int, default=8)
    parser.add_argument("--min-non-flat-fraction", type=float, default=0.002)
    parser.add_argument("--max-flat-fraction", type=float, default=0.998)
    parser.add_argument("--max-drawdown", type=float, default=0.35)
    parser.add_argument("--max-cvar", type=float, default=0.05)
    args = parser.parse_args(argv)

    numeric_positive = (
        args.trials,
        args.baseline_steps,
        args.optimization_steps,
        args.optimization_confirm_steps,
        args.final_steps,
        args.parallel_envs,
        args.batch_size,
        args.replay_capacity,
        args.gradient_steps,
        args.hidden_depth,
        args.metrics_interval,
    )
    if min(numeric_positive) < 1 or args.replay_capacity < args.batch_size:
        raise SystemExit("training workload values must be positive and replay capacity >= batch size")
    if args.warmup_steps < 0 or args.walk_forward_folds < 0 or args.walk_forward_seed_count < 1:
        raise SystemExit("warmup/walk-forward arguments are invalid")
    if not math.isfinite(args.max_market_sweeps) or args.max_market_sweeps <= 0:
        raise SystemExit("max-market-sweeps must be positive and finite")
    if not 0.0 <= args.max_gap_fraction <= 1.0:
        raise SystemExit("max-gap-fraction must be within [0,1]")

    seeds = parse_seeds(args.seeds)
    algorithms = tuple(
        dict.fromkeys(
            item.strip().lower().replace("-", "_")
            for item in args.algorithms.split(",")
            if item.strip()
        )
    )
    unknown = set(algorithms) - set(available_algorithms())
    if not algorithms or unknown:
        raise SystemExit(f"unknown/empty algorithms: {sorted(unknown)}")

    data_root = args.data_root.expanduser().resolve()
    if not data_root.is_dir():
        raise SystemExit(f"data root does not exist: {data_root}")
    exchange_risk_evidence = None
    if args.exchange_risk_evidence is not None:
        exchange_risk_evidence = load_exchange_risk_evidence(
            args.exchange_risk_evidence, expected_symbol=SYMBOL
        )
    if args.exchange_risk_mode == "verified" and exchange_risk_evidence is None:
        raise SystemExit(
            "exchange-risk-mode=verified requires --exchange-risk-evidence; manual assertion is not accepted"
        )
    exchange_risk_verified = bool(
        exchange_risk_evidence is not None and exchange_risk_evidence.verified
    )
    root = Path(__file__).resolve().parents[1]
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = (
        args.output_dir
        or root / "artifacts" / "hprl-eth-learning-integrity" / stamp
    ).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "validation-behavior").mkdir()

    thresholds = QualificationThresholds(
        min_non_flat_decisions=args.min_non_flat_decisions,
        min_non_flat_fraction=args.min_non_flat_fraction,
        max_flat_fraction=args.max_flat_fraction,
        max_drawdown=args.max_drawdown,
        max_cvar=args.max_cvar,
    )
    _write_json(output / "RUN-CONFIG.json", {"args": vars(args), "thresholds": asdict(thresholds), "identity": run_identity(root, args, seeds)})
    (output / "RUN-COMMAND.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")

    frame, feature_columns, source_manifest = build_feature_frame(
        data_root,
        primary=args.primary_timeframe,
        max_gap_fraction=args.max_gap_fraction,
        funding_file=args.funding_file,
        require_funding=args.require_funding,
    )
    splits = split_frame(frame, purge=args.purge)
    train_dataset = dataset_from_frame(splits.train, feature_columns)
    validation_dataset = dataset_from_frame(splits.validation, feature_columns)
    combined_dataset = dataset_from_frame(splits.combined, feature_columns)
    holdout_dataset = dataset_from_frame(splits.holdout, feature_columns)
    dataset_payload = {
        "primary_timeframe": args.primary_timeframe,
        "feature_columns": feature_columns,
        "source_manifest": source_manifest,
        "rows": {
            "full": len(frame),
            "train": len(splits.train),
            "validation": len(splits.validation),
            "combined": len(splits.combined),
            "holdout": len(splits.holdout),
        },
        "timestamp_ranges": {
            name: [str(value.timestamp.iloc[0]), str(value.timestamp.iloc[-1])]
            for name, value in {
                "train": splits.train,
                "validation": splits.validation,
                "combined": splits.combined,
                "holdout": splits.holdout,
            }.items()
        },
        "purge": args.purge,
        "holdout_role": args.holdout_role,
    }
    _write_json(output / "dataset.json", dataset_payload)

    baseline_candidate = Candidate(3e-5, 0.99, 0.003, 128, 0.40, 0.0015)
    baseline_algorithms: list[dict[str, Any]] = []
    print(f"[BASELINE] algorithms={','.join(algorithms)} seeds={seeds}", flush=True)
    for algorithm in algorithms:
        seed_results = []
        for seed in seeds:
            record, _, _ = train_and_validate(
                algorithm,
                baseline_candidate,
                train_dataset,
                validation_dataset,
                args,
                seed=seed,
                steps=args.baseline_steps,
                thresholds=thresholds,
            )
            seed_results.append(record)
            print(f"[{record['status']}] baseline {algorithm} seed={seed}", flush=True)
        aggregate = aggregate_group(seed_results, thresholds)
        aggregate["algorithm"] = algorithm
        aggregate["candidate"] = asdict(baseline_candidate)
        baseline_algorithms.append(aggregate)

    winner_row, winner_status = select_winner(
        baseline_algorithms,
        tie_tolerance=thresholds.objective_tie_tolerance,
    )
    if winner_row is None:
        payload = {
            "schema": SCHEMA,
            "status": winner_status.value,
            "generated_at": datetime.now(UTC).isoformat(),
            "baseline": baseline_algorithms,
            "dataset": dataset_payload,
        }
        _write_json(output / "report.json", payload)
        write_artifact_manifest(output)
        write_summary(output, {**payload, "holdout_role": args.holdout_role, "qualification": {"accepted": False, "release_qualified": False, "reasons": [winner_status.value]}})
        package(output)
        return 3
    winner = str(winner_row["algorithm"])

    # Screen the candidate grid with a single fixed seed.  Only qualified candidates proceed to
    # multi-seed confirmation, avoiding a 3x multiplication of already-invalid trials.
    screen_seed = seeds[0]
    trials: list[dict[str, Any]] = []
    print(f"[OPTIMIZE-SCREEN] winner={winner} trials={args.trials} seed={screen_seed}", flush=True)
    for index, candidate in enumerate(candidate_grid(args.trials, screen_seed), start=1):
        record, _, _ = train_and_validate(
            winner,
            candidate,
            train_dataset,
            validation_dataset,
            args,
            seed=screen_seed,
            steps=args.optimization_steps,
            thresholds=thresholds,
        )
        record["trial"] = index
        trials.append(record)
        print(f"[{record['status']}] trial={index}/{args.trials}", flush=True)

    if search_is_degenerate(trials):
        payload = {
            "schema": SCHEMA,
            "status": QualificationStatus.SEARCH_DEGENERATE.value,
            "generated_at": datetime.now(UTC).isoformat(),
            "baseline": baseline_algorithms,
            "optimization_trials": trials,
            "dataset": dataset_payload,
        }
        _write_json(output / "report.json", payload)
        write_artifact_manifest(output)
        write_summary(output, {**payload, "holdout_role": args.holdout_role, "qualification": {"accepted": False, "release_qualified": False, "reasons": ["optimization_objective_has_no_discriminating_signal"]}})
        package(output)
        return 4

    screened = [row for row in trials if row.get("accepted")]
    if not screened:
        payload = {
            "schema": SCHEMA,
            "status": QualificationStatus.NO_QUALIFIED_TRIAL.value,
            "generated_at": datetime.now(UTC).isoformat(),
            "baseline": baseline_algorithms,
            "optimization_trials": trials,
            "dataset": dataset_payload,
        }
        _write_json(output / "report.json", payload)
        write_artifact_manifest(output)
        write_summary(output, {**payload, "holdout_role": args.holdout_role, "qualification": {"accepted": False, "release_qualified": False, "reasons": ["no_screen_trial_passed_qualification"]}})
        package(output)
        return 5

    screened.sort(key=lambda row: float(row["objective"]), reverse=True)
    top_screened = screened[: min(args.optimization_confirm_top_k, len(screened))]
    confirmed: list[dict[str, Any]] = []
    print(f"[OPTIMIZE-CONFIRM] candidates={len(top_screened)} seeds={seeds}", flush=True)
    for rank, screen in enumerate(top_screened, start=1):
        candidate = Candidate(**screen["candidate"])
        seed_results = []
        for seed in seeds:
            record, _, _ = train_and_validate(
                winner,
                candidate,
                train_dataset,
                validation_dataset,
                args,
                seed=seed,
                steps=args.optimization_confirm_steps,
                thresholds=thresholds,
            )
            seed_results.append(record)
        aggregate = aggregate_group(seed_results, thresholds)
        aggregate.update({"rank_from_screen": rank, "candidate": asdict(candidate), "algorithm": winner})
        confirmed.append(aggregate)

    best_row, best_status = select_winner(
        confirmed,
        tie_tolerance=thresholds.objective_tie_tolerance,
    )
    if best_row is None:
        payload = {
            "schema": SCHEMA,
            "status": best_status.value.replace("ALGORITHM", "TRIAL"),
            "generated_at": datetime.now(UTC).isoformat(),
            "baseline": baseline_algorithms,
            "optimization_trials": trials,
            "optimization_confirmation": confirmed,
            "dataset": dataset_payload,
        }
        _write_json(output / "report.json", payload)
        write_artifact_manifest(output)
        write_summary(output, {**payload, "holdout_role": args.holdout_role, "qualification": {"accepted": False, "release_qualified": False, "reasons": [best_status.value]}})
        package(output)
        return 6
    best_candidate = Candidate(**best_row["candidate"])

    walk_forward = run_walk_forward(
        splits.combined,
        feature_columns,
        winner,
        best_candidate,
        args,
        thresholds,
        seeds,
    ) if args.walk_forward_folds else {"accepted": True, "status": "SKIPPED", "folds": []}
    if not walk_forward["accepted"]:
        payload = {
            "schema": SCHEMA,
            "status": QualificationStatus.REJECTED_ROBUSTNESS.value,
            "generated_at": datetime.now(UTC).isoformat(),
            "baseline": baseline_algorithms,
            "optimization_trials": trials,
            "optimization_confirmation": confirmed,
            "walk_forward": walk_forward,
            "selected_algorithm": winner,
            "selected_candidate": asdict(best_candidate),
            "dataset": dataset_payload,
        }
        _write_json(output / "report.json", payload)
        write_artifact_manifest(output)
        write_summary(output, {**payload, "holdout_role": args.holdout_role, "qualification": {"accepted": False, "release_qualified": False, "reasons": ["walk_forward_robustness_failed"]}})
        package(output)
        return 7

    final_seed = seeds[0]
    final_config = make_config(winner, best_candidate, args, seed=final_seed, parallel_envs=args.parallel_envs)
    benchmark_validation = benchmark_suite(splits.validation, validation_dataset, final_config)
    final_runtime = build_online_runtime(combined_dataset, final_config)
    warm_start_metadata: dict[str, Any] | None = None
    selected_checkpoint = output / "selected-validation.pt"
    if args.final_init == "warm_start_weights":
        # Train a dedicated selected model on train only.  It is used solely as a weight warm start;
        # optimizer/replay/normalizer/health state are intentionally not restored.
        selection_runtime = build_online_runtime(train_dataset, final_config)
        try:
            selection_training = selection_runtime.trainer.run(args.optimization_confirm_steps)
            selection_metrics, selection_actions = evaluate_agent(selection_runtime.agent, validation_dataset, final_config)
            selection_health = training_health(selection_training)
            selection_decision = qualify_candidate(
                metrics=selection_metrics,
                health=selection_health,
                actions=actions_from_evaluation_rows(selection_actions),
                thresholds=thresholds,
            )
            if not selection_decision.accepted:
                raise RuntimeError(f"dedicated warm-start model failed qualification: {selection_decision.reasons}")
            save_checkpoint(selected_checkpoint, selection_runtime.agent, {"resume_mode": "warm_start_weights", "candidate": asdict(best_candidate)})
        finally:
            selection_runtime.close(aggressive=True)
        warm_start_metadata = load_warm_start_weights(selected_checkpoint, final_runtime.agent)

    try:
        final_actual_steps, final_workload = bounded_training_steps(
            args.final_steps, combined_dataset, max_market_sweeps=args.max_market_sweeps
        )
        final_training = final_runtime.trainer.run(final_actual_steps)
        final_health = training_health(final_training)
        final_metrics, actions = evaluate_agent(final_runtime.agent, holdout_dataset, final_config)
        final_decision = qualify_candidate(
            metrics=final_metrics,
            health=final_health,
            actions=actions_from_evaluation_rows(actions),
            thresholds=thresholds,
        )
        research_accepted = bool(final_decision.accepted and walk_forward["accepted"])
        release_qualified = bool(
            research_accepted
            and args.holdout_role == "blind"
            and args.exchange_risk_mode == "verified"
            and exchange_risk_verified
            and bool(source_manifest.get("funding_enabled"))
        )
        model_name = "best-hprl-eth-strategy.pt" if release_qualified else "research-hprl-eth-strategy.pt" if research_accepted else "rejected-hprl-eth-strategy.pt"
        save_checkpoint(
            output / model_name,
            final_runtime.agent,
            {
                "algorithm": winner,
                "candidate": asdict(best_candidate),
                "final_init": args.final_init,
                "warm_start_metadata": warm_start_metadata,
                "final_training": asdict(final_training),
                "training_health": final_health,
                "holdout_backtest": final_metrics,
                "qualification": final_decision.to_dict(),
                "release_qualified": release_qualified,
                "feature_columns": feature_columns,
            },
        )
    finally:
        final_runtime.close(aggressive=True)

    write_action_csv(output / "holdout-actions.csv", actions)
    benchmark_holdout = benchmark_suite(splits.holdout, holdout_dataset, final_config)
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "PASS" if release_qualified else "RESEARCH_PASS" if research_accepted else "REJECTED",
        "generated_at": datetime.now(UTC).isoformat(),
        "holdout_role": args.holdout_role,
        "baseline": baseline_algorithms,
        "optimization_trials": trials,
        "optimization_space": {
            "grid_size": 288,
            "screen_trials": args.trials,
            "screen_fraction": args.trials / 288.0,
            "confirmation_top_k": args.optimization_confirm_top_k,
            "screen_seed": seeds[0],
            "confirmation_seeds": seeds,
        },
        "optimization_confirmation": confirmed,
        "walk_forward": walk_forward,
        "selected_algorithm": winner,
        "selected_candidate": asdict(best_candidate),
        "final_config": asdict(final_config),
        "final_workload": final_workload,
        "final_training": asdict(final_training),
        "final_training_health": final_health,
        "final_holdout_backtest": final_metrics,
        "final_activity": asdict(policy_activity(actions_from_evaluation_rows(actions))),
        "final_behavior_diagnostics": summarize_evaluation_behavior(actions),
        "benchmarks": {"validation": benchmark_validation, "holdout": benchmark_holdout},
        "qualification": {
            **final_decision.to_dict(),
            "walk_forward_ok": bool(walk_forward["accepted"]),
            "release_qualified": release_qualified,
            "holdout_role": args.holdout_role,
            "reasons": (
                list(final_decision.reasons)
                + ([] if args.holdout_role == "blind" else ["holdout_is_diagnostic_not_blind"])
                + ([] if exchange_risk_verified else ["exchange_liquidation_risk_not_externally_verified"])
                + ([] if source_manifest.get("funding_enabled") else ["funding_data_not_enabled"])
            ),
        },
        "dataset": dataset_payload,
        "training_semantics": {
            "parallel_envs_are_independent_accounts_not_independent_market_histories": True,
            "vector_transitions_are_not_unique_market_samples": True,
            "final_equity_mean_is_training_trajectory_diagnostic_not_holdout_performance": True,
            "max_market_sweeps": args.max_market_sweeps,
        },
        "risk_semantics": {
            "liquidations_field": "synthetic_bankruptcy_autoreset_proxy",
            "exchange_like_liquidation_engine": exchange_risk_verified,
            "exchange_risk_mode": args.exchange_risk_mode,
            "exchange_risk_evidence": (
                exchange_risk_evidence.to_dict() if exchange_risk_evidence is not None else None
            ),
            "funding_enabled": bool(source_manifest.get("funding_enabled")),
        },
    }
    _write_json(output / "report.json", payload)
    _write_json(
        output / "best-parameters.json",
        {
            "algorithm": winner,
            "candidate": asdict(best_candidate),
            "qualification_thresholds": asdict(thresholds),
        },
    )
    write_summary(output, payload)
    write_artifact_manifest(output)
    archive = package(output)
    print(f"Report: {output / 'report.json'}")
    print(f"Summary: {output / 'BACKTEST-SUMMARY.md'}")
    print(f"Package: {archive}")
    if release_qualified:
        return 0
    if research_accepted:
        return 8  # Deliberately non-zero: diagnostic holdout is not a release gate.
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
