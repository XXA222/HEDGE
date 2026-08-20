# ruff: noqa: E402, I001
from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
_here_text = str(HERE)
if _here_text in sys.path:
    sys.path.remove(_here_text)
sys.path.insert(0, _here_text)

from artifact_contract import (
    ARTIFACT_MANIFEST_SCHEMA,
    MODEL_METADATA_SCHEMA,
    atomic_save_npz,
    atomic_write_json,
    canonical_sha256,
    runtime_contract_payload,
    sha256_array,
    sha256_file,
    source_contract_payload,
    verify_manifest_files,
)
from features import FEATURE_VERSION, apply_scaler, fit_scaler, training_arrays_mtf
from suite_specs import (
    ACTION_KWARGS,
    COST_KWARGS,
    MODELS,
    MTF_ALIGNMENT_CONTRACT,
    PERIODS_PER_YEAR,
    SOURCE_WARMUP_CANDLES,
    TIMEFRAME_SECONDS,
    TRAIN_STEPS,
    WINDOW_STEPS,
    input_timeframes_for,
    reward_kwargs,
)


def _json(path: Path, payload: object) -> None:
    atomic_write_json(path, payload)


def _load_history(repo_root: Path, datadir: Path, timeframe: str, data_format: str):
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from freqtrade.data.history.history_utils import load_pair_history
    from freqtrade.enums import CandleType

    formats = (data_format,) if data_format != "auto" else ("feather", "parquet", "json", "jsongz")
    errors: list[str] = []
    for fmt in formats:
        try:
            frame = load_pair_history(
                pair="ETH/USDT:USDT",
                timeframe=timeframe,
                datadir=datadir,
                fill_up_missing=True,
                drop_incomplete=False,
                startup_candles=0,
                data_format=fmt,
                candle_type=CandleType.FUTURES,
            )
            if frame is not None and not frame.empty:
                return frame, fmt
        except Exception as exc:
            errors.append(f"{fmt}: {type(exc).__name__}: {exc}")
    raise FileNotFoundError(
        f"No local ETH/USDT:USDT {timeframe} futures OHLCV under {datadir}. "
        f"Tried formats={formats}; errors={errors}"
    )


def _source_frame(frame, timeframe: str, start: str, end: str):
    """Keep only the requested interval plus causal pre-start feature warmup."""
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize("UTC")
    else:
        start_ts = start_ts.tz_convert("UTC")
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")
    else:
        end_ts = end_ts.tz_convert("UTC")
    if not start_ts < end_ts:
        raise ValueError("training start must be earlier than training end")
    warmup = pd.Timedelta(seconds=TIMEFRAME_SECONDS[timeframe] * SOURCE_WARMUP_CANDLES)
    source_start = start_ts - warmup
    dates = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    clipped = frame.loc[(dates >= source_start) & (dates < end_ts)].copy()
    in_window = int(((dates >= start_ts) & (dates < end_ts)).sum())
    if in_window < 1:
        raise ValueError(
            f"Training source window for {timeframe} has no in-range candles: "
            f"{start_ts} .. {end_ts}"
        )
    return clipped


def _native(repo_root: Path):
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from freqtrade.hedge.hprl.action_space import configure_agent_action_levels
    from freqtrade.hedge.hprl.checkpoint import save_checkpoint
    from freqtrade.hedge.hprl.config import (
        HPRLActionConfig,
        HPRLCostConfig,
        HPRLEnvironmentConfig,
        HPRLMemoryConfig,
        HPRLRewardConfig,
        HPRLTrainingConfig,
    )
    from freqtrade.hedge.hprl.data import TensorMarketDataset
    from freqtrade.hedge.hprl.env import VectorizedHedgeEnv
    from freqtrade.hedge.hprl.registry import create_agent
    from freqtrade.hedge.hprl.replay import TensorReplayBuffer
    from freqtrade.hedge.hprl.trainer import DiscountedReturnNormalizer, OfflineTrainer

    return {
        "configure_agent_action_levels": configure_agent_action_levels,
        "save_checkpoint": save_checkpoint,
        "HPRLActionConfig": HPRLActionConfig,
        "HPRLCostConfig": HPRLCostConfig,
        "HPRLEnvironmentConfig": HPRLEnvironmentConfig,
        "HPRLMemoryConfig": HPRLMemoryConfig,
        "HPRLRewardConfig": HPRLRewardConfig,
        "HPRLTrainingConfig": HPRLTrainingConfig,
        "TensorMarketDataset": TensorMarketDataset,
        "VectorizedHedgeEnv": VectorizedHedgeEnv,
        "create_agent": create_agent,
        "TensorReplayBuffer": TensorReplayBuffer,
        "DiscountedReturnNormalizer": DiscountedReturnNormalizer,
        "OfflineTrainer": OfflineTrainer,
    }


def _configs(api, spec, timeframe: str, device: str, parallel_envs: int, steps: int):
    action = api["HPRLActionConfig"](**ACTION_KWARGS)
    costs = api["HPRLCostConfig"](**COST_KWARGS)
    reward = api["HPRLRewardConfig"](**reward_kwargs(spec))
    env = api["HPRLEnvironmentConfig"](
        initial_equity=1000.0,
        parallel_envs=parallel_envs,
        annualization_periods=PERIODS_PER_YEAR[timeframe],
        cvar_alpha=0.05,
        terminate_equity_ratio=0.20,
        runtime_checks=False,
        info_mode="training",
        action=action,
        costs=costs,
        reward=reward,
    )
    train = api["HPRLTrainingConfig"](
        algorithm=spec.algorithm,
        seed=42,
        device=device,
        replay_device="same",
        batch_size=spec.batch_size,
        replay_capacity=spec.replay_capacity,
        warmup_steps=spec.warmup_transitions,
        gradient_steps=1,
        gamma=spec.gamma,
        tau=spec.tau,
        learning_rate=spec.learning_rate,
        hidden_dim=spec.hidden_dim,
        hidden_depth=spec.hidden_depth,
        gradient_clip_norm=spec.gradient_clip_norm,
        mixed_precision=False,
        compile_mode="off",
        expected_updates=steps,
        metrics_interval=max(100, steps // 10),
        tier_entropy_target_fraction=spec.tier_entropy_target_fraction,
        runtime_checks=False,
    )
    memory = api["HPRLMemoryConfig"](
        dataset_mode="auto",
        dataset_window_steps=16_384,
        dataset_gpu_fraction=0.20,
        replay_gpu_fraction=0.30,
        release_offline_source_after_tensorize=True,
    )
    return action, costs, reward, env, train, memory


def _dataset(api, torch, x, y, available):
    zeros = np.zeros(len(x), dtype=np.float32)
    return api["TensorMarketDataset"](
        features=torch.as_tensor(x[:, None, :], dtype=torch.float32),
        forward_returns=torch.as_tensor(y[:, None], dtype=torch.float32),
        funding_rates=torch.as_tensor(zeros[:, None], dtype=torch.float32),
        available_notional=torch.as_tensor(available[:, None], dtype=torch.float32),
        symbols=("ETHUSDT",),
    ).validate()


def _train_online_windowed(
    api, torch, agent, dataset, env_cfg, train_cfg, mem_cfg, steps, segment_steps
):
    env = api["VectorizedHedgeEnv"](
        dataset, env_cfg, device=train_cfg.device, memory_config=mem_cfg
    )
    buffer = api["TensorReplayBuffer"](
        train_cfg.replay_capacity,
        env.observation_dim,
        env.action_dim,
        device=str(agent.device),
        pin_memory=False,
        validate_inputs=False,
    )
    normalizer = None
    if getattr(agent, "reward_normalization", None) == "return_std":
        normalizer = api["DiscountedReturnNormalizer"](
            env.envs, train_cfg.gamma, device=agent.device, validate_inputs=False
        )
    rng = np.random.default_rng(train_cfg.seed)
    transition_count = 0
    update_count = 0
    last_metrics: dict[str, float | None] = {}
    obs = None
    segment_left = 0
    try:
        for _decision_step in range(int(steps)):
            if obs is None or segment_left <= 0:
                max_start = max(0, env.market.time_steps - segment_steps - 2)
                start = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
                obs, _ = env.reset(start_index=start)
                segment_left = min(segment_steps, env.market.time_steps - start - 1)
            action = (
                env.sample_random_action()
                if transition_count < train_cfg.warmup_steps
                else agent.act(obs, deterministic=False)
            )
            step = env.step(action)
            segment_left -= 1
            done = torch.logical_or(step.terminated, step.truncated)
            if segment_left <= 0:
                done = torch.ones_like(done, dtype=torch.bool)
            reward = step.reward if normalizer is None else normalizer.normalize(step.reward, done)
            executed = step.info.get("executed_action")
            if executed is None:
                raise RuntimeError("native HPRL environment did not expose executed_action")
            buffer.add(obs, executed, reward, step.observation, done)
            transition_count += env.envs
            obs = step.observation
            if len(buffer) >= train_cfg.batch_size and transition_count >= train_cfg.warmup_steps:
                batch = buffer.sample_reusable(train_cfg.batch_size)
                collect = update_count == 0 or (update_count + 1) % train_cfg.metrics_interval == 0
                metrics = agent.update(batch, collect_metrics=collect)
                if getattr(metrics, "values", None):
                    last_metrics = {
                        str(k): (float(v) if math.isfinite(float(v)) else None)
                        for k, v in dict(metrics.values).items()
                    }
                update_count += 1
            if bool(step.info.get("time_done", False)):
                obs = None
                segment_left = 0
        return {
            "decision_steps": int(steps),
            "transitions": int(transition_count),
            "updates": int(update_count),
            "last_metrics": last_metrics,
        }
    finally:
        cleanup_errors: list[str] = []
        for label, cleanup in (
            ("buffer", lambda: buffer.release(aggressive=False)),
            (
                "normalizer",
                lambda: None if normalizer is None else normalizer.release(),
            ),
            ("environment", lambda: env.close(aggressive=False)),
        ):
            try:
                cleanup()
            except Exception as exc:
                cleanup_errors.append(f"{label}:{type(exc).__name__}:{exc}")
        if cleanup_errors:
            print(
                "HPRL online cleanup warnings: " + "; ".join(cleanup_errors),
                file=sys.stderr,
            )


class _OfflineDataset:
    action_unit = "policy_code"

    def __init__(self, obs, action, reward, next_obs, done):
        self.obs = obs
        self.action = action
        self.reward = reward
        self.next_obs = next_obs
        self.done = done
        self.observation_dim = int(obs.shape[1])
        self.action_dim = int(action.shape[1])

    def __len__(self):
        return int(self.obs.shape[0])

    def tensors(self, device="cpu", *, chunk_rows=4096):
        del chunk_rows
        import torch
        target = torch.device(device)
        return {
            "obs": torch.as_tensor(self.obs, dtype=torch.float32, device=target),
            "action": torch.as_tensor(self.action, dtype=torch.float32, device=target),
            "reward": torch.as_tensor(self.reward, dtype=torch.float32, device=target),
            "next_obs": torch.as_tensor(self.next_obs, dtype=torch.float32, device=target),
            "done": torch.as_tensor(self.done, dtype=torch.float32, device=target),
        }

    def release_source(self):
        self.obs = np.empty((0, self.observation_dim), dtype=np.float32)
        self.action = np.empty((0, self.action_dim), dtype=np.float32)
        self.reward = np.empty((0, 1), dtype=np.float32)
        self.next_obs = np.empty((0, self.observation_dim), dtype=np.float32)
        self.done = np.empty((0, 1), dtype=np.float32)


def _offline_behavior(
    api,
    torch,
    dataset,
    env_cfg,
    train_cfg,
    mem_cfg,
    rows_target,
    segment_steps,
    signal_index,
):
    env = api["VectorizedHedgeEnv"](
        dataset, env_cfg, device=train_cfg.device, memory_config=mem_cfg
    )
    rng = np.random.default_rng(train_cfg.seed + 101)
    obs_rows: list[np.ndarray] = []
    action_rows: list[np.ndarray] = []
    reward_rows: list[np.ndarray] = []
    next_rows: list[np.ndarray] = []
    done_rows: list[np.ndarray] = []
    total = 0
    obs = None
    segment_left = 0
    market_index = 0
    levels = env.config.action.level_count
    try:
        while total < rows_target:
            if obs is None or segment_left <= 0:
                max_start = max(0, env.market.time_steps - segment_steps - 2)
                start = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
                obs, _ = env.reset(start_index=start)
                market_index = start
                segment_left = min(segment_steps, env.market.time_steps - start - 1)
            market_signal = float(dataset.features[market_index, 0, signal_index].item())
            strength = min(1.0, abs(math.tanh(market_signal / 2.0)))
            tier = max(0, min(levels - 1, round(strength * (levels - 1))))
            code = tier / float(levels - 1)
            base = np.array([code, 0.0] if market_signal >= 0 else [0.0, code], dtype=np.float32)
            actions = np.repeat(base[None, :], env.envs, axis=0)
            random_mask = rng.random(env.envs) < 0.22
            random_tier = rng.integers(0, levels, size=(env.envs, 2))
            actions[random_mask] = random_tier[random_mask] / float(levels - 1)
            action = torch.as_tensor(actions, dtype=torch.float32, device=obs.device)
            step = env.step(action)
            segment_left -= 1
            done = torch.logical_or(step.terminated, step.truncated)
            if segment_left <= 0:
                done = torch.ones_like(done, dtype=torch.bool)
            executed = step.info.get("executed_action")
            if executed is None:
                raise RuntimeError("native HPRL environment did not expose executed_action")
            obs_rows.append(obs.detach().float().cpu().numpy())
            action_rows.append(executed.detach().float().cpu().numpy())
            reward_rows.append(step.reward.reshape(-1, 1).detach().float().cpu().numpy())
            next_rows.append(step.observation.detach().float().cpu().numpy())
            done_rows.append(done.reshape(-1, 1).detach().float().cpu().numpy())
            total += env.envs
            obs = step.observation
            market_index += 1
            if bool(step.info.get("time_done", False)):
                obs = None
                segment_left = 0
        return _OfflineDataset(
            np.concatenate(obs_rows, axis=0)[:rows_target],
            np.concatenate(action_rows, axis=0)[:rows_target],
            np.concatenate(reward_rows, axis=0)[:rows_target],
            np.concatenate(next_rows, axis=0)[:rows_target],
            np.concatenate(done_rows, axis=0)[:rows_target],
        )
    finally:
        try:
            env.close(aggressive=False)
        except Exception as exc:
            print(
                f"HPRL offline behavior cleanup warning: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )


def _runtime_contract(
    spec, timeframe: str, feature_names: tuple[str, ...], *, repo_root: Path
) -> dict[str, object]:
    source_contract = source_contract_payload(suite_root=HERE, repo_root=repo_root)
    inputs = input_timeframes_for(timeframe)
    return runtime_contract_payload(
        model_key=spec.key,
        algorithm=spec.algorithm,
        strategy_class=spec.strategy_class,
        base_timeframe=timeframe,
        input_timeframes=inputs,
        feature_version=FEATURE_VERSION,
        feature_names=feature_names,
        alignment_contract=MTF_ALIGNMENT_CONTRACT,
        action_config=ACTION_KWARGS,
        cost_config=COST_KWARGS,
        reward_config=reward_kwargs(spec),
        model_spec=asdict(spec),
        source_contract=source_contract,
    )


def _artifacts_reusable(
    artifact_dir: Path,
    *,
    runtime_contract_sha256: str,
    artifact_contract_sha256: str,
) -> bool:
    manifest_path = artifact_dir / "artifact_manifest.json"
    metadata_path = artifact_dir / "metadata.json"
    if not manifest_path.is_file() or not metadata_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != ARTIFACT_MANIFEST_SCHEMA:
            return False
        if metadata.get("schema") != MODEL_METADATA_SCHEMA:
            return False
        if manifest.get("runtime_contract_sha256") != runtime_contract_sha256:
            return False
        if manifest.get("artifact_contract_sha256") != artifact_contract_sha256:
            return False
        if metadata.get("runtime_contract_sha256") != runtime_contract_sha256:
            return False
        if metadata.get("artifact_contract_sha256") != artifact_contract_sha256:
            return False
        verify_manifest_files(artifact_dir, manifest)
        return True
    except Exception:
        return False


def prepare_one(args) -> int:
    repo_root = Path(args.repo_root).resolve()
    datadir = Path(args.datadir).resolve()
    user_data = Path(args.user_data_dir).resolve()
    spec = MODELS[args.model]
    base_timeframe = args.timeframe
    input_timeframes = input_timeframes_for(base_timeframe)
    artifact_dir = user_data / "hprl_freqtrade_models" / spec.key / base_timeframe
    checkpoint = artifact_dir / "checkpoint.pt"
    checkpoint_sidecar = artifact_dir / "checkpoint.pt.json"
    scaler_path = artifact_dir / "scaler.npz"
    metadata_path = artifact_dir / "metadata.json"
    manifest_path = artifact_dir / "artifact_manifest.json"

    frames: dict[str, pd.DataFrame] = {}
    data_formats: dict[str, str] = {}
    for timeframe in input_timeframes:
        frame, resolved_format = _load_history(repo_root, datadir, timeframe, args.data_format)
        frames[timeframe] = _source_frame(frame, timeframe, args.train_start, args.train_end)
        data_formats[timeframe] = resolved_format
        del frame
    resolved_formats = set(data_formats.values())
    if len(resolved_formats) != 1:
        raise ValueError(
            "Formal Freqtrade backtesting accepts one OHLCV data handler per run, but the MTF "
            f"sources resolved to different formats: {data_formats}. Convert them to one format."
        )
    data_format = next(iter(resolved_formats))

    x_raw, y, available, timestamps, feature_names, _, alignment_diagnostics = (
        training_arrays_mtf(
            frames,
            base_timeframe=base_timeframe,
            input_timeframes=input_timeframes,
            start=args.train_start,
            end=args.train_end,
        )
    )
    if len(x_raw) < 256:
        raise ValueError(f"Only {len(x_raw)} usable causal MTF training rows")
    mean, std = fit_scaler(x_raw)
    x = apply_scaler(x_raw, mean, std)

    runtime_contract = _runtime_contract(
        spec, base_timeframe, feature_names, repo_root=repo_root
    )
    runtime_contract_sha = canonical_sha256(runtime_contract)
    training_data_contract = {
        "schema": "hprl-freqtrade-mtf-training-data-contract-v2",
        "train_start": args.train_start,
        "train_end": args.train_end,
        "base_timeframe": base_timeframe,
        "input_timeframes": list(input_timeframes),
        "training_rows": len(x_raw),
        "feature_count": int(x_raw.shape[1]),
        "features_sha256": sha256_array(x_raw),
        "forward_returns_sha256": sha256_array(y),
        "available_notional_sha256": sha256_array(available),
        "timestamps_sha256": sha256_array(
            np.asarray([pd.Timestamp(value).value for value in timestamps], dtype=np.int64)
        ),
        "data_formats": dict(data_formats),
        "alignment_diagnostics": alignment_diagnostics,
    }
    steps = TRAIN_STEPS[args.budget][base_timeframe]
    artifact_contract = {
        "schema": "hprl-freqtrade-mtf-artifact-contract-v2",
        "runtime_contract_sha256": runtime_contract_sha,
        "training_data": training_data_contract,
        "budget": args.budget,
        "train_steps": int(steps),
        "parallel_envs": int(args.parallel_envs),
    }
    artifact_contract_sha = canonical_sha256(artifact_contract)

    if not args.force and _artifacts_reusable(
        artifact_dir,
        runtime_contract_sha256=runtime_contract_sha,
        artifact_contract_sha256=artifact_contract_sha,
    ):
        print(json.dumps({
            "status": "SKIP",
            "artifact_dir": str(artifact_dir),
            "runtime_contract_sha256": runtime_contract_sha,
            "artifact_contract_sha256": artifact_contract_sha,
        }))
        return 0

    # artifact_manifest.json is the commit marker.  Removing it first makes interrupted
    # retraining fail closed instead of mixing old single-TF and new MTF members.
    manifest_path.unlink(missing_ok=True)

    api = _native(repo_root)
    import torch

    action_cfg, _cost_cfg, _reward_cfg, env_cfg, train_cfg, mem_cfg = _configs(
        api, spec, base_timeframe, args.device, args.parallel_envs, steps
    )
    probe_dataset = _dataset(api, torch, x, y, available)
    probe = api["VectorizedHedgeEnv"](
        probe_dataset, env_cfg, device=train_cfg.device, memory_config=mem_cfg
    )
    try:
        obs_dim, action_dim = probe.observation_dim, probe.action_dim
    finally:
        probe.close(aggressive=False)
    del probe, probe_dataset

    dataset = _dataset(api, torch, x, y, available)
    agent = api["create_agent"](spec.algorithm, obs_dim, action_dim, train_cfg, device=None)
    api["configure_agent_action_levels"](agent, action_cfg.level_count)

    if spec.algorithm == "rebrac_v2":
        signal_name = f"{base_timeframe}__momentum_12"
        signal_index = feature_names.index(signal_name) if signal_name in feature_names else 0
        rows_target = max(spec.batch_size * 8, min(60_000, steps * args.parallel_envs * 2))
        offline = _offline_behavior(
            api, torch, dataset, env_cfg, train_cfg, mem_cfg,
            rows_target, WINDOW_STEPS[base_timeframe], signal_index,
        )
        trainer = api["OfflineTrainer"](
            offline,
            agent,
            train_cfg,
            device=str(agent.device),
            memory_config=mem_cfg,
            action_config=action_cfg,
        )
        training_summary = asdict(trainer.run(steps))
    else:
        training_summary = _train_online_windowed(
            api, torch, agent, dataset, env_cfg, train_cfg, mem_cfg,
            steps, WINDOW_STEPS[base_timeframe],
        )

    artifact_dir.mkdir(parents=True, exist_ok=True)
    atomic_save_npz(
        scaler_path,
        mean=mean,
        std=std,
        feature_names=np.asarray(feature_names),
        runtime_contract_sha256=np.asarray(runtime_contract_sha),
        artifact_contract_sha256=np.asarray(artifact_contract_sha),
    )
    checkpoint_metadata = {
        "schema": MODEL_METADATA_SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "model": spec.key,
        "algorithm": spec.algorithm,
        "strategy_class": spec.strategy_class,
        "timeframe": base_timeframe,
        "base_timeframe": base_timeframe,
        "input_timeframes": list(input_timeframes),
        "pair": "ETH/USDT:USDT",
        "train_start": args.train_start,
        "train_end": args.train_end,
        "training_rows": len(x),
        "first_training_timestamp": str(timestamps[0]),
        "last_training_timestamp": str(timestamps[-1]),
        "feature_version": FEATURE_VERSION,
        "feature_names": list(feature_names),
        "feature_count": len(feature_names),
        "observation_dim": int(obs_dim),
        "action_dim": int(action_dim),
        "alignment_contract": MTF_ALIGNMENT_CONTRACT,
        "alignment_diagnostics": alignment_diagnostics,
        "action_config": dict(ACTION_KWARGS),
        "cost_config": dict(COST_KWARGS),
        "reward_config": reward_kwargs(spec),
        "model_spec": asdict(spec),
        "runtime_contract": runtime_contract,
        "runtime_contract_sha256": runtime_contract_sha,
        "training_data_contract": training_data_contract,
        "artifact_contract": artifact_contract,
        "artifact_contract_sha256": artifact_contract_sha,
        "budget": args.budget,
        "train_steps": int(steps),
        "parallel_envs": int(args.parallel_envs),
        "device_request": args.device,
        "data_format": data_format,
        "data_formats": dict(data_formats),
        "datadir": str(datadir),
        "training_summary": training_summary,
        "formal_backtest_authority": "freqtrade hedge-backtesting",
        "policy_input_authority": "Freqtrade base dataframe + DataProvider informative OHLCV",
    }
    api["save_checkpoint"](checkpoint, agent, checkpoint_metadata)

    checkpoint_sha = sha256_file(checkpoint)
    checkpoint_sidecar_sha = sha256_file(checkpoint_sidecar)
    scaler_sha = sha256_file(scaler_path)
    metadata = {
        **checkpoint_metadata,
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_sidecar_sha256": checkpoint_sidecar_sha,
        "scaler_sha256": scaler_sha,
    }
    _json(metadata_path, metadata)
    manifest = {
        "schema": ARTIFACT_MANIFEST_SCHEMA,
        "runtime_contract_sha256": runtime_contract_sha,
        "artifact_contract_sha256": artifact_contract_sha,
        "files": {
            "checkpoint.pt": checkpoint_sha,
            "checkpoint.pt.json": checkpoint_sidecar_sha,
            "scaler.npz": scaler_sha,
            "metadata.json": sha256_file(metadata_path),
        },
    }
    _json(manifest_path, manifest)
    print(json.dumps({
        "status": "PASS",
        "model": spec.key,
        "base_timeframe": base_timeframe,
        "input_timeframes": list(input_timeframes),
        "checkpoint": str(checkpoint),
        "training_rows": len(x),
        "feature_count": len(feature_names),
        "updates": training_summary.get("updates"),
        "runtime_contract_sha256": runtime_contract_sha,
        "artifact_contract_sha256": artifact_contract_sha,
    }, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Train one native HPRL model with causal Freqtrade-style "
            "multi-timeframe inputs."
        )
    )
    p.add_argument("--repo-root", required=True)
    p.add_argument("--user-data-dir", required=True)
    p.add_argument("--datadir", required=True)
    p.add_argument("--data-format", default="auto")
    p.add_argument("--model", required=True, choices=tuple(MODELS))
    p.add_argument("--timeframe", required=True, choices=tuple(PERIODS_PER_YEAR))
    p.add_argument("--train-start", default="2023-08-19T00:00:00Z")
    p.add_argument("--train-end", default="2024-08-19T00:00:00Z")
    p.add_argument("--budget", choices=tuple(TRAIN_STEPS), default="balanced")
    p.add_argument("--device", default="cpu")
    p.add_argument("--parallel-envs", type=int, default=16)
    p.add_argument("--force", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return prepare_one(args)
    except Exception as exc:
        traceback.print_exc()
        print(json.dumps({"status": "FAIL", "error": repr(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
