#!/usr/bin/env python3
"""Run a configurable, synthetic CPU training smoke for the Hedge risk-level model.

This command is deliberately offline: it creates deterministic synthetic market data,
uses the real Hedge environment and Stable-Baselines3 policy, and never creates an
exchange client.  Increase ``--timesteps`` locally for a longer training soak.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO

from freqtrade.freqai.hedge_rl.risk_environment import HedgeRiskLevelEnv
from freqtrade.freqai.prediction_models.HedgeRiskLevelReinforcementLearner import (
    HedgeRiskLevelReinforcementLearner,
)


def _market(rows: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if rows < 64:
        raise ValueError("--rows must be at least 64")
    x = np.linspace(0.0, 1.0, rows, dtype=np.float32)
    features = pd.DataFrame(
        {"f1": x, "f2": x * x, "f3": np.sin(x), "f4": np.cos(x)},
        dtype=np.float32,
    )
    base = 100.0 + np.linspace(0.0, 2.0, rows)
    prices = pd.DataFrame(
        {
            "open": base,
            "high": base + 1.0,
            "low": base - 1.0,
            "close": base + 0.1,
            "volume": np.ones(rows),
        }
    )
    return features, prices


def run(*, timesteps: int, rows: int, seed: int, output: Path | None) -> dict[str, object]:
    if timesteps < 1:
        raise ValueError("--timesteps must be positive")
    features, prices = _market(rows)
    config = {
        "freqai": {
            "hedge_rl_config": {
                "random_start": False,
                "observation_window": 16,
                "max_episode_steps": 64,
                "seed": seed,
            }
        }
    }
    env = HedgeRiskLevelEnv(df=features, prices=prices, config=config, window_size=16)
    try:
        observation, _ = env.reset(seed=seed)
        if tuple(env.action_space.nvec.tolist()) != (5, 5):
            raise AssertionError("risk-level action space must be 5x5")
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=32,
            batch_size=32,
            n_epochs=1,
            seed=seed,
            device="cpu",
            verbose=0,
        )
        model.learn(total_timesteps=timesteps)
        action, _ = model.predict(observation, deterministic=True)
        action_array = np.asarray(action)
        if action_array.shape != (2,):
            raise AssertionError("risk-level action must contain two levels")
        if not str(model.device).startswith("cpu"):
            raise AssertionError("training smoke must use CPU")
        if next(model.policy.parameters()).device.type != "cpu":
            raise AssertionError("policy must be on CPU")
        if HedgeRiskLevelReinforcementLearner.MyRLEnv is not HedgeRiskLevelEnv:
            raise AssertionError("risk-level learner environment contract changed")
        report: dict[str, object] = {
            "schema": "freqtrade-hedge-risklevel-training-smoke-v1",
            "generated_at": datetime.now(UTC).isoformat(),
            "status": "PASS",
            "offline": True,
            "device": str(model.device),
            "rows": rows,
            "timesteps": timesteps,
            "seed": seed,
            "action_space": tuple(int(item) for item in env.action_space.nvec),
            "predicted_action": tuple(int(item) for item in action_array),
            "torch_version": torch.__version__,
        }
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            model.save(output)
            report["model_path"] = str(output.with_suffix(".zip").resolve())
        return report
    finally:
        env.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=64)
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = run(
        timesteps=args.timesteps,
        rows=args.rows,
        seed=args.seed,
        output=args.output,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
