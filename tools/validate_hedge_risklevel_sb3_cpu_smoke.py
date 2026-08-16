#!/usr/bin/env python3
"""Real Stable-Baselines3 CPU smoke for the 5x5 Hedge risk-level environment."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO

from freqtrade.freqai.hedge_rl.risk_environment import HedgeRiskLevelEnv
from freqtrade.freqai.prediction_models.HedgeRiskLevelReinforcementLearner import (
    HedgeRiskLevelReinforcementLearner,
)


def main() -> int:
    rows = 256
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
    config = {
        "freqai": {
            "hedge_rl_config": {
                "random_start": False,
                "observation_window": 16,
                "max_episode_steps": 64,
            }
        }
    }
    env = HedgeRiskLevelEnv(df=features, prices=prices, config=config, window_size=16)
    try:
        obs, _ = env.reset(seed=7)
        if tuple(env.action_space.nvec.tolist()) != (5, 5):
            raise AssertionError("risk-level action space must be 5x5")
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=32,
            batch_size=32,
            n_epochs=1,
            device="cpu",
            verbose=0,
        )
        model.learn(total_timesteps=64)
        action, _ = model.predict(obs, deterministic=True)
        if np.asarray(action).shape != (2,):
            raise AssertionError("risk-level action must contain two levels")
        if not str(model.device).startswith("cpu"):
            raise AssertionError("SB3 smoke must use CPU")
        if next(model.policy.parameters()).device.type != "cpu":
            raise AssertionError("SB3 policy must be on CPU")
        if HedgeRiskLevelReinforcementLearner.MyRLEnv is not HedgeRiskLevelEnv:
            raise AssertionError("risk-level learner environment contract changed")
        print("HEDGE RISKLEVEL SB3 CPU SMOKE: PASS")
        print("torch=" + str(torch.__version__))
        print("cuda_available=" + str(torch.cuda.is_available()))
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
