from __future__ import annotations

import numpy as np
import pandas as pd

from freqtrade.freqai.hedge_rl.risk_environment import HedgeRiskLevelEnv


def _env() -> HedgeRiskLevelEnv:
    rows = 12
    features = pd.DataFrame(
        {
            "f1": np.linspace(-1.0, 1.0, rows, dtype=np.float32),
            "f2": np.linspace(1.0, -1.0, rows, dtype=np.float32),
        }
    )
    close = np.linspace(100.0, 102.0, rows)
    prices = pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close + 0.25,
            "volume": np.full(rows, 1000.0),
        }
    )
    config = {
        "freqai": {
            "hedge_rl_config": {
                "observation_window": 2,
                "max_episode_steps": 8,
                "random_start": False,
            }
        }
    }
    return HedgeRiskLevelEnv(df=features, prices=prices, config=config, window_size=2)


def test_reset_options_select_exact_audit_segment_without_changing_default_training_contract() -> (
    None
):
    env = _env()
    try:
        _, info = env.reset(
            seed=1,
            options={"start_tick": 3, "end_tick": 6, "max_episode_steps": 3},
        )
        assert info["tick"] == 3
        assert info["episode_start_tick"] == 3
        assert info["episode_end_tick"] == 6
        truncated = False
        for expected_tick in (4, 5, 6):
            _, _, terminated, truncated, info = env.step(np.asarray([0, 0], dtype=np.int64))
            assert not terminated
            assert info["tick"] == expected_tick
        assert truncated

        _, default_info = env.reset(seed=1)
        assert default_info["tick"] == env._start_tick
        assert default_info["episode_end_tick"] == env._end_tick
    finally:
        env.close()
