from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from freqtrade.freqai.hedge_rl.risk_environment import HedgeRiskLevelEnv
from freqtrade.freqai.hedge_rl.risk_levels import RiskLevelProfile
from freqtrade.freqai.hedge_rl.risk_memory import (
    CompactRiskMarketData,
    HedgeRLMemoryConfig,
    compact_feature_matrix,
    release_rl_phase_memory,
)
from freqtrade.freqai.hedge_rl.risk_observation import (
    HedgeRiskObservationBuilder,
    RiskObservationSchema,
)
from freqtrade.freqai.hedge_rl.risk_portfolio import LegSide, RiskAccountState
from freqtrade.freqai.hedge_rl.risk_reward import (
    HedgeRiskRewardModel,
    PendingOutcome,
    RiskRewardConfig,
)


def _market(rows: int = 256, cols: int = 8):
    features = pd.DataFrame(
        np.linspace(-2.0, 2.0, rows * cols).reshape(rows, cols),
        columns=[f"f{i}" for i in range(cols)],
    )
    base = np.linspace(100.0, 105.0, rows)
    prices = pd.DataFrame(
        {
            "open": base,
            "high": base + 1,
            "low": base - 1,
            "close": base + 0.1,
            "volume": np.ones(rows),
            "funding_rate": np.zeros(rows),
            "uncertainty_score": np.full(rows, 0.5),
            "unused_wide_column": np.arange(rows, dtype=np.float64),
        }
    )
    return features, prices


def _env(rows: int = 256, cols: int = 8, *, breakdown_interval: int = 64):
    features, prices = _market(rows, cols)
    return HedgeRiskLevelEnv(
        df=features,
        prices=prices,
        window_size=16,
        config={
            "freqai": {
                "hedge_rl_config": {
                    "random_start": False,
                    "max_episode_steps": 128,
                    "memory": {"reward_breakdown_interval": breakdown_interval},
                }
            }
        },
    )


def test_memory_policy_defaults_to_float32_features():
    assert HedgeRLMemoryConfig().feature_dtype == "float32"


def test_memory_policy_requires_full_dual_leg_pending_envelope():
    with pytest.raises(ValueError):
        HedgeRLMemoryConfig(max_pending_reward_outcomes=7)
    assert HedgeRLMemoryConfig(max_pending_reward_outcomes=8).max_pending_reward_outcomes == 8


def test_compact_feature_matrix_downcasts_and_is_readonly():
    features, _ = _market(32, 4)
    values = compact_feature_matrix(features)
    assert values.dtype == np.float32
    assert values.flags.c_contiguous
    assert not values.flags.writeable


def test_compact_feature_matrix_rejects_nonfinite_values():
    frame = pd.DataFrame({"x": [1.0, np.nan]})
    with pytest.raises(ValueError):
        compact_feature_matrix(frame)


def test_compact_market_retains_only_four_arrays():
    _, prices = _market(32, 4)
    compact = CompactRiskMarketData.from_prices(prices)
    assert set(compact.__dataclass_fields__) == {
        "open",
        "close",
        "funding_rate",
        "uncertainty_score",
    }


def test_compact_market_uses_float64_prices_and_float32_auxiliary():
    _, prices = _market(32, 4)
    compact = CompactRiskMarketData.from_prices(prices)
    assert compact.open.dtype == np.float64
    assert compact.close.dtype == np.float64
    assert compact.funding_rate.dtype == np.float32
    assert compact.uncertainty_score.dtype == np.float32


def test_environment_does_not_retain_price_dataframe():
    env = _env()
    assert not hasattr(env, "prices")
    assert isinstance(env.market, CompactRiskMarketData)


def test_environment_feature_matrix_is_float32():
    env = _env()
    assert env.features.dtype == np.float32


def test_environment_memory_telemetry_has_no_price_dataframe():
    env = _env()
    telemetry = env.memory_telemetry()
    assert telemetry["retained_price_dataframe"] == 0
    assert telemetry["feature_dtype"] == "float32"


def test_observation_builder_build_into_reuses_supplied_buffer():
    schema = RiskObservationSchema(("a", "b"), 4)
    builder = HedgeRiskObservationBuilder(schema)
    features = np.arange(40, dtype=np.float32).reshape(20, 2)
    out = np.empty(schema.flat_size, dtype=np.float32)
    result = builder.build_into(
        features,
        out,
        tick=3,
        account=RiskAccountState.initial(1000),
        mark=100,
        profile=RiskLevelProfile(),
        uncertainty_score=0.5,
        funding_rate=0,
        max_episode_steps=100,
    )
    assert result is out
    assert np.isfinite(result).all()


def test_observation_builder_clips_directly_to_float32():
    schema = RiskObservationSchema(("a",), 2)
    builder = HedgeRiskObservationBuilder(schema, feature_clip=1.0)
    result = builder.build(
        np.asarray([[-9.0], [9.0]], dtype=np.float32),
        tick=1,
        account=RiskAccountState.initial(1000),
        mark=100,
        profile=RiskLevelProfile(),
        uncertainty_score=0.5,
        funding_rate=0,
        max_episode_steps=100,
    )
    assert result.dtype == np.float32
    assert result[0] == pytest.approx(-1.0)
    assert result[1] == pytest.approx(1.0)


def test_action_mask_is_cached_and_readonly():
    env = _env()
    first = env.action_masks()
    second = env.action_masks()
    assert first is second
    assert not first.flags.writeable


def test_reset_reuses_simulator_and_reward_model_objects():
    env = _env()
    simulator_id = id(env.simulator)
    reward_id = id(env.reward_model)
    env.reset()
    env.step(np.asarray([1, 0], dtype=np.int64))
    env.reset()
    assert id(env.simulator) == simulator_id
    assert id(env.reward_model) == reward_id


def test_reward_reset_clears_pending_in_place():
    model = HedgeRiskRewardModel(RiskRewardConfig())
    pending_id = id(model._pending)
    model._append_pending(PendingOutcome("probe", LegSide.LONG, 1, 2, 1000, 0, 0, 0, 1))
    model.reset()
    assert model.pending_outcome_count == 0
    assert id(model._pending) == pending_id


def test_reward_pending_outcomes_are_hard_bounded():
    model = HedgeRiskRewardModel(RiskRewardConfig(), max_pending_outcomes=4)
    for index in range(4):
        model._append_pending(PendingOutcome("probe", LegSide.LONG, index, 99, 1000, 0, 0, 0, 1))
    with pytest.raises(RuntimeError):
        model._append_pending(PendingOutcome("probe", LegSide.LONG, 5, 99, 1000, 0, 0, 0, 1))


def test_reward_breakdown_is_not_materialized_every_step():
    env = _env(breakdown_interval=64)
    env.reset()
    _, _, _, _, info = env.step(np.asarray([1, 0], dtype=np.int64))
    assert info["reward_components"] == {}


def test_reward_breakdown_is_emitted_at_configured_interval():
    env = _env(rows=128, breakdown_interval=2)
    env.reset()
    env.step(np.asarray([1, 0], dtype=np.int64))
    _, _, _, _, info = env.step(np.asarray([1, 0], dtype=np.int64))
    assert "equity_log_return" in info["reward_components"]


def test_tensorboard_metrics_dict_is_reused():
    env = _env()
    env.reset()
    metrics_id = id(env.tensorboard_metrics)
    nested_id = id(env.tensorboard_metrics["hedge_risk"])
    for _ in range(4):
        env.step(np.asarray([1, 0], dtype=np.int64))
    assert id(env.tensorboard_metrics) == metrics_id
    assert id(env.tensorboard_metrics["hedge_risk"]) == nested_id


def test_repeated_episode_reset_does_not_grow_pending_rewards():
    env = _env(rows=512)
    for _ in range(20):
        env.reset()
        for _ in range(5):
            env.step(np.asarray([1, 0], dtype=np.int64))
        assert env.reward_model.pending_outcome_count <= 1
    assert env.reward_model.pending_outcome_count <= 1


def test_phase_release_helper_is_safe():
    gc.collect()
    assert isinstance(release_rl_phase_memory(trim_allocator=False), bool)


def test_learner_source_removes_unused_train_feature_deepcopy():
    source = (
        Path(__file__).resolve().parents[3]
        / "freqtrade"
        / "freqai"
        / "prediction_models"
        / "HedgeRiskLevelReinforcementLearner.py"
    ).read_text(encoding="utf-8")
    assert "copy.deepcopy" not in source
    assert "self.df_raw = DataFrame()" in source
    assert "_release_training_environments" in source


def test_environment_hot_step_does_not_call_gc_or_allocator_trim():
    source = (
        Path(__file__).resolve().parents[3]
        / "freqtrade"
        / "freqai"
        / "hedge_rl"
        / "risk_environment.py"
    ).read_text(encoding="utf-8")
    step_source = source.split("    def step(self, action: Sequence[int]):", 1)[1].split(
        "    def memory_telemetry", 1
    )[0]
    assert "release_rl_phase_memory" not in step_source
    assert "gc.collect" not in step_source
