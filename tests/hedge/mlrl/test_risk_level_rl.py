from __future__ import annotations

import math
from datetime import UTC, datetime
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from freqtrade.freqai.hedge_rl.risk_bridge import HedgeRiskLevelPolicyBridge, HedgeRiskPolicyContext
from freqtrade.freqai.hedge_rl.risk_environment import HedgeRiskLevelEnv
from freqtrade.freqai.hedge_rl.risk_levels import (
    HedgeRiskLevelAction,
    PositionRiskLevel,
    RiskLevelMapper,
    RiskLevelProfile,
)
from freqtrade.freqai.hedge_rl.risk_observation import (
    HedgeRiskObservationBuilder,
    RiskObservationSchema,
)
from freqtrade.freqai.hedge_rl.risk_planner_adapter import HedgeRiskLevelPlannerAdapter
from freqtrade.freqai.hedge_rl.risk_portfolio import RiskAccountState, TargetLevelPortfolioSimulator
from freqtrade.freqai.hedge_rl.risk_reward import HedgeRiskRewardModel, RiskRewardConfig
from freqtrade.hedge.planning.context import (
    LegPosition,
    MarketSnapshot,
    PlannerConfig,
    PlanningContext,
    PositionSide,
    WalletSnapshot,
)
from freqtrade.hedge.planning.target import calculate_target


def _sim(*, fee_rate=0, slippage_bps=0):
    return TargetLevelPortfolioSimulator(
        1000,
        profile=RiskLevelProfile(),
        fee_rate=fee_rate,
        slippage_bps=slippage_bps,
    )


def test_default_profile_matches_design():
    profile = RiskLevelProfile()
    assert profile.position_levels == (0.0, 0.05, 0.12, 0.25, 0.40)
    assert profile.fraction(PositionRiskLevel.HEAVY) == pytest.approx(0.40)
    assert profile.minimum_reserve_margin_fraction == pytest.approx(0.20)


def test_all_25_joint_actions_decode():
    for long_level in range(5):
        for short_level in range(5):
            action = HedgeRiskLevelAction.from_value((long_level, short_level))
            assert action.as_tuple() == (long_level, short_level)
            assert action.joint_id == long_level * 5 + short_level


def test_invalid_action_rejected():
    with pytest.raises(ValueError):
        HedgeRiskLevelAction.from_value((5, 0))
    with pytest.raises(ValueError):
        HedgeRiskLevelAction.from_value((1,))


def test_heavy_is_not_all_in_and_both_heavy_reserve_20_percent():
    target = RiskLevelMapper(RiskLevelProfile()).map((4, 4), equity=1000)
    assert target.long_margin_budget == pytest.approx(400)
    assert target.short_margin_budget == pytest.approx(400)
    assert target.reserve_margin_fraction == pytest.approx(0.20)


def test_margin_budget_is_separate_from_leverage():
    profile = RiskLevelProfile(long_leverage=3.0, short_leverage=2.0)
    target = RiskLevelMapper(profile).map((1, 2), equity=1000)
    assert target.long_margin_budget == pytest.approx(50)
    assert target.long_target_notional == pytest.approx(150)
    assert target.short_margin_budget == pytest.approx(120)
    assert target.short_target_notional == pytest.approx(240)


def test_profile_rejects_all_in_heavy():
    with pytest.raises(ValueError):
        RiskLevelProfile(position_levels=(0.0, 0.05, 0.12, 0.25, 0.60))


def test_profile_rejects_combined_reserve_violation():
    with pytest.raises(ValueError):
        RiskLevelProfile(
            position_levels=(0.0, 0.05, 0.12, 0.25, 0.45),
            max_combined_margin_fraction=0.80,
            minimum_reserve_margin_fraction=0.20,
            hard_max_margin_fraction_per_leg=0.50,
        )


def test_simulator_long_profit_and_short_loss_are_account_level():
    sim = TargetLevelPortfolioSimulator(
        1000,
        profile=RiskLevelProfile(),
        fee_rate=0,
        slippage_bps=0,
    )
    transition = sim.apply_target((3, 1), reference_price=100, mark_price=110)
    assert transition.long_unrealized > 0
    assert transition.short_unrealized < 0
    assert transition.equity == pytest.approx(1020.0)


def test_simulator_short_profit_on_falling_market():
    sim = _sim()
    transition = sim.apply_target((0, 3), reference_price=100, mark_price=90)
    assert transition.short_unrealized == pytest.approx(25.0)
    assert transition.equity == pytest.approx(1025.0)


def test_same_target_no_turnover_when_equity_and_price_unchanged():
    sim = _sim()
    sim.apply_target((2, 2), reference_price=100, mark_price=100)
    second = sim.apply_target((2, 2), reference_price=100, mark_price=100)
    assert second.traded_notional == pytest.approx(0.0, abs=1e-12)


def test_reduce_realizes_profit_without_double_counting():
    sim = _sim()
    sim.apply_target((3, 0), reference_price=100, mark_price=110)
    transition = sim.apply_target((1, 0), reference_price=110, mark_price=110)
    assert transition.realized_pnl > 0
    assert sim.state.long.quantity > 0
    assert sim.state.equity == pytest.approx(1025.0)


def test_positive_funding_long_pays_short_receives():
    long_sim = _sim()
    short_sim = _sim()
    long_t = long_sim.apply_target((2, 0), reference_price=100, mark_price=100, funding_rate=0.001)
    short_t = short_sim.apply_target(
        (0, 2),
        reference_price=100,
        mark_price=100,
        funding_rate=0.001,
    )
    assert long_t.funding_cashflow < 0
    assert short_t.funding_cashflow > 0


def test_reward_primary_term_is_equity_log_return_only():
    sim = _sim(fee_rate=0.001)
    transition = sim.apply_target((2, 0), reference_price=100, mark_price=100)
    model = HedgeRiskRewardModel(
        RiskRewardConfig(
            drawdown_weight=0,
            downside_exposure_weight=0,
            downside_ewma_weight=0,
            uncertainty_exposure_weight=0,
            reserve_pressure_weight=0,
            liquidation_buffer_weight=0,
            wrong_level_loss_weight=0,
            position_success_bonus_weight=0,
            adverse_scale_in_weight=0,
            upward_jump_weight=0,
            level_churn_weight=0,
            turnover_shaping_weight=0,
            repeated_probe_weight=0,
            risk_reduction_bonus_weight=0,
            profit_lock_bonus_weight=0,
            hedge_efficiency_weight=0,
            hedge_waste_weight=0,
            delayed_scale_bonus_weight=0,
            delayed_probe_bonus_weight=0,
        )
    )
    breakdown = model.calculate(
        transition=transition,
        account=sim.state,
        mark=100,
        uncertainty_score=0,
        reserve_margin_fraction=sim.state.reserve_margin_fraction(100, sim.profile),
    )
    expected = 100 * math.log(transition.equity / transition.previous_equity)
    assert breakdown.equity_log_return == pytest.approx(expected)
    assert breakdown.unclipped_reward == pytest.approx(expected)


def test_uncertainty_penalizes_large_exposure_more_than_small():
    profile = RiskLevelProfile()
    cfg = RiskRewardConfig(
        drawdown_weight=0,
        downside_exposure_weight=0,
        liquidation_buffer_weight=0,
        adverse_scale_in_weight=0,
        turnover_shaping_weight=0,
        repeated_probe_weight=0,
        risk_reduction_bonus_weight=0,
        profit_lock_bonus_weight=0,
        hedge_efficiency_weight=0,
        hedge_waste_weight=0,
        delayed_scale_bonus_weight=0,
        delayed_probe_bonus_weight=0,
    )
    small_sim = TargetLevelPortfolioSimulator(1000, profile=profile, fee_rate=0, slippage_bps=0)
    heavy_sim = TargetLevelPortfolioSimulator(1000, profile=profile, fee_rate=0, slippage_bps=0)
    small_t = small_sim.apply_target((1, 0), reference_price=100, mark_price=100)
    heavy_t = heavy_sim.apply_target((4, 0), reference_price=100, mark_price=100)
    small_r = HedgeRiskRewardModel(cfg).calculate(
        transition=small_t,
        account=small_sim.state,
        mark=100,
        uncertainty_score=1,
        reserve_margin_fraction=small_sim.state.reserve_margin_fraction(100, profile),
    )
    heavy_r = HedgeRiskRewardModel(cfg).calculate(
        transition=heavy_t,
        account=heavy_sim.state,
        mark=100,
        uncertainty_score=1,
        reserve_margin_fraction=heavy_sim.state.reserve_margin_fraction(100, profile),
    )
    assert heavy_r.uncertainty_exposure_penalty > small_r.uncertainty_exposure_penalty


def test_heavy_wrong_direction_has_extra_risk_penalty():
    profile = RiskLevelProfile()
    small_sim = TargetLevelPortfolioSimulator(1000, profile=profile, fee_rate=0, slippage_bps=0)
    heavy_sim = TargetLevelPortfolioSimulator(1000, profile=profile, fee_rate=0, slippage_bps=0)
    small_t = small_sim.apply_target((1, 0), reference_price=100, mark_price=95)
    heavy_t = heavy_sim.apply_target((4, 0), reference_price=100, mark_price=95)
    cfg = RiskRewardConfig(uncertainty_exposure_weight=0)
    small_r = HedgeRiskRewardModel(cfg).calculate(
        transition=small_t,
        account=small_sim.state,
        mark=95,
        uncertainty_score=0,
        reserve_margin_fraction=small_sim.state.reserve_margin_fraction(95, profile),
    )
    heavy_r = HedgeRiskRewardModel(cfg).calculate(
        transition=heavy_t,
        account=heavy_sim.state,
        mark=95,
        uncertainty_score=0,
        reserve_margin_fraction=heavy_sim.state.reserve_margin_fraction(95, profile),
    )
    assert heavy_r.downside_exposure_penalty > small_r.downside_exposure_penalty
    assert heavy_r.reward < small_r.reward


def test_repeated_failed_probe_penalty_escalates_only_after_confirmed_probe_failures():
    profile = RiskLevelProfile()
    sim = TargetLevelPortfolioSimulator(1000, profile=profile, fee_rate=0, slippage_bps=0)
    model = HedgeRiskRewardModel(
        RiskRewardConfig(
            uncertainty_exposure_weight=0,
            probe_confirmation_steps=1,
        )
    )
    penalties = []
    price = 100.0
    for _ in range(4):
        opened = sim.apply_target((1, 0), reference_price=price, mark_price=price)
        first = model.calculate(
            transition=opened,
            account=sim.state,
            mark=price,
            uncertainty_score=0,
            reserve_margin_fraction=sim.state.reserve_margin_fraction(price, profile),
        )
        assert first.repeated_probe_penalty == 0.0
        losing_price = price * 0.99
        closed = sim.apply_target((0, 0), reference_price=losing_price, mark_price=losing_price)
        resolved = model.calculate(
            transition=closed,
            account=sim.state,
            mark=losing_price,
            uncertainty_score=0,
            reserve_margin_fraction=sim.state.reserve_margin_fraction(losing_price, profile),
        )
        penalties.append(resolved.repeated_probe_penalty)
        price = losing_price
    assert penalties == pytest.approx([0.02, 0.024, 0.03, 0.04])
    assert model.consecutive_failed_probes == 4


def test_probe_holding_loss_does_not_increment_failure_every_candle():
    profile = RiskLevelProfile()
    sim = TargetLevelPortfolioSimulator(1000, profile=profile, fee_rate=0, slippage_bps=0)
    model = HedgeRiskRewardModel(
        RiskRewardConfig(uncertainty_exposure_weight=0, probe_confirmation_steps=3)
    )
    first = sim.apply_target((1, 0), reference_price=100, mark_price=99)
    r1 = model.calculate(
        transition=first,
        account=sim.state,
        mark=99,
        uncertainty_score=0,
        reserve_margin_fraction=sim.state.reserve_margin_fraction(99, profile),
    )
    assert r1.consecutive_failed_probes == 0
    second = sim.apply_target((1, 0), reference_price=99, mark_price=98)
    r2 = model.calculate(
        transition=second,
        account=sim.state,
        mark=98,
        uncertainty_score=0,
        reserve_margin_fraction=sim.state.reserve_margin_fraction(98, profile),
    )
    assert r2.consecutive_failed_probes == 0


def test_observation_schema_includes_account_state():
    schema = RiskObservationSchema(("a", "b"), 3)
    builder = HedgeRiskObservationBuilder(schema)
    features = np.arange(20, dtype=float).reshape(10, 2)
    account = RiskAccountState.initial(1000)
    vector = builder.build(
        features,
        tick=2,
        account=account,
        mark=100,
        profile=RiskLevelProfile(),
        uncertainty_score=0.5,
        funding_rate=0,
        max_episode_steps=100,
    )
    assert vector.shape == (schema.flat_size,)
    assert schema.flat_size == 3 * 2 + 20


def test_bridge_fails_closed_on_stale_projection():
    class ExplodingModel:
        def predict(self, *args, **kwargs):
            raise AssertionError("model must not be called for stale projection")

    bridge = HedgeRiskLevelPolicyBridge(
        feature_names=("x",),
        window_size=2,
        profile=RiskLevelProfile(),
    )
    context = HedgeRiskPolicyContext(
        account=RiskAccountState.initial(1000),
        mark=100,
        projection_fresh=False,
    )
    observation = bridge.observation(np.asarray([[0.0], [0.0]]), tick=1, context=context)
    action = bridge.predict_action(ExplodingModel(), observation, context=context)
    assert action.as_tuple() == (0, 0)


def test_bridge_accepts_two_level_model_output():
    class Model:
        def predict(self, observation, deterministic=True):
            assert deterministic
            return np.asarray([3, 1]), None

    bridge = HedgeRiskLevelPolicyBridge(
        feature_names=("x",),
        window_size=2,
        profile=RiskLevelProfile(),
    )
    context = HedgeRiskPolicyContext.flat(1000, mark=100)
    observation = bridge.observation(np.asarray([[0.0], [0.0]]), tick=1, context=context)
    assert bridge.predict_action(Model(), observation, context=context).as_tuple() == (3, 1)


def test_planner_adapter_emits_target_exposure_not_orders():
    adapter = HedgeRiskLevelPlannerAdapter(RiskLevelProfile(long_leverage=3, short_leverage=3))
    signal = adapter.from_action(HedgeRiskLevelAction.from_value((3, 1)), equity=1000)
    assert signal.long_margin_fraction == pytest.approx(0.25)
    assert signal.short_margin_fraction == pytest.approx(0.05)
    assert signal.long_target_notional == pytest.approx(750)
    assert signal.short_target_notional == pytest.approx(150)
    assert "order" not in " ".join(signal.strategy_columns().keys()).lower()


def test_risk_level_snapshot_preserves_equity_unit_and_exact_dual_leg_targets():
    adapter = HedgeRiskLevelPlannerAdapter(RiskLevelProfile(long_leverage=3, short_leverage=3))
    signal = adapter.from_action(HedgeRiskLevelAction.from_value((3, 1)), equity=1000)
    now = datetime(2026, 8, 17, tzinfo=UTC)
    snapshot = adapter.to_signal_snapshot(
        signal,
        pair="BTC/USDT:USDT",
        timeframe="1m",
        candle_close_time=now,
        feature_timestamp=now,
        model_version="test",
    )
    assert snapshot.target_net_ratio == Decimal("0.6")
    assert snapshot.target_long_notional == Decimal("750.0")
    assert snapshot.target_short_notional == Decimal("150.0")

    market = MarketSnapshot(
        symbol="BTC/USDT:USDT",
        timestamp=now,
        bid=Decimal("99"),
        ask=Decimal("101"),
        mark=Decimal("100"),
        qty_step=Decimal("0.01"),
    )
    wallet = WalletSnapshot(
        balance=Decimal("1000"),
        equity=Decimal("1000"),
        available_balance=Decimal("1000"),
        long=LegPosition(PositionSide.LONG),
        short=LegPosition(PositionSide.SHORT),
    )
    config = PlannerConfig(
        max_wallet_exposure_long=Decimal("0.8"),
        max_wallet_exposure_short=Decimal("0.8"),
        max_gross_wallet_exposure=Decimal("1"),
    )
    context = PlanningContext(
        market=market,
        wallet=wallet,
        config=config,
        target_long_quantity=snapshot.target_long_notional / market.mark,
        target_short_quantity=snapshot.target_short_notional / market.mark,
    )
    assert calculate_target(context, PositionSide.LONG).total_quantity == Decimal("7.50")
    assert calculate_target(context, PositionSide.SHORT).total_quantity == Decimal("1.50")


def test_reward_nonfinite_values_fail_closed():
    model = HedgeRiskRewardModel(RiskRewardConfig(reward_clip=5))
    assert model._safe_log_return(float("nan"), 1000) == -1.0
    assert model._safe_log_return(1000, float("inf")) == -1.0
    assert model._transform_reward(float("nan")) == -5.0
    assert model._transform_reward(float("inf")) == -5.0


def _market(rows=40):
    features = pd.DataFrame({"feature": np.linspace(-1, 1, rows)})
    prices = pd.DataFrame(
        {
            "open": np.full(rows, 100.0),
            "high": np.full(rows, 101.0),
            "low": np.full(rows, 99.0),
            "close": np.full(rows, 100.0),
            "volume": np.ones(rows),
            "uncertainty_score": np.linspace(0, 1, rows),
        }
    )
    return features, prices


def test_environment_uses_multidiscrete_5x5():
    features, prices = _market()
    env = HedgeRiskLevelEnv(
        df=features,
        prices=prices,
        window_size=4,
        config={"freqai": {"hedge_rl_config": {"random_start": False}}},
    )
    assert env.action_space.nvec.tolist() == [5, 5]
    observation, _ = env.reset()
    assert observation.shape == env.observation_space.shape
    _, reward, terminated, truncated, info = env.step(np.asarray([3, 1], dtype=np.int64))
    assert math.isfinite(float(reward))
    assert not terminated
    assert not truncated
    assert info["executed_long_level"] == 3
    assert info["executed_short_level"] == 1


def test_environment_rejects_float_or_out_of_range_action():
    features, prices = _market()
    env = HedgeRiskLevelEnv(
        df=features,
        prices=prices,
        window_size=4,
        config={"freqai": {"hedge_rl_config": {"random_start": False}}},
    )
    env.reset()
    with pytest.raises(ValueError):
        env.step(np.asarray([1.5, 0.0]))
    with pytest.raises(ValueError):
        env.step(np.asarray([5, 0], dtype=np.int64))


def test_environment_executes_on_next_bar_open():
    features, prices = _market()
    prices.loc[4, "open"] = 200.0
    prices.loc[4, "high"] = 200.0
    prices.loc[4, "low"] = 200.0
    prices.loc[4, "close"] = 200.0
    env = HedgeRiskLevelEnv(
        df=features,
        prices=prices,
        window_size=4,
        config={
            "freqai": {"hedge_rl_config": {"random_start": False, "fee_rate": 0, "slippage_bps": 0}}
        },
    )
    env.reset()
    env.step(np.asarray([1, 0], dtype=np.int64))
    expected_qty = 10000 * 0.05 / 200
    assert env.simulator.state.long.quantity == pytest.approx(expected_qty)
