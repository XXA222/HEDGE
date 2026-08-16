from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from freqtrade.freqai.hedge_rl.risk_bridge import HedgeRiskPolicyContext
from freqtrade.freqai.hedge_rl.risk_levels import (
    HedgeRiskLevelAction,
    RiskActionTopology,
    RiskLevelMapper,
    RiskLevelProfile,
)
from freqtrade.freqai.hedge_rl.risk_observation import (
    HedgeRiskObservationBuilder,
    RiskObservationSchema,
)
from freqtrade.freqai.hedge_rl.risk_portfolio import TargetLevelPortfolioSimulator
from freqtrade.freqai.hedge_rl.risk_reward import HedgeRiskRewardModel, RiskRewardConfig


def _sim(*, fee_rate=0, slippage_bps=0):
    return TargetLevelPortfolioSimulator(
        1000,
        profile=RiskLevelProfile(),
        fee_rate=fee_rate,
        slippage_bps=slippage_bps,
    )


def _reward(model, sim, transition, *, mark, uncertainty=0.0, reserve=None):
    if reserve is None:
        reserve = sim.state.reserve_margin_fraction(mark, sim.profile)
    return model.calculate(
        transition=transition,
        account=sim.state,
        mark=mark,
        uncertainty_score=uncertainty,
        reserve_margin_fraction=reserve,
    )


def _base_only_config(**overrides):
    values = {
        "drawdown_weight": 0,
        "downside_exposure_weight": 0,
        "downside_ewma_weight": 0,
        "uncertainty_exposure_weight": 0,
        "reserve_pressure_weight": 0,
        "liquidation_buffer_weight": 0,
        "wrong_level_loss_weight": 0,
        "position_success_bonus_weight": 0,
        "adverse_scale_in_weight": 0,
        "upward_jump_weight": 0,
        "level_churn_weight": 0,
        "turnover_shaping_weight": 0,
        "repeated_probe_weight": 0,
        "risk_reduction_bonus_weight": 0,
        "profit_lock_bonus_weight": 0,
        "hedge_efficiency_weight": 0,
        "hedge_waste_weight": 0,
        "delayed_scale_bonus_weight": 0,
        "delayed_probe_bonus_weight": 0,
    }
    values.update(overrides)
    return RiskRewardConfig(**values)


def test_action_parser_rejects_fractional_levels():
    with pytest.raises(ValueError):
        HedgeRiskLevelAction.from_value((1.9, 0))


def test_action_parser_rejects_bool_levels():
    with pytest.raises(TypeError):
        HedgeRiskLevelAction.from_value((True, 0))


def test_joint_id_roundtrip_all_25_states():
    for joint_id in range(25):
        action = HedgeRiskLevelAction.from_joint_id(joint_id)
        assert action.joint_id == joint_id


def test_joint_id_rejects_out_of_range():
    with pytest.raises(ValueError):
        HedgeRiskLevelAction.from_joint_id(25)


def test_joint_id_rejects_bool_input_as_type_error():
    with pytest.raises(TypeError):
        HedgeRiskLevelAction.from_joint_id(True)


@pytest.mark.parametrize("value", [1.5, float("nan"), "1.5"])
def test_joint_id_rejects_non_exact_integer_inputs(value):
    with pytest.raises(ValueError):
        HedgeRiskLevelAction.from_joint_id(value)


def test_risk_level_simulator_rejects_slippage_that_can_make_sell_fill_nonpositive():
    with pytest.raises(ValueError):
        TargetLevelPortfolioSimulator(
            1000,
            profile=RiskLevelProfile(),
            fee_rate=0,
            slippage_bps=10_000,
        )


def test_profile_signature_changes_when_risk_ladder_changes():
    a = RiskLevelProfile()
    b = RiskLevelProfile(position_levels=(0.0, 0.04, 0.11, 0.24, 0.40))
    assert a.signature != b.signature


def test_profile_signature_changes_when_leverage_changes():
    assert RiskLevelProfile().signature != RiskLevelProfile(long_leverage=2.0).signature


def test_topology_preserves_full_jump_without_hard_block():
    topology = RiskActionTopology(RiskLevelProfile())
    transition = topology.transition((0, 0), (4, 1))
    assert transition.manhattan_distance == 5
    assert transition.upward_jump_excess == 3
    assert transition.increases_risk


def test_topology_detects_risk_reduction():
    topology = RiskActionTopology(RiskLevelProfile())
    transition = topology.transition((4, 2), (2, 1))
    assert transition.reduces_risk
    assert transition.gross_margin_delta < 0


def test_margin_matrix_is_5_by_5_and_heavy_heavy_is_80_percent():
    matrix = RiskActionTopology(RiskLevelProfile()).margin_matrix()
    assert len(matrix) == 5
    assert all(len(row) == 5 for row in matrix)
    assert matrix[4][4] == pytest.approx(0.80)


def test_mapper_uses_current_equity_for_margin_budget():
    target = RiskLevelMapper(RiskLevelProfile()).map((3, 1), equity=800)
    assert target.long_margin_budget == pytest.approx(200)
    assert target.short_margin_budget == pytest.approx(40)


def test_next_open_gap_reprices_sizing_equity_before_same_level_target():
    profile = RiskLevelProfile(rebalance_deadband_fraction=0.0)
    sim = TargetLevelPortfolioSimulator(1000, profile=profile, fee_rate=0, slippage_bps=0)
    sim.apply_target((4, 0), reference_price=100, mark_price=100)
    t = sim.apply_target((4, 0), reference_price=80, mark_price=80)
    assert t.sizing_equity == pytest.approx(920.0)
    assert t.target.long_target_notional == pytest.approx(368.0)


def test_same_level_deadband_suppresses_tiny_equity_rebalance():
    profile = RiskLevelProfile(rebalance_deadband_fraction=0.0025)
    sim = TargetLevelPortfolioSimulator(1000, profile=profile, fee_rate=0, slippage_bps=0)
    sim.apply_target((2, 0), reference_price=100, mark_price=100)
    t = sim.apply_target((2, 0), reference_price=100.01, mark_price=100.01)
    assert t.traded_notional == pytest.approx(0.0)


def test_level_change_bypasses_deadband():
    profile = RiskLevelProfile(rebalance_deadband_fraction=0.05)
    sim = TargetLevelPortfolioSimulator(1000, profile=profile, fee_rate=0, slippage_bps=0)
    sim.apply_target((1, 0), reference_price=100, mark_price=100)
    t = sim.apply_target((2, 0), reference_price=100, mark_price=100)
    assert t.traded_notional > 0


def test_side_step_pnl_reconciles_account_equity_delta():
    profile = RiskLevelProfile(rebalance_deadband_fraction=0.0)
    sim = TargetLevelPortfolioSimulator(1000, profile=profile, fee_rate=0.001, slippage_bps=10)
    t = sim.apply_target((3, 1), reference_price=100, mark_price=102, funding_rate=0.0001)
    assert t.long_step_net_pnl + t.short_step_net_pnl == pytest.approx(t.equity - t.previous_equity)


def test_realizing_profit_at_same_mark_does_not_create_fake_step_profit():
    sim = _sim()
    sim.apply_target((3, 0), reference_price=100, mark_price=110)
    t = sim.apply_target((0, 0), reference_price=110, mark_price=110)
    assert t.long_realized_pnl > 0
    assert t.long_step_net_pnl == pytest.approx(0.0)


def test_base_only_reward_equals_equity_log_return_even_with_costs():
    sim = _sim(fee_rate=0.001, slippage_bps=5)
    t = sim.apply_target((3, 0), reference_price=100, mark_price=100)
    r = _reward(HedgeRiskRewardModel(_base_only_config()), sim, t, mark=100)
    assert r.unclipped_reward == pytest.approx(100 * math.log(t.equity / t.previous_equity))
    assert r.accounting_cost_ratio > 0


def test_wrong_level_penalty_is_monotonic_for_identical_adverse_move():
    penalties = []
    for level in (1, 2, 3, 4):
        sim = _sim()
        t = sim.apply_target((level, 0), reference_price=100, mark_price=95)
        r = _reward(
            HedgeRiskRewardModel(RiskRewardConfig(uncertainty_exposure_weight=0)),
            sim,
            t,
            mark=95,
        )
        penalties.append(r.wrong_level_loss_penalty)
    assert penalties == sorted(penalties)
    assert penalties[-1] > penalties[1] > penalties[0]


def test_heavy_win_extra_bonus_is_small_relative_to_base_profit():
    sim = _sim()
    t = sim.apply_target((4, 0), reference_price=100, mark_price=105)
    r = _reward(
        HedgeRiskRewardModel(RiskRewardConfig(uncertainty_exposure_weight=0)),
        sim,
        t,
        mark=105,
    )
    assert r.position_success_bonus > 0
    assert r.position_success_bonus < abs(r.equity_log_return) * 0.1


def test_upward_jump_penalty_only_prices_excess_jump_and_risk_context():
    sim = _sim()
    t = sim.apply_target((4, 0), reference_price=100, mark_price=100)
    low = _reward(
        HedgeRiskRewardModel(_base_only_config(upward_jump_weight=0.1)),
        sim,
        t,
        mark=100,
        uncertainty=0,
    )
    sim2 = _sim()
    t2 = sim2.apply_target((4, 0), reference_price=100, mark_price=100)
    high = _reward(
        HedgeRiskRewardModel(_base_only_config(upward_jump_weight=0.1)),
        sim2,
        t2,
        mark=100,
        uncertainty=1,
    )
    assert low.upward_jump_penalty == pytest.approx(0.0)
    assert high.upward_jump_penalty > 0


def test_level_churn_penalty_requires_actual_turnover():
    cfg = _base_only_config(level_churn_weight=0.1)
    sim = _sim()
    t1 = sim.apply_target((1, 0), reference_price=100, mark_price=100)
    r1 = _reward(HedgeRiskRewardModel(cfg), sim, t1, mark=100)
    assert r1.level_churn_penalty > 0


def test_convex_drawdown_penalty_grows_with_severity():
    cfg = _base_only_config(drawdown_weight=2.0)
    penalties = []
    for mark in (99.0, 90.0):
        sim = _sim()
        t = sim.apply_target((4, 0), reference_price=100, mark_price=mark)
        penalties.append(_reward(HedgeRiskRewardModel(cfg), sim, t, mark=mark).drawdown_penalty)
    assert penalties[1] > penalties[0] * 10


def test_downside_ewma_persists_then_decays_without_new_loss():
    cfg = _base_only_config(downside_ewma_weight=1.0, downside_ewma_alpha=0.5)
    sim = _sim()
    model = HedgeRiskRewardModel(cfg)
    t1 = sim.apply_target((4, 0), reference_price=100, mark_price=90)
    r1 = _reward(model, sim, t1, mark=90)
    t2 = sim.apply_target((4, 0), reference_price=90, mark_price=90)
    r2 = _reward(model, sim, t2, mark=90)
    assert r1.downside_semideviation > 0
    assert 0 < r2.downside_semideviation < r1.downside_semideviation


def test_reward_reset_clears_downside_memory():
    model = HedgeRiskRewardModel(_base_only_config(downside_ewma_weight=1.0))
    model.downside_semivariance_ewma = 9.0
    model.reset()
    assert model.downside_semideviation == 0.0


def test_reserve_pressure_is_soft_and_hard_buffer_is_separate():
    sim = _sim()
    t = sim.apply_target((4, 4), reference_price=100, mark_price=100)
    cfg = _base_only_config(
        preferred_reserve_margin_fraction=0.30,
        reserve_pressure_weight=1.0,
        minimum_liquidation_buffer_fraction=0.10,
        liquidation_buffer_weight=4.0,
    )
    r = _reward(HedgeRiskRewardModel(cfg), sim, t, mark=100, reserve=0.20)
    assert r.reserve_pressure_penalty > 0
    assert r.liquidation_buffer_penalty == 0


def test_hard_reserve_shortfall_penalty_is_convex():
    sim = _sim()
    t = sim.apply_target((1, 0), reference_price=100, mark_price=100)
    cfg = _base_only_config(liquidation_buffer_weight=4.0)
    model = HedgeRiskRewardModel(cfg)
    r1 = _reward(model, sim, t, mark=100, reserve=0.09)
    model.reset()
    r2 = _reward(model, sim, t, mark=100, reserve=0.05)
    assert r2.liquidation_buffer_penalty > r1.liquidation_buffer_penalty


def test_long_probe_failure_not_hidden_by_larger_profitable_short():
    cfg = _base_only_config(repeated_probe_weight=0.02, probe_confirmation_steps=1)
    sim = _sim()
    model = HedgeRiskRewardModel(cfg)
    t1 = sim.apply_target((1, 3), reference_price=100, mark_price=100)
    _reward(model, sim, t1, mark=100)
    t2 = sim.apply_target((1, 3), reference_price=100, mark_price=99)
    r2 = _reward(model, sim, t2, mark=99)
    assert t2.equity > t2.previous_equity
    assert r2.consecutive_failed_probes_long == 1
    assert r2.repeated_probe_penalty == pytest.approx(0.02)


def test_short_probe_failure_not_hidden_by_larger_profitable_long():
    cfg = _base_only_config(repeated_probe_weight=0.02, probe_confirmation_steps=1)
    sim = _sim()
    model = HedgeRiskRewardModel(cfg)
    t1 = sim.apply_target((3, 1), reference_price=100, mark_price=100)
    _reward(model, sim, t1, mark=100)
    t2 = sim.apply_target((3, 1), reference_price=100, mark_price=101)
    r2 = _reward(model, sim, t2, mark=101)
    assert t2.equity > t2.previous_equity
    assert r2.consecutive_failed_probes_short == 1


def test_probe_opening_cost_is_included_in_delayed_outcome():
    cfg = _base_only_config(repeated_probe_weight=0.02, probe_confirmation_steps=1)
    sim = _sim(fee_rate=0.01)
    model = HedgeRiskRewardModel(cfg)
    t1 = sim.apply_target((1, 0), reference_price=100, mark_price=100)
    _reward(model, sim, t1, mark=100)
    t2 = sim.apply_target((1, 0), reference_price=100, mark_price=100)
    r2 = _reward(model, sim, t2, mark=100)
    assert r2.consecutive_failed_probes_long == 1


def test_probe_close_resolves_before_horizon():
    cfg = _base_only_config(repeated_probe_weight=0.02, probe_confirmation_steps=10)
    sim = _sim()
    model = HedgeRiskRewardModel(cfg)
    t1 = sim.apply_target((1, 0), reference_price=100, mark_price=100)
    _reward(model, sim, t1, mark=100)
    assert model.pending_outcome_count == 1
    t2 = sim.apply_target((0, 0), reference_price=99, mark_price=99)
    r2 = _reward(model, sim, t2, mark=99)
    assert model.pending_outcome_count == 0
    assert r2.consecutive_failed_probes_long == 1


def test_profitable_short_does_not_schedule_scale_credit_for_losing_long_add():
    cfg = _base_only_config(scale_confirmation_steps=2, delayed_scale_bonus_weight=0.1)
    sim = _sim()
    model = HedgeRiskRewardModel(cfg)
    t1 = sim.apply_target((1, 3), reference_price=100, mark_price=95)
    _reward(model, sim, t1, mark=95)
    pending_before = model.pending_outcome_count
    t2 = sim.apply_target((3, 3), reference_price=95, mark_price=95)
    _reward(model, sim, t2, mark=95)
    assert model.pending_outcome_count == pending_before


def test_risk_reduction_bonus_is_side_specific():
    cfg = _base_only_config(risk_reduction_bonus_weight=1.0)
    sim = _sim()
    model = HedgeRiskRewardModel(cfg)
    t1 = sim.apply_target((3, 1), reference_price=100, mark_price=95)
    _reward(model, sim, t1, mark=95)
    # LONG is losing and SHORT is winning. Reducing only SHORT must not earn
    # loss-risk reduction credit.
    t2 = sim.apply_target((3, 0), reference_price=95, mark_price=95)
    r2 = _reward(model, sim, t2, mark=95)
    assert r2.risk_reduction_bonus == pytest.approx(0.0)
    t3 = sim.apply_target((1, 0), reference_price=95, mark_price=95)
    r3 = _reward(model, sim, t3, mark=95)
    assert r3.risk_reduction_bonus > 0


def test_profit_lock_bonus_requires_profit_on_reduced_side():
    cfg = _base_only_config(profit_lock_bonus_weight=1.0)
    sim = _sim()
    model = HedgeRiskRewardModel(cfg)
    t1 = sim.apply_target((3, 1), reference_price=100, mark_price=105)
    _reward(model, sim, t1, mark=105)
    # SHORT is losing. Reducing SHORT cannot borrow LONG's profit-lock credit.
    t2 = sim.apply_target((3, 0), reference_price=105, mark_price=105)
    r2 = _reward(model, sim, t2, mark=105)
    assert r2.profit_lock_bonus == pytest.approx(0.0)


def test_hedge_efficiency_rewards_actual_step_offset():
    cfg = _base_only_config(hedge_efficiency_weight=1.0)
    sim = _sim()
    model = HedgeRiskRewardModel(cfg)
    t1 = sim.apply_target((3, 1), reference_price=100, mark_price=100)
    _reward(model, sim, t1, mark=100)
    t2 = sim.apply_target((3, 1), reference_price=100, mark_price=95)
    r2 = _reward(model, sim, t2, mark=95)
    assert r2.hedge_efficiency_bonus > 0


def test_equal_long_short_does_not_receive_arbitrary_hedge_bonus():
    cfg = _base_only_config(hedge_efficiency_weight=1.0)
    sim = _sim()
    model = HedgeRiskRewardModel(cfg)
    t1 = sim.apply_target((2, 2), reference_price=100, mark_price=100)
    _reward(model, sim, t1, mark=100)
    t2 = sim.apply_target((2, 2), reference_price=100, mark_price=95)
    r2 = _reward(model, sim, t2, mark=95)
    assert r2.hedge_efficiency_bonus == 0


def test_hedge_waste_penalizes_drag_on_winning_dominant_leg():
    cfg = _base_only_config(hedge_waste_weight=1.0)
    sim = _sim()
    model = HedgeRiskRewardModel(cfg)
    t1 = sim.apply_target((3, 1), reference_price=100, mark_price=100)
    _reward(model, sim, t1, mark=100)
    t2 = sim.apply_target((3, 1), reference_price=100, mark_price=105)
    r2 = _reward(model, sim, t2, mark=105)
    assert r2.hedge_waste_penalty > 0


def test_positive_shaping_is_capped():
    cfg = _base_only_config(position_success_bonus_weight=100.0, max_positive_shaping=0.05)
    sim = _sim()
    t = sim.apply_target((4, 0), reference_price=100, mark_price=150)
    r = _reward(HedgeRiskRewardModel(cfg), sim, t, mark=150)
    assert r.unclipped_reward <= r.equity_log_return + 0.05 + 1e-12


def test_soft_clip_is_bounded_and_smooth():
    cfg = _base_only_config(reward_clip=10.0, soft_clip=True)
    sim = _sim()
    t = sim.apply_target((4, 0), reference_price=100, mark_price=105)
    r = _reward(HedgeRiskRewardModel(cfg), sim, t, mark=105)
    assert 0 < r.reward < 10.0
    assert r.reward < r.unclipped_reward


def test_observation_exposes_reward_memory_state():
    schema = RiskObservationSchema(("x",), 2)
    builder = HedgeRiskObservationBuilder(schema)
    out = builder.build(
        np.asarray([[0.0], [0.0]], dtype=np.float32),
        tick=1,
        account=TargetLevelPortfolioSimulator(1000, profile=RiskLevelProfile()).state,
        mark=100,
        profile=RiskLevelProfile(),
        uncertainty_score=0.5,
        funding_rate=0,
        max_episode_steps=100,
        failed_probe_long=4,
        failed_probe_short=2,
        downside_semideviation=2.5,
        pending_reward_fraction=0.25,
    )
    tail = out[-4:]
    assert tail.tolist() == pytest.approx([1.0, 0.5, 0.5, 0.25])


def test_policy_context_validates_reward_memory_state():
    with pytest.raises(ValueError):
        HedgeRiskPolicyContext.flat(1000, mark=100).__class__(
            account=TargetLevelPortfolioSimulator(1000, profile=RiskLevelProfile()).state,
            mark=100,
            failed_probe_long=-1,
        )


def test_reward_remains_finite_under_extreme_but_positive_equity_move():
    sim = _sim()
    t = sim.apply_target((4, 0), reference_price=100, mark_price=1)
    r = _reward(HedgeRiskRewardModel(RiskRewardConfig()), sim, t, mark=1, uncertainty=1)
    assert math.isfinite(r.reward)
    assert -10 <= r.reward <= 10


def test_leverage_risk_penalty_distinguishes_same_margin_budget():
    cfg = _base_only_config(leverage_exposure_weight=1.0)
    low_profile = RiskLevelProfile(long_leverage=2.0)
    high_profile = RiskLevelProfile(long_leverage=10.0)
    low = TargetLevelPortfolioSimulator(1000, profile=low_profile, fee_rate=0, slippage_bps=0)
    high = TargetLevelPortfolioSimulator(1000, profile=high_profile, fee_rate=0, slippage_bps=0)
    low_t = low.apply_target((4, 0), reference_price=100, mark_price=100)
    high_t = high.apply_target((4, 0), reference_price=100, mark_price=100)
    low_r = _reward(HedgeRiskRewardModel(cfg), low, low_t, mark=100, uncertainty=1)
    high_r = _reward(HedgeRiskRewardModel(cfg), high, high_t, mark=100, uncertainty=1)
    assert high_r.leverage_exposure_penalty > low_r.leverage_exposure_penalty > 0


def test_automatic_same_level_rebalance_does_not_earn_risk_reduction_bonus():
    cfg = _base_only_config(risk_reduction_bonus_weight=1.0)
    profile = RiskLevelProfile(rebalance_deadband_fraction=0.0)
    sim = TargetLevelPortfolioSimulator(1000, profile=profile, fee_rate=0, slippage_bps=0)
    model = HedgeRiskRewardModel(cfg)
    t1 = sim.apply_target((4, 0), reference_price=100, mark_price=90)
    _reward(model, sim, t1, mark=90)
    # Same level at a lower sizing equity automatically reduces notional, but the policy did not
    # request a lower risk level, so no behavioral risk-reduction bonus is allowed.
    t2 = sim.apply_target((4, 0), reference_price=90, mark_price=90)
    r2 = _reward(model, sim, t2, mark=90)
    assert t2.long_quantity_delta == pytest.approx(0.0)
    assert r2.risk_reduction_bonus == pytest.approx(0.0)


def test_positive_shaping_breakdown_reports_cap_application():
    cfg = _base_only_config(position_success_bonus_weight=100.0, max_positive_shaping=0.01)
    sim = _sim()
    t = sim.apply_target((4, 0), reference_price=100, mark_price=150)
    r = _reward(HedgeRiskRewardModel(cfg), sim, t, mark=150)
    assert r.positive_shaping_raw > r.positive_shaping_applied
    assert r.positive_shaping_applied == pytest.approx(0.01)


def test_reward_signature_changes_when_reward_contract_changes():
    base = RiskRewardConfig()
    changed = replace(base, wrong_level_loss_weight=base.wrong_level_loss_weight + 0.01)
    assert base.signature != changed.signature
    assert len(base.signature) == 16


def test_state_aware_planner_same_level_never_increases_after_loss():
    from freqtrade.freqai.hedge_rl.risk_planner_adapter import HedgeRiskLevelPlannerAdapter

    profile = RiskLevelProfile(long_leverage=3.0, short_leverage=3.0)
    sim = TargetLevelPortfolioSimulator(1000, profile=profile, fee_rate=0, slippage_bps=0)
    sim.apply_target((4, 0), reference_price=100, mark_price=90)
    current_notional = sim.state.long.notional(90)
    signal = HedgeRiskLevelPlannerAdapter(profile).from_account_action(
        HedgeRiskLevelAction.from_value((4, 0)), account=sim.state, mark=90
    )
    assert signal.long_target_notional <= current_notional + 1e-12
    assert not signal.long_increase_allowed
    assert signal.target_semantics == "RISK_CAP_NO_SAME_LEVEL_SCALE_IN"


def test_state_aware_planner_requires_level_raise_for_new_risk():
    from freqtrade.freqai.hedge_rl.risk_planner_adapter import HedgeRiskLevelPlannerAdapter

    profile = RiskLevelProfile(long_leverage=3.0, short_leverage=3.0)
    sim = TargetLevelPortfolioSimulator(1000, profile=profile, fee_rate=0, slippage_bps=0)
    adapter = HedgeRiskLevelPlannerAdapter(profile)
    signal = adapter.from_account_action(
        HedgeRiskLevelAction.from_value((1, 0)), account=sim.state, mark=100
    )
    assert signal.long_increase_allowed
    assert not signal.short_increase_allowed
    assert signal.allow_new_risk


def test_freqtrade_schema_exposes_hedge_action_and_reward_contracts():
    from freqtrade.config_schema.config_schema import CONF_SCHEMA

    rl = CONF_SCHEMA["definitions"]["freqai"]["properties"]["rl_config"]["properties"]
    assert rl["hedge_action_space"]["properties"]["position_levels"]["minItems"] == 5
    assert rl["hedge_action_space"]["properties"]["position_levels"]["maxItems"] == 5
    assert rl["hedge_reward"]["properties"]["loss_level_multipliers"]["minItems"] == 5
    assert rl["hedge_reward"]["additionalProperties"] is False
