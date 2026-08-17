from __future__ import annotations

from freqtrade.freqai.hedge_rl.risk_action_mask import RiskLevelActionMasker, RiskLevelMaskContext
from freqtrade.freqai.hedge_rl.risk_levels import HedgeRiskLevelAction, RiskLevelProfile
from freqtrade.freqai.hedge_rl.risk_planner_adapter import HedgeRiskLevelPlannerAdapter
from freqtrade.freqai.hedge_rl.risk_portfolio import LegSide, RiskAccountState, RiskLegState


def _account() -> RiskAccountState:
    return RiskAccountState(
        cash_balance=1000.0, equity=1000.0, peak_equity=1000.0,
        long=RiskLegState(LegSide.LONG, quantity=0.5, average_price=100.0),
        short=RiskLegState(LegSide.SHORT, quantity=0.5, average_price=100.0),
        long_level=1, short_level=1,
    )


def test_joint_mask_is_exactly_25_and_stale_is_flat_only() -> None:
    mask = RiskLevelActionMasker(RiskLevelProfile()).build(RiskLevelMaskContext(
        current_action=HedgeRiskLevelAction.from_value((1, 1)), projection_fresh=False
    ))
    assert len(mask.allowed) == 25
    assert mask.allowed_joint_ids == (0,)


def test_unknown_order_mask_blocks_each_leg_from_increasing() -> None:
    mask = RiskLevelActionMasker(RiskLevelProfile()).build(RiskLevelMaskContext(
        current_action=HedgeRiskLevelAction.from_value((1, 1)), unresolved_unknown=True
    ))
    assert not mask.permits(HedgeRiskLevelAction.from_value((2, 1)))
    assert not mask.permits(HedgeRiskLevelAction.from_value((1, 2)))
    assert mask.permits(HedgeRiskLevelAction.from_value((1, 1)))
    assert mask.permits(HedgeRiskLevelAction.from_value((0, 1)))


def test_joint_mask_rejects_upward_jump_above_configured_limit() -> None:
    mask = RiskLevelActionMasker(RiskLevelProfile()).build(RiskLevelMaskContext(
        current_action=HedgeRiskLevelAction.from_value((0, 0)), max_upward_levels=1
    ))
    assert mask.permits(HedgeRiskLevelAction.from_value((1, 0)))
    assert not mask.permits(HedgeRiskLevelAction.from_value((2, 0)))


def test_planner_replaces_masked_increase_with_current_exposure() -> None:
    signal = HedgeRiskLevelPlannerAdapter(RiskLevelProfile()).from_account_action(
        HedgeRiskLevelAction.from_value((2, 1)), account=_account(), mark=100.0,
        unresolved_unknown=True,
    )
    assert (signal.long_level, signal.short_level) == (1, 1)
    assert not signal.allow_new_risk
    assert "MASKED:UNKNOWN_ORDER_NO_INCREASE" in signal.reason
