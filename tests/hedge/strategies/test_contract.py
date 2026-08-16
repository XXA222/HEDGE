from decimal import Decimal

import pytest

from freqtrade.hedge.planning.context import PlannerConfig
from freqtrade.hedge.strategies.contract import (
    StrategyDirective,
    directive_from_values,
    planner_config_for_directive,
    target_net_quantity_for_directive,
)


def test_directive_prefers_bounded_ratio_over_exact_quantity():
    d = directive_from_values(
        {"hedge_target_net": "99", "hedge_target_net_ratio": "0.2", "hedge_confidence": "0.5"}
    )
    assert d.target_net_quantity is None and d.target_net_ratio == Decimal("0.2")


def test_dynamic_values_only_reduce_static_risk():
    base = PlannerConfig(
        max_wallet_exposure_long=Decimal("0.4"),
        max_wallet_exposure_short=Decimal("0.4"),
        max_gross_wallet_exposure=Decimal("0.65"),
        max_single_order_notional=Decimal(100),
    )
    d = StrategyDirective(
        confidence=Decimal("0.5"),
        risk_scale=Decimal("0.8"),
        long_exposure_scale=Decimal("0.5"),
        short_exposure_scale=Decimal(1),
    )
    out = planner_config_for_directive(base, d)
    assert out.max_wallet_exposure_long == Decimal("0.1")
    assert out.max_wallet_exposure_short == Decimal("0.2")
    assert out.max_gross_wallet_exposure == Decimal("0.325")
    assert out.max_single_order_notional == Decimal(50)


def test_target_ratio_uses_equity_mark_and_directional_scale():
    base = PlannerConfig(
        max_wallet_exposure_long=Decimal("0.4"),
        max_wallet_exposure_short=Decimal("0.4"),
        max_gross_wallet_exposure=Decimal("0.65"),
    )
    d = StrategyDirective(
        target_net_ratio=Decimal("0.5"),
        confidence=Decimal("0.5"),
        risk_scale=Decimal(1),
        long_exposure_scale=Decimal("0.5"),
    )
    # raw target 5, safe cap = 1000*.4*.5*.5/100 = 1
    assert target_net_quantity_for_directive(
        directive=d, base=base, equity=Decimal(1000), mark_price=Decimal(100)
    ) == Decimal("1.000")


def test_metadata_and_bounds_fail_closed():
    with pytest.raises(ValueError):
        StrategyDirective(confidence=Decimal("1.1"))
    with pytest.raises(ValueError):
        StrategyDirective(target_net_quantity=Decimal(1), target_net_ratio=Decimal("0.1"))


@pytest.mark.parametrize("value", (None, 0, 1, "yes", "garbage", [], {}))
def test_present_invalid_allow_new_risk_values_fail_closed(value: object):
    assert not directive_from_values({"hedge_allow_new_risk": value}).allow_new_risk


def test_missing_allow_new_risk_remains_compatible_and_invalid_scales_close_risk():
    assert directive_from_values({}).allow_new_risk
    assert directive_from_values({"hedge_allow_new_risk": "TRUE"}).allow_new_risk
    directive = directive_from_values(
        {
            "hedge_confidence": "nan",
            "hedge_risk_scale": "inf",
            "hedge_long_exposure_scale": "bad",
            "hedge_short_exposure_scale": None,
        }
    )
    assert directive.confidence == directive.risk_scale == Decimal(0)
    assert directive.long_exposure_scale == directive.short_exposure_scale == Decimal(0)
