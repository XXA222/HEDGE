from datetime import UTC, datetime
from decimal import Decimal

import pytest

from freqtrade.hedge.contracts.events import BarEvent, SignalEvent
from freqtrade.hedge.strategies.hprl_eth_dual_leg import HprlEthDualLegStrategy


def test_hprl_action_is_bounded_and_maps_to_dual_leg_directive() -> None:
    directive = HprlEthDualLegStrategy().directive_from_policy_action((0.99, 0.01))
    assert directive.regime == "HPRL"
    assert directive.long_score == Decimal("0.99")
    assert directive.short_score == Decimal("0.01")
    assert directive.target_net_ratio is not None and directive.target_net_ratio > 0
    assert directive.allow_new_risk is True


def test_hprl_strategy_requires_exactly_two_finite_actions() -> None:
    strategy = HprlEthDualLegStrategy()
    with pytest.raises(ValueError, match="exactly LONG and SHORT"):
        strategy.directive_from_policy_action((0.1,))
    with pytest.raises(ValueError, match="finite"):
        strategy.directive_from_policy_action(("NaN", 0.1))


def test_hprl_events_emit_signal_before_matching_bar() -> None:
    bar = BarEvent(
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        symbol="ETH/USDT:USDT",
        open=Decimal("2000"), high=Decimal("2010"), low=Decimal("1990"),
        close=Decimal("2005"), volume=Decimal("100"),
    )
    events = list(HprlEthDualLegStrategy().events((bar,), ((0.5, 0.25),)))
    assert isinstance(events[0], SignalEvent)
    assert events[0].timestamp == bar.timestamp
    assert events[1] is bar
