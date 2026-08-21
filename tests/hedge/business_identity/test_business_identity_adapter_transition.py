from decimal import Decimal
from types import SimpleNamespace

import pytest

from freqtrade.hedge.contracts.adapters import adapt_planner_intent
from freqtrade.hedge.contracts.types import IntentAction, OrderType, PositionSide
from freqtrade.hedge.planning.context import OrderSide


def planner_intent():
    return SimpleNamespace(
        intent_id="planner-r2",
        symbol="BTCUSDT",
        position_side=PositionSide.LONG,
        order_side=OrderSide.BUY,
        action=IntentAction.OPEN,
        quantity=Decimal("0.001"),
        price=Decimal(100000),
        reduce_only=False,
        order_type=OrderType.LIMIT,
        bucket="TACTICAL",
        layer=0,
        time_in_force="GTC",
        reason="transition-test",
        business_identity=None,
        order_role=None,
        order_revision=0,
        submission_generation=0,
    )


def test_adapter_keeps_legacy_path_until_binder_is_wired():
    result = adapt_planner_intent(
        planner_intent(),
        account_id="acct",
        exchange="paper",
    )
    assert result.business_trade_id is None
    assert result.business_lot_id is None
    assert result.idempotency_key.startswith("planner:")


def test_adapter_strict_gate_fails_closed_when_explicitly_enabled():
    with pytest.raises(ValueError, match="durable business identity"):
        adapt_planner_intent(
            planner_intent(),
            account_id="acct",
            exchange="paper",
            require_business_identity=True,
        )
