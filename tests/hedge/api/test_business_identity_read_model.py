from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from uuid import uuid4

from freqtrade.hedge.contracts.business_identity import BusinessIdentity, BusinessOrderRole
from freqtrade.hedge.execution.service import (
    ExecutionOrder,
    IntentAction,
    OrderIntent,
    OrderType,
    PositionSide,
)
from freqtrade.hedge.execution.state_machine import OrderLifecycle
from freqtrade.rpc.api_server.hedge_readonly import _execution_order_schema


def test_execution_read_model_projects_business_display_identity() -> None:
    identity = BusinessIdentity(
        business_trade_id=uuid4(),
        business_trade_seq=12,
        business_lot_id=uuid4(),
        lot_index=1,
        account_id="main",
        symbol="BTC/USDT:USDT",
        position_side="LONG",
    )
    intent = OrderIntent(
        account_id="main",
        symbol="BTC/USDT:USDT",
        position_side=PositionSide.LONG,
        action=IntentAction.OPEN,
        quantity=Decimal("0.01"),
        idempotency_key="phase5-read-model",
        order_type=OrderType.LIMIT,
        limit_price=Decimal(100000),
        business_trade_id=identity.business_trade_id,
        business_lot_id=identity.business_lot_id,
        business_trade_seq=identity.business_trade_seq,
        lot_index=identity.lot_index,
        order_role=BusinessOrderRole.ENTRY,
        order_revision=2,
        metadata={"exchange": "paper"},
    )
    now = datetime(2026, 8, 21, tzinfo=UTC)
    order = ExecutionOrder(
        intent=intent,
        client_order_id="phase5-client-order",
        approved_quantity=Decimal("0.01"),
        lifecycle=OrderLifecycle(updated_at=now),
        created_at=now,
    )
    schema = _execution_order_schema(order)
    assert schema.business_trade_id == str(identity.business_trade_id)
    assert schema.business_lot_id == str(identity.business_lot_id)
    assert schema.business_trade_seq == 12
    assert schema.lot_index == 1
    assert schema.order_role == "ENTRY"
    assert schema.business_display_id == "BTCUSDT-L-000012"
    assert schema.order_revision == 2
