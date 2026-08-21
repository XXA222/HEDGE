from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from uuid import uuid4

from freqtrade.hedge.contracts.business_identity import BusinessIdentity, BusinessOrderRole
from freqtrade.hedge.contracts.types import (
    ExecutionOrderIntent,
    IntentAction,
    OrderType,
    PositionSide,
)
from freqtrade.rpc.api_server.hedge_readonly import _execution_order_schema


def test_execution_read_model_projects_protection_identity() -> None:
    identity = BusinessIdentity(uuid4(), 50, uuid4(), 1, "main", "BTCUSDT", "LONG")
    group_id = uuid4()
    protection_id = uuid4()
    intent = ExecutionOrderIntent(
        account_id="main",
        symbol="BTCUSDT",
        position_side=PositionSide.LONG,
        action=IntentAction.REDUCE,
        quantity=Decimal("0.005"),
        idempotency_key="tp-read-model",
        order_type=OrderType.MARKET,
        reduce_only=True,
        business_trade_id=identity.business_trade_id,
        business_lot_id=identity.business_lot_id,
        business_trade_seq=identity.business_trade_seq,
        lot_index=identity.lot_index,
        order_role=BusinessOrderRole.TAKE_PROFIT,
        metadata={
            "protection_group_id": str(group_id),
            "protection_id": str(protection_id),
            "protection_kind": "TAKE_PROFIT",
            "protection_label": "TP1",
            "business_display_id": identity.display_id,
        },
    )
    now = datetime(2026, 8, 21, tzinfo=UTC)
    order = SimpleNamespace(
        client_order_id="hedge-protection-read-model",
        intent=intent,
        approved_quantity=Decimal("0.005"),
        created_at=now,
        lifecycle=SimpleNamespace(
            status=SimpleNamespace(value="ACKNOWLEDGED"),
            filled_quantity=Decimal(0),
            average_price=None,
            exchange_order_id="123",
            reason=None,
            updated_at=now,
        ),
    )
    schema = _execution_order_schema(order)
    assert schema.business_display_id == identity.display_id
    assert schema.protection_group_id == str(group_id)
    assert schema.protection_id == str(protection_id)
    assert schema.protection_kind == "TAKE_PROFIT"
    assert schema.protection_label == "TP1"
    assert schema.order_role == "TAKE_PROFIT"
