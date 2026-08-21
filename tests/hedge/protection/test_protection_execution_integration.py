from dataclasses import replace
from decimal import Decimal
from uuid import uuid4

from freqtrade.hedge.contracts.business_identity import BusinessIdentity, BusinessOrderRole
from freqtrade.hedge.contracts.types import (
    ExecutionOrderIntent,
    IntentAction,
    OrderType,
    PositionSide,
)


def test_protection_cancel_replace_preserves_business_and_protection_identity() -> None:
    identity = BusinessIdentity(uuid4(), 40, uuid4(), 1, "main", "BTCUSDT", "LONG")
    original = ExecutionOrderIntent(
        account_id="main",
        symbol="BTCUSDT",
        position_side=PositionSide.LONG,
        action=IntentAction.CLOSE,
        quantity=Decimal("0.01"),
        idempotency_key="protection-original",
        order_type=OrderType.MARKET,
        reduce_only=True,
        business_trade_id=identity.business_trade_id,
        business_lot_id=identity.business_lot_id,
        business_trade_seq=identity.business_trade_seq,
        lot_index=identity.lot_index,
        order_role=BusinessOrderRole.STOP_LOSS,
        metadata={
            "protection_group_id": str(uuid4()),
            "protection_id": str(uuid4()),
            "protection_kind": "STOP_LOSS",
            "protection_label": "SL",
        },
    )
    replacement = replace(
        original,
        intent_id=uuid4(),
        idempotency_key="protection-replacement",
        order_revision=original.order_revision + 1,
    )
    assert replacement.business_trade_id == original.business_trade_id
    assert replacement.business_lot_id == original.business_lot_id
    assert replacement.order_role is BusinessOrderRole.STOP_LOSS
    assert replacement.metadata == original.metadata
    assert replacement.order_revision == 1
