from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace
from uuid import uuid4

from freqtrade.hedge.business_reconciliation import (
    BusinessLotBalance,
    business_reconciliation_log_payload,
    reconcile_business_state,
)
from freqtrade.hedge.contracts.business_identity import BusinessIdentity


def identity(*, seq: int, side: str, lot_id=None) -> BusinessIdentity:
    return BusinessIdentity(
        business_trade_id=uuid4(),
        business_trade_seq=seq,
        business_lot_id=lot_id or uuid4(),
        lot_index=1,
        account_id="paper-main",
        symbol="BTC/USDT:USDT",
        position_side=side,
    )


def managed_order(business_identity: BusinessIdentity | None, *, reduce: bool = False):
    intent = SimpleNamespace(
        business_identity=business_identity,
        account_id="paper-main",
        symbol="BTC/USDT:USDT",
        position_side="LONG" if business_identity is None else business_identity.position_side,
        reduces_risk=reduce,
        reduce_only=reduce,
        action=SimpleNamespace(value="REDUCE" if reduce else "OPEN"),
    )
    return SimpleNamespace(intent=intent, client_order_id=f"order-{uuid4().hex[:8]}")


def test_exact_lot_sum_and_managed_identity_coverage_are_consistent() -> None:
    long_a = identity(seq=11, side="LONG")
    long_b = identity(seq=12, side="LONG")
    short_a = identity(seq=13, side="SHORT")
    result = reconcile_business_state(
        open_lots=(
            BusinessLotBalance(long_a, Decimal("0.40")),
            BusinessLotBalance(long_b, Decimal("0.60")),
            BusinessLotBalance(short_a, Decimal("0.20")),
        ),
        managed_orders=(managed_order(long_a), managed_order(short_a, reduce=True)),
        remote_long_quantity=Decimal("1.00"),
        remote_short_quantity=Decimal("0.20"),
        account_id="paper-main",
        symbol="BTCUSDT",
    )
    assert result.consistent
    assert result.lot_sum_consistent
    assert result.managed_order_identity_consistent
    assert result.managed_order_identity_coverage == Decimal(1)
    assert result.display_ids == (
        "BTCUSDT-L-000011",
        "BTCUSDT-L-000012",
        "BTCUSDT-S-000013",
    )
    payload = business_reconciliation_log_payload(result)
    assert payload["business_reconciliation_consistent"] is True
    assert payload["managed_identity_coverage"] == "1"


def test_lot_sum_drift_is_explicit_and_never_attributed() -> None:
    long_a = identity(seq=21, side="LONG")
    result = reconcile_business_state(
        open_lots=(BusinessLotBalance(long_a, Decimal("0.4")),),
        managed_orders=(),
        remote_long_quantity=Decimal("0.5"),
        remote_short_quantity=0,
        account_id="paper-main",
        symbol="BTCUSDT",
    )
    assert not result.consistent
    assert not result.lot_sum_consistent
    assert "LOT_SUM_LONG_MISMATCH" in result.issue_codes
    assert result.long_lot_quantity == Decimal("0.4")
    assert result.remote_long_quantity == Decimal("0.5")


def test_missing_managed_order_identity_fails_closed() -> None:
    long_a = identity(seq=31, side="LONG")
    result = reconcile_business_state(
        open_lots=(BusinessLotBalance(long_a, Decimal(1)),),
        managed_orders=(managed_order(long_a), managed_order(None)),
        remote_long_quantity=1,
        remote_short_quantity=0,
        account_id="paper-main",
        symbol="BTCUSDT",
    )
    assert not result.consistent
    assert not result.managed_order_identity_consistent
    assert result.managed_order_identity_coverage == Decimal("0.5")
    assert "MANAGED_ORDER_IDENTITY_MISSING" in result.issue_codes


def test_reduce_order_targeting_non_open_lot_fails_closed() -> None:
    open_identity = identity(seq=41, side="LONG")
    missing_identity = identity(seq=42, side="LONG")
    result = reconcile_business_state(
        open_lots=(BusinessLotBalance(open_identity, Decimal(1)),),
        managed_orders=(managed_order(missing_identity, reduce=True),),
        remote_long_quantity=1,
        remote_short_quantity=0,
        account_id="paper-main",
        symbol="BTCUSDT",
    )
    assert not result.consistent
    assert "TARGET_BUSINESS_LOT_NOT_OPEN" in result.issue_codes
