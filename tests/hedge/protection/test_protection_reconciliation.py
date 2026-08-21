from decimal import Decimal
from uuid import uuid4

from freqtrade.hedge.contracts.business_identity import BusinessIdentity
from freqtrade.hedge.protection import (
    BusinessLotProtectionSnapshot,
    build_protection_group,
    reconcile_protection_state,
)


def _lot(seq: int) -> BusinessLotProtectionSnapshot:
    identity = BusinessIdentity(uuid4(), seq, uuid4(), 1, "main", "BTCUSDT", "LONG")
    return BusinessLotProtectionSnapshot(identity, Decimal("0.01"), Decimal(90000))


def test_open_lot_protection_and_stop_coverage_are_explicit() -> None:
    first = _lot(20)
    second = _lot(21)
    group = build_protection_group(lot=first, stop_loss=Decimal(85000))
    result = reconcile_protection_state(open_lots=(first, second), groups=(group,))
    assert not result.consistent
    assert result.open_lot_count == 2
    assert result.protected_lot_count == 1
    assert result.stop_covered_lot_count == 1
    assert result.protection_coverage == Decimal("0.5")
    assert "BUSINESS_LOT_PROTECTION_MISSING" in result.issue_codes


def test_closed_or_unknown_lot_cannot_keep_an_active_protection_group() -> None:
    lot = _lot(22)
    group = build_protection_group(lot=lot, stop_loss=Decimal(85000))
    result = reconcile_protection_state(open_lots=(), groups=(group,))
    assert not result.consistent
    assert "PROTECTION_TARGET_LOT_NOT_OPEN" in result.issue_codes
