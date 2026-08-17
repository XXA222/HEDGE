from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from freqtrade.hedge.production.reservations import ExposureReservationBook, ReservationState


NOW = datetime(2026, 8, 17, tzinfo=UTC)


def test_unknown_reservation_retains_capacity_until_reconciliation() -> None:
    book = ExposureReservationBook(ttl=timedelta(seconds=1))
    reserved = book.reserve(
        client_order_id="client-1", notional=Decimal("100"), now=NOW,
        max_total_notional=Decimal("100"), max_orders=1,
    )
    unknown = book.mark_unknown(reserved.reservation_id, now=NOW)
    assert unknown.state is ReservationState.UNKNOWN
    snapshot = book.snapshot(now=NOW + timedelta(minutes=1))
    assert snapshot.held_notional == Decimal("100")
    with pytest.raises(PermissionError, match="ORDER_LIMIT"):
        book.reserve(
            client_order_id="client-2", notional=Decimal("1"), now=NOW + timedelta(minutes=1),
            max_total_notional=Decimal("100"), max_orders=1,
        )
    resolved = book.resolve_unknown(unknown.reservation_id, accepted=False, now=NOW + timedelta(minutes=1))
    assert resolved.state is ReservationState.RELEASED
