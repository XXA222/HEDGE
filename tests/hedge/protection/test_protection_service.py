from decimal import Decimal
from uuid import uuid4

import pytest

from freqtrade.hedge.contracts.business_identity import BusinessIdentity
from freqtrade.hedge.protection import (
    BusinessLotProtectionSnapshot,
    BusinessProtectionService,
    InMemoryProtectionRepository,
    ProtectionGroupStatus,
    ProtectionIntegrityError,
    ProtectionKind,
    ProtectionLegStatus,
    build_protection_group,
)


def _identity(side: str = "LONG", seq: int = 12) -> BusinessIdentity:
    return BusinessIdentity(uuid4(), seq, uuid4(), 1, "main", "BTCUSDT", side)


def test_tp_then_sl_targets_only_the_same_business_lot_and_uses_remaining_quantity() -> None:
    identity = _identity()
    lot = BusinessLotProtectionSnapshot(identity, Decimal("0.02"), Decimal(90000))
    group = build_protection_group(
        lot=lot,
        take_profits=(("TP1", Decimal(98000), Decimal("0.01")),),
        stop_loss=Decimal(85000),
    )
    repository = InMemoryProtectionRepository()
    service = BusinessProtectionService(repository)
    service.arm(group)

    tp = service.evaluate(group.protection_group_id, lot=lot, mark_price=Decimal(99000))
    assert tp.triggered and tp.leg is not None and tp.intent is not None
    assert tp.leg.kind is ProtectionKind.TAKE_PROFIT
    assert tp.intent.quantity == Decimal("0.01")
    assert tp.intent.business_lot_id == identity.business_lot_id
    assert tp.intent.order_role.value == "TAKE_PROFIT"

    # Crash after durable trigger but before submit: intent identity and idempotency are stable.
    replay = service.evaluate(group.protection_group_id, lot=lot, mark_price=Decimal(99000))
    assert replay.intent is not None
    assert replay.intent.intent_id == tp.intent.intent_id
    assert replay.intent.idempotency_key == tp.intent.idempotency_key

    service.mark_submitted(
        group.protection_group_id,
        protection_id=tp.leg.protection_id,
        client_order_id="tp-order",
    )
    after_tp = service.record_fill(
        group.protection_group_id,
        protection_id=tp.leg.protection_id,
        cumulative_filled_quantity=Decimal("0.01"),
        lot_open_quantity=Decimal("0.01"),
        terminal=True,
    )
    assert after_tp.status is ProtectionGroupStatus.ACTIVE

    remaining = BusinessLotProtectionSnapshot(
        identity,
        Decimal("0.01"),
        Decimal(90000),
    )
    stop = service.evaluate(
        group.protection_group_id,
        lot=remaining,
        mark_price=Decimal(84000),
    )
    assert stop.intent is not None and stop.leg is not None
    assert stop.intent.quantity == Decimal("0.01")
    assert stop.intent.business_lot_id == identity.business_lot_id
    assert stop.intent.order_role.value == "STOP_LOSS"

    service.mark_submitted(
        group.protection_group_id,
        protection_id=stop.leg.protection_id,
        client_order_id="sl-order",
    )
    closed = service.record_fill(
        group.protection_group_id,
        protection_id=stop.leg.protection_id,
        cumulative_filled_quantity=Decimal("0.01"),
        lot_open_quantity=Decimal(0),
        terminal=True,
    )
    assert closed.status is ProtectionGroupStatus.CLOSED
    assert all(leg.status.terminal for leg in closed.legs)


def test_targeted_take_profit_never_spills_when_lot_was_reduced_elsewhere() -> None:
    identity = _identity(seq=13)
    original = BusinessLotProtectionSnapshot(identity, Decimal("0.02"), Decimal(90000))
    group = build_protection_group(
        lot=original,
        take_profits=(("TP1", Decimal(98000), Decimal("0.015")),),
        stop_loss=Decimal(85000),
    )
    repository = InMemoryProtectionRepository()
    service = BusinessProtectionService(repository)
    service.arm(group)
    reduced_elsewhere = BusinessLotProtectionSnapshot(
        identity,
        Decimal("0.01"),
        Decimal(90000),
    )
    with pytest.raises(ProtectionIntegrityError, match="exceeds business lot open quantity"):
        service.evaluate(
            group.protection_group_id,
            lot=reduced_elsewhere,
            mark_price=Decimal(99000),
        )


def test_long_and_short_trailing_stop_are_symmetric() -> None:
    long_identity = _identity("LONG", 14)
    long_lot = BusinessLotProtectionSnapshot(long_identity, Decimal("0.01"), Decimal(100))
    long_group = build_protection_group(lot=long_lot, trailing_distance=Decimal(5))
    long_repo = InMemoryProtectionRepository()
    long_service = BusinessProtectionService(long_repo)
    long_service.arm(long_group)
    assert not long_service.evaluate(
        long_group.protection_group_id,
        lot=long_lot,
        mark_price=Decimal(110),
    ).triggered
    long_trigger = long_service.evaluate(
        long_group.protection_group_id,
        lot=long_lot,
        mark_price=Decimal(104),
    )
    assert long_trigger.leg is not None
    assert long_trigger.leg.kind is ProtectionKind.TRAILING_STOP

    short_identity = _identity("SHORT", 15)
    short_lot = BusinessLotProtectionSnapshot(
        short_identity,
        Decimal("0.01"),
        Decimal(100),
    )
    short_group = build_protection_group(lot=short_lot, trailing_distance=Decimal(5))
    short_repo = InMemoryProtectionRepository()
    short_service = BusinessProtectionService(short_repo)
    short_service.arm(short_group)
    assert not short_service.evaluate(
        short_group.protection_group_id,
        lot=short_lot,
        mark_price=Decimal(90),
    ).triggered
    short_trigger = short_service.evaluate(
        short_group.protection_group_id,
        lot=short_lot,
        mark_price=Decimal(96),
    )
    assert short_trigger.intent is not None
    assert short_trigger.intent.position_side.value == "SHORT"


def test_only_one_protection_execution_can_be_in_flight_for_one_lot() -> None:
    identity = _identity(seq=16)
    lot = BusinessLotProtectionSnapshot(identity, Decimal("0.02"), Decimal(90000))
    group = build_protection_group(
        lot=lot,
        take_profits=(("TP1", Decimal(98000), Decimal("0.01")),),
        stop_loss=Decimal(85000),
    )
    repository = InMemoryProtectionRepository()
    service = BusinessProtectionService(repository)
    service.arm(group)
    triggered = service.evaluate(group.protection_group_id, lot=lot, mark_price=Decimal(99000))
    assert triggered.leg is not None
    service.mark_submitted(
        group.protection_group_id,
        protection_id=triggered.leg.protection_id,
        client_order_id="tp-active",
    )
    blocked = service.evaluate(group.protection_group_id, lot=lot, mark_price=Decimal(84000))
    assert not blocked.triggered
    assert blocked.leg is not None
    assert blocked.leg.status is ProtectionLegStatus.SUBMITTED
