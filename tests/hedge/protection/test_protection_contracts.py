from decimal import Decimal
from uuid import uuid4

import pytest

from freqtrade.hedge.contracts.business_identity import BusinessIdentity
from freqtrade.hedge.protection import (
    BusinessLotProtectionSnapshot,
    ProtectionIntegrityError,
    ProtectionKind,
    ProtectionQuantityMode,
    build_protection_group,
    make_protection_leg,
)


def _identity(side: str = "LONG", seq: int = 12) -> BusinessIdentity:
    return BusinessIdentity(uuid4(), seq, uuid4(), 1, "main", "BTCUSDT", side)


def test_group_uses_same_business_trade_and_lot_for_every_protection_leg() -> None:
    identity = _identity()
    lot = BusinessLotProtectionSnapshot(identity, Decimal("0.02"), Decimal(90000))
    group = build_protection_group(
        lot=lot,
        take_profits=(
            ("TP1", Decimal(98000), Decimal("0.005")),
            ("TP2", Decimal(102000), Decimal("0.005")),
        ),
        stop_loss=Decimal(85000),
    )
    assert {leg.business_identity for leg in group.legs} == {identity}
    assert {leg.protection_group_id for leg in group.legs} == {group.protection_group_id}
    assert [leg.kind for leg in group.legs] == [
        ProtectionKind.TAKE_PROFIT,
        ProtectionKind.TAKE_PROFIT,
        ProtectionKind.STOP_LOSS,
    ]
    assert group.legs[-1].quantity_mode is ProtectionQuantityMode.REMAINING


def test_absolute_take_profit_budget_cannot_exceed_target_lot() -> None:
    identity = _identity()
    lot = BusinessLotProtectionSnapshot(identity, Decimal("0.01"), Decimal(90000))
    with pytest.raises(ProtectionIntegrityError):
        build_protection_group(
            lot=lot,
            take_profits=(
                ("TP1", Decimal(98000), Decimal("0.006")),
                ("TP2", Decimal(102000), Decimal("0.006")),
            ),
            stop_loss=Decimal(85000),
        )


def test_trigger_geometry_is_side_specific_and_fail_closed() -> None:
    long_lot = BusinessLotProtectionSnapshot(_identity("LONG"), Decimal("0.01"), Decimal(90000))
    short_lot = BusinessLotProtectionSnapshot(
        _identity("SHORT", 13), Decimal("0.01"), Decimal(90000)
    )
    with pytest.raises(ValueError, match="LONG take-profit"):
        build_protection_group(
            lot=long_lot,
            take_profits=(("TP1", Decimal(89000), Decimal("0.005")),),
            stop_loss=Decimal(85000),
        )
    with pytest.raises(ValueError, match="SHORT stop-loss"):
        build_protection_group(
            lot=short_lot,
            stop_loss=Decimal(88000),
        )


def test_remaining_mode_does_not_persist_a_stale_fixed_quantity() -> None:
    identity = _identity()
    with pytest.raises(ValueError, match="must not store a fixed quantity"):
        make_protection_leg(
            protection_group_id=uuid4(),
            business_identity=identity,
            kind=ProtectionKind.STOP_LOSS,
            label="SL",
            trigger_price=Decimal(85000),
            quantity=Decimal("0.01"),
            quantity_mode=ProtectionQuantityMode.REMAINING,
        )
