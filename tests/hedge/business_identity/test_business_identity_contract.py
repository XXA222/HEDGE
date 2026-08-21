from dataclasses import FrozenInstanceError
from uuid import uuid4

import pytest

from freqtrade.hedge.contracts.business_identity import (
    BusinessIdentity,
    BusinessOrderRole,
    business_display_id,
    validate_order_role,
)
from freqtrade.hedge.contracts.types import IntentAction, PositionSide


def make_identity(**changes):
    values = {
        "business_trade_id": uuid4(),
        "business_trade_seq": 12,
        "business_lot_id": uuid4(),
        "lot_index": 1,
        "account_id": "acct",
        "symbol": "BTC/USDT:USDT",
        "position_side": PositionSide.LONG,
    }
    values.update(changes)
    return BusinessIdentity(**values)


def test_business_identity_requires_positive_sequence():
    with pytest.raises(ValueError):
        make_identity(business_trade_seq=0)


def test_business_identity_is_immutable():
    identity = make_identity()
    with pytest.raises(FrozenInstanceError):
        identity.business_trade_seq = 13


def test_business_identity_display_id():
    assert business_display_id(make_identity()) == "BTCUSDT-L-000012"


def test_business_identity_rejects_side_mismatch():
    identity = make_identity()
    with pytest.raises(ValueError, match="side mismatch"):
        identity.assert_matches(
            account_id="acct",
            symbol="BTCUSDT",
            position_side=PositionSide.SHORT,
        )


def test_reduce_role_requires_reduce_only():
    with pytest.raises(ValueError):
        validate_order_role(
            BusinessOrderRole.TAKE_PROFIT,
            action=IntentAction.REDUCE,
            reduce_only=False,
        )


def test_entry_role_rejects_reduce_action():
    with pytest.raises(ValueError):
        validate_order_role(
            BusinessOrderRole.ENTRY,
            action=IntentAction.REDUCE,
            reduce_only=True,
        )


def test_identity_accepts_planner_side_enum_without_domain_import_cycle():
    identity = make_identity(position_side=PositionSide.LONG)
    assert identity.position_side == "LONG"
    assert identity.display_id == "BTCUSDT-L-000012"
