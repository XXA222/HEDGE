from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from uuid import uuid4

import pytest

from freqtrade.hedge.contracts.business_identity import BusinessIdentity
from freqtrade.hedge.integration.paper_projection import _BucketState
from freqtrade.hedge.planning.context import PositionBucket, PositionSide


def identity(seq: int) -> BusinessIdentity:
    return BusinessIdentity(
        business_trade_id=uuid4(),
        business_trade_seq=seq,
        business_lot_id=uuid4(),
        lot_index=1,
        account_id="acct",
        symbol="BTCUSDT",
        position_side=PositionSide.LONG,
    )


def test_paper_projection_preserves_two_independent_lots() -> None:
    state = _BucketState()
    first = identity(11)
    second = identity(12)
    at = datetime.now(UTC)
    state.increase(
        PositionBucket.TACTICAL,
        Decimal("0.01"),
        Decimal(100),
        at,
        business_identity=first,
    )
    state.increase(
        PositionBucket.TACTICAL,
        Decimal("0.02"),
        Decimal(110),
        at,
        business_identity=second,
    )
    lots = state.position_lots()
    assert [lot.business_identity.business_trade_seq for lot in lots] == [11, 12]
    assert state.tactical_quantity == Decimal("0.03")


def test_targeted_paper_reduce_never_spills_into_another_lot() -> None:
    state = _BucketState()
    first = identity(11)
    second = identity(12)
    at = datetime.now(UTC)
    for item in (first, second):
        state.increase(
            PositionBucket.TACTICAL,
            Decimal("0.01"),
            Decimal(100),
            at,
            business_identity=item,
        )
    with pytest.raises(ValueError, match="exceeds open quantity"):
        state.reduce(
            PositionBucket.TACTICAL,
            Decimal("0.02"),
            business_identity=first,
        )
    assert state.tactical_quantity == Decimal("0.02")
