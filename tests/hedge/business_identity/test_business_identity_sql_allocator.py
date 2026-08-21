from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from freqtrade.hedge.planning.context import PositionBucket, PositionSide
from freqtrade.persistence.hedge_business_identity import SqlBusinessIdentityAllocator
from freqtrade.persistence.hedge_models import (
    BusinessSequenceRow,
    BusinessTradeRow,
    HedgeModelBase,
    PositionLotRow,
)


def factory():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    HedgeModelBase.metadata.create_all(
        engine,
        tables=[
            BusinessSequenceRow.__table__,
            BusinessTradeRow.__table__,
            PositionLotRow.__table__,
        ],
    )
    return sessionmaker(bind=engine, expire_on_commit=False)


def test_allocator_retry_replays_same_durable_identity_without_burning_sequence() -> None:
    allocator = SqlBusinessIdentityAllocator(factory())
    first = allocator.allocate_entry(
        account_id="acct",
        exchange="paper",
        symbol="BTCUSDT",
        position_side=PositionSide.LONG,
        strategy_entry_key="cycle-1-entry-0",
        bucket=PositionBucket.TACTICAL,
    )
    replay = allocator.allocate_entry(
        account_id="acct",
        exchange="paper",
        symbol="BTCUSDT",
        position_side=PositionSide.LONG,
        strategy_entry_key="cycle-1-entry-0",
        bucket=PositionBucket.TACTICAL,
    )
    second = allocator.allocate_entry(
        account_id="acct",
        exchange="paper",
        symbol="BTCUSDT",
        position_side=PositionSide.LONG,
        strategy_entry_key="cycle-2-entry-0",
        bucket=PositionBucket.TACTICAL,
    )
    assert replay == first
    assert first.business_trade_seq == 1
    assert second.business_trade_seq == 2
    assert second.business_trade_id != first.business_trade_id
    assert second.business_lot_id != first.business_lot_id
