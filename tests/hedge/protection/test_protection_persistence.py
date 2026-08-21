from datetime import UTC, datetime
from decimal import Decimal
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from freqtrade.hedge.contracts.business_identity import BusinessIdentity
from freqtrade.hedge.protection import BusinessLotProtectionSnapshot, build_protection_group
from freqtrade.persistence.hedge_protection import (
    PROTECTION_GROUPS,
    PROTECTION_METADATA,
    SqlProtectionRepository,
)
from freqtrade.persistence.hedge_protection_migrations import (
    step_protection_constraints,
    step_protection_schema,
    step_verify_protection,
)


def _group(seq: int = 30):
    identity = BusinessIdentity(uuid4(), seq, uuid4(), 1, "main", "BTCUSDT", "LONG")
    lot = BusinessLotProtectionSnapshot(identity, Decimal("0.01"), Decimal(90000))
    return build_protection_group(lot=lot, stop_loss=Decimal(85000))


def test_sql_protection_repository_roundtrips_restart_state() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    PROTECTION_METADATA.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    repository = SqlProtectionRepository(factory)
    group = _group()
    repository.save_group(group)
    assert repository.load_group(group.protection_group_id) == group
    assert repository.groups_for_lot(group.business_lot_id) == (group,)


def test_migration_enforces_one_active_group_per_business_lot() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    step_protection_schema(engine)
    step_protection_constraints(engine)
    group = _group(31)
    identity = group.business_identity
    now = datetime.now(UTC).replace(tzinfo=None)
    values = {
        "protection_group_id": str(group.protection_group_id),
        "business_trade_id": str(identity.business_trade_id),
        "business_lot_id": str(identity.business_lot_id),
        "business_trade_seq": identity.business_trade_seq,
        "lot_index": identity.lot_index,
        "account_id": identity.account_id,
        "symbol": identity.symbol,
        "position_side": identity.position_side,
        "status": "ACTIVE",
        "revision": 0,
        "require_stop": True,
        "created_at": now,
        "updated_at": now,
    }
    with engine.begin() as connection:
        connection.execute(PROTECTION_GROUPS.insert().values(**values))
    values["protection_group_id"] = str(uuid4())
    with pytest.raises(IntegrityError):
        with engine.begin() as connection:
            connection.execute(PROTECTION_GROUPS.insert().values(**values))


def test_verify_protection_fails_closed_on_incomplete_trigger_state() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    step_protection_schema(engine)
    step_protection_constraints(engine)
    group = _group(32)
    identity = group.business_identity
    with engine.begin() as connection:
        connection.execute(
            text(
                "CREATE TABLE hedge_position_lots ("
                "business_lot_id VARCHAR(36) PRIMARY KEY,business_trade_id VARCHAR(36),"
                "account_id VARCHAR(128),symbol VARCHAR(128),position_side VARCHAR(8),"
                "status VARCHAR(32))"
            )
        )
        connection.execute(
            text(
                "INSERT INTO hedge_position_lots VALUES (:lot,:trade,:account,:symbol,:side,'OPEN')"
            ),
            {
                "lot": str(identity.business_lot_id),
                "trade": str(identity.business_trade_id),
                "account": identity.account_id,
                "symbol": identity.symbol,
                "side": identity.position_side,
            },
        )
    repository = SqlProtectionRepository(sessionmaker(bind=engine, expire_on_commit=False))
    repository.save_group(group)
    with engine.begin() as connection:
        connection.execute(
            text(
                "UPDATE hedge_protection_legs SET status='TRIGGERED' "
                "WHERE protection_group_id=:group_id"
            ),
            {"group_id": str(group.protection_group_id)},
        )
        connection.execute(
            text(
                "UPDATE hedge_protection_groups SET status='EXECUTING' "
                "WHERE protection_group_id=:group_id"
            ),
            {"group_id": str(group.protection_group_id)},
        )
    from freqtrade.persistence.hedge_migrations import HedgeMigrationConflict

    with pytest.raises(HedgeMigrationConflict):
        step_verify_protection(engine)
