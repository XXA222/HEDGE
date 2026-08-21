from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError

from freqtrade.persistence.hedge_migrations import HedgeMigrationConflict
from freqtrade.persistence.hedge_protection_migrations import (
    step_protection_constraints,
    step_protection_schema,
    step_verify_protection,
)


def _group_values(*, lot_id: str, group_id: str | None = None) -> dict[str, object]:
    now = datetime(2026, 8, 21, tzinfo=UTC).replace(tzinfo=None)
    return {
        "protection_group_id": group_id or str(uuid4()),
        "business_trade_id": str(uuid4()),
        "business_lot_id": lot_id,
        "business_trade_seq": 1,
        "lot_index": 1,
        "account_id": "main",
        "symbol": "BTCUSDT",
        "position_side": "LONG",
        "status": "ACTIVE",
        "revision": 0,
        "require_stop": False,
        "created_at": now,
        "updated_at": now,
    }


def test_protection_schema_and_constraints_are_idempotent() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    step_protection_schema(engine)
    step_protection_schema(engine)
    step_protection_constraints(engine)
    step_protection_constraints(engine)
    assert step_verify_protection(engine)["verified"] is True


def test_protection_active_group_is_unique_per_business_lot() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    step_protection_schema(engine)
    step_protection_constraints(engine)
    lot_id = str(uuid4())
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO hedge_protection_groups "
                "(protection_group_id,business_trade_id,business_lot_id,business_trade_seq,"
                "lot_index,account_id,symbol,position_side,status,revision,require_stop,"
                "created_at,updated_at) VALUES "
                "(:protection_group_id,:business_trade_id,:business_lot_id,:business_trade_seq,"
                ":lot_index,:account_id,:symbol,:position_side,:status,:revision,:require_stop,"
                ":created_at,:updated_at)"
            ),
            _group_values(lot_id=lot_id),
        )
    with pytest.raises(IntegrityError):
        with engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO hedge_protection_groups "
                    "(protection_group_id,business_trade_id,business_lot_id,business_trade_seq,"
                    "lot_index,account_id,symbol,position_side,status,revision,require_stop,"
                    "created_at,updated_at) VALUES "
                    "(:protection_group_id,:business_trade_id,:business_lot_id,:business_trade_seq,"
                    ":lot_index,:account_id,:symbol,:position_side,:status,:revision,:require_stop,"
                    ":created_at,:updated_at)"
                ),
                _group_values(lot_id=lot_id),
            )


def test_protection_verify_fails_closed_on_orphan_group() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    step_protection_schema(engine)
    with engine.begin() as connection:
        connection.execute(
            text(
                "CREATE TABLE hedge_position_lots ("
                "business_lot_id TEXT PRIMARY KEY,business_trade_id TEXT NOT NULL,"
                "account_id TEXT NOT NULL,symbol TEXT NOT NULL,position_side TEXT NOT NULL,"
                "status TEXT NOT NULL)"
            )
        )
        connection.execute(
            text(
                "INSERT INTO hedge_protection_groups "
                "(protection_group_id,business_trade_id,business_lot_id,business_trade_seq,"
                "lot_index,account_id,symbol,position_side,status,revision,require_stop,"
                "created_at,updated_at) VALUES "
                "(:protection_group_id,:business_trade_id,:business_lot_id,:business_trade_seq,"
                ":lot_index,:account_id,:symbol,:position_side,:status,:revision,:require_stop,"
                ":created_at,:updated_at)"
            ),
            _group_values(lot_id=str(uuid4())),
        )
    with pytest.raises(HedgeMigrationConflict, match="protection integrity"):
        step_verify_protection(engine)
