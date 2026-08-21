"""H3-035..H3-038 business-identity migrations.

These functions are imported by hedge_migrations.py so the existing migration
journal, backup, retry and checksum machinery remains the authority.
"""

from __future__ import annotations

import json
from collections import defaultdict
from uuid import NAMESPACE_URL, uuid5

from sqlalchemy import inspect, text


def _tables(engine) -> set[str]:
    return set(inspect(engine).get_table_names())


def _columns(engine, table: str) -> set[str]:
    if table not in _tables(engine):
        return set()
    return {str(row["name"]) for row in inspect(engine).get_columns(table)}


def _add_column(engine, table: str, name: str, ddl: str) -> bool:
    if table not in _tables(engine) or name in _columns(engine, table):
        return False
    quote = engine.dialect.identifier_preparer.quote
    with engine.begin() as connection:
        connection.execute(
            text(f"ALTER TABLE {quote(table)} ADD COLUMN {quote(name)} {ddl}")
        )
    return True


def step_business_identity_schema(engine) -> dict[str, object]:
    from freqtrade.persistence.hedge_models import (
        BusinessSequenceRow,
        BusinessTradeRow,
        PositionLotRow,
    )

    BusinessSequenceRow.__table__.create(engine, checkfirst=True)
    BusinessTradeRow.__table__.create(engine, checkfirst=True)
    PositionLotRow.__table__.create(engine, checkfirst=True)

    specs = {
        "hedge_order_intents": (
            ("business_trade_id", "VARCHAR(36)"),
            ("business_lot_id", "VARCHAR(36)"),
            ("order_role", "VARCHAR(32)"),
            ("order_revision", "INTEGER DEFAULT 0"),
        ),
        "hedge_order_snapshots": (
            ("business_trade_id", "VARCHAR(36)"),
            ("business_lot_id", "VARCHAR(36)"),
            ("order_role", "VARCHAR(32)"),
        ),
        "hedge_fill_events": (
            ("business_trade_id", "VARCHAR(36)"),
            ("business_lot_id", "VARCHAR(36)"),
            ("order_role", "VARCHAR(32)"),
        ),
        "hedge_current_orders": (
            ("business_trade_id", "VARCHAR(36)"),
            ("business_lot_id", "VARCHAR(36)"),
            ("order_role", "VARCHAR(32)"),
        ),
        "hedge_execution_order_states": (
            ("business_trade_id", "VARCHAR(36)"),
            ("business_lot_id", "VARCHAR(36)"),
            ("order_role", "VARCHAR(32)"),
            ("order_revision", "INTEGER DEFAULT 0"),
        ),
    }
    added: list[str] = []
    for table, columns in specs.items():
        for name, ddl in columns:
            if _add_column(engine, table, name, ddl):
                added.append(f"{table}.{name}")
    return {"created_business_tables": True, "added_columns": added}


def _deterministic_uuid(kind: str, *parts: object) -> str:
    material = "|".join([kind, *(str(part) for part in parts)])
    return str(uuid5(NAMESPACE_URL, f"freqtrade-hedge-business:{material}"))


def step_business_identity_backfill(engine) -> dict[str, object]:
    """Deterministically migrate legacy tactical lots; never guess ambiguous orders."""
    if "hedge_tactical_lots" not in _tables(engine):
        return {"tactical_lots": 0, "active_order_conflicts": 0}

    migrated = 0
    next_seq: dict[tuple[str, str], int] = defaultdict(lambda: 1)

    with engine.begin() as connection:
        if "hedge_execution_order_states" in _tables(engine):
            active = connection.execute(
                text(
                    "SELECT client_order_id, action, lifecycle_status "
                    "FROM hedge_execution_order_states "
                    "WHERE lifecycle_status NOT IN "
                    "('FILLED','CANCELED','CANCELLED','REJECTED','EXPIRED') "
                    "AND (business_trade_id IS NULL OR "
                    "business_lot_id IS NULL OR order_role IS NULL)"
                )
            ).mappings().all()
            conflicts = [
                {
                    "client_order_id": row["client_order_id"],
                    "action": row["action"],
                    "lifecycle_status": row["lifecycle_status"],
                    "reason": "active_managed_order_has_no_provable_business_identity",
                }
                for row in active
            ]
            if conflicts:
                from freqtrade.persistence.hedge_migrations import HedgeMigrationConflict
                raise HedgeMigrationConflict(
                    "Ambiguous active managed orders cannot be auto-backfilled.",
                    {"business_identity_conflicts": conflicts},
                )

        existing = connection.execute(
            text(
                "SELECT exchange, account_id, MAX(business_trade_seq) AS max_seq "
                "FROM hedge_business_trades GROUP BY exchange, account_id"
            )
        ).mappings().all()
        for row in existing:
            next_seq[(str(row["exchange"]), str(row["account_id"]))] = int(row["max_seq"] or 0) + 1

        rows = connection.execute(
            text(
                "SELECT lot_id, exchange, account_id, symbol, position_side, "
                "strategy_name, lot_type, quantity, entry_price, status, opened_at, "
                "closed_at, metadata_json FROM hedge_tactical_lots "
                "ORDER BY exchange, account_id, opened_at, lot_id"
            )
        ).mappings().all()

        for row in rows:
            exchange = str(row["exchange"])
            account_id = str(row["account_id"])
            legacy_lot_id = str(row["lot_id"])
            trade_id = _deterministic_uuid(
                "trade", exchange, account_id, legacy_lot_id
            )
            lot_id = _deterministic_uuid(
                "lot", exchange, account_id, legacy_lot_id
            )
            already = connection.execute(
                text(
                    "SELECT 1 FROM hedge_position_lots "
                    "WHERE business_lot_id=:lot_id"
                ),
                {"lot_id": lot_id},
            ).first()
            if already:
                continue
            seq = next_seq[(exchange, account_id)]
            next_seq[(exchange, account_id)] = seq + 1
            meta = {}
            try:
                meta = json.loads(str(row["metadata_json"] or "{}"))
            except (TypeError, ValueError, json.JSONDecodeError):
                meta = {}
            meta["provenance_legacy_lot_id"] = legacy_lot_id
            qty = str(row["quantity"] or "0")
            status = str(row["status"] or "OPEN").upper()
            open_qty = qty if status not in {"CLOSED", "CANCELED"} else "0"
            closed_qty = "0" if open_qty != "0" else qty
            connection.execute(
                text(
                    "INSERT INTO hedge_business_trades "
                    "(business_trade_id,business_trade_seq,exchange,account_id,symbol,"
                    "position_side,strategy_name,status,created_at,opened_at,closed_at,"
                    "metadata_json,record_version) "
                    "VALUES (:trade_id,:seq,:exchange,:account,:symbol,:side,:strategy,"
                    ":status,:opened,:opened,:closed,:meta,1)"
                ),
                {
                    "trade_id": trade_id,
                    "seq": seq,
                    "exchange": exchange,
                    "account": account_id,
                    "symbol": str(row["symbol"]),
                    "side": str(row["position_side"]),
                    "strategy": row["strategy_name"],
                    "status": "CLOSED" if open_qty == "0" else "OPEN",
                    "opened": row["opened_at"],
                    "closed": row["closed_at"],
                    "meta": json.dumps(meta, sort_keys=True, separators=(",", ":")),
                },
            )
            connection.execute(
                text(
                    "INSERT INTO hedge_position_lots "
                    "(business_lot_id,business_trade_id,lot_index,exchange,account_id,"
                    "symbol,position_side,bucket,status,original_quantity,"
                    "entry_filled_quantity,open_quantity,closed_quantity,entry_quote,"
                    "average_entry_price,realized_pnl,fees,funding,opened_at,closed_at,"
                    "metadata_json,record_version) "
                    "VALUES (:lot_id,:trade_id,1,:exchange,:account,:symbol,:side,"
                    "'TACTICAL',:status,:qty,:qty,:open_qty,:closed_qty,'0',:price,"
                    "'0','0','0',:opened,:closed,:meta,1)"
                ),
                {
                    "lot_id": lot_id,
                    "trade_id": trade_id,
                    "exchange": exchange,
                    "account": account_id,
                    "symbol": str(row["symbol"]),
                    "side": str(row["position_side"]),
                    "status": "CLOSED" if open_qty == "0" else "OPEN",
                    "qty": qty,
                    "open_qty": open_qty,
                    "closed_qty": closed_qty,
                    "price": str(row["entry_price"] or "0"),
                    "opened": row["opened_at"],
                    "closed": row["closed_at"],
                    "meta": json.dumps(meta, sort_keys=True, separators=(",", ":")),
                },
            )
            migrated += 1

        scopes = connection.execute(
            text(
                "SELECT exchange, account_id, MAX(business_trade_seq) AS max_seq "
                "FROM hedge_business_trades GROUP BY exchange, account_id"
            )
        ).mappings().all()
        synced = 0
        for scope in scopes:
            desired_next = int(scope["max_seq"] or 0) + 1
            current = connection.execute(
                text(
                    "SELECT id, next_trade_seq, revision FROM hedge_business_sequences "
                    "WHERE exchange=:exchange AND account_id=:account_id"
                ),
                {
                    "exchange": scope["exchange"],
                    "account_id": scope["account_id"],
                },
            ).mappings().first()
            if current is None:
                connection.execute(
                    text(
                        "INSERT INTO hedge_business_sequences "
                        "(exchange, account_id, next_trade_seq, revision, updated_at) "
                        "VALUES (:exchange, :account_id, :next_trade_seq, 1, CURRENT_TIMESTAMP)"
                    ),
                    {
                        "exchange": scope["exchange"],
                        "account_id": scope["account_id"],
                        "next_trade_seq": desired_next,
                    },
                )
                synced += 1
            elif int(current["next_trade_seq"]) < desired_next:
                connection.execute(
                    text(
                        "UPDATE hedge_business_sequences SET next_trade_seq=:next_trade_seq, "
                        "revision=:revision, updated_at=CURRENT_TIMESTAMP WHERE id=:id"
                    ),
                    {
                        "next_trade_seq": desired_next,
                        "revision": int(current["revision"]) + 1,
                        "id": current["id"],
                    },
                )
                synced += 1

    return {
        "tactical_lots": migrated,
        "active_order_conflicts": 0,
        "sequence_scopes_synced": synced,
    }


def step_business_identity_constraints(engine) -> dict[str, object]:
    specs = (
        (
            "hedge_business_trades",
            "uq_hedge_business_trade_account_seq",
            (
                "CREATE UNIQUE INDEX uq_hedge_business_trade_account_seq "
                "ON hedge_business_trades(exchange, account_id, business_trade_seq)"
            ),
        ),
        (
            "hedge_position_lots",
            "uq_hedge_position_lot_trade_index",
            (
                "CREATE UNIQUE INDEX uq_hedge_position_lot_trade_index "
                "ON hedge_position_lots(business_trade_id, lot_index)"
            ),
        ),
        (
            "hedge_position_lots",
            "ix_hedge_position_lot_open_side",
            (
                "CREATE INDEX ix_hedge_position_lot_open_side "
                "ON hedge_position_lots(account_id, symbol, position_side, status)"
            ),
        ),
    )
    created: list[str] = []
    for table, name, expression in specs:
        if table not in _tables(engine):
            continue
        indexes = {row["name"] for row in inspect(engine).get_indexes(table)}
        if name in indexes:
            continue
        with engine.begin() as connection:
            connection.execute(text(expression))
        created.append(name)
    return {"created_indexes": created}


def step_verify_business_identity(engine) -> dict[str, object]:
    required_tables = {
        "hedge_business_sequences",
        "hedge_business_trades",
        "hedge_position_lots",
    }
    missing_tables = sorted(required_tables - _tables(engine))
    required_columns = {
        "hedge_order_intents": {
            "business_trade_id",
            "business_lot_id",
            "order_role",
            "order_revision",
        },
        "hedge_order_snapshots": {"business_trade_id", "business_lot_id", "order_role"},
        "hedge_fill_events": {"business_trade_id", "business_lot_id", "order_role"},
        "hedge_current_orders": {"business_trade_id", "business_lot_id", "order_role"},
        "hedge_execution_order_states": {
            "business_trade_id", "business_lot_id", "order_role", "order_revision"
        },
    }
    missing_columns = [
        f"{table}.{column}"
        for table, expected in required_columns.items()
        for column in sorted(expected - _columns(engine, table))
    ]
    violations: dict[str, int] = {}
    with engine.connect() as connection:
        if "hedge_execution_order_states" in _tables(engine):
            violations["active_managed_missing_identity"] = int(
                connection.execute(
                    text(
                        "SELECT COUNT(*) FROM hedge_execution_order_states "
                        "WHERE lifecycle_status NOT IN "
                        "('FILLED','CANCELED','CANCELLED','REJECTED','EXPIRED') "
                        "AND (business_trade_id IS NULL OR "
                        "business_lot_id IS NULL OR order_role IS NULL)"
                    )
                ).scalar_one()
            )
        if "hedge_business_trades" in _tables(engine):
            violations["duplicate_trade_seq"] = int(
                connection.execute(
                    text(
                        "SELECT COUNT(*) FROM (SELECT exchange,account_id,business_trade_seq,"
                        "COUNT(*) c FROM hedge_business_trades GROUP BY exchange,account_id,"
                        "business_trade_seq HAVING COUNT(*)>1) q"
                    )
                ).scalar_one()
            )
        if "hedge_position_lots" in _tables(engine):
            violations["orphan_lots"] = int(
                connection.execute(
                    text(
                        "SELECT COUNT(*) FROM hedge_position_lots l "
                        "LEFT JOIN hedge_business_trades t "
                        "ON t.business_trade_id=l.business_trade_id "
                        "WHERE t.business_trade_id IS NULL"
                    )
                ).scalar_one()
            )
    violations = {key: value for key, value in violations.items() if value}
    if missing_tables or missing_columns or violations:
        from freqtrade.persistence.hedge_migrations import HedgeMigrationError
        raise HedgeMigrationError(
            json.dumps(
                {
                    "missing_tables": missing_tables,
                    "missing_columns": missing_columns,
                    "violations": violations,
                },
                sort_keys=True,
            )
        )
    return {"business_identity": "verified"}
