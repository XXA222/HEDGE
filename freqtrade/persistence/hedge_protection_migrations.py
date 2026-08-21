"""H3-039..H3-041 exact business-lot protection migrations."""

from __future__ import annotations

from sqlalchemy import inspect, text

from freqtrade.persistence.hedge_protection import PROTECTION_GROUPS, PROTECTION_LEGS


PROTECTION_TABLES = ("hedge_protection_groups", "hedge_protection_legs")


def _tables(engine) -> set[str]:
    return set(inspect(engine).get_table_names())


def step_protection_schema(engine) -> dict[str, object]:
    PROTECTION_GROUPS.create(engine, checkfirst=True)
    PROTECTION_LEGS.create(engine, checkfirst=True)
    return {"created": list(PROTECTION_TABLES)}


def step_protection_constraints(engine) -> dict[str, object]:
    if not set(PROTECTION_TABLES) <= _tables(engine):
        raise RuntimeError("protection tables must exist before constraints")
    with engine.begin() as connection:
        if engine.dialect.name == "sqlite":
            connection.execute(
                text(
                    "CREATE UNIQUE INDEX IF NOT EXISTS uq_hedge_protection_active_lot "
                    "ON hedge_protection_groups(business_lot_id) "
                    "WHERE status IN ('ACTIVE','EXECUTING')"
                )
            )
        elif engine.dialect.name == "postgresql":
            connection.execute(
                text(
                    "CREATE UNIQUE INDEX IF NOT EXISTS uq_hedge_protection_active_lot "
                    "ON hedge_protection_groups(business_lot_id) "
                    "WHERE status IN ('ACTIVE','EXECUTING')"
                )
            )
        else:
            # Other dialects still keep the repository/service invariant.  SQLite and
            # PostgreSQL are the production acceptance authorities for this project.
            return {"active_lot_unique_index": "application-enforced"}
    return {"active_lot_unique_index": "created"}


def step_verify_protection(engine) -> dict[str, object]:
    tables = _tables(engine)
    missing = sorted(set(PROTECTION_TABLES) - tables)
    if missing:
        raise RuntimeError(f"protection tables are missing: {missing}")

    problems: dict[str, object] = {}
    with engine.connect() as connection:
        if "hedge_position_lots" in tables:
            orphan = (
                connection.execute(
                    text(
                        "SELECT g.protection_group_id,g.business_lot_id "
                        "FROM hedge_protection_groups g "
                        "LEFT JOIN hedge_position_lots l "
                        "ON l.business_lot_id=g.business_lot_id "
                        "WHERE l.business_lot_id IS NULL"
                    )
                )
                .mappings()
                .all()
            )
            if orphan:
                problems["orphan_groups"] = [dict(row) for row in orphan]

            scope = (
                connection.execute(
                    text(
                        "SELECT g.protection_group_id,g.business_lot_id "
                        "FROM hedge_protection_groups g "
                        "JOIN hedge_position_lots l ON l.business_lot_id=g.business_lot_id "
                        "WHERE g.business_trade_id<>l.business_trade_id "
                        "OR g.account_id<>l.account_id OR g.symbol<>l.symbol "
                        "OR g.position_side<>l.position_side"
                    )
                )
                .mappings()
                .all()
            )
            if scope:
                problems["group_lot_scope_mismatch"] = [dict(row) for row in scope]

            active_closed = (
                connection.execute(
                    text(
                        "SELECT g.protection_group_id,g.business_lot_id,l.status "
                        "FROM hedge_protection_groups g "
                        "JOIN hedge_position_lots l ON l.business_lot_id=g.business_lot_id "
                        "WHERE g.status IN ('ACTIVE','EXECUTING') "
                        "AND l.status NOT IN ('PARTIAL_OPEN','OPEN','PARTIAL_CLOSED','ADOPTED')"
                    )
                )
                .mappings()
                .all()
            )
            if active_closed:
                problems["active_group_targets_closed_lot"] = [dict(row) for row in active_closed]

        missing_stop = (
            connection.execute(
                text(
                    "SELECT g.protection_group_id FROM hedge_protection_groups g "
                    "WHERE g.status IN ('ACTIVE','EXECUTING') AND g.require_stop=1 "
                    "AND NOT EXISTS (SELECT 1 FROM hedge_protection_legs l "
                    "WHERE l.protection_group_id=g.protection_group_id "
                    "AND l.kind IN ('STOP_LOSS','TRAILING_STOP') "
                    "AND l.status NOT IN ('CANCELED','FAILED'))"
                )
            )
            .mappings()
            .all()
        )
        if missing_stop:
            problems["required_stop_missing"] = [dict(row) for row in missing_stop]

        execution_counts = (
            connection.execute(
                text(
                    "SELECT g.protection_group_id,g.status,"
                    "SUM(CASE WHEN l.status IN "
                    "('TRIGGERED','SUBMITTED','PARTIAL') "
                    "THEN 1 ELSE 0 END) "
                    "AS active_execution_count "
                    "FROM hedge_protection_groups g "
                    "LEFT JOIN hedge_protection_legs l "
                    "ON l.protection_group_id=g.protection_group_id "
                    "WHERE g.status IN ('ACTIVE','EXECUTING') "
                    "GROUP BY g.protection_group_id,g.status"
                )
            )
            .mappings()
            .all()
        )
        invalid_execution = [
            dict(row)
            for row in execution_counts
            if (str(row["status"]) == "ACTIVE" and int(row["active_execution_count"] or 0) != 0)
            or (str(row["status"]) == "EXECUTING" and int(row["active_execution_count"] or 0) != 1)
        ]
        if invalid_execution:
            problems["group_execution_state_invalid"] = invalid_execution

        invalid_trigger = (
            connection.execute(
                text(
                    "SELECT protection_id,status FROM hedge_protection_legs "
                    "WHERE status IN ('TRIGGERED','SUBMITTED','PARTIAL','FILLED') "
                    "AND (execution_intent_id IS NULL OR trigger_quantity IS NULL)"
                )
            )
            .mappings()
            .all()
        )
        if invalid_trigger:
            problems["trigger_state_incomplete"] = [dict(row) for row in invalid_trigger]

        remaining_fixed = (
            connection.execute(
                text(
                    "SELECT protection_id FROM hedge_protection_legs "
                    "WHERE quantity_mode='REMAINING' AND quantity IS NOT NULL"
                )
            )
            .mappings()
            .all()
        )
        if remaining_fixed:
            problems["remaining_mode_has_fixed_quantity"] = [dict(row) for row in remaining_fixed]

    if problems:
        from freqtrade.persistence.hedge_migrations import HedgeMigrationConflict

        raise HedgeMigrationConflict(
            "Business-lot protection integrity verification failed.",
            {"protection_conflicts": problems},
        )
    return {"verified": True, "tables": list(PROTECTION_TABLES)}
