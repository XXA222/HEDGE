"""Durable SQL repository for exact business-lot protection groups."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
    select,
)

from freqtrade.hedge.contracts.business_identity import BusinessIdentity
from freqtrade.hedge.protection.contracts import (
    ProtectionGroup,
    ProtectionGroupStatus,
    ProtectionKind,
    ProtectionLeg,
    ProtectionLegStatus,
    ProtectionQuantityMode,
)


PROTECTION_METADATA = MetaData()

PROTECTION_GROUPS = Table(
    "hedge_protection_groups",
    PROTECTION_METADATA,
    Column("id", Integer, primary_key=True),
    Column("protection_group_id", String(36), nullable=False),
    Column("business_trade_id", String(36), nullable=False),
    Column("business_lot_id", String(36), nullable=False),
    Column("business_trade_seq", Integer, nullable=False),
    Column("lot_index", Integer, nullable=False),
    Column("account_id", String(128), nullable=False),
    Column("symbol", String(128), nullable=False),
    Column("position_side", String(8), nullable=False),
    Column("status", String(16), nullable=False),
    Column("revision", Integer, nullable=False, default=0),
    Column("require_stop", Boolean, nullable=False, default=True),
    Column("created_at", DateTime(timezone=False), nullable=False),
    Column("updated_at", DateTime(timezone=False), nullable=False),
    UniqueConstraint("protection_group_id", name="uq_hedge_protection_group_id"),
    CheckConstraint("position_side IN ('LONG','SHORT')", name="ck_hedge_protection_group_side"),
    CheckConstraint(
        "status IN ('ACTIVE','EXECUTING','CLOSED','CANCELED','ERROR')",
        name="ck_hedge_protection_group_status",
    ),
    Index(
        "ix_hedge_protection_group_lot_status",
        "business_lot_id",
        "status",
    ),
    Index(
        "ix_hedge_protection_group_scope",
        "account_id",
        "symbol",
        "position_side",
        "status",
    ),
)

PROTECTION_LEGS = Table(
    "hedge_protection_legs",
    PROTECTION_METADATA,
    Column("id", Integer, primary_key=True),
    Column("protection_id", String(36), nullable=False),
    Column(
        "protection_group_id",
        String(36),
        ForeignKey("hedge_protection_groups.protection_group_id"),
        nullable=False,
    ),
    Column("kind", String(32), nullable=False),
    Column("label", String(64), nullable=False),
    Column("quantity_mode", String(16), nullable=False),
    Column("quantity", String(80), nullable=True),
    Column("trigger_price", String(80), nullable=True),
    Column("trailing_distance", String(80), nullable=True),
    Column("status", String(16), nullable=False),
    Column("revision", Integer, nullable=False, default=0),
    Column("high_watermark", String(80), nullable=True),
    Column("low_watermark", String(80), nullable=True),
    Column("execution_intent_id", String(36), nullable=True),
    Column("trigger_quantity", String(80), nullable=True),
    Column("client_order_id", String(256), nullable=True),
    Column("exchange_order_id", String(256), nullable=True),
    Column("filled_quantity", String(80), nullable=False, default="0"),
    Column("last_error", Text, nullable=True),
    Column("created_at", DateTime(timezone=False), nullable=False),
    Column("updated_at", DateTime(timezone=False), nullable=False),
    UniqueConstraint("protection_id", name="uq_hedge_protection_id"),
    UniqueConstraint(
        "protection_group_id",
        "label",
        name="uq_hedge_protection_group_label",
    ),
    CheckConstraint(
        "kind IN ('TAKE_PROFIT','STOP_LOSS','TRAILING_STOP')",
        name="ck_hedge_protection_leg_kind",
    ),
    CheckConstraint(
        "quantity_mode IN ('ABSOLUTE','REMAINING')",
        name="ck_hedge_protection_quantity_mode",
    ),
    CheckConstraint(
        "status IN ('ARMED','TRIGGERED','SUBMITTED','PARTIAL','FILLED','CANCELED','FAILED')",
        name="ck_hedge_protection_leg_status",
    ),
    Index(
        "ix_hedge_protection_leg_group_status",
        "protection_group_id",
        "status",
    ),
    Index("ix_hedge_protection_leg_client_order", "client_order_id"),
)


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _decimal_text(value: Decimal | None) -> str | None:
    return None if value is None else format(value, "f")


def _uuid_text(value: UUID | None) -> str | None:
    return None if value is None else str(value)


def _identity_from_group_row(row: Any) -> BusinessIdentity:
    return BusinessIdentity(
        business_trade_id=row["business_trade_id"],
        business_trade_seq=int(row["business_trade_seq"]),
        business_lot_id=row["business_lot_id"],
        lot_index=int(row["lot_index"]),
        account_id=row["account_id"],
        symbol=row["symbol"],
        position_side=row["position_side"],
    )


def _leg_from_row(row: Any, *, identity: BusinessIdentity) -> ProtectionLeg:
    return ProtectionLeg(
        protection_group_id=row["protection_group_id"],
        business_identity=identity,
        kind=ProtectionKind(row["kind"]),
        label=row["label"],
        quantity_mode=ProtectionQuantityMode(row["quantity_mode"]),
        quantity=row["quantity"],
        trigger_price=row["trigger_price"],
        trailing_distance=row["trailing_distance"],
        protection_id=row["protection_id"],
        status=ProtectionLegStatus(row["status"]),
        revision=int(row["revision"]),
        high_watermark=row["high_watermark"],
        low_watermark=row["low_watermark"],
        execution_intent_id=row["execution_intent_id"],
        trigger_quantity=row["trigger_quantity"],
        client_order_id=row["client_order_id"],
        exchange_order_id=row["exchange_order_id"],
        filled_quantity=row["filled_quantity"],
        last_error=row["last_error"],
    )


class SqlProtectionRepository:
    """Session-factory repository with one transaction per group-state update."""

    def __init__(self, session_factory: Any) -> None:
        if not callable(session_factory):
            raise TypeError("session_factory must be callable")
        self._session_factory = session_factory

    def save_group(self, group: ProtectionGroup) -> None:
        if not isinstance(group, ProtectionGroup):
            raise TypeError("group must be ProtectionGroup")
        now = _now()
        identity = group.business_identity
        group_values = {
            "protection_group_id": str(group.protection_group_id),
            "business_trade_id": str(identity.business_trade_id),
            "business_lot_id": str(identity.business_lot_id),
            "business_trade_seq": identity.business_trade_seq,
            "lot_index": identity.lot_index,
            "account_id": identity.account_id,
            "symbol": identity.symbol,
            "position_side": identity.position_side,
            "status": group.status.value,
            "revision": group.revision,
            "require_stop": group.require_stop,
            "updated_at": now,
        }
        with self._session_factory.begin() as session:
            existing = (
                session.execute(
                    select(PROTECTION_GROUPS)
                    .where(
                        PROTECTION_GROUPS.c.protection_group_id == str(group.protection_group_id)
                    )
                    .with_for_update()
                )
                .mappings()
                .first()
            )
            if existing is None:
                session.execute(
                    PROTECTION_GROUPS.insert().values(
                        **group_values,
                        created_at=now,
                    )
                )
            else:
                if int(existing["revision"]) > group.revision:
                    raise RuntimeError("refusing to overwrite newer protection group revision")
                session.execute(
                    PROTECTION_GROUPS.update()
                    .where(
                        PROTECTION_GROUPS.c.protection_group_id == str(group.protection_group_id)
                    )
                    .values(**group_values)
                )

            for leg in group.legs:
                values = self._leg_values(leg, updated_at=now)
                current = (
                    session.execute(
                        select(PROTECTION_LEGS)
                        .where(PROTECTION_LEGS.c.protection_id == str(leg.protection_id))
                        .with_for_update()
                    )
                    .mappings()
                    .first()
                )
                if current is None:
                    session.execute(
                        PROTECTION_LEGS.insert().values(
                            **values,
                            created_at=now,
                        )
                    )
                else:
                    if str(current["protection_group_id"]) != str(group.protection_group_id):
                        raise RuntimeError("protection leg changed parent group")
                    if int(current["revision"]) > leg.revision:
                        raise RuntimeError("refusing to overwrite newer protection leg revision")
                    session.execute(
                        PROTECTION_LEGS.update()
                        .where(PROTECTION_LEGS.c.protection_id == str(leg.protection_id))
                        .values(**values)
                    )

    def load_group(self, protection_group_id: UUID | str) -> ProtectionGroup:
        key = str(UUID(str(protection_group_id)))
        with self._session_factory() as session:
            group_row = (
                session.execute(
                    select(PROTECTION_GROUPS).where(PROTECTION_GROUPS.c.protection_group_id == key)
                )
                .mappings()
                .first()
            )
            if group_row is None:
                raise KeyError(key)
            leg_rows = (
                session.execute(
                    select(PROTECTION_LEGS)
                    .where(PROTECTION_LEGS.c.protection_group_id == key)
                    .order_by(PROTECTION_LEGS.c.id)
                )
                .mappings()
                .all()
            )
        return self._group_from_rows(group_row, leg_rows)

    def groups_for_lot(
        self,
        business_lot_id: UUID | str,
        *,
        include_terminal: bool = False,
    ) -> tuple[ProtectionGroup, ...]:
        lot_id = str(UUID(str(business_lot_id)))
        with self._session_factory() as session:
            statement = select(PROTECTION_GROUPS).where(
                PROTECTION_GROUPS.c.business_lot_id == lot_id
            )
            if not include_terminal:
                statement = statement.where(PROTECTION_GROUPS.c.status.in_(("ACTIVE", "EXECUTING")))
            group_rows = (
                session.execute(statement.order_by(PROTECTION_GROUPS.c.id)).mappings().all()
            )
            if not group_rows:
                return ()
            group_ids = [str(row["protection_group_id"]) for row in group_rows]
            leg_rows = (
                session.execute(
                    select(PROTECTION_LEGS)
                    .where(PROTECTION_LEGS.c.protection_group_id.in_(group_ids))
                    .order_by(PROTECTION_LEGS.c.id)
                )
                .mappings()
                .all()
            )
        by_group: dict[str, list[Any]] = {group_id: [] for group_id in group_ids}
        for row in leg_rows:
            by_group[str(row["protection_group_id"])].append(row)
        return tuple(
            self._group_from_rows(row, by_group[str(row["protection_group_id"])])
            for row in group_rows
        )

    @staticmethod
    def _leg_values(leg: ProtectionLeg, *, updated_at: datetime) -> dict[str, object]:
        return {
            "protection_id": str(leg.protection_id),
            "protection_group_id": str(leg.protection_group_id),
            "kind": leg.kind.value,
            "label": leg.label,
            "quantity_mode": leg.quantity_mode.value,
            "quantity": _decimal_text(leg.quantity),
            "trigger_price": _decimal_text(leg.trigger_price),
            "trailing_distance": _decimal_text(leg.trailing_distance),
            "status": leg.status.value,
            "revision": leg.revision,
            "high_watermark": _decimal_text(leg.high_watermark),
            "low_watermark": _decimal_text(leg.low_watermark),
            "execution_intent_id": _uuid_text(leg.execution_intent_id),
            "trigger_quantity": _decimal_text(leg.trigger_quantity),
            "client_order_id": leg.client_order_id,
            "exchange_order_id": leg.exchange_order_id,
            "filled_quantity": _decimal_text(leg.filled_quantity),
            "last_error": leg.last_error,
            "updated_at": updated_at,
        }

    @staticmethod
    def _group_from_rows(group_row: Any, leg_rows: list[Any]) -> ProtectionGroup:
        identity = _identity_from_group_row(group_row)
        legs = tuple(_leg_from_row(row, identity=identity) for row in leg_rows)
        return ProtectionGroup(
            business_identity=identity,
            protection_group_id=group_row["protection_group_id"],
            legs=legs,
            status=ProtectionGroupStatus(group_row["status"]),
            revision=int(group_row["revision"]),
            require_stop=bool(group_row["require_stop"]),
        )
