"""Transaction-safe SQL authority for HEDGE business identity."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any, Protocol
from uuid import NAMESPACE_URL, UUID, uuid5

from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError

from freqtrade.hedge.contracts.business_identity import (
    BusinessIdentity,
    canonical_business_side,
    canonical_business_symbol,
)


class BusinessIdentityPersistenceError(RuntimeError):
    pass


class BusinessIdentitySession(Protocol):
    def execute(self, statement: object, params: object | None = None) -> object: ...

    def flush(self) -> None: ...


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _entry_uuid(kind: str, *, exchange: str, account_id: str, key: str) -> UUID:
    material = f"freqtrade-hedge-business-entry|{kind}|{exchange}|{account_id}|{key}"
    return uuid5(NAMESPACE_URL, material)


def _bucket(value: object) -> str:
    raw = str(getattr(value, "value", value)).strip().upper()
    if raw not in {"CORE", "TACTICAL"}:
        raise BusinessIdentityPersistenceError("business lot bucket must be CORE or TACTICAL")
    return raw


def _identity_from_rows(trade: object, lot: object) -> BusinessIdentity:
    return BusinessIdentity(
        business_trade_id=UUID(str(trade.business_trade_id)),
        business_trade_seq=int(trade.business_trade_seq),
        business_lot_id=UUID(str(lot.business_lot_id)),
        lot_index=int(lot.lot_index),
        account_id=str(lot.account_id),
        symbol=str(lot.symbol),
        position_side=str(lot.position_side),
    )


def allocate_business_identity(
    session: BusinessIdentitySession,
    *,
    exchange: str,
    account_id: str,
    symbol: str,
    position_side: object,
    strategy_entry_key: str,
    bucket: object,
    sequence_table: object,
    trade_table: object,
    lot_table: object,
) -> BusinessIdentity:
    """Allocate or replay one durable business trade in the caller transaction."""

    exchange_value = str(exchange).strip().lower()
    account_value = str(account_id).strip()
    entry_key = str(strategy_entry_key).strip()
    if not exchange_value or not account_value or not entry_key:
        raise BusinessIdentityPersistenceError(
            "exchange, account_id and strategy_entry_key are required"
        )
    if len(entry_key) > 512:
        raise BusinessIdentityPersistenceError("strategy_entry_key is too long")
    symbol_value = canonical_business_symbol(symbol)
    side_value = canonical_business_side(position_side)
    bucket_value = _bucket(bucket)
    trade_uuid = _entry_uuid(
        "trade",
        exchange=exchange_value,
        account_id=account_value,
        key=entry_key,
    )
    lot_uuid = _entry_uuid(
        "lot",
        exchange=exchange_value,
        account_id=account_value,
        key=entry_key,
    )

    existing_trade = (
        session.execute(
            select(trade_table).where(trade_table.c.business_trade_id == str(trade_uuid))
        )
        .mappings()
        .first()
    )
    if existing_trade is not None:
        existing_lot = (
            session.execute(select(lot_table).where(lot_table.c.business_lot_id == str(lot_uuid)))
            .mappings()
            .first()
        )
        if existing_lot is None:
            raise BusinessIdentityPersistenceError(
                "existing deterministic business trade has no matching lot"
            )
        identity = BusinessIdentity(
            business_trade_id=trade_uuid,
            business_trade_seq=int(existing_trade["business_trade_seq"]),
            business_lot_id=lot_uuid,
            lot_index=int(existing_lot["lot_index"]),
            account_id=str(existing_lot["account_id"]),
            symbol=str(existing_lot["symbol"]),
            position_side=str(existing_lot["position_side"]),
        )
        identity.assert_matches(
            account_id=account_value,
            symbol=symbol_value,
            position_side=side_value,
        )
        if str(existing_lot["bucket"]).upper() != bucket_value:
            raise BusinessIdentityPersistenceError(
                "deterministic business identity replay changed the lot bucket"
            )
        return identity

    row = (
        session.execute(
            select(sequence_table)
            .where(sequence_table.c.exchange == exchange_value)
            .where(sequence_table.c.account_id == account_value)
            .with_for_update()
        )
        .mappings()
        .first()
    )

    if row is None:
        max_seq = session.execute(
            select(func.max(trade_table.c.business_trade_seq)).where(
                trade_table.c.exchange == exchange_value,
                trade_table.c.account_id == account_value,
            )
        ).scalar_one_or_none()
        seq = int(max_seq or 0) + 1
        session.execute(
            sequence_table.insert().values(
                exchange=exchange_value,
                account_id=account_value,
                next_trade_seq=seq + 1,
                revision=1,
                updated_at=_now(),
            )
        )
    else:
        seq = int(row["next_trade_seq"])
        if seq <= 0:
            raise BusinessIdentityPersistenceError("business sequence is corrupt")
        session.execute(
            update(sequence_table)
            .where(sequence_table.c.exchange == exchange_value)
            .where(sequence_table.c.account_id == account_value)
            .values(
                next_trade_seq=seq + 1,
                revision=int(row["revision"]) + 1,
                updated_at=_now(),
            )
        )

    identity = BusinessIdentity(
        business_trade_id=trade_uuid,
        business_trade_seq=seq,
        business_lot_id=lot_uuid,
        lot_index=1,
        account_id=account_value,
        symbol=symbol_value,
        position_side=side_value,
    )
    now = _now()
    metadata = json.dumps(
        {"strategy_entry_key": entry_key, "allocation": "deterministic_uuid5"},
        sort_keys=True,
        separators=(",", ":"),
    )
    session.execute(
        trade_table.insert().values(
            business_trade_id=str(identity.business_trade_id),
            business_trade_seq=identity.business_trade_seq,
            exchange=exchange_value,
            account_id=identity.account_id,
            symbol=identity.symbol,
            position_side=canonical_business_side(identity.position_side),
            status="PLANNED",
            origin_decision_id=entry_key[:128],
            created_at=now,
            metadata_json=metadata,
            record_version=1,
        )
    )
    session.execute(
        lot_table.insert().values(
            business_lot_id=str(identity.business_lot_id),
            business_trade_id=str(identity.business_trade_id),
            lot_index=identity.lot_index,
            exchange=exchange_value,
            account_id=identity.account_id,
            symbol=identity.symbol,
            position_side=canonical_business_side(identity.position_side),
            bucket=bucket_value,
            status="PLANNED",
            original_quantity="0",
            entry_filled_quantity="0",
            open_quantity="0",
            closed_quantity="0",
            entry_quote="0",
            average_entry_price="0",
            realized_pnl="0",
            fees="0",
            funding="0",
            metadata_json=metadata,
            record_version=1,
        )
    )
    session.flush()
    return identity


class SqlBusinessIdentityAllocator:
    """Session-factory adapter used by the production BusinessIdentityBinder."""

    def __init__(self, session_factory: Any) -> None:
        if not callable(session_factory):
            raise TypeError("session_factory must be callable")
        self._session_factory = session_factory

    def allocate_entry(
        self,
        *,
        account_id: str,
        exchange: str,
        symbol: str,
        position_side: object,
        strategy_entry_key: str,
        bucket: object,
    ) -> BusinessIdentity:
        from freqtrade.persistence.hedge_models import (
            BusinessSequenceRow,
            BusinessTradeRow,
            PositionLotRow,
        )

        for attempt in range(2):
            try:
                with self._session_factory.begin() as session:
                    return allocate_business_identity(
                        session,
                        exchange=exchange,
                        account_id=account_id,
                        symbol=symbol,
                        position_side=position_side,
                        strategy_entry_key=strategy_entry_key,
                        bucket=bucket,
                        sequence_table=BusinessSequenceRow.__table__,
                        trade_table=BusinessTradeRow.__table__,
                        lot_table=PositionLotRow.__table__,
                    )
            except IntegrityError as exc:
                if attempt:
                    raise BusinessIdentityPersistenceError(
                        "business identity allocation lost a uniqueness race"
                    ) from exc
        raise BusinessIdentityPersistenceError("business identity allocation failed")

    def load_for_lot(self, business_lot_id: object) -> BusinessIdentity:
        from freqtrade.persistence.hedge_models import BusinessTradeRow, PositionLotRow

        key = str(business_lot_id).strip()
        if not key:
            raise ValueError("business_lot_id is required")
        with self._session_factory() as session:
            pair = session.execute(
                select(PositionLotRow, BusinessTradeRow)
                .join(
                    BusinessTradeRow,
                    BusinessTradeRow.business_trade_id == PositionLotRow.business_trade_id,
                )
                .where(PositionLotRow.business_lot_id == key)
            ).one_or_none()
            if pair is None:
                raise KeyError(key)
            lot, trade = pair
            return _identity_from_rows(trade, lot)

    def list_open_lots(
        self,
        *,
        account_id: str,
        symbol: str,
        position_side: object,
    ) -> tuple[tuple[BusinessIdentity, str, str], ...]:
        """Return identity, bucket and open quantity for reconciliation/recovery."""
        from freqtrade.persistence.hedge_models import BusinessTradeRow, PositionLotRow

        symbol_value = canonical_business_symbol(symbol)
        side_value = canonical_business_side(position_side)
        with self._session_factory() as session:
            rows = session.execute(
                select(PositionLotRow, BusinessTradeRow)
                .join(
                    BusinessTradeRow,
                    BusinessTradeRow.business_trade_id == PositionLotRow.business_trade_id,
                )
                .where(
                    PositionLotRow.account_id == str(account_id).strip(),
                    PositionLotRow.symbol == symbol_value,
                    PositionLotRow.position_side == side_value,
                    PositionLotRow.status.in_(
                        ("PARTIAL_OPEN", "OPEN", "PARTIAL_CLOSED", "ADOPTED")
                    ),
                )
                .order_by(BusinessTradeRow.business_trade_seq, PositionLotRow.lot_index)
            ).all()
        return tuple(
            (_identity_from_rows(trade, lot), str(lot.bucket), str(lot.open_quantity))
            for lot, trade in rows
        )
