"""Transactional BusinessTrade/PositionLot projector for execution fills."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from sqlalchemy import select

from freqtrade.hedge.contracts.business_identity import BusinessOrderRole


class BusinessLotProjectionError(RuntimeError):
    pass


def _decimal(value: object) -> Decimal:
    result = Decimal(str(value))
    if not result.is_finite():
        raise BusinessLotProjectionError("lot projection received non-finite decimal")
    return result


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def apply_fill_to_business_lot(  # noqa: C901 - atomic fill/lot transaction boundary
    session,
    *,
    position_lot_model,
    business_trade_model,
    order,
    fill,
) -> object:
    """Apply one newly-created fill row inside the caller's SQL transaction."""

    intent = order.intent
    trade_id = getattr(intent, "business_trade_id", None)
    lot_id = getattr(intent, "business_lot_id", None)
    role = getattr(intent, "order_role", None)
    if trade_id is None or lot_id is None or role is None:
        raise BusinessLotProjectionError("managed fill has no business identity")

    for name in ("business_trade_id", "business_lot_id"):
        fill_value = getattr(fill, name, None)
        order_value = getattr(intent, name)
        if fill_value is None or str(fill_value) != str(order_value):
            raise BusinessLotProjectionError(f"fill/order {name} mismatch")
    fill_role = getattr(fill, "order_role", None)
    if fill_role is None or str(getattr(fill_role, "value", fill_role)) != str(
        getattr(role, "value", role)
    ):
        raise BusinessLotProjectionError("fill/order order_role mismatch")

    row = session.execute(
        select(position_lot_model)
        .where(position_lot_model.business_lot_id == str(lot_id))
        .with_for_update()
    ).scalar_one_or_none()
    if row is None:
        raise BusinessLotProjectionError("target business lot does not exist")
    trade = session.execute(
        select(business_trade_model)
        .where(business_trade_model.business_trade_id == str(trade_id))
        .with_for_update()
    ).scalar_one_or_none()
    if trade is None:
        raise BusinessLotProjectionError("target business trade does not exist")
    if str(row.business_trade_id) != str(trade_id):
        raise BusinessLotProjectionError("business lot belongs to another trade")

    qty = _decimal(fill.quantity)
    price = _decimal(fill.price)
    fee = _decimal(getattr(fill, "fee", 0))
    if qty <= 0 or price <= 0 or fee < 0:
        raise BusinessLotProjectionError("invalid fill quantity/price/fee")

    open_qty = _decimal(row.open_quantity)
    filled_entry = _decimal(row.entry_filled_quantity)
    closed_qty = _decimal(row.closed_quantity)
    entry_quote = _decimal(row.entry_quote)
    fees = _decimal(row.fees)
    resolved_role = (
        role if isinstance(role, BusinessOrderRole) else BusinessOrderRole(str(role).upper())
    )

    if resolved_role in {
        BusinessOrderRole.ENTRY,
        BusinessOrderRole.ENTRY_REPLACE,
        BusinessOrderRole.INCREASE,
    }:
        new_filled = filled_entry + qty
        new_quote = entry_quote + qty * price
        new_open = open_qty + qty
        row.entry_filled_quantity = format(new_filled, "f")
        row.entry_quote = format(new_quote, "f")
        row.open_quantity = format(new_open, "f")
        row.original_quantity = format(max(_decimal(row.original_quantity), new_filled), "f")
        row.average_entry_price = format(new_quote / new_filled, "f")
        row.status = "OPEN" if new_open == new_filled else "PARTIAL_OPEN"
        if row.opened_at is None:
            row.opened_at = getattr(fill, "exchange_time", None) or _now()
        trade.status = "OPEN" if new_open > 0 else "OPENING"
        if trade.opened_at is None:
            trade.opened_at = row.opened_at
    else:
        if not resolved_role.reduces_risk:
            raise BusinessLotProjectionError(f"unsupported business order role: {resolved_role}")
        if qty > open_qty:
            raise BusinessLotProjectionError("targeted reduce exceeds business lot open quantity")
        remaining = open_qty - qty
        row.open_quantity = format(remaining, "f")
        row.closed_quantity = format(closed_qty + qty, "f")
        row.status = "CLOSED" if remaining == 0 else "PARTIAL_CLOSED"
        trade.status = "CLOSED" if remaining == 0 else "REDUCING"
        if remaining == 0:
            closed_at = getattr(fill, "exchange_time", None) or _now()
            row.closed_at = closed_at
            trade.closed_at = closed_at

    row.fees = format(fees + fee, "f")
    return row
