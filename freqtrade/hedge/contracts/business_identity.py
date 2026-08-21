"""Canonical business-trade identity contracts for HEDGE."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from uuid import UUID


_SYMBOL_ALLOWED = re.compile(r"^[A-Za-z0-9/_:-]+$")


class BusinessOrderRole(StrEnum):
    ENTRY = "ENTRY"
    ENTRY_REPLACE = "ENTRY_REPLACE"
    INCREASE = "INCREASE"
    TAKE_PROFIT = "TAKE_PROFIT"
    STOP_LOSS = "STOP_LOSS"
    TRAILING_STOP = "TRAILING_STOP"
    REDUCE = "REDUCE"
    CLOSE = "CLOSE"
    UNSTUCK = "UNSTUCK"
    LIQUIDATION = "LIQUIDATION"
    RECONCILIATION_REPAIR = "RECONCILIATION_REPAIR"

    @property
    def reduces_risk(self) -> bool:
        return self in {
            BusinessOrderRole.TAKE_PROFIT,
            BusinessOrderRole.STOP_LOSS,
            BusinessOrderRole.TRAILING_STOP,
            BusinessOrderRole.REDUCE,
            BusinessOrderRole.CLOSE,
            BusinessOrderRole.UNSTUCK,
            BusinessOrderRole.LIQUIDATION,
            BusinessOrderRole.RECONCILIATION_REPAIR,
        }


class BusinessTradeStatus(StrEnum):
    PLANNED = "PLANNED"
    OPENING = "OPENING"
    PARTIALLY_OPEN = "PARTIALLY_OPEN"
    OPEN = "OPEN"
    REDUCING = "REDUCING"
    CLOSED = "CLOSED"
    CANCELED = "CANCELED"
    ABORTED = "ABORTED"
    RECOVERY_REQUIRED = "RECOVERY_REQUIRED"
    ERROR = "ERROR"


class BusinessLotStatus(StrEnum):
    PLANNED = "PLANNED"
    PARTIAL_OPEN = "PARTIAL_OPEN"
    OPEN = "OPEN"
    PARTIAL_CLOSED = "PARTIAL_CLOSED"
    CLOSED = "CLOSED"
    ADOPTED = "ADOPTED"
    RECOVERY_REQUIRED = "RECOVERY_REQUIRED"


def _required_text(value: object, *, name: str, max_length: int) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    result = value.strip()
    if not result or len(result) > max_length:
        raise ValueError(f"{name} is invalid")
    return result


def canonical_business_symbol(value: object) -> str:
    result = _required_text(value, name="symbol", max_length=128).upper()
    if not _SYMBOL_ALLOWED.fullmatch(result):
        raise ValueError("symbol contains unsupported characters")
    market = result.split(":", 1)[0]
    compact = re.sub(r"[/_-]", "", market)
    if not compact or not compact.isascii() or not compact.isalnum():
        raise ValueError("symbol must be ASCII alphanumeric after normalization")
    return compact


def canonical_business_side(value: object) -> str:
    side = str(getattr(value, "value", value)).strip().upper()
    if side not in {"LONG", "SHORT"}:
        raise ValueError("position_side is invalid")
    return side


def _uuid(value: object, *, name: str) -> UUID:
    if isinstance(value, UUID):
        return value
    try:
        return UUID(str(value))
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError(f"{name} must be UUID") from exc


@dataclass(frozen=True, slots=True)
class BusinessIdentity:
    business_trade_id: UUID
    business_trade_seq: int
    business_lot_id: UUID
    lot_index: int
    account_id: str
    symbol: str
    position_side: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "business_trade_id", _uuid(self.business_trade_id, name="business_trade_id")
        )
        object.__setattr__(
            self, "business_lot_id", _uuid(self.business_lot_id, name="business_lot_id")
        )
        if (
            isinstance(self.business_trade_seq, bool)
            or not isinstance(self.business_trade_seq, int)
            or self.business_trade_seq <= 0
        ):
            raise ValueError("business_trade_seq must be a positive integer")
        if (
            isinstance(self.lot_index, bool)
            or not isinstance(self.lot_index, int)
            or self.lot_index <= 0
        ):
            raise ValueError("lot_index must be a positive integer")
        object.__setattr__(
            self,
            "account_id",
            _required_text(self.account_id, name="account_id", max_length=128),
        )
        object.__setattr__(self, "symbol", canonical_business_symbol(self.symbol))
        object.__setattr__(self, "position_side", canonical_business_side(self.position_side))

    @property
    def display_id(self) -> str:
        side = "L" if self.position_side == "LONG" else "S"
        return f"{self.symbol}-{side}-{self.business_trade_seq:06d}"

    def assert_matches(
        self,
        *,
        account_id: str,
        symbol: str,
        position_side: object,
    ) -> None:
        if self.account_id != _required_text(account_id, name="account_id", max_length=128):
            raise ValueError("business identity account mismatch")
        if self.symbol != canonical_business_symbol(symbol):
            raise ValueError("business identity symbol mismatch")
        if self.position_side != canonical_business_side(position_side):
            raise ValueError("business identity position side mismatch")


def validate_order_role(
    role: BusinessOrderRole | str,
    *,
    action: object,
    reduce_only: bool,
) -> BusinessOrderRole:
    resolved = role if isinstance(role, BusinessOrderRole) else BusinessOrderRole(str(role).upper())
    action_value = str(getattr(action, "value", action)).upper()
    reducing_action = action_value in {"REDUCE", "CLOSE", "UNSTUCK"}
    if resolved.reduces_risk and not reduce_only:
        raise ValueError(f"{resolved.value} must be reduce_only")
    if resolved.reduces_risk and not reducing_action:
        raise ValueError(f"{resolved.value} requires REDUCE/CLOSE execution action")
    if resolved in {
        BusinessOrderRole.ENTRY,
        BusinessOrderRole.ENTRY_REPLACE,
        BusinessOrderRole.INCREASE,
    } and (reduce_only or reducing_action):
        raise ValueError(f"{resolved.value} cannot reduce risk")
    return resolved


def business_display_id(identity: BusinessIdentity) -> str:
    if not isinstance(identity, BusinessIdentity):
        raise TypeError("identity must be BusinessIdentity")
    return identity.display_id
