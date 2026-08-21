"""Frozen cross-direction hedge identity and order semantics contracts."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation
from enum import Enum, StrEnum
from types import MappingProxyType
from typing import Any, cast
from uuid import UUID, uuid4

from .business_identity import BusinessIdentity, BusinessOrderRole
from .errors import HedgeContractError


_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_SYMBOL_ALLOWED = re.compile(r"^[A-Za-z0-9/_:-]+$")
_QUOTE_SUFFIXES = ("FDUSD", "USDT", "USDC", "BUSD", "TUSD", "USD", "BTC", "ETH", "EUR", "TRY")


class PositionSide(StrEnum):
    LONG = "LONG"
    SHORT = "SHORT"


class IntentAction(StrEnum):
    OPEN = "OPEN"
    INCREASE = "INCREASE"
    REDUCE = "REDUCE"
    CLOSE = "CLOSE"

    @property
    def reduces_risk(self) -> bool:
        return self in {IntentAction.REDUCE, IntentAction.CLOSE}


class OrderSide(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(StrEnum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class TimeInForce(StrEnum):
    GTC = "GTC"
    IOC = "IOC"
    FOK = "FOK"
    GTX = "GTX"


def required_text(value: object, *, field_name: str, max_length: int) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized or len(normalized) > max_length or _CONTROL.search(normalized):
        raise ValueError(f"{field_name} is invalid")
    return normalized


def finite_decimal(value: Decimal | str | int | float, *, field_name: str) -> Decimal:
    if isinstance(value, bool):
        # Floats were accepted by the frozen v1 contract; stringify to keep exact text.
        raise HedgeContractError(f"{field_name} must be a finite Decimal.")
    try:
        result = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise HedgeContractError(f"{field_name} must be a finite Decimal.") from exc
    if not result.is_finite():
        raise HedgeContractError(f"{field_name} must be a finite Decimal.")
    return result


def _canonical_pair(value: object) -> str:
    raw = (
        required_text(value, field_name="symbol", max_length=64)
        .upper()
        .replace("-", "/")
        .replace("_", "/")
    )
    if not _SYMBOL_ALLOWED.fullmatch(raw):
        raise ValueError("symbol contains unsupported characters")
    market, _, settle = raw.partition(":")
    if "/" in market:
        base, slash, quote = market.partition("/")
        if slash != "/" or not base or not quote or "/" in quote:
            raise ValueError("canonical_symbol must be a futures pair")
        settlement = settle or quote
        return f"{base}/{quote}:{settlement}"
    compact = re.sub(r"[/_-]", "", market)
    for quote in _QUOTE_SUFFIXES:
        if compact.endswith(quote) and len(compact) > len(quote):
            base = compact[: -len(quote)]
            return f"{base}/{quote}:{settle or quote}"
    if not compact or not compact.isascii() or not compact.isalnum():
        raise ValueError("symbol must contain ASCII alphanumeric characters")
    return compact


def canonical_symbol(value: object) -> str:
    """Return the frozen public Freqtrade futures-pair representation."""
    return _canonical_pair(value)


def _raw_symbol(value: object) -> str:
    canonical = _canonical_pair(value)
    market = canonical.split(":", 1)[0]
    return market.replace("/", "")


def expected_order_side(position_side: PositionSide, action: IntentAction) -> OrderSide:
    if position_side is PositionSide.LONG:
        return OrderSide.SELL if action.reduces_risk else OrderSide.BUY
    return OrderSide.BUY if action.reduces_risk else OrderSide.SELL


@dataclass(frozen=True, slots=True, order=True, init=False)
class PositionKey:
    exchange: str
    account_id: str
    symbol: str
    position_side: PositionSide
    _canonical_symbol: str = field(compare=False, repr=False)

    def __init__(
        self,
        exchange: str,
        account_id: str,
        symbol: str | None = None,
        position_side: PositionSide | str | None = None,
        *,
        canonical_symbol: str | None = None,
    ) -> None:
        if symbol is None:
            symbol = canonical_symbol
        elif canonical_symbol is not None and _raw_symbol(symbol) != _raw_symbol(canonical_symbol):
            raise ValueError("symbol and canonical_symbol identify different markets")
        if symbol is None:
            raise TypeError("symbol or canonical_symbol is required")
        exchange_value = required_text(exchange, field_name="exchange", max_length=64).lower()
        account_value = required_text(account_id, field_name="account_id", max_length=128)
        if position_side is None:
            raise TypeError("position_side is required")
        try:
            side = (
                position_side
                if isinstance(position_side, PositionSide)
                else PositionSide(str(position_side).upper())
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("position_side is invalid") from exc
        object.__setattr__(self, "exchange", exchange_value)
        object.__setattr__(self, "account_id", account_value)
        object.__setattr__(self, "symbol", _raw_symbol(symbol))
        object.__setattr__(self, "position_side", side)
        object.__setattr__(self, "_canonical_symbol", _canonical_pair(symbol))

    @property
    def canonical_symbol(self) -> str:
        return self._canonical_symbol

    @property
    def lock_name(self) -> str:
        return f"{self.exchange}:{self.account_id}:{self.symbol}:{self.position_side.value}"


@dataclass(frozen=True, slots=True)
class PositionRecord:
    key: PositionKey
    quantity: Decimal
    entry_price: Decimal
    observed_time_ms: int

    def __post_init__(self) -> None:
        if not isinstance(self.key, PositionKey):
            raise HedgeContractError("key must be a PositionKey")
        object.__setattr__(self, "quantity", finite_decimal(self.quantity, field_name="quantity"))
        object.__setattr__(
            self, "entry_price", finite_decimal(self.entry_price, field_name="entry_price")
        )
        if isinstance(self.observed_time_ms, bool) or int(self.observed_time_ms) < 0:
            raise HedgeContractError("observed_time_ms must be nonnegative")


@dataclass(frozen=True, slots=True)
class TargetPosition:
    key: PositionKey
    target_quantity: Decimal
    correlation_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "target_quantity",
            finite_decimal(self.target_quantity, field_name="target_quantity"),
        )


@dataclass(frozen=True, slots=True)
class OrderIntent:
    intent_id: str
    key: PositionKey
    quantity: Decimal
    idempotency_key: str
    correlation_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "quantity", finite_decimal(self.quantity, field_name="quantity"))


def _execution_text(
    value: object,
    *,
    field_name: str,
    max_length: int,
    optional: bool = False,
) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    result = value.strip()
    if not result:
        if optional:
            return None
        raise ValueError(f"{field_name} is required")
    if len(result) > max_length or _CONTROL.search(result):
        raise ValueError(f"{field_name} is invalid")
    return result


def _execution_decimal(value: object, *, field_name: str) -> Decimal:
    if isinstance(value, (bool, float)):
        raise ValueError(  # noqa: TRY004 - exact-decimal validation is a value-domain contract
            f"{field_name} must use an exact decimal value"
        )
    try:
        result = Decimal(value)  # type: ignore[arg-type]
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a valid decimal") from exc
    if not result.is_finite():
        raise ValueError(f"{field_name} must be finite")
    return result


def _execution_normalize_symbol(value: object) -> str:
    raw = _execution_text(value, field_name="symbol", max_length=64)
    if raw is None:  # pragma: no cover - required text cannot return None
        raise ValueError("symbol is required")
    raw = raw.upper()
    if not _SYMBOL_ALLOWED.fullmatch(raw):
        raise ValueError("symbol contains unsupported characters")
    parts = raw.split(":")
    if len(parts) > 2:
        raise ValueError("symbol contains multiple settlement suffixes")
    normalized = re.sub(r"[/_-]", "", parts[0])
    if not normalized or not normalized.isascii() or not normalized.isalnum():
        raise ValueError("symbol must contain ASCII alphanumeric characters")
    if len(parts) == 2:
        settle = parts[1]
        if not settle.isascii() or not settle.isalnum() or not normalized.endswith(settle):
            raise ValueError("settlement suffix must match the normalized quote asset")
    return normalized


def _execution_uuid(value: object, *, field_name: str, optional: bool = False) -> UUID | None:
    if value is None and optional:
        return None
    if isinstance(value, UUID):
        return value
    try:
        return UUID(str(value))
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError(f"{field_name} must be a UUID") from exc


def _execution_stable_sort_key(value: object) -> str:
    return f"{type(value).__qualname__}:{value!r}"


def _execution_freeze_value(  # noqa: C901 - recursive execution payload boundary
    value: object, *, depth: int = 0
) -> object:
    if depth > 12:
        raise ValueError("metadata nesting is too deep")
    if value is None or isinstance(value, (str, bool, int, Decimal, UUID, date, Enum)):
        if isinstance(value, str) and (len(value) > 65536 or _CONTROL.search(value)):
            raise ValueError("metadata string is invalid")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("metadata float must be finite")
        return value
    if isinstance(value, Mapping):
        if len(value) > 1000:
            raise ValueError("metadata mapping is too large")
        converted: dict[str, object] = {}
        for raw_key, raw_value in value.items():
            key = _execution_text(raw_key, field_name="metadata key", max_length=256)
            if key is None:  # pragma: no cover
                continue
            if key in converted:
                raise ValueError("metadata keys collide after normalization")
            converted[key] = _execution_freeze_value(raw_value, depth=depth + 1)
        return MappingProxyType(converted)
    if isinstance(value, (list, tuple)):
        if len(value) > 1000:
            raise ValueError("metadata sequence is too large")
        return tuple(_execution_freeze_value(item, depth=depth + 1) for item in value)
    if isinstance(value, (set, frozenset)):
        if len(value) > 1000:
            raise ValueError("metadata set is too large")
        items = [_execution_freeze_value(item, depth=depth + 1) for item in value]
        return tuple(sorted(items, key=_execution_stable_sort_key))
    raise TypeError(f"unsupported metadata value: {type(value).__name__}")


def _execution_freeze_mapping(value: object, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    frozen = _execution_freeze_value(value)
    if not isinstance(frozen, Mapping):  # pragma: no cover
        raise TypeError(f"{field_name} must be a mapping")
    return cast(Mapping[str, Any], frozen)


@dataclass(frozen=True, slots=True)
class ExecutionOrderIntent:
    """Execution order intent owned by the frozen contracts layer."""

    account_id: str
    symbol: str
    position_side: PositionSide
    action: IntentAction
    quantity: Decimal
    idempotency_key: str
    order_type: OrderType = OrderType.LIMIT
    limit_price: Decimal | None = None
    reduce_only: bool = False
    intent_id: UUID = field(default_factory=uuid4)
    action_group_id: UUID | None = None
    business_trade_id: UUID | None = None
    business_lot_id: UUID | None = None
    business_trade_seq: int | None = None
    lot_index: int | None = None
    order_role: BusinessOrderRole | None = None
    order_revision: int = 0
    submission_generation: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:  # noqa: C901 - canonical contract validation
        account_id = _execution_text(self.account_id, field_name="account_id", max_length=128)
        key = _execution_text(self.idempotency_key, field_name="idempotency_key", max_length=256)
        symbol = _execution_normalize_symbol(self.symbol)
        try:
            side = (
                self.position_side
                if isinstance(self.position_side, PositionSide)
                else PositionSide(self.position_side)
            )
            action = (
                self.action if isinstance(self.action, IntentAction) else IntentAction(self.action)
            )
            order_type = (
                self.order_type
                if isinstance(self.order_type, OrderType)
                else OrderType(self.order_type)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("position_side, action or order_type is invalid") from exc
        quantity = _execution_decimal(self.quantity, field_name="quantity")
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        limit_price = None
        if self.limit_price is not None:
            limit_price = _execution_decimal(self.limit_price, field_name="limit_price")
            if limit_price <= 0:
                raise ValueError("limit_price must be positive")
        if order_type is OrderType.LIMIT and limit_price is None:
            raise ValueError("LIMIT intent requires positive limit_price")
        if order_type is OrderType.MARKET and limit_price is not None:
            raise ValueError("MARKET intent must not include limit_price")
        if not isinstance(self.reduce_only, bool):
            raise TypeError("reduce_only must be a boolean")
        reduce_only = self.reduce_only
        if action.reduces_risk:
            reduce_only = True
        elif reduce_only:
            raise ValueError("risk-increasing intent cannot be reduce_only")
        intent_id = _execution_uuid(self.intent_id, field_name="intent_id")
        group_id = _execution_uuid(
            self.action_group_id, field_name="action_group_id", optional=True
        )
        business_trade_id = _execution_uuid(
            self.business_trade_id, field_name="business_trade_id", optional=True
        )
        business_lot_id = _execution_uuid(
            self.business_lot_id, field_name="business_lot_id", optional=True
        )
        identity_values = (
            business_trade_id,
            business_lot_id,
            self.business_trade_seq,
            self.lot_index,
            self.order_role,
        )
        if any(value is not None for value in identity_values) and not all(
            value is not None for value in identity_values
        ):
            raise ValueError("business identity fields must be supplied as one complete set")
        role = None
        if self.order_role is not None:
            role = (
                self.order_role
                if isinstance(self.order_role, BusinessOrderRole)
                else BusinessOrderRole(str(self.order_role).upper())
            )
            if isinstance(self.business_trade_seq, bool) or int(self.business_trade_seq) <= 0:
                raise ValueError("business_trade_seq must be positive")
            if isinstance(self.lot_index, bool) or int(self.lot_index) <= 0:
                raise ValueError("lot_index must be positive")
            if role.reduces_risk and not reduce_only:
                raise ValueError("risk-reducing business order role must be reduce_only")
            if role.reduces_risk != action.reduces_risk:
                raise ValueError("business order role conflicts with execution action")
        for name, value in (
            ("order_revision", self.order_revision),
            ("submission_generation", self.submission_generation),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        metadata = _execution_freeze_mapping(self.metadata, field_name="metadata")
        object.__setattr__(self, "account_id", account_id)
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "idempotency_key", key)
        object.__setattr__(self, "position_side", side)
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "order_type", order_type)
        object.__setattr__(self, "quantity", quantity)
        object.__setattr__(self, "limit_price", limit_price)
        object.__setattr__(self, "reduce_only", reduce_only)
        object.__setattr__(self, "intent_id", intent_id)
        object.__setattr__(self, "action_group_id", group_id)
        object.__setattr__(self, "business_trade_id", business_trade_id)
        object.__setattr__(self, "business_lot_id", business_lot_id)
        object.__setattr__(self, "business_trade_seq", self.business_trade_seq)
        object.__setattr__(self, "lot_index", self.lot_index)
        object.__setattr__(self, "order_role", role)
        object.__setattr__(self, "metadata", metadata)

    @property
    def business_identity(self) -> BusinessIdentity | None:
        if self.business_trade_id is None:
            return None
        return BusinessIdentity(
            business_trade_id=self.business_trade_id,
            business_trade_seq=int(self.business_trade_seq),
            business_lot_id=self.business_lot_id,
            lot_index=int(self.lot_index),
            account_id=self.account_id,
            symbol=self.symbol,
            position_side=self.position_side,
        )

    @property
    def reduces_risk(self) -> bool:
        return self.action.reduces_risk


@dataclass(frozen=True, slots=True)
class ApprovedOrderIntent:
    intent: OrderIntent
    approval_id: str
    approved_time_ms: int


@dataclass(frozen=True, slots=True)
class OrderSnapshot:
    order_id: str
    key: PositionKey
    status: str
    filled_quantity: Decimal
    exchange_time_ms: int
    observed_time_ms: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "filled_quantity",
            finite_decimal(self.filled_quantity, field_name="filled_quantity"),
        )


@dataclass(frozen=True, slots=True)
class AccountRiskSnapshot:
    account_id: str
    equity: Decimal
    margin_ratio: Decimal
    exchange_time_ms: int
    observed_time_ms: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "equity", finite_decimal(self.equity, field_name="equity"))
        object.__setattr__(
            self, "margin_ratio", finite_decimal(self.margin_ratio, field_name="margin_ratio")
        )


@dataclass(frozen=True, slots=True)
class ReconciliationDiff:
    key: PositionKey
    local_quantity: Decimal | None
    remote_quantity: Decimal | None
    reason: str

    def __post_init__(self) -> None:
        if self.local_quantity is not None:
            object.__setattr__(
                self,
                "local_quantity",
                finite_decimal(self.local_quantity, field_name="local_quantity"),
            )
        if self.remote_quantity is not None:
            object.__setattr__(
                self,
                "remote_quantity",
                finite_decimal(self.remote_quantity, field_name="remote_quantity"),
            )
