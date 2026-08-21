"""Exact business-lot protective-order contracts for HEDGE."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from freqtrade.hedge.contracts.business_identity import BusinessIdentity, BusinessOrderRole


ZERO = Decimal(0)
ONE = Decimal(1)


class ProtectionIntegrityError(RuntimeError):
    """Raised when protection state cannot safely target one exact business lot."""


class ProtectionKind(StrEnum):
    TAKE_PROFIT = "TAKE_PROFIT"
    STOP_LOSS = "STOP_LOSS"
    TRAILING_STOP = "TRAILING_STOP"

    @property
    def order_role(self) -> BusinessOrderRole:
        return BusinessOrderRole(self.value)


class ProtectionQuantityMode(StrEnum):
    ABSOLUTE = "ABSOLUTE"
    REMAINING = "REMAINING"


class ProtectionLegStatus(StrEnum):
    ARMED = "ARMED"
    TRIGGERED = "TRIGGERED"
    SUBMITTED = "SUBMITTED"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    FAILED = "FAILED"

    @property
    def terminal(self) -> bool:
        return self in {
            ProtectionLegStatus.FILLED,
            ProtectionLegStatus.CANCELED,
            ProtectionLegStatus.FAILED,
        }


class ProtectionGroupStatus(StrEnum):
    ACTIVE = "ACTIVE"
    EXECUTING = "EXECUTING"
    CLOSED = "CLOSED"
    CANCELED = "CANCELED"
    ERROR = "ERROR"

    @property
    def terminal(self) -> bool:
        return self in {
            ProtectionGroupStatus.CLOSED,
            ProtectionGroupStatus.CANCELED,
            ProtectionGroupStatus.ERROR,
        }


def finite_decimal(value: object, *, name: str) -> Decimal:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a decimal value")
    try:
        result = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite decimal") from exc
    if not result.is_finite():
        raise ValueError(f"{name} must be a finite decimal")
    return result


def required_text(value: object, *, name: str, limit: int = 128) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    result = value.strip()
    if not result or len(result) > limit:
        raise ValueError(f"{name} is invalid")
    return result


def optional_text(value: object | None, *, name: str, limit: int = 256) -> str | None:
    if value is None:
        return None
    return required_text(value, name=name, limit=limit)


def as_uuid(value: object, *, name: str) -> UUID:
    if isinstance(value, UUID):
        return value
    try:
        return UUID(str(value))
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError(f"{name} must be UUID") from exc


def deterministic_protection_intent_id(protection_id: UUID, revision: int) -> UUID:
    return uuid5(NAMESPACE_URL, f"freqtrade-hedge-protection:{protection_id}:{revision}")


def deterministic_protection_idempotency_key(protection_id: UUID, revision: int) -> str:
    return f"hedge-protection:{protection_id}:{revision}"


@dataclass(frozen=True, slots=True)
class BusinessLotProtectionSnapshot:
    identity: BusinessIdentity
    open_quantity: Decimal
    average_entry_price: Decimal

    def __post_init__(self) -> None:
        if not isinstance(self.identity, BusinessIdentity):
            raise TypeError("identity must be BusinessIdentity")
        open_quantity = finite_decimal(self.open_quantity, name="open_quantity")
        average_entry_price = finite_decimal(
            self.average_entry_price,
            name="average_entry_price",
        )
        if open_quantity < ZERO:
            raise ValueError("open_quantity must be nonnegative")
        if open_quantity > ZERO and average_entry_price <= ZERO:
            raise ValueError("open lot requires positive average_entry_price")
        if open_quantity == ZERO and average_entry_price < ZERO:
            raise ValueError("average_entry_price must be nonnegative")
        object.__setattr__(self, "open_quantity", open_quantity)
        object.__setattr__(self, "average_entry_price", average_entry_price)


@dataclass(frozen=True, slots=True)
class ProtectionLeg:
    protection_group_id: UUID
    business_identity: BusinessIdentity
    kind: ProtectionKind
    label: str
    quantity_mode: ProtectionQuantityMode
    quantity: Decimal | None = None
    trigger_price: Decimal | None = None
    trailing_distance: Decimal | None = None
    protection_id: UUID = field(default_factory=uuid4)
    status: ProtectionLegStatus = ProtectionLegStatus.ARMED
    revision: int = 0
    high_watermark: Decimal | None = None
    low_watermark: Decimal | None = None
    execution_intent_id: UUID | None = None
    trigger_quantity: Decimal | None = None
    client_order_id: str | None = None
    exchange_order_id: str | None = None
    filled_quantity: Decimal = ZERO
    last_error: str | None = None

    def __post_init__(self) -> None:  # noqa: C901 - invariant boundary
        group_id = as_uuid(self.protection_group_id, name="protection_group_id")
        protection_id = as_uuid(self.protection_id, name="protection_id")
        if not isinstance(self.business_identity, BusinessIdentity):
            raise TypeError("business_identity must be BusinessIdentity")
        kind = (
            self.kind if isinstance(self.kind, ProtectionKind) else ProtectionKind(str(self.kind))
        )
        quantity_mode = (
            self.quantity_mode
            if isinstance(self.quantity_mode, ProtectionQuantityMode)
            else ProtectionQuantityMode(str(self.quantity_mode))
        )
        status = (
            self.status
            if isinstance(self.status, ProtectionLegStatus)
            else ProtectionLegStatus(str(self.status))
        )
        label = required_text(self.label, name="label", limit=64)
        if isinstance(self.revision, bool) or not isinstance(self.revision, int):
            raise TypeError("revision must be an integer")
        if self.revision < 0:
            raise ValueError("revision must be nonnegative")

        quantity = None
        if self.quantity is not None:
            quantity = finite_decimal(self.quantity, name="quantity")
            if quantity <= ZERO:
                raise ValueError("quantity must be positive")
        if quantity_mode is ProtectionQuantityMode.ABSOLUTE and quantity is None:
            raise ValueError("ABSOLUTE protection requires quantity")
        if quantity_mode is ProtectionQuantityMode.REMAINING and quantity is not None:
            raise ValueError("REMAINING protection must not store a fixed quantity")

        trigger_price = None
        if self.trigger_price is not None:
            trigger_price = finite_decimal(self.trigger_price, name="trigger_price")
            if trigger_price <= ZERO:
                raise ValueError("trigger_price must be positive")
        trailing_distance = None
        if self.trailing_distance is not None:
            trailing_distance = finite_decimal(self.trailing_distance, name="trailing_distance")
            if trailing_distance <= ZERO:
                raise ValueError("trailing_distance must be positive")

        if kind in {ProtectionKind.TAKE_PROFIT, ProtectionKind.STOP_LOSS}:
            if trigger_price is None or trailing_distance is not None:
                raise ValueError(f"{kind.value} requires trigger_price only")
        elif trigger_price is not None or trailing_distance is None:
            raise ValueError("TRAILING_STOP requires trailing_distance only")

        high = (
            None
            if self.high_watermark is None
            else finite_decimal(
                self.high_watermark,
                name="high_watermark",
            )
        )
        low = (
            None
            if self.low_watermark is None
            else finite_decimal(
                self.low_watermark,
                name="low_watermark",
            )
        )
        if high is not None and high <= ZERO:
            raise ValueError("high_watermark must be positive")
        if low is not None and low <= ZERO:
            raise ValueError("low_watermark must be positive")

        intent_id = (
            None
            if self.execution_intent_id is None
            else as_uuid(self.execution_intent_id, name="execution_intent_id")
        )
        trigger_quantity = None
        if self.trigger_quantity is not None:
            trigger_quantity = finite_decimal(self.trigger_quantity, name="trigger_quantity")
            if trigger_quantity <= ZERO:
                raise ValueError("trigger_quantity must be positive")
        filled = finite_decimal(self.filled_quantity, name="filled_quantity")
        if filled < ZERO:
            raise ValueError("filled_quantity must be nonnegative")
        if trigger_quantity is not None and filled > trigger_quantity:
            raise ValueError("filled_quantity exceeds trigger_quantity")

        if status in {
            ProtectionLegStatus.TRIGGERED,
            ProtectionLegStatus.SUBMITTED,
            ProtectionLegStatus.PARTIAL,
            ProtectionLegStatus.FILLED,
        } and (intent_id is None or trigger_quantity is None):
            raise ValueError("triggered protection state requires durable intent and quantity")
        if (
            status in {ProtectionLegStatus.SUBMITTED, ProtectionLegStatus.PARTIAL}
            and not self.client_order_id
        ):
            raise ValueError("submitted protection state requires client_order_id")
        if (
            status is ProtectionLegStatus.FILLED
            and trigger_quantity is not None
            and filled != trigger_quantity
        ):
            raise ValueError("FILLED protection must fill its trigger quantity")

        object.__setattr__(self, "protection_group_id", group_id)
        object.__setattr__(self, "protection_id", protection_id)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "quantity_mode", quantity_mode)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "quantity", quantity)
        object.__setattr__(self, "trigger_price", trigger_price)
        object.__setattr__(self, "trailing_distance", trailing_distance)
        object.__setattr__(self, "high_watermark", high)
        object.__setattr__(self, "low_watermark", low)
        object.__setattr__(self, "execution_intent_id", intent_id)
        object.__setattr__(self, "trigger_quantity", trigger_quantity)
        object.__setattr__(
            self,
            "client_order_id",
            optional_text(self.client_order_id, name="client_order_id", limit=256),
        )
        object.__setattr__(
            self,
            "exchange_order_id",
            optional_text(self.exchange_order_id, name="exchange_order_id", limit=256),
        )
        object.__setattr__(
            self,
            "last_error",
            optional_text(self.last_error, name="last_error", limit=1024),
        )
        object.__setattr__(self, "filled_quantity", filled)

    @property
    def order_role(self) -> BusinessOrderRole:
        return self.kind.order_role

    @property
    def active_execution(self) -> bool:
        return self.status in {
            ProtectionLegStatus.TRIGGERED,
            ProtectionLegStatus.SUBMITTED,
            ProtectionLegStatus.PARTIAL,
        }

    def with_watermark(self, mark_price: Decimal) -> ProtectionLeg:
        if (
            self.kind is not ProtectionKind.TRAILING_STOP
            or self.status is not ProtectionLegStatus.ARMED
        ):
            return self
        mark = finite_decimal(mark_price, name="mark_price")
        if mark <= ZERO:
            raise ValueError("mark_price must be positive")
        if self.business_identity.position_side == "LONG":
            high = mark if self.high_watermark is None else max(self.high_watermark, mark)
            return replace(self, high_watermark=high)
        low = mark if self.low_watermark is None else min(self.low_watermark, mark)
        return replace(self, low_watermark=low)


@dataclass(frozen=True, slots=True)
class ProtectionGroup:
    business_identity: BusinessIdentity
    legs: tuple[ProtectionLeg, ...]
    protection_group_id: UUID = field(default_factory=uuid4)
    status: ProtectionGroupStatus = ProtectionGroupStatus.ACTIVE
    revision: int = 0
    require_stop: bool = True

    def __post_init__(self) -> None:  # noqa: C901 - invariant boundary
        if not isinstance(self.business_identity, BusinessIdentity):
            raise TypeError("business_identity must be BusinessIdentity")
        group_id = as_uuid(self.protection_group_id, name="protection_group_id")
        status = (
            self.status
            if isinstance(self.status, ProtectionGroupStatus)
            else ProtectionGroupStatus(str(self.status))
        )
        if isinstance(self.revision, bool) or not isinstance(self.revision, int):
            raise TypeError("revision must be an integer")
        if self.revision < 0:
            raise ValueError("revision must be nonnegative")
        if not isinstance(self.require_stop, bool):
            raise TypeError("require_stop must be a boolean")
        legs = tuple(self.legs)
        if not legs:
            raise ValueError("protection group requires at least one leg")
        protection_ids: set[UUID] = set()
        labels: set[str] = set()
        active_execution = 0
        has_stop = False
        for leg in legs:
            if not isinstance(leg, ProtectionLeg):
                raise TypeError("legs must contain ProtectionLeg")
            if leg.protection_group_id != group_id:
                raise ValueError("protection leg belongs to another group")
            if leg.business_identity != self.business_identity:
                raise ValueError("protection leg business identity mismatch")
            if leg.protection_id in protection_ids:
                raise ValueError("protection ids must be unique inside a group")
            if leg.label in labels:
                raise ValueError("protection labels must be unique inside a group")
            protection_ids.add(leg.protection_id)
            labels.add(leg.label)
            active_execution += int(leg.active_execution)
            has_stop = has_stop or leg.kind in {
                ProtectionKind.STOP_LOSS,
                ProtectionKind.TRAILING_STOP,
            }
        if active_execution > 1:
            raise ProtectionIntegrityError(
                "one business lot cannot have multiple protection executions in flight"
            )
        if self.require_stop and not has_stop:
            raise ValueError("protected business lot requires STOP_LOSS or TRAILING_STOP")
        if status is ProtectionGroupStatus.EXECUTING and active_execution != 1:
            raise ValueError("EXECUTING group requires exactly one active protection execution")
        if status is ProtectionGroupStatus.ACTIVE and active_execution:
            raise ValueError("ACTIVE group cannot contain an in-flight protection execution")
        object.__setattr__(self, "protection_group_id", group_id)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "legs", legs)

    @property
    def business_lot_id(self) -> UUID:
        return self.business_identity.business_lot_id

    @property
    def display_id(self) -> str:
        return self.business_identity.display_id

    def leg(self, protection_id: UUID | str) -> ProtectionLeg:
        key = as_uuid(protection_id, name="protection_id")
        for leg in self.legs:
            if leg.protection_id == key:
                return leg
        raise KeyError(str(key))

    def replace_leg(
        self,
        updated: ProtectionLeg,
        *,
        status: ProtectionGroupStatus | None = None,
        revision: int | None = None,
    ) -> ProtectionGroup:
        """Replace one leg and, when needed, its group state atomically."""
        if updated.protection_group_id != self.protection_group_id:
            raise ValueError("replacement leg belongs to another protection group")
        replaced = False
        legs: list[ProtectionLeg] = []
        for current in self.legs:
            if current.protection_id == updated.protection_id:
                legs.append(updated)
                replaced = True
            else:
                legs.append(current)
        if not replaced:
            raise KeyError(str(updated.protection_id))
        return replace(
            self,
            legs=tuple(legs),
            status=self.status if status is None else status,
            revision=self.revision if revision is None else revision,
        )


def make_protection_leg(
    *,
    protection_group_id: UUID,
    business_identity: BusinessIdentity,
    kind: ProtectionKind,
    label: str,
    trigger_price: Decimal | str | int | None = None,
    trailing_distance: Decimal | str | int | None = None,
    quantity: Decimal | str | int | None = None,
    quantity_mode: ProtectionQuantityMode | None = None,
) -> ProtectionLeg:
    resolved_kind = kind if isinstance(kind, ProtectionKind) else ProtectionKind(str(kind))
    mode = quantity_mode
    if mode is None:
        mode = (
            ProtectionQuantityMode.ABSOLUTE
            if resolved_kind is ProtectionKind.TAKE_PROFIT
            else ProtectionQuantityMode.REMAINING
        )
    fixed = None if quantity is None else finite_decimal(quantity, name="quantity")
    trigger = None if trigger_price is None else finite_decimal(trigger_price, name="trigger_price")
    trailing = (
        None
        if trailing_distance is None
        else finite_decimal(trailing_distance, name="trailing_distance")
    )
    return ProtectionLeg(
        protection_group_id=protection_group_id,
        business_identity=business_identity,
        kind=resolved_kind,
        label=label,
        quantity_mode=mode,
        quantity=fixed,
        trigger_price=trigger,
        trailing_distance=trailing,
    )


def build_protection_group(
    *,
    lot: BusinessLotProtectionSnapshot,
    take_profits: tuple[tuple[str, Decimal | str | int, Decimal | str | int], ...] = (),
    stop_loss: Decimal | str | int | None = None,
    trailing_distance: Decimal | str | int | None = None,
    require_stop: bool = True,
) -> ProtectionGroup:
    if not isinstance(lot, BusinessLotProtectionSnapshot):
        raise TypeError("lot must be BusinessLotProtectionSnapshot")
    if lot.open_quantity <= ZERO:
        raise ProtectionIntegrityError("cannot protect a closed business lot")
    group_id = uuid4()
    legs: list[ProtectionLeg] = []
    for label, trigger, quantity in take_profits:
        legs.append(
            make_protection_leg(
                protection_group_id=group_id,
                business_identity=lot.identity,
                kind=ProtectionKind.TAKE_PROFIT,
                label=label,
                trigger_price=trigger,
                quantity=quantity,
                quantity_mode=ProtectionQuantityMode.ABSOLUTE,
            )
        )
    if stop_loss is not None:
        legs.append(
            make_protection_leg(
                protection_group_id=group_id,
                business_identity=lot.identity,
                kind=ProtectionKind.STOP_LOSS,
                label="SL",
                trigger_price=stop_loss,
                quantity_mode=ProtectionQuantityMode.REMAINING,
            )
        )
    if trailing_distance is not None:
        legs.append(
            make_protection_leg(
                protection_group_id=group_id,
                business_identity=lot.identity,
                kind=ProtectionKind.TRAILING_STOP,
                label="TRAIL",
                trailing_distance=trailing_distance,
                quantity_mode=ProtectionQuantityMode.REMAINING,
            )
        )
    group = ProtectionGroup(
        business_identity=lot.identity,
        protection_group_id=group_id,
        legs=tuple(legs),
        require_stop=require_stop,
    )
    validate_trigger_geometry(group, lot)
    fixed_tp = sum(
        (leg.quantity or ZERO) for leg in group.legs if leg.kind is ProtectionKind.TAKE_PROFIT
    )
    if fixed_tp > lot.open_quantity:
        raise ProtectionIntegrityError(
            "sum of absolute take-profit quantities exceeds business lot open quantity"
        )
    return group


def validate_trigger_geometry(
    group: ProtectionGroup,
    lot: BusinessLotProtectionSnapshot,
) -> None:
    if group.business_identity != lot.identity:
        raise ProtectionIntegrityError("protection group and lot identity differ")
    entry = lot.average_entry_price
    side = lot.identity.position_side
    for leg in group.legs:
        if leg.kind is ProtectionKind.TRAILING_STOP:
            continue
        trigger = leg.trigger_price
        if trigger is None:  # pragma: no cover - enforced by ProtectionLeg
            raise ProtectionIntegrityError("trigger price is missing")
        if side == "LONG":
            if leg.kind is ProtectionKind.TAKE_PROFIT and trigger <= entry:
                raise ValueError("LONG take-profit trigger must be above entry price")
            if leg.kind is ProtectionKind.STOP_LOSS and trigger >= entry:
                raise ValueError("LONG stop-loss trigger must be below entry price")
        else:
            if leg.kind is ProtectionKind.TAKE_PROFIT and trigger >= entry:
                raise ValueError("SHORT take-profit trigger must be below entry price")
            if leg.kind is ProtectionKind.STOP_LOSS and trigger <= entry:
                raise ValueError("SHORT stop-loss trigger must be above entry price")
