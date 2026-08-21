"""State machine for exact-lot take-profit, stop-loss and trailing protection."""

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal
from typing import Protocol
from uuid import UUID

from freqtrade.hedge.contracts.types import (
    ExecutionOrderIntent,
    IntentAction,
    OrderType,
    PositionSide,
)

from .contracts import (
    ZERO,
    BusinessLotProtectionSnapshot,
    ProtectionGroup,
    ProtectionGroupStatus,
    ProtectionIntegrityError,
    ProtectionKind,
    ProtectionLeg,
    ProtectionLegStatus,
    ProtectionQuantityMode,
    deterministic_protection_idempotency_key,
    deterministic_protection_intent_id,
    finite_decimal,
)


class ProtectionRepository(Protocol):
    def save_group(self, group: ProtectionGroup) -> None: ...

    def load_group(self, protection_group_id: UUID | str) -> ProtectionGroup: ...

    def groups_for_lot(
        self,
        business_lot_id: UUID | str,
        *,
        include_terminal: bool = False,
    ) -> tuple[ProtectionGroup, ...]: ...


class ProtectionExecutionPort(Protocol):
    def submit(self, intent: ExecutionOrderIntent) -> object: ...


@dataclass(frozen=True, slots=True)
class ProtectionTriggerDecision:
    group: ProtectionGroup
    leg: ProtectionLeg | None
    intent: ExecutionOrderIntent | None
    triggered: bool
    reason: str


class InMemoryProtectionRepository:
    def __init__(self) -> None:
        self._groups: dict[UUID, ProtectionGroup] = {}

    def save_group(self, group: ProtectionGroup) -> None:
        if not isinstance(group, ProtectionGroup):
            raise TypeError("group must be ProtectionGroup")
        self._groups[group.protection_group_id] = group

    def load_group(self, protection_group_id: UUID | str) -> ProtectionGroup:
        key = UUID(str(protection_group_id))
        try:
            return self._groups[key]
        except KeyError as exc:
            raise KeyError(str(key)) from exc

    def groups_for_lot(
        self,
        business_lot_id: UUID | str,
        *,
        include_terminal: bool = False,
    ) -> tuple[ProtectionGroup, ...]:
        key = UUID(str(business_lot_id))
        return tuple(
            group
            for group in self._groups.values()
            if group.business_lot_id == key and (include_terminal or not group.status.terminal)
        )


def _triggered(leg: ProtectionLeg, mark_price: Decimal) -> bool:
    side = leg.business_identity.position_side
    if leg.kind is ProtectionKind.TAKE_PROFIT:
        if leg.trigger_price is None:  # pragma: no cover
            raise ProtectionIntegrityError("take-profit trigger is missing")
        return (
            mark_price >= leg.trigger_price if side == "LONG" else mark_price <= leg.trigger_price
        )
    if leg.kind is ProtectionKind.STOP_LOSS:
        if leg.trigger_price is None:  # pragma: no cover
            raise ProtectionIntegrityError("stop-loss trigger is missing")
        return (
            mark_price <= leg.trigger_price if side == "LONG" else mark_price >= leg.trigger_price
        )
    if side == "LONG":
        if leg.high_watermark is None or leg.trailing_distance is None:
            return False
        return mark_price <= leg.high_watermark - leg.trailing_distance
    if leg.low_watermark is None or leg.trailing_distance is None:
        return False
    return mark_price >= leg.low_watermark + leg.trailing_distance


def _trigger_priority(leg: ProtectionLeg) -> tuple[int, str, str]:
    priority = {
        ProtectionKind.STOP_LOSS: 0,
        ProtectionKind.TRAILING_STOP: 1,
        ProtectionKind.TAKE_PROFIT: 2,
    }[leg.kind]
    return priority, leg.label, str(leg.protection_id)


def _intent_for_leg(
    leg: ProtectionLeg,
    *,
    trigger_quantity: Decimal,
) -> ExecutionOrderIntent:
    identity = leg.business_identity
    action = (
        IntentAction.CLOSE
        if leg.quantity_mode is ProtectionQuantityMode.REMAINING
        else IntentAction.REDUCE
    )
    intent_id = deterministic_protection_intent_id(leg.protection_id, leg.revision)
    return ExecutionOrderIntent(
        account_id=identity.account_id,
        symbol=identity.symbol,
        position_side=PositionSide(identity.position_side),
        action=action,
        quantity=trigger_quantity,
        idempotency_key=deterministic_protection_idempotency_key(
            leg.protection_id,
            leg.revision,
        ),
        order_type=OrderType.MARKET,
        reduce_only=True,
        intent_id=intent_id,
        business_trade_id=identity.business_trade_id,
        business_lot_id=identity.business_lot_id,
        business_trade_seq=identity.business_trade_seq,
        lot_index=identity.lot_index,
        order_role=leg.order_role,
        order_revision=leg.revision,
        metadata={
            "protection_group_id": str(leg.protection_group_id),
            "protection_id": str(leg.protection_id),
            "protection_kind": leg.kind.value,
            "protection_label": leg.label,
            "business_display_id": identity.display_id,
        },
    )


def _active_leg(group: ProtectionGroup) -> ProtectionLeg | None:
    active = tuple(leg for leg in group.legs if leg.active_execution)
    if len(active) > 1:
        raise ProtectionIntegrityError("multiple protection executions are active for one lot")
    return None if not active else active[0]


class BusinessProtectionService:
    """Coordinate one exact business lot without spillover to sibling lots."""

    def __init__(self, repository: ProtectionRepository) -> None:
        self._repository = repository

    def arm(self, group: ProtectionGroup) -> ProtectionGroup:
        if not isinstance(group, ProtectionGroup):
            raise TypeError("group must be ProtectionGroup")
        existing = self._repository.groups_for_lot(group.business_lot_id)
        if existing:
            raise ProtectionIntegrityError("business lot already has an active protection group")
        self._repository.save_group(group)
        return group

    def evaluate(
        self,
        protection_group_id: UUID | str,
        *,
        lot: BusinessLotProtectionSnapshot,
        mark_price: Decimal | str | int,
    ) -> ProtectionTriggerDecision:
        group = self._repository.load_group(protection_group_id)
        self._assert_lot(group, lot)
        if group.status.terminal:
            return ProtectionTriggerDecision(group, None, None, False, "group is terminal")
        if lot.open_quantity == ZERO:
            closed = self._close_group(group)
            self._repository.save_group(closed)
            return ProtectionTriggerDecision(closed, None, None, False, "business lot is closed")

        mark = finite_decimal(mark_price, name="mark_price")
        if mark <= ZERO:
            raise ValueError("mark_price must be positive")

        active = _active_leg(group)
        if active is not None:
            if active.status is ProtectionLegStatus.TRIGGERED:
                intent = self._resume_triggered_intent(active, lot)
                return ProtectionTriggerDecision(
                    group,
                    active,
                    intent,
                    True,
                    "resume durable triggered protection",
                )
            return ProtectionTriggerDecision(
                group,
                active,
                None,
                False,
                "protection execution already in flight",
            )

        refreshed_legs = tuple(leg.with_watermark(mark) for leg in group.legs)
        refreshed = replace(group, legs=refreshed_legs)
        candidates = sorted(
            (
                leg
                for leg in refreshed.legs
                if leg.status is ProtectionLegStatus.ARMED and _triggered(leg, mark)
            ),
            key=_trigger_priority,
        )
        if not candidates:
            if refreshed != group:
                self._repository.save_group(refreshed)
            return ProtectionTriggerDecision(refreshed, None, None, False, "no trigger")

        selected = candidates[0]
        trigger_quantity = self._resolve_trigger_quantity(selected, lot)
        intent_id = deterministic_protection_intent_id(selected.protection_id, selected.revision)
        triggered_leg = replace(
            selected,
            status=ProtectionLegStatus.TRIGGERED,
            execution_intent_id=intent_id,
            trigger_quantity=trigger_quantity,
        )
        executing = refreshed.replace_leg(
            triggered_leg,
            status=ProtectionGroupStatus.EXECUTING,
            revision=refreshed.revision + 1,
        )
        self._repository.save_group(executing)
        return ProtectionTriggerDecision(
            executing,
            triggered_leg,
            _intent_for_leg(triggered_leg, trigger_quantity=trigger_quantity),
            True,
            f"{triggered_leg.kind.value}:{triggered_leg.label}",
        )

    def mark_submitted(
        self,
        protection_group_id: UUID | str,
        *,
        protection_id: UUID | str,
        client_order_id: str,
        exchange_order_id: str | None = None,
    ) -> ProtectionGroup:
        group = self._repository.load_group(protection_group_id)
        leg = group.leg(protection_id)
        if leg.status is not ProtectionLegStatus.TRIGGERED:
            raise ProtectionIntegrityError("only TRIGGERED protection can be marked submitted")
        submitted = replace(
            leg,
            status=ProtectionLegStatus.SUBMITTED,
            client_order_id=client_order_id,
            exchange_order_id=exchange_order_id,
        )
        updated = group.replace_leg(submitted)
        self._repository.save_group(updated)
        return updated

    def record_fill(
        self,
        protection_group_id: UUID | str,
        *,
        protection_id: UUID | str,
        cumulative_filled_quantity: Decimal | str | int,
        lot_open_quantity: Decimal | str | int,
        terminal: bool,
    ) -> ProtectionGroup:
        group = self._repository.load_group(protection_group_id)
        leg = group.leg(protection_id)
        if leg.status not in {
            ProtectionLegStatus.SUBMITTED,
            ProtectionLegStatus.PARTIAL,
        }:
            raise ProtectionIntegrityError("fill targets a protection leg that is not submitted")
        if leg.trigger_quantity is None:  # pragma: no cover
            raise ProtectionIntegrityError("submitted protection has no trigger quantity")
        filled = finite_decimal(cumulative_filled_quantity, name="cumulative_filled_quantity")
        open_quantity = finite_decimal(lot_open_quantity, name="lot_open_quantity")
        if filled < ZERO or filled > leg.trigger_quantity:
            raise ProtectionIntegrityError("protection fill quantity is inconsistent")
        if open_quantity < ZERO:
            raise ProtectionIntegrityError("business lot open quantity cannot be negative")
        if filled < leg.filled_quantity:
            raise ProtectionIntegrityError("protection cumulative fill moved backwards")

        if terminal and filled != leg.trigger_quantity:
            raise ProtectionIntegrityError(
                "terminal protection order did not fill its exact trigger quantity"
            )
        new_status = ProtectionLegStatus.FILLED if terminal else ProtectionLegStatus.PARTIAL
        updated_leg = replace(leg, status=new_status, filled_quantity=filled)
        if open_quantity == ZERO:
            legs = tuple(
                updated_leg
                if current.protection_id == updated_leg.protection_id
                else (
                    current
                    if current.status in {ProtectionLegStatus.FILLED, ProtectionLegStatus.FAILED}
                    else replace(current, status=ProtectionLegStatus.CANCELED)
                )
                for current in group.legs
            )
            updated_group = replace(
                group,
                legs=legs,
                status=ProtectionGroupStatus.CLOSED,
                revision=group.revision + 1,
            )
        elif terminal:
            updated_group = group.replace_leg(
                updated_leg,
                status=ProtectionGroupStatus.ACTIVE,
                revision=group.revision + 1,
            )
        else:
            updated_group = group.replace_leg(updated_leg)
        self._repository.save_group(updated_group)
        return updated_group

    def mark_failed(
        self,
        protection_group_id: UUID | str,
        *,
        protection_id: UUID | str,
        error: str,
    ) -> ProtectionGroup:
        group = self._repository.load_group(protection_group_id)
        leg = group.leg(protection_id)
        failed = replace(
            leg,
            status=ProtectionLegStatus.FAILED,
            last_error=error,
        )
        updated = group.replace_leg(
            failed,
            status=ProtectionGroupStatus.ERROR,
            revision=group.revision + 1,
        )
        self._repository.save_group(updated)
        return updated

    def cancel_group(self, protection_group_id: UUID | str) -> ProtectionGroup:
        group = self._repository.load_group(protection_group_id)
        if group.status is ProtectionGroupStatus.CLOSED:
            return group
        legs = tuple(
            leg if leg.status.terminal else replace(leg, status=ProtectionLegStatus.CANCELED)
            for leg in group.legs
        )
        updated = replace(
            group,
            legs=legs,
            status=ProtectionGroupStatus.CANCELED,
            revision=group.revision + 1,
        )
        self._repository.save_group(updated)
        return updated

    def amend_armed_leg(
        self,
        protection_group_id: UUID | str,
        *,
        protection_id: UUID | str,
        trigger_price: Decimal | str | int | None = None,
        trailing_distance: Decimal | str | int | None = None,
        quantity: Decimal | str | int | None = None,
    ) -> ProtectionGroup:
        group = self._repository.load_group(protection_group_id)
        leg = group.leg(protection_id)
        if leg.status is not ProtectionLegStatus.ARMED:
            raise ProtectionIntegrityError("only ARMED protection can be amended in place")
        changes: dict[str, object] = {"revision": leg.revision + 1}
        if leg.kind is ProtectionKind.TRAILING_STOP:
            if trailing_distance is None:
                raise ValueError("trailing_distance is required")
            changes["trailing_distance"] = finite_decimal(
                trailing_distance,
                name="trailing_distance",
            )
        else:
            if trigger_price is None:
                raise ValueError("trigger_price is required")
            changes["trigger_price"] = finite_decimal(trigger_price, name="trigger_price")
        if leg.quantity_mode is ProtectionQuantityMode.ABSOLUTE:
            if quantity is None:
                raise ValueError("quantity is required for ABSOLUTE protection")
            changes["quantity"] = finite_decimal(quantity, name="quantity")
        amended = replace(leg, **changes)
        updated = replace(
            group.replace_leg(amended),
            revision=group.revision + 1,
        )
        self._repository.save_group(updated)
        return updated

    @staticmethod
    def _assert_lot(
        group: ProtectionGroup,
        lot: BusinessLotProtectionSnapshot,
    ) -> None:
        if group.business_identity != lot.identity:
            raise ProtectionIntegrityError(
                "protection group cannot target a different business lot"
            )

    @staticmethod
    def _resolve_trigger_quantity(
        leg: ProtectionLeg,
        lot: BusinessLotProtectionSnapshot,
    ) -> Decimal:
        quantity = (
            lot.open_quantity
            if leg.quantity_mode is ProtectionQuantityMode.REMAINING
            else leg.quantity
        )
        if quantity is None:  # pragma: no cover
            raise ProtectionIntegrityError("protection quantity is missing")
        if quantity <= ZERO:
            raise ProtectionIntegrityError("cannot trigger protection for a closed business lot")
        if quantity > lot.open_quantity:
            raise ProtectionIntegrityError(
                "targeted protection quantity exceeds business lot open quantity"
            )
        return quantity

    @staticmethod
    def _resume_triggered_intent(
        leg: ProtectionLeg,
        lot: BusinessLotProtectionSnapshot,
    ) -> ExecutionOrderIntent:
        if leg.trigger_quantity is None or leg.execution_intent_id is None:
            raise ProtectionIntegrityError("durable triggered state is incomplete")
        if leg.trigger_quantity > lot.open_quantity:
            raise ProtectionIntegrityError(
                "triggered protection quantity now exceeds business lot open quantity; "
                "reconcile first"
            )
        return _intent_for_leg(leg, trigger_quantity=leg.trigger_quantity)

    @staticmethod
    def _close_group(group: ProtectionGroup) -> ProtectionGroup:
        legs = tuple(
            leg
            if leg.status in {ProtectionLegStatus.FILLED, ProtectionLegStatus.FAILED}
            else replace(leg, status=ProtectionLegStatus.CANCELED)
            for leg in group.legs
        )
        return replace(
            group,
            legs=legs,
            status=ProtectionGroupStatus.CLOSED,
            revision=group.revision + 1,
        )
