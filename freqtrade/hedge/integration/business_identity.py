"""Planner -> Execution business-identity binding boundary.

This module is the only place allowed to allocate a new HEDGE business trade for a
planner order.  It runs before the canonical Planner -> Execution adapter and before
exchange submission.  Replacement orders inherit the durable identity of the active
order they replace; targeted reductions must resolve an existing open lot.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol, cast

from freqtrade.hedge.contracts.business_identity import (
    BusinessIdentity,
    BusinessOrderRole,
    validate_order_role,
)
from freqtrade.hedge.planning.context import ActiveOrder, OrderIntent, PlanningResult


class BusinessIdentityError(RuntimeError):
    """Business identity cannot be proven safely."""


class BusinessIdentityAllocator(Protocol):
    def allocate_entry(
        self,
        *,
        account_id: str,
        exchange: str,
        symbol: str,
        position_side: object,
        strategy_entry_key: str,
        bucket: object,
    ) -> BusinessIdentity: ...

    def load_for_lot(self, business_lot_id: object) -> BusinessIdentity: ...


_ROLE_BY_ACTION = {
    "OPEN": BusinessOrderRole.ENTRY,
    "INCREASE": BusinessOrderRole.INCREASE,
    "REDUCE": BusinessOrderRole.REDUCE,
    "CLOSE": BusinessOrderRole.CLOSE,
    "UNSTUCK": BusinessOrderRole.UNSTUCK,
}


@dataclass(slots=True)
class BusinessIdentityBinder:
    """Bind durable business identity to the final planner submit set."""

    allocator: BusinessIdentityAllocator
    account_id: str
    exchange: str = "binance"

    @staticmethod
    def _action(intent: object) -> str:
        raw = getattr(getattr(intent, "action", None), "value", getattr(intent, "action", ""))
        value = str(raw).strip().upper()
        if value not in _ROLE_BY_ACTION:
            raise BusinessIdentityError(f"unsupported planner action: {value!r}")
        return value

    def _role(self, intent: object) -> BusinessOrderRole:
        raw = getattr(intent, "order_role", None)
        role = (
            _ROLE_BY_ACTION[self._action(intent)]
            if raw is None
            else raw
            if isinstance(raw, BusinessOrderRole)
            else BusinessOrderRole(str(raw).upper())
        )
        try:
            return validate_order_role(
                role,
                action=intent.action,
                reduce_only=bool(intent.reduce_only),
            )
        except (TypeError, ValueError) as exc:
            raise BusinessIdentityError(str(exc)) from exc

    def _assert_identity(self, identity: BusinessIdentity, intent: object) -> None:
        try:
            identity.assert_matches(
                account_id=self.account_id,
                symbol=intent.symbol,
                position_side=intent.position_side,
            )
        except (TypeError, ValueError) as exc:
            raise BusinessIdentityError(str(exc)) from exc

    @staticmethod
    def _replace(
        intent: OrderIntent,
        identity: BusinessIdentity,
        role: BusinessOrderRole,
        *,
        order_revision: int | None = None,
    ) -> OrderIntent:
        changes: dict[str, object] = {
            "business_identity": identity,
            "order_role": role,
            "target_business_lot_id": (
                identity.business_lot_id if role.reduces_risk else intent.target_business_lot_id
            ),
        }
        if order_revision is not None:
            changes["order_revision"] = order_revision
        try:
            return replace(intent, **changes)
        except TypeError as exc:
            raise BusinessIdentityError(
                "planner OrderIntent does not expose the canonical business identity fields"
            ) from exc

    def bind_intent(
        self,
        intent: OrderIntent,
        active_order: ActiveOrder | None = None,
    ) -> OrderIntent:
        role = self._role(intent)
        current = getattr(intent, "business_identity", None)
        if isinstance(current, BusinessIdentity):
            self._assert_identity(current, intent)
            return self._replace(intent, current, role)

        if active_order is not None:
            inherited = getattr(active_order, "business_identity", None)
            if not isinstance(inherited, BusinessIdentity):
                raise BusinessIdentityError(
                    "replacement target is a managed active order without business identity"
                )
            self._assert_identity(inherited, intent)
            active_role = getattr(active_order, "order_role", None)
            if role.reduces_risk:
                replacement_role = (
                    active_role
                    if isinstance(active_role, BusinessOrderRole) and active_role.reduces_risk
                    else role
                )
            else:
                replacement_role = BusinessOrderRole.ENTRY_REPLACE
            revision = int(getattr(active_order, "order_revision", 0)) + 1
            return self._replace(
                intent,
                inherited,
                replacement_role,
                order_revision=revision,
            )

        target_lot = getattr(intent, "target_business_lot_id", None)
        if target_lot is not None:
            try:
                identity = self.allocator.load_for_lot(target_lot)
            except (KeyError, ValueError, RuntimeError) as exc:
                raise BusinessIdentityError(
                    f"target business lot cannot be resolved: {target_lot}"
                ) from exc
            self._assert_identity(identity, intent)
            if not role.reduces_risk:
                raise BusinessIdentityError("entry intent cannot target an existing business lot")
            return self._replace(intent, identity, role)

        if role in {BusinessOrderRole.ENTRY, BusinessOrderRole.INCREASE}:
            strategy_entry_key = str(
                getattr(intent, "strategy_entry_key", "") or getattr(intent, "intent_id", "")
            ).strip()
            if not strategy_entry_key:
                raise BusinessIdentityError("new entry has no durable allocation key")
            try:
                identity = self.allocator.allocate_entry(
                    account_id=self.account_id,
                    exchange=self.exchange,
                    symbol=intent.symbol,
                    position_side=intent.position_side,
                    strategy_entry_key=strategy_entry_key,
                    bucket=intent.bucket,
                )
            except Exception as exc:
                raise BusinessIdentityError("durable business identity allocation failed") from exc
            self._assert_identity(identity, intent)
            return self._replace(intent, identity, role)

        raise BusinessIdentityError(
            f"{role.value} requires an explicit target business lot; attribution is ambiguous"
        )

    def bind_planning_result(
        self,
        planning: PlanningResult,
        *,
        active_orders: tuple[ActiveOrder, ...],
    ) -> PlanningResult:
        """Bind only final submit orders, before cancel/submit side effects begin."""

        if not isinstance(planning, PlanningResult):
            raise TypeError("planning must be PlanningResult")
        by_order_id = {item.order_id: item for item in active_orders}
        replacement_map = dict(getattr(planning, "replacement_order_map", ()))
        bound: list[OrderIntent] = []
        for intent in planning.submit_orders:
            active_id = replacement_map.get(intent.intent_id)
            active = None if active_id is None else by_order_id.get(active_id)
            if active_id is not None and active is None:
                raise BusinessIdentityError(
                    f"replacement source disappeared before identity binding: {active_id}"
                )
            bound.append(self.bind_intent(intent, active_order=active))
        return replace(planning, submit_orders=tuple(bound))

    def assert_bound(self, intent: OrderIntent) -> BusinessIdentity:
        identity = cast(BusinessIdentity | None, getattr(intent, "business_identity", None))
        if not isinstance(identity, BusinessIdentity):
            raise BusinessIdentityError("planner intent is not bound to durable business identity")
        self._assert_identity(identity, intent)
        return identity
