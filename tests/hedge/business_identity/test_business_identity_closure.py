from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from uuid import uuid4

import pytest

from freqtrade.hedge.contracts.business_identity import BusinessIdentity, BusinessOrderRole
from freqtrade.hedge.integration.business_identity import (
    BusinessIdentityBinder,
    BusinessIdentityError,
)
from freqtrade.hedge.planning.context import (
    ActiveOrder,
    IntentAction,
    OrderIntent,
    OrderSide,
    PlanningResult,
    PositionBucket,
    PositionSide,
    StrategyLegState,
)


class FakeAllocator:
    def __init__(self) -> None:
        self.by_lot: dict[object, BusinessIdentity] = {}
        self.allocations: list[str] = []

    def allocate_entry(
        self,
        *,
        account_id,
        exchange,
        symbol,
        position_side,
        strategy_entry_key,
        bucket,
    ):
        del exchange, bucket
        self.allocations.append(strategy_entry_key)
        identity = BusinessIdentity(
            business_trade_id=uuid4(),
            business_trade_seq=len(self.allocations),
            business_lot_id=uuid4(),
            lot_index=1,
            account_id=account_id,
            symbol=symbol,
            position_side=position_side,
        )
        self.by_lot[identity.business_lot_id] = identity
        return identity

    def load_for_lot(self, business_lot_id):
        return self.by_lot[business_lot_id]


def entry_intent(*, key="entry-1", price="100") -> OrderIntent:
    return OrderIntent.deterministic(
        symbol="BTCUSDT",
        position_side=PositionSide.LONG,
        order_side=OrderSide.BUY,
        action=IntentAction.OPEN,
        bucket=PositionBucket.TACTICAL,
        quantity=Decimal("0.01"),
        price=Decimal(price),
        reduce_only=False,
        strategy_entry_key=key,
    )


def active_from(intent: OrderIntent) -> ActiveOrder:
    return ActiveOrder(
        order_id="active-1",
        client_order_id="client-1",
        symbol=intent.symbol,
        position_side=intent.position_side,
        order_side=intent.order_side,
        quantity=intent.quantity,
        price=intent.price,
        reduce_only=False,
        bucket=intent.bucket,
        action=intent.action,
        created_at=datetime.now(UTC),
        business_identity=intent.business_identity,
        order_role=intent.order_role,
        order_revision=intent.order_revision,
        strategy_entry_key=intent.strategy_entry_key,
    )


def test_new_entry_allocates_before_execution_adaptation() -> None:
    allocator = FakeAllocator()
    binder = BusinessIdentityBinder(allocator=allocator, account_id="acct", exchange="paper")
    bound = binder.bind_intent(entry_intent())
    assert bound.business_identity is not None
    assert bound.order_role is BusinessOrderRole.ENTRY
    assert allocator.allocations == ["entry-1"]


def test_cancel_replace_inherits_trade_and_lot_and_increments_revision() -> None:
    allocator = FakeAllocator()
    binder = BusinessIdentityBinder(allocator=allocator, account_id="acct", exchange="paper")
    original = binder.bind_intent(entry_intent())
    active = active_from(original)
    replacement = binder.bind_intent(entry_intent(key="entry-price-change", price="101"), active)
    assert replacement.business_identity == original.business_identity
    assert replacement.order_role is BusinessOrderRole.ENTRY_REPLACE
    assert replacement.order_revision == original.order_revision + 1
    assert allocator.allocations == ["entry-1"]


def test_targeted_reduce_resolves_exact_business_lot() -> None:
    allocator = FakeAllocator()
    binder = BusinessIdentityBinder(allocator=allocator, account_id="acct", exchange="paper")
    opened = binder.bind_intent(entry_intent())
    identity = opened.business_identity
    assert identity is not None
    reduce = OrderIntent.deterministic(
        symbol="BTCUSDT",
        position_side=PositionSide.LONG,
        order_side=OrderSide.SELL,
        action=IntentAction.REDUCE,
        bucket=PositionBucket.TACTICAL,
        quantity=Decimal("0.005"),
        price=Decimal(110),
        reduce_only=True,
        order_role=BusinessOrderRole.TAKE_PROFIT,
        target_business_lot_id=identity.business_lot_id,
    )
    bound = binder.bind_intent(reduce)
    assert bound.business_identity == identity
    assert bound.target_business_lot_id == identity.business_lot_id
    assert bound.order_role is BusinessOrderRole.TAKE_PROFIT


def test_ambiguous_reduce_fails_closed() -> None:
    binder = BusinessIdentityBinder(allocator=FakeAllocator(), account_id="acct", exchange="paper")
    reduce = OrderIntent.deterministic(
        symbol="BTCUSDT",
        position_side=PositionSide.LONG,
        order_side=OrderSide.SELL,
        action=IntentAction.REDUCE,
        bucket=PositionBucket.TACTICAL,
        quantity=Decimal("0.005"),
        price=Decimal(110),
        reduce_only=True,
    )
    with pytest.raises(BusinessIdentityError, match="explicit target business lot"):
        binder.bind_intent(reduce)


def test_planning_replacement_map_drives_identity_inheritance() -> None:
    allocator = FakeAllocator()
    binder = BusinessIdentityBinder(allocator=allocator, account_id="acct", exchange="paper")
    original = binder.bind_intent(entry_intent())
    active = active_from(original)
    desired = entry_intent(key="replacement", price="101")
    planning = PlanningResult(
        ideal_orders=(desired,),
        submit_orders=(desired,),
        cancel_order_ids=(active.order_id,),
        kept_order_ids=(),
        long_state=StrategyLegState(PositionSide.LONG),
        short_state=StrategyLegState(PositionSide.SHORT),
        modify_order_ids=(active.order_id,),
        replacement_order_map=((desired.intent_id, active.order_id),),
    )
    bound = binder.bind_planning_result(planning, active_orders=(active,))
    assert bound.submit_orders[0].business_identity == original.business_identity
    assert bound.submit_orders[0].order_role is BusinessOrderRole.ENTRY_REPLACE
