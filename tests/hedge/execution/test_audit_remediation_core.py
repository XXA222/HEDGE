from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal

from freqtrade.hedge.contracts.events import OutboxEvent
from freqtrade.hedge.execution.event_publisher import (
    HedgeEventHubPublisher,
    InMemoryEventPublisher,
)
from freqtrade.hedge.execution.fake_exchange import FakeExchangeExecutionPort
from freqtrade.hedge.execution.idempotency import InMemoryIdempotencyStore
from freqtrade.hedge.execution.kill_switch import KillSwitch
from freqtrade.hedge.execution.service import (
    AllowAllRiskApproval,
    ExecutionService,
    InMemoryAuditLog,
    InMemoryExecutionStore,
    IntentAction,
    OrderIntent,
    OrderType,
    PositionSide,
)
from freqtrade.hedge.execution.state_machine import OrderState
from freqtrade.hedge.execution.unknown_resolver import UnknownOrderResolver


def _intent(key: str, *, side: PositionSide = PositionSide.LONG) -> OrderIntent:
    return OrderIntent(
        account_id="main",
        symbol="ETH/USDT:USDT",
        position_side=side,
        action=IntentAction.OPEN,
        quantity=Decimal("0.1"),
        idempotency_key=key,
        order_type=OrderType.LIMIT,
        limit_price=Decimal(3000),
    )


def _service(*, retention: int = 5):
    exchange = FakeExchangeExecutionPort(terminal_retention=retention)
    store = InMemoryExecutionStore(terminal_retention=retention)
    idempotency = InMemoryIdempotencyStore(completed_retention=retention)
    service = ExecutionService(
        risk=AllowAllRiskApproval(),
        exchange=exchange,
        store=store,
        idempotency=idempotency,
        unknown_resolver=UnknownOrderResolver(exchange),
        kill_switch=KillSwitch(),
    )
    return service, exchange, store, idempotency


def test_execution_store_retention_and_indexes_are_bounded() -> None:
    service, exchange, store, idempotency = _service(retention=5)
    for index in range(20):
        exchange.queue_snapshot(
            OrderState.FILLED,
            filled_quantity=Decimal("0.1"),
            average_price=Decimal(3000),
            exchange_trade_id=f"trade-{index}",
        )
        service.submit(_intent(f"terminal-{index}"))

    gauges = store.collection_gauges()
    assert gauges["orders"] == 5
    assert gauges["open_orders"] == 0
    assert gauges["retained_terminal_orders"] == 5
    assert exchange.collection_gauges()["orders"] == 5
    assert idempotency.collection_gauges()["completed"] == 5


def test_execution_store_filter_pushdown_semantics_preserve_newest_first() -> None:
    service, exchange, store, _ = _service(retention=20)
    for index, side in enumerate((PositionSide.LONG, PositionSide.SHORT, PositionSide.LONG)):
        exchange.queue_snapshot(OrderState.ACKNOWLEDGED)
        service.submit(_intent(f"open-{index}", side=side))

    long_rows = service.list_orders(
        account_id="main",
        symbol="ETH/USDT:USDT",
        position_side=PositionSide.LONG,
        include_terminal=False,
        limit=1,
    )
    assert len(long_rows) == 1
    assert long_rows[0].intent.idempotency_key == "open-2"
    assert len(store.list_orders(include_terminal=False)) == 3


def test_execution_store_newest_first_survives_created_at_ties() -> None:
    service, exchange, _store, _ = _service(retention=20)
    submitted = []
    for index in range(3):
        exchange.queue_snapshot(OrderState.ACKNOWLEDGED)
        submitted.append(service.submit(_intent(f"tie-{index}")).order)

    fixed_time = datetime(2026, 8, 15, 0, 0, tzinfo=UTC)
    tied_store = InMemoryExecutionStore(terminal_retention=20)
    for order in submitted:
        tied_store.put(replace(order, created_at=fixed_time))

    newest = tied_store.list_orders(include_terminal=False, newest_first=True, limit=1)
    oldest = tied_store.list_orders(include_terminal=False, newest_first=False, limit=1)
    assert newest[0].intent.idempotency_key == "tie-2"
    assert oldest[0].intent.idempotency_key == "tie-0"


def test_unknown_index_is_removed_when_order_resolves() -> None:
    service, exchange, store, _ = _service(retention=20)
    exchange.queue_timeout()
    result = service.submit(_intent("unknown"))
    leg_key = result.order.leg_key
    assert store.has_unresolved_unknown(leg_key)

    exchange.acknowledge_order(result.order.client_order_id)
    service.refresh_order(result.order.client_order_id)
    assert not store.has_unresolved_unknown(leg_key)


def test_event_and_audit_buffers_are_capacity_bounded() -> None:
    publisher = InMemoryEventPublisher(capacity=3)
    observed: list[str] = []
    publisher.add_callback(lambda event: observed.append(event.event_type))
    for index in range(8):
        publisher.publish(OutboxEvent(f"ORDER_{index}", {"index": index}))
    assert len(publisher.events()) == 3
    assert [event.event_type for event in publisher.events()] == [
        "ORDER_5",
        "ORDER_6",
        "ORDER_7",
    ]
    assert len(observed) == 8

    audit = InMemoryAuditLog(capacity=2)
    for index in range(5):
        audit.emit("TEST", {"index": index})
    assert len(audit.records) == 2


def test_event_hub_publisher_never_creates_loop_from_sync_context(monkeypatch) -> None:
    class Hub:
        async def publish(self, event) -> None:
            raise AssertionError("sync path without owner loop must not run coroutine")

    def forbidden(*args, **kwargs):
        raise AssertionError("asyncio.run must not be used by HedgeEventHubPublisher")

    monkeypatch.setattr(asyncio, "run", forbidden)
    publisher = HedgeEventHubPublisher(Hub())
    publisher.publish(OutboxEvent("ORDER_SYNC", {}))


def test_event_hub_publisher_delivers_on_running_owner_loop() -> None:
    delivered: list[str] = []

    class Hub:
        async def publish(self, event) -> None:
            delivered.append(event.payload["event_type"])

    async def scenario() -> None:
        publisher = HedgeEventHubPublisher(Hub())
        publisher.publish(OutboxEvent("ORDER_ASYNC", {}))
        await asyncio.sleep(0)

    asyncio.run(scenario())
    assert delivered == ["ORDER_ASYNC"]
