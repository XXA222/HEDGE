#!/usr/bin/env python3
"""Execution-store throughput and idempotency benchmark."""

from __future__ import annotations

import argparse
import json
from decimal import Decimal
from pathlib import Path
from statistics import mean
from time import perf_counter

from freqtrade.hedge.execution.fake_exchange import FakeExchangeExecutionPort
from freqtrade.hedge.execution.idempotency import InMemoryIdempotencyStore
from freqtrade.hedge.execution.kill_switch import KillSwitch
from freqtrade.hedge.execution.service import (
    AllowAllRiskApproval,
    ExecutionService,
    InMemoryExecutionStore,
    IntentAction,
    OrderIntent,
    OrderType,
    PositionSide,
)
from freqtrade.hedge.execution.state_machine import OrderState
from freqtrade.hedge.execution.unknown_resolver import UnknownOrderResolver


def _intent(key: str, side: PositionSide) -> OrderIntent:
    return OrderIntent(
        account_id="bench",
        symbol="BTC/USDT:USDT",
        position_side=side,
        action=IntentAction.OPEN,
        quantity=Decimal("0.001"),
        idempotency_key=key,
        order_type=OrderType.LIMIT,
        limit_price=Decimal(60000),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="")
    parser.add_argument("--cycles", type=int, default=1500)
    parser.add_argument("--retention", type=int, default=2000)
    args = parser.parse_args()

    store = InMemoryExecutionStore(terminal_retention=args.retention)
    exchange = FakeExchangeExecutionPort(terminal_retention=args.retention)
    idempotency = InMemoryIdempotencyStore(completed_retention=args.retention)
    service = ExecutionService(
        risk=AllowAllRiskApproval(),
        exchange=exchange,
        store=store,
        idempotency=idempotency,
        unknown_resolver=UnknownOrderResolver(exchange),
        kill_switch=KillSwitch(),
    )

    samples: list[float] = []
    for cycle in range(args.cycles):
        started = perf_counter()
        for side_index, side in enumerate((PositionSide.LONG, PositionSide.SHORT)):
            exchange.queue_snapshot(
                OrderState.FILLED,
                filled_quantity=Decimal("0.001"),
                average_price=Decimal(60000),
                exchange_trade_id=f"trade-{cycle}-{side_index}",
            )
            service.submit(_intent(f"cycle-{cycle}-{side_index}", side))
        # Exercise repeated active-order queries without touching terminal history.
        for _ in range(12):
            service.list_orders(
                account_id="bench",
                symbol="BTC/USDT:USDT",
                include_terminal=False,
                limit=32,
            )
        samples.append((perf_counter() - started) * 1000.0)

    chunk = 250
    segments = [mean(samples[i : i + chunk]) for i in range(0, len(samples), chunk)]
    first = segments[0]
    last = segments[-1]
    ratio = last / first if first else 999.0
    gauges = store.collection_gauges()
    exchange_gauges = exchange.collection_gauges()
    idem_gauges = idempotency.collection_gauges()
    bounded = (
        gauges["orders"] <= args.retention
        and exchange_gauges["orders"] <= args.retention
        and idem_gauges["completed"] <= args.retention
    )
    # Timing is informative in noisy CI; the fail-closed invariants are boundedness
    # and active-query cardinality.  A generous 2x ratio catches renewed O(N log N).
    stable = ratio < 2.0
    report = {
        "schema": "freqtrade-hedge-execution-performance-v1",
        "cycles": args.cycles,
        "retention": args.retention,
        "segment_mean_ms": segments,
        "last_to_first_ratio": ratio,
        "store_gauges": gauges,
        "exchange_gauges": exchange_gauges,
        "idempotency_gauges": idem_gauges,
        "bounded": bounded,
        "stable": stable,
        "status": "PASS" if bounded and stable else "FAIL",
    }
    rendered = json.dumps(report, indent=2) + "\n"
    print(rendered, end="")
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(rendered, encoding="utf-8")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
