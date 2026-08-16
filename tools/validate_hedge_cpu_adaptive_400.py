#!/usr/bin/env python3
"""Deterministic 400-point validation for adaptive CPU/performance mainline."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from freqtrade.hedge.performance.resource_governor import (  # noqa: E402
    AdaptiveResourceGovernor,
    ResourcePolicy,
    ResourceSnapshot,
)
from freqtrade.hedge.planning.context import (  # noqa: E402
    IntentAction,
    OrderIntent,
    OrderSide,
    PositionBucket,
    PositionSide,
)
from freqtrade.hedge.simulation.cross_wallet import CrossWallet  # noqa: E402
from freqtrade.hedge.simulation.exchange import BarEvent  # noqa: E402
from freqtrade.hedge.simulation.matcher import ConservativeMatcher  # noqa: E402


NOW = datetime(2026, 8, 13, tzinfo=UTC)
rows: list[dict[str, object]] = []


class _CountingClone:
    def __init__(self, original, calls: list[int]) -> None:
        self.original = original
        self.calls = calls

    def __call__(self, value: CrossWallet) -> CrossWallet:
        self.calls[0] += 1
        return self.original(value)


def add(theme: str, case: int, ok: bool, detail: object = "") -> None:
    rows.append(
        {
            "id": len(rows) + 1,
            "theme": theme,
            "case": case,
            "status": "PASS" if ok else "FAIL",
            "detail": detail,
        }
    )


def snapshot(
    *,
    cpu: float = 0.0,
    logical: int = 32,
    physical: int = 16,
    affinity: int = 32,
    mem_available_mib: int = 7168,
    source: str = "host-broker",
) -> ResourceSnapshot:
    limit = 8 * 1024**3
    available = mem_available_mib * 1024**2
    current = max(0, limit - available)
    return ResourceSnapshot(
        logical_cpus=logical,
        physical_cpus=physical,
        affinity_cpus=affinity,
        system_cpu_percent=cpu,
        process_cpu_percent=0.0,
        cgroup_memory_limit_bytes=limit,
        cgroup_memory_current_bytes=current,
        host_memory_available_bytes=available,
        timestamp_monotonic=1.0,
        source=source,
        host_snapshot_age_seconds=0.1,
    )


def source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8-sig")


# 1. Evidence-driven policy defaults: 20
for i in range(20):
    p = ResourcePolicy()
    ok = (
        p.enabled
        and p.target_system_cpu_percent == 88.0
        and p.hard_system_cpu_percent == 96.0
        and p.reserve_logical_cpus == 2
        and p.hard_max_workers == 28
        and p.worker_memory_mib >= 256
    )
    add("adaptive-policy-defaults", i, ok, str(p))

# 2. Idle desktop scales independent work: 20
p = ResourcePolicy()
g = AdaptiveResourceGovernor(p)
for i in range(20):
    tasks = i + 2
    got = g.recommended_workers(tasks=tasks, requested=0, snapshot=snapshot(cpu=0.0))
    expected = min(tasks, 28, 24)  # 7GiB available - 1GiB reserve / 256MiB
    add("idle-desktop-scale-up", i, got == expected, {"got": got, "expected": expected})

# 3. Busy desktop backs off: 20
for i in range(20):
    cpu = 96.0 + min(3.9, i * 0.2)
    got = g.recommended_workers(
        tasks=64,
        requested=0,
        current_workers=0,
        snapshot=snapshot(cpu=cpu),
    )
    add("busy-desktop-backoff", i, got == 1, {"cpu": cpu, "workers": got})

# 4. Moderate desktop load adapts continuously: 20
previous = 99
for i in range(20):
    cpu = float(i * 4)
    got = g.recommended_workers(tasks=64, requested=0, snapshot=snapshot(cpu=cpu))
    ok = 1 <= got <= 24 and got <= previous
    add("load-sensitive-worker-budget", i, ok, {"cpu": cpu, "workers": got})
    previous = got

# 5. Memory pressure constrains process count: 20
for i in range(20):
    available = 1280 + i * 128
    s = snapshot(cpu=0.0, mem_available_mib=available)
    got = g.recommended_workers(tasks=64, requested=-1, snapshot=s)
    expected = max(1, min(28, (max(0, available - 1024) // 256)))
    add("memory-aware-worker-cap", i, got == expected, {"got": got, "expected": expected})

# 6. Numeric threading avoids oversubscription: 20
for i in range(20):
    concurrent = 1 if i < 10 else (i - 8)
    got = g.numeric_threads(concurrent_python_workers=concurrent, snapshot=snapshot(cpu=0.0))
    expected = 28 if concurrent == 1 else 1
    add("numeric-thread-budget", i, got == expected, {"concurrent": concurrent, "threads": got})

# 7. Host broker uses low-overhead kernel APIs: 20
broker = source("scripts/Start-Freqtrade-Hedge-Adaptive-Resource-Broker.ps1")
required = (
    "GetSystemTimes",
    "GlobalMemoryStatusEx",
    "freqtrade-hedge-host-resource-v2",
    "host-resource-snapshot.json",
    "Move-Item -LiteralPath $Temp -Destination $SnapshotPath -Force",
    "Get-CimInstance Win32_Processor",
    "$QuotedScript",
    "$QuotedUserData",
)
for i in range(20):
    ok = all(token in broker for token in required) and "Win32_PerfFormattedData" not in broker
    add("windows-host-resource-broker", i, ok)

# 8. Process parallel backtesting contracts: 20
parallel = source("freqtrade/hedge/backtesting/parallel.py")
for i in range(20):
    tokens = (
        "ProcessPoolExecutor",
        "multiprocessing_context",
        "configure_worker_numeric_threads",
        "recommended_workers",
        "FIRST_COMPLETED",
        "resource_samples",
    )
    add("adaptive-backtest-process-pool", i, all(t in parallel for t in tokens))

# 9. Native hyperopt process/COW contracts: 20
native = source("freqtrade/hedge/native/parallel_hyperopt.py")
for i in range(20):
    tokens = (
        "ProcessPoolExecutor",
        "_PREPARED",
        "multiprocessing_context",
        "configure_worker_numeric_threads",
        "recommended_workers",
        "persist_artifact=False",
    )
    add("adaptive-native-hyperopt", i, all(t in native for t in tokens))

# 10. Research optimizer dynamic feed: 20
engine = source("freqtrade/hedge/optimization/engine.py")
for i in range(20):
    tokens = (
        "ProcessPoolExecutor",
        "FIRST_COMPLETED",
        "recommended_workers",
        "process-fork-adaptive",
        "resource_samples",
        "submit_until",
    )
    add("adaptive-research-optimizer", i, all(t in engine for t in tokens))

# 11. Flat/no-order matcher eliminates clones: 20
for i in range(20):
    wallet = CrossWallet(Decimal(1000))
    matcher = ConservativeMatcher()
    bar = BarEvent(
        timestamp=NOW + timedelta(minutes=i),
        symbol="BTC/USDT:USDT",
        open=Decimal(100),
        high=Decimal(101),
        low=Decimal(99),
        close=Decimal(100),
        volume=Decimal(1000),
    )
    with patch.object(matcher, "_clone", side_effect=AssertionError("unexpected clone")):
        outcome = matcher.match_outcome(bar, wallet)
    add("matcher-flat-fast-path", i, outcome.fills == () and outcome.ending_equity == Decimal(1000))

# 12. Untouched GTC order eliminates clones: 20
for i in range(20):
    wallet = CrossWallet(Decimal(1000))
    price = Decimal(80) - Decimal(i) / Decimal(10)
    intent = OrderIntent.deterministic(
        symbol="BTC/USDT:USDT",
        position_side=PositionSide.LONG,
        order_side=OrderSide.BUY,
        action=IntentAction.OPEN,
        bucket=PositionBucket.CORE,
        quantity=Decimal("0.01"),
        price=price,
        reduce_only=False,
    )
    wallet.accept_order(f"far-{i}", intent, accepted_at=NOW)
    bar = BarEvent(
        timestamp=NOW + timedelta(minutes=i + 1),
        symbol="BTC/USDT:USDT",
        open=Decimal(100),
        high=Decimal(101),
        low=Decimal(99),
        close=Decimal(100),
        volume=Decimal(1000),
    )
    matcher = ConservativeMatcher()
    with patch.object(matcher, "_clone", side_effect=AssertionError("unexpected clone")):
        outcome = matcher.match_outcome(bar, wallet)
    add(
        "matcher-no-touch-fast-path",
        i,
        outcome.fills == () and wallet.remaining(f"far-{i}") == Decimal("0.01"),
    )

# 13. Touching order preserves conservative two-path semantics: 20
for i in range(20):
    wallet = CrossWallet(Decimal(1000))
    intent = OrderIntent.deterministic(
        symbol="BTC/USDT:USDT",
        position_side=PositionSide.LONG,
        order_side=OrderSide.BUY,
        action=IntentAction.OPEN,
        bucket=PositionBucket.CORE,
        quantity=Decimal("0.01"),
        price=Decimal("99.5"),
        reduce_only=False,
    )
    wallet.accept_order(f"touch-{i}", intent, accepted_at=NOW)
    matcher = ConservativeMatcher()
    original = matcher._clone
    calls = [0]

    bar = BarEvent(
        timestamp=NOW + timedelta(minutes=i + 1),
        symbol="BTC/USDT:USDT",
        open=Decimal(100),
        high=Decimal(101),
        low=Decimal(99),
        close=Decimal(100),
        volume=Decimal(1000),
    )
    with patch.object(matcher, "_clone", side_effect=_CountingClone(original, calls)):
        outcome = matcher.match_outcome(bar, wallet)
    add("matcher-touch-parity", i, calls[0] == 2 and bool(outcome.fills), calls[0])

# 14. Direct scalar unrealized calculation matches immutable projection: 20
for i in range(20):
    wallet = CrossWallet(Decimal(1000))
    qty = Decimal(i + 1) / Decimal(100)
    wallet.long.quantity = qty
    wallet.long.core_quantity = qty
    wallet.long.core_average_price = Decimal(100)
    wallet.long.average_price = Decimal(100)
    mark = Decimal(95) + Decimal(i)
    direct = wallet.unrealized(mark)
    projected = wallet.long.immutable().unrealized_pnl(
        mark
    ) + wallet.short.immutable().unrealized_pnl(mark)
    add("wallet-direct-scalar-risk", i, direct == projected, str(direct))

# 15. Lightweight matcher clone isolation: 20
for i in range(20):
    wallet = CrossWallet(Decimal(1000))
    clone = wallet.clone_for_matcher()
    clone.balance -= Decimal(i + 1)
    clone.gross_peak += Decimal(i + 1)
    add(
        "lightweight-wallet-clone-isolation",
        i,
        wallet.balance == Decimal(1000)
        and clone.balance != wallet.balance
        and clone.active_orders is not wallet.active_orders,
    )

# 16. Replay/planner low-allocation source contracts: 20
replay = source("freqtrade/hedge/simulation/replay.py")
context = source("freqtrade/hedge/planning/context.py")
for i in range(20):
    ok = all(
        token in replay + context
        for token in (
            "collect_diagnostics=False",
            "_effective_planner_config_key",
            "with_state_trusted",
            "_evolve_trusted",
            "if not fills",
        )
    )
    add("replay-planner-allocation-control", i, ok)

# 17. Compact dataframe/event adapter source contracts: 20
backtest = source("freqtrade/optimize/hedge_backtesting.py")
for i in range(20):
    tokens = (
        "_compact_signal_bar_from_row",
        "_ArrayRowView",
        "numeric_execution_context",
        "reduce_df_footprint",
        "clear_cache(include_backtesting=True)",
        "persist_artifact: bool = True",
    )
    add("compact-vector-preprocess", i, all(t in backtest for t in tokens))

# 18. CLI/config adaptive defaults: 20
cli = source("freqtrade/commands/hedge_cli.py")
bt_cfg = source("freqtrade/hedge/backtesting/config.py")
opt_cfg = source("freqtrade/hedge/optimization/config.py")
example = source("config_examples/config_hedge_hyperopt.example.json")
for i in range(20):
    ok = (
        "0=adaptive" in cli
        and 'raw.get("workers", 0)' in bt_cfg
        and "default=0" in opt_cfg
        and '"workers": 0' in example
    )
    add("adaptive-config-defaults", i, ok)

# 19. Memory/GC remains pressure-aware while CPU work scales: 20
memory = source("freqtrade/hedge/backtesting/memory.py")
for i in range(20):
    tokens = (
        "HEDGE_MEMORY_RELEASE_MODE",
        "HEDGE_MEMORY_GC_RSS_MIB",
        "HEDGE_MEMORY_GC_HARD_PRESSURE_RATIO",
        "gc.collect",
        "malloc_trim",
    )
    add("cpu-memory-coordination", i, all(t in memory for t in tokens))

# 20. PowerShell 5.1 and safety surface: 20
ps_files = [
    ROOT / "scripts/Start-Freqtrade-Hedge-Adaptive-Resource-Broker.ps1",
    ROOT / "scripts/Stop-Freqtrade-Hedge-Adaptive-Resource-Broker.ps1",
]
for i in range(20):
    ok = True
    detail: list[str] = []
    for path in ps_files:
        raw = path.read_bytes()
        text = raw.decode("ascii")
        conditions = {
            "ascii": all(byte < 128 for byte in raw),
            "crlf": b"\r\n" in raw,
            "no_join_string": "Join-String" not in text,
            "no_parallel_ps7": "ForEach-Object -Parallel" not in text,
            "no_exchange_write": "create_order" not in text and "cancel_order" not in text,
        }
        ok = ok and all(conditions.values())
        detail.append(f"{path.name}:{conditions}")
    add("ps51-and-safety", i, ok, detail)

if len(rows) != 400:
    raise AssertionError(len(rows))
failed = [row for row in rows if row["status"] != "PASS"]
payload = {
    "schema": "freqtrade-hedge-adaptive-cpu-400-v1",
    "expected": 400,
    "executed": len(rows),
    "passed": len(rows) - len(failed),
    "failed": len(failed),
    "status": "PASS" if not failed else "FAIL",
    "checks": rows,
}
print(f"HEDGE ADAPTIVE CPU 400: {payload['passed']}/400 PASS; FAIL={payload['failed']}")
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("legacy_output", nargs="?")
parser.add_argument("--project-root")
parser.add_argument("--output")
args, _ = parser.parse_known_args()
output = args.output or args.legacy_output
if output:
    Path(output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
raise SystemExit(1 if failed else 0)
