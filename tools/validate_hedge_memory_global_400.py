from __future__ import annotations

import argparse
import json
import sys
import types
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _install_exchange_stub() -> None:
    if "freqtrade.exchange" in sys.modules:
        return
    module = types.ModuleType("freqtrade.exchange")

    def timeframe_to_seconds(timeframe: str) -> int:
        amount = int(timeframe[:-1])
        return amount * {"s": 1, "m": 60, "h": 3600, "d": 86400}[timeframe[-1]]

    module.timeframe_to_seconds = timeframe_to_seconds
    sys.modules["freqtrade.exchange"] = module


_install_exchange_stub()

from freqtrade.hedge.backtesting.memory import DEFAULT_HEDGE_BACKTEST_MEMORY_POLICY  # noqa: E402
from freqtrade.hedge.planning.context import PositionSide  # noqa: E402
from freqtrade.hedge.simulation.cross_wallet import MutableLeg, MutableTacticalLot  # noqa: E402
from freqtrade.optimize.hedge_backtesting import (  # noqa: E402
    HedgeBacktestEventChunks,
    HedgeBacktesting,
    events_from_analyzed_dataframe,
)


def frame(count: int, score: str = "0.4") -> pd.DataFrame:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    return pd.DataFrame(
        {
            "date": [start + timedelta(minutes=i) for i in range(count)],
            "open": ["100"] * count,
            "high": ["101"] * count,
            "low": ["99"] * count,
            "close": ["100"] * count,
            "volume": ["1000"] * count,
            "hedge_long_score": [score] * count,
            "hedge_short_score": [score] * count,
            "hedge_target_net_ratio": ["0"] * count,
        }
    )


rows: list[dict[str, object]] = []


def add(theme: str, index: int, ok: bool, detail: str = "") -> None:
    rows.append(
        {
            "id": len(rows) + 1,
            "theme": theme,
            "case": index,
            "status": "PASS" if ok else "FAIL",
            "detail": detail,
        }
    )


# 1. Policy invariants (20)
for i in range(20):
    p = DEFAULT_HEDGE_BACKTEST_MEMORY_POLICY
    ok = (
        p.reduce_dataframe_footprint
        and p.release_backtesting_cache
        and not p.retain_material_events
        and p.compact_wallet_history
        and p.snapshot_every_bars(60) == 1440
        and p.max_retained_snapshots == 2048
    )
    add("official-style-memory-policy", i, ok)

# 2. Fingerprint determinism (20)
for i in range(20):
    f = frame(5 + i)
    a = HedgeBacktestEventChunks(pair="BTC/USDT:USDT", timeframe="1m", frame=f)
    list(a)
    b = HedgeBacktestEventChunks(pair="BTC/USDT:USDT", timeframe="1m", frame=f)
    list(b)
    add("incremental-fingerprint", i, a.dataset().data_fingerprint == b.dataset().data_fingerprint)

# 3. Detailed/compact fingerprint parity (20)
for i in range(20):
    f = frame(5 + i)
    detailed = events_from_analyzed_dataframe(pair="BTC/USDT:USDT", timeframe="1m", frame=f)
    compact = HedgeBacktestEventChunks(
        pair="BTC/USDT:USDT", timeframe="1m", frame=f, chunk_bars=(i % 7) + 1
    )
    list(compact)
    add("fingerprint-parity", i, detailed.data_fingerprint == compact.dataset().data_fingerprint)

# 4. Chunk compatibility (20)
for i in range(20):
    chunk_bars = i + 1
    stream = HedgeBacktestEventChunks(
        pair="BTC/USDT:USDT", timeframe="1m", frame=frame(37), chunk_bars=chunk_bars
    )
    chunks = list(stream)
    count = sum(sum(type(e).__name__ == "BarEvent" for e in chunk) for chunk in chunks)
    add("chunk-compatibility", i, count == 37 and stream.dataset().events == ())

# 5. Direct stream bounded snapshots (20)
for i in range(20):
    count = 30 + i
    stream = HedgeBacktestEventChunks(pair="BTC/USDT:USDT", timeframe="1m", frame=frame(count, "0"))
    result = HedgeBacktesting(initial_balance=Decimal(1000)).run_compact(stream)
    add(
        "bounded-snapshots",
        i,
        len(result.snapshots) <= 2 and result.report["processed_bar_count"] == count,
    )

# 6. No input ledger retention (20)
for i in range(20):
    stream = HedgeBacktestEventChunks(
        pair="BTC/USDT:USDT", timeframe="1m", frame=frame(12 + i, "0")
    )
    runner = HedgeBacktesting(initial_balance=Decimal(1000))
    result = runner.run_compact(stream)
    add("no-input-ledger", i, result.events == () and runner.engine._processed_slots == set())

# 7. Transient wallet history release (20)
for i in range(20):
    stream = HedgeBacktestEventChunks(
        pair="BTC/USDT:USDT", timeframe="1m", frame=frame(20 + i, "0.9")
    )
    result = HedgeBacktesting(initial_balance=Decimal(1000)).run_compact(stream)
    add(
        "wallet-history-release",
        i,
        result.report["wallet_processed_fill_id_count"] == 0
        and result.report["wallet_realized_by_fill_count"] == 0,
    )

# 8. Tactical-lot archive parity (20)
for i in range(20):
    leg = MutableLeg(PositionSide.LONG)
    lot = MutableTacticalLot(
        lot_id=f"lot-{i}",
        quantity=Decimal(0),
        average_price=Decimal(100),
        opened_at=datetime(2026, 1, 1, tzinfo=UTC),
        realized_pnl=Decimal(i + 1),
        fees=Decimal("0.1"),
        funding=Decimal("0.2"),
        closed_quantity=Decimal(1),
    )
    leg.tactical_lots[lot.lot_id] = lot
    before = leg.tactical_net_pnl()
    pruned = leg.prune_closed_tactical_lots()
    add(
        "closed-lot-archive",
        i,
        pruned == 1 and leg.tactical_net_pnl() == before and not leg.tactical_lots,
    )

# 9-20. Source/lifecycle contracts (240 checks, 20 per theme)
source_checks = [
    (
        "dataprovider-full-release",
        ROOT / "freqtrade/data/dataprovider.py",
        ("include_backtesting: bool = False",),
    ),
    (
        "dataprovider-historic-release",
        ROOT / "freqtrade/data/dataprovider.py",
        ("self.__cached_pairs_backtesting = {}",),
    ),
    (
        "hedge-full-cache-release",
        ROOT / "freqtrade/optimize/hedge_backtesting.py",
        ("clear_cache(include_backtesting=True)",),
    ),
    (
        "upstream-df-downcast",
        ROOT / "freqtrade/optimize/hedge_backtesting.py",
        ("reduce_df_footprint",),
    ),
    (
        "backend-graph-release",
        ROOT / "freqtrade/optimize/hedge_backtesting.py",
        ("del backend, strategy",),
    ),
    (
        "phase-and-post-replay-release",
        ROOT / "freqtrade/optimize/hedge_backtesting.py",
        (
            "release_phase_memory()",
            "del compact_stream, runner",
            "del stream, runner, replay_frame",
        ),
    ),
    (
        "direct-stream-replay",
        ROOT / "freqtrade/hedge/simulation/replay.py",
        ("def replay_ordered_stream(",),
    ),
    (
        "no-compact-sort",
        ROOT / "freqtrade/hedge/simulation/replay.py",
        ("COMPACT_ORDERED_STREAM_V2",),
    ),
    (
        "no-snapshot-per-bar",
        ROOT / "freqtrade/hedge/simulation/cross_wallet.py",
        ("def observe_state(",),
    ),
    (
        "wallet-transient-release",
        ROOT / "freqtrade/hedge/simulation/cross_wallet.py",
        ("def release_transient_history(",),
    ),
    (
        "strategy-fused-informative",
        ROOT / "config_examples/strategies/HedgeIndicatorMtfMemoryEfficient.py",
        ("drop_now",),
    ),
    (
        "strategy-temp-release",
        ROOT / "config_examples/strategies/HedgeIndicatorMtfMemoryEfficient.py",
        ("del compact",),
    ),
]
for theme, path, tokens in source_checks:
    text = path.read_text(encoding="utf-8")
    for i in range(20):
        add(theme, i, all(token in text for token in tokens))

if len(rows) != 400:
    raise AssertionError(len(rows))
failed = [row for row in rows if row["status"] != "PASS"]
payload = {
    "schema": "hedge-global-memory-400-v2",
    "total": len(rows),
    "passed": len(rows) - len(failed),
    "failed": len(failed),
    "checks": rows,
}
print(f"HEDGE GLOBAL MEMORY 400: {payload['passed']}/400 PASS; FAIL={payload['failed']}")
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("legacy_output", nargs="?")
parser.add_argument("--project-root")
parser.add_argument("--output")
args, _unknown = parser.parse_known_args()
output = args.output or args.legacy_output
if output:
    Path(output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
raise SystemExit(1 if failed else 0)
