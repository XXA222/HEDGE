from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import tracemalloc
import types
from pathlib import Path

import numpy as np
import pandas as pd


def _install_exchange_stub() -> None:
    if "freqtrade.exchange" in sys.modules:
        return
    module = types.ModuleType("freqtrade.exchange")

    def timeframe_to_seconds(timeframe: str) -> int:
        value = timeframe.strip().lower()
        amount = int(value[:-1])
        factor = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}[value[-1]]
        return amount * factor

    module.timeframe_to_seconds = timeframe_to_seconds
    sys.modules["freqtrade.exchange"] = module


def _frame(count: int) -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=count, freq="min", tz="UTC")
    price = np.full(count, 100.0, dtype="float32")
    return pd.DataFrame(
        {
            "date": dates,
            "open": price,
            "high": price + 1.0,
            "low": price - 1.0,
            "close": price,
            "volume": np.full(count, 100.0, dtype="float32"),
            "hedge_long_score": np.zeros(count, dtype="float32"),
            "hedge_short_score": np.zeros(count, dtype="float32"),
            "hedge_target_net_ratio": np.zeros(count, dtype="float32"),
        }
    )


def _measure(fn):
    gc.collect()
    tracemalloc.start()
    started = time.perf_counter()
    value = fn()
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return value, elapsed, peak


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bars", type=int, default=50000)
    parser.add_argument("--chunk-bars", type=int, default=2048)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.bars < 2 or args.chunk_bars < 1:
        raise SystemExit("bars must be >=2 and chunk-bars must be >=1")

    _install_exchange_stub()
    from freqtrade.optimize.hedge_backtesting import (
        HedgeBacktestEventChunks,
        events_from_analyzed_dataframe,
    )

    data = _frame(args.bars)

    def legacy():
        dataset = events_from_analyzed_dataframe(
            pair="BTC/USDT:USDT",
            timeframe="1m",
            frame=data,
        )
        return {
            "events": len(dataset.events),
            "fingerprint": dataset.data_fingerprint,
        }

    def compact():
        stream = HedgeBacktestEventChunks(
            pair="BTC/USDT:USDT",
            timeframe="1m",
            frame=data,
            chunk_bars=args.chunk_bars,
        )
        event_count = sum(len(chunk) for chunk in stream)
        dataset = stream.dataset()
        return {
            "events_seen": event_count,
            "events_retained": len(dataset.events),
            "max_chunk_input_events": stream.max_chunk_input_events,
            "fingerprint": dataset.data_fingerprint,
        }

    legacy_value, legacy_seconds, legacy_peak = _measure(legacy)
    compact_value, compact_seconds, compact_peak = _measure(compact)
    payload = {
        "schema": "hedge-memory-builder-benchmark-v1",
        "bars": args.bars,
        "chunk_bars": args.chunk_bars,
        "legacy": {
            **legacy_value,
            "seconds": legacy_seconds,
            "tracemalloc_peak_bytes": legacy_peak,
        },
        "compact": {
            **compact_value,
            "seconds": compact_seconds,
            "tracemalloc_peak_bytes": compact_peak,
        },
        "peak_memory_ratio_compact_to_legacy": (
            compact_peak / legacy_peak if legacy_peak else None
        ),
        "peak_memory_reduction_ratio": (1.0 - compact_peak / legacy_peak if legacy_peak else None),
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
