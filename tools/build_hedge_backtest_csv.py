#!/usr/bin/env python3
"""Build a PIT-safe Hedge backtest CSV from downloaded Feather candles."""

import argparse
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--symbol", default="ETH/USDT:USDT")
    parser.add_argument("--funding", type=Path)
    args = parser.parse_args()
    frame = pd.read_feather(args.input).sort_values("date").drop_duplicates("date")
    fast = frame["close"].ewm(span=20, adjust=False).mean().shift(1)
    slow = frame["close"].ewm(span=80, adjust=False).mean().shift(1)
    frame["timestamp"] = frame["date"].dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    frame["symbol"] = args.symbol
    frame["long_signal"] = (fast > slow).astype(int)
    frame["short_signal"] = (fast < slow).astype(int)
    if args.funding:
        funding = pd.read_feather(args.funding)
        frame = frame.merge(funding, on="date", how="left")
        frame["mark_price"] = frame["close"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "timestamp",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "long_signal",
        "short_signal",
    ]
    if args.funding:
        columns += ["funding_rate", "mark_price"]
    frame[columns].to_csv(args.output, index=False)
    print(f"wrote {args.output} rows={len(frame)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
