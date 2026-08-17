#!/usr/bin/env python3
"""Download public Binance USD-M monthly klines without API credentials."""

from __future__ import annotations

import argparse
from datetime import date
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen
from zipfile import ZipFile

import pandas as pd


COLUMNS = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_base",
    "taker_quote",
    "ignore",
)


def months(start: date, end: date):
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        yield year, month
        month += 1
        if month == 13:
            year, month = year + 1, 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="ETHUSDT")
    parser.add_argument("--timeframes", nargs="+", default=["5m", "1h"])
    parser.add_argument("--start", default="2024-08-01")
    parser.add_argument("--end", default="2026-07-31")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
    args.output.mkdir(parents=True, exist_ok=True)
    for timeframe in args.timeframes:
        frames = []
        for year, month in months(start, end):
            name = f"{args.symbol}-{timeframe}-{year:04d}-{month:02d}.zip"
            url = f"https://data.binance.vision/data/futures/um/monthly/klines/{args.symbol}/{timeframe}/{name}"
            try:
                with urlopen(url, timeout=60) as response:
                    archive = response.read()
            except HTTPError as exc:
                if exc.code == 404:
                    continue
                raise
            with ZipFile(BytesIO(archive)) as zipped:
                with zipped.open(zipped.namelist()[0]) as handle:
                    frame = pd.read_csv(handle, header=None, names=COLUMNS)
            if not frame.empty and str(frame.iloc[0]["open_time"]).strip().lower() == "open_time":
                frame = frame.iloc[1:].reset_index(drop=True)
            frames.append(frame)
            print(f"downloaded {name}")
        if not frames:
            raise RuntimeError(f"no public data found for {timeframe}")
        result = pd.concat(frames, ignore_index=True)
        result["open_time"] = pd.to_numeric(result["open_time"], errors="raise")
        result["date"] = pd.to_datetime(result["open_time"], unit="ms", utc=True)
        result = result[["date", "open", "high", "low", "close", "volume"]]
        for column in ("open", "high", "low", "close", "volume"):
            result[column] = pd.to_numeric(result[column], errors="raise")
        symbol_name = args.symbol.upper().removesuffix("USDT") + "_USDT_USDT"
        target = args.output / f"{symbol_name}-{timeframe}-futures.feather"
        result.to_feather(target)
        print(f"wrote {target} rows={len(result)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
