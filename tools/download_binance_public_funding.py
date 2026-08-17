#!/usr/bin/env python3
"""Download public Binance USD-M funding-rate history."""

import argparse
from datetime import date
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen
from zipfile import ZipFile

import pandas as pd
from download_binance_public_klines import months


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="ETHUSDT")
    parser.add_argument("--start", default="2024-08-01")
    parser.add_argument("--end", default="2026-07-31")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frames = []
    for year, month in months(date.fromisoformat(args.start), date.fromisoformat(args.end)):
        name = f"{args.symbol}-fundingRate-{year:04d}-{month:02d}.zip"
        url = f"https://data.binance.vision/data/futures/um/monthly/fundingRate/{args.symbol}/{name}"
        try:
            with urlopen(url, timeout=60) as response:
                archive = response.read()
        except HTTPError as exc:
            if exc.code == 404:
                continue
            raise
        with ZipFile(BytesIO(archive)) as zipped:
            frames.append(pd.read_csv(zipped.open(zipped.namelist()[0])))
    if not frames:
        raise RuntimeError("no public funding data found for the requested period")
    result = pd.concat(frames, ignore_index=True)
    timestamp = "calc_time" if "calc_time" in result else "fundingTime"
    rate = "last_funding_rate" if "last_funding_rate" in result else "fundingRate"
    # Binance settlement timestamps may include a few milliseconds of publication
    # jitter; bind the charge to its canonical five-minute decision slot.
    result["date"] = pd.to_datetime(
        pd.to_numeric(result[timestamp]),
        unit="ms",
        utc=True,
    ).dt.floor("5min")
    result["funding_rate"] = pd.to_numeric(result[rate])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result[["date", "funding_rate"]].drop_duplicates("date").to_feather(args.output)
    print(f"wrote {args.output} rows={len(result)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
