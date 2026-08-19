#!/usr/bin/env python3
"""Validate ETH multi-timeframe historical data without modifying the source dataset."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd


TIMEFRAME_MINUTES = {"1m": 1, "15m": 15, "1h": 60, "8h": 480, "1d": 1440}
REQUIRED_COLUMNS = {"timestamp", "open", "high", "low", "close", "volume"}


def _find_file(root: Path, timeframe: str) -> Path | None:
    candidates = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".csv", ".feather", ".parquet"}
        and re.search(rf"(?:^|[-_.]){re.escape(timeframe)}(?:[-_.]|$)", path.name.lower())
    )
    return candidates[0] if candidates else None


def _read(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".feather":
        return pd.read_feather(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"unsupported data file: {path}")


def _parse_timestamp(values: pd.Series) -> pd.Series:
    """Parse ISO timestamps and common epoch-second/millisecond feeds safely."""
    numeric = pd.to_numeric(values, errors="coerce")
    finite_numeric = numeric.dropna()
    if not finite_numeric.empty and float(finite_numeric.abs().median()) >= 1e11:
        return pd.to_datetime(numeric, unit="ms", utc=True, errors="coerce")
    if not finite_numeric.empty and float(finite_numeric.abs().median()) >= 1e9:
        return pd.to_datetime(numeric, unit="s", utc=True, errors="coerce")
    return pd.to_datetime(values, utc=True, errors="coerce")


def _validate(path: Path, timeframe: str) -> dict[str, object]:
    interval = TIMEFRAME_MINUTES[timeframe]
    frame = _read(path)
    columns = {str(column).lower(): column for column in frame.columns}
    missing = sorted(REQUIRED_COLUMNS - set(columns))
    result: dict[str, object] = {
        "timeframe": timeframe,
        "path": str(path),
        "rows": int(len(frame)),
        "missing_columns": missing,
        "duplicate_timestamps": 0,
        "non_monotonic": False,
        "gaps": 0,
        "invalid_ohlcv_rows": 0,
        "non_finite_rows": 0,
        "status": "FAIL",
    }
    if missing:
        return result
    renamed = frame.rename(columns={value: key for key, value in columns.items()})
    timestamp = _parse_timestamp(renamed["timestamp"])
    numeric = renamed[["open", "high", "low", "close", "volume"]].apply(
        pd.to_numeric, errors="coerce"
    )
    finite = numeric.notna().all(axis=1)
    result["invalid_timestamp_rows"] = int(timestamp.isna().sum())
    result["non_finite_rows"] = int((~finite).sum())
    result["duplicate_timestamps"] = int(timestamp.duplicated().sum())
    result["non_monotonic"] = bool(not timestamp.is_monotonic_increasing)
    if len(timestamp) > 1:
        deltas = timestamp.diff().dropna().dt.total_seconds().div(60.0)
        result["gaps"] = int((deltas > interval).sum())
        result["wrong_interval_rows"] = int((deltas != interval).sum())
    else:
        result["wrong_interval_rows"] = 0
    ohlc = numeric
    valid_ohlc = (
        finite
        & (ohlc["high"] >= ohlc[["open", "close", "low"]].max(axis=1))
        & (ohlc["low"] <= ohlc[["open", "close", "high"]].min(axis=1))
        & (ohlc["volume"] >= 0)
        & (ohlc["open"] > 0)
        & (ohlc["high"] > 0)
        & (ohlc["low"] > 0)
        & (ohlc["close"] > 0)
    )
    result["invalid_ohlcv_rows"] = int((~valid_ohlc).sum())
    if len(timestamp):
        result["start"] = timestamp.min().isoformat()
        result["end"] = timestamp.max().isoformat()
        result["duration_days"] = float(
            (timestamp.max() - timestamp.min()).total_seconds() / 86400.0
        )
    result["status"] = "PASS" if all(
        result[key] in (0, False) for key in (
            "duplicate_timestamps",
            "non_monotonic",
            "gaps",
            "wrong_interval_rows",
            "invalid_ohlcv_rows",
            "non_finite_rows",
            "invalid_timestamp_rows",
        )
    ) else "FAIL"
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-days", type=float, default=700.0)
    args = parser.parse_args(argv)
    root = args.data_root.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"data root does not exist: {root}")
    results: list[dict[str, object]] = []
    for timeframe in TIMEFRAME_MINUTES:
        path = _find_file(root, timeframe)
        if path is None:
            results.append({"timeframe": timeframe, "status": "FAIL", "reason": "file missing"})
            continue
        results.append(_validate(path, timeframe))
    for row in results:
        if row.get("status") == "PASS" and float(row.get("duration_days", 0.0)) < args.min_days:
            row["status"] = "FAIL"
            row["reason"] = f"duration shorter than {args.min_days} days"
    payload = {
        "schema": "hedge-eth-two-year-deep-validation-v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "data_root": str(root),
        "minimum_days": args.min_days,
        "timeframes": results,
        "status": "PASS" if results and all(row.get("status") == "PASS" for row in results) else "FAIL",
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
