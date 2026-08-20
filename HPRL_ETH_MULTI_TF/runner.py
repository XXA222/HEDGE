from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from strategies.common import (
    ACTION_KWARGS,
    BASE_REWARD_KWARGS,
    COST_KWARGS,
    PERIODS_PER_YEAR,
    TRAIN_STEPS,
    WINDOW_STEPS,
)

TIMEFRAMES = ("1m", "5m", "15m", "1h", "8h", "1d")
STRATEGY_MODULES = (
    "fast_td3_strategy",
    "fast_dsac_strategy",
    "simba_sac_strategy",
    "xqc_strategy",
    "rebrac_v2_strategy",
)
DEFAULT_START = "2024-08-19T00:00:00Z"
DEFAULT_END = "2026-08-19T00:00:00Z"
BINANCE_FAPI = "https://fapi.binance.com"


def utc_now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_utc_ms(text: str) -> int:
    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def interval_ms(tf: str) -> int:
    mapping = {
        "1m": 60_000,
        "5m": 5 * 60_000,
        "15m": 15 * 60_000,
        "1h": 60 * 60_000,
        "8h": 8 * 60 * 60_000,
        "1d": 24 * 60 * 60_000,
    }
    return mapping[tf]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def json_dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def safe_float(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def binance_json(path: str, params: dict, retries: int = 9):
    query = urllib.parse.urlencode(params)
    url = f"{BINANCE_FAPI}{path}?{query}"
    last = None
    for attempt in range(retries):
        req = urllib.request.Request(url, headers={"User-Agent": "HEDGE-HPRL-ETH-suite/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code not in (418, 429, 500, 502, 503, 504):
                raise
            retry_after = exc.headers.get("Retry-After")
            delay = float(retry_after) if retry_after else min(30.0, 1.5 ** attempt)
        except Exception as exc:
            last = exc
            delay = min(30.0, 1.5 ** attempt)
        time.sleep(delay)
    raise RuntimeError(f"Binance request failed after retries: {url}: {last}")



def _timeframe_name_match(name: str, tf: str) -> bool:
    lowered = name.lower()
    return re.search(rf"(^|[-_.]){re.escape(tf.lower())}([-_.]|$)", lowered) is not None


def _read_local_market_file(path: Path) -> pd.DataFrame:
    name = path.name.lower()
    if name.endswith(".feather"):
        return pd.read_feather(path)
    if name.endswith(".parquet"):
        return pd.read_parquet(path)
    if name.endswith(".csv") or name.endswith(".csv.gz"):
        return pd.read_csv(path)
    if name.endswith(".json") or name.endswith(".json.gz"):
        try:
            return pd.read_json(path)
        except ValueError:
            with path.open("rt", encoding="utf-8") as handle:
                raw = json.load(handle)
            return pd.DataFrame(raw)
    raise ValueError(f"Unsupported local market file: {path}")


def _normalize_ohlcv_frame(frame: pd.DataFrame, source: Path) -> pd.DataFrame:
    df = frame.copy()

    # Freqtrade JSON can be positional: date/open/high/low/close/volume.
    if all(isinstance(c, (int, np.integer)) for c in df.columns) and len(df.columns) >= 6:
        rename = {
            df.columns[0]: "date",
            df.columns[1]: "open",
            df.columns[2]: "high",
            df.columns[3]: "low",
            df.columns[4]: "close",
            df.columns[5]: "volume",
        }
        df = df.rename(columns=rename)

    aliases = {
        "timestamp": "date",
        "time": "date",
        "datetime": "date",
        "open_time": "date",
        "quoteVolume": "quote_volume",
        "quote_asset_volume": "quote_volume",
        "trade_count": "trades",
        "count": "trades",
        "taker_buy_quote_asset_volume": "taker_buy_quote",
    }
    for old, new in aliases.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    required = ("date", "open", "high", "low", "close", "volume")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{source} is missing OHLCV columns: {missing}; columns={list(df.columns)}")

    date = df["date"]
    if pd.api.types.is_datetime64_any_dtype(date):
        open_time = pd.to_datetime(date, utc=True).astype("int64") // 1_000_000
    else:
        numeric = pd.to_numeric(date, errors="coerce")
        # seconds vs milliseconds vs nanoseconds
        finite = numeric.dropna()
        if finite.empty:
            parsed = pd.to_datetime(date, utc=True, errors="coerce")
            open_time = parsed.astype("int64") // 1_000_000
        else:
            med = float(finite.abs().median())
            if med < 1e11:
                open_time = numeric * 1000.0
            elif med > 1e15:
                open_time = numeric / 1_000_000.0
            else:
                open_time = numeric

    out = pd.DataFrame({
        "open_time": pd.to_numeric(open_time, errors="coerce"),
        "open": pd.to_numeric(df["open"], errors="coerce"),
        "high": pd.to_numeric(df["high"], errors="coerce"),
        "low": pd.to_numeric(df["low"], errors="coerce"),
        "close": pd.to_numeric(df["close"], errors="coerce"),
        "volume": pd.to_numeric(df["volume"], errors="coerce"),
    })

    if "quote_volume" in df.columns:
        out["quote_volume"] = pd.to_numeric(df["quote_volume"], errors="coerce")
    else:
        # Conservative proxy sufficient for HPRL market-impact availability.
        out["quote_volume"] = out["volume"] * out["close"]

    if "trades" in df.columns:
        out["trades"] = pd.to_numeric(df["trades"], errors="coerce")
    else:
        out["trades"] = 0.0

    if "taker_buy_quote" in df.columns:
        out["taker_buy_quote"] = pd.to_numeric(df["taker_buy_quote"], errors="coerce")
    else:
        # Neutral fallback: 50% taker-buy share. This feature is explicitly visible
        # in metadata so a local file missing microstructure fields is not hidden.
        out["taker_buy_quote"] = out["quote_volume"] * 0.5

    out["close_time"] = out["open_time"]
    out["taker_buy_base"] = 0.0
    out["ignore"] = 0.0
    out = out.dropna(subset=["open_time", "open", "high", "low", "close", "volume"])
    out["open_time"] = out["open_time"].astype("int64")
    return out.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)


def discover_local_ohlcv(repo_root: Path, tf: str, start_ms: int, end_ms: int):
    data_root = repo_root / "user_data" / "data"
    if not data_root.exists():
        return None, None

    extensions = (".feather", ".parquet", ".json", ".json.gz", ".csv", ".csv.gz")
    candidates = []
    for path in data_root.rglob("*"):
        if not path.is_file():
            continue
        low = path.name.lower()
        if not low.endswith(extensions):
            continue
        if "eth" not in low or not _timeframe_name_match(low, tf):
            continue
        if "funding" in low or "mark" in low or "index" in low or "premium" in low:
            continue
        try:
            normalized = _normalize_ohlcv_frame(_read_local_market_file(path), path)
            clipped = normalized[
                (normalized["open_time"] >= start_ms) & (normalized["open_time"] < end_ms)
            ].copy()
            if len(clipped) < 100:
                continue
            coverage_start = int(clipped["open_time"].iloc[0])
            coverage_end = int(clipped["open_time"].iloc[-1])
            score = len(clipped)
            # Prefer files that cover both boundaries of the requested window.
            if int(normalized["open_time"].iloc[0]) <= start_ms:
                score += 10_000_000
            if int(normalized["open_time"].iloc[-1]) >= end_ms - interval_ms(tf):
                score += 10_000_000
            if "futures" in str(path).lower():
                score += 1_000_000
            candidates.append((score, path, clipped, coverage_start, coverage_end))
        except Exception:
            continue

    if not candidates:
        return None, None
    candidates.sort(key=lambda item: item[0], reverse=True)
    best = candidates[0]
    meta = {
        "source": "existing_freqtrade_data",
        "path": str(best[1]),
        "rows_in_requested_window": int(len(best[2])),
        "coverage_start_ms": best[3],
        "coverage_end_ms": best[4],
        "candidate_count": len(candidates),
    }
    return best[2], meta


def _normalize_funding_frame(frame: pd.DataFrame, source: Path) -> pd.DataFrame:
    df = frame.copy()
    if all(isinstance(c, (int, np.integer)) for c in df.columns) and len(df.columns) >= 2:
        df = df.rename(columns={df.columns[0]: "date", df.columns[1]: "funding_rate"})

    date_col = next(
        (c for c in ("fundingTime", "date", "timestamp", "time", "datetime") if c in df.columns),
        None,
    )
    rate_col = next(
        (c for c in ("fundingRate", "funding_rate", "rate", "close") if c in df.columns),
        None,
    )
    if date_col is None or rate_col is None:
        raise ValueError(f"{source} does not expose funding timestamp/rate columns")

    date = df[date_col]
    if pd.api.types.is_datetime64_any_dtype(date):
        t = pd.to_datetime(date, utc=True).astype("int64") // 1_000_000
    else:
        numeric = pd.to_numeric(date, errors="coerce")
        finite = numeric.dropna()
        if finite.empty:
            parsed = pd.to_datetime(date, utc=True, errors="coerce")
            t = parsed.astype("int64") // 1_000_000
        else:
            med = float(finite.abs().median())
            if med < 1e11:
                t = numeric * 1000.0
            elif med > 1e15:
                t = numeric / 1_000_000.0
            else:
                t = numeric

    out = pd.DataFrame({
        "fundingTime": pd.to_numeric(t, errors="coerce"),
        "fundingRate": pd.to_numeric(df[rate_col], errors="coerce"),
    }).dropna()
    out["fundingTime"] = out["fundingTime"].astype("int64")
    return out.drop_duplicates("fundingTime").sort_values("fundingTime").reset_index(drop=True)


def discover_local_funding(repo_root: Path, start_ms: int, end_ms: int):
    data_root = repo_root / "user_data" / "data"
    if not data_root.exists():
        return None, None
    extensions = (".feather", ".parquet", ".json", ".json.gz", ".csv", ".csv.gz")
    candidates = []
    for path in data_root.rglob("*"):
        if not path.is_file():
            continue
        low = path.name.lower()
        if not low.endswith(extensions) or "eth" not in low or "funding" not in low:
            continue
        try:
            normalized = _normalize_funding_frame(_read_local_market_file(path), path)
            clipped = normalized[
                (normalized["fundingTime"] >= start_ms) & (normalized["fundingTime"] < end_ms)
            ].copy()
            if len(clipped) < 2:
                continue
            score = len(clipped) + (1_000_000 if "futures" in str(path).lower() else 0)
            candidates.append((score, path, clipped))
        except Exception:
            continue
    if not candidates:
        return None, None
    candidates.sort(key=lambda item: item[0], reverse=True)
    best = candidates[0]
    return best[2], {
        "source": "existing_freqtrade_funding",
        "path": str(best[1]),
        "rows": int(len(best[2])),
        "candidate_count": len(candidates),
    }

def download_klines(tf: str, start_ms: int, end_ms: int, raw_path: Path) -> pd.DataFrame:
    if raw_path.exists():
        df = pd.read_csv(raw_path, compression="gzip")
        if len(df) > 2:
            return df

    cols = [
        "open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore",
    ]
    rows = []
    cursor = start_ms
    step = interval_ms(tf)
    while cursor < end_ms:
        batch = binance_json(
            "/fapi/v1/klines",
            {
                "symbol": "ETHUSDT",
                "interval": tf,
                "startTime": cursor,
                "endTime": end_ms - 1,
                "limit": 1000,
            },
        )
        if not batch:
            break
        rows.extend(batch)
        last_open = int(batch[-1][0])
        next_cursor = last_open + step
        if next_cursor <= cursor:
            raise RuntimeError(f"kline pagination stalled at {cursor}")
        cursor = next_cursor
        # Conservative pacing. The first 1m download is request-heavy.
        time.sleep(0.17)

    if not rows:
        raise RuntimeError(f"No ETHUSDT {tf} klines returned")
    df = pd.DataFrame(rows, columns=cols)
    df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce")
    df = df.dropna(subset=["open_time"]).drop_duplicates("open_time").sort_values("open_time")
    df = df[(df["open_time"] >= start_ms) & (df["open_time"] < end_ms)].copy()
    numeric = [c for c in cols if c != "ignore"]
    for c in numeric:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if len(df) < 100:
        raise RuntimeError(f"Too few ETHUSDT {tf} rows: {len(df)}")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(raw_path, index=False, compression="gzip")
    return df


def download_funding(start_ms: int, end_ms: int, path: Path) -> pd.DataFrame:
    if path.exists():
        df = pd.read_csv(path, compression="gzip")
        if len(df) > 2:
            return df

    rows = []
    cursor = start_ms
    while cursor < end_ms:
        batch = binance_json(
            "/fapi/v1/fundingRate",
            {
                "symbol": "ETHUSDT",
                "startTime": cursor,
                "endTime": end_ms - 1,
                "limit": 1000,
            },
        )
        if not batch:
            break
        rows.extend(batch)
        last_t = int(batch[-1]["fundingTime"])
        if last_t + 1 <= cursor:
            raise RuntimeError("funding pagination stalled")
        cursor = last_t + 1
        time.sleep(0.2)

    if not rows:
        raise RuntimeError("No ETHUSDT funding rows returned")
    df = pd.DataFrame(rows)
    df["fundingTime"] = pd.to_numeric(df["fundingTime"], errors="coerce")
    df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    df = df.dropna(subset=["fundingTime", "fundingRate"]).drop_duplicates("fundingTime")
    df = df.sort_values("fundingTime")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, compression="gzip")
    return df


def map_funding_to_bars(open_times: np.ndarray, funding: pd.DataFrame) -> np.ndarray:
    out = np.zeros(len(open_times), dtype=np.float64)
    # Settlement at an exact bar boundary is charged to the bar immediately before it,
    # i.e. to the position held into the settlement, avoiding use of future funding.
    for t, r in zip(funding["fundingTime"].to_numpy(), funding["fundingRate"].to_numpy()):
        i = int(np.searchsorted(open_times, int(t), side="left") - 1)
        if 0 <= i < len(out):
            out[i] += float(r)
    return out


def engineer_features(df: pd.DataFrame, funding_step: np.ndarray):
    open_ = df["open"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    quote = df["quote_volume"].astype(float).clip(lower=1.0)
    trades = df["trades"].astype(float).clip(lower=0.0)
    taker_quote = df["taker_buy_quote"].astype(float).clip(lower=0.0)

    log_close = np.log(close)
    lr = log_close.diff()
    feats = {}

    for lag in (1, 2, 3, 6, 12):
        feats[f"logret_{lag}"] = log_close.diff(lag)
    for lag in (6, 12, 24):
        feats[f"momentum_{lag}"] = close / close.shift(lag) - 1.0
    for win in (6, 12, 24):
        feats[f"vol_{win}"] = lr.rolling(win).std(ddof=0)

    feats["range_pct"] = (high - low) / close
    feats["body_pct"] = (close - open_) / open_.replace(0, np.nan)
    feats["upper_wick_pct"] = (high - np.maximum(open_, close)) / close
    feats["lower_wick_pct"] = (np.minimum(open_, close) - low) / close

    for span in (8, 21, 55):
        ema = close.ewm(span=span, adjust=False, min_periods=span).mean()
        feats[f"ema_spread_{span}"] = close / ema - 1.0

    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    feats["rsi14"] = (100 - 100 / (1 + rs)) / 50.0 - 1.0

    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    feats["atr14_pct"] = atr / close

    lq = np.log1p(quote)
    feats["quote_volume_z24"] = (lq - lq.rolling(24).mean()) / lq.rolling(24).std(ddof=0)
    lt = np.log1p(trades)
    feats["trades_z24"] = (lt - lt.rolling(24).mean()) / lt.rolling(24).std(ddof=0)
    feats["taker_buy_share"] = taker_quote / quote - 0.5
    feats["quote_volume_change"] = lq.diff()

    funding_s = pd.Series(funding_step, index=df.index, dtype=float)
    feats["funding_prev"] = funding_s.shift(1)
    feats["funding_mean_6"] = funding_s.shift(1).rolling(6, min_periods=1).mean()

    ts = pd.to_datetime(df["open_time"].astype("int64"), unit="ms", utc=True)
    hour = ts.dt.hour.to_numpy()
    dow = ts.dt.dayofweek.to_numpy()
    feats["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    feats["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    feats["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    feats["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

    feat_df = pd.DataFrame(feats, index=df.index).replace([np.inf, -np.inf], np.nan)
    fwd = close.shift(-1) / close - 1.0
    available = quote.to_numpy(dtype=np.float64)
    open_times = df["open_time"].to_numpy(dtype=np.int64)
    close_np = close.to_numpy(dtype=np.float64)

    mask = feat_df.notna().all(axis=1).to_numpy() & np.isfinite(fwd.to_numpy())
    x = feat_df.to_numpy(dtype=np.float32)[mask]
    y = fwd.to_numpy(dtype=np.float32)[mask]
    fr = funding_step.astype(np.float32)[mask]
    av = np.maximum(available[mask], 1.0).astype(np.float32)
    ot = open_times[mask]
    cp = close_np[mask]
    return x, y, fr, av, ot, cp, list(feat_df.columns)


def prepare_timeframe(args) -> int:
    cache = Path(args.cache_dir).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    start_ms, end_ms = parse_utc_ms(args.start), parse_utc_ms(args.end)
    tf = args.timeframe
    raw_path = cache / f"ETHUSDT_{tf}_{start_ms}_{end_ms}.csv.gz"
    funding_path = cache / f"ETHUSDT_funding_{start_ms}_{end_ms}.csv.gz"
    prepared_path = cache / f"ETHUSDT_{tf}_{start_ms}_{end_ms}_prepared.npz"
    meta_path = prepared_path.with_suffix(".json")

    if prepared_path.exists() and meta_path.exists() and not args.force:
        print(str(prepared_path))
        return 0

    repo_root = Path(args.repo_root).resolve()
    local_df, local_meta = discover_local_ohlcv(repo_root, tf, start_ms, end_ms)
    if local_df is not None:
        df = local_df
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(raw_path, index=False, compression="gzip")
        print(f"[LOCAL-DATA] {tf}: {local_meta['path']} rows={len(df)}")
    else:
        df = download_klines(tf, start_ms, end_ms, raw_path)
        local_meta = {"source": "binance_download", "path": str(raw_path)}
        print(f"[DOWNLOAD] {tf}: Binance USD-M klines rows={len(df)}")

    local_funding, funding_meta = discover_local_funding(repo_root, start_ms, end_ms)
    if local_funding is not None:
        funding = local_funding
        funding_path.parent.mkdir(parents=True, exist_ok=True)
        funding.to_csv(funding_path, index=False, compression="gzip")
        print(f"[LOCAL-FUNDING] {funding_meta['path']} rows={len(funding)}")
    else:
        try:
            funding = download_funding(start_ms, end_ms, funding_path)
            funding_meta = {"source": "binance_download", "path": str(funding_path)}
            print(f"[DOWNLOAD] funding rows={len(funding)}")
        except Exception as exc:
            # Funding is valuable but must not turn a valid OHLCV period into a hard
            # failure. Preserve the limitation explicitly in metadata/results.
            funding = pd.DataFrame(columns=["fundingTime", "fundingRate"])
            funding_meta = {
                "source": "missing_zero_fallback",
                "error": repr(exc),
                "rows": 0,
            }
            print(f"[WARNING] funding unavailable; using zeros: {exc}")

    funding_step = (
        map_funding_to_bars(df["open_time"].to_numpy(dtype=np.int64), funding)
        if len(funding)
        else np.zeros(len(df), dtype=np.float64)
    )
    x, y, fr, av, ot, cp, names = engineer_features(df, funding_step)

    np.savez(
        prepared_path,
        features=x,
        forward_returns=y,
        funding_rates=fr,
        available_notional=av,
        open_time=ot,
        close=cp,
        feature_names=np.asarray(names),
    )
    meta = {
        "symbol": "ETHUSDT",
        "market": "Binance USD-M perpetual",
        "timeframe": tf,
        "requested_start": args.start,
        "requested_end_exclusive": args.end,
        "rows_raw": int(len(df)),
        "rows_prepared": int(len(x)),
        "first_open_time_ms": int(ot[0]),
        "last_open_time_ms": int(ot[-1]),
        "features": names,
        "raw_sha256": sha256_file(raw_path),
        "prepared_sha256": sha256_file(prepared_path),
        "funding_sha256": sha256_file(funding_path),
        "funding_mapping": "settlement is applied to immediately preceding bar",
        "ohlcv_source": local_meta,
        "funding_source": funding_meta,
    }
    json_dump(meta_path, meta)
    print(str(prepared_path))
    return 0


def load_spec(module_name: str):
    mod = importlib.import_module(f"strategies.{module_name}")
    return mod.SPEC


def import_hprl(repo_root: Path):
    repo_root = repo_root.resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from freqtrade.hedge.hprl.action_space import configure_agent_action_levels
    from freqtrade.hedge.hprl.config import (
        HPRLActionConfig,
        HPRLCostConfig,
        HPRLEnvironmentConfig,
        HPRLMemoryConfig,
        HPRLRewardConfig,
        HPRLTrainingConfig,
    )
    from freqtrade.hedge.hprl.data import TensorMarketDataset
    from freqtrade.hedge.hprl.env import VectorizedHedgeEnv
    from freqtrade.hedge.hprl.evaluation import evaluate_trading, walk_forward_folds
    from freqtrade.hedge.hprl.registry import create_agent
    from freqtrade.hedge.hprl.replay import TensorReplayBuffer
    from freqtrade.hedge.hprl.trainer import DiscountedReturnNormalizer, OfflineTrainer

    return {
        "configure_agent_action_levels": configure_agent_action_levels,
        "HPRLActionConfig": HPRLActionConfig,
        "HPRLCostConfig": HPRLCostConfig,
        "HPRLEnvironmentConfig": HPRLEnvironmentConfig,
        "HPRLMemoryConfig": HPRLMemoryConfig,
        "HPRLRewardConfig": HPRLRewardConfig,
        "HPRLTrainingConfig": HPRLTrainingConfig,
        "TensorMarketDataset": TensorMarketDataset,
        "VectorizedHedgeEnv": VectorizedHedgeEnv,
        "evaluate_trading": evaluate_trading,
        "walk_forward_folds": walk_forward_folds,
        "create_agent": create_agent,
        "TensorReplayBuffer": TensorReplayBuffer,
        "DiscountedReturnNormalizer": DiscountedReturnNormalizer,
        "OfflineTrainer": OfflineTrainer,
    }


def make_folds(length: int, requested: int, walk_forward_folds):
    purge = 1
    if requested <= 1:
        train = int(length * 0.68)
        val = int(length * 0.10)
        test = length - train - val - 2 * purge
        return walk_forward_folds(
            length, train=train, validation=val, test=test, step=max(1, test), purge=purge
        )[:1]

    train = int(length * 0.55)
    val = int(length * 0.10)
    test = int(length * 0.10)
    max_start = max(1, length - (train + val + test + 2 * purge))
    step = max(1, max_start // (requested - 1))
    folds = walk_forward_folds(
        length, train=train, validation=val, test=test, step=step, purge=purge
    )
    if len(folds) <= requested:
        return folds
    # Evenly choose requested folds, preserving the final fold.
    idx = np.linspace(0, len(folds) - 1, requested).round().astype(int)
    return tuple(folds[i] for i in sorted(set(idx)))


def fit_scale(x: np.ndarray):
    mean = x.mean(axis=0, dtype=np.float64)
    std = x.std(axis=0, dtype=np.float64)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def apply_scale(x: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return np.clip((x - mean) / std, -8.0, 8.0).astype(np.float32)


def make_dataset(api, torch, x, y, fr, av):
    TensorMarketDataset = api["TensorMarketDataset"]
    return TensorMarketDataset(
        features=torch.as_tensor(x[:, None, :], dtype=torch.float32),
        forward_returns=torch.as_tensor(y[:, None], dtype=torch.float32),
        funding_rates=torch.as_tensor(fr[:, None], dtype=torch.float32),
        available_notional=torch.as_tensor(av[:, None], dtype=torch.float32),
        symbols=("ETHUSDT",),
    ).validate()


def make_configs(api, spec, tf, device, parallel_envs, compile_mode, expected_updates, info_mode):
    reward = dict(BASE_REWARD_KWARGS)
    reward.update(dict(spec.reward_overrides))
    action_cfg = api["HPRLActionConfig"](**ACTION_KWARGS)
    cost_cfg = api["HPRLCostConfig"](**COST_KWARGS)
    reward_cfg = api["HPRLRewardConfig"](**reward)
    env_cfg = api["HPRLEnvironmentConfig"](
        initial_equity=1000.0,
        parallel_envs=parallel_envs,
        annualization_periods=PERIODS_PER_YEAR[tf],
        cvar_alpha=0.05,
        terminate_equity_ratio=0.20,
        runtime_checks=False,
        info_mode=info_mode,
        action=action_cfg,
        costs=cost_cfg,
        reward=reward_cfg,
    )
    train_cfg = api["HPRLTrainingConfig"](
        algorithm=spec.algorithm,
        seed=42,
        device=device,
        replay_device="same",
        batch_size=spec.batch_size,
        replay_capacity=spec.replay_capacity,
        warmup_steps=spec.warmup_transitions,
        gradient_steps=1,
        gamma=spec.gamma,
        tau=spec.tau,
        learning_rate=spec.learning_rate,
        hidden_dim=spec.hidden_dim,
        hidden_depth=spec.hidden_depth,
        gradient_clip_norm=spec.gradient_clip_norm,
        mixed_precision=False,
        compile_mode=compile_mode,
        expected_updates=expected_updates,
        metrics_interval=max(100, expected_updates // 10),
        tier_entropy_target_fraction=spec.tier_entropy_target_fraction,
        runtime_checks=False,
    )
    mem_cfg = api["HPRLMemoryConfig"](
        dataset_mode="auto",
        dataset_window_steps=16_384,
        dataset_gpu_fraction=0.20,
        replay_gpu_fraction=0.30,
        release_offline_source_after_tensorize=True,
    )
    return action_cfg, env_cfg, train_cfg, mem_cfg


def train_online_windowed(api, torch, agent, dataset, env_cfg, train_cfg, mem_cfg, steps, segment_steps):
    Env = api["VectorizedHedgeEnv"]
    Replay = api["TensorReplayBuffer"]
    Normalizer = api["DiscountedReturnNormalizer"]
    env = Env(dataset, env_cfg, device=train_cfg.device, memory_config=mem_cfg)
    buffer = Replay(
        train_cfg.replay_capacity,
        env.observation_dim,
        env.action_dim,
        device=str(agent.device),
        pin_memory=False,
        validate_inputs=False,
    )
    normalizer = None
    if getattr(agent, "reward_normalization", None) == "return_std":
        normalizer = Normalizer(
            env.envs, train_cfg.gamma, device=agent.device, validate_inputs=False
        )

    rng = np.random.default_rng(train_cfg.seed)
    transition_count = 0
    update_count = 0
    last_metrics = {}
    obs = None
    seg_left = 0

    try:
        for decision_step in range(int(steps)):
            if obs is None or seg_left <= 0:
                max_start = max(0, env.market.time_steps - segment_steps - 2)
                start = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
                obs, _ = env.reset(start_index=start)
                seg_left = min(segment_steps, env.market.time_steps - start - 1)

            if transition_count < train_cfg.warmup_steps:
                action = env.sample_random_action()
            else:
                action = agent.act(obs, deterministic=False)

            step = env.step(action)
            seg_left -= 1
            done = torch.logical_or(step.terminated, step.truncated)
            if seg_left <= 0:
                done = torch.ones_like(done, dtype=torch.bool)

            replay_reward = step.reward
            if normalizer is not None:
                replay_reward = normalizer.normalize(step.reward, done)

            executed = step.info["executed_action"]
            buffer.add(obs, executed, replay_reward, step.observation, done)
            transition_count += env.envs
            obs = step.observation

            if len(buffer) >= train_cfg.batch_size and transition_count >= train_cfg.warmup_steps:
                batch = buffer.sample_reusable(train_cfg.batch_size)
                collect = update_count == 0 or (update_count + 1) % train_cfg.metrics_interval == 0
                metrics = agent.update(batch, collect_metrics=collect)
                if getattr(metrics, "values", None):
                    last_metrics = {
                        k: safe_float(v) for k, v in dict(metrics.values).items()
                    }
                update_count += 1

            if bool(step.info.get("time_done", False)) or bool(step.terminated.any().item()):
                obs = None
                seg_left = 0

        return {
            "decision_steps": int(steps),
            "transitions": int(transition_count),
            "updates": int(update_count),
            "last_metrics": last_metrics,
        }
    finally:
        try:
            buffer.release(aggressive=False)
        except Exception:
            pass
        try:
            if normalizer is not None:
                normalizer.release()
        except Exception:
            pass
        try:
            env.close(aggressive=False)
        except Exception:
            pass


class ArrayOfflineDataset:
    action_unit = "policy_code"

    def __init__(self, obs, action, reward, next_obs, done):
        self.obs = obs
        self.action = action
        self.reward = reward
        self.next_obs = next_obs
        self.done = done
        self.observation_dim = int(obs.shape[1])
        self.action_dim = int(action.shape[1])

    def __len__(self):
        return int(self.obs.shape[0])

    def tensors(self, device="cpu", *, chunk_rows=4096):
        import torch
        target = torch.device(device)
        return {
            "obs": torch.as_tensor(self.obs, dtype=torch.float32, device=target),
            "action": torch.as_tensor(self.action, dtype=torch.float32, device=target),
            "reward": torch.as_tensor(self.reward, dtype=torch.float32, device=target),
            "next_obs": torch.as_tensor(self.next_obs, dtype=torch.float32, device=target),
            "done": torch.as_tensor(self.done, dtype=torch.float32, device=target),
        }

    def release_source(self):
        self.obs = np.empty((0, self.observation_dim), dtype=np.float32)
        self.action = np.empty((0, self.action_dim), dtype=np.float32)
        self.reward = np.empty((0, 1), dtype=np.float32)
        self.next_obs = np.empty((0, self.observation_dim), dtype=np.float32)
        self.done = np.empty((0, 1), dtype=np.float32)


def build_offline_behavior_dataset(
    api, torch, dataset, env_cfg, train_cfg, mem_cfg, rows_target, segment_steps, signal_index
):
    Env = api["VectorizedHedgeEnv"]
    env = Env(dataset, env_cfg, device=train_cfg.device, memory_config=mem_cfg)
    rng = np.random.default_rng(train_cfg.seed + 101)
    obs_rows, action_rows, reward_rows, next_rows, done_rows = [], [], [], [], []
    obs = None
    seg_left = 0
    market_index = 0
    levels = env.config.action.level_count

    try:
        while sum(len(x) for x in obs_rows) < rows_target:
            if obs is None or seg_left <= 0:
                max_start = max(0, env.market.time_steps - segment_steps - 2)
                start = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
                obs, _ = env.reset(start_index=start)
                market_index = start
                seg_left = min(segment_steps, env.market.time_steps - start - 1)

            # Momentum-biased behavior with explicit random exploration.
            market_signal = float(dataset.features[market_index, 0, signal_index].item())
            strength = min(1.0, abs(math.tanh(market_signal / 2.0)))
            tier = int(round(strength * (levels - 1)))
            tier = max(0, min(levels - 1, tier))
            code = tier / float(levels - 1)
            base = np.array([code, 0.0] if market_signal >= 0 else [0.0, code], dtype=np.float32)
            acts = np.repeat(base[None, :], env.envs, axis=0)
            random_mask = rng.random(env.envs) < 0.22
            random_tier = rng.integers(0, levels, size=(env.envs, 2))
            acts[random_mask] = random_tier[random_mask] / float(levels - 1)
            action = torch.as_tensor(acts, dtype=torch.float32, device=agent_device(obs))

            step = env.step(action)
            seg_left -= 1
            done = torch.logical_or(step.terminated, step.truncated)
            if seg_left <= 0:
                done = torch.ones_like(done, dtype=torch.bool)

            executed = step.info["executed_action"]
            obs_rows.append(obs.detach().float().cpu().numpy())
            action_rows.append(executed.detach().float().cpu().numpy())
            reward_rows.append(step.reward.reshape(-1, 1).detach().float().cpu().numpy())
            next_rows.append(step.observation.detach().float().cpu().numpy())
            done_rows.append(done.reshape(-1, 1).detach().float().cpu().numpy())

            obs = step.observation
            market_index += 1
            if bool(step.info.get("time_done", False)) or bool(step.terminated.any().item()):
                obs = None
                seg_left = 0

            total = sum(a.shape[0] for a in obs_rows)
            if total >= rows_target:
                break

        obs_a = np.concatenate(obs_rows, axis=0)[:rows_target]
        act_a = np.concatenate(action_rows, axis=0)[:rows_target]
        rew_a = np.concatenate(reward_rows, axis=0)[:rows_target]
        nxt_a = np.concatenate(next_rows, axis=0)[:rows_target]
        done_a = np.concatenate(done_rows, axis=0)[:rows_target]
        return ArrayOfflineDataset(obs_a, act_a, rew_a, nxt_a, done_a)
    finally:
        try:
            env.close(aggressive=False)
        except Exception:
            pass


def agent_device(obs):
    return obs.device


def evaluate_agent(api, torch, agent, dataset, env_cfg, mem_cfg, timestamps, closes, tf, curve_path):
    Env = api["VectorizedHedgeEnv"]
    evaluate_trading = api["evaluate_trading"]
    env = Env(dataset, env_cfg, device=str(agent.device), memory_config=mem_cfg)
    obs, _ = env.reset()
    equity = [float(env_cfg.initial_equity)]
    turnover = 0.0
    fees = 0.0
    slippage = 0.0
    impact = 0.0
    funding_pnl = 0.0
    liquidations = 0
    trade_bars = 0
    sampled = []
    stride = max(1, (len(timestamps) - 1) // 50_000)
    prev_equity = float(env_cfg.initial_equity)

    try:
        for i in range(len(timestamps) - 1):
            action = agent.act(obs, deterministic=True)
            step = env.step(action)
            info = step.info
            eq = float(info["equity"][0].item())
            eq_safe = max(eq, 1e-12)
            equity.append(eq_safe)

            trn = float(info["turnover_ratio"][0].item())
            fee = float(info["fee_cost"][0].item())
            slip = float(info["slippage_cost"][0].item())
            imp = float(info["market_impact_cost"][0].item())
            f_ratio = float(info["funding_pnl_ratio"][0].item())
            turnover += trn
            fees += fee
            slippage += slip
            impact += imp
            funding_pnl += prev_equity * f_ratio
            if trn > 1e-10:
                trade_bars += 1
            prev_equity = eq_safe

            if i % stride == 0 or i == len(timestamps) - 2:
                margin = info["margin_position"][0, 0].detach().float().cpu().numpy()
                sampled.append({
                    "open_time_ms": int(timestamps[min(i + 1, len(timestamps) - 1)]),
                    "equity": eq_safe,
                    "long_margin": float(margin[0]),
                    "short_margin": float(margin[1]),
                    "turnover_ratio": trn,
                    "fee_cost": fee,
                    "slippage_cost": slip,
                    "impact_cost": imp,
                    "funding_pnl_ratio": f_ratio,
                })

            if bool(step.terminated[0].item()):
                liquidations += 1
                break
            if bool(info.get("time_done", False)):
                break
            obs = step.observation

        metrics = asdict(evaluate_trading(
            equity,
            periods_per_year=PERIODS_PER_YEAR[tf],
            alpha=0.05,
            turnover=turnover,
            fees=fees,
            funding=funding_pnl,
            liquidations=liquidations,
        ))
        rets = np.asarray(equity[1:], dtype=np.float64) / np.asarray(equity[:-1], dtype=np.float64) - 1
        metrics.update({
            "slippage": float(slippage),
            "market_impact": float(impact),
            "trade_bars": int(trade_bars),
            "evaluated_bars": int(len(equity) - 1),
            "win_bar_rate": float(np.mean(rets > 0)) if len(rets) else 0.0,
            "avg_bar_return": float(np.mean(rets)) if len(rets) else 0.0,
            "equity_final": float(equity[-1]),
            "buy_hold_return": float(closes[min(len(equity) - 1, len(closes) - 1)] / closes[0] - 1.0),
            "curve_sampling_stride": int(stride),
        })
        metrics["excess_vs_buy_hold"] = float(metrics["net_return"] - metrics["buy_hold_return"])
        pd.DataFrame(sampled).to_csv(curve_path, index=False)
        return metrics
    finally:
        try:
            env.close(aggressive=False)
        except Exception:
            pass


def worker(args) -> int:
    task_dir = Path(args.task_dir).resolve()
    task_dir.mkdir(parents=True, exist_ok=True)
    spec = load_spec(args.strategy)
    api = import_hprl(Path(args.repo_root))
    import torch

    cache = Path(args.cache_dir).resolve()
    start_ms, end_ms = parse_utc_ms(args.start), parse_utc_ms(args.end)
    prepared = cache / f"ETHUSDT_{args.timeframe}_{start_ms}_{end_ms}_prepared.npz"
    if not prepared.exists():
        raise FileNotFoundError(f"Prepared dataset missing: {prepared}")

    data = np.load(prepared, allow_pickle=False)
    x_raw = data["features"].astype(np.float32, copy=False)
    y = data["forward_returns"].astype(np.float32, copy=False)
    fr = data["funding_rates"].astype(np.float32, copy=False)
    av = data["available_notional"].astype(np.float32, copy=False)
    ts = data["open_time"].astype(np.int64, copy=False)
    closes = data["close"].astype(np.float64, copy=False)
    feature_names = [str(v) for v in data["feature_names"].tolist()]
    signal_index = feature_names.index("momentum_12") if "momentum_12" in feature_names else 0

    folds = make_folds(len(x_raw), args.folds, api["walk_forward_folds"])
    results = []
    fold_failures = []
    steps = TRAIN_STEPS[args.budget][args.timeframe]

    for fold_no, fold in enumerate(folds, start=1):
        fold_dir = task_dir / f"fold_{fold_no:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        try:
            mean, std = fit_scale(x_raw[fold.train_start:fold.train_end])
            x_scaled = apply_scale(x_raw, mean, std)

            train_slice = slice(fold.train_start, fold.train_end)
            test_slice = slice(fold.test_start, fold.test_end)

            train_ds = make_dataset(
                api, torch,
                x_scaled[train_slice], y[train_slice], fr[train_slice], av[train_slice],
            )
            test_ds = make_dataset(
                api, torch,
                x_scaled[test_slice], y[test_slice], fr[test_slice], av[test_slice],
            )

            action_cfg, train_env_cfg, train_cfg, mem_cfg = make_configs(
                api, spec, args.timeframe, args.device, args.parallel_envs,
                args.compile_mode, steps, "training",
            )
            _, eval_env_cfg, _, eval_mem_cfg = make_configs(
                api, spec, args.timeframe, args.device, 1,
                "off", 0, "full",
            )

            train_env_probe = api["VectorizedHedgeEnv"](
                train_ds, train_env_cfg, device=train_cfg.device, memory_config=mem_cfg
            )
            obs_dim, action_dim = train_env_probe.observation_dim, train_env_probe.action_dim
            train_env_probe.close()

            agent = api["create_agent"](
                spec.algorithm, obs_dim, action_dim, train_cfg, device=None
            )
            api["configure_agent_action_levels"](agent, action_cfg.level_count)

            if spec.algorithm == "rebrac_v2":
                rows_target = max(spec.batch_size * 8, min(60_000, steps * args.parallel_envs * 2))
                offline_ds = build_offline_behavior_dataset(
                    api, torch, train_ds, train_env_cfg, train_cfg, mem_cfg,
                    rows_target, WINDOW_STEPS[args.timeframe], signal_index,
                )
                trainer = api["OfflineTrainer"](
                    offline_ds, agent, train_cfg,
                    device=str(agent.device),
                    memory_config=mem_cfg,
                    action_config=action_cfg,
                )
                summary = trainer.run(steps)
                training_summary = asdict(summary)
            else:
                training_summary = train_online_windowed(
                    api, torch, agent, train_ds, train_env_cfg, train_cfg, mem_cfg,
                    steps, WINDOW_STEPS[args.timeframe],
                )

            curve_path = fold_dir / "equity_curve.csv"
            test_metrics = evaluate_agent(
                api, torch, agent, test_ds, eval_env_cfg, eval_mem_cfg,
                ts[test_slice], closes[test_slice], args.timeframe, curve_path,
            )

            fold_result = {
                "fold": fold_no,
                "train": {"start": fold.train_start, "end": fold.train_end},
                "validation": {"start": fold.validation_start, "end": fold.validation_end},
                "test": {"start": fold.test_start, "end": fold.test_end},
                "train_start_ms": int(ts[fold.train_start]),
                "train_end_ms": int(ts[fold.train_end - 1]),
                "test_start_ms": int(ts[fold.test_start]),
                "test_end_ms": int(ts[fold.test_end - 1]),
                "training_summary": training_summary,
                "test_metrics": test_metrics,
                "scale_mean": mean.tolist(),
                "scale_std": std.tolist(),
                "status": "SUCCESS",
            }
            json_dump(fold_dir / "metrics.json", fold_result)
            results.append(fold_result)

        except Exception as exc:
            tb = traceback.format_exc()
            failure = {
                "fold": fold_no,
                "error": repr(exc),
                "traceback": tb,
                "status": "FAILED",
            }
            json_dump(fold_dir / "error.json", failure)
            (fold_dir / "traceback.txt").write_text(tb, encoding="utf-8")
            fold_failures.append(failure)
            continue

    if results:
        returns = [float(r["test_metrics"]["net_return"]) for r in results]
        compounded = float(np.prod([1.0 + r for r in returns]) - 1.0)
        aggregate = {
            "compounded_oos_return": compounded,
            "mean_oos_return": float(np.mean(returns)),
            "mean_sharpe": float(np.mean([r["test_metrics"]["sharpe"] for r in results])),
            "mean_sortino": float(np.mean([r["test_metrics"]["sortino"] for r in results])),
            "mean_calmar": float(np.mean([r["test_metrics"]["calmar"] for r in results])),
            "worst_max_drawdown": float(max(r["test_metrics"]["max_drawdown"] for r in results)),
            "mean_cvar": float(np.mean([r["test_metrics"]["cvar"] for r in results])),
            "total_turnover": float(sum(r["test_metrics"]["turnover"] for r in results)),
            "total_fees": float(sum(r["test_metrics"]["fees"] for r in results)),
            "total_slippage": float(sum(r["test_metrics"]["slippage"] for r in results)),
            "total_market_impact": float(sum(r["test_metrics"]["market_impact"] for r in results)),
            "total_funding_pnl": float(sum(r["test_metrics"]["funding"] for r in results)),
            "total_liquidations": int(sum(r["test_metrics"]["liquidations"] for r in results)),
            "mean_buy_hold_return": float(np.mean([r["test_metrics"]["buy_hold_return"] for r in results])),
            "mean_excess_vs_buy_hold": float(np.mean([r["test_metrics"]["excess_vs_buy_hold"] for r in results])),
        }
        status = "SUCCESS" if not fold_failures else "PARTIAL"
    else:
        aggregate = {}
        status = "FAILED"

    payload = {
        "strategy": spec.name,
        "strategy_module": args.strategy,
        "algorithm": spec.algorithm,
        "timeframe": args.timeframe,
        "status": status,
        "requested_folds": args.folds,
        "successful_folds": len(results),
        "failed_folds": len(fold_failures),
        "budget": args.budget,
        "train_steps_per_fold": steps,
        "device_request": args.device,
        "compile_mode": args.compile_mode,
        "strategy_spec": asdict(spec),
        "action_config": ACTION_KWARGS,
        "cost_config": COST_KWARGS,
        "aggregate": aggregate,
        "folds": results,
        "fold_failures": fold_failures,
        "prepared_data_sha256": sha256_file(prepared),
    }
    json_dump(task_dir / "metrics.json", payload)
    print(json.dumps({
        "status": status,
        "strategy": spec.name,
        "timeframe": args.timeframe,
        "aggregate": aggregate,
    }, ensure_ascii=False))
    return 0 if results else 2


def subprocess_command(base_args, subcommand, extra):
    return [sys.executable, str(HERE / "runner.py"), subcommand, *base_args, *extra]


def run_process(command, log_path: Path, timeout: int = 0):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n\n")
        log.flush()
        try:
            cp = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=None if timeout <= 0 else timeout,
            )
            return {
                "returncode": int(cp.returncode),
                "seconds": float(time.time() - started),
                "timed_out": False,
            }
        except subprocess.TimeoutExpired:
            return {
                "returncode": 124,
                "seconds": float(time.time() - started),
                "timed_out": True,
            }
        except Exception as exc:
            log.write("\nMASTER SUBPROCESS ERROR:\n" + traceback.format_exc())
            return {
                "returncode": 125,
                "seconds": float(time.time() - started),
                "timed_out": False,
                "master_error": repr(exc),
            }


def copy_source_snapshot(dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(HERE / "runner.py", dst / "runner.py")
    shutil.copy2(HERE / "README_CN.md", dst / "README_CN.md")
    shutil.copytree(HERE / "strategies", dst / "strategies", dirs_exist_ok=True)


def collect_summary(results_root: Path):
    rows = []
    for metrics_path in sorted((results_root / "tasks").glob("*/*/metrics.json")):
        try:
            p = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        agg = p.get("aggregate") or {}
        rows.append({
            "strategy": p.get("strategy"),
            "algorithm": p.get("algorithm"),
            "timeframe": p.get("timeframe"),
            "status": p.get("status"),
            "successful_folds": p.get("successful_folds"),
            "failed_folds": p.get("failed_folds"),
            "compounded_oos_return": agg.get("compounded_oos_return"),
            "mean_sharpe": agg.get("mean_sharpe"),
            "mean_sortino": agg.get("mean_sortino"),
            "mean_calmar": agg.get("mean_calmar"),
            "worst_max_drawdown": agg.get("worst_max_drawdown"),
            "mean_cvar": agg.get("mean_cvar"),
            "total_turnover": agg.get("total_turnover"),
            "total_fees": agg.get("total_fees"),
            "total_slippage": agg.get("total_slippage"),
            "total_market_impact": agg.get("total_market_impact"),
            "total_funding_pnl": agg.get("total_funding_pnl"),
            "total_liquidations": agg.get("total_liquidations"),
            "mean_buy_hold_return": agg.get("mean_buy_hold_return"),
            "mean_excess_vs_buy_hold": agg.get("mean_excess_vs_buy_hold"),
        })
    pd.DataFrame(rows).to_csv(results_root / "summary.csv", index=False)
    json_dump(results_root / "summary.json", rows)
    return rows


def run_all(args) -> int:
    repo_root = Path(args.repo_root).resolve()
    cache_dir = Path(args.cache_dir or (repo_root / "user_data" / "hprl_eth_cache")).resolve()
    out_base = Path(args.output_dir or (repo_root / "hprl_results")).resolve()
    out_base.mkdir(parents=True, exist_ok=True)
    results_root = out_base / f"HPRL_ETH_2Y_RESULTS_{utc_now_tag()}"
    results_root.mkdir(parents=True, exist_ok=False)
    (results_root / "prepare_logs").mkdir()
    (results_root / "worker_logs").mkdir()
    (results_root / "tasks").mkdir()

    failures = []
    prep_status = {}
    base_args = [
        "--repo-root", str(repo_root),
        "--cache-dir", str(cache_dir),
        "--start", args.start,
        "--end", args.end,
    ]

    manifest = {
        "suite": "HPRL ETH 2Y multi-timeframe",
        "repository": "XXA222/HEDGE",
        "start_utc": args.start,
        "end_utc_exclusive": args.end,
        "timeframes": list(TIMEFRAMES),
        "strategies": list(STRATEGY_MODULES),
        "tasks_expected": len(TIMEFRAMES) * len(STRATEGY_MODULES),
        "budget": args.budget,
        "folds": args.folds,
        "device": args.device,
        "compile_mode": args.compile_mode,
        "parallel_envs": args.parallel_envs,
        "isolation": "each prepare and model×timeframe task runs in a separate subprocess",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
    }
    json_dump(results_root / "run_manifest.json", manifest)
    copy_source_snapshot(results_root / "source_snapshot")

    # Phase 1: data preparation. Each timeframe is isolated.
    for tf in TIMEFRAMES:
        cmd = subprocess_command(
            base_args,
            "prepare",
            ["--timeframe", tf] + (["--force"] if args.force_data else []),
        )
        status = run_process(cmd, results_root / "prepare_logs" / f"{tf}.log", args.task_timeout)
        prep_status[tf] = status
        if status["returncode"] != 0:
            failures.append({
                "scope": "prepare",
                "timeframe": tf,
                **status,
                "log": f"prepare_logs/{tf}.log",
            })

    # Phase 2: 30 isolated model×timeframe tasks.
    for strategy in STRATEGY_MODULES:
        for tf in TIMEFRAMES:
            task_dir = results_root / "tasks" / strategy / tf
            task_dir.mkdir(parents=True, exist_ok=True)
            if prep_status[tf]["returncode"] != 0:
                skip = {
                    "strategy_module": strategy,
                    "timeframe": tf,
                    "status": "SKIPPED_DATA_PREP_FAILED",
                    "prepare_status": prep_status[tf],
                }
                json_dump(task_dir / "metrics.json", skip)
                failures.append({
                    "scope": "worker-skipped",
                    "strategy": strategy,
                    "timeframe": tf,
                    "reason": "data preparation failed",
                })
                continue

            cmd = subprocess_command(
                base_args,
                "worker",
                [
                    "--strategy", strategy,
                    "--timeframe", tf,
                    "--task-dir", str(task_dir),
                    "--device", args.device,
                    "--budget", args.budget,
                    "--folds", str(args.folds),
                    "--parallel-envs", str(args.parallel_envs),
                    "--compile-mode", args.compile_mode,
                ],
            )
            log = results_root / "worker_logs" / f"{strategy}__{tf}.log"
            status = run_process(cmd, log, args.task_timeout)
            if status["returncode"] != 0:
                failures.append({
                    "scope": "worker",
                    "strategy": strategy,
                    "timeframe": tf,
                    **status,
                    "log": str(log.relative_to(results_root)),
                })

    rows = collect_summary(results_root)
    json_dump(results_root / "failures.json", failures)

    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    manifest["tasks_with_summary_rows"] = len(rows)
    manifest["failure_records"] = len(failures)
    manifest["prepare_status"] = prep_status
    json_dump(results_root / "run_manifest.json", manifest)

    # Zip after every task attempt, regardless of failures.
    zip_path = out_base / f"{results_root.name}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for p in sorted(results_root.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(results_root))

    print("\n=== HPRL SUITE COMPLETE ===")
    print(f"Results directory: {results_root}")
    print(f"Upload this ZIP:   {zip_path}")
    print(f"Failure records:   {len(failures)}")
    # Master returns success even with task failures: the requested invariant is that
    # one failed strategy/command must not abort the entire suite.
    return 0


def build_parser():
    p = argparse.ArgumentParser(description="HEDGE HPRL ETH multi-timeframe backtest suite")
    sub = p.add_subparsers(dest="command", required=True)

    def shared(sp):
        sp.add_argument("--repo-root", default=".")
        sp.add_argument("--cache-dir", default="")
        sp.add_argument("--start", default=DEFAULT_START)
        sp.add_argument("--end", default=DEFAULT_END)

    prep = sub.add_parser("prepare")
    shared(prep)
    prep.add_argument("--timeframe", required=True, choices=TIMEFRAMES)
    prep.add_argument("--force", action="store_true")

    worker_p = sub.add_parser("worker")
    shared(worker_p)
    worker_p.add_argument("--strategy", required=True, choices=STRATEGY_MODULES)
    worker_p.add_argument("--timeframe", required=True, choices=TIMEFRAMES)
    worker_p.add_argument("--task-dir", required=True)
    worker_p.add_argument("--device", default="auto")
    worker_p.add_argument("--budget", default="balanced", choices=tuple(TRAIN_STEPS))
    worker_p.add_argument("--folds", type=int, default=2)
    worker_p.add_argument("--parallel-envs", type=int, default=8)
    worker_p.add_argument("--compile-mode", default="off")

    all_p = sub.add_parser("run-all")
    shared(all_p)
    all_p.add_argument("--output-dir", default="")
    all_p.add_argument("--device", default="auto")
    all_p.add_argument("--budget", default="balanced", choices=tuple(TRAIN_STEPS))
    all_p.add_argument("--folds", type=int, default=2)
    all_p.add_argument("--parallel-envs", type=int, default=8)
    all_p.add_argument("--compile-mode", default="off")
    all_p.add_argument("--task-timeout", type=int, default=0, help="seconds; 0 disables timeout")
    all_p.add_argument("--force-data", action="store_true")
    return p


def main():
    args = build_parser().parse_args()
    if getattr(args, "cache_dir", "") == "":
        args.cache_dir = str(Path(args.repo_root).resolve() / "user_data" / "hprl_eth_cache")
    try:
        if args.command == "prepare":
            rc = prepare_timeframe(args)
        elif args.command == "worker":
            rc = worker(args)
        else:
            rc = run_all(args)
    except Exception:
        traceback.print_exc()
        rc = 2
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
