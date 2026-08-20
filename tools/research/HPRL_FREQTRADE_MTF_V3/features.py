from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from suite_specs import TIMEFRAME_SECONDS


FEATURE_VERSION = "hprl-freqtrade-mtf-closed-candle-v3"


def _dates(dataframe: pd.DataFrame, *, expected_seconds: int | None) -> pd.Series:
    if "date" not in dataframe.columns:
        raise ValueError("HPRL feature input is missing the date column")
    if not dataframe.index.is_unique:
        raise ValueError("HPRL feature input dataframe index must be unique")
    dates = pd.to_datetime(dataframe["date"], utc=True, errors="coerce")
    if dates.isna().any():
        raise ValueError("HPRL feature input contains an invalid candle timestamp")
    if len(dates) > 1:
        ns = dates.astype("int64").to_numpy(copy=False)
        deltas = np.diff(ns)
        if np.any(deltas <= 0):
            raise ValueError("HPRL candle timestamps must be strictly increasing and unique")
        if expected_seconds is not None:
            if isinstance(expected_seconds, bool) or int(expected_seconds) < 1:
                raise ValueError("expected_seconds must be a positive integer")
            expected_ns = int(expected_seconds) * 1_000_000_000
            mismatch = np.flatnonzero(deltas != expected_ns)
            if len(mismatch):
                i = int(mismatch[0])
                raise ValueError(
                    "HPRL source candles are not contiguous for the configured timeframe: "
                    f"{dates.iloc[i].isoformat()} -> {dates.iloc[i + 1].isoformat()}"
                )
    return dates


def _numeric_ohlcv(dataframe: pd.DataFrame):
    required = ("open", "high", "low", "close", "volume")
    missing = [name for name in required if name not in dataframe.columns]
    if missing:
        raise ValueError("HPRL feature input is missing OHLCV columns: " + ", ".join(missing))
    values = {
        name: pd.to_numeric(dataframe[name], errors="coerce").astype(float)
        for name in required
    }
    price_matrix = np.column_stack(
        [values[name].to_numpy(copy=False) for name in ("open", "high", "low", "close")]
    )
    if not np.isfinite(price_matrix).all() or np.any(price_matrix <= 0):
        raise ValueError("HPRL OHLC prices must be finite and positive")
    volume = values["volume"].to_numpy(copy=False)
    if not np.isfinite(volume).all() or np.any(volume < 0):
        raise ValueError("HPRL volume must be finite and non-negative")
    if np.any(values["high"].to_numpy(copy=False) < values["low"].to_numpy(copy=False)):
        raise ValueError("HPRL OHLC input contains high < low")
    return (
        values["open"],
        values["high"],
        values["low"],
        values["close"],
        values["volume"],
    )


def build_feature_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Build strictly causal features from one normal Freqtrade OHLCV dataframe."""
    open_, high, low, close, volume = _numeric_ohlcv(dataframe)

    log_close = np.log(close)
    lr = log_close.diff()
    feats: dict[str, pd.Series | np.ndarray] = {}

    for lag in (1, 2, 3, 6, 12):
        feats[f"logret_{lag}"] = log_close.diff(lag)
    for lag in (6, 12, 24):
        feats[f"momentum_{lag}"] = close / close.shift(lag) - 1.0
    for win in (6, 12, 24):
        feats[f"vol_{win}"] = lr.rolling(win).std(ddof=0)

    feats["range_pct"] = (high - low) / close
    feats["body_pct"] = (close - open_) / open_
    feats["upper_wick_pct"] = (high - np.maximum(open_, close)) / close
    feats["lower_wick_pct"] = (np.minimum(open_, close) - low) / close

    for span in (8, 21, 55):
        ema = close.ewm(span=span, adjust=False, min_periods=span).mean()
        feats[f"ema_spread_{span}"] = close / ema - 1.0

    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    movement = gain + loss
    denominator = movement.where(movement > 0, 1.0)
    rsi = 100.0 * gain / denominator
    # Zero movement after the RSI warmup is neutral.  Pre-warmup NaN remains NaN and is not
    # silently converted into a valid feature.  One-sided gain/loss still maps to 100/0.
    no_movement = movement.notna() & (movement <= 0)
    rsi = rsi.where(~no_movement, 50.0)
    feats["rsi14"] = rsi / 50.0 - 1.0

    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    feats["atr14_pct"] = atr / close

    lv = np.log1p(volume)
    volume_mean = lv.rolling(24).mean()
    volume_std = lv.rolling(24).std(ddof=0)
    safe_volume_std = volume_std.where(volume_std > 1e-12, 1.0)
    volume_z = (lv - volume_mean) / safe_volume_std
    volume_z = volume_z.where(volume_std > 1e-12, 0.0)
    feats["volume_z24"] = volume_z
    feats["volume_log_change"] = lv.diff()

    ts = pd.to_datetime(dataframe["date"], utc=True, errors="coerce")
    hour = ts.dt.hour.to_numpy(dtype=float)
    dow = ts.dt.dayofweek.to_numpy(dtype=float)
    feats["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    feats["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    feats["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    feats["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

    return pd.DataFrame(feats, index=dataframe.index).replace([np.inf, -np.inf], np.nan)


def _available_notional(volume: pd.Series, close: pd.Series) -> pd.Series:
    available = volume * close
    if not np.isfinite(available.to_numpy(copy=False)).all():
        raise ValueError("HPRL available-notional proxy contains a non-finite value")
    return available.clip(lower=1.0)


def _normalize_utc(value: object) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if result.tzinfo is None:
        return result.tz_localize("UTC")
    return result.tz_convert("UTC")


def _validate_timeframes(
    frames: Mapping[str, pd.DataFrame],
    base_timeframe: str,
    input_timeframes: Sequence[str],
) -> tuple[str, ...]:
    values = tuple(str(value) for value in input_timeframes)
    if not values or values[0] != base_timeframe:
        raise ValueError("HPRL MTF input_timeframes must start with the base timeframe")
    if len(set(values)) != len(values):
        raise ValueError("HPRL MTF input_timeframes must be unique")
    unsupported = [tf for tf in values if tf not in TIMEFRAME_SECONDS]
    if unsupported:
        raise ValueError("unsupported HPRL MTF timeframe(s): " + ", ".join(unsupported))
    missing = [tf for tf in values if tf not in frames]
    if missing:
        raise ValueError("HPRL MTF input is missing timeframe(s): " + ", ".join(missing))
    base_seconds = TIMEFRAME_SECONDS[base_timeframe]
    smaller = [tf for tf in values[1:] if TIMEFRAME_SECONDS[tf] < base_seconds]
    if smaller:
        raise ValueError(
            "informative timeframes must be equal to or higher than the base timeframe: "
            + ", ".join(smaller)
        )
    return values


def _feature_names_for(timeframes: Sequence[str], single_names: Sequence[str]) -> tuple[str, ...]:
    names: list[str] = []
    base = str(timeframes[0])
    for timeframe in timeframes:
        names.extend(f"{timeframe}__{name}" for name in single_names)
        if timeframe != base:
            names.append(f"{timeframe}__age_frac")
    return tuple(names)


def align_multi_timeframe_features(
    frames: Mapping[str, pd.DataFrame],
    *,
    base_timeframe: str,
    input_timeframes: Sequence[str],
    base_positions: np.ndarray | None = None,
) -> tuple[np.ndarray, tuple[str, ...], dict[str, object]]:
    """Align informative features to base-candle close without lookahead.

    Freqtrade OHLCV timestamps identify candle *open* time.  A source candle therefore becomes
    visible only at ``source_open + source_duration``.  For every base decision we choose the
    latest source candle whose close is <= the base candle close.  No partial higher-timeframe
    candle can enter the observation.

    This function writes directly into one float32 matrix instead of merging six wide pandas
    dataframes, which substantially reduces peak memory on multi-year 1m histories.
    """
    timeframes = _validate_timeframes(frames, base_timeframe, input_timeframes)
    base_frame = frames[base_timeframe]
    base_dates = _dates(base_frame, expected_seconds=TIMEFRAME_SECONDS[base_timeframe])
    base_ns = base_dates.astype("int64").to_numpy(copy=False)

    if base_positions is None:
        positions = np.arange(len(base_frame), dtype=np.int64)
    else:
        positions = np.asarray(base_positions, dtype=np.int64)
        if positions.ndim != 1:
            raise ValueError("base_positions must be one-dimensional")
        if len(positions) and (positions[0] < 0 or positions[-1] >= len(base_frame)):
            raise ValueError("base_positions contains an out-of-range row")
        if len(positions) > 1 and np.any(np.diff(positions) <= 0):
            raise ValueError("base_positions must be strictly increasing")

    base_seconds = TIMEFRAME_SECONDS[base_timeframe]
    decision_close_ns = base_ns[positions] + base_seconds * 1_000_000_000

    first_feature_frame = build_feature_frame(base_frame)
    single_names = tuple(str(name) for name in first_feature_frame.columns)
    feature_names = _feature_names_for(timeframes, single_names)
    output = np.full((len(positions), len(feature_names)), np.nan, dtype=np.float32)

    source_summary: dict[str, object] = {}
    column = 0
    for timeframe in timeframes:
        frame = frames[timeframe]
        dates = _dates(frame, expected_seconds=TIMEFRAME_SECONDS[timeframe])
        if timeframe == base_timeframe:
            feature_frame = first_feature_frame
        else:
            feature_frame = build_feature_frame(frame)
        if tuple(feature_frame.columns) != single_names:
            raise RuntimeError(f"HPRL feature layout changed across timeframe {timeframe}")

        source_values = feature_frame.to_numpy(dtype=np.float32, copy=False)
        source_open_ns = dates.astype("int64").to_numpy(copy=False)
        source_seconds = TIMEFRAME_SECONDS[timeframe]
        source_close_ns = source_open_ns + source_seconds * 1_000_000_000
        source_index = np.searchsorted(source_close_ns, decision_close_ns, side="right") - 1
        valid_index = source_index >= 0

        age_seconds = np.full(len(positions), np.nan, dtype=np.float64)
        if np.any(valid_index):
            selected = source_index[valid_index]
            selected_close = source_close_ns[selected]
            selected_age = (decision_close_ns[valid_index] - selected_close) / 1_000_000_000.0
            age_seconds[valid_index] = selected_age
            fresh = (selected_age >= 0.0) & (selected_age < float(source_seconds))
            target_rows = np.flatnonzero(valid_index)[fresh]
            source_rows = selected[fresh]
            output[target_rows, column : column + len(single_names)] = source_values[source_rows]

        source_summary[timeframe] = {
            "rows": len(frame),
            "first_open": None if len(dates) == 0 else dates.iloc[0].isoformat(),
            "last_open": None if len(dates) == 0 else dates.iloc[-1].isoformat(),
            "max_allowed_age_seconds": int(source_seconds - 1),
        }
        column += len(single_names)

        if timeframe != base_timeframe:
            age_frac = age_seconds / float(source_seconds)
            fresh_age = np.isfinite(age_frac) & (age_frac >= 0.0) & (age_frac < 1.0)
            output[fresh_age, column] = age_frac[fresh_age].astype(np.float32)
            column += 1

        # Avoid retaining one full float64 feature dataframe per timeframe.
        if timeframe != base_timeframe:
            del feature_frame

    if column != output.shape[1]:
        raise RuntimeError("HPRL MTF feature column accounting mismatch")
    del first_feature_frame

    diagnostics = {
        "base_timeframe": base_timeframe,
        "input_timeframes": list(timeframes),
        "decision_rows": len(positions),
        "feature_count": int(output.shape[1]),
        "source_summary": source_summary,
    }
    return output, feature_names, diagnostics


def _first_fully_valid_row(values: np.ndarray) -> int:
    if values.ndim != 2 or not len(values):
        raise ValueError("HPRL MTF feature matrix is empty")
    valid = np.isfinite(values).all(axis=1)
    positions = np.flatnonzero(valid)
    if not len(positions):
        raise ValueError("HPRL MTF features never become fully valid after startup warmup")
    first = int(positions[0])
    invalid_after = np.flatnonzero(~valid[first:])
    if len(invalid_after):
        bad = first + int(invalid_after[0])
        bad_columns = np.flatnonzero(~np.isfinite(values[bad]))[:8]
        raise ValueError(
            "HPRL MTF feature stream became invalid after startup warmup at aligned row "
            f"{bad}; bad_column_indices={bad_columns.tolist()}; refusing to compress, forward-fill "
            "stale informative data, or skip a market timestep"
        )
    return first


def training_arrays_mtf(
    frames: Mapping[str, pd.DataFrame],
    *,
    base_timeframe: str,
    input_timeframes: Sequence[str],
    start: object,
    end: object,
):
    timeframes = _validate_timeframes(frames, base_timeframe, input_timeframes)
    base = frames[base_timeframe]
    dates = _dates(base, expected_seconds=TIMEFRAME_SECONDS[base_timeframe])
    _, _, _, close, volume = _numeric_ohlcv(base)
    start_ts = _normalize_utc(start)
    end_ts = _normalize_utc(end)
    if not start_ts < end_ts:
        raise ValueError("HPRL training start must be earlier than training end")

    eligible = np.flatnonzero(
        ((dates >= start_ts) & (dates < end_ts)).to_numpy(dtype=bool, copy=False)
    ).astype(np.int64, copy=False)
    if len(eligible) < 2:
        raise ValueError("HPRL MTF training interval has fewer than two base candles")
    if np.any(np.diff(eligible) != 1):
        raise ValueError("HPRL MTF training base interval is not contiguous")

    decision_positions = eligible[:-1]
    x, feature_names, diagnostics = align_multi_timeframe_features(
        frames,
        base_timeframe=base_timeframe,
        input_timeframes=timeframes,
        base_positions=decision_positions,
    )
    if not np.isfinite(x).all():
        bad_row = int(np.flatnonzero(~np.isfinite(x).all(axis=1))[0])
        bad_columns = np.flatnonzero(~np.isfinite(x[bad_row]))[:8]
        raise ValueError(
            "HPRL MTF training interval lacks fully closed/fresh informative features at base "
            f"position={int(decision_positions[bad_row])}; "
            f"bad_column_indices={bad_columns.tolist()}. "
            "Provide additional pre-start history for every informative timeframe."
        )

    next_positions = decision_positions + 1
    current_close = close.iloc[decision_positions].to_numpy(dtype=np.float64, copy=False)
    next_close = close.iloc[next_positions].to_numpy(dtype=np.float64, copy=False)
    returns = next_close / current_close - 1.0
    if not np.isfinite(returns).all():
        raise ValueError("HPRL MTF training forward return contains a non-finite value")
    available = _available_notional(volume, close).iloc[decision_positions]
    timestamps = dates.iloc[decision_positions].to_numpy(copy=True)
    diagnostics["interval_start"] = start_ts.isoformat()
    diagnostics["interval_end"] = end_ts.isoformat()
    return (
        x,
        returns.astype(np.float32, copy=True),
        available.to_numpy(dtype=np.float32, copy=True),
        timestamps,
        feature_names,
        decision_positions,
        diagnostics,
    )


def inference_arrays_mtf(
    frames: Mapping[str, pd.DataFrame],
    *,
    base_timeframe: str,
    input_timeframes: Sequence[str],
):
    """Build a causal non-compressing MTF inference tape for a formal Freqtrade Strategy."""
    timeframes = _validate_timeframes(frames, base_timeframe, input_timeframes)
    base = frames[base_timeframe]
    dates = _dates(base, expected_seconds=TIMEFRAME_SECONDS[base_timeframe])
    _, _, _, close, volume = _numeric_ohlcv(base)
    positions = np.arange(len(base), dtype=np.int64)
    x_full, feature_names, diagnostics = align_multi_timeframe_features(
        frames,
        base_timeframe=base_timeframe,
        input_timeframes=timeframes,
        base_positions=positions,
    )
    first = _first_fully_valid_row(x_full)
    valid_positions = positions[first:]
    x = x_full[first:]
    if len(x) < 2:
        raise ValueError("Not enough causal HPRL MTF rows after startup warmup")

    forward_return = close.shift(-1) / close - 1.0
    returns = forward_return.iloc[valid_positions].to_numpy(dtype=np.float64, copy=True)
    if len(returns) > 1 and not np.isfinite(returns[:-1]).all():
        bad_local = int(np.flatnonzero(~np.isfinite(returns[:-1]))[0])
        bad = int(valid_positions[bad_local])
        raise ValueError(
            "HPRL MTF inference forward return is invalid before the terminal row at "
            f"position {bad}"
        )
    returns[-1] = 0.0
    available = _available_notional(volume, close).iloc[valid_positions]
    timestamps = dates.iloc[valid_positions].to_numpy(copy=True)
    diagnostics["first_valid_base_position"] = int(valid_positions[0])
    return (
        x.astype(np.float32, copy=False),
        returns.astype(np.float32, copy=False),
        available.to_numpy(dtype=np.float32, copy=True),
        timestamps,
        feature_names,
        valid_positions,
        diagnostics,
    )


def fit_scaler(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if x.ndim != 2 or not len(x):
        raise ValueError("training feature matrix must be non-empty and two-dimensional")
    if not np.isfinite(x).all():
        raise ValueError("training feature matrix contains a non-finite value")
    mean = x.mean(axis=0, dtype=np.float64)
    std = x.std(axis=0, dtype=np.float64)
    std[~np.isfinite(std) | (std < 1e-6)] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def apply_scaler(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    if x.ndim != 2 or mean.ndim != 1 or std.ndim != 1:
        raise ValueError("feature/scaler arrays have invalid dimensions")
    if x.shape[-1] != mean.shape[0] or mean.shape != std.shape:
        raise ValueError("feature/scaler dimensions do not match")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("HPRL scaler contains invalid parameters")
    result = (x.astype(np.float32, copy=False) - mean) / std
    if not np.isfinite(result).all():
        raise ValueError("scaled HPRL feature matrix contains a non-finite value")
    return result.astype(np.float32, copy=False)
