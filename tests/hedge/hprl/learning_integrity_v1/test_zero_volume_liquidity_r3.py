from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "freqtrade/hedge/hprl").is_dir() and (parent / "tools").is_dir():
            return parent
    raise AssertionError("cannot resolve installed HEDGE project root")


def load_workflow():
    path = project_root() / "tools/train_hprl_eth_two_year.py"
    spec = importlib.util.spec_from_file_location("hprl_eth_r3_zero_volume_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_candles(path: Path, timestamps: pd.DatetimeIndex, *, all_zero: bool = False) -> None:
    base = 1000.0 + np.arange(len(timestamps), dtype=float) * 0.1
    volume = np.full(len(timestamps), 100.0)
    if all_zero:
        volume[:] = 0.0
    else:
        volume[100] = 0.0
    pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": base,
            "high": base + 1.0,
            "low": base - 1.0,
            "close": base + 0.2,
            "volume": volume,
        }
    ).to_csv(path, index=False)


def test_zero_volume_candle_is_valid_but_all_zero_primary_is_rejected(tmp_path):
    tool = load_workflow()
    timestamps = pd.date_range("2020-01-01", periods=2100, freq="1D", tz="UTC")
    for timeframe in tool.TIMEFRAMES:
        write_candles(tmp_path / f"eth-{timeframe}.csv", timestamps)

    frame, _, manifest = tool.build_feature_frame(
        tmp_path,
        primary="1d",
        max_gap_fraction=1.0,
        funding_file=None,
        require_funding=False,
    )
    assert manifest["sources"]["1d"]["zero_volume_count"] == 1
    assert manifest["liquidity"]["zero_liquidity_count"] == 1
    assert manifest["liquidity"]["available_notional_floor"] == 1.0
    assert float(frame["available_notional"].min()) >= 1.0

    write_candles(tmp_path / "eth-1d.csv", timestamps, all_zero=True)
    try:
        tool.build_feature_frame(
            tmp_path,
            primary="1d",
            max_gap_fraction=1.0,
            funding_file=None,
            require_funding=False,
        )
    except ValueError as exc:
        assert "no positive-liquidity candles" in str(exc)
    else:
        raise AssertionError("all-zero primary liquidity must fail closed")
