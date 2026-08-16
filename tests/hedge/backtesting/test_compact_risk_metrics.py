from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace

import pytest

from freqtrade.hedge.native.backtest import HedgeBacktestResultAdapter
from freqtrade.hedge.simulation.cross_wallet import CrossWallet


def test_wallet_tracks_bar_return_moments_in_constant_memory() -> None:
    wallet = CrossWallet(initial_balance=Decimal(1000))
    start = datetime(2026, 1, 1, tzinfo=UTC)
    wallet.observe_bar_state(start, Decimal(100))
    wallet.balance = Decimal(1100)
    wallet.observe_bar_state(start + timedelta(minutes=1), Decimal(100))
    wallet.balance = Decimal(990)
    wallet.observe_bar_state(start + timedelta(minutes=2), Decimal(100))

    assert wallet.equity_return_count == 2
    assert wallet.equity_return_sum == Decimal("0.00")
    assert wallet.equity_return_sum_squares == Decimal("0.0200")
    assert wallet.equity_downside_square_sum == Decimal("0.0100")
    assert wallet.last_observed_equity == Decimal(990)


def test_native_adapter_prefers_compact_bar_return_moments() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    simulation = SimpleNamespace(
        snapshots=(
            {"timestamp": start, "equity": Decimal(1000)},
            {"timestamp": start + timedelta(days=1), "equity": Decimal(1050)},
        ),
        events=(),
        report={
            "initial_balance": Decimal(1000),
            "final_equity": Decimal(1050),
            "total_return_ratio": Decimal("0.05"),
            "max_drawdown": Decimal("0.10"),
            "replay_mode": "COMPACT_ORDERED_STREAM_V2",
            "snapshot_stride_bars": 1440,
            "equity_return_count": 2,
            "equity_return_sum": Decimal("0.05"),
            "equity_return_sum_squares": Decimal("0.0125"),
            "equity_downside_square_sum": Decimal("0.0025"),
        },
    )
    artifact = HedgeBacktestResultAdapter().build(
        simulation,
        strategy_name="Audit",
        pairs=("BTC/USDT:USDT",),
        timeframe="1m",
    )
    expected_mean = 0.025
    expected_variance = 0.0125 / 2 - expected_mean * expected_mean
    expected_sharpe = expected_mean / (expected_variance**0.5) * (365 * 24 * 60) ** 0.5
    assert artifact.metrics.sharpe == pytest.approx(expected_sharpe)
    assert artifact.metadata["risk_metric_source"] == "BAR_RETURN_MOMENTS"
    assert artifact.metadata["risk_periods_per_year"] == 365 * 24 * 60


def test_legacy_compact_snapshot_fallback_adjusts_stride() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    simulation = SimpleNamespace(
        snapshots=(
            {"timestamp": start, "equity": Decimal(1000)},
            {"timestamp": start + timedelta(days=1), "equity": Decimal(1010)},
            {"timestamp": start + timedelta(days=2), "equity": Decimal(1005)},
        ),
        events=(),
        report={
            "initial_balance": Decimal(1000),
            "final_equity": Decimal(1005),
            "total_return_ratio": Decimal("0.005"),
            "replay_mode": "COMPACT_ORDERED_STREAM_V2",
            "snapshot_stride_bars": 1440,
        },
    )
    artifact = HedgeBacktestResultAdapter().build(
        simulation,
        strategy_name="LegacyCompact",
        pairs=("BTC/USDT:USDT",),
        timeframe="1m",
    )
    assert artifact.metadata["risk_metric_source"] == "SNAPSHOT_RETURNS"
    assert artifact.metadata["risk_periods_per_year"] == 365


def test_native_adapter_uses_bar_return_moments_in_detailed_mode_too() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    simulation = SimpleNamespace(
        snapshots=(
            {"timestamp": start, "equity": Decimal(1000)},
            {"timestamp": start + timedelta(minutes=1), "equity": Decimal(1050)},
        ),
        events=(),
        report={
            "initial_balance": Decimal(1000),
            "final_equity": Decimal(1050),
            "total_return_ratio": Decimal("0.05"),
            "max_drawdown": Decimal("0.10"),
            "equity_return_count": 2,
            "equity_return_sum": Decimal("0.05"),
            "equity_return_sum_squares": Decimal("0.0125"),
            "equity_downside_square_sum": Decimal("0.0025"),
        },
    )
    artifact = HedgeBacktestResultAdapter().build(
        simulation,
        strategy_name="Detailed",
        pairs=("BTC/USDT:USDT",),
        timeframe="1m",
    )
    assert artifact.metadata["risk_metric_source"] == "BAR_RETURN_MOMENTS"
    assert artifact.metadata["risk_periods_per_year"] == 365 * 24 * 60


def test_non_bar_risk_observation_does_not_create_return_sample() -> None:
    wallet = CrossWallet(initial_balance=Decimal(1000))
    start = datetime(2026, 1, 1, tzinfo=UTC)
    wallet.observe_bar_state(start, Decimal(100))
    wallet.balance = Decimal(1010)
    wallet.observe_state(start + timedelta(seconds=30), Decimal(100))
    assert wallet.equity_return_count == 0
    wallet.observe_bar_state(start + timedelta(minutes=1), Decimal(100))
    assert wallet.equity_return_count == 1
    assert wallet.equity_return_sum == Decimal("0.01")
