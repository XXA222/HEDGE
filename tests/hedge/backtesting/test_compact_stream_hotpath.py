from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np

from freqtrade.optimize.hedge_backtesting import (
    _ArrayRowView,
    _missing_candle_count_seconds,
)


def test_indexed_array_row_view_reuses_array_references_without_row_tuple() -> None:
    names = ("open", "close")
    arrays = (
        np.asarray([100.0, 101.0, 102.0]),
        np.asarray([100.5, 101.5, 102.5]),
    )
    row = _ArrayRowView(names, arrays)
    assert row.bind_index(0).get("open") == 100.0
    assert row.bind_index(2).get("close") == 102.5
    assert row.get("missing", 7) == 7
    assert not hasattr(row, "_values")


def test_cached_timeframe_missing_count_matches_minute_contract() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    assert _missing_candle_count_seconds(start, start + timedelta(minutes=1), 60) == 0
    assert _missing_candle_count_seconds(start, start + timedelta(minutes=4), 60) == 3


def test_cached_timeframe_missing_count_rejects_misalignment() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    try:
        _missing_candle_count_seconds(start, start + timedelta(seconds=61), 60)
    except ValueError as exc:
        assert "aligned" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("misaligned candle interval was accepted")


def test_ordered_compact_funding_observation_matches_detailed_report_semantics() -> None:
    from decimal import Decimal

    from freqtrade.hedge.planning.context import PlannerConfig
    from freqtrade.hedge.simulation.exchange import BarEvent, FundingEvent
    from freqtrade.hedge.simulation.replay import EventReplayEngine

    start = datetime(2026, 1, 1, tzinfo=UTC)
    events = [
        BarEvent(
            start,
            "BTC/USDT:USDT",
            Decimal(100),
            Decimal(101),
            Decimal(99),
            Decimal(100),
            Decimal(1000),
        ),
        BarEvent(
            start + timedelta(minutes=1),
            "BTC/USDT:USDT",
            Decimal(100),
            Decimal(102),
            Decimal(99),
            Decimal(101),
            Decimal(1000),
        ),
        FundingEvent(
            start + timedelta(minutes=1, seconds=30),
            "BTC/USDT:USDT",
            Decimal("0.0001"),
            Decimal(101),
        ),
        BarEvent(
            start + timedelta(minutes=2),
            "BTC/USDT:USDT",
            Decimal(101),
            Decimal(103),
            Decimal(100),
            Decimal(102),
            Decimal(1000),
        ),
    ]
    cfg = PlannerConfig(cooldown_seconds=0, trailing_rebound=Decimal(0))
    detailed = EventReplayEngine(initial_balance=Decimal(1000), planner_config=cfg).replay(events)
    compact = EventReplayEngine(
        initial_balance=Decimal(1000), planner_config=cfg
    ).replay_ordered_stream(
        events,
        retain_material_events=False,
        snapshot_every_bars=2,
        max_retained_snapshots=16,
        compact_wallet_history=True,
    )
    assert detailed.report["equity_return_count"] == 2
    assert compact.report["equity_return_count"] == 2
    for key, expected in detailed.report.items():
        assert compact.report[key] == expected, key


def test_compact_slot_mask_and_flat_idle_bypass_are_reported() -> None:
    from decimal import Decimal

    from freqtrade.hedge.simulation.exchange import BarEvent, SignalEvent
    from freqtrade.hedge.simulation.replay import EventReplayEngine

    start = datetime(2026, 1, 1, tzinfo=UTC)
    events = [
        SignalEvent(
            start,
            "BTC/USDT:USDT",
            Decimal(0),
            Decimal(0),
            allow_new_risk=False,
            reason="flat-idle-bypass-test",
        ),
        *(
            BarEvent(
                start + timedelta(minutes=index),
                "BTC/USDT:USDT",
                Decimal(100),
                Decimal(100),
                Decimal(100),
                Decimal(100),
                Decimal(1000),
            )
            for index in range(3)
        ),
    ]
    result = EventReplayEngine(
        initial_balance=Decimal(1000),
        long_signal=Decimal(0),
        short_signal=Decimal(0),
    ).replay_ordered_stream(
        events,
        retain_material_events=False,
        snapshot_every_bars=2,
        max_retained_snapshots=16,
        compact_wallet_history=True,
    )
    assert result.report["slot_validation_mode"] == "BITMASK_SLOT_VALIDATION_V1"
    assert result.report["matcher_mode"] == "FLAT_IDLE_BYPASS_V1"
    assert result.report["flat_idle_matcher_bypass_count"] == 3
