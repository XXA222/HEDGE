from __future__ import annotations

import math
import sys
import types
import unittest
from datetime import UTC, datetime, timedelta
from decimal import Decimal

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


def _frame(count: int, *, long_score: str = "0", short_score: str = "0") -> pd.DataFrame:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    rows = []
    for index in range(count):
        price = Decimal(100) + Decimal(index % 5) / Decimal(10)
        rows.append(
            {
                "date": start + timedelta(minutes=index),
                "open": str(price),
                "high": str(price + Decimal(1)),
                "low": str(price - Decimal(1)),
                "close": str(price),
                "volume": "100",
                "hedge_long_score": long_score,
                "hedge_short_score": short_score,
                "hedge_target_net_ratio": "0",
            }
        )
    return pd.DataFrame(rows)


class MemoryOptimizedBacktestingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _install_exchange_stub()

    def test_stream_is_single_use_and_dataset_is_compact(self) -> None:
        from freqtrade.optimize.hedge_backtesting import HedgeBacktestEventChunks

        stream = HedgeBacktestEventChunks(
            pair="BTC/USDT:USDT",
            timeframe="1m",
            frame=_frame(17),
            chunk_bars=4,
        )
        chunks = list(stream)
        dataset = stream.dataset()
        self.assertEqual(dataset.events, ())
        self.assertEqual(dataset.bar_count, 17)
        self.assertEqual(dataset.signal_count, 17)
        self.assertEqual(len(chunks), math.ceil(17 / 4))
        self.assertLessEqual(stream.max_chunk_input_events, 8)
        with self.assertRaises(RuntimeError):
            list(stream)

    def test_stream_event_priority_signal_funding_bar(self) -> None:
        from freqtrade.hedge.simulation.exchange import BarEvent, FundingEvent, SignalEvent
        from freqtrade.optimize.hedge_backtesting import HedgeBacktestEventChunks

        frame = _frame(3)
        funding = pd.DataFrame(
            [
                {
                    "date": datetime(2026, 1, 1, 0, 1, tzinfo=UTC),
                    "open_fund": "0.0001",
                    "open_mark": "100",
                }
            ]
        )
        stream = HedgeBacktestEventChunks(
            pair="BTC/USDT:USDT",
            timeframe="1m",
            frame=frame,
            funding_frame=funding,
            chunk_bars=3,
        )
        events = [event for chunk in stream for event in chunk]
        same_timestamp = [
            event for event in events if event.timestamp == datetime(2026, 1, 1, 0, 1, tzinfo=UTC)
        ]
        self.assertEqual(
            [type(event) for event in same_timestamp],
            [SignalEvent, FundingEvent, BarEvent],
        )

    def test_stream_fingerprint_is_deterministic_and_signal_sensitive(self) -> None:
        from freqtrade.optimize.hedge_backtesting import HedgeBacktestEventChunks

        def fingerprint(score: str) -> str:
            stream = HedgeBacktestEventChunks(
                pair="BTC/USDT:USDT",
                timeframe="1m",
                frame=_frame(8, long_score=score),
                chunk_bars=3,
            )
            list(stream)
            return stream.dataset().data_fingerprint

        one = fingerprint("0.25")
        two = fingerprint("0.25")
        changed = fingerprint("0.75")
        self.assertEqual(one, two)
        self.assertNotEqual(one, changed)

    def test_stream_detects_missing_candles_without_date_tuple(self) -> None:
        from freqtrade.exceptions import OperationalException
        from freqtrade.optimize.hedge_backtesting import HedgeBacktestEventChunks

        frame = _frame(4).drop(index=2).reset_index(drop=True)
        stream = HedgeBacktestEventChunks(
            pair="BTC/USDT:USDT",
            timeframe="1m",
            frame=frame,
            max_missing_candles=0,
        )
        with self.assertRaises(OperationalException):
            list(stream)

    def test_compact_replay_matches_full_wallet_report(self) -> None:
        from freqtrade.optimize.hedge_backtesting import (
            HedgeBacktestEventChunks,
            HedgeBacktesting,
            events_from_analyzed_dataframe,
        )

        frame = _frame(24, long_score="0.8", short_score="0.8")
        detailed_dataset = events_from_analyzed_dataframe(
            pair="BTC/USDT:USDT",
            timeframe="1m",
            frame=frame,
        )
        detailed = HedgeBacktesting(initial_balance=Decimal(1000)).run(detailed_dataset.events)

        stream = HedgeBacktestEventChunks(
            pair="BTC/USDT:USDT",
            timeframe="1m",
            frame=frame,
            chunk_bars=5,
        )
        compact = HedgeBacktesting(initial_balance=Decimal(1000)).run_compact(stream)

        from freqtrade.hedge.backtesting.consistency import COMPACT_ONLY_REPORT_FIELDS

        compact_business = {
            key: value
            for key, value in compact.report.items()
            if key not in COMPACT_ONLY_REPORT_FIELDS
        }
        self.assertEqual(compact_business, detailed.report)
        self.assertEqual(compact.report["processed_bar_count"], 24)
        self.assertEqual(compact.report["replay_mode"], "COMPACT_ORDERED_STREAM_V2")

    def test_compact_replay_retains_bounded_snapshots_and_no_input_ledger(self) -> None:
        from freqtrade.hedge.simulation.exchange import BarEvent, FundingEvent, SignalEvent
        from freqtrade.optimize.hedge_backtesting import (
            HedgeBacktestEventChunks,
            HedgeBacktesting,
        )

        count = 101
        chunk_bars = 8
        stream = HedgeBacktestEventChunks(
            pair="BTC/USDT:USDT",
            timeframe="1m",
            frame=_frame(count),
            chunk_bars=chunk_bars,
        )
        result = HedgeBacktesting(initial_balance=Decimal(1000)).run_compact(stream)
        self.assertLessEqual(len(result.snapshots), math.ceil(count / chunk_bars))
        self.assertFalse(
            any(isinstance(event, (SignalEvent, FundingEvent, BarEvent)) for event in result.events)
        )
        self.assertEqual(result.report["processed_bar_count"], count)

    def test_compact_replay_releases_processed_slot_history(self) -> None:
        from freqtrade.optimize.hedge_backtesting import (
            HedgeBacktestEventChunks,
            HedgeBacktesting,
        )

        runner = HedgeBacktesting(initial_balance=Decimal(1000))
        stream = HedgeBacktestEventChunks(
            pair="BTC/USDT:USDT",
            timeframe="1m",
            frame=_frame(50),
            chunk_bars=7,
        )
        runner.run_compact(stream)
        self.assertEqual(runner.engine._processed_slots, set())

    def test_legacy_event_builder_still_available_for_detailed_mode(self) -> None:
        from freqtrade.optimize.hedge_backtesting import events_from_analyzed_dataframe

        dataset = events_from_analyzed_dataframe(
            pair="BTC/USDT:USDT",
            timeframe="1m",
            frame=_frame(5),
        )
        self.assertEqual(dataset.bar_count, 5)
        self.assertEqual(len(dataset.events), 10)
        self.assertTrue(dataset.data_fingerprint)


if __name__ == "__main__":
    unittest.main()
