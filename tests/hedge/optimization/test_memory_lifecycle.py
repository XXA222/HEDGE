from __future__ import annotations

import sys
import types
import unittest
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pandas as pd


def _install_exchange_stub() -> None:
    if "freqtrade.exchange" in sys.modules:
        return
    module = types.ModuleType("freqtrade.exchange")

    def timeframe_to_seconds(timeframe: str) -> int:
        amount = int(timeframe[:-1])
        return amount * {"s": 1, "m": 60, "h": 3600, "d": 86400}[timeframe[-1]]

    module.timeframe_to_seconds = timeframe_to_seconds
    sys.modules["freqtrade.exchange"] = module


def _frame(count: int, score: str = "0.8") -> pd.DataFrame:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    rows = []
    for index in range(count):
        base = Decimal(100) + Decimal(index % 3)
        rows.append(
            {
                "date": start + timedelta(minutes=index),
                "open": str(base),
                "high": str(base + Decimal(2)),
                "low": str(base - Decimal(2)),
                "close": str(base),
                "volume": "10000",
                "hedge_long_score": score,
                "hedge_short_score": score,
                "hedge_target_net_ratio": "0",
            }
        )
    return pd.DataFrame(rows)


class MemoryLifecycleV14Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _install_exchange_stub()

    def test_dataprovider_cache_contract_is_backward_compatible(self) -> None:
        source = (
            Path(__file__).resolve().parents[3] / "freqtrade" / "data" / "dataprovider.py"
        ).read_text(encoding="utf-8")
        self.assertIn("def clear_cache(self, *, include_backtesting: bool = False)", source)
        self.assertIn("if include_backtesting:", source)
        self.assertIn("self.__cached_pairs_backtesting = {}", source)

    def test_hedge_full_release_requests_historic_cache_clear(self) -> None:
        source = (
            Path(__file__).resolve().parents[3] / "freqtrade" / "optimize" / "hedge_backtesting.py"
        ).read_text(encoding="utf-8")
        self.assertIn("clear_cache(include_backtesting=True)", source)
        self.assertIn("reduce_df_footprint", source)
        self.assertIn("DEFAULT_HEDGE_BACKTEST_MEMORY_POLICY.reduce_dataframe_footprint", source)

    def test_ordered_stream_has_no_input_ledger(self) -> None:
        from freqtrade.optimize.hedge_backtesting import HedgeBacktestEventChunks, HedgeBacktesting

        stream = HedgeBacktestEventChunks(
            pair="BTC/USDT:USDT", timeframe="1m", frame=_frame(200, "0")
        )
        runner = HedgeBacktesting(initial_balance=Decimal(1000))
        result = runner.run_compact(stream)
        self.assertEqual(result.events, ())
        self.assertEqual(runner.engine._processed_slots, set())
        self.assertEqual(result.report["replay_mode"], "COMPACT_ORDERED_STREAM_V2")

    def test_ordered_stream_snapshot_count_is_bounded(self) -> None:
        from freqtrade.optimize.hedge_backtesting import HedgeBacktestEventChunks, HedgeBacktesting

        stream = HedgeBacktestEventChunks(
            pair="BTC/USDT:USDT", timeframe="1m", frame=_frame(5000, "0")
        )
        result = HedgeBacktesting(initial_balance=Decimal(1000)).run_compact(stream)
        self.assertLessEqual(len(result.snapshots), 6)
        self.assertEqual(result.report["processed_bar_count"], 5000)

    def test_ordered_stream_releases_wallet_transient_history(self) -> None:
        from freqtrade.optimize.hedge_backtesting import HedgeBacktestEventChunks, HedgeBacktesting

        stream = HedgeBacktestEventChunks(
            pair="BTC/USDT:USDT", timeframe="1m", frame=_frame(50, "0.9")
        )
        result = HedgeBacktesting(initial_balance=Decimal(1000)).run_compact(stream)
        self.assertEqual(result.report["wallet_processed_fill_id_count"], 0)
        self.assertEqual(result.report["wallet_realized_by_fill_count"], 0)

    def test_compact_and_detailed_fingerprint_match(self) -> None:
        from freqtrade.optimize.hedge_backtesting import (
            HedgeBacktestEventChunks,
            events_from_analyzed_dataframe,
        )

        frame = _frame(40)
        detailed = events_from_analyzed_dataframe(pair="BTC/USDT:USDT", timeframe="1m", frame=frame)
        stream = HedgeBacktestEventChunks(
            pair="BTC/USDT:USDT", timeframe="1m", frame=frame, chunk_bars=7
        )
        list(stream)
        self.assertEqual(detailed.data_fingerprint, stream.dataset().data_fingerprint)

    def test_tactical_closed_lot_pruning_preserves_net(self) -> None:
        from freqtrade.hedge.planning.context import PositionSide
        from freqtrade.hedge.simulation.cross_wallet import MutableLeg, MutableTacticalLot

        leg = MutableLeg(PositionSide.LONG)
        lot = MutableTacticalLot(
            lot_id="lot-a",
            quantity=Decimal(0),
            average_price=Decimal(100),
            opened_at=datetime(2026, 1, 1, tzinfo=UTC),
            realized_pnl=Decimal(12),
            fees=Decimal(2),
            funding=Decimal(1),
            closed_quantity=Decimal(3),
        )
        leg.tactical_lots[lot.lot_id] = lot
        before = leg.tactical_net_pnl()
        self.assertEqual(leg.prune_closed_tactical_lots(), 1)
        self.assertEqual(leg.tactical_lots, {})
        self.assertEqual(leg.tactical_net_pnl(), before)
        self.assertEqual(leg.tactical_closed_lot_count, 1)

    def test_wallet_online_realization_metrics_exist(self) -> None:
        from freqtrade.hedge.simulation.cross_wallet import CrossWallet

        wallet = CrossWallet(initial_balance=Decimal(1000))
        self.assertEqual(wallet.fill_count, 0)
        self.assertEqual(wallet.winning_realizations, 0)
        self.assertEqual(wallet.losing_realizations, 0)

    def test_stream_fingerprint_is_v3_and_deterministic(self) -> None:
        from freqtrade.optimize.hedge_backtesting import HedgeBacktestEventChunks

        values = []
        for _ in range(2):
            stream = HedgeBacktestEventChunks(
                pair="BTC/USDT:USDT", timeframe="1m", frame=_frame(20)
            )
            list(stream)
            values.append(stream.dataset().data_fingerprint)
        self.assertEqual(values[0], values[1])
        self.assertEqual(len(values[0]), 64)

    def test_stream_max_chunk_telemetry_is_scalar(self) -> None:
        from freqtrade.optimize.hedge_backtesting import HedgeBacktestEventChunks, HedgeBacktesting

        stream = HedgeBacktestEventChunks(
            pair="BTC/USDT:USDT", timeframe="1m", frame=_frame(100, "0")
        )
        result = HedgeBacktesting(initial_balance=Decimal(1000)).run_compact(stream)
        self.assertEqual(result.report["max_chunk_input_events"], 1)


if __name__ == "__main__":
    unittest.main()
