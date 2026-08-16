from __future__ import annotations

from copy import deepcopy

from freqtrade.hedge.backtesting.consistency import CORE_REPORT_FIELDS, compare_backtest_results


def _payload(compact: bool):
    report = {field: "0" for field in CORE_REPORT_FIELDS}
    report.update(
        {
            "fill_count": 0,
            "winning_realizations": 0,
            "losing_realizations": 0,
            "equity_return_count": 59,
            "liquidated": False,
            "liquidation_count": 0,
        }
    )
    payload = {
        "pair": "BTC/USDT:USDT",
        "timeframe": "1m",
        "start": "2026-01-01T00:01:00+00:00",
        "end": "2026-01-01T01:00:00+00:00",
        "bar_count": 60,
        "signal_count": 60,
        "funding_count": 0,
        "missing_candle_count": 0,
        "data_fingerprint": "abc",
        "execution_timing": "NEXT_BAR_NO_LOOKAHEAD",
        "report": report,
        "hedge_native": {
            "metrics": {"sharpe": 1.5, "sortino": 2.0, "volatility": 0.3},
            "metadata": {
                "risk_metric_source": "BAR_RETURN_MOMENTS",
                "risk_periods_per_year": 365 * 24 * 60,
            },
        },
    }
    if compact:
        report.update(
            {
                "replay_mode": "COMPACT_ORDERED_STREAM_V2",
                "stream_row_mode": "INDEXED_ARRAY_VIEW_V2",
                "chronology_mode": "CACHED_TIMEFRAME_SECONDS_V2",
                "slot_validation_mode": "BITMASK_SLOT_VALIDATION_V1",
                "matcher_mode": "FLAT_IDLE_BYPASS_V1",
                "flat_idle_matcher_bypass_count": 10,
                "processed_bar_count": 60,
                "retained_snapshot_count": 2,
            }
        )
    return payload


def test_compact_detailed_consistency_passes_only_with_exact_core_semantics() -> None:
    report = compare_backtest_results(_payload(True), _payload(False))
    assert report.passed
    assert not report.mismatches
    assert report.gates["risk_metrics_use_same_bar_moments"]


def test_compact_detailed_consistency_detects_accounting_drift() -> None:
    compact = _payload(True)
    detailed = deepcopy(_payload(False))
    detailed["report"]["final_equity"] = "1"
    report = compare_backtest_results(compact, detailed)
    assert not report.passed
    assert "report.final_equity" in report.mismatches


def test_compact_detailed_consistency_detects_risk_metric_drift() -> None:
    compact = _payload(True)
    detailed = deepcopy(_payload(False))
    detailed["hedge_native"]["metrics"]["sharpe"] = 1.4
    report = compare_backtest_results(compact, detailed)
    assert not report.passed
    assert "hedge_native.metrics.sharpe" in report.mismatches
