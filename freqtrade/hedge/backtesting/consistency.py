"""Deterministic compact-vs-detailed Hedge backtest consistency gate."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any


# Stable minimum surface used by tests and external tooling.  The comparator also checks
# every detailed report key dynamically, so newly-added semantic metrics cannot silently
# diverge between compact and detailed replay.
CORE_REPORT_FIELDS = (
    "initial_balance",
    "total_pnl",
    "total_return_ratio",
    "final_long_quantity",
    "final_short_quantity",
    "long_pnl",
    "short_pnl",
    "long_trading_pnl",
    "short_trading_pnl",
    "funding",
    "fees",
    "maker_fees",
    "taker_fees",
    "fill_count",
    "winning_realizations",
    "losing_realizations",
    "gross_realized_profit",
    "gross_realized_loss",
    "gross_peak",
    "max_drawdown",
    "equity_return_count",
    "equity_return_sum",
    "equity_return_sum_squares",
    "equity_downside_square_sum",
    "final_equity",
    "final_balance",
    "final_gross_notional",
    "liquidated",
    "liquidation_count",
    "pnl_reconciliation_error",
)

# Compact replay adds operational evidence that has no detailed-path equivalent.
COMPACT_ONLY_REPORT_FIELDS = frozenset(
    {
        "replay_mode",
        "processed_chunk_count",
        "processed_input_event_count",
        "processed_bar_count",
        "slot_validation_mode",
        "matcher_mode",
        "flat_idle_matcher_bypass_count",
        "retained_event_count",
        "retained_snapshot_count",
        "snapshot_stride_bars",
        "max_chunk_input_events",
        "material_event_count",
        "wallet_processed_fill_id_count",
        "wallet_realized_by_fill_count",
        "wallet_tactical_lot_object_count",
        "stream_row_mode",
        "chronology_mode",
    }
)

RISK_METRIC_FIELDS = ("sharpe", "sortino", "volatility")


@dataclass(frozen=True, slots=True)
class BacktestConsistencyReport:
    passed: bool
    gates: Mapping[str, bool]
    mismatches: Mapping[str, tuple[Any, Any]]
    compact_bar_count: int
    detailed_bar_count: int
    semantic_report_field_count: int

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["schema"] = "hedge-backtest-compact-detailed-consistency-v2"
        payload["status"] = "PASS" if self.passed else "FAIL"
        payload["gates"] = dict(self.gates)
        payload["mismatches"] = {
            key: {"compact": left, "detailed": right}
            for key, (left, right) in self.mismatches.items()
        }
        return payload


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be an object")
    return value


def _nested_mapping(root: Mapping[str, Any], *keys: str) -> Mapping[str, Any]:
    current: object = root
    path: list[str] = []
    for key in keys:
        path.append(key)
        if not isinstance(current, Mapping):
            raise TypeError(".".join(path[:-1]) + " must be an object")
        current = current.get(key)
    return _mapping(current, ".".join(keys))


def compare_backtest_results(
    compact: Mapping[str, Any],
    detailed: Mapping[str, Any],
) -> BacktestConsistencyReport:
    compact_report = _mapping(compact.get("report"), "compact.report")
    detailed_report = _mapping(detailed.get("report"), "detailed.report")
    mismatches: dict[str, tuple[Any, Any]] = {}

    top_level_fields = (
        "pair",
        "timeframe",
        "start",
        "end",
        "bar_count",
        "signal_count",
        "funding_count",
        "missing_candle_count",
        "data_fingerprint",
        "execution_timing",
    )
    for field in top_level_fields:
        left = compact.get(field)
        right = detailed.get(field)
        if left != right:
            mismatches[f"root.{field}"] = (left, right)

    semantic_fields = sorted(
        (set(detailed_report) | set(CORE_REPORT_FIELDS)) - COMPACT_ONLY_REPORT_FIELDS
    )
    for field in semantic_fields:
        left = compact_report.get(field)
        right = detailed_report.get(field)
        if left != right:
            mismatches[f"report.{field}"] = (left, right)

    # R4 makes detailed and compact risk-adjusted metrics derive from the same exact
    # bar-return moments.  Compare these separately because they live in native output.
    compact_metrics = _nested_mapping(compact, "hedge_native", "metrics")
    detailed_metrics = _nested_mapping(detailed, "hedge_native", "metrics")
    for field in RISK_METRIC_FIELDS:
        left = compact_metrics.get(field)
        right = detailed_metrics.get(field)
        if left != right:
            mismatches[f"hedge_native.metrics.{field}"] = (left, right)

    compact_metadata = _nested_mapping(compact, "hedge_native", "metadata")
    detailed_metadata = _nested_mapping(detailed, "hedge_native", "metadata")
    for field in ("risk_metric_source", "risk_periods_per_year"):
        left = compact_metadata.get(field)
        right = detailed_metadata.get(field)
        if left != right:
            mismatches[f"hedge_native.metadata.{field}"] = (left, right)

    compact_bars = int(compact_report.get("processed_bar_count", compact.get("bar_count", 0)))
    detailed_bars = int(detailed.get("bar_count", 0))
    compact_return_count = int(compact_report.get("equity_return_count", -1))
    detailed_return_count = int(detailed_report.get("equity_return_count", -1))
    compact_markers = bool(
        compact_report.get("replay_mode") == "COMPACT_ORDERED_STREAM_V2"
        and compact_report.get("stream_row_mode") == "INDEXED_ARRAY_VIEW_V2"
        and compact_report.get("chronology_mode") == "CACHED_TIMEFRAME_SECONDS_V2"
        and compact_report.get("slot_validation_mode") == "BITMASK_SLOT_VALIDATION_V1"
        and compact_report.get("matcher_mode") == "FLAT_IDLE_BYPASS_V1"
    )
    risk_moments = bool(
        compact_metadata.get("risk_metric_source") == "BAR_RETURN_MOMENTS"
        and detailed_metadata.get("risk_metric_source") == "BAR_RETURN_MOMENTS"
    )
    semantic_mismatches = {
        key: value
        for key, value in mismatches.items()
        if key.startswith("report.")
        or key.startswith("hedge_native.metrics.")
        or key.startswith("hedge_native.metadata.")
    }
    gates = {
        "same_canonical_dataset": (
            compact.get("data_fingerprint") == detailed.get("data_fingerprint")
        ),
        "same_bar_count": compact_bars == detailed_bars and compact_bars > 0,
        "same_semantic_report_and_risk_metrics": not semantic_mismatches,
        "same_top_level_identity": not any(key.startswith("root.") for key in mismatches),
        "compact_cpu_memory_path_verified": compact_markers,
        "bar_return_moment_count_exact": (
            compact_return_count == max(compact_bars - 1, 0)
            and detailed_return_count == max(detailed_bars - 1, 0)
        ),
        "risk_metrics_use_same_bar_moments": risk_moments,
    }
    return BacktestConsistencyReport(
        passed=all(gates.values()),
        gates=gates,
        mismatches=mismatches,
        compact_bar_count=compact_bars,
        detailed_bar_count=detailed_bars,
        semantic_report_field_count=len(semantic_fields),
    )
