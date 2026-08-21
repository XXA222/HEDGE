from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from freqtrade.hedge.operations.runtime import DryRunOperationsRuntime, OperationsCycleInput


def cycle_input(*, consistent: bool, coverage: Decimal) -> OperationsCycleInput:
    return OperationsCycleInput(
        timestamp=datetime(2026, 8, 21, tzinfo=UTC),
        symbol="BTCUSDT",
        timeframe_seconds=60,
        mark_price=Decimal(100),
        index_price=Decimal(100),
        equity=Decimal(1000),
        initial_equity=Decimal(1000),
        long_notional=Decimal(100),
        short_notional=Decimal(0),
        margin_used=Decimal(10),
        realized_pnl=Decimal(0),
        unrealized_pnl=Decimal(0),
        funding_pnl=Decimal(0),
        fees=Decimal(0),
        slippage_cost=Decimal(0),
        base_candles=2,
        informative_candles={},
        reconciliation_fresh=True,
        api_healthy=True,
        dashboard_healthy=True,
        business_reconciliation_consistent=consistent,
        managed_order_identity_coverage=coverage,
        business_trade_display_ids=("BTCUSDT-L-000012",),
        business_reconciliation_issues=(
            () if consistent else ("MANAGED_ORDER_IDENTITY_MISSING:order=o-1",)
        ),
    )


def test_operations_dashboard_exposes_business_identity_health() -> None:
    runtime = DryRunOperationsRuntime(
        account_id="paper-main",
        symbols=("BTCUSDT",),
        config={"hedge": {"operations": {"warmup_candles": 1}}},
    )
    snapshot = runtime.observe(cycle_input(consistent=True, coverage=Decimal(1)))
    summary = snapshot.summary()
    assert snapshot.business_identity_ready
    assert summary["business_reconciliation_consistent"] is True
    assert summary["managed_order_identity_coverage"] == "1"
    assert summary["business_trade_display_ids"] == ("BTCUSDT-L-000012",)
    assert runtime.certificate(at=datetime(2026, 8, 21, 0, 1, tzinfo=UTC)).ready


def test_business_identity_drift_disables_new_risk_and_readiness() -> None:
    runtime = DryRunOperationsRuntime(
        account_id="paper-main",
        symbols=("BTCUSDT",),
        config={"hedge": {"operations": {"warmup_candles": 1}}},
    )
    snapshot = runtime.observe(cycle_input(consistent=False, coverage=Decimal("0.5")))
    assert not snapshot.business_identity_ready
    assert not snapshot.new_risk_enabled
    assert not snapshot.ready
    assert "MANAGED_ORDER_IDENTITY_MISSING:order=o-1" in snapshot.diagnostics
    certificate = runtime.certificate(at=datetime(2026, 8, 21, 0, 1, tzinfo=UTC))
    assert not certificate.ready
    assert any(
        check.name == "BUSINESS_IDENTITY_RECONCILIATION" and not check.passed
        for check in certificate.checks
    )
