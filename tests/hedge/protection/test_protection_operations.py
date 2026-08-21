from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from freqtrade.hedge.operations.runtime import DryRunOperationsRuntime, OperationsCycleInput


def _cycle(*, protection_ok: bool) -> OperationsCycleInput:
    timestamp = datetime(2026, 8, 21, tzinfo=UTC)
    return OperationsCycleInput(
        timestamp,
        "BTCUSDT",
        60,
        Decimal(100),
        Decimal(100),
        Decimal(1000),
        Decimal(1000),
        Decimal(100),
        Decimal(50),
        Decimal(20),
        Decimal(1),
        Decimal(2),
        Decimal(0),
        Decimal(1),
        Decimal("0.1"),
        2,
        {},
        reconciliation_fresh=True,
        business_reconciliation_consistent=True,
        managed_order_identity_coverage=Decimal(1),
        protection_reconciliation_consistent=protection_ok,
        protection_coverage=Decimal(1) if protection_ok else Decimal("0.5"),
        stop_coverage=Decimal(1) if protection_ok else Decimal("0.5"),
        protection_reconciliation_issues=(
            () if protection_ok else ("BUSINESS_LOT_PROTECTION_MISSING",)
        ),
    )


def test_operations_new_risk_fails_closed_on_business_lot_protection_drift(
    tmp_path: Path,
) -> None:
    runtime = DryRunOperationsRuntime(
        account_id="main",
        symbols=("BTCUSDT",),
        config={"hedge": {"operations": {"warmup_candles": 2}}},
        state_path=tmp_path / "state.json",
    )
    snapshot = runtime.observe(_cycle(protection_ok=False))
    assert not snapshot.protection_ready
    assert not snapshot.new_risk_enabled
    assert not snapshot.ready
    assert "BUSINESS_LOT_PROTECTION_MISSING" in snapshot.diagnostics
    certificate = runtime.certificate(at=datetime(2026, 8, 21, 0, 1, tzinfo=UTC))
    checks = {item.name: item.passed for item in certificate.checks}
    assert checks["BUSINESS_IDENTITY_RECONCILIATION"]
    assert not checks["BUSINESS_LOT_PROTECTION"]


def test_operations_accepts_full_business_lot_protection_coverage(tmp_path: Path) -> None:
    runtime = DryRunOperationsRuntime(
        account_id="main",
        symbols=("BTCUSDT",),
        config={"hedge": {"operations": {"warmup_candles": 2}}},
        state_path=tmp_path / "state.json",
    )
    snapshot = runtime.observe(_cycle(protection_ok=True))
    assert snapshot.protection_ready
    assert snapshot.new_risk_enabled
    assert snapshot.summary()["protection_coverage"] == "1"
    assert snapshot.summary()["stop_coverage"] == "1"
