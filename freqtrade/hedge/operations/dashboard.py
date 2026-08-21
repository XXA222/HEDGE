"""immutable operations snapshot for API/UI projection."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from .alerts import AlertRecord
from .attribution import PerformanceAttribution
from .breaker import BreakerDecision
from .common import ensure_aware
from .risk import PortfolioRiskSnapshot


@dataclass(frozen=True, slots=True)
class OperationsDashboardSnapshot:
    generated_at: datetime
    session_id: str
    cycle_id: str | None
    symbols: tuple[str, ...]
    state: str
    market_ready: bool
    warmup_ready: bool
    risk: PortfolioRiskSnapshot | None
    breaker: BreakerDecision | None
    attribution: PerformanceAttribution | None
    active_alerts: tuple[AlertRecord, ...]
    diagnostics: tuple[str, ...]
    new_risk_enabled: bool
    last_candle_age_seconds: Decimal | None = None
    candle_gap_seconds: Decimal | None = None
    strategy_cycle_count: int = 0
    order_count: int = 0
    fill_count: int = 0
    active_order_count: int = 0
    reconciliation_fresh: bool = False
    runtime_quality_level: int = 1
    runtime_quality_state: str = "RUNNING_UNVERIFIED"
    business_reconciliation_consistent: bool | None = None
    managed_order_identity_coverage: Decimal | None = None
    business_trade_display_ids: tuple[str, ...] = ()
    business_reconciliation_issues: tuple[str, ...] = ()
    protection_reconciliation_consistent: bool | None = None
    protection_coverage: Decimal | None = None
    stop_coverage: Decimal | None = None
    protection_reconciliation_issues: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        ensure_aware(self.generated_at)
        coverage = self.managed_order_identity_coverage
        if coverage is not None:
            if not coverage.is_finite() or coverage < 0 or coverage > 1:
                raise ValueError("managed order identity coverage must be within [0, 1]")
        if len(self.business_trade_display_ids) != len(set(self.business_trade_display_ids)):
            raise ValueError("business trade display ids must be unique")
        if len(self.business_trade_display_ids) > 1000:
            raise ValueError("too many business trade display ids")
        if len(self.business_reconciliation_issues) > 100:
            raise ValueError("too many business reconciliation issues")
        for field_name in ("protection_coverage", "stop_coverage"):
            value = getattr(self, field_name)
            if value is not None and (
                not value.is_finite() or value < 0 or value > 1
            ):
                raise ValueError(f"{field_name} must be within [0, 1]")
        if len(self.protection_reconciliation_issues) > 100:
            raise ValueError("too many protection reconciliation issues")

    @property
    def business_identity_ready(self) -> bool:
        coverage = self.managed_order_identity_coverage
        return (
            self.business_reconciliation_consistent is not False
            and (coverage is None or coverage == Decimal(1))
        )

    @property
    def protection_ready(self) -> bool:
        return (
            self.protection_reconciliation_consistent is not False
            and (
                self.protection_coverage is None
                or self.protection_coverage == Decimal(1)
            )
            and (self.stop_coverage is None or self.stop_coverage == Decimal(1))
        )

    @property
    def ready(self) -> bool:
        return (
            self.state == "RUNNING"
            and self.market_ready
            and self.warmup_ready
            and (self.risk is None or self.risk.ready)
            and self.business_identity_ready
            and self.protection_ready
            and self.new_risk_enabled
        )

    def summary(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "cycle_id": self.cycle_id,
            "state": self.state,
            "symbols": self.symbols,
            "ready": self.ready,
            "new_risk_enabled": self.new_risk_enabled,
            "market_ready": self.market_ready,
            "warmup_ready": self.warmup_ready,
            "gross_ratio": None if self.risk is None else str(self.risk.gross_ratio),
            "margin_ratio": None if self.risk is None else str(self.risk.margin_ratio),
            "drawdown": None if self.breaker is None else str(self.breaker.drawdown),
            "net_pnl": None if self.attribution is None else str(self.attribution.net_pnl),
            "active_alert_count": len(self.active_alerts),
            "diagnostics": self.diagnostics,
            "generated_at": self.generated_at.isoformat(),
            "last_candle_age_seconds": (
                None if self.last_candle_age_seconds is None else str(self.last_candle_age_seconds)
            ),
            "candle_gap_seconds": (
                None if self.candle_gap_seconds is None else str(self.candle_gap_seconds)
            ),
            "strategy_cycle_count": self.strategy_cycle_count,
            "order_count": self.order_count,
            "fill_count": self.fill_count,
            "active_order_count": self.active_order_count,
            "reconciliation_fresh": self.reconciliation_fresh,
            "runtime_quality_level": self.runtime_quality_level,
            "runtime_quality_state": self.runtime_quality_state,
            "business_reconciliation_consistent": (
                self.business_reconciliation_consistent
            ),
            "managed_order_identity_coverage": (
                None
                if self.managed_order_identity_coverage is None
                else str(self.managed_order_identity_coverage)
            ),
            "business_trade_display_ids": self.business_trade_display_ids,
            "business_reconciliation_issues": self.business_reconciliation_issues,
            "business_identity_ready": self.business_identity_ready,
            "protection_reconciliation_consistent": (
                self.protection_reconciliation_consistent
            ),
            "protection_coverage": (
                None
                if self.protection_coverage is None
                else str(self.protection_coverage)
            ),
            "stop_coverage": (
                None if self.stop_coverage is None else str(self.stop_coverage)
            ),
            "protection_reconciliation_issues": self.protection_reconciliation_issues,
            "protection_ready": self.protection_ready,
        }
