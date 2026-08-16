"""Runtime projection publishing for the integrated Paper application."""

from __future__ import annotations

from typing import Any

from freqtrade.enums.hedge import PositionSide as RiskPositionSide
from freqtrade.hedge.planning.context import PositionSide
from freqtrade.hedge.position_book import PositionRecord
from freqtrade.hedge.runtime import HedgeProjectionSource, HedgeRuntime


class PaperRuntimePublisherMixin:
    """Publish Paper facts with Paper-specific health semantics."""

    def publish_runtime(
        self: Any,
        runtime: HedgeRuntime,
        *,
        market_data_fresh: bool = True,
        funding_source_healthy: bool = True,
        market_source: str = "UNKNOWN",
    ) -> None:
        market = self.last_market
        if market is None:
            return
        wallet = self.wallet(market)
        positions = []
        for side, leg in ((PositionSide.LONG, wallet.long), (PositionSide.SHORT, wallet.short)):
            if leg.quantity <= 0:
                continue
            positions.append(
                PositionRecord(
                    symbol=self.symbol,
                    position_side=RiskPositionSide(side.value),
                    amount=leg.quantity,
                    entry_price=leg.average_price,
                    mark_price=market.mark,
                    unrealized_pnl=leg.unrealized_pnl(market.mark),
                    leverage=self.leverage,
                    collateral=leg.quantity * market.mark / self.leverage,
                    source="PAPER_EXECUTION",
                    exchange="paper",
                    account_id=self.account_id,
                )
            )
        risk = self.risk_portfolio().account
        runtime.publish(
            source=HedgeProjectionSource.PAPER,
            positions=tuple(positions),
            risk=risk,
            reconciliation_status="NOT_APPLICABLE",
            reconciliation_at=None,
            reconciliation_details=("source=paper-execution-ledger",),
            stream_state="NOT_APPLICABLE",
            stream_last_event_at=None,
            stream_reconnect_count=0,
            checks={
                "common.persistence_healthy": self._state_durable,
                "paper.market_data_fresh": market_data_fresh,
                "paper.funding_source_healthy": funding_source_healthy,
                "paper.account_events_durable": (
                    self._state_durable and self.paper_config.account_events_enabled
                )
                or not self.paper_config.account_events_enabled,
                "paper.simulation_engine_healthy": True,
                "paper.ledger_durable": self._state_durable,
                "paper.risk_snapshot_valid": risk.effective_risk_data_valid,
            },
            reasons=(
                ()
                if self._state_durable and risk.effective_risk_data_valid
                else tuple(
                    dict.fromkeys(
                        (
                            *(() if self._state_durable else ("PAPER_LEDGER_NOT_DURABLE",)),
                            *risk.risk_data_errors,
                        )
                    )
                )
            ),
            source_version=f"{market_source}:{market.timestamp.isoformat()}",
            source_event_time=market.timestamp,
            stale=False,
        )
