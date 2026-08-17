from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from freqtrade.hedge.contracts import IntentAction, PositionSide
from freqtrade.hedge.simulation.calibration import (
    ExecutionTraceState, TraceCorpus, calibrate_simulation_trace,
)


HASH = "e" * 64


def _state(*, equity: str = "1000") -> ExecutionTraceState:
    return ExecutionTraceState(
        sequence=0, client_order_id="cid-1", observed_at=datetime(2026, 8, 17, tzinfo=UTC),
        position_side=PositionSide.LONG, action=IntentAction.OPEN, accepted=True,
        fill_quantity=Decimal("1"), fill_price=Decimal("100"), fee=Decimal("0.04"),
        funding=Decimal(0), wallet_balance=Decimal("900"), equity=Decimal(equity),
        gross_notional=Decimal("100"), net_notional=Decimal("100"), pending_order_ids=(),
        liquidation_buffer=Decimal("0.5"), long_quantity=Decimal("1"), short_quantity=Decimal(0),
    )


def _calibrate(recorded: TraceCorpus, simulated: TraceCorpus):
    return calibrate_simulation_trace(
        recorded=recorded, simulated=simulated, source_authority_sha256=HASH,
        simulator_schema="event-replay-v1", matcher_sha256=HASH, cost_model_sha256=HASH,
        funding_model_sha256=HASH, latency_model_sha256=HASH,
    )


def test_exact_recorded_trace_is_a_passing_calibration_artifact() -> None:
    trace = TraceCorpus("binance-hedge-v1", HASH, (_state(),))
    artifact = _calibrate(trace, trace)
    assert artifact.passed
    assert artifact.state_match_rate == Decimal(1)
    assert len(artifact.fingerprint) == 64


def test_any_state_divergence_fails_closed_and_records_pnl_error() -> None:
    recorded = TraceCorpus("binance-hedge-v1", HASH, (_state(),))
    simulated = TraceCorpus("binance-hedge-v1", HASH, (_state(equity="999"),))
    artifact = _calibrate(recorded, simulated)
    assert not artifact.passed
    assert artifact.max_pnl_error == Decimal(1)
