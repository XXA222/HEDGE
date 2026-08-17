from decimal import Decimal

from freqtrade.hedge.production.drift_v2 import DriftSnapshot, ModelHealthState, evaluate_drift


def test_multi_axis_drift_uses_worst_axis_for_risk_response() -> None:
    result = evaluate_drift(DriftSnapshot(Decimal("0.1"), Decimal("0.4"), Decimal("0.1"), Decimal("0.1")))
    assert result.health is ModelHealthState.DEGRADED
    assert result.pause_new_risk


def test_nonfinite_health_flag_halts_model() -> None:
    result = evaluate_drift(DriftSnapshot(Decimal(0), Decimal(0), Decimal(0), Decimal(0), finite=False))
    assert result.health is ModelHealthState.HALTED
    assert result.risk_multiplier == 0
