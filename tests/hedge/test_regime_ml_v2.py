from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from freqtrade.hedge.research.regime_ml_v2 import (
    DecisionForecast,
    DirectionalRegime,
    ForecastFamily,
    IntensityRegime,
    RegimeSnapshotV2,
    SupervisedTarget,
    TrendRegime,
    qualify_foundation_challenger,
)


NOW = datetime(2026, 8, 17, tzinfo=UTC)


def test_regime_snapshot_is_point_in_time_safe() -> None:
    snapshot = RegimeSnapshotV2(
        observed_at=NOW,
        available_at=NOW + timedelta(minutes=1),
        trend=TrendRegime.UP,
        volatility=IntensityRegime.NORMAL,
        liquidity=IntensityRegime.HIGH,
        funding=DirectionalRegime.POSITIVE,
        basis=DirectionalRegime.NEUTRAL,
        crowding=IntensityRegime.LOW,
        tail_risk=IntensityRegime.NORMAL,
        correlation=IntensityRegime.HIGH,
    )
    assert not snapshot.usable_at(NOW)
    assert snapshot.usable_at(NOW + timedelta(minutes=1))


def _forecast(**changes: object) -> DecisionForecast:
    values = {
        "model_id": "chronos-challenger",
        "family": ForecastFamily.FOUNDATION_CHALLENGER,
        "produced_at": NOW,
        "available_at": NOW + timedelta(seconds=5),
        "horizon": timedelta(hours=1),
        "values": {
            SupervisedTarget.RETURN_Q10: Decimal("-0.02"),
            SupervisedTarget.RETURN_Q50: Decimal("0.01"),
            SupervisedTarget.RETURN_Q90: Decimal("0.04"),
        },
    }
    values.update(changes)
    return DecisionForecast(**values)  # type: ignore[arg-type]


def test_decision_forecast_validates_probabilities_and_is_observation_only() -> None:
    forecast = _forecast()
    assert not forecast.usable_at(NOW)
    with pytest.raises(ValueError, match="within"):
        _forecast(values={SupervisedTarget.FILL_PROBABILITY: Decimal("1.1")})
    with pytest.raises(ValueError, match="observation-only"):
        _forecast(observation_only=False)


def test_foundation_model_requires_safe_outputs_and_incremental_evidence() -> None:
    passed, reasons = qualify_foundation_challenger(
        _forecast(),
        baseline_score=Decimal("0.5"),
        challenger_score=Decimal("0.56"),
        minimum_incremental_score=Decimal("0.05"),
    )
    assert passed
    assert reasons == ()

    unsafe = _forecast(values={SupervisedTarget.FILL_PROBABILITY: Decimal("0.8")})
    passed, reasons = qualify_foundation_challenger(
        unsafe,
        baseline_score=Decimal("0.5"),
        challenger_score=Decimal("0.51"),
        minimum_incremental_score=Decimal("0.05"),
    )
    assert not passed
    assert "UNSUPPORTED_FOUNDATION_OUTPUT:fill_probability" in reasons
    assert "NO_INCREMENTAL_EVIDENCE_OVER_BASELINE" in reasons
