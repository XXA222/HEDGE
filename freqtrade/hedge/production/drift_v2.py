"""Unified model drift health and risk response across feature, prediction, action and PnL."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum

from freqtrade.hedge.contracts import finite_decimal


class ModelHealthState(StrEnum):
    HEALTHY = "HEALTHY"
    WATCH = "WATCH"
    DEGRADED = "DEGRADED"
    HALTED = "HALTED"


@dataclass(frozen=True, slots=True)
class DriftSnapshot:
    feature_drift: Decimal
    prediction_drift: Decimal
    action_drift: Decimal
    performance_drift: Decimal
    finite: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.finite, bool):
            raise TypeError("finite must be bool")
        for name in ("feature_drift", "prediction_drift", "action_drift", "performance_drift"):
            value = finite_decimal(getattr(self, name), field_name=name)
            if value < 0:
                raise ValueError("drift metrics must be nonnegative")
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class DriftRiskResponse:
    health: ModelHealthState
    risk_multiplier: Decimal
    pause_new_risk: bool
    reasons: tuple[str, ...]


def evaluate_drift(snapshot: DriftSnapshot, *, watch: Decimal = Decimal("0.15"), degraded: Decimal = Decimal("0.30"), halted: Decimal = Decimal("0.50")) -> DriftRiskResponse:
    limits = tuple(finite_decimal(value, field_name="drift threshold") for value in (watch, degraded, halted))
    if not Decimal(0) <= limits[0] < limits[1] < limits[2]:
        raise ValueError("drift thresholds must be strictly increasing")
    if not snapshot.finite:
        return DriftRiskResponse(ModelHealthState.HALTED, Decimal(0), True, ("NONFINITE_DRIFT",))
    values = {name: getattr(snapshot, name) for name in ("feature_drift", "prediction_drift", "action_drift", "performance_drift")}
    peak = max(values.values())
    reasons = tuple(name.upper() for name, value in values.items() if value >= limits[0])
    if peak >= limits[2]:
        return DriftRiskResponse(ModelHealthState.HALTED, Decimal(0), True, reasons)
    if peak >= limits[1]:
        return DriftRiskResponse(ModelHealthState.DEGRADED, Decimal("0.25"), True, reasons)
    if peak >= limits[0]:
        return DriftRiskResponse(ModelHealthState.WATCH, Decimal("0.5"), False, reasons)
    return DriftRiskResponse(ModelHealthState.HEALTHY, Decimal(1), False, ())
