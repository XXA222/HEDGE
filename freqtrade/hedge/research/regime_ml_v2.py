"""Multi-dimensional regime and decision-relevant supervised forecast contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import StrEnum

from freqtrade.hedge.contracts import finite_decimal
from freqtrade.hedge.contracts.types import required_text


class TrendRegime(StrEnum):
    DOWN = "DOWN"
    RANGE = "RANGE"
    UP = "UP"


class IntensityRegime(StrEnum):
    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"
    EXTREME = "EXTREME"


class DirectionalRegime(StrEnum):
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"
    POSITIVE = "POSITIVE"


@dataclass(frozen=True, slots=True)
class RegimeSnapshotV2:
    observed_at: datetime
    available_at: datetime
    trend: TrendRegime
    volatility: IntensityRegime
    liquidity: IntensityRegime
    funding: DirectionalRegime
    basis: DirectionalRegime
    crowding: IntensityRegime
    tail_risk: IntensityRegime
    correlation: IntensityRegime

    def __post_init__(self) -> None:
        for name in ("observed_at", "available_at"):
            value = getattr(self, name)
            if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
                raise ValueError(f"{name} must be timezone-aware")
            object.__setattr__(self, name, value.astimezone(UTC))
        if self.available_at < self.observed_at:
            raise ValueError("available_at cannot precede observed_at")
        enum_fields = {
            "trend": TrendRegime,
            "volatility": IntensityRegime,
            "liquidity": IntensityRegime,
            "funding": DirectionalRegime,
            "basis": DirectionalRegime,
            "crowding": IntensityRegime,
            "tail_risk": IntensityRegime,
            "correlation": IntensityRegime,
        }
        if any(not isinstance(getattr(self, name), enum) for name, enum in enum_fields.items()):
            raise TypeError("regime dimensions have invalid enum types")

    def usable_at(self, decision_time: datetime) -> bool:
        if decision_time.tzinfo is None or decision_time.utcoffset() is None:
            raise ValueError("decision_time must be timezone-aware")
        return self.available_at <= decision_time.astimezone(UTC)


class SupervisedTarget(StrEnum):
    RETURN_ABOVE_THRESHOLD_PROBABILITY = "return_above_threshold_probability"
    RETURN_Q10 = "return_q10"
    RETURN_Q50 = "return_q50"
    RETURN_Q90 = "return_q90"
    FUTURE_REALIZED_VOLATILITY = "future_realized_volatility"
    MAXIMUM_ADVERSE_EXCURSION = "maximum_adverse_excursion"
    MAXIMUM_FAVORABLE_EXCURSION = "maximum_favorable_excursion"
    TAIL_LOSS_PROBABILITY = "tail_loss_probability"
    REGIME_TRANSITION_PROBABILITY = "regime_transition_probability"
    EXECUTION_COST = "execution_cost"
    FILL_PROBABILITY = "fill_probability"


PROBABILITY_TARGETS = frozenset(
    {
        SupervisedTarget.RETURN_ABOVE_THRESHOLD_PROBABILITY,
        SupervisedTarget.TAIL_LOSS_PROBABILITY,
        SupervisedTarget.REGIME_TRANSITION_PROBABILITY,
        SupervisedTarget.FILL_PROBABILITY,
    }
)


class ForecastFamily(StrEnum):
    STATISTICAL = "STATISTICAL"
    LOGISTIC = "LOGISTIC"
    LIGHTGBM = "LIGHTGBM"
    HMM = "HMM"
    GMM = "GMM"
    FOUNDATION_CHALLENGER = "FOUNDATION_CHALLENGER"


@dataclass(frozen=True, slots=True)
class DecisionForecast:
    model_id: str
    family: ForecastFamily
    produced_at: datetime
    available_at: datetime
    horizon: timedelta
    values: Mapping[SupervisedTarget, Decimal]
    observation_only: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "model_id",
            required_text(self.model_id, field_name="model_id", max_length=128),
        )
        if not isinstance(self.family, ForecastFamily):
            raise TypeError("family must be ForecastFamily")
        for name in ("produced_at", "available_at"):
            value = getattr(self, name)
            if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
                raise ValueError(f"{name} must be timezone-aware")
            object.__setattr__(self, name, value.astimezone(UTC))
        if self.available_at < self.produced_at:
            raise ValueError("available_at cannot precede produced_at")
        if not isinstance(self.horizon, timedelta) or self.horizon <= timedelta(0):
            raise ValueError("horizon must be positive")
        if not isinstance(self.values, Mapping) or not self.values:
            raise ValueError("forecast values must be nonempty")
        normalized: dict[SupervisedTarget, Decimal] = {}
        for target, raw_value in self.values.items():
            if not isinstance(target, SupervisedTarget):
                raise TypeError("forecast keys must be SupervisedTarget")
            value = finite_decimal(raw_value, field_name=target.value)
            if target in PROBABILITY_TARGETS and not Decimal(0) <= value <= Decimal(1):
                raise ValueError(f"{target.value} must be within [0, 1]")
            normalized[target] = value
        object.__setattr__(self, "values", normalized)
        if self.observation_only is not True:
            raise ValueError("decision forecasts are observation-only and cannot be venue actions")

    def usable_at(self, decision_time: datetime) -> bool:
        if decision_time.tzinfo is None or decision_time.utcoffset() is None:
            raise ValueError("decision_time must be timezone-aware")
        return self.available_at <= decision_time.astimezone(UTC)


FOUNDATION_OUTPUTS = frozenset(
    {
        SupervisedTarget.RETURN_Q10,
        SupervisedTarget.RETURN_Q50,
        SupervisedTarget.RETURN_Q90,
        SupervisedTarget.FUTURE_REALIZED_VOLATILITY,
    }
)


def qualify_foundation_challenger(
    forecast: DecisionForecast,
    *,
    baseline_score: Decimal,
    challenger_score: Decimal,
    minimum_incremental_score: Decimal,
) -> tuple[bool, tuple[str, ...]]:
    baseline = finite_decimal(baseline_score, field_name="baseline_score")
    challenger = finite_decimal(challenger_score, field_name="challenger_score")
    increment = finite_decimal(minimum_incremental_score, field_name="minimum_incremental_score")
    if increment < 0:
        raise ValueError("minimum_incremental_score must be nonnegative")
    reasons: list[str] = []
    if forecast.family is not ForecastFamily.FOUNDATION_CHALLENGER:
        reasons.append("NOT_FOUNDATION_CHALLENGER")
    unsupported = set(forecast.values) - FOUNDATION_OUTPUTS
    reasons.extend(
        f"UNSUPPORTED_FOUNDATION_OUTPUT:{target.value}"
        for target in sorted(unsupported, key=lambda item: item.value)
    )
    if challenger - baseline < increment:
        reasons.append("NO_INCREMENTAL_EVIDENCE_OVER_BASELINE")
    return not reasons, tuple(reasons)
