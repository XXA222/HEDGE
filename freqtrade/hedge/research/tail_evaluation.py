"""Distributional and tail-aware evaluation used by promotion gates."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from freqtrade.hedge.contracts import finite_decimal
from freqtrade.hedge.contracts.types import required_text


def _decimal_series(values: tuple[Decimal, ...], *, field_name: str) -> tuple[Decimal, ...]:
    if not values:
        raise ValueError(f"{field_name} must be nonempty")
    return tuple(finite_decimal(value, field_name=field_name) for value in values)


def _quantile(values: tuple[Decimal, ...], quantile: Decimal) -> Decimal:
    ordered = tuple(sorted(values))
    if len(ordered) == 1:
        return ordered[0]
    position = quantile * Decimal(len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - Decimal(lower)
    return ordered[lower] * (Decimal(1) - weight) + ordered[upper] * weight


@dataclass(frozen=True, slots=True)
class TailMetrics:
    observation_count: int
    downside_quantile: Decimal
    value_at_risk: Decimal
    conditional_value_at_risk: Decimal
    tail_loss_probability: Decimal
    downside_deviation: Decimal
    worst_return: Decimal


def evaluate_tail(
    returns: tuple[Decimal, ...],
    *,
    quantile: Decimal = Decimal("0.05"),
    tail_loss_threshold: Decimal = Decimal("-0.02"),
) -> TailMetrics:
    values = _decimal_series(returns, field_name="return")
    q = finite_decimal(quantile, field_name="quantile")
    threshold = finite_decimal(tail_loss_threshold, field_name="tail_loss_threshold")
    if not Decimal(0) < q < Decimal(1):
        raise ValueError("quantile must be within (0, 1)")
    downside_quantile = _quantile(values, q)
    tail = tuple(value for value in values if value <= downside_quantile)
    losses = tuple(max(Decimal(0), -value) for value in tail)
    negative_squares = tuple(min(value, Decimal(0)) ** 2 for value in values)
    return TailMetrics(
        observation_count=len(values),
        downside_quantile=downside_quantile,
        value_at_risk=max(Decimal(0), -downside_quantile),
        conditional_value_at_risk=sum(losses, Decimal(0)) / Decimal(len(losses)),
        tail_loss_probability=Decimal(sum(value <= threshold for value in values))
        / Decimal(len(values)),
        downside_deviation=(sum(negative_squares, Decimal(0)) / Decimal(len(values))).sqrt(),
        worst_return=min(values),
    )


@dataclass(frozen=True, slots=True)
class TailScenario:
    name: str
    expected_edge: Decimal
    uncertainty: Decimal
    metrics: TailMetrics

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "name",
            required_text(self.name, field_name="name", max_length=128),
        )
        for field_name in ("expected_edge", "uncertainty"):
            value = finite_decimal(getattr(self, field_name), field_name=field_name)
            if field_name == "uncertainty" and value < 0:
                raise ValueError("uncertainty must be nonnegative")
            object.__setattr__(self, field_name, value)
        if not isinstance(self.metrics, TailMetrics):
            raise TypeError("metrics must be TailMetrics")


def qualify_tail_scenarios(
    scenarios: tuple[TailScenario, ...],
    *,
    minimum_expected_edge: Decimal,
    maximum_cvar: Decimal,
    maximum_tail_loss_probability: Decimal,
    maximum_uncertainty: Decimal,
) -> tuple[bool, tuple[str, ...]]:
    if not scenarios:
        return False, ("TAIL_SCENARIOS_MISSING",)
    thresholds = {
        "minimum_expected_edge": finite_decimal(
            minimum_expected_edge, field_name="minimum_expected_edge"
        ),
        "maximum_cvar": finite_decimal(maximum_cvar, field_name="maximum_cvar"),
        "maximum_tail_loss_probability": finite_decimal(
            maximum_tail_loss_probability,
            field_name="maximum_tail_loss_probability",
        ),
        "maximum_uncertainty": finite_decimal(
            maximum_uncertainty, field_name="maximum_uncertainty"
        ),
    }
    if any(value < 0 for name, value in thresholds.items() if name != "minimum_expected_edge"):
        raise ValueError("tail maximum thresholds must be nonnegative")
    reasons: list[str] = []
    for scenario in scenarios:
        if scenario.expected_edge < thresholds["minimum_expected_edge"]:
            reasons.append(f"{scenario.name}:EDGE_BELOW_MINIMUM")
        if scenario.metrics.conditional_value_at_risk > thresholds["maximum_cvar"]:
            reasons.append(f"{scenario.name}:CVAR_ABOVE_MAXIMUM")
        if scenario.metrics.tail_loss_probability > thresholds["maximum_tail_loss_probability"]:
            reasons.append(f"{scenario.name}:TAIL_LOSS_PROBABILITY_ABOVE_MAXIMUM")
        if scenario.uncertainty > thresholds["maximum_uncertainty"]:
            reasons.append(f"{scenario.name}:UNCERTAINTY_ABOVE_MAXIMUM")
    return not reasons, tuple(reasons)
