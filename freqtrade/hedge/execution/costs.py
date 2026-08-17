"""Exact, stressable execution-cost contract used before risk-increasing orders."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from freqtrade.hedge.contracts import finite_decimal


def _cost(value: object, *, name: str) -> Decimal:
    result = finite_decimal(value, field_name=name)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


@dataclass(frozen=True, slots=True)
class ExecutionCostEstimate:
    """Expected account-currency costs; every component is explicit and additive."""

    maker_fee: Decimal
    taker_fee: Decimal
    spread: Decimal
    slippage: Decimal
    impact: Decimal
    funding: Decimal
    basis_cost: Decimal
    latency_penalty: Decimal
    adverse_selection: Decimal
    uncertainty: Decimal

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, _cost(getattr(self, name), name=name))

    @property
    def total_expected_cost(self) -> Decimal:
        return sum((getattr(self, name) for name in self.__dataclass_fields__), Decimal(0))

    def stressed(self, *, cost_multiplier: Decimal = Decimal(1), latency_multiplier: Decimal = Decimal(1)) -> "ExecutionCostEstimate":
        cost_multiplier = _cost(cost_multiplier, name="cost_multiplier")
        latency_multiplier = _cost(latency_multiplier, name="latency_multiplier")
        return ExecutionCostEstimate(
            maker_fee=self.maker_fee * cost_multiplier,
            taker_fee=self.taker_fee * cost_multiplier,
            spread=self.spread * cost_multiplier,
            slippage=self.slippage * cost_multiplier,
            impact=self.impact * cost_multiplier,
            funding=self.funding * cost_multiplier,
            basis_cost=self.basis_cost * cost_multiplier,
            latency_penalty=self.latency_penalty * latency_multiplier,
            adverse_selection=self.adverse_selection * cost_multiplier,
            uncertainty=self.uncertainty * cost_multiplier,
        )

    def net_expected_alpha(self, gross_alpha: Decimal) -> Decimal:
        return finite_decimal(gross_alpha, field_name="gross_alpha") - self.total_expected_cost


@dataclass(frozen=True, slots=True)
class ExecutionCostStressScenario:
    name: str
    cost_multiplier: Decimal
    latency_multiplier: Decimal
    funding_multiplier: Decimal

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name is required")
        for name in ("cost_multiplier", "latency_multiplier", "funding_multiplier"):
            value = _cost(getattr(self, name), name=name)
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)

    def apply(self, estimate: ExecutionCostEstimate) -> ExecutionCostEstimate:
        if not isinstance(estimate, ExecutionCostEstimate):
            raise TypeError("estimate must be ExecutionCostEstimate")
        base = estimate.stressed(
            cost_multiplier=self.cost_multiplier,
            latency_multiplier=self.latency_multiplier,
        )
        return ExecutionCostEstimate(
            maker_fee=base.maker_fee, taker_fee=base.taker_fee, spread=base.spread,
            slippage=base.slippage, impact=base.impact,
            funding=estimate.funding * self.funding_multiplier,
            basis_cost=base.basis_cost, latency_penalty=base.latency_penalty,
            adverse_selection=base.adverse_selection, uncertainty=base.uncertainty,
        )


STANDARD_COST_STRESS_GRID = (
    ExecutionCostStressScenario("BASE", Decimal(1), Decimal(1), Decimal(1)),
    ExecutionCostStressScenario("COST_1_5", Decimal("1.5"), Decimal(1), Decimal(1)),
    ExecutionCostStressScenario("COST_2_LATENCY_2", Decimal(2), Decimal(2), Decimal(2)),
)
