from __future__ import annotations

from decimal import Decimal

from freqtrade.hedge.execution.costs import (
    ExecutionCostEstimate, STANDARD_COST_STRESS_GRID,
)


def _estimate() -> ExecutionCostEstimate:
    return ExecutionCostEstimate(
        maker_fee=Decimal("1"), taker_fee=Decimal(0), spread=Decimal("2"),
        slippage=Decimal("3"), impact=Decimal("4"), funding=Decimal("5"),
        basis_cost=Decimal("6"), latency_penalty=Decimal("7"),
        adverse_selection=Decimal("8"), uncertainty=Decimal("9"),
    )


def test_cost_estimate_is_exact_and_net_alpha_is_after_all_costs() -> None:
    estimate = _estimate()
    assert estimate.total_expected_cost == Decimal("45")
    assert estimate.net_expected_alpha(Decimal("50")) == Decimal("5")


def test_cost_stress_grid_increases_cost_and_handles_latency_separately() -> None:
    stressed = STANDARD_COST_STRESS_GRID[-1].apply(_estimate())
    assert stressed.funding == Decimal("10")
    assert stressed.latency_penalty == Decimal("14")
    assert stressed.total_expected_cost > _estimate().total_expected_cost
