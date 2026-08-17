from __future__ import annotations

from decimal import Decimal

from freqtrade.hedge.contracts import ExecutionOrderIntent, IntentAction, OrderType, PositionSide
from freqtrade.hedge.execution.algorithms import ExecutionAlgorithm, ExecutionAlgorithmContext, plan_execution


def _intent(*, quantity: str = "10", action: IntentAction = IntentAction.OPEN) -> ExecutionOrderIntent:
    return ExecutionOrderIntent("acct", "BTC/USDT:USDT", PositionSide.LONG, action, Decimal(quantity), "algo-base", OrderType.MARKET)


def test_twap_slices_only_canonical_order_intents_with_unique_keys() -> None:
    plan = plan_execution(_intent(), ExecutionAlgorithmContext(Decimal("0.2"), Decimal(1), Decimal(12), Decimal(3), True))
    assert plan.algorithm is ExecutionAlgorithm.TWAP
    assert sum((item.quantity for item in plan.intents), Decimal(0)) == Decimal(10)
    assert len({item.idempotency_key for item in plan.intents}) == 4


def test_emergency_reduce_outranks_cost_optimization() -> None:
    plan = plan_execution(_intent(action=IntentAction.REDUCE), ExecutionAlgorithmContext(Decimal("0.9"), Decimal(20), Decimal(100), Decimal(1), True, True))
    assert plan.algorithm is ExecutionAlgorithm.EMERGENCY_REDUCE
    assert plan.intents[0].reduces_risk
