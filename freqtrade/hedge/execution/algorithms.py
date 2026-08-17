"""Execution-policy selection that only materializes canonical ExecutionOrderIntent objects."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING
from enum import StrEnum

from freqtrade.hedge.contracts import ExecutionOrderIntent, OrderType, finite_decimal


class ExecutionAlgorithm(StrEnum):
    IMMEDIATE = "IMMEDIATE"
    PASSIVE_MAKER = "PASSIVE_MAKER"
    TWAP = "TWAP"
    ADAPTIVE_REBALANCE = "ADAPTIVE_REBALANCE"
    EMERGENCY_REDUCE = "EMERGENCY_REDUCE"


@dataclass(frozen=True, slots=True)
class ExecutionAlgorithmContext:
    urgency: Decimal
    spread_bps: Decimal
    estimated_impact_bps: Decimal
    max_slice_quantity: Decimal
    maker_supported: bool
    emergency: bool = False

    def __post_init__(self) -> None:
        for name in ("urgency", "spread_bps", "estimated_impact_bps", "max_slice_quantity"):
            value = finite_decimal(getattr(self, name), field_name=name)
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")
            object.__setattr__(self, name, value)
        if self.urgency > 1 or self.max_slice_quantity <= 0:
            raise ValueError("urgency must be within [0,1] and max_slice_quantity positive")
        if not isinstance(self.maker_supported, bool) or not isinstance(self.emergency, bool):
            raise TypeError("maker_supported and emergency must be bool")


@dataclass(frozen=True, slots=True)
class ExecutionAlgorithmPlan:
    algorithm: ExecutionAlgorithm
    intents: tuple[ExecutionOrderIntent, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.algorithm, ExecutionAlgorithm) or not self.intents:
            raise ValueError("algorithm and nonempty canonical intents are required")
        if not all(isinstance(intent, ExecutionOrderIntent) for intent in self.intents):
            raise TypeError("intents must be ExecutionOrderIntent")


def select_execution_algorithm(intent: ExecutionOrderIntent, context: ExecutionAlgorithmContext) -> ExecutionAlgorithm:
    if not isinstance(intent, ExecutionOrderIntent) or not isinstance(context, ExecutionAlgorithmContext):
        raise TypeError("intent/context use canonical execution contracts")
    if intent.reduces_risk and (context.emergency or context.urgency >= Decimal("0.8")):
        return ExecutionAlgorithm.EMERGENCY_REDUCE
    if intent.reduces_risk:
        return ExecutionAlgorithm.IMMEDIATE
    if intent.quantity > context.max_slice_quantity or context.estimated_impact_bps >= Decimal(10):
        return ExecutionAlgorithm.TWAP
    if context.maker_supported and context.spread_bps >= Decimal(2) and context.urgency < Decimal("0.5"):
        return ExecutionAlgorithm.PASSIVE_MAKER
    return ExecutionAlgorithm.ADAPTIVE_REBALANCE


def plan_execution(intent: ExecutionOrderIntent, context: ExecutionAlgorithmContext) -> ExecutionAlgorithmPlan:
    algorithm = select_execution_algorithm(intent, context)
    if algorithm is not ExecutionAlgorithm.TWAP:
        return ExecutionAlgorithmPlan(algorithm, (_annotate(intent, algorithm, 1, 1),))
    count = int((intent.quantity / context.max_slice_quantity).to_integral_value(rounding=ROUND_CEILING))
    base, remainder = divmod(intent.quantity, Decimal(count))
    intents = tuple(
        _annotate(
            intent,
            algorithm,
            index + 1,
            count,
            quantity=base + (remainder if index == count - 1 else Decimal(0)),
        )
        for index in range(count)
    )
    return ExecutionAlgorithmPlan(algorithm, intents)


def _annotate(intent: ExecutionOrderIntent, algorithm: ExecutionAlgorithm, index: int, count: int, *, quantity: Decimal | None = None) -> ExecutionOrderIntent:
    suffix = f":{algorithm.value.lower()}:{index}"
    if len(intent.idempotency_key) + len(suffix) > 256:
        raise ValueError("idempotency_key cannot represent execution slice")
    metadata = dict(intent.metadata)
    metadata.update({"execution_algorithm": algorithm.value, "execution_slice": index, "execution_slice_count": count})
    return ExecutionOrderIntent(
        account_id=intent.account_id, symbol=intent.symbol, position_side=intent.position_side,
        action=intent.action, quantity=intent.quantity if quantity is None else quantity,
        idempotency_key=intent.idempotency_key + suffix, order_type=intent.order_type,
        limit_price=intent.limit_price, reduce_only=intent.reduce_only,
        action_group_id=intent.action_group_id, metadata=metadata,
    )
