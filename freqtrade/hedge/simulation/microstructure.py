"""Tier-A deterministic spread/latency/queue microstructure fill model."""

from dataclasses import dataclass
from decimal import Decimal

from freqtrade.hedge.contracts import finite_decimal


@dataclass(frozen=True, slots=True)
class MicrostructureState:
    bid: Decimal
    ask: Decimal
    available_quantity: Decimal
    queue_ahead_quantity: Decimal
    latency_ms: Decimal

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = finite_decimal(getattr(self, name), field_name=name)
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")
            object.__setattr__(self, name, value)
        if self.bid <= 0 or self.ask < self.bid:
            raise ValueError("book prices are invalid")


@dataclass(frozen=True, slots=True)
class MicrostructureFill:
    filled_quantity: Decimal
    fill_price: Decimal | None
    latency_penalty: Decimal


def simulate_taker_fill(state: MicrostructureState, *, buy: bool, quantity: Decimal, latency_bps_per_second: Decimal = Decimal(1)) -> MicrostructureFill:
    quantity = finite_decimal(quantity, field_name="quantity")
    penalty_rate = finite_decimal(latency_bps_per_second, field_name="latency_bps_per_second")
    if quantity <= 0 or penalty_rate < 0:
        raise ValueError("quantity positive and penalty nonnegative are required")
    filled = min(quantity, state.available_quantity)
    if not filled:
        return MicrostructureFill(Decimal(0), None, Decimal(0))
    price = state.ask if buy else state.bid
    penalty = price * penalty_rate * state.latency_ms / Decimal(10_000_000)
    return MicrostructureFill(filled, price + penalty if buy else price - penalty, penalty * filled)
