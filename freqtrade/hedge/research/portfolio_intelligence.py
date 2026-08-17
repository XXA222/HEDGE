"""Account-level hedge efficiency and perpetual crowding metrics."""

from dataclasses import dataclass
from decimal import Decimal

from freqtrade.hedge.contracts import finite_decimal


def _d(value: object, name: str) -> Decimal:
    return finite_decimal(value, field_name=name)


@dataclass(frozen=True, slots=True)
class PerpetualIntelligence:
    funding_zscore: Decimal
    basis_zscore: Decimal
    open_interest_zscore: Decimal
    liquidation_zscore: Decimal

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, _d(getattr(self, name), name))

    @property
    def crowding_score(self) -> Decimal:
        values = tuple(abs(getattr(self, name)) for name in self.__dataclass_fields__)
        return sum(values, Decimal(0)) / Decimal(len(values))


@dataclass(frozen=True, slots=True)
class HedgeEfficiency:
    gross_notional: Decimal
    net_notional: Decimal
    hedge_cost: Decimal
    risk_reduction_value: Decimal

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = _d(getattr(self, name), name)
            if name != "net_notional" and value < 0:
                raise ValueError(f"{name} must be nonnegative")
            object.__setattr__(self, name, value)
        if abs(self.net_notional) > self.gross_notional:
            raise ValueError("absolute net notional cannot exceed gross notional")

    @property
    def neutralization_ratio(self) -> Decimal:
        return Decimal(1) if self.gross_notional == 0 else Decimal(1) - abs(self.net_notional) / self.gross_notional

    @property
    def value_after_cost(self) -> Decimal:
        return self.risk_reduction_value - self.hedge_cost
