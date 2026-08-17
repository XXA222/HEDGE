"""Final hard-gate composition for a promotable HEDGE release."""

from dataclasses import dataclass
from decimal import Decimal

from freqtrade.hedge.contracts import finite_decimal


@dataclass(frozen=True, slots=True)
class ReleaseHardGates:
    source: bool
    data: bool
    semantics: bool
    risk: bool
    exchange: bool
    simulation: bool
    rl: bool
    production: bool
    recovery: bool
    performance: bool

    def __post_init__(self) -> None:
        if not all(isinstance(getattr(self, name), bool) for name in self.__dataclass_fields__):
            raise TypeError("all release gates must be bool")

    @property
    def passed(self) -> bool:
        return all(getattr(self, name) for name in self.__dataclass_fields__)

    @property
    def failed_gates(self) -> tuple[str, ...]:
        return tuple(name.upper() for name in self.__dataclass_fields__ if not getattr(self, name))


@dataclass(frozen=True, slots=True)
class QualificationScorecard:
    profitability: Decimal
    tail_risk: Decimal
    costs: Decimal
    hedge_behavior: Decimal
    position_management: Decimal
    safety: Decimal
    operational: Decimal

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = finite_decimal(getattr(self, name), field_name=name)
            if not Decimal(0) <= value <= Decimal(1):
                raise ValueError("scorecard values must be within [0,1]")
            object.__setattr__(self, name, value)

    @property
    def minimum_score(self) -> Decimal:
        return min(getattr(self, name) for name in self.__dataclass_fields__)


@dataclass(frozen=True, slots=True)
class ReleaseQualificationDecision:
    qualified: bool
    reasons: tuple[str, ...]


def qualify_release(gates: ReleaseHardGates, scorecard: QualificationScorecard, *, minimum_dimension_score: Decimal = Decimal("0.70")) -> ReleaseQualificationDecision:
    threshold = finite_decimal(minimum_dimension_score, field_name="minimum_dimension_score")
    if not Decimal(0) <= threshold <= Decimal(1):
        raise ValueError("minimum dimension score must be within [0,1]")
    reasons = list(gates.failed_gates)
    if scorecard.minimum_score < threshold:
        reasons.append("SCORECARD_DIMENSION_BELOW_THRESHOLD")
    return ReleaseQualificationDecision(not reasons, tuple(reasons))
