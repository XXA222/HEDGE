"""Fail-closed off-policy evaluation qualification contracts."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum

from freqtrade.hedge.contracts import finite_decimal


class OPEMethod(StrEnum):
    BEHAVIOR_BASELINE = "BEHAVIOR_BASELINE"
    IMPORTANCE_SAMPLING = "IMPORTANCE_SAMPLING"
    WEIGHTED_IMPORTANCE_SAMPLING = "WEIGHTED_IMPORTANCE_SAMPLING"
    DOUBLY_ROBUST = "DOUBLY_ROBUST"


@dataclass(frozen=True, slots=True)
class OPEEstimate:
    method: OPEMethod
    expected_return: Decimal
    lower_confidence_bound: Decimal
    effective_sample_size: Decimal
    max_importance_weight: Decimal
    finite: bool

    def __post_init__(self) -> None:
        if not isinstance(self.method, OPEMethod) or not isinstance(self.finite, bool):
            raise TypeError("method/finite have invalid types")
        for name in ("expected_return", "lower_confidence_bound", "effective_sample_size", "max_importance_weight"):
            value = finite_decimal(getattr(self, name), field_name=name)
            object.__setattr__(self, name, value)
        if self.effective_sample_size < 0 or self.max_importance_weight < 0:
            raise ValueError("sample size and importance weight must be nonnegative")


@dataclass(frozen=True, slots=True)
class OPEQualification:
    passed: bool
    reasons: tuple[str, ...]


def qualify_ope(estimates: tuple[OPEEstimate, ...], *, minimum_lcb: Decimal, minimum_ess: Decimal, maximum_weight: Decimal) -> OPEQualification:
    minimum_lcb = finite_decimal(minimum_lcb, field_name="minimum_lcb")
    minimum_ess = finite_decimal(minimum_ess, field_name="minimum_ess")
    maximum_weight = finite_decimal(maximum_weight, field_name="maximum_weight")
    if minimum_ess <= 0 or maximum_weight <= 0:
        raise ValueError("OPE thresholds must be positive")
    reasons: list[str] = []
    methods = {estimate.method for estimate in estimates}
    required = {OPEMethod.WEIGHTED_IMPORTANCE_SAMPLING, OPEMethod.DOUBLY_ROBUST}
    if not required.issubset(methods):
        reasons.append("REQUIRED_OPE_METHOD_MISSING")
    for estimate in estimates:
        if not estimate.finite:
            reasons.append(f"NONFINITE:{estimate.method.value}")
        if estimate.lower_confidence_bound < minimum_lcb:
            reasons.append(f"LCB_BELOW_THRESHOLD:{estimate.method.value}")
        if estimate.effective_sample_size < minimum_ess:
            reasons.append(f"ESS_BELOW_THRESHOLD:{estimate.method.value}")
        if estimate.max_importance_weight > maximum_weight:
            reasons.append(f"IMPORTANCE_WEIGHT_EXCEEDED:{estimate.method.value}")
    return OPEQualification(not reasons, tuple(reasons))
