from decimal import Decimal

from freqtrade.hedge.research.ope import OPEEstimate, OPEMethod, qualify_ope


def _estimate(method: OPEMethod) -> OPEEstimate:
    return OPEEstimate(method, Decimal("0.1"), Decimal("0.02"), Decimal(100), Decimal(3), True)


def test_ope_requires_weighted_is_and_doubly_robust_to_pass() -> None:
    result = qualify_ope((_estimate(OPEMethod.WEIGHTED_IMPORTANCE_SAMPLING), _estimate(OPEMethod.DOUBLY_ROBUST)), minimum_lcb=Decimal(0), minimum_ess=Decimal(50), maximum_weight=Decimal(5))
    assert result.passed


def test_ope_fails_closed_on_low_ess() -> None:
    weak = OPEEstimate(OPEMethod.DOUBLY_ROBUST, Decimal(1), Decimal(1), Decimal(1), Decimal(2), True)
    result = qualify_ope((_estimate(OPEMethod.WEIGHTED_IMPORTANCE_SAMPLING), weak), minimum_lcb=Decimal(0), minimum_ess=Decimal(50), maximum_weight=Decimal(5))
    assert not result.passed
