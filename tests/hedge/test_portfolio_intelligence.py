from decimal import Decimal

from freqtrade.hedge.research.portfolio_intelligence import HedgeEfficiency, PerpetualIntelligence


def test_hedge_efficiency_is_account_level_and_after_cost() -> None:
    metric = HedgeEfficiency(Decimal(200), Decimal(20), Decimal(3), Decimal(10))
    assert metric.neutralization_ratio == Decimal("0.9")
    assert metric.value_after_cost == Decimal(7)


def test_crowding_score_combines_perpetual_signals_symmetrically() -> None:
    signal = PerpetualIntelligence(Decimal(1), Decimal(-2), Decimal(3), Decimal(-4))
    assert signal.crowding_score == Decimal("2.5")
