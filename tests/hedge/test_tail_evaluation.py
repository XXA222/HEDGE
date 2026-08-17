from decimal import Decimal

from freqtrade.hedge.research.tail_evaluation import (
    TailScenario,
    evaluate_tail,
    qualify_tail_scenarios,
)


def test_tail_metrics_are_loss_positive_and_distributional() -> None:
    metrics = evaluate_tail(
        tuple(Decimal(value) for value in ("-0.05", "-0.02", "0.01", "0.02", "0.03")),
        quantile=Decimal("0.2"),
        tail_loss_threshold=Decimal("-0.02"),
    )
    assert metrics.downside_quantile == Decimal("-0.026")
    assert metrics.value_at_risk == Decimal("0.026")
    assert metrics.conditional_value_at_risk == Decimal("0.05")
    assert metrics.tail_loss_probability == Decimal("0.4")


def test_tail_qualification_checks_every_scenario() -> None:
    metrics = evaluate_tail((Decimal("-0.01"), Decimal("0.02"), Decimal("0.03")))
    passed, reasons = qualify_tail_scenarios(
        (TailScenario("base", Decimal("0.02"), Decimal("0.01"), metrics),),
        minimum_expected_edge=Decimal("0.01"),
        maximum_cvar=Decimal("0.02"),
        maximum_tail_loss_probability=Decimal("0.1"),
        maximum_uncertainty=Decimal("0.02"),
    )
    assert passed
    assert reasons == ()

    passed, reasons = qualify_tail_scenarios(
        (TailScenario("stress", Decimal("-0.01"), Decimal("0.2"), metrics),),
        minimum_expected_edge=Decimal(0),
        maximum_cvar=Decimal("0.001"),
        maximum_tail_loss_probability=Decimal("0.1"),
        maximum_uncertainty=Decimal("0.1"),
    )
    assert not passed
    assert reasons == (
        "stress:EDGE_BELOW_MINIMUM",
        "stress:CVAR_ABOVE_MAXIMUM",
        "stress:UNCERTAINTY_ABOVE_MAXIMUM",
    )


def test_tail_qualification_requires_scenarios() -> None:
    assert qualify_tail_scenarios(
        (),
        minimum_expected_edge=Decimal(0),
        maximum_cvar=Decimal(1),
        maximum_tail_loss_probability=Decimal(1),
        maximum_uncertainty=Decimal(1),
    ) == (False, ("TAIL_SCENARIOS_MISSING",))
