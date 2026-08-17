from decimal import Decimal

from freqtrade.hedge.research.position_management import (
    AblationAxis,
    AblationEvidence,
    PositionDecision,
    position_management_metrics,
    qualify_position_management,
)


def _ablation(axis: AblationAxis, *, preserved: bool = True) -> AblationEvidence:
    return AblationEvidence(
        axis=axis,
        variant=f"{axis.value}-variant",
        net_return_delta=Decimal("-0.01"),
        max_drawdown_delta=Decimal("0.01"),
        projection_rate_delta=Decimal(0),
        behavior_preserved=preserved,
    )


def test_position_management_behavior_metrics() -> None:
    rows = (
        PositionDecision(0, 0, 0, 0),
        PositionDecision(1, 0, 1, 0),
        PositionDecision(2, 0, 2, 0, realized_pnl=Decimal(2), risk_event=True),
        PositionDecision(1, 0, 1, 0),
        PositionDecision(0, 1, 0, 0, risk_rejected=True),
    )
    metrics = position_management_metrics(rows)
    assert metrics.observation_count == 5
    assert metrics.scale_up_frequency == Decimal("0.5")
    assert metrics.scale_down_frequency == Decimal("0.5")
    assert metrics.profit_lock_rate == Decimal(1)
    assert metrics.mean_de_risk_latency == Decimal(1)
    assert metrics.projection_rate == Decimal("0.2")
    assert metrics.risk_reject_rate == Decimal("0.2")


def test_position_management_gate_requires_all_ablations() -> None:
    metrics = position_management_metrics(
        (
            PositionDecision(0, 0, 0, 0),
            PositionDecision(1, 0, 1, 0),
            PositionDecision(0, 0, 0, 0),
        )
    )
    passed, reasons = qualify_position_management(
        metrics,
        tuple(_ablation(axis) for axis in AblationAxis),
        maximum_projection_rate=Decimal("0.1"),
        maximum_churn_rate=Decimal("0.1"),
        maximum_de_risk_latency=Decimal(1),
    )
    assert passed
    assert reasons == ()

    passed, reasons = qualify_position_management(
        metrics,
        (_ablation(AblationAxis.LEVEL_MAPPING),),
        maximum_projection_rate=Decimal("0.1"),
        maximum_churn_rate=Decimal("0.1"),
        maximum_de_risk_latency=Decimal(1),
    )
    assert not passed
    assert "ABLATION_MISSING:PROJECTOR" in reasons
    assert "ABLATION_MISSING:REWARD" in reasons


def test_projector_dependence_fails_closed() -> None:
    metrics = position_management_metrics(
        (
            PositionDecision(4, 4, 0, 0),
            PositionDecision(4, 4, 1, 0),
        )
    )
    passed, reasons = qualify_position_management(
        metrics,
        tuple(_ablation(axis) for axis in AblationAxis),
        maximum_projection_rate=Decimal("0.25"),
        maximum_churn_rate=Decimal(1),
        maximum_de_risk_latency=Decimal(1),
    )
    assert not passed
    assert "PROJECTOR_DEPENDENCE_TOO_HIGH" in reasons
