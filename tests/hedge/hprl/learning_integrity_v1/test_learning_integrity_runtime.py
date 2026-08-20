from __future__ import annotations

from pathlib import Path

import pytest

from freqtrade.hedge.hprl.qualification import (
    QualificationStatus,
    qualify_candidate,
    search_is_degenerate,
    select_winner,
)
from freqtrade.hedge.telemetry.training_health import CollapseThresholds, RollingCollapseDetector


def metrics(**overrides):
    values = {
        "net_return": 0.02,
        "sharpe": 1.0,
        "sortino": 1.0,
        "calmar": 1.0,
        "max_drawdown": 0.01,
        "cvar": 0.001,
        "turnover": 1.0,
        "fees": 1.0,
        "funding": 0.0,
        "liquidations": 0,
    }
    values.update(overrides)
    return values


def healthy():
    return {
        "training_health_ready": 1.0,
        "training_health_collapsed": 0.0,
        "policy_collapse": 0.0,
        "gradient_collapse": 0.0,
    }


def test_flat_policy_is_not_qualified():
    decision = qualify_candidate(metrics=metrics(net_return=0.0, turnover=0.0), health=healthy(), actions=[[0.0, 0.0]] * 3349)
    assert decision.status == QualificationStatus.REJECTED_INACTIVE


def test_pure_policy_collapse_is_fatal_before_gradient_death():
    detector = RollingCollapseDetector(CollapseThresholds(window=3, patience=2))
    sample = {
        "actor_grad_norm": 1e-3,
        "critic_grad_norm": 1.0,
        "actor_update_ratio": 1e-4,
        "critic_update_ratio": 1e-4,
        "policy_entropy": 0.0,
        "action_saturation": 1.0,
        "policy_action_mean": 0.0,
        "policy_action_std": 0.0,
        "flat_saturation": 1.0,
        "heavy_saturation": 0.0,
        "advantage_std": 0.1,
    }
    result = {}
    for _ in range(4):
        result = detector.update(sample)
    assert result["policy_flat_collapse"] == 1.0
    assert result["training_health_collapsed"] == 1.0
    assert result["gradient_collapse"] == 0.0


def test_ties_and_zero_signal_search_fail_closed():
    winner, status = select_winner([
        {"status": "PASS", "objective": 0.1},
        {"status": "PASS", "objective": 0.1},
    ])
    assert winner is None
    assert status == QualificationStatus.NO_DISTINGUISHABLE_WINNER
    assert search_is_degenerate([{"status": "PASS", "objective": 0.0}] * 8)


def test_nonfinite_and_liquidation_are_hard_rejections():
    bad = metrics(net_return=float("nan"))
    decision = qualify_candidate(metrics=bad, health=healthy(), actions=[[0.25, 0.0], [0.5, 0.0]] * 16)
    assert decision.status == QualificationStatus.REJECTED_NUMERICAL
    decision = qualify_candidate(metrics=metrics(liquidations=1), health=healthy(), actions=[[0.25, 0.0], [0.5, 0.0]] * 16)
    assert decision.status == QualificationStatus.REJECTED_RISK
