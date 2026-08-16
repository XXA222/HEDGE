from __future__ import annotations

from freqtrade.freqai.hedge_rl.risk_walkforward_audit import (
    RiskWalkForwardThresholds,
    aggregate_risk_walk_forward,
)


def _payload(
    start: str,
    end: str,
    fingerprint: str,
    fixed: float,
    perm: float,
    status: str = "PASS",
):
    return {
        "schema": "hedge-risk-level-learning-audit-v4",
        "status": status,
        "metadata": {
            "train_start": "2024-01-01T00:00:00+00:00",
            "train_end": (
                "2024-12-31T23:59:00+00:00"
                if start.startswith("2025-01")
                else (
                    "2025-01-31T23:59:00+00:00"
                    if start.startswith("2025-02")
                    else "2025-02-28T23:59:00+00:00"
                )
            ),
            "oos_start": start,
            "oos_end": end,
        },
        "policy_fingerprint": fingerprint,
        "action_signature": "same-action-contract",
        "observation_signature": "same-observation-contract",
        "reward_signature": "same-reward-contract",
        "feature_count": 64,
        "adaptive": {"steps": 1000},
        "evidence": {
            "adaptive_vs_best_fixed_edge": fixed,
            "adaptive_vs_permutation_quantile_edge": perm,
        },
    }


def test_walkforward_requires_non_overlapping_sequential_oos_and_retrained_models() -> None:
    audits = [
        (
            _payload(
                "2025-01-01T00:00:00+00:00",
                "2025-01-31T23:59:00+00:00",
                "a",
                0.01,
                0.02,
            ),
            "a.json",
        ),
        (
            _payload(
                "2025-02-01T00:00:00+00:00",
                "2025-02-28T23:59:00+00:00",
                "b",
                0.02,
                0.01,
            ),
            "b.json",
        ),
        (
            _payload(
                "2025-03-01T00:00:00+00:00",
                "2025-03-31T23:59:00+00:00",
                "c",
                0.01,
                0.01,
            ),
            "c.json",
        ),
    ]
    report = aggregate_risk_walk_forward(
        audits,
        thresholds=RiskWalkForwardThresholds(min_folds=3),
    )
    assert report.passed
    assert report.metrics["distinct_model_count"] == 3
    assert report.gates["oos_windows_are_non_overlapping"]


def test_walkforward_rejects_overlapping_oos_windows() -> None:
    audits = [
        (
            _payload(
                "2025-01-01T00:00:00+00:00",
                "2025-02-15T00:00:00+00:00",
                "a",
                0.01,
                0.01,
            ),
            "a.json",
        ),
        (
            _payload(
                "2025-02-01T00:00:00+00:00",
                "2025-02-28T00:00:00+00:00",
                "b",
                0.01,
                0.01,
            ),
            "b.json",
        ),
        (
            _payload(
                "2025-03-01T00:00:00+00:00",
                "2025-03-31T00:00:00+00:00",
                "c",
                0.01,
                0.01,
            ),
            "c.json",
        ),
    ]
    report = aggregate_risk_walk_forward(audits)
    assert not report.passed
    assert not report.gates["oos_windows_are_non_overlapping"]


def test_walkforward_rejects_single_reused_model_fingerprint() -> None:
    audits = [
        (
            _payload(
                "2025-01-01T00:00:00+00:00",
                "2025-01-31T00:00:00+00:00",
                "same",
                0.01,
                0.01,
            ),
            "a.json",
        ),
        (
            _payload(
                "2025-02-01T00:00:00+00:00",
                "2025-02-28T00:00:00+00:00",
                "same",
                0.01,
                0.01,
            ),
            "b.json",
        ),
        (
            _payload(
                "2025-03-01T00:00:00+00:00",
                "2025-03-31T00:00:00+00:00",
                "same",
                0.01,
                0.01,
            ),
            "c.json",
        ),
    ]
    report = aggregate_risk_walk_forward(audits)
    assert not report.passed
    assert not report.gates["models_are_retrained_across_folds"]


def test_walkforward_rejects_observation_contract_drift() -> None:
    first = _payload(
        "2025-01-01T00:00:00+00:00",
        "2025-01-31T00:00:00+00:00",
        "a",
        0.01,
        0.01,
    )
    second = _payload(
        "2025-02-01T00:00:00+00:00",
        "2025-02-28T00:00:00+00:00",
        "b",
        0.01,
        0.01,
    )
    third = _payload(
        "2025-03-01T00:00:00+00:00",
        "2025-03-31T00:00:00+00:00",
        "c",
        0.01,
        0.01,
    )
    third["observation_signature"] = "different-observation-contract"
    report = aggregate_risk_walk_forward([(first, "a.json"), (second, "b.json"), (third, "c.json")])
    assert not report.passed
    assert not report.gates["observation_contract_is_constant"]


def test_walkforward_rejects_training_window_touching_oos() -> None:
    import pytest

    payload = _payload(
        "2025-01-01T00:00:00+00:00",
        "2025-01-31T00:00:00+00:00",
        "a",
        0.01,
        0.01,
    )
    payload["metadata"]["train_end"] = "2025-01-01T00:00:00+00:00"
    with pytest.raises(ValueError, match="train_end must strictly precede oos_start"):
        aggregate_risk_walk_forward([(payload, "leak.json")])


def test_walkforward_rejects_nonadvancing_training_cutoff() -> None:
    first = _payload(
        "2025-01-01T00:00:00+00:00",
        "2025-01-31T00:00:00+00:00",
        "a",
        0.01,
        0.01,
    )
    second = _payload(
        "2025-02-01T00:00:00+00:00",
        "2025-02-28T00:00:00+00:00",
        "b",
        0.01,
        0.01,
    )
    third = _payload(
        "2025-03-01T00:00:00+00:00",
        "2025-03-31T00:00:00+00:00",
        "c",
        0.01,
        0.01,
    )
    # Still precedes fold-2 OOS, but does not advance beyond fold-1 training cutoff.
    second["metadata"]["train_end"] = first["metadata"]["train_end"]
    report = aggregate_risk_walk_forward([(first, "a.json"), (second, "b.json"), (third, "c.json")])
    assert not report.passed
    assert not report.gates["training_cutoffs_advance_across_folds"]
