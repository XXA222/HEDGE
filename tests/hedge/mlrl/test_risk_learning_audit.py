from __future__ import annotations

from freqtrade.freqai.hedge_rl.risk_learning_audit import (
    RiskLearningAuditThresholds,
    action_diagnostics,
    active_change_signature,
    fixed_level_transform,
    level_permutation_transform,
    sizing_score,
)
from freqtrade.freqai.hedge_rl.risk_levels import RiskLevelProfile


def test_level_permutation_preserves_active_and_change_timing() -> None:
    actions = [
        (0, 0),
        (1, 0),
        (2, 0),
        (2, 3),
        (0, 3),
        (4, 1),
        (4, 1),
        (0, 0),
    ]
    transformed = level_permutation_transform((4, 1, 3, 2))(actions)
    assert active_change_signature(transformed) == active_change_signature(actions)
    assert transformed != actions


def test_fixed_level_preserves_only_active_flat_path() -> None:
    actions = [(0, 0), (1, 0), (3, 2), (0, 2), (4, 0)]
    transformed = fixed_level_transform(2)(actions)
    assert transformed == [(0, 0), (2, 0), (2, 2), (0, 2), (2, 0)]


def test_action_collapse_gate_is_not_biased_by_sparse_flat_time() -> None:
    profile = RiskLevelProfile()
    actions = [(0, 0)] * 95 + [(1, 0), (2, 0), (3, 0), (4, 0), (2, 0)]
    diagnostics = action_diagnostics(actions, profile=profile)
    assert diagnostics.max_joint_action_share == 0.95
    assert diagnostics.max_active_joint_action_share == 0.40
    assert diagnostics.distinct_nonzero_levels == 4
    assert diagnostics.normalized_nonzero_level_entropy > 0.80


def test_magnitude_change_fraction_tracks_nonzero_choices_across_flat_gaps() -> None:
    profile = RiskLevelProfile()
    actions = [(1, 0), (0, 0), (3, 0), (0, 0), (2, 0), (0, 0), (2, 0)]
    diagnostics = action_diagnostics(actions, profile=profile)
    assert diagnostics.magnitude_transition_opportunities == 3
    assert diagnostics.magnitude_change_steps == 2
    assert diagnostics.magnitude_change_fraction == 2 / 3


def test_sizing_score_is_net_equity_and_drawdown_based() -> None:
    better = sizing_score(
        initial_equity=1000.0,
        final_equity=1100.0,
        max_drawdown=0.05,
        drawdown_weight=1.0,
    )
    worse = sizing_score(
        initial_equity=1000.0,
        final_equity=1100.0,
        max_drawdown=0.20,
        drawdown_weight=1.0,
    )
    assert better > worse


def test_thresholds_reject_invalid_permutation_budget() -> None:
    try:
        RiskLearningAuditThresholds(permutation_trials=24)
    except ValueError as exc:
        assert "permutation_trials" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("invalid permutation budget was accepted")


def test_post_fit_learning_audit_is_opt_in_and_non_blocking_by_default() -> None:
    from freqtrade.freqai.hedge_rl.risk_learning_audit import audit_enabled, audit_required

    assert not audit_enabled({})
    assert not audit_required({})


def test_default_attribution_uses_all_nonidentity_level_permutations() -> None:
    assert RiskLearningAuditThresholds().permutation_trials == 23


def test_standalone_audit_training_boundary_must_precede_oos() -> None:
    from freqtrade.commands.hedge_risk_learning_commands import _training_boundary_metadata
    from freqtrade.exceptions import OperationalException

    args = {
        "hedge_risk_audit_train_start": "2024-01-01T00:00:00+00:00",
        "hedge_risk_audit_train_end": "2025-01-01T00:00:00+00:00",
    }
    try:
        _training_boundary_metadata(
            args,
            oos_start="2025-01-01T00:00:00+00:00",
        )
    except OperationalException as exc:
        assert "train_end < oos_start" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("training/OOS leakage boundary was accepted")
