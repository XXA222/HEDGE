"""Qualification contracts for HPRL research candidates.

Economic reward, training health and model qualification deliberately remain separate.
A candidate is never promoted merely because it avoided losses by staying flat.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from statistics import fmean, median
from typing import Any, Iterable, Mapping, Sequence


class QualificationStatus(StrEnum):
    PASS = "PASS"
    REJECTED_NUMERICAL = "REJECTED_NUMERICAL"
    REJECTED_HEALTH = "REJECTED_HEALTH"
    REJECTED_INACTIVE = "REJECTED_INACTIVE"
    REJECTED_RISK = "REJECTED_RISK"
    REJECTED_PERFORMANCE = "REJECTED_PERFORMANCE"
    REJECTED_ROBUSTNESS = "REJECTED_ROBUSTNESS"
    NO_QUALIFIED_ALGORITHM = "NO_QUALIFIED_ALGORITHM"
    NO_QUALIFIED_TRIAL = "NO_QUALIFIED_TRIAL"
    NO_DISTINGUISHABLE_WINNER = "NO_DISTINGUISHABLE_WINNER"
    SEARCH_DEGENERATE = "SEARCH_DEGENERATE"


@dataclass(frozen=True, slots=True)
class QualificationThresholds:
    """Research qualification thresholds, intentionally configurable at the workflow boundary."""

    min_non_flat_decisions: int = 8
    min_non_flat_fraction: float = 0.002
    min_unique_joint_states: int = 2
    max_flat_fraction: float = 0.998
    max_drawdown: float = 0.35
    max_cvar: float = 0.05
    max_liquidations: int = 0
    flat_objective_improvement: float = 1e-6
    objective_tie_tolerance: float = 1e-9
    min_seed_pass_fraction: float = 2.0 / 3.0
    max_seed_collapse_fraction: float = 1.0 / 3.0

    def __post_init__(self) -> None:
        integer_values = (
            self.min_non_flat_decisions,
            self.min_unique_joint_states,
            self.max_liquidations,
        )
        if any(not isinstance(value, int) or isinstance(value, bool) for value in integer_values):
            raise TypeError("integer qualification thresholds must be integers")
        if self.min_non_flat_decisions < 0 or self.min_unique_joint_states < 1:
            raise ValueError("activity thresholds are invalid")
        if self.max_liquidations < 0:
            raise ValueError("max_liquidations cannot be negative")
        scalar_values = (
            self.min_non_flat_fraction,
            self.max_flat_fraction,
            self.max_drawdown,
            self.max_cvar,
            self.flat_objective_improvement,
            self.objective_tie_tolerance,
            self.min_seed_pass_fraction,
            self.max_seed_collapse_fraction,
        )
        if any(not math.isfinite(float(value)) for value in scalar_values):
            raise ValueError("qualification thresholds must be finite")
        if not 0.0 <= self.min_non_flat_fraction <= 1.0:
            raise ValueError("min_non_flat_fraction must be within [0, 1]")
        if not 0.0 <= self.max_flat_fraction <= 1.0:
            raise ValueError("max_flat_fraction must be within [0, 1]")
        if self.max_drawdown < 0.0 or self.max_cvar < 0.0:
            raise ValueError("risk thresholds cannot be negative")
        if self.flat_objective_improvement < 0.0 or self.objective_tie_tolerance < 0.0:
            raise ValueError("objective thresholds cannot be negative")
        if not 0.0 <= self.min_seed_pass_fraction <= 1.0:
            raise ValueError("min_seed_pass_fraction must be within [0, 1]")
        if not 0.0 <= self.max_seed_collapse_fraction <= 1.0:
            raise ValueError("max_seed_collapse_fraction must be within [0, 1]")


@dataclass(frozen=True, slots=True)
class PolicyActivityDiagnostics:
    decisions: int
    flat_decisions: int
    non_flat_decisions: int
    flat_fraction: float
    non_flat_fraction: float
    unique_joint_states: int
    joint_state_entropy: float
    long_nonzero_fraction: float
    short_nonzero_fraction: float
    mean_long_policy: float
    mean_short_policy: float
    mean_gross_policy: float
    mean_abs_net_policy: float
    transition_count: int
    transition_fraction: float
    joint_state_counts: Mapping[str, int] = field(default_factory=dict)
    long_level_counts: Mapping[str, int] = field(default_factory=dict)
    short_level_counts: Mapping[str, int] = field(default_factory=dict)

    @property
    def is_completely_flat(self) -> bool:
        return self.decisions > 0 and self.flat_decisions == self.decisions


def _finite_number(value: object) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("policy actions must be finite")
    return number


def _level_index(value: float, level_count: int) -> int:
    if level_count < 2:
        raise ValueError("level_count must be >= 2")
    clipped = min(1.0, max(0.0, float(value)))
    return min(level_count - 1, max(0, int(math.floor(clipped * (level_count - 1) + 0.5))))


def policy_activity(
    actions: Iterable[Sequence[object]],
    *,
    level_count: int = 5,
) -> PolicyActivityDiagnostics:
    """Summarize deterministic executed policy behavior without rewarding activity itself."""

    rows: list[tuple[float, float, int, int]] = []
    for action in actions:
        if len(action) != 2:
            raise ValueError("dual-leg policy action must contain exactly LONG and SHORT values")
        long_value = _finite_number(action[0])
        short_value = _finite_number(action[1])
        long_index = _level_index(long_value, level_count)
        short_index = _level_index(short_value, level_count)
        rows.append((long_value, short_value, long_index, short_index))
    if not rows:
        return PolicyActivityDiagnostics(
            decisions=0,
            flat_decisions=0,
            non_flat_decisions=0,
            flat_fraction=1.0,
            non_flat_fraction=0.0,
            unique_joint_states=0,
            joint_state_entropy=0.0,
            long_nonzero_fraction=0.0,
            short_nonzero_fraction=0.0,
            mean_long_policy=0.0,
            mean_short_policy=0.0,
            mean_gross_policy=0.0,
            mean_abs_net_policy=0.0,
            transition_count=0,
            transition_fraction=0.0,
            joint_state_counts={},
            long_level_counts={},
            short_level_counts={},
        )
    joint = [(row[2], row[3]) for row in rows]
    counts = Counter(joint)
    long_counts = Counter(row[2] for row in rows)
    short_counts = Counter(row[3] for row in rows)
    total = len(rows)
    flat = counts.get((0, 0), 0)
    probabilities = [count / total for count in counts.values()]
    entropy = -sum(value * math.log(value) for value in probabilities if value > 0.0)
    maximum_entropy = math.log(max(1, min(level_count * level_count, total)))
    normalized_entropy = entropy / maximum_entropy if maximum_entropy > 0.0 else 0.0
    transitions = sum(1 for previous, current in zip(joint, joint[1:]) if previous != current)
    long_values = [row[0] for row in rows]
    short_values = [row[1] for row in rows]
    return PolicyActivityDiagnostics(
        decisions=total,
        flat_decisions=flat,
        non_flat_decisions=total - flat,
        flat_fraction=flat / total,
        non_flat_fraction=(total - flat) / total,
        unique_joint_states=len(counts),
        joint_state_entropy=normalized_entropy,
        long_nonzero_fraction=sum(row[2] > 0 for row in rows) / total,
        short_nonzero_fraction=sum(row[3] > 0 for row in rows) / total,
        mean_long_policy=fmean(long_values),
        mean_short_policy=fmean(short_values),
        mean_gross_policy=fmean(long_value + short_value for long_value, short_value, _, _ in rows),
        mean_abs_net_policy=fmean(abs(long_value - short_value) for long_value, short_value, _, _ in rows),
        transition_count=transitions,
        transition_fraction=transitions / max(1, total - 1),
        joint_state_counts={f"{key[0]}/{key[1]}": value for key, value in sorted(counts.items())},
        long_level_counts={str(key): value for key, value in sorted(long_counts.items())},
        short_level_counts={str(key): value for key, value in sorted(short_counts.items())},
    )


def actions_from_evaluation_rows(rows: Iterable[Mapping[str, Any]]) -> list[Sequence[object]]:
    result: list[Sequence[object]] = []
    for row in rows:
        action = row.get("policy_action")
        if not isinstance(action, Sequence) or isinstance(action, (str, bytes)):
            raise ValueError("evaluation row is missing a policy_action sequence")
        result.append(action)
    return result


@dataclass(frozen=True, slots=True)
class QualificationDecision:
    accepted: bool
    status: QualificationStatus
    reasons: tuple[str, ...]
    objective: float
    health_ok: bool
    numerical_ok: bool
    activity_ok: bool
    risk_ok: bool
    performance_ok: bool
    activity: PolicyActivityDiagnostics

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        return payload


def _health_reasons(health: Mapping[str, object]) -> list[str]:
    reasons: list[str] = []
    checks = {
        "training_health_collapsed": "training_health_collapse",
        "policy_collapse": "policy_degeneracy",
        "policy_no_diversity": "policy_no_diversity",
        "policy_flat_collapse": "policy_flat_collapse",
        "policy_heavy_collapse": "policy_heavy_collapse",
        "gradient_collapse": "gradient_collapse",
        "policy_gradient_collapse": "policy_gradient_collapse",
        "value_gradient_collapse": "value_gradient_collapse",
        "update_collapse": "update_collapse",
        "advantage_collapse": "advantage_collapse",
    }
    ready = float(health.get("training_health_ready", 0.0) or 0.0) >= 1.0
    if not ready:
        return reasons
    for name, reason in checks.items():
        try:
            active = float(health.get(name, 0.0) or 0.0) >= 1.0
        except (TypeError, ValueError):
            active = True
        if active:
            reasons.append(reason)
    return reasons


def _metrics_are_finite(metrics: Mapping[str, object]) -> bool:
    required = (
        "net_return",
        "sharpe",
        "sortino",
        "calmar",
        "max_drawdown",
        "cvar",
        "turnover",
        "fees",
        "funding",
        "liquidations",
    )
    try:
        return all(math.isfinite(float(metrics[name])) for name in required)
    except (KeyError, TypeError, ValueError):
        return False


def trading_objective(metrics: Mapping[str, object]) -> float:
    """Economic objective used only after qualification eligibility has been established."""
    liquidations = float(metrics["liquidations"])
    if liquidations > 0:
        return -1000.0 - liquidations
    return (
        float(metrics["net_return"])
        - 1.5 * float(metrics["max_drawdown"])
        - 0.002 * float(metrics["turnover"])
    )


def qualify_candidate(
    *,
    metrics: Mapping[str, object],
    health: Mapping[str, object],
    actions: Iterable[Sequence[object]],
    thresholds: QualificationThresholds | None = None,
    flat_objective: float = 0.0,
    level_count: int = 5,
) -> QualificationDecision:
    """Apply fail-closed numerical -> health -> activity -> risk -> performance gates."""

    limits = thresholds or QualificationThresholds()
    activity = policy_activity(actions, level_count=level_count)
    finite = _metrics_are_finite(metrics)
    if not finite:
        return QualificationDecision(
            False,
            QualificationStatus.REJECTED_NUMERICAL,
            ("nonfinite_or_missing_evaluation_metrics",),
            float("-inf"),
            False,
            False,
            False,
            False,
            False,
            activity,
        )
    objective = trading_objective(metrics)
    health_reasons = _health_reasons(health)
    health_ok = not health_reasons
    activity_reasons: list[str] = []
    if activity.decisions == 0:
        activity_reasons.append("no_evaluation_decisions")
    if activity.non_flat_decisions < limits.min_non_flat_decisions:
        activity_reasons.append("insufficient_non_flat_decisions")
    if activity.non_flat_fraction < limits.min_non_flat_fraction:
        activity_reasons.append("insufficient_non_flat_fraction")
    if activity.unique_joint_states < limits.min_unique_joint_states:
        activity_reasons.append("insufficient_joint_action_diversity")
    if activity.flat_fraction > limits.max_flat_fraction:
        activity_reasons.append("flat_policy_dominates")
    activity_ok = not activity_reasons
    risk_reasons: list[str] = []
    if int(float(metrics["liquidations"])) > limits.max_liquidations:
        risk_reasons.append("liquidation_limit_exceeded")
    if float(metrics["max_drawdown"]) > limits.max_drawdown:
        risk_reasons.append("drawdown_limit_exceeded")
    if float(metrics["cvar"]) > limits.max_cvar:
        risk_reasons.append("cvar_limit_exceeded")
    risk_ok = not risk_reasons
    performance_reasons: list[str] = []
    required_objective = float(flat_objective) + limits.flat_objective_improvement
    if not objective > required_objective:
        performance_reasons.append("does_not_beat_flat_baseline")
    performance_ok = not performance_reasons

    if not health_ok:
        status = QualificationStatus.REJECTED_HEALTH
        reasons = tuple(health_reasons)
    elif not activity_ok:
        status = QualificationStatus.REJECTED_INACTIVE
        reasons = tuple(activity_reasons)
    elif not risk_ok:
        status = QualificationStatus.REJECTED_RISK
        reasons = tuple(risk_reasons)
    elif not performance_ok:
        status = QualificationStatus.REJECTED_PERFORMANCE
        reasons = tuple(performance_reasons)
    else:
        status = QualificationStatus.PASS
        reasons = ()
    return QualificationDecision(
        accepted=status is QualificationStatus.PASS,
        status=status,
        reasons=reasons,
        objective=objective,
        health_ok=health_ok,
        numerical_ok=True,
        activity_ok=activity_ok,
        risk_ok=risk_ok,
        performance_ok=performance_ok,
        activity=activity,
    )


def select_winner(
    rows: Iterable[Mapping[str, Any]],
    *,
    objective_key: str = "objective",
    status_key: str = "status",
    tie_tolerance: float = 1e-9,
) -> tuple[Mapping[str, Any] | None, QualificationStatus]:
    """Select only from PASS rows and fail closed on indistinguishable best candidates."""

    eligible = [row for row in rows if str(row.get(status_key, "")) == QualificationStatus.PASS.value]
    if not eligible:
        return None, QualificationStatus.NO_QUALIFIED_ALGORITHM
    scores = [float(row[objective_key]) for row in eligible]
    if any(not math.isfinite(value) for value in scores):
        raise ValueError("eligible candidate objective must be finite")
    best_score = max(scores)
    best_rows = [row for row, score in zip(eligible, scores, strict=True) if abs(score - best_score) <= tie_tolerance]
    if len(best_rows) != 1:
        return None, QualificationStatus.NO_DISTINGUISHABLE_WINNER
    return best_rows[0], QualificationStatus.PASS


def search_is_degenerate(
    rows: Iterable[Mapping[str, Any]],
    *,
    objective_key: str = "objective",
    tolerance: float = 1e-12,
) -> bool:
    eligible = [row for row in rows if str(row.get("status", "")) == QualificationStatus.PASS.value]
    if len(eligible) < 2:
        return False
    values = [float(row[objective_key]) for row in eligible]
    return max(values) - min(values) <= tolerance


def aggregate_seed_qualification(
    rows: Iterable[Mapping[str, Any]],
    *,
    thresholds: QualificationThresholds | None = None,
) -> dict[str, Any]:
    limits = thresholds or QualificationThresholds()
    values = list(rows)
    if not values:
        return {
            "accepted": False,
            "status": QualificationStatus.REJECTED_ROBUSTNESS.value,
            "reason": "no_seed_results",
        }
    accepted = [bool(row.get("accepted", str(row.get("status", "")) == "PASS")) for row in values]
    collapse = [
        any("collapse" in str(reason) for reason in row.get("reasons", ()))
        for row in values
    ]
    objectives = [float(row.get("objective", float("-inf"))) for row in values]
    pass_fraction = sum(accepted) / len(values)
    collapse_fraction = sum(collapse) / len(values)
    robust = (
        pass_fraction >= limits.min_seed_pass_fraction
        and collapse_fraction <= limits.max_seed_collapse_fraction
        and all(math.isfinite(value) for value in objectives)
    )
    return {
        "accepted": robust,
        "status": QualificationStatus.PASS.value if robust else QualificationStatus.REJECTED_ROBUSTNESS.value,
        "seed_count": len(values),
        "pass_fraction": pass_fraction,
        "collapse_fraction": collapse_fraction,
        "median_objective": median(objectives),
        "worst_objective": min(objectives),
        "best_objective": max(objectives),
    }
