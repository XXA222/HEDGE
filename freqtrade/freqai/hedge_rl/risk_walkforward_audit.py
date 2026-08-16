"""Aggregate sequential Risk-Level RL OOS audits into a true walk-forward gate."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import pairwise
from pathlib import Path
from statistics import median
from typing import Any


@dataclass(frozen=True, slots=True)
class RiskWalkForwardThresholds:
    min_folds: int = 3
    min_fold_pass_ratio: float = 0.67
    min_positive_fixed_edge_ratio: float = 0.67
    min_positive_permutation_edge_ratio: float = 0.67
    min_distinct_model_ratio: float = 1.0
    min_median_fixed_edge: float = 0.0
    min_median_permutation_edge: float = 0.0

    def __post_init__(self) -> None:
        if self.min_folds < 2:
            raise ValueError("min_folds must be at least 2")
        for name in (
            "min_fold_pass_ratio",
            "min_positive_fixed_edge_ratio",
            "min_positive_permutation_edge_ratio",
            "min_distinct_model_ratio",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and within [0, 1]")
        for name in ("min_median_fixed_edge", "min_median_permutation_edge"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")


@dataclass(frozen=True, slots=True)
class RiskWalkForwardFold:
    source: str
    status: str
    train_start: str
    train_end: str
    oos_start: str
    oos_end: str
    policy_fingerprint: str
    action_signature: str
    observation_signature: str
    reward_signature: str
    feature_count: int
    steps: int
    fixed_edge: float
    permutation_edge: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RiskWalkForwardReport:
    passed: bool
    gates: Mapping[str, bool]
    thresholds: RiskWalkForwardThresholds
    folds: tuple[RiskWalkForwardFold, ...]
    metrics: Mapping[str, float | int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "hedge-risk-level-walk-forward-audit-v2",
            "status": "PASS" if self.passed else "FAIL",
            "gates": dict(self.gates),
            "thresholds": asdict(self.thresholds),
            "folds": [fold.to_dict() for fold in self.folds],
            "metrics": dict(self.metrics),
        }


def _parse_time(value: object) -> datetime:
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text)


def _fold_from_payload(payload: Mapping[str, Any], source: str) -> RiskWalkForwardFold:
    if str(payload.get("schema")) != "hedge-risk-level-learning-audit-v4":
        raise ValueError(f"{source}: unsupported Risk-Level audit schema")
    metadata = payload.get("metadata", {})
    evidence = payload.get("evidence", {})
    adaptive = payload.get("adaptive", {})
    if not isinstance(metadata, Mapping) or not isinstance(evidence, Mapping):
        raise TypeError(f"{source}: malformed audit metadata/evidence")
    train_start = str(metadata.get("train_start", "")).strip()
    train_end = str(metadata.get("train_end", "")).strip()
    start = str(metadata.get("oos_start", "")).strip()
    end = str(metadata.get("oos_end", "")).strip()
    if not train_start or not train_end or not start or not end:
        raise ValueError(
            f"{source}: train_start/train_end/oos_start/oos_end are required for walk-forward proof"
        )
    if _parse_time(train_end) <= _parse_time(train_start):
        raise ValueError(f"{source}: training window end must be after start")
    if _parse_time(start) <= _parse_time(train_end):
        raise ValueError(f"{source}: train_end must strictly precede oos_start")
    # Fail here instead of silently accepting lexical ordering.
    if _parse_time(end) <= _parse_time(start):
        raise ValueError(f"{source}: OOS window end must be after start")
    fingerprint = str(payload.get("policy_fingerprint", "")).strip()
    signature = str(payload.get("action_signature", "")).strip()
    observation_signature = str(payload.get("observation_signature", "")).strip()
    reward_signature = str(payload.get("reward_signature", "")).strip()
    feature_count = int(payload.get("feature_count", 0))
    if not fingerprint or not signature or not observation_signature or not reward_signature:
        raise ValueError(f"{source}: policy/action/observation/reward fingerprints are required")
    if feature_count < 1:
        raise ValueError(f"{source}: feature_count must be positive")
    return RiskWalkForwardFold(
        source=source,
        status=str(payload.get("status", "FAIL")),
        train_start=train_start,
        train_end=train_end,
        oos_start=start,
        oos_end=end,
        policy_fingerprint=fingerprint,
        action_signature=signature,
        observation_signature=observation_signature,
        reward_signature=reward_signature,
        feature_count=feature_count,
        steps=int(adaptive.get("steps", 0)) if isinstance(adaptive, Mapping) else 0,
        fixed_edge=float(evidence.get("adaptive_vs_best_fixed_edge", -1e12)),
        permutation_edge=float(evidence.get("adaptive_vs_permutation_quantile_edge", -1e12)),
    )


def load_risk_learning_audit(path: str | Path) -> tuple[Mapping[str, Any], str]:
    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"{source}: audit root must be an object")
    return payload, str(source)


def aggregate_risk_walk_forward(
    audits: Sequence[tuple[Mapping[str, Any], str]],
    *,
    thresholds: RiskWalkForwardThresholds | None = None,
) -> RiskWalkForwardReport:
    limits = thresholds or RiskWalkForwardThresholds()
    folds = [_fold_from_payload(payload, source) for payload, source in audits]
    folds.sort(key=lambda item: _parse_time(item.oos_start))

    non_overlapping = True
    training_cutoffs_advance = True
    for previous, current in pairwise(folds):
        if _parse_time(current.oos_start) <= _parse_time(previous.oos_end):
            non_overlapping = False
        if _parse_time(current.train_end) <= _parse_time(previous.train_end):
            training_cutoffs_advance = False

    training_precedes_oos = all(
        _parse_time(fold.train_end) < _parse_time(fold.oos_start) for fold in folds
    )

    count = len(folds)
    passed_count = sum(fold.status == "PASS" for fold in folds)
    positive_fixed = sum(fold.fixed_edge > 0 for fold in folds)
    positive_permutation = sum(fold.permutation_edge > 0 for fold in folds)
    distinct_models = len({fold.policy_fingerprint for fold in folds})
    action_signatures = {fold.action_signature for fold in folds}
    observation_signatures = {fold.observation_signature for fold in folds}
    reward_signatures = {fold.reward_signature for fold in folds}
    feature_counts = {fold.feature_count for fold in folds}

    pass_ratio = passed_count / count if count else 0.0
    positive_fixed_ratio = positive_fixed / count if count else 0.0
    positive_permutation_ratio = positive_permutation / count if count else 0.0
    distinct_model_ratio = distinct_models / count if count else 0.0
    median_fixed = median([fold.fixed_edge for fold in folds]) if folds else -1e12
    median_permutation = median([fold.permutation_edge for fold in folds]) if folds else -1e12
    total_steps = sum(max(fold.steps, 0) for fold in folds)
    weighted_fixed = (
        sum(fold.fixed_edge * max(fold.steps, 0) for fold in folds) / total_steps
        if total_steps
        else -1e12
    )
    weighted_permutation = (
        sum(fold.permutation_edge * max(fold.steps, 0) for fold in folds) / total_steps
        if total_steps
        else -1e12
    )

    gates = {
        "enough_sequential_folds": count >= limits.min_folds,
        "training_cutoff_precedes_each_oos": training_precedes_oos,
        "training_cutoffs_advance_across_folds": training_cutoffs_advance,
        "oos_windows_are_non_overlapping": non_overlapping,
        "action_contract_is_constant": len(action_signatures) == 1 and bool(action_signatures),
        "observation_contract_is_constant": (
            len(observation_signatures) == 1
            and len(feature_counts) == 1
            and bool(observation_signatures)
        ),
        "reward_contract_is_constant": len(reward_signatures) == 1 and bool(reward_signatures),
        "models_are_retrained_across_folds": (
            distinct_model_ratio >= limits.min_distinct_model_ratio
        ),
        "fold_pass_ratio": pass_ratio >= limits.min_fold_pass_ratio,
        "positive_fixed_edge_ratio": (positive_fixed_ratio >= limits.min_positive_fixed_edge_ratio),
        "positive_permutation_edge_ratio": (
            positive_permutation_ratio >= limits.min_positive_permutation_edge_ratio
        ),
        "median_fixed_edge_is_positive": median_fixed > limits.min_median_fixed_edge,
        "median_permutation_edge_is_positive": (
            median_permutation > limits.min_median_permutation_edge
        ),
    }
    metrics: dict[str, float | int] = {
        "fold_count": count,
        "fold_pass_ratio": pass_ratio,
        "positive_fixed_edge_ratio": positive_fixed_ratio,
        "positive_permutation_edge_ratio": positive_permutation_ratio,
        "distinct_model_count": distinct_models,
        "distinct_model_ratio": distinct_model_ratio,
        "median_fixed_edge": median_fixed,
        "median_permutation_edge": median_permutation,
        "step_weighted_fixed_edge": weighted_fixed,
        "step_weighted_permutation_edge": weighted_permutation,
        "total_oos_steps": total_steps,
    }
    return RiskWalkForwardReport(
        passed=all(gates.values()),
        gates=gates,
        thresholds=limits,
        folds=tuple(folds),
        metrics=metrics,
    )


def write_risk_walk_forward_report(
    report: RiskWalkForwardReport,
    path: str | Path,
) -> Path:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output
