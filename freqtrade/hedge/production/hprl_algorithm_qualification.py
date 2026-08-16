"""Paired multi-seed, multi-window qualification for HPRL algorithms.

Algorithm promotion is based on repeatable out-of-sample evidence against a deterministic
baseline, not on a single profitable backtest or reward curve.  Behavior, replay parity
and inference latency remain mandatory independent gates.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import isfinite
from statistics import median, pstdev


def _sha256(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return sha256(raw).hexdigest()


def _valid_sha256(value: str) -> bool:
    text = value.lower()
    return len(text) == 64 and all(ch in "0123456789abcdef" for ch in text)


@dataclass(frozen=True, slots=True)
class AlgorithmTrialEvidence:
    algorithm: str
    seed: int
    window_id: str
    sample_count: int
    net_return: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    cvar95: float
    turnover: float
    fee_fraction: float
    funding_fraction: float
    slippage_fraction: float
    behavior_passed: bool
    behavior_sha256: str
    replay_parity_passed: bool
    inference_p95_ms: float
    source_sha256: str

    def __post_init__(self) -> None:
        if not self.algorithm.strip() or not self.window_id.strip():
            raise ValueError("algorithm and window_id are required")
        if self.seed < 0 or self.sample_count <= 0:
            raise ValueError("seed must be nonnegative and sample_count positive")
        for name in (
            "net_return",
            "sharpe",
            "sortino",
            "calmar",
            "max_drawdown",
            "cvar95",
            "turnover",
            "fee_fraction",
            "funding_fraction",
            "slippage_fraction",
            "inference_p95_ms",
        ):
            value = float(getattr(self, name))
            if not isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if (
            self.max_drawdown < 0
            or self.cvar95 < 0
            or self.turnover < 0
            or self.inference_p95_ms < 0
        ):
            raise ValueError("risk/turnover/latency metrics cannot be negative")
        if self.fee_fraction < 0 or self.funding_fraction < 0 or self.slippage_fraction < 0:
            raise ValueError("cost fractions cannot be negative")
        for name in ("behavior_sha256", "source_sha256"):
            value = str(getattr(self, name)).lower()
            if not _valid_sha256(value):
                raise ValueError(f"{name} must be SHA-256 hex")
            object.__setattr__(self, name, value)

    @property
    def pair_key(self) -> tuple[str, int]:
        return self.window_id, self.seed

    @property
    def total_cost_fraction(self) -> float:
        return self.fee_fraction + self.funding_fraction + self.slippage_fraction


@dataclass(frozen=True, slots=True)
class AlgorithmQualificationPolicy:
    baseline_algorithm: str = "deterministic"
    minimum_seeds: int = 3
    minimum_windows: int = 3
    minimum_samples_per_trial: int = 10_000
    minimum_pair_coverage: float = 1.0
    minimum_positive_pair_ratio: float = 0.55
    minimum_median_excess_return: float = 0.0
    maximum_worst_drawdown: float = 0.25
    maximum_median_cvar95: float = 0.08
    maximum_return_stddev: float = 0.25
    maximum_inference_p95_ms: float = 100.0
    require_all_behavior: bool = True
    require_all_replay_parity: bool = True

    def __post_init__(self) -> None:
        if (
            self.minimum_seeds <= 0
            or self.minimum_windows <= 0
            or self.minimum_samples_per_trial <= 0
        ):
            raise ValueError("minimum qualification counts must be positive")
        for name in ("minimum_pair_coverage", "minimum_positive_pair_ratio"):
            value = float(getattr(self, name))
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be within [0,1]")
        if self.maximum_worst_drawdown <= 0 or self.maximum_median_cvar95 <= 0:
            raise ValueError("risk ceilings must be positive")
        if self.maximum_return_stddev < 0 or self.maximum_inference_p95_ms <= 0:
            raise ValueError("stability/latency ceilings are invalid")


@dataclass(frozen=True, slots=True)
class AlgorithmQualificationReport:
    algorithm: str
    baseline_algorithm: str
    passed: bool
    trials: int
    paired_trials: int
    seeds: int
    windows: int
    pair_coverage: float
    positive_pair_ratio: float
    median_excess_return: float
    median_net_return: float
    median_sharpe: float
    median_sortino: float
    median_calmar: float
    worst_drawdown: float
    median_cvar95: float
    return_stddev: float
    median_total_cost_fraction: float
    inference_p95_ms: float
    behavior_pass_ratio: float
    replay_parity_pass_ratio: float
    score: float
    evidence_sha256: str
    reasons: tuple[str, ...]


def _unique_by_pair(
    rows: Iterable[AlgorithmTrialEvidence],
    *,
    label: str,
) -> dict[tuple[str, int], AlgorithmTrialEvidence]:
    result: dict[tuple[str, int], AlgorithmTrialEvidence] = {}
    for row in rows:
        if row.pair_key in result:
            raise ValueError(f"duplicate {label} trial for window/seed {row.pair_key}")
        result[row.pair_key] = row
    return result



def _qualification_identity(
    rows: tuple[AlgorithmTrialEvidence, ...],
    base_rows: tuple[AlgorithmTrialEvidence, ...],
    policy: AlgorithmQualificationPolicy,
) -> tuple[
    str,
    dict[tuple[str, int], AlgorithmTrialEvidence],
    dict[tuple[str, int], AlgorithmTrialEvidence],
    tuple[tuple[AlgorithmTrialEvidence, AlgorithmTrialEvidence], ...],
    float,
    int,
    int,
    list[str],
]:
    if not rows:
        raise ValueError("algorithm trials are required")
    names = {item.algorithm for item in rows}
    if len(names) != 1:
        raise ValueError("qualification input must contain exactly one candidate algorithm")
    algorithm = next(iter(names))
    if algorithm == policy.baseline_algorithm:
        raise ValueError("candidate algorithm cannot be the deterministic baseline")
    if any(item.algorithm != policy.baseline_algorithm for item in base_rows):
        raise ValueError("baseline rows do not match policy baseline_algorithm")
    reasons: list[str] = []
    if len({item.source_sha256 for item in (*rows, *base_rows)}) != 1:
        reasons.append("ALGORITHM_SOURCE_IDENTITY_MISMATCH")
    candidate_by = _unique_by_pair(rows, label="candidate")
    baseline_by = _unique_by_pair(base_rows, label="baseline")
    paired_keys = tuple(sorted(set(candidate_by).intersection(baseline_by)))
    paired = tuple((candidate_by[key], baseline_by[key]) for key in paired_keys)
    coverage = len(paired) / max(1, len(candidate_by))
    seeds = len({item.seed for item in rows})
    windows = len({item.window_id for item in rows})
    coverage_checks = (
        (coverage < policy.minimum_pair_coverage, "ALGORITHM_BASELINE_PAIR_COVERAGE_INSUFFICIENT"),
        (seeds < policy.minimum_seeds, "ALGORITHM_SEED_COVERAGE_INSUFFICIENT"),
        (windows < policy.minimum_windows, "ALGORITHM_WALKFORWARD_COVERAGE_INSUFFICIENT"),
        (
            any(item.sample_count < policy.minimum_samples_per_trial for item in rows),
            "ALGORITHM_TRIAL_SAMPLE_INSUFFICIENT",
        ),
    )
    reasons.extend(reason for failed, reason in coverage_checks if failed)
    return algorithm, candidate_by, baseline_by, paired, coverage, seeds, windows, reasons


def _performance_gate_reasons(
    *,
    positive_ratio: float,
    median_excess: float,
    worst_drawdown: float,
    median_cvar: float,
    return_stddev: float,
    policy: AlgorithmQualificationPolicy,
) -> list[str]:
    checks = (
        (positive_ratio < policy.minimum_positive_pair_ratio, "ALGORITHM_PAIRED_EDGE_INCONSISTENT"),
        (
            median_excess < policy.minimum_median_excess_return,
            "ALGORITHM_MEDIAN_EXCESS_RETURN_INSUFFICIENT",
        ),
        (worst_drawdown > policy.maximum_worst_drawdown, "ALGORITHM_MAX_DRAWDOWN_EXCEEDED"),
        (median_cvar > policy.maximum_median_cvar95, "ALGORITHM_CVAR_EXCEEDED"),
        (return_stddev > policy.maximum_return_stddev, "ALGORITHM_RETURN_INSTABILITY"),
    )
    return [reason for failed, reason in checks if failed]


def _runtime_gate_reasons(
    *,
    behavior_ratio: float,
    replay_ratio: float,
    latency: float,
    policy: AlgorithmQualificationPolicy,
) -> list[str]:
    checks = (
        (
            policy.require_all_behavior and behavior_ratio < 1.0,
            "ALGORITHM_POSITION_BEHAVIOR_NOT_UNIVERSALLY_PASSED",
        ),
        (
            policy.require_all_replay_parity and replay_ratio < 1.0,
            "ALGORITHM_REPLAY_PARITY_NOT_UNIVERSALLY_PASSED",
        ),
        (latency > policy.maximum_inference_p95_ms, "ALGORITHM_INFERENCE_LATENCY_EXCEEDED"),
    )
    return [reason for failed, reason in checks if failed]

def qualify_algorithm(
    trials: Iterable[AlgorithmTrialEvidence],
    baselines: Iterable[AlgorithmTrialEvidence],
    *,
    policy: AlgorithmQualificationPolicy | None = None,
) -> AlgorithmQualificationReport:
    p = policy or AlgorithmQualificationPolicy()
    rows = tuple(trials)
    base_rows = tuple(baselines)
    (
        algorithm,
        candidate_by,
        baseline_by,
        paired,
        coverage,
        seeds,
        windows,
        reasons,
    ) = _qualification_identity(rows, base_rows, p)

    excess = [candidate.net_return - baseline.net_return for candidate, baseline in paired]
    positive_ratio = sum(value > 0 for value in excess) / max(1, len(excess))
    median_excess = median(excess) if excess else float("-inf")
    worst_drawdown = max(item.max_drawdown for item in rows)
    median_cvar = median(item.cvar95 for item in rows)
    return_stddev = pstdev(item.net_return for item in rows) if len(rows) > 1 else 0.0
    reasons.extend(
        _performance_gate_reasons(
            positive_ratio=positive_ratio,
            median_excess=median_excess,
            worst_drawdown=worst_drawdown,
            median_cvar=median_cvar,
            return_stddev=return_stddev,
            policy=p,
        )
    )

    behavior_ratio = sum(item.behavior_passed for item in rows) / len(rows)
    replay_ratio = sum(item.replay_parity_passed for item in rows) / len(rows)
    latency = max(item.inference_p95_ms for item in rows)
    reasons.extend(
        _runtime_gate_reasons(
            behavior_ratio=behavior_ratio,
            replay_ratio=replay_ratio,
            latency=latency,
            policy=p,
        )
    )

    median_return = median(item.net_return for item in rows)
    median_sharpe = median(item.sharpe for item in rows)
    median_sortino = median(item.sortino for item in rows)
    median_calmar = median(item.calmar for item in rows)
    median_cost = median(item.total_cost_fraction for item in rows)
    score = (
        median_excess
        + 0.02 * median_sharpe
        + 0.01 * median_sortino
        + 0.01 * median_calmar
        - 0.50 * worst_drawdown
        - 0.25 * median_cvar
        - 0.10 * return_stddev
        - 0.25 * median_cost
    )
    paired_keys = tuple(sorted(set(candidate_by).intersection(baseline_by)))
    payload = {
        "policy": asdict(p),
        "algorithm": algorithm,
        "trials": [asdict(item) for item in sorted(rows, key=lambda x: (x.window_id, x.seed))],
        "baselines": [
            asdict(item) for item in sorted(base_rows, key=lambda x: (x.window_id, x.seed))
        ],
        "paired_keys": paired_keys,
        "reasons": reasons,
    }
    return AlgorithmQualificationReport(
        algorithm=algorithm,
        baseline_algorithm=p.baseline_algorithm,
        passed=not reasons,
        trials=len(rows),
        paired_trials=len(paired),
        seeds=seeds,
        windows=windows,
        pair_coverage=coverage,
        positive_pair_ratio=positive_ratio,
        median_excess_return=median_excess,
        median_net_return=median_return,
        median_sharpe=median_sharpe,
        median_sortino=median_sortino,
        median_calmar=median_calmar,
        worst_drawdown=worst_drawdown,
        median_cvar95=median_cvar,
        return_stddev=return_stddev,
        median_total_cost_fraction=median_cost,
        inference_p95_ms=latency,
        behavior_pass_ratio=behavior_ratio,
        replay_parity_pass_ratio=replay_ratio,
        score=score,
        evidence_sha256=_sha256(payload),
        reasons=tuple(dict.fromkeys(reasons)),
    )


@dataclass(frozen=True, slots=True)
class AlgorithmSelectionReport:
    passed: bool
    source_sha256: str
    qualified: tuple[AlgorithmQualificationReport, ...]
    rejected: tuple[AlgorithmQualificationReport, ...]
    champion: str | None
    ranking: tuple[tuple[str, float], ...]
    evidence_sha256: str


def qualify_candidate_set(
    trials: Iterable[AlgorithmTrialEvidence],
    *,
    policy: AlgorithmQualificationPolicy | None = None,
) -> AlgorithmSelectionReport:
    p = policy or AlgorithmQualificationPolicy()
    rows = tuple(trials)
    baseline = tuple(item for item in rows if item.algorithm == p.baseline_algorithm)
    if not baseline:
        raise ValueError("deterministic baseline evidence is required")
    algorithms = sorted({item.algorithm for item in rows if item.algorithm != p.baseline_algorithm})
    reports = tuple(
        qualify_algorithm(
            (item for item in rows if item.algorithm == algorithm),
            baseline,
            policy=p,
        )
        for algorithm in algorithms
    )
    qualified = tuple(
        sorted(
            (item for item in reports if item.passed),
            key=lambda x: x.score,
            reverse=True,
        )
    )
    rejected = tuple(item for item in reports if not item.passed)
    ranking = tuple((item.algorithm, item.score) for item in qualified)
    champion = qualified[0].algorithm if qualified else None
    baseline_sources = {item.source_sha256 for item in baseline}
    source_sha256 = next(iter(baseline_sources)) if len(baseline_sources) == 1 else ""
    digest = _sha256(
        {
            "source_sha256": source_sha256,
            "qualified": [(item.algorithm, item.evidence_sha256) for item in qualified],
            "rejected": [(item.algorithm, item.evidence_sha256) for item in rejected],
            "champion": champion,
        }
    )
    return AlgorithmSelectionReport(
        bool(qualified),
        source_sha256,
        qualified,
        rejected,
        champion,
        ranking,
        digest,
    )


@dataclass(frozen=True, slots=True)
class RealMarketModelQualificationReport:
    passed: bool
    source_sha256: str
    real_market_passed: bool
    production_evidence_eligible: bool
    zero_trade_writes: bool
    model_target_feed: bool
    behavior_passed: bool
    algorithm_selection_passed: bool
    champion: str | None
    evidence_sha256: str
    reasons: tuple[str, ...]


def qualify_real_market_model(
    real_market_report: object,
    behavior_report: object,
    algorithm_selection: AlgorithmSelectionReport,
) -> RealMarketModelQualificationReport:
    reasons: list[str] = []
    real_market_passed = bool(getattr(real_market_report, "passed", False))
    production_eligible = bool(
        getattr(real_market_report, "production_evidence_eligible", False)
    )
    writes = int(getattr(real_market_report, "real_trade_write_count", -1))
    zero_writes = writes == 0
    model_feed = bool(getattr(real_market_report, "model_target_feed", False))
    behavior_passed = bool(getattr(behavior_report, "passed", False))
    algorithm_passed = bool(algorithm_selection.passed and algorithm_selection.champion)
    if not real_market_passed:
        reasons.append("REAL_MARKET_ACCEPTANCE_FAILED")
    if not production_eligible:
        reasons.append("REAL_MARKET_NOT_PRODUCTION_EVIDENCE_ELIGIBLE")
    if not zero_writes:
        reasons.append("REAL_MARKET_ZERO_WRITE_INVARIANT_FAILED")
    if not model_feed:
        reasons.append("REAL_MARKET_MODEL_TARGET_FEED_MISSING")
    if not behavior_passed:
        reasons.append("REAL_MARKET_POSITION_BEHAVIOR_FAILED")
    if not algorithm_passed:
        reasons.append("ALGORITHM_SELECTION_NOT_QUALIFIED")
    source = algorithm_selection.source_sha256
    if not _valid_sha256(source):
        reasons.append("ALGORITHM_SOURCE_IDENTITY_INVALID")
    payload = {
        "source_sha256": source,
        "real_market": str(getattr(real_market_report, "evidence_sha256", "")),
        "behavior": str(getattr(behavior_report, "semantic_sha256", "")),
        "algorithm": algorithm_selection.evidence_sha256,
        "champion": algorithm_selection.champion,
        "writes": writes,
        "model_feed": model_feed,
        "reasons": reasons,
    }
    return RealMarketModelQualificationReport(
        passed=not reasons,
        source_sha256=source,
        real_market_passed=real_market_passed,
        production_evidence_eligible=production_eligible,
        zero_trade_writes=zero_writes,
        model_target_feed=model_feed,
        behavior_passed=behavior_passed,
        algorithm_selection_passed=algorithm_passed,
        champion=algorithm_selection.champion,
        evidence_sha256=_sha256(payload),
        reasons=tuple(reasons),
    )


def behavior_report_from_real_market(
    real_market_report: object,
    *,
    minimum_observations: int = 10_000,
) -> object:
    """Evaluate real Binance model-target observations using the canonical behavior analyzer."""
    from .risk_behavior import HprlBehaviorPolicy, analyze_hprl_position_behavior

    rows = tuple(getattr(real_market_report, "behavior_rows", ()))
    observations = tuple(getattr(item, "observation") for item in rows)
    return analyze_hprl_position_behavior(
        observations,
        policy=HprlBehaviorPolicy(minimum_observations=minimum_observations),
    )


def trial_from_mapping(payload: Mapping[str, object]) -> AlgorithmTrialEvidence:
    return AlgorithmTrialEvidence(**payload)  # type: ignore[arg-type]
