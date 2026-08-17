"""Fail-closed Fast Research to Exact Event Qualification funnel."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum

from freqtrade.hedge.contracts import finite_decimal
from freqtrade.hedge.contracts.types import required_text


def _sha256(value: object, *, field_name: str) -> str:
    digest = required_text(value, field_name=field_name, max_length=64).lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"{field_name} must be sha256")
    return digest


class ExactComponent(StrEnum):
    CANONICAL_MATCHER = "CANONICAL_MATCHER"
    FEES = "FEES"
    FUNDING = "FUNDING"
    PARTIAL_FILLS = "PARTIAL_FILLS"
    DUAL_LEGS = "DUAL_LEGS"
    RISK = "RISK"
    PLANNER = "PLANNER"
    EXECUTION_STATE = "EXECUTION_STATE"
    RECONCILIATION = "RECONCILIATION"


REQUIRED_EXACT_COMPONENTS = frozenset(ExactComponent)


@dataclass(frozen=True, slots=True)
class FastResearchCandidate:
    candidate_id: str
    dataset_sha256: str
    feature_set_sha256: str
    protocol_sha256: str
    rank: int
    score: Decimal
    passed: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidate_id",
            required_text(self.candidate_id, field_name="candidate_id", max_length=128),
        )
        for name in ("dataset_sha256", "feature_set_sha256", "protocol_sha256"):
            object.__setattr__(self, name, _sha256(getattr(self, name), field_name=name))
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank < 1:
            raise ValueError("rank must be a positive int")
        object.__setattr__(self, "score", finite_decimal(self.score, field_name="score"))
        if not isinstance(self.passed, bool):
            raise TypeError("passed must be bool")


@dataclass(frozen=True, slots=True)
class ExactQualificationResult:
    candidate_id: str
    dataset_sha256: str
    feature_set_sha256: str
    protocol_sha256: str
    source_sha256: str
    simulator_sha256: str
    cost_model_sha256: str
    result_sha256: str
    components: frozenset[ExactComponent]
    passed: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidate_id",
            required_text(self.candidate_id, field_name="candidate_id", max_length=128),
        )
        for name in (
            "dataset_sha256",
            "feature_set_sha256",
            "protocol_sha256",
            "source_sha256",
            "simulator_sha256",
            "cost_model_sha256",
            "result_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), field_name=name))
        components = frozenset(self.components)
        if any(not isinstance(component, ExactComponent) for component in components):
            raise TypeError("components must contain ExactComponent values")
        object.__setattr__(self, "components", components)
        if not isinstance(self.passed, bool):
            raise TypeError("passed must be bool")


@dataclass(frozen=True, slots=True)
class FunnelDecision:
    promotable_candidate_ids: tuple[str, ...]
    rejected: tuple[tuple[str, tuple[str, ...]], ...]

    @property
    def passed(self) -> bool:
        return bool(self.promotable_candidate_ids)


def _exact_rejection_reasons(
    fast: FastResearchCandidate | None,
    exact: ExactQualificationResult,
    *,
    fast_cutoff: int,
) -> list[str]:
    reasons: list[str] = []
    if fast is None:
        reasons.append("FAST_EVIDENCE_MISSING")
    else:
        if not fast.passed:
            reasons.append("FAST_RESEARCH_FAILED")
        if fast.rank > fast_cutoff:
            reasons.append("FAST_CUTOFF_EXCEEDED")
        for name in ("dataset_sha256", "feature_set_sha256", "protocol_sha256"):
            if getattr(fast, name) != getattr(exact, name):
                reasons.append(f"{name.upper()}_MISMATCH")
    missing = REQUIRED_EXACT_COMPONENTS - exact.components
    reasons.extend(
        f"EXACT_COMPONENT_MISSING:{component.value}"
        for component in sorted(missing, key=lambda item: item.value)
    )
    if not exact.passed:
        reasons.append("EXACT_QUALIFICATION_FAILED")
    return reasons


def qualify_research_funnel(
    fast_candidates: tuple[FastResearchCandidate, ...],
    exact_results: tuple[ExactQualificationResult, ...],
    *,
    fast_cutoff: int = 300,
    exact_cutoff: int = 30,
) -> FunnelDecision:
    """Return only candidates that passed Fast Research and complete exact qualification."""
    if isinstance(fast_cutoff, bool) or not isinstance(fast_cutoff, int) or fast_cutoff < 1:
        raise ValueError("fast_cutoff must be a positive int")
    if isinstance(exact_cutoff, bool) or not isinstance(exact_cutoff, int) or exact_cutoff < 1:
        raise ValueError("exact_cutoff must be a positive int")

    fast_by_id = {candidate.candidate_id: candidate for candidate in fast_candidates}
    if len(fast_by_id) != len(fast_candidates):
        raise ValueError("duplicate fast candidate_id")
    exact_by_id = {result.candidate_id: result for result in exact_results}
    if len(exact_by_id) != len(exact_results):
        raise ValueError("duplicate exact candidate_id")
    if len(exact_results) > exact_cutoff:
        raise ValueError("exact result count exceeds exact_cutoff")

    promotable: list[str] = []
    rejected: list[tuple[str, tuple[str, ...]]] = []
    for candidate_id, exact in sorted(exact_by_id.items()):
        reasons = _exact_rejection_reasons(
            fast_by_id.get(candidate_id),
            exact,
            fast_cutoff=fast_cutoff,
        )
        if reasons:
            rejected.append((candidate_id, tuple(reasons)))
        else:
            promotable.append(candidate_id)
    return FunnelDecision(tuple(promotable), tuple(rejected))
