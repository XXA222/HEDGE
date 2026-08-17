from decimal import Decimal

from freqtrade.hedge.research.qualification_funnel import (
    ExactComponent,
    ExactQualificationResult,
    FastResearchCandidate,
    qualify_research_funnel,
)


H = "a" * 64


def _fast(**changes: object) -> FastResearchCandidate:
    values = {
        "candidate_id": "candidate-1",
        "dataset_sha256": H,
        "feature_set_sha256": H,
        "protocol_sha256": H,
        "rank": 1,
        "score": Decimal("1.2"),
        "passed": True,
    }
    values.update(changes)
    return FastResearchCandidate(**values)  # type: ignore[arg-type]


def _exact(**changes: object) -> ExactQualificationResult:
    values = {
        "candidate_id": "candidate-1",
        "dataset_sha256": H,
        "feature_set_sha256": H,
        "protocol_sha256": H,
        "source_sha256": H,
        "simulator_sha256": H,
        "cost_model_sha256": H,
        "result_sha256": H,
        "components": frozenset(ExactComponent),
        "passed": True,
    }
    values.update(changes)
    return ExactQualificationResult(**values)  # type: ignore[arg-type]


def test_only_complete_exact_result_can_promote() -> None:
    decision = qualify_research_funnel((_fast(),), (_exact(),))
    assert decision.passed
    assert decision.promotable_candidate_ids == ("candidate-1",)


def test_funnel_rejects_missing_fast_and_incomplete_exact() -> None:
    exact = _exact(components=frozenset({ExactComponent.FEES}), passed=False)
    decision = qualify_research_funnel((), (exact,))
    reasons = decision.rejected[0][1]
    assert "FAST_EVIDENCE_MISSING" in reasons
    assert "EXACT_QUALIFICATION_FAILED" in reasons
    assert "EXACT_COMPONENT_MISSING:FUNDING" in reasons


def test_funnel_rejects_fingerprint_drift_and_cutoff_bypass() -> None:
    decision = qualify_research_funnel(
        (_fast(rank=301),),
        (_exact(dataset_sha256="b" * 64),),
    )
    assert decision.promotable_candidate_ids == ()
    assert decision.rejected == (
        (
            "candidate-1",
            ("FAST_CUTOFF_EXCEEDED", "DATASET_SHA256_MISMATCH"),
        ),
    )
