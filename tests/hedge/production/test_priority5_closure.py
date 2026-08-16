from datetime import UTC, datetime

from freqtrade.hedge.production.priority5_closure import (
    EvidenceClass,
    PriorityEvidence,
    PriorityGate,
    evaluate_priority5_closure,
)

SHA = "a" * 64
ART = "b" * 64
NOW = datetime(2026, 8, 16, tzinfo=UTC)


def _evidence(gate, evidence_class=EvidenceClass.MEASURED, passed=True, source=SHA):
    return PriorityEvidence(gate, passed, evidence_class, ART, source, NOW, "pytest")


def test_all_five_priorities_pass_with_measured_external_evidence():
    rows = (
        _evidence(PriorityGate.SOURCE_RUNTIME, EvidenceClass.OFFLINE),
        _evidence(PriorityGate.POSTGRESQL),
        _evidence(PriorityGate.MODEL_REAL_MARKET),
        _evidence(PriorityGate.FAULT_RECOVERY, EvidenceClass.OFFLINE),
        _evidence(PriorityGate.LONG_RUN),
    )
    report = evaluate_priority5_closure(rows, expected_source_sha256=SHA, now=NOW)
    assert report.passed


def test_offline_postgres_cannot_be_promoted():
    rows = tuple(_evidence(gate, EvidenceClass.OFFLINE) for gate in PriorityGate)
    report = evaluate_priority5_closure(rows, expected_source_sha256=SHA, now=NOW)
    assert not report.passed
    assert any("POSTGRESQL:MEASURED_EVIDENCE_REQUIRED" in item for item in report.reasons)


def test_source_identity_mismatch_fails_closed():
    rows = tuple(_evidence(gate, source="c" * 64) for gate in PriorityGate)
    report = evaluate_priority5_closure(rows, expected_source_sha256=SHA, now=NOW)
    assert not report.passed
    assert any("SOURCE_IDENTITY_MISMATCH" in item for item in report.reasons)
