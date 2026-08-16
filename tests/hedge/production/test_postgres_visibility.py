from dataclasses import dataclass

from freqtrade.hedge.production.postgres_visibility import (
    PostgresCompositePolicy,
    evaluate_postgres_composite,
)


@dataclass
class _Report:
    passed: bool
    evidence_sha256: str = "a" * 64


def test_composite_requires_all_real_evidence_by_default():
    report = evaluate_postgres_composite(
        core=_Report(True),
        visibility=None,
        backup=None,
        restore=None,
        failover=None,
    )
    assert not report.passed
    assert "POSTGRES_VISIBILITY_EVIDENCE_MISSING_OR_FAILED" in report.reasons
    assert "POSTGRES_FAILOVER_EVIDENCE_MISSING_OR_FAILED" in report.reasons


def test_composite_can_express_core_only_diagnostic_without_promotion():
    policy = PostgresCompositePolicy(
        require_backup=False,
        require_restore=False,
        require_failover=False,
        require_visibility=False,
    )
    report = evaluate_postgres_composite(core=_Report(True), visibility=None, policy=policy)
    assert report.passed
    assert not report.promotion_eligible
