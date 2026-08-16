from freqtrade.hedge.production.fault_campaign_qualification import (
    FaultQualificationPolicy,
    FaultScenarioEvidence,
    evaluate_fault_qualification,
)


def _row(name, **updates):
    values = dict(
        scenario=name,
        passed=True,
        duplicate_writes=0,
        final_converged=True,
        new_risk_blocked_during_fault=True,
        recovery_seconds=1.0,
        state_hash_match=True,
        outbox_drained=True,
        fencing_preserved=True,
    )
    values.update(updates)
    return FaultScenarioEvidence(**values)


def test_small_policy_passes_when_every_invariant_holds():
    policy = FaultQualificationPolicy(required_scenarios=("A", "B"))
    report = evaluate_fault_qualification((_row("A"), _row("B")), policy=policy)
    assert report.passed
    assert report.coverage_ratio == 1.0
    assert not report.full_catalog_qualified
    assert report.qualification_scope == "FOCUSED_OR_CUSTOM"


def test_duplicate_write_fails_even_when_scenario_claims_pass():
    policy = FaultQualificationPolicy(required_scenarios=("A",))
    report = evaluate_fault_qualification((_row("A", duplicate_writes=1),), policy=policy)
    assert not report.passed
    assert "DUPLICATE_WRITE:A" in report.reasons


def test_missing_scenario_cannot_be_counted_as_full_campaign():
    policy = FaultQualificationPolicy(required_scenarios=("A", "B"))
    report = evaluate_fault_qualification((_row("A"),), policy=policy)
    assert not report.passed
    assert "MISSING:B" in report.reasons
