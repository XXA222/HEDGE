from freqtrade.hedge.research.evidence_dag import EvidenceNode, EvidenceStage, evaluate_evidence_dag


def test_complete_single_source_evidence_chain_promotes() -> None:
    source = "a" * 64
    data = EvidenceNode("1" * 64, EvidenceStage.DATA, source, passed=True)
    train = EvidenceNode("2" * 64, EvidenceStage.TRAINING, source, (data.artifact_sha256,), True)
    prod = EvidenceNode("3" * 64, EvidenceStage.PRODUCTION, source, (train.artifact_sha256,), True)
    decision = evaluate_evidence_dag((data, train, prod), production_artifact_sha256=prod.artifact_sha256)
    assert decision.promotable
    assert decision.ordered_artifacts[-1] == prod.artifact_sha256


def test_missing_or_failed_evidence_fails_closed() -> None:
    source = "a" * 64
    prod = EvidenceNode("3" * 64, EvidenceStage.PRODUCTION, source, ("4" * 64,), False)
    decision = evaluate_evidence_dag((prod,), production_artifact_sha256=prod.artifact_sha256)
    assert not decision.promotable
    assert any(reason.startswith("MISSING_DEPENDENCY") for reason in decision.reasons)
