from dataclasses import replace

from freqtrade.hedge.production.evidence_composition_v2 import (
    OperationalProof,
    OperationalProofKind,
    QualificationStage,
    StageArtifactEvidence,
    qualify_production_evidence,
)
from freqtrade.hedge.production.model_artifact_v2 import ModelArtifactV2


def _artifact(**changes: str) -> ModelArtifactV2:
    values = {
        "model_id": "hprl-champion",
        "algorithm": "fast-td3",
        "model_sha256": "1" * 64,
        "source_authority_sha256": "2" * 64,
        "dataset_sha256": "3" * 64,
        "feature_set_sha256": "4" * 64,
        "risk_policy_sha256": "5" * 64,
        "simulator_calibration_sha256": "6" * 64,
        "benchmark_protocol_sha256": "7" * 64,
        "evidence_dag_sha256": "8" * 64,
    }
    values.update(changes)
    return ModelArtifactV2(**values)


def _stages(artifact: ModelArtifactV2) -> tuple[StageArtifactEvidence, ...]:
    return tuple(
        StageArtifactEvidence(stage, artifact, f"{index:x}" * 64, True)
        for index, stage in enumerate(QualificationStage, start=10)
    )


def _proofs(artifact: ModelArtifactV2) -> tuple[OperationalProof, ...]:
    return tuple(
        OperationalProof(kind, artifact.fingerprint, f"{index:x}" * 64, True)
        for index, kind in enumerate(OperationalProofKind, start=1)
    )


def test_same_immutable_artifact_and_all_operational_proofs_qualify() -> None:
    artifact = _artifact()
    decision = qualify_production_evidence(_stages(artifact), _proofs(artifact))
    assert decision.qualified
    assert decision.artifact_fingerprint == artifact.fingerprint
    assert decision.reasons == ()


def test_model_change_between_shadow_and_canary_requires_requalification() -> None:
    original = _artifact()
    stages = list(_stages(original))
    stages[-1] = replace(stages[-1], artifact=_artifact(model_sha256="9" * 64))
    decision = qualify_production_evidence(tuple(stages), _proofs(original))
    assert not decision.qualified
    assert "IMMUTABLE_ARTIFACT_CHANGED_BETWEEN_STAGES" in decision.reasons
    assert "PROOF_ARTIFACT_CANNOT_BE_RESOLVED" in decision.reasons


def test_missing_or_mismatched_operational_proof_fails_closed() -> None:
    artifact = _artifact()
    proofs = list(_proofs(artifact))
    proofs.pop()
    proofs[0] = replace(proofs[0], artifact_fingerprint="f" * 64)
    decision = qualify_production_evidence(_stages(artifact), tuple(proofs))
    assert not decision.qualified
    assert "PROOF_MISSING:FAULT_CAMPAIGN" in decision.reasons
    assert "PROOF_ARTIFACT_MISMATCH:RECOVERY" in decision.reasons
