"""Immutable artifact and operational-proof composition from replay through canary."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from freqtrade.hedge.contracts.types import required_text

from .model_artifact_v2 import ModelArtifactV2


def _sha256(value: object, *, field_name: str) -> str:
    digest = required_text(value, field_name=field_name, max_length=64).lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"{field_name} must be sha256")
    return digest


class QualificationStage(StrEnum):
    REPLAY = "REPLAY"
    SHADOW = "SHADOW"
    SOAK = "SOAK"
    CANARY = "CANARY"


@dataclass(frozen=True, slots=True)
class StageArtifactEvidence:
    stage: QualificationStage
    artifact: ModelArtifactV2
    evidence_sha256: str
    passed: bool

    def __post_init__(self) -> None:
        if not isinstance(self.stage, QualificationStage):
            raise TypeError("stage must be QualificationStage")
        if not isinstance(self.artifact, ModelArtifactV2):
            raise TypeError("artifact must be ModelArtifactV2")
        object.__setattr__(
            self,
            "evidence_sha256",
            _sha256(self.evidence_sha256, field_name="evidence_sha256"),
        )
        if not isinstance(self.passed, bool):
            raise TypeError("passed must be bool")


class OperationalProofKind(StrEnum):
    RECOVERY = "RECOVERY"
    POSTGRES_TRANSACTION_SEMANTICS = "POSTGRES_TRANSACTION_SEMANTICS"
    FAULT_CAMPAIGN = "FAULT_CAMPAIGN"


@dataclass(frozen=True, slots=True)
class OperationalProof:
    kind: OperationalProofKind
    artifact_fingerprint: str
    evidence_sha256: str
    passed: bool

    def __post_init__(self) -> None:
        if not isinstance(self.kind, OperationalProofKind):
            raise TypeError("kind must be OperationalProofKind")
        for name in ("artifact_fingerprint", "evidence_sha256"):
            object.__setattr__(self, name, _sha256(getattr(self, name), field_name=name))
        if not isinstance(self.passed, bool):
            raise TypeError("passed must be bool")


@dataclass(frozen=True, slots=True)
class ProductionEvidenceDecision:
    qualified: bool
    artifact_fingerprint: str | None
    reasons: tuple[str, ...]


def qualify_production_evidence(
    stages: tuple[StageArtifactEvidence, ...],
    proofs: tuple[OperationalProof, ...],
) -> ProductionEvidenceDecision:
    by_stage = {item.stage: item for item in stages}
    by_proof = {item.kind: item for item in proofs}
    reasons: list[str] = []
    if len(by_stage) != len(stages):
        reasons.append("DUPLICATE_QUALIFICATION_STAGE")
    if len(by_proof) != len(proofs):
        reasons.append("DUPLICATE_OPERATIONAL_PROOF")
    reasons.extend(
        f"STAGE_MISSING:{stage.value}"
        for stage in QualificationStage
        if stage not in by_stage
    )
    reasons.extend(
        f"PROOF_MISSING:{kind.value}"
        for kind in OperationalProofKind
        if kind not in by_proof
    )
    reasons.extend(
        f"STAGE_FAILED:{item.stage.value}" for item in stages if not item.passed
    )
    reasons.extend(f"PROOF_FAILED:{item.kind.value}" for item in proofs if not item.passed)

    fingerprints = {item.artifact.fingerprint for item in stages}
    artifact_fingerprint = next(iter(fingerprints)) if len(fingerprints) == 1 else None
    if len(fingerprints) > 1:
        reasons.append("IMMUTABLE_ARTIFACT_CHANGED_BETWEEN_STAGES")
    if artifact_fingerprint is not None:
        reasons.extend(
            f"PROOF_ARTIFACT_MISMATCH:{proof.kind.value}"
            for proof in proofs
            if proof.artifact_fingerprint != artifact_fingerprint
        )
    elif proofs:
        reasons.append("PROOF_ARTIFACT_CANNOT_BE_RESOLVED")

    evidence_hashes = {item.evidence_sha256 for item in stages}
    if len(evidence_hashes) != len(stages):
        reasons.append("STAGE_EVIDENCE_REUSED")
    return ProductionEvidenceDecision(not reasons, artifact_fingerprint, tuple(reasons))
