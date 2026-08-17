"""Content-addressed research-to-production qualification evidence DAG."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class EvidenceStage(StrEnum):
    DATA = "DATA"
    FEATURES = "FEATURES"
    SIMULATION = "SIMULATION"
    TRAINING = "TRAINING"
    BENCHMARK = "BENCHMARK"
    SHADOW = "SHADOW"
    CANARY = "CANARY"
    PRODUCTION = "PRODUCTION"


def _sha(value: str, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be str")
    value = value.lower().strip()
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be sha256")
    return value


@dataclass(frozen=True, slots=True)
class EvidenceNode:
    artifact_sha256: str
    stage: EvidenceStage
    source_authority_sha256: str
    dependencies: tuple[str, ...] = ()
    passed: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifact_sha256", _sha(self.artifact_sha256, "artifact_sha256"))
        object.__setattr__(self, "source_authority_sha256", _sha(self.source_authority_sha256, "source_authority_sha256"))
        if not isinstance(self.stage, EvidenceStage) or not isinstance(self.passed, bool):
            raise TypeError("stage/passed have invalid types")
        deps = tuple(_sha(item, "dependency") for item in self.dependencies)
        if len(set(deps)) != len(deps) or self.artifact_sha256 in deps:
            raise ValueError("dependencies must be unique and cannot self-reference")
        object.__setattr__(self, "dependencies", deps)


@dataclass(frozen=True, slots=True)
class EvidenceDAGDecision:
    promotable: bool
    reasons: tuple[str, ...]
    ordered_artifacts: tuple[str, ...]


def evaluate_evidence_dag(nodes: tuple[EvidenceNode, ...], *, production_artifact_sha256: str) -> EvidenceDAGDecision:
    target = _sha(production_artifact_sha256, "production_artifact_sha256")
    by_id = {node.artifact_sha256: node for node in nodes}
    if len(by_id) != len(nodes):
        return EvidenceDAGDecision(False, ("DUPLICATE_ARTIFACT",), ())
    reasons: list[str] = []
    ordered: list[str] = []
    visiting: set[str] = set()
    visited: set[str] = set()
    authority: str | None = None

    def visit(artifact: str) -> None:
        nonlocal authority
        if artifact in visiting:
            reasons.append("EVIDENCE_CYCLE")
            return
        if artifact in visited:
            return
        node = by_id.get(artifact)
        if node is None:
            reasons.append(f"MISSING_DEPENDENCY:{artifact}")
            return
        visiting.add(artifact)
        if authority is None:
            authority = node.source_authority_sha256
        elif authority != node.source_authority_sha256:
            reasons.append("SOURCE_AUTHORITY_MISMATCH")
        if not node.passed:
            reasons.append(f"ARTIFACT_FAILED:{artifact}")
        for dependency in node.dependencies:
            visit(dependency)
        visiting.remove(artifact)
        visited.add(artifact)
        ordered.append(artifact)

    visit(target)
    target_node = by_id.get(target)
    if target_node is None or target_node.stage is not EvidenceStage.PRODUCTION:
        reasons.append("PRODUCTION_TARGET_REQUIRED")
    unique = tuple(dict.fromkeys(reasons))
    return EvidenceDAGDecision(not unique, unique, tuple(ordered))
