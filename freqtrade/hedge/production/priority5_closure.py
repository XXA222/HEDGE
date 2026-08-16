"""Fail-closed evidence aggregation for the five HPRL development priorities.

This is a development qualification layer, not a replacement for Production Readiness.
It refuses to turn offline or synthetic checks into measured PostgreSQL, real-market or
long-run evidence.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from hashlib import sha256
import json
from pathlib import Path


class PriorityGate(StrEnum):
    SOURCE_RUNTIME = "SOURCE_RUNTIME"
    POSTGRESQL = "POSTGRESQL"
    MODEL_REAL_MARKET = "MODEL_REAL_MARKET"
    FAULT_RECOVERY = "FAULT_RECOVERY"
    LONG_RUN = "LONG_RUN"


class EvidenceClass(StrEnum):
    OFFLINE = "OFFLINE"
    MEASURED = "MEASURED"


MEASURED_ONLY_GATES = frozenset(
    {
        PriorityGate.POSTGRESQL,
        PriorityGate.MODEL_REAL_MARKET,
        PriorityGate.LONG_RUN,
    }
)


def _valid_sha256(value: str) -> bool:
    text = value.lower()
    return len(text) == 64 and all(ch in "0123456789abcdef" for ch in text)


@dataclass(frozen=True, slots=True)
class PriorityEvidence:
    gate: PriorityGate
    passed: bool
    evidence_class: EvidenceClass
    artifact_sha256: str
    source_sha256: str
    observed_at: datetime
    producer: str
    detail: str = ""

    def __post_init__(self) -> None:
        gate = self.gate if isinstance(self.gate, PriorityGate) else PriorityGate(self.gate)
        evidence_class = (
            self.evidence_class
            if isinstance(self.evidence_class, EvidenceClass)
            else EvidenceClass(self.evidence_class)
        )
        object.__setattr__(self, "gate", gate)
        object.__setattr__(self, "evidence_class", evidence_class)
        if not _valid_sha256(self.artifact_sha256) or not _valid_sha256(self.source_sha256):
            raise ValueError("artifact/source identity must be SHA-256")
        if self.observed_at.tzinfo is None or self.observed_at.utcoffset() is None:
            raise ValueError("observed_at must be timezone-aware")
        object.__setattr__(self, "observed_at", self.observed_at.astimezone(UTC))
        if not self.producer.strip():
            raise ValueError("producer is required")


@dataclass(frozen=True, slots=True)
class PriorityClosurePolicy:
    max_evidence_age: timedelta = timedelta(days=7)

    def __post_init__(self) -> None:
        if self.max_evidence_age <= timedelta(0):
            raise ValueError("max_evidence_age must be positive")


@dataclass(frozen=True, slots=True)
class PriorityGateDecision:
    gate: PriorityGate
    passed: bool
    artifact_sha256: str
    evidence_class: EvidenceClass | None
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Priority5ClosureReport:
    passed: bool
    expected_source_sha256: str
    decisions: tuple[PriorityGateDecision, ...]
    observed_at: datetime
    evidence_sha256: str
    reasons: tuple[str, ...]


def evaluate_priority5_closure(
    evidence: Iterable[PriorityEvidence],
    *,
    expected_source_sha256: str,
    now: datetime,
    policy: PriorityClosurePolicy | None = None,
) -> Priority5ClosureReport:
    if not _valid_sha256(expected_source_sha256):
        raise ValueError("expected_source_sha256 must be SHA-256")
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    current = now.astimezone(UTC)
    p = policy or PriorityClosurePolicy()
    rows = tuple(evidence)
    decisions: list[PriorityGateDecision] = []
    global_reasons: list[str] = []
    for gate in PriorityGate:
        candidates = sorted(
            (item for item in rows if item.gate is gate),
            key=lambda item: item.observed_at,
            reverse=True,
        )
        reasons: list[str] = []
        selected = candidates[0] if candidates else None
        if selected is None:
            reasons.append(f"{gate.value}:MISSING_EVIDENCE")
            decisions.append(PriorityGateDecision(gate, False, "", None, tuple(reasons)))
            global_reasons.extend(reasons)
            continue
        if selected.source_sha256 != expected_source_sha256:
            reasons.append(f"{gate.value}:SOURCE_IDENTITY_MISMATCH")
        age = current - selected.observed_at
        if age < timedelta(0) or age > p.max_evidence_age:
            reasons.append(f"{gate.value}:EVIDENCE_STALE_OR_FUTURE")
        if not selected.passed:
            reasons.append(f"{gate.value}:EVIDENCE_FAILED")
        if gate in MEASURED_ONLY_GATES and selected.evidence_class is not EvidenceClass.MEASURED:
            reasons.append(f"{gate.value}:MEASURED_EVIDENCE_REQUIRED")
        decision = PriorityGateDecision(
            gate=gate,
            passed=not reasons,
            artifact_sha256=selected.artifact_sha256,
            evidence_class=selected.evidence_class,
            reasons=tuple(reasons),
        )
        decisions.append(decision)
        global_reasons.extend(reasons)
    payload = {
        "expected_source_sha256": expected_source_sha256,
        "decisions": [asdict(item) for item in decisions],
        "observed_at": current.isoformat(),
    }
    return Priority5ClosureReport(
        passed=all(item.passed for item in decisions),
        expected_source_sha256=expected_source_sha256.lower(),
        decisions=tuple(decisions),
        observed_at=current,
        evidence_sha256=sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
        ).hexdigest(),
        reasons=tuple(global_reasons),
    )


def load_priority_evidence(path: str | Path) -> PriorityEvidence:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    payload["gate"] = PriorityGate(payload["gate"])
    payload["evidence_class"] = EvidenceClass(payload["evidence_class"])
    payload["observed_at"] = datetime.fromisoformat(payload["observed_at"])
    return PriorityEvidence(**payload)


def write_priority_evidence(path: str | Path, evidence: PriorityEvidence) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(evidence)
    payload["gate"] = evidence.gate.value
    payload["evidence_class"] = evidence.evidence_class.value
    payload["observed_at"] = evidence.observed_at.isoformat()
    target.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
