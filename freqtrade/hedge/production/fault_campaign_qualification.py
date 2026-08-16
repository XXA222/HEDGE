"""Coverage-aware qualification for the production fault catalog.

The canonical fault catalog remains the source of scenario truth.  This module adds
coverage groups, aggregate recovery SLOs and regression comparison so a small focused
campaign cannot be mistaken for full crash/UNKNOWN/DB/network qualification.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import ceil, isfinite


@dataclass(frozen=True, slots=True)
class FaultScenarioEvidence:
    scenario: str
    passed: bool
    duplicate_writes: int
    final_converged: bool
    new_risk_blocked_during_fault: bool
    recovery_seconds: float
    state_hash_match: bool
    outbox_drained: bool
    fencing_preserved: bool
    detail: str = ""

    def __post_init__(self) -> None:
        if not self.scenario.strip():
            raise ValueError("fault scenario is required")
        if self.duplicate_writes < 0:
            raise ValueError("duplicate_writes cannot be negative")
        if not isfinite(self.recovery_seconds) or self.recovery_seconds < 0:
            raise ValueError("recovery_seconds must be finite and nonnegative")

    @classmethod
    def from_fault_result(cls, result: object) -> "FaultScenarioEvidence":
        scenario = getattr(result, "scenario")
        value = str(getattr(scenario, "value", scenario))
        return cls(
            scenario=value,
            passed=bool(getattr(result, "passed")),
            duplicate_writes=int(getattr(result, "duplicate_writes")),
            final_converged=bool(getattr(result, "final_converged")),
            new_risk_blocked_during_fault=bool(getattr(result, "new_risk_blocked_during_fault")),
            recovery_seconds=float(getattr(result, "recovery_seconds")),
            state_hash_match=bool(getattr(result, "state_hash_match", True)),
            outbox_drained=bool(getattr(result, "outbox_drained", True)),
            fencing_preserved=bool(getattr(result, "fencing_preserved", True)),
            detail=str(getattr(result, "detail", "")),
        )


CORE_EXECUTION_SCENARIOS = (
    "HTTP_TIMEOUT_AFTER_ACCEPT",
    "QUERY_TIMEOUT",
    "PARTIAL_FILL",
    "PROCESS_CRASH_BEFORE_COMMIT",
    "PROCESS_CRASH_AFTER_COMMIT",
    "PROCESS_CRASH_AFTER_SUBMIT",
    "PROCESS_CRASH_AFTER_FILL",
    "CANCEL_FILL_RACE",
    "DUPLICATE_FILL",
)
NETWORK_STREAM_SCENARIOS = (
    "HTTP_429",
    "HTTP_5XX",
    "DNS_FAILURE",
    "TLS_FAILURE",
    "CONNECTION_RESET",
    "PARTIAL_HTTP_BODY",
    "WS_DISCONNECT",
    "WS_DUPLICATE",
    "WS_OUT_OF_ORDER",
    "WS_SEQUENCE_GAP",
    "USER_STREAM_LISTEN_KEY_EXPIRE",
    "REST_STALE_SNAPSHOT",
    "STALE_MARKET_DATA",
    "CLOCK_SKEW",
    "API_CLOCK_DRIFT",
)
DATABASE_RECOVERY_SCENARIOS = (
    "DB_CONNECTION_LOSS",
    "DB_DEADLOCK",
    "DB_PAUSE",
    "FENCING_TOKEN_STALE",
    "MULTI_WRITER_RACE",
    "OUTBOX_PUBLISHER_DOWN",
    "BACKUP_RESTORE_INTERRUPTED",
    "CHECKPOINT_CORRUPTION",
)
EXTERNAL_RISK_SCENARIOS = (
    "MANUAL_EXTERNAL_ORDER",
    "POSITION_DRIFT",
    "EXTERNAL_POSITION_CHANGE",
    "FUNDING_SPIKE",
    "MARK_PRICE_GAP",
    "EXCHANGE_FILTER_CHANGE",
    "LIQUIDATION_DATA_MISSING",
)
RESOURCE_MODEL_SCENARIOS = (
    "MODEL_TIMEOUT_STORM",
    "MODEL_NONFINITE",
    "DISK_FULL",
    "MEMORY_PRESSURE",
    "PROCESS_PAUSE",
)
FULL_QUALIFICATION_SCENARIOS = tuple(
    dict.fromkeys(
        CORE_EXECUTION_SCENARIOS
        + NETWORK_STREAM_SCENARIOS
        + DATABASE_RECOVERY_SCENARIOS
        + EXTERNAL_RISK_SCENARIOS
        + RESOURCE_MODEL_SCENARIOS
    )
)


@dataclass(frozen=True, slots=True)
class FaultQualificationPolicy:
    required_scenarios: tuple[str, ...] = FULL_QUALIFICATION_SCENARIOS
    maximum_recovery_seconds: float = 30.0
    maximum_p95_recovery_seconds: float = 15.0

    def __post_init__(self) -> None:
        if (
            not self.required_scenarios
            or len(set(self.required_scenarios)) != len(self.required_scenarios)
        ):
            raise ValueError("required_scenarios must be non-empty and unique")
        if self.maximum_recovery_seconds <= 0 or self.maximum_p95_recovery_seconds <= 0:
            raise ValueError("recovery ceilings must be positive")


@dataclass(frozen=True, slots=True)
class FaultQualificationReport:
    passed: bool
    qualification_scope: str
    required: int
    observed: int
    coverage_ratio: float
    catalog_required: int
    catalog_observed: int
    catalog_coverage_ratio: float
    full_catalog_qualified: bool
    p95_recovery_seconds: float
    maximum_recovery_seconds: float
    duplicate_write_free: bool
    all_converged: bool
    all_new_risk_fail_closed: bool
    all_state_hash_match: bool
    all_outbox_drained: bool
    all_fencing_preserved: bool
    group_coverage: tuple[tuple[str, int, int], ...]
    evidence_sha256: str
    reasons: tuple[str, ...]


def _percentile95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, ceil(0.95 * len(ordered)) - 1)
    return ordered[index]



def _fault_invariant_reasons(
    item: FaultScenarioEvidence,
    policy: FaultQualificationPolicy,
) -> list[str]:
    reasons: list[str] = []
    checks = (
        (not item.passed, f"FAILED:{item.scenario}"),
        (item.duplicate_writes != 0, f"DUPLICATE_WRITE:{item.scenario}"),
        (not item.final_converged, f"NOT_CONVERGED:{item.scenario}"),
        (
            not item.new_risk_blocked_during_fault,
            f"NEW_RISK_NOT_BLOCKED:{item.scenario}",
        ),
        (not item.state_hash_match, f"STATE_HASH_MISMATCH:{item.scenario}"),
        (not item.outbox_drained, f"OUTBOX_NOT_DRAINED:{item.scenario}"),
        (not item.fencing_preserved, f"FENCING_VIOLATION:{item.scenario}"),
        (
            item.recovery_seconds > policy.maximum_recovery_seconds,
            f"RECOVERY_MAX_SLA:{item.scenario}",
        ),
    )
    reasons.extend(reason for failed, reason in checks if failed)
    return reasons


def _fault_row_is_qualified(
    item: FaultScenarioEvidence,
    policy: FaultQualificationPolicy,
) -> bool:
    return not _fault_invariant_reasons(item, policy)

def evaluate_fault_qualification(
    rows: Iterable[FaultScenarioEvidence],
    *,
    policy: FaultQualificationPolicy | None = None,
) -> FaultQualificationReport:
    p = policy or FaultQualificationPolicy()
    materialized = tuple(rows)
    by: dict[str, FaultScenarioEvidence] = {}
    reasons: list[str] = []
    for item in materialized:
        if item.scenario in by:
            reasons.append(f"DUPLICATE_FAULT_EVIDENCE:{item.scenario}")
        by[item.scenario] = item
    required = tuple(p.required_scenarios)
    present = [name for name in required if name in by]
    missing = [name for name in required if name not in by]
    reasons.extend(f"MISSING:{name}" for name in missing)
    selected = [by[name] for name in present]
    for item in selected:
        reasons.extend(_fault_invariant_reasons(item, p))
    p95 = _percentile95([item.recovery_seconds for item in selected])
    if selected and p95 > p.maximum_p95_recovery_seconds:
        reasons.append("RECOVERY_P95_SLA")

    catalog_present = [name for name in FULL_QUALIFICATION_SCENARIOS if name in by]
    catalog_rows = [by[name] for name in catalog_present]
    catalog_p95 = _percentile95([item.recovery_seconds for item in catalog_rows])
    full_catalog_qualified = (
        len(catalog_present) == len(FULL_QUALIFICATION_SCENARIOS)
        and all(_fault_row_is_qualified(item, p) for item in catalog_rows)
        and catalog_p95 <= p.maximum_p95_recovery_seconds
    )
    qualification_scope = (
        "FULL"
        if set(required) == set(FULL_QUALIFICATION_SCENARIOS)
        else "FOCUSED_OR_CUSTOM"
    )

    groups = (
        ("core_execution", CORE_EXECUTION_SCENARIOS),
        ("network_stream", NETWORK_STREAM_SCENARIOS),
        ("database_recovery", DATABASE_RECOVERY_SCENARIOS),
        ("external_risk", EXTERNAL_RISK_SCENARIOS),
        ("resource_model", RESOURCE_MODEL_SCENARIOS),
    )
    coverage_rows = tuple(
        (name, sum(item in by for item in members), len(members))
        for name, members in groups
    )
    payload = {
        "policy": asdict(p),
        "rows": [asdict(item) for item in sorted(materialized, key=lambda x: x.scenario)],
        "group_coverage": coverage_rows,
    }
    return FaultQualificationReport(
        passed=not reasons,
        qualification_scope=qualification_scope,
        required=len(required),
        observed=len(present),
        coverage_ratio=len(present) / len(required),
        catalog_required=len(FULL_QUALIFICATION_SCENARIOS),
        catalog_observed=len(catalog_present),
        catalog_coverage_ratio=len(catalog_present) / len(FULL_QUALIFICATION_SCENARIOS),
        full_catalog_qualified=full_catalog_qualified,
        p95_recovery_seconds=p95,
        maximum_recovery_seconds=max((item.recovery_seconds for item in selected), default=0.0),
        duplicate_write_free=all(item.duplicate_writes == 0 for item in selected),
        all_converged=all(item.final_converged for item in selected),
        all_new_risk_fail_closed=all(item.new_risk_blocked_during_fault for item in selected),
        all_state_hash_match=all(item.state_hash_match for item in selected),
        all_outbox_drained=all(item.outbox_drained for item in selected),
        all_fencing_preserved=all(item.fencing_preserved for item in selected),
        group_coverage=coverage_rows,
        evidence_sha256=sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        reasons=tuple(dict.fromkeys(reasons)),
    )


def focused_runtime_campaign_report() -> FaultQualificationReport:
    """Run the canonical focused campaign, but label it only as focused coverage."""
    from .runtime_fault_injection import (
        FOCUSED_RUNTIME_SCENARIOS,
        run_focused_runtime_fault_campaign,
    )

    raw = run_focused_runtime_fault_campaign()
    required = tuple(str(getattr(item, "value", item)) for item in FOCUSED_RUNTIME_SCENARIOS)
    evidence = tuple(FaultScenarioEvidence.from_fault_result(item) for item in raw.results)
    return evaluate_fault_qualification(
        evidence,
        policy=FaultQualificationPolicy(required_scenarios=required),
    )


def compare_fault_campaigns(
    previous: FaultQualificationReport,
    current: FaultQualificationReport,
) -> tuple[bool, tuple[str, ...]]:
    reasons: list[str] = []
    if current.coverage_ratio < previous.coverage_ratio:
        reasons.append("FAULT_COVERAGE_REGRESSED")
    if (
        previous.p95_recovery_seconds > 0
        and current.p95_recovery_seconds > previous.p95_recovery_seconds * 1.25
    ):
        reasons.append("FAULT_RECOVERY_P95_REGRESSED")
    if previous.passed and not current.passed:
        reasons.append("FAULT_QUALIFICATION_REGRESSED")
    return not reasons, tuple(reasons)
