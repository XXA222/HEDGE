"""State-machine-level chaos and recovery qualification proof."""

from dataclasses import dataclass
from enum import StrEnum


class ChaosScenario(StrEnum):
    REST_TIMEOUT_UNKNOWN = "REST_TIMEOUT_UNKNOWN"
    USER_STREAM_GAP = "USER_STREAM_GAP"
    DATABASE_RESTART = "DATABASE_RESTART"
    PROCESS_RESTART_DUAL_LEG = "PROCESS_RESTART_DUAL_LEG"
    PARTIAL_FILL_CANCEL_REPLACE = "PARTIAL_FILL_CANCEL_REPLACE"
    MANUAL_EXTERNAL_ORDER = "MANUAL_EXTERNAL_ORDER"


@dataclass(frozen=True, slots=True)
class ChaosCaseResult:
    scenario: ChaosScenario
    no_duplicate_risk: bool
    reconciled: bool
    ledger_converged: bool
    passed: bool

    def __post_init__(self) -> None:
        if not isinstance(self.scenario, ChaosScenario):
            raise TypeError("scenario must be ChaosScenario")
        if not all(isinstance(getattr(self, name), bool) for name in ("no_duplicate_risk", "reconciled", "ledger_converged", "passed")):
            raise TypeError("chaos outcomes must be bool")
        if self.passed and not (self.no_duplicate_risk and self.reconciled and self.ledger_converged):
            raise ValueError("passed chaos case requires every invariant")


def qualify_chaos_recovery(results: tuple[ChaosCaseResult, ...]) -> tuple[bool, tuple[str, ...]]:
    by_scenario = {result.scenario: result for result in results}
    reasons = [f"MISSING:{scenario.value}" for scenario in ChaosScenario if scenario not in by_scenario]
    if len(by_scenario) != len(results):
        reasons.append("DUPLICATE_SCENARIO")
    reasons.extend(f"FAILED:{result.scenario.value}" for result in results if not result.passed)
    return not reasons, tuple(reasons)
