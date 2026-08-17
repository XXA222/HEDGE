"""Process lifecycle gate, distinct from individual exchange-order lifecycle."""

from __future__ import annotations

from enum import StrEnum
from threading import RLock


class ExecutionRuntimeState(StrEnum):
    RUNNING = "RUNNING"
    QUIESCING = "QUIESCING"
    CANCELING = "CANCELING"
    RECONCILING = "RECONCILING"
    STOPPED = "STOPPED"
    HALTED = "HALTED"


class ExecutionRuntimeBlockedError(PermissionError):
    pass


_TRANSITIONS = {
    ExecutionRuntimeState.RUNNING: {ExecutionRuntimeState.QUIESCING, ExecutionRuntimeState.HALTED},
    ExecutionRuntimeState.QUIESCING: {ExecutionRuntimeState.CANCELING, ExecutionRuntimeState.RECONCILING, ExecutionRuntimeState.HALTED},
    ExecutionRuntimeState.CANCELING: {ExecutionRuntimeState.RECONCILING, ExecutionRuntimeState.HALTED},
    ExecutionRuntimeState.RECONCILING: {ExecutionRuntimeState.RUNNING, ExecutionRuntimeState.STOPPED, ExecutionRuntimeState.HALTED},
    ExecutionRuntimeState.STOPPED: {ExecutionRuntimeState.RECONCILING},
    ExecutionRuntimeState.HALTED: {ExecutionRuntimeState.RECONCILING},
}


class ExecutionRuntimeLifecycle:
    """Single-writer process state; only RUNNING may originate risk-increasing orders."""

    def __init__(self, initial: ExecutionRuntimeState = ExecutionRuntimeState.RUNNING) -> None:
        if not isinstance(initial, ExecutionRuntimeState):
            raise TypeError("initial must be ExecutionRuntimeState")
        self._state = initial
        self._lock = RLock()

    @property
    def state(self) -> ExecutionRuntimeState:
        with self._lock:
            return self._state

    @property
    def allows_new_risk(self) -> bool:
        return self.state is ExecutionRuntimeState.RUNNING

    def transition(self, target: ExecutionRuntimeState) -> ExecutionRuntimeState:
        if not isinstance(target, ExecutionRuntimeState):
            raise TypeError("target must be ExecutionRuntimeState")
        with self._lock:
            if target is self._state:
                return self._state
            if target not in _TRANSITIONS[self._state]:
                raise ValueError(f"invalid execution runtime transition: {self._state}->{target}")
            self._state = target
            return self._state

    def assert_intent_allowed(self, *, reduces_risk: bool) -> None:
        if not isinstance(reduces_risk, bool):
            raise TypeError("reduces_risk must be bool")
        if not reduces_risk and not self.allows_new_risk:
            raise ExecutionRuntimeBlockedError(
                f"execution runtime {self.state.value} blocks new risk"
            )
