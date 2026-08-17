from __future__ import annotations

import pytest

from freqtrade.hedge.execution.runtime_lifecycle import (
    ExecutionRuntimeBlockedError, ExecutionRuntimeLifecycle, ExecutionRuntimeState,
)


def test_only_running_allows_new_risk_but_reduce_is_available_during_shutdown() -> None:
    lifecycle = ExecutionRuntimeLifecycle()
    lifecycle.transition(ExecutionRuntimeState.QUIESCING)
    with pytest.raises(ExecutionRuntimeBlockedError, match="QUIESCING"):
        lifecycle.assert_intent_allowed(reduces_risk=False)
    lifecycle.assert_intent_allowed(reduces_risk=True)


def test_runtime_must_reconcile_before_restart() -> None:
    lifecycle = ExecutionRuntimeLifecycle()
    lifecycle.transition(ExecutionRuntimeState.HALTED)
    with pytest.raises(ValueError, match="HALTED"):
        lifecycle.transition(ExecutionRuntimeState.RUNNING)
    lifecycle.transition(ExecutionRuntimeState.RECONCILING)
    assert lifecycle.transition(ExecutionRuntimeState.RUNNING) is ExecutionRuntimeState.RUNNING
