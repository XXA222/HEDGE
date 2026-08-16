"""Pure simulation risk gate.

The gate is a deterministic transformation of planning output. Keeping it in
the simulation layer lets replay depend on a domain operation rather than on
the integration composition root.
"""

from __future__ import annotations

from dataclasses import replace

from freqtrade.hedge.planning.context import PlanningResult, PositionSide, StrategyLegState


def apply_new_risk_gate(
    planning: PlanningResult,
    *,
    enabled: bool,
    current_long_state: StrategyLegState,
    current_short_state: StrategyLegState,
) -> tuple[PlanningResult, int]:
    """Remove risk-increasing submissions while preserving reduce-only actions."""

    if not isinstance(planning, PlanningResult):
        raise TypeError("planning must be PlanningResult")
    if not isinstance(enabled, bool):
        raise TypeError("enabled must be boolean")
    if not isinstance(current_long_state, StrategyLegState):
        raise TypeError("current_long_state must be StrategyLegState")
    if not isinstance(current_short_state, StrategyLegState):
        raise TypeError("current_short_state must be StrategyLegState")
    if enabled:
        return planning, 0
    blocked = tuple(item for item in planning.submit_orders if not item.reduce_only)
    if not blocked:
        return planning, 0
    blocked_sides = {item.position_side for item in blocked}
    return (
        replace(
            planning,
            submit_orders=tuple(item for item in planning.submit_orders if item.reduce_only),
            long_state=(
                current_long_state if PositionSide.LONG in blocked_sides else planning.long_state
            ),
            short_state=(
                current_short_state if PositionSide.SHORT in blocked_sides else planning.short_state
            ),
            diagnostics=planning.diagnostics + (f"NEW_RISK_STOPPED:{len(blocked)}",),
        ),
        len(blocked),
    )
