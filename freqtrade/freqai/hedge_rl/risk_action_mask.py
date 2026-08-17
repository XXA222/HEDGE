"""Joint 25-state feasibility mask for Hedge Risk-Level RL.

SB3's ``MultiDiscrete([5, 5])`` mask is factorised (ten bits) and therefore cannot
express coupled gross-margin or transition constraints.  This module is the canonical
25-state safety view used by planners; the live Hedge risk engine still revalidates
every executable order.
"""

from __future__ import annotations

from dataclasses import dataclass

from .risk_levels import HedgeRiskLevelAction, RiskActionTopology, RiskLevelProfile


@dataclass(frozen=True, slots=True)
class RiskLevelMaskContext:
    current_action: HedgeRiskLevelAction
    projection_fresh: bool = True
    unresolved_unknown: bool = False
    reconciliation_required: bool = False
    model_degraded: bool = False
    reduce_only: bool = False
    margin_stressed: bool = False
    max_upward_levels: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.current_action, HedgeRiskLevelAction):
            raise TypeError("current_action must be HedgeRiskLevelAction")
        for name in (
            "projection_fresh", "unresolved_unknown", "reconciliation_required",
            "model_degraded", "reduce_only", "margin_stressed",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool")
        if isinstance(self.max_upward_levels, bool) or not isinstance(self.max_upward_levels, int):
            raise TypeError("max_upward_levels must be int")
        if self.max_upward_levels < 0:
            raise ValueError("max_upward_levels must be nonnegative")


@dataclass(frozen=True, slots=True)
class RiskLevelJointActionMask:
    allowed: tuple[bool, ...]
    reasons: tuple[str | None, ...]

    def __post_init__(self) -> None:
        if len(self.allowed) != 25 or len(self.reasons) != 25:
            raise ValueError("joint action mask must contain exactly 25 states")
        if not all(isinstance(value, bool) for value in self.allowed):
            raise TypeError("joint action mask values must be bool")
        if not self.allowed[0]:
            raise ValueError("flat action must always remain available")

    def permits(self, action: HedgeRiskLevelAction) -> bool:
        return self.allowed[action.joint_id]

    def reason_for(self, action: HedgeRiskLevelAction) -> str | None:
        return self.reasons[action.joint_id]

    @property
    def allowed_joint_ids(self) -> tuple[int, ...]:
        return tuple(index for index, valid in enumerate(self.allowed) if valid)


class RiskLevelActionMasker:
    def __init__(self, profile: RiskLevelProfile) -> None:
        self.profile = profile
        self.topology = RiskActionTopology(profile)

    def build(self, context: RiskLevelMaskContext) -> RiskLevelJointActionMask:
        allowed: list[bool] = []
        reasons: list[str | None] = []
        for joint_id in range(25):
            candidate = HedgeRiskLevelAction.from_joint_id(joint_id)
            reason = self._reject_reason(context, candidate)
            allowed.append(reason is None)
            reasons.append(reason)
        # FLAT is intentionally available even when account state is stale or out of
        # profile: it is the only universally conservative target.
        allowed[0] = True
        reasons[0] = None
        return RiskLevelJointActionMask(tuple(allowed), tuple(reasons))

    def _reject_reason(self, context: RiskLevelMaskContext, candidate: HedgeRiskLevelAction) -> str | None:
        if not context.projection_fresh:
            return "STALE_PROJECTION"
        transition = self.topology.transition(context.current_action, candidate)
        if context.unresolved_unknown:
            if int(candidate.long_level) > int(context.current_action.long_level) or int(candidate.short_level) > int(context.current_action.short_level):
                return "UNKNOWN_ORDER_NO_INCREASE"
        if context.reconciliation_required:
            if int(candidate.long_level) > int(context.current_action.long_level) or int(candidate.short_level) > int(context.current_action.short_level):
                return "RECONCILIATION_REQUIRED"
        if context.model_degraded or context.reduce_only:
            if int(candidate.long_level) > int(context.current_action.long_level) or int(candidate.short_level) > int(context.current_action.short_level):
                return "REDUCE_ONLY"
        if context.margin_stressed and transition.increases_risk:
            return "MARGIN_STRESSED"
        upward_levels = max(
            0,
            int(candidate.long_level) - int(context.current_action.long_level),
            int(candidate.short_level) - int(context.current_action.short_level),
        )
        if upward_levels > context.max_upward_levels:
            return "UPWARD_JUMP_LIMIT"
        return None
