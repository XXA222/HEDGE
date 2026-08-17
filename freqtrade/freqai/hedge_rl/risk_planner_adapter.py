"""Planner-facing contract for target risk-level Hedge RL.

This module deliberately emits target exposure intent only.  It does not create orders,
client IDs, exchange requests, or bypass the Hedge risk engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
import math
from typing import cast

from .risk_levels import HedgeRiskLevelAction, RiskLevelMapper, RiskLevelProfile
from .risk_action_mask import RiskLevelActionMasker, RiskLevelJointActionMask, RiskLevelMaskContext
from .risk_portfolio import RiskAccountState


@dataclass(frozen=True, slots=True)
class HedgeRiskLevelPlannerSignal:
    target_equity: float
    long_level: int
    short_level: int
    long_margin_fraction: float
    short_margin_fraction: float
    long_target_notional: float
    short_target_notional: float
    reserve_margin_fraction: float
    combined_margin_fraction: float
    target_net_margin_fraction: float
    action_signature: str
    long_increase_allowed: bool
    short_increase_allowed: bool
    target_semantics: str
    allow_new_risk: bool
    reason: str
    requested_joint_id: int = 0
    action_mask_allowed_count: int = 25

    def strategy_columns(self) -> dict[str, float | int | bool | str]:
        return {
            "hedge_rl_long_level": self.long_level,
            "hedge_rl_short_level": self.short_level,
            "hedge_rl_long_margin_fraction": self.long_margin_fraction,
            "hedge_rl_short_margin_fraction": self.short_margin_fraction,
            "hedge_rl_long_target_notional": self.long_target_notional,
            "hedge_rl_short_target_notional": self.short_target_notional,
            "hedge_rl_reserve_margin_fraction": self.reserve_margin_fraction,
            "hedge_rl_combined_margin_fraction": self.combined_margin_fraction,
            "hedge_rl_target_net_margin_fraction": self.target_net_margin_fraction,
            "hedge_rl_action_signature": self.action_signature,
            "hedge_rl_long_increase_allowed": self.long_increase_allowed,
            "hedge_rl_short_increase_allowed": self.short_increase_allowed,
            "hedge_rl_target_semantics": self.target_semantics,
            "hedge_allow_new_risk": self.allow_new_risk,
            "hedge_rl_reason": self.reason,
            "hedge_rl_requested_joint_id": self.requested_joint_id,
            "hedge_rl_action_mask_allowed_count": self.action_mask_allowed_count,
        }


class HedgeRiskLevelPlannerAdapter:
    def __init__(self, profile: RiskLevelProfile) -> None:
        self.profile = profile
        self.mapper = RiskLevelMapper(profile)
        self.masker = RiskLevelActionMasker(profile)
        self._action_signature = profile.signature

    def _signal(
        self,
        action: HedgeRiskLevelAction,
        *,
        equity: float,
        projection_fresh: bool,
        current_long_level: int | None = None,
        current_short_level: int | None = None,
        current_long_notional: float | None = None,
        current_short_notional: float | None = None,
        requested_joint_id: int | None = None,
        mask_allowed_count: int = 25,
        masked_reason: str | None = None,
    ) -> HedgeRiskLevelPlannerSignal:
        if not math.isfinite(equity) or equity <= 0:
            raise ValueError("equity must be finite and positive")
        if not projection_fresh:
            action = HedgeRiskLevelAction.from_value((0, 0))
        target = self.mapper.map(action, equity=equity)
        state_known = all(
            value is not None
            for value in (
                current_long_level,
                current_short_level,
                current_long_notional,
                current_short_notional,
            )
        )
        long_increase = False
        short_increase = False
        if projection_fresh and state_known:
            long_increase = int(action.long_level) > cast(int, current_long_level)
            short_increase = int(action.short_level) > cast(int, current_short_level)
        long_notional = target.long_target_notional
        short_notional = target.short_target_notional
        if state_known:
            if not long_increase:
                long_notional = min(float(cast(float, current_long_notional)), long_notional)
            if not short_increase:
                short_notional = min(float(cast(float, current_short_notional)), short_notional)
        allow = projection_fresh and (long_increase or short_increase)
        return HedgeRiskLevelPlannerSignal(
            target_equity=equity,
            long_level=int(action.long_level),
            short_level=int(action.short_level),
            long_margin_fraction=target.long_margin_fraction,
            short_margin_fraction=target.short_margin_fraction,
            long_target_notional=long_notional,
            short_target_notional=short_notional,
            reserve_margin_fraction=target.reserve_margin_fraction,
            combined_margin_fraction=target.combined_margin_fraction,
            target_net_margin_fraction=(target.long_margin_fraction - target.short_margin_fraction),
            action_signature=self._action_signature,
            long_increase_allowed=long_increase,
            short_increase_allowed=short_increase,
            target_semantics="RISK_CAP_NO_SAME_LEVEL_SCALE_IN",
            allow_new_risk=allow,
            reason=(
                f"HEDGE_RL_RISK_LEVEL:L{int(action.long_level)}:S{int(action.short_level)}"
                + (f":MASKED:{masked_reason}" if masked_reason else "")
            ),
            requested_joint_id=action.joint_id if requested_joint_id is None else requested_joint_id,
            action_mask_allowed_count=mask_allowed_count,
        )

    def from_action(
        self,
        action: HedgeRiskLevelAction,
        *,
        equity: float,
        projection_fresh: bool = True,
    ) -> HedgeRiskLevelPlannerSignal:
        """Stateless advisory mapping.

        This method preserves the V1/V2 analysis API.  Because current position facts are
        unknown it never grants permission to add live risk.  Live/dry-run integration must
        use :meth:`from_account_action`.
        """

        return self._signal(action, equity=equity, projection_fresh=projection_fresh)

    def from_account_action(
        self,
        action: HedgeRiskLevelAction,
        *,
        account: RiskAccountState,
        mark: float,
        projection_fresh: bool = True,
        unresolved_unknown: bool = False,
        reconciliation_required: bool = False,
        model_degraded: bool = False,
        reduce_only: bool = False,
        margin_stressed: bool = False,
        max_upward_levels: int = 1,
    ) -> HedgeRiskLevelPlannerSignal:
        """Canonical state-aware target mapping for dry-run/live planners.

        Risk can increase only when the policy explicitly raises that leg's level.
        Same-level or lower-level actions can only keep/reduce the current notional,
        preventing equity/price drift from silently averaging down.
        """

        requested = HedgeRiskLevelAction.from_value(action)
        mask = self.action_mask(
            account=account,
            projection_fresh=projection_fresh,
            unresolved_unknown=unresolved_unknown,
            reconciliation_required=reconciliation_required,
            model_degraded=model_degraded,
            reduce_only=reduce_only,
            margin_stressed=margin_stressed,
            max_upward_levels=max_upward_levels,
        )
        selected = requested
        masked_reason = mask.reason_for(requested)
        if not mask.permits(requested):
            current = HedgeRiskLevelAction.from_value((account.long_level, account.short_level))
            selected = current if mask.permits(current) else HedgeRiskLevelAction.from_value((0, 0))
        return self._signal(
            selected,
            equity=account.equity,
            projection_fresh=projection_fresh,
            current_long_level=account.long_level,
            current_short_level=account.short_level,
            current_long_notional=account.long.notional(mark),
            current_short_notional=account.short.notional(mark),
            requested_joint_id=requested.joint_id,
            mask_allowed_count=len(mask.allowed_joint_ids),
            masked_reason=masked_reason,
        )

    def action_mask(
        self,
        *,
        account: RiskAccountState,
        projection_fresh: bool = True,
        unresolved_unknown: bool = False,
        reconciliation_required: bool = False,
        model_degraded: bool = False,
        reduce_only: bool = False,
        margin_stressed: bool = False,
        max_upward_levels: int = 1,
    ) -> RiskLevelJointActionMask:
        return self.masker.build(RiskLevelMaskContext(
            current_action=HedgeRiskLevelAction.from_value((account.long_level, account.short_level)),
            projection_fresh=projection_fresh,
            unresolved_unknown=unresolved_unknown,
            reconciliation_required=reconciliation_required,
            model_degraded=model_degraded,
            reduce_only=reduce_only,
            margin_stressed=margin_stressed,
            max_upward_levels=max_upward_levels,
        ))

    def to_signal_snapshot(
        self,
        signal: HedgeRiskLevelPlannerSignal,
        *,
        pair: str,
        timeframe: str,
        candle_close_time: datetime,
        feature_timestamp: datetime,
        model_version: str,
    ):
        """Adapt to the existing canonical SignalSnapshot without order coupling."""

        from freqtrade.hedge.integration.signal_provider import SignalSnapshot

        heavy = max(self.profile.position_levels[-1], 1e-12)
        long_score = min(1.0, signal.long_margin_fraction / heavy)
        short_score = min(1.0, signal.short_margin_fraction / heavy)
        long_notional = signal.long_target_notional
        short_notional = signal.short_target_notional
        # ``target_net_ratio`` is a net-notional/equity ratio.  Dividing by gross
        # notional changes the unit and inflates the target downstream.
        target_net_ratio = (long_notional - short_notional) / signal.target_equity
        return SignalSnapshot(
            symbol=pair,
            timeframe=timeframe,
            candle_close_time=candle_close_time,
            feature_timestamp=feature_timestamp,
            long_score=Decimal(str(long_score)),
            short_score=Decimal(str(short_score)),
            target_net=None,
            target_net_ratio=Decimal(str(target_net_ratio)),
            target_long_notional=Decimal(str(long_notional)),
            target_short_notional=Decimal(str(short_notional)),
            model_version=model_version,
            reason=signal.reason,
            confidence=Decimal(1),
            risk_scale=Decimal(1),
            long_exposure_scale=Decimal(str(long_score)),
            short_exposure_scale=Decimal(str(short_score)),
            allow_new_risk=signal.allow_new_risk,
            regime="HEDGE_RL_RISK_LEVEL",
            strategy_reason=signal.reason,
        )
