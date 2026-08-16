"""Account-level reward model for target risk-level Hedge RL.

Design rules:

* account equity log-return is the only primary profit term;
* fees, slippage, funding, realized and unrealized PnL are not booked twice;
* auxiliary shaping prices risk quality and position-management behavior only;
* delayed probe/scale credit is side-specific so one Hedge leg cannot hide the other;
* large-position losses are penalized asymmetrically while large-position wins receive
  only a very small extra shaping bonus;
* drawdown, downside-risk memory and reserve pressure are continuous/convex rather than
  rare cliff penalties.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from .risk_portfolio import LegSide, RiskAccountState, RiskPortfolioTransition


@dataclass(frozen=True, slots=True)
class RiskRewardConfig:
    equity_log_return_scale: float = 100.0
    drawdown_weight: float = 2.0
    downside_exposure_weight: float = 0.60
    downside_ewma_weight: float = 0.02
    downside_ewma_alpha: float = 0.05
    uncertainty_exposure_weight: float = 0.12
    leverage_exposure_weight: float = 0.01
    preferred_reserve_margin_fraction: float = 0.20
    reserve_pressure_weight: float = 0.02
    minimum_liquidation_buffer_fraction: float = 0.10
    liquidation_buffer_weight: float = 4.0
    wrong_level_loss_weight: float = 0.15
    position_success_bonus_weight: float = 0.02
    loss_level_multipliers: tuple[float, float, float, float, float] = (0.0, 1.0, 1.10, 1.30, 1.65)
    win_level_multipliers: tuple[float, float, float, float, float] = (0.0, 1.0, 1.03, 1.06, 1.10)
    adverse_scale_in_weight: float = 0.25
    upward_jump_weight: float = 0.01
    level_churn_weight: float = 0.0025
    turnover_shaping_weight: float = 0.005
    repeated_probe_weight: float = 0.02
    risk_reduction_bonus_weight: float = 0.02
    profit_lock_bonus_weight: float = 0.015
    hedge_efficiency_weight: float = 0.01
    hedge_waste_weight: float = 0.005
    delayed_scale_bonus_weight: float = 0.01
    delayed_probe_bonus_weight: float = 0.005
    scale_confirmation_steps: int = 3
    probe_confirmation_steps: int = 3
    probe_drawdown_limit: float = 0.01
    max_positive_shaping: float = 0.25
    reward_clip: float = 10.0
    soft_clip: bool = False

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if name in {"loss_level_multipliers", "win_level_multipliers", "soft_clip"}:
                continue
            if name.endswith("_steps"):
                if int(value) < 1:
                    raise ValueError(f"{name} must be positive")
                continue
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.reward_clip <= 0:
            raise ValueError("reward_clip must be positive")
        if not 0 < self.downside_ewma_alpha <= 1:
            raise ValueError("downside_ewma_alpha must be within (0, 1]")
        if not 0 <= self.minimum_liquidation_buffer_fraction < 1:
            raise ValueError("minimum_liquidation_buffer_fraction must be within [0, 1)")
        if not 0 <= self.preferred_reserve_margin_fraction < 1:
            raise ValueError("preferred_reserve_margin_fraction must be within [0, 1)")
        if self.preferred_reserve_margin_fraction < self.minimum_liquidation_buffer_fraction:
            raise ValueError("preferred reserve cannot be below the hard minimum reserve")
        self._validate_level_curve("loss_level_multipliers", self.loss_level_multipliers)
        self._validate_level_curve("win_level_multipliers", self.win_level_multipliers)

    @staticmethod
    def _validate_level_curve(name: str, values: tuple[float, ...]) -> None:
        if len(values) != 5:
            raise ValueError(f"{name} must contain exactly five values")
        if any(not math.isfinite(float(value)) or value < 0 for value in values):
            raise ValueError(f"{name} must contain finite non-negative values")
        if values[0] != 0.0:
            raise ValueError(f"{name}[0] must be 0")
        if tuple(sorted(values)) != tuple(values):
            raise ValueError(f"{name} must be monotonic non-decreasing")

    @property
    def signature(self) -> str:
        """Stable reward-contract identity for experiment/checkpoint provenance."""

        raw = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(raw.encode("ascii")).hexdigest()[:16]

    @classmethod
    def from_freqtrade_config(cls, config: Mapping[str, Any]) -> RiskRewardConfig:
        freqai = config.get("freqai", {}) if isinstance(config, Mapping) else {}
        if not isinstance(freqai, Mapping):
            return cls()
        rl = freqai.get("rl_config", {})
        if not isinstance(rl, Mapping):
            rl = {}
        hedge = rl.get("hedge_reward", {})
        if not isinstance(hedge, Mapping):
            hedge = {}
        valid = set(cls.__dataclass_fields__)
        values = {key: value for key, value in hedge.items() if key in valid}
        for name in ("loss_level_multipliers", "win_level_multipliers"):
            if name in values:
                values[name] = tuple(float(item) for item in values[name])
        return cls(**values)


@dataclass(frozen=True, slots=True)
class PendingOutcome:
    kind: str
    side: LegSide
    created_step: int
    due_step: int
    baseline_equity: float
    baseline_drawdown: float
    baseline_leg_net_pnl: float
    baseline_level: int
    target_level: int


@dataclass(frozen=True, slots=True)
class RiskRewardBreakdown:
    equity_log_return: float
    drawdown_penalty: float
    downside_exposure_penalty: float
    downside_ewma_penalty: float
    uncertainty_exposure_penalty: float
    leverage_exposure_penalty: float
    reserve_pressure_penalty: float
    liquidation_buffer_penalty: float
    wrong_level_loss_penalty: float
    position_success_bonus: float
    adverse_scale_in_penalty: float
    upward_jump_penalty: float
    level_churn_penalty: float
    turnover_shaping_penalty: float
    repeated_probe_penalty: float
    risk_reduction_bonus: float
    profit_lock_bonus: float
    hedge_efficiency_bonus: float
    hedge_waste_penalty: float
    delayed_scale_bonus: float
    delayed_probe_bonus: float
    positive_shaping_raw: float
    positive_shaping_applied: float
    accounting_cost_ratio: float
    downside_semideviation: float
    unclipped_reward: float
    reward: float
    consecutive_failed_probes: int
    consecutive_failed_probes_long: int
    consecutive_failed_probes_short: int

    def to_dict(self) -> dict[str, float | int]:
        # Manual conversion avoids dataclasses.asdict() deep-copy machinery in the hot loop.
        return {
            "equity_log_return": self.equity_log_return,
            "drawdown_penalty": self.drawdown_penalty,
            "downside_exposure_penalty": self.downside_exposure_penalty,
            "downside_ewma_penalty": self.downside_ewma_penalty,
            "uncertainty_exposure_penalty": self.uncertainty_exposure_penalty,
            "leverage_exposure_penalty": self.leverage_exposure_penalty,
            "reserve_pressure_penalty": self.reserve_pressure_penalty,
            "liquidation_buffer_penalty": self.liquidation_buffer_penalty,
            "wrong_level_loss_penalty": self.wrong_level_loss_penalty,
            "position_success_bonus": self.position_success_bonus,
            "adverse_scale_in_penalty": self.adverse_scale_in_penalty,
            "upward_jump_penalty": self.upward_jump_penalty,
            "level_churn_penalty": self.level_churn_penalty,
            "turnover_shaping_penalty": self.turnover_shaping_penalty,
            "repeated_probe_penalty": self.repeated_probe_penalty,
            "risk_reduction_bonus": self.risk_reduction_bonus,
            "profit_lock_bonus": self.profit_lock_bonus,
            "hedge_efficiency_bonus": self.hedge_efficiency_bonus,
            "hedge_waste_penalty": self.hedge_waste_penalty,
            "delayed_scale_bonus": self.delayed_scale_bonus,
            "delayed_probe_bonus": self.delayed_probe_bonus,
            "positive_shaping_raw": self.positive_shaping_raw,
            "positive_shaping_applied": self.positive_shaping_applied,
            "accounting_cost_ratio": self.accounting_cost_ratio,
            "downside_semideviation": self.downside_semideviation,
            "unclipped_reward": self.unclipped_reward,
            "reward": self.reward,
            "consecutive_failed_probes": self.consecutive_failed_probes,
            "consecutive_failed_probes_long": self.consecutive_failed_probes_long,
            "consecutive_failed_probes_short": self.consecutive_failed_probes_short,
        }


class HedgeRiskRewardModel:
    def __init__(self, config: RiskRewardConfig, *, max_pending_outcomes: int = 64) -> None:
        self.config = config
        self.max_pending_outcomes = int(max_pending_outcomes)
        if self.max_pending_outcomes < 4:
            raise ValueError("max_pending_outcomes must be at least 4")
        self.consecutive_failed_probes_long = 0
        self.consecutive_failed_probes_short = 0
        self.downside_semivariance_ewma = 0.0
        self._pending: list[PendingOutcome] = []

    @property
    def consecutive_failed_probes(self) -> int:
        return max(self.consecutive_failed_probes_long, self.consecutive_failed_probes_short)

    @property
    def pending_outcome_count(self) -> int:
        return len(self._pending)

    @property
    def downside_semideviation(self) -> float:
        return math.sqrt(max(0.0, self.downside_semivariance_ewma))

    def reset(self) -> None:
        self.consecutive_failed_probes_long = 0
        self.consecutive_failed_probes_short = 0
        self.downside_semivariance_ewma = 0.0
        self._pending.clear()

    def _append_pending(self, outcome: PendingOutcome) -> None:
        if len(self._pending) >= self.max_pending_outcomes:
            raise RuntimeError(
                "pending Hedge RL reward outcomes exceeded the configured memory cap"
            )
        self._pending.append(outcome)

    @staticmethod
    def _safe_log_return(previous_equity: float, equity: float) -> float:
        if previous_equity <= 0 or equity <= 0:
            return -1.0
        return math.log(equity / previous_equity)

    @staticmethod
    def _leg(account: RiskAccountState, side: LegSide):
        return account.long if side is LegSide.LONG else account.short

    def _failed_probe_count(self, side: LegSide) -> int:
        return (
            self.consecutive_failed_probes_long
            if side is LegSide.LONG
            else self.consecutive_failed_probes_short
        )

    def _set_failed_probe_count(self, side: LegSide, value: int) -> None:
        if side is LegSide.LONG:
            self.consecutive_failed_probes_long = int(value)
        else:
            self.consecutive_failed_probes_short = int(value)

    def _schedule_side_outcome(
        self,
        *,
        kind: str,
        side: LegSide,
        account: RiskAccountState,
        mark: float,
        baseline_equity: float,
        baseline_drawdown: float,
        baseline_leg_net_pnl: float,
        baseline_level: int,
        target_level: int,
        due_steps: int,
    ) -> None:
        self._append_pending(
            PendingOutcome(
                kind=kind,
                side=side,
                created_step=account.step,
                due_step=account.step + due_steps,
                baseline_equity=baseline_equity,
                baseline_drawdown=baseline_drawdown,
                baseline_leg_net_pnl=baseline_leg_net_pnl,
                baseline_level=baseline_level,
                target_level=target_level,
            )
        )

    def _schedule_outcomes(
        self,
        transition: RiskPortfolioTransition,
        account: RiskAccountState,
        mark: float,
    ) -> None:
        side_facts = (
            (
                LegSide.LONG,
                transition.previous_long_level,
                transition.long_level,
                transition.previous_long_unrealized,
                transition.previous_long_net_pnl,
            ),
            (
                LegSide.SHORT,
                transition.previous_short_level,
                transition.short_level,
                transition.previous_short_unrealized,
                transition.previous_short_net_pnl,
            ),
        )
        for side, previous_level, level, previous_unrealized, previous_net_pnl in side_facts:
            if level > previous_level and previous_level > 0 and previous_unrealized > 0:
                self._schedule_side_outcome(
                    kind="scale",
                    side=side,
                    account=account,
                    mark=mark,
                    baseline_equity=transition.previous_equity,
                    baseline_drawdown=transition.previous_drawdown,
                    baseline_leg_net_pnl=previous_net_pnl,
                    baseline_level=previous_level,
                    target_level=level,
                    due_steps=self.config.scale_confirmation_steps,
                )
            if previous_level == 0 and level == 1:
                self._schedule_side_outcome(
                    kind="probe",
                    side=side,
                    account=account,
                    mark=mark,
                    baseline_equity=transition.previous_equity,
                    baseline_drawdown=transition.previous_drawdown,
                    baseline_leg_net_pnl=previous_net_pnl,
                    baseline_level=0,
                    target_level=1,
                    due_steps=self.config.probe_confirmation_steps,
                )

    def _resolve_outcomes(
        self,
        account: RiskAccountState,
        *,
        mark: float,
    ) -> tuple[float, float, float]:
        scale_bonus = 0.0
        probe_bonus = 0.0
        probe_failure_penalty = 0.0
        write_index = 0
        pending = self._pending
        pending_count = len(pending)
        multipliers = (0.0, 1.0, 1.2, 1.5, 2.0)
        for read_index in range(pending_count):
            event = pending[read_index]
            current_level = (
                account.long_level if event.side is LegSide.LONG else account.short_level
            )
            finished_early = current_level < event.target_level
            if account.step < event.due_step and not finished_early:
                if write_index != read_index:
                    pending[write_index] = event
                write_index += 1
                continue
            if event.baseline_equity <= 0:
                continue
            side_delta = self._leg(account, event.side).net_pnl(mark) - event.baseline_leg_net_pnl
            side_return_pct = side_delta / event.baseline_equity * 100.0
            drawdown_increase = max(0.0, account.drawdown() - event.baseline_drawdown)
            if event.kind == "scale":
                if side_return_pct > 0:
                    scale_bonus += (
                        min(1.0, side_return_pct) * self.config.delayed_scale_bonus_weight
                    )
            elif event.kind == "probe":
                if side_return_pct > 0 and drawdown_increase <= self.config.probe_drawdown_limit:
                    probe_bonus += (
                        min(1.0, side_return_pct) * self.config.delayed_probe_bonus_weight
                    )
                    self._set_failed_probe_count(event.side, 0)
                else:
                    count = self._failed_probe_count(event.side) + 1
                    self._set_failed_probe_count(event.side, count)
                    index = min(count, len(multipliers) - 1)
                    probe_failure_penalty += self.config.repeated_probe_weight * multipliers[index]
        if write_index < pending_count:
            del pending[write_index:]
        return scale_bonus, probe_bonus, probe_failure_penalty

    def _update_downside_risk(self, base_log_return: float) -> tuple[float, float]:
        previous_semideviation = self.downside_semideviation
        downside_pct = max(0.0, -base_log_return * 100.0)
        alpha = self.config.downside_ewma_alpha
        self.downside_semivariance_ewma = (
            1.0 - alpha
        ) * self.downside_semivariance_ewma + alpha * downside_pct**2
        semideviation = self.downside_semideviation
        # Charge only an increase in downside-risk memory.  Persisting/decaying risk remains
        # observable but does not punish every future action for an already-booked loss.
        penalty = self.config.downside_ewma_weight * max(
            0.0, semideviation - previous_semideviation
        )
        return semideviation, penalty

    def _level_asymmetry(
        self,
        transition: RiskPortfolioTransition,
    ) -> tuple[float, float]:
        base = max(transition.previous_equity, 1e-12)
        wrong_penalty = 0.0
        success_bonus = 0.0
        for level, pnl in (
            (transition.long_level, transition.long_step_net_pnl),
            (transition.short_level, transition.short_step_net_pnl),
        ):
            pnl_pct = pnl / base * 100.0
            if pnl_pct < 0:
                extra = max(0.0, self.config.loss_level_multipliers[level] - 1.0)
                wrong_penalty += self.config.wrong_level_loss_weight * (-pnl_pct) * extra
            elif pnl_pct > 0:
                extra = max(0.0, self.config.win_level_multipliers[level] - 1.0)
                success_bonus += self.config.position_success_bonus_weight * pnl_pct * extra
        return wrong_penalty, success_bonus

    def _side_management_shaping(
        self,
        transition: RiskPortfolioTransition,
    ) -> tuple[float, float, float]:
        base = max(transition.previous_equity, 1e-12)
        adverse_scale = 0.0
        risk_reduction_bonus = 0.0
        profit_lock_bonus = 0.0
        side_facts = (
            (
                transition.previous_long_level,
                transition.long_level,
                transition.previous_long_margin_fraction,
                transition.long_margin_fraction,
                transition.previous_long_unrealized,
                transition.long_realized_pnl,
                transition.long_quantity_delta,
            ),
            (
                transition.previous_short_level,
                transition.short_level,
                transition.previous_short_margin_fraction,
                transition.short_margin_fraction,
                transition.previous_short_unrealized,
                transition.short_realized_pnl,
                transition.short_quantity_delta,
            ),
        )
        for (
            previous_level,
            level,
            previous_margin,
            margin,
            previous_unrealized,
            realized,
            quantity_delta,
        ) in side_facts:
            added_margin = max(0.0, margin - previous_margin)
            reduced_margin = max(0.0, previous_margin - margin)
            if level > previous_level and previous_unrealized < 0:
                loss_pct = abs(previous_unrealized) / base * 100.0
                adverse_scale += added_margin * loss_pct * max(1, level - previous_level)
            if (
                level < previous_level
                and reduced_margin > 0
                and previous_unrealized < 0
                and quantity_delta < 0
            ):
                risk_reduction_bonus += self.config.risk_reduction_bonus_weight * reduced_margin
            if level < previous_level and realized > 0:
                realized_pct = realized / base * 100.0
                profit_lock_bonus += self.config.profit_lock_bonus_weight * min(1.0, realized_pct)
        return (
            self.config.adverse_scale_in_weight * adverse_scale,
            risk_reduction_bonus,
            profit_lock_bonus,
        )

    def _hedge_shaping(self, transition: RiskPortfolioTransition) -> tuple[float, float]:
        long_size = transition.target.long_target_notional
        short_size = transition.target.short_target_notional
        if long_size <= 0 or short_size <= 0 or abs(long_size - short_size) <= 1e-12:
            return 0.0, 0.0
        if long_size > short_size:
            dominant = transition.long_step_net_pnl
            hedge = transition.short_step_net_pnl
        else:
            dominant = transition.short_step_net_pnl
            hedge = transition.long_step_net_pnl
        base = max(transition.previous_equity, 1e-12)
        hedge_bonus = 0.0
        hedge_waste = 0.0
        if dominant < 0 < hedge:
            offset_pct = min(abs(dominant), hedge) / base * 100.0
            hedge_bonus = self.config.hedge_efficiency_weight * offset_pct
        elif dominant > 0 > hedge:
            drag_pct = min(dominant, abs(hedge)) / base * 100.0
            hedge_waste = self.config.hedge_waste_weight * drag_pct
        return hedge_bonus, hedge_waste

    def _transform_reward(self, reward: float) -> float:
        clip = self.config.reward_clip
        if self.config.soft_clip:
            return clip * math.tanh(reward / clip)
        return max(-clip, min(clip, reward))

    def calculate(
        self,
        *,
        transition: RiskPortfolioTransition,
        account: RiskAccountState,
        mark: float,
        uncertainty_score: float,
        reserve_margin_fraction: float,
    ) -> RiskRewardBreakdown:
        uncertainty = min(1.0, max(0.0, float(uncertainty_score)))
        reserve = min(1.0, max(0.0, float(reserve_margin_fraction)))
        base_log = self._safe_log_return(transition.previous_equity, transition.equity)
        equity_reward = self.config.equity_log_return_scale * base_log

        # Incremental squared drawdown prices severity without charging the same static
        # drawdown again forever.
        drawdown_penalty = (
            self.config.drawdown_weight
            * max(0.0, transition.drawdown**2 - transition.previous_drawdown**2)
            * 100.0
        )
        gross_margin = max(0.0, transition.used_margin_fraction)
        downside = max(0.0, -equity_reward)
        downside_exposure_penalty = (
            self.config.downside_exposure_weight * downside * gross_margin**2
        )
        downside_semideviation, downside_ewma_penalty = self._update_downside_risk(base_log)
        uncertainty_penalty = (
            self.config.uncertainty_exposure_weight * uncertainty * gross_margin**2
        )
        gross_notional = max(0.0, account.gross_notional_ratio(mark))
        leverage_excess = max(0.0, gross_notional - gross_margin)
        leverage_penalty = (
            self.config.leverage_exposure_weight * leverage_excess**2 * (0.25 + 0.75 * uncertainty)
        )
        preferred_shortfall = max(0.0, self.config.preferred_reserve_margin_fraction - reserve)
        reserve_pressure_penalty = (
            self.config.reserve_pressure_weight * preferred_shortfall**2 * 100.0
        )
        hard_shortfall = max(0.0, self.config.minimum_liquidation_buffer_fraction - reserve)
        liquidation_penalty = self.config.liquidation_buffer_weight * hard_shortfall**2 * 100.0

        wrong_level_penalty, position_success_bonus = self._level_asymmetry(transition)
        adverse_scale_penalty, risk_reduction_bonus, profit_lock_bonus = (
            self._side_management_shaping(transition)
        )

        long_jump = max(0, transition.long_level - transition.previous_long_level - 1)
        short_jump = max(0, transition.short_level - transition.previous_short_level - 1)
        upward_jump_excess = long_jump + short_jump
        upward_jump_penalty = (
            self.config.upward_jump_weight
            * upward_jump_excess
            * (uncertainty + transition.previous_drawdown)
        )
        turnover_ratio = transition.traded_notional / max(transition.previous_equity, 1e-12)
        level_churn_penalty = (
            self.config.level_churn_weight * transition.level_distance * turnover_ratio
        )
        turnover_penalty = self.config.turnover_shaping_weight * turnover_ratio

        hedge_bonus, hedge_waste = self._hedge_shaping(transition)
        delayed_scale_bonus, delayed_probe_bonus, repeated_probe_penalty = self._resolve_outcomes(
            account, mark=mark
        )
        self._schedule_outcomes(transition, account, mark)

        positive_shaping_raw = (
            position_success_bonus
            + risk_reduction_bonus
            + profit_lock_bonus
            + hedge_bonus
            + delayed_scale_bonus
            + delayed_probe_bonus
        )
        positive_shaping_applied = min(self.config.max_positive_shaping, positive_shaping_raw)

        reward = (
            equity_reward
            - drawdown_penalty
            - downside_exposure_penalty
            - downside_ewma_penalty
            - uncertainty_penalty
            - leverage_penalty
            - reserve_pressure_penalty
            - liquidation_penalty
            - wrong_level_penalty
            - adverse_scale_penalty
            - upward_jump_penalty
            - level_churn_penalty
            - turnover_penalty
            - repeated_probe_penalty
            - hedge_waste
            + positive_shaping_applied
        )
        transformed = self._transform_reward(reward)
        accounting_cost_ratio = transition.accounting_cost / max(transition.previous_equity, 1e-12)
        return RiskRewardBreakdown(
            equity_log_return=equity_reward,
            drawdown_penalty=drawdown_penalty,
            downside_exposure_penalty=downside_exposure_penalty,
            downside_ewma_penalty=downside_ewma_penalty,
            uncertainty_exposure_penalty=uncertainty_penalty,
            leverage_exposure_penalty=leverage_penalty,
            reserve_pressure_penalty=reserve_pressure_penalty,
            liquidation_buffer_penalty=liquidation_penalty,
            wrong_level_loss_penalty=wrong_level_penalty,
            position_success_bonus=position_success_bonus,
            adverse_scale_in_penalty=adverse_scale_penalty,
            upward_jump_penalty=upward_jump_penalty,
            level_churn_penalty=level_churn_penalty,
            turnover_shaping_penalty=turnover_penalty,
            repeated_probe_penalty=repeated_probe_penalty,
            risk_reduction_bonus=risk_reduction_bonus,
            profit_lock_bonus=profit_lock_bonus,
            hedge_efficiency_bonus=hedge_bonus,
            hedge_waste_penalty=hedge_waste,
            delayed_scale_bonus=delayed_scale_bonus,
            delayed_probe_bonus=delayed_probe_bonus,
            positive_shaping_raw=positive_shaping_raw,
            positive_shaping_applied=positive_shaping_applied,
            accounting_cost_ratio=accounting_cost_ratio,
            downside_semideviation=downside_semideviation,
            unclipped_reward=reward,
            reward=transformed,
            consecutive_failed_probes=self.consecutive_failed_probes,
            consecutive_failed_probes_long=self.consecutive_failed_probes_long,
            consecutive_failed_probes_short=self.consecutive_failed_probes_short,
        )
