"""Target risk-level action space for dual-leg FreqAI Hedge reinforcement learning.

The policy selects a target *margin-risk budget* for LONG and SHORT independently.
The selection is not an order-size command.  A mapper converts the two levels into
margin budgets and leverage-adjusted target notionals while planner/risk/execution
remain responsible for executable intents.

V3 keeps the user-facing MultiDiscrete([5, 5]) contract and adds an explicit action
topology/transition contract.  This makes level distance, upward jumps, gross/net
margin intent and checkpoint signatures deterministic without changing the 25-state
semantic design.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import IntEnum
from typing import Any


class PositionRiskLevel(IntEnum):
    FLAT = 0
    VERY_LOW = 1
    LIGHT = 2
    MEDIUM = 3
    HEAVY = 4


# Tensorboard only needs an Enum-like catalogue.  The actual SB3 action is MultiDiscrete([5, 5]).
class JointRiskAction(IntEnum):
    LONG_0_SHORT_0 = 0
    LONG_0_SHORT_1 = 1
    LONG_0_SHORT_2 = 2
    LONG_0_SHORT_3 = 3
    LONG_0_SHORT_4 = 4
    LONG_1_SHORT_0 = 5
    LONG_1_SHORT_1 = 6
    LONG_1_SHORT_2 = 7
    LONG_1_SHORT_3 = 8
    LONG_1_SHORT_4 = 9
    LONG_2_SHORT_0 = 10
    LONG_2_SHORT_1 = 11
    LONG_2_SHORT_2 = 12
    LONG_2_SHORT_3 = 13
    LONG_2_SHORT_4 = 14
    LONG_3_SHORT_0 = 15
    LONG_3_SHORT_1 = 16
    LONG_3_SHORT_2 = 17
    LONG_3_SHORT_3 = 18
    LONG_3_SHORT_4 = 19
    LONG_4_SHORT_0 = 20
    LONG_4_SHORT_1 = 21
    LONG_4_SHORT_2 = 22
    LONG_4_SHORT_3 = 23
    LONG_4_SHORT_4 = 24


@dataclass(frozen=True, slots=True)
class HedgeRiskLevelAction:
    long_level: PositionRiskLevel
    short_level: PositionRiskLevel

    @classmethod
    def from_value(cls, value: Sequence[int] | HedgeRiskLevelAction) -> HedgeRiskLevelAction:
        if isinstance(value, HedgeRiskLevelAction):
            return value
        if len(value) != 2:
            raise ValueError("Hedge risk-level action must contain [long_level, short_level]")
        levels: list[PositionRiskLevel] = []
        try:
            for raw in value:
                if isinstance(raw, bool):
                    raise TypeError("boolean is not a risk level")
                numeric = float(raw)
                if not math.isfinite(numeric) or not numeric.is_integer():
                    raise ValueError("risk level must be an exact integer")
                levels.append(PositionRiskLevel(int(numeric)))
            return cls(levels[0], levels[1])
        except TypeError as exc:
            raise TypeError("risk levels must be exact integers within [0, 4]") from exc
        except (ValueError, OverflowError) as exc:
            raise ValueError("risk levels must be exact integers within [0, 4]") from exc

    def as_tuple(self) -> tuple[int, int]:
        return int(self.long_level), int(self.short_level)

    @property
    def joint_id(self) -> int:
        return int(self.long_level) * 5 + int(self.short_level)

    @classmethod
    def from_joint_id(cls, joint_id: int) -> HedgeRiskLevelAction:
        if isinstance(joint_id, bool):
            raise TypeError("joint risk action id must be an exact integer within [0, 24]")
        try:
            numeric = float(joint_id)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "joint risk action id must be an exact integer within [0, 24]"
            ) from exc
        if not math.isfinite(numeric) or not numeric.is_integer():
            raise ValueError("joint risk action id must be an exact integer within [0, 24]")
        value = int(numeric)
        if not 0 <= value < 25:
            raise ValueError("joint risk action id must be within [0, 24]")
        return cls.from_value(divmod(value, 5))


@dataclass(frozen=True, slots=True)
class RiskLevelProfile:
    """Configuration for target margin budgets.

    The default profile intentionally preserves 20% of account equity as unallocated
    cross-margin budget even when both legs request HEAVY.  Fractions are margin
    budgets, not notionals, and are multiplied by side leverage by RiskLevelMapper.

    ``rebalance_deadband_fraction`` is not part of policy semantics.  It is an
    execution-simulation tolerance used only when a leg remains at the same level, so
    normal mark-to-market equity drift does not generate tiny artificial trades on
    every candle.
    """

    position_levels: tuple[float, float, float, float, float] = (0.0, 0.05, 0.12, 0.25, 0.40)
    long_leverage: float = 1.0
    short_leverage: float = 1.0
    max_combined_margin_fraction: float = 0.80
    minimum_reserve_margin_fraction: float = 0.20
    hard_max_margin_fraction_per_leg: float = 0.50
    rebalance_deadband_fraction: float = 0.0025

    def __post_init__(self) -> None:
        values = tuple(float(item) for item in self.position_levels)
        object.__setattr__(self, "position_levels", values)
        self._validate_levels(values)
        self._validate_leverage()
        self._validate_combined_limits(values)
        self._validate_deadband()

    def _validate_levels(self, values: tuple[float, ...]) -> None:
        if len(values) != 5:
            raise ValueError("position_levels must contain exactly five values")
        if values[0] != 0.0:
            raise ValueError("position_levels[0] must be 0.0 (FLAT)")
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("position_levels must be finite and non-negative")
        if tuple(sorted(set(values))) != values:
            raise ValueError("position_levels must be strictly increasing")
        if not 0 < self.hard_max_margin_fraction_per_leg <= 1:
            raise ValueError("hard_max_margin_fraction_per_leg must be within (0, 1]")
        if values[-1] > self.hard_max_margin_fraction_per_leg + 1e-12:
            raise ValueError("HEAVY margin fraction exceeds hard per-leg safety cap")

    def _validate_leverage(self) -> None:
        for name in ("long_leverage", "short_leverage"):
            leverage = float(getattr(self, name))
            if not math.isfinite(leverage) or leverage <= 0:
                raise ValueError(f"{name} must be finite and positive")

    def _validate_combined_limits(self, values: tuple[float, ...]) -> None:
        if not 0 < self.max_combined_margin_fraction <= 1:
            raise ValueError("max_combined_margin_fraction must be within (0, 1]")
        if not 0 <= self.minimum_reserve_margin_fraction < 1:
            raise ValueError("minimum_reserve_margin_fraction must be within [0, 1)")
        available = 1 - self.minimum_reserve_margin_fraction + 1e-12
        if self.max_combined_margin_fraction > available:
            raise ValueError("combined margin cap conflicts with minimum reserve")
        if 2 * values[-1] > self.max_combined_margin_fraction + 1e-12:
            raise ValueError("two HEAVY legs exceed max_combined_margin_fraction")

    def _validate_deadband(self) -> None:
        deadband = float(self.rebalance_deadband_fraction)
        if not math.isfinite(deadband):
            raise ValueError("rebalance_deadband_fraction must be finite")
        if not 0 <= deadband <= 0.05:
            raise ValueError("rebalance_deadband_fraction must be within [0, 0.05]")

    def fraction(self, level: int | PositionRiskLevel) -> float:
        return self.position_levels[int(PositionRiskLevel(int(level)))]

    @property
    def heavy_fraction(self) -> float:
        return self.position_levels[-1]

    @property
    def signature(self) -> str:
        raw = (
            "multidiscrete-5x5|"
            + ",".join(f"{value:.12g}" for value in self.position_levels)
            + f"|L{self.long_leverage:.12g}|S{self.short_leverage:.12g}"
            + f"|C{self.max_combined_margin_fraction:.12g}"
            + f"|R{self.minimum_reserve_margin_fraction:.12g}"
        )
        return hashlib.sha256(raw.encode("ascii")).hexdigest()[:16]

    @classmethod
    def from_freqtrade_config(cls, config: Mapping[str, Any]) -> RiskLevelProfile:
        freqai = config.get("freqai", {}) if isinstance(config, Mapping) else {}
        if not isinstance(freqai, Mapping):
            raise TypeError("freqai config must be an object")
        rl_config = freqai.get("rl_config", {})
        hedge_rl = freqai.get("hedge_rl_config", {})
        if not isinstance(rl_config, Mapping):
            rl_config = {}
        if not isinstance(hedge_rl, Mapping):
            hedge_rl = {}
        action_cfg = rl_config.get("hedge_action_space", {})
        if not isinstance(action_cfg, Mapping):
            action_cfg = {}

        defaults = cls()
        raw_levels = action_cfg.get(
            "position_levels", hedge_rl.get("position_levels", defaults.position_levels)
        )
        values = tuple(float(item) for item in raw_levels)
        if len(values) != 5:
            raise ValueError("position_levels must contain exactly five values")
        position_levels = (values[0], values[1], values[2], values[3], values[4])
        return cls(
            position_levels=position_levels,
            long_leverage=float(
                action_cfg.get("long_leverage", hedge_rl.get("long_leverage", 1.0))
            ),
            short_leverage=float(
                action_cfg.get("short_leverage", hedge_rl.get("short_leverage", 1.0))
            ),
            max_combined_margin_fraction=float(
                action_cfg.get(
                    "max_combined_margin_fraction",
                    hedge_rl.get("max_combined_margin_fraction", 0.80),
                )
            ),
            minimum_reserve_margin_fraction=float(
                action_cfg.get(
                    "minimum_reserve_margin_fraction",
                    hedge_rl.get("minimum_reserve_margin_fraction", 0.20),
                )
            ),
            hard_max_margin_fraction_per_leg=float(
                action_cfg.get(
                    "hard_max_margin_fraction_per_leg",
                    hedge_rl.get("hard_max_margin_fraction_per_leg", 0.50),
                )
            ),
            rebalance_deadband_fraction=float(
                action_cfg.get(
                    "rebalance_deadband_fraction",
                    hedge_rl.get(
                        "rebalance_deadband_fraction", defaults.rebalance_deadband_fraction
                    ),
                )
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["position_levels"] = list(self.position_levels)
        result["signature"] = self.signature
        return result


@dataclass(frozen=True, slots=True)
class RiskActionTransition:
    previous: HedgeRiskLevelAction
    requested: HedgeRiskLevelAction
    long_level_delta: int
    short_level_delta: int
    manhattan_distance: int
    upward_jump_excess: int
    previous_gross_margin_fraction: float
    requested_gross_margin_fraction: float
    gross_margin_delta: float
    previous_net_margin_fraction: float
    requested_net_margin_fraction: float
    net_margin_delta: float

    @property
    def changed(self) -> bool:
        return self.manhattan_distance > 0

    @property
    def increases_risk(self) -> bool:
        return self.gross_margin_delta > 1e-12

    @property
    def reduces_risk(self) -> bool:
        return self.gross_margin_delta < -1e-12


class RiskActionTopology:
    """Deterministic geometry of the 25 target states.

    The topology is deliberately descriptive rather than prescriptive.  It does not
    forbid 0->4 transitions; reward/risk layers can price a large jump according to
    uncertainty and drawdown while preserving the full action space.
    """

    def __init__(self, profile: RiskLevelProfile) -> None:
        self.profile = profile

    def transition(
        self,
        previous: Sequence[int] | HedgeRiskLevelAction,
        requested: Sequence[int] | HedgeRiskLevelAction,
    ) -> RiskActionTransition:
        before = HedgeRiskLevelAction.from_value(previous)
        after = HedgeRiskLevelAction.from_value(requested)
        long_delta = int(after.long_level) - int(before.long_level)
        short_delta = int(after.short_level) - int(before.short_level)
        before_long = self.profile.fraction(before.long_level)
        before_short = self.profile.fraction(before.short_level)
        after_long = self.profile.fraction(after.long_level)
        after_short = self.profile.fraction(after.short_level)
        before_gross = before_long + before_short
        after_gross = after_long + after_short
        before_net = before_long - before_short
        after_net = after_long - after_short
        upward_jump_excess = max(0, long_delta - 1) + max(0, short_delta - 1)
        return RiskActionTransition(
            previous=before,
            requested=after,
            long_level_delta=long_delta,
            short_level_delta=short_delta,
            manhattan_distance=abs(long_delta) + abs(short_delta),
            upward_jump_excess=upward_jump_excess,
            previous_gross_margin_fraction=before_gross,
            requested_gross_margin_fraction=after_gross,
            gross_margin_delta=after_gross - before_gross,
            previous_net_margin_fraction=before_net,
            requested_net_margin_fraction=after_net,
            net_margin_delta=after_net - before_net,
        )

    def margin_matrix(self) -> tuple[tuple[float, ...], ...]:
        return tuple(
            tuple(self.profile.fraction(long) + self.profile.fraction(short) for short in range(5))
            for long in range(5)
        )


@dataclass(frozen=True, slots=True)
class TargetExposure:
    action: HedgeRiskLevelAction
    equity: float
    long_margin_fraction: float
    short_margin_fraction: float
    long_margin_budget: float
    short_margin_budget: float
    long_target_notional: float
    short_target_notional: float
    combined_margin_fraction: float
    reserve_margin_fraction: float


class RiskLevelMapper:
    """Convert policy levels to margin budgets and leverage-adjusted notionals."""

    def __init__(self, profile: RiskLevelProfile) -> None:
        self.profile = profile

    def map(self, action: Sequence[int] | HedgeRiskLevelAction, *, equity: float) -> TargetExposure:
        selected = HedgeRiskLevelAction.from_value(action)
        equity_value = float(equity)
        if not math.isfinite(equity_value) or equity_value <= 0:
            raise ValueError("equity must be finite and positive")
        long_fraction = self.profile.fraction(selected.long_level)
        short_fraction = self.profile.fraction(selected.short_level)
        combined = long_fraction + short_fraction
        if combined > self.profile.max_combined_margin_fraction + 1e-12:
            raise ValueError("requested action exceeds combined cross-margin budget")
        reserve = 1.0 - combined
        if reserve + 1e-12 < self.profile.minimum_reserve_margin_fraction:
            raise ValueError("requested action violates minimum reserve margin")
        long_budget = equity_value * long_fraction
        short_budget = equity_value * short_fraction
        return TargetExposure(
            action=selected,
            equity=equity_value,
            long_margin_fraction=long_fraction,
            short_margin_fraction=short_fraction,
            long_margin_budget=long_budget,
            short_margin_budget=short_budget,
            long_target_notional=long_budget * self.profile.long_leverage,
            short_target_notional=short_budget * self.profile.short_leverage,
            combined_margin_fraction=combined,
            reserve_margin_fraction=reserve,
        )
