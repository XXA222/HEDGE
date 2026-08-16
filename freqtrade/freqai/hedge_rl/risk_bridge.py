"""Shape-safe inference bridge for the MultiDiscrete Hedge risk-level policy."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from .risk_levels import HedgeRiskLevelAction, RiskLevelProfile
from .risk_observation import HedgeRiskObservationBuilder, RiskObservationSchema
from .risk_portfolio import LegSide, RiskAccountState, RiskLegState


@dataclass(frozen=True, slots=True)
class HedgeRiskPolicyContext:
    account: RiskAccountState
    mark: float
    uncertainty_score: float = 0.5
    funding_rate: float = 0.0
    feature_age_steps: int = 0
    projection_fresh: bool = True
    failed_probe_long: int = 0
    failed_probe_short: int = 0
    downside_semideviation: float = 0.0
    pending_reward_fraction: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.mark)) or self.mark <= 0:
            raise ValueError("mark must be finite and positive")
        if not 0 <= float(self.uncertainty_score) <= 1:
            raise ValueError("uncertainty_score must be within [0, 1]")
        if not math.isfinite(float(self.funding_rate)):
            raise ValueError("funding_rate must be finite")
        if self.feature_age_steps < 0:
            raise ValueError("feature_age_steps cannot be negative")
        if self.failed_probe_long < 0 or self.failed_probe_short < 0:
            raise ValueError("failed probe counters cannot be negative")
        if not math.isfinite(float(self.downside_semideviation)) or self.downside_semideviation < 0:
            raise ValueError("downside_semideviation must be finite and non-negative")
        if not 0 <= float(self.pending_reward_fraction) <= 1:
            raise ValueError("pending_reward_fraction must be within [0, 1]")

    @classmethod
    def flat(cls, starting_balance: float, *, mark: float = 1.0) -> HedgeRiskPolicyContext:
        return cls(RiskAccountState.initial(starting_balance), mark)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> HedgeRiskPolicyContext:
        equity = float(values.get("equity", values.get("cash_balance", 0.0)))
        cash = float(values.get("cash_balance", equity))
        peak = float(values.get("peak_equity", max(equity, cash)))
        long = RiskLegState(
            LegSide.LONG,
            float(values.get("long_quantity", 0.0)),
            float(values.get("long_average_price", 0.0)),
        )
        short = RiskLegState(
            LegSide.SHORT,
            float(values.get("short_quantity", 0.0)),
            float(values.get("short_average_price", 0.0)),
        )
        account = RiskAccountState(
            cash_balance=cash,
            equity=equity,
            peak_equity=peak,
            long=long,
            short=short,
            long_level=int(values.get("long_level", 0)),
            short_level=int(values.get("short_level", 0)),
            step=int(values.get("step", 0)),
            turnover=float(values.get("turnover", 0.0)),
        )
        return cls(
            account=account,
            mark=float(values["mark"]),
            uncertainty_score=float(values.get("uncertainty_score", 0.5)),
            funding_rate=float(values.get("funding_rate", 0.0)),
            feature_age_steps=int(values.get("feature_age_steps", 0)),
            projection_fresh=bool(values.get("projection_fresh", True)),
            failed_probe_long=int(values.get("failed_probe_long", 0)),
            failed_probe_short=int(values.get("failed_probe_short", 0)),
            downside_semideviation=float(values.get("downside_semideviation", 0.0)),
            pending_reward_fraction=float(values.get("pending_reward_fraction", 0.0)),
        )


class HedgeRiskLevelPolicyBridge:
    def __init__(
        self,
        *,
        feature_names: tuple[str, ...],
        window_size: int,
        profile: RiskLevelProfile,
        feature_clip: float = 10.0,
        max_episode_steps: int = 2048,
        max_feature_age_steps: int = 1,
    ) -> None:
        self.profile = profile
        self.max_episode_steps = int(max_episode_steps)
        self.max_feature_age_steps = int(max_feature_age_steps)
        self.schema = RiskObservationSchema(feature_names, int(window_size))
        self.builder = HedgeRiskObservationBuilder(self.schema, feature_clip=feature_clip)

    def observation(
        self,
        features: npt.ArrayLike,
        *,
        tick: int,
        context: HedgeRiskPolicyContext,
    ) -> npt.NDArray[np.float32]:
        return self.builder.build(
            features,
            tick=tick,
            account=context.account,
            mark=context.mark,
            profile=self.profile,
            uncertainty_score=context.uncertainty_score,
            funding_rate=context.funding_rate,
            max_episode_steps=self.max_episode_steps,
            failed_probe_long=context.failed_probe_long,
            failed_probe_short=context.failed_probe_short,
            downside_semideviation=context.downside_semideviation,
            pending_reward_fraction=context.pending_reward_fraction,
        )

    def predict_action(
        self,
        model: Any,
        observation: npt.ArrayLike,
        *,
        context: HedgeRiskPolicyContext,
    ) -> HedgeRiskLevelAction:
        if not context.projection_fresh or context.feature_age_steps > self.max_feature_age_steps:
            return HedgeRiskLevelAction.from_value((0, 0))
        vector = np.asarray(observation, dtype=np.float32)
        if vector.shape != (self.schema.flat_size,) or not np.isfinite(vector).all():
            raise ValueError("policy observation shape or values are invalid")
        result, _ = model.predict(vector, deterministic=True)
        values = np.asarray(result).reshape(-1)
        if values.size != 2:
            raise ValueError("Hedge risk-level policy must return exactly two levels")
        if not np.all(np.equal(values, np.floor(values))):
            raise ValueError("Hedge risk-level policy returned non-integer levels")
        return HedgeRiskLevelAction.from_value((int(values[0]), int(values[1])))
