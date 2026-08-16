"""Shape-safe inference bridge for the MultiDiscrete Hedge risk-level policy."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import numpy as np
import numpy.typing as npt

from .risk_levels import HedgeRiskLevelAction, RiskLevelProfile
from .risk_observation import HedgeRiskObservationBuilder, RiskObservationSchema
from .risk_portfolio import LegSide, RiskAccountState, RiskLegState


def _strict_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float | Decimal):
        raise TypeError(f"{field} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _strict_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    return value


def _strict_bool(value: object, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field} must be bool")
    return value


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
        for name in ("feature_age_steps", "failed_probe_long", "failed_probe_short"):
            _strict_int(getattr(self, name), field=name)
        _strict_bool(self.projection_fresh, field="projection_fresh")
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
        equity = _strict_float(
            values.get("equity", values.get("cash_balance", 0.0)), field="equity"
        )
        cash = _strict_float(values.get("cash_balance", equity), field="cash_balance")
        peak = _strict_float(values.get("peak_equity", max(equity, cash)), field="peak_equity")
        long = RiskLegState(
            LegSide.LONG,
            _strict_float(values.get("long_quantity", 0.0), field="long_quantity"),
            _strict_float(values.get("long_average_price", 0.0), field="long_average_price"),
        )
        short = RiskLegState(
            LegSide.SHORT,
            _strict_float(values.get("short_quantity", 0.0), field="short_quantity"),
            _strict_float(values.get("short_average_price", 0.0), field="short_average_price"),
        )
        account = RiskAccountState(
            cash_balance=cash,
            equity=equity,
            peak_equity=peak,
            long=long,
            short=short,
            long_level=_strict_int(values.get("long_level", 0), field="long_level"),
            short_level=_strict_int(values.get("short_level", 0), field="short_level"),
            step=_strict_int(values.get("step", 0), field="step"),
            turnover=_strict_float(values.get("turnover", 0.0), field="turnover"),
        )
        return cls(
            account=account,
            mark=_strict_float(values["mark"], field="mark"),
            uncertainty_score=_strict_float(
                values.get("uncertainty_score", 0.5), field="uncertainty_score"
            ),
            funding_rate=_strict_float(values.get("funding_rate", 0.0), field="funding_rate"),
            feature_age_steps=_strict_int(
                values.get("feature_age_steps", 0), field="feature_age_steps"
            ),
            projection_fresh=_strict_bool(
                values.get("projection_fresh", True), field="projection_fresh"
            ),
            failed_probe_long=_strict_int(
                values.get("failed_probe_long", 0), field="failed_probe_long"
            ),
            failed_probe_short=_strict_int(
                values.get("failed_probe_short", 0), field="failed_probe_short"
            ),
            downside_semideviation=_strict_float(
                values.get("downside_semideviation", 0.0), field="downside_semideviation"
            ),
            pending_reward_fraction=_strict_float(
                values.get("pending_reward_fraction", 0.0), field="pending_reward_fraction"
            ),
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
