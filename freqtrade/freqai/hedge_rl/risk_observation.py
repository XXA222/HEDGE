"""Allocation-conscious causal observation contract for target risk-level Hedge RL."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .risk_levels import RiskLevelProfile
from .risk_portfolio import RiskAccountState


ACCOUNT_FEATURE_NAMES = (
    "long_level",
    "short_level",
    "long_margin_ratio",
    "short_margin_ratio",
    "long_unrealized_ratio",
    "short_unrealized_ratio",
    "gross_notional_ratio",
    "net_notional_ratio",
    "used_margin_fraction",
    "reserve_margin_fraction",
    "drawdown",
    "long_leverage",
    "short_leverage",
    "uncertainty_score",
    "funding_rate",
    "episode_progress",
    "failed_probe_long_norm",
    "failed_probe_short_norm",
    "downside_semideviation_norm",
    "pending_reward_fraction",
)


@dataclass(frozen=True, slots=True)
class RiskObservationSchema:
    market_feature_names: tuple[str, ...]
    window_size: int

    def __post_init__(self) -> None:
        if self.window_size < 2:
            raise ValueError("window_size must be at least 2")
        if not self.market_feature_names:
            raise ValueError("at least one market feature is required")
        if len(set(self.market_feature_names)) != len(self.market_feature_names):
            raise ValueError("market feature names must be unique")

    @property
    def market_flat_size(self) -> int:
        return len(self.market_feature_names) * self.window_size

    @property
    def flat_size(self) -> int:
        return self.market_flat_size + len(ACCOUNT_FEATURE_NAMES)

    @property
    def signature(self) -> str:
        raw = "|".join((*self.market_feature_names, *ACCOUNT_FEATURE_NAMES, str(self.window_size)))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


class HedgeRiskObservationBuilder:
    """Build directly into one float32 output vector.

    V1 converted the complete feature matrix to float64 on every call, clipped into a
    temporary array, allocated a second account array, concatenated both, then cast to
    float32.  V2 keeps the source matrix compact and writes market/account values directly
    into the final observation buffer.
    """

    def __init__(self, schema: RiskObservationSchema, *, feature_clip: float = 10.0) -> None:
        if feature_clip <= 0:
            raise ValueError("feature_clip must be positive")
        self.schema = schema
        self.feature_clip = float(feature_clip)

    def build_into(
        self,
        features: npt.ArrayLike,
        output: npt.NDArray[np.float32],
        *,
        tick: int,
        account: RiskAccountState,
        mark: float,
        profile: RiskLevelProfile,
        uncertainty_score: float,
        funding_rate: float,
        max_episode_steps: int,
        failed_probe_long: int = 0,
        failed_probe_short: int = 0,
        downside_semideviation: float = 0.0,
        pending_reward_fraction: float = 0.0,
    ) -> npt.NDArray[np.float32]:
        values = np.asarray(features)
        expected_width = len(self.schema.market_feature_names)
        if values.ndim != 2 or values.shape[1] != expected_width:
            raise ValueError("market feature matrix has incompatible shape")
        if output.shape != (self.schema.flat_size,) or output.dtype != np.float32:
            raise ValueError("observation output buffer has incompatible shape or dtype")
        start = tick - self.schema.window_size + 1
        if start < 0 or tick >= len(values):
            raise ValueError("tick does not contain a complete causal observation window")
        window = values[start : tick + 1]
        market_output = output[: self.schema.market_flat_size].reshape(window.shape)
        np.clip(window, -self.feature_clip, self.feature_clip, out=market_output)

        base = max(abs(account.equity), 1e-12)
        used = account.used_margin_fraction(mark, profile)
        reserve = max(0.0, 1.0 - used)
        offset = self.schema.market_flat_size
        output[offset + 0] = account.long_level / 4.0
        output[offset + 1] = account.short_level / 4.0
        output[offset + 2] = account.long.notional(mark) / profile.long_leverage / base
        output[offset + 3] = account.short.notional(mark) / profile.short_leverage / base
        output[offset + 4] = account.long.unrealized_pnl(mark) / base
        output[offset + 5] = account.short.unrealized_pnl(mark) / base
        output[offset + 6] = account.gross_notional_ratio(mark)
        output[offset + 7] = account.net_notional_ratio(mark)
        output[offset + 8] = used
        output[offset + 9] = reserve
        output[offset + 10] = account.drawdown()
        output[offset + 11] = profile.long_leverage / 20.0
        output[offset + 12] = profile.short_leverage / 20.0
        output[offset + 13] = min(1.0, max(0.0, float(uncertainty_score)))
        output[offset + 14] = float(funding_rate)
        output[offset + 15] = min(1.0, account.step / max(int(max_episode_steps), 1))
        output[offset + 16] = min(1.0, max(0.0, int(failed_probe_long) / 4.0))
        output[offset + 17] = min(1.0, max(0.0, int(failed_probe_short) / 4.0))
        output[offset + 18] = min(1.0, max(0.0, float(downside_semideviation) / 5.0))
        output[offset + 19] = min(1.0, max(0.0, float(pending_reward_fraction)))
        np.clip(output[offset:], -self.feature_clip, self.feature_clip, out=output[offset:])
        if not np.isfinite(output[offset:]).all():
            raise ValueError("account observation contains non-finite values")
        return output

    def build(
        self,
        features: npt.ArrayLike,
        *,
        tick: int,
        account: RiskAccountState,
        mark: float,
        profile: RiskLevelProfile,
        uncertainty_score: float,
        funding_rate: float,
        max_episode_steps: int,
        failed_probe_long: int = 0,
        failed_probe_short: int = 0,
        downside_semideviation: float = 0.0,
        pending_reward_fraction: float = 0.0,
    ) -> npt.NDArray[np.float32]:
        output = np.empty(self.schema.flat_size, dtype=np.float32)
        return self.build_into(
            features,
            output,
            tick=tick,
            account=account,
            mark=mark,
            profile=profile,
            uncertainty_score=uncertainty_score,
            funding_rate=funding_rate,
            max_episode_steps=max_episode_steps,
            failed_probe_long=failed_probe_long,
            failed_probe_short=failed_probe_short,
            downside_semideviation=downside_semideviation,
            pending_reward_fraction=pending_reward_fraction,
        )
