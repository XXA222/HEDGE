"""Memory-efficient Gymnasium environment for MultiDiscrete([5, 5]) Hedge targets."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from .risk_gym import gym, spaces
from .risk_levels import JointRiskAction, RiskLevelProfile
from .risk_memory import (
    CompactRiskMarketData,
    HedgeRLMemoryConfig,
    compact_feature_matrix,
    release_rl_phase_memory,
)
from .risk_observation import HedgeRiskObservationBuilder, RiskObservationSchema
from .risk_portfolio import TargetLevelPortfolioSimulator
from .risk_reward import HedgeRiskRewardModel, RiskRewardConfig


@dataclass(frozen=True, slots=True)
class HedgeRiskEnvConfig:
    observation_window: int = 32
    max_episode_steps: int = 2048
    starting_balance: float = 10_000.0
    fee_rate: float = 0.0004
    slippage_bps: float = 1.0
    funding_rate_per_step: float = 0.0
    drawdown_stop: float = 0.35
    maintenance_margin_fraction: float = 0.05
    feature_clip: float = 10.0
    default_uncertainty_score: float = 0.50
    random_start: bool = True
    seed: int = 1

    def __post_init__(self) -> None:
        if self.observation_window < 2 or self.max_episode_steps < 1:
            raise ValueError("invalid episode dimensions")
        if not math.isfinite(float(self.starting_balance)) or self.starting_balance <= 0:
            raise ValueError("starting_balance must be finite and positive")
        if not math.isfinite(float(self.fee_rate)) or not 0 <= self.fee_rate < 0.1:
            raise ValueError("fee_rate must be finite and within [0, 0.1)")
        if not math.isfinite(float(self.slippage_bps)) or not 0 <= self.slippage_bps < 10_000:
            raise ValueError("slippage_bps must be finite and within [0, 10000)")
        if not math.isfinite(float(self.funding_rate_per_step)):
            raise ValueError("funding_rate_per_step must be finite")
        if not 0 < self.drawdown_stop < 1:
            raise ValueError("drawdown_stop must be within (0, 1)")
        if not 0 <= self.maintenance_margin_fraction < 1:
            raise ValueError("maintenance_margin_fraction must be within [0, 1)")
        if not math.isfinite(float(self.feature_clip)) or self.feature_clip <= 0:
            raise ValueError("feature_clip must be finite and positive")
        if not 0 <= self.default_uncertainty_score <= 1:
            raise ValueError("default_uncertainty_score must be within [0, 1]")

    @classmethod
    def from_freqtrade_config(cls, config: Mapping[str, Any]) -> HedgeRiskEnvConfig:
        freqai = config.get("freqai", {}) if isinstance(config, Mapping) else {}
        hedge = freqai.get("hedge_rl_config", {}) if isinstance(freqai, Mapping) else {}
        if not isinstance(hedge, Mapping):
            hedge = {}
        aliases = {
            "observation_window": hedge.get("observation_window", 32),
            "max_episode_steps": hedge.get("max_episode_steps", 2048),
            "starting_balance": hedge.get("starting_balance", 10_000.0),
            "fee_rate": hedge.get("fee_rate", 0.0004),
            "slippage_bps": hedge.get("slippage_bps", 1.0),
            "funding_rate_per_step": hedge.get("funding_rate_per_step", 0.0),
            "drawdown_stop": hedge.get("drawdown_stop", 0.35),
            "maintenance_margin_fraction": hedge.get("maintenance_margin_ratio", 0.05),
            "feature_clip": hedge.get("feature_clip", 10.0),
            "default_uncertainty_score": hedge.get("default_uncertainty_score", 0.50),
            "random_start": hedge.get("random_start", True),
            "seed": hedge.get("seed", 1),
        }
        return cls(**aliases)


class HedgeRiskLevelEnv(gym.Env):
    """Causal dual-leg environment with compact retained arrays.

    Observation at tick t contains only information through closed candle t.  The target
    action is executed at t+1 open and marked at t+1 close.  The retained hot-loop data is
    a compact feature matrix plus four one-dimensional market arrays; the input pandas
    DataFrames are not retained.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        df,
        prices,
        reward_kwargs: dict[str, Any] | None = None,
        window_size: int | None = None,
        starting_point: bool = True,
        id: str = "hedge-risk-level-v3-action-reward",  # noqa: A002
        seed: int = 1,
        config: dict[str, Any] | None = None,
        live: bool = False,
        fee: float | None = None,
        can_short: bool = True,
        pair: str = "",
        df_raw=None,
    ) -> None:
        del reward_kwargs, starting_point, live, can_short, df_raw
        config_dict = config or {"freqai": {}}
        self.id = id
        self.pair = pair
        self.env_config = HedgeRiskEnvConfig.from_freqtrade_config(config_dict)
        if window_size is not None:
            self.env_config = replace(self.env_config, observation_window=int(window_size))
        if fee is not None:
            self.env_config = replace(self.env_config, fee_rate=float(fee))
        if seed != 1:
            self.env_config = replace(self.env_config, seed=int(seed))
        self.memory_config = HedgeRLMemoryConfig.from_freqtrade_config(config_dict)
        self.profile = RiskLevelProfile.from_freqtrade_config(config_dict)
        self.reward_config = RiskRewardConfig.from_freqtrade_config(config_dict)
        # Do not retain the full Freqtrade configuration object in each environment.
        del config_dict

        self._rng = np.random.default_rng(self.env_config.seed)
        self.feature_names, self.features, self.market = self._prepare_market_data(
            df,
            prices,
            observation_window=self.env_config.observation_window,
            feature_dtype=self.memory_config.numpy_feature_dtype,
        )
        self.schema = RiskObservationSchema(self.feature_names, self.env_config.observation_window)
        self.observation_builder = HedgeRiskObservationBuilder(
            self.schema, feature_clip=self.env_config.feature_clip
        )
        self.action_space = spaces.MultiDiscrete(np.asarray([5, 5], dtype=np.int64))
        self.observation_space = spaces.Box(
            low=-self.env_config.feature_clip,
            high=self.env_config.feature_clip,
            shape=(self.schema.flat_size,),
            dtype=np.float32,
        )
        self._action_mask = np.ones(10, dtype=np.bool_)
        self._action_mask.flags.writeable = False
        self.simulator = TargetLevelPortfolioSimulator(
            self.env_config.starting_balance,
            profile=self.profile,
            fee_rate=self.env_config.fee_rate,
            slippage_bps=self.env_config.slippage_bps,
        )
        self.reward_model = HedgeRiskRewardModel(
            self.reward_config,
            max_pending_outcomes=self.memory_config.max_pending_reward_outcomes,
        )
        # Immutable contract signatures are hot-loop telemetry; compute them once.
        self._observation_signature = self.schema.signature
        self._action_signature = self.profile.signature
        self._reward_signature = self.reward_config.signature
        self._start_tick = self.env_config.observation_window - 1
        self._end_tick = len(self.market) - 1
        self._current_tick = self._start_tick
        self._episode_steps = 0
        self._episode_count = 0
        self._terminated = False
        self._truncated = False
        self._episode_start_tick = self._start_tick
        self._episode_end_tick = self._end_tick
        self._episode_step_limit = self.env_config.max_episode_steps
        self.tensorboard_metrics: dict[str, dict[str, float]] = {
            "hedge_risk": {
                "long_level": 0.0,
                "short_level": 0.0,
                "used_margin_fraction": 0.0,
                "reserve_margin_fraction": 1.0,
                "pending_reward_outcomes": 0.0,
                "failed_probes_long": 0.0,
                "failed_probes_short": 0.0,
                "downside_semideviation": 0.0,
            }
        }

    @staticmethod
    def _prepare_market_data(
        df,
        prices,
        *,
        observation_window: int,
        feature_dtype: np.dtype | None = None,
    ):
        if feature_dtype is None:
            feature_dtype = np.dtype(np.float32)
        feature_names = tuple(str(column) for column in pd.DataFrame(df, copy=False).columns)
        features = compact_feature_matrix(df, dtype=feature_dtype, readonly=True)
        market = CompactRiskMarketData.from_prices(prices)
        if len(features) != len(market):
            raise ValueError("feature and price rows must have identical length")
        if len(features) <= observation_window:
            raise ValueError("dataset must contain more rows than observation_window")
        return feature_names, features, market

    def _row_uncertainty(self, tick: int) -> float:
        raw = float(self.market.uncertainty_score[tick])
        if np.isnan(raw):
            return self.env_config.default_uncertainty_score
        return min(1.0, max(0.0, raw))

    def _funding(self, tick: int) -> float:
        raw = float(self.market.funding_rate[tick])
        return raw if raw != 0.0 else self.env_config.funding_rate_per_step

    def _observation(self) -> np.ndarray:
        tick = self._current_tick
        return self.observation_builder.build(
            self.features,
            tick=tick,
            account=self.simulator.state,
            mark=float(self.market.close[tick]),
            profile=self.profile,
            uncertainty_score=self._row_uncertainty(tick),
            funding_rate=self._funding(tick),
            max_episode_steps=self.env_config.max_episode_steps,
            failed_probe_long=self.reward_model.consecutive_failed_probes_long,
            failed_probe_short=self.reward_model.consecutive_failed_probes_short,
            downside_semideviation=self.reward_model.downside_semideviation,
            pending_reward_fraction=(
                self.reward_model.pending_outcome_count
                / max(self.reward_model.max_pending_outcomes, 1)
            ),
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        try:
            super().reset(seed=seed)
        except TypeError:
            super().reset(seed=seed, options=options)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._episode_count += 1
        gc_every = self.memory_config.gc_collect_every_episodes
        if gc_every and self._episode_count > 1 and self._episode_count % gc_every == 0:
            # Episode boundary only; never in the per-candle step loop.
            release_rl_phase_memory(trim_allocator=False)

        requested = dict(options or {})
        explicit_start = requested.get("start_tick")
        explicit_end = requested.get("end_tick")
        explicit_limit = requested.get("max_episode_steps")

        end_tick = self._end_tick if explicit_end is None else int(explicit_end)
        if not self._start_tick < end_tick <= self._end_tick:
            raise ValueError("audit end_tick must be within (_start_tick, _end_tick]")

        if explicit_start is not None:
            start_tick = int(explicit_start)
            if not self._start_tick <= start_tick < end_tick:
                raise ValueError(
                    "audit start_tick must contain a full observation and precede end_tick"
                )
        else:
            default_limit = self.env_config.max_episode_steps
            max_start = max(self._start_tick, end_tick - default_limit)
            if self.env_config.random_start and max_start > self._start_tick:
                start_tick = int(self._rng.integers(self._start_tick, max_start + 1))
            else:
                start_tick = self._start_tick

        step_limit = (
            self.env_config.max_episode_steps if explicit_limit is None else int(explicit_limit)
        )
        if step_limit < 1:
            raise ValueError("max_episode_steps reset option must be positive")
        self._episode_start_tick = start_tick
        self._episode_end_tick = end_tick
        self._episode_step_limit = min(step_limit, end_tick - start_tick)
        self._current_tick = start_tick
        self._episode_steps = 0
        self._terminated = False
        self._truncated = False
        self.simulator.reset(self.env_config.starting_balance)
        self.reward_model.reset()
        info = self._info(requested=(0, 0), executed=(0, 0), reward_components=None)
        return self._observation(), info

    def get_actions(self):
        return JointRiskAction

    def action_masks(self) -> np.ndarray:
        # Factorized MaskablePPO mask is static for the V1 risk profile. Returning the
        # read-only cached array avoids a 10-element allocation on every mask query.
        return self._action_mask

    def _info(self, *, requested, executed, reward_components):
        tick = self._current_tick
        mark = float(self.market.close[tick])
        state = self.simulator.state
        return {
            "tick": tick,
            "episode_start_tick": self._episode_start_tick,
            "episode_end_tick": self._episode_end_tick,
            "episode_step_limit": self._episode_step_limit,
            "requested_long_level": int(requested[0]),
            "requested_short_level": int(requested[1]),
            "executed_long_level": int(executed[0]),
            "executed_short_level": int(executed[1]),
            "equity": float(state.equity),
            "drawdown": float(state.drawdown()),
            "gross_notional_ratio": float(state.gross_notional_ratio(mark)),
            "net_notional_ratio": float(state.net_notional_ratio(mark)),
            "used_margin_fraction": float(state.used_margin_fraction(mark, self.profile)),
            "reserve_margin_fraction": float(state.reserve_margin_fraction(mark, self.profile)),
            "pending_reward_outcomes": self.reward_model.pending_outcome_count,
            "reward_components": reward_components or {},
            "observation_schema": self._observation_signature,
            "action_signature": self._action_signature,
            "reward_signature": self._reward_signature,
        }

    def _emit_reward_breakdown(self, *, terminated: bool, truncated: bool) -> bool:
        interval = self.memory_config.reward_breakdown_interval
        return bool(terminated or truncated or (interval and self._episode_steps % interval == 0))

    def step(self, action: Sequence[int]):
        if self._terminated or self._truncated:
            raise RuntimeError("step() called after episode completion; call reset()")
        action_array = np.asarray(action)
        if not np.issubdtype(action_array.dtype, np.integer):
            raise ValueError("risk-level action must contain integer levels")
        action_array = action_array.astype(np.int64, copy=False)
        if not self.action_space.contains(action_array):
            raise ValueError(f"action {action!r} is outside MultiDiscrete([5, 5])")
        requested = (int(action_array[0]), int(action_array[1]))
        executed = requested
        next_tick = self._current_tick + 1
        transition = self.simulator.apply_target(
            executed,
            reference_price=float(self.market.open[next_tick]),
            mark_price=float(self.market.close[next_tick]),
            funding_rate=self._funding(next_tick),
        )
        self._current_tick = next_tick
        self._episode_steps += 1
        state = self.simulator.state
        mark = float(self.market.close[next_tick])
        reserve = state.reserve_margin_fraction(mark, self.profile)
        breakdown = self.reward_model.calculate(
            transition=transition,
            account=state,
            mark=mark,
            uncertainty_score=self._row_uncertainty(next_tick),
            reserve_margin_fraction=reserve,
        )
        self._terminated = bool(
            state.equity <= 0
            or state.drawdown() >= self.env_config.drawdown_stop
            or reserve <= self.env_config.maintenance_margin_fraction
        )
        self._truncated = bool(
            self._current_tick >= self._episode_end_tick
            or self._episode_steps >= self._episode_step_limit
        )
        reward_components = (
            breakdown.to_dict()
            if self._emit_reward_breakdown(
                terminated=self._terminated,
                truncated=self._truncated,
            )
            else None
        )
        info = self._info(
            requested=requested,
            executed=executed,
            reward_components=reward_components,
        )
        metrics = self.tensorboard_metrics["hedge_risk"]
        metrics["long_level"] = float(executed[0])
        metrics["short_level"] = float(executed[1])
        metrics["used_margin_fraction"] = float(info["used_margin_fraction"])
        metrics["reserve_margin_fraction"] = float(info["reserve_margin_fraction"])
        metrics["pending_reward_outcomes"] = float(self.reward_model.pending_outcome_count)
        metrics["failed_probes_long"] = float(self.reward_model.consecutive_failed_probes_long)
        metrics["failed_probes_short"] = float(self.reward_model.consecutive_failed_probes_short)
        metrics["downside_semideviation"] = float(self.reward_model.downside_semideviation)
        return self._observation(), breakdown.reward, self._terminated, self._truncated, info

    def memory_telemetry(self) -> dict[str, int | str]:
        """Small, non-retaining memory contract used by validators and diagnostics."""

        return {
            "feature_dtype": str(self.features.dtype),
            "feature_bytes": int(self.features.nbytes),
            "market_bytes": self.market.nbytes,
            "retained_price_dataframe": 0,
            "pending_reward_outcomes": self.reward_model.pending_outcome_count,
        }
