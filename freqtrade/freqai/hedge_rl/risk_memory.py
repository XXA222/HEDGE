"""Memory lifecycle helpers for Hedge risk-level reinforcement learning.

The design follows the same lifecycle principle used by Freqtrade's memory-sensitive
backtesting path: keep the hot-loop surface narrow, downcast where precision permits,
release phase-sized object graphs at explicit boundaries, and never run heavy garbage
collection in the per-candle hot loop.
"""

from __future__ import annotations

import gc
import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class HedgeRLMemoryConfig:
    """Memory policy for the target-risk RL path.

    ``float32`` is sufficient for normalized FreqAI features and halves the feature
    matrix footprint relative to the previous float64 environment copy. Price arrays
    remain float64 because fills/accounting should not inherit feature precision.
    """

    feature_dtype: str = "float32"
    reward_breakdown_interval: int = 64
    gc_collect_every_episodes: int = 64
    release_training_envs_after_fit: bool = True
    release_phase_memory_after_fit: bool = True
    release_phase_memory_after_predict: bool = False
    max_pending_reward_outcomes: int = 64

    def __post_init__(self) -> None:
        if self.feature_dtype not in {"float32", "float64"}:
            raise ValueError("feature_dtype must be float32 or float64")
        for name in (
            "reward_breakdown_interval",
            "gc_collect_every_episodes",
            "max_pending_reward_outcomes",
        ):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} cannot be negative")
        # With two legs and five levels, a monotonic 0->1->2->3->4 trajectory can retain
        # one probe plus three scale-confirmation events per leg.  Eight is therefore the
        # smallest production cap that cannot reject a valid dual-leg action trajectory.
        if self.max_pending_reward_outcomes < 8:
            raise ValueError("max_pending_reward_outcomes must be at least 8")

    @property
    def numpy_feature_dtype(self) -> np.dtype:
        return np.dtype(self.feature_dtype)

    @classmethod
    def from_freqtrade_config(cls, config: Mapping[str, Any]) -> HedgeRLMemoryConfig:
        freqai = config.get("freqai", {}) if isinstance(config, Mapping) else {}
        hedge = freqai.get("hedge_rl_config", {}) if isinstance(freqai, Mapping) else {}
        memory = hedge.get("memory", {}) if isinstance(hedge, Mapping) else {}
        if not isinstance(memory, Mapping):
            memory = {}
        valid = set(cls.__dataclass_fields__)
        return cls(**{key: value for key, value in memory.items() if key in valid})


def release_rl_phase_memory(*, trim_allocator: bool | None = None) -> bool:
    """Release large phase garbage and reuse the V1.4 global allocator trim when present."""

    try:
        from freqtrade.hedge.backtesting.memory import release_phase_memory

        return bool(release_phase_memory(trim_allocator=trim_allocator))
    except (ImportError, AttributeError):
        gc.collect()
        return False


def compact_feature_matrix(
    frame: Any,
    *,
    dtype: np.dtype[Any] | str = "float32",
    require_finite: bool = True,
    readonly: bool = True,
) -> npt.NDArray[np.floating]:
    """Return a compact C-contiguous numeric feature matrix with at most one cast copy."""

    target_dtype = np.dtype(dtype)
    if isinstance(frame, pd.DataFrame):
        try:
            values = frame.to_numpy(dtype=target_dtype, copy=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("Hedge RL features must be numeric") from exc
    else:
        try:
            values = np.asarray(frame, dtype=target_dtype)
        except (TypeError, ValueError) as exc:
            raise ValueError("Hedge RL features must be numeric") from exc
    if values.ndim != 2:
        raise ValueError("Hedge RL feature matrix must be two-dimensional")
    if not values.flags.c_contiguous:
        values = np.ascontiguousarray(values, dtype=target_dtype)
    if require_finite and not np.isfinite(values).all():
        raise ValueError("Hedge RL features must be finite")
    if readonly:
        try:
            values.flags.writeable = False
        except ValueError:
            logger.debug("Unable to mark compact Hedge RL features read-only", exc_info=True)
    return values


def compact_training_dataframe(frame: pd.DataFrame, *, dtype: str = "float32") -> pd.DataFrame:
    """Downcast transformed FreqAI RL features without introducing an extra persistent copy."""

    if frame.empty:
        return frame
    target = np.dtype(dtype)
    if target == np.float64:
        return frame
    # FreqAI's transformed RL features are numeric.  astype returns a compact frame and the
    # caller immediately replaces the previous reference, allowing the old blocks to die.
    try:
        result = frame.astype(target, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("transformed Hedge RL features must be numeric") from exc
    return result


@dataclass(frozen=True, slots=True)
class CompactRiskMarketData:
    """Narrow price surface retained by the RL hot loop.

    HIGH/LOW/VOLUME are validated during construction and then released.  The simulator only
    needs next-bar OPEN, marking CLOSE, funding, and optional uncertainty.
    """

    open: npt.NDArray[np.float64]
    close: npt.NDArray[np.float64]
    funding_rate: npt.NDArray[np.float32]
    uncertainty_score: npt.NDArray[np.float32]

    def __len__(self) -> int:
        return int(self.open.shape[0])

    @property
    def nbytes(self) -> int:
        return int(
            self.open.nbytes
            + self.close.nbytes
            + self.funding_rate.nbytes
            + self.uncertainty_score.nbytes
        )

    @staticmethod
    def _numeric_column(
        frame: pd.DataFrame,
        name: str,
        *,
        dtype: np.dtype,
        default: float | None = None,
    ) -> npt.NDArray[np.floating]:
        if name not in frame:
            if default is None:
                raise ValueError(f"prices are missing required column: {name}")
            values = np.full(len(frame), default, dtype=dtype)
        else:
            series = pd.to_numeric(frame[name], errors="coerce")
            values = series.to_numpy(dtype=dtype, copy=False)
            if not values.flags.c_contiguous:
                values = np.ascontiguousarray(values, dtype=dtype)
        return values

    @classmethod
    def from_prices(cls, prices: Any) -> CompactRiskMarketData:
        frame = pd.DataFrame(prices, copy=False)
        open_values = cls._numeric_column(frame, "open", dtype=np.dtype(np.float64))
        high_values = cls._numeric_column(frame, "high", dtype=np.dtype(np.float64))
        low_values = cls._numeric_column(frame, "low", dtype=np.dtype(np.float64))
        close_values = cls._numeric_column(frame, "close", dtype=np.dtype(np.float64))
        volume_values = cls._numeric_column(
            frame, "volume", dtype=np.dtype(np.float64), default=0.0
        )
        funding = cls._numeric_column(
            frame, "funding_rate", dtype=np.dtype(np.float32), default=0.0
        )
        uncertainty = cls._numeric_column(
            frame, "uncertainty_score", dtype=np.dtype(np.float32), default=math.nan
        )

        ohlc_finite = (
            np.isfinite(open_values).all()
            and np.isfinite(high_values).all()
            and np.isfinite(low_values).all()
            and np.isfinite(close_values).all()
        )
        if not ohlc_finite:
            raise ValueError("OHLC prices must be finite")
        if (
            (open_values <= 0).any()
            or (high_values <= 0).any()
            or (low_values <= 0).any()
            or (close_values <= 0).any()
        ):
            raise ValueError("OHLC prices must be positive")
        if (high_values < np.maximum.reduce((open_values, low_values, close_values))).any():
            raise ValueError("invalid high price")
        if (low_values > np.minimum.reduce((open_values, high_values, close_values))).any():
            raise ValueError("invalid low price")
        if not np.isfinite(volume_values).all() or (volume_values < 0).any():
            raise ValueError("volume must be finite and non-negative")
        if not np.isfinite(funding).all():
            raise ValueError("funding_rate must be finite")
        invalid_uncertainty = ~np.isnan(uncertainty) & (
            ~np.isfinite(uncertainty) | (uncertainty < 0) | (uncertainty > 1)
        )
        if invalid_uncertainty.any():
            raise ValueError("uncertainty_score must be NaN or within [0, 1]")

        # Ensure retained arrays do not pin a large heterogeneous DataFrame block.
        retained: list[np.ndarray] = []
        for values, dtype in (
            (open_values, np.float64),
            (close_values, np.float64),
            (funding, np.float32),
            (uncertainty, np.float32),
        ):
            array = np.asarray(values, dtype=dtype)
            if array.base is not None or not array.flags.owndata:
                array = array.copy()
            array.flags.writeable = False
            retained.append(array)
        return cls(*retained)
