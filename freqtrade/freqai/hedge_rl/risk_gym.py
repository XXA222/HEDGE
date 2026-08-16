"""Gymnasium import shim used by the risk-level environment.

Freqtrade installs gymnasium in production.  The tiny fallback keeps static/unit tests
for the pure Hedge overlay runnable in minimal build environments; it is not used by SB3.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np


try:  # pragma: no cover - exercised in the real Freqtrade environment
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - lightweight source-validation fallback

    class _Env:
        def reset(self, *, seed=None, options=None):
            del seed, options
            return None

    class _Box:
        def __init__(self, low, high, shape, dtype):
            self.low = low
            self.high = high
            self.shape = tuple(shape)
            self.dtype = dtype

        def contains(self, value):
            arr = np.asarray(value)
            return arr.shape == self.shape and np.isfinite(arr).all()

    class _MultiDiscrete:
        def __init__(self, nvec):
            self.nvec = np.asarray(nvec, dtype=np.int64)
            self.shape = self.nvec.shape

        def contains(self, value):
            arr = np.asarray(value)
            return (
                arr.shape == self.shape
                and np.issubdtype(arr.dtype, np.integer)
                and (arr >= 0).all()
                and (arr < self.nvec).all()
            )

    class _Spaces:
        Box = _Box
        MultiDiscrete = _MultiDiscrete

    class _Gym:
        Env = _Env

    gym = cast(Any, _Gym())
    spaces = cast(Any, _Spaces())
