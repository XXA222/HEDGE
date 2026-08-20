from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class StrategySpec:
    name: str
    algorithm: str
    description: str
    learning_rate: float
    hidden_dim: int
    hidden_depth: int
    gamma: float
    tau: float
    batch_size: int
    replay_capacity: int
    warmup_transitions: int
    gradient_clip_norm: float = 10.0
    tier_entropy_target_fraction: float = 0.65
    reward_overrides: Mapping[str, float] = field(default_factory=dict)


# Unified execution/risk contract across models.  This makes the five algorithms
# directly comparable: they face the same tier grid, leverage envelope and costs.
ACTION_KWARGS = {
    "mode": "tiered",
    "position_levels": (0.0, 0.03, 0.07, 0.12, 0.20),
    "leverage": 2.0,
    "max_leg_margin_ratio": 0.20,
    "max_gross_margin_ratio": 0.35,
    "max_abs_net_margin_ratio": 0.20,
    "max_increase_levels": 1,
    "max_decrease_levels": -1,
    "tier_hysteresis": 0.02,
}

COST_KWARGS = {
    "maker_fee_bps": 2.0,
    "taker_fee_bps": 5.0,
    "base_slippage_bps": 0.5,
    "impact_coefficient_bps": 2.0,
    "max_participation": 0.05,
}

BASE_REWARD_KWARGS = {
    "return_scale": 100.0,
    "equity": 1.0,
    "drawdown": 0.45,
    "downside": 0.20,
    "cvar": 0.12,
    "turnover": 0.002,
    "fees": 0.0,
    "slippage": 0.0,
    "market_impact": 0.0,
    "funding": 0.0,
    "quantization_alignment": 0.005,
    "risk_projection": 0.08,
    "gross_margin_risk": 0.08,
    "hedge_overlap": 0.02,
    "opportunity_cost": 0.0,
    "terminal_loss": 3.0,
    "gross_margin_soft_limit": 0.25,
    "reward_clip": 5.0,
}

PERIODS_PER_YEAR = {
    "1m": 525_600,
    "5m": 105_120,
    "15m": 35_040,
    "1h": 8_760,
    "8h": 1_095,
    "1d": 365,
}

# Number of environment decision steps / optimizer updates per fold.
# The windowed trainer samples starts across the entire training history, so
# even the 1m model is not confined to the first few days of a 2-year tape.
TRAIN_STEPS = {
    "fast": {
        "1m": 1200, "5m": 1100, "15m": 1000, "1h": 900, "8h": 700, "1d": 600,
    },
    "balanced": {
        "1m": 4500, "5m": 4000, "15m": 3500, "1h": 3000, "8h": 2200, "1d": 1800,
    },
    "deep": {
        "1m": 14000, "5m": 12000, "15m": 10000, "1h": 8000, "8h": 5500, "1d": 4000,
    },
}

WINDOW_STEPS = {
    "1m": 256,
    "5m": 256,
    "15m": 192,
    "1h": 168,
    "8h": 90,
    "1d": 60,
}
