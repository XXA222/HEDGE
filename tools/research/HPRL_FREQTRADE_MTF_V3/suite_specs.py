from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class ModelSpec:
    key: str
    strategy_class: str
    strategy_file: str
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


# All five algorithms deliberately share the same position/risk envelope.
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

# The order is semantic: each base timeframe consumes itself plus every timeframe to its right.
TIMEFRAMES = ("1m", "5m", "15m", "1h", "8h", "1d")
TIMEFRAME_SECONDS = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "1h": 3600,
    "8h": 28_800,
    "1d": 86_400,
}
PERIODS_PER_YEAR = {
    "1m": 525_600,
    "5m": 105_120,
    "15m": 35_040,
    "1h": 8_760,
    "8h": 1_095,
    "1d": 365,
}

# 96 candles covers the longest feature warmup (EMA55/rolling24) with margin and also tells
# Freqtrade how much history to request for every informative timeframe.
SOURCE_WARMUP_CANDLES = 96

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

# This contract is hashed into every checkpoint.  It intentionally describes the temporal
# semantics, not just a list of columns, so an alignment change forces retraining.
MTF_ALIGNMENT_CONTRACT = {
    "decision_clock": "base_candle_close",
    "informative_timestamp_semantics": "candle_open_time",
    "informative_visibility": "source_open_plus_source_duration_le_base_close",
    "join": "searchsorted_last_closed",
    "closed_candle_only": True,
    "forward_fill": "only_last_closed_within_one_source_period",
    "stale_source_policy": "fail_closed",
    "age_feature": "normalized_seconds_since_last_source_close",
    "same_pair_only": True,
    "missing_candle_policy": "freqtrade_no_action_fill_then_strict_contiguity",
}


def input_timeframes_for(base_timeframe: str) -> tuple[str, ...]:
    try:
        index = TIMEFRAMES.index(str(base_timeframe))
    except ValueError as exc:
        raise ValueError(f"unsupported HPRL base timeframe: {base_timeframe!r}") from exc
    return TIMEFRAMES[index:]


def informative_timeframes_for(base_timeframe: str) -> tuple[str, ...]:
    return input_timeframes_for(base_timeframe)[1:]


def timeframe_signature(base_timeframe: str) -> str:
    return "__".join(input_timeframes_for(base_timeframe))


MODELS: dict[str, ModelSpec] = {
    "fast_td3": ModelSpec(
        key="fast_td3",
        strategy_class="HPRLFastTD3ETHStrategy",
        strategy_file="hprl_fast_td3_eth.py",
        algorithm="fast_td3",
        description="Deterministic distributional TD3; lower LR and stronger turnover discipline.",
        learning_rate=2.0e-4,
        hidden_dim=256,
        hidden_depth=3,
        gamma=0.995,
        tau=0.005,
        batch_size=512,
        replay_capacity=80_000,
        warmup_transitions=4096,
        reward_overrides={"drawdown": 0.50, "downside": 0.22, "turnover": 0.0030, "cvar": 0.14},
    ),
    "fast_dsac": ModelSpec(
        key="fast_dsac",
        strategy_class="HPRLFastDSACETHStrategy",
        strategy_file="hprl_fast_dsac_eth.py",
        algorithm="fast_dsac",
        description=(
            "Distributional maximum-entropy SAC; balanced exploration and downside control."
        ),
        learning_rate=3.0e-4,
        hidden_dim=256,
        hidden_depth=3,
        gamma=0.992,
        tau=0.006,
        batch_size=512,
        replay_capacity=90_000,
        warmup_transitions=4096,
        tier_entropy_target_fraction=0.58,
        reward_overrides={"drawdown": 0.48, "downside": 0.24, "cvar": 0.16, "turnover": 0.0025},
    ),
    "simba_sac": ModelSpec(
        key="simba_sac",
        strategy_class="HPRLSimbaSACETHStrategy",
        strategy_file="hprl_simba_sac_eth.py",
        algorithm="simba_sac",
        description="SimBa-style SAC; wider network with conservative gross-margin shaping.",
        learning_rate=2.5e-4,
        hidden_dim=384,
        hidden_depth=3,
        gamma=0.994,
        tau=0.005,
        batch_size=512,
        replay_capacity=100_000,
        warmup_transitions=4096,
        tier_entropy_target_fraction=0.55,
        reward_overrides={
            "drawdown": 0.52,
            "downside": 0.22,
            "cvar": 0.15,
            "gross_margin_risk": 0.11,
            "turnover": 0.0022,
        },
    ),
    "xqc": ModelSpec(
        key="xqc",
        strategy_class="HPRLXQCETHStrategy",
        strategy_file="hprl_xqc_eth.py",
        algorithm="xqc",
        description=(
            "XQC categorical critic with conditioning constraints; stability-oriented profile."
        ),
        learning_rate=2.0e-4,
        hidden_dim=320,
        hidden_depth=3,
        gamma=0.995,
        tau=0.004,
        batch_size=512,
        replay_capacity=100_000,
        warmup_transitions=4096,
        tier_entropy_target_fraction=0.52,
        reward_overrides={
            "drawdown": 0.55,
            "downside": 0.25,
            "cvar": 0.18,
            "risk_projection": 0.10,
            "turnover": 0.0025,
        },
    ),
    "rebrac_v2": ModelSpec(
        key="rebrac_v2",
        strategy_class="HPRLReBRACv2ETHStrategy",
        strategy_file="hprl_rebrac_v2_eth.py",
        algorithm="rebrac_v2",
        description="Offline ReBRAC-v2 flow policy trained on a momentum/random behavior mixture.",
        learning_rate=2.0e-4,
        hidden_dim=320,
        hidden_depth=3,
        gamma=0.995,
        tau=0.005,
        batch_size=512,
        replay_capacity=80_000,
        warmup_transitions=4096,
        reward_overrides={
            "drawdown": 0.58,
            "downside": 0.26,
            "cvar": 0.18,
            "turnover": 0.0028,
            "hedge_overlap": 0.03,
        },
    ),
}


def reward_kwargs(spec: ModelSpec) -> dict[str, float]:
    values = dict(BASE_REWARD_KWARGS)
    values.update(spec.reward_overrides)
    return values
