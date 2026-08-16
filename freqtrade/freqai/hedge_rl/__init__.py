"""Dual-leg machine-learning and reinforcement-learning components for Hedge mode."""
# Public re-export grouping is intentional.
# ruff: noqa: I001

from .config import HedgeRLConfig, RewardWeights
from .contracts import ConfigSchemaVersion, SeedLedger
from .features import FeatureSchema
from .risk_levels import HedgeRiskLevelAction, PositionRiskLevel, RiskLevelProfile
from .risk_reward import HedgeRiskRewardModel, RiskRewardConfig
from .risk_runtime import RiskRLAdaptiveCpuConfig, RiskRLAdaptiveCpuController

HEDGE_MLRL_SOURCE_VERSION = "clean-mainline"
HEDGE_MLRL_COMPLETED_ROUNDS = 80
HEDGE_MLRL_ADVANCED_ROUNDS = 80
HEDGE_RISK_LEVEL_RL_CONTRACT = "multidiscrete-5x5-risk-cap-v3"

__all__ = [
    "ConfigSchemaVersion",
    "FeatureSchema",
    "HEDGE_MLRL_ADVANCED_ROUNDS",
    "HEDGE_MLRL_COMPLETED_ROUNDS",
    "HEDGE_MLRL_SOURCE_VERSION",
    "HEDGE_RISK_LEVEL_RL_CONTRACT",
    "HedgeRLConfig",
    "HedgeRiskLevelAction",
    "HedgeRiskRewardModel",
    "PositionRiskLevel",
    "RiskLevelProfile",
    "RiskRewardConfig",
    "RiskRLAdaptiveCpuConfig",
    "RiskRLAdaptiveCpuController",
    "RewardWeights",
    "SeedLedger",
]
