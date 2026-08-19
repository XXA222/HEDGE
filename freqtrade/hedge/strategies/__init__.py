"""Reference Hedge strategies."""

from freqtrade.hedge.strategies.simple_ma_hedge import (
    SimpleDualLegMaConfig,
    SimpleDualLegMaHedgeStrategy,
)
from freqtrade.hedge.strategies.hprl_eth_dual_leg import (
    HprlEthDualLegConfig,
    HprlEthDualLegStrategy,
)


__all__ = [
    "HprlEthDualLegConfig",
    "HprlEthDualLegStrategy",
    "SimpleDualLegMaConfig",
    "SimpleDualLegMaHedgeStrategy",
]
